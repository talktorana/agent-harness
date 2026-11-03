"""Cron-lite scheduler: unattended missions without an OS cron entry.

The cloud story for this harness is a container with no systemd and no crontab
guarantee, so the schedule lives in a JSON file next to the code and a tiny
daemon loop decides what is due. Jobs are plain mission strings -- the same
text the interactive CLI takes -- so a scheduled pass and a manual pass are
the same code path.

schedule.json:

    {
      "jobs": [
        {"id": "monitor", "every_min": 30, "enabled": true,
         "message": "run a full monitoring pass and report"},
        {"id": "nightly", "at": "02:30", "enabled": true,
         "message": "summarise yesterday's sessions into memory"}
      ]
    }

``every_min`` fires on a fixed interval from the last run (or from daemon
start). ``at`` fires once per calendar day at HH:MM local time. A job that
raises is recorded as failed and retried on its next due time -- a broken
mission must not take the scheduler down with it.

WHY STATE LIVES IN A SECOND FILE
--------------------------------
The declaration is configuration: it is reviewed, versioned, and in a gated
deployment mounted read-only AND on the applier's forbidden list, because an
agent able to rewrite its own mission text owns its own guardrails. Runtime
state (``last_run``, ``last_error``) is neither, and this module used to write
it back into the declaration. On a read-only mount that raised ``OSError``
inside the poll loop, where it was caught and printed as a generic
``[scheduler] loop error`` -- a message that reads as transient while the truth
is that no mission ever ran, and the reason (an unwritable file) never appeared.

So state goes to ``config.SCHEDULE_STATE_FILE`` (a sibling of the session log by
default, i.e. agent-writable). A declaration on a read-only mount schedules
normally.

AND WHY AN UNWRITABLE STATE FILE IS FATAL, NOT IGNORED
------------------------------------------------------
``last_run`` is stamped BEFORE the job runs, so an overlapping poll cannot start
a second copy of a long mission. If that stamp cannot be persisted the
scheduler has no memory: the job is due again on the next poll, and again after
that -- an unattended loop spending tokens on the same mission every 30 seconds.
That is worse than not running at all, so ``run_due_jobs`` raises
``SchedulerStateError`` and runs nothing. The daemon names the problem and the
fix on its own log line, and ``--once`` exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import run_settings

_LOCK = threading.Lock()

#: A repeating log line is rate-limited to at most one per this many seconds.
REPEAT_LOG_SEC = float(os.environ.get("AGENT_SCHEDULER_REPEAT_LOG_SEC", "600"))


#: Raised when per-job runtime state cannot be persisted. Distinct from OSError
#: so the daemon can name the actual problem (and the fix) instead of reporting
#: a generic write failure that reads like a transient fault.
class SchedulerStateError(RuntimeError):
    """Runtime state is unwritable, so scheduling is unsafe."""


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a JSON object, or None when it is absent/corrupt/not an object.

    Degrading rather than raising is deliberate: for the declaration that means
    "no jobs" (inert), and for state it means "nothing has run yet", which the
    next pass corrects by rewriting the file. Neither is worth killing a daemon
    over.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _load() -> Dict[str, Any]:
    """The declaration: jobs, cadence, message. Never written to."""
    data = _read_json(config.SCHEDULE_FILE)
    if data is None or not isinstance(data.get("jobs"), list):
        return {"jobs": []}
    return data


def _state_path() -> Path:
    return Path(config.SCHEDULE_STATE_FILE)


def _load_state() -> Dict[str, Any]:
    """Per-job runtime state, keyed by job id."""
    data = _read_json(_state_path()) or {}
    jobs = data.get("jobs")
    return {"jobs": jobs if isinstance(jobs, dict) else {}}


def _save_state(state: Dict[str, Any]) -> None:
    """Persist runtime state atomically, or raise SchedulerStateError."""
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise SchedulerStateError(
            f"cannot persist scheduler state to {path}: {exc}. Set "
            f"AGENT_SCHEDULE_STATE_FILE to a writable path (the declaration at "
            f"{config.SCHEDULE_FILE} may legitimately be read-only). Refusing "
            f"to run jobs: without a persisted last_run, every poll would "
            f"re-trigger the same mission.") from exc


def _job_key(job: Dict[str, Any], index: int) -> str:
    """A stable identity for state.

    An anonymous job falls back to its position. That is a weak identity (an
    insert renumbers it), but refusing to schedule an id-less job would be a
    worse surprise than scheduling it under a key that shows up as ``#2`` in
    the state file, where the weakness is visible.
    """
    ident = job.get("id")
    return str(ident) if ident else f"#{index}"


def _interval_label(job: Dict[str, Any]) -> str:
    """How often this job runs, in words a glance can read."""
    every = job.get("every_min")
    if every:
        try:
            minutes = float(every)
        except (TypeError, ValueError):
            return str(every)
        if minutes >= 60 and minutes % 60 == 0:
            return f"every {int(minutes // 60)}h"
        return f"every {minutes:g} min"
    if job.get("at"):
        return f"daily at {job['at']}"
    return "not scheduled"


def _parse_stamp(value: Any) -> Optional[datetime]:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp


def _comparable(stamp: Optional[datetime],
                reference: datetime) -> Optional[datetime]:
    """Put ``stamp`` on the same clock as ``reference`` so maths cannot raise.

    The scheduler writes NAIVE local stamps (``datetime.now()``) while sessions
    are named in UTC, so the two surfaces genuinely mix awareness. Subtracting
    them raises ``TypeError: can't subtract offset-naive and offset-aware`` --
    and a console that crashes on a timestamp is worse than one that shows
    nothing, because the operator cannot even tell which surface is broken.
    A naive stamp is read as UTC, which is exact on any host whose local time
    is UTC (this deployment) and is the conservative reading elsewhere.
    """
    if stamp is None:
        return None
    if stamp.tzinfo is None and reference.tzinfo is not None:
        return stamp.replace(tzinfo=timezone.utc)
    if stamp.tzinfo is not None and reference.tzinfo is None:
        return stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp



def status(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Everything an operator needs to see whether the loop is alive.

    The console could show what happened but not WHY nothing was happening, so
    "paused", "disabled", "running every 30 minutes" and "the scheduler is
    wedged" all looked like an empty page. This is the single place that
    assembles that answer, so the UI cannot drift from what the loop obeys --
    ``_due`` decides whether a job fires, and this reports the same inputs.

    ``now`` defaults to NAIVE local, matching the clock the run loop and the
    state file use. Using UTC here instead would make ``due_now`` disagree with
    what the loop actually does near a boundary, i.e. the view would lie about
    the very thing it exists to show.
    """
    when = now or datetime.now()
    doc = _load()
    state = _load_state()
    held = hold()
    jobs: List[Dict[str, Any]] = []

    for index, job in enumerate(doc.get("jobs") or []):
        if not isinstance(job, dict):
            continue
        key = _job_key(job, index)
        entry = (state.get("jobs") or {}).get(key) or {}
        message = str(job.get("message") or "")
        last_run = _comparable(_parse_stamp(entry.get("last_run")), when)
        finished = _comparable(_parse_stamp(entry.get("finished")), when)

        #: The interval in force is the CONSOLE OVERRIDE if one is set, else
        #: what the declaration says. Both are reported: an operator has to be
        #: able to see that the page's 45 is what governs, not the 30 in the
        #: file, otherwise the declaration reads as the truth and the override
        #: looks like a display bug.
        every, every_source = run_settings.effective_every_min(job.get("every_min"))
        next_due: Optional[datetime] = None
        seconds_until: Optional[float] = None
        if job.get("enabled", True) and every and last_run:
            try:
                next_due = last_run + timedelta(minutes=float(every))
                seconds_until = (next_due - when).total_seconds()
            except (TypeError, ValueError):
                next_due = None

        #: ``last_run`` and ``finished`` are NOT a pair while a run is going.
        #: last_run is stamped when the job FIRES; finished is only rewritten
        #: when it RETURNS. So during a run, finished still describes the
        #: PREVIOUS one and is earlier than last_run -- and subtracting them
        #: produced a large negative duration. Live 2026-11-02 the console
        #: showed "took -1072.7s" for a healthy, in-progress mission, which
        #: reads as a broken clock rather than a run in progress.
        #:
        #: So report the duration only when the two stamps actually describe the
        #: same run, and say plainly that a run is in progress otherwise.
        running = bool(last_run and finished and finished < last_run)
        duration: Optional[float] = None
        if last_run and finished and not running:
            try:
                duration = round((finished - last_run).total_seconds(), 1)
            except (TypeError, ValueError):
                duration = None

        #: WHEN the run that is in progress started, and how long it has been
        #: going. ``next_due`` is ALWAYS the next FIRE time (last_run + the
        #: interval), including while a run is live -- but a reader who sees
        #: ``running: true`` next to a future ``next_due`` has no way to tell
        #: whether that stamp means "when this run ends" or "when the next one
        #: starts", and guessing is worse than not asking.
        #:
        #: Live 2026-11-02 the 09:20 pass hit exactly this: it read a positive
        #: ``seconds_until`` with ``running: true``, reasoned aloud about both
        #: interpretations, and reported its own next-wake time as
        #: **UNVERIFIED**. A fact the console already held cost turns and ended
        #: as a hedge in the one field an operator scans first. So the console
        #: now states the basis outright rather than leaving it to be derived.
        in_progress_since = entry.get("last_run") if running else None
        in_progress_sec: Optional[float] = None
        if running and last_run:
            try:
                in_progress_sec = round((when - last_run).total_seconds(), 1)
            except (TypeError, ValueError):
                in_progress_sec = None

        jobs.append({
            "id": key,
            "interval": _interval_label({**job, "every_min": every}),
            "every_min": every,
            #: Where the interval came from: "console" (an operator set it on
            #: the page), "schedule" (the declaration) or "none" (an `at` job).
            "every_min_source": every_source,
            #: The declared value, so a page can say "45 in force, declaration
            #: says 30" -- without it an override is invisible in the payload
            #: that reports scheduling, and the next operator re-discovers it
            #: by watching runs fire at an interval nobody set.
            "every_min_declared": job.get("every_min"),
            "enabled": bool(job.get("enabled", True)),
            # The prompt is the thing an operator most wants to check -- "what
            # is it actually being told to do every half hour?" -- so it is
            # served in full, with a preview for list rendering.
            "prompt": message,
            "prompt_preview": " ".join(message.split())[:160],
            "prompt_chars": len(message),
            "last_run": entry.get("last_run"),
            "finished": entry.get("finished"),
            "last_ok": entry.get("last_ok"),
            "last_error": entry.get("last_error"),
            "next_due": next_due.isoformat() if next_due else None,
            #: Say WHAT next_due means, so it cannot be read as the end of the
            #: run in flight. A future stamp with `running: true` is otherwise
            #: genuinely ambiguous, and the field an operator looks at first is
            #: the wrong place to leave an inference.
            "next_due_basis": ("last_run + every_min; this is the next FIRE "
                               "time, even while a run is in progress"),
            "seconds_until": (round(seconds_until)
                              if seconds_until is not None else None),
            "due_now": _due(job, when, state, key),
            "overdue": bool(seconds_until is not None and seconds_until < 0),
            #: A run in progress: finished has not caught up with last_run yet.
            "running": running,
            #: When it started, and for how long. Non-null only while running,
            #: so a reader never has to decide between "the run in flight" and
            #: "the previous one" from `last_run`/`finished` alone.
            "in_progress_since": in_progress_since,
            "in_progress_sec": in_progress_sec,
            #: Duration of the LAST COMPLETED run, never negative.
            "duration_sec": duration,
            #: A run that died with a previous process. Shown so an operator
            #: can see the pass was KILLED, not skipped: a deploy that eats
            #: missions should be visible in the view that reports schedules.
            "interrupted": entry.get("interrupted"),
            "interrupted_count": entry.get("interrupted_count"),
        })

    return {
        "now": when.isoformat(),
        "schedule_file": str(config.SCHEDULE_FILE),
        "state_file": str(config.SCHEDULE_STATE_FILE),
        "hold": held,
        "paused": bool(held.get("held")),
        "jobs": jobs,
        "any_enabled": any(j["enabled"] for j in jobs),
    }


def hold() -> Dict[str, Any]:
    """Whether an operator has PAUSED the mission loop, and why.

    Presence of the file is the signal, not its contents. That makes the two
    things an operator actually reaches for work without reading documentation:
    ``touch`` pauses, ``rm`` resumes. The JSON body is only read for the "who and
    when" that makes a pause explainable later; an unreadable body still pauses,
    because presence already expressed the intent and failing to PARSE must not
    be the one thing that silently resumes an agent that can run shell commands.

    Deliberately not the declaration's ``enabled`` flag: the declaration is
    reviewed configuration on a read-only mount, and in a gated deployment it is
    on the applier's refuse list -- so the pause has to live somewhere an
    operator (not the agent) can write, while the agent can still read it.
    """
    raw = getattr(config, "SCHEDULE_HOLD", "") or ""
    out: Dict[str, Any] = {"configured": bool(raw), "held": False,
                           "path": str(raw) if raw else None,
                           "reason": "", "since": None, "by": None}
    if not raw:
        return out
    path = Path(raw)
    if not path.exists():
        return out
    out["held"] = True
    body = _read_json(path) or {}
    out["by"] = body.get("by")
    out["since"] = body.get("since") or body.get("at")
    out["reason"] = str(body.get("reason") or
                        "scheduler paused by an operator hold file")
    return out


def _due(job: Dict[str, Any], now: datetime,
         state: Optional[Dict[str, Any]] = None, key: str = "") -> bool:
    """Whether a job should fire now, given its persisted last_run."""
    if not job.get("enabled", True):
        return False
    stamp = ((state or {}).get("jobs", {}).get(key) or {}).get("last_run") \
        or job.get("last_run")   # an inline stamp in an old-style declaration
    #: The console override wins over the declaration AT FIRE TIME, so cadence
    #: is tunable while the daemon runs and the declaration stays read-only.
    #: An unreadable override must not stop the loop: fall back to what the
    #: declaration says and let the console report the read error.
    every = job.get("every_min")
    try:
        every, _source = run_settings.effective_every_min(every)
    except Exception:  # noqa: BLE001
        pass
    if every:
        if not stamp:
            return True
        try:
            previous = datetime.fromisoformat(str(stamp))
        except ValueError:
            return True          # unreadable stamp: fire once, then rewrite it
        #: Normalise before subtracting. A stamp and a clock of different
        #: awareness raise TypeError, and this function gates every run -- so a
        #: crash here stops the mission loop entirely rather than skipping one
        #: job. Found when the schedule view passed an aware clock.
        previous = _comparable(previous, now) or previous
        return (now - previous).total_seconds() >= float(every) * 60
    at = job.get("at")
    if at:
        hh, mm = (int(x) for x in str(at).split(":")[:2])
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        done_today = str(stamp or "")[:10] == now.date().isoformat()
        return now >= target and not done_today
    return False


def due_jobs(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """The jobs due now, each annotated with its state key (``_key``).

    A hold returns nothing at all, before the declaration is even consulted: the
    pause is meant to stop the loop, so it must not depend on which jobs happen
    to be enabled today.
    """
    if hold()["held"]:
        return []
    now = now or datetime.now()
    data, state = _load(), _load_state()
    due: List[Dict[str, Any]] = []
    for index, job in enumerate(data.get("jobs", [])):
        if not isinstance(job, dict):
            continue
        key = _job_key(job, index)
        if _due(job, now, state, key):
            due.append({**job, "_key": key})
    return due


def reclaim_interrupted(now: Optional[datetime] = None) -> List[str]:
    """Re-open the slot of any run that died with the PREVIOUS process.

    ``last_run`` is stamped BEFORE a job runs, and ``finished`` is written only
    when it RETURNS. That ordering is correct -- it is what stops an overlapping
    poll from starting the same mission twice -- but it leaves a signature that
    nothing used to interpret:

        last_run stamped, finished never brought forward  ==  the run died.

    On this deployment that is not a rare event: recreating the sidecar kills
    whatever is running, and rebuilding the image is how a fix is deployed, so
    every deploy risks one. Live 2026-11-02 the 06:13 pass was killed 5 minutes
    in; its slot was treated as served, the next pass waited until 06:49, and
    the console showed the job as ``running`` for 36 minutes because
    ``finished`` still described the run before it.

    THE DETECTION IS EXACT, not a timeout heuristic. Jobs run in THIS process,
    so at startup there is no live run to protect: a persisted stamp with no
    finishing stamp can only have been left by a process that is gone. That is
    why this runs once at startup and never mid-loop -- mid-loop the same
    signature means a run in progress, and stealing that slot would double-run
    the mission.

    The slot is RE-OPENED (``last_run`` removed) and the fact is RECORDED
    (``interrupted``), so the next poll fires immediately AND an operator can
    see the pass was killed rather than lost. A silent re-run would hide a
    deploy that keeps eating missions; a silent loss is what we had.
    """
    when = now or datetime.now()
    reclaimed: List[str] = []
    with _LOCK:
        state = _load_state()
        jobs = state.get("jobs") or {}
        for key, entry in jobs.items():
            if not isinstance(entry, dict):
                continue
            last = _parse_stamp(entry.get("last_run"))
            finished = _parse_stamp(entry.get("finished"))
            if last is None:
                continue
            if finished is not None and finished >= last:
                continue          # that run ended; nothing to reclaim
            entry["interrupted"] = {
                "started": entry.get("last_run"),
                "detected": when.isoformat(timespec="seconds"),
                "reason": "the process ended before this run finished, so its "
                          "slot is re-opened and the pass runs again",
            }
            #: ``count`` is cumulative for the operator, not per interruption.
            try:
                entry["interrupted_count"] = int(
                    entry.get("interrupted_count") or 0) + 1
            except (TypeError, ValueError):
                entry["interrupted_count"] = 1
            entry.pop("last_run", None)
            reclaimed.append(key)
        if reclaimed:
            _save_state(state)
    return reclaimed


def run_due_jobs(verbose: bool = False) -> List[Dict[str, Any]]:
    """Run every job that is due NOW. Returns one record per executed job.

    Raises SchedulerStateError BEFORE running anything when the stamp cannot be
    persisted -- see the module docstring: a mission the scheduler cannot
    remember starting will start again on the next poll.
    """
    from agent import run_task

    ran: List[Dict[str, Any]] = []
    with _LOCK:
        now = datetime.now()
        due = due_jobs(now)
        if not due:
            return ran
        state = _load_state()
        stamp = now.isoformat(timespec="seconds")
        for job in due:
            state["jobs"].setdefault(job["_key"], {})["last_run"] = stamp
        _save_state(state)

    for job in due:
        record: Dict[str, Any] = {"id": job["_key"],
                                  "started": datetime.now().isoformat()}
        try:
            #: Tagged so the run these sessions share can still say which rows
            #: the scheduler produced. Untagged, a six-second console smoke
            #: test sits in the same list, green, and reads as a mission that
            #: did no work -- see SessionLog.origin.
            record["answer"] = run_task(job.get("message", ""),
                                        verbose=verbose, origin="schedule")
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished"] = datetime.now().isoformat()
        ran.append(record)
        # A FAILED job still consumed its slot. Retrying on the next poll would
        # turn a broken mission into a 30-second retry loop burning tokens.
        _record_outcome(job["_key"], record)
    return ran


def _record_outcome(key: str, record: Dict[str, Any]) -> None:
    """Write ok/error back into state. A failure here must not kill the loop."""
    with _LOCK:
        state = _load_state()
        entry = state["jobs"].setdefault(key, {})
        entry["last_ok"] = bool(record.get("ok"))
        entry["finished"] = record.get("finished")
        if record.get("error"):
            entry["last_error"] = str(record["error"])[:500]
        else:
            entry.pop("last_error", None)
        try:
            _save_state(state)
        except SchedulerStateError as exc:
            print(f"[scheduler] could not record the outcome for {key}: {exc}",
                  flush=True)


def state_writable() -> bool:
    """Can runtime state be written? Reported by --list; never assumed."""
    path = _state_path()
    probe = path.with_suffix(".probe")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        return True
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def describe() -> Dict[str, Any]:
    """The declaration merged with its runtime state (for --list and the UI)."""
    data, state = _load(), _load_state()
    paused = hold()
    jobs = []
    for index, job in enumerate(data.get("jobs", [])):
        if not isinstance(job, dict):
            continue
        key = _job_key(job, index)
        jobs.append({**job, "_key": key,
                     "_state": state.get("jobs", {}).get(key, {}),
                     #: A held job is NOT due, whatever its cadence says. The
                     #: declaration still reads `enabled: true` -- the pause is
                     #: an outside file, not an edit -- so reporting `_due: true`
                     #: here would describe a job that will not run, which is the
                     #: exact confusion this hold exists to remove.
                     "_due": (False if paused["held"]
                              else _due(job, datetime.now(), state, key))})
    return {"schedule_file": str(config.SCHEDULE_FILE),
            "schedule_readable": _read_json(config.SCHEDULE_FILE) is not None,
            "state_file": str(_state_path()),
            "state_writable": state_writable(),
            #: Reported, not inferred. A paused loop and a broken loop both run
            #: no jobs, and telling them apart is the whole question an operator
            #: is asking when they open this.
            "hold": hold(),
            "jobs": jobs}


def _rate_limited(message: str, last: List[Any]) -> Optional[str]:
    """First occurrence prints at once; repeats print at most hourly.

    A poll loop that re-reports one unchanged failure every 30 seconds buries
    the evidence under ~2,880 identical lines a day, which is how the
    read-only-mount failure stayed invisible: it was in the log, indistinguishable
    from noise. A repeat that DOES print says so, so a reader knows the
    condition is ongoing rather than flapping.
    """
    if not message:
        return None
    now = time.monotonic()
    if last and last[0] == message:
        if now - float(last[1]) < REPEAT_LOG_SEC:
            return None
        last[1] = now
        return (f"{message} (unchanged for "
                f"{(now - float(last[2])) / 60:.0f} min; repeats at most every "
                f"{REPEAT_LOG_SEC / 60:.0f} min)")
    if last and last[0] and last[0] != message:
        previous = last[0]
        last[:] = [message, now, now]
        return f"{message} (previous condition cleared: {previous})"
    last[:] = [message, now, now]
    return message


def daemon(poll_sec: float = 30.0) -> None:
    """Block forever, running due jobs. One bad job never kills the loop."""
    print(f"[scheduler] watching {config.SCHEDULE_FILE} every {poll_sec:.0f}s "
          f"(state: {_state_path()})", flush=True)
    if not state_writable():
        print(f"[scheduler] WARNING: {_state_path()} is not writable; jobs will "
              f"NOT run until AGENT_SCHEDULE_STATE_FILE points somewhere "
              f"writable.", flush=True)
    #: A run killed by a recreate must not cost a whole cycle. This is the one
    #: moment the signature is unambiguous -- see reclaim_interrupted.
    try:
        for key in reclaim_interrupted():
            entry = (_load_state().get("jobs") or {}).get(key) or {}
            interrupted = entry.get("interrupted") or {}
            print(f"[scheduler] {key}: the previous run started "
                  f"{interrupted.get('started')} and never finished -- this "
                  f"process is new, so it died with the old one. Its slot is "
                  f"re-opened and the pass will run now "
                  f"(interruptions so far: "
                  f"{entry.get('interrupted_count')}).", flush=True)
    except SchedulerStateError as exc:
        print(f"[scheduler] could not reclaim an interrupted run: {exc}",
              flush=True)
    #: [message, last_printed_monotonic, first_seen_monotonic]
    last_error: List[Any] = []
    while True:
        message: Optional[str] = None
        try:
            paused = hold()
            if paused["held"]:
                #: Said once per transition, not once per poll: a hold that
                #: repeats every 30 seconds is 2,880 lines a day, which is how
                #: the read-only-mount failure stayed invisible.
                message = (f"[scheduler] PAUSED by hold file {paused['path']}"
                           + (f" (by {paused['by']}, since {paused['since']})"
                              if paused["by"] or paused["since"] else "")
                           + "; no jobs will run until it is removed")
            for record in run_due_jobs():
                state = "ok" if record.get("ok") else \
                    f"FAILED: {record.get('error')}"
                print(f"[scheduler] job {record.get('id')}: {state}", flush=True)
        except SchedulerStateError as exc:
            message = f"[scheduler] REFUSING TO RUN: {exc}"
        except Exception as exc:  # noqa: BLE001
            message = f"[scheduler] loop error: {type(exc).__name__}: {exc}"
        line = _rate_limited(message or "", last_error)
        if line:
            print(line, flush=True)
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="Run due jobs once and exit.")
    parser.add_argument("--list", action="store_true",
                        help="Show the declaration, its state, and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what is due now; run nothing.")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(describe(), indent=2))
        return 0
    if args.dry_run:
        due = due_jobs()
        print(json.dumps([{"id": j["_key"], "every_min": j.get("every_min"),
                           "at": j.get("at"), "enabled": j.get("enabled"),
                           "message_chars": len(j.get("message") or "")}
                          for j in due], indent=2))
        return 0
    if args.once:
        try:
            records = run_due_jobs(verbose=True)
        except SchedulerStateError as exc:
            # Non-zero: a pass that did not happen is not a success. A cron
            # unit that reports 0 while refusing to run is indistinguishable
            # from a quiet system.
            print(f"[scheduler] REFUSING TO RUN: {exc}", flush=True)
            return 2
        print(json.dumps(records, indent=2, default=str)
              if records else "(nothing due)")
        return 0
    daemon(args.poll_sec)
    return 0
