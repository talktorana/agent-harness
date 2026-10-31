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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

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


def _due(job: Dict[str, Any], now: datetime,
         state: Optional[Dict[str, Any]] = None, key: str = "") -> bool:
    """Whether a job should fire now, given its persisted last_run."""
    if not job.get("enabled", True):
        return False
    stamp = ((state or {}).get("jobs", {}).get(key) or {}).get("last_run") \
        or job.get("last_run")   # an inline stamp in an old-style declaration
    every = job.get("every_min")
    if every:
        if not stamp:
            return True
        try:
            previous = datetime.fromisoformat(str(stamp))
        except ValueError:
            return True          # unreadable stamp: fire once, then rewrite it
        return (now - previous).total_seconds() >= float(every) * 60
    at = job.get("at")
    if at:
        hh, mm = (int(x) for x in str(at).split(":")[:2])
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        done_today = str(stamp or "")[:10] == now.date().isoformat()
        return now >= target and not done_today
    return False


def due_jobs(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """The jobs due now, each annotated with its state key (``_key``)."""
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
            record["answer"] = run_task(job.get("message", ""), verbose=verbose)
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
    jobs = []
    for index, job in enumerate(data.get("jobs", [])):
        if not isinstance(job, dict):
            continue
        key = _job_key(job, index)
        jobs.append({**job, "_key": key,
                     "_state": state.get("jobs", {}).get(key, {}),
                     "_due": _due(job, datetime.now(), state, key)})
    return {"schedule_file": str(config.SCHEDULE_FILE),
            "schedule_readable": _read_json(config.SCHEDULE_FILE) is not None,
            "state_file": str(_state_path()),
            "state_writable": state_writable(),
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
    #: [message, last_printed_monotonic, first_seen_monotonic]
    last_error: List[Any] = []
    while True:
        message: Optional[str] = None
        try:
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
