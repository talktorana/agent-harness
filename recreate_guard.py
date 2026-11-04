#!/usr/bin/env python3
"""Recreate a running agent sidecar WITHOUT killing a run in flight.

WHY THIS EXISTS
---------------
Deploying a fix to a containerized agent usually means rebuilding its image and
recreating the container -- and recreating the container kills whatever agent
run is in progress. The step that ships a fix is therefore also the step that
can destroy the pass that is about to inherit it. On one consumer deployment
(2026-11-03) that happened twice inside ten minutes: two live monitoring
passes, at turn 46 and turn 24 of a 60-turn budget, died with no ``session_end``
record because two recreates landed on top of them. The consumer's mission text
already *described* the required check; the passes hand-rolled throwaway
recreate scripts anyway, and one of those scripts contained a bare ``sleep(5)``
and no probe of any kind. A guard every operator or agent has to remember to
re-implement is not a guard -- so it lives here, as code, with tests.

WHAT IT DECIDES, AND WHY NOT "BOTH SIGNALS MUST AGREE"
------------------------------------------------------
``scheduler.running`` (served by ``webui._scheduler_view`` from
``scheduler.status()``) is AUTHORITATIVE for "is a run live right now". It
derives from the scheduler's own state file, where the ``finished`` stamp is
only rewritten when a run RETURNS, so ``finished < last_run`` can only mean a
run is genuinely in flight. The per-run rows are NOT authoritative: a row says
``running`` while its session file was touched recently, so a FINISHED run keeps
reading as running for minutes, then flips to ``no_end_record`` while the
scheduler is still mid-run -- trusting the row in that direction recreates the
container and kills a live pass.

A second, independent signal -- "the newest session file contains
``session_end``" -- sounds like a good cross-check, and requiring BOTH signals
to agree is what earlier mission text said. That rule DEADLOCKS: a run killed
by an earlier recreate never gains a ``session_end``, so the newest file can
lack one forever while nothing at all is running. An operator obeying "wait
until both agree" waits out the next run too, and the fix being deployed never
ships. So here the session file CORROBORATES but never vetoes: a missing
``session_end`` only blocks when it belongs to the very run the scheduler
reports in flight (matched on the start stamp).

Building an image is always safe -- it replaces no container. Only the
recreate kills, so only the recreate is gated, and it is gated twice: once
before waiting/deciding and once again seconds before the destructive command,
because a single sample can land in the gap before a run starts ("idle for 12
seconds" is not "ended").

WHERE IT RUNS
-------------
On the HOST (or wherever the container runtime is reachable), not inside the
container -- it needs the container CLI and the session files as the host sees
them. Session paths reported by the console are the paths the CONSOLE sees
(inside the container); ``--mount CONTAINER_PREFIX=HOST_PREFIX`` maps them onto
the host filesystem. With no mapping the paths are used as-is, which is the
correct behaviour for an agent running un-containerized on the same machine.

Usage (probe only -- prints the decision, changes nothing, always safe):
    python3 recreate_guard.py --mount /agent-data=/srv/agent-data
The real thing (builds the image, recreates only if idle, verifies in-container):
    python3 recreate_guard.py --mount /agent-data=/srv/agent-data \
        --docker "sudo -n docker" --image my-agent:1.0.0 \
        --build-context /srv/agent-src --compose-dir /srv/deploy \
        --compose-file agent.yml --container my-agent --build --apply

Exit codes (stable contract -- callers branch on these, not on prose):
    0  ok (or a clean check-only run)
    1  usage/configuration error (a required setting is missing)
    2  refused: the scheduler could not answer, so "idle" was never established
    3  busy: a run was in flight when the budget ran out; nothing was recreated
    4  a build/recreate/verify command failed, or the recreate did not replace
       the container (the new code is therefore not running)

Environment fallbacks exist for every setting (``AGENT_RECREATE_*``); the
command line wins. ``decide()`` is pure -- no I/O, no clock -- so the rule can
be unit-tested and mutation-checked without a live deployment.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Defaults are neutral on purpose: every deployment-specific value arrives as
#: configuration. The console default matches this harness's own default bind
#: address; the docker command assumes a CLI already usable without a prefix.
CONSOLE = os.environ.get("AGENT_RECREATE_CONSOLE", "http://127.0.0.1:8787")
BASE_PATH = os.environ.get("AGENT_RECREATE_BASE_PATH", "/agent")
DOCKER = os.environ.get("AGENT_RECREATE_DOCKER", "docker")
IMAGE = os.environ.get("AGENT_RECREATE_IMAGE", "")
CONTAINER = os.environ.get("AGENT_RECREATE_CONTAINER", "")
BUILD_CONTEXT = os.environ.get("AGENT_RECREATE_BUILD_CONTEXT", "")
COMPOSE_DIR = os.environ.get("AGENT_RECREATE_COMPOSE_DIR", "")
COMPOSE_FILE = os.environ.get("AGENT_RECREATE_COMPOSE_FILE", "")
ENV_PREFIX = os.environ.get("AGENT_RECREATE_ENV_PREFIX", "AGENT_")
MOUNTS = [m for m in (os.environ.get("AGENT_RECREATE_MOUNTS") or "").split(",") if m]

#: How long to wait for a run in flight to end before giving up, and the grace
#: between the two probes taken immediately before the recreate. The grace
#: closes the race a real 2026-11-02 incident exposed: a single sample said
#: "no run in flight", the recreate went ahead, and a run that had started in
#: the gap was killed five minutes in.
DEFAULT_WAIT_SEC = 1500.0
CONFIRM_SEC = 5.0
POLL_SEC = 30.0


def parse_mounts(specs: Sequence[str]) -> Dict[str, str]:
    """Parse ``CONTAINER_PREFIX=HOST_PREFIX`` pairs into a mapping.

    A spec without ``=``, or with an empty side, is a configuration error:
    silently ignoring it would probe the WRONG filesystem and report every
    session missing, which reads as "no run has ever started".
    """
    mounts: Dict[str, str] = {}
    for spec in specs:
        container, sep, host = spec.partition("=")
        if not sep or not container or not host:
            raise ValueError("bad --mount %r: expected CONTAINER_PREFIX=HOST_PREFIX" % spec)
        mounts[container] = host
    return mounts


def host_path(path: str, mounts: Dict[str, str]) -> str:
    """Map a console-reported (container) path onto this host's filesystem.

    Longest matching prefix wins, so ``/data`` and ``/data/sessions`` can be
    mapped independently. ``posixpath``, never ``os.path``: these are container
    and Linux-host paths, POSIX on every machine -- ``os.path.join`` would emit
    backslashes if the guard were ever run or tested from a Windows
    workstation, and the resulting path would silently match no file. That bug
    was caught by a test before this module ever reached a host.
    """
    best = ""
    for prefix in mounts:
        if (path == prefix or path.startswith(prefix.rstrip("/") + "/")) \
                and len(prefix) > len(best):
            best = prefix
    if not best:
        return path
    remainder = path[len(best):].lstrip("/")
    return posixpath.join(mounts[best], remainder) if remainder else mounts[best]


def session_has_end(path: str) -> Tuple[bool, bool]:
    """Return ``(has_session_end, readable)`` for a session jsonl file.

    Substring match over raw lines, not a JSON parse: the file may end mid-line
    (that is exactly the killed-run case), and the question is only whether an
    end record was ever written.
    """
    if not os.path.exists(path):
        return False, False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"session_end"' in line:
                    return True, True
    except OSError:
        return False, False
    return False, True


def newest_session(runs: Dict[str, Any],
                   mounts: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """The most recently STARTED run row, plus the two facts about its file."""
    rows = [r for r in (runs.get("runs") or []) if r.get("file")]
    if not rows:
        return None
    row = max(rows, key=lambda r: str(r.get("started") or ""))
    host = host_path(
        posixpath.join(str(runs.get("sessions_dir") or ""), str(row["file"])),
        mounts or {})
    has_end, readable = session_has_end(host)
    return {
        "file": row.get("file"),
        "started": row.get("started"),
        "status": row.get("status"),
        "host_path": host,
        "readable": readable,
        "has_session_end": has_end,
    }


def decide(sched: Dict[str, Any],
           newest: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Decide whether a recreate is safe, from two independent facts.

    Pure -- no I/O and no clock -- so the rule can be unit-tested and
    mutation-checked. Returns ``("proceed" | "wait" | "refuse", reason)``.
    """
    if not sched.get("available"):
        return "refuse", ("the scheduler cannot answer (%s); an unknown must never "
                          "be read as idle" % (sched.get("reason") or "no reason given"))

    if sched.get("running"):
        return "wait", ("the scheduler reports a run in flight since %s (%ss ago)"
                        % (sched.get("in_progress_since"), sched.get("in_progress_sec")))

    # The scheduler says idle. Its answer is authoritative -- see the module
    # docstring -- so the session file corroborates, it does not veto.
    if not newest:
        return "proceed", ("scheduler idle; no session file to cross-check, so the "
                           "scheduler's own answer stands")

    if not newest.get("readable"):
        return "proceed", ("scheduler idle; the newest session (%s) could not be read, "
                           "so it corroborates nothing either way"
                           % newest.get("file"))

    if newest.get("has_session_end"):
        return "proceed", ("scheduler idle and the newest session ended cleanly (%s)"
                           % newest.get("file"))

    if newest.get("started") and newest.get("started") == sched.get("in_progress_since"):
        return "wait", ("the newest session (%s) IS the run the scheduler reports in "
                        "flight" % newest.get("file"))

    return "proceed", ("scheduler idle; the newest session (%s) has no session_end, but it "
                       "is not the live run -- a run an earlier recreate killed never gains "
                       "one, so treating it as a blocker would wait forever"
                       % newest.get("file"))


def describe_run(sched: Dict[str, Any],
                 newest: Optional[Dict[str, Any]]) -> str:
    """Name exactly what a forced recreate will destroy.

    ``--force`` is an escape hatch for a run that is not merely slow but is the
    thing blocking the schedule, so it is deliberately narrow: only "a run is
    in flight" can be forced past. That is also why it states the facts -- a
    command that silently kills a 75-minute mission is indistinguishable from
    the accidental kills this guard exists to prevent.
    """
    if not sched.get("running"):
        return "no run in flight"
    return ("run in flight since %s (%ss ago)" % (
        sched.get("in_progress_since"), sched.get("in_progress_sec"))
        + (", session file %s" % (newest or {}).get("file")
           if (newest or {}).get("file") else ""))


def line_log(*values: Any) -> None:
    """Log one line and FLUSH it.

    ``print`` block-buffers when stdout is not a tty, and this tool is meant to
    be run detached -- it can legitimately wait tens of minutes for a run in
    flight to end, so its normal invocation is
    ``nohup ... > guard.log 2>&1 &``. Unflushed, that produces an EMPTY log
    while the tool is working perfectly: an operator (or an agent) watching the
    file cannot tell waiting from hung, and "hung" gets killed, which is how a
    guard against killing runs ends up being killed itself. Observed live on
    2026-11-03: 0 bytes after three minutes, with the build finished and the
    wait loop polling happily.

    Subprocess output is separate: ``sh`` streams it through this function,
    which flushes per call, so a long build still shows progress.
    """
    print(*values, flush=True)


def fetch_json(console: str, base_path: str, path: str,
               timeout: float = 20.0) -> Dict[str, Any]:
    with urllib.request.urlopen(console.rstrip("/") + base_path.rstrip("/") + path,
                                timeout=timeout) as fh:
        return json.load(fh)


def probe(console: str, base_path: str,
          mounts: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any],
                                                            Optional[Dict[str, Any]]]:
    """Ask the scheduler, and gather the corroborating session facts."""
    runs = fetch_json(console, base_path, "/runs.json")
    return (runs.get("scheduler") or {}), newest_session(runs, mounts)


def wait_until_idle(probe_fn, wait_sec: float, log=line_log,
                    poll_sec: float = POLL_SEC) -> Tuple[str, str]:
    """Poll until the scheduler is idle, or give up. Never recreates."""
    deadline = time.monotonic() + wait_sec
    while True:
        sched, newest = probe_fn()
        action, reason = decide(sched, newest)
        log("[probe] %s -- %s" % (action.upper(), reason))
        if action != "wait":
            return action, reason
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "wait", ("still in flight after %.0fs; gave up WITHOUT recreating, "
                            "because a killed run is worse than a late fix" % wait_sec)
        time.sleep(min(poll_sec, max(1.0, remaining)))


def confirm_still_idle(probe_fn, log=line_log,
                       confirm_sec: float = CONFIRM_SEC) -> Tuple[str, str]:
    """Re-probe once after a grace, so a run starting in the gap is caught."""
    time.sleep(confirm_sec)
    sched, newest = probe_fn()
    action, reason = decide(sched, newest)
    log("[confirm] %s -- %s" % (action.upper(), reason))
    return action, reason


def sh(cmd: str, log=line_log, cwd: Optional[str] = None,
       quiet_out: bool = False) -> Tuple[int, str]:
    log("+ %s" % cmd)
    proc = subprocess.run(cmd, shell=True, cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = (proc.stdout or "").strip()
    if out and not quiet_out:
        log(out)
    return proc.returncode, out


#: Name fragments that mark an environment variable as a secret. A name-based
#: rule (not an allow-list of known variable names) because a deployment's
#: credentials arrive under provider-chosen names the guard cannot enumerate.
SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def mask_if_secret(name: str, value: str) -> str:
    """The variable's value, or a redaction marker if its NAME looks secret.

    Redacts on the NAME, never heuristics over the value: guessing from the
    value both over-redacts (a path containing "token") and -- worse -- misses
    a credential that looks like ordinary text.
    """
    upper = name.upper()
    if any(hint in upper for hint in SECRET_NAME_HINTS):
        return "<masked %d chars>" % len(value)
    return value


def container_started(docker: str, container: str,
                      log=line_log) -> Optional[str]:
    """The container's StartedAt stamp, or None if it cannot be read."""
    rc, out = sh('%s inspect %s --format "{{.State.StartedAt}}"' % (docker, container),
                 log=log)
    return out.strip('"') if rc == 0 and out else None


def verify(docker: str, container: str, before: Optional[str],
           env_prefix: str = ENV_PREFIX, log=line_log) -> int:
    """Prove the running container is the NEW one and read its env from inside.

    Verification is from INSIDE the running container -- not from a build log,
    not from a receipt. A build log proves an image was produced; it cannot
    prove the container anyone talks to was replaced.
    """
    after = container_started(docker, container, log=log)
    if after is None:
        log("VERIFY FAILED: cannot inspect %s" % container)
        return 4
    if before is not None and after == before:
        log("VERIFY FAILED: running container predates the recreate "
            "(StartedAt %s unchanged) -- the new code is NOT running" % after)
        return 4
    log("container replaced: StartedAt %s (was %s)" % (after, before))
    # quiet_out: `env` prints EVERY variable including credentials (provider
    # API keys are ordinary env names). The raw dump must never reach a log --
    # an earlier version of this function streamed it, and watched a real
    # provider key land in the operator's transcript. The command line IS
    # logged (that is not secret); only the body is withheld, and what gets
    # re-reported below is masked on the variable NAME.
    rc, out = sh("%s exec %s env" % (docker, container), log=log, quiet_out=True)
    if rc != 0:
        log("VERIFY FAILED: could not exec into %s" % container)
        return 4
    seen = {}
    for line in out.splitlines():
        if line.startswith(env_prefix) and "=" in line:
            key, _, val = line.partition("=")
            seen[key.strip()] = val.strip()
    for key in sorted(seen):
        log("  in-container %s=%s" % (key, mask_if_secret(key, seen[key])))
    if not seen:
        log("  (no %s* variables in the container env)" % env_prefix)
    return 0


def _require(args: argparse.Namespace, names: List[str]) -> Optional[str]:
    missing = [n for n in names if not getattr(args, n)]
    if missing:
        return "missing required setting(s): %s" % ", ".join("--" + n.replace("_", "-")
                                                            for n in missing)
    return None


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="rebuild the image (always safe; not gated)")
    ap.add_argument("--apply", action="store_true",
                    help="recreate the container -- GATED on no run being in flight")
    ap.add_argument("--wait-sec", type=float, default=DEFAULT_WAIT_SEC,
                    help="how long to wait for a run in flight to end (default %(default)s)")
    ap.add_argument("--confirm-sec", type=float, default=CONFIRM_SEC,
                    help="grace before the destructive re-probe (default %(default)s)")
    ap.add_argument("--console", default=CONSOLE, help="console origin (default %(default)s)")
    ap.add_argument("--base-path", default=BASE_PATH,
                    help="console base path (default %(default)s)")
    ap.add_argument("--mount", action="append", default=None, metavar="CONTAINER=HOST",
                    help="map console-reported session paths onto the host filesystem "
                         "(repeatable; env AGENT_RECREATE_MOUNTS is a comma-separated list)")
    ap.add_argument("--docker", default=DOCKER,
                    help="container CLI command, prefix included (default %(default)s)")
    ap.add_argument("--image", default=IMAGE, help="image name:tag to build")
    ap.add_argument("--build-context", default=BUILD_CONTEXT, help="directory to build the image from")
    ap.add_argument("--compose-dir", default=COMPOSE_DIR, help="directory holding the compose file")
    ap.add_argument("--compose-file", default=COMPOSE_FILE, help="compose file name")
    ap.add_argument("--container", default=CONTAINER, help="container name to recreate and verify")
    ap.add_argument("--recreate-cmd", default=os.environ.get("AGENT_RECREATE_CMD", ""),
                    help="override the whole recreate command (non-compose runtimes)")
    ap.add_argument("--env-prefix", default=ENV_PREFIX,
                    help="env vars to report from inside the container (default %(default)s)")
    ap.add_argument("--json", action="store_true", help="print the probe as JSON and exit")
    ap.add_argument("--force", action="store_true",
                    help="recreate even with a run in flight, DESTROYING it. "
                         "Refuses if the scheduler cannot be reached, because "
                         "forcing must never mean acting blind about what is "
                         "about to be killed")
    args = ap.parse_args(argv)

    log = (lambda *a: None) if args.json else line_log

    specs: List[str] = list(args.mount or []) or list(MOUNTS)
    try:
        mounts = parse_mounts(specs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def _probe():
        return probe(args.console, args.base_path, mounts)

    if args.json:
        try:
            sched, newest = _probe()
        except Exception as exc:  # noqa: BLE001 -- the JSON answer must not be a traceback
            print(json.dumps({"action": "refuse",
                              "reason": "probe failed: %s: %s" % (type(exc).__name__, exc),
                              "scheduler": None, "newest_session": None}))
            return 2
        action, reason = decide(sched, newest)
        result = {"action": action, "reason": reason, "scheduler": sched,
                  "newest_session": newest}
        if action == "wait" and getattr(args, "force", False):
            result["forced"] = {
                "would_destroy": describe_run(sched, newest),
                "scheduler_available": bool(sched.get("available")),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return {"proceed": 0, "refuse": 2, "wait": 3}[action]

    usage_error = None
    if args.build:
        usage_error = _require(args, ["image"]) or \
            (None if args.build_context else "missing required setting(s): --build-context")
    if args.apply and not usage_error:
        usage_error = _require(args, ["container"])
        if not args.recreate_cmd and not usage_error:
            usage_error = _require(args, ["compose_dir", "compose_file"])
    if usage_error:
        print(usage_error, file=sys.stderr)
        return 1

    try:
        sched, newest = _probe()
    except Exception as exc:  # noqa: BLE001
        log("REFUSING: probe failed (%s: %s); an unknown must never be read as idle"
            % (type(exc).__name__, exc))
        return 2
    action, reason = decide(sched, newest)
    log("[probe] %s -- %s" % (action.upper(), reason))

    if action == "refuse" and args.apply:
        log("REFUSING to recreate: %s" % reason)
        return 2

    # Building is safe whether or not a run is live: it replaces no container.
    if args.build:
        rc, _ = sh("%s build -t %s ." % (args.docker, args.image), log=log,
                   cwd=args.build_context)
        if rc != 0:
            log("BUILD FAILED (rc=%d); not recreating" % rc)
            return 4

    if not args.apply:
        log("check-only: nothing built or recreated")
        return 0

    if action == "wait":
        if getattr(args, "force", False):
            #: Only "a run is in flight" can be forced. If the scheduler cannot
            #: answer we do not know WHAT is running, and forcing through that
            #: would be acting blind -- so an unavailable scheduler still stops
            #: us here.
            if not sched.get("available"):
                log("REFUSING to force: %s" % reason)
                return 2
            victim = describe_run(sched, newest)
            log("[probe] FORCE -- proceeding despite an in-flight run; "
                "--force was given. It will DESTROY: %s" % victim)
            action = "proceed"
            reason = "forced past an in-flight run (%s)" % victim
        else:
            action, reason = wait_until_idle(_probe, args.wait_sec, log=log)
            if action == "refuse":
                log("REFUSING to recreate: %s" % reason)
                return 2
            if action == "wait":
                log("NOT recreating: %s" % reason)
                return 3

    if not getattr(args, "force", False):
        # Close the race: re-ask once, immediately before the destructive step.
        # Skipped under --force because the run we would wait for is exactly the
        # one we have already decided to destroy, so re-asking can only ever
        # return "wait".
        action, reason = confirm_still_idle(_probe, log=log,
                                           confirm_sec=args.confirm_sec)
        if action == "refuse":
            log("REFUSING to recreate: %s" % reason)
            return 2
        if action == "wait":
            log("NOT recreating: %s" % reason)
            return 3

    before = container_started(args.docker, args.container, log=log)
    if args.recreate_cmd:
        rc, _ = sh(args.recreate_cmd, log=log)
    else:
        rc, _ = sh("%s compose -f %s up -d" % (args.docker, args.compose_file),
                   log=log, cwd=args.compose_dir)
    if rc != 0:
        log("RECREATE FAILED (rc=%d)" % rc)
        return 4

    return verify(args.docker, args.container, before,
                  env_prefix=args.env_prefix, log=log)


if __name__ == "__main__":
    sys.exit(main())
