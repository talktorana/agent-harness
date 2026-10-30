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
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

_LOCK = threading.Lock()


def _load() -> Dict[str, Any]:
    p = Path(config.SCHEDULE_FILE)
    if not p.is_file():
        return {"jobs": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"jobs": []}
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        return {"jobs": []}
    return data


def _save(data: Dict[str, Any]) -> None:
    p = Path(config.SCHEDULE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _due(job: Dict[str, Any], now: datetime) -> bool:
    if not job.get("enabled", True):
        return False
    last = job.get("last_run")
    every = job.get("every_min")
    if every:
        if last is None:
            return True
        return (now - datetime.fromisoformat(last)).total_seconds() \
            >= float(every) * 60
    at = job.get("at")
    if at:
        hh, mm = (int(x) for x in str(at).split(":")[:2])
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        done_today = (job.get("last_run") or "")[:10] == now.date().isoformat()
        return now >= target and not done_today
    return False


def run_due_jobs(verbose: bool = False) -> List[Dict[str, Any]]:
    """Run every job that is due NOW. Returns one record per executed job."""
    from agent import run_task

    ran: List[Dict[str, Any]] = []
    with _LOCK:
        data = _load()
        now = datetime.now()
        due = [j for j in data.get("jobs", []) if _due(j, now)]
        if not due:
            return ran
        for job in due:
            job["last_run"] = now.isoformat(timespec="seconds")
        _save(data)
    for job in due:
        record = {"id": job.get("id"), "started": datetime.now().isoformat()}
        try:
            record["answer"] = run_task(job.get("message", ""),
                                        verbose=verbose)
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished"] = datetime.now().isoformat()
        ran.append(record)
    return ran


def daemon(poll_sec: float = 30.0) -> None:
    """Block forever, running due jobs. One bad job never kills the loop."""
    print(f"[scheduler] watching {config.SCHEDULE_FILE} "
          f"every {poll_sec:.0f}s", flush=True)
    while True:
        try:
            for record in run_due_jobs():
                state = "ok" if record.get("ok") else f"FAILED: " \
                    f"{record.get('error')}"
                print(f"[scheduler] job {record.get('id')}: {state}",
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[scheduler] loop error: {exc}", flush=True)
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="Run due jobs once and exit.")
    parser.add_argument("--list", action="store_true",
                        help="Show the schedule and exit.")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(_load(), indent=2))
        return 0
    if args.once:
        records = run_due_jobs(verbose=True)
        print(json.dumps(records, indent=2, default=str)
              if records else "(nothing due)")
        return 0
    daemon(args.poll_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
