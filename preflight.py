#!/usr/bin/env python3
"""One read-only orientation snapshot for a scheduled monitoring mission.

The generic harness contributes schedule, live settings and the previous
SCHEDULED delivery. Application health belongs to the deployment, supplied as
``AGENT_PREFLIGHT_COMMAND``: one command, executed WITHOUT a shell, that prints
JSON with ``fast_path_eligible`` (bool) and ``attention_required`` (strings).

A clean result lets a mission take a short healthy path instead of spending
 dozens of model turns rediscovering settled facts. Any attention item selects
the normal investigation path; this module never suppresses a finding and never
writes application or runtime state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import shutil
import subprocess
from typing import Any, Callable
import urllib.parse
import urllib.request

import config

JsonFetcher = Callable[[str], Any]
Collector = Callable[[], tuple[dict[str, Any], list[str]]]
MAX_PROJECT_OUTPUT_CHARS = 100_000


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def _console_data(fetch: JsonFetcher) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {}
    errors: list[str] = []
    for name, path in (("schedule", "/agent/schedule"),
                       ("settings", "/agent/settings"),
                       ("runs", "/agent/runs.json")):
        try:
            result[name] = fetch(config.PREFLIGHT_CONSOLE_URL + path)
        except Exception as exc:  # an unavailable fact is attention, not a crash
            errors.append(f"console {name}: {type(exc).__name__}: {exc}")
    return result, errors


def _run_id(row: dict[str, Any]) -> str:
    explicit = row.get("session_id") or row.get("id")
    if explicit:
        return str(explicit)
    from pathlib import Path
    filename = Path(str(row.get("file") or "")).stem
    return filename.rsplit("_", 1)[-1] if filename else ""


def previous_scheduled_delivery(console: dict[str, Any],
                                fetch: JsonFetcher) -> tuple[dict[str, Any], list[str]]:
    """Latest completed timed delivery, never a console/chat smoke test."""
    runs = console.get("runs") or {}
    rows = runs.get("runs") if isinstance(runs, dict) else []
    scheduled = [row for row in rows or [] if isinstance(row, dict)
                 and row.get("origin") == "schedule"
                 and row.get("status") != "running"]
    scheduled.sort(key=lambda row: str(row.get("started") or row.get("file") or ""),
                   reverse=True)
    if not scheduled:
        return {}, ["no completed scheduled delivery with an answer is available"]
    row = scheduled[0]
    answer = row.get("answer") or ""
    errors: list[str] = []
    run_id = _run_id(row)
    if not answer and run_id:
        try:
            detail = fetch(config.PREFLIGHT_CONSOLE_URL + "/agent/run/"
                           + urllib.parse.quote(run_id, safe=""))
            answer = detail.get("answer") or ""
        except Exception as exc:
            errors.append(f"previous scheduled answer: {type(exc).__name__}: {exc}")
    summary = {key: row.get(key) for key in
               ("session_id", "started", "finished", "status", "steps", "file")
               if row.get(key) is not None}
    summary["answer"] = str(answer)[:max(0, config.PREFLIGHT_ANSWER_CHARS)]
    if not str(answer).strip():
        errors.append("the latest completed scheduled delivery has no answer")
    return summary, errors


def project_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Reject a collector response that could otherwise invent false quiet."""
    if not payload:
        return []
    errors: list[str] = []
    if not isinstance(payload.get("fast_path_eligible"), bool):
        errors.append("project health collector needs boolean fast_path_eligible")
    attention = payload.get("attention_required")
    if not isinstance(attention, list) or any(not isinstance(item, str)
                                              for item in attention):
        errors.append("project health collector needs attention_required as a string list")
    if payload.get("fast_path_eligible") is False and not attention:
        errors.append("project health collector reports fast_path_eligible=false "
                      "without naming attention_required evidence")
    return errors


def run_project_collector() -> tuple[dict[str, Any], list[str]]:
    """Run the configured application-health command without a shell."""
    command = config.PREFLIGHT_COMMAND
    if not command:
        if config.PREFLIGHT_REQUIRE_COMMAND:
            return {}, ["project health collector is not configured "
                        "(AGENT_PREFLIGHT_COMMAND)"]
        return {}, []
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return {}, [f"project health collector command is invalid: {exc}"]
    if not argv:
        return {}, ["project health collector command is empty"]
    try:
        proc = subprocess.run(
            argv, cwd=str(config.PROJECT_ROOT), shell=False,
            capture_output=True, text=True, timeout=config.PREFLIGHT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return {}, [f"project health collector timed out after "
                    f"{config.PREFLIGHT_TIMEOUT_SEC:g}s"]
    except Exception as exc:
        return {}, [f"project health collector could not start: "
                    f"{type(exc).__name__}: {exc}"]
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()[-4000:]
        return {}, [f"project health collector exited {proc.returncode}: {detail}"]
    if len(proc.stdout) > MAX_PROJECT_OUTPUT_CHARS:
        return {}, [f"project health collector output exceeds "
                    f"{MAX_PROJECT_OUTPUT_CHARS} characters"]
    try:
        payload = json.loads(proc.stdout)
    except ValueError as exc:
        return {}, [f"project health collector did not print one JSON object: {exc}"]
    if not isinstance(payload, dict):
        return {}, ["project health collector JSON must be an object"]
    return payload, []


#: A console cadence override at or above this multiple of the DECLARED interval
#: is a hold on the mission, not a tuning choice.
CADENCE_HOLD_RATIO = 4.0


def _positive_number(value: Any) -> float | None:
    """A usable positive number, or None -- never a guess and never a zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def cadence_override_attention(settings: Any, schedule: Any) -> list[str]:
    """Name a console cadence override that is silently holding the mission.

    ``overdue`` structurally cannot catch this: it compares the elapsed gap to
    the override itself, so raising the interval moves the goalposts and a
    stopped mission reads healthy. Only the DECLARED interval is independent of
    the override, so that is the comparison that can see it.

    Measured 2026-11-23: a pass POSTed ``every_min=4500`` against a declared 45
    (100x) at 02:08:45 -- one minute into its own run -- and never restored it,
    then reported SUCCESS. The mission stopped for 62 hours while ``overdue``
    stayed False, because 62h is less than the 75h it had just invented.

    An operator hold explains a pause and is never reported as a defect.
    """
    if not isinstance(settings, dict):
        return []
    if settings.get("every_min_source") != "console":
        return []                       # a declared cadence is owner intent
    hold = schedule.get("hold") if isinstance(schedule, dict) else None
    if isinstance(hold, dict) and hold.get("held"):
        return []                       # a real operator pause: deliberate
    live = _positive_number(settings.get("every_min"))
    if live is None:
        return []
    declared = _positive_number(settings.get("declared_every_min"))
    if declared is None:
        return ["the console cadence override cannot be compared: "
                "declared_every_min is not readable, so a hold is not detectable"]
    ratio = live / declared
    if ratio < CADENCE_HOLD_RATIO:
        return []
    saved = settings.get("saved_at") or "unknown time"
    return [f"console cadence override is holding the mission: every_min={live:g} "
            f"against declared {declared:g} ({ratio:.0f}x, saved {saved}); no "
            f"operator hold is present, so this is an unrestored hold and not a "
            f"pause -- restore every_min to {declared:g} and verify next_due"]


def build_snapshot(*, now: datetime | None = None,
                   fetch: JsonFetcher = fetch_json,
                   collector: Collector = run_project_collector,
                   include_console: bool = True) -> dict[str, Any]:
    """Merge generic orientation and project health into one bounded decision."""
    now = now or datetime.now(timezone.utc)
    console: dict[str, Any] = {}
    previous: dict[str, Any] = {}
    attention: list[str] = []
    if include_console:
        console, errors = _console_data(fetch)
        attention.extend(errors)
        previous, errors = previous_scheduled_delivery(console, fetch)
        attention.extend(errors)
    project, errors = collector()
    attention.extend(errors)
    attention.extend(project_contract_errors(project))
    project_attention = project.get("attention_required") or []
    if isinstance(project_attention, list):
        attention.extend(str(item) for item in project_attention)

    jobs = (console.get("schedule") or {}).get("jobs") or []
    job = jobs[0] if jobs and isinstance(jobs[0], dict) else {}
    if job.get("last_ok") is False:
        attention.append("the previous scheduled run reports last_ok=false")
    schedule_keys = ("id", "every_min", "last_run", "finished", "last_ok", "next_due",
                     "seconds_until", "running", "in_progress_since", "in_progress_sec",
                     "duration_sec", "overdue_reason")
    settings = console.get("settings") or {}
    attention.extend(cadence_override_attention(settings, console.get("schedule")))
    try:
        usage = shutil.disk_usage(config.PROJECT_ROOT)
        storage = {"free_gb": round(usage.free / 1e9, 2),
                   "used_percent": round(100 * usage.used / usage.total, 1)}
    except OSError as exc:
        storage = {"error": f"{type(exc).__name__}: {exc}"}
        attention.append("project storage is not measurable")
    return {
        "generated_at": now.isoformat(),
        "fast_path_eligible": not attention,
        "attention_required": attention,
        "schedule": {key: job.get(key) for key in schedule_keys if key in job},
        #: declared_every_min and saved_at are the ONLY independent evidence of
        #: an unrestored cadence hold; without them the projection shows the
        #: override with nothing to compare it to.
        "settings": {key: settings.get(key) for key in
                     ("max_steps", "max_steps_source", "every_min", "every_min_source",
                      "declared_every_min", "saved_at")
                     if isinstance(settings, dict) and key in settings},
        "previous_scheduled_delivery": previous,
        "project_health": project,
        "storage": storage,
        "scope_note": "Read-only orientation. Project health evidence comes from "
                      "the deployment-configured collector; absence is not proof of health.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-console", action="store_true",
                        help="Skip console endpoints (diagnostics only)")
    args = parser.parse_args()
    snapshot = build_snapshot(include_console=not args.no_console)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
