#!/usr/bin/env python3
"""Generate schedule.json from a canonical mission text.

The harness scheduler takes a plain ``message`` string per job, so a mission
brief has to live inside JSON. Embedding a long prompt by hand is how the two
copies drift apart: the reviewed brief says one thing and the schedule runs
another. So ``mission.md`` is the single source, this script is the only thing
that serialises it, and the generated file is committed so it can be reviewed
in a diff.

Everything after the ``---`` divider in the mission file is the prompt; the
part above it is documentation for whoever reads the file in a repo browser.

Usage:
    python3 make_schedule.py                  # write schedule.json
    python3 make_schedule.py --check           # verify, do not write (CI)
    python3 make_schedule.py --enable          # opt the mission in
    python3 make_schedule.py --every-min 30    # set the cadence
    python3 make_schedule.py --out other.json  # write elsewhere
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

#: The canonical mission text. Documentation above the divider, prompt below.
MISSION_FILE = Path(__file__).resolve().parent / "mission.md"
SCHEDULE_FILE = Path(__file__).resolve().parent / "schedule.json"

#: The mission runs on this cadence. A cadence shorter than the work it
#: schedules is a cadence that is fiction: every report of "next due" would be
#: wrong and the page would show a job permanently overdue. Declaring the
#: interval long enough to contain the work is the honest number.
MISSION_EVERY_MIN = 45


def read_mission(path: Path = MISSION_FILE) -> str:
    """The mission prompt: everything after the `---` divider."""
    if not path.is_file():
        raise FileNotFoundError(f"mission file missing: {path}")
    text = path.read_text(encoding="utf-8")
    if "\n---\n" not in text:
        raise ValueError(
            f"{path} has no '---' divider separating docs from the mission "
            f"text; refusing to embed the whole document.")
    body = text.split("\n---\n", 1)[1]
    mission = body.strip()
    # The mission must still be recognisable as a brief, not a stub that
    # survived some editing accident.
    for probe in ("MISSION:", "TERMINAL SAFELIGHTS", "OUTPUT FORMAT"):
        if probe not in mission:
            raise ValueError(f"mission text lost {probe!r} -- refusing to "
                             f"schedule a truncated brief")
    return mission


def build(mission: str, *, enabled: bool = False,
          every_min: int = MISSION_EVERY_MIN) -> Dict[str, Any]:
    """The schedule document.

    ``enabled`` defaults to False ON PURPOSE. Shipping an enabled job means
    the agent starts running missions the moment the container starts. Enable
    it deliberately: --enable, or the console's enable toggle.
    """
    jobs: List[Dict[str, Any]] = [{
        "id": "monitoring-pass",
        "every_min": every_min,
        "enabled": bool(enabled),
        "message": mission,
    }]
    return {
        "_comment": [
            "GENERATED from mission.md by make_schedule.py.",
            "Edit the mission file and regenerate; do not hand-edit this JSON.",
            "The mission text is embedded verbatim so the scheduler and the",
            "reviewed brief cannot drift apart.",
            "",
            "Ships disabled: enable with --enable or the console toggle.",
            "This file must stay valid JSON -- the scheduler parses it with",
            "json.loads, so no comments outside _comment.",
        ],
        "jobs": jobs,
    }


def serialise(doc: Dict[str, Any]) -> str:
    """Pure JSON. A `#` header would make this file unreadable to json.loads,
    which is exactly how the scheduler loads a schedule, so the "do not
    hand-edit" note lives in ``_comment``, where it survives parsing."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="Verify the committed file is current; do not write.")
    parser.add_argument("--enable", action="store_true",
                        help="Write the schedule with the mission enabled.")
    parser.add_argument("--every-min", type=int, default=MISSION_EVERY_MIN)
    parser.add_argument("--out", default=str(SCHEDULE_FILE))
    args = parser.parse_args(argv)

    try:
        mission = read_mission()
        doc = build(mission, enabled=args.enable, every_min=args.every_min)
        text = serialise(doc)
        out = Path(args.out)
        if args.check:
            if not out.is_file():
                print(f"FAIL: {out} does not exist; run make_schedule.py",
                      file=sys.stderr)
                return 1
            existing = out.read_text(encoding="utf-8")
            if existing == text:
                print(f"OK: {out} is current ({len(text)} bytes)")
                return 0
            print(f"FAIL: {out} is stale; run make_schedule.py",
                  file=sys.stderr)
            return 1
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}: {len(text)} bytes, "
              f"every_min={args.every_min}, enabled={args.enable}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
