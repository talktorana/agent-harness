#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the recreate guard's confirmation-gap distinction.

The bug these pin down was found while shipping a real fix at 06:30 on
2026-11-04. The armed recreate was waiting for a run that had been in flight 43
minutes; the job's cadence is 60 minutes but its runs take 80+, so when that run
finally ends the job is ALREADY OVERDUE and the scheduler fires the next mission
within its own 30s poll. The guard's ``confirm_still_idle`` re-probes after a 5s
grace -- so it would very plausibly see that BRAND-NEW run and return "wait",
exit 3, and ship nothing.

The refusal itself is correct: `up -d` kills whatever is running, and killing a
mission 5 seconds old is just as expensive as killing one 40 minutes old. What
was wrong is the REASON. It named only the new run's timestamp, so a caller
reads it as "my wait budget was still too short", raises ``--wait-sec``, re-arms
-- and races the same window again. The image is rebuilt a second time and the
fix still does not ship. That is the third instance of built-but-not-shipped
that this deployment has had to diagnose from a log line.

The distinction is ``in_progress_since``: same stamp means the run you waited
for is still going (budget really was short); a different stamp means it ENDED
and the scheduler moved on (the wait succeeded -- re-arm, don't raise).

Nothing here writes to the scheduler's state file, opens a console port, or
touches Docker.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import recreate_guard as rg      # noqa: E402


WAITED = "2026-11-04T05:49:16"
NEWER = "2026-11-04T07:19:02"


class TestConfirmGapDistinction(unittest.TestCase):
    """The two "wait" outcomes must read differently, and mean differently."""

    def test_same_run_gives_no_different_run_clause(self):
        """A budget that really is too short must still say only that."""
        sched = {"available": True, "running": True,
                 "in_progress_since": WAITED, "in_progress_sec": 4500.0}
        lines = []
        action, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lines.append,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertEqual(action, "wait")
        self.assertIn("in flight", reason)
        self.assertNotIn("DIFFERENT run", reason,
                         "the same run must not be reported as a new one: that"
                         " would tell an operator their budget was fine when it"
                         " was not, and the fix silently never ships")
        self.assertEqual(len(lines), 1)
        self.assertIn("[confirm]", lines[0])
        self.assertIn(WAITED, lines[0], "the log must name the run it saw")

    def test_different_run_is_reported_as_such(self):
        """The waited-for run ended; a new one took the slot."""
        sched = {"available": True, "running": True,
                 "in_progress_since": NEWER, "in_progress_sec": 3.0}
        action, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertEqual(action, "wait",
                         "it must STILL refuse: killing a fresh mission is"
                         " exactly what this guard exists to prevent")
        self.assertIn("DIFFERENT run", reason)

    def test_different_run_names_both_timestamps(self):
        sched = {"available": True, "running": True,
                 "in_progress_since": NEWER, "in_progress_sec": 3.0}
        _, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertIn(WAITED, reason, "must name the run that finished")
        self.assertIn(NEWER, reason, "must name the run that replaced it")

    def test_different_run_says_the_wait_succeeded(self):
        """The whole point: it must correct the misdiagnosis."""
        sched = {"available": True, "running": True,
                 "in_progress_since": NEWER, "in_progress_sec": 3.0}
        _, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertIn("finished", reason)
        self.assertIn("wait succeeded", reason)
        self.assertIn("re-arm", reason)

    def test_different_run_warns_against_raising_the_budget(self):
        """Raising --wait-sec here is the mistake the old reason invites."""
        sched = {"available": True, "running": True,
                 "in_progress_since": NEWER, "in_progress_sec": 3.0}
        _, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertIn("--wait-sec", reason)
        self.assertTrue("--wait-sec, the budget was not the problem" in reason
                        or "budget was not the problem" in reason,
                        "must say the budget was NOT the problem, or the operator"
                        " raises it again and races the same window")

    def test_proceed_is_unaffected_by_the_new_argument(self):
        """No run at all: the normal success path must not gain a clause."""
        sched = {"available": True, "running": False}
        action, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertEqual(action, "proceed")
        self.assertNotIn("DIFFERENT run", reason)

    def test_no_waited_from_adds_nothing(self):
        """Back-compat: existing callers that pass nothing behave as before."""
        for sched, want in (({"available": True, "running": False}, "proceed"),
                            ({"available": True, "running": True,
                              "in_progress_since": NEWER}, "wait")):
            action, reason = rg.confirm_still_idle(
                lambda s=sched: (s, None), log=lambda *a, **k: None,
                confirm_sec=0.0)
            self.assertEqual(action, want)
            self.assertNotIn("DIFFERENT run", reason)

    def test_refuse_is_unaffected(self):
        """A scheduler we cannot ask stays a refusal, clause-free."""
        sched = {"available": False, "error": "boom"}
        action, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertEqual(action, "refuse")
        self.assertNotIn("DIFFERENT run", reason)

    def test_missing_stamp_does_not_claim_a_new_run(self):
        """running with no in_progress_since: unknown, so say nothing extra."""
        sched = {"available": True, "running": True}
        action, reason = rg.confirm_still_idle(
            lambda: (sched, None), log=lambda *a, **k: None,
            confirm_sec=0.0, waited_from=WAITED)
        self.assertEqual(action, "wait")
        self.assertNotIn("DIFFERENT run", reason,
                         "without a stamp we do not know which run it is;"
                         " asserting 'the one you waited for finished' would be"
                         " a guess presented as a fact")


class TestWiring(unittest.TestCase):
    """The distinction is worthless if main() never passes waited_from."""

    SRC = (pathlib.Path(__file__).resolve().parent
           / "recreate_guard.py").read_text(encoding="utf-8")

    def test_main_captures_waited_from(self):
        self.assertIn("waited_from: Optional[str] = (sched.get(\"in_progress_since\")",
                      self.SRC)

    def test_main_only_captures_when_running(self):
        self.assertIn("if sched.get(\"running\") else None)", self.SRC,
                      "capturing a stamp while idle would make every later wait"
                      " look like a different run")

    def test_main_passes_it_to_confirm(self):
        self.assertIn("waited_from=waited_from", self.SRC)

    def test_signature_defaults_to_none(self):
        self.assertIn("waited_from: Optional[str] = None", self.SRC,
                      "must stay optional so existing callers and tests work")


if __name__ == "__main__":
    unittest.main(verbosity=2)
