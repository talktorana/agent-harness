"""The scheduler LOOP must name an unrestored cadence hold -- nothing else can.

``preflight.cadence_override_attention`` runs INSIDE a mission. A console
override large enough to stop missions therefore stops the only check that
would name it. That is not a hypothetical:

  * 2026-11-21 02:08Z: a pass left ``every_min=4500`` (declared 45) and
    reported SUCCESS. 62 hours of silence, ~83 missed fires.
  * A preflight guard for exactly this shipped on 2026-11-23 -- and the SAME
    override recurred at 22:05Z the same night, because no mission ran to
    execute the guard.

The daemon loop runs whether or not missions do, so the detection belongs
there. These tests drive the REAL ``daemon()`` for a bounded number of polls
(``time.sleep`` raises a sentinel) and read what it actually printed, so they
prove the line reaches the log through the loop itself rather than only that a
helper returns a string.

Pinned:
  * an unrestored override prints ONE cadence-hold line, then stays quiet
    (rate-limited like the operator-hold line, never a log flood);
  * an operator hold SUPPRESSES the cadence line -- that pause is deliberate;
  * a healthy cadence, or an override within the tuning band, prints nothing;
  * the loop keeps scheduling normally while reporting -- a diagnostic must
    never be the reason missions stop.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import run_settings  # noqa: E402
import scheduler  # noqa: E402
from test_scheduler import SchedulerPaths, _job, _write  # noqa: E402


class _StopLoop(BaseException):
    """Raised from the patched sleep to end the daemon after N polls."""


def _run_daemon(polls: int) -> str:
    """Run scheduler.daemon() for exactly ``polls`` iterations; return stdout."""
    buf = io.StringIO()
    remaining = {"n": polls}

    def fake_sleep(_sec):
        if remaining["n"] <= 0:
            raise _StopLoop()
        remaining["n"] -= 1

    with redirect_stdout(buf), mock.patch.object(scheduler.time, "sleep",
                                                 side_effect=fake_sleep):
        try:
            scheduler.daemon(poll_sec=0)
        except _StopLoop:
            pass
    return buf.getvalue()


class CadenceHoldInTheLoopTest(SchedulerPaths):
    def setUp(self) -> None:
        super().setUp()
        self.settings = self.dir / "run_settings.json"
        self._saved_settings = run_settings.SETTINGS_FILE
        run_settings.SETTINGS_FILE = self.settings
        #: A job far from due, so the loop's own work is quiet and only the
        #: cadence line can appear.
        _write(self.declaration, [_job(every_min=45,
                                       last_run="2099-01-01T00:00:00")])

    def tearDown(self) -> None:
        run_settings.SETTINGS_FILE = self._saved_settings
        super().tearDown()

    def _override(self, every_min: int) -> None:
        run_settings.save(every_min=every_min)

    def test_an_unrestored_override_is_named_in_the_daemon_log(self):
        self._override(4500)
        out = _run_daemon(polls=4)
        self.assertIn("[scheduler] CADENCE HOLD", out)
        self.assertIn("4500", out)
        self.assertIn("missions will not fire", out)

    def test_it_is_printed_once_not_every_poll(self):
        self._override(4500)
        out = _run_daemon(polls=12)
        self.assertEqual(out.count("[scheduler] CADENCE HOLD"), 1, out)

    def test_a_healthy_cadence_prints_nothing(self):
        out = _run_daemon(polls=4)
        self.assertNotIn("CADENCE HOLD", out)

    def test_an_override_within_the_tuning_band_is_quiet(self):
        # 60 vs a declared 45 is an operator choice, not a hold.
        self._override(60)
        out = _run_daemon(polls=4)
        self.assertNotIn("CADENCE HOLD", out)

    def test_an_operator_hold_suppresses_the_cadence_line(self):
        # A deliberate pause explains the silence; it is not a defect.
        self._override(4500)
        self.hold.write_text('{"by": "owner"}', encoding="utf-8")
        out = _run_daemon(polls=4)
        self.assertNotIn("CADENCE HOLD", out)
        self.assertIn("[scheduler] PAUSED by hold file", out)

    def test_the_loop_keeps_scheduling_while_a_hold_is_reported(self):
        # The diagnostic must never be the reason missions stop: a job that IS
        # due still runs on the very next poll.
        self._override(4500)
        self.stub_task()
        _write(self.declaration, [_job(every_min=45)])  # never run => due now
        out = _run_daemon(polls=2)
        self.assertIn("[scheduler] CADENCE HOLD", out)
        self.assertEqual(len(self.ran), 1, self.ran)

    def test_the_line_names_the_restore_target(self):
        # Actionable without a second investigation: it says what to set.
        self._override(4500)
        out = _run_daemon(polls=2)
        self.assertIn("restore every_min to 45", out)


class CadenceHoldHelperTest(SchedulerPaths):
    """The helper alone, including the paths that must never raise."""

    def setUp(self) -> None:
        super().setUp()
        self.settings = self.dir / "run_settings.json"
        self._saved_settings = run_settings.SETTINGS_FILE
        run_settings.SETTINGS_FILE = self.settings
        _write(self.declaration, [_job(every_min=45)])

    def tearDown(self) -> None:
        run_settings.SETTINGS_FILE = self._saved_settings
        super().tearDown()

    def test_healthy_is_empty_string(self):
        self.assertEqual(scheduler._cadence_hold_message(), "")

    def test_an_unreadable_schedule_does_not_raise(self):
        self.declaration.write_text("{not json", encoding="utf-8")
        run_settings.save(every_min=4500)
        # Either names the hold from the declared-less path or reports that
        # the check failed -- but it returns a string, it never raises.
        self.assertIsInstance(scheduler._cadence_hold_message(), str)

    def test_a_broken_preflight_is_reported_not_raised(self):
        run_settings.save(every_min=4500)
        with mock.patch("preflight.cadence_override_attention",
                        side_effect=RuntimeError("boom")):
            msg = scheduler._cadence_hold_message()
        self.assertIn("cadence-hold check failed", msg)
        self.assertIn("RuntimeError", msg)


if __name__ == "__main__":
    unittest.main()
