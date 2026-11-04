"""The guard's wait must be able to contain the run it waits for.

``recreate_guard`` refuses to recreate while a mission is in flight, which is the
behaviour that stopped this deployment killing its own passes. But "refuse and
wait" carries a second failure mode that is quieter and, over a week, more
expensive: the WAIT is a fixed budget supplied by the caller, while the run it
waits for is bounded by a different number -- ``AGENT_MAX_RUN_SEC``, reported as
``run_bound_sec`` -- that the guard never read.

With the shipped default of 1500 s and a 5400 s bound, a healthy mission that had
been running 20 minutes could still legally run another 70. So ``--build
--apply`` rebuilt the image, waited, gave up, and exited 3: the fix was built and
the container was NOT replaced. The log said "still in flight ... gave up WITHOUT
recreating, because a killed run is worse than a late fix", which is true and
sounds principled, and hides the only actionable fact -- that no amount of
patience was ever going to be enough at that budget. The next pass then reports
the same drift and cannot tell whether the previous one failed or was never given
the time.

These tests pin the comparison and the three ways it can legitimately be silent.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recreate_guard as RG  # noqa: E402


def busy(running=True, in_progress_sec=2900.0, bound=5400.0, available=True,
         with_key=True):
    """A scheduler payload with a run in flight.

    ``with_key=False`` models a console that predates ``run_bound_sec``. That is
    a DIFFERENT situation from "no bound is configured" and the guard must not
    confuse them: one is an old deployment where guessing would warn spuriously,
    the other is an unbounded run where no wait is provably sufficient.
    """
    sched = {"available": available, "running": running,
             "in_progress_since": "2026-11-04T03:15:42",
             "in_progress_sec": in_progress_sec}
    if with_key:
        sched["run_bound_sec"] = bound
    return sched


def session(has_end=True, started="2026-11-04T03:15:42", readable=True):
    return {"file": "x.jsonl", "started": started, "has_session_end": has_end,
            "readable": readable, "host_path": "/tmp/x.jsonl"}


GARBAGE_PAYLOADS = [
    ({"running": True, "run_bound_sec": "soon", "in_progress_sec": 10},
     1500.0),
    ({"running": True, "run_bound_sec": 5400.0, "in_progress_sec": None},
     1500.0),
    ({"running": True, "run_bound_sec": None, "in_progress_sec": "lots"},
     "1500"),
    ({"running": True, "run_bound_sec": float("nan"),
      "in_progress_sec": 10}, 1500.0),
]

APPLY = ["--apply", "--container", "agent-sidecar", "--compose-dir", "/d",
         "--compose-file", "c.yml"]


class WaitBudgetVerdictTest(unittest.TestCase):
    """The pure rule, so each branch can be read separately."""

    def test_a_wait_that_can_contain_the_run_says_nothing(self):
        """A run 100 s from its bound with a 1500 s wait needs no warning."""
        self.assertIsNone(RG.wait_budget_verdict(
            busy(in_progress_sec=5300.0, bound=5400.0), 1500.0))

    def test_a_wait_shorter_than_the_run_can_still_needed_is_said(self):
        v = RG.wait_budget_verdict(busy(in_progress_sec=2900.0, bound=5400.0),
                                   1500.0)
        self.assertIsNotNone(v)
        #: Both numbers have to be in the sentence, or the operator cannot see
        #: the mismatch -- only the conclusion.
        self.assertIn("1500", v)
        self.assertIn("2500", v)
        self.assertIn("5400", v)
        self.assertIn("2900", v)
        self.assertIn("--wait-sec 2560", v,
                      "it must name a working budget, not just object to this "
                      "one")
        #: And the escape hatch for a run that genuinely will not end.
        self.assertIn("--force", v)

    def test_the_suggested_budget_exceeds_what_the_run_can_still_take(self):
        """A warning that recommends another insufficient budget is useless.

        Only cases where a warning IS due are listed: with 1 s left and a 1 s
        wait the verdict is correctly silent, and asserting a warning there
        would be asserting the bug this fix prevents.
        """
        import re
        for elapsed, bound, wait in ((100.0, 5400.0, 300.0),
                                     (5300.0, 5400.0, 10.0),
                                     (0.0, 3600.0, 60.0)):
            with self.subTest(elapsed=elapsed, bound=bound, wait=wait):
                v = RG.wait_budget_verdict(
                    busy(in_progress_sec=elapsed, bound=bound), wait)
                self.assertIsNotNone(v)
                remaining = bound - elapsed
                #: The sentence mentions --wait-sec TWICE: the budget being
                #: judged, then the fix. Take the suggestion after "Re-run with",
                #: or the parse reads back the caller's own number and the
                #: assertion passes for the wrong reason.
                suggested = float(
                    re.search(r"Re-run with --wait-sec (\d+)", v).group(1))
                self.assertGreater(suggested, remaining,
                                   "the suggested wait must itself be enough")

    def test_exact_equality_is_not_a_warning(self):
        """A wait that exactly matches what is left is sufficient."""
        self.assertIsNone(RG.wait_budget_verdict(
            busy(in_progress_sec=4400.0, bound=5400.0), 1000.0))

    def test_an_absent_bound_key_is_silence_not_a_guess(self):
        """A console that predates the field must not be warned about.

        Reporting "your wait may be too short" on a deployment that cannot
        answer would be a guess dressed as a measurement, and a guard that cries
        wolf gets run with --force, which is the dangerous direction.
        """
        self.assertIsNone(
            RG.wait_budget_verdict(busy(with_key=False), 1500.0))

    def test_an_unbounded_run_says_so_and_points_at_force(self):
        """`run_bound_sec: null` means no wall-clock bound, not "unknown"."""
        v = RG.wait_budget_verdict(busy(bound=None), 1500.0)
        self.assertIsNotNone(v)
        self.assertIn("no wall-clock bound", v)
        self.assertIn("--force", v)
        self.assertNotIn("--wait-sec", v,
                         "no number can fix an unbounded run, so suggesting one "
                         "invites retrying the same failing command")

    def test_no_run_in_flight_is_silence(self):
        self.assertIsNone(
            RG.wait_budget_verdict(busy(running=False), 1500.0))

    def test_it_never_raises_on_garbage_the_console_could_send(self):
        """The guard runs at 4am with nobody to read a traceback."""
        for sched, wait in GARBAGE_PAYLOADS:
            with self.subTest(sched=sched):
                RG.wait_budget_verdict(sched, wait)   # must not raise

    def test_a_null_elapsed_is_treated_as_zero_not_as_unknown(self):
        """A run that just started has no elapsed reading yet."""
        v = RG.wait_budget_verdict(
            busy(in_progress_sec=None, bound=5400.0), 1500.0)
        self.assertIsNotNone(v)
        self.assertIn("5400", v)

    def test_a_nan_bound_is_silence_and_not_a_crash(self):
        """``int(nan)`` raises: this is a crash, found by the garbage sweep.

        NaN is reachable, not theoretical -- Python's json EMITS and PARSES it by
        default, so a console that computed an unbounded value can send
        ``run_bound_sec: NaN`` straight through ``float()``. The function that
        exists to help an operator at 4am must not be the one that raises on
        them, and it must not print "nan" in a sentence that looks like a
        measurement.
        """
        self.assertIsNone(RG.wait_budget_verdict(
            busy(bound=float("nan"), in_progress_sec=100.0), 1500.0))
        self.assertIsNone(RG.wait_budget_verdict(
            busy(bound=float("inf"), in_progress_sec=100.0), 1500.0))
        v = RG.wait_budget_verdict(
            busy(bound=5400.0, in_progress_sec=float("nan")), 1500.0)
        self.assertTrue(v is None or "nan" not in v.lower(), v)


class TheWaitLoopSaysWhyItGaveUpTest(unittest.TestCase):
    """The give-up reason must distinguish "out of patience" from "never had a
    chance"."""

    def test_giving_up_on_a_run_that_exceeded_the_budget_says_so(self):
        probe = mock.Mock(return_value=(busy(in_progress_sec=100.0,
                                             bound=5400.0), None))
        action, reason = RG.wait_until_idle(probe, 0.0, log=lambda *a: None)
        self.assertEqual(action, "wait")
        self.assertIn("gave up WITHOUT recreating", reason)
        self.assertIn("never going to be enough", reason,
                      "the reason reads as a principled refusal unless it also "
                      "says the budget could not have worked")
        self.assertIn("--wait-sec", reason)

    def test_giving_up_when_the_wait_would_have_worked_says_only_that(self):
        """A run that ends just after the wait is NOT a budget error.

        This needs a FAKE CLOCK rather than ``wait_sec=0``: a zero wait can never
        contain anything, so with 0 the budget verdict correctly complains and
        the test would be asserting its own artifact. Patching ``monotonic`` to
        jump past the deadline models "the operator waited the full budget and
        the run was about to finish", which is the case that must not blame the
        number.
        """
        probe = mock.Mock(return_value=(busy(in_progress_sec=5390.0,
                                             bound=5400.0), None))
        clock = iter([1000.0, 9999.0, 9999.0, 9999.0])
        with mock.patch.object(RG.time, "monotonic",
                               side_effect=lambda: next(clock)):
            _action, reason = RG.wait_until_idle(
                probe, 1500.0, log=lambda *a: None)
        self.assertIn("gave up WITHOUT recreating", reason)
        self.assertNotIn("never going to be enough", reason,
                         "blaming the budget when 10s more would have done "
                         "sends the operator to fix the wrong number")
        self.assertNotIn("CANNOT contain", reason)

    def test_the_verdict_is_logged_once_not_every_poll(self):
        """A 90-minute wait polls ~180 times; one warning is the point of it."""
        probe = mock.Mock(return_value=(busy(in_progress_sec=100.0,
                                             bound=5400.0), None))
        lines = []
        RG.wait_until_idle(probe, 0.0, log=lambda *a: lines.append(" ".join(
            str(x) for x in a)))
        joined = "\n".join(lines)
        self.assertLessEqual(joined.count("CANNOT contain"), 1, joined)


class MainWarnsBeforeBuildingTest(unittest.TestCase):
    """The warning has to arrive BEFORE the build, or it arrives too late.

    ``--build --apply`` spends minutes building first. An operator who learns
    only afterwards that the wait was never going to be long enough has wasted a
    build; more importantly the next invocation is likely to be the same command
    with the same budget.
    """

    def _main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = RG.main(argv)
        return rc, buf.getvalue()

    def test_a_short_wait_with_build_and_apply_warns_before_building(self):
        """Call order is the evidence: the warning arrives before the build runs.

        ``sh`` is patched, so the usual ``+ <cmd>`` log line never appears --
        ordering is recorded from the call itself instead of from the log text.
        """
        order = []

        def recorder(*values):
            order.append(("log", " ".join(str(v) for v in values)))

        def fake_sh(cmd, **kwargs):
            order.append(("sh", cmd))
            return 0, ""

        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=100.0,
                                                  bound=5400.0),
                                             session())), \
                mock.patch.object(RG, "line_log", side_effect=recorder), \
                mock.patch.object(RG, "wait_until_idle",
                                  return_value=("wait", "gave up")) as w, \
                mock.patch.object(RG, "sh", side_effect=fake_sh):
            rc = RG.main(["--build", "--build-context", "/ctx",
                          "--image", "harness:1.0.0", "--wait-sec",
                          "1500"] + APPLY)
        self.assertEqual(rc, 3)
        self.assertEqual(w.call_args.args[1], 1500.0,
                         "the wait it used is the wait it must judge")
        self.assertTrue(any(kind == "sh" for kind, _ in order),
                        "the build never ran, so this test proves nothing: %r"
                        % (order,))
        warn_at = next((i for i, (kind, text) in enumerate(order)
                        if kind == "log" and "CANNOT contain" in text), None)
        build_at = next((i for i, (kind, cmd) in enumerate(order)
                         if kind == "sh" and "build" in cmd), None)
        self.assertIsNotNone(warn_at)
        self.assertLess(warn_at, build_at,
                        "the warning arrived after the build, so the operator "
                        "learns too late that the wait was never sufficient")
        self.assertIn("--wait-sec 5360", order[warn_at][1])

    def test_a_sufficient_wait_does_not_warn(self):
        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=5390.0,
                                                  bound=5400.0), session())), \
                mock.patch.object(RG, "wait_until_idle",
                                  return_value=("wait", "gave up")), \
                mock.patch.object(RG, "sh", return_value=(0, "")):
            _rc, out = self._main(["--wait-sec", "3600"] + APPLY)
        self.assertNotIn("CANNOT contain", out)

    def test_force_does_not_warn_about_a_wait_it_will_not_do(self):
        """Forcing skips the wait, so judging its budget is noise."""
        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=100.0,
                                                  bound=5400.0), session())), \
                mock.patch.object(RG, "sh", return_value=(0, "ok")), \
                mock.patch.object(RG, "container_started", return_value="b"), \
                mock.patch.object(RG, "verify", return_value=0):
            _rc, out = self._main(APPLY + ["--force"])
        self.assertNotIn("CANNOT contain", out)
        self.assertIn("FORCE", out)

    def test_a_probe_only_run_does_not_warn(self):
        """Without --apply nothing destructive is pending, so nothing is at stake."""
        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=100.0,
                                                  bound=5400.0), session())):
            _rc, out = self._main([])
        self.assertNotIn("CANNOT contain", out)

    def test_an_idle_run_never_warns(self):
        with mock.patch.object(RG, "probe",
                               return_value=(busy(running=False), session())), \
                mock.patch.object(RG, "sh", return_value=(0, "")), \
                mock.patch.object(RG, "container_started", return_value="b"), \
                mock.patch.object(RG, "verify", return_value=0):
            _rc, out = self._main(APPLY)
        self.assertNotIn("CANNOT contain", out)

    def test_json_mode_carries_the_verdict_for_a_scripted_caller(self):
        """A cron job driving this gets the same fact a human reading logs gets."""
        buf = io.StringIO()
        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=100.0,
                                                  bound=5400.0), session())), \
                redirect_stdout(buf):
            rc = RG.main(["--json", "--wait-sec", "1500"] + APPLY)
        self.assertEqual(rc, 3)
        payload = json.loads(buf.getvalue())
        self.assertIn("wait_too_short", payload)
        self.assertIn("--wait-sec 5360", payload["wait_too_short"])

    def test_json_mode_omits_the_verdict_when_there_is_nothing_to_say(self):
        buf = io.StringIO()
        with mock.patch.object(RG, "probe",
                               return_value=(busy(in_progress_sec=5390.0,
                                                  bound=5400.0), session())), \
                redirect_stdout(buf):
            RG.main(["--json", "--wait-sec", "3600"] + APPLY)
        self.assertNotIn("wait_too_short", json.loads(buf.getvalue()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
