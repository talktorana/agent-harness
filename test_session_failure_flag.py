"""The action timeline's ok/⚠️ flag: it must not lie about what worked.

WHY THIS EXISTS
---------------
``SessionLog.action`` records every tool call and derives ``ok`` from
``session._failed(result)``. That flag drives three things a person and a pass
both rely on:

* the ⚠️ marker beside every entry in the console timeline;
* ``self._errors``, printed in the run summary;
* the "failed tool calls" count a monitoring pass uses to judge its own
  efficiency -- the mission text explicitly tells it to quantify that.

So a false ⚠️ is not cosmetic. Live 2026-11-02 the 11:21 pass was scored with
five failed calls, and three of them were successful ``grep`` commands whose
findings happened to be tracebacks:

    grep -nE 'CRITICAL|ERROR|Traceback' watchdog.log      -> exit=0

``_failed`` scanned the output for the phrase "Traceback (most recent call
last)" and marked them failed. The counter was therefore counting the act of
LOOKING FOR errors as an error, which is the one search this deployment's
monitoring exists to perform.

The rule these tests pin: AN EXIT CODE IS A VERDICT AND TEXT MUST NOT OVERRULE
IT. The body scan stays for the tools that have no exit code (generated Python
reports failure as a traceback block), because that is what it was written for.

Run:  python -m unittest test_session_failure_flag
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import session  # noqa: E402

TRACEBACK = ("Traceback (most recent call last):\n"
             "  File \"x.py\", line 1, in <module>\n"
             "ValueError: boom\n")


class ExitCodeIsAVerdictTest(unittest.TestCase):
    """``exit=<rc>`` is authoritative; the output body is DATA."""

    def test_a_successful_grep_for_tracebacks_is_not_a_failure(self):
        """The exact command that produced three false ⚠️ in one pass."""
        result = f"exit=0\n1553:COLLECTOR-ERRORS:\n1554:{TRACEBACK}"
        self.assertFalse(session._failed(result),
                         "a successful search FOR errors is not an error")

    def test_a_successful_command_mentioning_exceptions_is_not_a_failure(self):
        self.assertFalse(session._failed(
            "exit=0\n[exception] handling is described in docstring line 4"))
        self.assertFalse(session._failed("exit=0\nno problems found"))

    def test_a_nonzero_exit_is_a_failure(self):
        self.assertTrue(session._failed("exit=1\nsome error text"))
        self.assertTrue(session._failed(f"exit=2\n{TRACEBACK}"))
        self.assertTrue(session._failed("exit=130\n"))

    def test_a_large_negative_exit_still_parses(self):
        """Signals report as negative codes; the sign must not confuse it."""
        self.assertTrue(session._failed("exit=-9\nkilled"))

    def test_whitespace_before_the_exit_line_does_not_hide_it(self):
        self.assertFalse(session._failed(f"  exit=0\n{TRACEBACK}"))


class BodyScanStillAppliesWithoutAnExitCodeTest(unittest.TestCase):
    """Generated code has no exit code, so the traceback scan must survive."""

    def test_a_traceback_block_is_a_failure(self):
        self.assertTrue(session._failed(TRACEBACK))

    def test_an_exception_marker_is_a_failure(self):
        self.assertTrue(session._failed("[exception] ValueError: boom"))

    def test_plain_output_is_success(self):
        self.assertFalse(session._failed("413"))

    def test_the_error_prefixes_still_win(self):
        for prefix in ("ERROR", "TOOL ERROR", "UNKNOWN TOOL", "TIMEOUT",
                       "NOT FOUND", "BAD ARGUMENTS", "DECLINED", "EMPTY",
                       "IS A DIRECTORY", "EMPTY RANGE"):
            with self.subTest(prefix=prefix):
                self.assertTrue(session._failed(f"{prefix}: something"))

    def test_an_unparsable_exit_code_is_not_evidence_of_failure(self):
        """Refuse to guess: a garbled code must not become a ⚠️."""
        self.assertFalse(session._failed("exit=oops\nran fine"))


class TheCounterFollowsTheFlagTest(unittest.TestCase):
    """``_errors`` is derived from the flag, so the two cannot disagree."""

    def test_errors_counts_only_real_failures(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = session.SessionLog("probe", sessions_dir=Path(tmp.name))
        #: finish() closes the jsonl handle. The log is a long-lived writer by
        #: design, so a test that abandons it must close it explicitly or the
        #: temp directory cannot be removed on Windows (WinError 32) and the
        #: leak surfaces as a teardown error that looks like a defect in the
        #: thing under test.
        self.addCleanup(log._jsonl.close)
        log.action(1, "run_command", {"command": "grep Traceback log"},
                   f"exit=0\n{TRACEBACK}", 0.1)
        self.assertEqual(log._errors, 0,
                         "a successful grep must not raise the error count")
        log.action(2, "run_command", {"command": "false"},
                   "exit=1\nit failed", 0.1)
        self.assertEqual(log._errors, 1)
        self.assertTrue(log._actions[0]["ok"])
        self.assertFalse(log._actions[1]["ok"])
        #: Close now, so cleanup order is close-then-rmtree.
        log._jsonl.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
