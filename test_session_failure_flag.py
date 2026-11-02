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


class ActionIdentityTest(unittest.TestCase):
    """A call's identity must be exact; its rendering may be truncated.

    The mission tells a monitoring pass to "count the repeated identical calls
    -- same tool + same arguments". It can only do that if identity is
    recoverable from the record, and ``detail`` cannot supply it: each argument
    is cut at 90 characters, so two invocations differing only later render
    identically and both end in an ellipsis.

    Live 2026-11-02 I read two dashboard-suite runs as a duplicate pair for
    exactly this reason -- the second was a verbose re-run and the flag that
    distinguished it fell past the cut. The waste figure I reported was wrong,
    in the direction that makes an agent look wasteful when it was not.

    ``args_hash`` is the fix, and it is a FINGERPRINT, not a copy: 12 hex
    characters per action, so the whole jsonl grows negligibly.
    """

    def test_calls_differing_only_past_the_display_cut_are_distinguishable(self):
        #: A generic path on purpose: this repository holds no deployment's
        #: vocabulary, and a fixture that quotes one turns the real command into
        #: the shape the code is tested against. test_genericity enforces it.
        base = ("cd /project/app/widgets && "
                "PYTHONPYCACHEPREFIX=/agent-data/work/pyc python3 -m unittest "
                "discover -s . -p test_*.py -t .")
        a = {"command": base}
        b = {"command": base + " -v"}
        self.assertEqual(session._one_line(a["command"], 90),
                         session._one_line(b["command"], 90),
                         "precondition: the rendered forms collide")
        self.assertNotEqual(session._args_hash(a), session._args_hash(b),
                            "two different commands must not share a hash")

    def test_the_same_call_hashes_the_same(self):
        a = {"command": "ls -la", "reason": "look"}
        self.assertEqual(session._args_hash(a), session._args_hash(dict(a)))

    def test_key_order_does_not_matter(self):
        """Arguments are a mapping; insertion order is not information."""
        self.assertEqual(session._args_hash({"x": 1, "y": 2}),
                         session._args_hash({"y": 2, "x": 1}))

    def test_different_arguments_hash_differently(self):
        self.assertNotEqual(session._args_hash({"command": "ls"}),
                            session._args_hash({"command": "ls -l"}))

    def test_no_arguments_is_a_stable_hash(self):
        self.assertEqual(session._args_hash(None), session._args_hash({}))
        self.assertEqual(len(session._args_hash({})),
                         session.ARGS_HASH_CHARS)

    def test_a_non_serialisable_argument_does_not_raise(self):
        """A diagnostic field must never be why a run crashes."""
        self.assertTrue(session._args_hash({"weird": {1, 2, 3}}))

    def test_the_record_carries_the_hash(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = session.SessionLog("probe", sessions_dir=Path(tmp.name))
        self.addCleanup(log._jsonl.close)
        log.action(1, "run_command", {"command": "ls -la"}, "exit=0\nok", 0.1)
        log.action(2, "run_command", {"command": "ls -la"}, "exit=0\nok", 0.1)
        log.action(3, "run_command", {"command": "ls -la -v"}, "exit=0\nok", 0.1)
        keys = [(a["tool"], a["args_hash"]) for a in log._actions]
        self.assertEqual(keys[0], keys[1], "an identical repeat must be visible")
        self.assertNotEqual(keys[0], keys[2],
                            "a different call must not look like a repeat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
