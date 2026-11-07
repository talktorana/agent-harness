"""An exact repeated tool call must be REFUSED past a generous allowance.

Why a cap and not just the existing notice: measured 2026-11-06, run
``25599ed1`` made 1,285 tool calls containing 123 exact repeats -- every one of
them already carrying a ``[duplicate call]`` notice -- across 382 turns, staging
no deploy, and self-reporting FAILED DELIVERY. Its successor made 981 calls
(43 repeats) and hit the wall clock. Informing did not bind; refusing does.

    python -m unittest test_duplicate_call_limit
"""
from __future__ import annotations

import unittest
from unittest import mock

import agent
import config
import session as session_mod


class DuplicateCallLimitTest(unittest.TestCase):
    def setUp(self):
        self.subject = agent.Agent.__new__(agent.Agent)
        self.subject._call_fingerprints = {}
        self.subject._transcript = None
        self.subject.session = None
        self.subject.steps = 1
        self.subject.verbose = False
        self.events: list = []
        self.subject.on_event = self.events.append
        self.calls: list = []

        def fake_handler(**kwargs):
            self.calls.append(kwargs)
            return f"ok {kwargs.get('target', '')}"

        patcher = mock.patch.dict(agent.HANDLERS, {"fake_read": fake_handler},
                                  clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_call(self, target="a", limit=3):
        with mock.patch.object(config, "DUPLICATE_CALL_LIMIT", limit):
            return self.subject._run_tool("fake_read", {"target": target})

    def test_first_calls_execute_normally_up_to_the_allowance(self):
        for _ in range(3):
            result, _content = self.run_call(limit=3)
        self.assertEqual(len(self.calls), 3)
        self.assertNotIn("REFUSED", str(result))

    def test_the_call_past_the_allowance_is_refused_without_executing(self):
        for _ in range(3):
            self.run_call(limit=3)
        result, content = self.run_call(limit=3)
        self.assertEqual(len(self.calls), 3,
                         "a refused repeat must not run the tool at all")
        self.assertIn("REFUSED", str(result))
        self.assertIn("NOT executed", str(result))
        self.assertEqual(str(content), str(result))

    def test_refusal_is_returned_as_text_and_never_raises(self):
        for _ in range(5):
            result, _ = self.run_call(limit=1)
        self.assertIsInstance(result, str)

    def test_the_refusal_is_recorded_for_audit(self):
        self.run_call(limit=1)
        self.run_call(limit=1)
        blocked = [e for e in self.events
                   if e.get("event") == "duplicate_call_blocked"]
        self.assertTrue(blocked, "a refusal that leaves no record is not auditable")
        self.assertEqual(blocked[-1]["tool"], "fake_read")
        self.assertEqual(blocked[-1]["limit"], 1)
        self.assertEqual(blocked[-1]["args_hash"],
                         session_mod.args_hash({"target": "a"}))

    def test_zero_disables_the_cap_and_restores_notice_only_behaviour(self):
        for _ in range(12):
            result, _ = self.run_call(limit=0)
        self.assertEqual(len(self.calls), 12)
        self.assertNotIn("REFUSED", str(result))

    def test_different_arguments_are_not_the_same_repeat(self):
        for target in ("a", "b", "a", "b", "a", "b"):
            result, _ = self.run_call(target=target, limit=3)
        self.assertEqual(len(self.calls), 6,
                         "the cap is per exact argument fingerprint")
        self.assertNotIn("REFUSED", str(result))

    def test_the_counter_increments_once_per_attempt(self):
        """One increment site: a refusal must not double-count."""
        self.run_call(limit=1)          # attempt 1 executes
        self.run_call(limit=1)          # attempt 2 refused
        self.assertEqual(self.subject._repeat_count(
            "fake_read", {"target": "a"}), 2)

    def test_refusal_tells_the_model_what_to_do_instead(self):
        for _ in range(2):
            result, _ = self.run_call(limit=1)
        text = str(result)
        for expected in ("change the arguments", "stage the fix",
                         "final report", "AGENT_DUPLICATE_CALL_LIMIT=0"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
