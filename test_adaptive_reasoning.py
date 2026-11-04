"""Opt-in adaptive reasoning: cheap routine turns, deep turns on evidence."""
from types import SimpleNamespace
import unittest
from unittest import mock

import agent
import config


class BuildRequestEffortTest(unittest.TestCase):
    def test_default_preserves_configured_effort(self):
        with mock.patch.object(config, "REASONING_EFFORT", "low"):
            request = agent.build_request("model", [], False)
        self.assertEqual(request["reasoning_effort"], "low")

    def test_explicit_deep_effort_reaches_the_wire(self):
        with mock.patch.object(config, "REASONING_EFFORT", "low"):
            request = agent.build_request("model", [], False, reasoning_effort="max")
        self.assertEqual(request["reasoning_effort"], "max")

    def test_explicit_empty_effort_omits_the_field(self):
        with mock.patch.object(config, "REASONING_EFFORT", "low"):
            request = agent.build_request("model", [], False, reasoning_effort="")
        self.assertNotIn("reasoning_effort", request)


class AdaptiveReasoningTest(unittest.TestCase):
    def setUp(self):
        self.subject = agent.Agent.__new__(agent.Agent)
        self.subject._deep_reasoning = False
        self.subject._active_reasoning_effort = "low"
        self.subject._transcript = None
        self.subject.on_event = None
        self.subject.session = None
        self.patches = [
            mock.patch.object(config, "REASONING_EFFORT", "low"),
            mock.patch.object(config, "DEEP_REASONING_EFFORT", "max", create=True),
            mock.patch.object(config, "DEEP_REASONING_TRIGGER",
                              '"fast_path_eligible": false||FAILED (errors=', create=True),
            mock.patch.object(config, "DEEP_REASONING_TOOLS", ("deploy_request",), create=True),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def test_routine_result_stays_at_base_effort(self):
        self.subject._update_reasoning_depth("run_command", '{"fast_path_eligible": true}')
        self.assertFalse(self.subject._deep_reasoning)
        self.assertEqual(self.subject._next_reasoning_effort(), "low")

    def test_attention_marker_escalates_for_the_rest_of_the_run(self):
        events = []
        self.subject.on_event = events.append
        self.subject._update_reasoning_depth(
            "run_command", 'exit=0\n{"fast_path_eligible": false, "attention_required": ["red"]}')
        self.assertTrue(self.subject._deep_reasoning)
        self.assertEqual(self.subject._next_reasoning_effort(), "max")
        self.assertEqual(events[0]["event"], "reasoning_escalated")
        self.assertEqual(events[0]["from"], "low")
        self.assertEqual(events[0]["to"], "max")
        self.subject._update_reasoning_depth("run_command", "all healthy now")
        self.assertEqual(self.subject._next_reasoning_effort(), "max")

    def test_second_marker_form_is_supported(self):
        self.subject._update_reasoning_depth("run_command", "FAILED (errors=1)")
        self.assertEqual(self.subject._next_reasoning_effort(), "max")

    def test_configured_tool_name_escalates_without_result_text_matching(self):
        self.subject._update_reasoning_depth("deploy_request", "request staged")
        self.assertEqual(self.subject._next_reasoning_effort(), "max")

    def test_disabled_deep_effort_preserves_legacy_behaviour(self):
        with mock.patch.object(config, "DEEP_REASONING_EFFORT", "", create=True):
            self.subject._update_reasoning_depth("deploy_request", "FAILED (errors=1)")
        self.assertFalse(self.subject._deep_reasoning)
        self.assertEqual(self.subject._next_reasoning_effort(), "low")

    def test_usage_records_actual_not_base_effort(self):
        events = []
        self.subject.on_event = events.append
        self.subject._active_reasoning_effort = "max"
        details = SimpleNamespace(reasoning_tokens=12)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                completion_tokens_details=details,
                                prompt_tokens_details=None)
        self.subject._record_usage(usage, "route")
        self.assertEqual(events[0]["reasoning_effort"], "max")
        self.assertEqual(events[0]["reasoning_tokens"], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
