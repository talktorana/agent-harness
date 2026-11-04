"""Tests for bounded history and prompt-cache evidence.

The regression these pin is unbounded request growth: every model call used to
resend the entire run, so later turns cost several times the first turn. These
assert that history is bounded WITHOUT breaking the tool-call protocol or
throwing away the original mission, and that cache usage is visible in the
record an operator reads.
"""
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent as agent_mod  # noqa: E402
import config  # noqa: E402
from history import MARKER, compact_history, history_chars  # noqa: E402


def task_messages():
    return [
        {"role": "system", "content": "SYSTEM PROMPT -- must survive verbatim"},
        {"role": "user", "content": "MISSION TEXT -- must survive verbatim"},
    ]


def turn(n, result_chars=1200):
    call_id = "call_%d" % n
    return [
        {"role": "assistant", "content": "investigating %d" % n,
         "tool_calls": [{"id": call_id, "type": "function",
                         "function": {"name": "run_command",
                                      "arguments": json.dumps({"command": "suite %d" % n})}}]},
        {"role": "tool", "tool_call_id": call_id,
         "content": "exit=0\nRan %d tests OK\n" % n + ("evidence-%d " % n) * result_chars},
    ]


def assert_protocol_valid(testcase, messages):
    """Every retained tool call has its response, and no tool is orphaned."""
    pending = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                pending.add(call["id"])
        elif role == "tool":
            testcase.assertIn(message["tool_call_id"], pending,
                              "a tool response lost its assistant call")
            pending.discard(message["tool_call_id"])
    testcase.assertEqual(pending, set(), "an assistant tool call lost its result")


class CompactionTest(unittest.TestCase):
    def test_disabled_returns_every_message_verbatim(self):
        messages = task_messages() + turn(1)
        result, info = compact_history(messages, max_chars=0, keep_chars=10)
        self.assertEqual(result, messages)
        self.assertIsNone(info)

    def test_under_the_limit_does_nothing(self):
        messages = task_messages() + turn(1)
        result, info = compact_history(messages, max_chars=1000000,
                                       keep_chars=100)
        self.assertEqual(result, messages)
        self.assertIsNone(info)

    def test_system_and_mission_survive_verbatim(self):
        messages = task_messages() + sum((turn(i) for i in range(12)), [])
        result, info = compact_history(messages, max_chars=1000,
                                       keep_chars=100, summary_chars=2000)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[1], messages[1])
        self.assertIsNotNone(info)

    def test_history_is_bounded_and_protocol_stays_valid(self):
        messages = task_messages() + sum((turn(i) for i in range(20)), [])
        result, info = compact_history(messages, max_chars=1000,
                                       keep_chars=100, summary_chars=2500)
        self.assertLess(history_chars(result), history_chars(messages))
        self.assertLessEqual(len(result), len(messages))
        assert_protocol_valid(self, result)
        self.assertGreater(info["groups_condensed"], 0)

    def test_summary_points_at_the_full_session_log(self):
        messages = task_messages() + sum((turn(i) for i in range(10)), [])
        result, _ = compact_history(messages, max_chars=1000,
                                    keep_chars=100, summary_chars=2000,
                                    session_file="run.jsonl")
        summaries = [m for m in result if m.get("role") == "system"
                     and str(m.get("content", "")).startswith(MARKER)]
        self.assertEqual(len(summaries), 1)
        self.assertIn("run.jsonl", summaries[0]["content"])
        self.assertIn("session_read", summaries[0]["content"])
        self.assertIn("lossy index", summaries[0]["content"])

    def test_older_evidence_is_indexed_not_erased(self):
        messages = task_messages() + sum((turn(i) for i in range(10)), [])
        result, _ = compact_history(messages, max_chars=1000,
                                    keep_chars=100, summary_chars=3000)
        summary = result[2]["content"]
        self.assertIn("run_command", summary)
        self.assertIn("suite 1", summary)
        self.assertIn("exit=0", summary)

    def test_recent_groups_are_kept_verbatim(self):
        messages = task_messages() + sum((turn(i) for i in range(10)), [])
        result, _ = compact_history(messages, max_chars=1000,
                                    keep_chars=100, summary_chars=2000)
        #: The last complete assistant/tool pair must be byte-for-byte present.
        self.assertIn(messages[-2], result)
        self.assertIn(messages[-1], result)

    def test_multimodal_history_drops_pixels_but_keeps_caption(self):
        messages = task_messages() + [
            {"role": "assistant", "content": "look", "tool_calls": [
                {"id": "img", "type": "function",
                 "function": {"name": "view_image", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "img", "content": [
                {"type": "text", "text": "screenshot of the console"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]},
        ] + sum((turn(i) for i in range(5)), [])
        result, _ = compact_history(messages, max_chars=100,
                                    keep_chars=10, summary_chars=2000)
        summary = result[2]["content"]
        self.assertIn("screenshot of the console", summary)
        self.assertNotIn("base64", summary)
        self.assertNotIn("AAAA", summary)

    def test_repeated_compaction_has_one_summary_and_stays_bounded(self):
        messages = task_messages() + sum((turn(i) for i in range(10)), [])
        first, _ = compact_history(messages, max_chars=1000,
                                   keep_chars=100, summary_chars=2000)
        second, info = compact_history(first + sum((turn(i + 20) for i in range(8)), []),
                                       max_chars=1000, keep_chars=100,
                                       summary_chars=2000)
        markers = [m for m in second if str(m.get("content", "")).startswith(MARKER)]
        self.assertEqual(len(markers), 1)
        self.assertLessEqual(len(markers[0]["content"]), 2100)
        assert_protocol_valid(self, second)
        self.assertIsNotNone(info)

    def test_summary_size_is_capped_and_newest_evidence_wins(self):
        messages = task_messages() + sum((turn(i, result_chars=4000)
                                          for i in range(30)), [])
        result, _ = compact_history(messages, max_chars=100,
                                    keep_chars=10, summary_chars=1800)
        self.assertLessEqual(len(result[2]["content"]), 1850)
        #: The newest evidence is preserved VERBATIM in the retained tail, not
        #: expected to be repeated inside the lossy index of OLDER turns.
        self.assertIn("suite 29",
                      result[-2]["tool_calls"][0]["function"]["arguments"])
        self.assertIn("Ran 0 tests OK", result[2]["content"])

    def test_one_giant_group_is_not_split_into_an_invalid_request(self):
        messages = task_messages() + turn(1, result_chars=100000)
        result, info = compact_history(messages, max_chars=10,
                                       keep_chars=10)
        self.assertEqual(result, messages)
        self.assertIsNone(info)
        assert_protocol_valid(self, result)

    def test_non_string_content_does_not_crash_or_fabricate_text(self):
        messages = task_messages() + [
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "user", "content": {"unexpected": "shape"}},
        ] + sum((turn(i) for i in range(4)), [])
        result, _ = compact_history(messages, max_chars=10,
                                    keep_chars=10)
        self.assertTrue(all(isinstance(m.get("content"), (str, list, dict, type(None)))
                            for m in result))

    def test_bad_budgets_are_refused_loudly(self):
        with self.assertRaises(ValueError):
            compact_history(task_messages() + turn(1), max_chars=1,
                            keep_chars=0)
        with self.assertRaises(ValueError):
            compact_history(task_messages() + turn(1), max_chars=1,
                            keep_chars=1, summary_chars=100)


class RequestCacheTest(unittest.TestCase):
    def test_no_cache_key_by_default_preserves_the_old_payload(self):
        with mock.patch.object(config, "PROMPT_CACHE_KEY", ""):
            request = agent_mod.build_request("vendor/model", [], True)
        self.assertNotIn("prompt_cache_key", request)

    def test_configured_cache_key_reaches_the_wire(self):
        with mock.patch.object(config, "PROMPT_CACHE_KEY", "stable-prefix-v1"):
            request = agent_mod.build_request("vendor/model", [], True)
        self.assertEqual(request["prompt_cache_key"], "stable-prefix-v1")

    def test_cached_token_counts_are_recorded_in_both_sinks(self):
        class Details:
            cached_tokens = 19712
            cache_write_tokens = 140
        class Usage:
            prompt_tokens = 19852
            completion_tokens = 8
            prompt_tokens_details = Details()
            completion_tokens_details = None

        class Session:
            def __init__(self):
                self.events = []
            def event(self, kind, **fields):
                self.events.append((kind, fields))

        instance = agent_mod.Agent.__new__(agent_mod.Agent)
        instance.session = Session()
        transcript = []
        instance._log = transcript.append
        instance._record_usage(Usage(), "provider/model")
        self.assertEqual(instance.session.events[0][1]["cached_prompt_tokens"],
                         19712)
        self.assertEqual(instance.session.events[0][1]["cache_write_tokens"], 140)
        self.assertEqual(transcript[0]["cached_prompt_tokens"], 19712)
        self.assertEqual(transcript[0]["cache_write_tokens"], 140)

    def test_absent_cache_details_stay_absent_not_fake_zero(self):
        class Usage:
            prompt_tokens = 100
            completion_tokens = 5
            prompt_tokens_details = None
            completion_tokens_details = None
        instance = agent_mod.Agent.__new__(agent_mod.Agent)
        instance.session = None
        transcript = []
        instance._log = transcript.append
        instance._record_usage(Usage(), "provider/model")
        self.assertNotIn("cached_prompt_tokens", transcript[0])
        self.assertNotIn("cache_write_tokens", transcript[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
