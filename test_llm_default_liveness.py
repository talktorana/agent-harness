"""Promoting a model to DEFAULT must prove it can complete an agentic turn.

Resolving is not working. On 2026-11-24 a newly added model resolved cleanly
and even answered a bare 32-token chat, so it was made the default. Its first
mission then stalled on EVERY real turn -- four calls, zero completions, each
dying at the 120 s timeout -- and survived only by falling back to the
previous model at two minutes per turn. The difference was the real request:
the full tool schema and a real completion budget.

So ``save_default_choice`` now runs ``probe_turn``, which sends what a mission
sends (via ``agent.build_request``, the one authority for that payload) with a
short timeout. These tests pin:

  * a model that TIMES OUT on a tool-enabled turn is refused as default, and
    the previous default stays in force (THE incident);
  * a model that answers in text without calling a tool is refused -- it
    cannot drive an agent loop;
  * a model that returns malformed tool arguments is refused;
  * a model that completes a tool call is accepted;
  * the probe is bounded well below the mission timeout, so a stall costs
    seconds, not minutes;
  * the probe itself never raises -- callers report, they do not crash.

No network: the provider client is replaced at ``llm.client_for``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import llm  # noqa: E402

CATALOG = {
    "default": {"provider": "alpha", "model": "fast"},
    "providers": {"alpha": {
        "base_url": "https://alpha.test/v1",
        "api_key_env": "TEST_ALPHA_KEY",
        "models": {"fast": "alpha-fast-1", "smart": "alpha-smart-2"},
    }},
}


def _call(name="read_file", arguments='{"path": "x"}'):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(function=fn)


def _client(result=None, *, raises=None, seen=None, delay=0.0):
    """A stand-in for the provider client with one scripted behaviour."""
    def create(**request):
        if seen is not None:
            seen.append(request)
        if delay:
            time.sleep(delay)
        if raises is not None:
            raise raises
        return result
    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _response(calls):
    msg = SimpleNamespace(content=None, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _Timeout(Exception):
    pass


class SaveDefaultProbesLiveness(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.choice = Path(tmp.name) / "model_choice.json"
        for patcher in (
            mock.patch.object(llm, "PROVIDERS_FILE",
                              Path(tmp.name) / "none.json"),
            mock.patch.object(config, "MODEL_CHOICE_FILE", self.choice),
            mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "env-secret"}),
            mock.patch.dict(llm.providers(), {}, clear=False),
            mock.patch.object(llm, "_load_file", return_value=CATALOG),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_a_model_that_stalls_is_refused_and_the_default_survives(self):
        # THE INCIDENT: resolves cleanly, times out on every real turn.
        llm.save_default_choice("alpha", "fast", probe=False)
        with mock.patch.object(llm, "client_for",
                               return_value=_client(raises=_Timeout(
                                   "Request timed out."))):
            with self.assertRaises(llm.ProviderError) as caught:
                llm.save_default_choice("alpha", "smart")
        self.assertIn("cannot complete an agentic turn", str(caught.exception))
        self.assertIn("timeout", str(caught.exception))
        self.assertIn("previous default is unchanged", str(caught.exception))
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))
        saved = json.loads(self.choice.read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "fast",
                         "the refused choice must not have been written")

    def test_a_text_only_answer_is_refused(self):
        with mock.patch.object(llm, "client_for",
                               return_value=_client(_response([]))):
            with self.assertRaises(llm.ProviderError) as caught:
                llm.save_default_choice("alpha", "smart")
        self.assertIn("no_tool_call", str(caught.exception))
        self.assertFalse(self.choice.exists())

    def test_malformed_tool_arguments_are_refused(self):
        bad = _response([_call(arguments="{not json")])
        with mock.patch.object(llm, "client_for",
                               return_value=_client(bad)):
            with self.assertRaises(llm.ProviderError) as caught:
                llm.save_default_choice("alpha", "smart")
        self.assertIn("bad_response", str(caught.exception))

    def test_a_model_that_completes_a_tool_call_is_accepted(self):
        good = _response([_call()])
        with mock.patch.object(llm, "client_for",
                               return_value=_client(good)):
            res = llm.save_default_choice("alpha", "smart")
        self.assertEqual(res.model_id, "alpha-smart-2")
        self.assertEqual(llm.default_choice(), ("alpha", "smart"))

    def test_any_tool_call_is_accepted_not_one_hard_coded_name(self):
        # The gate asks for "a tool", so it is not coupled to one tool's name.
        good = _response([_call(name="run_command",
                                arguments='{"command": "echo hi"}')])
        with mock.patch.object(llm, "client_for",
                               return_value=_client(good)):
            res = llm.save_default_choice("alpha", "smart")
        self.assertEqual(res.model, "smart")

    def test_probe_false_skips_the_network_entirely(self):
        with mock.patch.object(llm, "client_for",
                               side_effect=AssertionError("must not be called")):
            llm.save_default_choice("alpha", "smart", probe=False)
        self.assertEqual(llm.default_choice(), ("alpha", "smart"))


class TheProbeSendsWhatAMissionSends(unittest.TestCase):
    """The probe must not test a lighter payload than the one that matters."""

    def setUp(self):
        for patcher in (
            mock.patch.object(llm, "PROVIDERS_FILE", Path("none.json")),
            mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "env-secret"}),
            mock.patch.object(llm, "_load_file", return_value=CATALOG),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_it_offers_the_real_tool_schema_and_forces_a_tool_call(self):
        seen = []
        good = _response([_call()])
        with mock.patch.object(llm, "client_for",
                               return_value=_client(good, seen=seen)):
            verdict = llm.probe_turn("alpha", "smart")
        self.assertTrue(verdict["ok"], verdict)
        request = seen[0]
        self.assertTrue(request.get("tools"), "the real tool schema must be sent")
        self.assertEqual(request["tool_choice"], "required",
                         "a model that only chats must not pass by ignoring tools")
        self.assertEqual(request["model"], "alpha-smart-2",
                         "it must call the WIRE id, not the alias")

    def test_the_probe_sends_the_models_real_budget_not_a_capped_one(self):
        # THE 2026-11-24 INCIDENT: a model's declared maxOutputTokens (262144,
        # copied from a sibling) was above the provider's limit (131072). The
        # first probe capped max_tokens at 512, so it PASSED, the model was
        # promoted, and every real mission turn was a 400. The probe must send
        # what a mission sends, or it cannot catch a refused capability.
        seen = []
        with mock.patch.object(llm, "client_for",
                               return_value=_client(_response([_call()]),
                                                    seen=seen)):
            llm.probe_turn("alpha", "smart")
        self.assertNotIn("max_tokens", seen[0],
                         "a plain-string alias declares no budget; the probe "
                         "must not invent a small one that a mission never sends")

    def test_a_declared_budget_the_provider_refuses_fails_the_probe(self):
        refused = json.loads(json.dumps(CATALOG))
        refused["providers"]["alpha"]["models"]["smart"] = {
            "id": "alpha-smart-2", "maxOutputTokens": 262144,
            "toolCalling": True}

        class _Refused(Exception):
            status_code = 400

        def create(**request):
            if request.get("max_tokens") == 262144:
                raise _Refused("The max_tokens parameter is illegal: "
                               "limit [1,131072]")
            return _response([_call()])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create)))
        with mock.patch.object(llm, "_load_file", return_value=refused), \
                mock.patch.object(llm, "client_for", return_value=client):
            verdict = llm.probe_turn("alpha", "smart")
        self.assertFalse(verdict["ok"], verdict)
        self.assertEqual(verdict["error"], "http")
        self.assertIn("max_tokens", verdict["detail"])

    def test_the_request_timeout_is_far_below_the_mission_timeout(self):
        seen_timeouts = []
        good = _response([_call()])

        def create(**request):
            seen_timeouts.append(request.get("timeout"))
            return good

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create)))
        with mock.patch.object(llm, "client_for", return_value=client):
            llm.probe_turn("alpha", "smart")
        self.assertEqual(len(seen_timeouts), 1)
        self.assertLess(seen_timeouts[0], config.API_TIMEOUT_SEC,
                        "a stall must cost seconds, not the mission timeout")
        self.assertGreater(seen_timeouts[0], 0)

    def test_the_probe_is_bounded_by_its_timeout_not_the_mission_one(self):
        self.assertLess(llm.PROBE_TIMEOUT_SEC, config.API_TIMEOUT_SEC)


class TheProbeNeverRaises(unittest.TestCase):
    def setUp(self):
        for patcher in (
            mock.patch.object(llm, "PROVIDERS_FILE", Path("none.json")),
            mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "env-secret"}),
            mock.patch.object(llm, "_load_file", return_value=CATALOG),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_an_unknown_model_is_reported_not_raised(self):
        verdict = llm.probe_turn("alpha", "no-such-model")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["error"], "exception")
        self.assertIn("no model", verdict["detail"])

    def test_an_http_error_is_classified_as_http(self):
        class _HTTP(Exception):
            status_code = 429
        with mock.patch.object(llm, "client_for",
                               return_value=_client(raises=_HTTP("rate limited"))):
            verdict = llm.probe_turn("alpha", "smart")
        self.assertEqual(verdict["error"], "http")
        self.assertFalse(verdict["ok"])

    def test_a_timeout_is_classified_as_timeout(self):
        with mock.patch.object(llm, "client_for",
                               return_value=_client(raises=_Timeout("x"))):
            verdict = llm.probe_turn("alpha", "smart")
        self.assertEqual(verdict["error"], "timeout")

    def test_a_model_declaring_no_tool_calling_is_refused_without_a_request(self):
        no_tools = json.loads(json.dumps(CATALOG))
        no_tools["providers"]["alpha"]["models"]["smart"] = {
            "id": "alpha-smart-2", "toolCalling": False}
        with mock.patch.object(llm, "_load_file", return_value=no_tools), \
                mock.patch.object(llm, "client_for",
                                  side_effect=AssertionError("must not be called")):
            verdict = llm.probe_turn("alpha", "smart")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["error"], "bad_response")
        self.assertIn("toolCalling", verdict["detail"])


if __name__ == "__main__":
    unittest.main()
