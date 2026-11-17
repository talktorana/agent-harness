"""Per-model capabilities must reach the wire, not stay documentation.

Owner request 2026-11-17: the remote harness must actually utilise a model's
declared maxInputTokens / maxOutputTokens / supportsReasoningEffort (with the
route effort held at xhigh), plus toolCalling / vision / thinking /
reasoningEffortFormat. Before this, providers.json model entries were plain
strings and every one of those fields was a comment -- the request always used
the run-wide AGENT_MAX_TOKENS and temperature 0.2 regardless of the model.

These pin the request builder (pure, no network) and the resolver/catalogue
(temp providers file), so a regression that drops a capability back to
"documentation" fails here.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent
import config
import llm


QWEN_CAPS = {
    "maxInputTokens": 1000000,
    "maxOutputTokens": 262144,
    "toolCalling": True,
    "vision": True,
    "thinking": True,
    "supportsReasoningEffort": ["none", "low", "high", "xhigh"],
    "reasoningEffortFormat": "chat-completions",
}


class BuildRequestCapabilitiesTest(unittest.TestCase):
    def _req(self, caps, effort=None, use_tools=True):
        with mock.patch.object(config, "MAX_TOKENS", 32000), \
                mock.patch.object(config, "REASONING_EFFORT", "low"), \
                mock.patch.object(config, "PROMPT_CACHE_KEY", ""):
            return agent.build_request("ali/qwen4-max", [], use_tools,
                                       reasoning_effort=effort, caps=caps)

    def test_max_output_tokens_overrides_global_budget(self):
        r = self._req(QWEN_CAPS, effort="xhigh")
        self.assertEqual(r["max_tokens"], 262144)

    def test_thinking_omits_temperature(self):
        self.assertNotIn("temperature", self._req(QWEN_CAPS, effort="xhigh"))

    def test_non_thinking_keeps_temperature(self):
        caps = dict(QWEN_CAPS, thinking=False)
        self.assertEqual(self._req(caps, effort="xhigh")["temperature"], 0.2)

    def test_supported_effort_is_sent_verbatim(self):
        r = self._req(QWEN_CAPS, effort="xhigh")
        self.assertEqual(r["reasoning_effort"], "xhigh")

    def test_unsupported_deep_effort_clamps_to_highest_supported(self):
        # A deep escalation asking for "max" must not be sent to a model whose
        # top level is xhigh; it clamps down to xhigh (always-xhigh invariant).
        r = self._req(QWEN_CAPS, effort="max")
        self.assertEqual(r["reasoning_effort"], "xhigh")

    def test_empty_supports_sends_no_effort(self):
        caps = dict(QWEN_CAPS, supportsReasoningEffort=[])
        self.assertNotIn("reasoning_effort", self._req(caps, effort="xhigh"))

    def test_tool_calling_false_withholds_tools(self):
        caps = dict(QWEN_CAPS, toolCalling=False)
        self.assertNotIn("tools", self._req(caps, effort="xhigh"))

    def test_tools_present_when_capable(self):
        self.assertIn("tools", self._req(QWEN_CAPS, effort="xhigh"))

    def test_no_caps_preserves_legacy_behaviour(self):
        r = self._req(None, effort=None)
        self.assertEqual(r["temperature"], 0.2)
        self.assertEqual(r["max_tokens"], 32000)
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertIn("tools", r)


class ClampEffortTest(unittest.TestCase):
    def test_clamps_down_to_highest_at_or_below(self):
        self.assertEqual(agent.clamp_effort("max", ["none", "low", "high", "xhigh"]), "xhigh")
        self.assertEqual(agent.clamp_effort("medium", ["none", "low", "high", "xhigh"]), "low")

    def test_below_all_supported_uses_lowest(self):
        self.assertEqual(agent.clamp_effort("none", ["low", "high"]), "low")

    def test_empty_supports_returns_empty(self):
        self.assertEqual(agent.clamp_effort("max", []), "")


class ResolveCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "providers.json"
        doc = {"providers": {"inferhub": {
            "base_url": "https://api.inferhub.dev/v1",
            "api_key": "test-key",
            "models": {
                "inferhub-qwen4-max": dict(QWEN_CAPS, id="ali/qwen4-max"),
                "inferhub-plain": "ali/plain-model",
            },
        }}}
        self.path.write_text(json.dumps(doc), encoding="utf-8")
        self._patches = [
            mock.patch.object(llm, "PROVIDERS_FILE", self.path),
            mock.patch.object(config, "BASE_URL", ""),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_object_entry_yields_wire_id_and_capabilities(self):
        res = llm.resolve("inferhub", "inferhub-qwen4-max")
        self.assertEqual(res.model_id, "ali/qwen4-max")
        self.assertEqual(res.capabilities["maxOutputTokens"], 262144)
        self.assertEqual(res.capabilities["supportsReasoningEffort"],
                         ["none", "low", "high", "xhigh"])

    def test_string_entry_stays_capability_free(self):
        res = llm.resolve("inferhub", "inferhub-plain")
        self.assertEqual(res.model_id, "ali/plain-model")
        self.assertEqual(res.capabilities, {})

    def test_raw_wire_id_recovers_capabilities(self):
        res = llm.resolve("inferhub", "ali/qwen4-max")
        self.assertEqual(res.capabilities["maxOutputTokens"], 262144)

    def test_catalog_exposes_ids_and_capabilities_key_free(self):
        cat = llm.catalog()
        inf = cat["providers"]["inferhub"]
        self.assertEqual(inf["model_ids"]["inferhub-qwen4-max"], "ali/qwen4-max")
        self.assertEqual(inf["capabilities"]["inferhub-qwen4-max"]["maxOutputTokens"], 262144)
        self.assertNotIn("api_key", json.dumps(cat))


if __name__ == "__main__":
    unittest.main()
