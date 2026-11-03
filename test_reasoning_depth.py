"""Reasoning depth: the payload must carry what was configured, and nothing else.

WHY THIS EXISTS (2026-11-03). The deployed model is ``ali/deepseek-v4-pro-0813``
and a request to run it at "maximum reasoning" turned out to be two different
questions depending on the provider:

  * ``reasoning_effort`` is accepted by the gateway but only TYPE validated --
    it accepts the string "banana". On this model it was measured to do
    NOTHING: with one prompt and a 900-token budget, ``reasoning_tokens`` was
    900 for none, "low" and "max", and wall time differed by under two seconds.
  * the model thinks out of its COMPLETION budget, so ``max_tokens`` is what
    actually governs how long it reasons.

So the payload has to carry both, and must omit each when unset. Omission
matters as much as inclusion: a gateway that accepts a nonsense effort string
and ignores it means a typo silently buys nothing, and the agent would believe
it had been configured to think harder when it had not.
"""

import unittest
from unittest import mock

import agent
import config


class ReasonDepthPayloadTest(unittest.TestCase):
    def _build(self, **env):
        patches = [mock.patch.object(config, "MAX_TOKENS",
                                     env.get("max_tokens", 0)),
                   mock.patch.object(config, "REASONING_EFFORT",
                                     env.get("effort", ""))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return agent.build_request("some-model", [{"role": "user"}], True)

    def test_both_settings_reach_the_wire_when_configured(self):
        req = self._build(max_tokens=32000, effort="max")
        self.assertEqual(req["max_tokens"], 32000)
        self.assertEqual(req["reasoning_effort"], "max")

    def test_an_unset_effort_is_OMITTED_not_sent_empty(self):
        """An empty string would be type-valid at the gateway and do nothing.

        Sending it would put ``reasoning_effort: ""`` on the wire and in the
        receipt of a request that asked for depth -- indistinguishable from a
        working setting until somebody reads the response.
        """
        req = self._build(max_tokens=8000, effort="")
        self.assertNotIn("reasoning_effort", req)
        self.assertEqual(req["max_tokens"], 8000)

    def test_an_unset_budget_is_OMITTED_so_the_provider_default_applies(self):
        req = self._build(max_tokens=0, effort="max")
        self.assertNotIn("max_tokens", req)
        self.assertEqual(req["reasoning_effort"], "max")

    def test_zero_or_negative_budget_never_sends_a_useless_value(self):
        for bad in (0, -1):
            with self.subTest(max_tokens=bad):
                req = self._build(max_tokens=bad)
                self.assertNotIn("max_tokens", req)

    def test_the_tool_surface_is_unchanged_by_the_depth_settings(self):
        """Depth must not accidentally start offering tools on a final answer."""
        req = self._build(max_tokens=4000, effort="max")
        self.assertIn("tools", req)
        self.assertEqual(req["tool_choice"], "auto")
        no_tools = agent.build_request("m", [{"role": "user"}], False)
        self.assertNotIn("tools", no_tools)
        self.assertNotIn("tool_choice", no_tools)


class TruncatedTurnIsRefusedTest(unittest.TestCase):
    """A turn that spent its whole budget reasoning must not read as "done".

    The loop treats "no tool calls" as "the model has finished". A capped
    reasoning turn returns exactly that shape -- finish_reason "length", no
    content, no tool calls -- so before this guard, setting a too-small
    AGENT_MAX_TOKENS produced a silent empty answer that looked like a clean
    stop. That becomes reachable the moment the budget is set explicitly, which
    is precisely what asking for deeper reasoning requires.
    """

    def _agent(self):
        cls = agent.Agent
        inst = cls.__new__(cls)
        inst._log = lambda *a, **k: None      # no session file in a unit test
        return inst

    def _msg(self, finish_reason, content=None, tool_calls=None,
             reasoning_chars=0):
        from types import SimpleNamespace
        return SimpleNamespace(finish_reason=finish_reason, content=content,
                               tool_calls=tool_calls,
                               reasoning_chars=reasoning_chars)

    def test_a_capped_turn_with_nothing_to_show_is_refused(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._agent()._refuse_truncated_turn(self._msg("length"), "route")
        self.assertIn("token cap", str(ctx.exception))
        self.assertIn("AGENT_MAX_TOKENS", str(ctx.exception),
                      "the error must name the setting to change")

    def test_a_capped_turn_that_still_answered_is_KEPT(self):
        """A partial answer is real work; discarding it would lose it."""
        self._agent()._refuse_truncated_turn(
            self._msg("length", content="partial answer"), "route")

    def test_a_capped_turn_that_still_called_a_tool_is_KEPT(self):
        """The tool call is usable even if the prose was cut off."""
        self._agent()._refuse_truncated_turn(
            self._msg("length", tool_calls=[object()]), "route")

    def test_a_normal_finish_is_never_touched(self):
        for fr in ("stop", "tool_calls", None):
            with self.subTest(finish_reason=fr):
                self._agent()._refuse_truncated_turn(self._msg(fr), "route")

    def test_assemble_stream_reports_why_the_turn_ended(self):
        """Otherwise the guard has nothing to read."""
        from types import SimpleNamespace

        def chunk(finish=None, content=None):
            delta = SimpleNamespace(content=content, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(
                delta=delta, finish_reason=finish)])

        msg = agent.assemble_stream([chunk(content="hi"),
                                     chunk(finish="length")])
        self.assertEqual(msg.finish_reason, "length")
        self.assertEqual(msg.content, "hi")


if __name__ == "__main__":
    unittest.main()
