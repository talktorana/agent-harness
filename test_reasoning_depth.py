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


class DepthReachesTheWireTest(unittest.TestCase):
    """The payload dict is not the request. THIS is the request.

    ``build_request`` returning a dict containing ``reasoning_effort`` proves
    the dict has a key. The call site then does ``create(**payload)``, which
    splats the dict into a TYPED signature: a parameter the installed SDK does
    not model is dropped on the floor, and every assertion above still passes
    against a request that never asked for depth at all.

    That gap is not hypothetical, it is how this was first reported as working.
    So the check here stands up a loopback endpoint, points the REAL client at
    it through the same ``llm.Resolved``/``client_for`` path production uses,
    and asserts on the JSON body that actually arrived.
    """

    def setUp(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        captured = []
        self.captured = captured

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                captured.append(self.rfile.read(length))
                body = json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0,
                    "model": "capture",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _on_the_wire(self, effort, budget):
        import json

        import llm

        with mock.patch.object(config, "REASONING_EFFORT", effort), \
             mock.patch.object(config, "MAX_TOKENS", budget):
            request = agent.build_request("wire-model",
                                          [{"role": "user", "content": "hi"}],
                                          True)
            res = llm.Resolved(provider="capture", model="m",
                               model_id="wire-model",
                               base_url=f"http://127.0.0.1:{self.port}/v1",
                               api_key="k")
            llm.client_for(res).chat.completions.create(**request)
        return json.loads(self.captured[-1].decode("utf-8"))

    def test_the_effort_setting_survives_the_sdk_and_arrives(self):
        wire = self._on_the_wire("max", 32000)
        self.assertEqual(
            wire.get("reasoning_effort"), "max",
            "reasoning_effort never reached the wire -- the SDK dropped it, so "
            "every payload assertion above was testing a dict nobody sent")
        self.assertEqual(wire.get("max_tokens"), 32000)

    def test_an_unset_effort_sends_no_field_at_all(self):
        """A type-valid empty string would look configured and do nothing."""
        wire = self._on_the_wire("", 32000)
        self.assertNotIn("reasoning_effort", wire)


class ReasoningSpendIsRecordedTest(unittest.TestCase):
    """The provider's own token counts must reach the record.

    Without them "configured for max effort" is the only checkable claim, and a
    setting that does nothing looks identical to one that works -- which is how
    this was reported as done when it had never been observed.
    """

    def _agent(self, events):
        cls = agent.Agent
        inst = cls.__new__(cls)
        inst._transcript = None
        inst.on_event = events.append
        inst._log = lambda record: events.append(record)
        #: Modelled explicitly, and left as None by default. `_record_usage`
        #: writes BOTH sinks, so a helper that omits this attribute does not
        #: produce a passing test -- it produces an AttributeError that the
        #: suite reports as three unrelated failures.
        inst.session = None
        return inst

    def _usage(self, reasoning, completion=300, prompt=120):
        from types import SimpleNamespace
        return SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning))

    def test_the_reasoning_tokens_reach_the_record_with_the_setting(self):
        events = []
        inst = self._agent(events)
        with mock.patch.object(config, "REASONING_EFFORT", "max"), \
             mock.patch.object(config, "MAX_TOKENS", 32000):
            inst._record_usage(self._usage(153), "inferhub/flash")
        self.assertEqual(len(events), 1)
        rec = events[0]
        self.assertEqual(rec["event"], "model_usage")
        self.assertEqual(rec["reasoning_tokens"], 153)
        self.assertEqual(rec["completion_tokens"], 300)
        self.assertEqual(rec["reasoning_effort"], "max")
        self.assertEqual(rec["max_tokens"], 32000)
        self.assertEqual(rec["route"], "inferhub/flash")

    def test_a_provider_reporting_ZERO_reasoning_records_zero(self):
        """Zero is a finding, not a missing value.

        Dropping it would hide the one case worth acting on: effort configured,
        nothing spent.
        """
        events = []
        inst = self._agent(events)
        inst._record_usage(self._usage(0), "r")
        self.assertEqual(events[0]["reasoning_tokens"], 0)

    def test_no_usage_block_records_nothing_rather_than_a_fake_zero(self):
        """An absent block must not be reported as 0 -- that reads as a fault."""
        events = []
        inst = self._agent(events)
        inst._record_usage(None, "r")
        self.assertEqual(events, [])

    def test_a_provider_without_the_details_field_still_records_totals(self):
        from types import SimpleNamespace
        events = []
        inst = self._agent(events)
        inst._record_usage(
            SimpleNamespace(prompt_tokens=5, completion_tokens=7), "r")
        rec = events[0]
        self.assertEqual(rec["completion_tokens"], 7)
        self.assertNotIn("reasoning_tokens", rec)

    def test_streamed_usage_is_read_even_when_the_chunk_has_no_choices(self):
        """The usage chunk is choice-less BY DESIGN; reading it after the
        choices guard would discard the only counts the provider sends."""
        from types import SimpleNamespace

        seen = []
        usage = self._usage(42)
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content="hi", reasoning_content=None))]),
            # The real final chunk: usage present, choices empty.
            SimpleNamespace(choices=[], usage=usage),
        ]
        agent.assemble_stream(chunks, on_usage=seen.append)
        self.assertEqual(seen, [usage])

    def test_the_record_reaches_the_SESSION_FILE_not_only_the_transcript(self):
        """THE BUG THIS ACTUALLY SHIPPED WITH.

        `_log()` writes the transcript. The session `.jsonl` is a DIFFERENT file,
        written by `SessionLog.event()`, and it is the one `/agent/run/<id>` and
        every run-quality analysis read. The first version of this recorded
        model_usage through `_log` alone, so it never appeared in a session and
        the console showed no usage at all -- the record was being written and
        was invisible, which looks exactly like the fix not working.

        A test that stubs only `_log` cannot see the difference, which is why
        this one gives the agent a fake SessionLog and asserts on THAT.
        """
        class FakeSession:
            def __init__(self):
                self.events = []

            def event(self, kind, **fields):
                self.events.append({"event": kind, **fields})

        events = []
        inst = self._agent(events)
        inst.session = FakeSession()
        inst._record_usage(self._usage(153), "prov/alias")

        self.assertEqual(len(inst.session.events), 1,
                         "the session log never received a model_usage event")
        rec = inst.session.events[0]
        self.assertEqual(rec["event"], "model_usage")
        self.assertEqual(rec["reasoning_tokens"], 153)
        self.assertEqual(rec["route"], "prov/alias")
        #: And the transcript copy still happens: the two sinks answer
        #: different questions and neither replaces the other.
        self.assertTrue(any(e.get("event") == "model_usage" for e in events),
                        "the transcript copy was lost when the session one "
                        "was added")

    def test_a_session_that_raises_never_costs_the_call(self):
        """Logging must not be able to break a live turn."""
        class BoomSession:
            def event(self, *a, **k):
                raise RuntimeError("disk full")

        events = []
        inst = self._agent(events)
        inst.session = BoomSession()
        inst._record_usage(self._usage(11), "r")     # must not raise
        self.assertTrue(any(e.get("event") == "model_usage" for e in events))

    def test_complete_actually_calls_the_recorder(self):
        """THE CALL SITE, not the helper.

        Every test above passes if `_record_usage` works AND nobody calls it,
        which is the exact shape of the defect being fixed: the numbers were
        always available on the response and were simply thrown away. So this
        drives the real `_complete` through a stub client and asserts the
        record came out the other end.
        """
        from types import SimpleNamespace

        events = []
        inst = self._agent(events)
        usage = self._usage(153)
        message = SimpleNamespace(content="done", tool_calls=None,
                                  finish_reason="stop", reasoning_chars=0)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=usage)

        class Create:
            def create(self, **kwargs):
                return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=Create()))

        inst.route_pairs = [("prov", "alias")]
        inst.messages = [{"role": "user", "content": "hi"}]
        inst.verbose = False
        inst.on_token = None
        inst.session = None            # no per-run session log in a unit test
        inst._client_for = lambda _resolved: client
        with mock.patch.object(config, "STREAM", False), \
             mock.patch("llm.resolve",
                        return_value=SimpleNamespace(model_id="wire",
                                                     provider="prov",
                                                     model="alias")):
            inst._complete(use_tools=False)

        usage_events = [e for e in events
                        if isinstance(e, dict)
                        and e.get("event") == "model_usage"]
        self.assertEqual(len(usage_events), 1,
                         f"the recorder was never called; events={events}")
        self.assertEqual(usage_events[0]["reasoning_tokens"], 153)
        self.assertEqual(usage_events[0]["route"], "prov/alias")


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
