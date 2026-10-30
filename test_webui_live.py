"""Regressions for the live streaming chat surface (webui.py) and the
cancellation/event hooks it depends on in agent.py.

The page is stdlib-only on purpose, which means correctness lives entirely in
the task record the page polls -- so these tests drive the SERVER contract
rather than eyeballing HTML: a running task streams tokens into the record,
publishes structured events, survives a concurrent stop request, and never
leaks its internal agent handle through the API.

Run:  python -m unittest test_webui_live
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeAgent:
    """Stands in for Agent so the server contract is testable offline."""

    instances: list = []

    def __init__(self, verbose=True, on_token=None, on_event=None,
                 provider=None, model=None):
        self.verbose = verbose
        self.on_token = on_token
        self.on_event = on_event
        self.provider = provider
        self.model = model
        self.stop_requested = False
        self.started = threading.Event()
        FakeAgent.instances.append(self)

    def run(self, task):
        self.started.set()
        # Emit the exact shapes the real agent emits, in its order: the
        # model_call event is logged BEFORE the request, then tokens stream.
        if self.on_event:
            self.on_event({"event": "model_call", "model": "m",
                           "route": "prov/alias", "attempt": 1})
        for chunk in ("STREAM", "-", "OK"):
            if self.on_token:
                self.on_token(chunk)
            time.sleep(0.02)
        # Honour cooperative cancellation the way Agent.run does.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.stop_requested:
                return "STOPPED by user request."
            time.sleep(0.02)
        return "ANSWER"


class WebuiServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import webui
        cls.webui = webui
        cls.port = _free_port()
        cls.srv = ThreadingHTTPServer(("127.0.0.1", cls.port), webui.Handler)
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        # Route Agent construction to the fake.
        patcher = mock.patch("agent.Agent", FakeAgent)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        cls.addClassCleanup(cls.srv.shutdown)

    def setUp(self):
        FakeAgent.instances.clear()
        with self.webui._LOCK:
            self.webui._TASKS.clear()

    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def get(self, path):
        with urllib.request.urlopen(self.base() + path, timeout=10) as r:
            return r.status, r.read().decode("utf-8", "replace")

    def post(self, path, body=None):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            self.base() + path, data=data, method="POST",
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def post_task(self, text="hello"):
        code, body = self.post("/chat", {"text": text, "provider": "prov",
                                        "model": "alias"})
        self.assertEqual(code, 200, body)
        return json.loads(body)["id"]

    def task(self, task_id):
        code, body = self.get(f"/task/{task_id}")
        self.assertEqual(code, 200, body)
        return json.loads(body)

    def wait_for(self, task_id, pred, timeout=6.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rec = self.task(task_id)
            if pred(rec):
                return rec
            time.sleep(0.05)
        return self.task(task_id)

    # -- pages ---------------------------------------------------------

    def test_live_page_is_served_with_its_controls(self):
        code, body = self.get("/live")
        self.assertEqual(code, 200)
        for probe in ("id=\"stop\"", "id=\"regen\"", "id=\"copy\"",
                      "id=\"msgs\"", "agent activity", "route chain"):
            self.assertIn(probe, body, f"live page missing {probe!r}")

    def test_index_links_to_the_live_page(self):
        code, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("/live", body)

    def test_models_endpoint_is_key_free(self):
        code, body = self.get("/models")
        self.assertEqual(code, 200)
        self.assertNotIn("api_key\":", body.replace("api_key_env", ""))

    # -- streaming contract --------------------------------------------

    def test_running_task_exposes_streamed_tokens_and_events(self):
        task_id = self.post_task()
        rec = self.wait_for(task_id, lambda r: bool(r.get("stream")))
        self.assertIn("STREAM", rec.get("stream", ""))
        self.assertEqual(rec["status"], "running")
        # The activity feed the /live page renders comes from these events.
        events = rec.get("events") or []
        self.assertTrue(events, "no structured events published")
        self.assertIn("model_call", [e.get("event") for e in events])

    def test_stream_clears_and_answer_lands_on_completion(self):
        task_id = self.post_task()
        rec = self.wait_for(task_id,
                            lambda r: r.get("status") in ("done", "stopped"),
                            timeout=8.0)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(rec["answer"], "ANSWER")
        self.assertEqual(rec.get("stream", ""), "",
                         "a finished task must not keep a half-stream up")
        self.assertEqual(rec["lines"][-1]["text"], "ANSWER")

    def test_errors_surface_as_a_terminal_error_status(self):
        class Boom(FakeAgent):
            def run(self, task):
                self.started.set()
                raise RuntimeError("gateway exploded")

        with mock.patch("agent.Agent", Boom):
            task_id = self.post_task()
            rec = self.wait_for(task_id, lambda r: r["status"] == "error")
        self.assertEqual(rec["status"], "error")
        self.assertIn("gateway exploded", rec["lines"][-1]["text"])

    # -- cancellation ----------------------------------------------------

    def test_stop_requests_a_cooperative_cancel(self):
        task_id = self.post_task()
        rec = self.wait_for(task_id, lambda r: bool(r.get("stream")))
        code, body = self.post(f"/stop/{task_id}")
        self.assertEqual(code, 200, body)
        self.assertTrue(json.loads(body)["stopped"], body)
        rec = self.wait_for(task_id,
                            lambda r: r["status"] in ("stopped", "done"),
                            timeout=8.0)
        self.assertEqual(rec["status"], "stopped",
                         "a cancelled run must report stopped, not done")
        self.assertIn("STOPPED", rec["answer"])

    def test_stop_is_requested_not_forced(self):
        """Cancellation never kills a thread mid-tool."""
        task_id = self.post_task()
        rec = self.wait_for(task_id, lambda r: bool(r.get("stream")))
        self.post(f"/stop/{task_id}")
        agent = FakeAgent.instances[-1]
        self.assertTrue(agent.stop_requested)
        self.assertTrue(agent.started.is_set())
        # The run still gets to finish: status only goes terminal once the
        # agent itself returns, which is what makes stopping safe.
        self.wait_for(task_id, lambda r: r["status"] in ("stopped", "done"),
                      timeout=8.0)
        with self.webui._LOCK:
            self.assertIn(task_id, self.webui._TASKS)

    def test_unknown_and_finished_tasks_refuse_to_stop(self):
        code, body = self.post("/stop/doesnotexist")
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(body)["stopped"])

        task_id = self.post_task()
        self.wait_for(task_id, lambda r: r["status"] == "done", timeout=8.0)
        code, body = self.post(f"/stop/{task_id}")
        self.assertFalse(json.loads(body)["stopped"],
                         "a finished task cannot be stopped again")

    # -- hygiene ---------------------------------------------------------

    def test_task_payload_never_leaks_the_agent_handle(self):
        task_id = self.post_task()
        self.wait_for(task_id, lambda r: r["status"] == "done", timeout=8.0)
        code, body = self.get(f"/task/{task_id}")
        self.assertNotIn('"agent"', body)
        self.assertNotIn("FakeAgent", body)

    def test_event_feed_is_bounded_under_a_long_run(self):
        """A chatty task must not grow the poll payload without limit."""
        task_id = self.post_task()
        agent = None
        end = time.monotonic() + 5.0
        while time.monotonic() < end and agent is None:
            agent = FakeAgent.instances[-1] if FakeAgent.instances else None
            if agent is None:
                time.sleep(0.02)
                continue
            break
        self.assertIsNotNone(agent)

        def spam():
            for i in range(1200):
                if agent.on_event and not agent.stop_requested:
                    agent.on_event({"event": "noisy", "i": i})
                time.sleep(0)
            agent.stop_requested = True

        threading.Thread(target=spam, daemon=True).start()
        rec = self.wait_for(task_id,
                            lambda r: len(r.get("events") or []) > 0,
                            timeout=8.0)
        self.assertLessEqual(len(rec["events"]), 400,
                             "event feed grew past its documented cap")

    def test_empty_text_is_refused(self):
        code, body = self.post("/chat", {"text": "   "})
        self.assertEqual(code, 400, body)


class AgentStopSemanticsTest(unittest.TestCase):
    """Agent.run must bail BETWEEN turns, never mid-tool."""

    def _agent(self, **cfg_over):
        import agent as agent_mod
        import llm

        resolved = llm.Resolved(provider="prov", model="alias",
                                model_id="alias-id",
                                base_url="https://prov.example/v1",
                                api_key="k")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(llm, "route_chain",
                                              return_value=[("prov", "alias")]))
        stack.enter_context(mock.patch.object(llm, "resolve",
                                              return_value=resolved))
        stack.enter_context(mock.patch.object(llm, "client_for",
                                              return_value=object()))
        stack.enter_context(mock.patch.object(config, "STREAM", False))
        stack.enter_context(mock.patch.object(config, "REPO_MAP_ENABLED", False))
        for key, val in cfg_over.items():
            stack.enter_context(mock.patch.object(config, key, val))
        return agent_mod.Agent(verbose=False)

    def test_preset_stop_returns_before_any_model_call(self):
        a = self._agent(MAX_STEPS=5)
        calls = []
        a._complete = lambda use_tools=True: calls.append(use_tools)
        a.stop_requested = True
        out = a.run("do something")
        self.assertIn("STOPPED", out)
        self.assertEqual(calls, [], "no model call should be made once stopped")

    def test_stop_during_a_run_exits_at_the_next_step_boundary(self):
        a = self._agent(MAX_STEPS=8)
        calls = []
        ran = []
        # A tool call keeps the loop alive; a bare message would END the run as
        # the final answer before the next boundary is ever reached.
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="read_file", arguments="{}"))

        def fake_complete(use_tools=True):
            calls.append(use_tools)
            a.stop_requested = True
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="thinking", tool_calls=[tool_call]))])

        def fake_run_tool(name, args):
            ran.append(name)
            return "ran " + name, "ok"

        a._complete = fake_complete
        a._run_tool = fake_run_tool
        out = a.run("do something")
        self.assertIn("STOPPED", out)
        self.assertEqual(len(calls), 1,
                         "the loop must notice the stop on the very next turn")
        self.assertEqual(ran, ["read_file"],
                         "the in-flight tool finished; nothing was cut mid-call")

    def test_events_fan_out_to_the_ui_sink(self):
        seen = []
        a = self._agent(MAX_STEPS=1, TRANSCRIPT=None)
        a.on_event = seen.append
        a._log({"event": "probe", "detail": "x"})
        a._complete = lambda use_tools=True: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="done", tool_calls=None))])
        a.run("do something")
        kinds = [e.get("event") for e in seen]
        self.assertIn("probe", kinds)
        self.assertIn("task", kinds)

    def test_a_failing_event_sink_never_breaks_the_run(self):
        a = self._agent(MAX_STEPS=1, TRANSCRIPT=None)

        def boom(_evt):
            raise RuntimeError("ui sink exploded")

        a.on_event = boom
        a._log({"event": "probe"})
        a._complete = lambda use_tools=True: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="done", tool_calls=None))])
        out = a.run("do something")
        self.assertNotIn("exploded", out)


if __name__ == "__main__":
    unittest.main()
