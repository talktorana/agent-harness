"""A run must not be able to hold the schedule indefinitely.

The scheduler fires jobs ONE AT A TIME. That is correct -- two concurrent
missions would both write edits and both judge the same defects -- but it means
the length of one run is the length of every mission behind it. Three facts
about a live deployment on 2026-11-03 made that arithmetic visible:

  * a mission had been running 75 minutes, 14 model timeouts, and had reached
    only turn 62 of 200;
  * it never once fell back: ``AGENT_ROUTES`` is a chain, but a timeout is
    classified transient, so the slow primary was retried forever and
    ``route_switch`` never fired;
  * with ``next_due`` 40 minutes in the past, the console reported the job as
    **overdue** with no explanation, and the operator's reading -- an entirely
    reasonable one -- was "nothing seems to be running, the next scheduled
    mission never started, major defect".

Nothing was broken in the sense of a crashed process. What was missing was a
BOUND, a fallback that actually fires, and a sentence saying which of those it
was. Each of the four below is the repair for one of those gaps.
"""
from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from openai import APITimeoutError

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import config  # noqa: E402
import recreate_guard as RG  # noqa: E402
import scheduler  # noqa: E402
import webui  # noqa: E402
from test_webui_runs import TempDirs  # noqa: E402


# --------------------------------------------------------------- fake clocks


def _real_timeout():
    """The SDK exception a slow provider actually raises.

    A plain ``TimeoutError`` would be simpler, but `_is_transient` only accepts
    the openai connection/timeout types -- so a test that raised the builtin
    would never reach the retry path at all, and would pass against a demotion
    rule that the production exception never triggers. Using the real type is
    what makes this test say something about deployed behaviour.
    """
    return APITimeoutError(httpx.Request("POST", "https://prov.example/v1"))


class FakeClock:
    """A clock the test advances, so no test has to wait for real seconds.

    ``time.monotonic`` is what the bound reads, and patching the module-level
    ``agent.time`` is what makes a 1000-second ceiling testable in
    milliseconds. Waiting real seconds would make this the slowest module in the
    suite and, worse, encourage a smaller bound in the test than in production
    -- a bound that is only ever checked at 1s is a bound nobody has tested.
    """

    def __init__(self, step_sec: float = 300.0):
        self.now = 0.0
        self.step_sec = step_sec

    def advance(self):
        self.now += self.step_sec

    def monotonic(self):
        return self.now

    def sleep(self, _s):
        return None

    def time(self):
        return 1_700_000_000.0 + self.now

    def perf_counter(self):
        return self.now


# ------------------------------------------------------------- fake sessions


class FakeSession:
    """Everything ``_run_impl`` asks of a session, recording it."""

    counter = 0

    def __init__(self, task="", sessions_dir=None, model="", base_url="",
                 origin=""):
        FakeSession.counter += 1
        self.session_id = "s%07d" % FakeSession.counter
        self.task, self.model, self.origin = task, model, origin
        self.md_path = Path("fake-%s.md" % self.session_id)
        self.jsonl_path = Path("fake-%s.jsonl" % self.session_id)
        self.events = []
        self.answers = []
        self.notes = []
        self.finished = None

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def action(self, *a, **k):
        self.events.append(("action", k))

    def assistant(self, text):
        self.notes.append(text)

    def model_call(self, *a, **k):
        pass

    def api_retry(self, *a, **k):
        pass

    def api_error(self, *a, **k):
        pass

    def answer(self, text):
        self.answers.append(text)

    def finish(self, status="ok"):
        self.finished = status


def bare_agent_with_loop(clock, *, max_steps=9, warn_steps=1,
                         route_pairs=None, complete=None,
                         salvaged_answer="the salvaged report"):
    """An Agent wired for the loop, with no network and no real session file.

    Built through ``__new__`` rather than ``Agent(...)``: the constructor
    resolves the route chain and needs a configured provider, and a test for
    "does the run stop when the clock runs out" must not depend on a key being
    present on the machine running the suite.
    """
    inst = agent.Agent.__new__(agent.Agent)
    session = FakeSession()
    inst.session = session
    inst.routes = route_pairs or [("prov", "alias")]
    inst.route_pairs = inst.routes
    inst.resolved = SimpleNamespace(
        provider=(inst.routes[0][0]), model=(inst.routes[0][1]),
        model_id="wire-1", base_url="https://prov.example/v1", api_key="k",
        label=lambda: "%s/%s" % inst.routes[0])
    inst.client = object()
    inst._route_clients = {}
    inst.messages = []
    inst.verbose = False
    inst.on_token = None
    inst.on_event = None
    inst.stop_requested = False
    inst.steps = 0
    inst.max_steps = 0
    inst.origin = ""
    inst._transcript = None
    inst._thinking_route = None
    inst._thinking_chars = 0
    inst._last_think_event = 0.0
    inst._run_started_monotonic = clock.monotonic()
    inst._route_timeouts = {}
    inst._run_elapsed_at_stop = None
    inst.logged = []
    inst._log = lambda record: inst.logged.append(record)
    inst._say = lambda *a, **k: None
    inst._close = lambda: None
    inst._bound_storage = lambda: None
    #: Tools are stubbed for the same reason the session is: what is under test
    #: is the loop's arithmetic, and a real dispatcher would write audit files
    #: into whatever directory the ambient environment points at.
    inst._run_tool = lambda name, args: ("ok", "tool said ok")

    calls = {"n": 0}
    if complete is None:
        def complete(use_tools=True):          # noqa: ARG001
            calls["n"] += 1
            clock.advance()
            if calls["n"] > 30:                # never let a test spin forever
                raise AssertionError("the loop did not stop")
            tc = SimpleNamespace(
                id="call_%d" % calls["n"],
                function=SimpleNamespace(name="stub_tool", arguments="{}"))
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="thinking",
                                        tool_calls=[tc]))])
    inst._complete = complete
    inst._salvaged = salvaged_answer

    def _final(reason="You have run out of tool steps."):
        inst.final_reason = reason
        return salvaged_answer

    inst._final_answer = _final
    return inst


def run_loop(inst, clock, *, max_run_sec, max_steps=9, warn_steps=1):
    """Drive the real loop with the given clock and ceiling."""
    patches = [
        mock.patch.object(config, "MAX_RUN_SEC", max_run_sec),
        mock.patch.object(config, "REPO_MAP_ENABLED", False),
        mock.patch.object(agent, "time", clock),
        mock.patch.object(agent, "SessionLog", FakeSession),
        mock.patch.object(agent, "_effective_max_steps", lambda: max_steps),
        mock.patch.object(agent, "_effective_answer_warn_steps",
                          lambda _m: warn_steps),
        mock.patch.object(agent, "build_system",
                          lambda *a, **k: "SYSTEM PROMPT"),
        mock.patch.object(agent, "build_repo_map", lambda: ""),
        mock.patch.object(agent, "MemoryStore",
                          lambda session_id=None: SimpleNamespace(
                              head=lambda kind: "",
                              path_for=lambda kind: Path("mem"),
                              remember=lambda *a, **k: None)),
        mock.patch.object(agent, "set_memory_session", lambda sid: None),
        mock.patch.object(agent, "storage",
                          SimpleNamespace(
                              rotate_transcript=lambda *a, **k: None)),
    ]
    for p in patches:
        p.start()
    try:
        return inst.run("the mission")
    finally:
        for p in reversed(patches):
            p.stop()


# --------------------------------------------------------- 1. the wall clock


class WallClockBoundTest(unittest.TestCase):
    """A run ends when its time is up, whatever turn it is on."""

    def test_a_run_past_its_ceiling_stops_while_turns_remain(self):
        """The defect: MAX_STEPS bounded turns, and nothing bounded minutes.

        62 of 200 turns after 75 minutes is a run that would have kept going for
        hours -- with a sequential scheduler, hours during which no other
        mission can start and the console can only say "overdue".
        """
        clock = FakeClock(step_sec=300.0)
        inst = bare_agent_with_loop(clock)
        out = run_loop(inst, clock, max_run_sec=1000.0)

        self.assertEqual(inst.session.finished, "time_limit_with_answer",
                         "a time cut-off must not be reported as a step "
                         "cut-off; the next pass needs to know the clock, not "
                         "the turns, was the limit")
        self.assertEqual(out, "the salvaged report")
        self.assertLess(inst.steps, 9,
                        "the run must stop EARLY, with turns to spare")
        kinds = [name for name, _ in inst.session.events]
        self.assertIn("route_health", kinds)

    def test_the_time_limit_is_recorded_with_the_budget_that_caused_it(self):
        clock = FakeClock(step_sec=300.0)
        inst = bare_agent_with_loop(clock)
        run_loop(inst, clock, max_run_sec=1000.0)
        hit = [r for r in inst.logged if r.get("event") == "time_limit"]
        self.assertEqual(len(hit), 1, inst.logged[-6:])
        self.assertEqual(hit[0]["budget_sec"], 1000.0)
        self.assertGreaterEqual(hit[0]["elapsed_sec"], 1000.0)

    def test_the_model_is_warned_before_the_clock_runs_out(self):
        """A warning that arrives after the bound is not a warning.

        This is the same reasoning as the step warning firing 20 turns out: the
        salvage call has to fit inside what is left, or the run ends with
        nothing to deliver.
        """
        clock = FakeClock(step_sec=300.0)
        inst = bare_agent_with_loop(clock)
        run_loop(inst, clock, max_run_sec=1000.0)
        warned = [r for r in inst.logged if r.get("event") == "time_budget_warning"]
        hits = [r for r in inst.logged if r.get("event") == "time_limit"]
        self.assertTrue(warned, "the run was never told its time was short")
        self.assertTrue(hits)
        self.assertLess(warned[0]["elapsed_sec"], hits[0]["elapsed_sec"],
                        "the warning must precede the cut-off, not follow it")
        self.assertIn("You have spent your 1000s wall-clock budget",
                      getattr(inst, "final_reason", ""))

    def test_a_bound_of_zero_leaves_a_long_run_alone(self):
        """`0` means unbounded, so this must not be a silent regression.

        A deployment that wants a three-hour mission has to be able to have one;
        the bound is a choice, not an assumption.
        """
        clock = FakeClock(step_sec=60.0)
        calls = {"answered": False}

        def complete(use_tools=True):          # noqa: ARG001
            calls["answered"] = True
            clock.advance()
            #: No tool calls: the model is answering, so the loop ends.
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="the answer",
                                        tool_calls=None))])

        inst = bare_agent_with_loop(clock, complete=complete)
        out = run_loop(inst, clock, max_run_sec=0.0)
        self.assertTrue(calls["answered"])
        self.assertEqual(out, "the answer")
        self.assertEqual(inst.session.finished, "ok")

    def test_a_run_that_finishes_in_time_is_untouched(self):
        clock = FakeClock(step_sec=1.0)

        def complete(use_tools=True):          # noqa: ARG001
            clock.advance()
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None))])

        inst = bare_agent_with_loop(clock, complete=complete)
        out = run_loop(inst, clock, max_run_sec=1000.0)
        self.assertEqual(out, "done")
        self.assertEqual(inst.session.finished, "ok")
        self.assertIsNone(inst._run_elapsed_at_stop)

    def test_nothing_is_salvageable_when_the_time_runs_out(self):
        clock = FakeClock(step_sec=300.0)
        inst = bare_agent_with_loop(clock, salvaged_answer="")
        out = run_loop(inst, clock, max_run_sec=1000.0)
        self.assertEqual(inst.session.finished, "time_limit",
                         "an overrun with no answer is a distinct outcome from "
                         "one that still delivered")
        self.assertEqual(inst.session.answers, ["[no answer salvaged]"])
        self.assertIn("wall-clock budget", out)


# --------------------------------------------------- 2. routes that keep timing out


class SlowRouteIsSteppedOverTest(unittest.TestCase):
    """A chain is only a fallback if something makes the chain move."""

    def _inst(self, failing_routes, working_route_response):
        inst = agent.Agent.__new__(agent.Agent)
        inst.session = FakeSession()
        inst.route_pairs = [("p1", "m1"), ("p2", "m2")]
        inst.resolved = SimpleNamespace(provider="p1", model="m1",
                                        model_id="wire-1",
                                        base_url="https://p1.example/v1",
                                        api_key="k", label=lambda: "p1/m1")
        inst.messages = []
        inst.verbose = False
        inst.on_token = None
        inst.on_event = None
        inst._transcript = None
        inst.logged = []
        inst._log = lambda r: inst.logged.append(r)
        inst._say = lambda *a, **k: None
        inst.steps = 0
        inst._thinking_route = None
        inst._thinking_chars = 0
        inst._last_think_event = 0.0
        inst._route_timeouts = {}
        inst.client = None

        def client_for(resolved):
            route = "%s/%s" % (resolved.provider, resolved.model)
            if route in failing_routes:
                class _Bad:
                    class chat:
                        class completions:
                            @staticmethod
                            def create(**_kw):
                                raise _real_timeout()
                return _Bad()

            class _Good:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**_kw):
                            return working_route_response
            return _Good()

        inst._client_for = client_for
        return inst

    def test_the_fourth_timeout_moves_the_run_to_the_next_route(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="from p2", tool_calls=None))])
        inst = self._inst({"p1/m1"}, response)

        resolved_map = {"p1/m1": SimpleNamespace(
            provider="p1", model="m1", model_id="wire-1",
            base_url="https://p1.example/v1", api_key="k",
            label=lambda: "p1/m1"),
            "p2/m2": SimpleNamespace(
                provider="p2", model="m2", model_id="wire-2",
                base_url="https://p2.example/v1", api_key="k",
                label=lambda: "p2/m2")}

        def resolve(prov, alias):
            return resolved_map["%s/%s" % (prov, alias)]

        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 4), \
                mock.patch.object(config, "API_RETRIES", 6), \
                mock.patch.object(config, "API_BACKOFF_SEC", 0), \
                mock.patch.object(agent.llm, "resolve", resolve), \
                mock.patch.object(agent, "build_request",
                                  lambda *a, **k: {}), \
                mock.patch.object(agent.time, "sleep", lambda s: None):
            out = inst._complete(use_tools=False)

        self.assertEqual(out.choices[0].message.content, "from p2",
                         "the chain must reach the second route; a slow primary "
                         "that never yields is what held the mission for 75 min")
        dead = [r for r in inst.logged if r.get("event") == "route_dead"
                and r.get("route") == "p1/m1"]
        self.assertTrue(dead, "no route_dead was recorded for the slow route")
        self.assertEqual(dead[0]["timeouts"], 4)
        self.assertIn("4 timeouts", dead[0]["reason"])
        self.assertEqual(inst._route_timeouts, {"p1/m1": 4})

    def test_fewer_timeouts_than_the_limit_keeps_retrying_the_route(self):
        """Demotion must not fire on the first hiccup.

        A transient slowdown is exactly what retries are for; stepping to
        another route after one timeout would throw away a perfectly good
        primary and make the chain's order meaningless.
        """
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="from p2", tool_calls=None))])
        inst = self._inst({"p1/m1"}, response)
        resolved_map = {
            "p1/m1": SimpleNamespace(provider="p1", model="m1",
                                     model_id="w1",
                                     base_url="https://p1.example/v1",
                                     api_key="k", label=lambda: "p1/m1"),
            "p2/m2": SimpleNamespace(provider="p2", model="m2",
                                     model_id="w2",
                                     base_url="https://p2.example/v1",
                                     api_key="k", label=lambda: "p2/m2")}
        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 4), \
                mock.patch.object(config, "API_RETRIES", 2), \
                mock.patch.object(config, "API_BACKOFF_SEC", 0), \
                mock.patch.object(agent.llm, "resolve",
                                  lambda p, a: resolved_map["%s/%s" % (p, a)]), \
                mock.patch.object(agent, "build_request",
                                  lambda *a, **k: {}), \
                mock.patch.object(agent.time, "sleep", lambda s: None):
            inst._complete(use_tools=False)

        self.assertEqual(inst._route_timeouts, {"p1/m1": 2})
        self.assertNotIn("route_dead",
                         [r.get("event") for r in inst.logged],
                         "two timeouts under a limit of four must not demote")

    def test_a_limit_of_zero_disables_demotion(self):
        self.assertTrue(hasattr(config, "ROUTE_TIMEOUT_LIMIT"))
        self.assertTrue(agent._is_timeout(_real_timeout()))
        self.assertFalse(agent._is_timeout(ValueError("bad payload")),
                         "a refusal is not a timeout: it is not the route being "
                         "slow, it is the request being wrong")

    def test_zero_disables_the_demotion_rule_entirely(self):
        """`0` must leave the old behaviour in place, not demote at once.

        Without this a deployment that wants retries forever gets a route
        abandoned after a single slow call -- the opposite error to the one
        being fixed.
        """
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="slow but fine", tool_calls=None))])
        inst = self._inst({"p1/m1"}, response)
        #: p1 always times out, so the only way this returns is via the chain.
        inst.route_pairs = [("p1", "m1")]
        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 0), \
                mock.patch.object(config, "API_RETRIES", 3), \
                mock.patch.object(config, "API_BACKOFF_SEC", 0), \
                mock.patch.object(agent.llm, "resolve",
                                  lambda p, a: SimpleNamespace(
                                      provider=p, model=a, model_id="w",
                                      base_url="https://%s.example/v1" % p,
                                      api_key="k",
                                      label=lambda: "%s/%s" % (p, a))), \
                mock.patch.object(agent, "build_request",
                                  lambda *a, **k: {}), \
                mock.patch.object(agent.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError, msg="no route available"):
                inst._complete(use_tools=False)
        self.assertNotIn("route_dead",
                         [r.get("event") for r in inst.logged],
                         "a disabled limit must not demote a route")
        self.assertIn(inst._route_timeouts.get("p1/m1"), (None, 0),
                      "timeouts are not counted once demotion is off")

    def test_the_timeout_detector_matches_what_providers_actually_send(self):
        """The messages observed in production, not invented ones.

        The live session recorded `"Request timed out."` fourteen times, so that
        exact string has to classify as a timeout or none of the demotion logic
        would ever fire on this deployment.
        """
        self.assertTrue(agent._is_timeout(_real_timeout()))
        for text in ("Connection timed out", "read timeout", "timeout"):
            self.assertTrue(agent._is_timeout(RuntimeError(text)), text)
        self.assertTrue(agent._is_timeout(TimeoutError("plain socket")))
        self.assertFalse(agent._is_timeout(RuntimeError("rate limited")))
        self.assertFalse(agent._is_timeout(RuntimeError("connection refused")))


# ------------------------------------------------- 3. saying why it is late


class OverdueReasonTest(TempDirs):
    """Three situations report `overdue: true`; only one is a fault."""

    def test_a_run_in_flight_makes_lateness_expected_not_broken(self):
        #: `running` is defined as finished BEHIND last_run -- last_run is
        #: stamped when the job fires, finished only when it returns.
        self.write_state(last_run="2020-01-01T00:10:00",
                         finished="2020-01-01T00:00:00")
        job = scheduler.status()["jobs"][0]
        self.assertTrue(job["running"], "finished == last_run must read as a run "
                                      "still in flight")
        self.assertTrue(job["overdue"])
        self.assertIn("SEQUENTIAL", job["overdue_reason"])
        self.assertIn("not lost", job["overdue_reason"],
                      "the reason must answer the question the page raises: "
                      "is the next firing missed or merely waiting?")
        self.assertNotIn("IS a fault", job["overdue_reason"])

    def test_nothing_running_and_late_is_named_as_a_fault(self):
        #: last_run and finished BOTH old and finished >= last_run => not
        #: running, so lateness has no benign explanation.
        self.write_state(last_run="2020-01-01T00:00:00",
                         finished="2020-01-01T00:05:00")
        job = scheduler.status()["jobs"][0]
        self.assertFalse(job["running"])
        self.assertTrue(job["overdue"])
        self.assertIn("IS a fault", job["overdue_reason"])
        self.assertIn("not polling", job["overdue_reason"])

    def test_an_operator_hold_is_named_as_the_reason(self):
        self.write_state(last_run="2020-01-01T00:00:00",
                         finished="2020-01-01T00:05:00")
        self.hold.write_text('{"reason": "investigating"}', encoding="utf-8")
        doc = scheduler.status()
        job = doc["jobs"][0]
        self.assertTrue(doc["paused"])
        self.assertIn("hold file", job["overdue_reason"])
        self.assertNotIn("IS a fault", job["overdue_reason"])

    def test_an_on_time_job_offers_no_reason_at_all(self):
        self.write_job(every_min=1)
        job = scheduler.status()["jobs"][0]
        self.assertFalse(job["overdue"])
        self.assertIsNone(job["overdue_reason"],
                        "a reason on a healthy job is noise, and noise in the "
                        "one field an operator scans gets ignored")

    def test_the_console_carries_the_reason_not_just_the_scheduler(self):
        self.write_state(last_run="2020-01-01T00:10:00",
                         finished="2020-01-01T00:00:00")
        view = webui._scheduler_view()
        self.assertTrue(view["available"])
        self.assertTrue(view["overdue"])
        self.assertIn("SEQUENTIAL", view["overdue_reason"])

    def test_the_runs_page_prints_the_reason(self):
        """A payload field no page shows is a field nobody reads."""
        self.assertIn("overdue_reason", webui.RUNS_PAGE)

    def test_the_console_reports_which_cadence_is_governing(self):
        self.write_job(every_min=30)
        with mock.patch.object(scheduler.run_settings, "effective_every_min",
                               return_value=(60, "console")):
            view = webui._scheduler_view()
        self.assertEqual(view["every_min"], 60)
        self.assertEqual(view["every_min_source"], "console",
                         "the override that governs must be visible next to the "
                         "spacing it produces")


# ------------------------------------ 4. a sanctioned way to end a wedged run


def sched(running=False, available=True, in_progress_since=None,
          in_progress_sec=None, reason=None):
    return {"available": available, "running": running,
            "in_progress_since": in_progress_since,
            "in_progress_sec": in_progress_sec, "reason": reason}


def session(has_end=True, started="2026-11-03T22:27:52", readable=True):
    return {"file": "2026-11-03T22-27-52_dead.jsonl", "started": started,
            "has_session_end": has_end, "readable": readable,
            "host_path": "/tmp/x.jsonl"}


class ForceRecreateTest(unittest.TestCase):
    """`--force` exists for the case the wait loop cannot solve."""

    #: Generic placeholder names on purpose: this repo is the shared harness,
    #: and `test_genericity` fails a generic file that names a specific
    #: deployment -- including in a comment, which is how its own guard caught
    #: this line on the first run.
    APPLY = ["--apply", "--container", "agent-sidecar",
             "--compose-dir", "/d", "--compose-file", "c.yml"]

    def _main(self, argv, out=None):
        buf = out if out is not None else io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            rc = RG.main(argv)
        return rc, buf.getvalue()

    def test_without_force_a_live_run_still_blocks_everything(self):
        """The existing safety is not weakened: default stays exit 3."""
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_since="2026-11-03T22:27:52",
                                                   in_progress_sec=4300),
                                             session())):
            rc, _ = self._main(self.APPLY + ["--wait-sec", "0"])
        self.assertEqual(rc, 3)

    def test_force_proceeds_past_a_live_run(self):
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_since="2026-11-03T22:27:52",
                                                   in_progress_sec=4300),
                                             session())), \
                mock.patch.object(RG, "sh", return_value=(0, "recreated")), \
                mock.patch.object(RG, "container_started",
                                  return_value="2026-11-03T22:20:51"), \
                mock.patch.object(RG, "verify", return_value=0):
            out = io.StringIO()
            rc, _ = self._main(self.APPLY + ["--force"], out=out)
        self.assertEqual(rc, 0)
        self.assertIn("FORCE", out.getvalue())

    def test_force_says_exactly_what_it_destroyed(self):
        """The log is the only record that a run ended because of this command.

        Without the facts -- which session, and how long it had been going -- a
        forced recreate is indistinguishable from the accidental kills this
        guard exists to prevent, and the next pass inherits a no_end_record it
        has to investigate all over again.
        """
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_since="2026-11-03T22:27:52",
                                                   in_progress_sec=4300),
                                             session())), \
                mock.patch.object(RG, "sh", return_value=(0, "recreated")), \
                mock.patch.object(RG, "container_started",
                                  return_value="2026-11-03T22:20:51"), \
                mock.patch.object(RG, "verify", return_value=0):
            out = io.StringIO()
            self._main(self.APPLY + ["--force"], out=out)
        said = out.getvalue()
        self.assertIn("2026-11-03T22:27:52", said)
        self.assertIn("4300", said)
        self.assertIn("2026-11-03T22-27-52_dead.jsonl", said)
        self.assertIn("DESTROY", said)

    def test_force_does_not_mean_acting_blind(self):
        """If the scheduler cannot answer, nobody can say what would be killed.

        An unreachable console is a REFUSE, not a wait, so ``--force`` never
        even reaches the destructive path -- which is the point: forcing is
        scoped to "a run is in flight" and nothing else.
        """
        with mock.patch.object(RG, "probe",
                               return_value=(sched(available=False,
                                                   reason="console down"),
                                             session())):
            out = io.StringIO()
            rc, _ = self._main(self.APPLY + ["--force"], out=out)
        self.assertEqual(rc, 2)
        said = out.getvalue()
        self.assertIn("cannot answer", said)
        self.assertIn("REFUSING", said)
        self.assertEqual(RG.decide(sched(available=False), None)[0], "refuse",
                         "force must not be able to convert an unknown into an "
                         "idle verdict")

    def test_force_still_runs_the_build_and_the_verify(self):
        """Forcing bypasses the WAIT, not the safety checks around it."""
        calls = []
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_sec=10),
                                             session())), \
                mock.patch.object(RG, "sh",
                                  side_effect=lambda *a, **k: (
                                      calls.append((a or k.get("cwd"))[0])
                                      or (0, "ok"))), \
                mock.patch.object(RG, "container_started",
                                  return_value="before"), \
                mock.patch.object(RG, "verify", return_value=0):
            rc, _ = self._main(["--build", "--build-context", "/ctx",
                                "--image", "agent-harness:1.0.0",
                                "--container", "c",
                                "--compose-dir", "/d",
                                "--compose-file", "c.yml",
                                "--apply", "--force"])
        self.assertEqual(rc, 0)
        self.assertTrue(any("build" in str(c) for c in calls), calls)

    def test_describe_run_names_the_facts_it_has(self):
        text = RG.describe_run(sched(running=True,
                                     in_progress_since="2026-11-03T22:27:52",
                                     in_progress_sec=4300), session())
        self.assertIn("4300", text)
        self.assertIn("2026-11-03T22-27-52_dead.jsonl", text)
        self.assertEqual(RG.describe_run(sched(running=False), None),
                         "no run in flight")


if __name__ == "__main__":
    unittest.main(verbosity=2)
