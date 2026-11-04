"""Per-run route demotion must PERSIST across turns, not just the turn it fires.

Regression for the defect measured live 2026-11-04 14:29: a route hit its
per-run timeout limit, ``route_dead`` fired, the run failed over correctly --
then the VERY NEXT turn called the same dead route again from attempt 1. A
persistently slow primary was therefore re-paid (up to API_RETRIES x
API_TIMEOUT_SEC) every turn and the run never really left it: a 4-tool pass that
normally runs ~90 s took 535 s. ``ROUTE_TIMEOUT_LIMIT`` is documented (config.py
and the deployment brief) as "steps over a route after that many timeouts IN
THIS RUN" -- i.e. demotion is per-RUN, but was implemented per-TURN.

Also pins the opposite failure: if EVERY route is demoted (a broad provider
outage), skipping them all would leave nothing to call, so demotion is a
preference and the chain is tried anyway -- a slow answer beats a hard api_error.
"""
from types import SimpleNamespace
import unittest
from unittest import mock

import agent
import config


def _real_timeout():
    from openai import APITimeoutError
    return APITimeoutError.__new__(APITimeoutError)


class _FakeSession:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))

    def model_call(self, *a, **k):
        pass

    def api_retry(self, *a, **k):
        pass


def _inst(failing_routes, working_response, seed_timeouts=None,
          routes=(("p1", "m1"), ("p2", "m2"))):
    """An Agent whose listed routes always time out; the rest return cleanly."""
    inst = agent.Agent.__new__(agent.Agent)
    inst.session = _FakeSession()
    inst.route_pairs = list(routes)
    inst.resolved = SimpleNamespace(provider="p1", model="m1", model_id="w1",
                                    base_url="https://p1.example/v1", api_key="k",
                                    label=lambda: "p1/m1")
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
    inst._route_timeouts = dict(seed_timeouts or {})
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
                        return working_response
        return _Good()

    inst._client_for = client_for
    return inst


class RouteDemotedTest(unittest.TestCase):
    def test_demoted_at_or_above_limit_only(self):
        inst = _inst(set(), None)
        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 3):
            inst._route_timeouts = {"p1/m1": 2}
            self.assertFalse(inst._route_demoted("p1/m1"))
            inst._route_timeouts = {"p1/m1": 3}
            self.assertTrue(inst._route_demoted("p1/m1"))

    def test_zero_limit_never_demotes(self):
        inst = _inst(set(), None)
        inst._route_timeouts = {"p1/m1": 99}
        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 0):
            self.assertFalse(inst._route_demoted("p1/m1"))


class DemotionPersistsAcrossTurnsTest(unittest.TestCase):
    def _run(self, seed_timeouts, limit=3):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="from p2", tool_calls=None))])
        inst = _inst({"p1/m1"}, response, seed_timeouts=seed_timeouts)
        resolved_map = {
            "p1/m1": SimpleNamespace(provider="p1", model="m1", model_id="w1",
                                     base_url="https://p1.example/v1", api_key="k",
                                     label=lambda: "p1/m1"),
            "p2/m2": SimpleNamespace(provider="p2", model="m2", model_id="w2",
                                     base_url="https://p2.example/v1", api_key="k",
                                     label=lambda: "p2/m2")}
        with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", limit), \
                mock.patch.object(config, "API_RETRIES", 6), \
                mock.patch.object(config, "API_BACKOFF_SEC", 0), \
                mock.patch.object(agent.llm, "resolve",
                                  lambda p, a: resolved_map["%s/%s" % (p, a)]), \
                mock.patch.object(agent, "build_request", lambda *a, **k: {}), \
                mock.patch.object(agent.time, "sleep", lambda s: None):
            out = inst._complete(use_tools=False)
        return inst, out

    def test_a_parked_route_is_skipped_not_retried_on_a_later_turn(self):
        """THE REGRESSION. p1 hit its limit on an earlier turn; this turn must
        not call it again from attempt 1."""
        inst, out = self._run(seed_timeouts={"p1/m1": 3})
        self.assertEqual(out.choices[0].message.content, "from p2")
        calls = [r for r in inst.logged if r.get("event") == "model_call"]
        self.assertFalse(any(c["model"] == "w1" for c in calls),
                         "the parked route was called again this turn: %r" % calls)
        skipped = [r for r in inst.logged if r.get("event") == "route_skipped"]
        self.assertTrue(skipped and skipped[0]["route"] == "p1/m1",
                        "no route_skipped was recorded for the parked route")
        self.assertEqual(inst._route_timeouts.get("p1/m1"), 3,
                         "a skipped route must not accrue new timeouts")

    def test_a_route_below_its_limit_is_still_tried(self):
        """Do not over-skip: 1 timeout under a limit of 3 is a hiccup, not a park."""
        inst, out = self._run(seed_timeouts={"p1/m1": 1})
        self.assertEqual(out.choices[0].message.content, "from p2")
        calls = [r for r in inst.logged if r.get("event") == "model_call"]
        self.assertTrue(any(c["model"] == "w1" for c in calls),
                        "p1 is below its limit and must still be attempted first")
        self.assertFalse(any(r.get("event") == "route_skipped" for r in inst.logged))

    def test_all_routes_demoted_tries_the_chain_anyway(self):
        """Last resort: if everything is parked, skipping would leave nothing to
        call, so the chain is walked anyway and still returns a real answer."""
        inst, out = self._run(seed_timeouts={"p1/m1": 3, "p2/m2": 3})
        self.assertEqual(out.choices[0].message.content, "from p2",
                         "with every route demoted the run must still get an "
                         "answer, not a hard failure")
        calls = [r for r in inst.logged if r.get("event") == "model_call"]
        self.assertTrue(calls, "no model_call happened: the chain was fully skipped")
        self.assertFalse(any(r.get("event") == "route_skipped" for r in inst.logged),
                         "when ALL routes are demoted none may be skipped")

    def test_no_prior_demotion_preserves_normal_chain_behaviour(self):
        """First turn of a run: nothing is parked, so the slow primary still
        gets its retries before falling through (unchanged behaviour)."""
        inst, out = self._run(seed_timeouts={})
        self.assertEqual(out.choices[0].message.content, "from p2")
        self.assertFalse(any(r.get("event") == "route_skipped" for r in inst.logged))
        dead = [r for r in inst.logged if r.get("event") == "route_dead"
                and r.get("route") == "p1/m1"]
        self.assertTrue(dead, "the in-turn demotion still fires on a fresh run")


if __name__ == "__main__":
    unittest.main(verbosity=2)
