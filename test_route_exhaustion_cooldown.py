"""A route that ends a turn UNSERVED must be stepped over on later turns.

Regression for the defect measured live 2026-11-24 14:19 (a scheduled
monitoring mission, 876 s for what is normally a ~3-minute fast path): the
default model -- made the default that morning -- answered

    Error code: 429 - 292 of 310 providers for this model are out of quota or
    cooling down until their limits reset; every provider within your price
    limit is busy or rate-limited right now; retry shortly. 2 provider(s)
    priced above your limit have free capacity; to use them, raise ...

on ALL 15 attempts of the run. Not one provider-exhaustion phrase matched, so
`_is_transient` owned it as an ordinary 429: six attempts a turn, each held
~35 s before failing plus ~65 s of backoff, then a fallback for THAT TURN
ONLY. The next turn called the same route again from attempt 1 -- the
across-turn re-payment `test_route_demotion_persists` pins for TIMEOUTS, in a
shape the timeout counter cannot see (0 `route_dead`, 15 `api_retry`).

Two things fix it, and both halves are pinned here:

  * the pool-exhaustion WORDING of a 429 is a dead route for the turn (one
    call, no backoff ladder), exactly like the 503 wording already was; and
    * a route that ended a turn unserved is PARKED for a cooldown window, so
    later turns skip it, and is probed again once the window lapses.

The dangerous version of this fix would park a route on any failure or skip a
healthy one on an ordinary rate limit; those negatives are pinned too.
"""
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest import mock

import agent
import config


LIVE_429 = (
    "Error code: 429 - 292 of 310 providers for this model are out of quota "
    "or cooling down until their limits reset; every provider within your "
    "price limit is busy or rate-limited right now; retry shortly. 2 "
    "provider(s) priced above your limit have free capacity; to use them, "
    "raise x-max-input-price / x-max-output-price"
)
PLAIN_429 = "Error code: 429 - Too Many Requests: rate limit exceeded, retry later"


class _PoolGone(Exception):
    """Carries the provider's own pool-exhaustion wording."""


class _Busy(Exception):
    """An ordinary transient rate limit: one key is busy, the pool is not."""


class _Refused(Exception):
    """The route refuses THIS request's shape (a 400, say), not the pool."""


class _FakeSession:
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))

    def model_call(self, *a, **k):
        pass

    def api_retry(self, *a, **k):
        pass


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=None))])


def _inst(failures, routes=(("p1", "m1"), ("p2", "m2"))):
    """An Agent whose routes raise what ``failures`` says; the rest answer.

    ``failures`` maps ``provider`` to an exception, to ``None``, or to a
    LIST of exceptions consumed one per call (an empty list serves).
    """
    inst = agent.Agent.__new__(agent.Agent)
    inst.session = _FakeSession()
    inst.route_pairs = list(routes)
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
    inst._route_parked_until = {}
    inst.client = None
    inst._resolved = {
        "%s/%s" % (p, a): SimpleNamespace(
            provider=p, model=a, model_id="w-%s" % p,
            base_url="https://%s.example/v1" % p, api_key="k",
            label=lambda p=p, a=a: "%s/%s" % (p, a))
        for p, a in routes}
    inst.wire_calls = []

    def client_for(resolved):
        route = "%s/%s" % (resolved.provider, resolved.model)

        class _Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kw):
                        inst.wire_calls.append(route)
                        exc = failures.get(route)
                        if isinstance(exc, list):
                            exc = exc.pop(0) if exc else None
                        if exc is not None:
                            raise exc
                        return _response("from %s" % route)
        return _Client()

    inst._client_for = client_for
    return inst


def _turn(inst, retries=6, cooldown=600, transient=None):
    """One model turn through the real ``_complete`` chain walk."""
    maybe_transient = (mock.patch.object(agent, "_is_transient", transient)
                       if transient else nullcontext())
    with mock.patch.object(config, "ROUTE_TIMEOUT_LIMIT", 3), \
            mock.patch.object(config, "API_RETRIES", retries), \
            mock.patch.object(config, "API_BACKOFF_SEC", 0), \
            mock.patch.object(config, "ROUTE_EXHAUST_COOLDOWN_SEC", cooldown), \
            mock.patch.object(agent.llm, "resolve",
                              lambda p, a: inst._resolved["%s/%s" % (p, a)]), \
            mock.patch.object(agent, "build_request", lambda *a, **k: {}), \
            mock.patch.object(agent.time, "sleep", lambda s: None), \
            maybe_transient:
        return inst._complete(use_tools=False)


def _events(inst, kind):
    return [r for r in inst.logged if r.get("event") == kind]


def _busy_only(exc):
    return isinstance(exc, _Busy)


class PoolExhausted429IsRouteDeadTest(unittest.TestCase):
    def test_the_live_429_wording_is_pool_exhaustion_and_route_dead(self):
        """THE INCIDENT: this exact text cost fifteen backoff retries."""
        self.assertTrue(agent._is_provider_exhausted(Exception(LIVE_429)))
        self.assertTrue(agent._is_route_dead(Exception(LIVE_429)))

    def test_the_shorter_live_variants_match(self):
        for text in ("Error code: 429 - 289 of 310 providers for this model "
                     "are out of quota or cooling down",
                     "every provider within your price limit is busy or "
                     "rate-limited right now; retry shortly"):
            with self.subTest(text=text):
                self.assertTrue(agent._is_provider_exhausted(Exception(text)))

    def test_an_ordinary_rate_limit_is_not_pool_exhaustion(self):
        """ANTI-VACUITY: a single-key 429 is a blip. It must keep its retries
        rather than skip a healthy route on the status code alone."""
        self.assertFalse(agent._is_provider_exhausted(Exception(PLAIN_429)),
                         "a plain rate limit must not read as pool exhaustion")
        self.assertFalse(agent._is_route_dead(Exception(PLAIN_429)))


class UnservedRouteIsParkedTest(unittest.TestCase):
    def test_pool_exhaustion_costs_one_call_and_parks_the_route(self):
        inst = _inst({"p1/m1": _PoolGone(LIVE_429)})
        out = _turn(inst)
        self.assertEqual(out.choices[0].message.content, "from p2/m2")
        self.assertEqual(inst.wire_calls, ["p1/m1", "p2/m2"],
                         "a pool-exhaustion refusal must cost ONE call, not a "
                         "backoff ladder")
        self.assertEqual(_events(inst, "api_retry"), [],
                         "an exhausted pool must not be retried with backoff")
        parked = _events(inst, "route_parked")
        self.assertTrue(parked and parked[0]["route"] == "p1/m1", parked)
        self.assertEqual([k for k, _ in inst.session.events
                          if k == "route_parked"], ["route_parked"],
                         "the park must be visible in the SESSION record")

    def test_the_next_turn_skips_the_parked_route(self):
        """THE REGRESSION: turn 2 must not call the unserved route again."""
        inst = _inst({"p1/m1": _PoolGone(LIVE_429)})
        _turn(inst)
        inst.wire_calls.clear()
        out = _turn(inst)
        self.assertEqual(out.choices[0].message.content, "from p2/m2")
        self.assertEqual(inst.wire_calls, ["p2/m2"],
                         "the parked route was called again this turn")
        skipped = _events(inst, "route_skipped")
        self.assertTrue(skipped and skipped[0]["route"] == "p1/m1", skipped)
        self.assertGreater(skipped[0]["cooldown_left_sec"], 0)
        self.assertIn("unserved", skipped[0]["reason"])

    def test_a_route_that_stays_starved_is_probed_once_per_window(self):
        inst = _inst({"p1/m1": _PoolGone(LIVE_429)})
        _turn(inst)
        inst._route_parked_until["p1/m1"] = agent.time.monotonic() - 1.0
        inst.wire_calls.clear()
        out = _turn(inst)
        self.assertEqual(out.choices[0].message.content, "from p2/m2")
        self.assertEqual(inst.wire_calls, ["p1/m1", "p2/m2"],
                         "once the window lapses the route is probed again")
        self.assertEqual(_events(inst, "route_skipped"), [])
        self.assertGreater(inst._route_cooldown_left("p1/m1"), 0,
                           "a failed re-probe re-parks the route")

    def test_a_recovered_route_serves_again_after_the_window(self):
        inst = _inst({})
        with mock.patch.object(config, "ROUTE_EXHAUST_COOLDOWN_SEC", 600):
            inst._park_route("p1/m1", "test")
        self.assertTrue(inst._route_demoted("p1/m1"))
        self.assertEqual(list(inst._parked_routes()), ["p1/m1"])
        self.assertAlmostEqual(inst._parked_routes()["p1/m1"], 600.0, delta=5.0)
        inst._route_parked_until["p1/m1"] = agent.time.monotonic() - 1.0
        self.assertFalse(inst._route_demoted("p1/m1"))
        self.assertEqual(inst._parked_routes(), {})
        out = _turn(inst)
        self.assertEqual(out.choices[0].message.content, "from p1/m1",
                         "the operator's first choice must be used again once "
                         "the pool has capacity")

    def test_a_new_run_inherits_no_park(self):
        src = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("self._route_parked_until = {}", src,
                      "a second run on one Agent must not inherit the first "
                      "run's parks")

    def test_every_route_parked_walks_the_chain_anyway(self):
        """Last resort: with the whole chain parked, skipping would leave
        nothing to call, so the chain is walked and still answers."""
        routes = (("p1", "m1"), ("p2", "m2"), ("p3", "m3"))
        inst = _inst({}, routes=routes)
        with mock.patch.object(config, "ROUTE_EXHAUST_COOLDOWN_SEC", 600):
            for prov, alias in routes:
                inst._park_route("%s/%s" % (prov, alias), "test")
        out = _turn(inst)
        self.assertEqual(out.choices[0].message.content, "from p1/m1")
        self.assertEqual(inst.wire_calls, ["p1/m1"])
        self.assertEqual(_events(inst, "route_skipped"), [],
                         "when EVERY route is parked none may be skipped")


class RetriesExhaustedParksTest(unittest.TestCase):
    def test_retries_exhausted_on_a_transient_error_parks_the_route(self):
        """An ordinary 429 keeps its retries -- but once ALL are spent the
        route served nothing, and the next turn must not pay for them again."""
        inst = _inst({"p1/m1": _Busy(PLAIN_429)})
        out = _turn(inst, retries=3, transient=_busy_only)
        self.assertEqual(out.choices[0].message.content, "from p2/m2")
        self.assertEqual(inst.wire_calls, ["p1/m1"] * 3 + ["p2/m2"])
        parked = _events(inst, "route_parked")
        self.assertTrue(parked and "all 3 attempts" in parked[0]["reason"], parked)
        inst.wire_calls.clear()
        _turn(inst, retries=3, transient=_busy_only)
        self.assertEqual(inst.wire_calls, ["p2/m2"],
                         "the route that spent every retry was retried in full")

    def test_a_transient_error_that_recovers_in_turn_does_not_park(self):
        """Do not over-park: one failure that the next attempt answers is a
        hiccup, and the route stays first in line."""
        inst = _inst({"p1/m1": [_Busy(PLAIN_429)]})
        out = _turn(inst, retries=3, transient=_busy_only)
        self.assertEqual(out.choices[0].message.content, "from p1/m1")
        self.assertEqual(inst.wire_calls, ["p1/m1", "p1/m1"])
        self.assertEqual(_events(inst, "route_parked"), [])
        self.assertEqual(inst._parked_routes(), {})

    def test_zero_cooldown_disables_the_park(self):
        inst = _inst({"p1/m1": _PoolGone(LIVE_429)})
        _turn(inst, cooldown=0)
        inst.wire_calls.clear()
        out = _turn(inst, cooldown=0)
        self.assertEqual(out.choices[0].message.content, "from p2/m2")
        self.assertEqual(inst.wire_calls, ["p1/m1", "p2/m2"],
                         "with the park disabled each turn tries the route "
                         "again (the previous behaviour)")
        self.assertEqual(_events(inst, "route_skipped"), [])
        self.assertEqual(inst._parked_routes(), {})

    def test_a_refused_payload_stays_a_per_turn_decision(self):
        """A refusal about THIS request (bad payload, wrong key) steps over for
        the turn, as before. It is not a statement about the pool, so it must
        not park the route on later turns."""
        inst = _inst({"p1/m1": _Refused("max_tokens is illegal")})
        self.assertFalse(agent._is_provider_exhausted(_Refused("x")))
        with mock.patch.object(agent, "_is_route_dead",
                               lambda exc: isinstance(exc, _Refused)):
            out = _turn(inst)
            self.assertEqual(out.choices[0].message.content, "from p2/m2")
            self.assertEqual(inst.wire_calls, ["p1/m1", "p2/m2"])
            self.assertEqual(_events(inst, "route_parked"), [],
                             "a refused payload must not park a route")
            inst.wire_calls.clear()
            out = _turn(inst)
        self.assertEqual(inst.wire_calls, ["p1/m1", "p2/m2"],
                         "a refusal is re-evaluated every turn, as before")
        self.assertEqual(inst._parked_routes(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
