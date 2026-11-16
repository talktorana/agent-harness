"""A provider saying "no offers left" is a DEAD route, not a transient blip.

Regression for the defect measured live 2026-11-16, run 220876b9
(19:35:57Z -> the 90-minute wall bound):

    api_retry events            99
    503 "no available provider" 95   <-- all on ali/deepseek-v4-pro-0813
    genuine timeouts             4
    sleep_minutes              1236 s total  (20.6 min = ~23% of the run budget)

`_is_transient` retries any status >= 500, and it must: a real 503 blip
deserves one more try. But the provider's message here -- *"no available
provider ... (all offers exhausted or in cooldown)"* -- states that it has
nothing to serve with, so waiting cannot help and only the NEXT ROUTE can.
`_is_route_dead` did not recognise that shape, so the run retried it 95 times
and slept out a fifth of its budget on a route that served ZERO calls.

The route-timeout stepping could not save it either: `ROUTE_TIMEOUT_LIMIT`
counts TIMEOUTS, and 95 of these 99 failures were not timeouts.

The fix is deliberately narrow -- it needs the provider-exhaustion WORDING, so
an ordinary 503 (gateway restart, momentary hiccup) keeps its retry. Both
halves are pinned here, because the dangerous version of this fix would skip a
healthy route on any 5xx.
"""
import unittest

import agent


LIVE_503 = (
    "Error code: 503 - no available provider for ali/deepseek-v4-pro-0813 "
    "(all offers exhausted or in cooldown); trace=abc123"
)


class ProviderExhaustedIsRouteDeadTest(unittest.TestCase):
    def test_the_live_503_message_is_treated_as_dead(self):
        """THE INCIDENT: this exact text was retried 95 times in one run."""
        self.assertTrue(agent._is_provider_exhausted(
            Exception(LIVE_503)))
        self.assertTrue(agent._is_route_dead(Exception(LIVE_503)))

    def test_a_short_capacity_refusal_matches(self):
        for text in ("no available provider for ali/foo",
                     "all offers exhausted",
                     "model is in cooldown",
                     "503: offers exhausted or in cooldown"):
            with self.subTest(text=text):
                self.assertTrue(agent._is_route_dead(Exception(text)))

    def test_a_plain_503_stays_retryable(self):
        """ANTI-VACUITY: a momentary 5xx must NOT skip a healthy route."""
        for text in ("Error code: 503 - Service Unavailable",
                     "Error code: 502 - Bad Gateway",
                     "Connection reset by peer",
                     "Request timed out."):
            with self.subTest(text=text):
                self.assertFalse(agent._is_route_dead(Exception(text)),
                                 "a generic 5xx/timeout must keep its retry")

    def test_the_transient_classifier_still_owns_a_bare_503(self):
        """The split is intentional: transient retries, dead skips."""
        exc = Exception("Error code: 503 - no available provider for ali/x")
        self.assertTrue(agent._is_route_dead(exc))
        # A dead route is checked BEFORE transient in the caller, so ordering
        # is what makes the fix effective; assert the caller's order holds.
        import inspect
        src = inspect.getsource(agent)
        dead_at = src.index("if _is_route_dead(exc):")
        transient_at = src.index("if not _is_transient(exc):")
        self.assertLess(dead_at, transient_at,
                        "the route-dead check must run before the transient "
                        "retry, or a dead route is retried anyway")


if __name__ == "__main__":
    unittest.main()
