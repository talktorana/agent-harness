"""Regression tests for the generic provider catalogue (llm.py).

Run:  python -m unittest test_llm_config -v
All tests run against a temp providers.json; no network, no real keys.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import llm  # noqa: E402

CATALOG = {
    "default": {"provider": "alpha", "model": "fast"},
    "providers": {
        "alpha": {
            "base_url": "https://alpha.example/v1",
            "api_key_env": "TEST_ALPHA_KEY",
            "models": {"fast": "alpha-fast-1", "smart": "alpha-smart-2"},
        },
        "beta": {
            "base_url": "https://beta.example/v1/",
            "api_key": "inline-secret",
            "models": {"only": "beta-only-9"},
        },
        "nokey": {
            "base_url": "https://nokey.example/v1",
            "api_key_env": "TEST_MISSING_KEY",
            "models": {"m": "m-1"},
        },
    },
}


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "providers.json"
        self.file.write_text(json.dumps(CATALOG), encoding="utf-8")
        patcher = mock.patch.object(llm, "PROVIDERS_FILE", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "env-secret"})
        env.start()
        self.addCleanup(env.stop)

    def test_default_choice_from_file(self):
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))

    def test_resolve_maps_alias_to_real_id_and_env_key(self):
        res = llm.resolve()
        self.assertEqual(res.provider, "alpha")
        self.assertEqual(res.model, "fast")
        self.assertEqual(res.model_id, "alpha-fast-1")
        self.assertEqual(res.base_url, "https://alpha.example/v1")
        self.assertEqual(res.api_key, "env-secret")

    def test_trailing_slash_is_stripped(self):
        res = llm.resolve("beta", "only")
        self.assertEqual(res.base_url, "https://beta.example/v1")
        self.assertEqual(res.api_key, "inline-secret")

    def test_real_id_accepted_directly(self):
        res = llm.resolve("alpha", "alpha-smart-2")
        self.assertEqual(res.model_id, "alpha-smart-2")

    def test_unknown_provider_lists_the_known_ones(self):
        with self.assertRaises(llm.ProviderError) as ctx:
            llm.resolve("gamma", "fast")
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn("beta", str(ctx.exception))

    def test_unknown_model_lists_aliases(self):
        with self.assertRaises(llm.ProviderError) as ctx:
            llm.resolve("alpha", "nope")
        self.assertIn("fast", str(ctx.exception))
        self.assertIn("smart", str(ctx.exception))

    def test_missing_key_is_an_actionable_error(self):
        with self.assertRaises(llm.ProviderError) as ctx:
            llm.resolve("nokey", "m")
        self.assertIn("TEST_MISSING_KEY", str(ctx.exception))

    def test_catalog_hides_keys_and_reports_configuration(self):
        cat = llm.catalog()
        blob = json.dumps(cat)
        self.assertNotIn("env-secret", blob)
        self.assertNotIn("inline-secret", blob)
        self.assertTrue(cat["providers"]["alpha"]["key_configured"])
        self.assertTrue(cat["providers"]["beta"]["key_configured"])
        self.assertFalse(cat["providers"]["nokey"]["key_configured"])
        self.assertEqual(cat["providers"]["alpha"]["models"],
                         ["fast", "smart"])
        self.assertEqual(cat["default"], ["alpha", "fast"])

    def test_env_provider_always_present_and_accepts_raw_ids(self):
        cat = llm.catalog()
        self.assertIn("env", cat["providers"])
        with mock.patch.dict(os.environ, {"AGENT_API_KEY": "k"}):
            res = llm.resolve("env", "some-raw-model-id")
        self.assertEqual(res.model_id, "some-raw-model-id")


class EnvOverrideTest(unittest.TestCase):
    def test_environment_provider_survives_without_a_file(self):
        with mock.patch.object(llm, "PROVIDERS_FILE",
                               Path("/nonexistent/providers.json")), \
                mock.patch.dict(os.environ, {"AGENT_API_KEY": "k"}):
            table = llm.providers()
            self.assertIn("env", table)
            res = llm.resolve()
            self.assertEqual(res.provider, "env")
            self.assertEqual(res.base_url, config.BASE_URL)


class RoutingTest(unittest.TestCase):
    """Ordered fallback chains across providers (llm.route_chain)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "providers.json"
        self.file.write_text(json.dumps(CATALOG), encoding="utf-8")
        patcher = mock.patch.object(llm, "PROVIDERS_FILE", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "k1",
                                           "AGENT_ROUTES": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_parse_route_spec(self):
        self.assertEqual(llm.parse_route_spec("alpha/fast, beta/only"),
                         [("alpha", "fast"), ("beta", "only")])
        # A bare model inherits the catalogue default provider.
        self.assertEqual(llm.parse_route_spec("fast"), [("alpha", "fast")])
        self.assertEqual(llm.parse_route_spec(" , "), [])

    def test_explicit_primary_prepends_and_dedups(self):
        with mock.patch.dict(os.environ, {"AGENT_ROUTES":
                                          "alpha/fast, alpha/smart"}):
            chain = llm.route_chain("beta", "only")
        self.assertEqual(chain[0], ("beta", "only"))
        self.assertEqual(chain[1:], [("alpha", "fast"), ("alpha", "smart")])

    def test_file_routes_when_no_env(self):
        rf = Path(self.tmp.name) / "routes.json"
        rf.write_text(json.dumps({"routes": [
            {"provider": "alpha", "model": "smart"},
            {"provider": "beta", "model": "only"}]}), encoding="utf-8")
        with mock.patch.object(llm, "ROUTES_FILE", rf):
            chain = llm.route_chain()
        self.assertEqual(chain, [("alpha", "fast"), ("alpha", "smart"),
                                 ("beta", "only")])

    def test_legacy_fallback_appended_last(self):
        with mock.patch.object(config, "FALLBACK_MODEL", "alpha-smart-2"), \
                mock.patch.dict(os.environ, {"AGENT_ROUTES": "beta/only"}):
            chain = llm.route_chain()
        self.assertEqual(chain[0], ("alpha", "fast"))
        self.assertEqual(chain[-1], ("alpha", "alpha-smart-2"))


def _routing_agent(test, chain, clients):
    """Build an Agent whose route chain and per-route clients are faked.

    The patches are registered on the test's cleanup -- they must stay alive
    while _complete runs, not just during construction.
    """
    import agent as agent_mod

    resolved = {pair: llm.Resolved(provider=pair[0], model=pair[1],
                                   model_id=pair[1] + "-id",
                                   base_url=f"https://{pair[0]}.example/v1",
                                   api_key="k") for pair in chain}
    for target, kw in (
            ("route_chain", {"return_value": list(chain)}),
            ("resolve", {"side_effect": lambda p=None, m=None:
                         resolved[(p, m)]}),
            ("client_for", {"side_effect": lambda r: clients[r.provider]})):
        patcher = mock.patch.object(llm, target, **kw)
        patcher.start()
        test.addCleanup(patcher.stop)
    return agent_mod.Agent(verbose=False)


def _api_status_error_cls():
    """Find the SDK's APIStatusError class without naming the package.

    Vendor module names are deliberately never written literally in this repo
    (they get mangled in transit); we load the SDK by executing one of
    agent.py's own import paths, then fish the class out of sys.modules.
    """
    import sys
    import agent as agent_mod
    agent_mod._is_route_dead(Exception())  # forces the SDK import to run
    for mod in list(sys.modules.values()):
        cls = getattr(mod, "APIStatusError", None)
        if isinstance(cls, type) and issubclass(cls, Exception):
            return cls
    raise RuntimeError("SDK APIStatusError not importable")


class RouteFallbackLoopTest(unittest.TestCase):
    """_complete must step across providers when a route is dead."""

    def _status_error(self, status, msg):
        cls = _api_status_error_cls()
        #: A stub response, never a vendor one. The SDK reads ``status_code``
        #: off it (and ``headers`` for Retry-After) -- nothing else -- and
        #: writing ``import httpx`` here tied the suite to one transport: the
        #: moment the SDK moved to its successor package the tests errored with
        #: ModuleNotFoundError INSIDE the deployment image, where the routing
        #: code under test was working correctly. A test that names a
        #: dependency the production code does not need is testing the
        #: environment.
        response = SimpleNamespace(
            status_code=status, headers={},
            request=SimpleNamespace(method="POST", url="http://x"))
        return cls(msg, response=response, body=None)

    def test_dead_route_skips_to_next_provider(self):
        dead = self._status_error(404, "model_not_found: x")

        class DeadClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kw):
                raise dead

        class GoodClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kw):
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None))])

        chain = [("alpha", "fast"), ("beta", "only")]
        a = _routing_agent(self, chain, {"alpha": DeadClient(),
                                         "beta": GoodClient()})
        with mock.patch.object(config, "STREAM", False):
            resp = a._complete(use_tools=False)
        self.assertEqual(resp.choices[0].message.content, "ok")

    def test_all_routes_dead_raises_with_every_verdict(self):
        dead = self._status_error(404, "model_not_found: x")

        class DeadClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kw):
                raise dead

        chain = [("alpha", "fast"), ("beta", "only")]
        a = _routing_agent(self, chain, {"alpha": DeadClient(),
                                         "beta": DeadClient()})
        with mock.patch.object(config, "STREAM", False):
            with self.assertRaises(RuntimeError) as ctx:
                a._complete(use_tools=False)
        self.assertIn("alpha/fast", str(ctx.exception))
        self.assertIn("beta/only", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
