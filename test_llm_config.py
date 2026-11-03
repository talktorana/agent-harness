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
import webui  # noqa: E402

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

    def test_catalog_exposes_alias_to_real_id_mapping(self):
        """The UI can only show what this payload carries.

        Selecting a model used to expose nothing but the alias, so an operator
        could not confirm which real model a choice reached -- the wire id
        appeared to be missing from the dropdown when it was merely aliased.
        """
        cat = llm.catalog()
        self.assertEqual(cat["providers"]["alpha"]["model_ids"],
                         {"fast": "alpha-fast-1", "smart": "alpha-smart-2"})
        self.assertEqual(cat["providers"]["beta"]["model_ids"],
                         {"only": "beta-only-9"})
        # Every offered alias must be resolvable through the mapping, or the
        # dropdown would render an option with no real id behind it.
        for name, block in cat["providers"].items():
            for alias in block["models"]:
                self.assertIn(alias, block["model_ids"],
                              f"{name}/{alias} has no real id")

    def test_the_webui_renders_the_real_id_not_just_the_alias(self):
        """Guard the JS too: a payload nobody displays fixes nothing."""
        source = Path(webui.__file__).read_text(encoding="utf-8")
        self.assertIn("model_ids", source,
                      "webui never reads model_ids from the catalogue")
        # Three dropdowns now: the console, the live view, and the runs page's
        # "model for scheduled runs" panel. The count is asserted so that a
        # future dropdown added without the alias->real-id mapping is caught
        # rather than shipping alias-only and looking like the model is absent.
        self.assertEqual(source.count("p.model_ids"), 3,
                         "expected the mapping wired into all three dropdowns")


class ConsoleModelChoiceTest(unittest.TestCase):
    """The model the remote harness uses, as set from the console.

    A checkbox on a page is only worth anything if it reaches the code that
    starts a mission. Scheduled runs call the runner with no provider and no
    model, so the whole feature rests on default_choice() -- which is what
    these tests drive directly, without a scheduler, a network or an HTTP
    server.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "providers.json"
        self.file.write_text(json.dumps(CATALOG), encoding="utf-8")
        self.choice = Path(self.tmp.name) / "model_choice.json"
        for patcher in (
            mock.patch.object(llm, "PROVIDERS_FILE", self.file),
            # Patched through config, not llm: the module reads the path at
            # call time precisely so a test can point it somewhere harmless.
            mock.patch.object(config, "MODEL_CHOICE_FILE", self.choice),
            mock.patch.dict(os.environ, {"TEST_ALPHA_KEY": "env-secret"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_with_nothing_saved_the_catalogue_default_is_used(self):
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))
        self.assertEqual(llm.choice_source(), "catalogue")

    def test_a_saved_choice_outranks_the_catalogue_default(self):
        llm.save_default_choice("alpha", "smart")
        self.assertEqual(llm.default_choice(), ("alpha", "smart"))
        self.assertEqual(llm.choice_source(), "console")
        # And it is what a caller that names nothing actually resolves to --
        # the property a scheduled mission depends on.
        res = llm.resolve()
        self.assertEqual((res.provider, res.model, res.model_id),
                         ("alpha", "smart", "alpha-smart-2"))

    def test_the_saved_file_records_the_wire_id(self):
        llm.save_default_choice("alpha", "smart")
        saved = json.loads(self.choice.read_text(encoding="utf-8"))
        self.assertEqual(saved["model_id"], "alpha-smart-2",
                         "the file must show the id the wire will see, not "
                         "just the alias")
        self.assertNotIn("env-secret", self.choice.read_text(encoding="utf-8"),
                         "no key material may be written beside the choice")

    def test_saving_a_bad_choice_leaves_the_previous_one_in_force(self):
        """The failure that would matter: a typo quietly bricking the mission.

        Rejecting at SAVE time is the whole reason this is not a free-text
        field. If the bad value were written first, the next unattended run
        would die on it with nobody watching, and the console would still be
        displaying the selected name as if it were working.
        """
        llm.save_default_choice("alpha", "fast")
        with self.assertRaises(llm.ProviderError):
            llm.save_default_choice("alpha", "no-such-alias")
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))

    def test_a_provider_without_a_key_cannot_be_saved(self):
        """Same rule for the other half of a route: a keyless provider is a
        configuration error, so it must never become the persisted default."""
        with self.assertRaises(llm.ProviderError):
            llm.save_default_choice("nokey", "m")
        self.assertFalse(self.choice.exists(),
                         "a rejected choice must not create the file")

    def test_clearing_returns_to_the_catalogue_default(self):
        llm.save_default_choice("alpha", "smart")
        self.assertTrue(llm.clear_default_choice())
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))
        self.assertEqual(llm.choice_source(), "catalogue")
        # Idempotent, and honest about it: the second clear reports False so
        # the page can say "nothing was set" instead of claiming success.
        self.assertFalse(llm.clear_default_choice())

    def test_an_empty_saved_choice_is_ignored_not_fatal(self):
        """A truncated or templated file must not stop every run.

        This is the one case where falling back is right: the file carries no
        choice at all, so there is nothing to contradict.
        """
        self.choice.write_text(json.dumps({"provider": "", "model": ""}),
                               encoding="utf-8")
        self.assertEqual(llm.default_choice(), ("alpha", "fast"))

    def test_an_unreadable_saved_choice_raises_instead_of_reverting(self):
        """SILENT fallback here would be the worst outcome available.

        The console would keep showing the saved model, scheduled missions
        would quietly run on a different one, and no log would mention it.
        """
        self.choice.write_text("{not json", encoding="utf-8")
        with self.assertRaises(llm.ProviderError) as ctx:
            llm.default_choice()
        self.assertIn(str(self.choice), str(ctx.exception),
                      "the error must name the file to delete")

    def test_the_page_state_reports_the_effective_model_and_its_source(self):
        """The panel is rendered from this payload, so a stale or absent
        source here becomes a page that lies about which model is in force."""
        llm.save_default_choice("alpha", "smart")
        state = webui._model_state()
        self.assertEqual(state["source"], "console")
        self.assertEqual(state["model"], "smart")
        self.assertEqual(state["model_id"], "alpha-smart-2")
        self.assertIsNone(state["error"])

    def test_the_page_state_exposes_the_process_wide_settings(self):
        """A model choice does NOT change reasoning effort or the token
        budget -- both are container environment. The panel says so, and it
        can only say so if the payload carries them."""
        with mock.patch.object(config, "REASONING_EFFORT", "max"), \
             mock.patch.object(config, "MAX_TOKENS", 32000):
            state = webui._model_state()
        self.assertEqual(state["reasoning_effort"], "max")
        self.assertEqual(state["max_tokens"], 32000)

    def test_the_page_state_reports_a_broken_default_instead_of_hiding_it(self):
        """An unresolvable default means missions are about to fail. The page
        must be able to say that; swallowing it would present a healthy
        panel over a dead scheduler."""
        self.choice.write_text("{not json", encoding="utf-8")
        state = webui._model_state()
        self.assertIsNotNone(state["error"])
        self.assertIn("model_choice.json", state["error"])

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


class DotenvProviderKeyTest(unittest.TestCase):
    """.env loading of the key names providers.json declares.

    Provider keys are NOT ``AGENT_``-prefixed, so a whitelist limited to that
    prefix silently ignored them: the key sat in .env while the run died with
    "environment variable TEST_ALPHA_KEY is not set" -- naming the very file
    that already held it. The loader still whitelists rather than importing
    everything, so unrelated credentials cannot reach generated code.
    """

    #: Names the tests may write into os.environ; each is removed first and
    #: restored after, because "already exported" changes loader behaviour.
    TOUCHED = ("TEST_ALPHA_KEY", "TEST_MISSING_KEY", "BROKER_SECRET",
               "QUOTED", "AGENT_MODEL")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = self.root / "providers.json"
        self.catalog.write_text(json.dumps(CATALOG), encoding="utf-8")
        self.env_file = self.root / ".env"
        self._clear_touched()

    def _clear_touched(self):
        """Empty these keys for the test, then restore the original state.

        Cleanups run in reverse order of registration, so the wipe is
        registered LAST and executes FIRST; the restore follows it. Without
        the wipe the loader's own writes would leak into later test classes.
        """
        saved = {}
        for key in self.TOUCHED:
            if key in os.environ:
                saved[key] = os.environ.pop(key)
        self.addCleanup(os.environ.update, saved)
        self.addCleanup(
            lambda: [os.environ.pop(key, None) for key in self.TOUCHED])

    def _catalogue_env(self):
        return mock.patch.dict(os.environ,
                               {"AGENT_PROVIDERS_FILE": str(self.catalog)})

    def test_key_names_come_from_the_catalogue(self):
        with self._catalogue_env():
            self.assertEqual(config._provider_key_names(),
                             {"TEST_ALPHA_KEY", "TEST_MISSING_KEY"})

    def test_unreadable_catalogue_yields_no_names(self):
        with mock.patch.dict(os.environ, {"AGENT_PROVIDERS_FILE":
                                          str(self.root / "absent.json")}):
            self.assertEqual(config._provider_key_names(), set())
        self.catalog.write_text("{not json", encoding="utf-8")
        with self._catalogue_env():
            self.assertEqual(config._provider_key_names(), set())

    def test_whitelisted_key_loads_unlisted_secret_does_not(self):
        self.env_file.write_text(
            "AGENT_MODEL=from-file\n"
            "TEST_ALPHA_KEY=k-from-env-file\n"
            "QUOTED='still works'\n"
            "BROKER_SECRET=must-not-leak\n"
            "# a comment\n"
            "\n"
            "NO_EQUALS_LINE\n",
            encoding="utf-8")
        with self._catalogue_env():
            allowed = config._provider_key_names()
            self.assertIn("TEST_ALPHA_KEY", allowed)
            config._load_dotenv(self.env_file, allowed=allowed)
            # Assertions live INSIDE the patch: patch.dict restores os.environ
            # on exit, which would discard exactly the keys just loaded.
            self.assertEqual(os.environ["TEST_ALPHA_KEY"], "k-from-env-file")
            self.assertEqual(os.environ["AGENT_MODEL"], "from-file")
            self.assertNotIn("BROKER_SECRET", os.environ)
            self.assertNotIn("QUOTED", os.environ)
            self.assertNotIn("NO_EQUALS_LINE", os.environ)
            self.assertEqual(config.source_of("TEST_ALPHA_KEY"), ".env")

    def test_loaded_key_reaches_resolve(self):
        """The point of the whole exercise: llm.resolve() sees it."""
        self.env_file.write_text("TEST_ALPHA_KEY=k-from-env-file\n",
                                 encoding="utf-8")
        with self._catalogue_env(), \
                mock.patch.object(llm, "PROVIDERS_FILE", self.catalog):
            config._load_dotenv(self.env_file,
                                allowed=config._provider_key_names())
            res = llm.resolve("alpha", "fast")
        self.assertEqual(res.api_key, "k-from-env-file")
        self.assertEqual(res.model_id, "alpha-fast-1")

    def test_exported_key_wins_and_shadowing_is_reported(self):
        self.env_file.write_text("TEST_ALPHA_KEY=from-file\n", encoding="utf-8")
        os.environ["TEST_ALPHA_KEY"] = "from-shell"
        self.addCleanup(os.environ.pop, "TEST_ALPHA_KEY", None)
        with self._catalogue_env():
            config._load_dotenv(self.env_file,
                                allowed=config._provider_key_names())
            # Inside the patch for the same reason: the exported value must be
            # what survives the load, and the shadowing must be recorded.
            self.assertEqual(os.environ["TEST_ALPHA_KEY"], "from-shell")
            self.assertIn("shadows", config.source_of("TEST_ALPHA_KEY"))


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
