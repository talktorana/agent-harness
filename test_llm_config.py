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


if __name__ == "__main__":
    unittest.main()
