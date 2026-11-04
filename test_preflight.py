"""Generic one-call preflight: harness facts plus deployment-owned health."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import config
import preflight
import tools
import tools_preflight


class PreflightSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            mock.patch.object(config, "PROJECT_ROOT", self.root),
            mock.patch.object(config, "PREFLIGHT_CONSOLE_URL", "http://console.test"),
            mock.patch.object(config, "PREFLIGHT_ANSWER_CHARS", 6000),
        ]
        for patch in self.patches:
            patch.start()
        self.now = datetime(2026, 11, 4, 10, 30, tzinfo=timezone.utc)

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.temp.cleanup()

    def fetch(self, url):
        if url.endswith("/agent/schedule"):
            return {"jobs": [{"id": "mission", "last_ok": True, "every_min": 60,
                              "next_due": "2026-11-04T11:30:00+00:00"}]}
        if url.endswith("/agent/settings"):
            return {"max_steps": 200, "every_min": 60}
        if url.endswith("/agent/runs.json"):
            return {"runs": [
                {"origin": "chat", "started": "2026-11-04T10:29:00", "status": "ok",
                 "answer": "not a scheduled delivery"},
                {"origin": "schedule", "started": "2026-11-04T10:00:00", "status": "ok",
                 "session_id": "scheduled1", "answer": "scheduled answer"},
            ]}
        raise AssertionError("unexpected URL: " + url)

    @staticmethod
    def healthy():
        return {"fast_path_eligible": True, "attention_required": [],
                "application": "example evidence"}, []

    def snapshot(self, collector=healthy):
        return preflight.build_snapshot(now=self.now, fetch=self.fetch, collector=collector)

    def test_healthy_snapshot_merges_harness_and_project_evidence(self):
        snap = self.snapshot()
        self.assertTrue(snap["fast_path_eligible"], snap["attention_required"])
        self.assertEqual(snap["previous_scheduled_delivery"]["session_id"], "scheduled1")
        self.assertEqual(snap["previous_scheduled_delivery"]["answer"], "scheduled answer")
        self.assertEqual(snap["project_health"]["application"], "example evidence")
        self.assertEqual(snap["settings"]["max_steps"], 200)

    def test_project_attention_forces_deep_path(self):
        snap = self.snapshot(lambda: ({"fast_path_eligible": False,
                                       "attention_required": ["queue is stuck"]}, []))
        self.assertFalse(snap["fast_path_eligible"])
        self.assertIn("queue is stuck", snap["attention_required"])

    def test_false_project_verdict_without_evidence_is_not_silent(self):
        snap = self.snapshot(lambda: ({"fast_path_eligible": False,
                                       "attention_required": []}, []))
        self.assertFalse(snap["fast_path_eligible"])
        self.assertTrue(any("without naming" in item for item in snap["attention_required"]))

    def test_console_failure_is_attention_not_a_crash(self):
        def unavailable(url):
            raise TimeoutError("test timeout")
        snap = preflight.build_snapshot(now=self.now, fetch=unavailable,
                                        collector=self.healthy)
        self.assertFalse(snap["fast_path_eligible"])
        self.assertTrue(any("console schedule" in item for item in snap["attention_required"]))

    def test_missing_answer_is_fetched_by_run_id(self):
        def fallback(url):
            if url.endswith("/agent/run/scheduled1"):
                return {"answer": "answer from detail endpoint"}
            data = self.fetch(url)
            if url.endswith("/agent/runs.json"):
                data["runs"][1].pop("answer")
            return data
        options = {"now": self.now, "fetch": fallback, "collector": self.healthy}
        snap = preflight.build_snapshot(**options)
        self.assertEqual(snap["previous_scheduled_delivery"]["answer"],
                         "answer from detail endpoint")
        self.assertTrue(snap["fast_path_eligible"], snap["attention_required"])

    def test_failed_previous_schedule_run_forces_deep_path(self):
        def failed(url):
            data = self.fetch(url)
            if url.endswith("/agent/schedule"):
                data["jobs"][0]["last_ok"] = False
            return data
        options = {"now": self.now, "fetch": failed, "collector": self.healthy}
        snap = preflight.build_snapshot(**options)
        self.assertFalse(snap["fast_path_eligible"])
        self.assertIn("the previous scheduled run reports last_ok=false",
                      snap["attention_required"])

    def test_file_only_mode_does_not_invent_console_findings(self):
        snap = preflight.build_snapshot(now=self.now, collector=self.healthy,
                                        include_console=False)
        self.assertTrue(snap["fast_path_eligible"], snap["attention_required"])
        self.assertEqual(snap["schedule"], {})


class ProjectCollectorTest(unittest.TestCase):
    def setUp(self):
        self.patches = [
            mock.patch.object(config, "PROJECT_ROOT", Path(".")),
            mock.patch.object(config, "PREFLIGHT_COMMAND", "health-check --json"),
            mock.patch.object(config, "PREFLIGHT_REQUIRE_COMMAND", True),
            mock.patch.object(config, "PREFLIGHT_TIMEOUT_SEC", 12.0),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def test_missing_required_collector_is_explicit(self):
        with mock.patch.object(config, "PREFLIGHT_COMMAND", ""):
            payload, errors = preflight.run_project_collector()
        self.assertEqual(payload, {})
        self.assertIn("AGENT_PREFLIGHT_COMMAND", errors[0])

    def test_optional_collector_can_be_disabled_for_harness_only_use(self):
        with (mock.patch.object(config, "PREFLIGHT_COMMAND", ""),
              mock.patch.object(config, "PREFLIGHT_REQUIRE_COMMAND", False)):
            payload, errors = preflight.run_project_collector()
        self.assertEqual((payload, errors), ({}, []))

    def test_command_runs_without_a_shell_and_parses_json(self):
        completed = mock.Mock(returncode=0, stdout='{"fast_path_eligible": true, '
                                                   '"attention_required": []}', stderr="")
        with mock.patch.object(preflight.subprocess, "run", return_value=completed) as run:
            payload, errors = preflight.run_project_collector()
        self.assertEqual(errors, [])
        self.assertTrue(payload["fast_path_eligible"])
        kwargs = run.call_args.kwargs
        self.assertEqual(run.call_args.args[0], ["health-check", "--json"])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["timeout"], 12.0)

    def test_nonzero_exit_includes_diagnostic_output(self):
        completed = mock.Mock(returncode=7, stdout="", stderr="collector boom")
        with mock.patch.object(preflight.subprocess, "run", return_value=completed):
            payload, errors = preflight.run_project_collector()
        self.assertEqual(payload, {})
        self.assertIn("exited 7: collector boom", errors[0])

    def test_invalid_contract_cannot_read_as_healthy(self):
        errors = preflight.project_contract_errors({"attention_required": []})
        self.assertIn("boolean fast_path_eligible", errors[0])

    def test_timeout_is_reported(self):
        with mock.patch.object(preflight.subprocess, "run",
                               side_effect=preflight.subprocess.TimeoutExpired("health", 12)):
            payload, errors = preflight.run_project_collector()
        self.assertEqual(payload, {})
        self.assertIn("timed out after 12s", errors[0])


class ToolRegistrationTest(unittest.TestCase):
    def test_schema_and_handler_share_one_name(self):
        names = {item["function"]["name"] for item in tools.TOOLS}
        self.assertIn("mission_preflight", names)
        self.assertIn("mission_preflight", tools.HANDLERS)

    def test_tool_failure_returns_deep_path_evidence(self):
        with mock.patch.object(tools_preflight.preflight, "build_snapshot",
                               side_effect=RuntimeError("test failure")):
            result = json.loads(tools_preflight.mission_preflight())
        self.assertFalse(result["fast_path_eligible"])
        self.assertIn("test failure", result["attention_required"][0])


class PublicMissionTemplateTest(unittest.TestCase):
    def test_template_contains_the_bounded_healthy_and_deep_policy(self):
        path = Path(__file__).resolve().parent / "mission.md.example"
        if not path.is_file():
            self.skipTest("the deployed image ships Python source, not documentation")
        text = path.read_text(encoding="utf-8")
        self.assertIn("mission_preflight", text)
        self.assertIn("HEALTHY PATH", text)
        self.assertIn("DEEP PATH", text)
        self.assertIn("report and STOP", text)
        self.assertIn("ONE GREEN CHECK IS FINAL", text)

    def test_operations_guide_says_cache_is_not_recurring_maintenance(self):
        path = Path(__file__).resolve().parent / "docs" / "PERFORMANCE.md"
        if not path.is_file():
            self.skipTest("the deployed image ships Python source, not documentation")
        text = path.read_text(encoding="utf-8")
        self.assertIn("one-time harness/configuration fix", text)
        self.assertIn("cache is ephemeral", text)
        self.assertIn("AGENT_PREFLIGHT_COMMAND", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
