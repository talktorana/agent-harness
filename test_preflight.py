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
    #: These tests skip (not fail) inside the deployed image, which ships
    #: Python source but not documentation artifacts.
    def _template_text(self):
        path = Path(__file__).resolve().parent / "mission.md.example"
        if not path.is_file():
            self.skipTest("the deployed image ships Python source, not documentation")
        return path.read_text(encoding="utf-8")

    def test_template_contains_the_bounded_healthy_and_deep_policy(self):
        text = self._template_text()
        self.assertIn("mission_preflight", text)
        self.assertIn("HEALTHY PATH", text)
        self.assertIn("DEEP PATH", text)
        self.assertIn("report and STOP", text)
        self.assertIn("ONE GREEN CHECK IS FINAL", text)

    def test_template_carries_the_no_handoff_rule_as_a_failed_delivery(self):
        """A diagnosed incident that ends in a handover checklist is a FAILED
        delivery. Pinned at PARAGRAPH level, and scoped to the BODY below the
        divider (the part the model actually receives -- the documentation
        above it may only mention the rule; only the body binds a run)."""
        text = self._template_text()
        body = text.split("\n---\n", 1)[1]
        paragraphs = [p for p in body.split("\n\n") if "NO-HANDOFF RULE" in p]
        self.assertTrue(paragraphs, "NO-HANDOFF RULE missing from the "
                                    "mission body")
        self.assertIn("FAILED delivery", paragraphs[0],
                      "the no-handoff paragraph must call the handover-only "
                      "outcome a FAILED delivery, not merely discourage it")
        self.assertIn("deployed", paragraphs[0].lower())
        self.assertIn("ZERO times", paragraphs[0],
                      "the paragraph must name the observed failure shape: a "
                      "full budget spent with zero deploy calls")

    def test_template_contains_first_then_diagnoses_from_stable_ground(self):
        text = self._template_text()
        self.assertIn("HALF the wall budget", text)
        self.assertIn("CONTAINMENT fix FIRST", text)
        self.assertIn("Contain, then diagnose", text)

    def test_template_carries_evidence_discipline_labels(self):
        text = self._template_text()
        for label in ("**VERIFIED**", "**INHERITED**",
                      "UNVERIFIABLE FROM HERE"):
            self.assertIn(label, text)
        self.assertIn("THIS pass", text, "suite counts must come from this pass")
        self.assertIn("prove the probe can fail", text)
        self.assertIn("RETRACT", text)

    def test_template_carries_the_drift_direction_rule(self):
        text = self._template_text()
        self.assertIn("SUSPECT DRIFT BEFORE CODE", text)
        self.assertIn("BACKPORT first", text)
        self.assertIn("you are behind", text,
                      "the rule must name the usual direction, or a pass will "
                      "again overwrite a host-ahead fix")

    def test_template_separates_pushed_from_shipped(self):
        text = self._template_text()
        self.assertIn("Pushed is STILL not shipped", text)
        self.assertIn("RUNNING process/container", text)
        self.assertIn("never from a build log", text)

    def test_template_keeps_the_parallel_actor_rule(self):
        text = self._template_text()
        self.assertIn("another actor", text)
        self.assertIn("Stage exactly your own files", text)

    def test_output_format_names_bounds_gaps_and_processes_touched(self):
        text = self._template_text()
        self.assertIn("Write the report FIRST", text)
        self.assertIn("Executed mission(s)", text)
        self.assertIn("wall clock", text,
                      "the report must distinguish WHICH bound ended a run")
        self.assertIn("Production processes touched", text)

    def test_template_body_keeps_every_make_schedule_probe(self):
        """`read_mission()` refuses a brief missing any probe string; the
        template's prompt half must carry all of them so `cp mission.md.example
        mission.md` stays a working starting point."""
        from make_schedule import read_mission  # noqa: local import on purpose
        text = self._template_text()
        body = text.split("\n---\n", 1)[1]
        for probe in ("MISSION:", "TERMINAL SAFELIGHTS", "OUTPUT FORMAT"):
            self.assertIn(probe, body)
        # And the whole file must round-trip through the real reader.
        tmp = Path(tempfile.mkdtemp()) / "mission.md"
        tmp.write_text(text, encoding="utf-8")
        mission = read_mission(tmp)
        self.assertIn("NO-HANDOFF RULE", mission)
        self.assertNotIn("What to customise", mission,
                         "documentation above the divider must not be embedded")

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
