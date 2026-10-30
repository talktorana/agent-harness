"""Regressions for the agent-facing deploy tools (tools_deploy.py).

These tools are the ONLY way an agent proposes a change to a deployment, so the
tests pin two things that must never drift:

* every refusal the host applier would make is also refused EARLY, with a
  message that says what to do instead -- an agent should not discover a
  refusal three steps later by reading a receipt;
* in the default GATED mode nothing is applied and no applier is invoked: the
  request is staged and the host decides. Direct mode invokes the same applier
  with the same policy, so "direct" changes who runs the gate, never whether
  one runs.

Run:  python -m unittest test_deploy_tools
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import tools  # noqa: E402
import tools_deploy  # noqa: E402

HARNESS_DIR = Path(__file__).resolve().parent
APPLIER = HARNESS_DIR / "deploy_applier.py"

#: A generic readiness policy, the shape a deployment would ship. Nothing here
#: is trading-specific: the domain vocabulary lives in the deployment repo.
POLICY_DOC = {
    "readiness": {
        "status_file": "data/readiness.json",
        "field": "state",
        "label": "readiness",
        "safe_states": ["ready", "idle"],
        "max_age_sec": 900,
    },
    "limits": {"max_age_sec": 900},
    "forbid": {"names": ["runtime-state.json", "owner.svg"]},
}


class DeployTestBase(unittest.TestCase):
    """A throwaway deployment root with a safety policy and a safe status."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "engine").mkdir()
        self.outbox = self.root / ".deploy" / "outbox"
        self.outbox.mkdir(parents=True)
        self.policy = self.root / "deploy_policy.json"
        self.policy.write_text(json.dumps(POLICY_DOC, indent=2),
                               encoding="utf-8")
        self.status = self.root / "data" / "readiness.json"
        self.set_status("ready")

        stack = mock.patch.object(config, "DEPLOY_ROOT", str(self.root))
        stack.start()
        self.addCleanup(stack.stop)
        for attr, value in (
                ("DEPLOY_OUTBOX", str(self.outbox)),
                ("DEPLOY_APPLIER", str(APPLIER)),
                ("DEPLOY_POLICY", str(self.policy)),
                ("DEPLOY_ALLOW_RESTART", False),
                ("DEPLOY_MAX_FILES", 20),
                ("DEPLOY_MAX_BYTES", 4 << 20),
                ("DEPLOY_FORBIDDEN", ("runtime-config.json",))):
            patcher = mock.patch.object(config, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def set_status(self, state, stale_sec=None):
        self.status.write_text(json.dumps({"state": state, "reason": "ok"}),
                               encoding="utf-8")
        if stale_sec:
            mtime = self.status.stat().st_mtime - stale_sec
            os.utime(self.status, (mtime, mtime))

    def mode(self, name):
        patcher = mock.patch.object(config, "DEPLOY_MODE", name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def sha(self, payload):
        raw = (payload if isinstance(payload, bytes)
               else payload.encode("utf-8"))
        return hashlib.sha256(raw).hexdigest()


class ComposeTest(DeployTestBase):
    def test_a_request_is_staged_with_its_hashes_and_purpose(self):
        self.mode("gated")
        req = tools_deploy.compose_deploy_request(
            purpose="fix the retry loop",
            files=[{"target": "engine/fix.py", "content": "print(1)\n"}])
        path = Path(req["_request_path"])
        self.assertTrue(path.is_file())
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["purpose"], "fix the retry loop")
        self.assertEqual(doc["mode"], "gated")
        self.assertTrue(doc["created_at"])
        self.assertEqual(len(doc["files"]), 1)
        row = doc["files"][0]
        self.assertEqual(row["target"], "engine/fix.py")
        self.assertEqual(row["sha256"], self.sha("print(1)\n"))
        self.assertEqual(row["bytes"], len("print(1)\n"))

    def test_a_source_file_is_staged_as_bytes_not_as_a_container_path(self):
        """The host cannot see paths that only exist inside the sidecar."""
        src = self.root / "engine" / "staged_src.py"
        src.write_text("SOURCE BODY\n", encoding="utf-8")
        # Compare against the bytes ON DISK: a text-mode write translates
        # newlines on Windows, so hashing the source string would assert
        # something about the platform rather than about the staging.
        expected = self.sha(src.read_bytes())
        req = tools_deploy.compose_deploy_request(
            purpose="ship an edited module",
            files=[{"target": "engine/applied.py", "source": str(src)}])
        doc = json.loads(Path(req["_request_path"]).read_text(encoding="utf-8"))
        row = doc["files"][0]
        self.assertEqual(row["sha256"], expected)
        staged = Path(row["payload_path"])
        self.assertTrue(staged.is_file(), "payload must be staged on disk")
        self.assertEqual(staged.read_bytes(), src.read_bytes())
        self.assertEqual(row["bytes"], len(src.read_bytes()))

    def test_an_absolute_target_is_refused_before_anything_is_staged(self):
        absolute = ("C:/Windows/system.ini" if os.name == "nt"
                    else "/etc/passwd")
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="p", files=[{"target": absolute, "content": "x"}])
        self.assertIn("relative to the deploy root", str(ctx.exception))
        self.assertEqual(list(self.outbox.glob("*.deploy.json")), [])

    def test_traversal_and_dotgit_are_refused(self):
        for target in ("../escape.py", "engine/../../escape.py",
                       ".git/config", "sub/.git/objects/x"):
            with self.subTest(target=target):
                with self.assertRaises(tools_deploy.DeployError):
                    tools_deploy.compose_deploy_request(
                        purpose="p", files=[{"target": target,
                                             "content": "x"}])

    def test_secrets_are_refused_whatever_the_project_configured(self):
        """A project that forgets to list .env still cannot have it written."""
        with mock.patch.object(config, "DEPLOY_FORBIDDEN", ()):
            for name in ("config/.env", "config/.env.production",
                         "certs/server.pem", "certs/signing.key"):
                with self.subTest(name=name):
                    with self.assertRaises(tools_deploy.DeployError) as ctx:
                        tools_deploy.compose_deploy_request(
                            purpose="p",
                            files=[{"target": name, "content": "x"}])
                    self.assertIn("never be written", str(ctx.exception))

    def test_project_configured_refusals_are_honoured(self):
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="p",
                files=[{"target": "data/runtime-config.json",
                        "content": "{}"}])
        self.assertIn("runtime-config.json", str(ctx.exception))

    def test_a_missing_purpose_is_refused(self):
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="   ",
                files=[{"target": "engine/x.py", "content": "x"}])
        self.assertIn("purpose", str(ctx.exception))

    def test_an_empty_or_oversized_request_is_refused(self):
        with self.assertRaises(tools_deploy.DeployError):
            tools_deploy.compose_deploy_request(purpose="p", files=[])
        with mock.patch.object(config, "DEPLOY_MAX_FILES", 2):
            with self.assertRaises(tools_deploy.DeployError) as ctx:
                tools_deploy.compose_deploy_request(
                    purpose="p",
                    files=[{"target": f"engine/f{i}.py", "content": "x"}
                           for i in range(3)])
            self.assertIn("too many files", str(ctx.exception))
        with mock.patch.object(config, "DEPLOY_MAX_BYTES", 10):
            with self.assertRaises(tools_deploy.DeployError) as ctx:
                tools_deploy.compose_deploy_request(
                    purpose="p",
                    files=[{"target": "engine/big.py",
                            "content": "x" * 500}])
            self.assertIn("byte ceiling", str(ctx.exception))

    def test_duplicate_targets_and_missing_sources_are_refused(self):
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="p",
                files=[{"target": "engine/a.py", "content": "x"},
                       {"target": "engine/a.py", "content": "y"}])
        self.assertIn("same file twice", str(ctx.exception))
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="p",
                files=[{"target": "engine/a.py",
                        "source": str(self.root / "nope.py")}])
        self.assertIn("does not exist", str(ctx.exception))

    def test_services_and_verify_command_are_recorded(self):
        req = tools_deploy.compose_deploy_request(
            purpose="p", files=[{"target": "engine/x.py", "content": "x"}],
            services=["market-collector"], verify_command="true")
        doc = json.loads(Path(req["_request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["services"], ["market-collector"])
        self.assertEqual(doc["verify_command"], "true")


class GuardTest(DeployTestBase):
    def test_gated_mode_refuses_direct_edits_inside_the_root(self):
        self.mode("gated")
        reason = tools_deploy.guard_project_write(
            str(self.root / "engine" / "x.py"))
        self.assertIsNotNone(reason)
        self.assertIn("gated", reason)
        self.assertIn("deploy_request", reason)

    def test_gated_mode_allows_the_outbox_and_paths_outside_the_root(self):
        self.mode("gated")
        self.assertIsNone(tools_deploy.guard_project_write(
            str(self.outbox / "req-1.deploy.json")))
        outside = Path(tempfile.gettempdir()) / "agent-scratch.txt"
        self.assertIsNone(tools_deploy.guard_project_write(str(outside)))

    def test_direct_mode_does_not_block_the_edit_tools(self):
        self.mode("direct")
        self.assertIsNone(tools_deploy.guard_project_write(
            str(self.root / "engine" / "x.py")))

    def test_write_file_is_refused_in_gated_mode(self):
        """The gate must hold on the tool the agent would reach for first."""
        self.mode("gated")
        target = self.root / "engine" / "guarded.py"
        out = tools.write_file(str(target), "print('hi')\n")
        self.assertIn("REFUSED", out)
        self.assertIn("deploy_request", out)
        self.assertFalse(target.exists())

    def test_replace_in_file_is_refused_in_gated_mode(self):
        self.mode("gated")
        target = self.root / "engine" / "existing.py"
        target.write_text("old\n", encoding="utf-8")
        out = tools.replace_in_file(str(target), "old", "new")
        self.assertIn("REFUSED", out)
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_multi_replace_in_file_is_refused_in_gated_mode(self):
        self.mode("gated")
        target = self.root / "engine" / "multi.py"
        target.write_text("alpha beta\n", encoding="utf-8")
        out = tools.multi_replace_in_file(str(target),
                                          [{"old": "alpha", "new": "A"},
                                           {"old": "beta", "new": "B"}])
        self.assertIn("REFUSED", out)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha beta\n")


class ToolOutputTest(DeployTestBase):
    def test_gated_mode_stages_and_does_not_invoke_the_applier(self):
        self.mode("gated")
        with mock.patch.object(tools_deploy, "run_applier") as run:
            out = tools_deploy.deploy_request(
                purpose="patch the parser",
                files=[{"target": "engine/p.py", "content": "x"}])
        run.assert_not_called()
        self.assertIn("STAGED", out)
        self.assertIn("Gated mode", out)
        self.assertIn("engine/p.py", out)
        self.assertFalse((self.root / "engine" / "p.py").exists())

    def test_dry_run_evaluates_gates_and_writes_no_target(self):
        self.mode("direct")
        out = tools_deploy.deploy_request(
            purpose="rehearse the patch",
            files=[{"target": "engine/dry.py", "content": "x"}],
            dry_run=True)
        self.assertIn("status: dry_run", out)
        self.assertIn("DRY RUN", out)
        self.assertIn("readiness gate", out)
        self.assertIn("ready", out)
        self.assertFalse((self.root / "engine" / "dry.py").exists())
        self.assertTrue(list(self.outbox.glob("*.receipt.json")))

    def test_direct_mode_applies_through_the_applier_with_the_policy(self):
        self.mode("direct")
        out = tools_deploy.deploy_request(
            purpose="ship the fix",
            files=[{"target": "engine/live.py", "content": "FIXED\n"}],
            verify_command="true")
        self.assertIn("status: applied", out)
        self.assertEqual((self.root / "engine" / "live.py").read_text(
            encoding="utf-8"), "FIXED\n")
        self.assertIn("backups:", out)

    def test_an_unsafe_readiness_state_is_refused_by_the_host_gate(self):
        self.mode("direct")
        self.set_status("draining")
        out = tools_deploy.deploy_request(
            purpose="ship the fix",
            files=[{"target": "engine/blocked.py", "content": "x"}])
        self.assertIn("status: refused", out)
        self.assertIn("draining", out)
        self.assertFalse((self.root / "engine" / "blocked.py").exists())

    def test_a_stale_status_file_is_refused_even_when_it_says_safe(self):
        self.mode("direct")
        self.set_status("ready", stale_sec=7200)
        out = tools_deploy.deploy_request(
            purpose="ship the fix",
            files=[{"target": "engine/stale.py", "content": "x"}])
        self.assertIn("status: refused", out)
        self.assertIn("old", out)

    def test_a_failing_verification_command_leaves_nothing_applied(self):
        self.mode("direct")
        target = self.root / "engine" / "kept.py"
        target.write_text("ORIGINAL\n", encoding="utf-8")
        out = tools_deploy.deploy_request(
            purpose="ship a broken fix",
            files=[{"target": "engine/kept.py", "content": "BROKEN\n"}],
            verify_command="exit 3")
        self.assertIn("status: failed", out)
        self.assertIn("verification failed", out)
        self.assertEqual(target.read_text(encoding="utf-8"), "ORIGINAL\n")

    def test_a_missing_policy_is_refused_rather_than_run_without_gates(self):
        """Failing open is the one failure mode a gate may not have."""
        self.mode("direct")
        with mock.patch.object(config, "DEPLOY_POLICY",
                               str(self.root / "absent.json")):
            out = tools_deploy.deploy_request(
                purpose="ship the fix",
                files=[{"target": "engine/nopolicy.py", "content": "x"}])
        # "unapplied" is the honest status: the request was staged but no
        # applier ran, so no gate was evaluated and nothing was written.
        self.assertIn("status: unapplied", out)
        self.assertIn("missing file", out)
        self.assertIn("readiness gate", out)
        self.assertFalse((self.root / "engine" / "nopolicy.py").exists())

    def test_deploy_status_reports_pending_and_decided(self):
        self.mode("gated")
        tools_deploy.deploy_request(
            purpose="staged change",
            files=[{"target": "engine/s.py", "content": "x"}])
        out = tools_deploy.deploy_status()
        self.assertIn("mode: gated", out)
        self.assertIn("pending requests: 1", out)
        self.assertIn("staged change", out)
        self.assertIn("engine/s.py", out)

        self.mode("direct")
        tools_deploy.deploy_request(
            purpose="applied change",
            files=[{"target": "engine/a.py", "content": "y"}])
        out2 = tools_deploy.deploy_status()
        self.assertIn("decided receipts:", out2)
        self.assertIn("applied", out2)

    def test_compose_refusals_come_back_as_text_not_exceptions(self):
        """A tool must never raise into the model's transcript."""
        self.mode("gated")
        out = tools_deploy.deploy_request(purpose="p", files=[])
        self.assertIn("DEPLOY REQUEST REFUSED", out)
        out2 = tools_deploy.deploy_request(
            purpose="p",
            files=[{"target": "/etc/passwd", "content": "x"}])
        self.assertIn("REFUSED", out2)


class RegistrationTest(unittest.TestCase):
    """The tools must be reachable by the model with a valid schema."""

    def _schema(self, name):
        for entry in tools.TOOLS:
            fn = entry.get("function") or {}
            if fn.get("name") == name:
                return fn
        return None

    def test_both_tools_are_registered_as_handlers(self):
        self.assertIn("deploy_request", tools.HANDLERS)
        self.assertIn("deploy_status", tools.HANDLERS)
        self.assertIs(tools.HANDLERS["deploy_request"],
                      tools_deploy.deploy_request)
        self.assertIs(tools.HANDLERS["deploy_status"],
                      tools_deploy.deploy_status)

    def test_schemas_declare_the_arguments_the_handlers_accept(self):
        schema = self._schema("deploy_request")
        self.assertIsNotNone(schema, "deploy_request missing from TOOLS")
        params = schema["parameters"]
        self.assertEqual(sorted(params["required"]), ["files", "purpose"])
        props = params["properties"]
        for key in ("purpose", "files", "services", "verify_command",
                    "dry_run"):
            self.assertIn(key, props)
        self.assertEqual(props["files"]["type"], "array")
        self.assertEqual(
            sorted((props["files"]["items"]["properties"]).keys()),
            ["content", "source", "target"])

    def test_the_status_tool_takes_no_arguments(self):
        schema = self._schema("deploy_status")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["parameters"]["required"], [])
        self.assertEqual(schema["parameters"]["properties"], {})

    def test_tool_descriptions_explain_the_gate_not_just_the_syntax(self):
        """A model that does not know it cannot self-apply will not use this."""
        schema = self._schema("deploy_request")
        text = schema["description"].lower()
        for probe in ("host", "sha256", "dry_run"):
            self.assertIn(probe, text)


if __name__ == "__main__":
    unittest.main()
