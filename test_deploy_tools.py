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
import inspect
import json
import shutil
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
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(purpose="p", files=[])
        #: The message must name the third legal shape, or a pass that COULD
        #: have used a rebuild-only request is told only that it is wrong.
        self.assertIn("service", str(ctx.exception))
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


    def test_a_rehearsal_through_the_tool_leaves_nothing_pending(self):
        """The CALL SITE is what matters, not the helper.

        An earlier version of this coverage called `_retire_rehearsal` directly
        and passed with the call removed from `deploy_request` -- the same
        "verified the helper, never the caller" shape this deployment has been
        bitten by before. Worse, the caller returned early when the applier was
        unavailable, which is the NORMAL case inside the sidecar, so the
        rehearsal stayed pending exactly where it was most dangerous.

        Here the applier is made to REFUSE (as it does in the sidecar) and the
        request must STILL leave the pending set.
        """
        with mock.patch.object(tools_deploy, "apply_request",
                               side_effect=tools_deploy.DeployError("no applier")):
            text = tools_deploy.deploy_request(
                purpose="rehearsal", dry_run=True,
                action={"kind": "set_instance_field",
                        "instance_id": "bei-spx",
                        "set": {"risk_profile": "balanced"}})
        self.assertIn("REHEARSAL ONLY", text)
        self.assertEqual([r for r in tools_deploy.pending_requests()
                          if r.get("pending")], [],
                         "a rehearsal is still in the pending set, so the host "
                         "timer would apply it for real")
        self.assertEqual(len(list((self.outbox / "rehearsed").glob("*"))), 1,
                         "the rehearsal was not kept for the audit trail")

    def test_a_dry_run_cannot_be_applied_later_by_the_host(self):
        """A rehearsal must not become the change.

        ``apply_request(dry_run=True)`` writes a receipt and leaves the request
        in the outbox -- which is exactly where the host applier looks, so on
        the next timer pass (three minutes in this deployment) it would apply
        the very change the agent only meant to test. The agent would report
        "rehearsed, nothing changed" and the change would land anyway: the worst
        shape of failure here, because the operator is told it is inert.
        """
        path = self.outbox / f"req-rehearse{tools_deploy.REQUEST_SUFFIX}"
        path.write_text(json.dumps({
            "schema_version": 1, "id": "req-rehearse",
            "created_at": "2026-11-01T00:00:00+00:00",
            "purpose": "rehearsal",
            "action": {"kind": "set_instance_field",
                       "instance_id": "bei-spx",
                       "set": {"risk_profile": "balanced"}},
        }), encoding="utf-8")
        moved = tools_deploy._retire_rehearsal(str(path))
        self.assertIsNotNone(moved, "the rehearsal stayed in the pending set")
        self.assertFalse(path.exists(), "the request is still pending")
        self.assertTrue((self.outbox / "rehearsed" / path.name).is_file(),
                        "the rehearsal was deleted instead of kept for audit")
        self.assertEqual(
            [p.name for p in self.outbox.parent.glob(f"*outbox/*{tools_deploy.REQUEST_SUFFIX}")],
            [], "a request the host would still apply")

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

    def test_direct_mode_against_a_read_only_root_explains_itself(self):
        """The sidecar mounts the root read-only, so `direct` cannot apply.

        The failure must name the fix. An OSError bubbling up from three layers
        down ("Read-only file system") leaves the model guessing, and guessing
        is how an agent starts trying to work around a safety property.
        """
        self.mode("direct")
        with mock.patch.object(tools_deploy, "root_is_writable",
                               return_value=False):
            out = tools_deploy.deploy_request(
                purpose="apply in place",
                files=[{"target": "engine/nope.py", "content": "x"}])
        self.assertIn("status: unapplied", out)
        self.assertIn("read-only", out)
        self.assertIn("AGENT_ROOT_MOUNT_MODE=rw", out)
        self.assertIn("ops/apply_deploys.py", out,
                      "must offer the gated path as the working alternative")
        self.assertFalse((self.root / "engine" / "nope.py").exists())

    def test_a_dry_run_is_still_allowed_on_a_read_only_root(self):
        """Rehearsal touches no target, so read-only must not block it."""
        self.mode("direct")
        with mock.patch.object(tools_deploy, "root_is_writable",
                               return_value=False):
            out = tools_deploy.deploy_request(
                purpose="rehearse only",
                files=[{"target": "engine/dry2.py", "content": "y"}],
                dry_run=True)
        self.assertIn("status: dry_run", out)
        self.assertNotIn("status: unapplied", out)
        self.assertFalse((self.root / "engine" / "dry2.py").exists())

    def test_gated_mode_is_unaffected_by_a_read_only_root(self):
        """Gated writes nothing itself, so the mount mode is irrelevant to it."""
        self.mode("gated")
        with mock.patch.object(tools_deploy, "root_is_writable",
                               return_value=False):
            out = tools_deploy.deploy_request(
                purpose="stage a proposal",
                files=[{"target": "engine/gated.py", "content": "z"}])
        self.assertIn("STAGED", out)
        self.assertIn("Gated mode", out)
        self.assertFalse((self.root / "engine" / "gated.py").exists())

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

    def setUp(self):
        """Scratch isolation, even though these tests only expect a refusal.

        Three of them call ``compose_deploy_request`` inside ``assertRaises``:
        the refusal is meant to happen before anything is staged, and today it
        does. That is an implementation detail of the validator, not a property
        of this test class -- and the moment a check is reordered so that
        staging happens first, these calls write into whatever
        ``config.DEPLOY_OUTBOX`` points at, which inside a deployed container is
        the live outbox. A class that can stage is a class that must be
        isolated; ``test_outbox_isolation`` enforces exactly that.
        """
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        root = Path(self.tmpdir.name)
        outbox = root / ".deploy" / "outbox"
        outbox.mkdir(parents=True)
        for attr, value in (("PROJECT_ROOT", str(root)),
                            ("DEPLOY_ROOT", str(root)),
                            ("DEPLOY_OUTBOX", str(outbox)),
                            ("DEPLOY_MODE", "direct")):
            patcher = mock.patch.object(config, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

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
        #: ``files`` is no longer REQUIRED, because an ACTION request carries no
        #: files. What must stay true is that a request cannot omit BOTH -- and
        #: that is enforced where it can be tested (compose refuses an empty
        #: request), rather than by a schema that makes one of the two shapes
        #: impossible to express.
        self.assertEqual(sorted(params["required"]), ["purpose"])
        props = params["properties"]
        for key in ("purpose", "files", "services", "verify_command",
                    "dry_run", "action"):
            self.assertIn(key, props)
        self.assertEqual(props["files"]["type"], "array")
        self.assertEqual(
            sorted((props["files"]["items"]["properties"]).keys()),
            ["content", "delete", "source", "target"],
            "the deletion operation must be declared or the model cannot form it")
        #: The action shape is what lets the agent change runtime state, so its
        #: three parts must be declared or the model cannot form the call.
        self.assertEqual(props["action"]["type"], "object")
        self.assertEqual(sorted((props["action"]["properties"]).keys()),
                         ["instance_id", "kind", "set"])

    def test_a_request_cannot_be_both_a_deploy_and_an_action(self):
        """Two kinds of change in one receipt cannot be reviewed as one unit."""
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="p",
                files=[{"target": "engine/x.py", "content": "x"}],
                action={"kind": "set_instance_field",
                        "instance_id": "bei-spx",
                        "set": {"risk_profile": "balanced"}})
        self.assertIn("not both", str(ctx.exception))

    def test_an_action_request_needs_a_kind_instance_and_a_scalar_value(self):
        for bogus in (
                {"instance_id": "bei-spx", "set": {"risk_profile": "safe"}},
                {"kind": "set_instance_field", "set": {"risk_profile": "safe"}},
                {"kind": "set_instance_field", "instance_id": "bei-spx"},
                {"kind": "set_instance_field", "instance_id": "bei-spx",
                 "set": {}},
                {"kind": "set_instance_field", "instance_id": "bei-spx",
                 "set": {"risk_profile": {"nested": 1}}}):
            with self.subTest(action=bogus):
                with self.assertRaises(tools_deploy.DeployError):
                    tools_deploy.compose_deploy_request(
                        purpose="p", action=bogus)

    def test_an_action_field_name_must_be_an_identifier(self):
        """A field name is a name, not a path into something else."""
        for bogus in ("risk/profile", "../risk_profile", "risk profile",
                      "risk_profile;rm"):
            with self.subTest(field=bogus):
                with self.assertRaises(tools_deploy.DeployError):
                    tools_deploy.compose_deploy_request(
                        purpose="p",
                        action={"kind": "set_instance_field",
                                "instance_id": "bei-spx",
                                "set": {bogus: "safe"}})

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


class PolicyForbidTest(DeployTestBase):
    """The early refusal must read the policy the HOST gate enforces.

    One list, two readers. Live 2026-10-30 the two copies had drifted -- the
    policy forbade the gate's own control-plane files, ``AGENT_DEPLOY_FORBIDDEN``
    did not -- so the container staged a proposal to rewrite
    ``ops/deploy_policy.json`` and only the host gate caught it. Nothing was
    ever unsafe: the host is authoritative and it refused. But two lists that
    can disagree is exactly how a second layer's failure becomes a silent
    bypass, so the container now reads the policy itself and the env list is
    only ever additive.
    """

    #: What a deployment ships: control-plane files, runtime state, directories.
    POLICY_FORBID = {"names": ["deploy_policy.json", "routes.json", "owner.svg"],
                     "dirs": ["secrets"],
                     "prefixes": ["collector.log"],
                     "suffixes": [".jsonl"]}

    def setUp(self):
        super().setUp()
        (self.root / "secrets").mkdir()
        (self.root / "ops").mkdir()
        doc = dict(POLICY_DOC, forbid=self.POLICY_FORBID)
        self.policy.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        # Start blank: prove the policy supplies the names, not the env list.
        self.patch_forbidden(("only-in-env.json",))
        self.clear_cache()
        self.addCleanup(self.clear_cache)

    @staticmethod
    def clear_cache():
        tools_deploy.reset_policy_cache()

    def patch_forbidden(self, specs):
        patcher = mock.patch.object(config, "DEPLOY_FORBIDDEN", specs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clear_cache()

    def tighten_policy(self, extra_name):
        """Rewrite the policy AND move its mtime clearly forward.

        Without the utime bump this test passes on the cache being absent and
        fails spuriously on a filesystem that records whole seconds, because the
        cache is keyed on mtime to stay live without a re-parse per target.
        """
        doc = dict(POLICY_DOC, forbid=dict(
            self.POLICY_FORBID, names=list(self.POLICY_FORBID["names"]) + [extra_name]))
        self.policy.write_text(json.dumps(doc), encoding="utf-8")
        os.utime(self.policy, (2 ** 31, 2 ** 31))
        self.clear_cache()

    def refuse_reason(self, target, content="payload"):
        with self.assertRaises(tools_deploy.DeployError) as exc:
            tools_deploy.compose_deploy_request(
                purpose="p", files=[{"target": target, "content": content}])
        return str(exc.exception)

    def test_policy_names_refuse_early_alone_the_env_list_does_not_carry_them(self):
        """The regression from the live drift, stated so it can never recur."""
        joined = " ".join(config.DEPLOY_FORBIDDEN or ())
        self.assertNotIn("deploy_policy.json", joined)
        self.assertEqual(tools_deploy.policy_forbid().get("names"),
                         self.POLICY_FORBID["names"])
        for target in ("ops/deploy_policy.json", "ops/agent/routes.json"):
            message = self.refuse_reason(target)
            self.assertIn("may never be written", message, target)
            self.assertTrue(tools_deploy.is_forbidden(target), target)

    def test_policy_directories_are_refused(self):
        """Runtime data and secrets are off limits by DIRECTORY, not by name."""
        for target in ("secrets/totp.txt", "secrets/nested/deep.json"):
            message = self.refuse_reason(target)
            self.assertIn("secrets", message, target)
            self.assertIn("policy", message.lower(), target)
        self.assertIn("secrets", tools_deploy.forbidden_dirs())

    def test_git_stays_refused_even_though_the_policy_never_mentions_it(self):
        """A deployment policy cannot opt a project into VCS corruption."""
        self.assertIn(".git", tools_deploy.forbidden_dirs())
        self.assertIn(".git", self.refuse_reason(".git/hooks/pre-commit"))

    def test_policy_prefixes_and_suffixes_apply(self):
        self.assertTrue(tools_deploy.is_forbidden("data/collector.log"))
        self.assertTrue(tools_deploy.is_forbidden("data/collector.log.1"))
        self.assertTrue(tools_deploy.is_forbidden("data/closed_positions.jsonl"))

    def test_env_layer_adds_refusals_the_policy_lacks(self):
        self.assertTrue(tools_deploy.is_forbidden("only-in-env.json"),
                        "the env list must still refuse on its own")

    def test_generic_secret_refusals_survive_a_policy_that_omits_them(self):
        """A policy can add to the generic floor, never subtract from it."""
        for target in ("server.key", "cert.pem", ".env"):
            self.assertTrue(tools_deploy.is_forbidden(target), target)

    def test_a_tightened_policy_takes_effect_without_a_restart(self):
        """The web UI is long-lived; an operator's edit must be seen live."""
        self.assertFalse(tools_deploy.is_forbidden("ops/new_control.json"))
        self.tighten_policy("new_control.json")
        self.assertTrue(tools_deploy.is_forbidden("ops/new_control.json"))

    def test_an_unreadable_policy_degrades_to_the_env_floor_it_does_not_fail(self):
        """The host gate stays authoritative, so this must never be a crash."""
        patcher = mock.patch.object(config, "DEPLOY_POLICY",
                                    str(self.root / "absent.json"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clear_cache()
        self.assertEqual({}, tools_deploy.policy_forbid())
        self.assertFalse(tools_deploy.is_forbidden("engine/collector.py"))
        self.assertTrue(tools_deploy.is_forbidden("server.key"),
                        "generic floor must apply with no policy at all")

    def test_a_malformed_policy_is_ignored_rather_than_raising(self):
        self.policy.write_text("{not json", encoding="utf-8")
        self.clear_cache()
        self.assertEqual({}, tools_deploy.policy_forbid())
        self.assertFalse(tools_deploy.is_forbidden("engine/collector.py"))

    def test_a_policy_without_a_forbid_section_is_legal(self):
        self.policy.write_text(json.dumps(
            {"readiness": POLICY_DOC["readiness"]}), encoding="utf-8")
        self.clear_cache()
        self.assertEqual({}, tools_deploy.policy_forbid())
        self.assertTrue(tools_deploy.is_forbidden(".env"))

    def test_direct_mode_runs_the_same_gate_so_a_proposal_is_refused_there_too(self):
        """Direct mode changes WHO runs the gate, never WHETHER one does.

        ``deploy_request`` is the tool boundary, so a refusal arrives as TEXT
        for the model to read rather than as a raise: an exception there would
        surface as a crash mid-task instead of as an explanation of what to do
        instead. ``compose_deploy_request`` below is the raising layer.
        """
        self.mode("direct")
        out = tools_deploy.deploy_request(
            purpose="widen my own safe states",
            files=[{"target": "ops/deploy_policy.json",
                    "content": '{"readiness": {"safe_states": ["busy"]}}'}],
            verify_command="true")
        self.assertIn("DEPLOY REQUEST REFUSED", out)
        self.assertIn("may never be written", out)
        self.assertFalse((self.root / "ops" / "deploy_policy.json").exists(),
                         "a refused deploy must not have written anything")
        self.assertEqual([], list(self.outbox.glob("*.deploy.json")),
                         "a refusal must not stage a request either")

    def test_the_gate_itself_cannot_be_rewritten_by_a_proposal(self):
        """Everything control-plane, whichever layer would catch it.

        Only names THIS fixture's policy forbids belong in the ``is_forbidden``
        loop: DIRECTORY refusals live one layer up, in ``_check_target``, which
        raises rather than returning a bool. ``is_forbidden('secrets/x')`` is
        False by design -- it inspects a leaf name -- while the proposal is
        still refused end to end, as the next two tests pin down. Asserting
        directory rules here would encode a false belief about the layering.
        """
        for target in ("ops/deploy_policy.json", "ops/agent/routes.json",
                       "engine/collector.log"):
            self.assertTrue(tools_deploy.is_forbidden(target), target)
            out = tools_deploy.deploy_request(
                purpose="probe", files=[{"target": target, "content": "x"}])
            self.assertIn("DEPLOY REQUEST REFUSED", out, target)
            self.assertEqual([], list(self.outbox.glob("*.deploy.json")),
                             f"{target} must not be staged")

    def test_policy_directories_refuse_at_the_layer_that_enforces_them(self):
        """``forbid.dirs`` is enforced by _check_target, which raises.

        Verified live 2026-10-30 rather than assumed: ``is_forbidden`` answers
        False for ``secrets/totp.txt`` because it only sees a leaf name, yet the
        proposal is refused and nothing is staged.
        """
        for target in ("secrets/totp.txt", "secrets/nested/deep.json"):
            self.assertFalse(tools_deploy.is_forbidden(target),
                             f"{target}: leaf-name check is not the dir layer")
            with self.assertRaises(tools_deploy.DeployError) as exc:
                tools_deploy.compose_deploy_request(
                    purpose="p", files=[{"target": target, "content": "x"}])
            self.assertIn("secrets", str(exc.exception), target)
            self.assertIn("policy", str(exc.exception).lower(), target)
        self.assertEqual([], list(self.outbox.glob("*.deploy.json")),
                         "a directory refusal must stage nothing")

    def test_git_is_refused_end_to_end_even_though_the_policy_never_mentions_it(self):
        """A deployment policy cannot opt a project into VCS corruption."""
        with self.assertRaises(tools_deploy.DeployError):
            tools_deploy.compose_deploy_request(
                purpose="p", files=[{"target": ".git/hooks/pre-commit",
                                     "content": "x"}])
        self.assertIn(".git", tools_deploy.forbidden_dirs())

    def test_a_policy_name_the_env_list_lacks_is_still_refused_end_to_end(self):
        """The drift regression, driven through the tool an agent actually calls."""
        self.patch_forbidden(())          # env list contributes nothing
        self.clear_cache()
        out = tools_deploy.deploy_request(
            purpose="widen my own safe states",
            files=[{"target": "ops/deploy_policy.json",
                    "content": '{"readiness": {"safe_states": ["busy"]}}'}])
        self.assertIn("DEPLOY REQUEST REFUSED", out)
        self.assertIn("deploy_policy.json", out)
        self.assertEqual([], list(self.outbox.glob("*.deploy.json")))
        self.assertFalse((self.root / "ops" / "deploy_policy.json").exists())


class ReceiptCompatibilityTest(unittest.TestCase):
    """The agent-facing summary must read a receipt of EITHER schema.

    A receipt is a persisted audit trail written by whichever applier version
    was installed at the time, and the readiness verdict used to be filed under
    a domain-flavoured key. ``deploy_status`` is how an agent learns what the
    host decided, so a summary that only knew the new name would report
    "readiness gate absent" for every verdict written before the rename -- the
    agent concluding the gate never ran, on a host where it did.
    """

    def test_both_schemas_report_the_same_thing(self):
        retired = {"status": "refused", "reason": "state is 'busy'",
                   "mode": "gated",
                   "broker": {"gate": "queue", "state": "busy",
                              "age_sec": 2.5}}
        current = {"status": "refused", "reason": "state is 'busy'",
                   "mode": "gated",
                   "readiness": {"gate": "queue", "state": "busy",
                                 "age_sec": 2.5}}
        lines_old = tools_deploy._summarise_outcome(retired).splitlines()
        lines_new = tools_deploy._summarise_outcome(current).splitlines()
        self.assertEqual(lines_old, lines_new)
        gate_line = [ln for ln in lines_new if ln.startswith("readiness gate")]
        self.assertEqual(len(gate_line), 1, lines_new)
        self.assertIn("queue", gate_line[0])
        self.assertIn("'busy'", gate_line[0])

    def test_a_receipt_with_no_verdict_says_nothing_about_one(self):
        """Silence is correct; a fabricated line would claim a gate ran."""
        self.assertNotIn(
            "readiness gate",
            tools_deploy._summarise_outcome({"status": "applied",
                                            "reason": "ok"}))

    def test_the_key_history_matches_the_schemas_owner(self):
        """Two modules, one list: tools_deploy cannot import the applier, so
        the duplicate is compared rather than trusted."""
        import deploy_applier
        self.assertEqual(tools_deploy.READINESS_KEYS,
                         deploy_applier.READINESS_KEYS)
        self.assertEqual(tools_deploy.READINESS_KEYS[0], "readiness")


class GitVerdictPassthroughTest(unittest.TestCase):
    """``deploy_status`` must tell the agent whether the change reached git.

    The applier records the outcome in the receipt, but that is host-side and
    invisible to the agent: unless the summary carries it through, an agent
    cannot say where the applied code now lives, and "deployed but never
    pushed" -- exactly the drift this feature ends -- reads as a clean deploy.
    """

    def test_a_pushed_commit_is_reported_with_its_sha(self):
        out = tools_deploy._summarise_outcome(
            {"status": "applied", "reason": "ok",
             "git": {"state": "pushed", "sha": "abc123"}})
        self.assertIn("git: pushed abc123", out, out)

    def test_an_unpushed_commit_is_reported_with_a_retry(self):
        """A silent local-only commit is the failure mode to avoid."""
        out = tools_deploy._summarise_outcome(
            {"status": "applied", "reason": "ok",
             "git": {"state": "committed_not_pushed", "sha": "def456",
                     "reason": "push failed", "retry": "cd /opt/m && git push"}})
        self.assertIn("git: committed_not_pushed", out)
        self.assertIn("retry:", out)

    def test_a_receipt_without_a_git_block_says_nothing_about_git(self):
        """Silence is correct: an old receipt must not imply a push."""
        out = tools_deploy._summarise_outcome({"status": "applied",
                                               "reason": "ok"})
        self.assertNotIn("git:", out, out)

    def test_the_status_dict_carries_the_block_through(self):
        """The dict callers rely on must expose it, not just the text."""
        import inspect
        src = inspect.getsource(tools_deploy)
        self.assertIn('"git": receipt.get("git")', src)


class StagedAgainstReachesTheAgentTest(unittest.TestCase):
    """A field the applier records must reach the ONLY reader that can act on it.

    The applier writes `staged_against` when a request overwrites files that
    changed after it was staged. That fact is only useful to the agent, hours
    later, deciding whether its staged change is still the right one -- and it
    can only learn it through `deploy_status`.

    Both ends of that chain were missing when the field was introduced: the
    outcome builder never copied it out of the receipt, and the summariser never
    printed it. The check was therefore written, persisted and UNREAD -- which is
    the same as not writing it, and worse than not writing it, because the
    receipt then implies somebody was told.

    These tests pin the chain in both directions so a future field cannot repeat
    the pattern: carried through, and rendered as a warning rather than a fact,
    because the applier applies the requested bytes either way and the only
    correct response is to LOOK.
    """

    RECEIPT = {
        "status": "applied",
        "reason": "all gates passed and payloads written",
        "files": [{"target": "/x/engine/mod.py", "bytes": 12,
                   "sha256": "abcdef1234567890"}],
        "staged_against": [{
            "target": "/x/engine/mod.py",
            "staged_against": "324556bcabde9945",
            "found_instead": "dde83b4a4c8aa00f",
            "would_overwrite": "5fbbe6d00381c8e6",
        }],
        "staged_against_warning": "the deployment changed these files AFTER "
                                  "this request was staged: /x/engine/mod.py",
    }

    def test_the_outcome_carries_the_field_out_of_the_receipt(self):
        src = inspect.getsource(tools_deploy)
        self.assertIn('"staged_against": receipt.get("staged_against")', src,
                      "the outcome builder must copy it, or the summariser has "
                      "nothing to print")

    def test_the_summary_warns_that_a_newer_change_was_overwritten(self):
        text = tools_deploy._summarise_outcome(dict(self.RECEIPT))
        self.assertIn("changed AFTER this request was staged", text)
        self.assertIn("324556bcabde9945", text)
        self.assertIn("dde83b4a4c8aa00f", text)
        self.assertIn("undid a newer fix", text,
                      "the reader must be told what to DO about it, not just "
                      "given hashes")

    def test_a_receipt_without_the_field_prints_no_warning(self):
        """The common case must stay quiet, or the warning becomes noise."""
        clean = dict(self.RECEIPT)
        clean.pop("staged_against")
        text = tools_deploy._summarise_outcome(clean)
        self.assertNotIn("changed AFTER this request was staged", text)

    def test_an_empty_list_is_treated_as_nothing_to_report(self):
        clean = dict(self.RECEIPT)
        clean["staged_against"] = []
        text = tools_deploy._summarise_outcome(clean)
        self.assertNotIn("changed AFTER this request was staged", text)

    # -- the same question, the OTHER answer -----------------------------
    #
    # A file whose base was never recorded cannot be compared. That is not a
    # change, and reporting it as one (which the applier did until 2026-11-02)
    # sends a pass hunting a revert that never happened. The distinction is
    # only useful if it reaches the agent, so it is carried and printed here.

    def test_the_outcome_carries_the_unknown_list_out_of_the_receipt(self):
        src = inspect.getsource(tools_deploy)
        self.assertIn('"staged_against_unknown": receipt.get(', src,
                      "carried but unprinted is invisible; unread is the same "
                      "as not written")

    def test_an_unknown_base_prints_a_note_that_denies_being_a_change(self):
        receipt = dict(self.RECEIPT)
        receipt["staged_against"] = []
        receipt.pop("staged_against_warning")
        receipt["staged_against_unknown"] = [{
            "target": "/x/engine/mod.py",
            "reason": "no base hash was recorded",
            "found_instead": "dde83b4a4c8aa00f",
            "would_overwrite": None,
        }]
        text = tools_deploy._summarise_outcome(receipt)
        self.assertIn("NOT checked", text)
        self.assertIn("not a report that they changed", text)
        self.assertIn("dde83b4a4c8aa00f", text)
        self.assertNotIn("changed AFTER this request was staged", text,
                         "an unknown base must never be summarised as a "
                         "change -- that is the defect this test pins")

    def test_a_receipt_with_a_known_change_prints_only_the_change(self):
        """No phantom note on the ordinary superseded path."""
        text = tools_deploy._summarise_outcome(dict(self.RECEIPT))
        self.assertIn("changed AFTER this request was staged", text)
        self.assertNotIn("NOT checked", text)


class RebuildOnlyStagingTest(unittest.TestCase):
    """The agent must be able to COMPOSE what the host will ACCEPT.

    Found live on 2026-11-02, after the applier had already been taught to take
    a fileless rebuild request: staging it still failed with "a deploy request
    must name at least one file, or carry an 'action'". The host would have
    applied it; the agent could not create it. Fixing one end of a seam and not
    the other leaves the same dead end, just harder to see -- and the pass that
    hits it reads the error as "this repair is not permitted".
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp) / "project"
        self.outbox = self.root / ".deploy" / "outbox"
        self.outbox.mkdir(parents=True)
        self._patches = [
            mock.patch.object(config, "PROJECT_ROOT", str(self.root)),
            mock.patch.object(config, "DEPLOY_ROOT", str(self.root)),
            mock.patch.object(config, "DEPLOY_MODE", "direct"),
            #: MANDATORY, and its absence was not a cosmetic gap.
            #: `outbox_dir()` prefers `config.DEPLOY_OUTBOX` over anything
            #: derived from PROJECT_ROOT/DEPLOY_ROOT. Locally that value is
            #: empty so the scratch outbox above was reached and every test
            #: passed; inside the sidecar AGENT_DEPLOY_OUTBOX points at the
            #: DEPLOYMENT'S LIVE OUTBOX, so these tests staged real requests
            #: there -- and the host applier applied them (it wrote a fixture
            #: `engine/x.py` into the live tree and attempted a rebuild of a
            #: service that does not exist). Same file, same tests, green on one
            #: box and destructive on the other. Pinning the outbox makes the
            #: environment irrelevant: see test_outbox_isolation.py, which
            #: fails if ANY staging test class forgets it.
            mock.patch.object(config, "DEPLOY_OUTBOX", str(self.outbox)),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_fileless_rebuild_request_can_be_staged(self):
        req = tools_deploy.compose_deploy_request(
            purpose="rebuild the stale image", files=[], services=["svc-a"])
        doc = json.loads(Path(req["_request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["files"], [])
        self.assertEqual(doc["services"], ["svc-a"])
        self.assertEqual(doc["purpose"], "rebuild the stale image")

    def test_the_summary_says_rebuild_only_rather_than_showing_nothing(self):
        """An empty target list must be explained, not left blank."""
        text = tools_deploy.deploy_request(
            purpose="rebuild the stale image", files=[], services=["svc-a"])
        self.assertIn("rebuild only", text)
        self.assertIn("svc-a", text)
        self.assertNotIn("targets: \n", text)

    def test_no_files_no_action_no_services_is_still_refused(self):
        """The relaxation stops at "asks for nothing"."""
        with self.assertRaises(tools_deploy.DeployError):
            tools_deploy.compose_deploy_request(purpose="p")

    # -- a duplicate must not be stageable while one is pending ------------
    #
    # The brief says "never re-stage an identical request". A monitoring pass did
    # it anyway on 2026-11-02 -- twice in 61 seconds, honestly flagging the
    # second as its own error. Both were rebuild-only requests for the same
    # service, so each runs `docker compose build` AND `up -d`: the container is
    # recreated TWICE for one repair. Harmless only because the book was closed.
    #
    # A prose rule does not survive a model that has lost track of what it
    # staged, and `deploy_status` is a snapshot it may have read earlier. So the
    # check lives where the duplicate is created, and the refusal names the
    # request already waiting.

    def test_staging_the_same_rebuild_twice_refuses_the_second(self):
        first = tools_deploy.compose_deploy_request(
            purpose="rebuild", files=[], services=["svc-a"])
        self.assertTrue(first.get("_request_path"))
        with self.assertRaises(tools_deploy.DeployError) as ctx:
            tools_deploy.compose_deploy_request(
                purpose="rebuild again please", files=[], services=["svc-a"])
        self.assertIn("already PENDING", str(ctx.exception))
        self.assertIn(first["id"], str(ctx.exception),
                      "the refusal must name the request already waiting, or "
                      "the reader has to search the outbox to find it")

    def test_rewording_does_not_make_it_a_different_request(self):
        """A retry usually comes with a fresh sentence; that is still a dup."""
        first = tools_deploy.compose_deploy_request(
            purpose="rebuild the stale image", files=[], services=["svc-a"])
        with self.assertRaises(tools_deploy.DeployError):
            tools_deploy.compose_deploy_request(
                purpose="REBUILD the stale image (retry)", files=[],
                services=["svc-a"])
        self.assertIn("svc-a", first["services"])

    def test_a_different_service_is_not_a_duplicate(self):
        tools_deploy.compose_deploy_request(purpose="rebuild", files=[],
                                           services=["svc-a"])
        second = tools_deploy.compose_deploy_request(
            purpose="rebuild", files=[], services=["svc-b"])
        self.assertTrue(second.get("_request_path"))

    def test_a_different_payload_is_not_a_duplicate(self):
        (self.root / "engine").mkdir(parents=True, exist_ok=True)
        (self.root / "engine" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (self.root / "engine" / "b.py").write_text("B = 1\n", encoding="utf-8")
        tools_deploy.compose_deploy_request(
            purpose="ship", files=[{"target": "engine/x.py",
                                    "source": str(self.root / "engine" / "a.py")}])
        second = tools_deploy.compose_deploy_request(
            purpose="ship", files=[{"target": "engine/x.py",
                                    "source": str(self.root / "engine" / "b.py")}])
        self.assertTrue(second.get("_request_path"),
                        "different content is different work")

    def test_a_decided_request_does_not_block_a_new_one(self):
        """Once decided, the same work may legitimately be staged again."""
        first = tools_deploy.compose_deploy_request(
            purpose="rebuild", files=[], services=["svc-a"])
        applied = self.root / ".deploy" / "outbox" / "applied"
        applied.mkdir(parents=True, exist_ok=True)
        shutil.move(first["_request_path"], str(applied / Path(
            first["_request_path"]).name))
        second = tools_deploy.compose_deploy_request(
            purpose="rebuild", files=[], services=["svc-a"])
        self.assertTrue(second.get("_request_path"))


if __name__ == "__main__":
    unittest.main()
