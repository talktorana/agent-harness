"""Deletion through the deploy gate: a scoped, reversible file operation.

An agent that can only CREATE or OVERWRITE source cannot remove a stale module.
It reports the same cleanup every pass while the deployed file keeps being
imported. Deletion therefore belongs in the existing audited request pipeline,
not in a new shell permission:

* the target is relative to the deploy root and passes every existing path gate;
* it carries the hash of the exact file reviewed when the request was staged;
* a changed base is REFUSED, because deleting a newer revision erases a fix;
* the old bytes are backed up before removal and restored by blocking verification;
* an absent target is idempotent, but still mirrored -- the repository may be
  the only place the stale file still exists;
* the git mirror stages ONLY the named paths, and a real temporary clone proves
  that removal is committed and pushed.

Run:  python -m unittest test_deploy_delete
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import config
import deploy_applier
import tools_deploy


class DeployDeleteBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "engine").mkdir()
        (self.root / "data").mkdir()
        self.outbox = self.root / "outbox"
        self.outbox.mkdir()
        (self.root / "data" / "readiness.json").write_text(
            json.dumps({"state": "ready"}), encoding="utf-8")
        self.policy = deploy_applier.Policy({
            "readiness": {"status_file": "data/readiness.json",
                          "field": "state", "safe_states": ["ready"],
                          "max_age_sec": 900},
            "limits": {"max_age_sec": 900},
        })
        for key, value in (
                ("DEPLOY_ROOT", str(self.root)),
                ("DEPLOY_OUTBOX", str(self.outbox)),
                ("DEPLOY_MODE", "gated"),
                ("DEPLOY_FORBIDDEN", ()),
                ("DEPLOY_MAX_FILES", 20),
                ("DEPLOY_MAX_BYTES", 4 << 20),
                ("DEPLOY_POLICY", "")):
            patch = mock.patch.object(config, key, value)
            patch.start()
            self.addCleanup(patch.stop)

    def target(self, name="obsolete.py", body=b"OLD = True\n") -> Path:
        path = self.root / "engine" / name
        path.write_bytes(body)
        return path

    @staticmethod
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def request(self, entries, rid="req-delete", **over):
        files = []
        for entry in entries:
            row = dict(entry)
            if row.get("delete") is True:
                target = self.root / row["target"]
                row.setdefault("staged_against_sha256", self.sha(target))
            files.append(row)
        doc = {"schema_version": 1, "id": rid, "purpose": "remove stale module",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "files": files}
        doc.update(over)
        path = self.outbox / f"{rid}.deploy.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path


class ComposeDeletionTest(DeployDeleteBase):
    def test_delete_is_staged_with_the_reviewed_base_hash(self):
        target = self.target()
        req = tools_deploy.compose_deploy_request(
            purpose="remove an obsolete module",
            files=[{"target": "engine/obsolete.py", "delete": True}])
        row = req["files"][0]
        self.assertIs(row["delete"], True)
        self.assertEqual(row["staged_against_sha256"], self.sha(target))
        for forbidden in ("content", "source", "payload_path", "sha256", "bytes"):
            self.assertNotIn(forbidden, row,
                             "a deletion must not carry a fake payload identity")

    def test_delete_requires_an_existing_readable_file(self):
        for name in ("absent.py", "missing-directory/nested.py"):
            with self.subTest(target=name):
                with self.assertRaises(tools_deploy.DeployError) as ctx:
                    tools_deploy.compose_deploy_request(
                        purpose="p",
                        files=[{"target": "engine/" + name, "delete": True}])
                self.assertIn("existing readable", str(ctx.exception))

    def test_delete_cannot_be_combined_with_a_payload_in_one_entry(self):
        self.target()
        with self.assertRaises(tools_deploy.DeployError):
            tools_deploy.compose_deploy_request(
                purpose="p", files=[{"target": "engine/obsolete.py",
                                     "delete": True, "content": "surprise"}])

    def test_delete_uses_all_existing_path_refusals(self):
        for target in ("../outside.py", "engine/../../outside.py"):
            with self.subTest(target=target):
                with self.assertRaises(tools_deploy.DeployError):
                    tools_deploy.compose_deploy_request(
                        purpose="p", files=[{"target": target, "delete": True}])
        self.assertEqual(list(self.outbox.glob("*.deploy.json")), [])


class ApplyDeletionTest(DeployDeleteBase):
    def test_apply_backs_up_then_deletes_and_reports_the_operation(self):
        target = self.target(body=b"RECOVERABLE = 'yes'\n")
        path = self.request([{"target": "engine/obsolete.py", "delete": True}])
        result = deploy_applier.handle_request(
            path, self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(result["status"], "applied", result.get("reason"))
        self.assertFalse(target.exists())
        self.assertEqual(result["files"][0]["operation"], "delete")
        self.assertIn("1 file(s) deleted", result["reason"])
        backups = list((self.outbox / "backups" / "req-delete").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"RECOVERABLE = 'yes'\n")

    def test_a_changed_base_is_refused_and_the_newer_file_survives(self):
        target = self.target(body=b"REVIEWED = 1\n")
        path = self.request([{"target": "engine/obsolete.py", "delete": True}])
        target.write_bytes(b"NEWER_FIX = 2\n")
        result = deploy_applier.handle_request(
            path, self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(result["status"], "refused")
        self.assertIn("changed after staging", result["reason"])
        self.assertEqual(target.read_bytes(), b"NEWER_FIX = 2\n")

    def test_a_directory_cannot_be_deleted(self):
        target = self.root / "engine" / "package"
        target.mkdir()
        (target / "module.py").write_text("x = 1\n", encoding="utf-8")
        doc = {"schema_version": 1, "id": "req-dir", "purpose": "bad delete",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "files": [{"target": "engine/package", "delete": True,
                          "staged_against_sha256": "not-a-directory-hash"}]}
        path = self.outbox / "req-dir.deploy.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        result = deploy_applier.handle_request(
            path, self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(result["status"], "refused")
        self.assertIn("directory", result["reason"])
        self.assertTrue((target / "module.py").is_file())

    def test_dry_run_evaluates_the_gates_without_deleting(self):
        target = self.target()
        path = self.request([{"target": "engine/obsolete.py", "delete": True}])
        result = deploy_applier.handle_request(
            path, self.outbox, self.root, dry_run=True, policy=self.policy)
        self.assertEqual(result["status"], "dry_run")
        self.assertTrue(target.is_file())
        self.assertEqual(result["files"][0]["operation"], "delete")

    def test_blocking_verification_restores_the_deleted_bytes(self):
        target = self.target(body=b"ORIGINAL = True\n")
        path = self.request(
            [{"target": "engine/obsolete.py", "delete": True}],
            verify_command="exit 1")
        with mock.patch.object(deploy_applier, "run_command",
                               return_value=(1, "verifier failed")):
            result = deploy_applier.handle_request(
                path, self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(target.read_bytes(), b"ORIGINAL = True\n")
        self.assertEqual(result["verify_rc"], 1)

    def test_write_and_delete_can_share_one_atomic_request_and_recreate(self):
        deleted = self.target("old.py", b"OLD = True\n")
        path = self.request(
            [{"target": "engine/old.py", "delete": True},
             {"target": "engine/new.py", "content": "NEW = True\n",
              "sha256": hashlib.sha256(b"NEW = True\n").hexdigest()}],
            services=["worker"])
        with mock.patch.object(deploy_applier, "restart_services",
                               return_value=[{"service": "worker",
                                              "cmd": "up", "rc": 0}]) as restart:
            result = deploy_applier.handle_request(
                path, self.outbox, self.root, dry_run=False, policy=self.policy,
                allow_restart=True)
        self.assertEqual(result["status"], "applied", result.get("reason"))
        self.assertFalse(deleted.exists())
        self.assertEqual((self.root / "engine" / "new.py").read_bytes(),
                         b"NEW = True\n")
        restart.assert_called_once()
        self.assertIn("1 payload(s) written", result["reason"])
        self.assertIn("1 file(s) deleted", result["reason"])

    def test_an_already_absent_target_still_reaches_the_git_mirror(self):
        target = self.root / "engine" / "obsolete.py"
        # The request was reviewed against this content, then somebody removed
        # the deployed file before the applier ran. That is an idempotent host
        # outcome, not a reason to skip repository cleanup.
        base = hashlib.sha256(b"OLD = True\n").hexdigest()
        doc = {"schema_version": 1, "id": "req-absent", "purpose": "cleanup",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "files": [{"target": "engine/obsolete.py", "delete": True,
                          "staged_against_sha256": base}]}
        path = self.outbox / "req-absent.deploy.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with mock.patch.object(deploy_applier, "sync_applied_files",
                               return_value={"state": "disabled"}) as sync:
            result = deploy_applier.handle_request(
                path, self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(result["status"], "applied")
        self.assertFalse(target.exists())
        sync.assert_called_once()


@unittest.skipUnless(shutil.which("git"), "git is required for the mirror test")
class StatusDeletionTest(DeployDeleteBase):
    """The agent can see WHAT was removed, its backup, and the service result."""

    def test_the_status_summary_names_deletion_backup_and_recreate(self):
        target = self.target()
        path = self.request([{"target": "engine/obsolete.py", "delete": True}],
                            services=["worker"])
        with mock.patch.object(deploy_applier, "restart_services",
                               return_value=[{"service": "worker", "cmd": "up",
                                              "rc": 0}]):
            result = deploy_applier.handle_request(
                path, self.outbox, self.root, dry_run=False, policy=self.policy,
                allow_restart=True)
        # In gated mode apply_request deliberately does not invoke an applier;
        # the host has already applied this request, so read its receipt back
        # through the same helper the status tool uses.
        with mock.patch.object(tools_deploy, "is_gated", return_value=True), \
             mock.patch.object(tools_deploy, "run_applier", return_value=(0, "")):
            outcome = tools_deploy.apply_request(
                {"_request_path": str(path)}, dry_run=False)
        text = tools_deploy._summarise_outcome(outcome)
        self.assertEqual(outcome["status"], "applied")
        self.assertIn("obsolete.py (DELETE)", text)
        self.assertIn("backups", text)
        self.assertIn("req-delete", text)
        self.assertIn("service worker: up -> rc=0", text)
        self.assertFalse(target.exists())


class GitMirrorDeletionTest(unittest.TestCase):
    """A REAL clone and bare remote: no mocked copy or commit outcome."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.remote = self.base / "remote.git"
        self.mirror = self.base / "mirror"
        self.root = self.base / "deployment"
        self.root.mkdir()
        self.git("init", "--bare", str(self.remote))
        self.git("clone", str(self.remote), str(self.mirror))
        self.git("checkout", "-b", "main", cwd=self.mirror)
        tracked = self.mirror / "pkg" / "engine" / "obsolete.py"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("OLD = True\n", encoding="utf-8")
        self.git("add", "--", "pkg/engine/obsolete.py", cwd=self.mirror)
        self.git("-c", "user.name=test", "-c", "user.email=test@example.invalid",
                 "commit", "-m", "seed", cwd=self.mirror)
        self.git("push", "-u", "origin", "main", cwd=self.mirror)
        self.policy = deploy_applier.Policy({"git": {
            "enabled": True, "mirror_dir": str(self.mirror), "subdir": "pkg",
            "remote": "origin", "branch": "main",
            "user_name": "test", "user_email": "test@example.invalid"}})

    @staticmethod
    def git(*args, cwd=None):
        proc = subprocess.run(["git", *map(str, args)], cwd=str(cwd) if cwd else None,
                              capture_output=True, text=True)
        if proc.returncode:
            raise AssertionError(f"git {args} failed: {proc.stdout}{proc.stderr}")
        return (proc.stdout or "").strip()

    def test_an_applied_delete_is_committed_and_pushed_not_copied(self):
        target = self.root / "engine" / "obsolete.py"
        self.assertFalse(target.exists(), "the deployed deletion already happened")
        result = deploy_applier.sync_applied_files(
            [target], self.root, policy=self.policy, rid="req-delete-mirror",
            purpose="remove obsolete source")
        self.assertEqual(result["state"], "pushed", result.get("reason"))
        tracked = self.mirror / "pkg" / "engine" / "obsolete.py"
        self.assertFalse(tracked.exists(), "the mirror still carries the deleted file")
        subject = self.git("log", "-1", "--pretty=%s", cwd=self.mirror)
        self.assertIn("req-delete-mirror", subject)
        local = self.git("rev-parse", "HEAD", cwd=self.mirror)
        remote = self.git("--git-dir", str(self.remote), "rev-parse", "main")
        self.assertEqual(local, remote, "the deletion commit was not pushed")

    def test_a_later_unrelated_deploy_cannot_resurrect_a_deleted_file(self):
        """A stale file in the mirror is not authority to recreate it.

        A subsequent request may change only another file. The mirror refreshes
        from the remote first, so an earlier applied deletion is present before
        that request stages; without this ordering the old file could quietly
        come back while the receipt describes only the newer change.
        """
        target = self.root / "engine" / "obsolete.py"
        first = deploy_applier.sync_applied_files(
            [target], self.root, policy=self.policy, rid="req-remove",
            purpose="remove obsolete source")
        self.assertEqual(first["state"], "pushed", first.get("reason"))

        other = self.root / "engine" / "current.py"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("CURRENT = 2\n", encoding="utf-8")
        second = deploy_applier.sync_applied_files(
            [other], self.root, policy=self.policy, rid="req-unrelated",
            purpose="change only the current source")
        self.assertEqual(second["state"], "pushed", second.get("reason"))
        self.assertFalse((self.mirror / "pkg" / "engine" / "obsolete.py").exists(),
                         "an unrelated later deploy recreated the deleted file")
        self.assertTrue((self.mirror / "pkg" / "engine" / "current.py").is_file())


if __name__ == "__main__":
    unittest.main()
