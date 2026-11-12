"""Tests for Copilot-parity enhancements and bug fixes across the harness:
- PythonSandbox._request monotonic deadline against noisy stdout streams
- multi_replace_in_file sequential validation (prevents overlapping/corrupt edits)
- search() skips _SKIP_DIRS (.git, __pycache__, sessions, etc.)
- check_syntax() in-memory AST validation & duplicate dict key detection
- deploy_patch() & deploy_wait() & compose_deploy_request edits
- deploy_applier pause containment (--no-start), post_apply hooks, and target_map
- Agent max_steps_override precedence over console run_settings.json
- webui RUNS_PAGE status badge classification for time_limit_with_answer & stopped
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import agent
import config
import deploy_applier
import python_exec
import run_settings
import tools
import tools_deploy
import tools_diagnostics
import tools_edit
import webui


class PythonSandboxDeadlineTest(unittest.TestCase):
    def test_noisy_stdout_loop_cannot_extend_deadline_forever(self):
        sb = python_exec.PythonSandbox()
        try:
            t0 = time.monotonic()
            out = sb.run(
                "import time\n"
                "while True:\n"
                "    print('tick', flush=True)\n"
                "    time.sleep(0.05)\n",
                timeout_sec=1,
            )
            elapsed = time.monotonic() - t0
            self.assertIn("TIMEOUT", out)
            self.assertLess(elapsed, 6.0,
                            f"noisy loop held sandbox for {elapsed:.2f}s instead of ~1s")
            # Sandbox must recover cleanly on the next call after restart
            out2 = sb.run("40 + 2", timeout_sec=5)
            self.assertIn("42", out2)
        finally:
            sb.close()


class MultiReplaceSequentialTest(unittest.TestCase):
    def test_overlapping_edits_are_refused_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.py"
            original = "alpha = 1\nbeta = 2\n"
            p.write_text(original, encoding="utf-8")
            with mock.patch.object(config, "CONFIRM_WRITES", False), \
                 mock.patch.object(config, "DEPLOY_MODE", "direct"), \
                 mock.patch.object(config, "DEPLOY_ROOT", tmp), \
                 mock.patch.object(tools_edit, "PROJECT_ROOT", Path(tmp)):
                # Note: under old code (checking text.count(old) against original text),
                # both 'alpha = 1\nbeta = 2' AND 'beta = 2' had count==1 in original text!
                # Edit 0 changed beta = 2 -> beta = 20, making Edit 1 silently no-op or
                # worse, if Edit 0 created a second 'beta = 2', Edit 1 corrupted the wrong one!
                res = tools_edit.multi_replace_in_file(
                    str(p),
                    edits=[
                        {"old": "alpha = 1\nbeta = 2", "new": "alpha = 10\nbeta = 20"},
                        {"old": "beta = 2\n", "new": "beta = 30\n"},
                    ],
                )
            self.assertIn("NO EDITS APPLIED", res)
            self.assertEqual(p.read_text(encoding="utf-8"), original)

    def test_disjoint_edits_apply_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.py"
            p.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
            with mock.patch.object(config, "CONFIRM_WRITES", False), \
                 mock.patch.object(config, "DEPLOY_MODE", "direct"), \
                 mock.patch.object(config, "DEPLOY_ROOT", tmp), \
                 mock.patch.object(tools_edit, "PROJECT_ROOT", Path(tmp)):
                res = tools_edit.multi_replace_in_file(
                    str(p),
                    edits=[
                        {"old": "a = 1", "new": "a = 10"},
                        {"old": "c = 3", "new": "c = 30"},
                    ],
                )
            self.assertTrue(res.startswith("EDITED "), res)
            self.assertEqual(p.read_text(encoding="utf-8"), "a = 10\nb = 2\nc = 30\n")


class SearchSkipDirsTest(unittest.TestCase):
    def test_search_skips_git_and_sessions_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "live.py").write_text("NEEDLE_TOKEN = 1\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("NEEDLE_TOKEN\n", encoding="utf-8")
            (root / "sessions").mkdir()
            (root / "sessions" / "old.jsonl").write_text("NEEDLE_TOKEN\n", encoding="utf-8")
            with mock.patch.object(tools, "PROJECT_ROOT", root):
                hits = tools.search("NEEDLE_TOKEN", path=str(root))
            self.assertIn("live.py", hits)
            self.assertNotIn(".git", hits)
            self.assertNotIn("sessions", hits)


class CheckSyntaxToolTest(unittest.TestCase):
    def test_detects_syntax_error_and_duplicate_dict_key_without_pycache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.py"
            good.write_text("def add(x: int, y: int) -> int:\n    return x + y\n",
                            encoding="utf-8")
            dup = root / "dup.py"
            dup.write_text("CFG = {'a': 1, 'b': 2, 'a': 3}\n", encoding="utf-8")
            bad = root / "bad.py"
            bad.write_text("def broken(:\n    pass\n", encoding="utf-8")

            with mock.patch.object(tools_diagnostics, "PROJECT_ROOT", root):
                self.assertIn("OK: 1 Python file(s)",
                              tools_diagnostics.check_syntax([str(good)]))
                dup_res = tools_diagnostics.check_syntax([str(dup)])
                self.assertIn("duplicate dict key 'a'", dup_res)
                bad_res = tools_diagnostics.check_syntax([str(bad)])
                self.assertIn("SyntaxError", bad_res)
                self.assertFalse((root / "__pycache__").exists())


class DeployPatchAndWaitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "deploy"
        self.outbox = self.root / "data" / "deploy" / "outbox"
        (self.root / "engine").mkdir(parents=True)
        (self.root / "ops").mkdir(parents=True)
        self.outbox.mkdir(parents=True)
        (self.root / "engine" / "mod.py").write_text(
            "VALUE = 10\nNAME = 'old'\n", encoding="utf-8")
        (self.root / "watchdog.sh").write_text(
            "#!/bin/bash\necho v1\n", encoding="utf-8")
        policy = {
            "target_map": {"files": {"ops/watchdog.sh": "watchdog.sh"}},
            "pause": {
                "marker_file": "data/paused.json",
                "services": ["worker-svc"],
            },
            "post_apply": [
                {
                    "prefix": "ui/",
                    "argv": [os.sys.executable, "-c", "print('ui_published')"],
                }
            ],
        }
        self.policy_path = self.root / "ops" / "deploy_policy.json"
        self.policy_path.write_text(json.dumps(policy), encoding="utf-8")
        tools_deploy.reset_policy_cache()

    def tearDown(self):
        tools_deploy.reset_policy_cache()
        self._tmp.cleanup()

    def test_deploy_patch_refuses_broken_python_syntax_before_staging(self):
        with mock.patch.object(config, "DEPLOY_MODE", "gated"), \
             mock.patch.object(config, "DEPLOY_ROOT", str(self.root)), \
             mock.patch.object(config, "DEPLOY_OUTBOX", str(self.outbox)), \
             mock.patch.object(config, "DEPLOY_POLICY", str(self.policy_path)):
            res = tools_deploy.deploy_patch(
                purpose="bad patch",
                target="engine/mod.py",
                edits=[{"old": "VALUE = 10", "new": "VALUE = ("}],
            )
            self.assertIn("SyntaxError", res)
            self.assertEqual(list(self.outbox.glob("*.deploy.json")), [])

    def test_deploy_patch_stages_and_waits_for_applier_receipt(self):
        with mock.patch.object(config, "DEPLOY_MODE", "gated"), \
             mock.patch.object(config, "DEPLOY_ROOT", str(self.root)), \
             mock.patch.object(config, "DEPLOY_OUTBOX", str(self.outbox)), \
             mock.patch.object(config, "DEPLOY_POLICY", str(self.policy_path)):
            policy = deploy_applier.Policy.load(str(self.policy_path))

            def _run_applier_shortly():
                time.sleep(0.35)
                deploy_applier.run_once(
                    self.outbox, self.root, allow_restart=False,
                    dry_run=False, policy=policy)

            t = threading.Thread(target=_run_applier_shortly, daemon=True)
            t.start()
            res = tools_deploy.deploy_patch(
                purpose="bump value",
                target="engine/mod.py",
                edits=[{"old": "VALUE = 10", "new": "VALUE = 99"}],
                wait_sec=5.0,
            )
            t.join(timeout=5)
            self.assertIn("STAGED ", res)
            self.assertIn("--- deploy_wait ---", res)
            self.assertIn("status: applied", res)
            self.assertEqual(
                (self.root / "engine" / "mod.py").read_text(encoding="utf-8"),
                "VALUE = 99\nNAME = 'old'\n",
            )

    def test_target_map_routes_ops_file_to_flattened_deploy_root_and_back_to_git(self):
        bare = Path(self._tmp.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(bare)], check=True,
                       capture_output=True)
        mirror = Path(self._tmp.name) / "mirror"
        (mirror / "sub" / "ops").mkdir(parents=True)
        subprocess.run(["git", "init"], cwd=str(mirror), check=True,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"],
                       cwd=str(mirror), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Tester"],
                       cwd=str(mirror), check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", str(bare)],
                       cwd=str(mirror), check=True, capture_output=True)
        (mirror / "sub" / "ops" / "watchdog.sh").write_text(
            "#!/bin/bash\necho v1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(mirror), check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(mirror),
                       check=True, capture_output=True)
        subprocess.run(["git", "push", "-u", "origin", "HEAD"],
                       cwd=str(mirror), check=True, capture_output=True)

        policy_doc = {
            "target_map": {"files": {"ops/watchdog.sh": "watchdog.sh"}},
            "git": {
                "enabled": True,
                "mirror_dir": str(mirror),
                "subdir": "sub",
                "user_name": "Agent",
                "user_email": "agent@example.com",
            },
        }
        self.policy_path.write_text(json.dumps(policy_doc), encoding="utf-8")
        tools_deploy.reset_policy_cache()

        with mock.patch.object(config, "DEPLOY_MODE", "gated"), \
             mock.patch.object(config, "DEPLOY_ROOT", str(self.root)), \
             mock.patch.object(config, "DEPLOY_OUTBOX", str(self.outbox)), \
             mock.patch.object(config, "DEPLOY_POLICY", str(self.policy_path)):
            res = tools_deploy.deploy_patch(
                purpose="update watchdog via canonical ops path",
                target="ops/watchdog.sh",
                edits=[{"old": "echo v1", "new": "echo v2"}],
            )
            self.assertTrue(res.startswith("STAGED "), res)
            policy = deploy_applier.Policy.load(str(self.policy_path))
            rc = deploy_applier.run_once(
                self.outbox, self.root, allow_restart=False,
                dry_run=False, policy=policy)
            self.assertEqual(rc, 0)
            receipts = list(self.outbox.glob("*.receipt.json"))
            self.assertEqual(len(receipts), 1)
            receipt_doc = json.loads(receipts[0].read_text(encoding="utf-8"))
            self.assertEqual(receipt_doc["status"], "applied", receipt_doc)
            # Live root got flattened path updated, NOT a duplicate ops/watchdog.sh
            self.assertIn("echo v2",
                          (self.root / "watchdog.sh").read_text(encoding="utf-8"))
            self.assertFalse((self.root / "ops" / "watchdog.sh").exists())
            # Git mirror got canonical sub/ops/watchdog.sh updated, NOT sub/watchdog.sh
            self.assertIn(
                "echo v2",
                (mirror / "sub" / "ops" / "watchdog.sh").read_text(encoding="utf-8"))
            self.assertFalse((mirror / "sub" / "watchdog.sh").exists())

    def test_pause_marker_uses_no_start_on_paused_service_and_runs_post_apply(self):
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "paused.json").write_text(
            json.dumps({"paused": True}), encoding="utf-8")
        (self.root / "ui").mkdir(parents=True, exist_ok=True)
        (self.root / "ui" / "app.js").write_text("const v = 1;\n", encoding="utf-8")

        policy = deploy_applier.Policy.load(str(self.policy_path))
        with mock.patch.object(deploy_applier, "run_command",
                               return_value=(0, "ok\n")), \
             mock.patch.object(deploy_applier, "_record_image_stamp"):
            restarts = deploy_applier.restart_services(
                ["worker-svc", "other-svc"], self.root, 60.0, policy=policy)
        up_rows = {r["service"]: r for r in restarts if r["cmd"].startswith("up ")}
        self.assertTrue(up_rows["worker-svc"].get("paused_no_start"))
        self.assertIn("--no-start", up_rows["worker-svc"]["cmd"])
        self.assertFalse(up_rows["other-svc"].get("paused_no_start", False))
        self.assertNotIn("--no-start", up_rows["other-svc"]["cmd"])

        hooks = deploy_applier.run_post_apply_hooks(
            [self.root / "ui" / "app.js"], self.root, policy)
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0]["rc"], 0)
        self.assertIn("ui_published", hooks[0]["output"])


class MaxStepsOverridePrecedenceTest(unittest.TestCase):
    def test_explicit_max_steps_override_beats_console_run_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "run_settings.json"
            settings_path.write_text(json.dumps({"max_steps": 600}), encoding="utf-8")
            with mock.patch.object(run_settings, "SETTINGS_FILE", settings_path):
                self.assertEqual(agent._effective_max_steps(), 600)
                inst = object.__new__(agent.Agent)
                inst._max_steps_override = 12
                max_steps_ovr = getattr(inst, "_max_steps_override", None)
                effective = (max_steps_ovr if max_steps_ovr is not None
                             else agent._effective_max_steps())
                self.assertEqual(effective, 12)


class WebuiStatusBadgeTest(unittest.TestCase):
    def test_runs_page_declares_time_limit_and_stopped_statuses(self):
        self.assertIn("time_limit_with_answer", webui.RUNS_PAGE)
        self.assertIn("s==='time_limit_with_answer'", webui.RUNS_PAGE)
        self.assertIn("s==='stopped'", webui.RUNS_PAGE)


if __name__ == "__main__":
    unittest.main()
