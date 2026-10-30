"""Regression tests for the Copilot-parity layers.

Run:  python -m unittest test_harness_tools -v
Everything runs against temp directories; no network, no git repo required
except the git_guard unit checks (pure string classification).
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
import tools_edit  # noqa: E402
import tools_safety  # noqa: E402
import tools_session  # noqa: E402
import tools_todo  # noqa: E402
from tools_repo_map import build_repo_map  # noqa: E402


class SafetyTest(unittest.TestCase):
    def test_destructive_patterns_are_irreversible(self):
        for cmd in ("rm -rf build/", "git push --force origin main",
                    "git reset --hard HEAD~3", "DROP TABLE trades;",
                    "delete from trades", "docker volume rm data",
                    "git clean -fd", "dd if=/dev/zero of=/dev/sda"):
            risk, why = tools_safety.classify_command(cmd)
            self.assertEqual(risk, "irreversible", cmd)
            self.assertTrue(why, cmd)

    def test_delete_with_where_is_not_flagged(self):
        risk, _ = tools_safety.classify_command(
            "DELETE FROM trades WHERE id = 7;")
        self.assertEqual(risk, "ok")

    def test_caution_class_actions_pass_with_a_note(self):
        allowed, why = tools_safety.guard_command("git commit -m x")
        self.assertTrue(allowed)
        self.assertIn("caution", why)

    def test_irreversible_refused_without_approval(self):
        allowed, why = tools_safety.guard_command("rm -rf /tmp/x",
                                                  approve=False)
        self.assertFalse(allowed)
        self.assertIn("REFUSED", why)

    def test_irreversible_refused_when_approved_but_unattended(self):
        with mock.patch.object(config, "AUTO_APPROVE", False):
            allowed, why = tools_safety.guard_command(
                "rm -rf /tmp/x", approve=True, interactive=False)
        self.assertFalse(allowed)
        self.assertIn("no approval path", why)

    def test_auto_approve_allows_when_operator_opted_in(self):
        with mock.patch.object(config, "AUTO_APPROVE", True):
            allowed, why = tools_safety.guard_command(
                "rm -rf /tmp/x", approve=True, interactive=False)
        self.assertTrue(allowed)
        self.assertIn("auto-approved", why)

    def test_git_guard_gates_destructive_verbs(self):
        with mock.patch.object(config, "ALLOW_DESTRUCTIVE_GIT", False):
            allowed, why = tools_safety.git_guard("reset --hard")
        self.assertFalse(allowed)
        with mock.patch.object(config, "ALLOW_DESTRUCTIVE_GIT", True):
            allowed, _ = tools_safety.git_guard("reset --hard")
        self.assertTrue(allowed)


class EditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(tools_edit, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        tools_edit._EDIT_FAILS.clear()

    def _write(self, name, text):
        p = self.root / name
        p.write_text(text, encoding="utf-8")
        return p

    def test_replace_applies_unique_anchor(self):
        self._write("a.py", "x = 1\ny = 2\n")
        out = tools_edit.replace_in_file("a.py", "y = 2", "y = 3")
        self.assertIn("EDITED", out)
        self.assertEqual((self.root / "a.py").read_text(), "x = 1\ny = 3\n")

    def test_replace_refuses_ambiguous_and_missing(self):
        self._write("a.py", "dup\ndup\n")
        self.assertIn("NO EDIT APPLIED",
                      tools_edit.replace_in_file("a.py", "dup", "x"))
        self.assertIn("NO EDIT APPLIED",
                      tools_edit.replace_in_file("a.py", "nope", "x"))
        self.assertEqual((self.root / "a.py").read_text(), "dup\ndup\n")

    def test_multi_replace_is_atomic(self):
        self._write("a.py", "one\ntwo\n")
        out = tools_edit.multi_replace_in_file("a.py", [
            {"old": "one", "new": "1"},
            {"old": "missing", "new": "?"},
        ])
        self.assertIn("NO EDITS APPLIED", out)
        self.assertEqual((self.root / "a.py").read_text(), "one\ntwo\n")

    def test_three_failures_suggest_rollback(self):
        self._write("a.py", "keep\n")
        for _ in range(3):
            out = tools_edit.replace_in_file("a.py", "ghost", "x")
        self.assertIn("[rollback]", out)
        self.assertIn("git restore", out)
        # A success clears the counter.
        tools_edit.replace_in_file("a.py", "keep", "kept")
        out = tools_edit.replace_in_file("a.py", "ghost", "x")
        self.assertNotIn("[rollback]", out)


class RepoMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text(
            "class Thing(Base):\n"
            "    def method(self, a: int, b=2) -> str:\n"
            "        return 'x'\n"
            "def top_level(x):\n    pass\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "ignored.py").write_text(
            "class Secret:\n    pass\n", encoding="utf-8")

    def test_python_outline_has_signatures(self):
        m = build_repo_map(str(self.root))
        self.assertIn("class Thing(Base)", m)
        self.assertIn("def method(self, a: int, b=...) -> str", m)
        self.assertIn("def top_level(x)", m)

    def test_vcs_dirs_are_skipped(self):
        self.assertNotIn("Secret", build_repo_map(str(self.root)))

    def test_map_truncates_honestly(self):
        (self.root / "pkg" / "mod2.py").write_text(
            "class Other:\n    def m(self):\n        pass\n", encoding="utf-8")
        m = build_repo_map(str(self.root), max_chars=60)
        self.assertIn("truncated", m)
        # The first block still lands: "empty" would be a lie.
        self.assertIn("pkg/mod.py", m)


class TodoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(tools_todo, "SESSIONS_DIR",
                                    Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        tools_todo.set_todo_session("t1")

    def test_plan_round_trips_and_renders(self):
        out = tools_todo.manage_todo_list([
            {"id": 1, "title": "inspect", "status": "completed"},
            {"id": 2, "title": "fix", "status": "in-progress"},
        ])
        self.assertIn("[x] 1. inspect", out)
        self.assertIn("[>] 2. fix", out)
        self.assertIn("1/2 completed", out)
        again = tools_todo.todo_show()
        self.assertEqual(again, out)

    def test_two_in_progress_is_refused(self):
        out = tools_todo.manage_todo_list([
            {"id": 1, "title": "a", "status": "in-progress"},
            {"id": 2, "title": "b", "status": "in-progress"},
        ])
        self.assertIn("BAD PLAN", out)
        self.assertIn("(no todo list yet)", tools_todo.todo_show())

    def test_bad_status_is_refused(self):
        out = tools_todo.manage_todo_list(
            [{"id": 1, "title": "a", "status": "done-ish"}])
        self.assertIn("BAD STATUS", out)


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(tools_session, "SESSIONS_DIR",
                                    Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        (Path(self.tmp.name) / "s1.md").write_text(
            "task: fix the thing\nroot cause was a NaN\n", encoding="utf-8")

    def test_list_and_search(self):
        self.assertIn("s1.md", tools_session.session_list())
        hits = tools_session.session_search("NaN")
        self.assertIn("s1.md:2", hits)
        self.assertIn("(no matches", tools_session.session_search("zzz"))

    def test_read_reports_missing(self):
        self.assertIn("NOT FOUND", tools_session.session_read("nope"))


class DelegateTest(unittest.TestCase):
    def test_depth_limit_refuses(self):
        import tools_delegate
        with mock.patch.dict(os.environ, {"AGENT_DEPTH": "5"}):
            out = tools_delegate.run_subagent("do a thing")
        self.assertIn("REFUSED", out)

    def test_empty_prompt_refused(self):
        import tools_delegate
        with mock.patch.dict(os.environ, {"AGENT_DEPTH": "0"}):
            self.assertIn("REFUSED", tools_delegate.run_subagent("   "))


class CommandGuardIntegrationTest(unittest.TestCase):
    def test_run_command_refuses_destructive_without_approval(self):
        import tools
        out = tools.run_command("rm -rf ./nope")
        self.assertIn("REFUSED", out)

    def test_run_command_runs_safe_commands(self):
        import tools
        out = tools.run_command("echo hello-guard")
        self.assertIn("hello-guard", out)
        self.assertIn("exit=0", out)


if __name__ == "__main__":
    unittest.main()
