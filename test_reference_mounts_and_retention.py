"""Generic support for reference material and retained scratch work.

Two deployment-host failures are ported here without application-specific rules:

* an additional read-only mount must live OUTSIDE the read-only project mount;
  a nested mountpoint can make Docker reject the container before startup;
* scratch written for an asynchronous deploy must not be deleted while a pending
  request still names it, but abandoned scratch needs an age-based bound.

Run:  python -m unittest test_reference_mounts_and_retention
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import config
import retention
import sync_source

ROOT = Path(__file__).resolve().parent


class ReadOnlyReferenceMountTest(unittest.TestCase):
    """The example production compose must be safe to instantiate."""

    def _document(self):
        path = ROOT / "docker-compose.agent.yml.example"
        text = path.read_text(encoding="utf-8")
        self.assertIn("/agent-references:ro", text)
        self.assertIn("AGENT_REFERENCES_DIR", text)
        return text

    def test_reference_mount_is_outside_the_read_only_project_root(self):
        text = self._document()
        volume_lines = [line.strip() for line in text.splitlines()
                        if line.strip().startswith("- ${")]
        self.assertTrue(volume_lines)
        for line in volume_lines:
            spec = line[2:]
            # The last field is the container path. A nested reference mount
            # under a read-only parent is precisely the startup failure.
            if "/agent-references:ro" in spec:
                self.assertIn("/agent-references", spec)
                self.assertNotIn(":/project/", spec,
                                 "a reference mount nested below /project "
                                 "cannot be created through its read-only bind")
        self.assertTrue(any("/project:ro" in line for line in volume_lines),
                        "the target application must remain read-only")

    def test_reference_mount_does_not_disable_the_project_or_config_guards(self):
        text = self._document()
        self.assertIn("${AGENT_CONFIG_DIR:-./config}:/agent-config:ro", text)
        self.assertIn("${PROJECT_DIR:-../project}:/project:ro", text)

    @unittest.skipUnless(shutil.which("docker"), "docker is not installed")
    def test_rendered_compose_document_has_the_expected_mounts(self):
        """Validate the real YAML with Compose, not just line grepping."""
        proc = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.agent.yml.example",
             "config", "--format", "json"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            self.skipTest("docker compose is unavailable: "
                          + (proc.stderr or "")[-300:])
        import json
        doc = json.loads(proc.stdout)
        mounts = doc["services"]["agent"]["volumes"]
        by_target = {row["target"]: row for row in mounts}
        self.assertEqual(by_target["/project"]["read_only"], True)
        self.assertEqual(by_target["/agent-config"]["read_only"], True)
        self.assertEqual(by_target["/agent-references"]["read_only"], True)
        self.assertEqual(by_target["/agent-data"]["read_only"], False)


class ScratchRetentionTest(unittest.TestCase):
    """Age bounds old scratch without breaking a pending asynchronous deploy."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.outbox = self.root / "outbox"
        self.work.mkdir()
        self.outbox.mkdir()
        self.now = 2_000_000_000.0
        for key in ("AGENT_SCRATCH_DIR", "AGENT_SCRATCH_MAX_AGE_HOURS",
                    "AGENT_SCRATCH_GRACE_HOURS", "AGENT_SCRATCH_MIN_AGE_HOURS"):
            patch = mock.patch.dict(os.environ, {key: ""})
            patch.start()
            self.addCleanup(patch.stop)
        for key, value in (
                ("AGENT_SCRATCH_DIR", str(self.work)),
                ("AGENT_SCRATCH_MAX_AGE_HOURS", "168"),
                ("AGENT_SCRATCH_GRACE_HOURS", "48")):
            patch = mock.patch.dict(os.environ, {key: value})
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, name, age_hours=200, body=b"OLD = True\n"):
        path = self.work / name
        path.mkdir()
        (path / "module.py").write_bytes(body)
        stamp = self.now - age_hours * 3600
        for item in [path, path / "module.py"]:
            os.utime(item, (stamp, stamp))
        return path

    def test_old_abandoned_scratch_is_a_bounded_deletion(self):
        path = self.entry("abandoned")
        result = retention.prune_scratch(now=self.now, pending_request_ids=[],
                                         pending_request_texts=[])
        self.assertEqual(result["deleted"], ["abandoned"])
        self.assertFalse(path.exists())

    def test_a_recent_entry_is_kept_even_when_the_top_level_dir_is_old(self):
        """Mtime is checked recursively: a fresh fix inside an old dir survives.

        The newest file is 10 minutes old, inside the 48h grace, so disabling
        the grace branch is a real mutation: it would delete this active work
        despite the recent write.
        """
        path = self.entry("active")
        (path / "module.py").write_bytes(b"NEWER = True\n")
        os.utime(path / "module.py", (self.now - 600, self.now - 600))
        result = retention.prune_scratch(now=self.now, pending_request_ids=[],
                                         pending_request_texts=[])
        self.assertEqual(result["kept"], ["active"])
        self.assertTrue((path / "module.py").exists())

    def test_the_grace_gate_holds_at_a_very_old_parent_timestamp(self):
        """A recent file wins even if its parent directory is ancient.

        This is deliberately a separate fixture: it makes the grace boundary a
        direct input rather than depending on a directory mtime inherited from
        another test's setup.
        """
        path = self.entry("recent-inside-old", age_hours=10000)
        (path / "module.py").write_bytes(b"ACTIVE = True\n")
        os.utime(path / "module.py", (self.now - 600, self.now - 600))
        result = retention.prune_scratch(now=self.now, pending_request_ids=[],
                                         pending_request_texts=[])
        self.assertIn("recent-inside-old", result["kept"])
        self.assertTrue((path / "module.py").exists())


    def test_pending_request_reference_wins_over_age(self):
        path = self.entry("fix", body=b"UNAPPLIED = True\n")
        result = retention.prune_scratch(
            now=self.now, pending_request_ids=["req-1"],
            pending_request_texts=['{"source":"/agent-data/work/fix/module.py"}'])
        self.assertEqual(result["kept"], ["fix"])
        self.assertEqual(result["protected"], ["fix"])
        self.assertTrue((path / "module.py").exists())

    def test_pending_request_naming_the_tree_protects_every_entry(self):
        first = self.entry("a")
        second = self.entry("b")
        result = retention.prune_scratch(
            now=self.now, pending_request_ids=["req-1"],
            pending_request_texts=['{"source":"/agent-data/work"}'])
        self.assertEqual(result["deleted"], [])
        self.assertTrue(first.exists() and second.exists())

    def test_a_name_is_matched_as_a_path_segment_not_a_substring(self):
        kept = self.entry("prefix", body=b"OTHER = True\n")
        self.entry("fix", body=b"UNAPPLIED = True\n")
        result = retention.prune_scratch(
            now=self.now, pending_request_ids=["req-1"],
            pending_request_texts=['{"source":"/agent-data/work/prefix/module.py"}'])
        self.assertIn("fix", result["deleted"])
        self.assertIn("prefix", result["kept"])
        self.assertTrue(kept.exists())

    def test_dry_run_never_deletes(self):
        path = self.entry("abandoned")
        result = retention.prune_scratch(
            now=self.now, pending_request_ids=[], pending_request_texts=[],
            dry_run=True)
        self.assertEqual(result["deleted"], [])
        self.assertEqual(result["would_delete"], ["abandoned"])
        self.assertTrue(path.exists())

    def test_zero_max_age_is_refused(self):
        with mock.patch.dict(os.environ, {"AGENT_SCRATCH_MAX_AGE_HOURS": "0"}):
            with self.assertRaises(ValueError):
                retention.prune_scratch(now=self.now, pending_request_ids=[],
                                        pending_request_texts=[])


class ContentHashTest(unittest.TestCase):
    """Line endings must not masquerade as source drift."""

    def test_crlf_and_lf_hash_to_the_same_content(self):
        self.assertEqual(sync_source.content_md5(b"a = 1\r\nb = 2\r\n"),
                         sync_source.content_md5(b"a = 1\nb = 2\n"))

    def test_real_content_changes_do_not_collapse(self):
        self.assertNotEqual(sync_source.content_md5(b"a = 1\n"),
                            sync_source.content_md5(b"a = 2\n"))

    def test_the_file_hasher_uses_the_normalized_content_rule(self):
        import inspect
        source = inspect.getsource(sync_source.md5)
        self.assertIn("content_md5", source)
        self.assertNotIn("read_bytes()).hexdigest()[:16]", source)


if __name__ == "__main__":
    unittest.main()
