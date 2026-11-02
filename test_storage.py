"""Retention for everything the harness writes: bounded, visible, never fatal.

The property under test is the one that is invisible until it isn't: a
scheduled harness writes two session files per run and appends to a transcript
that is opened once and never closed. Left alone that grows until the disk
fills -- and a full disk on a deployment host takes the trading stack down with
it. Nothing in the code deletes a byte, so the only question worth asking is
whether the bounds actually bind.

Each test here states the failure it prevents in its docstring, because a
retention test that merely asserts "some files were removed" passes just as
happily against a rule that deletes the wrong ones.

Run:  python -m unittest test_storage
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import storage  # noqa: E402

#: Imported lazily, NOT at module scope. ``agent`` needs the OpenAI SDK, and this
#: file has to be runnable on a bare host python to prove the deployed tree
#: matches the workstation -- the host has no SDK installed (the sidecar image
#: does). An import at module scope made the WHOLE file fail to load there, so
#: the storage checks that need nothing but the stdlib were unreachable exactly
#: where they most need to be verifiable: on the machine that runs them.
agent = None
AGENT_IMPORT_ERROR = None
try:
    import agent  # noqa: E402
except Exception as exc:  # noqa: BLE001 -- report, do not fail collection
    AGENT_IMPORT_ERROR = exc


def _make_run(root: Path, stem: str, jsonl_bytes: int = 0,
              md_bytes: int = 0, mtime: float = 0.0) -> None:
    """Write one run's pair. Half a pair would defeat the point of the test."""
    j = root / f"{stem}.jsonl"
    m = root / f"{stem}.md"
    j.write_text("x" * jsonl_bytes, encoding="utf-8")
    m.write_text("y" * md_bytes, encoding="utf-8")
    if mtime:
        os.utime(j, (mtime, mtime))
        os.utime(m, (mtime, mtime))


class SessionRetentionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        #: Oldest first by mtime, so "keep the newest N" is unambiguous. The
        #: stamps read like dates, so the mtimes have to agree with them or the
        #: test asserts whichever run it happened to name last.
        self.stems = [f"2026-11-0{i}T00-00-0{i}_run{i}" for i in range(1, 7)]
        for age, stem in enumerate(self.stems):
            _make_run(self.root, stem, 100, 50, mtime=1000.0 + age * 10)

    def tearDown(self):
        self._tmp.cleanup()

    def _remaining(self):
        return sorted(p.name for p in self.root.iterdir())

    def test_old_runs_go_away_pair_by_pair(self):
        """A surviving half-pair is the worst outcome of a naive prune.

        A ``.md`` with no event stream is a log nothing can be cross-checked
        against; a ``.jsonl`` with no ``.md`` renders nowhere. Deleting by
        filename pattern rather than by stem leaves exactly that.
        """
        out = storage.prune_sessions(self.root, keep=2, max_bytes=0)
        self.assertEqual(out["kept"], 2)
        self.assertEqual(out["removed"], 8, out)      # 4 runs x 2 files
        self.assertEqual(out["freed_bytes"], 4 * 150)
        for stem in self.stems[:4]:
            self.assertNotIn(f"{stem}.jsonl", self._remaining())
            self.assertNotIn(f"{stem}.md", self._remaining())
        for stem in self.stems[-2:]:
            self.assertIn(f"{stem}.jsonl", self._remaining())
            self.assertIn(f"{stem}.md", self._remaining())

    def test_the_newest_run_is_never_deleted(self):
        """That is the run in progress -- or the one the operator is reading.

        A byte budget of 1 would otherwise reduce the directory to nothing,
        including the log of the task currently executing. The floor is not a
        convenience: retention that can delete the evidence of what just
        happened is worse than no retention.
        """
        out = storage.prune_sessions(self.root, keep=0, max_bytes=1)
        self.assertEqual(out["removed"], len(self.stems) * 2 - 2)
        newest = self.stems[-1]
        self.assertIn(f"{newest}.jsonl", self._remaining())
        self.assertIn(f"{newest}.md", self._remaining())

    def test_a_byte_budget_cuts_the_oldest_first(self):
        """The count and the byte budget answer different questions.

        A box that runs a mission every half hour needs the COUNT; one run that
        dumped enormous tool output needs the BUDGET. Applying both, count
        first, is what makes neither case escape.
        """
        out = storage.prune_sessions(self.root, keep=0, max_bytes=300)
        self.assertEqual(out["kept"], 2)               # 2 x 150 == 300
        remaining = self._remaining()
        for stem in self.stems[:4]:
            self.assertNotIn(stem, " ".join(remaining))
        self.assertIn(f"{self.stems[-1]}.jsonl", remaining)

    def test_latest_md_survives_because_it_is_the_handle_on_the_last_run(self):
        _make_run(self.root, "irrelevant", 10, 10)
        (self.root / "latest.md").write_text("copy", encoding="utf-8")
        storage.prune_sessions(self.root, keep=1, max_bytes=1)
        self.assertIn("latest.md", self._remaining())

    def test_a_partial_write_is_not_mistaken_for_a_completed_run(self):
        """.md.tmp is how the writer lands a file; it is not a session."""
        (self.root / "half.jsonl").write_text("x", encoding="utf-8")
        (self.root / "half.md.tmp").write_text("y", encoding="utf-8")
        before = self._remaining()
        storage.prune_sessions(self.root, keep=2, max_bytes=0)
        after = self._remaining()
        self.assertIn("half.md.tmp", after)
        self.assertIn("half.md.tmp", before)

    def test_explicit_unlimited_means_unlimited(self):
        """0 is a deliberate choice, not a typo that silently keeps everything."""
        out = storage.prune_sessions(self.root, keep=0, max_bytes=0)
        self.assertEqual(out["removed"], 0)
        self.assertEqual(out["kept"], len(self.stems) + 0)
        self.assertEqual(len(self._remaining()), len(self.stems) * 2)

    def test_a_delete_failure_is_reported_not_raised_and_not_swallowed(self):
        """Housekeeping must never kill a run -- and must never hide itself.

        Raising here would abort a mission for tidying's sake; swallowing it is
        the broad-except class that let a phantom method kill a live warning for
        three days. The failure has to reach the caller and a warning sink.
        """
        warnings = []
        real_unlink = Path.unlink

        def flaky(self, *a, **k):
            if self.suffix == ".md":
                raise OSError(28, "No space left on device")
            return real_unlink(self, *a, **k)

        with mock.patch.object(Path, "unlink", flaky):
            out = storage.prune_sessions(self.root, keep=1, max_bytes=0,
                                         on_warn=warnings.append)
        self.assertEqual(len(out["errors"]), 5)
        self.assertEqual(len(warnings), 5)
        self.assertIn("No space left", warnings[0])

    def test_a_missing_directory_is_not_an_error(self):
        out = storage.prune_sessions(self.root / "nope", keep=1, max_bytes=0)
        self.assertEqual(out["removed"], 0)
        self.assertEqual(out["errors"], [])

    def test_stems_come_back_as_names_newest_first(self):
        """Pinned as a SHAPE, because the shape is the bug that was here.

        Returning ``(mtime, stem)`` tuples while the caller unpacked
        ``(stem, mtime)`` made the pruner delete files named after a float. It
        reported success, kept ``kept`` accurate, and removed nothing -- so
        nothing downstream could see it, and a retention rule that retains is
        indistinguishable from no retention rule at all.
        """
        stems = storage.session_stems(self.root)
        self.assertEqual(len(stems), 6)
        self.assertTrue(all(isinstance(s, str) for s in stems), stems)
        self.assertEqual(stems[0], self.stems[-1], "newest must come first")
        self.assertEqual(stems[-1], self.stems[0])


class RetentionWindowTest(unittest.TestCase):
    """The count bound must not silently define the RETENTION WINDOW.

    Live 2026-11-02: the operator asked "is history not persisted?" after
    /agent/runs showed nothing older than about a day. Nothing was broken --
    they were reading the count bound. ``SESSIONS_KEEP`` was 50, the scheduler
    runs a mission every thirty minutes, so 50 runs is ~25 hours, and every pass
    deleted the run that aged out. The same directory held 8.1 MB against a
    200 MB allowance, i.e. 96% of the budget unused while history was destroyed.

    So these tests pin the RELATIONSHIP between the two knobs, not the numbers:
    the byte budget is the policy and the count is a backstop high enough that it
    cannot become the policy by accident.
    """

    def test_the_default_count_cannot_set_the_window_by_itself(self):
        """At the live cadence, the DEFAULT count must outlast the byte budget.

        Derived rather than asserted as a magic number: at the ~160 KB/run seen
        on the deployment, the 200 MB budget holds ~1280 runs. If the count
        default is below that, the count is deciding retention and a run that is
        "within budget" is still being deleted.
        """
        live_bytes_per_run = 160 * 1024          # measured: 8.1 MB / 50 runs
        runs_the_budget_holds = (config.SESSIONS_MAX_BYTES
                                 // live_bytes_per_run)
        self.assertGreaterEqual(
            config.SESSIONS_KEEP, runs_the_budget_holds,
            f"the count bound ({config.SESSIONS_KEEP}) is smaller than what the "
            f"byte budget holds (~{runs_the_budget_holds} runs), so the count is "
            f"silently the retention policy -- which is how a day-old history "
            f"came to be deleted from a directory using 4% of its allowance")

    def test_the_default_window_covers_more_than_a_week(self):
        """A monitoring pass every 30 min is 48 runs/day; a week is 336."""
        per_day = 48
        self.assertGreaterEqual(config.SESSIONS_KEEP / per_day, 7,
                                "the default retains less than a week of runs")

    def test_the_byte_budget_still_bounds_a_runaway_directory(self):
        """Raising the count must not make retention unbounded."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                _make_run(root, f"2026-11-0{i % 9 + 1}T00-00-0{i}_run{i}",
                          100_000, 100_000, mtime=1000.0 + i)
            out = storage.prune_sessions(root, keep=1000, max_bytes=500_000)
            self.assertGreater(out["kept"], 0, "it deleted everything")
            remaining = sum(
                (root / f"{s}{sfx}").stat().st_size
                for s in storage.session_stems(root)
                for sfx in (".jsonl", ".md")
                if (root / f"{s}{sfx}").exists())
            self.assertLessEqual(
                remaining, 500_000 + 200_000,
                "the byte budget stopped being enforced once the count was "
                "raised")

    def test_the_newest_run_survives_any_bound(self):
        """The rule that must hold at every setting: never delete the live run."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(5):
                _make_run(root, f"2026-11-0{i + 1}T00-00-00_run{i}", 10, 10,
                          mtime=1000.0 + i)
            newest = storage.session_stems(root)[0]
            storage.prune_sessions(root, keep=1, max_bytes=1)
            self.assertTrue((root / f"{newest}.jsonl").exists(),
                            "retention deleted the newest run -- the one that "
                            "may still be in progress")


class RetentionReportingTest(unittest.TestCase):
    """The runs page must SAY how far back it reaches, not just list rows."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "sessions").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _facts(self):
        import webui            # imported lazily: it pulls in prompts/config
        with mock.patch.object(config, "SESSIONS_DIR",
                               str(self.root / "sessions")):
            return webui.retention_facts()

    def test_an_empty_directory_reports_zero_not_an_error(self):
        facts = self._facts()
        self.assertEqual(facts["runs"], 0)
        self.assertIsNone(facts["oldest"])

    def test_it_reports_the_oldest_and_newest_run_it_can_still_see(self):
        for i in range(3):
            _make_run(self.root / "sessions",
                      f"2026-11-0{i + 1}T0{i}-00-00_run{i}", 10, 10,
                      mtime=1000.0 + i)
        facts = self._facts()
        self.assertEqual(facts["runs"], 3)
        self.assertEqual(facts["oldest"], "2026-11-01T00:00:00")
        self.assertEqual(facts["newest"], "2026-11-03T02:00:00")

    def test_a_timestamp_with_dashes_in_the_clock_is_still_a_time(self):
        """``2026:11:02T04:31:41`` is what a naive replace produces."""
        import webui
        stamp = webui._stamp_of(Path("x/2026-11-02T04-31-41_ab12cd34.jsonl"))
        self.assertEqual(stamp, "2026-11-02T04:31:41")
        self.assertNotIn("2026:11", stamp)

    def test_it_names_the_bound_so_nobody_has_to_read_the_pruner(self):
        _make_run(self.root / "sessions", "2026-11-02T00-00-00_run", 10, 10)
        facts = self._facts()
        self.assertIn("keep", facts)
        self.assertIn("max_bytes_mb", facts)
        self.assertGreater(facts["keep"], 0)
        self.assertGreater(facts["max_bytes_mb"], 0)


class TranscriptRotationTest(unittest.TestCase):
    """Rotating the raw API transcript, independent of session retention."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "transcript.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_under_budget_is_left_alone(self):
        self.path.write_text("a" * 100, encoding="utf-8")
        self.assertIsNone(storage.rotate_transcript(self.path, 1000, 2))
        self.assertEqual(self.path.read_text(encoding="utf-8"), "a" * 100)

    def test_over_budget_shifts_generations_and_drops_the_overflow(self):
        """``.1`` newest, ``.N`` oldest, and ``.N`` cannot accumulate forever."""
        self.path.write_text("gen-live", encoding="utf-8")
        (self.path.parent / "transcript.jsonl.1").write_text(
            "gen-1", encoding="utf-8")
        (self.path.parent / "transcript.jsonl.2").write_text(
            "gen-2", encoding="utf-8")
        moved = storage.rotate_transcript(self.path, max_bytes=1, backups=2)
        self.assertEqual(moved, self.path.parent / "transcript.jsonl.1")
        #: The live path is gone, so the next append starts a fresh file.
        self.assertFalse(self.path.exists())
        read = lambda n: (self.path.parent / n).read_text(encoding="utf-8")  # noqa: E731
        self.assertEqual(read("transcript.jsonl.1"), "gen-live")
        self.assertEqual(read("transcript.jsonl.2"), "gen-1")
        self.assertFalse((self.path.parent / "transcript.jsonl.3").exists(),
                         "the oldest generation must be dropped, not stacked")

    def test_backups_is_floored_at_one(self):
        """Rotation that discards the only copy is a deletion, not a rotation."""
        self.path.write_text("precious" * 10, encoding="utf-8")
        storage.rotate_transcript(self.path, max_bytes=1, backups=0)
        kept = self.path.parent / "transcript.jsonl.1"
        self.assertTrue(kept.exists())
        self.assertEqual(kept.read_text(encoding="utf-8"), "precious" * 10)

    def test_zero_budget_disables_rotation_entirely(self):
        self.path.write_text("z" * 10_000, encoding="utf-8")
        self.assertIsNone(storage.rotate_transcript(self.path, 0, 2))
        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.stat().st_size, 10_000)

    def test_a_missing_or_unwritable_transcript_does_not_raise(self):
        self.assertIsNone(storage.rotate_transcript(self.path, 10, 2),
                          "no file yet is the normal first-run case")
        with mock.patch.object(Path, "replace", side_effect=OSError(13, "no")):
            self.path.write_text("q" * 500, encoding="utf-8")
            warnings = []
            self.assertIsNone(
                storage.rotate_transcript(self.path, 1, 2,
                                          on_warn=warnings.append))
        self.assertEqual(len(warnings), 1)
        self.assertIn("could not rotate", warnings[0])

    def test_the_agent_rotates_before_it_opens_the_transcript(self):
        """Ordering, pinned by source position rather than by a stub.

        Renaming a file whose append handle is already open keeps writing into
        the renamed inode: the live file grows on and the rotation looks
        successful in the listing. A functional test cannot see this without a
        real filesystem, so the order itself is the assertion.
        """
        if agent is None:
            self.skipTest(f"the OpenAI SDK is not importable here: "
                          f"{AGENT_IMPORT_ERROR}")
        import inspect
        src = inspect.getsource(agent.Agent.__init__)
        rot = src.index("storage.rotate_transcript")
        opened = src.index('open(config.TRANSCRIPT')
        self.assertLess(rot, opened,
                        "the transcript is opened before it can be rotated")
        #: ...and the rotation is told where warnings go.
        self.assertIn("on_warn=self._warn", src)


class StorageReportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "memories" / "repo").mkdir(parents=True)
        (self.root / "sessions").mkdir()
        (self.root / "memories" / "repo" / "p.md").write_text(
            "n" * 4000, encoding="utf-8")
        _make_run(self.root / "sessions", "s1", 500, 200)
        (self.root / "transcript.jsonl").write_text("t" * 3000,
                                                    encoding="utf-8")
        #: A file type this report has never seen: it must still be measured.
        (self.root / "schedule_state.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_top_level_entry_is_measured(self):
        """An unlisted entry is a blind spot exactly where growth hides."""
        out = storage.report(self.root)
        self.assertEqual(out["sizes"]["memories"], 4000)
        self.assertEqual(out["sizes"]["sessions"], 700)
        self.assertEqual(out["sizes"]["transcript.jsonl"], 3000)
        self.assertEqual(out["sizes"]["schedule_state.json"], 2)
        self.assertEqual(out["total_bytes"], 4000 + 700 + 3000 + 2)
        self.assertIn("agent-data", storage.one_line(out))

    def test_memory_over_budget_warns_and_changes_nothing(self):
        """Knowledge is reported, never trimmed by a cleanup routine.

        Auto-trimming notes would defeat the entire purpose of memory, and it
        would do so silently -- the file would keep looking correct.
        """
        notes = self.root / "memories" / "repo" / "p.md"
        before = notes.read_bytes()
        out = storage.report(self.root, memory_soft_max_bytes=1000)
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("soft budget", out["warnings"][0])
        self.assertIn("nothing is trimmed automatically", out["warnings"][0])
        self.assertEqual(notes.read_bytes(), before)

    def test_memory_under_budget_says_nothing(self):
        out = storage.report(self.root, memory_soft_max_bytes=1 << 20)
        self.assertEqual(out["warnings"], [])

    def test_an_unreadable_root_reports_zeroes_rather_than_raising(self):
        out = storage.report(self.root / "missing")
        self.assertEqual(out["total_bytes"], 0)
        self.assertEqual(out["warnings"], [])


class AgentWiringTest(unittest.TestCase):
    """The bounds must actually run, on every surface that starts a run."""

    def setUp(self):
        if agent is None:
            self.skipTest(f"the OpenAI SDK is not importable here: "
                          f"{AGENT_IMPORT_ERROR}")

    def _bare_agent(self, events):
        #: Constructed without __init__ so no route resolution or API client is
        #: needed -- the methods under test read config globals, not credentials.
        a = agent.Agent.__new__(agent.Agent)
        a._transcript = None
        a.on_event = events.append
        a.session = None
        return a

    def test_a_run_emits_a_storage_event_with_the_retention_result(self):
        events = []
        a = self._bare_agent(events)
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp) / "sessions"
            sessions.mkdir()
            root = Path(tmp)
            _make_run(sessions, "old", 10, 10, mtime=1.0)
            _make_run(sessions, "new", 10, 10, mtime=9.0)
            with mock.patch.object(config, "SESSIONS_DIR", sessions), \
                 mock.patch.object(config, "SESSIONS_KEEP", 1), \
                 mock.patch.object(config, "SESSIONS_MAX_BYTES", 0), \
                 mock.patch.object(config, "MEMORY_ROOT", root / "memories"), \
                 mock.patch.object(config, "MEMORY_SOFT_MAX_BYTES", 0):
                a._bound_storage()
        storage_events = [e for e in events if e["event"] == "storage"]
        self.assertEqual(len(storage_events), 1, events)
        self.assertEqual(storage_events[0]["sessions_removed"], 2)
        self.assertEqual(storage_events[0]["sessions_kept"], 1)
        self.assertIn("agent-data", storage_events[0]["summary"])

    def test_a_retention_failure_is_reported_and_the_run_continues(self):
        events = []
        a = self._bare_agent(events)
        with mock.patch.object(storage, "prune_sessions",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(storage, "report", side_effect=RuntimeError("x")):
            a._bound_storage()          # must not raise
        kinds = [e["event"] for e in events]
        self.assertIn("storage_warning", kinds)
        self.assertIn("storage", kinds)
        self.assertIn("boom", " ".join(e.get("detail", "") for e in events))

    def test_a_warning_reaches_the_session_log_when_one_exists(self):
        """On a scheduled mission there is no terminal to write to.

        The transcript gets it first and the session log second, so the warning
        is in the readable log an operator actually opens. And a failure of the
        SECOND sink must not abort the run either -- `_warn` keeps going, which
        is why the recorder here is a plain function: `side_effect=list.append`
        raises on the kwargs and would exercise the swallow path instead of the
        property being tested.
        """
        events = []
        a = self._bare_agent(events)
        seen = []

        def recorder(*args, **kwargs):
            seen.append((args, kwargs))

        a.session = mock.MagicMock()
        a.session.event.side_effect = recorder
        a._warn("disk is 97% full")
        self.assertEqual(len(seen), 1, seen)
        self.assertEqual(seen[0][0], ("storage_warning",))
        self.assertEqual(seen[0][1]["detail"], "disk is 97% full")
        self.assertEqual(events[0]["event"], "storage_warning")

    def test_a_broken_session_sink_does_not_take_the_run_down_with_it(self):
        events = []
        a = self._bare_agent(events)
        a.session = mock.MagicMock()
        a.session.event.side_effect = OSError(28, "No space left on device")
        a._warn("still reported")          # must not raise
        self.assertEqual([e["event"] for e in events], ["storage_warning"])

    def test_a_run_actually_invokes_retention(self):
        """The call site, not just the helper.

        Deleting ``self._bound_storage()`` from the run body left every test
        above green -- they all drive the helper directly -- which means the
        harness would have shipped with retention that never ran, and the only
        symptom would have been a disk that filled months later. A behavioural
        test of a helper cannot see a missing caller; this asserts the caller.

        The body moved to ``_run_impl`` when ``run()`` gained a guard that
        closes the session on an unexpected exception (a crash used to look
        identical to the process being killed). So the guard must be checked
        too: inspecting only one of the two methods would let a refactor delete
        the delegation and silently stop every run.
        """
        import inspect
        wrapper = inspect.getsource(agent.Agent.run)
        body = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("_run_impl(task, images)", wrapper,
                      "run() no longer delegates -- _run_impl would never run")
        self.assertIn("_bound_storage()", body,
                      "a run never bounds its own storage")

    def test_retention_runs_after_the_session_log_exists(self):
        """Ordering: the newest pair is this run's, so it must not be a candidate.

        Pruning BEFORE the log is created is also survivable, but then the
        retention warning has nowhere to be recorded, and "the run that just
        deleted the evidence of its own failure" is not a shape worth allowing.
        """
        import inspect
        src = inspect.getsource(agent.Agent._run_impl)
        self.assertLess(src.index("SessionLog("), src.index("_bound_storage()"))


class ConfigKnobTest(unittest.TestCase):
    def test_the_defaults_bound_rather_than_disable(self):
        """A default of 0 would mean 'keep everything forever' -- the bug."""
        self.assertGreater(config.SESSIONS_KEEP, 0)
        self.assertGreater(config.SESSIONS_MAX_BYTES, 0)
        self.assertGreater(config.TRANSCRIPT_MAX_BYTES, 0)
        self.assertGreaterEqual(config.TRANSCRIPT_BACKUPS, 1)

    def test_a_mistyped_budget_is_refused_not_silently_unlimited(self):
        with mock.patch.dict(os.environ, {"AGENT_SESSIONS_MAX_MB": "2O0"}):
            with self.assertRaises(ValueError) as ctx:
                config._non_negative_int("AGENT_SESSIONS_MAX_MB", 200)
        self.assertIn("AGENT_SESSIONS_MAX_MB", str(ctx.exception))

    def test_zero_means_unlimited_explicitly(self):
        with mock.patch.dict(os.environ, {"AGENT_TRANSCRIPT_MAX_MB": "0"}):
            self.assertEqual(
                config._non_negative_int("AGENT_TRANSCRIPT_MAX_MB", 8) << 20, 0)

    def test_the_documented_knobs_are_read_from_the_environment(self):
        import importlib
        #: The restore reload MUST happen after the environment patch is gone.
        #:
        #: It used to sit in a finally INSIDE the ``mock.patch.dict`` block, so
        #: it reloaded config while the patched variables were still set and
        #: "restored" it to AGENT_SESSIONS_KEEP=7 -- permanently, for the rest of
        #: the run. Nothing caught it because no other test asserted a value
        #: above 7; the retention-window tests added on 2026-11-02 did, and
        #: failed only in a FULL suite run (each passed alone). A cleanup that
        #: runs inside the thing it is cleaning up is not a cleanup.
        try:
            with mock.patch.dict(os.environ, {
                    "AGENT_SESSIONS_KEEP": "7",
                    "AGENT_SESSIONS_MAX_MB": "3",
                    "AGENT_TRANSCRIPT_MAX_MB": "1",
                    "AGENT_TRANSCRIPT_BACKUPS": "4",
                    "AGENT_MEMORY_SOFT_MAX_MB": "2"}):
                fresh = importlib.reload(config)
                self.assertEqual(fresh.SESSIONS_KEEP, 7)
                self.assertEqual(fresh.SESSIONS_MAX_BYTES, 3 << 20)
                self.assertEqual(fresh.TRANSCRIPT_MAX_BYTES, 1 << 20)
                self.assertEqual(fresh.TRANSCRIPT_BACKUPS, 4)
                self.assertEqual(fresh.MEMORY_SOFT_MAX_BYTES, 2 << 20)
        finally:
            #: Now the environment is the real one again, so this reload puts
            #: the DEFAULTS back rather than the patch.
            importlib.reload(config)
            importlib.reload(storage)
            #: The other modules captured config's attributes at import, so
            #: reload them too. Only reload what actually imported: on a host
            #: without the SDK `agent` is None, and reloading None raises
            #: TypeError -- which is how this test, written to prove the knobs
            #: work, became the only failure in the suite on the one machine it
            #: must also run on.
            if agent is not None:
                importlib.reload(agent)
        self.assertGreater(
            config.SESSIONS_KEEP, 7,
            "the suite left config patched for every later test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
