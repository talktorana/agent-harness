"""Regressions for scheduler.py: mission cadence when the declaration is
read-only, and the failure modes around persisting runtime state.

The topology these tests exist for is the one a gated deployment actually runs:
the mission schedule is a reviewed file mounted READ-ONLY inside a sidecar (and
on the applier's forbidden list, so the agent cannot rewrite its own brief),
while ``last_run`` must still be remembered between polls. Writing state into
the declaration -- which is what this module used to do -- fails on that mount,
and the failure was caught by the poll loop's generic handler and printed as
``[scheduler] loop error``, which reads as transient. No mission ran, and the
reason never surfaced.

Two properties are pinned here that are easy to lose and expensive to miss:

* a read-only declaration schedules normally, with state in its own file; and
* an unwritable state file is FATAL and loud, not ignored -- because
  ``last_run`` is what stops the next poll from starting the same long mission
  again. A scheduler with no memory is a token-burning retry loop.

Run:  python -m unittest test_scheduler
"""
from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import scheduler  # noqa: E402


def _write(path: Path, jobs) -> None:
    path.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")


def _job(ident="monitor", every_min=30, message="run a pass", **extra):
    job = {"id": ident, "every_min": every_min, "enabled": True,
           "message": message}
    job.update(extra)
    return job


class SchedulerPaths(unittest.TestCase):
    """Points the scheduler at a scratch directory for one test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.declaration = self.dir / "schedule.json"
        self.state = self.dir / "state" / "schedule_state.json"
        self._saved = (config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE)
        config.SCHEDULE_FILE = self.declaration
        config.SCHEDULE_STATE_FILE = self.state
        #: every mission a test "runs", recorded verbatim
        self.ran: list = []

    def tearDown(self) -> None:
        config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE = self._saved
        # Undo any read-only bit before TemporaryDirectory cleanup, which on
        # Windows refuses to delete a file marked read-only.
        for p in self.dir.rglob("*"):
            try:
                p.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        self._tmp.cleanup()

    def stub_task(self, *, raises: Exception = None, spy=None):
        """Replace agent.run_task, optionally inspecting state mid-run."""
        outer = self

        def fake_run_task(message, verbose=True, **kwargs):
            outer.ran.append(message)
            if spy is not None:
                spy(message)
            if raises is not None:
                raise raises
            return f"answered: {message}"

        patcher = mock.patch("agent.run_task", side_effect=fake_run_task)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestDeclarationIsNeverWritten(SchedulerPaths):
    def test_a_run_leaves_the_declaration_byte_identical(self):
        """The declaration is configuration; the scheduler only reads it."""
        _write(self.declaration, [_job()])
        before = self.declaration.read_bytes()
        mtime_before = self.declaration.stat().st_mtime_ns
        self.stub_task()

        scheduler.run_due_jobs()

        self.assertEqual(self.ran, ["run a pass"])
        self.assertEqual(self.declaration.read_bytes(), before,
                         "the scheduler rewrote its own mission file")
        self.assertEqual(self.declaration.stat().st_mtime_ns, mtime_before,
                         "rewriting the declaration is how the reviewed brief "
                         "drifts from the brief that actually runs")

    def test_a_read_only_declaration_still_schedules(self):
        """The production mount: declaration read-only, state elsewhere.

        Skipped only on a platform that ignores the read-only bit -- and the
        skip says so loudly, because a test that skips quietly proves nothing.
        """
        _write(self.declaration, [_job()])
        self.declaration.chmod(0o444)
        try:
            self.declaration.write_text("{}")
        except OSError:
            readonly = True
        else:
            readonly = False
            self.declaration.chmod(0o444)
        if not readonly:
            self.skipTest("this platform ignores the read-only bit; the "
                          "content-unchanged assertion above covers the "
                          "property instead")
        self.stub_task()

        scheduler.run_due_jobs()

        self.assertEqual(self.ran, ["run a pass"])
        self.assertTrue(self.state.is_file(),
                        "state was not written, so cadence cannot survive a "
                        "restart")

    def test_state_lives_beside_the_session_log_by_default(self):
        """Default must land somewhere an agent can write, not in a repo."""
        with mock.patch.dict(os.environ, {
                "AGENT_SESSIONS_DIR": str(self.dir / "sessions"),
                "AGENT_SCHEDULE_FILE": str(self.dir / "sched.json")},
                clear=False):
            os.environ.pop("AGENT_SCHEDULE_STATE_FILE", None)
            for mod in list(sys.modules):
                if mod == "config":
                    del sys.modules[mod]
            import importlib
            fresh = importlib.import_module("config")
            self.addCleanup(lambda: sys.modules.update({"config": config}))
        default = Path(fresh.SCHEDULE_STATE_FILE)
        self.assertEqual(default.parent, self.dir / "sessions",
                         str(default))
        self.assertNotEqual(default, Path(fresh.SCHEDULE_FILE),
                            "state and declaration must not share a file")


class TestUnwritableStateIsFatal(SchedulerPaths):
    def test_nothing_runs_when_state_cannot_be_persisted(self):
        """Without a stamp, the next poll starts the same mission again.

        The state path's parent is a REGULAR FILE, so mkdir fails on every
        platform -- the portable way to simulate an unwritable location.
        """
        _write(self.declaration, [_job()])
        blocker = self.dir / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        config.SCHEDULE_STATE_FILE = blocker / "state.json"
        calls: list = []
        self.stub_task(spy=lambda m: calls.append(m))

        with self.assertRaises(scheduler.SchedulerStateError) as ctx:
            scheduler.run_due_jobs()

        self.assertEqual(self.ran, [], "a mission ran without a stamp: the "
                                       "next poll would run it again")
        self.assertIn("AGENT_SCHEDULE_STATE_FILE", str(ctx.exception),
                      "the error must name the fix, not the errno")

    def test_state_writable_reports_the_truth(self):
        blocker = self.dir / "blocker2"
        blocker.write_text("x", encoding="utf-8")
        config.SCHEDULE_STATE_FILE = blocker / "state.json"
        self.assertFalse(scheduler.state_writable())

        config.SCHEDULE_STATE_FILE = self.dir / "ok" / "state.json"
        self.assertTrue(scheduler.state_writable())

    def test_list_shows_the_fault_before_anyone_has_to_read_logs(self):
        """--list is the diagnostic; it must say the state is unwritable."""
        blocker = self.dir / "blocker3"
        blocker.write_text("x", encoding="utf-8")
        config.SCHEDULE_STATE_FILE = blocker / "state.json"
        _write(self.declaration, [_job()])

        described = scheduler.describe()

        self.assertFalse(described["state_writable"])
        self.assertEqual(str(described["state_file"]),
                         str(blocker / "state.json"))
        self.assertEqual(len(described["jobs"]), 1)

    def test_once_exits_non_zero_when_it_refuses(self):
        """A skipped pass must not look like a successful one to cron."""
        blocker = self.dir / "blocker4"
        blocker.write_text("x", encoding="utf-8")
        config.SCHEDULE_STATE_FILE = blocker / "state.json"
        _write(self.declaration, [_job()])
        self.stub_task()

        with mock.patch.object(sys, "argv", ["scheduler.py", "--once"]):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = scheduler.main()

        self.assertEqual(code, 2)
        self.assertIn("REFUSING TO RUN", buf.getvalue())
        self.assertEqual(self.ran, [])


class TestCadenceFromPersistedState(SchedulerPaths):
    def test_a_job_is_not_due_again_before_its_interval(self):
        _write(self.declaration, [_job(every_min=30)])
        self.stub_task()

        scheduler.run_due_jobs()
        self.assertEqual(len(self.ran), 1)

        # A fresh process (the sidecar restarts) must not re-run it.
        self.assertEqual(scheduler.due_jobs(datetime.now() + timedelta(minutes=5)),
                         [], "due again 5 minutes into a 30-minute interval")
        self.assertEqual(len(scheduler.due_jobs(
            datetime.now() + timedelta(minutes=31))), 1,
            "the interval elapsed but the job is still not due")
        scheduler.run_due_jobs()
        self.assertEqual(len(self.ran), 1, "run_due_jobs ignored the stamp")

    def test_the_stamp_is_written_before_the_mission_starts(self):
        """A mission can outlast the poll interval; the stamp is the lock."""
        _write(self.declaration, [_job(every_min=30)])
        seen: dict = {}

        def spy(_message):
            seen["state_at_call_time"] = json.loads(
                self.state.read_text(encoding="utf-8"))

        self.stub_task(spy=spy)
        scheduler.run_due_jobs()

        entry = seen["state_at_call_time"]["jobs"]["monitor"]
        self.assertIn("last_run", entry,
                      "the mission started before its stamp was persisted, so "
                      "an overlapping poll would start a second copy")

    def test_a_failed_job_records_the_error_and_consumes_its_slot(self):
        """A broken mission must not become a 30-second retry loop."""
        _write(self.declaration, [_job(every_min=30)])
        self.stub_task(raises=RuntimeError("provider 429"))

        records = scheduler.run_due_jobs()

        self.assertFalse(records[0]["ok"])
        self.assertIn("RuntimeError", records[0]["error"])
        state = json.loads(self.state.read_text(encoding="utf-8"))
        entry = state["jobs"]["monitor"]
        self.assertFalse(entry["last_ok"])
        self.assertIn("provider 429", entry["last_error"])
        self.assertEqual(scheduler.due_jobs(), [],
                         "a failure must wait for the next interval, not the "
                         "next poll")

    def test_a_disabled_job_never_runs(self):
        _write(self.declaration, [_job(enabled=False)])
        self.stub_task()

        scheduler.run_due_jobs()

        self.assertEqual(self.ran, [])
        self.assertFalse(self.state.is_file(),
                         "an inert schedule must not create state")


class TestDailyJobs(SchedulerPaths):
    def test_a_daily_job_fires_once_per_calendar_day(self):
        _write(self.declaration, [{"id": "nightly", "at": "02:30",
                                   "enabled": True, "message": "summarise"}])
        self.stub_task()
        today = datetime.now().replace(hour=3, minute=0, second=0,
                                       microsecond=0)

        self.assertEqual(len(scheduler.due_jobs(today)), 1)
        with mock.patch.object(scheduler, "datetime",
                               wraps=datetime) as fake_dt:
            fake_dt.now.return_value = today
            fake_dt.fromisoformat = datetime.fromisoformat
            scheduler.run_due_jobs()
        self.assertEqual(len(self.ran), 1)

        later_same_day = today.replace(hour=23, minute=0)
        with mock.patch.object(scheduler, "datetime", wraps=datetime) as fake_dt:
            fake_dt.now.return_value = later_same_day
            fake_dt.fromisoformat = datetime.fromisoformat
            self.assertEqual(scheduler.due_jobs(), [],
                             "a daily job fired twice in one day")

    def test_a_daily_job_waits_for_its_time(self):
        _write(self.declaration, [{"id": "nightly", "at": "02:30",
                                   "enabled": True, "message": "summarise"}])
        early = datetime.now().replace(hour=1, minute=0, second=0,
                                       microsecond=0)
        self.assertEqual(scheduler.due_jobs(early), [])


class TestDegradedInputs(SchedulerPaths):
    def test_a_missing_declaration_means_no_jobs(self):
        self.stub_task()
        self.assertEqual(scheduler.run_due_jobs(), [])

    def test_a_corrupt_declaration_means_no_jobs_not_a_crash(self):
        self.declaration.write_text("{not json", encoding="utf-8")
        self.stub_task()
        self.assertEqual(scheduler.run_due_jobs(), [])

    def test_corrupt_state_is_replaced_not_fatal(self):
        """Unreadable state means 'nothing has run'; the pass rewrites it."""
        _write(self.declaration, [_job()])
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text("{corrupt", encoding="utf-8")
        self.stub_task()

        scheduler.run_due_jobs()

        self.assertEqual(len(self.ran), 1)
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))
                         ["jobs"]["monitor"].get("last_ok"), True)

    def test_an_anonymous_job_is_still_scheduled_under_a_visible_key(self):
        _write(self.declaration, [{"every_min": 1, "enabled": True,
                                   "message": "no id here"}])
        self.stub_task()

        scheduler.run_due_jobs()

        self.assertEqual(self.ran, ["no id here"])
        self.assertIn("#0", json.loads(
            self.state.read_text(encoding="utf-8"))["jobs"],
            "an id-less job must not silently share state with the next one")


class TestRepeatLogging(unittest.TestCase):
    """One unchanged failure must not become 2,880 identical log lines."""

    def test_first_report_is_immediate_and_repeats_are_suppressed(self):
        last: list = []
        first = scheduler._rate_limited("[scheduler] REFUSING TO RUN", last)
        self.assertIsNotNone(first)
        self.assertIsNone(scheduler._rate_limited("[scheduler] REFUSING TO RUN",
                                                  last),
                          "the same condition printed again seconds later")

    def test_a_repeat_prints_again_after_the_window_and_says_it_persists(self):
        last: list = []
        scheduler._rate_limited("boom", last)
        last[1] -= scheduler.REPEAT_LOG_SEC + 1
        last[2] -= scheduler.REPEAT_LOG_SEC + 1
        again = scheduler._rate_limited("boom", last)
        self.assertIsNotNone(again)
        self.assertIn("unchanged for", again)

    def test_a_new_message_prints_at_once_and_names_the_cleared_one(self):
        last: list = []
        scheduler._rate_limited("first fault", last)
        changed = scheduler._rate_limited("second fault", last)
        self.assertIsNotNone(changed)
        self.assertIn("first fault", changed,
                      "a reader must know the previous condition cleared")

    def test_no_message_is_not_a_message(self):
        last: list = []
        self.assertIsNone(scheduler._rate_limited("", last))

    def test_the_daemon_warns_at_startup_when_state_is_unwritable(self):
        """The config fault must be visible before the first poll, not after."""
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            saved = (config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE)
            config.SCHEDULE_FILE = Path(tmp) / "schedule.json"
            config.SCHEDULE_STATE_FILE = blocker / "state.json"
            try:
                _write(config.SCHEDULE_FILE, [_job()])
                buf = io.StringIO()
                with redirect_stdout(buf), \
                        mock.patch.object(scheduler.time, "sleep",
                                          side_effect=KeyboardInterrupt):
                    with self.assertRaises((KeyboardInterrupt, SystemExit)):
                        scheduler.daemon(poll_sec=0)
                out = buf.getvalue()
                self.assertIn("not writable", out)
                self.assertIn("AGENT_SCHEDULE_STATE_FILE", out)
            finally:
                config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
