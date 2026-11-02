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
import types
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
        self.hold = self.dir / "scheduler.hold"
        self._saved = (config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE,
                       getattr(config, "SCHEDULE_HOLD", ""))
        config.SCHEDULE_FILE = self.declaration
        config.SCHEDULE_STATE_FILE = self.state
        #: Pointed at a scratch path so a hold left behind by a real run cannot
        #: make an unrelated test pass or fail for the wrong reason.
        config.SCHEDULE_HOLD = str(self.hold)
        #: every mission a test "runs", recorded verbatim
        self.ran: list = []

    def tearDown(self) -> None:
        (config.SCHEDULE_FILE, config.SCHEDULE_STATE_FILE,
         config.SCHEDULE_HOLD) = self._saved
        # Undo any read-only bit before TemporaryDirectory cleanup, which on
        # Windows refuses to delete a file marked read-only.
        for p in self.dir.rglob("*"):
            try:
                p.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        self._tmp.cleanup()

    def stub_task(self, *, raises: Exception = None, spy=None):
        """Replace agent.run_task, optionally inspecting state mid-run.

        The stub goes in through ``sys.modules`` rather than
        ``mock.patch("agent.run_task")``. The real ``agent`` module imports the
        provider SDK, so patching it by name fails on any machine that has the
        harness source but not its dependencies -- which is exactly the host
        that runs the applier, where these tests were then reporting 13 ERRORS
        that looked like scheduler defects and were a missing ``openai``.
        ``run_due_jobs`` imports lazily, so a pre-registered module satisfies it
        without touching the real one.
        """
        outer = self

        def fake_run_task(message, verbose=True, **kwargs):
            outer.ran.append(message)
            if spy is not None:
                spy(message)
            if raises is not None:
                raise raises
            return f"answered: {message}"

        fake = types.ModuleType("agent")
        fake.run_task = fake_run_task
        patcher = mock.patch.dict(sys.modules, {"agent": fake})
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


class TestOperatorHoldPausesTheLoop(SchedulerPaths):
    """A hold file stops missions, and its PRESENCE is the whole signal.

    The requirement behind this: an operator needs to stop the mission loop
    without editing the declaration (read-only, and on the applier's refuse list
    -- an agent that can rewrite its own brief owns its own guardrails) and
    without stopping the container (which also kills the console, so nobody can
    inspect what they paused). A file satisfies both, and placing it inside the
    read-only root means the agent can read it but cannot clear its own pause.
    """

    def test_no_hold_configured_means_never_held(self):
        """A plain checkout has nothing to point at, so nothing to forget."""
        _write(self.declaration, [_job()])
        config.SCHEDULE_HOLD = ""
        self.assertFalse(scheduler.hold()["held"])
        self.assertEqual(len(scheduler.due_jobs()), 1,
                         "no hold must not change scheduling")

    def test_a_hold_file_stops_every_job_without_touching_the_declaration(self):
        _write(self.declaration, [_job(), _job("second", every_min=1)])
        before = self.declaration.read_bytes()
        self.hold.write_text("{}", encoding="utf-8")
        self.stub_task()

        self.assertEqual(scheduler.due_jobs(), [],
                         "a paused loop must schedule nothing")
        self.assertEqual(scheduler.run_due_jobs(), [])
        self.assertEqual(self.ran, [], "a mission ran while paused")
        self.assertEqual(self.declaration.read_bytes(), before,
                         "pausing must be an OUTSIDE file, not an edit to the "
                         "reviewed declaration")

    def test_touch_pauses_and_rm_resumes(self):
        """The two things an operator actually reaches for must just work.

        An empty file pauses. That is not a convenience: a pause mechanism whose
        documented path is "run this script with these three flags" is one
        somebody will mis-type at 2am.
        """
        _write(self.declaration, [_job()])
        self.hold.write_text("", encoding="utf-8")
        self.assertTrue(scheduler.hold()["held"])
        self.hold.unlink()
        self.assertFalse(scheduler.hold()["held"])
        self.assertEqual(len(scheduler.due_jobs()), 1)

    def test_an_unreadable_hold_body_still_pauses(self):
        """Presence is the intent; a bad body must not resume the agent.

        Failing to PARSE cannot be the one thing that silently releases a
        shell-capable agent. The failure direction is chosen: a hold nobody can
        read keeps the loop stopped and says so, rather than starting missions
        the operator believes are paused.
        """
        _write(self.declaration, [_job()])
        self.hold.write_text("{this is not json", encoding="utf-8")
        state = scheduler.hold()
        self.assertTrue(state["held"])
        self.assertIn("paused", state["reason"])
        self.assertEqual(scheduler.due_jobs(), [])

    def test_the_pause_records_who_and_when_for_later_explanation(self):
        _write(self.declaration, [_job()])
        self.hold.write_text(json.dumps(
            {"by": "dashboard", "since": "2026-11-01T12:00:00+00:00",
             "reason": "owner asked for a pause"}), encoding="utf-8")
        state = scheduler.hold()
        self.assertTrue(state["held"])
        self.assertEqual(state["by"], "dashboard")
        self.assertEqual(state["since"], "2026-11-01T12:00:00+00:00")
        self.assertEqual(state["reason"], "owner asked for a pause")

    def test_describe_reports_the_hold_so_paused_is_not_confused_with_broken(self):
        """A paused loop and a broken loop both run nothing.

        Telling them apart is the entire question an operator is asking, so the
        state is reported explicitly rather than inferred from an empty job list.
        """
        _write(self.declaration, [_job()])
        self.hold.write_text("{}", encoding="utf-8")
        doc = scheduler.describe()
        self.assertTrue(doc["hold"]["held"])
        self.assertTrue(doc["hold"]["configured"])
        self.assertTrue(doc["schedule_readable"],
                        "the declaration is still readable when paused")
        #: ...and the jobs themselves still show as enabled, because the
        #: declaration was not edited. Only `_due` reflects the pause.
        self.assertFalse(doc["jobs"][0]["_due"])

    def test_removing_the_hold_lets_the_next_pass_run(self):
        _write(self.declaration, [_job()])
        self.hold.write_text("{}", encoding="utf-8")
        self.stub_task()
        self.assertEqual(scheduler.run_due_jobs(), [])
        self.hold.unlink()
        self.assertEqual(len(scheduler.run_due_jobs()), 1)
        self.assertEqual(self.ran, ["run a pass"])


class TestRunKilledWithTheProcess(SchedulerPaths):
    """A pass that dies mid-run must not cost a whole cycle.

    ``last_run`` is stamped BEFORE the mission and ``finished`` only when it
    returns -- the ordering that stops an overlapping poll from double-starting
    it. The consequence is a signature nothing used to interpret: a stamp with
    no finishing stamp means the run DIED. Live 2026-11-02 the 06:13 pass was
    killed by a recreate five minutes in, its slot was treated as served, the
    next pass waited until 06:49, and the console showed ``running`` for 36
    minutes because ``finished`` still described the run before it.

    The detection is exact rather than a timeout because jobs run in this
    process: at startup no run can be live, so a stranded stamp can only have
    been left by a process that is gone. The tests pin that boundary too --
    reclaiming MID-loop would steal the slot of a run that is still working.
    """

    def stranded(self, started_min_ago: float = 5.0) -> None:
        """State as a killed run leaves it: last_run, no finishing stamp.

        The default is 5 minutes into a 30-minute cycle -- the real shape of the
        06:13 kill. A stamp OLDER than the interval would already be due again
        and would prove nothing, which is exactly how the first version of this
        fixture passed for the wrong reason.
        """
        self.state.parent.mkdir(parents=True, exist_ok=True)
        started = (datetime.now()
                   - timedelta(minutes=started_min_ago)).isoformat()
        previous = (datetime.now()
                    - timedelta(minutes=started_min_ago + 30)).isoformat()
        #: MERGED into whatever is already there, not replaced: a second
        #: interruption must accumulate on the first, which is the whole point
        #: of interrupted_count. Replacing the document wiped it and made the
        #: accumulation test pass by reading a fresh 1.
        try:
            doc = json.loads(self.state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
        doc.setdefault("jobs", {}).setdefault("monitor", {}).update({
            "last_run": started,
            "last_ok": True,
            "finished": previous,
        })
        self.state.write_text(json.dumps(doc), encoding="utf-8")

    def test_a_stranded_slot_is_re_opened(self):
        _write(self.declaration, [_job(every_min=30)])
        self.stranded()
        self.assertEqual(scheduler.due_jobs(), [],
                         "precondition: the slot is consumed, which is the bug")
        self.assertEqual(scheduler.reclaim_interrupted(), ["monitor"])
        self.assertEqual(len(scheduler.due_jobs()), 1,
                         "the lost pass must be due immediately")

    def test_the_interruption_is_recorded_not_silently_retried(self):
        """A deploy that keeps eating missions should be visible."""
        _write(self.declaration, [_job(every_min=30)])
        self.stranded()
        scheduler.reclaim_interrupted()
        entry = json.loads(self.state.read_text(encoding="utf-8"))["jobs"]["monitor"]
        self.assertNotIn("last_run", entry, "the slot must be re-opened")
        self.assertIn("started", entry["interrupted"])
        self.assertIn("re-opened", entry["interrupted"]["reason"])
        self.assertEqual(entry["interrupted_count"], 1)

    def test_the_count_accumulates_across_interruptions(self):
        _write(self.declaration, [_job(every_min=30)])
        self.stranded()
        scheduler.reclaim_interrupted()
        self.stranded(started_min_ago=3)
        scheduler.reclaim_interrupted()
        entry = json.loads(self.state.read_text(encoding="utf-8"))["jobs"]["monitor"]
        self.assertEqual(entry["interrupted_count"], 2)

    def test_a_completed_run_is_left_alone(self):
        """finished >= last_run means it ended; there is nothing to reclaim."""
        _write(self.declaration, [_job(every_min=30)])
        self.stub_task()
        scheduler.run_due_jobs()
        self.assertEqual(scheduler.reclaim_interrupted(), [])
        self.assertEqual(scheduler.due_jobs(), [],
                         "a completed run must still wait for its interval")

    def test_the_reclaimed_pass_actually_runs(self):
        _write(self.declaration, [_job(every_min=30)])
        self.stranded()
        self.stub_task()
        scheduler.reclaim_interrupted()
        records = scheduler.run_due_jobs()
        self.assertEqual(len(records), 1)
        self.assertEqual(self.ran, ["run a pass"])

    def test_the_console_stops_saying_running_once_reclaimed(self):
        """The 36-minute 'running' was the same defect seen from the console."""
        _write(self.declaration, [_job(every_min=30)])
        self.stranded()
        self.assertTrue(scheduler.status()["jobs"][0]["running"],
                        "precondition: a stranded stamp reads as in progress")
        scheduler.reclaim_interrupted()
        job = scheduler.status()["jobs"][0]
        self.assertFalse(job["running"])
        self.assertIsNotNone(job["interrupted"],
                             "the view must say the pass was killed, not lost")
        self.assertEqual(job["interrupted_count"], 1)

    def test_a_failed_run_is_not_mistaken_for_an_interrupted_one(self):
        """A failure CONSUMES its slot on purpose (no 30-second retry loop)."""
        _write(self.declaration, [_job(every_min=30)])
        self.stub_task(raises=RuntimeError("provider 429"))
        scheduler.run_due_jobs()
        self.assertEqual(scheduler.reclaim_interrupted(), [],
                         "a recorded failure must not be re-queued as if the "
                         "process had died")
        self.assertEqual(scheduler.due_jobs(), [])

    def test_nothing_to_reclaim_when_the_state_is_empty(self):
        _write(self.declaration, [_job()])
        self.assertEqual(scheduler.reclaim_interrupted(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
