"""The schedule and run-history surfaces of the console.

Why these exist: a scheduled mission runs every 30 minutes while nobody is
watching, and the console showed only the task in the CURRENT tab. So two
questions an operator actually asks -- "is it still running every half hour?"
and "what did it do on that run?" -- were unanswerable from the UI even though
the evidence sat on disk. Worse, "paused", "disabled" and "the scheduler died"
all looked identical: an empty page.

Each test therefore asserts the DATA reaches the surface, not merely that a
route exists. A page that 200s with an empty table fixes nothing.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import scheduler  # noqa: E402
import webui  # noqa: E402


class TempDirs(unittest.TestCase):
    """A scratch schedule, state file and sessions directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sessions = self.root / "sessions"
        self.sessions.mkdir()
        self.schedule = self.root / "agent-schedule.json"
        self.state = self.root / "schedule_state.json"
        self.hold = self.root / "paused.json"

        for name, value in (("SCHEDULE_FILE", self.schedule),
                            ("SCHEDULE_STATE_FILE", self.state),
                            ("SCHEDULE_HOLD", str(self.hold)),
                            ("SESSIONS_DIR", self.sessions)):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.write_job()

    def write_job(self, *, every_min=30, enabled=True, message="do the mission",
                  jobs_extra=None):
        jobs = [{"id": "pass", "every_min": every_min, "enabled": enabled,
                 "message": message}]
        jobs.extend(jobs_extra or [])
        self.schedule.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

    def write_state(self, last_run=None, finished=None, ok=True, error=None):
        jobs = {"pass": {}}
        if last_run:
            jobs["pass"]["last_run"] = last_run
        if finished:
            jobs["pass"]["finished"] = finished
        if ok is not None:
            jobs["pass"]["last_ok"] = ok
        if error:
            jobs["pass"]["last_error"] = error
        self.state.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

    def write_session(self, sid="abcd1234", stamp="2026-11-02T03-17-48",
                      prompt="the scheduled mission", status="ok", steps=12,
                      elapsed=31.5, answer="the run summary"):
        path = self.sessions / f"{stamp}_{sid}.jsonl"
        lines = [json.dumps({"event": "session_start", "at": f"{stamp}.000000"
                             "+00:00", "task": prompt,
                             "model": "inferhub/deepseek"})]
        lines.append(json.dumps({"event": "assistant", "at": f"{stamp}.01+00:00",
                                 "text": "thinking about it"}))
        lines.append(json.dumps({"event": "action", "at": f"{stamp}.02+00:00",
                                 "step": 1, "tool": "run_command",
                                 "detail": "ls", "seconds": 0.1, "ok": True,
                                 "result_preview": "exit=0"}))
        if answer:
            lines.append(json.dumps({"event": "answer", "at": f"{stamp}.03+00:00",
                                     "text": answer, "steps": steps}))
        if status is not None:
            lines.append(json.dumps({"event": "session_end",
                                     "at": f"{stamp}.04+00:00",
                                     "status": status, "steps": steps,
                                     "elapsed_sec": elapsed}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class ScheduleStatusTest(TempDirs):
    def test_the_interval_and_last_run_are_reported(self):
        """The literal question: is it running every 30 minutes?"""
        self.write_state(last_run="2026-11-02T03:47:52")
        #: NAIVE, matching the clock the run loop and the state file use.
        now = datetime(2026, 11, 2, 3, 52, 0)
        doc = scheduler.status(now=now)
        self.assertEqual(len(doc["jobs"]), 1)
        job = doc["jobs"][0]
        self.assertEqual(job["id"], "pass")
        self.assertEqual(job["interval"], "every 30 min")
        self.assertEqual(job["every_min"], 30)
        self.assertTrue(job["enabled"])
        self.assertEqual(job["last_run"], "2026-11-02T03:47:52")
        self.assertFalse(doc["paused"])

    def test_an_aware_clock_does_not_crash_the_schedule_view(self):
        """Mixed awareness must degrade, not raise.

        ``_due`` gates every run, so a TypeError there stops the mission loop
        entirely -- a far worse outcome than a blank page. The state file holds
        naive local stamps while sessions are named in UTC, so the two really do
        meet.
        """
        self.write_state(last_run="2026-11-02T03:47:52")
        now = datetime(2026, 11, 2, 3, 52, 0, tzinfo=timezone.utc)
        job = scheduler.status(now=now)["jobs"][0]
        self.assertEqual(job["seconds_until"], 1552)
        self.assertFalse(job["due_now"])

    def test_due_itself_tolerates_a_mixed_clock(self):
        """Guard the production path, not just the reporting wrapper."""
        job = {"id": "j", "every_min": 30, "enabled": True}
        state = {"jobs": {"j": {"last_run": "2026-11-02T03:47:52"}}}
        self.assertFalse(
            scheduler._due(job, datetime(2026, 11, 2, 3, 52, tzinfo=timezone.utc),
                           state, "j"))
        self.assertTrue(
            scheduler._due(job, datetime(2026, 11, 2, 4, 20, tzinfo=timezone.utc),
                           state, "j"))

    def test_next_due_is_computed_from_the_last_run(self):
        self.write_state(last_run="2026-11-02T03:47:52")
        now = datetime(2026, 11, 2, 3, 52, 0, tzinfo=timezone.utc)
        job = scheduler.status(now=now)["jobs"][0]
        self.assertEqual(job["next_due"], "2026-11-02T04:17:52+00:00")
        #: 25m52s from 03:52.
        self.assertEqual(job["seconds_until"], 1552)
        self.assertFalse(job["overdue"])

    def test_an_overdue_job_is_flagged_so_a_wedge_is_visible(self):
        """A scheduler that stopped firing must look different from a quiet one."""
        self.write_state(last_run="2026-11-02T01:00:00")
        now = datetime(2026, 11, 2, 3, 52, 0, tzinfo=timezone.utc)
        job = scheduler.status(now=now)["jobs"][0]
        self.assertTrue(job["overdue"])
        self.assertLess(job["seconds_until"], 0)

    def test_a_hold_file_reports_paused_with_who_and_when(self):
        self.hold.write_text(json.dumps({"by": "dashboard Agent console",
                                        "since": "2026-11-02T00:10:00",
                                        "reason": "maintenance"}),
                             encoding="utf-8")
        doc = scheduler.status()
        self.assertTrue(doc["paused"])
        self.assertEqual(doc["hold"]["by"], "dashboard Agent console")
        self.assertEqual(doc["hold"]["reason"], "maintenance")

    def test_an_unreadable_hold_file_still_pauses(self):
        """Presence is the signal; a parse failure must not resume the agent."""
        self.hold.write_text("{not json", encoding="utf-8")
        doc = scheduler.status()
        self.assertTrue(doc["paused"])
        self.assertIn("paused", doc["hold"]["reason"])

    def test_disabled_is_distinct_from_paused(self):
        self.write_job(enabled=False)
        doc = scheduler.status()
        self.assertFalse(doc["paused"])
        self.assertFalse(doc["jobs"][0]["enabled"])
        self.assertFalse(doc["any_enabled"])

    def test_the_full_prompt_is_served_not_just_a_preview(self):
        """The prompt is the thing an operator most wants to audit."""
        long_prompt = "### MISSION ###\n" + ("line of instruction\n" * 40)
        self.write_job(message=long_prompt)
        job = scheduler.status()["jobs"][0]
        self.assertEqual(job["prompt"], long_prompt)
        self.assertEqual(job["prompt_chars"], len(long_prompt))
        #: The preview is one line, so the table does not explode.
        self.assertNotIn("\n", job["prompt_preview"])
        self.assertLessEqual(len(job["prompt_preview"]), 160)

    def test_a_failed_last_run_reports_its_error(self):
        self.write_state(last_run="2026-11-02T03:00:00", ok=False,
                         error="applier refused: no readiness file")
        job = scheduler.status()["jobs"][0]
        self.assertFalse(job["last_ok"])
        self.assertIn("readiness", job["last_error"])

    def test_a_job_that_never_ran_has_no_next_due(self):
        """Inventing one would claim a schedule that does not exist yet."""
        self.write_state()
        job = scheduler.status()["jobs"][0]
        self.assertIsNone(job["last_run"])
        self.assertIsNone(job["next_due"])

    def test_no_jobs_is_reported_rather_than_crashing(self):
        self.schedule.write_text(json.dumps({"jobs": []}), encoding="utf-8")
        doc = scheduler.status()
        self.assertEqual(doc["jobs"], [])
        self.assertFalse(doc["any_enabled"])

    def test_an_hourly_cadence_reads_as_hours(self):
        self.write_job(every_min=60)
        self.assertEqual(scheduler.status()["jobs"][0]["interval"], "every 1h")


class RunHistoryTest(TempDirs):
    def test_a_run_is_summarised_with_its_prompt_and_answer(self):
        self.write_session()
        rows = webui.run_summaries()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["session_id"], "abcd1234")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["steps"], 12)
        self.assertEqual(row["elapsed_sec"], 31.5)
        self.assertEqual(row["prompt"], "the scheduled mission")
        self.assertIn("the run summary", row["answer_preview"])
        self.assertEqual(row["model"], "inferhub/deepseek")

    def test_runs_are_newest_first(self):
        """Cadence is read off this order; oldest-first would invert the story."""
        self.write_session(sid="older111", stamp="2026-11-02T02-17-26")
        self.write_session(sid="newer222", stamp="2026-11-02T03-17-48")
        rows = webui.run_summaries()
        self.assertEqual([r["session_id"] for r in rows],
                         ["newer222", "older111"])

    def test_a_run_with_no_end_record_is_not_reported_as_ok(self):
        """A killed run is exactly what needs to be visible."""
        path = self.write_session(status=None)
        # Make it look old so it is not mistaken for one still running.
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).timestamp()
        import os
        os.utime(path, (old, old))
        row = webui.run_summaries()[0]
        self.assertEqual(row["status"], "no_end_record")
        self.assertNotEqual(row["status"], "ok")

    def test_a_run_still_writing_is_reported_as_running(self):
        self.write_session(status=None)
        self.assertEqual(webui.run_summaries()[0]["status"], "running")

    def test_a_failed_run_reports_its_status(self):
        self.write_session(status="error")
        self.assertEqual(webui.run_summaries()[0]["status"], "error")

    def test_a_run_with_no_answer_says_so(self):
        self.write_session(answer=None)
        row = webui.run_summaries()[0]
        self.assertFalse(row["has_answer"])
        self.assertEqual(row["answer_preview"], "")

    def test_the_detail_view_returns_the_full_prompt_and_steps(self):
        long_prompt = "### MISSION ###\n" + ("x" * 500)
        self.write_session(prompt=long_prompt)
        detail = webui.run_detail("abcd1234")
        self.assertIsNotNone(detail)
        self.assertEqual(detail["prompt"], long_prompt)
        self.assertEqual(detail["answer"], "the run summary")
        tools = [e for e in detail["events"] if e.get("event") == "action"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["tool"], "run_command")
        #: The session_start/end bookkeeping is not repeated as a "step".
        self.assertNotIn("session_start",
                         [e.get("event") for e in detail["events"]])

    def test_an_unknown_run_is_not_found_rather_than_empty(self):
        self.write_session()
        self.assertIsNone(webui.run_detail("nosuchid"))

    def test_a_half_written_record_does_not_break_the_listing(self):
        """The list is read while a run is appending; it must not throw."""
        path = self.write_session()
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"event": "action", "step": 9')   # no newline, incomplete
        rows = webui.run_summaries()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "abcd1234")


class PageSurfaceTest(TempDirs):
    """The data must reach the PAGE, not just the JSON endpoint."""

    def test_the_runs_page_renders_the_schedule_and_the_table(self):
        page = webui.RUNS_PAGE
        for needed in ("/schedule", "/runs.json", "/run/", "id=\"runs\"",
                       "id=\"sched\"", "started (UTC)"):
            self.assertIn(needed, page, f"the page never uses {needed!r}")

    def test_the_page_distinguishes_paused_from_disabled(self):
        page = webui.RUNS_PAGE
        self.assertIn("PAUSED", page)
        self.assertIn("disabled", page)
        self.assertIn("overdue", page)

    def test_the_page_escapes_values_it_renders(self):
        """A prompt or answer carrying HTML must not be injected."""
        page = webui.RUNS_PAGE
        self.assertIn("function esc(", page)
        self.assertIn("esc(j.prompt)", page)
        self.assertIn("esc(j.answer)", page)

    def test_both_existing_pages_link_to_the_new_view(self):
        for name in ("PAGE", "LIVE_PAGE"):
            self.assertIn("/runs", getattr(webui, name),
                          f"{name} has no link to the schedule view")

    def test_every_page_is_served_with_the_base_prefix_substituted(self):
        for name in ("PAGE", "LIVE_PAGE", "RUNS_PAGE"):
            html = webui._render(getattr(webui, name))
            self.assertNotIn("__AGENT_BASE__", html,
                             f"{name} still contains the placeholder")


class HandlerRouteTest(TempDirs):
    """Drive the real HTTP handler.

    Asserting only that the helpers return data leaves the WIRING untested: a
    route could serve the wrong page, or not exist at all, and every helper test
    would still pass. These start the actual server and ask it.
    """

    def setUp(self):
        super().setUp()
        import threading
        import urllib.error
        import urllib.request
        from http.server import ThreadingHTTPServer
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port),
                                          webui.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

        def _stop():
            # ORDER MATTERS: shutdown() asks serve_forever to return, join()
            # waits for it, and only then is it safe to close the socket.
            # Closing first makes the selector raise
            # `OSError: [WinError 10038] ... not a socket` from the server
            # thread, and reversing shutdown/join stalls the suite until the
            # join timeout expires.
            self.server.shutdown()
            self.thread.join(5)
            self.server.server_close()

        self.addCleanup(_stop)
        self._urlopen = urllib.request.urlopen
        self._http_error = urllib.error.HTTPError

    def get(self, path: str):
        import urllib.error
        import urllib.request
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.status, response.read().decode("utf-8",
                                                               "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def test_runs_serves_the_runs_page_not_the_console(self):
        self.write_session()
        status, body = self.get("/runs")
        self.assertEqual(status, 200)
        self.assertIn("schedule &amp; runs", body)
        self.assertNotIn('id="pick"', body, "served the console page instead")

    def test_schedule_returns_the_job_with_its_interval(self):
        self.write_state(last_run="2026-11-02T03:47:52")
        status, body = self.get("/schedule")
        self.assertEqual(status, 200)
        doc = json.loads(body)
        self.assertEqual(doc["jobs"][0]["interval"], "every 30 min")
        self.assertEqual(doc["jobs"][0]["last_run"], "2026-11-02T03:47:52")
        self.assertFalse(doc["paused"])

    def test_runs_json_lists_actual_runs(self):
        self.write_session()
        status, body = self.get("/runs.json")
        self.assertEqual(status, 200)
        doc = json.loads(body)
        self.assertEqual(len(doc["runs"]), 1)
        self.assertEqual(doc["runs"][0]["session_id"], "abcd1234")
        self.assertEqual(doc["runs"][0]["status"], "ok")

    def test_a_single_run_returns_its_prompt_and_answer(self):
        self.write_session(prompt="the scheduled mission")
        status, body = self.get("/run/abcd1234")
        self.assertEqual(status, 200)
        doc = json.loads(body)
        self.assertEqual(doc["prompt"], "the scheduled mission")
        self.assertEqual(doc["answer"], "the run summary")

    def test_an_unknown_run_is_404(self):
        status, _body = self.get("/run/deadbeef")
        self.assertEqual(status, 404)

    def test_the_console_and_live_pages_still_serve(self):
        for path in ("/", "/live"):
            with self.subTest(path=path):
                status, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertIn("agent-harness", body)


if __name__ == "__main__":
    unittest.main()
