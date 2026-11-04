"""The cadence a page advertises must not be able to be broken by its own bound.

The scheduler loop is sequential on purpose -- two concurrent missions would edit
the same files and judge the same defects twice -- which makes the length of ONE
run the length of every mission behind it. So the interval is only ever a floor
on the gap between firings, never a promise about duration.

Nothing said so. The console printed `every_min` beside `next_due` and both
looked authoritative, while the number that actually decides how long a slot can
be held (the wall-clock bound) lived in an environment variable the page never
read. The result was a page that reads as healthy in two states that are not:

  * a bound of 5400 s against a 3600 s interval -- one run may legally outlast
    the interval it belongs to, so the next firing slips and every "next due"
    the page reported was wrong by up to 30 minutes;
  * no bound at all (`AGENT_MAX_RUN_SEC=0`, the default) -- the only ceiling is
    turns, and turns have no duration, so a provider that times out can hold the
    slot for hours. That is the case that produced the 120-minute run, 14
    timeouts and a `next_due` an hour in the past which the operator read, quite
    reasonably, as "nothing is running and the mission never started".

An absent field would have read as "nothing to worry about", so the unbounded
case is reported too, and styled as a note rather than an error: a bound longer
than the cadence is a sizing choice an operator may have made deliberately, and
colouring it red would train the eye away from the statuses that are genuinely
broken.
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import agent  # noqa: E402
import scheduler  # noqa: E402
import webui  # noqa: E402
from test_webui_runs import TempDirs  # noqa: E402


class BoundVersusCadenceTest(TempDirs):
    """`scheduler.status()` states the relationship, not just the two numbers."""

    def job(self, every_min=60, bound=0.0):
        self.write_job(every_min=every_min, enabled=True)
        with mock.patch.object(config, "MAX_RUN_SEC", bound):
            return scheduler.status()["jobs"][0]

    def test_a_bound_longer_than_the_cadence_is_said_out_loud(self):
        """5400 s bound on a 60-minute job: the interval cannot be a promise."""
        j = self.job(every_min=60, bound=5400.0)
        self.assertTrue(j["bound_overruns_cadence"])
        self.assertEqual(j["run_bound_sec"], 5400.0)
        self.assertEqual(j["every_min_seconds"], 3600.0)
        reason = j["bound_overrun_reason"]
        self.assertIn("5400s", reason)
        self.assertIn("3600s", reason)
        #: It has to name the two knobs, or the operator is told there is a
        #: problem without being told what to change -- the same complaint the
        #: brief makes about a report that restates intent.
        self.assertIn("AGENT_MAX_RUN_SEC", reason)
        self.assertIn("interval", reason)

    def test_a_bound_that_fits_inside_the_cadence_reports_no_problem(self):
        j = self.job(every_min=60, bound=3000.0)
        self.assertFalse(j["bound_overruns_cadence"])
        self.assertIsNone(j["bound_overrun_reason"],
                          "a coherent pair must not emit a reason: noise in the "
                          "one field an operator reads gets ignored")

    def test_a_bound_exactly_equal_to_the_cadence_is_not_overrun(self):
        """Equality is the boundary case, and it is NOT an overrun.

        A run that lasts exactly as long as the interval still leaves the next
        firing on time, so flagging it would be a false positive on the one
        tuning an operator is most likely to pick.
        """
        j = self.job(every_min=60, bound=3600.0)
        self.assertFalse(j["bound_overruns_cadence"])
        self.assertIsNone(j["bound_overrun_reason"])

    def test_no_bound_at_all_is_reported_because_absence_would_read_as_fine(self):
        j = self.job(every_min=30, bound=0.0)
        self.assertFalse(j["bound_overruns_cadence"],
                         "unbounded is not 'overruns by N seconds'; it is the "
                         "other failure mode and must not share the flag")
        self.assertIsNone(j["run_bound_sec"], "0 must not be printed as a bound")
        reason = j["bound_overrun_reason"]
        self.assertIn("AGENT_MAX_RUN_SEC=0", reason)
        self.assertIn("no wall-clock bound", reason)
        #: The reason has to carry the mechanism, not just the verdict.
        self.assertIn("DURATION", reason)

    def test_the_reason_is_per_job_because_the_cadence_is_per_job(self):
        """A console override applies to one job; an `at` job has no interval.

        Reporting one global pair would be wrong the moment there is more than
        one job, which is the shape this file exists to support.
        """
        jobs = [{"id": "every-min-job", "every_min": 30, "enabled": True,
                 "message": "m"},
                {"id": "nightly", "at": "02:30", "enabled": True, "message": "n"}]
        self.schedule.write_text(__import__("json").dumps({"jobs": jobs}),
                                encoding="utf-8")
        with mock.patch.object(config, "MAX_RUN_SEC", 5400.0):
            got = {j["id"]: j for j in scheduler.status()["jobs"]}
        self.assertTrue(got["every-min-job"]["bound_overruns_cadence"])
        self.assertEqual(got["every-min-job"]["every_min_seconds"], 1800.0)
        #: A daily job has no interval, so no cadence to overtake. It must say
        #: the bound is unbounded-by-cadence rather than claim a safe margin.
        self.assertIsNone(got["nightly"]["every_min_seconds"])
        self.assertFalse(got["nightly"]["bound_overruns_cadence"])

    def test_the_cadence_compared_is_the_live_one_not_the_declared_one(self):
        """A console override of 60 replaces a declared 45 for firing purposes."""
        self.write_job(every_min=45, enabled=True)
        with mock.patch.object(config, "MAX_RUN_SEC", 3000.0), \
                mock.patch.object(scheduler.run_settings, "effective_every_min",
                                  return_value=(60, "console")):
            j = scheduler.status()["jobs"][0]
        self.assertEqual(j["every_min"], 60)
        self.assertEqual(j["every_min_source"], "console")
        self.assertEqual(j["every_min_seconds"], 3600.0)
        self.assertFalse(j["bound_overruns_cadence"],
                         "3000 s fits the LIVE 60-min cadence; comparing the "
                         "declared 45 instead would report a false overrun")

    def test_a_disabled_job_still_reports_the_pair(self):
        """So that arming it later does not surprise anyone."""
        self.write_job(every_min=10, enabled=False)
        with mock.patch.object(config, "MAX_RUN_SEC", 5400.0):
            j = scheduler.status()["jobs"][0]
        self.assertFalse(j["enabled"])
        self.assertTrue(j["bound_overruns_cadence"])


class ConsoleShowsTheBoundTest(TempDirs):
    """A payload field no page prints is a field nobody reads."""

    def test_the_scheduler_view_carries_the_bound_and_the_reason(self):
        self.write_job(every_min=30, enabled=True)
        with mock.patch.object(config, "MAX_RUN_SEC", 5400.0):
            view = webui._scheduler_view()
        self.assertTrue(view["available"])
        self.assertEqual(view["run_bound_sec"], 5400.0)
        self.assertEqual(view["every_min_seconds"], 1800.0)
        self.assertTrue(view["bound_overruns_cadence"])
        self.assertIn("5400s", view["bound_overrun_reason"])

    def test_the_unbounded_case_reaches_the_view_with_the_flag_false(self):
        """Flag false plus a reason is a deliberate shape, not an oversight.

        The view must not collapse the two cases into one, or the page can only
        choose between hiding the unbounded state and crying wolf about a safe
        pair.
        """
        self.write_job(every_min=30, enabled=True)
        with mock.patch.object(config, "MAX_RUN_SEC", 0.0):
            view = webui._scheduler_view()
        self.assertFalse(view["bound_overruns_cadence"])
        self.assertIsNone(view["run_bound_sec"])
        self.assertIn("AGENT_MAX_RUN_SEC=0", view["bound_overrun_reason"])

    def test_the_page_renders_the_bound_and_the_warning(self):
        page = webui.RUNS_PAGE
        self.assertIn("run_bound_sec", page)
        self.assertIn("bound_overrun_reason", page)
        self.assertIn("wall-clock bound", page)
        self.assertIn("unbounded", page,
                      "the page must say when there is no bound; printing "
                      "nothing next to a cadence reads as 'safe'")

    def test_the_warning_is_a_note_not_a_red_status(self):
        """Colouring a sizing choice red competes with genuinely broken runs."""
        page = webui.RUNS_PAGE
        self.assertIn("class=\"warn\"", page)
        #: The bound note must not use the `bad` class that statuses like
        #: crashed / no_end_record use.
        bound_block = page.split("let bound=''", 1)[1].split("</div>';", 1)[0]
        self.assertNotIn("'bad'", bound_block,
                         "the bound note uses the error colour; it must stay a "
                         "warning so red still means broken")

    def test_the_reason_is_escaped_before_the_dom_gets_it(self):
        """It contains the operator's own numbers and knob names; never trust it.

        Every other interpolated string on this page goes through esc(), and the
        one new field is exactly where someone would paste a value they did not
        write.
        """
        self.assertIn("esc(job.bound_overrun_reason)", webui.RUNS_PAGE)


class SchedulerReadsTheLiveConfigTest(unittest.TestCase):
    """The bound must be read at call time, never copied at import.

    ``MAX_RUN_SEC`` is a deployment env var, and the console can change the
    cadence at RUNTIME (``run_settings``), so a module-level snapshot of either
    would silently disagree with the loop that actually enforces them.
    """

    def test_status_reads_through_the_config_object(self):
        src = inspect.getsource(scheduler.status)
        self.assertIn('getattr(config, "MAX_RUN_SEC"', src,
                      "status must read config.MAX_RUN_SEC when called")
        self.assertNotIn("MAX_RUN_SEC =", src.replace(
            "run_bound_sec = float(getattr(config, \"MAX_RUN_SEC\"", ""))

    def test_the_bound_the_agent_enforces_is_the_one_status_reports(self):
        """Same attribute, both sides, or the page lies about the loop."""
        loop = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("config.MAX_RUN_SEC", loop,
                      "the loop stopped reading config.MAX_RUN_SEC; the "
                      "scheduler's report would then describe a bound nothing "
                      "enforces")


if __name__ == "__main__":
    unittest.main(verbosity=2)
