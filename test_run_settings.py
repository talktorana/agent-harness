"""``run_settings``: the cadence and the step ceiling are live settings.

WHY THIS SUITE EXISTS
---------------------
Two numbers decide how an unattended mission behaves -- how often it fires and
how many turns it may take -- and both used to be changeable ONLY by editing a
declaration file and recreating the container. On a deployment where recreating
kills a run in flight, that pairing is the worst possible one: the operator who
watches a pass die at its ceiling every half hour cannot fix it without a
redeploy, and the redeploy itself risks killing the next pass.

So the settings are read at USE time from a file the process can write, and
these tests pin the properties that make that safe rather than merely
convenient:

  1. VALIDATION REFUSES, NEVER CLAMPS. A ceiling of 0 ends every run on turn
     one; an interval of 0 fires on every poll and burns tokens forever. Both
     are rejected with a reason. Clamping would let "apply" look successful
     while doing something else -- and the operator who typed 5000 has no way
     to learn they got 1000.
  2. PROVENANCE IS PART OF THE ANSWER. Every reader returns ``(value,
     source)``, so the page can say "override in force, the environment ships
     40" instead of showing a number nobody can attribute. A redeploy silently
     reverting a console choice is exactly the surprise provenance prevents.
  3. A CORRUPT FILE IS AN ERROR, NOT A DEFAULT. Returning ``{}`` on a parse
     failure would restore the shipped ceiling without a word, and runs would
     start dying at the old limit while the console showed the value the
     operator set.
  4. THE WARNING DISTANCE SCALES. ``ANSWER_WARN_STEPS`` fires ONCE, at an exact
     equality, so it is a run's only chance to stop and write an answer. A
     constant tuned for 40 turns is 1.5% of a 200-turn budget -- too late to be
     of any use. It is DERIVED from the ceiling rather than stored as a second
     number an operator must remember to change.
  5. NOTHING IS WRITTEN WHEN A VALUE IS BAD. ``save`` validates every argument
     before touching the file, so one good value plus one bad one changes
     nothing, rather than half-applying and leaving the operator to guess which
     half took.

Run:  python -m unittest test_run_settings
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import run_settings


class SettingsFileTest(unittest.TestCase):
    """Every test gets its own settings file, so none can leak into another."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "run_settings.json"
        patch = mock.patch.object(run_settings, "SETTINGS_FILE", self.path)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    # -- 1. validation refuses, never clamps -------------------------------
    def test_a_zero_ceiling_is_refused_not_clamped(self):
        with self.assertRaises(run_settings.SettingsError) as ctx:
            run_settings.save(max_steps=0)
        self.assertIn("must be between", str(ctx.exception))

    def test_a_negative_ceiling_is_refused(self):
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(max_steps=-40)

    def test_an_absurd_ceiling_is_refused_rather_than_silently_capped(self):
        """5000 turns is not a tuning choice; clamping it to 4000 would hide that."""
        with self.assertRaises(run_settings.SettingsError) as ctx:
            run_settings.save(max_steps=run_settings.MAX_STEPS_MAX + 1)
        self.assertIn("must be between", str(ctx.exception))

    def test_a_zero_interval_is_refused(self):
        """An every_min of 0 fires on every poll -- a token-burning loop."""
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(every_min=0)

    def test_json_true_is_not_accepted_as_the_number_one(self):
        """`isinstance(True, int)` is True, so a boolean would arrive as 1.

        A ceiling of 1 would stop every run after a single turn, and the value
        looks like it came from a legitimate request.
        """
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(max_steps=True)

    def test_a_decimal_is_refused_for_a_count_of_turns(self):
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(max_steps=150.7)

    def test_a_numeric_string_is_accepted(self):
        """HTML inputs arrive as strings; refusing them would break the panel."""
        run_settings.save(max_steps="150", every_min="45")
        self.assertEqual(run_settings.effective_max_steps()[0], 150)

    def test_an_empty_string_is_refused_with_a_reason(self):
        with self.assertRaises(run_settings.SettingsError) as ctx:
            run_settings.save(max_steps="   ")
        self.assertIn("must not be empty", str(ctx.exception))

    def test_nonsense_text_is_refused_with_a_reason(self):
        with self.assertRaises(run_settings.SettingsError) as ctx:
            run_settings.save(every_min="forty-five")
        self.assertIn("whole number", str(ctx.exception))

    # -- 5. nothing is written when a value is bad -------------------------
    def test_one_bad_value_does_not_half_apply_a_request(self):
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(max_steps=150, every_min=0)
        self.assertFalse(self.path.exists(),
                         "a refused request still wrote the settings file, so "
                         "the good value took effect while the operator was "
                         "told the request failed")

    def test_a_refused_save_leaves_the_previous_values_in_force(self):
        run_settings.save(max_steps=120, every_min=30)
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save(max_steps=0, every_min=90)
        self.assertEqual(run_settings.effective_max_steps(), (120, "console"))
        self.assertEqual(run_settings.effective_every_min()[0], 30)

    # -- 2. precedence and provenance --------------------------------------
    def test_a_saved_ceiling_beats_the_environment(self):
        run_settings.save(max_steps=150)
        with mock.patch.object(config, "MAX_STEPS", 40):
            self.assertEqual(run_settings.effective_max_steps(), (150, "console"))

    def test_the_environment_wins_when_nothing_was_saved(self):
        """Provenance comes from ``config.source_of``, this repo's own record of
        where each setting was read from -- pinning against ``os.environ`` alone
        would pass for a value nobody exported."""
        with mock.patch.object(config, "MAX_STEPS", 40), \
             mock.patch.object(config, "SOURCES",
                               {"AGENT_MAX_STEPS": "environment (exported)"}):
            self.assertEqual(run_settings.effective_max_steps(),
                             (40, "environment"))

    def test_a_dotenv_sourced_ceiling_also_counts_as_configured(self):
        """``source_of`` distinguishes an export from a .env file; both mean
        "the deployment configured this", and both must beat the built-in."""
        with mock.patch.object(config, "MAX_STEPS", 72), \
             mock.patch.object(config, "SOURCES",
                               {"AGENT_MAX_STEPS": ".env"}):
            self.assertEqual(run_settings.effective_max_steps(),
                             (72, "environment"))

    def test_the_source_says_default_when_neither_is_configured(self):
        with mock.patch.object(config, "MAX_STEPS", 200), \
             mock.patch.object(config, "SOURCES", {}):
            self.assertEqual(run_settings.effective_max_steps(), (200, "default"))

    def test_the_declared_interval_is_used_when_nothing_was_saved(self):
        self.assertEqual(run_settings.effective_every_min(30), (30.0, "schedule"))

    def test_a_saved_interval_beats_the_declaration(self):
        run_settings.save(every_min=45)
        self.assertEqual(run_settings.effective_every_min(30), (45.0, "console"))

    def test_a_job_with_no_interval_at_all_reports_none_not_zero(self):
        """An `at` job has no interval; returning 0 would fire it every poll."""
        value, source = run_settings.effective_every_min(None)
        self.assertIsNone(value)
        self.assertEqual(source, "none")

    def test_clearing_one_setting_does_not_touch_the_other(self):
        """An operator tuning the ceiling must not silently lose a cadence set
        last week."""
        run_settings.save(max_steps=150, every_min=45)
        run_settings.clear("max_steps")
        self.assertEqual(run_settings.effective_every_min(), (45.0, "console"))
        # ...and the ceiling fell back to whatever the deployment configures,
        # with the file no longer claiming it.
        with mock.patch.object(config, "MAX_STEPS", 40), \
             mock.patch.object(config, "SOURCES",
                               {"AGENT_MAX_STEPS": "environment (exported)"}):
            self.assertEqual(run_settings.effective_max_steps(), (40, "environment"))

    def test_clearing_everything_removes_the_file(self):
        """A file with only `saved_at` in it would read as an override."""
        run_settings.save(max_steps=150)
        run_settings.clear()
        self.assertFalse(self.path.exists())

    def test_clearing_an_unknown_name_is_refused(self):
        with self.assertRaises(run_settings.SettingsError):
            run_settings.clear("reasoning_effort")

    def test_saving_nothing_at_all_is_refused(self):
        with self.assertRaises(run_settings.SettingsError):
            run_settings.save()

    # -- 3. a corrupt file is an error, not a default ----------------------
    def test_a_corrupt_settings_file_raises_instead_of_silently_defaulting(self):
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(run_settings.SettingsError) as ctx:
            run_settings.effective_max_steps()
        self.assertIn("not readable JSON", str(ctx.exception))
        self.assertIn(str(self.path), str(ctx.exception),
                      "the error must name the file, or nobody can fix it")

    def test_a_json_array_file_raises_instead_of_being_ignored(self):
        self.path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(run_settings.SettingsError):
            run_settings.effective_every_min()

    def test_a_saved_value_of_the_wrong_type_raises_on_read(self):
        """The file is on a shared volume; anything can land in it."""
        self.path.write_text(json.dumps({"max_steps": True}), encoding="utf-8")
        with self.assertRaises(run_settings.SettingsError):
            run_settings.effective_max_steps()

    def test_state_reports_the_error_rather_than_hiding_it(self):
        """The page must say the ceiling is unknown, not show a plausible one."""
        self.path.write_text("{oops", encoding="utf-8")
        payload = run_settings.state()
        self.assertIn("not readable JSON", payload["error"] or "")
        self.assertEqual(payload["max_steps_source"], "error")

    # -- 4. the warning distance scales with the ceiling -------------------
    def test_the_warning_distance_scales_with_the_ceiling(self):
        """3 of 40 is 7.5%; of a 200-turn budget it is 1.5% -- too late to use."""
        base = config.ANSWER_WARN_STEPS
        scaled = run_settings.effective_answer_warn_steps(config.MAX_STEPS * 4)
        self.assertGreater(scaled, base,
                           "a four-times-larger budget got the same warning "
                           "distance, so the run is warned with almost nothing "
                           "left to act on")
        self.assertEqual(scaled, int(round(base * 4)))

    def test_the_configured_minimum_applies_whenever_there_is_room_for_it(self):
        """The floor is honoured at every ceiling that can accommodate it.

        It cannot be honoured at ceiling 1 or 2 -- see the reachability test --
        and pretending otherwise is how a warning ends up configured at a
        distance the run can never reach.
        """
        floor = config.ANSWER_WARN_STEPS
        for ceiling in (floor * 2, 100, config.MAX_STEPS, 400, 4000):
            self.assertGreaterEqual(run_settings.effective_answer_warn_steps(ceiling),
                                    floor,
                                    "ceiling %d warned below the configured "
                                    "minimum %d" % (ceiling, floor))

    def test_the_warning_is_always_reachable_inside_the_run(self):
        """The warning fires on an exact equality (``remaining == warn``), so a
        distance at or above the ceiling is never reached: the run is cut off
        with no warning and no answer. Every ceiling must yield a distance the
        loop can actually hit -- except a 1-turn ceiling, which has no turn to
        spare and correctly yields 0 (the caller compares against
        ``max(1, warn)``, so nothing breaks; the run simply ends at its limit).
        """
        for ceiling in (1, 2, 3, 5, 40, 200, 4000):
            warn = run_settings.effective_answer_warn_steps(ceiling)
            self.assertGreaterEqual(warn, 0)
            self.assertLessEqual(warn, ceiling,
                                 "ceiling %d warns %d turns out -- unreachable, "
                                 "so the run is cut off silently"
                                 % (ceiling, warn))
            if ceiling == 1:
                self.assertEqual(warn, 0,
                                 "a one-turn run has no turn to warn on")
            else:
                self.assertGreaterEqual(warn, 1, "ceiling %d never warns" % ceiling)
            if ceiling > 2:
                self.assertLess(warn, ceiling,
                                "ceiling %d warns %d turns out -- the warning "
                                "fires before any work happens"
                                % (ceiling, warn))

    def test_the_warning_distance_follows_a_saved_ceiling(self):
        run_settings.save(max_steps=200)
        self.assertEqual(run_settings.effective_answer_warn_steps(),
                         run_settings.effective_answer_warn_steps(200))

    # -- persistence --------------------------------------------------------
    def test_a_save_survives_a_reload_of_the_module_state(self):
        run_settings.save(max_steps=150)
        self.assertEqual(run_settings.effective_max_steps()[0], 150)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["max_steps"], 150)
        self.assertIn("saved_at", payload,
                      "without a timestamp nobody can tell a live choice from "
                      "a file left behind by an old container")

    def test_a_second_save_updates_rather_than_duplicates(self):
        run_settings.save(max_steps=100)
        run_settings.save(max_steps=200)
        run_settings.save(every_min=45)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["max_steps"], 200)
        self.assertEqual(payload["every_min"], 45)

    def test_an_unwritable_location_raises_with_a_cause(self):
        """Pinned by making the write fail, not by guessing an unwritable path:
        a path that is unwritable on Linux may be creatable on Windows, and a
        test that passes only on one platform is a test that lies on the other.
        """
        with tempfile.TemporaryDirectory() as tmp:
            doomed = Path(tmp) / "settings.json"
            with mock.patch.object(run_settings, "SETTINGS_FILE", doomed), \
                 mock.patch("tempfile.mkstemp",
                            side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(run_settings.SettingsError) as ctx:
                    run_settings.save(max_steps=150)
        self.assertIn("cannot write", str(ctx.exception))
        self.assertIn("No space left", str(ctx.exception),
                      "the underlying cause was swallowed, so the operator "
                      "would be told only that a save failed")

    def test_a_failed_save_is_not_reported_as_success(self):
        """The console says "applied" on a 200. A write that did not land must
        therefore raise, or the page lies about a value that is not in force."""
        with tempfile.TemporaryDirectory() as tmp:
            doomed = Path(tmp) / "settings.json"
            with mock.patch.object(run_settings, "SETTINGS_FILE", doomed), \
                 mock.patch("os.replace", side_effect=OSError(5, "Access denied")):
                with self.assertRaises(run_settings.SettingsError):
                    run_settings.save(max_steps=150)
            self.assertFalse(doomed.exists())


class ShippedDefaultsTest(unittest.TestCase):
    """The defaults this repo ships, pinned so a change is a decision.

    These are not arbitrary: the previous values were too small for the one
    mission type this harness actually runs unattended, and every raise so far
    was caused by a run that died exactly at the ceiling mid-delivery.
    """

    def test_a_fresh_checkout_ships_a_ceiling_large_enough_for_a_mission(self):
        """Reading the source, not the imported value: the environment this
        process happens to run in must not decide what a deployment ships."""
        source = Path(config.__file__).read_text(encoding="utf-8")
        self.assertRegex(
            source, r'AGENT_MAX_STEPS",\s*"(\d+)"')
        shipped = int(__import__("re").search(
            r'AGENT_MAX_STEPS",\s*"(\d+)"', source).group(1))
        self.assertGreaterEqual(shipped, 200,
                                "the shipped ceiling was lowered; a mission "
                                "that audits, fixes and reports needs more "
                                "turns than this, and every past cut-off lost "
                                "real work")

    def test_the_shipped_warning_distance_is_a_usable_fraction_of_the_ceiling(self):
        source = Path(config.__file__).read_text(encoding="utf-8")
        import re as _re
        ceiling = int(_re.search(r'AGENT_MAX_STEPS",\s*"(\d+)"', source).group(1))
        warn = int(_re.search(r'AGENT_ANSWER_WARN_STEPS",\s*"(\d+)"',
                             source).group(1))
        fraction = warn / ceiling
        self.assertGreaterEqual(fraction, 0.05,
                                "the warning arrives with under 5% of the run "
                                "left -- not enough turns to write an answer")
        self.assertLess(fraction, 0.5,
                        "warned for the second half of every run trains the "
                        "model to ignore the warning")


class SchedulerHonoursTheOverrideTest(unittest.TestCase):
    """The cadence override has to change WHEN JOBS FIRE, not just the report."""

    def test_a_longer_override_delays_a_due_job(self):
        import scheduler
        from datetime import datetime, timedelta
        now = datetime(2026, 11, 3, 20, 40)
        # Declared every 30; last ran 35 minutes ago -> due by the declaration.
        job = {"id": "j", "every_min": 30, "enabled": True}
        state = {"jobs": {"j": {"last_run": (now - timedelta(minutes=35))
                                .isoformat()}}}
        self.assertTrue(scheduler._due(job, now, state, "j"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                run_settings.save(every_min=45)
                self.assertFalse(scheduler._due(job, now, state, "j"),
                                 "a 45-minute override did not hold back a job "
                                 "the 30-minute declaration made due")
                # ...and 50 minutes after the last run it IS due again.
                later = now + timedelta(minutes=15)
                self.assertTrue(scheduler._due(job, now + timedelta(minutes=15),
                                               state, "j"),
                                "the override never lets the job fire")
                del later

    def test_a_shorter_override_lets_a_job_fire_sooner(self):
        import scheduler
        from datetime import datetime, timedelta
        now = datetime(2026, 11, 3, 20, 40)
        job = {"id": "j", "every_min": 60, "enabled": True}
        state = {"jobs": {"j": {"last_run": (now - timedelta(minutes=20))
                                .isoformat()}}}
        self.assertFalse(scheduler._due(job, now, state, "j"))
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                run_settings.save(every_min=15)
                self.assertTrue(scheduler._due(job, now, state, "j"))

    def test_an_unreadable_override_does_not_stop_the_loop(self):
        """A corrupt settings file must not silently end every scheduled run."""
        import scheduler
        from datetime import datetime, timedelta
        now = datetime(2026, 11, 3, 20, 40)
        job = {"id": "j", "every_min": 30, "enabled": True}
        state = {"jobs": {"j": {"last_run": (now - timedelta(minutes=35))
                                .isoformat()}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text("{broken", encoding="utf-8")
            with mock.patch.object(run_settings, "SETTINGS_FILE", path):
                self.assertTrue(scheduler._due(job, now, state, "j"),
                                "the declared interval was ignored because the "
                                "override file was corrupt -- one bad file "
                                "silently disabled the mission loop")

    def test_the_status_report_names_where_the_interval_came_from(self):
        """Provenance in the payload, or an override is invisible to the very
        page that reports scheduling."""
        import scheduler
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"), \
                 mock.patch.object(scheduler, "_load",
                                   return_value={"jobs": [
                                     {"id": "j", "every_min": 30,
                                      "enabled": True, "message": "hi"}]}), \
                 mock.patch.object(scheduler, "_load_state",
                                   return_value={"jobs": {}}):
                run_settings.save(every_min=45)
                doc = scheduler.status()
        job = doc["jobs"][0]
        self.assertEqual(job["every_min"], 45)
        self.assertEqual(job["every_min_source"], "console")
        self.assertEqual(job["every_min_declared"], 30)


class AgentReadsTheLiveCeilingTest(unittest.TestCase):
    """The loop must use the ceiling in force, not the one baked in at import."""

    def test_the_helper_returns_the_override(self):
        import agent
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                run_settings.save(max_steps=150)
                self.assertEqual(agent._effective_max_steps(), 150)

    def test_the_helper_falls_back_to_the_environment_on_a_bad_file(self):
        """One corrupt settings file must not stop runs altogether."""
        import agent
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text("{broken", encoding="utf-8")
            with mock.patch.object(run_settings, "SETTINGS_FILE", path), \
                 mock.patch.object(config, "MAX_STEPS", 99):
                self.assertEqual(agent._effective_max_steps(), 99)

    def test_the_loop_bound_comes_from_the_helper(self):
        """Structural: the run loop must not read config.MAX_STEPS directly.

        Pinning the call is what makes "the console ceiling is live" a property
        rather than an intention -- a later edit that reverts to the constant
        fails here instead of silently ignoring the panel.

        ``_run_impl``, not ``run``: ``run`` is the crash-guard wrapper that only
        forwards, and looking there found nothing to check -- which is how a
        structural test passes vacuously. The loop bound lives in the impl.
        """
        import inspect
        import agent
        source = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("_effective_max_steps()", source,
                      "the run loop no longer reads the live ceiling, so a value "
                      "set in the console does nothing")
        self.assertNotIn("range(1, config.MAX_STEPS + 1)", source,
                         "the run loop still uses the import-time constant for "
                         "its bound")

    def test_the_cutoff_messages_quote_the_ceiling_that_was_actually_in_force(self):
        """The stop messages must not name a different number than the loop used.

        ``step_limit_with_answer`` runs are diagnosed by comparing the reported
        ceiling against the run's highest step; a message quoting the shipped
        default while the loop ran a console override sends whoever reads it to
        the wrong number.
        """
        import inspect
        import agent
        source = inspect.getsource(agent.Agent._run_impl)
        self.assertNotIn("config.MAX_STEPS", source,
                         "_run_impl still quotes the import-time ceiling in its "
                         "stop messages, which can disagree with the bound the "
                         "loop actually used")


class DeclaredCadenceIsReportedTest(unittest.TestCase):
    """The panel must show the cadence that is actually in force.

    Verified live on 2026-11-03: ``GET /agent/schedule`` reported jobs[0]
    ``every_min 45, every_min_source "schedule"`` while ``GET /agent/settings``
    reported ``null / "none"`` -- the same fact, two endpoints, and the one the
    panel reads was the wrong one. ``run_settings`` owns no schedule, so the
    console (which serves both) has to tell it what the declaration says.
    """

    def test_state_reports_a_declared_cadence_it_is_told_about(self):
        payload = run_settings.state(every_declared=45)
        self.assertEqual(payload["every_min"], 45.0)
        self.assertEqual(payload["every_min_source"], "schedule")
        self.assertEqual(payload["declared_every_min"], 45.0)

    def test_without_a_declaration_the_panel_says_unset(self):
        payload = run_settings.state()
        self.assertIsNone(payload["every_min"])
        self.assertEqual(payload["every_min_source"], "none")

    def test_an_override_beats_the_declaration_and_both_are_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                run_settings.save(every_min=60)
                payload = run_settings.state(every_declared=45)
        self.assertEqual(payload["every_min"], 60.0)
        self.assertEqual(payload["every_min_source"], "console")
        self.assertEqual(payload["declared_every_min"], 45.0,
                         "the declaration must stay visible next to an "
                         "override, or recreating the sidecar looks like it "
                         "silently changed the cadence")

    def test_a_non_numeric_declaration_is_reported_as_unset_not_crashed(self):
        payload = run_settings.state(every_declared="every 45 min")
        self.assertIsNone(payload["declared_every_min"])
        self.assertIsNone(payload["error"])


class ConsoleSuppliesTheDeclarationTest(unittest.TestCase):
    """The console wiring itself -- the part that was missing."""

    def _status(self, jobs):
        return {"now": "x", "jobs": jobs, "paused": False}

    def test_the_settings_endpoint_reads_the_schedulers_declared_cadence(self):
        import webui
        with mock.patch("scheduler.status", return_value=self._status(
                [{"id": "j", "every_min": 45.0, "every_min_declared": 45,
                  "enabled": True}])):
            payload = webui._settings_state()
        self.assertEqual(payload["every_min"], 45.0)
        self.assertEqual(payload["every_min_source"], "schedule")
        self.assertEqual(payload["declared_every_min"], 45.0)
        self.assertIsNone(payload.get("note"))

    def test_several_jobs_say_so_instead_of_showing_one_number(self):
        """An override is GLOBAL; one job's declaration must not stand in for
        all of them."""
        import webui
        with mock.patch("scheduler.status", return_value=self._status(
                [{"id": "a", "every_min": 30, "every_min_declared": 30},
                 {"id": "b", "every_min": 60, "every_min_declared": 60}])):
            payload = webui._settings_state()
        self.assertIn("ALL 2 jobs", payload.get("note") or "")
        self.assertIsNone(payload["declared_every_min"],
                          "picking one of two differing declarations to show "
                          "as 'the' cadence misdescribes what the override does")

    def test_an_unreadable_schedule_is_reported_not_fatal(self):
        """The override and the environment value must still resolve."""
        import webui
        with mock.patch("scheduler.status", side_effect=RuntimeError("no file")):
            payload = webui._settings_state()
        self.assertIn("schedule unreadable", payload.get("note") or "")
        self.assertIsNotNone(payload["max_steps"])
        self.assertIsNone(payload["error"])

    def test_the_panel_renders_the_declared_cadence_from_the_endpoint(self):
        """Structural: the page the operator reads must show it."""
        import webui
        page = webui.RUNS_PAGE if hasattr(webui, "RUNS_PAGE") else ""
        source = page or webui.__dict__.get("_RUNS_HTML", "")
        if not source:
            # The page template lives in a module-level string whose name has
            # changed before; find it rather than pinning the name.
            source = "\n".join(v for v in webui.__dict__.values()
                               if isinstance(v, str) and "loadRunSettings" in v)
        self.assertIn("loadRunSettings", source)


class ConsoleRoutesTest(unittest.TestCase):
    """The panel's endpoints, exercised through the handler."""

    def _get(self, route):
        import urllib.request
        return urllib.request

    def test_state_payload_carries_both_the_live_and_declared_values(self):
        """The page must be able to say "150 in force, the environment ships
        200" -- without the declared value a redeploy looks like a bug."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                run_settings.save(max_steps=150, every_min=45)
                payload = run_settings.state()
        self.assertEqual(payload["max_steps"], 150)
        self.assertEqual(payload["max_steps_source"], "console")
        self.assertEqual(payload["every_min"], 45)
        self.assertEqual(payload["declared_max_steps"], config.MAX_STEPS)
        self.assertEqual(payload["declared_answer_warn_steps"],
                         config.ANSWER_WARN_STEPS)
        self.assertEqual(payload["answer_warn_steps"],
                         run_settings.effective_answer_warn_steps(150))
        self.assertIsNone(payload["error"])

    def test_state_reports_the_accepted_ranges_so_the_inputs_can_constrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run_settings, "SETTINGS_FILE",
                                   Path(tmp) / "s.json"):
                payload = run_settings.state()
        self.assertEqual(payload["limits"]["max_steps"],
                         [run_settings.MAX_STEPS_MIN, run_settings.MAX_STEPS_MAX])
        self.assertEqual(payload["limits"]["every_min"],
                         [run_settings.EVERY_MIN_MIN, run_settings.EVERY_MIN_MAX])


if __name__ == "__main__":
    unittest.main()
