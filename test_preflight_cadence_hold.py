"""An unrestored console cadence override must never read as a healthy schedule.

The incident this pins, measured live 2026-11-23: a scheduled pass POSTed
``every_min=4500`` against a declared 45 at 2026-11-21T02:08:45 -- one minute
into its own run -- and finished at 02:09:50 reporting SUCCESS without restoring
it. The operator hold marker was later removed, so that leftover override became
the only reason the mission was stopped: 62 hours of silence, ~83 missed fires.

``overdue`` stayed False the whole time, because it compares the gap against the
override (75h), not against the cadence anyone actually chose (45min). Raising
the interval moves the goalposts, so the one field meant to catch lateness is
structurally blind to this. These tests pin the comparison that CAN see it.
"""
from datetime import datetime, timezone
import unittest
from unittest import mock

import config
import preflight

DECLARED = 45.0
LEFTOVER = 4500


def settings(**over):
    base = {"every_min": LEFTOVER, "every_min_source": "console",
            "declared_every_min": DECLARED, "saved_at": "2026-11-21T02:08:45",
            "max_steps": 600, "max_steps_source": "console"}
    base.update(over)
    return base


def schedule(held=False, **over):
    base = {"hold": {"configured": True, "held": held, "path": "/x/agent_paused.json"},
            "jobs": [{"id": "mission", "every_min": LEFTOVER}]}
    base.update(over)
    return base


class CadenceOverrideIsVisibleTest(unittest.TestCase):
    def test_the_real_incident_shape_is_named(self):
        items = preflight.cadence_override_attention(settings(), schedule())
        self.assertEqual(len(items), 1, items)
        text = items[0]
        # Both numbers, the ratio and the restore target: a reader must be able
        # to act without another turn of investigation.
        for token in ("4500", "45", "100x", "2026-11-21T02:08:45"):
            self.assertIn(token, text)
        self.assertIn("unrestored", text)

    def test_a_declared_cadence_is_owner_intent_not_a_finding(self):
        # Same enormous interval, but DECLARED: nobody overrode anything.
        self.assertEqual(preflight.cadence_override_attention(
            settings(every_min_source="declared"), schedule()), [])

    def test_an_operator_hold_explains_the_pause_and_is_not_a_defect(self):
        self.assertEqual(preflight.cadence_override_attention(
            settings(), schedule(held=True)), [])

    def test_a_modest_override_is_tuning_not_a_hold(self):
        self.assertEqual(preflight.cadence_override_attention(
            settings(every_min=60), schedule()), [])

    def test_a_faster_override_is_never_a_hold(self):
        self.assertEqual(preflight.cadence_override_attention(
            settings(every_min=30), schedule()), [])

    def test_the_threshold_is_inclusive(self):
        at = preflight.cadence_override_attention(
            settings(every_min=DECLARED * preflight.CADENCE_HOLD_RATIO), schedule())
        self.assertEqual(len(at), 1, at)

    def test_just_below_the_threshold_is_quiet(self):
        self.assertEqual(preflight.cadence_override_attention(
            settings(every_min=DECLARED * preflight.CADENCE_HOLD_RATIO - 1),
            schedule()), [])

    def test_an_unreadable_declared_cadence_says_so_instead_of_going_quiet(self):
        for bad in (None, 0, -5, "abc"):
            items = preflight.cadence_override_attention(
                settings(declared_every_min=bad), schedule())
            self.assertEqual(len(items), 1, (bad, items))
            self.assertIn("not detectable", items[0])

    def test_a_missing_hold_block_does_not_hide_the_override(self):
        # The marker file was ABSENT in the incident; absence must not read as
        # "an operator paused it".
        self.assertEqual(len(preflight.cadence_override_attention(
            settings(), {"jobs": []})), 1)
        self.assertEqual(len(preflight.cadence_override_attention(
            settings(), None)), 1)

    def test_absent_or_malformed_input_never_crashes_or_invents(self):
        self.assertEqual(preflight.cadence_override_attention(None, schedule()), [])
        self.assertEqual(preflight.cadence_override_attention({}, schedule()), [])
        self.assertEqual(preflight.cadence_override_attention(
            settings(every_min=None), schedule()), [])


class SnapshotCarriesTheFindingTest(unittest.TestCase):
    """The finding must reach the mission at orientation, at zero turn cost."""

    def setUp(self):
        self.patches = [
            mock.patch.object(config, "PREFLIGHT_CONSOLE_URL", "http://console.test"),
            mock.patch.object(config, "PREFLIGHT_ANSWER_CHARS", 6000),
        ]
        for patch in self.patches:
            patch.start()
        self.now = datetime(2026, 11, 23, 16, 30, tzinfo=timezone.utc)

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    @staticmethod
    def healthy():
        return {"fast_path_eligible": True, "attention_required": []}, []

    def fetch(self, url):
        if url.endswith("/agent/schedule"):
            return schedule()
        if url.endswith("/agent/settings"):
            return settings()
        if url.endswith("/agent/runs.json"):
            return {"runs": [{"origin": "schedule", "started": "2026-11-21T02:07:12",
                              "status": "ok", "session_id": "s1", "answer": "SUCCESS"}]}
        raise AssertionError("unexpected URL: " + url)

    def snapshot(self):
        return preflight.build_snapshot(now=self.now, fetch=self.fetch,
                                        collector=self.healthy)

    def test_the_held_mission_is_not_fast_path_eligible(self):
        snap = self.snapshot()
        self.assertFalse(snap["fast_path_eligible"],
                         "a mission stopped by a leftover override must not read healthy")
        self.assertTrue(any("unrestored" in a for a in snap["attention_required"]),
                        snap["attention_required"])

    def test_the_projection_publishes_what_the_comparison_needs(self):
        got = self.snapshot()["settings"]
        self.assertEqual(got.get("declared_every_min"), DECLARED)
        self.assertEqual(got.get("saved_at"), "2026-11-21T02:08:45")

    def test_a_restored_cadence_is_healthy_again(self):
        def restored(url):
            if url.endswith("/agent/settings"):
                return settings(every_min=DECLARED)
            return self.fetch(url)

        snap = preflight.build_snapshot(now=self.now, fetch=restored,
                                        collector=self.healthy)
        self.assertTrue(snap["fast_path_eligible"], snap["attention_required"])


if __name__ == "__main__":
    unittest.main()
