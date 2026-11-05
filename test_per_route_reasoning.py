"""Per-route reasoning effort: an explicitly selected model can be max.

The new cbcn/kimi-k3.1-pro option is a route, not a global behavior change.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent
import llm


class PerRouteReasoningTest(unittest.TestCase):
    def test_only_the_named_route_gets_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text(json.dumps({"routes": [
                {"provider": "inferhub", "model": "other"},
                {"provider": "inferhub", "model": "kimi",
                 "reasoning_effort": "max"},
            ]}), encoding="utf-8")
            with mock.patch.object(llm, "ROUTES_FILE", path), \
                    mock.patch("os.environ.get", side_effect=lambda k, d="": d):
                efforts = llm.route_reasoning_efforts()
        self.assertEqual(efforts[("inferhub", "kimi")], "max")
        self.assertNotIn(("inferhub", "other"), efforts)

    def test_selected_route_uses_max_and_other_route_keeps_low(self):
        subject = agent.Agent.__new__(agent.Agent)
        subject._route_reasoning_efforts = {("inferhub", "kimi"): "max"}
        subject._deep_reasoning = False
        with mock.patch.object(agent.config, "REASONING_EFFORT", "low"):
            for provider, alias, expected in (
                    ("inferhub", "kimi", "max"),
                    ("inferhub", "other", "low")):
                route_effort = subject._route_reasoning_efforts.get(
                    (provider, alias), "")
                effort = subject._next_reasoning_effort()
                if route_effort and not subject._deep_reasoning:
                    effort = route_effort
                self.assertEqual(effort, expected)

    def test_deep_escalation_still_wins_over_a_low_route_override(self):
        subject = agent.Agent.__new__(agent.Agent)
        subject._route_reasoning_efforts = {("inferhub", "other"): "low"}
        subject._deep_reasoning = True
        with mock.patch.object(agent.config, "REASONING_EFFORT", "low"), \
                mock.patch.object(agent.config, "DEEP_REASONING_EFFORT", "max"):
            route_effort = subject._route_reasoning_efforts.get(
                ("inferhub", "other"), "")
            effort = subject._next_reasoning_effort()
            if route_effort and not subject._deep_reasoning:
                effort = route_effort
        self.assertEqual(effort, "max")


if __name__ == "__main__":
    unittest.main()
