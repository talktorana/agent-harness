"""The duplicate-call notice must name WHICH call repeated, not just the tool.

Measured live 2026-11-23, mission 16:30:56: 28 tool calls, 17 of them
``run_command``. The notice said "you have run `run_command` with these EXACT
arguments 3 times" -- true, and unactionable, because nothing in it identified
the command. That pass reported its own waste as "I ran the dashboard suite
twice with identical arguments". The measured repeat was a python heredoc
reading closed_positions.jsonl, THREE times, and the dashboard suite was never
repeated at all.

So the notice was correct and still produced a false report. The brief demands
a pass quantify and name its waste; a notice that withholds the call's identity
makes that impossible and invites a confident guess. These tests pin the fix:
the notice carries the same ``args_hash`` the audit groups on, plus the same
detail rendering the action row records -- one renderer, so the thing the model
was warned about and the thing a later pass counts cannot be described
differently.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import session  # noqa: E402
from test_duplicate_call_notice import make_agent, with_tool  # noqa: E402

HEREDOC = {"command": "cd /project/app && python3 - <<'PY'\nimport json\n"
                      "rows=[json.loads(l) for l in open('closed_positions.jsonl')]\nPY"}
SUITE = {"command": "cd dashboard && python3 -m unittest discover -p 'test_*.py'"}


class DetailRendererIsSharedTest(unittest.TestCase):
    """One renderer for the audit row and the runtime notice."""

    def test_the_action_row_and_the_public_renderer_agree(self):
        inst = make_agent(self)
        with_tool(self, "run_command", "exit=0")
        inst._run_tool("run_command", HEREDOC)
        rows = [r for r in inst.session._actions if r["tool"] == "run_command"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["detail"], session.detail_line(HEREDOC))

    def test_no_arguments_renders_as_something_readable(self):
        self.assertEqual(session.detail_line({}), "(no arguments)")
        self.assertEqual(session.detail_line(None), "(no arguments)")

    def test_the_renderer_truncates_for_reading_like_the_row_does(self):
        long = {"command": "x" * 500}
        self.assertLess(len(session.detail_line(long)), 200)
        self.assertIn("…", session.detail_line(long))


class NoticeNamesTheCallTest(unittest.TestCase):

    def setUp(self):
        self.inst = make_agent(self)
        self.handler = with_tool(self, "run_command", "exit=0\nrows 159")

    def notice_for(self, args, times=2):
        out = None
        for _ in range(times):
            out, _content = self.inst._run_tool("run_command", args)
        return out

    def test_the_notice_carries_the_hash_the_audit_groups_on(self):
        out = self.notice_for(HEREDOC)
        self.assertIn(session.args_hash(HEREDOC), out)

    def test_the_notice_carries_the_command_itself(self):
        out = self.notice_for(HEREDOC)
        self.assertIn("closed_positions.jsonl", out)

    def test_the_notice_does_not_name_a_call_that_did_not_repeat(self):
        # The real misreport: it blamed the dashboard suite. A notice about the
        # heredoc must not mention the suite, or it manufactures that guess.
        self.inst._run_tool("run_command", SUITE)
        out = self.notice_for(HEREDOC)
        self.assertIn("[duplicate call]", out)
        self.assertNotIn("dashboard", out)

    def test_two_commands_differing_past_the_detail_cut_are_still_told_apart(self):
        # detail truncates at 90 chars per argument; the hash does not. Both
        # commands share a long prefix, so only the hash can separate them.
        prefix = "cd /project/app && python3 -m unittest " + "z" * 120
        a = {"command": prefix + " engine.test_one"}
        b = {"command": prefix + " engine.test_two"}
        self.assertEqual(session.detail_line(a), session.detail_line(b))
        self.assertNotEqual(session.args_hash(a), session.args_hash(b))
        out = self.notice_for(a)
        self.assertIn(session.args_hash(a), out)
        self.assertNotIn(session.args_hash(b), out)

    def test_the_count_still_climbs_and_the_tool_is_still_named(self):
        out = self.notice_for(HEREDOC, times=3)
        self.assertIn("run_command", out)
        self.assertIn("3 times", out)

    def test_a_first_call_is_still_silent(self):
        out, _ = self.inst._run_tool("run_command", HEREDOC)
        self.assertNotIn("[duplicate call]", out)

    def test_it_informs_and_never_blocks(self):
        self.notice_for(HEREDOC, times=3)
        self.assertEqual(self.handler.calls, 3)

    def test_the_notice_says_which_call_to_report(self):
        out = self.notice_for(HEREDOC)
        self.assertIn("Name THIS call", out)

    def test_disabling_the_feature_still_silences_it(self):
        with mock.patch.object(config, "DUPLICATE_CALL_NOTICE", 0):
            out = self.notice_for(HEREDOC)
        self.assertNotIn("[duplicate call]", out)

    def test_the_logged_event_and_the_notice_carry_the_same_hash(self):
        self.notice_for(HEREDOC)
        events = [r for r in self.inst.logged if r.get("event") == "duplicate_call"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["args_hash"], session.args_hash(HEREDOC))


if __name__ == "__main__":
    unittest.main()
