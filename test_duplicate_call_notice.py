"""An exact-repeat tool call must be named to the agent making it.

The harness fingerprinted every tool call's arguments from the day ``args_hash``
was added -- and then only ever WROTE the fingerprint down. Its single consumer
was a later audit pass counting waste after the fact, which is how waste stayed a
number in a report instead of a thing that gets fixed.

Measured live on 2026-11-04 across five consecutive missions:

    04:39  in flight, turn 49   -- 0 repeats of 75 calls
    03:15  ok, 192 calls        -- 7 repeats, `deploy_status` x6
    02:06  ok, 197 calls        -- 2 repeats, `deploy_status` x3
    01:06  ok, 181 calls        -- 5 repeats, `deploy_status` x6
    00:33  ok,  80 calls        -- 1 repeat,  `deploy_status` x2

One argument-less status call is repeated in EVERY run, and the 03:15 pass also
re-ran a 60-second dashboard suite three times -- inside a mission whose own text
carries an explicit "RUN IT ONCE PER PASS" rule naming that identical mistake from
2026-11-02. A rule written in prose did not stop it, so the fix is not more prose:
the notice arrives in the tool result the model is already reading, at the moment
it is deciding what to do next.

INFORMS, NEVER BLOCKS. Repeating a call is also how you legitimately poll an
asynchronous result -- a deploy receipt the host applier has not drained yet.
Only the model knows which case it is in, so the notice states the count, says it
was not blocked, and offers both readings rather than assuming the wasteful one.

These tests drive a REAL SessionLog in a throwaway directory rather than a stub.
A stub that only records what the test expects would happily pass against a notice
written to the wrong channel, or against one that corrupts the ``ok`` flag the
mission brief mandates reading failures from.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import config  # noqa: E402
import session  # noqa: E402
import tools  # noqa: E402
import tools_media  # noqa: E402


class CallHandler:
    """A tool handler with a canned result, counting its invocations."""

    def __init__(self, result="the result"):
        self.result = result
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.result


def make_agent(test):
    """A bare Agent writing to a REAL session log in a throwaway directory.

    Built through ``__new__`` because ``Agent(...)`` resolves a provider, and a
    test about repeated tool calls must not depend on a key existing on whatever
    machine runs the suite. Everything else -- session, recorder, tool funnel --
    is the production code.
    """
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    inst = agent.Agent.__new__(agent.Agent)
    inst.session = session.SessionLog(task="duplicate-call probe",
                                      sessions_dir=Path(tmp.name))

    def _close():
        #: finish() owns closing the handle; without a cleanup every test here
        #: leaks an open file and the suite fills with ResourceWarnings that a
        #: reader could mistake for a failure.
        try:
            handle = getattr(inst.session, "_jsonl", None)
            if handle is not None and not handle.closed:
                inst.session.finish("ok")
        except Exception:  # noqa: BLE001
            pass

    test.addCleanup(_close)
    inst.steps = 1
    inst.logged = []
    inst._log = lambda record: inst.logged.append(record)
    inst._say = lambda *a, **k: None
    inst._call_fingerprints = {}
    inst._route_timeouts = {}
    return inst


def with_tool(test, name="probe_tool", result="the result"):
    """Register a fake tool for the duration of one test."""
    handler = CallHandler(result)
    patcher = mock.patch.dict(tools.HANDLERS, {name: handler}, clear=False)
    patcher.start()
    test.addCleanup(patcher.stop)
    return handler


def records_of(inst):
    """Every record of the run's session file, in order."""
    return [json.loads(line) for line in
            inst.session.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class FirstCallIsSilentTest(unittest.TestCase):

    def setUp(self):
        self.inst = make_agent(self)
        self.handler = with_tool(self, "probe_tool", "first answer")

    def test_a_first_call_gets_no_notice_and_its_result_untouched(self):
        out, content = self.inst._run_tool("probe_tool", {"q": 1})
        self.assertEqual(out, "first answer")
        self.assertEqual(content, "first answer")
        self.assertEqual([r.get("event") for r in self.inst.logged], ["tool"])
        self.assertEqual(
            self.inst._call_fingerprints[
                session.fingerprint("probe_tool", {"q": 1})], 1)

    def test_the_call_was_still_recorded_normally(self):
        self.inst._run_tool("probe_tool", {"q": 1})
        acts = [r for r in records_of(self.inst) if r.get("event") == "action"]
        self.assertEqual(len(acts), 1)
        self.assertTrue(acts[0]["ok"])
        self.assertNotIn("duplicate_call",
                         [r["event"] for r in records_of(self.inst)])


class RepeatIsNamedTest(unittest.TestCase):

    def setUp(self):
        self.inst = make_agent(self)
        with_tool(self, "probe_tool", "the answer")

    def test_the_second_identical_call_says_so_with_the_count(self):
        self.inst._run_tool("probe_tool", {"q": 1})
        out, _content = self.inst._run_tool("probe_tool", {"q": 1})
        self.assertIn("[duplicate call]", out)
        self.assertIn("probe_tool", out)
        self.assertIn("2 times", out)

    def test_the_count_keeps_escalating_so_a_loop_is_visible(self):
        """A notice that always said "twice" would hide a runaway."""
        outs = [self.inst._run_tool("probe_tool", {})[0] for _ in range(5)]
        self.assertNotIn("[duplicate call]", outs[0])
        for index, expected in ((1, 2), (2, 3), (3, 4), (4, 5)):
            self.assertIn("%d times" % expected, outs[index],
                          "call %d reported the wrong repeat count"
                          % (index + 1))

    def test_the_original_result_is_kept_and_the_notice_appended(self):
        """A notice must never replace the tool's own answer.

        The model still needs the payload; the notice is commentary beside it,
        and a notice that displaced the result would be worse than the waste it
        prevents.
        """
        with_tool(self, "other", "PAYLOAD-123")
        self.inst._run_tool("other", {})
        out, _ = self.inst._run_tool("other", {})
        self.assertTrue(out.startswith("PAYLOAD-123"), out[:50])
        self.assertIn("[duplicate call]", out)

    def test_the_notice_reaches_the_model_content_not_only_the_log(self):
        """The model reads ``content``, not the recorded text.

        Writing the notice only into the transcript would keep the audit trail
        intact and still leave the model blind -- the mistake the usage record
        made when it first went to the transcript alone.
        """
        _t1, content1 = self.inst._run_tool("probe_tool", {})
        self.assertEqual(content1, "the answer")
        self.assertNotIn("[duplicate call]", str(content1))
        _t2, content2 = self.inst._run_tool("probe_tool", {})
        self.assertIn("[duplicate call]", str(content2))

    def test_it_says_the_call_was_not_blocked(self):
        """The wording must not read as a refusal, or the model backs off.

        It also has to offer the legitimate reading -- polling an asynchronous
        result -- because the most-repeated call on this deployment IS a status
        poll waiting on a host-side applier the agent cannot see.
        """
        self.inst._run_tool("probe_tool", {})
        out, _ = self.inst._run_tool("probe_tool", {})
        self.assertIn("not blocked", out)
        self.assertIn("polling", out)


class DistinctCallsAreNotRepeatsTest(unittest.TestCase):
    """Only an EXACT repeat counts; near-misses are different work."""

    def setUp(self):
        self.inst = make_agent(self)
        with_tool(self, "probe_tool")

    def test_different_arguments_are_not_a_duplicate(self):
        self.inst._run_tool("probe_tool", {"cmd": "suite A"})
        out, _ = self.inst._run_tool("probe_tool", {"cmd": "suite B"})
        self.assertNotIn("[duplicate call]", out)

    def test_a_long_argument_differing_past_90_chars_is_still_distinct(self):
        """Why this hashes instead of comparing the rendered ``detail``.

        ``detail`` truncates each argument, so two commands differing only past
        the cut look identical -- the misreading that produced a false "one
        command repeated 19-25 times" report on 2026-11-02. Nagging a model that
        is correctly re-running a suite with different flags would train it to
        ignore the notice entirely.
        """
        head = "x" * 200
        self.inst._run_tool("probe_tool", {"cmd": head + " --one"})
        out, _ = self.inst._run_tool("probe_tool", {"cmd": head + " --two"})
        self.assertNotIn("[duplicate call]", out)

    def test_a_different_tool_with_identical_arguments_is_not_a_duplicate(self):
        with_tool(self, "second_tool")
        self.inst._run_tool("probe_tool", {"a": 1})
        out, _ = self.inst._run_tool("second_tool", {"a": 1})
        self.assertNotIn("[duplicate call]", out)

    def test_argument_key_order_does_not_create_or_hide_a_duplicate(self):
        """Arguments are a mapping, so the fingerprint is order-independent."""
        self.inst._run_tool("probe_tool", {"a": 1, "b": 2})
        out, _ = self.inst._run_tool("probe_tool", {"b": 2, "a": 1})
        self.assertIn("[duplicate call]", out,
                      "reordering keys must not reset the count")


class DisabledTest(unittest.TestCase):

    def test_the_notice_can_be_turned_off_and_then_says_nothing(self):
        inst = make_agent(self)
        with_tool(self, "probe_tool")
        with mock.patch.object(config, "DUPLICATE_CALL_NOTICE", 0):
            inst._run_tool("probe_tool", {})
            out, content = inst._run_tool("probe_tool", {})
        self.assertNotIn("[duplicate call]", out)
        self.assertNotIn("[duplicate call]", str(content))
        self.assertTrue(out.startswith("the result"))

    def test_the_default_is_on(self):
        self.assertEqual(config.DUPLICATE_CALL_NOTICE, 1)


class AuditabilityTest(unittest.TestCase):
    """A warning nobody can see afterwards cannot be audited or improved."""

    def setUp(self):
        self.inst = make_agent(self)
        with_tool(self, "probe_tool")

    def test_the_session_records_that_the_agent_was_told(self):
        self.inst._run_tool("probe_tool", {})
        self.inst._run_tool("probe_tool", {})
        hits = [r for r in records_of(self.inst)
                if r.get("event") == "duplicate_call"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["tool"], "probe_tool")
        self.assertEqual(hits[0]["repeats"], 2)
        self.assertEqual(len(hits[0]["args_hash"]), session.ARGS_HASH_CHARS)

    def test_the_event_carries_the_same_hash_the_action_rows_do(self):
        """One identity, two renderings; if they disagree neither is usable."""
        self.inst._run_tool("probe_tool", {"a": 1})
        self.inst._run_tool("probe_tool", {"a": 1})
        recs = records_of(self.inst)
        acts = [r for r in recs if r.get("event") == "action"]
        dup = next(r for r in recs if r.get("event") == "duplicate_call")
        self.assertEqual({a["args_hash"] for a in acts}, {dup["args_hash"]})

    def test_the_transcript_log_gets_the_same_event(self):
        self.inst._run_tool("probe_tool", {})
        self.inst._run_tool("probe_tool", {})
        self.assertIn("duplicate_call",
                      [r.get("event") for r in self.inst.logged])

    def test_a_notice_cannot_make_a_successful_call_look_failed(self):
        """``action()`` derives ``ok`` by scanning the result body.

        The mission brief mandates reading failure from ``ok`` and never from text,
        so a notice containing a phrase the detector looks for would flip a
        passing call to a failure -- silently inflating every failure count a
        later pass computes from this run.
        """
        self.inst._run_tool("probe_tool", {})
        out, _ = self.inst._run_tool("probe_tool", {})
        self.assertIn("[duplicate call]", out)
        self.assertFalse(session._failed(out),
                         "the notice text trips the failure detector")
        self.assertEqual([a["ok"] for a in self.inst.session._actions],
                         [True, True])


class MultimodalResultsSurviveTest(unittest.TestCase):
    """The notice must not degrade an image result."""

    def setUp(self):
        self.inst = make_agent(self)
        self.envelope = tools_media.encode_image_bytes(
            b"\x89PNG\r\n\x1a\n....", "image/png", path="x.png")
        with_tool(self, "shot", self.envelope)

    def test_a_first_image_arrives_as_text_plus_image(self):
        _r, content = self.inst._run_tool("shot", {"n": 1})
        self.assertIsInstance(content, list)
        self.assertEqual([p["type"] for p in content], ["text", "image_url"])
        self.assertNotIn("[duplicate call]", str(content))

    def test_a_repeated_image_keeps_its_pixels(self):
        """The envelope is parsed BEFORE the notice is appended, on purpose.

        Appending first would defeat ``parse_image_envelope``, which matches an
        exact prefix -- so every screenshot after the first would silently
        degrade to text and the model would lose the pixels it needs to judge its
        own UI work. The ordering is the whole fix.
        """
        self.inst._run_tool("shot", {"n": 1})
        _r2, content = self.inst._run_tool("shot", {"n": 1})
        self.assertIsInstance(content, list)
        images = [p for p in content if p.get("type") == "image_url"]
        self.assertEqual(len(images), 1,
                         "the image part was lost when the notice was added")
        self.assertTrue(images[0]["image_url"]["url"].startswith(
            "data:image/png"))
        notices = [p for p in content if p.get("type") == "text"
                   and "[duplicate call]" in p.get("text", "")]
        self.assertEqual(len(notices), 1,
                         "the notice must arrive as its own text part beside the "
                         "image, not by corrupting the envelope")


class PerRunIsolationTest(unittest.TestCase):

    def test_the_counter_is_reset_when_a_run_starts(self):
        """One Agent object can run more than one task.

        Same trap as ``_route_timeouts``: without the reset a second run inherits
        the first run's history and its very first tool call is announced as a
        duplicate. Asserted on source because driving two real runs needs a live
        model.
        """
        self.assertIn("self._call_fingerprints = {}",
                      inspect.getsource(agent.Agent._run_impl),
                      "the counter is not reset at run start")

    def test_the_attribute_is_declared_so_a_normal_agent_has_it(self):
        """Miss this and every tool call raises AttributeError in production.

        The reset inside ``_run_impl`` alone would not protect a path that runs a
        tool before that point, and a missing declaration is invisible to a suite
        that builds its Agent only through ``make_agent``.
        """
        self.assertIn("_call_fingerprints",
                      inspect.getsource(agent.Agent.__init__))

    def test_fingerprint_is_one_definition_used_by_both_sides(self):
        """The audit key and the live warning must be the same key."""
        self.assertEqual(session.fingerprint("t", {"a": 1}),
                         "t " + session.args_hash({"a": 1}))
        self.assertEqual(session.args_hash({"a": 1}),
                         session.args_hash({"a": 1}))
        self.assertIn("_args_hash(args)",
                      inspect.getsource(session.SessionLog.action))


if __name__ == "__main__":
    unittest.main(verbosity=2)
