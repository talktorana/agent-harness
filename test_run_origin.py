"""Who started a run, and what one model call actually cost.

Both of these came from real misreadings of this deployment's own history, and
both are the kind that make an audit report confidently wrong.

1. ORIGIN. Every entry point -- a timed job, a browser chat, a script posting to
   the console, the command line -- writes its session into the SAME directory
   and therefore into the SAME run list. Two console smoke tests ("reply with
   one word: ok", six seconds, zero tool calls) sat beside forty-minute mission
   runs and were indistinguishable from them, so a monitoring pass read them as
   missions that ran and did nothing -- a finding invented entirely out of a
   missing field. The tag now travels on the session record, reaches the run
   history JSON and gets its own column, so the question "did the SCHEDULED
   mission run?" has an answer that does not require reading every prompt.

2. USAGE. ``assemble_stream`` handed every usage-bearing chunk to its callback.
   One provider sends that block twice with identical counts 0.2 ms apart, so a
   single model call recorded two ``model_usage`` rows -- and any per-run token
   or cost total read exactly double. Collapsing repeats is now the stream
   folder's job, while a provider that streams CUMULATIVE counts still has each
   distinct value recorded.
"""
from __future__ import annotations

import json
import inspect
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import scheduler  # noqa: E402
import session  # noqa: E402
import webui  # noqa: E402
from test_webui_runs import TempDirs  # noqa: E402


def records(path) -> list:
    """Every record of one session file, in order."""
    return [json.loads(line) for line
            in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def first_of(path, event: str) -> dict:
    hits = [r for r in records(path) if r.get("event") == event]
    assert hits, f"no {event!r} record in {path}"
    return hits[0]


# --------------------------------------------------------------------- origin


class SessionStartOriginTest(TempDirs):
    """The tag is written once, at session start, and nowhere is invented."""

    def test_the_scheduler_entry_point_tags_its_session(self):
        log = session.SessionLog(task="the mission text",
                                 sessions_dir=self.sessions,
                                 origin="schedule")
        log.finish("ok")
        self.assertEqual(first_of(log.jsonl_path, "session_start")["origin"],
                         "schedule")

    def test_the_console_entry_point_tags_its_session(self):
        log = session.SessionLog(task="Reply with exactly one word: ok",
                                 sessions_dir=self.sessions, origin="chat")
        log.finish("ok")
        self.assertEqual(first_of(log.jsonl_path, "session_start")["origin"],
                         "chat")

    def test_the_command_line_entry_point_tags_its_session(self):
        log = session.SessionLog(task="a one-off question",
                                 sessions_dir=self.sessions, origin="cli")
        log.finish("ok")
        self.assertEqual(first_of(log.jsonl_path, "session_start")["origin"],
                         "cli")

    def test_an_untagged_caller_records_no_origin_key_at_all(self):
        """Back-compatibility is the POINT of writing the key conditionally.

        An empty tag written as ``origin: ""`` and a missing key read the same
        through ``.get()``, but a reader that inspects the raw record -- the
        markdown export, an older console build -- must see the shape it always
        saw. Inventing a value would be worse than admitting there is none.
        """
        log = session.SessionLog(task="legacy caller",
                                 sessions_dir=self.sessions)
        log.finish("ok")
        self.assertNotIn("origin", first_of(log.jsonl_path, "session_start"))

    def test_the_origin_still_reaches_the_record_when_the_task_is_empty(self):
        log = session.SessionLog(task="", sessions_dir=self.sessions,
                                 origin="schedule")
        log.finish("ok")
        self.assertEqual(first_of(log.jsonl_path, "session_start")["origin"],
                         "schedule")


class RunHistoryOriginTest(TempDirs):
    """The tag has to survive from the file into what a human or agent reads."""

    def test_the_two_entry_points_are_two_different_kinds_of_row(self):
        mission = session.SessionLog(task="monitor the deployment",
                                     sessions_dir=self.sessions,
                                     origin="schedule")
        mission.finish("ok")
        probe = session.SessionLog(task="Reply with exactly one word: ok",
                                   sessions_dir=self.sessions, origin="chat")
        probe.finish("ok")

        rows = {r["prompt"]: r["origin"] for r in webui.run_summaries()}
        self.assertEqual(rows["monitor the deployment"], "schedule")
        self.assertEqual(rows["Reply with exactly one word: ok"], "chat")

    def test_a_session_written_before_the_field_reports_unknown(self):
        """Old files are labelled unknown, NOT silently promoted to schedule.

        Defaulting them would plant fake mission runs in the very history that
        a later automated pass audits, which is the error this tag exists to
        prevent -- just pointed the other way.
        """
        self.write_session(prompt="a run from an older build")
        row = next(r for r in webui.run_summaries()
                   if r["prompt"] == "a run from an older build")
        self.assertEqual(row["origin"], "unknown")

    def test_the_shape_that_once_read_as_a_mission_with_no_work(self):
        """status ok + zero tool calls, from the console rather than the clock.

        These exact rows are what a monitoring pass mistook for two failed
        deliveries. The data must stay honest on both axes: the run really did
        succeed, and it really was not a mission.
        """
        probe = session.SessionLog(task="Reply with exactly one word: ok",
                                   sessions_dir=self.sessions, origin="chat")
        probe.finish("ok")
        row = webui.run_summaries()[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["steps"], 0)
        self.assertEqual(row["origin"], "chat")


class RunsPageOriginColumnTest(TempDirs):
    """A field in the JSON that no page shows is a field nobody reads."""

    def test_the_header_has_one_cell_per_body_cell(self):
        head = (webui.RUNS_PAGE
                .split("<table><thead><tr>", 1)[1]
                .split("</tr></thead>", 1)[0])
        self.assertIn("origin", head)
        #: started, status, origin, steps, took, prompt, answer. If this count
        #: and the colspan below ever disagree again the empty-state row stops
        #: spanning the table -- the visible symptom that catches it.
        self.assertEqual(head.count("<th"), 7, f"header was: {head!r}")
        self.assertIn('colspan="7"', webui.RUNS_PAGE)

    def test_a_row_renders_its_origin(self):
        self.assertIn("r.origin", webui.RUNS_PAGE)

    def test_a_scheduled_run_that_made_no_tool_call_is_said_so(self):
        """The marker only fires for `schedule`, and only at zero.

        A chat question legitimately needs no tools; flagging it would train an
        operator to ignore a marker that matters -- the same argument the status
        colours already make about not colouring a partial success red.
        """
        page = webui.RUNS_PAGE
        self.assertIn("no work", page)
        self.assertIn("r.origin==='schedule'&&r.steps===0", page)

    def test_the_non_schedule_rows_carry_a_title_saying_why(self):
        self.assertIn('not started by the ', webui.RUNS_PAGE)


class EntryPointTaggingTest(TempDirs):
    """Each entry point declares itself, rather than being guessed at later."""

    def test_the_scheduler_launches_with_origin_schedule(self):
        with mock.patch.object(agent, "run_task",
                               return_value="the report") as launched:
            scheduler.run_due_jobs()
        self.assertTrue(launched.called, "the due job never ran")
        self.assertEqual(launched.call_args.kwargs.get("origin"), "schedule")

    def test_the_console_chat_launches_with_origin_chat(self):
        """The path the smoke tests took: POST /chat -> _start_task -> Agent.

        The worker runs on its own thread, so the record is polled until the
        task leaves `running`; asserting immediately would race and pass for
        the wrong reason most of the time.
        """
        captured: dict = {}

        def fake_agent(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                run=lambda text, images=None: "ok", stop_requested=False)

        with mock.patch.object(agent, "Agent", side_effect=fake_agent):
            task_id = webui._start_task("Reply with exactly one word: ok")
            for _ in range(200):
                if webui._TASKS[task_id]["status"] != "running":
                    break
                time.sleep(0.05)

        self.assertEqual(webui._TASKS[task_id]["status"], "done")
        self.assertEqual(captured.get("origin"), "chat")

    def test_run_task_forwards_the_origin_to_the_agent(self):
        with mock.patch.object(agent, "Agent") as ctor:
            inst = ctor.return_value
            inst.run.return_value = "the answer"
            agent.run_task("do a thing", origin="schedule")
        self.assertEqual(ctor.call_args.kwargs.get("origin"), "schedule")

    def test_the_command_line_declares_itself(self):
        source = Path(__file__).with_name("run.py").read_text(encoding="utf-8")
        self.assertIn('origin="cli"', source)

    def test_a_run_forwards_its_origin_into_the_session_it_opens(self):
        """The tag is useless if the agent drops it between init and session.

        Asserted on the source of ``_run_impl`` -- the same convention
        ``test_llm_config`` uses for configuration wiring -- because invoking a
        real run would need a live model to prove one keyword argument arrived.
        """
        source = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("SessionLog(", source)
        self.assertIn("origin=self.origin", source)
        self.assertIn("self.origin = origin or \"\"",
                      inspect.getsource(agent.Agent.__init__))


# ---------------------------------------------------------------------- usage


def usage(prompt=1, completion=3, reasoning=None):
    """A provider usage block. Reasoning detail is optional, as in the wild."""
    details = (SimpleNamespace(reasoning_tokens=reasoning)
               if reasoning is not None else None)
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           completion_tokens_details=details)


def chunk(content=None, usage_block=None, finish=None, tool_calls=None):
    """One SSE chunk. A usage-only chunk has EMPTY choices, as providers do."""
    choices = []
    if content is not None or tool_calls is not None or finish is not None:
        choices = [SimpleNamespace(
            finish_reason=finish,
            delta=SimpleNamespace(content=content, tool_calls=tool_calls,
                                  reasoning_content=None))]
    return SimpleNamespace(choices=choices, usage=usage_block)


class UsageRecordedOnceTest(unittest.TestCase):
    """One model call is one measurement, however many chunks carry it."""

    def test_a_repeated_usage_block_is_recorded_once(self):
        seen: list = []
        block = usage(18883, 3)
        msg = agent.assemble_stream(
            [chunk(content="ok"), chunk(usage_block=block),
             chunk(usage_block=block)],
            on_usage=seen.append)
        self.assertEqual(len(seen), 1, "one call recorded twice: every "
                                       "per-run token total would read double")
        self.assertEqual(seen[0].prompt_tokens, 18883)
        self.assertEqual(msg.content, "ok")

    def test_cumulative_counts_still_record_every_distinct_value(self):
        """The dedupe must not degrade to "first wins".

        A provider that streams running totals sends genuinely different
        measurements; discarding all but the first would under-report, and the
        last value is the total the run actually spent.
        """
        seen: list = []
        agent.assemble_stream(
            [chunk(content="a"), chunk(usage_block=usage(10, 1)),
             chunk(usage_block=usage(10, 4)), chunk(usage_block=usage(10, 4))],
            on_usage=seen.append)
        self.assertEqual([u.completion_tokens for u in seen], [1, 4])

    def test_reasoning_tokens_participate_in_the_signature(self):
        """A turn whose reasoning arrived late is a different measurement.

        ``reasoning_effort`` is audited from exactly this field, so collapsing
        a later reasoning count into an earlier zero would erase the evidence
        that the depth setting reached the wire.
        """
        seen: list = []
        agent.assemble_stream(
            [chunk(usage_block=usage(1, 2, reasoning=0)),
             chunk(usage_block=usage(1, 2, reasoning=512))],
            on_usage=seen.append)
        self.assertEqual(len(seen), 2)
        self.assertEqual(
            seen[-1].completion_tokens_details.reasoning_tokens, 512)

    def test_a_stream_without_usage_records_nothing(self):
        seen: list = []
        msg = agent.assemble_stream([chunk(content="hi")],
                                    on_usage=seen.append)
        self.assertEqual(seen, [])
        self.assertEqual(msg.content, "hi")

    def test_an_exploding_callback_cannot_break_the_fold(self):
        """Recording cost must never be the reason an answer is lost."""
        def boom(_u):
            raise RuntimeError("callback died")

        msg = agent.assemble_stream(
            [chunk(content="hi"), chunk(usage_block=usage())], on_usage=boom)
        self.assertEqual(msg.content, "hi")

    def test_tool_calls_and_finish_reason_survive_the_dedupe(self):
        frag_a = SimpleNamespace(
            index=0, id="call_1",
            function=SimpleNamespace(name="run_command", arguments='{"cmd":'))
        frag_b = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments='"ls"}'))
        seen: list = []
        block = usage(7, 9)
        msg = agent.assemble_stream(
            [chunk(tool_calls=[frag_a]), chunk(tool_calls=[frag_b]),
             chunk(finish="tool_calls"), chunk(usage_block=block),
             chunk(usage_block=block)],
            on_usage=seen.append)
        self.assertEqual(len(seen), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "run_command")
        self.assertEqual(msg.tool_calls[0].function.arguments, '{"cmd":"ls"}')
        self.assertEqual(msg.finish_reason, "tool_calls")

    def test_reasoning_only_chunks_do_not_produce_a_phantom_usage(self):
        seen: list = []
        msg = agent.assemble_stream(
            [SimpleNamespace(choices=[SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content=None, tool_calls=None,
                                      reasoning_content="x" * 500))],
                usage=None)],
            on_usage=seen.append)
        self.assertEqual(seen, [])
        self.assertEqual(msg.reasoning_chars, 500)
        self.assertIsNone(msg.content)


if __name__ == "__main__":
    unittest.main()
