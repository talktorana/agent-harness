"""``recreate_guard``: the gate that stops a recreate killing a run in flight.

WHY THIS SUITE EXISTS
---------------------
A recreate is how every agent fix is deployed, and it kills whatever run is in
progress. On one consumer deployment two recreates landed on two live passes
inside ten minutes (2026-11-03): both runs ended with no ``session_end``, at
turn 46 and turn 24 of a 60-turn budget -- killed, not cut off. The consumer's
mission text already described the check; the recreate was still performed by a
hand-written throwaway script with a bare sleep and no probe. Prose is not a
guard. This module is, and these tests pin the three decisions that make it
one.

  1. THE SCHEDULER IS AUTHORITATIVE. ``scheduler.running`` derives from the
     scheduler's own state (its ``finished`` stamp is only rewritten when a run
     RETURNS); the per-run rows are mtime-based and lie in both directions.
     The guard must WAIT on the scheduler even when the newest session file
     looks idle, and must not be argued out of it.
  2. A KILLED RUN MUST NOT DEADLOCK IT. Demanding that the newest session file
     contain ``session_end`` before proceeding never clears after a kill -- a
     killed run never gains one -- so an obedient operator waits out every
     future run too and the fix never ships. A missing ``session_end`` blocks
     ONLY when it belongs to the run the scheduler reports in flight.
  3. UNKNOWN IS NEVER IDLE. A scheduler that cannot answer must refuse, not
     proceed: an absent or unreadable field reads as "no run in flight" to any
     caller doing ``.get()``, which is how a real 2026-11-02 recreate killed a
     run five minutes in.

Plus the contract callers branch on (exit codes), the path mapping that stops
the session cross-check silently probing the wrong filesystem, and the proof
that a recreate actually REPLACED the container -- a compose run that changes
nothing reports success while the old code keeps serving.

Run:  python -m unittest test_recreate_guard
"""
from __future__ import annotations

import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

import recreate_guard as RG

GUARD = Path(__file__).resolve().parent / "recreate_guard.py"


def sched(running=False, available=True, in_progress_since=None, reason=None):
    return {"available": available, "running": running,
            "in_progress_since": in_progress_since, "reason": reason}


def session(has_end=True, started="2026-11-03T17:07:46", readable=True):
    return {"file": "x.jsonl", "started": started, "has_session_end": has_end,
            "readable": readable, "host_path": "/tmp/x.jsonl"}


class TheSchedulerIsAuthoritativeTest(unittest.TestCase):
    """A live run must be waited for, however the session file happens to look."""

    def test_a_run_in_flight_is_waited_for_even_when_the_session_file_ended(self):
        """The file can lag a live run; the scheduler's state cannot.

        A file that has ended while the scheduler still reports a run in flight
        is the moment a naive guard recreates and kills the NEXT pass. The
        scheduler's answer wins.
        """
        action, _ = RG.decide(sched(running=True, in_progress_since="2026-11-03T17:07:46"),
                              session(has_end=True))
        self.assertEqual(action, "wait")

    def test_a_run_in_flight_is_waited_for_when_the_file_has_no_end_either(self):
        action, reason = RG.decide(sched(running=True, in_progress_since="2026-11-03T17:07:46"),
                                   session(has_end=False))
        self.assertEqual(action, "wait")
        self.assertIn("in flight", reason)

    def test_idle_with_a_cleanly_ended_session_proceeds(self):
        action, _ = RG.decide(sched(), session(has_end=True))
        self.assertEqual(action, "proceed")


class AKilledRunMustNotDeadlockTheGuardTest(unittest.TestCase):
    """The case that makes 'require both signals to agree' unusable."""

    def test_a_killed_runs_missing_session_end_does_not_block_forever(self):
        """A killed run never gains a ``session_end``.

        Demanding one as a precondition means the guard waits out the next run,
        and the next, and the fix being deployed never ships -- so a missing
        ``session_end`` on a run the scheduler does NOT report in flight is a
        historical fact, not a blocker.
        """
        action, reason = RG.decide(sched(running=False, in_progress_since=None),
                                   session(has_end=False, started="2026-11-03T15:58:11"))
        self.assertEqual(action, "proceed")
        self.assertIn("never gains one", reason)

    def test_the_missing_end_is_still_reported_not_swallowed(self):
        """Proceeding must not be silent: the reason names the file."""
        _, reason = RG.decide(sched(), session(has_end=False, started="2026-11-03T15:58:11"))
        self.assertIn("x.jsonl", reason)

    def test_a_missing_end_on_the_run_the_scheduler_reports_waits(self):
        """If it IS the live run, the agreement check still holds."""
        action, _ = RG.decide(sched(running=False, in_progress_since="2026-11-03T17:07:46"),
                              session(has_end=False, started="2026-11-03T17:07:46"))
        self.assertEqual(action, "wait")


class UnknownIsNeverIdleTest(unittest.TestCase):

    def test_an_unavailable_scheduler_refuses_rather_than_proceeding(self):
        action, reason = RG.decide(sched(available=False, reason="scheduler.status() raised"),
                                   session(has_end=True))
        self.assertEqual(action, "refuse")
        self.assertIn("cannot answer", reason)

    def test_the_refusal_echoes_why_the_scheduler_could_not_answer(self):
        _, reason = RG.decide(sched(available=False, reason="cannot import the scheduler"),
                              session())
        self.assertIn("cannot import the scheduler", reason)

    def test_an_unreadable_session_file_does_not_block_a_confident_scheduler(self):
        action, reason = RG.decide(sched(), session(has_end=False, readable=False))
        self.assertEqual(action, "proceed")
        self.assertIn("could not be read", reason)

    def test_no_session_files_at_all_falls_back_to_the_schedulers_answer(self):
        action, _ = RG.decide(sched(), None)
        self.assertEqual(action, "proceed")


class MountSpecTest(unittest.TestCase):
    """A bad mapping must fail loudly, not silently probe the wrong filesystem."""

    def test_a_well_formed_spec_parses(self):
        self.assertEqual(RG.parse_mounts(["/agent-data=/srv/agent-data"]),
                         {"/agent-data": "/srv/agent-data"})

    def test_a_spec_without_an_equals_sign_raises(self):
        """Silently ignoring it would report every session missing."""
        with self.assertRaises(ValueError):
            RG.parse_mounts(["/agent-data:/srv/agent-data"])

    def test_a_spec_with_an_empty_side_raises(self):
        with self.assertRaises(ValueError):
            RG.parse_mounts(["=/srv/agent-data"])


class HostPathMappingTest(unittest.TestCase):
    """Console-reported paths are CONTAINER paths; probing them raw finds nothing."""

    MOUNTS = {"/agent-data": "/srv/host-data"}

    def test_a_mapped_prefix_swaps_onto_the_host_path(self):
        self.assertEqual(RG.host_path("/agent-data/sessions", self.MOUNTS),
                         "/srv/host-data/sessions")

    def test_the_longest_matching_prefix_wins(self):
        mounts = {"/agent-data": "/srv/a", "/agent-data/sessions": "/srv/b"}
        self.assertEqual(RG.host_path("/agent-data/sessions/x.jsonl", mounts),
                         "/srv/b/x.jsonl")

    def test_an_unmapped_path_is_left_alone_rather_than_guessed_at(self):
        """Without a mapping the path is used as-is -- correct for an agent
        running un-containerized on the same machine."""
        self.assertEqual(RG.host_path("/tmp/sessions", {}), "/tmp/sessions")

    def test_mapping_emits_posix_separators_on_every_operating_system(self):
        """``os.path`` here would emit backslashes on Windows, and the resulting
        path would silently match no file -- this bug was caught by a test
        before the module ever reached a host."""
        self.assertNotIn("\\", RG.host_path("/agent-data/sessions/run.jsonl", self.MOUNTS))
        self.assertEqual(RG.host_path("/agent-data", self.MOUNTS), "/srv/host-data")

    def test_newest_session_reads_the_file_through_the_mapping(self):
        """End to end over the mapping, because a blind mapping reads as 'no file'."""
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp) / "sessions"
            sessions.mkdir()
            (sessions / "run.jsonl").write_text(
                json.dumps({"event": "session_start"}) + "\n"
                + json.dumps({"event": "session_end", "status": "ok"}) + "\n",
                encoding="utf-8")
            runs = {"sessions_dir": "/agent-data/sessions",
                    "runs": [{"file": "run.jsonl", "started": "2026-11-03T17:07:46",
                              "status": "ok"}]}
            got = RG.newest_session(runs, {"/agent-data": tmp})
        self.assertTrue(got["readable"], "the mapping did not resolve to a real file")
        self.assertTrue(got["has_session_end"])
        self.assertEqual(got["file"], "run.jsonl")

    def test_a_killed_session_file_is_reported_as_readable_but_lacking_an_end(self):
        """Half a file is not an unreadable file: a killed run may end mid-line."""
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp) / "sessions"
            sessions.mkdir()
            (sessions / "dead.jsonl").write_text(
                json.dumps({"event": "session_start"}) + "\n{\"event\": \"act",
                encoding="utf-8")
            runs = {"sessions_dir": "/agent-data/sessions",
                    "runs": [{"file": "dead.jsonl", "started": "2026-11-03T15:58:11"}]}
            got = RG.newest_session(runs, {"/agent-data": tmp})
        self.assertTrue(got["readable"])
        self.assertFalse(got["has_session_end"])


class ExitCodesTest(unittest.TestCase):
    """The contract a calling script branches on: 3 means 'busy, nothing done'."""

    def _main(self, argv):
        """Run ``main`` with its report captured, so the suite stays readable."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return RG.main(argv)

    def test_json_mode_says_busy_when_a_run_is_in_flight(self):
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_since="2026-11-03T17:07:46"),
                                             session())):
            self.assertEqual(self._main(["--json"]), 3)

    def test_json_mode_says_ok_when_idle(self):
        with mock.patch.object(RG, "probe", return_value=(sched(), session())):
            self.assertEqual(self._main(["--json"]), 0)

    def test_json_mode_refuses_when_the_probe_itself_explodes(self):
        """An unreachable console is 'unknown', and unknown is never idle."""
        with mock.patch.object(RG, "probe", side_effect=ConnectionError("refused")):
            self.assertEqual(self._main(["--json"]), 2)

    def test_a_missing_container_is_a_config_error_before_any_probe(self):
        self.assertEqual(self._main(["--apply", "--compose-dir", "/d",
                                     "--compose-file", "c.yml"]), 1)

    def test_a_missing_image_is_a_config_error_before_any_build(self):
        self.assertEqual(self._main(["--build", "--build-context", "/ctx"]), 1)

    def test_apply_refuses_and_returns_2_when_the_scheduler_cannot_answer(self):
        with mock.patch.object(RG, "probe",
                               return_value=(sched(available=False, reason="boom"),
                                             session())):
            self.assertEqual(self._main(["--apply", "--container", "c",
                                         "--compose-dir", "/d",
                                         "--compose-file", "c.yml"]), 2)

    def test_apply_recreates_nothing_when_a_run_is_in_flight(self):
        """The whole point: busy means no subprocess is ever run."""
        with mock.patch.object(RG, "probe",
                               return_value=(sched(running=True,
                                                   in_progress_since="2026-11-03T17:07:46"),
                                             session())), \
             mock.patch.object(RG, "wait_until_idle", return_value=("wait", "still busy")), \
             mock.patch.object(RG, "sh") as sh:
            rc = self._main(["--apply", "--container", "c",
                             "--compose-dir", "/d", "--compose-file", "c.yml"])
        self.assertEqual(rc, 3)
        sh.assert_not_called()

    def test_apply_re_rechecks_immediately_before_the_destructive_step(self):
        """The race: a run that starts between the check and the recreate."""
        calls = []

        def fake_sh(cmd, log=print, cwd=None):
            calls.append(cmd)
            return 0, ""

        with mock.patch.object(RG, "probe", return_value=(sched(), session())), \
             mock.patch.object(RG, "confirm_still_idle",
                               return_value=("wait", "a run started in the gap")), \
             mock.patch.object(RG, "sh", side_effect=fake_sh):
            rc = self._main(["--apply", "--container", "c",
                             "--compose-dir", "/d", "--compose-file", "c.yml"])
        self.assertEqual(rc, 3)
        self.assertEqual(calls, [], "the recreate ran despite the confirm saying busy")


class VerifyProvesTheReplacementTest(unittest.TestCase):
    """A compose run that changed nothing reports rc 0 while the OLD code serves.

    Verification therefore compares the container's StartedAt across the
    recreate instead of trusting the command's exit code.
    """

    def test_an_unchanged_started_at_fails_even_though_every_command_succeeded(self):
        with mock.patch.object(RG, "sh", return_value=(0, "ts-same")) as sh:
            out = io.StringIO()
            rc = RG.verify("docker", "agent", "ts-same", log=out.write)
        self.assertEqual(rc, 4)
        self.assertIn("predates the recreate", out.getvalue())
        self.assertEqual(sh.call_count, 1, "it must not proceed to exec env")

    def test_a_replaced_container_is_confirmed_and_its_env_read_from_inside(self):
        with mock.patch.object(RG, "sh",
                               side_effect=[(0, "ts-new"),
                                            (0, "AGENT_MAX_STEPS=72\nPATH=/usr/bin")]) as sh:
            out = io.StringIO()
            rc = RG.verify("docker", "agent", "ts-old", log=out.write)
        self.assertEqual(rc, 0)
        self.assertIn("container replaced", out.getvalue())
        self.assertIn("AGENT_MAX_STEPS=72", out.getvalue())
        self.assertIn("exec agent env", sh.call_args_list[1][0][0])

    def test_a_secret_env_value_is_never_logged_masked_on_the_name(self):
        """`env` prints EVERY variable, credentials included.

        A real deployment watched a provider API key stream from this function
        into an operator transcript. The dump is withheld from the log and the
        re-reported value is masked; non-secret values stay readable, because
        "MAX_STEPS=72 took" is the entire point of the report.
        """
        raw = "AGENT_MAX_STEPS=72\nAGENT_API_KEY=sk-live-supersecret\nPATH=/usr/bin"
        with mock.patch.object(RG, "sh",
                               side_effect=[(0, "ts-new"), (0, raw)]) as sh:
            out = io.StringIO()
            rc = RG.verify("docker", "agent", "ts-old", log=out.write)
        logged = out.getvalue()
        self.assertEqual(rc, 0)
        self.assertNotIn("sk-live-supersecret", logged)
        self.assertIn("AGENT_API_KEY=<masked", logged)
        self.assertIn("AGENT_MAX_STEPS=72", logged)
        # The mock above cannot exercise sh()'s own suppression, so pin the
        # CONTRACT instead: verify must ASK for quiet output on the env call.
        env_call = sh.call_args_list[1]
        self.assertTrue(env_call.kwargs.get("quiet_out", False) or
                        (len(env_call.args) > 3 and env_call.args[3] is True),
                        "verify read env without quiet_out -- the raw dump would "
                        "stream every secret straight into the log")

    def test_sh_with_quiet_out_logs_the_command_but_never_the_body(self):
        """The suppression itself, at its own level."""
        done = mock.Mock(returncode=0, stdout="INFERHUB_API_KEY=sk-provider\n")
        with mock.patch.object(RG.subprocess, "run", return_value=done):
            out = io.StringIO()
            rc, body = RG.sh("docker exec agent env", log=out.write, quiet_out=True)
        self.assertEqual((rc, body), (0, "INFERHUB_API_KEY=sk-provider"))
        self.assertIn("+ docker exec agent env", out.getvalue(),
                      "the command line is not secret and stays visible")
        self.assertNotIn("sk-provider", out.getvalue())

    def test_sh_without_quiet_out_still_logs_the_body(self):
        """Suppression is opt-in: ordinary commands keep their output."""
        done = mock.Mock(returncode=0, stdout="done\n")
        with mock.patch.object(RG.subprocess, "run", return_value=done):
            out = io.StringIO()
            RG.sh("docker build -t x .", log=out.write)
        self.assertIn("done", out.getvalue())

    def test_non_prefixed_variables_are_not_reported_at_all(self):
        """The env prefix is a filter, not a redaction hint: an unprefixed
        provider key (the shape that actually leaked) never appears."""
        raw = "INFERHUB_API_KEY=sk-provider-leak\nPATH=/usr/bin"
        with mock.patch.object(RG, "sh",
                               side_effect=[(0, "ts-new"), (0, raw)]):
            out = io.StringIO()
            RG.verify("docker", "agent", "ts-old", log=out.write)
        logged = out.getvalue()
        self.assertNotIn("sk-provider-leak", logged)
        self.assertNotIn("INFERHUB", logged)

    def test_an_uninspectable_container_fails_closed(self):
        with mock.patch.object(RG, "sh", return_value=(1, "No such object")):
            out = io.StringIO()
            rc = RG.verify("docker", "agent", "ts-old", log=out.write)
        self.assertEqual(rc, 4)
        self.assertIn("cannot inspect", out.getvalue())


class DetachedRunShowsProgressTest(unittest.TestCase):
    """A guard nobody can watch is a guard that gets killed for looking hung.

    Observed live on 2026-11-03: run detached while it waited out a pass in
    flight, the log file was STILL 0 bytes three minutes in -- ``print``
    block-buffers when stdout is not a tty. The tool was working correctly and
    was indistinguishable from a hang.
    """

    def test_line_log_flushes_so_a_redirected_log_is_not_empty(self):
        import builtins
        with mock.patch.object(builtins, "print") as p:
            RG.line_log("[probe] WAIT", "--", "reason")
        self.assertTrue(p.call_args.kwargs.get("flush"),
                        "line_log did not pass flush=True, so a detached run "
                        "buffers everything until it exits")

    def test_helpers_log_through_the_flushing_logger_by_default(self):
        """The default must be the flushing one, not ``print``."""
        import inspect
        for fn in (RG.sh, RG.wait_until_idle, RG.confirm_still_idle, RG.verify):
            default = inspect.signature(fn).parameters["log"].default
            self.assertIs(default, RG.line_log,
                          "%s still defaults to a buffering logger" % fn.__name__)

    def test_a_detached_run_writes_progress_while_it_is_still_running(self):
        """End to end: the file has the WAIT line BEFORE the process exits."""
        payload = json.dumps({
            "scheduler": {"available": True, "running": True,
                          "in_progress_since": "2026-11-03T19:39:08",
                          "in_progress_sec": 42.0},
            "runs": [], "sessions_dir": "/nonexistent-sessions",
        }).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = os.path.join(tmp, "guard.log")
                with open(log_path, "wb") as sink:
                    proc = subprocess.Popen(
                        [sys.executable, str(GUARD), "--apply", "--wait-sec", "40",
                         "--console", "http://127.0.0.1:%d" % server.server_address[1],
                         "--compose-dir", tmp, "--compose-file", "none.yml",
                         "--container", "no-such-container"],
                        stdout=sink, stderr=subprocess.STDOUT)
                    seen = b""
                    deadline = time.time() + 15
                    while time.time() < deadline and proc.poll() is None:
                        time.sleep(0.4)
                        with open(log_path, "rb") as fh:
                            seen = fh.read()
                        if b"WAIT" in seen:
                            break
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:  # pragma: no cover
                        proc.kill()
        finally:
            server.shutdown()
        self.assertIn(b"WAIT", seen,
                      "a detached guard wrote nothing while still running -- "
                      "stdout is block-buffered and every probe line is stuck "
                      "in the buffer")


if __name__ == "__main__":
    unittest.main()
