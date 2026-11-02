"""A run that hits an unexpected exception must still CLOSE its session.

Why this matters: the console reports "no end record" to mean the process died
mid-run -- a container restart, an OOM -- because that is the one case where
nothing could have been written. Every normal exit from the step loop already
calls ``session.finish``, but an exception raised where the loop does not wrap
escaped ``Agent.run`` entirely, leaving the .jsonl open with no
``session_end``. A plain crash therefore looked identical to a kill, and the
console could not point at a cause.

Four runs were investigated as "killed" on 2026-11-02 and every one had died at
a container recreate; without this guard the same evidence would have been
ambiguous forever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402


class _FakeSession:
    """Records what the guard asked of it."""

    def __init__(self):
        self.events = []
        self.finished = None

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def finish(self, status="ok"):
        self.finished = status


def _bare_agent(monkeypatch_target=None):
    """An Agent instance without running Agent.__init__ (no network, no keys)."""
    instance = agent.Agent.__new__(agent.Agent)
    instance.session = _FakeSession()
    instance.steps = 3
    instance._log = lambda record: instance.logged.append(record)
    instance.logged = []
    instance._close = lambda: instance.closed.append(True)
    instance.closed = []
    return instance


class CrashGuardTest(unittest.TestCase):
    def test_an_unexpected_exception_closes_the_session_and_reraises(self):
        instance = _bare_agent()
        boom = RuntimeError("a tool blew up")

        def explode(*_a, **_kw):
            raise boom

        instance._run_impl = explode
        with self.assertRaises(RuntimeError) as ctx:
            instance.run("do the thing")
        self.assertIs(ctx.exception, boom, "the original exception must surface")

        self.assertEqual(instance.session.finished, "crashed",
                         "the session must be closed, or a crash reads as a kill")
        kinds = [name for name, _ in instance.session.events]
        self.assertIn("crashed", kinds)
        self.assertTrue(instance.closed, "the http client must be closed too")

    def test_the_crash_is_recorded_with_its_type_and_message(self):
        instance = _bare_agent()
        instance._run_impl = mock.Mock(side_effect=ValueError("bad arguments"))
        with self.assertRaises(ValueError):
            instance.run("task")
        kind, fields = instance.session.events[-1]
        self.assertEqual(kind, "crashed")
        self.assertEqual(fields["type"], "ValueError")
        self.assertIn("bad arguments", fields["error"])

    def test_a_normal_return_is_untouched(self):
        """The guard must not alter the happy path in any way."""
        instance = _bare_agent()
        instance._run_impl = lambda task, images=None: "the answer"
        self.assertEqual(instance.run("task"), "the answer")
        self.assertIsNone(instance.session.finished,
                          "a completed run finishes its own session")

    def test_run_really_delegates_to_run_impl(self):
        """Guard the delegation itself, not just the guard body.

        If ``run()`` stopped calling ``_run_impl``, every test here would still
        pass against a stub and no session would ever be created.
        """
        import inspect
        wrapper = inspect.getsource(agent.Agent.run)
        self.assertIn("_run_impl(task, images)", wrapper)
        #: And the real body must still exist with the loop in it.
        body = inspect.getsource(agent.Agent._run_impl)
        self.assertIn("SessionLog(", body)
        self.assertIn("for step in range", body)

    def test_an_exception_before_the_session_exists_still_raises(self):
        """SessionLog construction is the first thing run() does."""
        instance = _bare_agent()
        instance.session = None
        instance._run_impl = mock.Mock(side_effect=OSError("disk full"))
        with self.assertRaises(OSError):
            instance.run("task")

    def test_a_failing_record_attempt_does_not_mask_the_real_error(self):
        """Recording the crash must never become a second crash."""
        instance = _bare_agent()

        class _BadSession(_FakeSession):
            def event(self, kind, **fields):
                raise RuntimeError("cannot write")

            def finish(self, status="ok"):
                raise RuntimeError("cannot write")

        instance.session = _BadSession()
        instance._run_impl = mock.Mock(side_effect=KeyError("original"))
        with self.assertRaises(KeyError):
            instance.run("task")

    def test_the_session_finish_reaches_the_jsonl(self):
        """End to end through the REAL SessionLog: the file is closed properly."""
        import tempfile
        import session as session_module

        with tempfile.TemporaryDirectory() as tmp:
            real = session_module.SessionLog(task="t", sessions_dir=Path(tmp))
            instance = _bare_agent()
            instance.session = real
            instance._run_impl = mock.Mock(side_effect=RuntimeError("boom"))
            with self.assertRaises(RuntimeError):
                instance.run("t")
            records = [json.loads(l) for l in
                       real.jsonl_path.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
        end = [r for r in records if r.get("event") == "session_end"]
        self.assertEqual(len(end), 1, records[-3:])
        self.assertEqual(end[0]["status"], "crashed")


if __name__ == "__main__":
    unittest.main()
