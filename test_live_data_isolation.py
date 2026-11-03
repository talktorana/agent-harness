"""No test may write into the deployment's live data.

Two leaks, one shape, both found on 2026-11-03 by a monitoring pass that read
its own console and could not explain what it saw. Inside a deployed container
the environment points the harness at REAL data -- the deploy outbox a host
timer drains and applies, and the sessions directory that is the console's run
history. On a workstation those variables are empty, so every scratch directory
wins and the suite is green. The same tests, the same file, `Ran 694 ... OK`
here and writing into production there, with no signal at either end.

THE OUTBOX. `RebuildOnlyStagingTest` patched PROJECT_ROOT, DEPLOY_ROOT and
DEPLOY_MODE but not `DEPLOY_OUTBOX`, which `outbox_dir()` prefers over any
derived path. Its staging calls composed real requests into the live outbox;
the applier applied them, writing a fixture file into the deployment tree and
attempting a rebuild of a service that does not exist.

THE SESSIONS DIR. `session.py` did `from config import SESSIONS_DIR`, binding
the value at import, so patching `config.SESSIONS_DIR` redirected NOTHING. A
test class that builds a real Agent then wrote its fixture sessions -- and its
lines in the single shared global transcript -- into production run history.
Twelve fixture session files and six transcript lines were the evidence.

Three layers keep this from recurring, because a missing isolation patch looks
like nothing on the only machine where it passes:

1. A structural scan of every test module: any class that CALLS a function
   which can write live data must pin the relevant config value in its own
   setUp or inherit a base that does. Calls, not mentions -- `mock.patch.object`
   and `assertIs` name these functions without using them, and a guard that
   cries wolf gets ignored, which is how a real offender survives.
2. Behavioural reproductions: point the config value at a stand-in for the live
   location, run the REAL test through its own setUp, and assert the stand-in
   stayed empty.
3. Direct tests of the lazy-resolution contract that the sessions leak broke.
"""
from __future__ import annotations

import ast as _ast
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import session
import test_deploy_tools
import test_webui_live
import tools_deploy
import tools_session

HERE = Path(__file__).resolve().parent


#: Functions that can write a request where a host applier will find it.
STAGING_CALLS = frozenset({
    "compose_deploy_request", "deploy_request", "apply_request",
    "run_applier", "apply_pending", "stage_request",
})
#: Functions that create a run, and therefore a session file and transcript.
RUN_CALLS = frozenset({"Agent", "run_task", "_start_task", "run_agent"})
#: Constructed directly, and isolated either by a pin or by an explicit
#: `sessions_dir=` on every call in the class.
SESSION_LOG_CALLS = frozenset({"SessionLog"})

PIN_OUTBOX = ("DEPLOY_OUTBOX",)
PIN_SESSIONS = ("SESSIONS_DIR", "TRANSCRIPT")


# --------------------------------------------------------------------- scanner


def _called_names(node) -> set:
    """Function names actually CALLED inside this AST node."""
    names = set()
    for sub in _ast.walk(node):
        if not isinstance(sub, _ast.Call):
            continue
        fn = sub.func
        name = (fn.attr if isinstance(fn, _ast.Attribute)
                else fn.id if isinstance(fn, _ast.Name) else None)
        if name:
            names.add(name)
    return names


def _explicit_sessions_dir(node) -> bool:
    """Does EVERY SessionLog(...) call here pass sessions_dir=?"""
    calls = [sub for sub in _ast.walk(node)
             if isinstance(sub, _ast.Call)
             and ((isinstance(sub.func, _ast.Attribute)
                   and sub.func.attr in SESSION_LOG_CALLS)
                  or (isinstance(sub.func, _ast.Name)
                      and sub.func.id in SESSION_LOG_CALLS))]
    if not calls:
        return False
    return all(any(kw.arg == "sessions_dir" for kw in c.keywords)
               for c in calls)


def _reads_config_sessions(value) -> bool:
    """True when the expression is exactly `config.SESSIONS_DIR`."""
    return (isinstance(value, _ast.Attribute)
            and value.attr == "SESSIONS_DIR"
            and isinstance(value.value, _ast.Name)
            and value.value.id == "config")


class _Index:
    """Every test class in the repo, with cross-module bases resolved.

    Cross-module inheritance is the reason this is an index and not a per-file
    pass: test classes routinely inherit a scratch-directory base that lives in
    another module, and a scan that cannot follow it reports those classes as
    offenders. A guard that names safe classes gets switched off, and the one
    real offender goes back to being invisible.
    """

    def __init__(self):
        self.classes = {}          # "module::Class" -> (path, node, source)
        self.by_name = {}          # "Class" -> "module::Class"
        self.sources = {}
        for path in sorted(HERE.glob("test_*.py")):
            src = path.read_text(encoding="utf-8")
            self.sources[path.name] = src
            tree = _ast.parse(src, filename=str(path))
            for node in tree.body:
                if isinstance(node, _ast.ClassDef):
                    key = f"{path.name}::{node.name}"
                    seg = _ast.get_source_segment(src, node) or ""
                    self.classes[key] = (path, node, seg)
                    self.by_name.setdefault(node.name, key)

    def pins(self, key: str, attrs, seen=None) -> bool:
        """Does this class, or any base it inherits, pin one of `attrs`?"""
        seen = seen or set()
        if key in seen or key not in self.classes:
            return False
        seen.add(key)
        _, node, seg = self.classes[key]
        for attr in attrs:
            if f'"{attr}"' in seg or f"'{attr}'" in seg:
                return True
        for base in node.bases:
            name = base.id if isinstance(base, _ast.Name) else None
            if name and name in self.by_name:
                if self.pins(self.by_name[name], attrs, seen):
                    return True
        return False

    def offenders(self):
        out = []
        for key, (path, node, _) in self.classes.items():
            #: NOTE: this module is deliberately NOT skipped. It names every
            #: staging function and every run-creating function above, so if the
            #: scanner flagged mentions it would report this file first -- and
            #: the self-test below is what proves it flags calls instead.
            called = _called_names(node)
            if called & STAGING_CALLS and not self.pins(key, PIN_OUTBOX):
                out.append((key, "deploy outbox",
                            sorted(called & STAGING_CALLS)))
            creates_runs = bool(called & RUN_CALLS)
            logs_sessions = bool(called & SESSION_LOG_CALLS)
            if creates_runs and not self.pins(key, PIN_SESSIONS):
                out.append((key, "sessions dir", sorted(called & RUN_CALLS)))
            elif logs_sessions and not (self.pins(key, PIN_SESSIONS)
                                        or _explicit_sessions_dir(node)):
                out.append((key, "sessions dir",
                            sorted(called & SESSION_LOG_CALLS)))
        return sorted(out)


class StructuralIsolationTest(unittest.TestCase):
    """Every class that can write live data pins where it writes."""

    def test_no_test_class_can_write_into_live_data(self):
        offenders = _Index().offenders()
        rendered = [f"{key} [{surface}, calls {','.join(calls)}]"
                    for key, surface, calls in offenders]
        self.assertEqual(
            rendered, [],
            "These test classes call something that writes live data without "
            "pinning the config value that decides where. Inside a deployed "
            "container that is the REAL location -- a host applier drains the "
            "outbox, and the sessions directory is the console's run history "
            "both missions audit. Pin it in the class setUp (or inherit a base "
            "that does):\n  " + "\n  ".join(rendered))

    def test_the_scan_does_not_flag_this_module(self):
        """It names every staging function above, and calls none of them.

        This module is indexed like any other, so the assertion is real: a
        scanner that flagged mentions -- `mock.patch.object(tools_deploy,
        "apply_request")`, `assertIs(HANDLERS["deploy_request"], ...)` -- would
        report this file. A guard that cries wolf gets switched off, and the one
        real offender goes back to being invisible.
        """
        index = _Index()
        own = Path(__file__).name
        indexed_here = [key for key, (path, _, _) in index.classes.items()
                        if path.name == own]
        self.assertTrue(indexed_here,
                        "expected this module to be scanned, not skipped")
        self.assertEqual(
            [key for key, _, _ in index.offenders()
             if key.startswith(own + "::")], [],
            "this module names the functions but does not call them unguarded")

    def test_the_scan_catches_an_unpinned_class(self):
        """Prove it bites, on throwaway modules it scans directly."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leaky = root / "test_bogus_leaky.py"
            leaky.write_text(
                "import unittest\nimport tools_deploy\n"
                "class LeakyOutbox(unittest.TestCase):\n"
                "    def test_it(self):\n"
                "        tools_deploy.compose_deploy_request(purpose='p',\n"
                "                                            files=[])\n",
                encoding="utf-8")
            names = _called_names(_ast.parse(leaky.read_text(
                encoding="utf-8")).body[-1])
            self.assertIn("compose_deploy_request", names)
            self.assertTrue(names & STAGING_CALLS)

            safe = root / "test_bogus_safe.py"
            safe.write_text(
                "import unittest\nfrom unittest import mock\n"
                "import config, tools_deploy\n"
                "class Safe(unittest.TestCase):\n"
                "    def setUp(self):\n"
                '        mock.patch.object(config, "DEPLOY_OUTBOX",'
                ' "/tmp/x").start()\n'
                '        mock.patch.object(config, "SESSIONS_DIR",'
                ' "/tmp/y").start()\n'
                "    def test_it(self):\n"
                "        tools_deploy.compose_deploy_request(purpose='p',\n"
                "                                            files=[])\n",
                encoding="utf-8")
            node = _ast.parse(safe.read_text(encoding="utf-8")).body[-1]
            self.assertTrue(_called_names(node) & STAGING_CALLS)
            src = safe.read_text(encoding="utf-8")
            index = _Index()
            index.classes["test_bogus_safe.py::Safe"] = (
                safe, node, _ast.get_source_segment(src, node) or "")
            self.assertTrue(index.pins("test_bogus_safe.py::Safe",
                                       PIN_OUTBOX),
                            "a class that patches the outbox must read as safe")
            #: And the same machinery must flag the one that does not.
            leaky_src = leaky.read_text(encoding="utf-8")
            leaky_node = _ast.parse(leaky_src).body[-1]
            index.classes["test_bogus_leaky.py::LeakyOutbox"] = (
                leaky, leaky_node,
                _ast.get_source_segment(leaky_src, leaky_node) or "")
            self.assertFalse(index.pins("test_bogus_leaky.py::LeakyOutbox",
                                        PIN_OUTBOX))


class AmbientOutboxTest(unittest.TestCase):
    """Behavioural reproduction #1: the deploy outbox."""

    def test_outbox_dir_prefers_the_configured_outbox(self):
        """Why the leak happened: the configured value wins over derived paths.

        That is correct production behaviour -- a deployment must be able to aim
        the agent at its own outbox -- and exactly why patching PROJECT_ROOT and
        DEPLOY_ROOT is not isolation.
        """
        with tempfile.TemporaryDirectory() as td:
            live, scratch = Path(td) / "live", Path(td) / "scratch"
            live.mkdir()
            scratch.mkdir()
            with mock.patch.object(config, "DEPLOY_OUTBOX", str(live)):
                self.assertEqual(tools_deploy.outbox_dir(), live)
            with mock.patch.object(config, "DEPLOY_OUTBOX", ""), \
                    mock.patch.object(config, "DEPLOY_ROOT", str(scratch)), \
                    mock.patch.object(config, "PROJECT_ROOT", str(scratch)):
                self.assertNotEqual(tools_deploy.outbox_dir(), live)

    def test_rebuild_staging_never_reaches_an_ambient_live_outbox(self):
        """Run the REAL test with the container's environment shape.

        Without the class pinning DEPLOY_OUTBOX the stand-in collects a request
        document, which is precisely what the host applier applied.
        """
        with tempfile.TemporaryDirectory() as td:
            live = Path(td) / "live-outbox"
            live.mkdir()
            with mock.patch.object(config, "DEPLOY_OUTBOX", str(live)):
                suite = unittest.TestLoader().loadTestsFromName(
                    "test_a_fileless_rebuild_request_can_be_staged",
                    test_deploy_tools.RebuildOnlyStagingTest)
                result = unittest.TextTestRunner(
                    stream=io.StringIO(), verbosity=0).run(suite)
                leaked = sorted(p.name for p in live.iterdir())
        self.assertTrue(result.wasSuccessful(),
                        "the test under reproduction failed; this assertion is "
                        "about isolation, not about that test passing")
        self.assertEqual(
            leaked, [],
            f"a staging test wrote into the ambient live outbox: {leaked}")


class AmbientSessionsTest(unittest.TestCase):
    """Behavioural reproduction #2: the run history and the transcript."""

    def test_a_real_agent_run_never_reaches_an_ambient_sessions_dir(self):
        """Run the REAL Agent test with config.SESSIONS_DIR pointing at live.

        Pre-fix this class wrote fixture sessions ("do something", model
        prov:alias) into the deployment's sessions directory and its lines into
        the shared global transcript -- the run history both monitoring
        missions audit.
        """
        with tempfile.TemporaryDirectory() as td:
            live = Path(td) / "live-sessions"
            live.mkdir()
            transcript = Path(td) / "live-transcript.jsonl"
            with mock.patch.object(config, "SESSIONS_DIR", live), \
                    mock.patch.object(config, "TRANSCRIPT", str(transcript)):
                suite = unittest.TestLoader().loadTestsFromName(
                    "test_preset_stop_returns_before_any_model_call",
                    test_webui_live.AgentStopSemanticsTest)
                result = unittest.TextTestRunner(
                    stream=io.StringIO(), verbosity=0).run(suite)
                leaked = sorted(p.name for p in live.rglob("*"))
                wrote_transcript = transcript.exists()
        self.assertTrue(result.wasSuccessful(),
                        "the test under reproduction failed; this assertion is "
                        "about isolation, not about that test passing")
        self.assertEqual(
            leaked, [],
            f"a test wrote session files into the ambient live directory: "
            f"{leaked} -- these appear in the console's run history")
        self.assertFalse(
            wrote_transcript,
            "a test appended to the shared global transcript")

    def test_a_session_honours_a_patched_sessions_dir(self):
        """The lazy-resolution contract that was broken.

        `from config import SESSIONS_DIR` binds at import, so patching the
        config value -- what every test does, and what a deployment override
        relies on -- redirects nothing. This is the regression test for that
        one line, and it fails if the import-time binding ever returns.
        """
        with tempfile.TemporaryDirectory() as td:
            scratch = Path(td) / "sessions"
            with mock.patch.object(config, "SESSIONS_DIR", scratch):
                log = session.SessionLog(task="isolation probe")
                log.finish("ok")
                self.assertTrue(
                    Path(log.jsonl_path).is_relative_to(scratch),
                    f"session written to {log.jsonl_path}, not to the patched "
                    f"{scratch}: an import-time binding is back")

    def test_the_session_tools_read_the_configured_directory(self):
        """The same trap lived in tools_session and tools_todo."""
        with tempfile.TemporaryDirectory() as td:
            scratch = Path(td) / "sessions"
            with mock.patch.object(config, "SESSIONS_DIR", scratch):
                log = session.SessionLog(task="listing probe")
                log.finish("ok")
                listing = tools_session.session_list(limit=5)
                self.assertIn(log.md_path.name, listing,
                              "session_list read a stale import-time directory")

    def test_no_module_binds_the_sessions_dir_at_import(self):
        """Belt and braces: forbid the pattern that caused this.

        Both spellings are caught, because a mutation experiment showed only the
        first was: ``from config import SESSIONS_DIR`` and the equivalent
        module-level copy ``SESSIONS_DIR = config.SESSIONS_DIR`` bind the same
        value at the same moment and are equally immune to every later patch.
        """
        offenders = []
        for path in sorted(HERE.glob("*.py")):
            if path.name.startswith("test_"):
                continue
            tree = _ast.parse(path.read_text(encoding="utf-8"),
                              filename=str(path))
            for node in tree.body:
                #: `from config import SESSIONS_DIR`
                if (isinstance(node, _ast.ImportFrom)
                        and node.module == "config"):
                    for alias in node.names:
                        if alias.name == "SESSIONS_DIR":
                            offenders.append(f"{path.name} (from-import)")
                #: `SESSIONS_DIR = config.SESSIONS_DIR`, or any module-level
                #: name bound to it and then used as the default.
                if isinstance(node, _ast.Assign) and _reads_config_sessions(
                        node.value):
                    offenders.append(f"{path.name} (module-level copy)")
        self.assertEqual(
            offenders, [],
            "Binding config.SESSIONS_DIR at import time freezes the live "
            "sessions directory into a module constant, after which no "
            "configuration change and no test patch can redirect a session "
            "write. Read config.SESSIONS_DIR at call time:\n  "
            + "\n  ".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
