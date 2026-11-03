"""No test may write into the deployment's live outbox.

THE INCIDENT THIS FILE EXISTS FOR. Inside the sidecar, `AGENT_DEPLOY_OUTBOX`
points at the real deploy outbox, which a host timer drains and APPLIES. On the
workstation that variable is empty, so `outbox_dir()` falls back to a path
derived from `PROJECT_ROOT`/`DEPLOY_ROOT` and every test's scratch directory
wins. Result: one test class patched `PROJECT_ROOT`, `DEPLOY_ROOT` and
`DEPLOY_MODE` -- and not the outbox -- passed everywhere locally, and inside the
container it staged real requests into production. The applier then applied
them: it wrote a fixture file into the live deployment tree and attempted a
service rebuild for a service that does not exist. The same file, the same
tests, `Ran 694 ... OK` on one box and destructive on the other, with no signal
at either end.

Two layers:

1. A structural scan of every test module. Any class that CALLS a staging
   function must pin `config.DEPLOY_OUTBOX` in its own setUp or inherit a base
   that does. This is the guard that would have named the offender before it
   ever ran inside a container, and it keeps failing if someone "cleans up" an
   isolation patch that looks redundant on a workstation.
2. A behavioural reproduction: run the real test with the ambient outbox
   pointed at a live-looking directory, and assert that directory stayed empty.
"""
from __future__ import annotations

import ast as _ast
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import test_deploy_tools
import tools_deploy

HERE = Path(__file__).resolve().parent

#: Functions that can write a request where the host applier will find it.
#: `deploy_request` is included because it stages AND may invoke the applier.
STAGING_CALLS = frozenset({
    "compose_deploy_request", "deploy_request", "apply_request",
    "run_applier", "apply_pending", "stage_request",
})
OUTBOX_ATTR = "DEPLOY_OUTBOX"


def _staging_calls(node: _ast.AST) -> set:
    """Names actually CALLED here, not merely referenced.

    Call-based on purpose: `mock.patch.object(tools_deploy, "apply_request")`
    and `assertIs(HANDLERS["deploy_request"], tools_deploy.deploy_request)`
    mention staging functions without staging anything, and flagging those
    would make a guard that cries wolf get ignored -- which is how a real
    offender survives.
    """
    names = set()
    for sub in _ast.walk(node):
        if not isinstance(sub, _ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, _ast.Attribute) and fn.attr in STAGING_CALLS:
            names.add(fn.attr)
        elif isinstance(fn, _ast.Name) and fn.id in STAGING_CALLS:
            names.add(fn.id)
    return names


def _offenders_in(path: Path) -> list:
    """Classes in one test module that stage without pinning the outbox."""
    source = path.read_text(encoding="utf-8")
    tree = _ast.parse(source, filename=str(path))
    classes = {n.name: n for n in tree.body if isinstance(n, _ast.ClassDef)}

    def pins(name: str, seen=None) -> bool:
        """Does this class, or a base in the same module, pin the outbox?"""
        seen = seen or set()
        if name in seen or name not in classes:
            return False
        seen.add(name)
        node = classes[name]
        seg = _ast.get_source_segment(source, node) or ""
        if f'"{OUTBOX_ATTR}"' in seg or f"'{OUTBOX_ATTR}'" in seg:
            return True
        return any(pins(b.id, seen) for b in node.bases
                   if isinstance(b, _ast.Name))

    offenders = []
    for name, node in classes.items():
        calls = _staging_calls(node)
        if calls and not pins(name):
            offenders.append(f"{path.name}::{name} "
                             f"(calls {','.join(sorted(calls))})")
    return offenders


class StagingIsolationTest(unittest.TestCase):
    """The structural guard."""

    def test_every_staging_test_class_pins_the_outbox(self):
        offenders = []
        for path in sorted(HERE.glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue
            offenders.extend(_offenders_in(path))
        self.assertEqual(
            offenders, [],
            "These test classes call a staging function without pinning "
            f"config.{OUTBOX_ATTR}, so they write to whatever the ambient "
            "environment points at -- inside a deployed container that is the "
            "LIVE outbox, which a host timer applies. Add the patch to the "
            "class setUp (or inherit a base that has it):\n  "
            + "\n  ".join(offenders))

    def test_the_scan_reads_calls_not_mentions(self):
        """A guard that flags `mock.patch.object(..., "apply_request")` would.

        Pointing it at this module proves the discrimination: the docstring and
        the constant above NAME every staging function, and none of that is a
        call, so this module must not appear in its own offender list.
        """
        self.assertEqual(_offenders_in(Path(__file__).resolve()), [])

    def test_the_scan_catches_a_class_that_forgets_the_patch(self):
        """Prove the guard bites, on a throwaway module."""
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "test_invented_offender.py"
            bad.write_text(
                "import config, tools_deploy\n"
                "class Leaky(unittest.TestCase):\n"
                "    def setUp(self):\n"
                "        pass\n"
                "    def test_it(self):\n"
                "        tools_deploy.compose_deploy_request(purpose='p',\n"
                "                                            files=[])\n",
                encoding="utf-8")
            found = _offenders_in(bad)
            self.assertEqual(len(found), 1, found)
            self.assertIn("Leaky", found[0])

            good = Path(td) / "test_invented_fixed.py"
            good.write_text(
                bad.read_text(encoding="utf-8").replace(
                    "        pass\n",
                    '        mock.patch.object(config, "DEPLOY_OUTBOX", '
                    '"/tmp/x").start()\n'),
                encoding="utf-8")
            self.assertEqual(_offenders_in(good), [])


class AmbientOutboxTest(unittest.TestCase):
    """The behavioural reproduction, and the mechanism it depends on."""

    def test_outbox_dir_prefers_the_configured_outbox(self):
        """Why the leak happened at all: the configured value wins.

        This is correct production behaviour -- a deployment MUST be able to
        aim the agent at its own outbox -- and it is exactly why a test that
        patches only PROJECT_ROOT/DEPLOY_ROOT is not isolated.
        """
        with tempfile.TemporaryDirectory() as td:
            live, scratch = Path(td) / "live-outbox", Path(td) / "scratch"
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

        The ambient `config.DEPLOY_OUTBOX` is pointed at a stand-in for the
        deployment's outbox; the class under test then runs its own setUp and
        stages. Without the class pinning `DEPLOY_OUTBOX`, the stand-in
        directory collects a real request -- which is what the host applier
        applied on 2026-11-03.
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
                        "about isolation, not about the test passing")
        self.assertEqual(
            leaked, [],
            f"a staging test wrote into the ambient live outbox: {leaked}. "
            "Inside a container these files are applied by the host applier.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
