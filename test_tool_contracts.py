"""Every tool must be callable with exactly the arguments its schema requires.

Found live 2026-11-02 by reading run transcripts instead of trusting the status
field: eight scheduled passes hit

    BAD ARGUMENTS for memory_append: memory_append() missing 1 required
    positional argument: 'scope'

The schema for ``memory_append`` listed only ``text`` as required and described
``scope`` as "'repo' (default)", and the function BODY already read
``scope or "repo"`` -- but the SIGNATURE demanded ``scope``. A model that reads
the schema and does exactly what it says therefore failed every time, and the
failure came back as tool text, so the answer still read as a success. The agent
had done the work and could not record the lesson, which is the one thing its
memory layer exists for.

``memory_revise`` and ``memory_retire`` had the identical defect; the two
readers were fine because someone had given them a default.

So the invariant, checked mechanically for EVERY registered tool rather than for
the three that happened to break:

    a tool whose schema declares a set of REQUIRED properties must be callable
    with exactly those properties -- no parameter may be required by the
    signature but absent from the schema's required list.

That is the only relationship the model can act on: it sees the schema, never
the Python.

Run:  python test_tool_contracts.py
"""
from __future__ import annotations

import inspect
import unittest
from typing import Any, Dict, List

import tools


def _schemas() -> Dict[str, Dict[str, Any]]:
    """name -> schema, from the list the model is actually given."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry in tools.TOOLS:
        fn = (entry or {}).get("function") or {}
        name = fn.get("name")
        if name:
            out[name] = fn.get("parameters") or {}
    return out


def _dummy_for(spec: Dict[str, Any]) -> Any:
    """A value of the declared type, so binding exercises the SHAPE only."""
    kind = (spec or {}).get("type")
    if kind == "string":
        return "x"
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "x"


def required_but_unmapped(name: str) -> List[str]:
    """Parameters the signature requires that the schema does not.

    A required POSITIONAL parameter with no default is the killer: the model
    cannot know to pass it, and the call fails before the tool body runs.
    """
    handler = tools.HANDLERS.get(name)
    if handler is None:
        return []
    schema = _schemas().get(name, {})
    declared = set(schema.get("required") or [])
    sig = inspect.signature(handler)
    missing: List[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if pname in declared:
            continue
        if param.default is inspect.Parameter.empty:
            #: ``self`` never appears here (handlers are module functions), but
            #: be explicit rather than clever.
            missing.append(pname)
    return missing


class ToolSchemaContractTest(unittest.TestCase):
    def test_every_handler_is_registered_the_other_way_round(self):
        """A handler with no schema is invisible; a schema with no handler 500s."""
        schemas = set(_schemas())
        handlers = set(tools.HANDLERS)
        self.assertEqual(schemas - handlers, set(),
                         "schema without a handler: the model would be told a "
                         "tool exists and get an error when it calls it")
        #: The reverse is allowed only for deliberate internals.
        self.assertEqual(
            handlers - schemas,
            set(getattr(tools, "INTERNAL_ONLY", set())),
            "handler without a schema: the model can never call it")

    def test_no_tool_requires_an_argument_its_schema_does_not_declare(self):
        """THE defect this file exists for. See the module docstring."""
        offenders = {name: needed for name in sorted(tools.HANDLERS)
                     if (needed := required_but_unmapped(name))}
        self.assertEqual(
            offenders, {},
            "these tools demand parameters their schema does not list, so a "
            "model that follows the schema gets a TypeError instead of a "
            f"result: {offenders}")

    def test_every_tool_can_be_called_with_its_declared_required_arguments(self):
        """Behavioural, not just structural: actually BIND the call.

        Signature inspection can be satisfied by a ``**kwargs`` while the body
        still explodes; binding a real argument list proves the shape the model
        will produce is accepted.
        """
        failures = {}
        for name, handler in sorted(tools.HANDLERS.items()):
            schema = _schemas().get(name)
            if schema is None:
                continue
            props = schema.get("properties") or {}
            args = {req: _dummy_for(props.get(req) or {})
                    for req in (schema.get("required") or [])}
            try:
                inspect.signature(handler).bind(**args)
            except TypeError as exc:
                failures[name] = f"{exc} (passed {sorted(args)})"
        self.assertEqual(failures, {}, f"uncallable as declared: {failures}")


class MemoryWriterTest(unittest.TestCase):
    """The three writers, called the way the schema says they may be."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        #: Point the memory layer at a scratch tree so these tests never touch
        #: the real notes the agent depends on.
        self._orig = getattr(tools._MEMORY, "_root", None)
        try:
            tools._MEMORY._root = self.root
        except Exception:            # noqa: BLE001 - shape may differ
            self.skipTest("memory layer does not expose a writable root")

    def tearDown(self):
        if self._orig is not None:
            try:
                tools._MEMORY._root = self._orig
            except Exception:    # noqa: BLE001
                pass

    def test_append_works_without_a_scope(self):
        """The exact call the model made: text + heading, no scope."""
        out = tools.memory_append(text="a note body", heading="# Note")
        self.assertNotIn("missing 1 required positional", str(out))
        self.assertNotIn("BAD ARGUMENTS", str(out))
        self.assertNotIn("TypeError", str(out))

    def test_revise_and_retire_work_without_a_scope(self):
        for name, kwargs in (("memory_revise", {"old": "nothing here",
                                                "new": "x"}),
                             ("memory_retire", {"heading": "# Nope"})):
            with self.subTest(tool=name):
                out = getattr(tools, name)(**kwargs)
                self.assertNotIn("missing 1 required positional", str(out))

    def test_an_empty_note_is_refused_with_a_reason_not_a_crash(self):
        """A silent no-op here reads exactly like a saved note."""
        out = tools.memory_append(text="   ")
        self.assertIn("NOT SAVED", str(out))
        out = tools.memory_revise(old="")
        self.assertIn("NOT DONE", str(out))
        out = tools.memory_retire(heading="")
        self.assertIn("NOT DONE", str(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
