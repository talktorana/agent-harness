"""A persistent Python interpreter for the agent.

Why a separate process rather than ``exec`` in the harness process: a hard
timeout has to be able to KILL the running code, and you cannot kill a thread.
A ``while True`` in generated analysis code would otherwise hang the whole
monitoring pass. A subprocess also keeps a broken ``sys.path`` or a poisoned
``sys`` module out of the harness itself.

The interpreter SURVIVES across calls within one task, so the model can load a
JSONL once and then aggregate, filter and diff it in later steps instead of
re-reading the data every time. If the process is killed for a timeout the
namespace is lost and the next call starts fresh -- that is reported honestly
rather than hidden.

Protocol (one JSON object per line each way):
    parent -> child   {"id": int, "code": str}
    child  -> parent  {"id": int, "stdout": str, "stderr": str,
                       "error": str | None, "value": str | None,
                       "names": [str]}

The child reserves its OWN stdout for this protocol and swaps ``sys.stdout`` /
``sys.stderr`` for buffers while user code runs, so a stray ``print`` cannot
corrupt the channel.
"""

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

from config import (
    PROJECT_ROOT,
    PYTHON_MAX_OUTPUT,
    PYTHON_TIMEOUT_SEC,
)

_SENTINEL = "\x00HARNESS\x00"

#: Runs inside the child. Kept as a string so there is no second file to ship
#: and no import path to get wrong.
_BOOTSTRAP = r'''
import ast, io, json, sys, traceback

PROJECT_ROOT = sys.argv[1]
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_NS = {"__name__": "__agent__", "__builtins__": __builtins__}
_PROTO = sys.stdout          # reserved channel; user code must not see it


def _run(code):
    """Execute one block, returning the value of a trailing expression.

    Splitting the last statement out matters more than it looks: the natural
    way to write analysis code is imports first and the thing you want to see
    last, and an ``exec`` of the whole block silently throws that value away.
    Without this the model gets "(ran cleanly, no output)" for the number it
    was actually asking for.
    """
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    value = None
    error = None
    try:
        tree = ast.parse(code, "<agent>", "exec")
        tail = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = tree.body[-1].value
            tree.body = tree.body[:-1]
        if tree.body:
            exec(compile(tree, "<agent>", "exec"), _NS)
        if tail is not None:
            expr = ast.Expression(body=tail)
            ast.fix_missing_locations(expr)
            value = eval(compile(expr, "<agent>", "eval"), _NS)
    except BaseException:
        error = traceback.format_exc(limit=12)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return out.getvalue(), err.getvalue(), error, value


def _names():
    return sorted(n for n in _NS
                  if not n.startswith("_") and n != "__builtins__")


while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    stdout, stderr, error, value = _run(req.get("code", ""))
    vrepr = None
    if value is not None:
        try:
            vrepr = repr(value)
        except Exception:
            vrepr = "<unreprable " + type(value).__name__ + ">"
        if vrepr and len(vrepr) > 2000:
            vrepr = vrepr[:2000] + " ...[repr truncated]"
    resp = {
        "id": req.get("id"),
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "value": vrepr,
        "names": _names(),
    }
    _PROTO.write(json.dumps(resp) + "\n")
    _PROTO.flush()
'''


class PythonSandbox:
    """A long-lived child interpreter with a persistent namespace."""

    def __init__(self, timeout_sec: int = None, max_output: int = None):
        self.timeout = timeout_sec or PYTHON_TIMEOUT_SEC
        self.max_output = max_output or PYTHON_MAX_OUTPUT
        self.proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._next_id = 0
        self._restarts = 0

    # ------------------------------------------------------------- lifecycle

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _BOOTSTRAP, str(PROJECT_ROOT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        """Move child stdout lines onto the queue so reads can be timed out."""
        try:
            for line in self.proc.stdout:
                self._queue.put(line)
        except Exception:  # noqa: BLE001 -- process died; thread just ends
            pass

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ run

    def _truncate(self, text: str) -> str:
        """Keep the tail: tracebacks and error messages live at the end."""
        if len(text) <= self.max_output:
            return text
        head = self.max_output // 4
        tail = self.max_output - head
        return (text[:head]
                + f"\n\n... [truncated {len(text) - self.max_output} chars] ...\n\n"
                + text[-tail:])

    def _request(self, code: str, timeout: int) -> str:
        """Send one block and wait for its reply line."""
        self._next_id += 1
        payload = json.dumps({"id": self._next_id, "code": code}) + "\n"
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

        deadline_left = float(timeout)
        while True:
            try:
                line = self._queue.get(timeout=deadline_left)
            except queue.Empty:
                return ""  # caller treats "" as a timeout
            stripped = line.strip()
            if not stripped:
                continue
            try:
                resp = json.loads(stripped)
            except json.JSONDecodeError:
                # User code reached sys.__stdout__ and wrote junk. Skip it and
                # keep waiting for the real reply rather than losing the task.
                continue
            if isinstance(resp, dict) and resp.get("id") == self._next_id:
                return resp

    def run(self, code: str, timeout_sec: int = None) -> str:
        """Execute a block and return a legible report of what happened."""
        timeout = timeout_sec or self.timeout

        if self.proc is None or self.proc.poll() is not None:
            self._spawn()

        resp = self._request(code, timeout)

        if resp == "":  # timed out, or the child vanished mid-request
            alive = self.proc.poll() is None
            self.close()
            self.proc = None
            self._restarts += 1
            return (
                f"TIMEOUT after {timeout}s (or the interpreter died).\n"
                f"The persistent namespace was DISCARDED and a fresh one is "
                f"now running -- any variables you had built up are gone.\n"
                f"interpreter was {'still alive' if alive else 'already dead'} "
                f"before restart. Restart #{self._restarts}.\n"
                f"Fix the code (add a bound to a loop, read less data) and "
                f"re-run the parts you still need."
            )

        parts: List[str] = []
        if resp.get("stdout"):
            parts.append("[stdout]\n" + resp["stdout"])
        if resp.get("stderr"):
            parts.append("[stderr]\n" + resp["stderr"])
        if resp.get("value") is not None:
            parts.append("[value] " + str(resp["value"]))
        if resp.get("error"):
            parts.append("[exception]\n" + resp["error"])
        if not parts:
            parts.append("(ran cleanly, no output)")
        names = resp.get("names") or []
        if names:
            shown = ", ".join(names[:40])
            more = "" if len(names) <= 40 else f", ... (+{len(names) - 40})"
            parts.append(f"[in memory] {shown}{more}")
        return self._truncate("\n".join(parts))
