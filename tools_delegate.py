"""Subagents: hand a self-contained subtask to a fresh, bounded agent.

Some work is a detour -- "find every caller of X", "summarise this 4000-line
log" -- that floods the parent's context with noise. Copilot solves this with
subagents; here a subagent is a child ``run.py`` process with its own session,
its own step budget and its own memory writes, returning only its final
answer. Depth is capped so a subagent cannot spawn an army.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT

MAX_DEPTH = int(os.environ.get("AGENT_MAX_SUBAGENT_DEPTH", "2"))
SUBAGENT_TIMEOUT_SEC = int(os.environ.get("AGENT_SUBAGENT_TIMEOUT_SEC", "900"))


def run_subagent(prompt: str, steps: int = 12) -> str:
    """Run a bounded child agent and return ONLY its final answer.

    The child inherits the environment (same model, same memory tree) but gets
    a fresh session and a small step budget. Use it for research or
    summarisation that would otherwise fill your own context; do the deciding
    yourself afterwards -- a subagent's answer is evidence, not a verdict.
    """
    depth = int(os.environ.get("AGENT_DEPTH", "0"))
    if depth + 1 > MAX_DEPTH:
        return (f"REFUSED: subagent depth {depth + 1} exceeds "
                f"AGENT_MAX_SUBAGENT_DEPTH={MAX_DEPTH}. Do the work yourself.")
    if not (prompt or "").strip():
        return "REFUSED: empty subagent prompt."
    env = dict(os.environ)
    env["AGENT_DEPTH"] = str(depth + 1)
    env["AGENT_MAX_STEPS"] = str(max(2, int(steps)))
    env["AGENT_CONFIRM_WRITES"] = "0"      # child is read-only by instruction
    env["AGENT_CONFIRM_PYTHON_MUTATIONS"] = "0"
    entry = Path(__file__).resolve().parent / "run.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(entry), "--quiet", prompt],
            cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True,
            timeout=SUBAGENT_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return (f"SUBAGENT TIMEOUT after {SUBAGENT_TIMEOUT_SEC}s; it was "
                f"killed. Do the work yourself or use a smaller scope.")
    except Exception as exc:  # noqa: BLE001
        return f"SUBAGENT ERROR: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and not out:
        return f"SUBAGENT FAILED (exit {proc.returncode}): {err[-1500:]}"
    tail = out[-12000:]
    note = f"\n[child exit={proc.returncode}]"
    return tail + note
