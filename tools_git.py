"""Git awareness: the verbs a coding agent actually needs, guarded.

Copilot-style flows lean on git for three things: knowing what changed
(``status``/``diff``/``log``), recording work (``commit``), and undoing a bad
edit (``restore``). The destructive verbs (``reset --hard``, ``clean -f``,
force-push) are gated in :mod:`tools_safety` so "fix everything" can never
become "discard everything".
"""
from __future__ import annotations

import subprocess
from typing import List, Optional

from config import PROJECT_ROOT
from tools_safety import git_guard


def _git(args: List[str], timeout: int = 60) -> str:
    ok, why = git_guard(" ".join(args))
    if not ok:
        return why
    try:
        proc = subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s: git {' '.join(args)}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR running git: {type(exc).__name__}: {exc}"
    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
    return f"exit={proc.returncode}\n{out.strip() or '(no output)'}"


def git_status() -> str:
    """Working-tree state: branch, staged/unstaged/untracked files.

    Run this BEFORE editing anything so you know the starting point, and
    after edits so the diff you describe is the diff that exists.
    """
    return _git(["status", "--short", "--branch"])


def git_diff(staged: bool = False, path: str = "") -> str:
    """The current diff. ``staged=True`` limits to the index; ``path`` scopes
    to one file. Use after editing to verify WHAT changed, not just that
    something did."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    return _git(args)


def git_log(limit: int = 15, path: str = "") -> str:
    """Recent commits (oneline + date). Use to find the last known-good
    revision before rolling a file back."""
    args = ["log", f"--max-count={max(1, int(limit))}",
            "--pretty=%h %ad %s", "--date=short"]
    if path:
        args += ["--", path]
    return _git(args)


def git_show(revision: str, path: str = "") -> str:
    """Show one commit (or ``rev:path`` for a file at a revision)."""
    args = ["show", "--stat", "--patch", revision]
    if path:
        args += ["--", path]
    return _git(args)


def git_commit(message: str, paths: Optional[List[str]] = None) -> str:
    """Stage ``paths`` (or everything when omitted) and commit.

    Committing is a CAUTION-class action: it is recorded and narrated but not
    refused. Write a message that says WHY, not just WHAT.
    """
    if not (message or "").strip():
        return "REFUSED: empty commit message."
    if paths:
        staged = _git(["add", "--", *paths])
    else:
        staged = _git(["add", "-A"])
    if staged.startswith(("REFUSED", "ERROR", "TIMEOUT")):
        return staged
    return _git(["commit", "-m", message])


def git_restore(path: str, from_revision: str = "HEAD") -> str:
    """Roll ONE file back to a revision (default HEAD) -- the safe undo.

    This is the recovery path the safety rails point at after three failed
    edits of the same file. It discards that file's uncommitted changes, so
    it is reported as a caution, never run silently.
    """
    if not (path or "").strip():
        return "REFUSED: git_restore needs a path."
    return _git(["restore", "--source", from_revision, "--", path])
