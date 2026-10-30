"""Precision editing and path search: the tools Copilot's edit UX is built on.

``write_file`` (whole-file overwrite) stays for new files, but a small change
to a large file through it is the single most common way an agent destroys
content it never read. ``replace_in_file`` requires the EXACT old text and
refuses when it matches zero or several places, so an edit either lands where
intended or reports why it could not -- there is no silent third outcome.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Dict, List

from config import PROJECT_ROOT
from tools_safety import failure_note

#: Directories that are never part of a search or a repo map: walking them
#: burns the tool timeout and returns noise.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".mypy_cache", ".pytest_cache", "dist", "build", ".idea",
              ".vscode", "sessions", ".aider.trash"}

#: Consecutive failed edit attempts per file. Three in a row means the model
#: is guessing at content it has not read; the note tells it to stop and roll
#: back instead of burning a fourth step on the same wrong anchor.
_EDIT_FAILS: Dict[str, int] = {}


def _fail(path: str, result: str) -> str:
    key = str(_resolve(path))
    count = _EDIT_FAILS.get(key, 0) + 1
    _EDIT_FAILS[key] = count
    return result + failure_note(key, count)


def _succeed(path: str) -> None:
    _EDIT_FAILS.pop(str(_resolve(path)), None)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / path)


def replace_in_file(path: str, old: str, new: str,
                    expected_replacements: int = 1) -> str:
    """Replace an EXACT string in a file.

    ``old`` must appear exactly ``expected_replacements`` times (default 1).
    Zero matches means the file changed under you -- re-read it. Several means
    the anchor is ambiguous -- include more surrounding lines. Nothing is
    written unless the count matches, so a failed edit cannot half-apply.
    """
    p = _resolve(path)
    if not p.is_file():
        return _fail(path, f"NOT FOUND: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"
    if not old:
        return "EMPTY old string -- refusing to insert blindly."
    count = text.count(old)
    if count != expected_replacements:
        return _fail(path, (
            f"NO EDIT APPLIED: the old string appears {count} time(s) in "
            f"{p.name}, expected {expected_replacements}. "
            + ("Re-read the file; it changed since you read it."
               if count == 0 else
               "Add more surrounding lines to make the anchor unique.")))
    p.write_text(text.replace(old, new), encoding="utf-8")
    _succeed(path)
    return f"EDITED {p} ({count} replacement(s))"


def multi_replace_in_file(path: str, edits: List[Dict[str, str]]) -> str:
    """Apply several exact replacements to one file, in order, atomically.

    Each edit is {"old": ..., "new": ...}. If ANY edit fails its uniqueness
    check the WHOLE batch is refused and the file is untouched -- a partial
    application is worse than none, because the model then reasons about a
    file state that never existed.
    """
    p = _resolve(path)
    if not p.is_file():
        return f"NOT FOUND: {p}"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"
    problems = []
    for i, edit in enumerate(edits or []):
        old = (edit or {}).get("old", "")
        if not old:
            problems.append(f"edit {i}: empty old string")
            continue
        if text.count(old) != 1:
            problems.append(
                f"edit {i}: old string appears {text.count(old)} time(s)")
    if problems:
        return _fail(path, "NO EDITS APPLIED (atomic batch): "
                     + "; ".join(problems))
    for edit in edits:
        text = text.replace(edit["old"], edit.get("new", ""), 1)
    p.write_text(text, encoding="utf-8")
    _succeed(path)
    return f"EDITED {p} ({len(edits)} replacement(s))"


def glob_files(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files by path pattern, e.g. '**/*.py' or 'engine/validate_*.py'.

    Use this to discover WHERE things live before reading them; use search()
    for content. Skips VCS and cache directories so a repo-wide glob stays
    fast.
    """
    root = _resolve(path)
    if not root.exists():
        return f"NOT FOUND: {root}"
    out: List[str] = []
    try:
        for p in root.glob(pattern):
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            out.append(str(p))
            if len(out) >= max_results:
                break
    except (ValueError, OSError) as exc:
        return f"BAD PATTERN: {exc}"
    if not out:
        return f"(no files match {pattern!r} under {root})"
    out.sort()
    body = "\n".join(out)
    more = "" if len(out) < max_results else f"\n... (capped at {max_results})"
    return f"{len(out)} file(s):\n{body}{more}"


def matches_skip(path: Path) -> bool:
    """Shared skip test for repo map and search walkers."""
    return any(part in _SKIP_DIRS for part in path.parts)


def fnmatch_any(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)
