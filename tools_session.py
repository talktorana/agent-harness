"""Session history: remember what previous runs of this harness did.

Copilot keeps a searchable history of every chat. The harness already writes
one readable transcript per task (``sessions/*.md`` plus the machine-readable
``*.jsonl``); these tools make that history *usable* by the agent itself, so
"have we done this before?" is a lookup instead of a re-derivation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from config import SESSIONS_DIR


def _files(suffix: str) -> List[Path]:
    root = Path(SESSIONS_DIR)
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*{suffix}"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def session_list(limit: int = 20) -> str:
    """Most recent sessions, newest first: name, size, mtime.

    The name encodes the start time and session id, so this is also the
    timeline of what the harness has been asked to do.
    """
    files = _files(".md")[:max(1, int(limit))]
    if not files:
        return f"(no sessions yet in {SESSIONS_DIR})"
    rows = []
    for p in files:
        st = p.stat()
        rows.append(f"{p.name}  {st.st_size} bytes")
    return "\n".join(rows)


def session_read(name: str, max_chars: int = 8000) -> str:
    """Read one session transcript (the .md name, with or without suffix).

    Returns the HEAD and TAIL: the task and first actions say what was
    attempted, the tail says how it ended. The middle is reachable with
    session_search.
    """
    root = Path(SESSIONS_DIR)
    p = root / name if name.endswith((".md", ".jsonl")) else root / f"{name}.md"
    if not p.is_file():
        return f"NOT FOUND: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"
    limit = max(500, int(max_chars))
    if len(text) <= limit:
        return text
    half = limit // 2
    return (text[:half]
            + f"\n\n... [{len(text) - limit} chars omitted; use "
              f"session_search to find what is in the middle] ...\n\n"
            + text[-half:])


def session_search(pattern: str, limit: int = 30) -> str:
    """Search every session transcript for text or a regex.

    Returns ``file:line: text`` hits, newest sessions first. Use it for
    "what did we try last time this failed?" -- the answer is usually already
    written down.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    hits: List[str] = []
    for p in _files(".md") + _files(".jsonl"):
        try:
            lines = p.read_text(encoding="utf-8",
                                errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n... (capped at {limit})"
    return "\n".join(hits) if hits else "(no matches in any session)"
