"""Bounds on everything this harness writes, so a long-running one cannot fill a disk.

Two files grow forever by construction and neither had a limit:

``AGENT_TRANSCRIPT``
    A single JSONL, opened in append mode for the life of the process. On a
    deployment where the console and the mission scheduler run side by side it
    is never closed, and it reached 6.3 MB in two days -- about 1 GB a year
    from a box whose whole system disk is 45 GB.

``AGENT_SESSIONS_DIR``
    Two files per run, and runs are scheduled. Nothing ever removed one. At
    ~12 runs a day that is thousands of pairs a year, and the pairs are read
    by nobody once the next run has started.

Rotation is NOT deletion of anything the operator asked for. The transcript is
the raw record of "what did the API see" and a session log is "what did the
agent do"; both are kept, just not forever, and the readable twin of every run
(``latest.md``) always survives. Memory is deliberately NOT bounded here: it is
knowledge rather than a log, its token cost is already capped by
``MEMORY_HEAD_LINES``, and silently dropping notes would be the exact failure
mode memory exists to prevent. What memory gets is a REPORT, so growth is
visible instead of something discovered by a full disk.

Every function here fails OPEN -- housekeeping must never cost a run -- but it
reports the failure through ``on_warn`` rather than swallowing it. A cleanup
that raises would abort a mission for tidying's sake; one that fails silently
is the 51-broad-excepts class of defect this project has already paid for.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

#: Names inside a sessions dir that are not a run's own pair.
_SESSION_KEEP = frozenset({"latest.md"})

Warn = Callable[[str], None]


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _mb(n: int | float) -> str:
    return f"{n / (1 << 20):.1f} MB"


# --------------------------------------------------------------------------
# session logs
# --------------------------------------------------------------------------

def session_stems(root: Path) -> List[str]:
    """The run stamps present in ``root``, newest first.

    A run writes ``<stamp>_<id>.jsonl`` and ``<stamp>_<id>.md``, so the pair is
    keyed on the shared stem and deleted together. Half a pair left behind is
    the worst outcome of a naive prune: a ``.md`` with no event stream, or a
    ``.jsonl`` nothing renders.

    Names only, not ``(mtime, name)`` tuples. The first version of this returned
    tuples and the caller unpacked them the other way round, so the pruner went
    looking for a file called ``1050.0.jsonl`` and deleted nothing while
    recording success -- a retention rule that silently retains. A value only
    one caller needs should not carry a shape that reads two ways.
    """
    stamps: Dict[str, float] = {}
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return []
    for p in entries:
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        if p.name in _SESSION_KEEP or p.name.endswith(".tmp"):
            continue
        if p.suffix not in (".jsonl", ".md"):
            continue
        stem = p.stem
        mtime = _mtime(p)
        if mtime > stamps.get(stem, 0.0):
            stamps[stem] = mtime
    return [stem for stem, _at in sorted(stamps.items(),
                                        key=lambda kv: kv[1], reverse=True)]


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def prune_sessions(root: Path, keep: int, max_bytes: int,
                   on_warn: Optional[Warn] = None) -> Dict[str, object]:
    """Delete old session pairs until both budgets are satisfied.

    ``keep`` is a COUNT of runs (<= 0 = unlimited); ``max_bytes`` is a total
    budget across the directory (<= 0 = unlimited). Both are applied, count
    first, because they answer different questions: the count bounds file
    explosion on a box that runs a mission every half hour, the byte budget
    bounds one pathological run that logged enormous tool output.

    The newest pair is always left alone. It is the run that just finished, or
    the one in progress, and a retention rule that can delete the log of the
    task currently running is a rule nobody would keep.
    """
    warn = on_warn or (lambda _msg: None)
    out: Dict[str, object] = {"removed": 0, "freed_bytes": 0,
                              "kept": 0, "errors": []}
    root = Path(root)
    if not root.is_dir():
        return out

    stems = session_stems(root)
    if keep is not None and keep > 0:
        doomed = stems[keep:]
        stems = stems[:keep]
    else:
        doomed = []

    def total_bytes(items: List[str]) -> int:
        n = 0
        for stem in items:
            for suffix in (".jsonl", ".md"):
                n += _safe_size(root / f"{stem}{suffix}")
        return n

    if max_bytes and max_bytes > 0:
        size = total_bytes(stems)
        #: Oldest first -- which, with stems newest-first, is the tail.
        while size > max_bytes and len(stems) > 1:
            doomed.append(stems.pop())
            size = total_bytes(stems)

    for stem in doomed:
        for suffix in (".jsonl", ".md"):
            p = root / f"{stem}{suffix}"
            if not p.exists():
                continue
            try:
                freed = _safe_size(p)
                p.unlink()
                out["removed"] = int(out["removed"]) + 1
                out["freed_bytes"] = int(out["freed_bytes"]) + freed
            except OSError as exc:
                msg = f"could not remove {p.name}: {exc}"
                out["errors"].append(msg)
                warn(msg)

    out["kept"] = len(stems)
    return out


# --------------------------------------------------------------------------
# transcript
# --------------------------------------------------------------------------

def rotate_transcript(path: str | Path, max_bytes: int, backups: int = 2,
                      on_warn: Optional[Warn] = None) -> Optional[Path]:
    """Rotate an oversized JSONL transcript; return the path it moved to.

    Called BEFORE the handle is opened, so the process starts appending to a
    fresh file rather than racing a rename behind an open descriptor (which on
    POSIX keeps writing into the renamed inode -- the rotation would appear to
    work while the live file grew on).

    ``backups`` is floored at 1: a rotation that discards the only existing
    record is a deletion, and this function's contract is that history is
    kept, not thrown away. ``.1`` is newest, ``.N`` oldest, and ``.N`` is
    dropped when the shift would overflow it.
    """
    warn = on_warn or (lambda _msg: None)
    if not path:
        return None
    if not max_bytes or max_bytes <= 0:
        return None
    backups = max(1, int(backups or 1))
    p = Path(path)
    try:
        if not p.is_file():
            return None
        if p.stat().st_size <= max_bytes:
            return None
        overflow = p.with_name(f"{p.name}.{backups}")
        if overflow.exists():
            overflow.unlink()
        for i in range(backups - 1, 0, -1):
            src = p.with_name(f"{p.name}.{i}")
            if src.exists():
                src.replace(p.with_name(f"{p.name}.{i + 1}"))
        p.replace(p.with_name(f"{p.name}.1"))
        return p.with_name(f"{p.name}.1")
    except OSError as exc:
        #: Fail open: a transcript that cannot be rotated still works, it just
        #: keeps growing -- and the operator is told, so it is a fixable
        #: warning rather than a disk that fills without explanation.
        warn(f"could not rotate {p.name}: {exc}")
        return None


# --------------------------------------------------------------------------
# the whole tree: what is using the space
# --------------------------------------------------------------------------

def dir_bytes(root: Path) -> int:
    """Total size of a directory tree. Returns 0 for anything unreadable."""
    root = Path(root)
    if not root.exists():
        return 0
    if root.is_file():
        return _safe_size(root)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            total += _safe_size(Path(dirpath) / name)
    return total


def report(root: Path, memory_soft_max_bytes: int = 0,
           on_warn: Optional[Warn] = None) -> Dict[str, object]:
    """A one-call size report of the agent's own data root.

    Logged at the start of every run, so growth shows up as a trend across
    session logs instead of arriving as "No space left on device" at the
    moment a mission needs to write one. Memory over its soft budget is a
    warning HERE, and nowhere else in the harness -- trimming notes is a
    knowledge decision for a human or an explicit ``memory_retire``, not
    something a cleanup routine does quietly.
    """
    warn = on_warn or (lambda _msg: None)
    root = Path(root)
    sizes: Dict[str, int] = {}
    memories = root / "memories"
    sizes["memories"] = dir_bytes(memories)
    sizes["sessions"] = dir_bytes(root / "sessions")
    for name in ("transcript.jsonl",):
        p = root / name
        if p.is_file():
            sizes[name] = _safe_size(p)
        for i in range(1, 10):
            b = root / f"{name}.{i}"
            if not b.is_file():
                break
            sizes[b.name] = _safe_size(b)
    others = {n: dir_bytes(root / n) for n in sorted_ignored(root, sizes)}
    sizes.update(others)

    warnings: List[str] = []
    if memory_soft_max_bytes and memory_soft_max_bytes > 0:
        if sizes.get("memories", 0) > memory_soft_max_bytes:
            msg = (f"memory is {_mb(sizes['memories'])}, over its "
                   f"{_mb(memory_soft_max_bytes)} soft budget -- retire or "
                   f"consolidate notes; nothing is trimmed automatically")
            warnings.append(msg)
            warn(msg)
    return {"sizes": sizes,
            "total_bytes": sum(sizes.values()),
            "warnings": warnings}


def sorted_ignored(root: Path, already: Dict[str, int]) -> List[str]:
    """Top-level entries of ``root`` not already measured, as file/dir names.

    Named for what it does rather than ``_rest``: the point is that a new kind
    of agent-local file appears in the report the day it starts being written,
    instead of growing in a directory nobody measures.
    """
    try:
        entries = list(Path(root).iterdir())
    except OSError:
        return []
    out = []
    for p in entries:
        name = p.name
        if name in already or name in ("memories", "sessions"):
            continue
        #: Rotated transcripts are measured above by their numbered names.
        if name.startswith("transcript.jsonl."):
            continue
        out.append(name)
    return sorted(out)


def one_line(summary: Dict[str, object]) -> str:
    """Human-readable form of :func:`report`, for a log line."""
    sizes: Dict[str, int] = dict(summary.get("sizes") or {})  # type: ignore[arg-type]
    parts = [f"{k}={_mb(v)}" for k, v in sorted(sizes.items()) if v]
    total = _mb(int(summary.get("total_bytes") or 0))
    return f"agent-data {total}: " + ", ".join(parts[:6])
