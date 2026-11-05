"""Bounded cleanup of scratch created while preparing asynchronous changes.

A monitoring agent commonly writes candidate files into a persistent scratch
directory, then hands their paths to a separate applier. That applier may run
minutes or hours later. Cleanup must therefore answer two independent questions:

* Is the entry old enough to be abandoned?
* Does any still-pending request name it?

A pending reference wins over age. Within an entry, the newest file's mtime is
what matters: an old directory containing a freshly edited file is active work,
not garbage. No application-specific directories or request formats are encoded
here; callers provide the scratch path and pending request contents.

Environment:
    AGENT_SCRATCH_DIR               scratch root (unset = no automatic cleanup)
    AGENT_SCRATCH_MAX_AGE_HOURS     age bound, default 168 (7 days)
    AGENT_SCRATCH_GRACE_HOURS       recent-write grace, default 48

A zero age bound is refused: it would delete a fix still being written. The
function returns evidence and never silently claims a deletion succeeded.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

DEFAULT_MAX_AGE_HOURS = 168.0
DEFAULT_GRACE_HOURS = 48.0


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; 0 would delete active work")
    return value


def _entry_age_seconds(path: Path, now: float) -> float:
    """Age since the NEWEST file in this entry, not just its directory."""
    newest = 0.0
    for candidate in (path, *path.rglob("*")):
        try:
            newest = max(newest, candidate.stat().st_mtime)
        except OSError:
            continue
    return max(0.0, now - newest)


def prune_scratch(*, now: float,
                  pending_request_ids: Iterable[str],
                  pending_request_texts: Iterable[str],
                  dry_run: bool = False) -> dict[str, Any]:
    """Prune top-level scratch entries older than the configured age bound.

    ``pending_request_ids`` are retained as audit labels. The actual protection
    searches pending request CONTENT for the entry's name: a request naming a
    nested source file still protects its parent entry, and one naming the whole
    scratch directory protects everything. A caller with unreadable requests
    should supply their raw text, not an empty list that would imply safety.
    """
    root_raw = os.environ.get("AGENT_SCRATCH_DIR", "").strip()
    if not root_raw:
        return {"available": False, "deleted": [], "kept": [],
                "protected": [], "would_delete": [], "errors": []}
    root = Path(root_raw).expanduser()
    if not root.is_dir():
        return {"available": True, "deleted": [], "kept": [], "protected": [],
                "would_delete": [], "errors": [], "note": "scratch directory absent"}

    max_age_hours = _positive_float(
        "AGENT_SCRATCH_MAX_AGE_HOURS", DEFAULT_MAX_AGE_HOURS)
    grace_hours = _positive_float(
        "AGENT_SCRATCH_GRACE_HOURS", DEFAULT_GRACE_HOURS)
    texts = [str(text) for text in pending_request_texts]
    ids = [str(item) for item in pending_request_ids]
    result: dict[str, Any] = {"available": True, "deleted": [], "kept": [],
                             "protected": [], "would_delete": [], "errors": [],
                             "pending_request_ids": ids}
    # Segment-boundary matching: a pending `fix/module.py` protects `fix`, but
    # `prefix/module.py` must not be protected by the same two-letter match.
    # Root protection is based on the COMPLETE path token, not on the basename
    # appearing somewhere: `work/fix/module.py` is not a reference to `work/`.
    root_posix = str(root).replace("\\", "/").rstrip("/")
    slash_paths = [
        match.group(0).replace("\\", "/").rstrip("/").rstrip('"\'')
        for text in texts
        for match in re.finditer(r"(?<![A-Za-z0-9_.-])[^\s\"']*/[^\s\"']*", text)
    ]
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        # Only direct children are units. This function never recurses into a
        # parent beyond the entry it just listed and judged.
        name_pattern = re.compile(r"(?<![A-Za-z0-9_.-])"
                                  + re.escape(path.name)
                                  + r"(?![A-Za-z0-9_.-])")
        root_is_named = any(
            token == root_posix
            or (root.name and token.endswith("/" + root.name))
            for token in slash_paths
        )
        protected = any(name_pattern.search(text) for text in texts) or root_is_named
        if protected:
            result["protected"].append(path.name)
            result["kept"].append(path.name)
            continue
        try:
            age = _entry_age_seconds(path, now)
        except OSError as exc:
            result["kept"].append(path.name)
            result["errors"].append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if age < grace_hours * 3600 or age <= max_age_hours * 3600:
            result["kept"].append(path.name)
            continue
        if dry_run:
            result["would_delete"].append(path.name)
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            if path.exists():
                raise OSError("path still exists after removal")
            result["deleted"].append(path.name)
        except OSError as exc:
            result["kept"].append(path.name)
            result["errors"].append(f"{path.name}: {type(exc).__name__}: {exc}")
    return result
