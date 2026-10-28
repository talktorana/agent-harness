"""Agentic memory: three scopes, loaded automatically, written back by the agent.

Mirrors the editor's memory layout so the harness and Copilot share ONE set of
notes instead of accumulating two that drift apart:

    user.md        cross-project preferences and hard-won lessons  (persistent)
    repo/<project>.md  project facts, conventions, incidents, fixes (persistent)
    session/*.md   notes for the current task only                 (ephemeral)

Two properties make this "agentic" rather than a scratch file:

  1. The agent does not have to remember to READ. The head of the repo notes
     goes in front of it at the start of every task, and the prompt tells it
     how to search the rest.

  2. It WRITES BACK before finishing. Lessons that are recorded once stop
     being re-derived every session -- which is the whole difference between a
     harness that runs a task and one that gets better at running them.

Writes PREPEND a section rather than appending to the end: the file is read
from the top, so the newest, most relevant material is what loads first.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    MEMORY_HEAD_LINES,
    MEMORY_REPO_FILE,
    MEMORY_ROOT,
    MEMORY_SESSION_DIR,
    MEMORY_USER_FILE,
)


class MemoryStore:
    """Read/write access to the memory tree, scoped."""

    SCOPES = ("repo", "session", "user")

    def __init__(self, session_id: str = "adhoc", root: Path = None,
                 repo_file: Path = None):
        self.session_id = session_id
        self.root = Path(root) if root else Path(MEMORY_ROOT)
        self.repo_file = Path(repo_file) if repo_file else Path(MEMORY_REPO_FILE)
        self.user_file = self.root / "user.md"
        self.session_dir = self.root / "session"

    # ------------------------------------------------------------- paths

    def path_for(self, scope: str) -> Path:
        if scope == "repo":
            return self.repo_file
        if scope == "user":
            return self.user_file
        if scope == "session":
            return self.session_dir / f"{self.session_id}.md"
        raise ValueError(f"unknown memory scope {scope!r}; use one of {self.SCOPES}")

    def _resolve(self, scope: str) -> Path:
        scope = (scope or "repo").strip().lower()
        if scope not in self.SCOPES:
            raise ValueError(f"unknown memory scope {scope!r}; use one of {self.SCOPES}")
        return self.path_for(scope)

    def _checked(self, scope: str):
        """Resolve a scope, or return the reason it could not be.

        A typo must say so. "Unknown scope" and "no such note" are different
        answers and only one of them is true -- silently returning empty here
        is what made a search report "no matches" when the caller had simply
        looked in the wrong place.
        """
        try:
            return self._resolve(scope), None
        except ValueError as exc:
            return None, str(exc)

    # -------------------------------------------------------------- read

    def read(self, scope: str = "repo", start_line: int = 1,
             end_line: int = 200) -> str:
        """Numbered lines, like read_file, so later edits can be precise."""
        p, err = self._checked(scope)
        if err:
            return err
        if not p.exists():
            return (f"NO MEMORY YET at {p} ({scope} scope).\n"
                    f"This is normal on a first run -- use memory_append to "
                    f"start it.")
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"ERROR reading {p}: {exc}"
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line))
        if start > end:
            return f"EMPTY RANGE: {p} has {len(lines)} lines"
        body = "\n".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start - 1, end))
        more = "" if end >= len(lines) else f"\n... ({len(lines) - end} more lines)"
        return f"{p} ({len(lines)} lines)\n{body}{more}"

    def head(self, scope: str = "repo", n_lines: int = None) -> str:
        """The top of a scope, for automatic loading into the prompt.

        Degrades to empty rather than raising: this runs while the system
        prompt is being assembled, and a bad scope there would abort the task
        before the model got a chance to do anything useful.
        """
        n = n_lines or MEMORY_HEAD_LINES
        try:
            p = self._resolve(scope)
        except ValueError:
            return ""
        if not p.exists():
            return ""
        try:
            lines = p.read_text(encoding="utf-8",
                                errors="replace").splitlines()[:n]
        except OSError:
            return ""
        return "\n".join(lines).strip()

    # ------------------------------------------------------------ search

    def search(self, pattern: str, scope: str = None,
               max_results: int = 30) -> str:
        """Find text across memory. Pure Python -- no grep, which is absent on
        Windows and would silently report 'no matches'.

        A mistyped scope must say so rather than returning an empty result:
        "no matches" and "you looked in the wrong place" are different answers
        and only one of them is true.
        """
        if scope is not None and str(scope).strip():
            want = str(scope).strip().lower()
            if want not in self.SCOPES:
                return (f"unknown memory scope {scope!r}; use one of "
                        f"{self.SCOPES}, or omit scope to search them all.")
            scopes = [want]
        else:
            scopes = list(self.SCOPES)
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))

        out: List[str] = []
        for sc in scopes:
            p = self.path_for(sc)
            if not p.is_file():
                continue
            try:
                lines = p.read_text(encoding="utf-8",
                                    errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    out.append(f"[{sc}] {p.name}:{i}: {line.strip()[:200]}")
                    if len(out) >= max_results:
                        return "\n".join(out) + f"\n... (capped at {max_results})"
        return "\n".join(out) if out else "(no matches in memory)"

    # ------------------------------------------------------------- write

    def append(self, scope: str, text: str, heading: str = None) -> str:
        """Prepend a titled section to a scope. Never overwrites existing notes.

        Prepending (rather than appending) keeps the file newest-first, which
        is what makes the automatic head-loading useful: the freshest lessons
        are the ones the next task sees without asking.
        """
        body = (text or "").strip()
        if not body:
            return "EMPTY note -- nothing written. Give the note some content."
        p, err = self._checked(scope)
        if err:
            return err
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"ERROR creating {p.parent}: {exc}"

        title = (heading or "").strip().lstrip("#").strip()
        if not title:
            title = f"NOTE {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        block = f"# {title}\n\n{body}\n"

        existing = ""
        if p.exists():
            try:
                existing = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return f"ERROR reading {p} before writing: {exc}"

        try:
            p.write_text(block + "\n" + existing, encoding="utf-8")
        except OSError as exc:
            return f"ERROR writing {p}: {exc}"
        return (f"RECORDED to {scope} memory: {p}\n"
                f"({len(body.splitlines())} lines prepended as '{title}'; "
                f"file is now {len((block + existing).splitlines())} lines)")

    # ------------------------------------------------- revise and retire

    def _backup(self, p: Path) -> Optional[str]:
        """Snapshot before a destructive edit so a wrong rewrite is undoable.

        One rolling ``.bak``, not a growing archive: the need is to recover
        from a bad edit made moments ago. History is git's job.
        """
        try:
            bak = p.with_suffix(p.suffix + ".bak")
            bak.write_text(p.read_text(encoding="utf-8", errors="replace"),
                           encoding="utf-8")
            return str(bak)
        except OSError:
            return None

    def revise(self, scope: str, old: str, new: str) -> str:
        """Correct a note in place instead of stacking a correction on top.

        ``old`` must appear EXACTLY ONCE. Zero matches is a miss; more than one
        is a guess about which to change, and changing the wrong one is worse
        than changing none -- so both are refused and the caller is told what
        to do about it. Set ``new`` to empty to delete the matched text.
        """
        p, err = self._checked(scope)
        if err:
            return err
        old = old or ""
        if not old.strip():
            return ("REFUSED: 'old' is empty. Name the exact text to replace -- "
                    "use memory_read to see it first.")
        if not p.is_file():
            return f"NO MEMORY YET at {p} ({scope} scope) -- nothing to revise."
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR reading {p}: {exc}"

        count = text.count(old)
        if count == 0:
            near = [ln for ln in text.splitlines()
                    if old.splitlines()[0].strip()[:40]
                    and old.splitlines()[0].strip()[:40] in ln][:5]
            hint = ""
            if near:
                hint = ("\nLines containing the first words of 'old':\n  "
                        + "\n  ".join(n.strip()[:120] for n in near))
            return (f"NOT FOUND in {scope} memory: the text does not appear "
                    f"exactly. Copy it from memory_read -- whitespace and "
                    f"punctuation must match.{hint}")
        if count > 1:
            return (f"AMBIGUOUS: that text appears {count} times in {scope} "
                    f"memory. Include enough surrounding lines to make it "
                    f"unique, then try again. Nothing was changed.")

        updated = text.replace(old, new or "", 1)
        bak = self._backup(p)
        try:
            p.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"ERROR writing {p}: {exc}"
        before, after = len(text.splitlines()), len(updated.splitlines())
        return (f"REVISED {scope} memory: {p}\n"
                f"Replaced {len(old)} chars with {len(new or '')} chars; "
                f"{before} -> {after} lines."
                + (f"\nPrevious version saved to {bak}" if bak else ""))

    def retire(self, scope: str, heading: str) -> str:
        """Remove one whole titled note from a scope.

        Use when a note is simply wrong or superseded. This is the difference
        between memory that accumulates knowledge and memory that accumulates
        clutter: append-only leaves every bad conclusion in the file forever,
        still readable, still wrong.
        """
        p, err = self._checked(scope)
        if err:
            return err
        title = (heading or "").strip().lstrip("#").strip()
        if not title:
            return "REFUSED: give the heading of the note to retire."
        if not p.is_file():
            return f"NO MEMORY YET at {p} ({scope} scope) -- nothing to retire."
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines(True)
        except OSError as exc:
            return f"ERROR reading {p}: {exc}"

        starts = [i for i, ln in enumerate(lines)
                  if ln.startswith("# ") and ln[2:].strip() == title]
        if not starts:
            top = [ln.strip() for ln in lines if ln.startswith("# ")][:12]
            hint = ("\nTop-level notes in this scope:\n  " + "\n  ".join(top)
                    if top else "")
            return (f"NOT FOUND: no note titled '# {title}' in {scope} memory. "
                    f"Headings are matched exactly (one '#' plus the title)."
                    f"{hint}")
        if len(starts) > 1:
            return (f"AMBIGUOUS: {len(starts)} notes share the heading "
                    f"'# {title}'. Nothing was changed.")

        begin = starts[0]
        end = len(lines)
        for j in range(begin + 1, len(lines)):
            if lines[j].startswith("# "):
                end = j
                break
        removed = "".join(lines[begin:end])
        updated = "".join(lines[:begin] + lines[end:])
        bak = self._backup(p)
        try:
            p.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return f"ERROR writing {p}: {exc}"
        return (f"RETIRED '{title}' from {scope} memory: {p}\n"
                f"Removed {end - begin} lines; file is now "
                f"{len(updated.splitlines())} lines."
                + (f"\nPrevious version saved to {bak}" if bak else ""))

    # ------------------------------------------------------------ summary

    def summary(self) -> str:
        rows = []
        for sc in self.SCOPES:
            try:
                p = self._resolve(sc)
            except ValueError:
                continue
            if p.is_file():
                try:
                    stat = p.stat()
                    n = len(p.read_text(encoding="utf-8",
                                        errors="replace").splitlines())
                    when = datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")
                except OSError:
                    n, when = -1, "?"
                rows.append(f"  {sc:8} {n:6} lines  updated {when}  {p}")
            else:
                rows.append(f"  {sc:8}   (none)                     {p}")
        return ("Memory tree at " + str(self.root) + "\n" + "\n".join(rows)
                + "\n\nScopes: 'repo' = project knowledge (the one that matters), "
                  "'session' = this task only, 'user' = cross-project preferences.")
