"""Tool registry: JSON schemas for the model, handlers for the harness.

The DESCRIPTIONS matter more than the code. The model chooses tools from the
description alone, so a vague one produces vague behaviour. Each description
says when to use the tool AND what the result will look like, because the
model's next decision depends on knowing what it will get back.

Handlers always return a STRING and never raise -- a failed tool call is data
the model should react to, not a crash that kills the loop.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from config import (CONFIRM_PYTHON_MUTATIONS, CONFIRM_WRITES, MAX_TOOL_CHARS,
                    PROJECT_ROOT, TOOL_TIMEOUT_SEC)
from memory import MemoryStore
from python_exec import PythonSandbox
from tools_delegate import run_subagent
from tools_edit import glob_files, multi_replace_in_file, replace_in_file
from tools_browser import (browser_click, browser_close, browser_navigate,
                           browser_open, browser_read, browser_screenshot,
                           browser_snapshot, browser_type)
from tools_git import (git_commit, git_diff, git_log, git_restore, git_show,
                       git_status)
from tools_media import view_image
from tools_repo_map import repo_map
from tools_safety import guard_command
from tools_session import session_list, session_read, session_search
from tools_todo import manage_todo_list, todo_show

#: Prefer an explicit bash. On Windows ``shell=True`` gives CMD, which
#: understands none of the project's tooling -- pipes, ``$(...)``, ``&&`` --
#: and fails silently enough that a wrong answer is possible. Found by testing
#: this harness rather than by reading it.
BASH = shutil.which("bash")


def _detect_shell():
    """Identify the shell run_command will actually use.

    Windows can hold several. ``shutil.which("bash")`` finds Git Bash, but a
    bare ``"bash"`` in a subprocess resolves to WSL instead -- and the two
    disagree about everything that matters here: WSL spells a drive
    ``/mnt/c/`` while Git Bash spells it ``/c/``, and WSL's ``find /`` walks
    the whole C: drive. The code once computed the explicit path and then
    called the bare name, so it ran a different shell from the one it had
    checked. Detection is done once, against the binary that is actually
    invoked, so what the prompt says can never drift from what runs.
    """
    if not BASH:
        return ("no bash found; falling back to shell=True",
                os.name == "nt")
    try:
        out = subprocess.run(
            [BASH, "-lc", "uname -s"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        out = ""
    kind = out or ("Windows (uname unavailable)"
                   if os.name == "nt" else "unknown")
    return kind, ("mingw" in out.lower() or "msys" in out.lower())


SHELL_KIND, IS_MINGW = _detect_shell()


def shell_description() -> str:
    """One accurate paragraph for the system prompt."""
    if IS_MINGW:
        return (f"Shell: {SHELL_KIND} (Git Bash). Commands are POSIX, but a "
                f"Windows drive is written `/c/...` -- `/mnt/c/...` is WSL "
                f"syntax and will NOT work.")
    if "linux" in SHELL_KIND.lower():
        return (f"Shell: {SHELL_KIND} (WSL/Linux). A Windows drive is "
                f"written `/mnt/c/...`.")
    return f"Shell: {SHELL_KIND}."

#: One interpreter per task. Persistent on purpose: loading a 20k-line JSONL
#: once and then aggregating it across steps is the whole reason this is a
#: tool and not a thin wrapper around ``python -c``.
_SANDBOX = PythonSandbox()

#: WSL spells a Windows drive ``/mnt/c/``; Git Bash (MINGW) spells it ``/c/``.
#: The agent reaching for the wrong one is SILENT rather than loud: in
#:   cd /mnt/c/x && first ; second
#: the failed ``cd`` skips ``first``, ``second`` runs from the ORIGINAL
#: directory and supplies the exit code -- so the call reports success while
#: doing something else entirely. Rewriting the path in the tool removes a
#: whole class of "it looked like it worked".
_MNT_DRIVE = re.compile(r"(?<![\w.])/mnt/([A-Za-z])/")#: Shared memory tree, the same one the editor writes to.
_MEMORY = MemoryStore()


def set_memory_session(session_id: str) -> None:
    """Point the session memory scope at the current task's notes file."""
    _MEMORY.session_id = session_id
    # The visible plan lives with the session too, so a crash still shows
    # what was intended and what was left undone.
    tools_todo.set_todo_session(session_id)



def _truncate(text: str) -> str:
    """Keep the TAIL of a long result: failures, tracebacks and error messages
    live at the end of test and build output, and that is what the model needs."""
    if len(text) <= MAX_TOOL_CHARS:
        return text
    head = MAX_TOOL_CHARS // 4
    tail = MAX_TOOL_CHARS - head
    return (text[:head]
            + f"\n\n... [truncated {len(text) - MAX_TOOL_CHARS} chars] ...\n\n"
            + text[-tail:])


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / path)


# ---------------------------------------------------------------- handlers

def _normalize_paths(command: str):
    """Translate WSL drive paths to the form this shell accepts.

    Only on Windows under a MINGW shell: under WSL ``/mnt/c/`` is correct and
    must be left alone. The rewrite is reported in the result so the model
    learns the accepted form instead of the tool quietly doing something
    different from what it asked for.
    """
    if not IS_MINGW:
        return command, ""
    rewritten = _MNT_DRIVE.sub(lambda m: "/" + m.group(1).lower() + "/",
                               command)
    if rewritten == command:
        return command, ""
    return rewritten, ("[paths rewritten: /mnt/<drive>/ is WSL syntax and "
                       "this is Git Bash, so they were changed to "
                       "/<drive>/]\n")


def run_command(command: str, cwd: str = None, timeout_sec: int = None,
                approve: bool = False, reason: str = "") -> str:
    """Run a shell command. Covers git, tests, grep, ssh, docker -- use this
    instead of guessing at file contents or repository state.

    Irreversible commands (rm -rf, DROP TABLE, force-push, docker volume rm,
    ...) are REFUSED unless ``approve`` is true AND an approval path exists
    (AGENT_AUTO_APPROVE or an interactive terminal). A refusal tells you the
    exact command to hand to a human instead.
    """
    allowed, why = guard_command(command, approve=bool(approve))
    if not allowed:
        return why
    timeout = timeout_sec or TOOL_TIMEOUT_SEC
    command, note = _normalize_paths(command)
    note = (why + "\n" if why else "") + note
    try:
        if BASH:
            proc = subprocess.run(
                # The RESOLVED path, not the bare name: a bare "bash" picks up
                # a different shell on Windows than shutil.which found.
                [BASH, "-lc", command],
                cwd=str(_resolve(cwd)) if cwd else str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=timeout,
            )
        else:
            proc = subprocess.run(
                command, shell=True,
                cwd=str(_resolve(cwd)) if cwd else str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s\ncommand: {command}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR running command: {type(exc).__name__}: {exc}"

    out = proc.stdout or ""
    if proc.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
    if not out.strip():
        out = "(no output)"
    return f"exit={proc.returncode}\n{note}{_truncate(out)}"


def read_file(path: str, start_line: int = 1, end_line: int = 300) -> str:
    """Read a file. Returns numbered lines so you can quote exact content when
    reasoning about it or editing it. Use this rather than `cat` -- the line
    numbers make later edits unambiguous."""
    p = _resolve(path)
    if not p.exists():
        return f"NOT FOUND: {p}"
    if p.is_dir():
        return f"IS A DIRECTORY: {p}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR reading {p}: {exc}"
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))
    if start > end:
        return f"EMPTY RANGE: file has {len(lines)} lines"
    body = "\n".join(f"{i + 1:6d}\t{lines[i]}" for i in range(start - 1, end))
    more = "" if end >= len(lines) else f"\n... ({len(lines) - end} more lines)"
    return f"{p} ({len(lines)} lines total)\n{body}{more}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file. Overwrites the WHOLE file, so read it first
    and pass the full intended content -- never assume you know what is there."""
    p = _resolve(path)
    if CONFIRM_WRITES:
        preview = content if len(content) < 400 else content[:400] + "..."
        print(f"\n[write_file] {p}  ({len(content)} chars)")
        print("-" * 60)
        print(preview)
        print("-" * 60)
        answer = input("apply? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return "DECLINED by the operator. The file was NOT changed."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"ERROR writing {p}: {exc}"
    return f"WROTE {p} ({len(content)} chars)"


def search(pattern: str, include: str = "*", path: str = ".",
           max_results: int = 40) -> str:
    """Search file contents and return matching lines as file:line: text.

    Pure Python rather than a ``grep`` shell-out: grep is absent on Windows,
    and a search that silently returns nothing would let the agent conclude a
    symbol has no other callers when it has several.
    """
    root = _resolve(path)
    if not root.exists():
        return f"NOT FOUND: {root}"
    try:
        rx = re.compile(pattern)
    except re.error:
        rx = re.compile(re.escape(pattern))
    out = []
    files = [root] if root.is_file() else [p for p in root.rglob(include)
                                            if p.is_file()]
    for p in files:
        try:
            lines = p.read_text(encoding="utf-8",
                                errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                out.append(f"{p}:{i}: {line.strip()[:200]}")
                if len(out) >= max_results:
                    return "\n".join(out) + f"\n... (capped at {max_results})"
    return "\n".join(out) if out else "(no matches)"


# ------------------------------------------------- generated-code execution

#: Clues that a ``run_python`` block will change something outside its own
#: interpreter. A heuristic, and deliberately over-broad -- every unnecessary
#: "are you sure" costs a keystroke, whereas an unnoticed write to a live
#: position registry costs real money.
_MUTATING_PATTERNS = (
    "open(", ".write(", ".write_text", ".write_bytes", ".mkdir(",
    ".unlink(", ".touch(", ".rename(", ".replace(", ".rmdir(",
    "os.remove", "os.rename", "os.replace", "os.rmdir", "os.system",
    "shutil.", "subprocess", "multiprocessing",
    "requests.", "urllib", "http.client", "socket.",
    "placeorder", "ib.placeorder", "ib.connect", "ib.disconnect",
    ".to_csv(", ".to_json(", ".to_parquet(", ".to_excel(",
    "json.dump(", "pickle.dump(", "shelve.",
)


def _looks_mutating(code: str) -> bool:
    low = (code or "").lower()
    return any(p in low for p in _MUTATING_PATTERNS)


def run_python(code: str, timeout_sec: int = None) -> str:
    """Run Python in a persistent interpreter that keeps state across calls.

    Use this for ANALYSIS that would take many tool calls otherwise: parsing a
    JSONL, aggregating P&L, diffing a registry against broker state, computing
    stop geometry. Variables survive from one call to the next within the same
    task, so load data once and then slice it. The result reports stdout, the
    value of a trailing expression, any exception, and the names now in memory.

    This is NOT a substitute for edit-and-test: change source files with
    write_file and verify with run_command. Blocks that look like they mutate
    the filesystem, the network or a broker session ask for confirmation.
    """
    if not (code or "").strip():
        return "EMPTY code block -- nothing to run."
    if CONFIRM_PYTHON_MUTATIONS and _looks_mutating(code):
        preview = code if len(code) < 700 else code[:700] + "\n..."
        print("\n[run_python] this block looks like it CHANGES something:")
        print("-" * 60)
        print(preview)
        print("-" * 60)
        answer = input("run it anyway? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return "DECLINED by the operator. Nothing was executed."
    return _SANDBOX.run(code, timeout_sec)


# -------------------------------------------------------------- memory I/O

def memory_read(scope: str = "repo", start_line: int = 1,
                end_line: int = 200) -> str:
    """Read memory as numbered lines.

    Scopes: 'repo' = project knowledge, conventions and past incidents (start
    here); 'session' = notes from the current task; 'user' = cross-project
    preferences. Read before concluding something is unknown -- if it was
    learned before, it is written down.
    """
    return _MEMORY.read(scope, start_line, end_line)


def memory_search(pattern: str, scope: str = None,
                  max_results: int = 30) -> str:
    """Search all memory for text or a regex, returning scope:file:line hits.

    The repo notes run to thousands of lines, so search before reading. Use it
    to answer "have we seen this before?": a past incident, a deploy recipe, a
    gotcha that already cost an afternoon.
    """
    return _MEMORY.search(pattern, scope, max_results)


def memory_append(scope: str, text: str, heading: str = None) -> str:
    """Record something durable in memory. Prepends, never overwrites.

    Call this BEFORE finishing whenever you learned something that would save
    time next session: a root cause, a rule, a deploy recipe, a trap that
    nearly cost money. Prefer a short titled note over a transcript -- the
    point is that the lesson stops being re-derived. Default scope is 'repo'.
    """
    return _MEMORY.append(scope or "repo", text, heading)


def memory_revise(scope: str, old: str, new: str) -> str:
    """Correct an existing note in place, rather than stacking a correction
    on top of a wrong one.

    'old' must appear EXACTLY ONCE in the scope -- copy it from memory_read.
    Zero matches is a miss; several is a guess about which to change, and both
    are refused. Set 'new' empty to delete the matched text. A backup of the
    previous file is written alongside it first.
    """
    return _MEMORY.revise(scope or "repo", old, new)


def memory_retire(scope: str, heading: str) -> str:
    """Remove one whole titled note from a scope.

    Use when a note is simply wrong or superseded -- otherwise the bad
    conclusion stays in the file forever, still readable and still wrong.
    Matches the heading exactly ('#' plus the title). A backup is written
    first.
    """
    return _MEMORY.retire(scope or "repo", heading)


def memory_status() -> str:
    """Show what memory exists in each scope, and where it lives."""
    return _MEMORY.summary()


# ---------------------------------------------------------------- schema

def _fn(name: str, description: str, props: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOLS = [
    _fn(
        "run_command",
        "Run a shell command and get exit code, stdout and stderr. Use for "
        "git (status/diff/log), running the test suites, ls, grep, ssh to a "
        "remote host, docker, and anything else you would type at a terminal. "
        "Prefer this over guessing at repository or market state. Long output "
        "is truncated keeping the tail, where errors live.",
        {
            "command": {"type": "string",
                        "description": "The shell command to run."},
            "cwd": {"type": "string",
                    "description": "Working directory. Defaults to project root."},
            "timeout_sec": {"type": "integer",
                            "description": "Max seconds to wait. Default 180."},
            "approve": {"type": "boolean",
                        "description": "Set true ONLY to request an "
                                       "irreversible command you have decided "
                                       "is required; it may still be refused "
                                       "in unattended runs."},
            "reason": {"type": "string",
                       "description": "One line: why the irreversible action "
                                      "is necessary."},
        },
        ["command"],
    ),
    _fn(
        "read_file",
        "Read a file as numbered lines. Use before editing so you know what is "
        "actually there. Line numbers make later edits unambiguous.",
        {
            "path": {"type": "string", "description": "File path."},
            "start_line": {"type": "integer", "description": "First line, 1-based."},
            "end_line": {"type": "integer", "description": "Last line, inclusive."},
        },
        ["path"],
    ),
    _fn(
        "write_file",
        "Create or overwrite a file with the given content. Writes the WHOLE "
        "file, so read the file first and send its complete intended content. "
        "For a small edit, read, modify in memory, and write the full result.",
        {
            "path": {"type": "string", "description": "File path."},
            "content": {"type": "string", "description": "The complete file content."},
        },
        ["path", "content"],
    ),
    _fn(
        "search",
        "Search file contents and return matching lines as file:line: text. "
        "Use before renaming a symbol or changing a shared helper -- find every "
        "use first.",
        {
            "pattern": {"type": "string", "description": "Text or regex to find."},
            "include": {"type": "string",
                        "description": "Glob filter, e.g. '*.py'. Default '*'."},
            "path": {"type": "string", "description": "Directory to search."},
            "max_results": {"type": "integer", "description": "Cap on matches."},
        },
        ["pattern"],
    ),
    _fn(
        "run_python",
        "Run Python in a persistent interpreter. STATE IS KEPT between calls "
        "within one task, so load a big file once and then aggregate, filter "
        "and diff it in later calls instead of re-reading it. Returns stdout, "
        "the value of a trailing expression, any traceback, and the variable "
        "names now held. Use for analysis (parse logs, compute P&L, diff two "
        "states, arithmetic over a chain) rather than for editing source -- "
        "edit with write_file and verify with run_command. Blocks that write "
        "files, shell out or touch the network ask for confirmation first.",
        {
            "code": {"type": "string",
                     "description": "Python to run. A trailing expression is "
                                    "evaluated and its repr returned."},
            "timeout_sec": {"type": "integer",
                            "description": "Hard kill after this many seconds. "
                                           "Default 60."},
        },
        ["code"],
    ),
    _fn(
        "memory_read",
        "Read a scope of long-term memory as numbered lines. Scopes: 'repo' "
        "(project knowledge, conventions, past incidents -- read this first), "
        "'session' (notes from this task), 'user' (cross-project preferences). "
        "Before concluding something is unknown, check whether it was learned "
        "before and written down.",
        {
            "scope": {"type": "string",
                      "description": "'repo', 'session' or 'user'. Default 'repo'."},
            "start_line": {"type": "integer", "description": "First line, 1-based."},
            "end_line": {"type": "integer", "description": "Last line, inclusive."},
        },
        [],
    ),
    _fn(
        "memory_search",
        "Search all memory scopes for text or a regex. Returns scope:file:line "
        "hits. Repo memory runs to thousands of lines, so search before "
        "reading it wholesale. Use it to ask 'have we seen this before?' -- a "
        "past incident, a deploy recipe, a trap that already cost an afternoon.",
        {
            "pattern": {"type": "string", "description": "Text or regex to find."},
            "scope": {"type": "string",
                      "description": "Limit to one scope, or omit for all."},
            "max_results": {"type": "integer", "description": "Cap on hits."},
        },
        ["pattern"],
    ),
    _fn(
        "memory_append",
        "Record a durable lesson, rule, root cause or recipe in memory. "
        "Prepends a titled section; never overwrites existing notes. Do this "
        "BEFORE finishing whenever the session produced something worth not "
        "re-deriving -- a defect and its fix, a gotcha, a deploy recipe, a "
        "decision and why. Keep it short and specific; a note that saves the "
        "next session an hour is worth more than a paragraph of narration.",
        {
            "scope": {"type": "string",
                      "description": "'repo' (default), 'session' or 'user'."},
            "text": {"type": "string", "description": "The note body, markdown."},
            "heading": {"type": "string",
                        "description": "Short title. Defaults to a dated NOTE."},
        },
        ["text"],
    ),
    _fn(
        "memory_revise",
        "Correct an existing memory note IN PLACE. Prefer this over "
        "memory_append when a note is wrong: appending a correction leaves the "
        "bad conclusion in the file where it will be read and trusted again. "
        "'old' must appear EXACTLY ONCE -- copy it verbatim from memory_read. "
        "If it is missing you get NOT FOUND; if it appears several times you "
        "get AMBIGUOUS and should add surrounding lines to make it unique. "
        "'new' may be empty to delete the matched text. A backup is saved "
        "alongside the file first.",
        {
            "scope": {"type": "string",
                      "description": "'repo' (default), 'session' or 'user'."},
            "old": {"type": "string",
                    "description": "The exact text to replace, verbatim."},
            "new": {"type": "string",
                    "description": "Replacement text. Empty to delete it."},
        },
        ["old", "new"],
    ),
    _fn(
        "memory_retire",
        "Delete one whole titled note from memory. Use when a note is simply "
        "wrong or has been superseded, so the file accumulates knowledge "
        "instead of clutter. Give the heading exactly as it appears -- one "
        "'#' plus the title. With no match you get NOT FOUND and a list of "
        "the headings that ARE there. A backup is saved first.",
        {
            "scope": {"type": "string",
                      "description": "'repo' (default), 'session' or 'user'."},
            "heading": {"type": "string",
                        "description": "The note's title, e.g. "
                                       "'SOME NOTE (2026-10-28)'."},
        },
        ["heading"],
    ),
    _fn(
        "memory_status",
        "Show where memory lives and how much is in each scope. Use when you "
        "are unsure what has been recorded.",
        {},
        [],
    ),
    _fn(
        "replace_in_file",
        "Edit a file by replacing an EXACT string. The old text must appear "
        "exactly once (or exactly expected_replacements times); zero matches "
        "means re-read the file, several means your anchor is ambiguous. "
        "Nothing is written unless the count matches, so an edit can never "
        "half-apply. PREFER THIS over write_file for any change smaller than "
        "a whole-file rewrite.",
        {
            "path": {"type": "string", "description": "File path."},
            "old": {"type": "string",
                    "description": "Exact existing text, with surrounding "
                                   "context lines."},
            "new": {"type": "string", "description": "Replacement text."},
            "expected_replacements": {"type": "integer",
                                      "description": "Default 1."},
        },
        ["path", "old", "new"],
    ),
    _fn(
        "multi_replace_in_file",
        "Apply several exact replacements to ONE file atomically: if any edit's "
        "anchor is missing or ambiguous, NOTHING is applied. Use when a change "
        "touches several separated places in the same file.",
        {
            "path": {"type": "string", "description": "File path."},
            "edits": {"type": "array",
                      "items": {"type": "object",
                                "properties": {
                                    "old": {"type": "string"},
                                    "new": {"type": "string"}},
                                "required": ["old", "new"]},
                      "description": "Ordered exact replacements."},
        },
        ["path", "edits"],
    ),
    _fn(
        "glob_files",
        "Find files by path pattern, e.g. '**/*.py' or 'engine/validate_*.py'. "
        "Use to discover WHERE things live before reading; use search() for "
        "content. Skips .git, caches and vendor directories.",
        {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Root dir. Default '.'."},
            "max_results": {"type": "integer", "description": "Cap. Default 200."},
        },
        ["pattern"],
    ),
    _fn(
        "repo_map",
        "Structural overview of a subtree: every class and function signature, "
        "no bodies. Read THIS first to decide which files matter; it is far "
        "cheaper than reading them all.",
        {
            "path": {"type": "string", "description": "Subtree. Default '.'."},
            "max_chars": {"type": "integer",
                          "description": "Cap on map size. Default from config."},
        },
        [],
    ),
    _fn(
        "git_status",
        "Working-tree state: branch plus staged/unstaged/untracked files. Run "
        "before editing (know the baseline) and after (know what you changed).",
        {}, [],
    ),
    _fn(
        "git_diff",
        "The current diff, optionally staged-only or scoped to one path. Use "
        "after edits to verify WHAT changed, not just that something did.",
        {"staged": {"type": "boolean", "description": "Limit to the index."},
         "path": {"type": "string", "description": "Scope to one file."}},
        [],
    ),
    _fn(
        "git_log",
        "Recent commits (hash, date, subject), optionally for one path. Use to "
        "find the last known-good revision before rolling a file back.",
        {"limit": {"type": "integer", "description": "Default 15."},
         "path": {"type": "string", "description": "Scope to one file."}},
        [],
    ),
    _fn(
        "git_show",
        "Show one commit with its patch, or a file at a revision ('rev:path').",
        {"revision": {"type": "string"},
         "path": {"type": "string"}},
        ["revision"],
    ),
    _fn(
        "git_commit",
        "Stage paths (or everything) and commit. Say WHY in the message. A "
        "caution-class action: recorded and narrated, not refused.",
        {"message": {"type": "string"},
         "paths": {"type": "array", "items": {"type": "string"}}},
        ["message"],
    ),
    _fn(
        "git_restore",
        "Roll ONE file back to a revision (default HEAD) -- the safe undo the "
        "safety rails point at after repeated failed edits. Discards that "
        "file's uncommitted changes.",
        {"path": {"type": "string"},
         "from_revision": {"type": "string", "description": "Default HEAD."}},
        ["path"],
    ),
    _fn(
        "session_list",
        "Recent agent sessions, newest first. The name encodes start time and "
        "session id -- this is the timeline of what the harness has been asked.",
        {"limit": {"type": "integer", "description": "Default 20."}}, [],
    ),
    _fn(
        "session_read",
        "Read one past session transcript (head and tail). Use for 'how did "
        "the last attempt at this end?'.",
        {"name": {"type": "string",
                  "description": "Session file name or stem."},
         "max_chars": {"type": "integer", "description": "Default 8000."}},
        ["name"],
    ),
    _fn(
        "session_search",
        "Search ALL session transcripts for text or regex. 'What did we try "
        "last time this failed?' is usually already written down.",
        {"pattern": {"type": "string"},
         "limit": {"type": "integer", "description": "Default 30."}},
        ["pattern"],
    ),
    _fn(
        "manage_todo_list",
        "Publish your plan and its progress. Replace the WHOLE list each time "
        "with items {id, title, status in not-started|in-progress|completed}; "
        "at most one in-progress. Update it as you finish each item -- an "
        "operator watching an unattended run reads this to know where you are.",
        {"items": {"type": "array",
                   "items": {"type": "object",
                             "properties": {
                                 "id": {"type": "integer"},
                                 "title": {"type": "string"},
                                 "status": {"type": "string",
                                            "enum": ["not-started",
                                                     "in-progress",
                                                     "completed"]}},
                             "required": ["id", "title", "status"]}}},
        ["items"],
    ),
    _fn(
        "todo_show",
        "Render the current plan without changing it.", {}, [],
    ),
    _fn(
        "run_subagent",
        "Hand a SELF-CONTAINED subtask (research, summarising a huge log, "
        "finding every caller) to a fresh child agent with its own session "
        "and step budget; only its final answer comes back. Use it to keep "
        "noise out of your own context, then decide yourself -- a subagent's "
        "answer is evidence, not a verdict.",
        {"prompt": {"type": "string",
                    "description": "Complete instructions; the child cannot "
                                   "ask you follow-ups."},
         "steps": {"type": "integer", "description": "Child step budget. "
                                                     "Default 12."}},
        ["prompt"],
    ),
    _fn(
        "view_image",
        "Attach an image file (png/jpg/jpeg/gif/webp) so you can SEE it -- "
        "charts, screenshots, diagrams, UI mocks. The pixels arrive with the "
        "tool result; the transcript keeps only a caption.",
        {"path": {"type": "string", "description": "Image file path."},
         "note": {"type": "string",
                  "description": "Optional one-line note about what to look "
                                 "for."}},
        ["path"],
    ),
    _fn(
        "browser_open",
        "Launch the headless browser (Playwright chromium) and load a URL "
        "(http/https/file). Returns the first snapshot. If Playwright is not "
        "installed the tool says so; report that instead of retrying.",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _fn(
        "browser_snapshot",
        "Numbered interactive elements (refs e1, e2, ...) plus a bounded page "
        "text excerpt. Refs are rebuilt on every snapshot and valid until the "
        "next one -- click/type by ref rather than guessing selectors.",
        {}, [],
    ),
    _fn(
        "browser_click",
        "Click by snapshot ref, CSS selector, or visible text.",
        {"ref": {"type": "string"}, "selector": {"type": "string"},
         "text": {"type": "string"}},
        [],
    ),
    _fn(
        "browser_type",
        "Type into an input by ref or selector. clear replaces content; "
        "submit presses Enter afterwards.",
        {"ref": {"type": "string"}, "selector": {"type": "string"},
         "text": {"type": "string"}, "submit": {"type": "boolean"},
         "clear": {"type": "boolean"}},
        ["text"],
    ),
    _fn(
        "browser_navigate",
        "Go to a URL, or 'back' / 'forward' / 'reload'.",
        {"target": {"type": "string"}},
        ["target"],
    ),
    _fn(
        "browser_read",
        "The page's full visible text, bounded. Use when the snapshot excerpt "
        "is not enough.",
        {"max_chars": {"type": "integer", "description": "Default 20000."}},
        [],
    ),
    _fn(
        "browser_screenshot",
        "Capture the viewport and ATTACH it so you can see the rendered page. "
        "The PNG is also saved under sessions/ for the record. This is how "
        "you verify a UI change visually.",
        {"path": {"type": "string",
                  "description": "Optional destination PNG path."}},
        [],
    ),
    _fn(
        "browser_close",
        "Release the browser and every handle. Call when done browsing.",
        {}, [],
    ),
]


HANDLERS = {
    "run_command": run_command,
    "read_file": read_file,
    "write_file": write_file,
    "search": search,
    "run_python": run_python,
    "memory_read": memory_read,
    "memory_search": memory_search,
    "memory_append": memory_append,
    "memory_revise": memory_revise,
    "memory_retire": memory_retire,
    "memory_status": memory_status,
    "replace_in_file": replace_in_file,
    "multi_replace_in_file": multi_replace_in_file,
    "glob_files": glob_files,
    "repo_map": repo_map,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_show": git_show,
    "git_commit": git_commit,
    "git_restore": git_restore,
    "session_list": session_list,
    "session_read": session_read,
    "session_search": session_search,
    "manage_todo_list": manage_todo_list,
    "todo_show": todo_show,
    "run_subagent": run_subagent,
    "view_image": view_image,
    "browser_open": browser_open,
    "browser_snapshot": browser_snapshot,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_navigate": browser_navigate,
    "browser_read": browser_read,
    "browser_screenshot": browser_screenshot,
    "browser_close": browser_close,
}
