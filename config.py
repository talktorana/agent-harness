"""Configuration for the agent harness.

Everything comes from the environment so the same code runs against DeepInfra,
OpenAI, a local model, or any OpenAI-compatible proxy -- including a switch to
a different provider without touching the harness.
"""

import os
from pathlib import Path
from typing import Dict

#: Where each AGENT_* setting actually came from. An exported variable
#: silently beating the .env you just edited is exactly the kind of thing that
#: costs an hour to notice -- the key looks right in the file and the run still
#: fails -- so provenance is recorded and shown rather than left to be guessed.
SOURCES: Dict[str, str] = {}


def _load_dotenv(*candidates) -> None:
    """Read .env files into the environment WITHOUT overriding what is
    already exported.

    The README tells you to copy .env.example to .env, and until now nothing
    read it -- so a key written there was silently ignored and the run failed
    with an auth error that looked like a bad key. A tiny parser rather than
    a dependency: ``KEY=value`` per line, ``#`` comments, no quoting rules to
    get subtly wrong.

    Two rules learned the hard way:

    * ONLY ``AGENT_*`` keys are read. A project's own .env routinely holds
      broker credentials and tokens for other systems; importing those would
      make them visible to generated code through ``os.environ`` for no
      benefit to this harness at all.
    * Exported variables win, and the shadowing is reported. Silently losing
      to a stale export is what made a valid key in .env produce a 401.
    """
    for raw in candidates:
        p = Path(raw)
        if not p.exists():
            continue
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key.startswith("AGENT_"):
                continue
            if key in os.environ:
                SOURCES[key] = f"environment (exported) -- shadows {p.name}"
                continue
            os.environ[key] = value
            SOURCES[key] = p.name


for _exported in os.environ:
    if _exported.startswith("AGENT_"):
        SOURCES[_exported] = "environment (exported)"

# ONLY this file. Reading a .env from the working directory as well meant a
# project's own config (credentials and tokens for other systems) was pulled
# agent's environment -- and a stale AGENT_* export in the shell could change
# where the agent looked without a word about it.
_load_dotenv(Path(__file__).resolve().parent / ".env")


def source_of(key: str) -> str:
    """Where a setting came from: an exported variable or a specific .env."""
    return SOURCES.get(key, "built-in default")

# --- model endpoint (must be OpenAI-compatible) ---------------------------
BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.deepinfra.com/v1/openai")
API_KEY = os.environ.get("AGENT_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")

# DeepSeek-V4.1-Flash-1017, the agentic model behind this default:
#   1,048,576-token context, JSON function calling, multimodal.
#   $0.098 in / $0.294 out / $0.003 cached per 1M tokens (51% promo off).
# Cheap enough that a 40-step monitoring pass costs cents, which is the whole
# point of an unattended harness.

# --- api resilience -------------------------------------------------------
# A busy engine answers 429 "engine_overloaded", and a dropped connection
# looks the same to the caller. Transient failures retry with exponential
# backoff; a wrong key or unknown model id fails immediately because waiting
# cannot fix those. An unattended agent that dies on the first 429 is useless.
API_RETRIES = int(os.environ.get("AGENT_API_RETRIES", "6"))
API_BACKOFF_SEC = float(os.environ.get("AGENT_API_BACKOFF_SEC", "2.0"))
# Bound ONE model call. Without this the OpenAI client waits its own ten-minute
# default, and a hung connection looks exactly like a busy model -- the session
# log simply goes quiet with nothing to say where it stopped. On a real run
# that cost twenty minutes of staring at nothing.
API_TIMEOUT_SEC = float(os.environ.get("AGENT_API_TIMEOUT_SEC", "120"))
# Tried only after the primary model has been retried to exhaustion. Keep it
# empty unless you want a second model to catch overflow.
FALLBACK_MODEL = os.environ.get("AGENT_FALLBACK_MODEL", "").strip()

# --- loop limits ----------------------------------------------------------
# The agent stops when the model stops calling tools. MAX_STEPS is the
# backstop so a confused model cannot loop forever and burn tokens.
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "40"))
# Warn the model this many steps before the cut-off. Without a warning it
# keeps investigating to the very end and the run returns "no answer" --
# fifty steps of good research thrown away for want of a sentence.
ANSWER_WARN_STEPS = int(os.environ.get("AGENT_ANSWER_WARN_STEPS", "3"))
TOOL_TIMEOUT_SEC = int(os.environ.get("AGENT_TOOL_TIMEOUT_SEC", "180"))
# Tool results are truncated to this many characters. Large enough to carry a
# test failure or a diff, small enough that a 200 KB log dump cannot drown the
# context window in one turn.
MAX_TOOL_CHARS = int(os.environ.get("AGENT_MAX_TOOL_CHARS", "8000"))

# --- what the agent is allowed to touch -----------------------------------
_HARNESS_DIR = Path(__file__).resolve().parent
_root = Path(os.environ.get("AGENT_PROJECT_ROOT", ".")).expanduser()
# A RELATIVE project root resolves against the directory holding this module,
# not the current working directory. Against the cwd, `AGENT_PROJECT_ROOT=..`
# meant one thing when run from agent-harness/ and a different thing when run
# from the project root -- the agent pointed at the wrong tree with nothing to
# explain why. Absolute paths are used as-is, so a container's /project is
# unaffected.
PROJECT_ROOT = (_root if _root.is_absolute()
                else (_HARNESS_DIR / _root)).resolve()
# When true, write_file asks on the terminal before changing anything. Turn it
# off only for unattended runs where you have already seen what it does.
CONFIRM_WRITES = os.environ.get("AGENT_CONFIRM_WRITES", "1") == "1"
# run_python can do anything, so confirming every block would train you to
# click through without reading -- which is worse than no gate at all. So the
# harness asks ONLY when the block looks like it changes state: writes or
# removes a file, shells out, or reaches for the network. The list is
# deliberately broad. A needless "are you sure" costs a keystroke; a missed
# mutation on a live book costs real money.
CONFIRM_PYTHON_MUTATIONS = os.environ.get(
    "AGENT_CONFIRM_PYTHON_MUTATIONS", "1") == "1"

# --- generated-code execution ---------------------------------------------
# run_python keeps ONE interpreter per task so the model can load data once
# and then analyse it across several steps. The timeout is a hard kill -- a
# runaway loop in generated code must not hang the whole monitoring pass.
PYTHON_TIMEOUT_SEC = int(os.environ.get("AGENT_PYTHON_TIMEOUT_SEC", "60"))
PYTHON_MAX_OUTPUT = int(os.environ.get("AGENT_PYTHON_MAX_OUTPUT", "6000"))

# --- session log ----------------------------------------------------------
# Always on. The record of what the agent actually DID is not optional when
# the alternative is reconstructing it from memory after something goes wrong.
SESSIONS_DIR = Path(os.environ.get(
    "AGENT_SESSIONS_DIR", str(Path(__file__).resolve().parent / "sessions")))

# --- copilot-parity layers --------------------------------------------------
# Exact-string edits are the safe default; a whole-file write can silently
# drop content the model never saw. replace_in_file refuses ambiguous matches.
EDIT_MAX_MATCHES = int(os.environ.get("AGENT_EDIT_MAX_MATCHES", "1"))
# Structural overview injected into the system prompt so the model knows the
# shape of the codebase before it reads anything. Bounded: a 200k-line monorepo
# must not eat the context window.
REPO_MAP_MAX_CHARS = int(os.environ.get("AGENT_REPO_MAP_CHARS", "6000"))
REPO_MAP_ENABLED = os.environ.get("AGENT_REPO_MAP", "1") == "1"
# Safety rails. Irreversible shell actions (rm -rf, DROP TABLE, force-push,
# docker rm -v ...) are REFUSED unless the caller passes approve=True AND one
# of these is set: interactive confirmation, or AGENT_AUTO_APPROVE for a
# trusted unattended run. Destructive git verbs get their own gate.
AUTO_APPROVE = os.environ.get("AGENT_AUTO_APPROVE", "0") == "1"
ALLOW_DESTRUCTIVE_GIT = os.environ.get("AGENT_ALLOW_DESTRUCTIVE_GIT", "0") == "1"
# Three consecutive failures on the same file trigger a rollback suggestion
# (git restore from HEAD) rather than a fourth blind edit.
MAX_FILE_FAILURES = int(os.environ.get("AGENT_MAX_FILE_FAILURES", "3"))
# Streaming output for the interactive CLI and the web UI.
STREAM = os.environ.get("AGENT_STREAM", "1") == "1"
# Web chat UI (optional dependency group: requirements-web.txt).
WEBUI_HOST = os.environ.get("AGENT_WEBUI_HOST", "127.0.0.1")
WEBUI_PORT = int(os.environ.get("AGENT_WEBUI_PORT", "8787"))
# Scheduler: cron-lite job file so unattended missions run in the cloud.
SCHEDULE_FILE = Path(os.environ.get(
    "AGENT_SCHEDULE_FILE",
    str(Path(__file__).resolve().parent / "schedule.json")))
#: Where the scheduler persists per-job runtime state (last_run, last_error).
#:
#: NOT the schedule file. A deployment usually mounts the schedule read-only --
#: it is reviewed configuration, and in a gated deploy it is on the applier's
#: forbidden list, because an agent that can rewrite its own mission text owns
#: its own guardrails. Writing `last_run` back into that same file therefore
#: fails on a read-only mount, and the failure surfaced as a generic
#: "[scheduler] loop error" that looked transient while no mission ever ran.
#: Defaults to a sibling of the session log, which is agent-writable by design.
SCHEDULE_STATE_FILE = Path(os.environ.get(
    "AGENT_SCHEDULE_STATE_FILE",
    str(SESSIONS_DIR / "schedule_state.json")))

# --- browser automation (optional: requirements-browser.txt) -----------------
# Headless chromium via Playwright. The tools degrade to a clear
# NOT AVAILABLE message when the dependency is absent, so a slim image keeps
# working without it.
BROWSER_HEADLESS = os.environ.get("AGENT_BROWSER_HEADLESS", "1") == "1"
BROWSER_TIMEOUT_MS = int(os.environ.get("AGENT_BROWSER_TIMEOUT_MS", "30000"))
BROWSER_SNAPSHOT_CHARS = int(os.environ.get("AGENT_BROWSER_SNAPSHOT_CHARS",
                                            "4000"))
BROWSER_MAX_REFS = int(os.environ.get("AGENT_BROWSER_MAX_REFS", "120"))

# --- multimodal input --------------------------------------------------------
IMAGES_ENABLED = os.environ.get("AGENT_IMAGES", "1") == "1"
IMAGE_MAX_BYTES = int(os.environ.get("AGENT_IMAGE_MAX_BYTES",
                                     str(8 * 1024 * 1024)))

# --- memory ---------------------------------------------------------------
def _find_memory_tree(project_name: str):
    """Locate the memory tree the editor's Copilot writes to.

    VS Code keeps it under workspaceStorage/<hash>/GitHub.copilot-chat/
    memory-tool/memories, and that hash is not guessable. More than one
    workspace can have a tree, so pick by CONTENT, not recency: the tree
    holding ``repo/<project_name>.md`` is THIS project's memory, and a more
    recently written tree belonging to some other workspace is just noise.
    (A plain "newest wins" rule silently loaded the wrong project's notes --
    found by testing this rather than by reading it.)
    """
    override_root = os.environ.get("AGENT_MEMORY_ROOT", "").strip()
    override_repo = os.environ.get("AGENT_MEMORY_REPO_FILE", "").strip()
    if override_root or override_repo:
        root = (Path(override_root).expanduser() if override_root
                else Path(__file__).resolve().parent / "memories")
        if override_repo:
            return root, Path(override_repo).expanduser()
        return root, _pick_repo_file(root, f"{project_name}.md")

    wanted = f"{project_name}.md"
    named, other = [], []
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        base = Path(appdata) / "Code" / "User" / "workspaceStorage"
        if base.is_dir():
            glob = "*/GitHub.copilot-chat/memory-tool/memories"
            for marker in base.glob(glob):
                if not marker.is_dir():
                    continue
                repo = marker / "repo" / wanted
                try:
                    if repo.is_file():
                        named.append((repo.stat().st_mtime, marker, repo))
                    else:
                        other.append((marker.stat().st_mtime, marker))
                except OSError:
                    continue

    # Fallback lives INSIDE the project rather than beside this module, so the
    # tree is part of the deployable tree and can be carried to another host
    # with the code. In a container /project is often read-only -- set
    # AGENT_MEMORY_ROOT to a mounted volume there.
    fallback = PROJECT_ROOT / "memories"
    if named:
        _, root, repo = max(named)
        return root, repo
    if other:
        _, root = max(other)
        return root, _pick_repo_file(root, wanted)
    return fallback, _pick_repo_file(fallback, wanted)


def _pick_repo_file(root: Path, wanted: str) -> Path:
    """``repo/<project>.md`` when it exists, else the newest ``repo/*.md``.

    A starter tree ships with the harness, and it cannot know your project's
    name. Insisting on the exact filename would mean a first run reports "no
    memory yet" while the file explaining the format sits right there --
    so any repo note is a valid starting point until a project-named one is
    written.
    """
    repo_dir = root / "repo"
    exact = repo_dir / wanted
    if exact.is_file() or not repo_dir.is_dir():
        return exact
    found = []
    for p in repo_dir.glob("*.md"):
        try:
            found.append((p.stat().st_mtime, p))
        except OSError:
            continue
    return max(found)[1] if found else exact


MEMORY_ROOT, MEMORY_REPO_FILE = _find_memory_tree(PROJECT_ROOT.name)
MEMORY_USER_FILE = MEMORY_ROOT / "user.md"
MEMORY_SESSION_DIR = MEMORY_ROOT / "session"

# How much of repo memory is loaded into the prompt automatically. The file is
# thousands of lines; the head carries the recent material and memory_search
# covers the rest.
MEMORY_HEAD_LINES = int(os.environ.get("AGENT_MEMORY_HEAD_LINES", "120"))

# --- transcript ----------------------------------------------------------
TRANSCRIPT = os.environ.get("AGENT_TRANSCRIPT", "")

# --- deployment (agent -> host applier) -----------------------------------
# The agent never restarts a production container itself. It writes a REQUEST
# into an outbox; a host-side applier decides whether it may be applied
# (checksums, path containment, broker-state gate, backups, rollback). That
# split is the safety property, so "gated" is the default and "direct" only
# changes WHO invokes the applier, never whether one runs.
DEPLOY_MODE = os.environ.get("AGENT_DEPLOY_MODE", "gated").strip().lower()
if DEPLOY_MODE not in ("gated", "direct"):
    raise ValueError(
        f"AGENT_DEPLOY_MODE must be 'gated' or 'direct', got {DEPLOY_MODE!r}")
DEPLOY_ROOT = os.environ.get("AGENT_DEPLOY_ROOT", "").strip()
DEPLOY_OUTBOX = os.environ.get("AGENT_DEPLOY_OUTBOX", "").strip()
DEPLOY_APPLIER = os.environ.get("AGENT_DEPLOY_APPLIER", "").strip()
DEPLOY_ALLOW_RESTART = os.environ.get(
    "AGENT_DEPLOY_ALLOW_RESTART", "0") == "1"
# Payload ceilings. A deploy request is a handful of source files; a request
#: this large means the agent is trying to stage something else entirely.
DEPLOY_MAX_FILES = int(os.environ.get("AGENT_DEPLOY_MAX_FILES", "20"))
DEPLOY_MAX_BYTES = int(os.environ.get("AGENT_DEPLOY_MAX_BYTES", str(4 << 20)))
# The refuse list is DEPLOYMENT knowledge, not harness knowledge: which files
# are runtime state rather than code depends on the project. Comma-separated
# names, prefixes ("prefix=") and suffixes ("suffix="). Secrets and .git are
# always refused regardless, and the host applier re-checks whatever the agent
# proposes -- this copy only makes a refusal early and readable.
#
#   AGENT_DEPLOY_FORBIDDEN=positions.json,suffix=.sqlite,prefix=runtime-
DEPLOY_FORBIDDEN = tuple(
    part.strip() for part in
    os.environ.get("AGENT_DEPLOY_FORBIDDEN", "").split(",") if part.strip())
# Policy file handed to the host applier (readiness gate + domain refusals).
DEPLOY_POLICY = os.environ.get("AGENT_DEPLOY_POLICY", "").strip()
