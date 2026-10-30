# agent-harness

A small, self-contained **agent harness**: a tool-calling loop around a language
model that can read and edit your project, run commands, analyse data in a
persistent Python interpreter, and accumulate what it learns in long-term
memory.

It exists because the interesting behaviour of an agent is not the loop — that
is forty lines. It is the parts that make a run *survivable*: a hard timeout
that can actually kill runaway code, a session record of what was really done,
retry that distinguishes "busy" from "your key is wrong", and memory that is
correctable rather than append-only.

No framework, no database, no service to run. A handful of Python files and an
OpenAI-compatible endpoint.

## Quickstart

```bash
cp .env.example .env          # add your key and point AGENT_PROJECT_ROOT at your project
python run.py --check         # proves the key with a real completion
python run.py "explain how the retry policy works"
python run.py --serve         # browser chat UI on http://127.0.0.1:8787
python run.py --schedule      # cron-lite daemon over schedule.json
```

`--check` matters. Listing models is **not** an auth gate on at least one
provider — a bogus key lists them all — so the check makes a one-token
completion instead.

## Choosing a model: the provider catalogue

The harness speaks to any OpenAI-compatible endpoint DIRECTLY — no proxy in
between. `providers.json` (copy `providers.json.example`) names them:

```json
{
  "default": {"provider": "provider", "model": "provider-v4.1-flash"},
  "providers": {
    "provider": {
      "base_url": "https://provider",
      "api_key_env": "OPENROUTER_API_KEY",
      "models": {"provider-v4.1-flash": "provider"}
    },
    "ollama-local": {
      "base_url": "http://localhost:11434/v1",
      "api_key": "ollama",
      "models": {"provider3-coder": "qwen3-coder:latest"}
    }
  }
}
```

* **Keys** come from the environment variable named by `api_key_env`
  (preferred) or an inline `api_key`. The example ships empty; the web UI's
  `/models` endpoint never returns key material.
* **Aliases** on the left of `models` are what you type and select; the right
  side is the provider's real id on the wire. A provider rename is one line.
* **Precedence:** explicit `--provider` (or the web UI dropdowns) >
  `providers.json` default > the legacy `AGENT_BASE_URL`/`AGENT_API_KEY`/
  `AGENT_MODEL` triple, which still works untouched as provider `env`.
* **Failures are configuration errors, raised before the first tool call:**
  unknown provider, unknown alias, missing key — each names what to fix.

```bash
python run.py --models                       # list the catalogue
python run.py --provider provider --model provider-v4.1-flash "task..."
python run.py --serve                        # pick provider+model in the browser
```

## The three ideas

### 1. Generated code that remembers

`run_python` runs Python in a **subprocess** with state that survives across
calls in a task. Load a large JSONL once, then aggregate, filter and diff it
in later steps.

It is a subprocess because a timeout has to be able to *kill* the code, and
you cannot kill a thread. A `while True` in generated analysis must not hang
the whole run. If the timeout fires the namespace is discarded and the tool
says so.

Blocks that look like they change state — writing or deleting a file, shelling
out, hitting the network — ask for confirmation first. Analysis passes
silently. Confirming everything would train you to click through without
reading, which is worse than no gate.

### 2. Memory that is correctable

Three scopes, plain markdown, no service:

| Scope | File | Lifetime |
|---|---|---|
| `repo` | `memories/repo/<project>.md` | conventions, incidents, recipes |
| `session` | `memories/session/<id>.md` | this task only |
| `user` | `memories/user.md` | cross-project preferences |

The head of the repo notes is put in front of the model at the start of every
task — it never has to remember to go looking — and the prompt tells it to
write back before finishing. Notes are **prepended**, so the freshest load
first.

It is correctable on purpose. Append-only memory keeps a wrong conclusion in
the file forever, still readable and still wrong, and the next session trusts
it as readily as a good one:

| Tool | Use when |
|---|---|
| `memory_append` | record a lesson, rule or recipe |
| `memory_revise` | the note exists but is wrong in detail |
| `memory_retire` | the whole note is wrong or superseded |

`memory_revise` requires the target text to appear **exactly once** and
refuses both `NOT FOUND` and `AMBIGUOUS` rather than guessing which instance
to change. Both write a rolling `.bak` first.

### 3. A record of what was actually done

Every task writes `sessions/<timestamp>_<id>.md` and `.jsonl`, plus
`latest.md`. The markdown opens with an **Actions** index — every tool call in
order, with ⚠️ on the ones that failed — then the detail, then the answer.

When a run looks hung, this is how you tell whether it is thinking, waiting on
the network, or dead. A summary cannot answer that; the timeline can.

## Making it yours: `system.md`

The harness is **domain-free**. It does not know what your system does or what
is dangerous to change in it. That belongs in `<project>/system.md`, which is
appended to the prompt under "This project":

```markdown
Payments ledger for a multi-tenant SaaS.

DO NOT CHANGE WITHOUT APPROVAL: the `Posting` wire format, the
double-entry invariants in `ledger/validate.py`, or any migration that
rewrites historical rows.

Voucher = an immutable group of postings. Settlement is a separate,
idempotent step. Never assume one implies the other.
```

Keep it short: what must not change without approval, and what the vocabulary
means. `system.md.example` has a fuller template. Point
`AGENT_PROMPT_FILE` elsewhere to keep it out of the tree the agent edits.

Without one the harness still works — it just has to be told each session what
is load-bearing.

## Configuration, and where it comes from

Settings are read from **`.env` beside the harness only**, not from the working
directory. A project's own `.env` usually holds credentials for other systems,
and pulling those into an agent that runs generated Python would expose them
through `os.environ` for no benefit. Only `AGENT_*` keys are ever taken.

Exported `AGENT_*` variables win over the file, and the harness **says so**:

```
key    : set (93 chars)   [from .env]
WARNING: exported variables are overriding your .env:
   AGENT_API_KEY: environment (exported) -- shadows .env
```

That matters because a stale export from an earlier session silently beats the
key you just pasted in — the file looks right and the run still returns 401.

## Tools

| Tool | What it is for |
|---|---|
| `run_command` | anything you would type: git, tests, grep, ssh, docker |
| `read_file` | read a file as numbered lines |
| `write_file` | create or overwrite a file (asks first) |
| `search` | regex search returning `file:line: text` |
| `run_python` | analysis in a persistent interpreter |
| `memory_read` / `memory_search` | read and search memory |
| `memory_append` / `memory_revise` / `memory_retire` | write and correct it |
| `memory_status` | what memory exists and where it lives |

Adding one is a description and a function. **The description is the product**
— the model chooses tools from it, so say when to use it and what it returns.

## Resilience

Transient failures (429 "engine overloaded", 5xx, dropped connections, empty
completions) retry with exponential backoff, honouring `Retry-After`.
Permanent ones — a bad key, an unknown model — fail immediately, because
waiting cannot fix those. One model call is bounded (`AGENT_API_TIMEOUT_SEC`),
and the SDK's own retries are disabled so they cannot stack on ours.

The step budget warns the model before the cut-off, and on exhaustion makes
one final call with **no tools offered** to salvage an answer from what it has
already seen. A run that ends "no answer" has thrown away every step it took.

## Container

```bash
docker compose build
docker compose run --rm agent "explain the retry policy"
```

The project is mounted **read-only** at `/project`. The container has no
docker socket and no ssh keys — it cannot restart services or reach the host.
Widen that with a scoped key or a small API, not with the socket.

**Memory needs a volume**, or every recreate starts the agent knowing nothing:

```yaml
environment:
  AGENT_MEMORY_ROOT: /data/memories
volumes:
  - ./memories:/data/memories
```

`--memory-export` / `--memory-import` move the tree between hosts.

## Copilot-parity layers

The loop above is what an editor agent feels like from the inside. These are
the surfaces that make it feel like one from the outside:

| Layer | Tool / entry point | What it replaces |
|---|---|---|
| Precision edits | `replace_in_file`, `multi_replace_in_file` | whole-file overwrites; ambiguous anchors are refused, three failed edits suggest `git_restore` |
| Orientation | `repo_map`, `glob_files` | the editor's symbol index |
| Git awareness | `git_status`, `git_diff`, `git_log`, `git_show`, `git_commit`, `git_restore` | the SCM view |
| Safety rails | `run_command(approve=...)` + `tools_safety` | the human in the loop: irreversible verbs are refused and reported instead of run |
| Visible plan | `manage_todo_list`, `todo_show` | the editor's todo panel |
| History recall | `session_list`, `session_read`, `session_search` | chat history search |
| Subagents | `run_subagent` | delegated research with a bounded child session |
| Chat UI | `python run.py --serve` | the editor chat pane (stdlib HTTP, streaming tokens) |
| Scheduler | `python run.py --schedule`, `schedule.json` | cron for unattended missions |
| Browser | `browser_*` tools (Playwright, optional) | the editor's browser preview: snapshot refs, click/type, screenshots the model can SEE |
| Images | `view_image`, `run.py --image PATH` | multimodal input: charts, mocks, screenshots as evidence |
| Streaming | on by default (`AGENT_STREAM=1`) | tokens print as they arrive; the web UI shows the answer forming |

The safety rails are the load-bearing part. `tools_safety.classify_command`
sorts every shell command into `ok` / `caution` / `irreversible`; the last
needs the model's explicit `approve=true` **and** an approval path
(`AGENT_AUTO_APPROVE=1` or a TTY). In an unattended run there is usually no
approval path, so the refusal text tells the model to hand the exact command
to a human -- which is what you want a robot to do at 3am.

Browser automation is an optional dependency:

```bash
pip install -r requirements-browser.txt
playwright install chromium
```

Without it every `browser_*` tool returns one clear NOT AVAILABLE line and
the rest of the harness is unaffected. `browser_screenshot` returns the PNG
as a multimodal envelope, so the model verifies UI work with its eyes rather
than by trusting the DOM.

## Documentation

| | |
|---|---|
| [Examples](docs/EXAMPLES.md) | ten worked use cases, with the commands to run |
| [DevOps & infrastructure](docs/DEVOPS.md) | GKE, GCP, Terraform, monitoring — and what to let it do alone |
| [Guide](docs/GUIDE.md) | writing tasks and `system.md`, memory hygiene, troubleshooting |
| [Architecture](docs/ARCHITECTURE.md) | how it works, and why each piece is that shape |

Start with the examples — they show the range faster than any description. If
you are pointing this at infrastructure, read the DevOps one first: it is
mostly about the autonomy boundary, and that is the part that matters.

## Files

```
run.py             CLI (--check, --serve, --schedule*, --memory-export/-import)
agent.py           the loop
webui.py           zero-dependency browser chat UI
scheduler.py       cron-lite daemon over schedule.json
tools.py           tool schemas and handlers
tools_edit.py      exact-string edits, glob, per-file failure tracking
tools_git.py       guarded git verbs
tools_repo_map.py  bounded structural overview (AST for Python)
tools_safety.py    irreversible-command classification and gating
tools_session.py   searchable history of past runs
tools_todo.py      the visible plan
tools_delegate.py  bounded subagents
tools_browser.py   headless chromium (Playwright, optional)
tools_media.py     multimodal image envelopes
prompts.py         the system prompt, plus your system.md
python_exec.py     persistent interpreter for run_python
memory.py          three-scope memory
session.py         per-task action log
config.py          settings, provenance, memory discovery
llm.py             generic provider catalogue: resolve, keys, web-UI catalog
```

## What this is not

Not a framework and not production multi-tenant infrastructure. It is a
single-user tool that runs one task at a time and asks before it writes.
Unattended use (`--schedule`, `AGENT_AUTO_APPROVE`) is opt-in and stays inside
the safety rails above: what the rails refuse, the run reports instead of
doing.

## Licence

MIT. See [LICENSE](LICENSE).
