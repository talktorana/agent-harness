# Using the harness well

The mechanics are in [the README](../README.md). This is the judgement part:
how to get good answers out of it, and what to do when it misbehaves.

---

## First run

```bash
cp .env.example .env      # set AGENT_API_KEY, AGENT_PROJECT_ROOT
python run.py --check
python run.py "explain how the pieces of this project fit together"
```

`--check` makes a real one-token completion. Run it after **any** change to
the endpoint, key or model — it costs one cheap call and saves a task.

Read the session record afterwards:

```bash
cat sessions/latest.md
```

That is what the agent actually did, in order. Worth a look on the first run
so you know what "normal" looks like.

---

## Configuration comes from one place

`AGENT_*` settings are read from **`.env` beside the harness only** — not from
the working directory. That is deliberate: a project's own `.env` usually
holds credentials for other systems, and an agent that runs arbitrary Python
should not have those in its environment.

Exported variables win over the file, and the harness tells you when they do:

```
key    : set (93 chars)   [from .env]
WARNING: exported variables are overriding your .env:
   AGENT_API_KEY: environment (exported) -- shadows .env
```

If a run behaves as though your config is being ignored, look for that line.
A stale `export AGENT_API_KEY=...` from an earlier shell session is the usual
cause, and it is invisible without the warning.

---

## Write a `system.md`

The harness is domain-free. Without a project prompt the agent has no idea
what is load-bearing in your codebase and will have to be told every session.

Put it at `<project>/system.md`, or point `AGENT_PROMPT_FILE` at it. Keep it
short — the agent reads it before every task:

```markdown
# What this is
Multi-tenant billing service. Every number here is real money.

# Do not change without explicit approval
- The `Invoice` wire format — external clients pin it.
- The double-entry invariants in `ledger/validate.py`.
- Anything that makes a failure look like success.

# Vocabulary
- **Charge** — what the customer owes. Immutable once posted.
- **Credit** — a future discount. Not a refund. Never assume one implies the other.

# Working agreements
- Tests: `python -m pytest -q` (unit 4s, integration 90s with the DB up).
- Do not edit generated files under `build/`.
```

Three things belong there and nowhere else:

1. **What must not change without approval.** If the agent cannot tell
   load-bearing behaviour from an ordinary refactor, it will eventually
   refactor one.
2. **The vocabulary.** Every codebase has words that mean something specific.
3. **The workflow.** Which tests are slow, what needs a service running.

`system.md.example` is a fuller template.

---

## Memory: what to record and how to keep it healthy

The agent writes to memory before it finishes. What ends up there is up to
the prompt and to you.

**Worth recording:** root causes and fixes, non-obvious rules, deploy recipes,
traps that cost time, decisions and *why*. Short and specific.

**Not worth recording:** transcripts, restatements of what the code already
says, anything that will be stale next week.

The health problem is real: **append-only memory accumulates wrong
conclusions**, and a wrong note gets trusted as readily as a right one. So
the memory is correctable and you should use it:

```bash
harness "our note about the deploy recipe is out of date — the build moved to
Nix. Fix that note in place rather than adding a new one."
```

The agent has `memory_revise` (fix a detail) and `memory_retire` (remove a
whole note). Both keep a `.bak`.

**Occasionally audit it:**

```bash
harness "read our repo memory and list anything that looks contradicted by
the current code"
```

This is the highest-value maintenance task on a long-running project.

---

## Safety knobs

Two gates, and they are deliberately different:

| Setting | Behaviour |
|---|---|
| `AGENT_CONFIRM_WRITES=1` | `write_file` asks before changing anything |
| `AGENT_CONFIRM_PYTHON_MUTATIONS=1` | `run_python` asks **only** when the block looks like it changes state |

Confirming every block of generated code would train you to click through
without reading, which is worse than no gate at all. So analysis passes
silently and mutations ask.

**For unattended runs** (cron, CI) you will want both off — but run it against
a **read-only** mount and with the memory on a volume you can restore:

```yaml
environment:
  AGENT_CONFIRM_WRITES: "0"
  AGENT_CONFIRM_PYTHON_MUTATIONS: "0"
volumes:
  - ../project:/project:ro      # it cannot write the code anyway
  - ./memories:/data/memories
```

The prompt tells the agent not to change load-bearing behaviour without
approval. **That is guidance, not enforcement.** If you need enforcement, put
it in the tool — a denylist in `tools.py` — because the place to enforce is
the tool, not the prompt.

---

## Reading the session log

```
sessions/2026-10-28T01-33-06_a43aaa2c.md
```

| Section | What it answers |
|---|---|
| header table | how long, how many actions, how many failed |
| **Actions** | *what did it do?* — every tool call in order, ⚠️ on failures |
| Thinking | the model's own narration between calls |
| Detail | each action with its arguments and full result |
| Answer | the final answer |

`latest.md` is the most recent run. `tail -f sessions/*.jsonl` streams live.

**When a run looks hung**, the last event tells you where it stopped: a
`model_call` means it is waiting on the network; an ordinary tool result means
it is mid-analysis; nothing new means it is probably dead.

---

## Containers

```bash
docker compose build
docker compose run --rm agent "explain the retry policy"
```

The project mounts **read-only** at `/project`. There is no docker socket and
no ssh key, so the container cannot restart services or reach the host.

**Memory needs a volume** or every recreate starts the agent knowing nothing.
Move it between hosts with:

```bash
python run.py --memory-export /backup/memories
python run.py --memory-import /backup/memories   # keeps the old tree as .bak
```

---

## Troubleshooting

### `401 Unauthorized` on a task, but `--check` passed

`--check` makes a real completion, so if it passes the key is good. A 401 on
the task then means something else supplied a different key. Look for the
`WARNING: exported variables are overriding your .env` line, and `unset` the
variable.

Historical note: an earlier version of `--check` validated against `GET
/models`, which is **not authenticated** on at least one provider — a bogus
key listed every model and the check said "key accepted". That is why it now
completes a real request.

### `model ... is NOT in the list`

Model ids are per provider. `deepseek-ai/DeepSeek-V4.1-Flash-1017` and
`deepseek/deepseek-v4.1-flash` are the same family on two hosts with two
names. `--check` prints the closest matches; set `AGENT_MODEL` or use
`--model`.

### `429 engine overloaded` repeatedly

Normal under load. The harness backs off exponentially and honours
`Retry-After`. If it persists, set `AGENT_FALLBACK_MODEL` to a second model,
or raise `AGENT_API_RETRIES`.

### The run stops at the step limit

It should still answer — the budget warning tells it to wrap up, and on
exhaustion one final call with no tools salvages an answer from what it saw.
If you get nothing, the model was likely mid-investigation on something too
broad. Ask a narrower question.

### A run seems hung

```bash
tail -f sessions/*.jsonl
```

If the last event is `model_call`, it is waiting on the API (bounded by
`AGENT_API_TIMEOUT_SEC`). If it is a tool result, it is working. If there is
nothing recent and no `model_call`, kill it and read the log.

### Commands behave as though the path is wrong

The harness detects its shell and rewrites `/mnt/<drive>/` to `/<drive>/` on
Windows Git Bash, reporting the rewrite in the result. On WSL/Linux it leaves
`/mnt/c/` alone. If you see paths being rewritten that should not be, check
which bash is first on `PATH`.

### `./harness: bad interpreter` or exit 127

CRLF line endings. Git's `autocrlf` rewrites shell scripts on checkout. This
repo pins `*.sh` and `harness` to LF in `.gitattributes`; if you copy the
files without it, the shebang breaks.

---

## Cost

At current pricing for a small model, a typical 20-step task with a few
`run_python` calls costs well under a cent. A 50-step deep investigation of a
large file is still cents.

The thing that actually costs money is **an agent that does not stop**. The
step budget, the tool timeouts and the API timeout all exist for that reason
as much as for reliability.
