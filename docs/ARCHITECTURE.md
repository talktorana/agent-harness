# Architecture

Nine Python files, no framework, no database, no service. This is how it
fits together and — more usefully — *why* each piece is the shape it is.

---

## The loop

```
            ┌──────────────────────────────────────────┐
            │                                          │
            ▼                                          │
   system prompt + task ──► model ──► tool call? ──────┤
                                │                      │
                                │ no                   │ yes
                                ▼                      ▼
                             answer              run the tool
                                                    │
                                                    ▼
                                            result → messages ──┘
```

There is nothing clever here, and that is the point. The behaviour that makes
an agent feel like it *reads the response and decides* is entirely this loop.
The harness's job is narrower than it looks:

1. give the model tools whose results are **legible**,
2. never let a tool **crash** the loop,
3. **stop** it eventually.

Everything else in this document is about doing those three things without
losing the user's afternoon.

---

## Tools: the description is the product

The model chooses tools from their descriptions alone. A vague description
produces vague behaviour, so each one says **when to use it** and **what comes
back** — because the next decision depends on knowing what you will get.

Handlers always return a string and **never raise**. A failed call is data the
model should react to, not a crash that kills the run.

`search` is pure Python rather than a `grep` shell-out. On some systems `grep`
is absent, and a search that silently returns nothing lets the agent conclude
a symbol has no callers when it has several. A tool that can fail *silently*
is worse than no tool.

---

## `run_python`: why a subprocess

Three decisions worth explaining.

**A subprocess, not `exec()`.** A timeout has to be able to *kill* the code,
and you cannot kill a thread. A `while True` in generated analysis must not
hang the whole run. So the interpreter is a child process with a hard timeout;
if it is killed the namespace is discarded and the tool says so rather than
pretending otherwise.

**State persists across calls.** Otherwise the model re-reads a 20,000-line
file on every step. Load once, then aggregate, filter and diff in later calls.

**The trailing expression is returned.** The natural way to write analysis is
imports at the top and the thing you want to see at the bottom. A plain
`exec()` throws that value away and the model gets "(no output)" for the
number it was asking for. The block is parsed with `ast`; if the last
statement is an expression it is evaluated separately.

**The mutation gate.** `run_python` can do anything, so confirming every block
would train the user to click through without reading. Instead the harness
looks for state-changing constructs — file writes, deletions, subprocess,
network, external calls — and asks only then. A needless "are you sure" costs
a keystroke; an unnoticed write to something load-bearing costs real money.

---

## Memory: correctable, not append-only

Three scopes of plain markdown:

| Scope | File | Lifetime |
|---|---|---|
| `repo` | `memories/repo/<project>.md` | conventions, incidents, recipes |
| `session` | `memories/session/<id>.md` | this task only |
| `user` | `memories/user.md` | cross-project |

Two properties make it *agentic* rather than a scratch file:

1. **The agent does not have to remember to read.** A bounded head of the repo
   notes is placed in front of it at task start, exactly as an editor's memory
   works. The rest is reachable through `memory_search`.
2. **It writes back before finishing.** A session that records its lessons
   stops paying to re-derive them.

Notes **prepend**, so the newest material is what loads first.

The interesting design constraint is that **append-only memory has a failure
mode**: a note that turns out wrong stays in the file, still readable and
still wrong, and the next session trusts it as readily as a good one. So it is
correctable — `memory_revise` for a detail, `memory_retire` for a whole note —
and both write a rolling `.bak` first.

`memory_revise` requires the target text to appear **exactly once** and
refuses `NOT FOUND` and `AMBIGUOUS`. Changing the wrong instance is worse than
changing none.

---

## Session record: always on

Every task writes `<timestamp>_<id>.md` and `.jsonl`, plus `latest.md`. The
markdown opens with an **Actions** index before any detail.

This exists because of a real failure: a run looked hung and there was no way
to tell whether the agent was thinking, waiting on the network, or dead. A
summary cannot answer that; a timeline can.

So the log is written **as it happens** — a crash still leaves the record —
and each action is recorded *after* it returns, with the elapsed time. A
`model_call` event is written *before* the request, so a hung API call leaves
a trail naming what was being waited on.

A traceback is flagged as a failed action even though it does not start with
"ERROR" — otherwise a block that raised looks identical to one that printed a
number.

---

## API resilience: transient vs permanent

| | Examples | Behaviour |
|---|---|---|
| **Transient** | 429 "engine overloaded", 5xx, dropped connection, empty completion | exponential backoff, honouring `Retry-After` |
| **Permanent** | 401 bad key, 400 bad request, 404 unknown model | fail immediately |

Waiting cannot turn a 401 into a 200. Burning minutes of backoff before
failing anyway is worse than saying so at once.

Two details matter:

- **`max_retries=0` on the client.** The SDK has its own retries; stacked on
  ours, one hung call became many minutes of silence.
- **One call is bounded** by `AGENT_API_TIMEOUT_SEC`. Without it the client
  waited its own ten-minute default, and the log went quiet with nothing to
  say where it stopped.

---

## The step budget

A step cap with no warning throws the whole run away: the model keeps
investigating to the last step and returns nothing. So:

1. **Warn before the cut-off** — a `system` message N steps out saying answer
   now. A model that knows the budget is nearly spent writes its answer.
2. **Salvage on exhaustion** — one final call with **no tools offered**, so
   the model answers from what it has already seen. Asking while tools are
   still available lets it reach for one more call it will never get to make.

An answer with caveats is strictly more useful than no answer after fifty
steps of good research.

---

## Environment: detect, do not guess

Without being told the platform, the model guessed — it reached for WSL paths
on Windows and then ran `find /` looking for a file already in the tree. One
such call cost 74% of a real session's wall clock.

So the prompt carries an **Environment** block, and it is **derived from
detection, not written by hand**:

```
Shell: MINGW64_NT-10.0 (Git Bash). A Windows drive is written `/c/...`
-- `/mnt/c/...` is WSL syntax and will NOT work.
Project root: `C:\work\ai\ibkrai`. Write paths in one of these forms:
  `C:\work\ai\ibkrai` or `/c/work/ai/ibkrai`
```

That last part is the lesson: **a prompt assertion that is not derived from
reality will drift from it.** This one was confidently written as "Git Bash"
while a different shell was actually executing.

And a suggestion in a prompt is not enforcement — so `run_command` *rewrites*
`/mnt/<drive>/` paths on a Git Bash host and reports the rewrite.

---

## Failure modes actually encountered

Recorded because each one was found by testing, not by reading.

**`--check` was a false green.** It validated the key against `GET /models`,
which is not authenticated on at least one provider — a deliberately bogus key
listed all 94 models and the check printed "key accepted". It surfaced as a
401 mid-task. Now it makes a real completion. *Verify a credential against the
endpoint you will actually use.*

**An exported variable silently beat `.env`.** The loader never overrides the
environment, so a stale `AGENT_API_KEY` from an old shell beat the key just
pasted into the file — which looked correct. Provenance is now recorded and
printed, with an explicit warning when an export shadows the file. *Any config
that layers sources must report which one won.*

**A project's `.env` leaked into the agent.** Reading `.env` from the working
directory imported another system's credentials into the process, where
generated Python could read them from `os.environ`. Only `AGENT_*` keys are
taken now, and only from the harness's own file.

**The tool ran a different shell from the one it detected.** `shutil.which`
found Git Bash; the bare name in the subprocess resolved to WSL. The two
disagree about drive paths and about what `find /` walks. Now the resolved
path is invoked, and the prompt's shell note comes from `uname` on *that*
binary.

**A failed `cd` looked like success.** In `cd /missing && a ; b`, the failed
`cd` skips `a`, `b` runs from the original directory, and `b`'s exit code is
reported. The call did something other than what it said. Rewriting the bad
path removes the whole class.

**Windows line endings broke a shebang.** `autocrlf` rewrote a shell script on
checkout; `./harness` failed with exit 127, which looks exactly like the file
being missing. `.gitattributes` pins shell sources to LF.

**The image copied modules by name.** An explicit `COPY a.py b.py` list meant a
new module worked locally and the container died on import. Now `COPY *.py`.

**A step cap with no warning discarded a good run.** Twice.

**The console crashed after the work was done.** Windows `cp1252` could not
encode an arrow character in the answer, so `print` raised *after* the task had
completed — the work was finished and the result thrown away.

**Memory discovery loaded another project's notes.** Picking the
most-recently-written tree matched the wrong workspace. Now matched by project
name. *Match on content, not on recency.*

**A mistyped memory scope reported "no matches".** Different from "you looked
in the wrong place", and only one is true.

---

## What it is not

Not a framework, not multi-tenant, not an unattended worker. It runs one task
at a time and asks before it writes. That is the right shape for an assistant
you supervise, and the wrong one to leave running against systems you care
about.
