# Example use cases

Each one is a real shape of task, with the command, what the agent actually
does, and what you get back. Adapt the wording — the quality of the answer
follows the quality of the question.

For infrastructure — GKE, GCP, Terraform, monitoring and incident work — see
[DEVOPS.md](DEVOPS.md). Those examples carry an **autonomy level**, which is
the part worth reading before you run anything.

Assume `harness` is `python run.py`, and `AGENT_PROJECT_ROOT` points at the
codebase you mean.

---

## 1. Understand code you did not write

```bash
harness "explain how the retry policy works, and where it is applied"
```

The agent reads the relevant files, traces the call sites, and reports. Ask
for a specific thing rather than "explain this project" — narrow questions get
narrow, verifiable answers.

**Good follow-ups:**

```bash
harness "which callers of retry() do NOT handle a permanent failure?"
harness "what happens to an in-flight request when the timeout fires?"
```

**What makes this work:** `search` finds every use before the agent concludes
anything, and the answer cites file:line so you can check it.

---

## 2. Find the cause of a bug

Give it the **symptom**, not your theory.

```bash
harness "the export button reports success but the file is empty. Find why."
```

Better than "fix the empty export bug", because the agent has to establish the
cause before changing anything. It will read the handler, trace the write,
and usually find the real reason — which is frequently not the one you assumed.

**Ask it to state the failing test first:**

```bash
harness "the export produces an empty file. State the failing test that would
reproduce it, then fix it and show the test going green."
```

That is the discipline the system prompt asks for: a fix with no reproducing
test is a guess.

---

## 3. Analyse logs and data

This is where `run_python` earns its place. It keeps state across calls, so
the agent loads a file once and then slices it.

```bash
harness "in var/log/app.jsonl, how many requests took over 2s yesterday,
grouped by endpoint? Show the top 5."
```

Typical shape of what it does:

```python
# call 1 - load once
import json
rows = [json.loads(l) for l in open("var/log/app.jsonl")]
slow = [r for r in rows if r["duration_ms"] > 2000]

# call 2 - slice the in-memory data, no re-reading
from collections import Counter
Counter(r["path"] for r in slow).most_common(5)
```

**Ask follow-ups without repeating the load:**

```bash
harness "now split that by status code, and show whether the slow ones cluster
in a time window"
```

The interpreter is still warm, so the second answer costs a fraction of the
first.

---

## 4. Before a rename or a signature change

```bash
harness "I want to rename UserCache to SessionCache. Find every use first and
tell me if anything would break."
```

Do this **before** asking for the change. The answer is often "you cannot",
because the name is a wire format or a config key. Finding that out first is
cheaper than reverting.

Then:

```bash
harness "rename UserCache to SessionCache everywhere, including tests and docs"
```

`write_file` will ask before each file changes, so you see the blast radius as
it goes.

---

## 5. Write tests for code that has none

```bash
harness "add tests for billing/proration.py. Cover the edge cases: a plan
change mid-cycle, a cancellation, and a partial refund."
```

Then check what it produced before trusting it:

```bash
harness "which of the new tests would still pass if proration were deleted?"
```

That question is the one that matters. A test which passes against broken code
proves nothing, and the agent is asked to check for exactly this.

---

## 6. Audit for a class of bug

Much more useful than asking about one file.

```bash
harness "find every place that treats a failed dependency as success. I am
looking for code that fails OPEN — an except block that returns a default,
a cache miss that looks like a cache hit, a retry that returns None."
```

```bash
harness "audit for concurrency: which shared state is read and written
without a lock?"
```

The agent greps, reads the surrounding code, and reports a list with reasons —
including the ones that turned out to be fine.

---

## 7. Explain an incident from evidence

```bash
harness "at 14:32 we started returning 500s for 6 minutes. Look at
var/log/app.jsonl and the git log for that day and tell me what changed."
```

Better than asking "what went wrong", because it forces correlation between
two independent sources — which is where the answer usually is.

---

## 8. Cost or complexity analysis

```bash
harness "run_python to compute, for each module, lines of code, number of
functions, and cyclomatic complexity of the worst function. Sort by complexity."
```

```bash
harness "which dependencies do we import directly but also get transitively?
Could any be removed?"
```

Good use of the persistent interpreter: several steps of parsing, kept in
memory between them.

---

## 9. Documentation from code

```bash
harness "write an ARCHITECTURE.md for this service: the main components, how
they talk to each other, and where state lives. Base it on the code, not on
the existing README."
```

The "base it on the code" clause is what keeps it honest. Documentation
copied from other documentation inherits its errors.

---

## 10. The compounding case: a project you keep coming back to

This is what memory is for. Run the same shape of task over weeks:

```bash
harness "review the changes since the last release and flag anything risky"
```

The first run is expensive. By the tenth, the agent already knows the
deploy recipe, which tests are slow, what the invariants are, and which
"obvious" improvements were tried and reverted. It writes all of that down
before it finishes, and reads it before it starts.

You can watch this happen:

```bash
cat memories/repo/my-project.md | head -60
```

The file is the accumulated judgement of every session — and it is yours, in
markdown, in git, not locked inside a tool.

---

# Writing a task that gets a good answer

| Instead of | Try |
|---|---|
| "fix the bug" | "the export is empty. Find out why, then fix it with a test." |
| "explain the code" | "explain how the retry policy works and where it applies" |
| "make it faster" | "profile the import path and tell me the three slowest steps" |
| "add tests" | "add tests for `proration.py`, especially mid-cycle plan changes" |
| "check for problems" | "find every place that fails open when a dependency is down" |

Three things that reliably help:

1. **Name the symptom, not your theory.** The agent may find a different cause
   than the one you assumed, and it is not anchored if you do not anchor it.
2. **Ask for the evidence first.** "state the failing test, then fix it" keeps
   the agent honest and you informed.
3. **Say what "done" looks like.** "Be concise", "a short answer is better
   than a complete one", "show me the diff" — all of these change the output
   more than you would expect.

If the answer is too long, that is a prompt problem, not a model problem.
