# Agent prompt: using the agent-harness

You are an AI agent. You have been given access to a tool called **agent-harness**,
a self-contained tool-calling loop around a language model. This document tells you
how to use it to monitor, diagnose, and fix an application.

Read this document once, then follow it. Do not ask for clarification on anything
it covers. If something is not covered, use your judgement and say what you assumed.

---

## What agent-harness is

A Python program that wraps a language model in a tool-calling loop. The model
reads files, runs commands, edits code, and accumulates knowledge in memory. It
runs as a container alongside your application, on a schedule, with a console
you can read and a gate that controls what it can change.

The harness is **domain-free**. It does not know what your application does or
what is dangerous to change in it. You tell it through a mission brief and a
project prompt.

---

## How to run it

### Quick check

```bash
python run.py --check    # proves the model endpoint works
```

### One task

```bash
python run.py "explain how the retry policy works"
```

### Console + scheduler (production)

```bash
python run.py --serve --schedule
```

The console is at `http://127.0.0.1:8787/agent` (loopback only -- put an
authenticated proxy in front of it for remote access).

### Docker

```bash
docker compose -f docker-compose.agent.yml up -d
```

---

## The tools the model has

The model calls these tools. You do not call them directly; you describe what
you want in the mission brief and the model decides which tools to use.

| Tool | What it does | When the model uses it |
|---|---|---|
| `read_file` | Read a file's contents | To understand code, config, or logs |
| `write_file` | Create or edit a file | To fix a defect or add a test |
| `search` | Find text across files | To find callers, references, or patterns |
| `run_command` | Run a shell command | To run tests, check status, or inspect the system |
| `run_python` | Run Python in a persistent interpreter | To analyse data, compute metrics, or process logs |
| `memory_append` | Record a lesson or rule | To save what it learned for the next pass |
| `memory_search` | Search past lessons | To recall what it learned before |
| `deploy_request` | Stage a change through the gate | To change the application safely |
| `deploy_status` | Read the gate's verdict | To check whether a change was applied |

---

## How to write a mission

A mission is a Markdown file with a `---` divider. Documentation above, prompt
below. The prompt is what the model reads on every firing.

### Structure

```markdown
# Documentation (not sent to the model)
...

---

### SYSTEM INSTRUCTION EXTENSION ###
You are running as an autonomous operator on [APPLICATION]. Do not pause for
approval and do not ask questions.

### MISSION: [WHAT TO CHECK] ###
[Specific, measurable objective]

### SCOPE ###
[What is in bounds and what is out of bounds]

### STEP BUDGET ###
[Read the live ceiling; do not hardcode a number]

### WHAT TO CHECK ###
1. [First check]
2. [Second check]
3. [Third check]

### AUTHORITY ###
[What it can change and what it must only report]

### TERMINAL SAFELIGHTS ###
- Never place, cancel or amend a production order or transaction.
- If a command fails 3 times on the same file: stop, revert that file, report.
- Irreversible actions: do not run them; print the exact command.
- A suite you could not run is a stated gap, never a pass.

### OUTPUT FORMAT ###
[What the report should look like]
```

### Generate the schedule

```bash
python make_schedule.py                    # writes schedule.json (disabled)
python make_schedule.py --enable           # writes it enabled
python make_schedule.py --every-min 30     # set the cadence
python make_schedule.py --check            # verify it is current (CI)
```

---

## The deployment gate

The agent can stage changes to the application through a gate. The gate is a
host process that reads staged requests, checks a policy, and applies or refuses
them. The agent never has docker, ssh, or any way to bypass the gate.

### How it works

1. The agent calls `deploy_request` with the files it wants to change
2. The request is staged into a shared outbox directory
3. The host applier reads the request and checks the policy
4. If the policy allows it, the applier writes the files and runs a verifier
5. If the verifier fails, the write is rolled back
6. The receipt is written to the outbox
7. The agent reads the receipt with `deploy_status`

### Policy

The policy (`deploy_policy.json`) controls:

- **Readiness**: the application must be in a healthy state before deploying
- **Verification**: a command that proves the change works
- **Forbidden paths**: files the agent may never write (credentials, state, control plane)
- **Actions**: bounded host operations (restart a service, run a migration)
- **Git sync**: best-effort commit and push of applied files

### Configure it

```bash
cp deploy_policy.json.example deploy_policy.json
# Edit: set readiness checks, verify commands, forbidden paths
```

---

## The console

The console is a web UI at `http://127.0.0.1:8787/agent`. It shows:

- **Runs**: every mission that has run, with its status, duration, and result
- **Schedule**: when the next mission fires, whether it is paused, and why
- **Settings**: the live step ceiling and cadence (changeable without a redeploy)
- **Model**: which model the runs use (changeable from the dropdown)
- **Live**: what the current run is doing right now

### Key endpoints

| Endpoint | What it returns |
|---|---|
| `GET /agent/runs.json` | Run history with status, duration, origin |
| `GET /agent/schedule` | Schedule state, next firing, overdue reason |
| `GET /agent/settings` | Live settings (max_steps, every_min) |
| `GET /agent/model` | Which model the runs use |
| `POST /agent/settings` | Change the step ceiling or cadence |
| `POST /agent/model` | Change the model for runs |

---

## Memory

The agent accumulates knowledge in three scopes:

| Scope | Lifetime | What goes there |
|---|---|---|
| `repo` | Permanent | Conventions, incidents, recipes, traps |
| `session` | This task only | In-progress notes |
| `user` | Cross-project | Preferences, patterns |

The agent reads the repo notes at the start of every task and writes back
before finishing. Notes are prepended, so the freshest load first.

Memory is **correctable**: `memory_revise` for a detail, `memory_retire` for a
whole note. Append-only memory keeps a wrong conclusion in the file forever.

---

## Failure modes and what to do

### The model times out

The harness retries with exponential backoff. After `ROUTE_TIMEOUT_LIMIT`
timeouts on one route, it steps to the next route in the chain. If all routes
fail, the run ends with an error.

### The run hits the step ceiling

The model is warned `ANSWER_WARN_STEPS` turns before the cut-off and told to
write its answer. If it still runs out, one final call with no tools offered
salvages an answer from what it already knows. The status is
`step_limit_with_answer`.

### The run hits the time ceiling

Same as the step ceiling, but on wall clock. The status is
`time_limit_with_answer`.

### The run crashes

The session file ends with `crashed` and the traceback. The console shows it
as red.

### The console shows "overdue"

Three causes, only one is a fault:

1. **A run is in flight** -- the loop is sequential, so the next firing waits.
   This is normal.
2. **An operator hold** -- someone paused the scheduler. Deliberate.
3. **Nothing running** -- the daemon stopped polling. This IS a fault.

The console now says which one it is.

---

## Scenario: monitoring a web application

### Mission

```
Check that the payment processor's reconciliation ran in the last hour.
Check that no duplicate charges appeared in the ledger.
Check that the retry queue is draining (not growing).
If any check fails, diagnose the cause and fix it if you can.
Report what you found and what you did.
```

### What the agent does

1. Reads the reconciliation log
2. Queries the ledger for duplicates
3. Checks the retry queue depth
4. If a check fails, reads the relevant code and proposes a fix
5. If the fix is safe, stages it through the deploy gate
6. Writes a report to the console

### What you see

A report in the console's run history, with the status, the findings, and
what was changed.

---

## Scenario: monitoring a Kubernetes cluster

See [KUBERNETES.md](KUBERNETES.md) for the full setup.

### Mission

```
Check all namespaces for pods that are not Running or Completed.
For each one, read its events and last 50 log lines.
Check node resource pressure with kubectl top nodes.
Report any node above 80% memory or CPU.
Do not change anything.
Cite the exact command and output for every finding.
```

### What the agent does

1. Runs `kubectl get pods --all-namespaces`
2. For each non-running pod, reads events and logs
3. Runs `kubectl top nodes`
4. Reports findings with evidence

---

## Checklist for a new deployment

- [ ] `python run.py --check` passes (the model endpoint works)
- [ ] `mission.md` is written and specific (not "monitor the application")
- [ ] `make_schedule.py` generates `schedule.json`
- [ ] `deploy_policy.json` is configured (if using the gate)
- [ ] `docker-compose.agent.yml` points at the right project
- [ ] The console is reachable on loopback
- [ ] A mission fires and produces a session
- [ ] The session shows real work, not just "the process is up"
- [ ] Memory is being written (check `agent-data/memories/`)
- [ ] The applier is draining the outbox (if using the gate)

---

## What NOT to do

- **Do not hardcode model names, step ceilings, or cadence numbers in the
  mission.** They are live settings. Read them from the environment or console.
- **Do not give the agent docker or ssh.** The deploy gate is the only way to
  change the application. If you give it docker, the gate is theatre.
- **Do not put credentials in the mission, the schedule, or the console.**
  They belong in `.env` (gitignored) or a secrets manager.
- **Do not run the harness as root.** The container runs as a non-root user.
- **Do not expose the console on a public IP.** It has no authentication.
- **Do not let the agent write to the project directory directly.** The mount
  is read-only. The deploy gate is the only write path.
