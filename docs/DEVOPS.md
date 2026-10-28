# DevOps and infrastructure automation

The harness works the same way against infrastructure as against a codebase:
`run_command` reaches `gcloud`, `kubectl`, `terraform`, `helm`, `gh` — anything
on the path. What changes is the blast radius, so the first section is about
that rather than about clever prompts.

Every example below states its **autonomy level**. Read that before copying
the command.

---

## The autonomy ladder

| Level | What it does | Safe unattended? | How |
|---|---|---|---|
| **L0 — inspect** | reads state, reports, changes nothing | **Yes** | read-only credentials |
| **L1 — propose** | writes a plan, a diff or a runbook; you apply it | **Yes** | read-only credentials |
| **L2 — apply on approval** | makes the change after you say yes | You are present | `AGENT_CONFIRM_WRITES=1` |
| **L3 — apply unattended** | makes the change without asking | Only with guardrails | see below |

The honest position: **L0 and L1 are where almost all of the value is.**
Reading a live system quickly and proposing an accurate change is the slow
part of ops work. Applying it is usually the fast part, and it is where the
damage lives.

Most teams that try L3 end up back at L1 plus a human, because the cost of
being wrong on infrastructure is asymmetric.

### How much it does on its own, case by case

| # | Task | Level | What the agent does by itself |
|---|---|---|---|
| 1 | Triage a failing GKE deployment | L0 | reads `describe`, events, logs; reports cause **and** the fix command for you to run |
| 2 | Diagnose a crashlooping pod | L0 | correlates pod logs, previous container, deploy history and git |
| 3 | Node-pool capacity and pressure | L0 | computes requested-vs-allocatable across pools; flags what blocks scheduling |
| 4 | Terraform plan review | L0/L1 | runs `plan`, classifies each change by risk and reversibility, never applies |
| 5 | Drift detection | L0 | plans every root module and summarises drift per module, not per resource |
| 6 | Find what is costing money | L0 | queries billing and asset inventory; writes the deletion **script** for review (L1) |
| 7 | IAM and security audit | L0 | enumerates over-privileged bindings and stale keys; cheap enough to run weekly |
| 8 | Investigate a monitoring alert | L0 | rebuilds the timeline from logs and metrics; separates evidence from inference |
| 9 | SLO and error-budget burn | L0 | does the arithmetic in `run_python`, ranks by burn rate, correlates with deploys |
| 10 | Runbook into a checked script | L1 | **writes** a script with preconditions and refusals; you run it |
| 11 | Post-incident report | L0 | builds the timeline from three independent sources; marks what it inferred |
| 12 | Daily health report | L0 | reads the whole estate on a timer; writes a report while you sleep |
| 13 | Pre-deploy gate | L0/L1 | runs tests and plans; gives a ship/no-ship opinion, does not deploy |
| 14 | Auto-remediation | L2/L3 | acts — only with scoped credentials, a bounded condition and an idempotent remedy |

The pattern: **eleven of these are read-only.** That is not a limitation —
reading a live estate accurately and quickly is most of the work, and it is
the half that does not have a rollback procedure.

### Making L3 survivable, if you must

Separate **plan** from **apply**, and make the plan the artefact:

```bash
# L1 - produce the plan, reviewable, never applied
harness "plan the node-pool resize from n2-standard-4 to n2-standard-8.
Do not apply anything. Give me the terraform diff and the blast radius."

# L2 - apply only after you have read the plan
harness "apply that plan"
```

Then, and only then:

- **scoped credentials** — a service account that can touch exactly one
  project and exactly one resource type
- **a dry-run first, always** — `terraform plan`, `kubectl apply --dry-run=server`,
  `gcloud ... --async` with a check step
- **the session log is your audit trail** — it records every command and its
  output, so a run that did something surprising is reconstructable
- **state you can restore** — snapshot before, or use infrastructure that is
  cheap to rebuild

---

## Setting it up for GCP

### Credentials — use the narrowest that works

| Shape | Use for | Notes |
|---|---|---|
| Your local `gcloud auth` | L0/L1 while you are at the machine | simplest; the agent has whatever you have |
| A mounted **viewer** service account key | L0/L1 unattended | read-only is a real boundary |
| A scoped **deployer** SA, one project | L2/L3 | never `roles/owner`, never org-wide |

A read-only account is not a formality. It is what makes "run it while I
sleep" an acceptable sentence.

**Never mount your personal key, the CI admin key, or anything with
`roles/owner`.** The harness runs generated Python; anything in its
environment is reachable from it.

For the container, mount the key rather than baking it in:

```yaml
volumes:
  - ../project:/project:ro
  - ./memories:/data/memories
  - ${HOME}/.config/gcloud:/home/agent/.config/gcloud:ro
environment:
  GOOGLE_APPLICATION_CREDENTIALS: /home/agent/.config/gcloud/sa.json
  AGENT_CONFIRM_WRITES: "0"          # L0/L1 only
  AGENT_CONFIRM_PYTHON_MUTATIONS: "0"
```

### What to put in `system.md`

Infrastructure has more load-bearing surface than application code. This is
the file that stops an agent refactoring your network:

```markdown
# Do not change without explicit approval
- ANYTHING in the `network/` Terraform module — VPC, subnets, firewall rules.
  A wrong change cuts off every workload at once.
- IAM bindings on the `prod` project. Least-privilege is load-bearing here,
  not a nicety.
- Deletion of any GCS bucket, any Cloud SQL instance, any GKE cluster.
- Anything with a `--delete`, `--purge` or `destroy` in it.

# Vocabulary
- **Stack** = one Terraform root module + the GKE cluster it provisions.
- **Blast radius** = which services go down if this change is wrong.

# Working agreements
- Always `terraform plan` and show me the diff before any apply.
- `kubectl` defaults to the `prod` context. Check `kubectl config
  current-context` and say what it is before running anything.
- Destructive commands are read twice and confirmed once.
```

---

## Examples

### 1. Triage a failing GKE deployment — L0

```bash
harness "the `checkout` deployment in GKE is not reaching Ready. Find out why.
Do not change anything. Report the cause and the exact command you would run
to fix it."
```

What it does: `kubectl describe deploy/checkout`, reads events, checks the
replica set, looks at pod status, pulls the last log lines, and correlates —
then reports. The "do not change anything" plus "give me the command" split
is deliberate: you get the diagnosis and the remedy, and you stay the one who
runs it.

**Autonomy:** safe to leave running. It reads.

---

### 2. Why is this pod crashlooping? — L0

```bash
harness "pods with app=api are crashlooping since about 03:00 UTC. Pull the
logs for the failing ones (previous container too), find the common cause,
and tell me whether it started with a deploy."
```

Useful follow-up that ties two systems together:

```bash
harness "compare the crash start time against the last 5 deploys on that
service and against the git log. Did a release cause this?"
```

That correlation — logs, deploy history, git — is the slow part of an
incident and the part an agent does well.

---

### 3. Node pool capacity and pressure — L0

```bash
harness "why are pods pending in the `workers` node pool? Look at allocatable
vs requested resources, node taints, and any cluster autoscaler events."
```

```bash
harness "run_python to compute, for each node pool, requested CPU and memory
as a percentage of allocatable, and flag any pool over 80% that also has
unschedulable pods. Use `kubectl get nodes -o json` as the input."
```

The second one is the `run_python` shape: fetch once, then several steps of
arithmetic over the data.

---

### 4. Terraform plan review — L0/L1

```bash
harness "run `terraform plan -no-color` in infra/ and review the output.
For each change, say what it does, what could go wrong, and whether it is
reversible. Do not apply."
```

Better than reading the plan yourself when it is 400 lines. Ask it to
classify:

```bash
harness "in that plan, separate the changes into: cosmetic, additive,
destructive, and 'this will cause downtime'. I only care about the last two."
```

**Autonomy:** L0/L1. Produces a review, never applies.

---

### 5. Drift detection — L0

```bash
harness "run terraform plan across every root module under infra/ and tell me
which ones have drifted from what is deployed. Summarise the drift per module,
not per resource."
```

The per-module summary is the point: 200 drifted resources usually means a
hand-edited console change or a partial apply, and the summary surfaces that
pattern.

---

### 6. Find what is costing money — L0

```bash
harness "using gcloud billing and the asset inventory, find resources in
project `acme-prod` that look unused: disks unattached for 30+ days, static
IPs with no forwarding rule, GKE node pools with zero workloads, BigQuery
tables not queried in 90 days. Give me the list and the estimated monthly
saving. Change nothing."
```

```bash
harness "now split that list into 'safe to delete', 'needs a human', and
'probably still needed'. Give me the deletion commands in a script I can
review."
```

The second command is L1: it *writes the script* rather than running it. That
is the right shape for anything destructive.

---

### 7. IAM and security audit — L0

```bash
harness "audit IAM on project `acme-prod`: who has `roles/owner`,
`roles/editor`, or any service-account-key-creating role? Are there keys older
than 90 days? Any allUsers or allAuthenticatedUsers bindings on storage or
BigQuery? Do not change anything."
```

This is a good periodic task. The value is that it is cheap enough to run
weekly rather than once a quarter.

---

### 8. Investigate a monitoring alert — L0

```bash
harness "the alert 'p99 latency > 2s on api-gateway' fired at 14:32 and
resolved 14:38. Using the logs in var/log/ and `gcloud logging read`, tell me
what happened and whether it will recur. Do not change anything."
```

Ask for the honest version:

```bash
harness "and say plainly which parts of that are supported by evidence and
which are inference. I would rather have three facts than a confident story."
```

That clause is worth adding to any incident question. The gap between what
the evidence shows and what the narrative implies is where post-mortems go
wrong.

---

### 9. SLO and error-budget analysis — L0

```bash
harness "compute the 30-day error budget burn for each SLO we define in
slo/*.yaml using the metrics in that file, and rank them by how close they
are to exhaustion. Use run_python for the arithmetic."
```

```bash
harness "for any SLO burning faster than 2x, tell me which endpoints or
status codes are responsible and whether it correlates with the deploys in
the last fortnight."
```

---

### 10. Turn a runbook into an executable procedure — L1

If you have a runbook in markdown:

```bash
harness "read docs/runbooks/scale-up.md and turn it into a script with
explicit checkpoints. Each step should verify its own precondition and refuse
to continue if the state is not what the runbook assumes. Do not run it."
```

The "verify its own precondition and refuse" clause is what makes the script
better than a transcript of commands. It is also what makes it safe to hand to
someone else at 3am.

---

### 11. Post-incident report from evidence — L0

```bash
harness "write the incident report for the 2026-10-28 14:32 outage. Build the
timeline from the logs, the deploy history and the alert stream. Separate
timeline, impact, root cause, and follow-ups. Mark anything you inferred
rather than observed."
```

Better than writing it yourself, and the "mark anything you inferred" keeps
the report honest.

---

### 12. Scheduled autonomous health report — L0

The clearest case for running it unattended, because it only reads:

```bash
# cron, 07:00 daily
harness "daily infrastructure report. Cover: GKE clusters and their node
health, any CrashLoopBackOff or unschedulable pods, deployments not at their
desired replica count, terraform drift, budget burn vs last week, and any new
IAM bindings since yesterday. Write it to reports/YYYY-MM-DD.md and also
summarise in three sentences."
```

**Autonomy:** L0 — safe on a timer with read-only credentials. It writes only
its own report.

That is a genuinely useful autonomous task: it costs cents, it runs while you
sleep, and it turns "I should check" into "here is what I found".

---

### 13. Pre-deploy gate — L0/L1

```bash
harness "before I deploy `api` to prod: run the tests, check the terraform
plan, verify no schema migration is pending that the code assumes, and tell
me whether you would ship it. Do not deploy."
```

That "tell me whether *you* would ship it" gets a real opinion rather than a
list of facts. Worth trying once and seeing whether its judgement matches
yours.

---

### 14. Auto-remediation — L2/L3

```bash
harness "pods with app=api have been crashlooping for 20 minutes. Roll back
to the previous revision. Confirm the pods reach Ready and report the new
revision."
```

This is the one everybody wants and the one to be most careful with.

**Autonomy:** the only example here where the agent acts. Everything before
it produces information; this one produces a *change*. The difference is not
the prompt, it is the credentials and the guardrails below.

- **scoped credentials** — a rollback SA that can only do `deployments/rollback`
  on one service, nothing else
- **a bounded condition** — "crashlooping for 20 minutes" is a testable
  predicate, not a judgement call
- **an idempotent remedy** — rolling back twice is harmless
- **the session log is the audit trail** — read it afterwards
- **a way to stop it** — a kill switch that is not the agent itself

The failure mode is an agent that "fixes" the wrong thing repeatedly. Bounded
conditions and idempotent remedies are what stop that being expensive.

---

## Running it unattended

```bash
# nightly report, read-only
0 7 * * *  cd /opt/agent-harness && ./harness "daily infrastructure report..." \
           >> /var/log/agent/report.log 2>&1
```

Or in CI, as a check that does not merge:

```yaml
- run: python run.py "review this terraform plan and flag anything destructive"
```

What makes unattended operation acceptable:

| | |
|---|---|
| Read-only credentials | the real boundary — everything else is convenience |
| `--memory-export` to version control | so what it learned is reviewable |
| The session log | every command and its output, in order |
| A bounded step budget | it stops; it does not loop |
| Narrow tasks | "report on X" beats "look after the infra" |

What makes it a bad idea:

- credentials that can delete things
- tasks with a vague objective ("keep the cluster healthy")
- no one reading the reports

---

## Things it should never do on its own

Write these into `system.md` so they are in front of the model every run:

- `terraform destroy`, or any plan that removes state
- `gcloud ... delete`, `kubectl delete`, dropping databases
- IAM changes, especially granting `owner`/`editor`
- Anything irreversible that is not idempotent
- Deploying without a plan first

The system prompt tells the agent not to change load-bearing behaviour without
approval. **That is guidance.** If you need it to be a rule, put it in the
tool — a denylist in `tools.py` that refuses a command matching
`terraform destroy|gcloud .* delete|kubectl delete (ns|cluster)` — because the
place to enforce a boundary is the code that executes it, not the prose that
suggests it.
