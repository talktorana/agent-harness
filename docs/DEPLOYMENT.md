# Deployment runbook

A complete guide to deploying the agent harness as a production sidecar that
monitors, diagnoses, and optionally fixes an application.

This covers the full lifecycle: from zero to a running, scheduled, gated agent
with a console, persistent state, and a host applier.

---

## Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Docker + Docker Compose | The agent runs in a container | `docker compose version` |
| A target application | What the agent monitors | A directory with code, tests, or config |
| An [OI]-compatible endpoint | The model the agent talks to | `python run.py --check` |
| A writable persistent volume | Agent state survives recreates | A directory on the host |
| (Optional) A reverse proxy | Authenticated console access | Caddy, nginx, or equivalent |

---

## Step 1: Configure providers

The harness talks to any [OI]-compatible endpoint. You need at least one.

### Option A: Single provider (simplest)

```bash
cp .env.example .env
# Edit .env: set AGENT_BASE_URL, AGENT_API_KEY, AGENT_MODEL
```

### Option B: Multiple providers with fallback (recommended for production)

```bash
cp providers.json.example providers.json
# Edit providers.json: add your providers, set the default

cp routes.json.example routes.json
# Edit routes.json: set the fallback order
```

Keys come from environment variables named by `api_key_env` in providers.json.
Put them in `.env` (gitignored) or export them. Never commit keys.

### Verify

```bash
python run.py --check    # makes a real one-token completion
```

---

## Step 2: Write your mission

The mission is the prompt the scheduler sends on every firing. It is a
Markdown file with a `---` divider: documentation above, prompt below.

```bash
cp mission.md.example mission.md
# Edit mission.md: set the objective, scope, authority, and output format
```

Keep it specific. "Monitor the application" is too vague. "Check that the
payment processor's reconciliation ran, that no duplicate charges appeared,
and that the retry queue is draining" is a mission.

### Generate the schedule

```bash
python make_schedule.py                    # writes schedule.json (disabled)
python make_schedule.py --enable           # writes it enabled
python make_schedule.py --check            # CI: verify it is current
python make_schedule.py --every-min 30     # set the cadence
```

The schedule ships **disabled** by default. Enable it deliberately.

---

## Step 3: Configure the deployment gate

The gate is how the agent changes the target application without having docker
or ssh. It stages requests into a shared outbox; a host applier drains it.

```bash
cp deploy_policy.json.example deploy_policy.json
# Edit deploy_policy.json: set readiness checks, verify commands, forbidden paths
```

Key decisions:

| Decision | Default | When to change |
|---|---|---|
| `readiness.enforce` | `true` | `false` to record without blocking (migration period) |
| `verify.blocking` | `true` | `false` to record without rolling back (testing) |
| `forbid.dirs` | `[".git", "data", "logs", "memories"]` | Add your runtime state dirs |
| `git.enabled` | `false` | `true` when you have a mirror clone and deploy key |

---

## Step 4: Deploy the sidecar

### Copy the production compose file

```bash
cp docker-compose.agent.yml.example docker-compose.agent.yml
```

### Set the environment

The compose file reads from your `.env` and environment. Key variables:

| Variable | Default | What it controls |
|---|---|---|
| `PROJECT_DIR` | `../project` | The target application (mounted read-only) |
| `AGENT_CONFIG_DIR` | `./config` | Providers, routes, schedule, policy (read-only) |
| `AGENT_DATA_DIR` | `./agent-data` | Agent state: sessions, memories, transcripts (persistent) |
| `AGENT_MAX_STEPS` | `200` | Turn ceiling (backstop, not a target) |
| `AGENT_MAX_RUN_SEC` | `0` | Wall-clock bound (0 = unbounded) |
| `AGENT_ROUTE_TIMEOUT_LIMIT` | `4` | Timeouts before stepping to the next route |
| `AGENT_WEBUI_HOST` | `127.0.0.1` | Console bind address |
| `AGENT_WEBUI_PORT` | `8787` | Console port |
| `AGENT_HISTORY_MAX_CHARS` | `0` | Condense older turns past this size (0 = unbounded) |
| `AGENT_HISTORY_KEEP_CHARS` | `35000` | Recent turn groups kept verbatim after compaction |
| `AGENT_HISTORY_SUMMARY_CHARS` | `6000` | Maximum size of the condensed older-turn index |
| `AGENT_CACHE_ID` | empty | Non-secret stable id for provider-side prefix caching |
| `AGENT_PROMPT_CACHE_KEY` | empty | Optional [OI]-style override; wins over `AGENT_CACHE_ID` |

For a long mission, prompt size is often the dominant latency cost: every model
call resends the system prompt, mission, tool schemas and all prior turns.
Enable history compaction when runs become slow. Compaction never rewrites the
original system prompt or task, keeps tool calls paired with their results,
and leaves the full evidence in the session log. It is a request-size
optimisation, not deletion of the run's record.

`AGENT_CACHE_ID` is independent of compaction. A stable id can let the
provider reuse the unchanged prefix across turns and scheduled runs, but support
is endpoint-specific. It is not a credential, so it is safe to declare in a
reviewed compose file; the alternative `AGENT_PROMPT_CACHE_KEY` takes precedence
when both are set. Prove support before relying on it: repeat a large request and
check that `usage.prompt_tokens_details.cached_tokens` is nonzero on the second
call. The agent records `cached_prompt_tokens` and `cache_write_tokens` in each
session so production cache hits can be measured rather than assumed.

### Start

```bash
mkdir -p agent-data/sessions agent-data/memories
docker compose -f docker-compose.agent.yml up -d
docker compose -f docker-compose.agent.yml logs -f
```

### Verify the console

```bash
curl http://127.0.0.1:8787/agent/models    # should return the catalogue
curl http://127.0.0.1:8787/agent/schedule  # should return the schedule
curl http://127.0.0.1:8787/agent/runs.json # should return run history
```

---

## Step 5: Install the host applier

The applier runs on the host, outside the container, and drains the outbox.

### One-shot (manual)

```bash
python deploy_applier.py --once \
    --deploy-root /path/to/project \
    --outbox /path/to/project/.deploy/outbox \
    --policy /path/to/deploy_policy.json
```

### Continuous (systemd)

```ini
# /etc/systemd/system/agent-applier.service
[Unit]
Description=Agent harness deploy applier
After=docker.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/agent-harness/deploy_applier.py \
    --watch \
    --deploy-root /opt/my-project \
    --outbox /opt/my-project/.deploy/outbox \
    --policy /opt/my-project/deploy_policy.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now agent-applier
```

### Continuous (cron, simpler)

```cron
*/3 * * * * /usr/bin/python3 /opt/agent-harness/deploy_applier.py --once --deploy-root /opt/my-project --outbox /opt/my-project/.deploy/outbox --policy /opt/my-project/deploy_policy.json >> /var/log/agent-applier.log 2>&1
```

---

## Step 6: Verify end-to-end

1. The console is reachable: `curl http://127.0.0.1:8787/agent/runs.json`
2. The scheduler is ticking: `curl http://127.0.0.1:8787/agent/schedule`
3. A mission fires and produces a session: `ls agent-data/sessions/`
4. A deploy request is staged: `ls agent-data/../project/.deploy/outbox/`
5. The applier drains it: check the applier log

---

## Recreating the sidecar

When you update the harness code, the sidecar needs a new image and a recreate.
The recreate kills any run in flight, so it is gated.

```bash
# Probe: is a run in flight? (changes nothing)
python recreate_guard.py --json

# Build and recreate only if idle
python recreate_guard.py --build --apply

# Force (destroys the in-flight run, names what it killed)
python recreate_guard.py --build --apply --force
```

The guard reads the scheduler's own state, not the session file's mtime. A
session that ended 5 minutes ago still reads as "running" for ~600 s because
the row is mtime-based; the scheduler's state is authoritative.

---

## Backup and restore

The agent's state is in `AGENT_DATA_DIR` (default `./agent-data`):

| What | Where | Backup priority |
|---|---|---|
| Sessions | `agent-data/sessions/` | High -- the audit trail |
| Memories | `agent-data/memories/` | High -- accumulated knowledge |
| Transcript | `agent-data/transcript.jsonl` | Medium -- redundant with sessions |
| Run settings | `agent-data/sessions/run_settings.json` | Low -- console overrides |
| Model choice | `agent-data/sessions/model_choice.json` | Low -- console picker |
| Schedule state | `agent-data/schedule_state.json` | Low -- derived from schedule.json |

```bash
tar czf agent-backup-$(date +%Y%m%d).tar.gz agent-data/
```

---

## Troubleshooting

### The console is unreachable

```bash
docker compose -f docker-compose.agent.yml ps
docker compose -f docker-compose.agent.yml logs agent
```

### The scheduler is not firing

```bash
curl http://127.0.0.1:8787/agent/schedule
# Check: enabled, every_min, next_due, paused, hold.held
```

### A run is stuck

```bash
curl http://127.0.0.1:8787/agent/runs.json
# Check: scheduler.running, in_progress_sec, next_due
# Read the session file for the run's last action
```

### The applier is not draining

```bash
# Check the outbox
ls /path/to/project/.deploy/outbox/*.deploy.json
# Check the applier log
journalctl -u agent-applier --since "-10min"
# Check the policy
python deploy_applier.py --once --dry-run --policy deploy_policy.json
```
