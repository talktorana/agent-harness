# Mission latency and prompt-cache operations

Long scheduled runs usually become slow for several independent reasons. Treat
them separately instead of repeatedly tightening one setting.

## Is cache optimization recurring work?

**No. It is a one-time harness/configuration fix with automatic runtime behavior.**
You do not need a periodic "cache optimization" mission or a scheduled cache flush.

| Mechanism | What happens after setup | When to revisit it |
|---|---|---|
| Bounded history | Older complete turn groups are condensed before model calls; full evidence stays in the session log | Requests still grow unexpectedly, or important recent evidence is lost |
| Provider prompt caching | A stable id is sent when configured; the provider reuses eligible prefixes automatically | Provider/model changes, cache-hit rate near zero, or the endpoint rejects the field |
| Adaptive reasoning | Routine calls use the base effort; configured evidence escalates the rest of that run | Measured quality falls, or deep work never/routinely escalates |
| Mission preflight | One read-only call merges harness and application evidence | The health contract changes, or checks become stale/duplicative |

A provider's cache is ephemeral. Expiry is normal: the next request can be a
cache miss and the run still remains correct. Do not pay for synthetic warm-up
calls unless measurement proves they save more than they cost. A stable cache id
is not a credential and does not need periodic rotation; use a different id only
to separate deliberately different prompt prefixes.

Measure production effectiveness from `model_usage` records:

- `prompt_tokens` and `cached_prompt_tokens` show request size and actual hits;
- `reasoning_effort` and `reasoning_tokens` show the depth actually used;
- `history_compacted` shows when conversation history was bounded;
- `reasoning_escalated` shows which evidence selected deeper reasoning.

Absent cache details mean the endpoint did not report them, not that caching is
working or broken. Prove support with a repeated large request before enabling a
dependency on it. Caching is an optimization, never a correctness requirement.

## One-call mission preflight

Set one read-only deployment command:

```bash
AGENT_PREFLIGHT_COMMAND="python3 ops/application_health.py --json"
AGENT_PREFLIGHT_REQUIRE_COMMAND=1
AGENT_PREFLIGHT_TIMEOUT_SEC=45
```

It runs **without a shell**, from `AGENT_PROJECT_ROOT`, and must print one JSON
object:

```json
{
  "fast_path_eligible": true,
  "attention_required": [],
  "application_evidence": {"any": "JSON value"}
}
```

`attention_required` must be a list of strings. A false verdict with no named
evidence is itself an attention condition; it cannot create false quiet. A
nonzero exit, timeout, malformed response, unavailable console, missing previous
scheduled answer, or failed previous scheduled run also selects the deep path.

The harness contributes schedule/settings, the previous **scheduled** delivery
(chat and smoke tests are excluded), and storage. The deployment owns all
application-specific health rules and filenames. Set
`AGENT_PREFLIGHT_REQUIRE_COMMAND=0` only for an intentionally harness-only
monitor; otherwise a missing collector is an attention item rather than an
unexamined application being called healthy.

### Model tool and CLI

```text
mission_preflight()          # one tool call at mission start
python preflight.py          # the same snapshot as JSON for operators
```

Call it once. Do not issue its individual console/file reads again afterward.
The snapshot contains `fast_path_eligible`, `attention_required`, schedule,
settings, previous scheduled delivery, project health, and storage.

## Healthy and deep paths

Use this policy in the scheduled mission:

1. Call `mission_preflight` once.
2. **Healthy path:** if `attention_required` is empty, run only the mission's
   required routine verification once, then report and stop. Do not reopen old
   fixes or launch a broad search because turns remain.
3. **Deep path:** any attention item, failed routine verification, or carried
   open defect selects normal investigation and remediation. Close after the
   named evidence is resolved; do not begin an unrelated audit.
4. Changing a command's timeout, narration, environment decoration, or output
   pipe does not make it a new check. A green result remains final until the
   relevant source changes.

A fast healthy pass is a completed delivery, not a skipped audit. A long pass
that landed no change and merely repeated settled checks is an execution
finding even if its status is `ok`.

## Suggested rollout

1. Keep history compaction and caching optional initially; verify session logs.
2. Configure and run the project collector directly; validate its JSON contract.
3. Run `python preflight.py` inside the deployed environment.
4. Adopt the healthy/deep policy in `mission.md`; regenerate the schedule.
5. If using adaptive reasoning, set a cheap base effort and a deeper effort with
   explicit triggers. Benchmark both against the real mission before rollout.
6. Compare completed runs by wall time, turns, provider time, duplicate calls,
   cache hits and delivered changes. Do not infer speed from configuration.
