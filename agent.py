"""The agent loop.

There is nothing clever here, and that is the point. The behaviour the user
recognises in Copilot -- reads a result, decides whether to keep digging or
stop, fixes or investigates further -- is entirely this loop:

    model sees conversation + tool results -> model calls a tool OR writes
    the answer -> if a tool ran, its result joins the conversation -> repeat.

The model is what decides. The harness's job is to (a) give it tools whose
results are legible, (b) never let a tool crash the loop, and (c) stop it
eventually.
"""

import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config
import llm
import run_settings
import storage
import tools_media
from memory import MemoryStore
from prompts import build_system
from session import SessionLog


def _effective_max_steps() -> int:
    """The step ceiling in force for a NEW task.

    Read through ``run_settings`` so a value an operator just set in the
    console applies to the next run -- no redeploy, no recreate. A settings
    file that cannot be parsed raises, and a run must not die because of that,
    so the import-time value from the environment is the fallback. It is used
    only when the override is UNREADABLE, never when it is merely absent.
    """
    try:
        return run_settings.effective_max_steps()[0]
    except Exception as exc:  # noqa: BLE001 -- a bad settings file is reported,
        import sys                # not fatal: the env value still works
        print(f"[warn] run settings unreadable, using AGENT_MAX_STEPS="
              f"{config.MAX_STEPS}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return config.MAX_STEPS


def _effective_answer_warn_steps(max_steps: int) -> int:
    """The wind-down warning distance that goes with ``max_steps``."""
    try:
        return run_settings.effective_answer_warn_steps(max_steps)
    except Exception:  # noqa: BLE001
        return config.ANSWER_WARN_STEPS
from tools import HANDLERS, TOOLS, set_memory_session
from tools_repo_map import build_repo_map

# Statuses that mean "the service is momentarily unable", not "your request is
# wrong". Only these are worth waiting out.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    """True when retrying could plausibly succeed.

    A bad key or an unknown model name is permanent: no amount of waiting turns
    a 401 into a 200, so those must surface at once instead of burning minutes
    of backoff before failing anyway.
    """
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _TRANSIENT_STATUS or bool(status and status >= 500)
    return False

# Statuses that mean "THIS route can never serve the request": unknown model,
# wrong or missing key for that provider, payload the provider refuses (image
# parts on a text-only model, oversized context). Retrying them burns backoff
# budget; the NEXT ROUTE, however, may serve it fine -- which is the entire
# point of the chain.
_ROUTE_DEAD_STATUS = {400, 401, 402, 403, 404, 405, 410, 413, 422}


def _is_route_dead(exc: BaseException) -> bool:
    """True when the current route should be skipped without further retries."""
    from openai import APIStatusError

    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in _ROUTE_DEAD_STATUS:
            return True
    text = str(exc).lower()
    if "model_not_found" in text or "does not exist" in text:
        return True
    return ("not found" in text) and ("model" in text)

def _retry_after(exc: BaseException) -> Optional[float]:
    """Seconds the server asked us to wait, if it said so."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
        return None if raw is None else max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, retry_after: Optional[float]) -> float:
    delay = config.API_BACKOFF_SEC * (2 ** attempt) + random.uniform(0.0, 1.0)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, 60.0)


def assemble_stream(chunks, emit=None, on_think=None, on_usage=None) -> SimpleNamespace:
    """Fold SSE chunks into one message-shaped object.

    Streaming tool calls arrive FRAGMENTED: one chunk may carry the id, the
    next the name, a dozen more slices of the JSON arguments. They are
    reassembled per ``delta.tool_calls[].index`` so the caller sees exactly
    the shape the non-streaming path returns -- the loop below cannot tell
    the difference, which is the point.

    Reasoning models spend their opening minutes emitting
    ``delta.reasoning_content`` and NO ``delta.content`` at all. Discarding
    those chunks silently left the web UI showing a blank bubble for 5m22s of a
    real run -- indistinguishable, to the operator watching it, from a dead
    process. They are now counted and reported through ``on_think`` so a
    liveness signal exists: the reasoning text is never mixed into the answer
    and never printed.
    """
    content: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    thinking_chars = 0
    #: The provider's own verdict on why the turn ended. "length" means the
    #: budget ran out -- possibly entirely inside reasoning -- and the turn is
    #: NOT a completed answer. Recorded rather than ignored because a capped
    #: turn and a finished turn are otherwise indistinguishable downstream.
    finish_reason = None
    #: Signature of the usage measurement already handed to ``on_usage``, so a
    #: provider that sends the same block twice is not recorded twice. See the
    #: emit site inside the loop for why a signature and not a boolean.
    usage_signature: Optional[Any] = None
    for chunk in chunks:
        #: Read usage BEFORE the choices guard below. The chunk that carries
        #: usage is the final one and it has EMPTY choices by design, so a
        #: check that skips choice-less chunks first would discard the only
        #: token counts the provider ever sends -- and the record would show a
        #: call that reasoned for nothing.
        usage = getattr(chunk, "usage", None)
        if usage is not None and on_usage is not None:
            #: ONCE PER DISTINCT MEASUREMENT, not once per chunk. This used to
            #: assume the usage-bearing chunk is unique and final; one provider
            #: sent two chunks with IDENTICAL counts 0.2 ms apart, so a single
            #: call wrote two ``model_usage`` records and every per-run token
            #: total read double -- a silent doubling in exactly the evidence
            #: used to audit what a run cost.
            #:
            #: Comparing a SIGNATURE rather than remembering "already emitted"
            #: keeps both provider styles honest: a stream that repeats one
            #: block is recorded once, and a stream that sends cumulative
            #: counts still records each distinct value, so the last record is
            #: always the total. Emitting as soon as the value appears (rather
            #: than after the loop) preserves the old behaviour of surviving an
            #: exception raised later in the stream.
            details = (getattr(usage, "completion_tokens_details", None)
                       or getattr(usage, "completion_tokens_detail", None))
            sig = (getattr(usage, "prompt_tokens", None),
                   getattr(usage, "completion_tokens", None),
                   getattr(details, "reasoning_tokens", None))
            if sig != usage_signature:
                usage_signature = sig
                try:
                    on_usage(usage)
                except Exception:  # noqa: BLE001
                    pass
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        fr = getattr(choices[0], "finish_reason", None)
        if fr:
            finish_reason = fr
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            thinking_chars += len(reasoning)
            if on_think is not None:
                try:
                    on_think(len(reasoning))
                except Exception:  # noqa: BLE001
                    pass
        text = getattr(delta, "content", None)
        if text:
            content.append(text)
            if emit is not None:
                emit(text)
        for tc in (getattr(delta, "tool_calls", None) or []):
            # index 0 is FALSY: `or`-fallbacks would open a new slot for every
            # fragment of the first parallel call. Only a missing index falls
            # back to the running count.
            idx = getattr(tc, "index", None)
            slot = calls.setdefault(
                idx if isinstance(idx, int) else len(calls),
                {"id": None, "name": None, "args": []})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"].append(fn.arguments)
    tool_calls = None
    if calls:
        tool_calls = [
            SimpleNamespace(
                id=slot["id"] or f"call_{i}",
                function=SimpleNamespace(
                    name=slot["name"] or "",
                    arguments="".join(slot["args"]) or "{}"))
            for i, slot in sorted(calls.items())
        ]
    return SimpleNamespace(content="".join(content) or None,
                           tool_calls=tool_calls,
                           reasoning_chars=thinking_chars,
                           finish_reason=finish_reason)


def build_request(model: str, messages: list, use_tools: bool) -> dict:
    """The chat-completions payload for one model call.

    Extracted from the call site so the payload -- which is where reasoning
    depth is decided -- can be unit-tested directly instead of grepped.

    REASONING DEPTH, two knobs, each sent only when configured:

    ``max_tokens`` is the COMPLETION budget. A reasoning model thinks out of
    that budget before writing anything, so it is the hard ceiling on depth and
    the only lever that binds when a model will not stop on its own.

    ``reasoning_effort`` is sent for providers that honour it. WHETHER THEY DO
    IS A PROPERTY OF THE MODEL, NOT OF THIS PAYLOAD, so it has to be measured
    per model rather than assumed from the last one. Two models on one gateway,
    measured with the budget held constant:

      * one was INERT: 900 reasoning tokens at none, low and max alike against
        a 900 budget, and wall time varied under 2 s;
      * the other HONOURED it, but modestly: mean reasoning tokens 104 (none) /
        134 (low) / 153 (max), so roughly +50% from none to max. Adjacent
        settings OVERLAP at three samples each, which is why a single sample per
        setting is not evidence and why an earlier "+73%" reading of the same
        data was an overstatement.

    So it is never the only thing standing between the agent and a deeper
    think, and it is omitted entirely when empty: the gateway validates the
    field's TYPE but not its VALUE (it accepts "banana"), so a bogus string is
    accepted and silently ignored -- an inert setting is indistinguishable from
    a working one from outside.

    That indistinguishability is why ``_record_usage`` logs the provider's own
    reasoning_tokens for every call: with it, "configured for max effort" and
    "actually reasoned at max effort" become separately checkable facts.

    Omitting ``tools`` entirely is what forces a text answer. Asking with tools
    still offered lets the model reach for one more call it will never get to
    make.
    """
    request: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if config.MAX_TOKENS > 0:
        request["max_tokens"] = config.MAX_TOKENS
    if config.REASONING_EFFORT:
        request["reasoning_effort"] = config.REASONING_EFFORT
    if use_tools:
        request["tools"] = TOOLS
        request["tool_choice"] = "auto"
    return request


class Agent:
    #: Minimum seconds between two liveness events from a silently-reasoning
    #: model. Frequent enough that a 5-minute think still proves it is alive;
    #: coarse enough that a fast stream cannot flood the event feed.
    _THINK_EVENT_SEC = 3.0

    def __init__(self, verbose: bool = True, on_token=None,
                 on_event=None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 origin: Optional[str] = None):
        # max_retries=0 because _complete() does its own backoff. The SDK's
        # internal retries stacked on top of ours turned one hung call into
        # many minutes of silence with nothing in the transcript to explain it.
        # The endpoint is resolved per task from the provider catalogue
        # (llm.py): explicit choice > providers.json default > environment.
        # ROUTING: the task carries an ordered chain of (provider, model)
        # fallbacks (llm.route_chain). Non-primary routes resolve lazily, per
        # turn, so one half-configured route degrades the chain instead of
        # killing the task. The primary resolves now: a configuration error
        # there must fail loudly and early, as before.
        self.route_pairs = llm.route_chain(provider, model)
        self.resolved = llm.resolve(*self.route_pairs[0])
        self.client = llm.client_for(self.resolved)
        self._route_clients: Dict[Any, Any] = {}
        # The system prompt is built per task, once the session and its memory
        # scope exist -- so this starts empty and run() fills it in.
        self.messages: List[Dict[str, Any]] = []
        self.verbose = verbose
        # Optional sink for streamed tokens (the web UI appends them to the
        # live transcript; the CLI prints them as they arrive).
        self.on_token = on_token
        # Optional sink for structured lifecycle events (tool calls, route
        # switches, retries) so a UI can render agent activity as it happens.
        self.on_event = on_event
        #: Who asked for this run. Recorded on the session at start so that the
        #: run history -- which every entry point shares -- can still say which
        #: rows were the timed mission. See SessionLog.origin.
        self.origin = origin or ""
        # Cooperative cancellation: the web UI's Stop button sets this and the
        # step loop exits before the next model call. Set, never kill.
        self.stop_requested = False
        self.steps = 0
        # Liveness during a reasoning model's silent phase. `_on_think` reports
        # progress from the chunk loop; the counters reset per model call.
        self._thinking_route: Optional[str] = None
        self._thinking_chars = 0
        self._last_think_event = 0.0
        self.session: Optional[SessionLog] = None
        self._transcript = None
        if config.TRANSCRIPT:
            #: Rotation happens BEFORE the handle is opened. Renaming a file an
            #: open descriptor still points at keeps writing into the renamed
            #: inode, so the rotation would look correct while the live file
            #: grew on -- the whole point of rotating would be silently lost.
            storage.rotate_transcript(
                config.TRANSCRIPT, config.TRANSCRIPT_MAX_BYTES,
                config.TRANSCRIPT_BACKUPS, on_warn=self._warn)
            self._transcript = open(config.TRANSCRIPT, "a", encoding="utf-8")

    def _close(self) -> None:
        if self._transcript:
            self._transcript.close()
            self._transcript = None

    # ------------------------------------------------------------------ io

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _emit_token(self, text: str) -> None:
        """One streamed token: print without a newline, fan out to the sink."""
        if self.verbose:
            print(text, end="", flush=True)
        if self.on_token is not None:
            try:
                self.on_token(text)
            except Exception:  # noqa: BLE001
                pass

    def _refuse_truncated_turn(self, message: SimpleNamespace,
                               label: str) -> None:
        """Raise when a turn ended on the token cap with nothing to show for it.

        WHY THIS EXISTS (2026-11-03). A reasoning model spends its completion
        budget thinking. If the budget is too small the turn comes back with
        ``finish_reason="length"``, NO content and NO tool calls -- and the
        loop reads "no tool calls" as "the model has finished", so the run ends
        with a thin or empty answer while the transcript looks like a clean
        stop. Nothing in the code inspected ``finish_reason``, so configuring a
        large reasoning budget by mistake was indistinguishable from the model
        having nothing more to say.

        This matters the moment ``AGENT_MAX_TOKENS`` is set explicitly, which
        is exactly what "reason more deeply" requires. Raising here routes it
        through the existing retry/route fallback, so the failure is loud and
        attributable instead of silent.

        Deliberately narrow: only a turn with NO content and NO tool calls is
        refused. A turn that produced an answer and merely hit the cap is a
        partial answer worth keeping -- discarding it would lose real work.
        """
        if getattr(message, "finish_reason", None) != "length":
            return
        if message.content or message.tool_calls:
            return
        self._log({"event": "turn_truncated", "route": label,
                   "max_tokens": config.MAX_TOKENS or None,
                   "reasoning_chars": getattr(message, "reasoning_chars", 0)})
        raise RuntimeError(
            "model turn hit the token cap with no content and no tool call "
            f"(reasoning_chars={getattr(message, 'reasoning_chars', 0)}): "
            f"raise AGENT_MAX_TOKENS above {config.MAX_TOKENS} so the "
            "reasoning can finish and leave room for the answer")

    def _record_usage(self, usage: Any, label: str) -> None:
        """Write what the provider says this call actually cost and SPENT.

        WHY THIS EXISTS. "Configured for maximum reasoning" and "reasoned at
        maximum effort" are different claims, and until now only the first was
        checkable: the request said ``reasoning_effort: max`` and nobody could
        see whether a single extra token was spent on thinking. A setting whose
        effect is invisible cannot be audited, so it gets re-litigated on the
        next pass -- which is exactly what happened.

        Recording the provider's own ``reasoning_tokens`` turns depth into
        evidence in every run's transcript: a call that reports reasoning
        tokens near zero while ``effort=max`` is configured is a fact that can
        be acted on, and a call that reports thousands is proof the setting
        reached the model.

        Field names vary by provider, so every read is defensive: an absent
        usage block records nothing rather than a misleading zero. A zero is
        recorded when the provider genuinely reports zero, because "it thought
        about nothing" is a real and important observation.
        """
        if usage is None:
            return
        details = (getattr(usage, "completion_tokens_details", None)
                   or getattr(usage, "completion_tokens_detail", None))
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        record: Dict[str, Any] = {
            "event": "model_usage",
            "route": label,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": reasoning,
            #: The configured depth travels with the measurement. Without it a
            #: reader has a number and no context for whether it is expected.
            "reasoning_effort": config.REASONING_EFFORT or None,
            "max_tokens": config.MAX_TOKENS or None,
        }
        clean = {k: v for k, v in record.items() if v is not None}

        #: TWO SINKS, because they answer different questions and neither can
        #: stand in for the other. `_log` reaches the transcript, which is the
        #: long stream a human greps. The SESSION file is what `/agent/run/<id>`
        #: and every run-quality analysis read -- and a first version of this
        #: wrote only to the transcript, so the console showed no usage at all
        #: and the record looked absent even though it was being written.
        if self.session is not None:
            try:
                fields = {k: v for k, v in clean.items() if k != "event"}
                self.session.event("model_usage", **fields)
            except Exception:  # noqa: BLE001
                pass
        self._log(clean)

    def _on_think(self, n_chars: int) -> None:
        """Liveness signal from a model that is still reasoning.

        A reasoning model emits ``reasoning_content`` for minutes before the
        first ``content`` token. With nothing to show, the web UI sat on a
        blank bubble and the operator could not tell a busy model from a dead
        process -- which is exactly the confusion that cost an evening here.
        This emits a throttled ``thinking`` event carrying how much reasoning
        has streamed so far; it never touches the answer.
        """
        self._thinking_chars += n_chars
        now = time.monotonic()
        if now - self._last_think_event < self._THINK_EVENT_SEC:
            return
        self._last_think_event = now
        self._log({"event": "thinking", "route": self._thinking_route,
                   "reasoning_chars": self._thinking_chars})

    def _log(self, record: Dict[str, Any]) -> None:
        if self._transcript:
            self._transcript.write(json.dumps(record, default=str) + "\n")
            self._transcript.flush()
        if self.on_event is not None:
            try:
                self.on_event(dict(record))
            except Exception:  # noqa: BLE001
                pass

    def _warn(self, message: str) -> None:
        """One channel for "a thing degraded but the run continues".

        Housekeeping calls this. It goes to the transcript AND the session log
        rather than stderr only, because on a scheduled mission there is no
        terminal: a retention failure that printed to a console nobody watches
        would leave the operator with an unexplained disk problem later and no
        record of when it started.
        """
        self._log({"event": "storage_warning", "detail": str(message)[:400]})
        if self.session is not None:
            try:
                self.session.event("storage_warning",
                                   detail=str(message)[:400])
            except Exception:  # noqa: BLE001 -- never let logging kill a run
                pass

    def _bound_storage(self) -> None:
        """Retention for everything this harness writes, once per run.

        Deliberately run AFTER the session log exists: that makes the newest
        pair (this run's) the one retention never deletes, and any warning is
        recorded in the very log the operator will read.

        Failures are reported, never raised -- a mission must not die because
        tidying an old log failed.
        """
        try:
            pruned = storage.prune_sessions(
                config.SESSIONS_DIR, config.SESSIONS_KEEP,
                config.SESSIONS_MAX_BYTES, on_warn=self._warn)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"session retention failed: {exc}")
            pruned = {}

        root = (Path(config.MEMORY_ROOT).parent
                if config.MEMORY_ROOT else config.SESSIONS_DIR.parent)
        try:
            summary = storage.report(
                root, config.MEMORY_SOFT_MAX_BYTES, on_warn=self._warn)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"storage report failed: {exc}")
            summary = {"sizes": {}, "total_bytes": 0, "warnings": []}

        self._log({"event": "storage",
                   "summary": storage.one_line(summary),
                   "sessions_removed": pruned.get("removed", 0),
                   "sessions_kept": pruned.get("kept", 0),
                   "bytes_freed": pruned.get("freed_bytes", 0),
                   "warnings": list(summary.get("warnings") or [])})

    # --------------------------------------------------------------- tools

    def _run_tool(self, name: str, args: Dict[str, Any]):
        """Run one tool. Returns (log_text, message_content).

        The two differ only for multimodal results: an image envelope logs as
        a one-line caption but reaches the model as a real image part, so the
        transcript stays readable while the model still SEES the pixels.
        """
        handler = HANDLERS.get(name)
        if handler is None:
            return (f"UNKNOWN TOOL: {name}. Available: {sorted(HANDLERS)}",
                    f"UNKNOWN TOOL: {name}. Available: {sorted(HANDLERS)}")
        self._say(f"\n\033[36m[{name}]\033[0m "
                  f"{json.dumps(args, default=str)[:300]}")
        started = time.monotonic()
        try:
            result = handler(**args)
        except TypeError as exc:
            result = f"BAD ARGUMENTS for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001
            # A tool that blows up must return text, not raise: the model
            # needs to see the failure and choose a different route.
            result = f"TOOL ERROR ({type(exc).__name__}): {exc}"
        elapsed = time.monotonic() - started
        content: Any = result
        payload = tools_media.parse_image_envelope(result)
        if payload is not None:
            result = tools_media.caption(payload)
            content = [
                {"type": "text", "text": result},
                {"type": "image_url",
                 "image_url": {"url": tools_media.data_url(payload)}},
            ]
        self._say(f"\033[90m{str(result)[:1200]}\033[0m"
                  + (f"\n\033[90m... [{elapsed:.1f}s]\033[0m"
                     if elapsed > 2 else ""))
        self._log({"event": "tool", "name": name, "args": args,
                   "result": str(result)[:20000], "seconds": round(elapsed, 2)})
        if self.session:
            self.session.action(self.steps, name, args, str(result), elapsed)
        return result, content

    # ----------------------------------------------------------------- api

    def _client_for(self, resolved):
        """The (cached) client for one resolved route endpoint."""
        key = (resolved.base_url, resolved.provider, resolved.model_id)
        client = self._route_clients.get(key)
        if client is None:
            client = llm.client_for(resolved)
            self._route_clients[key] = client
        return client

    def _complete(self, use_tools: bool = True):
        """One model turn, ridden through transient failure and dead routes.

        Two tiers of resilience, learned from overnight runs:

        * TRANSIENT (429, 5xx, timeouts, empty completions) -- retry the SAME
          route with exponential backoff; the engine is briefly busy and will
          answer shortly.
        * ROUTE DEAD (unknown model, wrong key, refused payload) or retries
          exhausted -- move to the NEXT (provider, model) route. A whole
          gateway flapping is a provider-level disease, and stepping across
          providers is exactly what the chain is for.

        Only when every route has spoken do we give up -- and the raised error
        lists every route's verdict, so the log says which door was tried.
        """
        chain = self.route_pairs
        problems: List[str] = []

        for ridx, (prov, alias) in enumerate(chain):
            label = f"{prov}/{alias}"
            try:
                resolved = llm.resolve(prov, alias)
            except llm.ProviderError as exc:
                problems.append(f"{label}: config: {exc}")
                self._say(f"\n\033[33m[route] {label} unusable "
                          f"({str(exc)[:160]}) -- trying next route\033[0m")
                self._log({"event": "route_dead", "route": label,
                           "reason": str(exc)[:500]})
                continue
            client = self._client_for(resolved)
            model = resolved.model_id

            #: Bind the route label by DEFAULT ARGUMENT, not by closure. The
            #: loop rebinds `label` on every iteration, so a closure would
            #: attribute a later attempt's token counts to whichever route
            #: happened to be current when the callback finally fired.
            def on_usage(usage, _label=label):
                self._record_usage(usage, _label)

            for attempt in range(config.API_RETRIES):
                # Recorded BEFORE the request so that a hung call leaves a
                # trail: the last event in the session says which model we
                # were waiting on, instead of the log simply stopping.
                self._log({"event": "model_call", "model": model,
                           "route": label, "attempt": attempt + 1,
                           "timeout_sec": config.API_TIMEOUT_SEC,
                           #: Recorded so a report can state the depth it ran
                           #: at instead of assuming the configured values
                           #: reached the wire. A setting nobody can see from
                           #: the transcript is a setting nobody can audit.
                           "max_tokens": config.MAX_TOKENS or None,
                           "reasoning_effort": config.REASONING_EFFORT or None})
                if self.session:
                    self.session.model_call(model, attempt + 1)
                try:
                    request = build_request(model, self.messages, use_tools)
                    if config.STREAM and (self.verbose or self.on_token):
                        # Streamed: tokens print as they arrive and the web
                        # UI shows the answer forming. assemble_stream folds
                        # the chunks back into the same message shape the
                        # batched path returns, so nothing downstream differs.
                        # Reset per call: the counters belong to THIS turn, and
                        # a stale route would be blamed for a new one's silence.
                        self._thinking_route = label
                        self._thinking_chars = 0
                        self._last_think_event = 0.0
                        stream = client.chat.completions.create(
                            **request, stream=True)
                        message = assemble_stream(stream, self._emit_token,
                                                  on_think=self._on_think,
                                                  on_usage=on_usage)
                        if self.verbose:
                            print(flush=True)
                        self._refuse_truncated_turn(message, label)
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=message)])
                    response = client.chat.completions.create(**request)
                except Exception as exc:  # noqa: BLE001
                    if _is_route_dead(exc):
                        problems.append(f"{label}: {type(exc).__name__}: "
                                        f"{str(exc)[:200]}")
                        self._say(f"\n\033[33m[route] {label} refused "
                                  f"({type(exc).__name__}: {str(exc)[:120]}) "
                                  f"-- trying next route\033[0m")
                        self._log({"event": "route_dead", "route": label,
                                   "error": str(exc)[:500]})
                        break
                    if not _is_transient(exc):
                        raise
                    if attempt == config.API_RETRIES - 1:
                        problems.append(
                            f"{label}: {type(exc).__name__}: {exc}")
                        break
                    delay = _backoff(attempt, _retry_after(exc))
                    self._say(f"\n\033[33m[{type(exc).__name__}] "
                              f"{str(exc)[:160]} -- retry {attempt + 1}/"
                              f"{config.API_RETRIES} in {delay:.1f}s\033[0m")
                    self._log({"event": "api_retry", "model": model,
                               "route": label, "attempt": attempt + 1,
                               "error": str(exc)[:500],
                               "sleep_sec": round(delay, 2)})
                    if self.session:
                        self.session.api_retry(model, attempt + 1, str(exc), delay)
                    time.sleep(delay)
                    continue

                # Some busy engines return 200 with no choices at all. That is
                # the same disease as a 429, so treat it as retryable rather
                # than crashing on response.choices[0].
                if getattr(response, "choices", None):
                    #: Record what this call cost and spent BEFORE judging it.
                    #: A capped turn still consumed reasoning, and that spend
                    #: is the evidence that the depth setting is live -- so it
                    #: must survive the turn being refused below.
                    self._record_usage(getattr(response, "usage", None), label)
                    #: Same guard as the streamed path: a batched turn that
                    #: spent its whole budget reasoning is not an answer.
                    self._refuse_truncated_turn(
                        response.choices[0].message, label)
                    return response
                if attempt == config.API_RETRIES - 1:
                    problems.append(f"{label}: empty completion")
                    break
                delay = _backoff(attempt, None)
                self._say(f"\n\033[33m[empty completion] retry {attempt + 1}/"
                          f"{config.API_RETRIES} in {delay:.1f}s\033[0m")
                time.sleep(delay)

            if ridx < len(chain) - 1:
                nxt = f"{chain[ridx + 1][0]}/{chain[ridx + 1][1]}"
                self._say(f"\n\033[33m[route] {label} exhausted -- falling "
                          f"back to {nxt}\033[0m")
                self._log({"event": "route_switch", "from": label, "to": nxt})

        raise RuntimeError("; ".join(problems) or "no route available")

    def _final_answer(self) -> str:
        """One last request, with tools disabled, to salvage an answer.

        Running out of steps used to return "no answer" -- which discards
        every piece of research already done. Whatever the model can say from
        what it has seen is strictly more useful, as long as it is told to
        mark what it could not verify.
        """
        self.messages.append({
            "role": "user",
            "content": (
                "You have run out of tool steps. Write your final answer NOW "
                "from what you have already seen -- no more tools are "
                "available. Be concise. Mark anything you did not manage to "
                "verify as UNVERIFIED rather than omitting it."),
        })
        try:
            response = self._complete(use_tools=False)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            self._log({"event": "final_answer_failed",
                       "error": str(exc)[:500]})
            return ""

    # ---------------------------------------------------------------- loop

    def run(self, task: str, images: Optional[List[str]] = None) -> str:
        """``_run_impl`` with a guarantee that the session is CLOSED.

        Every normal exit from the loop already calls ``session.finish``, but an
        UNEXPECTED exception (one raised in ``_run_tool``, say, or anywhere the
        loop does not wrap) escaped the whole method with the session still
        open -- so the .jsonl ended with no ``session_end``.

        That mattered because "no end record" is what the console uses to mean
        "the process died", and it was also what a plain crash looked like. The
        two are very different: one means nothing was written for a reason, the
        other means the reason is IN the file. Closing the session on the way
        out makes a missing end record mean exactly one thing -- the process
        went away mid-run (a container restart, an OOM) -- which is the signal
        that actually points at the cause.

        The exception is re-raised so the caller still reports it; this only
        makes sure the record is honest before it does.
        """
        try:
            return self._run_impl(task, images)
        except BaseException as exc:      # noqa: BLE001 - incl. KeyboardInterrupt
            session = getattr(self, "session", None)
            if session is not None:
                try:
                    self._log({"event": "crashed",
                               "type": type(exc).__name__,
                               "error": str(exc)[:1000]})
                    session.event("crashed", type=type(exc).__name__,
                                  error=str(exc)[:1000])
                    session.finish("crashed")
                except Exception:         # noqa: BLE001
                    # Never let the attempt to record a crash become a second
                    # crash; the original exception is the one worth raising.
                    pass
            try:
                self._close()
            except Exception:             # noqa: BLE001
                pass
            raise

    def _run_impl(self, task: str, images: Optional[List[str]] = None) -> str:
        # Session record and memory come first: the model should see its own
        # notes before it acts, and the record of what it did should exist even
        # if this dies on the very first API call.
        self.session = SessionLog(task=task, model=self.resolved.label(),
                                  base_url=self.resolved.base_url,
                                  origin=self.origin)
        set_memory_session(self.session.session_id)
        self._bound_storage()

        memory = MemoryStore(session_id=self.session.session_id)
        # The structural map is built once per task, not per step: it is the
        # same codebase for the whole run, and rebuilding it every turn would
        # pay AST-parse cost for nothing. A failure degrades to no map -- the
        # task must never die while assembling its own prompt.
        try:
            repo_map = (build_repo_map() if config.REPO_MAP_ENABLED else "")
        except Exception as exc:  # noqa: BLE001
            repo_map = ""
            self._log({"event": "repo_map_failed", "error": str(exc)[:300]})
        system = build_system(memory.head("repo"),
                              str(memory.path_for("repo")),
                              str(config.PROJECT_ROOT),
                              repo_map=repo_map)
        self.messages.append({"role": "system", "content": system})
        if images:
            # Attach the pixels up front: a task that says "look at the
            # screenshot" is unanswerable if the screenshot arrives later.
            parts: List[Dict[str, Any]] = [{"type": "text", "text": task}]
            for image in images:
                envelope = tools_media.view_image(image)
                payload = tools_media.parse_image_envelope(envelope)
                if payload is not None:
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": tools_media.data_url(payload)}})
                else:
                    parts.append({"type": "text",
                                  "text": f"[image {image} unavailable: "
                                          f"{envelope}]"})
            self.messages.append({"role": "user", "content": parts})
        else:
            self.messages.append({"role": "user", "content": task})
        self._log({"event": "task", "task": task,
                   "images": list(images or [])})

        #: Read the ceiling ONCE per task, not per turn. A console override can
        #: legitimately change while a run is in flight (an operator watching a
        #: pass head for the cut-off raises it), but a run that re-reads every
        #: turn would also have its warning distance move under it, and the
        #: warning fires on an exact equality -- so the run could skip the only
        #: chance it gets to write an answer. Pinning at task start keeps the
        #: budget coherent for the run and applies the new value to the next.
        max_steps = _effective_max_steps()
        warn_steps = _effective_answer_warn_steps(max_steps)
        self.max_steps = max_steps

        for step in range(1, max_steps + 1):
            self.steps = step
            if self.stop_requested:
                # Cooperative stop: bail between turns, never mid-tool, so a
                # half-applied write can't be left behind by a cancel.
                self._say("\n\033[33m[stopped] cancelled by user\033[0m")
                self._log({"event": "stopped", "step": step})
                if self.session:
                    self.session.finish("stopped")
                self._close()
                return "STOPPED by user request."
            remaining = max_steps - step
            if remaining == max(1, warn_steps):
                # Told before it runs out, not after. A model that knows the
                # budget is nearly spent writes its answer; one that does not
                # keeps exploring and loses everything to the cut-off.
                self.messages.append({
                    "role": "system",
                    "content": (
                        f"Step budget: {remaining} tool call(s) left before "
                        f"this task ends. Stop investigating and write your "
                        f"final answer now from what you already know. Do not "
                        f"start anything new. If a point is unverified, say "
                        f"so rather than leaving it out."),
                })
                self._say(f"\n\033[33m[step budget] {remaining} left -- "
                          f"answer now\033[0m")
                self._log({"event": "budget_warning",
                           "remaining": remaining})
                if self.session:
                    self.session.assistant(f"[step budget] {remaining} left")
            try:
                response = self._complete()
            except Exception as exc:  # noqa: BLE001
                self._say(f"\n[api error] {type(exc).__name__}: {exc}")
                self._log({"event": "api_error", "type": type(exc).__name__,
                           "error": str(exc)[:1000]})
                if self.session:
                    self.session.api_error(type(exc).__name__, str(exc))
                    self.session.finish("api_error")
                self._close()
                return f"API ERROR: {exc}"

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                # The model has stopped asking for tools: it is answering.
                answer = msg.content or ""
                self._log({"event": "answer", "steps": step, "text": answer})
                if self.session:
                    self.session.answer(answer)
                    self.session.finish("ok")
                self._close()
                return answer

            # Record the assistant turn (its reasoning and its tool calls),
            # then run each tool and record the result. Order matters: the API
            # requires every tool_call id to be answered in the same turn.
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            })
            if msg.content:
                self._say(f"\033[33m{msg.content}\033[0m")
                if self.session:
                    self.session.assistant(msg.content)

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"__raw__": tc.function.arguments}
                result, content = self._run_tool(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })

        self._say(f"\n[stopped] reached {max_steps} steps")
        salvaged = self._final_answer()
        if salvaged:
            self._say("\n\033[33m" + salvaged + "\033[0m")
            self._log({"event": "answer", "steps": self.steps,
                       "text": salvaged, "salvaged": True})
            if self.session:
                self.session.answer(salvaged)
                self.session.finish("step_limit_with_answer")
            self._close()
            return salvaged
        if self.session:
            self.session.finish("step_limit")
        self._close()
        return (f"Reached the {max_steps}-step limit and could not "
                f"produce a final answer. The work is in the transcript at "
                f"{self.session.md_path if self.session else 'the session log'}.")


def run_task(task: str, verbose: bool = True,
             images: Optional[List[str]] = None,
             on_token=None,
             provider: Optional[str] = None,
             model: Optional[str] = None,
             origin: Optional[str] = None) -> str:
    """Convenience entry point.

    ``origin`` names the entry point (``schedule``, ``chat``, ``cli``). It is
    optional so an existing caller keeps working, and it is recorded on the
    session: the run history is shared by every entry point, and without the
    tag a console smoke test is indistinguishable from a scheduled mission.
    """
    return Agent(verbose=verbose, on_token=on_token, provider=provider,
                 model=model, origin=origin).run(task, images=images)
