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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config
import llm
import tools_media
from memory import MemoryStore
from prompts import build_system
from session import SessionLog
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


def assemble_stream(chunks, emit=None) -> SimpleNamespace:
    """Fold SSE chunks into one message-shaped object.

    Streaming tool calls arrive FRAGMENTED: one chunk may carry the id, the
    next the name, a dozen more slices of the JSON arguments. They are
    reassembled per ``delta.tool_calls[].index`` so the caller sees exactly
    the shape the non-streaming path returns -- the loop below cannot tell
    the difference, which is the point.
    """
    content: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    for chunk in chunks:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
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
                           tool_calls=tool_calls)


class Agent:
    def __init__(self, verbose: bool = True, on_token=None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None):
        # max_retries=0 because _complete() does its own backoff. The SDK's
        # internal retries stacked on top of ours turned one hung call into
        # many minutes of silence with nothing in the transcript to explain it.
        # The endpoint is resolved per task from the provider catalogue
        # (llm.py): explicit choice > providers.json default > environment.
        self.resolved = llm.resolve(provider, model)
        self.client = llm.client_for(self.resolved)
        # The system prompt is built per task, once the session and its memory
        # scope exist -- so this starts empty and run() fills it in.
        self.messages: List[Dict[str, Any]] = []
        self.verbose = verbose
        # Optional sink for streamed tokens (the web UI appends them to the
        # live transcript; the CLI prints them as they arrive).
        self.on_token = on_token
        self.steps = 0
        self.session: Optional[SessionLog] = None
        self._transcript = None
        if config.TRANSCRIPT:
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

    def _log(self, record: Dict[str, Any]) -> None:
        if self._transcript:
            self._transcript.write(json.dumps(record, default=str) + "\n")
            self._transcript.flush()

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

    def _complete(self, use_tools: bool = True):
        """One model turn, ridden through transient failure.

        The endpoint that answers a check call can still be busy when a real
        task lands. Retry the primary model, then (if configured) the fallback,
        and only give up once both are exhausted -- so a busy engine costs time,
        not the whole monitoring pass.
        """
        chain = [self.resolved.model_id]
        if config.FALLBACK_MODEL:
            chain.append(config.FALLBACK_MODEL)
        problems: List[str] = []

        for model in chain:
            for attempt in range(config.API_RETRIES):
                # Recorded BEFORE the request so that a hung call leaves a
                # trail: the last event in the session says which model we
                # were waiting on, instead of the log simply stopping.
                self._log({"event": "model_call", "model": model,
                           "attempt": attempt + 1,
                           "timeout_sec": config.API_TIMEOUT_SEC})
                if self.session:
                    self.session.model_call(model, attempt + 1)
                try:
                    request = {
                        "model": model,
                        "messages": self.messages,
                        "temperature": 0.2,
                    }
                    # Omitting `tools` entirely is what forces a text answer.
                    # Asking with tools still offered lets the model reach for
                    # one more call it will never get to make.
                    if use_tools:
                        request["tools"] = TOOLS
                        request["tool_choice"] = "auto"
                    if config.STREAM and (self.verbose or self.on_token):
                        # Streamed: tokens print as they arrive and the web
                        # UI shows the answer forming. assemble_stream folds
                        # the chunks back into the same message shape the
                        # batched path returns, so nothing downstream differs.
                        stream = self.client.chat.completions.create(
                            **request, stream=True)
                        message = assemble_stream(stream, self._emit_token)
                        if self.verbose:
                            print(flush=True)
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=message)])
                    response = self.client.chat.completions.create(**request)
                except Exception as exc:  # noqa: BLE001
                    if not _is_transient(exc):
                        raise
                    if attempt == config.API_RETRIES - 1:
                        problems.append(f"{model}: {type(exc).__name__}: {exc}")
                        break
                    delay = _backoff(attempt, _retry_after(exc))
                    self._say(f"\n\033[33m[{type(exc).__name__}] "
                              f"{str(exc)[:160]} -- retry {attempt + 1}/"
                              f"{config.API_RETRIES} in {delay:.1f}s\033[0m")
                    self._log({"event": "api_retry", "model": model,
                               "attempt": attempt + 1,
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
                    return response
                if attempt == config.API_RETRIES - 1:
                    problems.append(f"{model}: empty completion")
                    break
                delay = _backoff(attempt, None)
                self._say(f"\n\033[33m[empty completion] retry {attempt + 1}/"
                          f"{config.API_RETRIES} in {delay:.1f}s\033[0m")
                time.sleep(delay)

            if model != chain[-1]:
                self._say(f"\n\033[33m[{model}] exhausted, falling back to "
                          f"{chain[chain.index(model) + 1]}\033[0m")

        raise RuntimeError("; ".join(problems) or "no model available")

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
        # Session record and memory come first: the model should see its own
        # notes before it acts, and the record of what it did should exist even
        # if this dies on the very first API call.
        self.session = SessionLog(task=task, model=self.resolved.label(),
                                  base_url=self.resolved.base_url)
        set_memory_session(self.session.session_id)

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

        for step in range(1, config.MAX_STEPS + 1):
            self.steps = step
            remaining = config.MAX_STEPS - step
            if remaining == max(1, config.ANSWER_WARN_STEPS):
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

        self._say(f"\n[stopped] reached {config.MAX_STEPS} steps")
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
        return (f"Reached the {config.MAX_STEPS}-step limit and could not "
                f"produce a final answer. The work is in the transcript at "
                f"{self.session.md_path if self.session else 'the session log'}.")


def run_task(task: str, verbose: bool = True,
             images: Optional[List[str]] = None,
             on_token=None,
             provider: Optional[str] = None,
             model: Optional[str] = None) -> str:
    """Convenience entry point."""
    return Agent(verbose=verbose, on_token=on_token, provider=provider,
                 model=model).run(task, images=images)
