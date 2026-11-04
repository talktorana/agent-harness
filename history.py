"""Bounded conversation history for the agent loop.

Every chat-completions call resends ``messages``. Before this module existed the
list grew for the whole run, so each turn made every later turn more expensive:
a real 2026-11-04 mission sent 25,298 prompt tokens on its first call and
121,248 on its last, processing 9.56 million prompt tokens across 123 calls.
Provider time was 82% of the run's 82 minutes.

The fix is NOT to delete old turns blindly. The agent needs recent evidence,
and the [OI] protocol needs every assistant ``tool_call`` answered by its tool
message. So history is compacted at COMPLETE turn-group boundaries:

* the original system prompt and task are never touched;
* the newest groups are kept verbatim within a character budget;
* older groups become one small, deterministic index with excerpts and a
  pointer to the full session log;
* an assistant and all its tool responses are kept or condensed TOGETHER, so
  the request is always protocol-valid;
* image bytes are never copied into the index (their captions survive).

This is deliberately LOCAL and deterministic: no extra model call, no invented
summary of what the evidence "meant", and no silent truncation of the session
record itself. The complete, unabridged tool results remain in the SessionLog
and are reachable through ``session_read`` / ``session_search``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MARKER = "[history compaction]"


def _text(content: Any) -> str:
    """Text of a message's content, excluding image payloads.

    Multimodal content is a list of parts. Only text parts belong in a size
    estimate or a condensed index: copying a base64 image into every later
    request would defeat the whole exercise.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return "" if content is None else str(content)


def history_chars(messages: Sequence[Dict[str, Any]]) -> int:
    """Characters currently carried by messages, excluding image bytes."""
    return sum(len(_text(m.get("content"))) for m in messages)


def _clip(text: str, limit: int) -> str:
    """Keep the beginning AND the failure-rich end of a result."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    if limit <= 40:
        return text[:limit]
    head = max(20, int(limit * 0.4))
    tail = max(20, limit - head - 30)
    return (text[:head] + "\n... [condensed %d chars] ...\n"
            % (len(text) - head - tail) + text[-tail:])


def _groups(messages: Sequence[Dict[str, Any]]) -> Tuple[List[List[Dict[str, Any]]],
                                                         Optional[str]]:
    """Split after the fixed system+task pair into self-contained groups.

    Returns ``(groups, prior_summary)``. A group beginning with an assistant
    carrying ``tool_calls`` includes every following tool response up to the
    next non-tool message. Thus dropping or keeping a whole group can never
    leave an unanswered tool call in the request.
    """
    groups: List[List[Dict[str, Any]]] = []
    current: Optional[List[Dict[str, Any]]] = None
    prior_summary: Optional[str] = None
    for message in messages[2:]:
        role = message.get("role")
        content = message.get("content")
        if (role == "system" and isinstance(content, str)
                and content.startswith(MARKER)):
            prior_summary = content
            current = None
            continue
        if role == "assistant" and message.get("tool_calls"):
            current = [message]
            groups.append(current)
        elif role == "tool" and current is not None:
            current.append(message)
        else:
            groups.append([message])
            current = None
    return groups, prior_summary


def _summarise_group(group: Sequence[Dict[str, Any]]) -> str:
    """A compact factual index of one group; no inferred conclusions."""
    lines: List[str] = []
    for message in group:
        role = message.get("role")
        text = _text(message.get("content"))
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if text.strip():
                lines.append("assistant: " + _clip(text.strip(), 600))
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or "?"
                args = fn.get("arguments") or "{}"
                lines.append("call %s(%s)" % (name, _clip(args, 300)))
        elif role == "tool":
            lines.append("result: " + (_clip(text, 900) or "[no text]"))
        else:
            lines.append("%s: %s" % (role or "message", _clip(text, 800)))
    return "\n".join(lines)


def compact_history(messages: Sequence[Dict[str, Any]], *,
                    max_chars: int,
                    keep_chars: int,
                    summary_chars: int = 6000,
                    session_file: str = "") -> Tuple[List[Dict[str, Any]],
                                                      Optional[Dict[str, Any]]]:
    """Return ``(messages, info)``, compacting only when ``max_chars`` is hit.

    ``max_chars <= 0`` disables the feature and returns the original list
    object unchanged. Otherwise the fixed system+task messages are preserved,
    recent whole groups are kept within ``keep_chars``, and older groups are
    replaced by one bounded system message. ``info`` is ``None`` when no
    compaction was needed, so callers can log only real events.
    """
    if max_chars <= 0:
        return list(messages), None
    if keep_chars <= 0 or summary_chars < 500:
        raise ValueError("keep_chars must be positive and summary_chars >= 500")
    before_chars = history_chars(messages)
    if before_chars <= max_chars:
        return list(messages), None

    groups, prior_summary = _groups(messages)
    if len(groups) < 2:
        #: One enormous turn cannot be split without breaking tool-call pairing.
        #: It is left intact rather than risking an invalid request.
        return list(messages), None

    kept: List[List[Dict[str, Any]]] = []
    used = 0
    for group in reversed(groups):
        size = sum(len(_text(m.get("content"))) for m in group)
        if kept and used + size > keep_chars:
            break
        kept.insert(0, group)
        used += size
    dropped_count = len(groups) - len(kept)
    if dropped_count <= 0:
        return list(messages), None

    dropped = groups[:dropped_count]
    new_index = "\n\n".join(_summarise_group(g) for g in dropped)
    body = ((prior_summary + "\n\n" if prior_summary else "") + new_index).strip()
    header = (MARKER + " %d earlier turn group(s) condensed locally to bound "
              "prompt size. This is a lossy index, not the full evidence. "
              "The complete untruncated session is %s; use session_read or "
              "session_search for exact earlier outputs and reasoning.\n\n"
              % (dropped_count, session_file or "the current session file"))
    allowance = max(200, summary_chars - len(header))
    summary = header + _clip(body, allowance)

    result = list(messages[:2])
    result.append({"role": "system", "content": summary})
    for group in kept:
        result.extend(group)
    info = {
        "before_messages": len(messages),
        "after_messages": len(result),
        "before_chars": before_chars,
        "after_chars": history_chars(result),
        "groups_condensed": dropped_count,
        "groups_kept": len(kept),
        "summary_chars": len(summary),
        "session_file": session_file,
    }
    return result, info
