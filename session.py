"""Per-session action log, in the shape a coding session has.

The raw ``AGENT_TRANSCRIPT`` JSONL answers "what did the API see". This answers
"what did the agent DO": an ordered, human-readable trail of every action and
its outcome, written as it happens so a crash or a timeout still leaves the
record intact.

Each session produces two files in ``AGENT_SESSIONS_DIR``:
    <timestamp>_<id>.jsonl   the machine-readable event stream
    <timestamp>_<id>.md      the readable transcript, copied to latest.md
"""

import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import SESSIONS_DIR


def _fence(text: str, lang: str = "") -> str:
    """Wrap in a code fence, switching fence length if the text has backticks."""
    text = str(text)
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _one_line(text: str, limit: int = 160) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _failed(result: str) -> bool:
    """Did this action fail? Drives the ⚠️ markers in the timeline.

    Generated code reports failure as a traceback block rather than an ERROR
    prefix, so it needs its own signal -- otherwise a block that raised looks
    identical to one that printed a number, and the timeline lies about what
    worked.
    """
    s = str(result)
    head = s[:200].lstrip()
    if head.startswith(("ERROR", "TOOL ERROR", "UNKNOWN TOOL", "TIMEOUT",
                        "NOT FOUND", "BAD ARGUMENTS", "DECLINED", "EMPTY",
                        "IS A DIRECTORY", "EMPTY RANGE")):
        return True
    return ("[exception]" in s[:2000]
            or "Traceback (most recent call last)" in s[:2000])


class SessionLog:
    """One run of one task: the actions, in order, with their outcomes."""

    def __init__(self, task: str, sessions_dir: Path = None,
                 model: str = "", base_url: str = ""):
        root = Path(sessions_dir or SESSIONS_DIR)
        root.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        self.session_id = uuid.uuid4().hex[:8]
        self.started = now
        self.started_monotonic = time.monotonic()
        stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
        stem = f"{stamp}_{self.session_id}"

        self.jsonl_path = root / f"{stem}.jsonl"
        self.md_path = root / f"{stem}.md"
        self.model = model
        self.base_url = base_url
        self.task = task

        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._actions: List[Dict[str, Any]] = []
        self._assistant_notes: List[str] = []
        self._answer: Optional[str] = None
        self._errors = 0

        self.event("session_start", task=task, model=model, base_url=base_url)
        self._write_markdown()  # so the file exists even if we crash instantly

    # ------------------------------------------------------------- low level

    def event(self, kind: str, **fields: Any) -> None:
        record = {"event": kind, "at": datetime.now(timezone.utc).isoformat()}
        record.update(fields)
        self._jsonl.write(json.dumps(record, default=str) + "\n")
        self._jsonl.flush()

    # -------------------------------------------------------------- records

    def action(self, step: int, name: str, args: Dict[str, Any],
               result: str, seconds: float) -> None:
        """Record one tool call. This is the spine of the timeline."""
        detail = " ".join(f"{k}={_one_line(v, 90)}"
                          for k, v in (args or {}).items()) or "(no arguments)"
        ok = not _failed(result)
        if not ok:
            self._errors += 1
        entry = {
            "step": step,
            "tool": name,
            "detail": detail,
            "result": result,
            "seconds": round(float(seconds), 2),
            "ok": ok,
        }
        self._actions.append(entry)
        self.event("action", **{k: v for k, v in entry.items()
                                if k != "result"},
                   result_preview=_one_line(result, 300))
        self._write_markdown()

    def assistant(self, text: str) -> None:
        """The model's narration between tool calls -- the 'why' of the next action."""
        if text and text.strip():
            self._assistant_notes.append(text.strip())
            self.event("assistant", text=text[:4000])
            self._write_markdown()

    def model_call(self, model: str, attempt: int) -> None:
        """Record a model request BEFORE it is sent.

        Without this a hung API call leaves the transcript stopping mid-air:
        the last event is an ordinary tool result and there is no way to tell
        whether the agent is thinking, waiting on the network, or dead.
        """
        self.event("model_call", model=model, attempt=attempt)

    def api_retry(self, model: str, attempt: int, error: str,
                  sleep_sec: float) -> None:
        self.event("api_retry", model=model, attempt=attempt,
                   error=error[:300], sleep_sec=round(sleep_sec, 2))

    def api_error(self, kind: str, error: str) -> None:
        self._errors += 1
        self.event("api_error", type=kind, error=error[:1000])

    def answer(self, text: str) -> None:
        self._answer = text
        self.event("answer", text=text[:8000])

    def finish(self, status: str = "ok") -> None:
        self.event("session_end", status=status,
                   steps=len(self._actions),
                   elapsed_sec=round(time.monotonic() - self.started_monotonic, 1))
        self._write_markdown(status=status)
        self._jsonl.close()
        # latest.md is the convenient handle for "what did the last run do".
        try:
            shutil.copyfile(self.md_path, self.md_path.parent / "latest.md")
        except OSError:
            pass

    # -------------------------------------------------------------- markdown

    def _write_markdown(self, status: str = "running") -> None:
        elapsed = time.monotonic() - self.started_monotonic
        lines: List[str] = []
        w = lines.append

        w(f"# Session `{self.session_id}`")
        w("")
        w(f"**Task:** {self.task}")
        w("")
        w("| | |")
        w("|---|---|")
        w(f"| Started | {self.started.strftime('%Y-%m-%d %H:%M:%S')} UTC |")
        w(f"| Status | **{status}** |")
        w(f"| Elapsed | {elapsed:.1f}s |")
        w(f"| Model | `{self.model}` |")
        w(f"| Endpoint | `{self.base_url}` |")
        w(f"| Actions | {len(self._actions)} "
          f"({self._errors} failed) |")
        w("")

        # The action index first: the answer to "what did it do?" before any detail.
        # Numbered by POSITION, not by model turn -- one turn can issue several
        # tool calls, so reusing the turn number made the list read
        # 1, 1, 2, 3, ... which is worse than no numbering at all.
        if self._actions:
            w("## Actions")
            w("")
            for i, a in enumerate(self._actions, 1):
                mark = "" if a["ok"] else " ⚠️"
                w(f"{i}. `{a['tool']}` — {a['detail']} "
                  f"— {a['seconds']}s{mark}")
            w("")

        if self._assistant_notes:
            w("## Thinking")
            w("")
            for note in self._assistant_notes[-20:]:
                w(f"> {note.strip().splitlines()[0]}")
            w("")

        if self._actions:
            w("## Detail")
            w("")
            for i, a in enumerate(self._actions, 1):
                flag = "" if a["ok"] else " ⚠️"
                turn = (f" · turn {a['step']}" if a['step'] != i else "")
                w(f"### {i}. `{a['tool']}`{flag} · {a['seconds']}s{turn}")
                w("")
                w(_fence(a["detail"], "text"))
                w("")
                w(_fence(str(a["result"])[:6000], "text"))
                w("")

        if self._answer:
            w("## Answer")
            w("")
            w(self._answer)
            w("")

        tmp = self.md_path.with_suffix(".md.tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(self.md_path)
