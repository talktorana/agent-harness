"""A visible plan: the todo list the operator can watch.

A long autonomous run that never says what it intends next is indistinguishable
from one that is lost. Copilot shows a todo list for exactly this reason. Here
the list lives in the session (persisted beside the transcript, so a crash
still shows the plan) and is replaced wholesale on every update -- the same
semantics the editor uses, because a partially-updated plan is a stale plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from config import SESSIONS_DIR

VALID = ("not-started", "in-progress", "completed")

_state: Dict[str, object] = {"session_id": None, "items": []}

_ICON = {"not-started": "[ ]", "in-progress": "[>]", "completed": "[x]"}


def set_todo_session(session_id: str) -> None:
    """Bind the list to the current session and load any saved plan."""
    _state["session_id"] = session_id
    _state["items"] = _load()


def _path() -> Optional[Path]:
    sid = _state.get("session_id")
    if not sid:
        return None
    return Path(SESSIONS_DIR) / f"todo_{sid}.json"


def _load() -> List[Dict[str, object]]:
    p = _path()
    if p is None or not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save() -> None:
    p = _path()
    if p is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_state["items"], indent=2), encoding="utf-8")


def manage_todo_list(items: List[Dict[str, object]]) -> str:
    """Replace the whole plan with ``items``.

    Each item: {"id": n, "title": str, "status": one of not-started /
    in-progress / completed}. At most ONE item may be in-progress -- two
    simultaneous "current" steps means the plan is not a plan. Returns the
    rendered list so the operator sees the change immediately.
    """
    cleaned: List[Dict[str, object]] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", "not-started")).strip().lower()
        if status not in VALID:
            return (f"BAD STATUS {status!r}; use one of {VALID}. "
                    f"The plan was NOT changed.")
        cleaned.append({"id": raw.get("id"),
                        "title": str(raw.get("title", "")).strip(),
                        "status": status})
    running = [i for i in cleaned if i["status"] == "in-progress"]
    if len(running) > 1:
        return ("BAD PLAN: more than one in-progress item "
                f"({[i['title'] for i in running]}). The plan was NOT changed.")
    _state["items"] = cleaned
    _save()
    return todo_show()


def todo_show() -> str:
    """Render the current plan. Empty means no plan was declared yet."""
    items = _state.get("items") or []
    if not items:
        return "(no todo list yet)"
    lines = [f"{_ICON[i['status']]} {i['id']}. {i['title']}" for i in items]
    done = sum(1 for i in items if i["status"] == "completed")
    lines.append(f"-- {done}/{len(items)} completed")
    return "\n".join(lines)
