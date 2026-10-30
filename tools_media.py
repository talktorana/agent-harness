"""Multimodal input: let the model SEE images.

OpenAI-style chat messages carry images as base64 ``image_url`` parts, but a
tool result is a string. The bridge is an ENVELOPE: ``view_image`` returns a
string that starts with ``IMAGE_ENVELOPE`` followed by JSON (mime + base64).
The agent loop detects the envelope and converts that one tool message into a
multimodal content list; everything else stays plain text, so session logs and
transcripts never carry megabytes of base64.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

from config import IMAGE_MAX_BYTES, IMAGES_ENABLED, PROJECT_ROOT

#: Prefix marking a tool result as "this string IS an image payload".
IMAGE_ENVELOPE = "\x00IMG\x00"

_ALLOWED = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / path)


def encode_image_bytes(data: bytes, mime: str, path: str = "",
                       note: str = "") -> str:
    """Build the envelope. Used by view_image AND browser_screenshot."""
    payload = {"mime": mime, "data": base64.b64encode(data).decode("ascii"),
               "bytes": len(data)}
    if path:
        payload["path"] = path
    if note:
        payload["note"] = note
    return IMAGE_ENVELOPE + json.dumps(payload)


def parse_image_envelope(result: str):
    """Decode an envelope, or None when the result is ordinary text."""
    if not isinstance(result, str) or not result.startswith(IMAGE_ENVELOPE):
        return None
    try:
        payload = json.loads(result[len(IMAGE_ENVELOPE):])
    except ValueError:
        return None
    if not isinstance(payload, dict) or "data" not in payload:
        return None
    return payload


def data_url(payload: dict) -> str:
    return f"data:{payload.get('mime', 'image/png')};base64,{payload['data']}"


def caption(payload: dict) -> str:
    """The text a transcript shows instead of the base64 blob."""
    parts = [f"[image {payload.get('mime', 'image/png')}",
             f"{payload.get('bytes', 0)} bytes"]
    if payload.get("path"):
        parts.append(payload["path"])
    if payload.get("note"):
        parts.append(payload["note"])
    return " ".join(parts) + "]"


def view_image(path: str, note: str = "") -> str:
    """Attach an image file so the model can see it.

    Use for charts, screenshots, diagrams, UI mocks -- anything where the
    PIXELS are the evidence. Returns an envelope the loop converts into a
    multimodal message; the session log records only the caption.
    """
    if not IMAGES_ENABLED:
        return "IMAGES DISABLED (AGENT_IMAGES=0)."
    p = _resolve(path)
    if not p.is_file():
        return f"NOT FOUND: {p}"
    if p.suffix.lower() not in _ALLOWED:
        return (f"UNSUPPORTED IMAGE TYPE {p.suffix or '(none)'} for {p.name}; "
                f"use png/jpg/jpeg/gif/webp.")
    size = p.stat().st_size
    if size > IMAGE_MAX_BYTES:
        return (f"TOO LARGE: {size} bytes exceeds AGENT_IMAGE_MAX_BYTES="
                f"{IMAGE_MAX_BYTES}. Downscale or crop it first.")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    try:
        data = p.read_bytes()
    except OSError as exc:
        return f"ERROR reading {p}: {exc}"
    return encode_image_bytes(data, mime, str(p), note)
