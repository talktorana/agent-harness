"""Deployment requests: how the agent proposes a change it cannot apply.

An autonomous agent pointed at a live trading system must never apply its own
edits directly. The split that makes that safe is proposal vs execution:

* the agent COMPOSES a request -- a schema-versioned JSON document naming
  every target path, its intended bytes and their sha256, plus a purpose,
  optional compose services to restart and an optional verification command;
* a HOST-side applier (``ops/deploy_applier.py`` in the deployment repo)
  decides whether it may be applied, re-verifying path containment,
  checksums, request age, the broker-state gate and single-flight locking,
  and it backs up and can roll back.

``AGENT_DEPLOY_MODE`` only changes WHO invokes that applier, never whether one
runs: ``gated`` (default) leaves the request in the outbox for the host to pick
up, ``direct`` runs the applier in-process after composing, with the same
gates. Nothing in this module can bypass them.

The project mount is read-only in gated mode, so the guard below is
belt-and-braces: it also protects a host-mounted deployment where the agent
does have write access, by routing the change through the same audited path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

SCHEMA_VERSION = 1
REQUEST_SUFFIX = ".deploy.json"
RECEIPT_SUFFIX = ".receipt.json"

def _parse_forbidden(specs) -> tuple:
    """Split the configured refuse list into names, prefixes and suffixes."""
    names, prefixes, suffixes = set(), set(), set()
    for spec in specs or ():
        item = str(spec).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered.startswith("prefix="):
            prefixes.add(item[len("prefix="):])
        elif lowered.startswith("suffix="):
            suffixes.add(item[len("suffix="):])
        else:
            names.add(item)
    return names, tuple(prefixes), tuple(suffixes)


#: Secrets and VCS metadata are refused by EVERY deployment, whatever
#: AGENT_DEPLOY_FORBIDDEN says: a policy can add to this list, never subtract.
GENERIC_FORBIDDEN_NAMES = frozenset({".env"})
GENERIC_FORBIDDEN_PREFIXES = (".env",)
GENERIC_FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
GENERIC_FORBIDDEN_DIRS = frozenset({".git"})

def is_forbidden(target: str) -> bool:
    """True when this target may never be written by an agent, in any mode.

    Deployment-specific names come from the configuration; the generic secret
    and VCS refusals do not, so forgetting to list ``.env`` in a project's
    config cannot make it writable. The configuration is read on every call
    rather than cached at import: a long-running web UI process must see a
    changed refusal list, and a caller that narrows it for one request must
    not have its change ignored because a module-level constant was frozen
    first.
    """
    names, prefixes, suffixes = _parse_forbidden(config.DEPLOY_FORBIDDEN)
    name = Path(target.replace("\\", "/")).name
    lowered = name.lower()
    if name in GENERIC_FORBIDDEN_NAMES or name in names:
        return True
    if any(lowered.startswith(p.lower())
           for p in GENERIC_FORBIDDEN_PREFIXES + prefixes):
        return True
    return any(lowered.endswith(s.lower())
               for s in GENERIC_FORBIDDEN_SUFFIXES + suffixes)


class DeployError(RuntimeError):
    """A deploy request was malformed, refused, or failed to apply."""


def is_gated() -> bool:
    return config.DEPLOY_MODE == "gated"


def outbox_dir() -> Path:
    """Where requests are staged for the host applier to collect."""
    if config.DEPLOY_OUTBOX:
        return Path(config.DEPLOY_OUTBOX).expanduser()
    root = deploy_root()
    return root / ".deploy" / "outbox"


def deploy_root() -> Path:
    """The deployment root the applier will apply relative to."""
    if config.DEPLOY_ROOT:
        return Path(config.DEPLOY_ROOT).expanduser()
    return Path(config.PROJECT_ROOT).expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _check_target(target: str) -> None:
    """Refuse the shapes the host applier would refuse anyway.

    Targets are paths RELATIVE to the deployment root. An absolute path is the
    mistake an agent is most likely to make (it pastes the host path it saw in
    a log), and on Windows ``Path(root) / "/tmp/x"`` silently joins under the
    current drive instead of escaping -- so refusing by shape, not by
    containment, is what keeps this portable.
    """
    raw = (target or "").strip()
    if not raw:
        raise DeployError("deploy target must be a non-empty relative path")
    if raw.startswith("/") or raw.startswith("\\\\"):
        raise DeployError(
            f"deploy target must be relative to the deploy root, not absolute: "
            f"{target!r}")
    drive, _tail = os.path.splitdrive(raw)
    if drive:
        raise DeployError(
            f"deploy target must be relative to the deploy root: {target!r}")
    norm = raw.replace("\\", "/")
    if any(part == ".." for part in norm.split("/")):
        raise DeployError(
            f"deploy target must not traverse outside the deploy root: "
            f"{target!r}")
    name = Path(norm).name
    if is_forbidden(norm):
        raise DeployError(
            f"{name!r} is runtime state or a secret and may never be written "
            f"by an agent, in any mode")
    if ".git" in Path(norm).parts:
        raise DeployError(f"deploy target is inside .git: {target!r}")


def compose_deploy_request(purpose: str,
                           files: List[Dict[str, Any]],
                           services: Optional[List[str]] = None,
                           verify_command: Optional[str] = None,
                           request_id: Optional[str] = None
                           ) -> Dict[str, Any]:
    """Stage one deploy request in the outbox. Returns the request document.

    ``files`` entries are ``{"target": "engine/x.py", "content": "..."}`` or
    ``{"target": ..., "source": "/abs/or/project/path"}`` when the payload
    already exists on disk (staged into a sibling .payload file so the host
    sees bytes, not a path only meaningful inside this container).
    """
    if not purpose or not purpose.strip():
        raise DeployError("'purpose' is required: receipts must say why")
    if not files:
        raise DeployError("a deploy request must name at least one file")
    if len(files) > config.DEPLOY_MAX_FILES:
        raise DeployError(
            f"too many files in one request ({len(files)} > "
            f"{config.DEPLOY_MAX_FILES}); split the change so each request is "
            f"reviewable")

    outbox = outbox_dir()
    outbox.mkdir(parents=True, exist_ok=True)
    rid = request_id or f"req-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"

    entries: List[Dict[str, Any]] = []
    total = 0
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise DeployError(f"files[{i}] must be an object")
        target = str(entry.get("target") or "")
        _check_target(target)
        if "content" in entry:
            content = entry["content"]
            if not isinstance(content, str):
                raise DeployError(f"files[{i}].content must be a string")
            payload = content.encode("utf-8")
            row: Dict[str, Any] = {"target": target,
                                   "content": content}
        else:
            source = entry.get("source")
            if not source:
                raise DeployError(
                    f"files[{i}] needs either 'content' or 'source'")
            src_path = Path(str(source)).expanduser()
            if not src_path.is_absolute():
                src_path = Path(config.PROJECT_ROOT) / src_path
            if not src_path.is_file():
                raise DeployError(f"files[{i}].source does not exist: {src_path}")
            payload = src_path.read_bytes()
            staged = outbox / f"{rid}.{i}.payload"
            staged.write_bytes(payload)
            row = {"target": target, "payload_path": str(staged)}
        total += len(payload)
        row["sha256"] = sha256_bytes(payload)
        row["bytes"] = len(payload)
        entries.append(row)

    if total > config.DEPLOY_MAX_BYTES:
        raise DeployError(
            f"request payload is {total} bytes, over the "
            f"{config.DEPLOY_MAX_BYTES} byte ceiling")

    targets = [str(e.get("target")) for e in entries]
    if len(set(targets)) != len(targets):
        raise DeployError("a request may not target the same file twice")

    request = {
        "schema_version": SCHEMA_VERSION,
        "id": rid,
        "created_at": _now(),
        "purpose": purpose.strip(),
        "files": entries,
        "mode": config.DEPLOY_MODE,
    }
    if services:
        bad = [s for s in services if not isinstance(s, str) or not s.strip()]
        if bad:
            raise DeployError(f"services must be non-empty names: {bad}")
        request["services"] = [s.strip() for s in services]
    if verify_command and verify_command.strip():
        request["verify_command"] = verify_command.strip()

    path = outbox / f"{rid}{REQUEST_SUFFIX}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(request, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    request["_request_path"] = str(path)
    request["_outbox"] = str(outbox)
    return request


def run_applier(extra: Optional[List[str]] = None,
                dry_run: bool = False,
                timeout: float = 1200.0) -> Tuple[int, str]:
    """Invoke the HOST applier over the outbox. Same gates, every mode."""
    applier = config.DEPLOY_APPLIER
    if not applier:
        candidate = deploy_root() / "ops" / "deploy_applier.py"
        applier = str(candidate) if candidate.is_file() else ""
    if not applier or not Path(applier).is_file():
        raise DeployError(
            "no host applier found; set AGENT_DEPLOY_APPLIER to the path of "
            "ops/deploy_applier.py. Direct mode cannot apply changes without "
            "it, and gated mode leaves requests staged for the host.")
    cmd = [sys.executable, applier, "--root", str(deploy_root()),
           "--outbox", str(outbox_dir()), "--once"]
    if dry_run:
        cmd.append("--dry-run")
    if config.DEPLOY_ALLOW_RESTART:
        cmd.append("--allow-restart")
    # The policy carries the deployment's OWN gate (which status file, which
    # states are safe, which filenames are not code). Without it the applier
    # falls back to the generic gates and RECORDS the readiness gate as
    # skipped -- so passing it is what makes the domain gate actually run.
    if config.DEPLOY_POLICY:
        policy_file = Path(config.DEPLOY_POLICY).expanduser()
        if not policy_file.is_absolute():
            policy_file = Path(config.PROJECT_ROOT) / policy_file
        if policy_file.is_file():
            cmd.extend(["--policy", str(policy_file)])
        else:
            raise DeployError(
                f"AGENT_DEPLOY_POLICY points at a missing file: {policy_file}. "
                f"Refusing to run the applier without the deployment's policy, "
                f"since that would silently drop the readiness gate.")
    cmd.extend(extra or [])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-3000:]
    return proc.returncode, tail


def apply_request(request: Dict[str, Any], *, dry_run: bool = False
                  ) -> Dict[str, Any]:
    """Compose (if needed) then run the applier. Returns an outcome dict.

    ``dry_run`` is the rehearsal path: every gate is evaluated and a receipt is
    written, but no target file is touched. That is how a deployment is proven
    before anything live is at risk.
    """
    path = request.get("_request_path")
    if not path:
        composed = compose_deploy_request(
            purpose=str(request.get("purpose") or ""),
            files=list(request.get("files") or []),
            services=request.get("services"),
            verify_command=request.get("verify_command"))
        path = composed["_request_path"]
    receipt_path = Path(str(path)[:-len(REQUEST_SUFFIX)] + RECEIPT_SUFFIX)
    try:
        rc, output = run_applier(dry_run=dry_run)
    except DeployError as exc:
        return {"status": "unapplied", "reason": str(exc),
                "request_path": str(path), "mode": config.DEPLOY_MODE,
                "dry_run": dry_run, "receipt_path": str(receipt_path),
                "applier_output": ""}
    receipt: Dict[str, Any] = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {"status": "unreadable"}
    return {
        "status": receipt.get("status") or ("ok" if rc == 0 else "failed"),
        "reason": receipt.get("reason", ""),
        "applier_rc": rc,
        "request_path": str(path),
        "receipt_path": str(receipt_path) if receipt_path.exists() else "",
        "mode": config.DEPLOY_MODE,
        "dry_run": dry_run,
        "applier_output": output[-1500:],
        "files": receipt.get("files") or [],
        "broker": receipt.get("broker"),
        "backups": receipt.get("backups") or [],
    }


def pending_requests() -> List[Dict[str, Any]]:
    """Summarise what is waiting for the host, and what it last decided."""
    outbox = outbox_dir()
    rows: List[Dict[str, Any]] = []
    if not outbox.is_dir():
        return rows
    for path in sorted(outbox.glob(f"*{REQUEST_SUFFIX}")):
        row: Dict[str, Any] = {"request": path.name, "pending": True}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            row["purpose"] = doc.get("purpose")
            row["created_at"] = doc.get("created_at")
            row["targets"] = [f.get("target") for f in doc.get("files") or []]
            row["services"] = doc.get("services") or []
        except (OSError, ValueError):
            row["error"] = "request unreadable"
        rows.append(row)
    for path in sorted(outbox.glob(f"*{RECEIPT_SUFFIX}")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({"request": doc.get("request") or path.name,
                     "pending": False,
                     "status": doc.get("status"),
                     "reason": doc.get("reason"),
                     "at": doc.get("at"),
                     "dry_run": doc.get("dry_run")})
    return rows


def guard_project_write(path: str) -> Optional[str]:
    """Return a refusal reason if this write must go through a deploy request.

    In gated mode anything landing inside the deployment root is refused: the
    agent's job is to PROPOSE, and the proposal is auditable. Writes outside
    the root (scratch, memory, the outbox itself) are always allowed.
    """
    if not is_gated():
        return None
    try:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = Path(config.PROJECT_ROOT) / target
        resolved = target.resolve()
    except (OSError, ValueError):
        return None
    root = deploy_root().resolve()
    inside = root == resolved or root in resolved.parents
    if not inside:
        return None
    # The outbox lives under the root in the standard layout, and staging a
    # request there is exactly what gated mode wants.
    outbox = outbox_dir().resolve()
    if outbox == resolved or outbox in resolved.parents:
        return None
    return (
        f"AGENT_DEPLOY_MODE=gated: direct edits inside the deployment root "
        f"({root}) are refused. Compose a deploy request instead "
        f"(deploy_request tool) and the host applier will validate, back up "
        f"and apply it. Target was {resolved}.")


# =================================================================== tools

def _summarise_outcome(outcome: Dict[str, Any]) -> str:
    """Render an applier outcome as text the model can act on."""
    lines = [f"status: {outcome.get('status')}",
             f"mode: {outcome.get('mode')}"
             f"{' (DRY RUN - nothing written)' if outcome.get('dry_run') else ''}",
             f"reason: {outcome.get('reason') or '(none)'}"]
    if outcome.get("applier_rc") is not None:
        lines.append(f"applier exit code: {outcome['applier_rc']}")
    broker = outcome.get("broker") or {}
    if broker:
        if broker.get("skipped"):
            lines.append(f"readiness gate: SKIPPED ({broker.get('reason')})")
        else:
            lines.append(f"readiness gate: {broker.get('gate')} state="
                         f"{broker.get('state')!r} "
                         f"age={broker.get('age_sec')}s")
    for row in outcome.get("files") or []:
        lines.append(f"  target {row.get('target')} "
                     f"({row.get('bytes')} bytes, sha256 {row.get('sha256')}...)")
    backups = outcome.get("backups") or []
    if backups:
        lines.append(f"backups: {len(backups)} file(s) copied aside")
    if outcome.get("request_path"):
        lines.append(f"request: {outcome['request_path']}")
    if outcome.get("receipt_path"):
        lines.append(f"receipt: {outcome['receipt_path']}")
    if outcome.get("applier_output"):
        tail = str(outcome["applier_output"])[-900:]
        lines.append("applier output (tail):")
        lines.extend("  " + ln for ln in tail.splitlines())
    return "\n".join(lines)


def deploy_request(purpose: str, files: List[Dict[str, Any]],
                   services: Optional[List[str]] = None,
                   verify_command: Optional[str] = None,
                   dry_run: bool = False) -> str:
    """Propose a change to the deployment; never apply it directly.

    ``files`` entries are {"target": "engine/x.py", "content": "..."} or
    {"target": ..., "source": "path/to/existing/file"}. In the default
    ``gated`` mode this STAGES the request in the outbox and returns
    immediately: a host-side applier decides later whether to apply it, and
    reports its verdict in a receipt. In ``direct`` mode the same applier is
    invoked here, with identical gates.

    ``dry_run`` rehearses the whole thing -- every gate evaluated, a receipt
    written, no target file touched. Use it to prove a deployment before
    anything live is at risk.
    """
    try:
        request = compose_deploy_request(
            purpose=purpose, files=files, services=services,
            verify_command=verify_command)
    except DeployError as exc:
        return f"DEPLOY REQUEST REFUSED while composing: {exc}"
    staged = f"STAGED {request['_request_path']}"
    targets = ", ".join(str(f.get("target")) for f in request["files"])
    header = (f"{staged}\n  purpose: {request['purpose']}\n"
              f"  targets: {targets}\n  mode: {config.DEPLOY_MODE}")
    if services:
        header += f"\n  services to restart: {', '.join(services)}"
    if is_gated() and not dry_run:
        return (header + "\n\nGated mode: the request waits in the outbox for "
                "the HOST applier, which re-checks containment, checksums, "
                "request age and the readiness gate, backs every target up "
                "and can roll back. Read its verdict with deploy_status.")
    try:
        outcome = apply_request(request, dry_run=dry_run)
    except DeployError as exc:
        return header + f"\n\nAPPLIER ERROR: {exc}"
    return header + "\n\n" + _summarise_outcome(outcome)


def deploy_status() -> str:
    """Show what is waiting for the host, and what the host last decided."""
    rows = pending_requests()
    lines = [f"mode: {config.DEPLOY_MODE}",
             f"deploy root: {deploy_root()}",
             f"outbox: {outbox_dir()}",
             f"restart permission: "
             f"{'yes' if config.DEPLOY_ALLOW_RESTART else 'no'}",
             f"policy: {config.DEPLOY_POLICY or '(none -- generic gates only)'}"]
    waiting = [r for r in rows if r.get("pending")]
    decided = [r for r in rows if not r.get("pending")]
    lines.append(f"\npending requests: {len(waiting)}")
    for row in waiting[-10:]:
        lines.append(f"  {row.get('request')}: {row.get('purpose')}")
        lines.append(f"      targets: {row.get('targets')}")
    lines.append(f"decided receipts: {len(decided)}")
    for row in decided[-10:]:
        lines.append(f"  {row.get('request')} -> {row.get('status')}"
                     f"{'' if row.get('dry_run') is not True else ' (dry run)'}"
                     f": {str(row.get('reason') or '')[:160]}")
    if not rows:
        lines.append("(outbox is empty: nothing staged, nothing decided)")
    return "\n".join(lines)
