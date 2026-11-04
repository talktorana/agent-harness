"""Deployment requests: how the agent proposes a change it cannot apply.

An autonomous agent pointed at a live system must never apply its own edits
directly. The split that makes that safe is proposal vs execution:

* the agent COMPOSES a request -- a schema-versioned JSON document naming
  every target path, its intended bytes and their sha256, plus a purpose,
  optional compose services to restart and an optional verification command;
* a HOST-side applier (``ops/deploy_applier.py`` in the deployment repo)
  decides whether it may be applied, re-verifying path containment,
  checksums, request age, the readiness gate and single-flight locking,
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
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import config

SCHEMA_VERSION = 1
REQUEST_SUFFIX = ".deploy.json"
RECEIPT_SUFFIX = ".receipt.json"

#: A field name an ACTION may set. Plain identifiers only: the name is compared
#: against the deployment's allow-list, and a name carrying punctuation or a
#: path separator is not a field -- it is an attempt to reach something else.
_SCALAR_ACTION_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

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

    Three layers, all of which can only ADD refusals:

    * ``GENERIC_*`` -- secrets and VCS metadata, refused by every deployment
      regardless of configuration, so forgetting to list ``.env`` somewhere
      cannot make it writable;
    * the deployment's POLICY FILE -- the same source of truth the host applier
      enforces (see ``policy_forbid``);
    * ``AGENT_DEPLOY_FORBIDDEN`` -- an extra env-level layer for a process that
      has no readable policy, or wants refusals even earlier.

    Read on every call rather than cached at import: a long-running web UI must
    see a changed list, and a caller narrowing it for one request must not have
    that ignored because a module constant froze first.
    """
    names, prefixes, suffixes = _parse_forbidden(config.DEPLOY_FORBIDDEN)
    policy = policy_forbid()
    names = names | {str(n) for n in (policy.get("names") or ())
                     if isinstance(n, str) and n.strip()}
    prefixes += tuple(str(x) for x in (policy.get("prefixes") or ())
                      if isinstance(x, str) and x)
    suffixes += tuple(str(x) for x in (policy.get("suffixes") or ())
                      if isinstance(x, str) and x)
    name = Path(target.replace("\\", "/")).name
    lowered = name.lower()
    if name in GENERIC_FORBIDDEN_NAMES or name in names:
        return True
    if any(lowered.startswith(p.lower())
           for p in GENERIC_FORBIDDEN_PREFIXES + prefixes):
        return True
    return any(lowered.endswith(s.lower())
               for s in GENERIC_FORBIDDEN_SUFFIXES + suffixes)


#: Cache for the policy's refuse list, invalidated on mtime. Re-read every call
#: would stat and parse a file per target; caching forever would miss an
#: operator tightening the policy while the web UI process keeps running.
_POLICY_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "forbid": None}


def reset_policy_cache() -> None:
    """Return the policy cache to its empty state.

    Distinct from ``_POLICY_CACHE.clear()``: clearing empties the dict, and a
    later read does ``cache["path"]`` -- a KeyError on a shape that exists only
    between the two. Tests and config reloads must reset keys, not drop them.
    """
    _POLICY_CACHE.update({"path": None, "mtime": None, "forbid": None})


def _policy_path() -> Optional[Path]:
    raw = (config.DEPLOY_POLICY or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (Path(config.PROJECT_ROOT) / p)


def policy_forbid() -> Dict[str, Any]:
    """The refuse list from the deployment's own policy file.

    Reading the SAME file the host applier enforces is what stops the two
    lists drifting apart. Live 2026-10-30: the policy gained the gate's own
    control-plane filenames, and the ``AGENT_DEPLOY_FORBIDDEN`` env list did
    not, so the agent could stage a request to rewrite ``ops/deploy_policy.json``
    -- caught only later, by the host gate. Defence in depth still held, but
    two disagreeing lists is precisely the condition that turns a second-layer
    failure into a silent bypass.

    Never raises: the host gate remains authoritative, so an unreadable policy
    degrades to "no extra early refusals" rather than making every deploy fail.
    """
    p = _policy_path()
    if p is None:
        return {}
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    cache = _POLICY_CACHE
    if cache["path"] == str(p) and cache["mtime"] == mtime \
            and isinstance(cache["forbid"], dict):
        return cache["forbid"]
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    forbid = doc.get("forbid") if isinstance(doc, dict) else None
    result = forbid if isinstance(forbid, dict) else {}
    cache.update(path=str(p), mtime=mtime, forbid=result)
    return result


def forbidden_dirs() -> frozenset:
    """Directory components no target may live under."""
    policy_dirs = {str(d) for d in (policy_forbid().get("dirs") or ())
                   if isinstance(d, str) and d.strip()}
    return GENERIC_FORBIDDEN_DIRS | frozenset(policy_dirs)


def root_is_writable() -> bool:
    """Can this process actually write inside the deployment root?"""
    try:
        return os.access(str(deploy_root()), os.W_OK)
    except OSError:
        return False


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


#: What ``staged_against_sha256`` says when the target did not exist yet.
#:
#: A distinct literal rather than null, so a reader can tell "there was no file
#: and I am creating one" from "I could not read it" -- the same three-valued
#: discipline the readiness gate uses for a skipped verdict. A null that meant
#: three things at once is how a check quietly stops checking.
STAGED_AGAINST_ABSENT = "absent"
STAGED_AGAINST_UNREADABLE = "unreadable"


def _current_sha256(target: str) -> str:
    """Hash of the target as it stands NOW, for the request to carry.

    Reads through :func:`deploy_root` so it hashes the copy the host will
    overwrite -- not a copy in some other checkout that happens to be nearby.
    This used to call an undefined ``_deploy_root()``; the broad exception then
    recorded EVERY base as "unreadable", so writes could not detect being
    overwritten by a newer request and deletion preconditions could not work.
    """
    try:
        path = deploy_root() / target
    except Exception:  # noqa: BLE001 - a bad root must not stop staging
        return STAGED_AGAINST_UNREADABLE
    try:
        if not path.is_file():
            return STAGED_AGAINST_ABSENT
        return sha256_bytes(path.read_bytes())
    except OSError:
        return STAGED_AGAINST_UNREADABLE


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
    banned_dirs = forbidden_dirs()
    hit = next((part for part in Path(norm).parts[:-1] if part in banned_dirs),
               None)
    if hit is not None:
        raise DeployError(
            f"deploy target is under {hit!r}, which the deployment's policy "
            f"refuses as a target directory: {target!r}")
    if ".git" in Path(norm).parts:
        raise DeployError(f"deploy target is inside .git: {target!r}")


def compose_deploy_request(purpose: str,
                           files: Optional[List[Dict[str, Any]]] = None,
                           services: Optional[List[str]] = None,
                           verify_command: Optional[str] = None,
                           request_id: Optional[str] = None,
                           action: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """Stage one deploy request in the outbox. Returns the request document.

    Two shapes, and exactly one per request:

    * FILES -- ``{"target": "engine/x.py", "content": "..."}``, or
      ``{"target": ..., "source": "/abs/or/project/path"}`` when the payload
      already exists on disk (staged into a sibling .payload file so the host
      sees bytes, not a path only meaningful inside this container);
    * ACTION -- ``{"kind": ..., "instance_id": ..., "set": {...}}``, asking the
      host to run one of its OWN sanctioned commands. That is the only way to
      change runtime state, which lives in files the running service owns and
      caches: writing them behind its back is overwritten on the next save.

    The kinds, the settable fields and the commands are all decided by the
    DEPLOYMENT's policy; this module only carries the request. An action kind
    the deployment has not declared is refused there, and a request that names
    both is refused here -- a change should be one reviewable thing.
    """
    if not purpose or not purpose.strip():
        raise DeployError("'purpose' is required: receipts must say why")
    if action is not None and files:
        raise DeployError(
            "a request is either a file deploy or an action, not both: two "
            "kinds of change in one receipt cannot be reviewed or rolled back "
            "as one unit")
    if action is None and not files and not services:
        #: ...BUT ``services`` ALONE is a complete request: "rebuild these and
        #: change no file". That is the stale-image repair, and refusing to
        #: COMPOSE it was the other half of the gap fixed in the applier on
        #: 2026-11-02 -- the host would have accepted the request while the
        #: agent could not create one, which is the same dead end seen from the
        #: other side. Nothing is relaxed for the empty case: no files, no
        #: action and no services still means the request asks for nothing.
        raise DeployError("a deploy request must name at least one file, an "
                          "'action', or at least one service to rebuild")

    outbox = outbox_dir()
    outbox.mkdir(parents=True, exist_ok=True)
    rid = request_id or f"req-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"

    request: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": rid,
        "created_at": _now(),
        "purpose": purpose.strip(),
        "mode": config.DEPLOY_MODE,
    }

    if action is not None:
        if not isinstance(action, dict):
            raise DeployError("'action' must be an object")
        kind = str(action.get("kind") or "").strip()
        if not kind:
            raise DeployError("action.kind is required")
        instance_id = str(action.get("instance_id") or "").strip()
        if not instance_id:
            raise DeployError("action.instance_id is required")
        changes = action.get("set")
        if not isinstance(changes, dict) or not changes:
            raise DeployError("action.set must be a non-empty object")
        #: Refuse a nested value HERE as well as at the gate. The host would
        #: reject it too, but a request that cannot be valid should fail while
        #: the agent still has budget to correct itself rather than costing a
        #: round trip through the outbox and a quarantine.
        for field, value in changes.items():
            if isinstance(value, (dict, list)):
                raise DeployError(
                    f"action.set[{field!r}] must be a scalar")
            if not _SCALAR_ACTION_FIELD.match(str(field)):
                raise DeployError(
                    f"action.set field name {field!r} is not a plain "
                    f"identifier")
        request["action"] = {"kind": kind, "instance_id": instance_id,
                             "set": {str(k): v for k, v in changes.items()}}
        path = outbox / f"{rid}{REQUEST_SUFFIX}"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(request, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        request["_request_path"] = str(path)
        request["_outbox"] = str(outbox)
        return request

    entries: List[Dict[str, Any]] = []
    total = 0
    #: RESTORED, and it was briefly lost by this very change: restructuring the
    #: function for the action branch dropped this ceiling, and the suite caught
    #: it. A request is reviewable only while it is small enough to read, so the
    #: limit belongs next to the thing it limits.
    if len(files or []) > config.DEPLOY_MAX_FILES:
        raise DeployError(
            f"too many files in one request ({len(files or [])} > "
            f"{config.DEPLOY_MAX_FILES}); split the change so each request is "
            f"reviewable")
    for i, entry in enumerate(files or []):
        if not isinstance(entry, dict):
            raise DeployError(f"files[{i}] must be an object")
        target = str(entry.get("target") or "")
        _check_target(target)
        if entry.get("delete") is True:
            #: DELETE is a file operation, not a new top-level request kind.
            #: An OLD applier rejects this entry because it carries no payload,
            #: so a mixed write+delete request cannot be half-applied by a host
            #: that predates deletion support. That fail-closed property is why
            #: the schema version stays 1.
            if any(k in entry for k in ("content", "source", "sha256",
                                        "payload_path")):
                raise DeployError(
                    f"files[{i}] delete cannot also carry content, source or "
                    f"a payload hash")
            base = _current_sha256(target)
            if base in (STAGED_AGAINST_ABSENT, STAGED_AGAINST_UNREADABLE):
                raise DeployError(
                    f"files[{i}] delete target must be an existing readable "
                    f"file in the deployment, got {base}: {target!r}")
            row = {"target": target, "delete": True,
                   #: A deletion has no payload hash to verify; the BASE hash is
                   #: the payload identity. The host refuses if it changed
                   #: between staging and apply, instead of deleting a newer fix.
                   "staged_against_sha256": base}
            entries.append(row)
            continue
        if "delete" in entry:
            raise DeployError(f"files[{i}].delete must be true when present")
        if "content" in entry:
            content = entry["content"]
            if not isinstance(content, str):
                raise DeployError(f"files[{i}].content must be a string")
            payload = content.encode("utf-8")
            row = {"target": target, "content": content}
        else:
            source = entry.get("source")
            if not source:
                raise DeployError(
                    f"files[{i}] needs 'content', 'source', or 'delete': true")
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
        row["staged_against_sha256"] = _current_sha256(target)
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
    #: REFUSE A DUPLICATE of a request that is still PENDING.
    #:
    #: The brief already says "never re-stage an identical request", and on
    #: 2026-11-02 a monitoring pass did it anyway -- twice in 61 seconds
    #: (req-...-eeadab2d and req-...-4c23fbc2), honestly flagging the second as
    #: its own error. Both were rebuild-only requests for the same service, so
    #: each one runs `docker compose build` AND `docker compose up -d`, i.e. the
    #: trading container is recreated TWICE for one repair. Harmless when the
    #: book is closed, which is the only reason that instance cost nothing.
    #:
    #: A rule stated only in prose does not survive contact with a model that
    #: has lost track of what it already staged -- and the pass cannot always
    #: tell, because `deploy_status` is a snapshot it may read at any point. So
    #: the check lives here, where the duplicate is actually created, and the
    #: refusal NAMES the request already waiting: that is more useful than a
    #: second identical file in the outbox.
    duplicate = _pending_duplicate(outbox, request)
    if duplicate is not None:
        raise DeployError(
            "an identical request is already PENDING in the outbox: "
            f"{duplicate['id']} (staged {duplicate.get('created_at')}). It has "
            "not been decided yet -- read its verdict with deploy_status "
            "instead of staging it again. Two pending copies mean two "
            "rebuild/recreate cycles for one repair.")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(request, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    request["_request_path"] = str(path)
    request["_outbox"] = str(outbox)
    return request


def _request_fingerprint(req: Mapping[str, Any]) -> Tuple[Any, ...]:
    """What makes two requests THE SAME WORK.

    Deliberately NOT the id or the timestamp -- those differ by construction and
    are how a duplicate hides. Identity is the payload: which files with which
    content, which services, the action, and the verifier. ``purpose`` is
    excluded too: re-wording the reason does not change the work, and a model
    that retries usually rewords.
    """
    files = []
    for entry in req.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        files.append((entry.get("target"), entry.get("sha256")))
    action = req.get("action")
    if isinstance(action, Mapping):
        action = (action.get("kind"), action.get("instance_id"),
                  json.dumps(action.get("set"), sort_keys=True, default=repr))
    return (
        tuple(sorted(files)),
        tuple(sorted(str(s) for s in (req.get("services") or []))),
        action,
        req.get("verify_command"),
    )


def _pending_duplicate(outbox: Path,
                       request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """An undecided request in ``outbox`` that asks for exactly this work."""
    want = _request_fingerprint(request)
    try:
        candidates = sorted(outbox.glob(f"*{REQUEST_SUFFIX}"))
    except OSError:
        return None
    for path in candidates:
        try:
            other = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A half-written or unreadable request is not evidence of a
            # duplicate; the applier will deal with it.
            continue
        if isinstance(other, Mapping) and _request_fingerprint(other) == want:
            return dict(other, id=str(other.get("id") or path.stem))
    return None


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
    if not dry_run and not is_gated() and not root_is_writable():
        # direct mode applies IN PLACE, so a read-only root makes it impossible.
        # The default sidecar mount is deliberately read-only: gated mode's
        # refusal lives in this process, but the agent also has a shell, and a
        # redirect would step over an in-process check. Failing here with the
        # reason beats an opaque OSError from three layers down.
        return {
            "status": "unapplied",
            "reason": (
                "AGENT_DEPLOY_MODE=direct needs to write inside the deployment "
                f"root, but {deploy_root()} is read-only (the sidecar mounts it "
                ":ro by default, which is what makes the no-direct-write "
                "guarantee a kernel one rather than only an in-process one). "
                "Either stage the request and let the HOST apply it -- "
                "`python3 ops/apply_deploys.py --once`, which is what gated "
                "mode is for -- or recreate the container with "
                "AGENT_ROOT_MOUNT_MODE=rw, accepting that a shell inside it can "
                "then rewrite the deployment bypassing the gate."),
            "request_path": str(path), "mode": config.DEPLOY_MODE,
            "dry_run": dry_run, "receipt_path": str(receipt_path),
            "applier_output": ""}
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
        "services": receipt.get("services") or [],
        "restart": receipt.get("restart") or [],
        "verify_rc": receipt.get("verify_rc"),
        "verify_advisory": receipt.get("verify_advisory"),
        "verify_warning": receipt.get("verify_warning"),
        #: Readiness verdict, accepting the key an OLDER applier filed it under:
        #: a receipt is a persisted audit trail, so one written before a field
        #: rename still has to be reportable. ``deploy_applier.READINESS_KEYS``
        #: owns the same list; this module cannot import it (the applier is the
        #: host's side of the seam, invoked as a program, not imported), so the
        #: agreement is pinned by test instead of assumed -- see
        #: test_deploy_tools.py.
        "readiness": _readiness_of(receipt),
        "backups": receipt.get("backups") or [],
        #: DID THIS REQUEST OVERWRITE SOMETHING NEWER? Copied through from the
        #: receipt, because the applier records it and this is the only path by
        #: which the agent can learn about it. Two gaps existed on this line:
        #: the summariser never printed it, and this dict never carried it, so
        #: the check was invisible end to end -- written, persisted, unread.
        "staged_against": receipt.get("staged_against") or [],
        "staged_against_warning": receipt.get("staged_against_warning"),
        #: Files whose BASE was never recorded, so a change could not be
        #: detected either way. Carried separately from ``staged_against``
        #: because that key means "this DID change" -- and the two must not be
        #: conflated in an audit trail. Live 2026-11-02 a request staged with
        #: ``staged_against_sha256: "unreadable"`` produced a receipt claiming
        #: the deployment had changed under it, which sent a pass hunting a
        #: revert that never happened.
        "staged_against_unknown": receipt.get("staged_against_unknown") or [],
        "staged_against_unknown_warning": receipt.get(
            "staged_against_unknown_warning"),
        #: Did the applied change reach the repository? Reported so the agent
        #: can tell the operator WHERE the code now lives. A deploy that landed
        #: but never got pushed is still a real deploy -- it just means the
        #: repository is behind the host, which is exactly the drift that used
        #: to go unnoticed. Absent on receipts written before this field
        #: existed, so an old receipt reads as None rather than "not pushed".
        "git": receipt.get("git"),
    }


#: Field names a receipt may hold the readiness verdict under, newest first.
#: Mirrors ``deploy_applier.READINESS_KEYS`` for the reason given at the call
#: site; the two are compared by test, since nothing here can import the
#: applier.
READINESS_KEYS: Tuple[str, ...] = ("readiness", "broker")


def _readiness_of(doc: Any) -> Optional[Dict[str, Any]]:
    """The verdict from a receipt, whichever schema wrote it, or None."""
    if not isinstance(doc, dict):
        return None
    for key in READINESS_KEYS:
        found = doc.get(key)
        if isinstance(found, dict):
            return found
    return None


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
    #: Accept EITHER schema, not just the one the outcome builder emits. This
    #: function is the agent's only window into what the host decided, and a
    #: receipt written before the readiness field was renamed is still a
    #: receipt: reading only the new key printed nothing for those, which the
    #: agent reads as "the readiness gate never ran" on a host where it did.
    readiness = _readiness_of(outcome) or {}
    if readiness:
        if readiness.get("skipped"):
            lines.append(
                f"readiness gate: SKIPPED ({readiness.get('reason')})")
        else:
            lines.append(f"readiness gate: {readiness.get('gate')} state="
                         f"{readiness.get('state')!r} "
                         f"age={readiness.get('age_sec')}s")
    for row in outcome.get("files") or []:
        if row.get("operation") == "delete":
            lines.append(f"  target {row.get('target')} (DELETE)")
        else:
            lines.append(f"  target {row.get('target')} "
                         f"({row.get('bytes')} bytes, sha256 {row.get('sha256')}...)")
    for row in outcome.get("restart") or []:
        lines.append(f"  service {row.get('service')}: {row.get('cmd')} -> "
                     f"rc={row.get('rc')}" + (f" ({row['error']})"
                                              if row.get("error") else ""))
    if outcome.get("verify_rc") is not None:
        lines.append(f"verification rc={outcome['verify_rc']}"
                     + (" (ADVISORY; files kept)" if outcome.get("verify_advisory")
                        else ""))
    #: DID THIS REQUEST OVERWRITE SOMETHING NEWER?
    #:
    #: The applier records this and nothing else read it, which made the whole
    #: check one-directional: the fact existed in the receipt and the only
    #: consumer that could act on it -- the agent, hours later, deciding whether
    #: its staged change is still the right one -- could not see it. A field
    #: written for a reader who never reads it is the same as not writing it.
    #:
    #: Printed as a warning rather than a fact, because the applier applies the
    #: requested bytes either way (the owner's 2026-11-02 directive) and the
    #: only correct response is for the agent to LOOK, not to assume.
    superseded = outcome.get("staged_against") or []
    if superseded:
        lines.append(
            f"WARNING: {len(superseded)} file(s) changed AFTER this request was "
            f"staged, and the staged content overwrote them:")
        for row in superseded:
            lines.append(f"  {row.get('target')}: staged against "
                         f"{row.get('staged_against')}, found "
                         f"{row.get('found_instead')}")
        lines.append("  Check whether this undid a newer fix before reporting "
                     "the change as closed.")
    #: The other half of the same question, and NOT the same answer. A file
    #: whose base was never recorded cannot be compared, so it is reported as
    #: unknown rather than as a change. Printed because an unprinted field is
    #: an invisible one, and because the failure mode here is an agent hunting
    #: for a revert the receipt never evidenced.
    unknown = outcome.get("staged_against_unknown") or []
    if unknown:
        lines.append(
            f"NOTE: {len(unknown)} file(s) had NO recorded base, so a change "
            f"since staging was NOT checked (this is not a report that they "
            f"changed):")
        for row in unknown:
            lines.append(f"  {row.get('target')}: now {row.get('found_instead')}")
        lines.append("  Do not hunt for a revert on this line alone -- stage "
                     "an explicit verify_command if the content matters.")
    backups = outcome.get("backups") or []
    if backups:
        lines.append(f"backups: {len(backups)} file(s) copied aside")
        for row in backups:
            if row.get("backup"):
                lines.append(f"  {row.get('target')} -> {row.get('backup')}")
            else:
                lines.append(f"  {row.get('target')} (no pre-existing file)")
    #: Whether the applied change reached the repository. Omitted entirely when
    #: the receipt predates the field (or git is not configured), because
    #: "git: (none)" would read as a failed push rather than "not applicable".
    git_info = outcome.get("git")
    if isinstance(git_info, dict) and git_info.get("state"):
        state = git_info["state"]
        detail = git_info.get("sha") or git_info.get("reason") or ""
        lines.append(f"git: {state}" + (f" {detail}" if detail else ""))
        if git_info.get("retry"):
            lines.append(f"  retry: {git_info['retry']}")
    if outcome.get("request_path"):
        lines.append(f"request: {outcome['request_path']}")
    if outcome.get("receipt_path"):
        lines.append(f"receipt: {outcome['receipt_path']}")
    if outcome.get("applier_output"):
        tail = str(outcome["applier_output"])[-900:]
        lines.append("applier output (tail):")
        lines.extend("  " + ln for ln in tail.splitlines())
    return "\n".join(lines)


def _retire_rehearsal(request_path: Optional[str]) -> Optional[str]:
    """Move a DRY-RUN request out of the pending set. Returns where it went.

    A rehearsal must not become the change. ``apply_request(dry_run=True)``
    evaluates every gate and leaves the request IN THE OUTBOX -- which is exactly
    where the host applier looks, so on the next timer pass (three minutes here)
    it would apply the very change the agent only meant to test. The receipt
    would then read "applied" against a request the agent believed was inert.

    That is the worst shape of failure this design has: the operator is told a
    change was rehearsed, and it lands anyway. So the request is moved to
    ``outbox/rehearsed/``, which no gate scans, keeping it for the audit trail.

    Failures are swallowed: the rehearsal already produced its verdict, and
    reporting a tidy-up error would obscure that.
    """
    if not request_path:
        return None
    try:
        src = Path(str(request_path))
        if not src.is_file():
            return None
        dest_dir = Path(outbox_dir()) / "rehearsed"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        os.replace(src, dest)
        return str(dest)
    except OSError:
        return None


def deploy_request(purpose: str,
                   files: Optional[List[Dict[str, Any]]] = None,
                   services: Optional[List[str]] = None,
                   verify_command: Optional[str] = None,
                   dry_run: bool = False,
                   action: Optional[Dict[str, Any]] = None) -> str:
    """Propose a change to the deployment; never apply it directly.

    Two kinds of change, one tool:

    * FILES -- ``{"target": "engine/x.py", "content": "..."}`` or
      ``{"target": ..., "source": "path/to/existing/file"}``. To remove an
      obsolete file, use ``{"target": "engine/x.py", "delete": True}``: the
      host enforces an unchanged reviewed base, backs it up first, and mirrors
      the removal into git. This is the supported deletion path; never replace
      it with a direct ``rm`` or ``git rm``;
    * ACTION -- ``{"kind": ..., "instance_id": ..., "set": {...}}``: ask the
      host to run one of its OWN commands to change runtime state. Use this
      when the value lives in a file the running service owns and caches (an
      instance's selected risk profile, for example). Writing such a file as a
      deploy TARGET is refused by policy, and would be overwritten anyway.

    In the default ``gated`` mode this STAGES the request in the outbox and
    returns immediately: a host-side applier decides later whether to apply it
    and reports its verdict in a receipt. In ``direct`` mode the same applier
    is invoked here, with identical gates.

    A REBUILD-ONLY request is ``files=[]`` plus ``services=[...]``: no file
    changes, the host just runs ``docker compose build`` and recreates those
    services. Use it when the deployment tree is already correct and the
    RUNNING image is stale -- the one repair a read-only sidecar can diagnose
    but could not previously express. See ``compose_deploy_request``.

    ``dry_run`` rehearses the whole thing -- every gate evaluated, a receipt
    written, nothing changed. Use it to prove a change before anything live is
    at risk; for an action it also reveals whether the deployment accepts the
    value.
    """
    try:
        request = compose_deploy_request(
            purpose=purpose, files=files, services=services,
            verify_command=verify_command, action=action)
    except DeployError as exc:
        return f"DEPLOY REQUEST REFUSED while composing: {exc}"
    staged = f"STAGED {request['_request_path']}"
    if request.get("action"):
        act = request["action"]
        detail = ", ".join(f"{k}={v!r}" for k, v in act["set"].items())
        header = (f"{staged}\n  purpose: {request['purpose']}\n"
                  f"  action: {act['kind']} on {act['instance_id']}\n"
                  f"  set: {detail}\n  mode: {config.DEPLOY_MODE}")
        tail = ("\n\nThe host applier re-checks that the action kind is "
                "permitted, the field is one the policy exposes, the instance "
                "already exists and the deployment accepts the value -- then "
                "verifies by re-reading. An action is NOT a file write: do not "
                "also try to edit the file it comes from.")
    else:
        targets = ", ".join(str(f.get("target")) for f in request["files"])
        if targets:
            header = (f"{staged}\n  purpose: {request['purpose']}\n"
                      f"  targets: {targets}\n  mode: {config.DEPLOY_MODE}")
        else:
            #: REBUILD-ONLY. Say so explicitly: the receipt for this request
            #: has an empty ``files`` list, and a reader who is not told why
            #: reads that as a broken request rather than a deliberate one.
            header = (f"{staged}\n  purpose: {request['purpose']}\n"
                      f"  targets: (none -- rebuild only)\n"
                      f"  mode: {config.DEPLOY_MODE}")
        tail = ""
    if services:
        header += f"\n  services to restart: {', '.join(services)}"
    if is_gated() and not dry_run:
        return (header + "\n\nGated mode: the request waits in the outbox for "
                "the HOST applier, which re-checks containment, checksums, "
                "request age and the readiness gate, backs every target up "
                "and can roll back. Read its verdict with deploy_status.")
    #: A rehearsal must not outlive itself, whether or not the applier ran.
    #: This deployment's sidecar cannot reach the host applier at all, so the
    #: common case is that apply_request REFUSES -- and returning early on that
    #: used to leave the staged request sitting in the outbox, where the host
    #: timer would apply the very change the agent only meant to test. The
    #: retirement is therefore in a finally, not on the success path.
    outcome: Dict[str, Any] = {}
    error = ""
    try:
        outcome = apply_request(request, dry_run=dry_run)
    except DeployError as exc:
        error = str(exc)
    finally:
        moved = _retire_rehearsal(
            outcome.get("request_path") or request.get("_request_path")) \
            if dry_run else None

    if error:
        return header + f"\n\nAPPLIER ERROR: {error}" + (
            "\n\nREHEARSAL ONLY: the request was moved out of the pending set"
            + (f" to {moved}" if moved else "")
            + ", so the host will NOT apply it." if dry_run else "")
    note = ""
    if dry_run:
        note = ("\n\nREHEARSAL ONLY: the request was moved out of the pending "
                "set"
                + (f" to {moved}" if moved else "")
                + ", so the host will NOT apply it. Re-issue it without "
                  "dry_run to actually make the change.")
    return header + tail + "\n\n" + _summarise_outcome(outcome) + note


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
