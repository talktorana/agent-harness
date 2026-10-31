#!/usr/bin/env python3
"""Host-side applier for agent-staged deploy requests: the GATE.

The agent harness runs in a sidecar with the project mounted READ-ONLY, no
docker socket and no ssh key, so it cannot touch production even if it wants
to. Instead it writes a *request* into an outbox directory, and THIS script --
running on the host, as the user that owns the deployment -- decides whether
the request may be applied. Splitting proposal from execution is the whole
point: an autonomous agent that can both edit code and restart containers is
an unreviewed change applied to a live production system.

Gates, checked in this order, all of which must pass:

1. SCHEMA     the request must be a well-formed, versioned document.
2. PATHS      every target must resolve INSIDE the deploy root; no symlink
              target outside it, no ``..`` traversal, nothing git-managed
              metadata, nothing on the refuse list.
3. CHECKSUMS  the staged payload must hash to exactly what the request says,
              so a truncated or tampered file cannot be applied silently.
4. AGE        a request older than the window is refused: applying a stale
              intent hours later is not what anyone asked for.
5. READINESS  the deployment's own status file (named by the policy) must be
              recent AND in one of the policy's safe states. Applying a change
              while the system is mid-operation -- or while its state is merely
              unknown because the status file went stale -- is how a restart
              lands on something nobody is watching.
6. LOCK       one applier at a time. Concurrent applies would race backups
              and could leave two different files staged.

The applier holds NO domain knowledge. What counts as a safe state, which file
reports it and which filenames are not code all come from a policy file
(``--policy``), so the same gate can front a trading system, a web app or a
batch pipeline. Secrets and ``.git`` are refused unconditionally: a policy can
add to the refuse list, never subtract from it.

Then: back up every target that exists, run the request's verification
command and ABORT+RESTORE on failure, apply atomically via tmp+rename
preserving mode, and optionally rebuild/restart the named compose services --
restart only with ``--allow-restart``, and never if any gate failed.

Every outcome -- applied, refused, failed -- writes a receipt next to the
request, so the requester (and the morning review) can read what the host
decided and why. ``--dry-run`` runs the full gate sequence and writes a receipt
marked ``dry_run`` while changing nothing on disk except that receipt.

    python3 deploy_applier.py --once --root /srv/app --policy policy.json
    python3 deploy_applier.py --once --root /tmp/sandbox --dry-run
    python3 deploy_applier.py --watch --root /srv/app --interval 30

Exit codes: 0 all requests handled, 2 refused/failed (nothing applied),
3 a misconfiguration prevented evaluation at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
REQUEST_SUFFIX = ".deploy.json"
RECEIPT_SUFFIX = ".receipt.json"

#: Refused in EVERY deployment, whatever the policy says. Secrets and VCS
#: metadata are never a legitimate deploy payload, so they are not left to the
#: policy author to remember. ``--policy`` can only ADD to this list.
GENERIC_FORBIDDEN_NAMES = frozenset({".env"})
GENERIC_FORBIDDEN_PREFIXES = (".env",)
GENERIC_FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
GENERIC_FORBIDDEN_DIRS = (".git",)

#: A request this old is stale intent, not a pending task.
DEFAULT_MAX_AGE_SEC = 900.0

#: How fresh the readiness status file must be to believe it. A stale status
#: means we do not actually know the state the system is in, which is a reason
#: to refuse -- not a reason to assume it is safe.
DEFAULT_MAX_STATUS_AGE_SEC = 900.0


class Policy:
    """Every DOMAIN-SPECIFIC decision this applier makes, in one place.

    The applier itself knows nothing about brokers, trading sessions or any
    particular product. What it knows is: a deploy may only proceed when the
    system declares itself in one of a set of safe states, in a status file
    that is fresh, and it may never write a file the deployment says is not
    code. Those three facts are supplied by the policy, so the same gate can
    front a trading system, a web app or a batch pipeline.

    With no ``readiness.status_file`` configured the readiness gate reports
    itself skipped rather than silently passing -- a receipt must say which
    gates were actually evaluated.
    """

    def __init__(self, doc: Optional[Dict[str, Any]] = None,
                 name: str = "default") -> None:
        doc = doc if isinstance(doc, dict) else {}
        self.name = name
        readiness = doc.get("readiness")
        readiness = readiness if isinstance(readiness, dict) else {}
        self.status_file = str(readiness.get("status_file") or "").strip()
        self.status_field = str(readiness.get("field") or "state")
        self.status_label = str(readiness.get("label") or "readiness")
        states = readiness.get("safe_states") or []
        self.safe_states: Tuple[str, ...] = tuple(
            str(s) for s in states if isinstance(s, str) and s.strip())
        self.max_status_age_sec = float(
            readiness.get("max_age_sec") or DEFAULT_MAX_STATUS_AGE_SEC)
        #: The status document's OWN timestamp field, e.g. ``checked_at``.
        #:
        #: Optional. When declared, freshness is judged by BOTH the file's
        #: mtime and this stamp and the STRICTER one decides. mtime alone is not
        #: proof: it is reset by a copy, a restore, an rsync or a plain
        #: ``touch``, all routine on a deploy host, so a freshly-touched file
        #: carrying a verdict from an hour ago passes a gate whose entire job is
        #: to demand recent evidence. Found on a live rehearsal 2026-10-31.
        self.timestamp_field = str(
            readiness.get("timestamp_field") or "").strip()

        limits = doc.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        self.max_age_sec = float(limits.get("max_age_sec")
                                 or DEFAULT_MAX_AGE_SEC)

        forbid = doc.get("forbid")
        forbid = forbid if isinstance(forbid, dict) else {}
        extra_names = {str(n) for n in (forbid.get("names") or [])
                       if isinstance(n, str) and n.strip()}
        self.forbidden_names = frozenset(GENERIC_FORBIDDEN_NAMES | extra_names)
        self.forbidden_prefixes = tuple(GENERIC_FORBIDDEN_PREFIXES) + tuple(
            str(p) for p in (forbid.get("prefixes") or [])
            if isinstance(p, str) and p)
        self.forbidden_suffixes = tuple(GENERIC_FORBIDDEN_SUFFIXES) + tuple(
            str(s) for s in (forbid.get("suffixes") or [])
            if isinstance(s, str) and s)
        extra_dirs = [str(d) for d in (forbid.get("dirs") or [])
                      if isinstance(d, str) and d.strip()]
        self.forbidden_dirs = tuple(dict.fromkeys(
            list(GENERIC_FORBIDDEN_DIRS) + extra_dirs))
        # A status file is an input, never a deploy target.
        if self.status_file:
            self.forbidden_names = frozenset(
                self.forbidden_names | {Path(self.status_file).name})

    @property
    def readiness_enabled(self) -> bool:
        return bool(self.status_file)

    def is_forbidden_name(self, name: str) -> bool:
        if name in self.forbidden_names:
            return True
        lowered = name.lower()
        if any(lowered.startswith(p) for p in self.forbidden_prefixes):
            return True
        return any(lowered.endswith(s) for s in self.forbidden_suffixes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.name,
            "readiness": {
                "enabled": self.readiness_enabled,
                "status_file": self.status_file,
                "field": self.status_field,
                "label": self.status_label,
                "safe_states": list(self.safe_states),
                "max_status_age_sec": self.max_status_age_sec,
                "timestamp_field": self.timestamp_field or None,
            },
            "max_age_sec": self.max_age_sec,
            "forbidden_names": sorted(self.forbidden_names),
            "forbidden_dirs": list(self.forbidden_dirs),
        }

    @classmethod
    def load(cls, path: Optional[str]) -> "Policy":
        """Load a policy file; an absent path means the generic default.

        A malformed policy is a hard error, never a silent downgrade to
        permissive defaults: failing open is the one failure mode a gate must
        not have.
        """
        if not path:
            return cls({}, name="default")
        target = Path(path).expanduser()
        if not target.is_file():
            raise DeployError(f"policy file does not exist: {target}")
        data, err = _read_json(target)
        if data is None:
            raise DeployError(f"policy file unreadable: {err}")
        return cls(data, name=target.name)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is not readable JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path.name} must contain a JSON object"
    return data, None


class DeployError(Exception):
    """A gate refused the request, or applying it failed."""


class Refused(DeployError):
    """A gate refused the request BEFORE anything was written."""


# =================================================================== gates

def gate_schema(req: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Validate shape. Returns the (target, payload) pairs to apply."""
    if int(req.get("schema_version") or 0) != SCHEMA_VERSION:
        raise Refused(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{req.get('schema_version')!r}")
    files = req.get("files")
    if not isinstance(files, list) or not files:
        raise Refused("'files' must be a non-empty list")
    pairs: List[Tuple[str, Any]] = []
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise Refused(f"files[{i}] must be an object")
        target = entry.get("target")
        if not isinstance(target, str) or not target.strip():
            raise Refused(f"files[{i}].target must be a non-empty string")
        if "content" in entry:
            content = entry["content"]
            if not isinstance(content, str):
                raise Refused(f"files[{i}].content must be a string")
            pairs.append((target, content.encode("utf-8")))
        elif entry.get("payload_path"):
            payload = Path(str(entry["payload_path"]))
            if not payload.is_file():
                raise Refused(
                    f"files[{i}] payload_path does not exist: {payload}")
            pairs.append((target, payload.read_bytes()))
        else:
            raise Refused(
                f"files[{i}] needs either 'content' or 'payload_path'")
    # A request that says nothing about why is not reviewable later.
    purpose = req.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        raise Refused("'purpose' is required: receipts must say why")
    return pairs


def _is_absolute_target(target: str) -> bool:
    """True for any target the requester means to be absolute.

    Checked explicitly rather than left to ``Path`` semantics, which differ by
    platform: ``Path(root) / "/etc/passwd"`` DISCARDS root on POSIX, while
    ``Path(root) / "/tmp/x"`` on Windows joins under the CURRENT DRIVE and so
    lands back INSIDE root. Relying on containment alone would therefore allow
    on Windows precisely the shape that is refused on Linux -- and a deploy
    target is always meant to be relative to the deployment root anyway.
    """
    raw = target.strip()
    if raw.startswith("/") or raw.startswith("\\\\"):
        return True
    drive, _tail = os.path.splitdrive(raw)
    return bool(drive)


def gate_paths(pairs: List[Tuple[str, Any]], root: Path,
               policy: Optional[Policy] = None) -> List[Path]:
    """Resolve every target inside root; refuse absolute, traversal, refused.

    Containment is enforced by path comparison rather than string prefix,
    since ``/opt/x-evil`` starts with ``/opt/x`` but is not inside it.
    """
    policy = policy or Policy()
    try:
        base = root.resolve()
    except OSError as exc:
        raise Refused(f"deploy root is not resolvable: {exc}") from exc
    resolved: List[Path] = []
    for target, _payload in pairs:
        if _is_absolute_target(target):
            raise Refused(
                f"target must be relative to the deploy root, not an absolute "
                f"path: {target!r}")
        raw = target.strip().replace("\\", "/")
        unresolved = base / raw.lstrip("/")
        # Symlink FIRST, on the path as written. Containment resolves the link,
        # so a target pointing outside the root would otherwise be reported as
        # an escape -- a true statement that hides the actual mechanism, and a
        # receipt is what an operator reads to decide whether to trust the
        # agent again. Refusing a link is also refusing to write through a name
        # that does not say where the bytes land.
        if unresolved.is_symlink():
            raise Refused(f"target is a symlink, refusing to write through it: "
                          f"{target!r}")
        candidate = unresolved.resolve()
        if base != candidate and base not in candidate.parents:
            raise Refused(
                f"target escapes the deploy root: {target!r} -> {candidate}")
        parts = candidate.parts
        if any(p in policy.forbidden_dirs for p in parts):
            raise Refused(f"target is inside a forbidden directory: {target!r}")
        if policy.is_forbidden_name(candidate.name):
            raise Refused(
                f"target is on the refuse list and may never be written by an "
                f"agent: {candidate.name}")
        resolved.append(candidate)
    duplicates = {str(p) for p in resolved if resolved.count(p) > 1}
    if duplicates:
        raise Refused(f"request targets the same file twice: {sorted(duplicates)}")
    return resolved


def gate_checksums(req: Dict[str, Any], pairs: List[Tuple[str, Any]]) -> None:
    """Declared sha256 must match the payload byte for byte."""
    declared = req.get("files") or []
    for entry, (_target, payload) in zip(declared, pairs):
        want = entry.get("sha256")
        if not want:
            raise Refused(
                f"files[{entry.get('target')!r}] declares no sha256; refusing "
                f"an unverified payload")
        got = sha256_bytes(payload)
        if str(want).lower() != got.lower():
            raise Refused(
                f"checksum mismatch for {entry.get('target')!r}: declared "
                f"{str(want)[:16]}..., staged {got[:16]}...")


def gate_age(req: Dict[str, Any], path: Path, max_age: float) -> None:
    """Refuse stale intent: hours-old requests are not pending work."""
    stamp = req.get("created_at")
    created: Optional[float] = None
    if isinstance(stamp, str):
        try:
            created = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            created = None
    if created is None:
        created = path.stat().st_mtime if path.exists() else time.time()
    age = time.time() - created
    if age > max_age:
        raise Refused(
            f"request is {age / 60:.1f} min old, older than the "
            f"{max_age / 60:.0f} min window; re-stage it rather than apply "
            f"a stale intent")


def gate_broker(root: Path, policy: Policy,
                status_path: Optional[Path] = None) -> Dict[str, Any]:
    """The readiness gate: only deploy from a KNOWN, safe state.

    What "safe" means belongs to the deployment, so the state list, the status
    file and its field all come from the policy. The gate itself is generic and
    has two refusal modes that both matter:

    * a state outside the safe set (the system says it is busy/broken);
    * a STALE status file. "We have not heard in 20 minutes" is not evidence
      that the state is safe, and treating silence as agreement is exactly how
      a restart lands on a system mid-operation.

    A policy with no status file records the gate as skipped, so the receipt
    shows which gates actually ran instead of implying coverage it lacks.
    """
    if not policy.readiness_enabled and status_path is None:
        return {"gate": policy.status_label, "skipped": True,
                "reason": "no readiness status_file configured in the policy"}
    label = policy.status_label
    if not policy.safe_states:
        # With no declared safe states there is nothing to compare the file
        # against. Refusing everything would strand every request, and passing
        # would be a gate that only looks like one -- so the receipt says the
        # gate did not run, which is the honest third option.
        return {"gate": label, "skipped": True,
                "reason": "policy declares no safe_states, so the readiness "
                          "gate cannot evaluate"}
    path = status_path or _resolve_status(root, policy.status_file)
    if not path.is_file():
        raise Refused(
            f"no {label} status file at {path}; cannot prove the system is "
            f"safe to disturb")
    data, err = _read_json(path)
    if data is None:
        raise Refused(f"{label} status unreadable: {err}")
    state = str(data.get(policy.status_field) or "")
    age = time.time() - path.stat().st_mtime
    if age > policy.max_status_age_sec:
        raise Refused(
            f"{label} status is {age / 60:.1f} min old (limit "
            f"{policy.max_status_age_sec / 60:.0f} min): the state is unknown, "
            f"so a deploy is refused")
    #: The verdict's OWN timestamp, when the policy names one. File mtime
    #: alone can be reset by a copy, a restore or a touch, leaving a stale
    #: verdict looking current -- which is exactly the evidence this gate
    #: exists to demand. Both must be inside the limit; a stamp that cannot be
    #: read is a refusal rather than a pass, and one from the future means the
    #: writer's clock is wrong, so nothing about the state can be trusted.
    verdict_age = _verdict_age(data, policy.timestamp_field, label,
                               policy.max_status_age_sec)
    if state not in policy.safe_states:
        raise Refused(
            f"{label} state is {state!r}; only {list(policy.safe_states)} may "
            f"be disturbed by a deploy")
    return {"gate": label, "state": state, "age_sec": round(age, 1),
            "verdict_age_sec": (round(verdict_age, 1)
                                if verdict_age is not None else None),
            "field": policy.status_field,
            "stamp_field": policy.timestamp_field or None,
            "checked_at": data.get(policy.timestamp_field)
            if policy.timestamp_field else data.get("checked_at")}


#: A status stamp this far ahead of the host clock means the writer's clock is
#: wrong, so the age cannot be trusted in either direction.
MAX_CLOCK_SKEW_SEC = 300.0


def _parse_stamp(raw: Any) -> Optional[float]:
    """Epoch seconds from an ISO-8601 string or a number, or None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _verdict_age(data: Dict[str, Any], stamp_field: str, label: str,
                 limit_sec: float) -> Optional[float]:
    """Age in seconds of the verdict's own timestamp, or None if undeclared.

    Refuses rather than passing when a declared field is missing, blank or
    unparsable: a status document that cannot date its own verdict is not
    evidence of anything, and the receipt would otherwise claim a freshness
    check that never happened.
    """
    if not stamp_field:
        return None
    stamp = _parse_stamp(data.get(stamp_field))
    if stamp is None:
        raise Refused(
            f"{label} status carries no parsable {stamp_field!r} "
            f"(got {data.get(stamp_field)!r}); the verdict cannot be dated, "
            f"so the state is unknown and a deploy is refused")
    verdict_age = time.time() - stamp
    if verdict_age < -MAX_CLOCK_SKEW_SEC:
        raise Refused(
            f"{label} status is stamped {-verdict_age / 60:.1f} min in the "
            f"FUTURE ({data.get(stamp_field)!r}): the writer's clock disagrees "
            f"with this host, so its age proves nothing")
    if verdict_age > limit_sec:
        raise Refused(
            f"{label} verdict is {verdict_age / 60:.1f} min old (limit "
            f"{limit_sec / 60:.0f} min) even though the status FILE was "
            f"written recently -- a copied, restored or touched file does not "
            f"make a stale verdict current, so a deploy is refused")
    return verdict_age


def _resolve_status(root: Path, status_file: str) -> Path:
    """Resolve the policy's status file, relative to the deploy root."""
    candidate = Path(status_file).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / status_file


class ApplierLock:
    """Single-flight lock so two appliers cannot race backups and writes."""

    def __init__(self, outbox: Path, stale_sec: float = 3600.0) -> None:
        self.path = outbox / ".applier.lock"
        self.stale = stale_sec
        self._held = False

    def __enter__(self) -> "ApplierLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < self.stale:
                raise Refused(
                    f"another applier holds the lock ({age:.0f}s); refusing to "
                    f"race it")
            # A crashed applier left it behind; take it over.
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Refused("lock was taken by another applier first") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "at": _now()}))
        self._held = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False


# ================================================================== apply

def backup_targets(targets: List[Path], backup_dir: Path) -> List[Dict[str, str]]:
    """Copy every existing target aside BEFORE writing anything."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    made: List[Dict[str, str]] = []
    for target in targets:
        if not target.exists():
            made.append({"target": str(target), "backup": "", "created": "true"})
            continue
        flat = str(target).replace(":", "_").replace("/", "_").replace("\\", "_")
        dest = backup_dir / flat
        shutil.copy2(target, dest)
        made.append({"target": str(target), "backup": str(dest),
                     "created": "false"})
    return made


def restore_backup(made: List[Dict[str, str]]) -> None:
    """Put the originals back after a failed apply."""
    for row in made:
        target = Path(row["target"])
        if row.get("created") == "true":
            target.unlink(missing_ok=True)
        elif row.get("backup"):
            shutil.copy2(Path(row["backup"]), target)


def write_payloads(targets: List[Path], pairs: List[Tuple[str, Any]]) -> None:
    """Atomic tmp+rename so a crash cannot leave a half-written module.

    Mode AND owner are preserved. Without the owner half, a root-run applier
    -- root being the default for a systemd unit, which is what this kind of
    host uses, since it has no cron at all -- flips deployed code from its
    normal owner to root:root. The mixed ownership that leaves behind is the
    same failure mode that made an outbox readable-but-unwritable to its next
    non-root caller. New files inherit their parent directory's owner, so a
    deploy never introduces a lone root-owned file into an otherwise
    user-owned tree.

    Best-effort by design: the write has already succeeded, so a chown the
    platform refuses (Windows, or an unprivileged uid) must not be reported as
    a failed deploy.
    """
    for target, (_name, payload) in zip(targets, pairs):
        target.parent.mkdir(parents=True, exist_ok=True)
        stat_before = target.stat() if target.exists() else None
        tmp = target.with_name(target.name + ".deploy-tmp")
        tmp.write_bytes(payload)

        owner: Optional[Tuple[int, int]] = None
        if stat_before is not None:
            owner = (stat_before.st_uid, stat_before.st_gid)
            try:
                os.chmod(tmp, stat_before.st_mode & 0o7777)
            except OSError:
                pass
        else:
            try:
                parent = target.parent.stat()
                owner = (parent.st_uid, parent.st_gid)
            except OSError:
                pass
        if owner is not None and hasattr(os, "chown"):
            try:
                os.chown(tmp, *owner)
            except OSError:
                pass
        os.replace(tmp, target)


#: Prefix for every verification command. Without it a verifier that never ran
#: reports SUCCESS: `run_tests | tail` exits with tail's status, so a missing
#: interpreter upstream is invisible. Proved in the sandbox dry-run 2026-10-30,
#: where the receipt read verify_rc=0 next to "No such file or directory" --
#: the one outcome that must never pass a gate is a green light from a verifier
#: that was never executed.
VERIFY_SHELL_PREFIX = "set -o pipefail; "


def run_command(cmd: List[str], cwd: Path,
                timeout: float) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-2500:]
    return proc.returncode, tail


def restart_services(services: List[str], root: Path,
                     timeout: float) -> List[Dict[str, Any]]:
    """Rebuild+recreate the named compose services. Only with --allow-restart."""
    out: List[Dict[str, Any]] = []
    for svc in services:
        for args in (["build", svc], ["up", "-d", "--no-deps", svc]):
            cmd = ["docker", "compose"] + args
            try:
                rc, tail = run_command(cmd, root, timeout)
            except subprocess.TimeoutExpired:
                out.append({"service": svc, "cmd": " ".join(args),
                            "rc": None, "error": "timeout"})
                return out
            out.append({"service": svc, "cmd": " ".join(args), "rc": rc,
                        "output": tail[-600:]})
            if rc != 0:
                return out
    return out


# ============================================================== receipts

def _receipt_name(request_path: Path) -> str:
    """The receipt filename for one request: ``X.deploy.json`` -> ``X.receipt.json``.

    Derived by STRIPPING the request suffix rather than by ``Path.stem``, which
    only removes the final ``.json`` and so yields ``X.deploy.receipt.json``.
    The agent-side tools compute the same name to read the verdict back, and
    the two silently disagreeing turned every real outcome into a generic
    "ok"/"failed" -- the applier had refused a deploy while its caller reported
    success. One rule, spelled out here.
    """
    name = request_path.name
    if name.endswith(REQUEST_SUFFIX):
        return name[:-len(REQUEST_SUFFIX)] + RECEIPT_SUFFIX
    return request_path.stem + RECEIPT_SUFFIX


def write_receipt(outbox: Path, request_path: Path, status: str,
                  reason: str, extra: Optional[Dict[str, Any]] = None,
                  dry_run: bool = False) -> Path:
    payload: Dict[str, Any] = {
        "request": request_path.name,
        "status": status,
        "reason": reason,
        "at": _now(),
        "dry_run": bool(dry_run),
    }
    if extra:
        payload.update(extra)
    dest = outbox / _receipt_name(request_path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def handle_request(path: Path, outbox: Path, root: Path, *, dry_run: bool,
                   policy: Optional[Policy] = None,
                   allow_restart: bool = False,
                   test_timeout: float = 900.0,
                   status_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run every gate, then apply (or not). Always returns a receipt dict."""
    policy = policy or Policy()
    req, err = _read_json(path)
    if req is None:
        receipt = write_receipt(outbox, path, "refused", err or "unreadable",
                                dry_run=dry_run)
        return json.loads(receipt.read_text(encoding="utf-8"))

    rid = str(req.get("id") or path.stem)
    try:
        pairs = gate_schema(req)
        targets = gate_paths(pairs, root, policy)
        gate_checksums(req, pairs)
        gate_age(req, path, policy.max_age_sec)
        broker = gate_broker(root, policy, status_path=status_path)
    except Refused as exc:
        receipt = write_receipt(
            outbox, path, "refused", str(exc), {"id": rid,
                                                "dry_run": dry_run},
            dry_run=dry_run)
        return json.loads(receipt.read_text(encoding="utf-8"))

    plan = {
        "id": rid,
        "purpose": req.get("purpose"),
        "broker": broker,
        "files": [{"target": str(t), "bytes": len(payload),
                   "sha256": sha256_bytes(payload)[:16]}
                  for t, (_n, payload) in zip(targets, pairs)],
        "services": list(req.get("services") or []),
        "verify_command": req.get("verify_command"),
        "would_restart": bool(allow_restart and (req.get("services") or [])),
    }

    if dry_run:
        receipt = write_receipt(outbox, path, "dry_run",
                                "all gates passed; nothing written",
                                plan, dry_run=True)
        return json.loads(receipt.read_text(encoding="utf-8"))

    backup_dir = outbox / "backups" / rid
    made: List[Dict[str, str]] = []
    try:
        made = backup_targets(targets, backup_dir)
        write_payloads(targets, pairs)

        verify = req.get("verify_command")
        if isinstance(verify, str) and verify.strip():
            rc, tail = run_command(
                ["bash", "-lc", VERIFY_SHELL_PREFIX + verify], root,
                test_timeout)
            if rc != 0:
                restore_backup(made)
                # 126/127 mean the shell could not RUN the verifier (not found,
                # not executable). That is a configuration error in the request,
                # and reporting it as "tests failed" sends the next reader
                # hunting for a broken suite that was never invoked -- live
                # 2026-10-30: a Windows-pathed interpreter under `bash -lc`
                # produced exactly that (rc=127, "No such file or directory").
                if rc in (126, 127):
                    reason = (
                        f"verification command could not be executed "
                        f"(rc={rc}); originals restored. The verifier never "
                        f"ran, so this is a request/config fault, not a test "
                        f"failure: {tail.strip()[:200]}")
                else:
                    reason = (f"verification failed (rc={rc}); originals "
                              f"restored")
                receipt = write_receipt(
                    outbox, path, "failed", reason,
                    {**plan, "verify_rc": rc, "verify_command": verify,
                     "verify_output": tail, "backups": made})
                return json.loads(receipt.read_text(encoding="utf-8"))
            plan["verify_rc"] = rc
            plan["verify_command"] = verify
            plan["verify_output"] = tail[-800:]

        services = list(req.get("services") or [])
        if services and allow_restart:
            plan["restart"] = restart_services(services, root, test_timeout)
            if any(r.get("rc") not in (0, None) or r.get("error")
                   for r in plan["restart"]):
                receipt = write_receipt(
                    outbox, path, "failed",
                    "files applied but a service restart failed; inspect the "
                    "receipt before retrying",
                    {**plan, "backups": made})
                return json.loads(receipt.read_text(encoding="utf-8"))
        elif services:
            plan["restart_skipped"] = (
                "run with --allow-restart to rebuild/recreate these services")

        receipt = write_receipt(outbox, path, "applied",
                                "all gates passed and payloads written",
                                {**plan, "backups": made})
        # The request has been consumed; move it aside so it is never applied
        # twice, but keep it for the audit trail.
        done_dir = outbox / "applied"
        done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(done_dir / path.name))
        return json.loads(receipt.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        try:
            restore_backup(made)
            restored = True
        except Exception:  # noqa: BLE001
            restored = False
        receipt = write_receipt(
            outbox, path, "failed",
            f"{type(exc).__name__}: {exc}"
            + ("" if restored else " [RESTORE FAILED - inspect backups]"),
            {**plan, "backups": made, "restored": restored})
        return json.loads(receipt.read_text(encoding="utf-8"))


def pending_requests(outbox: Path) -> List[Path]:
    if not outbox.is_dir():
        return []
    return sorted(p for p in outbox.glob(f"*{REQUEST_SUFFIX}")
                  if p.is_file() and not p.name.startswith("."))


def check_outbox_writable(outbox: Path) -> None:
    """Refuse to run at all if receipts cannot be written.

    The receipt IS the audit trail: it is the only record of what was applied,
    which gate saw what, and where the backups went. An outbox that exists but
    is not writable by this user therefore fails the worst possible way -- the
    change lands and the evidence does not. So the whole applier refuses up
    front rather than discovering it mid-apply.

    Found by the live host prep on 2026-10-30: the outbox was created as an
    intermediate of ``install -d .../outbox/applied``, which GNU install leaves
    at root:root 755, so the ubuntu-run applier could read it but not write it.
    Nothing caught it because a dry run against a separate writable directory
    passed, and an empty outbox returns before any write is attempted.
    """
    try:
        outbox.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DeployError(
            f"outbox is not usable: cannot create {outbox} "
            f"({type(exc).__name__}: {exc})") from exc
    probe = outbox / ".write-probe"
    try:
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise DeployError(
            f"outbox {outbox} is not writable by uid {os.getuid() if hasattr(os, 'getuid') else '?'} "
            f"({type(exc).__name__}: {exc}). Refusing to run: without a "
            f"writable outbox there is no receipt, so an applied change would "
            f"leave no audit trail and no rollback record. Fix ownership, "
            f"e.g. `sudo chown -R <run-user>:<run-group> {outbox}`.") from exc


def run_once(outbox: Path, root: Path, **kwargs: Any) -> int:
    """Process every pending request. Returns the worst exit code."""
    requests = pending_requests(outbox)
    if not requests:
        print(f"[applier] no pending requests in {outbox}")
        return 0
    # Before ANY payload is written: see check_outbox_writable. Reported as
    # rc=3 (installation fault) rather than propagated, so `--watch` logs one
    # clear line per pass and a library caller gets an exit code, not a
    # traceback, for a misconfiguration that is identical every time.
    try:
        check_outbox_writable(outbox)
    except DeployError as exc:
        print(f"[applier] {exc}", file=sys.stderr)
        return 3
    worst = 0
    with ApplierLock(outbox):
        for path in requests:
            receipt = handle_request(path, outbox, root, **kwargs)
            status = receipt.get("status")
            print(f"[applier] {path.name} -> {status}: "
                  f"{str(receipt.get('reason'))[:160]}")
            for row in receipt.get("files") or []:
                print(f"          {row['target']} "
                      f"({row['bytes']} bytes, sha256 {row['sha256']}...)")
            if status == "applied":
                continue
            worst = max(worst, 2 if status in ("refused", "failed") else 0)
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply agent-staged deploy requests behind safety gates.")
    parser.add_argument("--root", required=True,
                        help="Deployment root (e.g. /srv/myapp).")
    parser.add_argument("--outbox", default=None,
                        help="Request directory (default: ROOT/.deploy/outbox).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run every gate and write receipts; change nothing.")
    parser.add_argument("--once", action="store_true",
                        help="Process what is pending, then exit.")
    parser.add_argument("--watch", action="store_true",
                        help="Poll the outbox until interrupted.")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--allow-restart", action="store_true",
                        help="Permit compose build/up for requested services.")
    parser.add_argument("--policy", default=os.environ.get(
                            "AGENT_DEPLOY_POLICY", "").strip() or None,
                        help="Policy JSON: readiness status file, safe states, "
                             "extra refused filenames. No policy means only the "
                             "generic gates run (and the readiness gate is "
                             "recorded as skipped).")
    parser.add_argument("--safe-states", default=None,
                        help="Override the policy's safe states "
                             "(comma-separated).")
    parser.add_argument("--max-age-sec", type=float, default=None,
                        help="Override how old a request may be.")
    parser.add_argument("--max-status-age-sec", type=float, default=None,
                        help="Override how stale the readiness status may be.")
    parser.add_argument("--status-path", default=None,
                        help="Override the readiness status file (tests).")
    parser.add_argument("--test-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    if not args.once and not args.watch:
        args.once = True

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"[applier] deploy root does not exist: {root}", file=sys.stderr)
        return 3
    outbox = Path(args.outbox).expanduser() if args.outbox \
        else root / ".deploy" / "outbox"
    try:
        check_outbox_writable(outbox)
    except DeployError as exc:
        # Fail CLOSED and loudly on a broken installation. An unwritable
        # outbox discovered only at first use means the first real request
        # would be applied with no receipt and no rollback record.
        print(f"[applier] {exc}", file=sys.stderr)
        return 3

    try:
        policy = Policy.load(args.policy)
    except DeployError as exc:
        # A gate that cannot load its policy must fail CLOSED, loudly.
        print(f"[applier] policy error: {exc}", file=sys.stderr)
        return 3
    if args.safe_states is not None:
        policy.safe_states = tuple(s.strip()
                                   for s in args.safe_states.split(",")
                                   if s.strip())
    if args.max_age_sec is not None:
        policy.max_age_sec = args.max_age_sec
    if args.max_status_age_sec is not None:
        policy.max_status_age_sec = args.max_status_age_sec

    kwargs: Dict[str, Any] = {
        "dry_run": args.dry_run,
        "policy": policy,
        "allow_restart": args.allow_restart,
        "test_timeout": args.test_timeout,
        "status_path": Path(args.status_path) if args.status_path else None,
    }

    print(f"[applier] root={root} outbox={outbox} dry_run={args.dry_run} "
          f"allow_restart={args.allow_restart} policy={json.dumps(policy.to_dict())}")
    if args.watch:
        while True:
            try:
                run_once(outbox, root, **kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[applier] pass failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            try:
                time.sleep(max(2.0, args.interval))
            except KeyboardInterrupt:
                return 0
        return 0
    return run_once(outbox, root, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
