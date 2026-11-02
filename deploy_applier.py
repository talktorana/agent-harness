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
import re
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


def _argv_list(raw: Any) -> Tuple[str, ...]:
    """A policy-supplied command as an ARGV tuple, or empty.

    Argv, never a shell string. The substituted values (an instance id, a
    filename, a JSON document) are operator-adjacent data, and joining them into
    a shell command would make every one of them a quoting problem and a
    potential injection. With a list there is no shell to inject into.
    """
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(part) for part in raw if isinstance(part, str) and part)

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
        #: The status document's OWN timestamp field -- whatever the writer
        #: calls it (``checked_at``, ``updated_utc``, ``as_of``); the policy
        #: names it, the applier never guesses one.
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

        #: Which container a typed action's commands run against. Named by the
        #: policy because only the deployment knows its own service topology --
        #: the applier would otherwise have to guess a container name, and a
        #: guess that is wrong runs the command against nothing (or, worse,
        #: against another product's container on the same host).
        actions_doc = doc.get("actions")
        actions_doc = actions_doc if isinstance(actions_doc, dict) else {}
        self.container = str(actions_doc.get("container") or "").strip()

        #: TYPED ACTIONS the gate will perform on the deployment's behalf.
        #:
        #: A request that names an action is not a file deploy: it asks the host
        #: to run one of the deployment's OWN sanctioned commands. That is the
        #: only shape that can change runtime state safely -- state lives in
        #: files this applier is forbidden to write (and should be: they are
        #: owned by the running service, which caches them), so the write must
        #: go THROUGH that service, not around it.
        #:
        #: Declared in the policy, never hardcoded here: the applier must stay
        #: free of any product's vocabulary. Each entry is
        #:   {allowed_fields: [...], read_argv: [...], apply_argv: [...],
        #:    validate_argv: [...], run_timeout_sec: n}
        #: where ``argv`` templates may use {container}, {instance_id},
        #: {profile} and {json}. ARGV LISTS, not shell strings, so nothing is
        #: re-quoted and a value can never become a command.
        actions = doc.get("actions")
        actions = actions if isinstance(actions, dict) else {}
        self.actions: Dict[str, Dict[str, Any]] = {}
        for kind, spec in actions.items():
            if isinstance(kind, str) and kind.startswith("_"):
                continue  # _comment and friends
            if not isinstance(spec, dict):
                continue
            self.actions[str(kind)] = {
                "allowed_fields": tuple(
                    str(f) for f in (spec.get("allowed_fields") or [])
                    if isinstance(f, str) and f.strip()),
                "read_argv": _argv_list(spec.get("read_argv")),
                "apply_argv": _argv_list(spec.get("apply_argv")),
                "validate_argv": _argv_list(spec.get("validate_argv")),
                "run_timeout_sec": float(spec.get("run_timeout_sec") or 120.0),
            }

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

        #: Where an applied change should be COMMITTED. Optional and disabled by
        #: default: a deployment may have no repository, and a deploy must not
        #: depend on one. Shape is documented at GIT_DEFAULTS.
        git_doc = doc.get("git")
        self.git: Dict[str, Any] = git_doc if isinstance(git_doc, dict) else {}

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
            # Say whether an applied change reaches the repository, and where.
            # A receipt claiming a commit with no mirror configured would be
            # the same class of silent lie as a gate reporting itself passed
            # when it never ran.
            "git": {"enabled": _truthy(self.git.get("enabled")),
                    "mirror_dir": str(self.git.get("mirror_dir") or ""),
                    "remote": str(self.git.get("remote") or ""),
                    "branch": str(self.git.get("branch") or "")},
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


class PendingWindow(Refused):
    """A refusal about the WORLD, not about the request.

    The system being busy, or the status file being stale, changes on its own:
    the same request may well be appliable in an hour, so it must stay pending.
    Every other refusal is about the request itself -- its checksums, its paths,
    its age -- and re-reading it a thousand times gives the same answer, so
    leaving it in the outbox means re-refusing it on every pass FOREVER.

    That was not hypothetical. Ten such requests sat in the live outbox for a
    day, which kept the applier's systemd unit permanently FAILED (exit 2 on
    every pass) and its log full of the same ten lines every three minutes. A
    unit that is always red is an alarm nobody reads, so the day a real deploy is
    refused for a real reason it looks exactly like the noise. Quarantining what
    cannot change makes red mean something again.
    """


def _move_staged_siblings(path: Path, outbox: Path, dest_dir: Path) -> None:
    """Move the payloads staged beside a DECIDED request into ``dest_dir``.

    The request and its ``.N.payload`` siblings are one unit: once the host has
    decided, no gate will read the payload again, so leaving it in the outbox
    root keeps a file that looks indistinguishable from pending work.

    That is not hypothetical. On 2026-11-01 an agent tasked with a ROUTINE
    CONFIG CHANGE sized ``req-...0.payload`` (631008 B, a copy of
    ``engine/collector.py`` from an APPLIED request) and concluded the deployed
    collector was that file, then reported the live feature as "never deployed"
    and warned the owner their working tree was dangerous. The deployed file was
    620781 B. The payload had sat in the root for eight hours because the
    REFUSED path moved siblings and the APPLIED path did not -- so "everything
    in the root is pending" was true for refusals and false for successes.

    The direction of that error matters: a false "the code you think is running
    is not running" makes an operator distrust a deployment that is fine, and
    the natural next move is to redeploy over something. Keeping the root
    meaning exactly one thing -- undecided requests -- is worth more than the
    convenience of finding a payload where it was staged.

    Failures are swallowed: this is tidy-up, and a pass that already applied the
    change must not report failure because a file could not be moved.
    """
    receipt = _receipt_name(path)
    base = (path.name[:-len(REQUEST_SUFFIX)]
            if path.name.endswith(REQUEST_SUFFIX) else path.stem)
    try:
        for sibling in sorted(Path(outbox).glob(f"{base}.*")):
            if sibling.name == receipt or not sibling.is_file():
                continue
            #: A directory (e.g. a per-request backup dir sharing the prefix)
            #: is NOT ours to move; only the staged payload files are.
            if sibling.is_symlink() or not sibling.resolve().is_file():
                continue
            shutil.move(str(sibling), str(dest_dir / sibling.name))
    except OSError:
        pass


def _quarantine_refused(path: Path, outbox: Path) -> Optional[Path]:
    """Move a permanently-refused request out of the pending set.

    The RECEIPT stays where ``write_receipt`` put it: it is the audit trail, and
    the deployment's dashboard reads receipts from the outbox root to show what
    the gate decided. Only the request and the payloads staged beside it move, so
    nothing is deleted and moving a file back is how a request gets reconsidered.

    Failures are swallowed by design. Quarantine is noise control: if it cannot
    move the file, the request stays pending and is refused again next pass,
    which is merely the inconvenience this function exists to remove. Aborting an
    otherwise-complete pass over it would be worse.
    """
    try:
        dest_dir = Path(outbox) / "refused"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        shutil.move(str(path), str(dest))
        #: The staged payloads travel with their request. Left behind they are
        #: files nothing will ever read again, accumulating in the directory the
        #: gates scan on every pass.
        _move_staged_siblings(path, outbox, dest_dir)
        return dest
    except OSError:
        return None


# =================================================================== gates

def _payload_name(raw: Any) -> str:
    """The FILENAME of a staged payload, whatever path the requester recorded.

    The sidecar stages a payload beside its request and records the path IT
    sees -- inside a container that may be ``/project/app/data/deploy/outbox/
    req-X.0.payload``, while the host applier reads the SAME directory at a
    different path (``/srv/app/data/deploy/outbox`` in one consumer's compose
    file). Two mount views of one directory, deliberately: the outbox is the
    seam, and the seam is the reason the views differ. Resolving the recorded
    string as a path therefore looks for the REQUESTER's view on a host that
    does not have it, and every request large enough to be staged as a file
    was refused with "payload_path does not exist" -- live for a full day in
    one consumer deployment, ten requests, zero appliable. The FILENAME is the
    only part that survives the crossing, so that is the only part used.
    """
    text = str(raw or "").strip()
    #: Both separators, because the two halves of the seam may not run the same
    #: OS and a Windows-staged request must not smuggle its whole path into a
    #: name (``Path("C:\\a\\b").name`` is ``b`` on Windows and the entire string
    #: on POSIX).
    return text.replace("\\", "/").rsplit("/", 1)[-1]


def _resolve_payload(raw: Any, outbox: Optional[Path], index: int) -> Path:
    """Resolve a staged payload to a file INSIDE the outbox.

    The name-only rule fixes the mount crossing above and closes a hole that was
    open the whole time: a request could previously name ANY path the applier
    process could read -- ``/etc/passwd``, a credential file, the host's ``.env``
    -- and the applier would copy its bytes into the deployment. Targets are
    contained to the deployment root by ``gate_paths``; the SOURCE was contained
    to nothing at all. Readable-by-the-applier is a wide set: it runs as the
    user that owns production.

    Containment here is by CONSTRUCTION rather than by checking: the path is
    built as ``outbox / name``, and a name with no separator in it cannot leave
    the directory. There is no traversal to test for because there is no way to
    express one.
    """
    name = _payload_name(raw)
    if not name or name in (".", ".."):
        raise Refused(
            f"files[{index}] payload_path names no file: {str(raw)!r}")
    outbox_dir = Path(outbox) if outbox is not None else Path(".")
    try:
        outbox_dir = outbox_dir.resolve()
    except OSError:                                  # pragma: no cover
        pass
    payload = outbox_dir / name
    if not payload.is_file():
        raise Refused(
            f"files[{index}] payload_path is not staged beside the request: "
            f"{name} (looked in {payload})")
    #: Defence in depth behind a construction that cannot escape: if resolve()
    #: says the file is elsewhere (a symlink planted in the outbox), refuse
    #: rather than follow it.
    try:
        real = payload.resolve()
    except OSError:                                  # pragma: no cover
        raise Refused(f"files[{index}] payload_path is unreadable: {name}")
    if real != payload and not str(real).startswith(str(outbox_dir) + os.sep):
        raise Refused(
            f"files[{index}] payload_path {name} resolves outside the outbox "
            f"({real}); refusing to read it")
    return payload


def gate_schema(req: Dict[str, Any],
                outbox: Optional[Path] = None) -> List[Tuple[str, Any]]:
    """Validate shape. Returns the (target, payload) pairs to apply.

    ``outbox`` is where staged payloads are looked up by name; see
    :func:`_resolve_payload`. It is optional so a direct caller can still use
    this gate alone, in which case payloads are resolved relative to the
    CURRENT directory -- never as the absolute path the request recorded.
    """
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
            payload = _resolve_payload(entry["payload_path"], outbox, i)
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


def gate_readiness(root: Path, policy: Policy,
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
        raise PendingWindow(
            f"no {label} status file at {path}; cannot prove the system is "
            f"safe to disturb")
    data, err = _read_json(path)
    if data is None:
        raise PendingWindow(f"{label} status unreadable: {err}")
    state = str(data.get(policy.status_field) or "")
    age = time.time() - path.stat().st_mtime
    if age > policy.max_status_age_sec:
        raise PendingWindow(
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
        #: PendingWindow, not Refused: a busy system is the NORMAL state of a
        #: live one most of the day, and a change proposed during it is deferred
        #: rather than rejected. This is the one refusal that a later pass can
        #: legitimately overturn.
        raise PendingWindow(
            f"{label} state is {state!r}; only {list(policy.safe_states)} may "
            f"be disturbed by a deploy")
    #: ``stamp`` is the verdict's own timestamp AS THE POLICY'S FIELD supplied
    #: it, and None when the policy names no such field. An earlier version
    #: guessed a field name here when none was declared, which made the receipt
    #: look like it had dated the verdict while silently reading whatever one
    #: consumer happened to call it. ``stamp_field: None`` alongside ``stamp:
    #: None`` is the honest shape: freshness was judged by mtime alone.
    return {"gate": label, "state": state, "age_sec": round(age, 1),
            "verdict_age_sec": (round(verdict_age, 1)
                                if verdict_age is not None else None),
            "field": policy.status_field,
            "stamp_field": policy.timestamp_field or None,
            "stamp": (data.get(policy.timestamp_field)
                      if policy.timestamp_field else None)}


#: The field-name HISTORY of one receipt key, newest first.
#:
#: Receipts outlive the code that writes them: they are the audit trail, kept
#: beside the request they describe, so a host's outbox holds verdicts from every
#: applier version it has ever run. The readiness verdict used to be filed under
#: a name borrowed from one consumer's domain (this repo is generic -- see
#: test_genericity.py), and every receipt written before that rename still says
#: so.
#:
#: A reader that knew only the current name would print "no verdict" for all of
#: them, i.e. an audit trail that goes blank at the upgrade. That silence is the
#: exact thing a receipt exists to prevent, so the old name stays readable here
#: and ONLY here: a writer never emits it.
READINESS_KEYS: Tuple[str, ...] = ("readiness", "broker")


def readiness_of(doc: Any) -> Dict[str, Any]:
    """The readiness verdict out of a receipt, whichever schema wrote it.

    Returns ``{}`` when there is none, so a caller cannot mistake "not recorded"
    for "recorded as fine" -- the same three-valued discipline as a skipped
    gate. Non-dict input is treated as no verdict rather than raised on, so one
    corrupt receipt in an outbox cannot break a reader listing many.
    """
    if not isinstance(doc, dict):
        return {}
    for key in READINESS_KEYS:
        found = doc.get(key)
        if isinstance(found, dict):
            return found
    return {}


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


# ============================================================== typed actions

#: Characters allowed in an instance identifier a request may name.
#: Deliberately conservative: the value is substituted into an argv, echoed into
#: a receipt and compared against ids read from the deployment, so there is no
#: reason to accept anything that is not a plain slug.
_INSTANCE_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _fmt_argv(template: Tuple[str, ...], **values: str) -> List[str]:
    """Substitute {name} placeholders in each element. No shell, no re-quoting."""
    out: List[str] = []
    for part in template:
        for key, value in values.items():
            part = part.replace("{" + key + "}", value)
        out.append(part)
    return out


def _run_argv(argv: List[str], timeout: float,
              stdin_text: Optional[str] = None) -> Tuple[int, str]:
    """Run a policy command and return (rc, combined output tail).

    Never raises on a failing command -- the caller turns rc into a verdict.
    A missing executable is rc 127, which is distinguishable from a refusal in
    the same way the verify gate distinguishes it: the command never ran, so its
    output is not evidence about the deployment.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            input=stdin_text, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        return 127, f"{argv[0]}: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {' '.join(argv[:4])} ..."
    except OSError as exc:
        return 126, f"{type(exc).__name__}: {exc}"
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def gate_action(action: Dict[str, Any],
                policy: "Policy") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Validate a TYPED action and resolve the row it would change.

    Returns ``(kind, spec, current_row)``. Raises :class:`Refused` for anything
    permanent -- a bad kind, a field the policy does not expose, an instance
    that does not exist, or a value the deployment's OWN validator rejects.

    The instance must ALREADY EXIST. Allowing creation would turn "set a field
    on this book" into "start a new trading book", which is a materially larger
    decision than the one the operator was asked about.

    Values are validated by ASKING THE DEPLOYMENT (``validate_argv``), not by
    this module: profile names, allowed ranges and every other domain rule live
    with the code that enforces them at run time. A second copy here would be
    the one that goes out of date, and it would do so silently.
    """
    kind = str(action.get("kind") or "").strip()
    if not kind:
        raise Refused("action.kind is required")
    spec = policy.actions.get(kind)
    if spec is None:
        known = ", ".join(sorted(policy.actions)) or "(none configured)"
        raise Refused(
            f"action kind {kind!r} is not permitted by this deployment's "
            f"policy; permitted: {known}")

    instance_id = str(action.get("instance_id") or "").strip()
    if not _INSTANCE_ID_OK.match(instance_id):
        raise Refused(
            f"action.instance_id must be a plain identifier, got "
            f"{instance_id!r}")

    changes = action.get("set")
    if not isinstance(changes, dict) or not changes:
        raise Refused("action.set must be a non-empty object")
    allowed = spec["allowed_fields"]
    if not allowed:
        raise Refused(f"action kind {kind!r} exposes no fields to set")
    extra = sorted(set(changes) - set(allowed))
    if extra:
        raise Refused(
            f"action.set names field(s) {extra} that {kind!r} does not expose; "
            f"settable: {sorted(allowed)}")

    for field, value in changes.items():
        #: Scalars only. A nested object would let a request replace a whole
        #: sub-structure by naming one field, which is the opposite of the
        #: bounded change this exists to make.
        if isinstance(value, (dict, list)):
            raise Refused(
                f"action.set[{field!r}] must be a scalar, not "
                f"{type(value).__name__}")

    if not spec["read_argv"] or not spec["apply_argv"]:
        raise Refused(f"action kind {kind!r} is not fully configured "
                      f"(read_argv / apply_argv missing)")

    rc, out = _run_argv(
        _fmt_argv(spec["read_argv"], container=policy.container,
                  instance_id=instance_id),
        spec["run_timeout_sec"])
    if rc != 0:
        raise Refused(
            f"could not read the deployment's current configuration "
            f"(rc={rc}): {out[:300]}")
    try:
        doc = json.loads(out[out.index("{"):]) if "{" in out else {}
    except (ValueError, TypeError):
        raise Refused(f"deployment returned unreadable configuration: "
                      f"{out[:300]}") from None
    rows = doc.get("instances") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        raise Refused("deployment returned no instance list")
    current = next((r for r in rows
                    if isinstance(r, dict)
                    and str(r.get("instance_id")) == instance_id), None)
    if current is None:
        known = sorted(str(r.get("instance_id")) for r in rows
                       if isinstance(r, dict))[:10]
        raise Refused(
            f"instance {instance_id!r} does not exist on this deployment "
            f"(known: {known}); this action changes existing instances and "
            f"never creates one")

    if "risk_profile" in changes and spec["validate_argv"]:
        profile = "" if changes["risk_profile"] is None else str(
            changes["risk_profile"])
        rc, out = _run_argv(
            _fmt_argv(spec["validate_argv"], container=policy.container,
                      instance_id=instance_id, profile=profile),
            spec["run_timeout_sec"])
        if rc != 0:
            raise Refused(
                f"the deployment rejected risk_profile={profile!r} "
                f"(rc={rc}): {out[:300]}")

    return kind, spec, current


def apply_action(kind: str, spec: Dict[str, Any], current: Dict[str, Any],
                 changes: Dict[str, Any], policy: "Policy",
                 instance_id: str) -> Tuple[Dict[str, Any], str]:
    """Merge the allowed fields into the CURRENT row and write it back.

    Read-modify-write, not "send what you want": the deployment's write path is
    an UPSERT that REPLACES the whole row, so a request carrying only the field
    it means to change would silently reset every other field to its default.
    That is the single most likely way this feature could lose an operator's
    configuration, and doing the merge here means the caller never has to
    reproduce a row it cannot see.

    Only ``allowed_fields`` are copied, so the row read back cannot smuggle in
    anything the policy does not expose.
    """
    merged = dict(current)
    merged.update({k: v for k, v in changes.items()
                   if k in spec["allowed_fields"]})
    argv = _fmt_argv(spec["apply_argv"], container=policy.container,
                     instance_id=instance_id, json=json.dumps(merged),
                     profile=str(changes.get("risk_profile") or ""))
    rc, out = _run_argv(argv, spec["run_timeout_sec"])
    if rc != 0:
        raise Refused(f"the deployment refused the change (rc={rc}): "
                      f"{out[:400]}")

    #: Verify by READING BACK, not by trusting the exit code. The write path
    #: reports success for an upsert even when nothing changed, and a change
    #: that silently did not take is exactly the failure this whole feature
    #: exists to make impossible.
    rc, after = _run_argv(
        _fmt_argv(spec["read_argv"], container=policy.container,
                  instance_id=instance_id),
        spec["run_timeout_sec"])
    if rc != 0:
        return merged, (f"written, direction unverified: re-read failed "
                        f"(rc={rc}): {after[:200]}")
    try:
        doc = json.loads(after[after.index("{"):]) if "{" in after else {}
        rows = doc.get("instances") if isinstance(doc, dict) else None
        row = next((r for r in (rows or [])
                    if isinstance(r, dict)
                    and str(r.get("instance_id")) == instance_id), None)
    except (ValueError, TypeError):
        return merged, "written, direction unverified: re-read unparseable"
    if row is None:
        return merged, "written, but the instance is gone on re-read"
    for field, wanted in changes.items():
        got = row.get(field)
        if str(got) != str(wanted):
            return merged, (f"WRITE DID NOT TAKE: {field} reads {got!r}, "
                            f"expected {wanted!r}")
    return merged, "verified by re-read"


# ======================================================= git mirror sync
#
# An applied deploy changes files the repository never hears about. The deploy
# root is deliberately NOT a git checkout (a `git checkout` there would delete
# every agent-applied, not-yet-committed file -- the exact drift this is meant
# to end), and the sidecar's copy of it is read-only. So the commit is made in a
# SEPARATE clone and only the applied paths are copied across.
#
# Design rules, each one earned:
#
# * BEST EFFORT, NEVER FATAL. A missing mirror, an absent deploy key or a
#   rejected push must not turn a successful deploy into a failed one. The
#   trading change already landed and was verified; refusing it now would be
#   worse than an un-pushed commit. Every outcome is recorded in the receipt.
# * ONLY THE APPLIED PATHS ARE STAGED. Never `git add -A`: the mirror is a
#   working tree that may hold unrelated drift, and sweeping it into a commit
#   would attribute someone else's edit to this request id.
# * THE COMMIT IS NEVER DISCARDED. If the push fails, the commit stays so the
#   next successful push carries it. Rolling back would destroy the only copy.
# * NOTHING HERE KNOWS THE PRODUCT. Paths, remote, branch and identity all come
#   from the policy's ``git`` block, exactly like every other domain decision.

GIT_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "mirror_dir": "",
    #: Where the DEPLOY ROOT sits inside the repository.
    #:
    #: Almost never empty in practice, and getting it wrong is silent: the
    #: mirror is the repo root, while a deploy root is usually one directory
    #: within it (a deployment at ``/srv/app`` whose repo keeps that tree under
    #: ``app/``). Copying ``engine/x.py`` to the mirror root instead of
    #: ``app/engine/x.py`` does not error -- git happily commits a NEW
    #: top-level ``engine/`` directory, so the repository quietly gains a
    #: duplicate tree that no one asked for. Found by inspection against a
    #: live mirror on 2026-11-02, before it had ever run.
    "subdir": "",
    "remote": "origin",
    "branch": "",
    "user_name": "deploy-applier",
    "user_email": "deploy-applier@localhost",
    "timeout_sec": 120.0,
}

#: Paths that must never enter a commit, whatever the policy says. The applier
#: already refuses to WRITE these, so an applied target can never be one of
#: them; this is the second lock on the same door, because a mirror is a
#: different machine surface from the deploy root.
GIT_NEVER_STAGE_NAMES = (".env", ".git-credentials", "id_rsa", "id_ed25519")


def git_config(policy: "Policy") -> Dict[str, Any]:
    """The policy's ``git`` block, normalised. Absent means disabled."""
    doc = getattr(policy, "git", None)
    doc = doc if isinstance(doc, dict) else {}
    cfg = dict(GIT_DEFAULTS)
    for key in GIT_DEFAULTS:
        if key in doc and doc[key] not in (None, ""):
            cfg[key] = doc[key]
    cfg["enabled"] = _truthy(doc.get("enabled"))
    cfg["mirror_dir"] = str(cfg["mirror_dir"] or "").strip()
    cfg["remote"] = str(cfg["remote"] or "origin").strip() or "origin"
    cfg["branch"] = str(cfg["branch"] or "").strip()
    #: Trailing slashes only. Stripping the LEADING one would quietly turn an
    #: absolute path into a relative one, so "/etc" would validate as "etc" and
    #: the safety check in _safe_subdir would never see what was configured.
    cfg["subdir"] = str(cfg["subdir"] or "").strip().rstrip("/")
    cfg["timeout_sec"] = float(cfg["timeout_sec"] or 120.0)
    return cfg


def _safe_subdir(subdir: str) -> Optional[str]:
    """None when usable, else why not. Refuses anything that escapes the clone.

    Deliberately not ``Path.is_absolute()``: that is platform-dependent, and
    ``"/abs/path"`` is ABSOLUTE on the Linux host but NOT on Windows (where it
    means "current drive"). A safety check that a rehearsal on one OS passes
    and production on another fails is worse than no check, so the test is on
    the string: no leading separator, no drive letter, no traversal parts.
    """
    if not subdir:
        return None
    if subdir.startswith(("/", "\\")):
        return f"git.subdir must be relative to the repository root, got {subdir!r}"
    if re.match(r"^[A-Za-z]:", subdir):
        return f"git.subdir must not name a drive, got {subdir!r}"
    if any(part in ("..", ".") for part in Path(subdir).parts):
        return f"git.subdir may not contain '.' or '..': {subdir!r}"
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _git(mirror: Path, args: List[str], timeout: float,
         ident: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run git inside the mirror. ``-c`` identity keeps it off global config."""
    cmd = ["git"]
    if ident:
        cmd += ["-c", f"user.name={ident['user_name']}",
                "-c", f"user.email={ident['user_email']}"]
    cmd += list(args)
    try:
        return run_command(cmd, mirror, timeout)
    except FileNotFoundError:
        return 127, "git is not installed on this host"
    except subprocess.TimeoutExpired:
        return 124, f"git {args[0]} timed out after {timeout:.0f}s"


def _mirror_ready(mirror: Path, cfg: Dict[str, Any]) -> Optional[str]:
    """None when usable, else the reason it is not."""
    if not mirror.is_dir():
        return f"mirror directory does not exist: {mirror}"
    if not (mirror / ".git").exists():
        return (f"{mirror} is not a git checkout (no .git) -- cloned with "
                f"--depth would still have one, so this is not a clone at all")
    return None


def _stageable(path: Path, root: Path, policy: "Policy") -> bool:
    """Is this applied target allowed into a commit? Belt and braces."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    parts = [p.lower() for p in rel.parts]
    if parts and parts[0] == "data":
        return False
    if rel.name.lower() in GIT_NEVER_STAGE_NAMES:
        return False
    if rel.name in getattr(policy, "forbidden_names", ()):
        return False
    return True


def sync_applied_files(targets: List[Path], root: Path, *,
                       policy: "Policy", rid: str,
                       purpose: Optional[str] = None) -> Dict[str, Any]:
    """Mirror the applied files into the git clone, commit and push.

    Returns a dict for the receipt. NEVER raises and never signals failure by
    return value -- the caller has already applied the change, so this is
    bookkeeping, and bookkeeping must not overturn a verified deploy.
    """
    cfg = git_config(policy)
    if not cfg["enabled"]:
        return {"state": "disabled"}

    mirror = Path(cfg["mirror_dir"]).expanduser()
    problem = _mirror_ready(mirror, cfg)
    if problem:
        return {"state": "no_mirror", "mirror": str(mirror),
                "reason": problem,
                "setup": (f"git clone <repo-url> {mirror} && "
                          f"cd {mirror} && git remote -v")}

    #: A subdir that does not exist in the clone means the mirror points at the
    #: wrong repository (or the wrong subtree). Report it instead of creating
    #: a parallel tree that would never be reviewed.
    bad_subdir = _safe_subdir(cfg["subdir"])
    if bad_subdir:
        return {"state": "failed", "mirror": str(mirror),
                "reason": bad_subdir}
    prefix = Path(cfg["subdir"]) if cfg["subdir"] else None
    if prefix is not None and not (mirror / prefix).is_dir():
        return {"state": "no_mirror", "mirror": str(mirror),
                "reason": (f"git.subdir {cfg['subdir']!r} does not exist in "
                           f"{mirror}; the mirror is not this deployment's "
                           f"repository, or the subdir is wrong. Refusing to "
                           f"create a parallel tree."),
                "setup": (f"cd {mirror} && ls -d {cfg['subdir']} # must exist")}

    rels = [p for p in targets if _stageable(p, root, policy)]
    #: Relative and POSIX-joined on purpose: these strings go into the receipt
    #: and the commit, so an OS-specific separator would make a rehearsal on
    #: Windows describe a different path from the one the Linux host commits.
    skipped = [p.relative_to(root.resolve()).as_posix()
               for p in targets if p not in rels]
    if not rels:
        return {"state": "no_change", "mirror": str(mirror),
                "reason": "no applied target is stageable (all runtime state)",
                "skipped": skipped}

    ident = {"user_name": cfg["user_name"], "user_email": cfg["user_email"]}
    copied: List[str] = []
    for src in rels:
        rel = src.resolve().relative_to(root.resolve())
        #: Repo-relative, i.e. where the file belongs in the CLONE -- which is
        #: not the same as where it sits under the deploy root.
        in_repo = (prefix / rel) if prefix is not None else rel
        dest = mirror / in_repo
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(in_repo.as_posix())
        except OSError as exc:
            return {"state": "failed", "mirror": str(mirror),
                    "reason": f"copy {rel} failed: {type(exc).__name__}: {exc}",
                    "copied": copied}

    rc, out = _git(mirror, ["add", "--"] + copied, cfg["timeout_sec"], ident)
    if rc != 0:
        return {"state": "failed", "mirror": str(mirror),
                "reason": f"git add failed (rc={rc}): {out[-300:]}",
                "copied": copied}

    #: Nothing to commit is a SUCCESS, not a failure: re-applying an identical
    #: file is a normal outcome and must not read as a broken mirror.
    rc, _out = _git(mirror, ["diff", "--cached", "--quiet"], cfg["timeout_sec"],
                    ident)
    if rc == 0:
        return {"state": "no_change", "mirror": str(mirror),
                "reason": "applied files already match the mirror",
                "copied": copied, "skipped": skipped}

    subject = f"agent deploy {rid}: {_one_line(purpose) or 'applied change'}"
    body = (f"Applied by the deploy applier behind the readiness gate.\n\n"
            f"Request: {rid}\nFiles: {len(copied)}\n")
    rc, out = _git(mirror, ["commit", "-m", subject, "-m", body,
                            "--"] + copied, cfg["timeout_sec"], ident)
    if rc != 0:
        return {"state": "failed", "mirror": str(mirror),
                "reason": f"git commit failed (rc={rc}): {out[-300:]}",
                "copied": copied}

    rc, out = _git(mirror, ["rev-parse", "HEAD"], cfg["timeout_sec"], ident)
    sha = out.strip().splitlines()[-1] if rc == 0 and out.strip() else ""
    result: Dict[str, Any] = {"state": "committed", "mirror": str(mirror),
                              "sha": sha, "files": copied,
                              "subject": subject, "skipped": skipped}

    push_args = ["push", cfg["remote"]]
    if cfg["branch"]:
        push_args.append(cfg["branch"])
    rc, out = _git(mirror, push_args, cfg["timeout_sec"], ident)
    if rc == 0:
        result["state"] = "pushed"
        result["remote"] = cfg["remote"]
        return result

    #: Keep the commit. The change is real and this is the only copy of the
    #: record, so discarding it to make the receipt tidy would be backwards.
    result["state"] = "committed_not_pushed"
    result["reason"] = (f"push failed (rc={rc}): {out[-400:]}. The commit is "
                        f"kept locally and the next successful push carries it.")
    result["retry"] = f"cd {mirror} && git push {cfg['remote']}"
    return result


def _one_line(text: Optional[str], limit: int = 72) -> str:
    """A commit subject line: one line, no controls, bounded."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "\u2026"


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


def _handle_action(path: Path, outbox: Path, root: Path, *,
                   req: Dict[str, Any], action: Dict[str, Any],
                   policy: "Policy", dry_run: bool, rid: str,
                   status_path: Optional[Path]) -> Dict[str, Any]:
    """Gate, then apply, one typed action. Always returns a receipt dict.

    Same shape as a file deploy -- permanent gates first (quarantined on refusal,
    because they will fail identically forever), then the readiness window
    (defers, because the world changes) -- so an operator reads one receipt
    format regardless of which kind of change was asked for.
    """
    try:
        if int(req.get("schema_version") or 0) != SCHEMA_VERSION:
            raise Refused(
                f"schema_version must be {SCHEMA_VERSION}, got "
                f"{req.get('schema_version')!r}")
        purpose = req.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise Refused("'purpose' is required: receipts must say why")
        gate_age(req, path, policy.max_age_sec)
        kind, spec, current = gate_action(action, policy)
    except Refused as exc:
        receipt = write_receipt(outbox, path, "refused", str(exc),
                                {"id": rid, "action": action, "dry_run": dry_run},
                                dry_run=dry_run)
        if not dry_run:
            _quarantine_refused(path, outbox)
        return json.loads(receipt.read_text(encoding="utf-8"))

    try:
        readiness = gate_readiness(root, policy, status_path=status_path)
    except PendingWindow as exc:
        receipt = write_receipt(outbox, path, "refused", str(exc),
                                {"id": rid, "action": action,
                                 "dry_run": dry_run, "pending_window": True},
                                dry_run=dry_run)
        return json.loads(receipt.read_text(encoding="utf-8"))
    except Refused as exc:
        receipt = write_receipt(outbox, path, "refused", str(exc),
                                {"id": rid, "action": action,
                                 "dry_run": dry_run},
                                dry_run=dry_run)
        if not dry_run:
            _quarantine_refused(path, outbox)
        return json.loads(receipt.read_text(encoding="utf-8"))

    changes = dict(action.get("set") or {})
    before = {k: current.get(k) for k in changes}
    plan = {
        "id": rid,
        "purpose": req.get("purpose"),
        "readiness": readiness,
        "action": {"kind": kind, "instance_id": action.get("instance_id"),
                   "set": changes},
        #: Before/after in the receipt, because a runtime-state change has no
        #: diff to review afterwards: the file it touched is rewritten in place
        #: and the previous value is gone.
        "before": before,
    }

    if dry_run:
        receipt = write_receipt(outbox, path, "dry_run",
                                "all gates passed; nothing changed", plan,
                                dry_run=True)
        return json.loads(receipt.read_text(encoding="utf-8"))

    try:
        _merged, note = apply_action(
            kind, spec, current, changes, policy,
            str(action.get("instance_id")))
    except Refused as exc:
        receipt = write_receipt(outbox, path, "failed", str(exc),
                                {**plan, "purpose": req.get("purpose")})
        return json.loads(receipt.read_text(encoding="utf-8"))

    done_dir = outbox / "applied"
    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(done_dir / path.name))
    _move_staged_siblings(path, outbox, done_dir)
    receipt = write_receipt(outbox, path, "applied",
                            "all gates passed and the change was applied",
                            {**plan, "after": changes, "verification": note})
    return json.loads(receipt.read_text(encoding="utf-8"))


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
        #: An unparseable request is a property of the FILE, so no later pass
        #: will make it readable. Quarantine it rather than re-refusing forever.
        if not dry_run:
            _quarantine_refused(path, outbox)
        return json.loads(receipt.read_text(encoding="utf-8"))

    rid = str(req.get("id") or path.stem)

    #: A TYPED ACTION is a different kind of request: it asks the host to run one
    #: of the deployment's own commands, and carries no files at all. Branching
    #: here (rather than inside gate_schema) keeps the file gates honest -- they
    #: require a non-empty ``files`` list and must keep doing so, because
    #: "a deploy that writes nothing" is not a deploy.
    action = req.get("action")
    if isinstance(action, dict):
        return _handle_action(path, outbox, root, req=req, action=action,
                              policy=policy, dry_run=dry_run, rid=rid,
                              status_path=status_path)

    #: The permanent gates are separated from the readiness gate because their
    #: refusals mean different things about the FUTURE. A wrong checksum will be
    #: wrong on every pass; "the system is busy" will not be. Only the second
    #: one is worth keeping in the pending set -- see PendingWindow.
    try:
        pairs = gate_schema(req, outbox)
        targets = gate_paths(pairs, root, policy)
        gate_checksums(req, pairs)
        gate_age(req, path, policy.max_age_sec)
    except Refused as exc:
        receipt = write_receipt(
            outbox, path, "refused", str(exc), {"id": rid,
                                                "dry_run": dry_run},
            dry_run=dry_run)
        if not dry_run:
            _quarantine_refused(path, outbox)
        return json.loads(receipt.read_text(encoding="utf-8"))

    try:
        readiness = gate_readiness(root, policy, status_path=status_path)
    except PendingWindow as exc:
        #: Transient BY TYPE, not by matching the refusal text: a message rewrite
        #: would otherwise silently turn "wait for the window" into "drop the
        #: request", which is the worst possible failure -- a legitimate change
        #: vanishing with nothing but a receipt to show for it.
        receipt = write_receipt(
            outbox, path, "refused", str(exc), {"id": rid,
                                                "dry_run": dry_run,
                                                "pending_window": True},
            dry_run=dry_run)
        return json.loads(receipt.read_text(encoding="utf-8"))
    except Refused as exc:
        receipt = write_receipt(
            outbox, path, "refused", str(exc), {"id": rid,
                                                "dry_run": dry_run},
            dry_run=dry_run)
        if not dry_run:
            _quarantine_refused(path, outbox)
        return json.loads(receipt.read_text(encoding="utf-8"))

    plan = {
        "id": rid,
        "purpose": req.get("purpose"),
        "readiness": readiness,
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
                                {**plan, "backups": made,
                                 # Recorded BEFORE the receipt is written, so the
                                 # audit trail carries the commit as applied.
                                 # Best effort by design: it cannot change the
                                 # status above, because the deploy really did
                                 # land and reverting it to satisfy a git
                                 # problem would be worse than an un-pushed
                                 # commit.
                                 "git": sync_applied_files(
                                     targets, root, policy=policy, rid=rid,
                                     purpose=req.get("purpose"))})
        # The request has been consumed; move it aside so it is never applied
        # twice, but keep it for the audit trail.
        done_dir = outbox / "applied"
        done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(done_dir / path.name))
        #: ...and its staged payloads with it, for the reason spelled out in
        #: _move_staged_siblings: the outbox root must mean "undecided", and it
        #: stopped meaning that because ONLY the refusal path moved siblings.
        #: An agent read a leftover payload as the deployed file and reported a
        #: live feature as missing.
        _move_staged_siblings(path, outbox, done_dir)
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
            if receipt.get("pending_window"):
                #: DEFERRED is not FAILED. The readiness window is closed (the
                #: system is busy, or the verdict is stale) and the request
                #: stays queued for a later pass -- which is the normal state of
                #: a live system most of the day.
                #:
                #: Counting it as a failure made the applier's unit red every two
                #: minutes for as long as the system stayed busy. A unit that is
                #: always red is an alarm nobody reads, and the pass that refuses
                #: a REAL deploy then looks exactly like the noise -- which is
                #: the very problem quarantining was meant to solve,
                #: reintroduced from the other side. A permanent refusal still
                #: exits 2 exactly once, because it is quarantined and never seen
                #: again.
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
