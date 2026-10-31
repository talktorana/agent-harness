"""Regressions for deploy_applier.py: the host-side GATE.

The applier is the only path by which an autonomous agent's edit can reach a
live deployment, so its refusals matter more than its successes. Each test
asserts ONE gate holds while the others pass, because a gate satisfied by
accident -- a stale status read as safe, a missing file treated as agreement --
is worse than no gate at all.

Domain knowledge lives in the policy, so these tests use a GENERIC one: a
``readiness`` status file with states like ``ready``/``idle``. The
trading-specific policy (broker flat/matched, refused trading-state files) is
regressed in the deployment repo that owns it.

Run:  python -m unittest test_deploy_applier
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deploy_applier import (  # noqa: E402
    REQUEST_SUFFIX,
    ApplierLock,
    DeployError,
    Policy,
    Refused,
    _receipt_name,
    check_outbox_writable,
    gate_paths,
    gate_schema,
    handle_request,
    main,
    pending_requests,
    sha256_bytes,
)

#: A generic readiness vocabulary. Nothing here implies a broker or a book.
SAFE_STATES = ("ready", "idle")
#: Files a deployment would declare "not code" in its own policy.
POLICY_FORBIDDEN = ["app-secret.json", "runtime-ledger.dat", "owner.svg"]


def _status_doc(**over):
    return {**{"state": "ready", "reason": "ok"}, **over}


def make_policy(**over):
    """A policy document as a deployment would ship it, then loaded."""
    doc = {
        "readiness": {
            "status_file": "data/readiness.json",
            "field": "state",
            "label": "readiness",
            "safe_states": list(SAFE_STATES),
            "max_age_sec": 900,
        },
        "limits": {"max_age_sec": 900},
        "forbid": {"names": list(POLICY_FORBIDDEN)},
    }
    doc.update(over)
    return Policy(doc, name="test-policy")


class ApplierTestBase(unittest.TestCase):
    """A throwaway deployment root with a fresh, safe readiness status."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "engine").mkdir()
        #: Scrub the ambient deploy environment. The CLI defaults to it, so
        #: inside a sidecar -- where AGENT_DEPLOY_POLICY points at the REAL
        #: trading policy -- these tests judged a sandbox against production
        #: requirements and failed on "no broker status file" while passing
        #: everywhere else. A test whose verdict depends on the machine it
        #: happens to run on cannot tell a regression from its surroundings.
        ambient = {k: v for k, v in os.environ.items()
                   if k.startswith("AGENT_DEPLOY_") or k == "AGENT_ROOT"}
        if ambient:
            patcher = mock.patch.dict(os.environ, {})
            patcher.start()
            self.addCleanup(patcher.stop)
            for key in ambient:
                os.environ.pop(key, None)
        self.policy = make_policy()
        # A real policy file on disk, so the CLI path is tested end to end --
        # main() must load it, and a missing/unreadable one must fail CLOSED.
        self.policy_path = self.root / "deploy_policy.json"
        self.policy_path.write_text(json.dumps({
            "readiness": {
                "status_file": "data/readiness.json", "field": "state",
                "label": "readiness", "safe_states": list(SAFE_STATES),
                "max_age_sec": 900},
            "limits": {"max_age_sec": 900},
            "forbid": {"names": list(POLICY_FORBIDDEN)},
        }, indent=2), encoding="utf-8")
        self.status = self.root / "data" / "readiness.json"
        self.write_status(_status_doc())
        self.outbox = self.root / ".deploy" / "outbox"
        self.outbox.mkdir(parents=True)

    def write_status(self, doc, stale_sec=None):
        self.status.write_text(json.dumps(doc), encoding="utf-8")
        mtime = self.status.stat().st_mtime - (stale_sec or 0)
        os.utime(self.status, (mtime, mtime))

    def stage(self, files, **over):
        """Write a request document and return its path."""
        entries = []
        for entry in files:
            payload = entry.get("payload")
            if payload is None:
                raw = Path(str(entry["payload_path"])).read_bytes()
            else:
                raw = (payload if isinstance(payload, bytes)
                       else payload.encode("utf-8"))
            row = {"target": entry["target"], "sha256": sha256_bytes(raw)}
            if payload is None:
                row["payload_path"] = str(entry["payload_path"])
            else:
                row["content"] = (payload if isinstance(payload, str)
                                  else payload.decode("utf-8"))
            entries.append(row)
        req = {
            "schema_version": 1,
            "id": over.get("id", "req-test"),
            "purpose": over.get("purpose", "test-purpose"),
            "created_at": over.get(
                "created_at", datetime.now(timezone.utc).isoformat()),
            "files": entries,
        }
        for key in ("services", "verify_command"):
            if key in over:
                req[key] = over[key]
        path = self.outbox / f"{req['id']}{REQUEST_SUFFIX}"
        path.write_text(json.dumps(req), encoding="utf-8")
        return path, req

    def handle(self, path, **kw):
        opts = {"dry_run": False, "policy": self.policy,
                "allow_restart": False, "test_timeout": 20.0,
                "status_path": self.status}
        opts.update(kw)
        return handle_request(path, self.outbox, self.root, **opts)


class SchemaGateTest(ApplierTestBase):
    def test_well_formed_request_passes_every_gate(self):
        path, _ = self.stage([{"target": "engine/fix.py",
                               "payload": "print('hi')\n"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        target = self.root / "engine" / "fix.py"
        self.assertEqual(target.read_text(encoding="utf-8"), "print('hi')\n")

    def test_wrong_schema_version_is_refused(self):
        path, req = self.stage([{"target": "engine/x.py", "payload": "x"}])
        req["schema_version"] = 99
        path.write_text(json.dumps(req), encoding="utf-8")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("schema_version", receipt["reason"])
        self.assertFalse((self.root / "engine" / "x.py").exists())

    def test_missing_purpose_is_refused(self):
        path, _ = self.stage([{"target": "engine/x.py", "payload": "x"}],
                             purpose="")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("purpose", receipt["reason"])

    def test_empty_file_list_is_refused(self):
        with self.assertRaises(Refused):
            gate_schema({"schema_version": 1, "purpose": "p", "files": []})

    def test_declared_checksum_must_match_the_payload(self):
        path, req = self.stage([{"target": "engine/x.py", "payload": "good"}])
        req["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(req), encoding="utf-8")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("checksum mismatch", receipt["reason"])
        self.assertFalse((self.root / "engine" / "x.py").exists())

    def test_payload_is_read_from_payload_path_and_hashed(self):
        payload = self.root / "staged.bin"
        payload.write_bytes(b"BINARY\x00PAYLOAD")
        path, _ = self.stage(
            [{"target": "engine/blob.bin", "payload_path": payload}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "blob.bin").read_bytes(),
                         b"BINARY\x00PAYLOAD")


class PathGateTest(ApplierTestBase):
    def test_traversal_out_of_the_root_is_refused(self):
        for target in ("../outside.py", "engine/../../escape.py",
                       "sub/../../etc/hosts"):
            with self.subTest(target=target):
                with self.assertRaises(Refused) as ctx:
                    gate_paths([(target, b"x")], self.root)
                self.assertIn("escapes", str(ctx.exception))

    def test_an_absolute_target_is_refused_not_silently_rebased(self):
        """Absolute targets are refused by SHAPE, not merely by containment.

        ``Path(root) / "/etc/passwd"`` DISCARDS the root on POSIX, and
        ``Path(root) / "/tmp/x"`` joins under the root's DRIVE on Windows and
        lands back inside -- so a containment-only check would accept on
        Windows exactly the shape POSIX refuses. Deploy targets are always
        relative to the root, and this is the mistake an agent is most likely
        to make by pasting a host path.
        """
        absolute = ("C:/Windows/system32/drivers/etc/hosts"
                    if os.name == "nt" else "/etc/passwd")
        for target in (absolute, "/tmp/deploy-escape/x.py",
                       "//host/share/x.py"):
            with self.subTest(target=target):
                with self.assertRaises(Refused) as ctx:
                    gate_paths([(target, b"x")], self.root)
                self.assertIn("absolute path", str(ctx.exception))

    def test_a_relative_target_that_resolves_inside_is_accepted(self):
        """The shape check must not over-refuse legitimate relative paths."""
        targets = gate_paths([("engine/a.py", b"x"),
                              ("ops/sub/b.json", b"y"),
                              ("dashboard/c.py", b"z")], self.root)
        names = sorted(p.name for p in targets)
        self.assertEqual(names, ["a.py", "b.json", "c.py"])
        for target, resolved in zip(
                ("engine/a.py", "ops/sub/b.json"), targets):
            self.assertIn(str(self.root.resolve()), str(resolved),
                          f"{target} did not resolve under the root")

    def test_a_dotdot_that_stays_inside_is_canonicalised_and_accepted(self):
        """``..`` is judged by where it LANDS, not by its presence.

        ``engine/../engine/x.py`` resolves inside the root, so it is safe; the
        escaping forms (``../outside.py``, ``engine/../../escape.py``) are
        covered by the traversal test above. Canonicalising means the request
        cannot target two different-looking paths at one file and dodge the
        duplicate check either.
        """
        targets = gate_paths([("engine/../engine/x.py", b"z")], self.root)
        self.assertEqual(len(targets), 1)
        # Compare canonicalised: resolve() returns the 8.3 short name on some
        # Windows paths, so a raw equality check fails for a path that is
        # actually correct.
        self.assertEqual(os.path.normcase(str(targets[0])),
                         os.path.normcase(str((self.root / "engine" / "x.py")
                                              .resolve())))
        self.assertTrue(os.path.isdir(self.root / "engine"))
        with self.assertRaises(Refused) as ctx:
            gate_paths([("engine/x.py", b"a"),
                        ("engine/../engine/x.py", b"b")], self.root)
        self.assertIn("twice", str(ctx.exception))

    def test_a_sibling_directory_sharing_a_prefix_is_still_outside(self):
        """``/opt/x-evil`` is not inside ``/opt/x``: prefix checks lie."""
        with self.assertRaises(Refused):
            gate_paths([("../%s-evil/pwn.py" % self.root.name, b"x")],
                       self.root)

    def test_policy_refuse_list_blocks_declared_non_code_files(self):
        """A deployment declares which of its files are not code.

        The applier knows nothing about them; the policy does. These are the
        runtime-state and owner files a deployment must never let an agent
        replace.
        """
        for name in POLICY_FORBIDDEN:
            with self.subTest(name=name):
                with self.assertRaises(Refused) as ctx:
                    gate_paths([(f"data/{name}", b"x")], self.root,
                               self.policy)
                self.assertIn("refuse list", str(ctx.exception))

    def test_secrets_and_env_files_are_refused_by_any_policy(self):
        """Generic refusals hold even with an EMPTY policy.

        A policy can add to the refuse list, never subtract: forgetting to
        name ``.env`` in a policy must not make it writable.
        """
        bare = Policy({}, name="empty")
        for name in (".env", ".env.production", ".env.local",
                     "server.pem", "signing.key", "bundle.p12"):
            with self.subTest(name=name):
                with self.assertRaises(Refused) as ctx:
                    gate_paths([(f"config/{name}", b"x")], self.root, bare)
                self.assertIn("refuse list", str(ctx.exception))

    def test_the_readiness_status_file_is_itself_unwritable(self):
        """The file the gate reads must not be a deploy target.

        Otherwise a request could rewrite its own verdict: write
        ``state: ready``, then pass the gate it just satisfied.
        """
        with self.assertRaises(Refused) as ctx:
            gate_paths([("data/readiness.json", b'{"state": "ready"}')],
                       self.root, self.policy)
        self.assertIn("refuse list", str(ctx.exception))

    def test_git_metadata_is_never_writable(self):
        with self.assertRaises(Refused) as ctx:
            gate_paths([(".git/config", b"x")], self.root)
        self.assertIn("forbidden directory", str(ctx.exception))

    def test_symlinked_target_is_refused_rather_than_followed(self):
        outside = self.root.parent / "secret_target.txt"
        link = self.root / "engine" / "link.py"
        try:
            outside.write_text("host file", encoding="utf-8")
            os.symlink(str(outside), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable here: {exc}")
        try:
            with self.assertRaises(Refused) as ctx:
                gate_paths([("engine/link.py", b"x")], self.root)
            self.assertIn("symlink", str(ctx.exception))
            self.assertEqual(outside.read_text(encoding="utf-8"), "host file")
        finally:
            link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)

    def test_a_symlink_inside_the_root_is_also_refused(self):
        """The case the containment check cannot see, and nearly did not.

        A link pointing at another file INSIDE the root resolves to a path that
        is perfectly well contained, so it sails through every gate the
        containment check owns -- and the write lands on a file the request
        never named. Found running the suite on Linux: the old code resolved the
        path FIRST and then asked ``is_symlink()`` of the result, which resolve()
        had already dereferenced, so that guard could never fire. The refusal
        now happens on the path as written.
        """
        real = self.root / "engine" / "collector.py"
        real.write_text("# the real file\n", encoding="utf-8")
        link = self.root / "engine" / "rename_me.py"
        try:
            os.symlink(str(real), str(link))
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable here: {exc}")
        try:
            with self.assertRaises(Refused) as ctx:
                gate_paths([("engine/rename_me.py", b"# payload\n")], self.root)
            self.assertIn("symlink", str(ctx.exception))
            self.assertEqual(real.read_text(encoding="utf-8"),
                             "# the real file\n",
                             "a payload written through a link rewrote a file "
                             "the request did not name")
        finally:
            link.unlink(missing_ok=True)

    def test_duplicate_targets_are_refused(self):
        with self.assertRaises(Refused) as ctx:
            gate_paths([("engine/a.py", b"x"), ("engine/a.py", b"y")],
                       self.root)
        self.assertIn("twice", str(ctx.exception))

    def test_new_subdirectories_are_created_for_a_legal_target(self):
        path, _ = self.stage([{"target": "engine/new/pkg/mod.py",
                               "payload": "ok"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertTrue((self.root / "engine" / "new" / "pkg"
                         / "mod.py").is_file())


class ReadinessGateTest(ApplierTestBase):
    def test_an_unsafe_state_refuses_the_deploy(self):
        for state in ("busy", "draining", "degraded", "starting",
                      "mismatch", "unknown"):
            with self.subTest(state=state):
                self.write_status(_status_doc(state=state))
                path, _ = self.stage([{"target": "engine/x.py",
                                       "payload": "x"}])
                receipt = self.handle(path)
                self.assertEqual(receipt["status"], "refused")
                self.assertIn(state, receipt["reason"])
                self.assertFalse((self.root / "engine" / "x.py").exists())

    def test_every_declared_safe_state_is_accepted(self):
        for state in SAFE_STATES:
            with self.subTest(state=state):
                self.write_status(_status_doc(state=state))
                path, _ = self.stage([{"target": f"engine/{state}.py",
                                       "payload": "y"}], id=f"req-{state}")
                self.assertEqual(self.handle(path)["status"], "applied")

    def test_stale_status_is_refused_even_when_it_declares_safe(self):
        """An old ``ready`` is not evidence the system is ready NOW.

        Silence is not agreement: a process that died an hour ago leaves a
        status file that still says safe.
        """
        self.write_status(_status_doc(state="ready"), stale_sec=3600.0)
        path, _ = self.stage([{"target": "engine/z.py", "payload": "z"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("old", receipt["reason"])
        self.assertFalse((self.root / "engine" / "z.py").exists())

    def test_a_policy_status_age_limit_is_honoured(self):
        """The limit comes from the policy, not from a hardcoded default."""
        strict = make_policy(readiness={
            "status_file": "data/readiness.json", "field": "state",
            "label": "readiness", "safe_states": list(SAFE_STATES),
            "max_age_sec": 30})
        self.write_status(_status_doc(state="ready"), stale_sec=120.0)
        path, _ = self.stage([{"target": "engine/strict.py",
                               "payload": "s"}], id="req-strict")
        receipt = self.handle(path, policy=strict)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("readiness status is", receipt["reason"])

    def test_missing_status_file_refuses_instead_of_assuming_safe(self):
        self.status.unlink()
        path, _ = self.stage([{"target": "engine/w.py", "payload": "w"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("no readiness status file", receipt["reason"])

    def test_unreadable_status_is_refused(self):
        self.status.write_text("{not json", encoding="utf-8")
        path, _ = self.stage([{"target": "engine/v.py", "payload": "v"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("unreadable", receipt["reason"])

    def test_an_empty_state_string_is_not_a_safe_state(self):
        self.write_status(_status_doc(state=""))
        path, _ = self.stage([{"target": "engine/u.py", "payload": "u"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")

    def test_the_status_field_name_comes_from_the_policy(self):
        """A deployment may call its field something other than ``state``."""
        alt = make_policy(readiness={
            "status_file": "data/health.json", "field": "phase",
            "label": "health", "safe_states": ["serving"],
            "max_age_sec": 900})
        health = self.root / "data" / "health.json"
        health.write_text(json.dumps({"phase": "serving"}), encoding="utf-8")
        path, _ = self.stage([{"target": "engine/alt.py", "payload": "a"}],
                             id="req-alt")
        receipt = self.handle(path, policy=alt, status_path=health)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        health.write_text(json.dumps({"phase": "draining"}), encoding="utf-8")
        path2, _ = self.stage([{"target": "engine/alt2.py", "payload": "b"}],
                              id="req-alt2")
        refused = self.handle(path2, policy=alt, status_path=health)
        self.assertEqual(refused["status"], "refused")
        self.assertIn("draining", refused["reason"])

    def test_a_policy_with_no_status_file_records_the_gate_as_skipped(self):
        """Skipping must be REPORTED, never implied as a pass."""
        bare = Policy({}, name="empty")
        path, _ = self.stage([{"target": "engine/noGate.py",
                               "payload": "n"}], id="req-nogate")
        receipt = self.handle(path, policy=bare)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertTrue(receipt["broker"].get("skipped"), receipt["broker"])

    def test_safe_states_are_configurable_per_request_handling(self):
        self.write_status(_status_doc(state="draining"))
        path, _ = self.stage([{"target": "engine/t.py", "payload": "t"}])
        permissive = make_policy(
            readiness={"status_file": "data/readiness.json",
                       "field": "state", "label": "readiness",
                       "safe_states": ["ready", "draining"],
                       "max_age_sec": 900})
        receipt = self.handle(path, policy=permissive)
        self.assertEqual(receipt["status"], "applied")


class VerdictStampTest(ApplierTestBase):
    """Freshness measured from the VERDICT, not only from the file's mtime.

    Found on a live rehearsal against the trading host (2026-10-31): a status
    file written seconds ago, declaring a verdict from 90 minutes earlier,
    sailed through the gate as "all gates passed". mtime is reset by a copy, a
    restore, an rsync or a plain ``touch`` -- all routine on a deploy host -- so
    on its own it measures when the FILE arrived, not how old the evidence
    inside it is. A policy can now name the document's own timestamp field, and
    then BOTH ages have to be inside the limit.

    A declared field that is missing or unparsable REFUSES rather than falling
    back to mtime. That is the whole point: a policy asking for dated verdicts
    and getting an undatable one has no evidence, and quietly passing would
    leave a freshness line in the receipt that never actually ran.
    """

    def _policy(self, **readiness):
        base = {"status_file": "data/readiness.json", "field": "state",
                "label": "readiness", "safe_states": list(SAFE_STATES),
                "max_age_sec": 900}
        base.update(readiness)
        return make_policy(readiness=base)

    def test_a_fresh_file_carrying_a_stale_verdict_is_refused(self):
        policy = self._policy(timestamp_field="checked_at")
        old = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=old))
        path, _ = self.stage([{"target": "engine/stale_verdict.py",
                               "payload": "x"}], id="req-stale-verdict")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "refused")
        self.assertIn("verdict is 90", receipt["reason"])
        self.assertIn("copied, restored or touched", receipt["reason"],
                      "the reason must explain why a fresh file was not enough")
        self.assertFalse((self.root / "engine" / "stale_verdict.py").exists())

    def test_a_declared_but_missing_stamp_refuses_instead_of_falling_back(self):
        """No stamp is not 'not stale'; it is 'cannot be dated'."""
        policy = self._policy(timestamp_field="checked_at")
        self.write_status(_status_doc(state="ready"))
        path, _ = self.stage([{"target": "engine/no_stamp.py",
                               "payload": "x"}], id="req-no-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "refused")
        self.assertIn("no parsable 'checked_at'", receipt["reason"])
        self.assertFalse((self.root / "engine" / "no_stamp.py").exists())

    def test_an_unparsable_stamp_refuses(self):
        policy = self._policy(timestamp_field="checked_at")
        self.write_status(_status_doc(state="ready", checked_at="an hour ago"))
        path, _ = self.stage([{"target": "engine/bad_stamp.py",
                               "payload": "x"}], id="req-bad-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "refused")
        self.assertIn("an hour ago", receipt["reason"])

    def test_a_stamp_from_the_future_refuses(self):
        """Two clocks disagreeing means the age proves nothing."""
        policy = self._policy(timestamp_field="checked_at")
        ahead = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=ahead))
        path, _ = self.stage([{"target": "engine/future_stamp.py",
                               "payload": "x"}], id="req-future-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "refused")
        self.assertIn("FUTURE", receipt["reason"])

    def test_a_current_stamp_and_a_current_file_pass_and_report_both(self):
        policy = self._policy(timestamp_field="checked_at")
        now = datetime.now(timezone.utc).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=now))
        path, _ = self.stage([{"target": "engine/fresh.py", "payload": "x"}],
                             id="req-fresh-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        broker = receipt["broker"]
        self.assertEqual(broker["stamp_field"], "checked_at")
        self.assertIsNotNone(broker["verdict_age_sec"],
                             "the receipt must show the age it measured")
        self.assertLess(broker["verdict_age_sec"], 60)

    def test_mtime_semantics_are_unchanged_when_no_stamp_field_is_declared(self):
        """Back-compat: an undeclared stamp is not enforced, and says so."""
        policy = self._policy()          # no timestamp_field
        old = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=old))
        path, _ = self.stage([{"target": "engine/undeclared.py",
                               "payload": "x"}], id="req-undeclared-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIsNone(receipt["broker"]["stamp_field"])
        self.assertIsNone(receipt["broker"]["verdict_age_sec"],
                          "claiming an age it never measured would be a lie in "
                          "the receipt")

    def test_a_naive_stamp_is_read_as_utc_not_as_host_local(self):
        """Guessing local time could turn a stale verdict into a fresh one."""
        policy = self._policy(timestamp_field="checked_at")
        naive_old = (datetime.now(timezone.utc)
                     - timedelta(minutes=90)).replace(tzinfo=None).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=naive_old))
        path, _ = self.stage([{"target": "engine/naive_stamp.py",
                               "payload": "x"}], id="req-naive-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "refused")
        self.assertIn("verdict is", receipt["reason"])


class AgeGateTest(ApplierTestBase):
    def test_a_stale_request_is_refused(self):
        old = (datetime.now(timezone.utc)
               - timedelta(minutes=40)).isoformat()
        path, _ = self.stage([{"target": "engine/a.py", "payload": "a"}],
                             created_at=old)
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("stale", receipt["reason"])
        self.assertFalse((self.root / "engine" / "a.py").exists())

    def test_a_fresh_request_is_accepted(self):
        path, _ = self.stage([{"target": "engine/b.py", "payload": "b"}])
        self.assertEqual(self.handle(path)["status"], "applied")

    def test_an_unparseable_stamp_falls_back_to_the_file_mtime(self):
        path, _ = self.stage([{"target": "engine/c.py", "payload": "c"}],
                             created_at="not-a-date")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))

    def test_a_very_old_file_mtime_is_refused_when_no_stamp(self):
        path, req = self.stage([{"target": "engine/d.py", "payload": "d"}])
        del req["created_at"]
        path.write_text(json.dumps(req), encoding="utf-8")
        old = time.time() - 7200
        os.utime(path, (old, old))
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("stale", receipt["reason"])


class ApplySemanticsTest(ApplierTestBase):
    def test_verification_failure_restores_the_originals(self):
        target = self.root / "engine" / "keep.py"
        target.write_text("original\n", encoding="utf-8")
        path, _ = self.stage([{"target": "engine/keep.py",
                               "payload": "replacement\n"}],
                             verify_command="exit 7")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("verification failed", receipt["reason"])
        self.assertEqual(target.read_text(encoding="utf-8"), "original\n",
                         "a failed verification must not leave the change in")

    def test_a_verifier_that_could_not_run_is_not_reported_as_test_failure(self):
        """127/126 mean the shell never reached the verifier.

        Live 2026-10-30: a verify_command naming a Windows-pathed interpreter
        came back rc=127 under ``bash -lc`` ("No such file or directory"), and
        the receipt said "verification failed" -- which reads as a broken test
        suite and sends the next person debugging code that was never run. The
        file is still restored, but the reason must name the real fault.
        """
        target = self.root / "engine" / "v127.py"
        target.write_text("original\n", encoding="utf-8")
        path, _ = self.stage([{"target": "engine/v127.py",
                               "payload": "replacement\n"}],
                             verify_command="nonexistent-verifier-xyz --all")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["verify_rc"], 127)
        self.assertIn("could not be executed", receipt["reason"])
        self.assertNotIn("verification failed (rc=127)", receipt["reason"],
                         "must not read like a failing test suite")
        self.assertEqual(receipt["verify_command"],
                         "nonexistent-verifier-xyz --all")
        self.assertEqual(target.read_text(encoding="utf-8"), "original\n",
                         "an unrun verifier must still roll the file back")

    def test_a_verifier_that_never_ran_cannot_report_success_through_a_pipe(self):
        """`missing_cmd | tail` exits with tail's 0 -- a green light from nothing.

        Proved in the sandbox dry-run: the receipt read ``verify_rc: 0`` beside
        "No such file or directory", so a deploy was applied on the strength of a
        verifier that was never executed. The applier therefore runs every
        verifier with pipefail, making the missing command fatal.
        """
        path, _ = self.stage([{"target": "engine/pipe.py", "payload": "p\n"}],
                             verify_command="nonexistent-verifier-xyz | tail -1")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "failed",
                         "a verifier that never ran must not pass the gate")
        self.assertEqual(receipt["verify_rc"], 127)
        self.assertIn("could not be executed", receipt["reason"])
        self.assertFalse((self.root / "engine" / "pipe.py").exists(),
                         "nothing may be applied on an unverified change")

    def test_successful_verification_keeps_the_change(self):
        path, _ = self.stage([{"target": "engine/ok.py", "payload": "new\n"}],
                             verify_command="exit 0")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual(receipt["verify_rc"], 0)
        self.assertEqual((self.root / "engine" / "ok.py").read_text(
            encoding="utf-8"), "new\n")

    def test_existing_file_mode_is_preserved(self):
        if os.name == "nt":
            self.skipTest("Windows has no POSIX mode bits to preserve")
        target = self.root / "engine" / "mode.py"
        target.write_text("old", encoding="utf-8")
        os.chmod(target, 0o750)
        path, _ = self.stage([{"target": "engine/mode.py", "payload": "new"}])
        self.handle(path)
        self.assertEqual(os.stat(target).st_mode & 0o777, 0o750)

    def test_existing_file_owner_is_preserved(self):
        """A root-run applier must not flip deployed code to root-owned.

        Hosts with systemd and no cron run this unit as root (it needs docker
        for restarts), and a naive write leaves every applied file root:root.
        Mixed ownership is what made the outbox readable-but-unwritable to its
        next non-root caller, so ownership is carried across deliberately.

        Checked by recording the chown calls rather than asserting real uids,
        so the test is meaningful without root on any platform.
        """
        target = self.root / "engine" / "owned.py"
        target.write_text("old", encoding="utf-8")
        st = os.stat(target)
        calls = []
        real_chown = getattr(os, "chown", None)
        if real_chown is None:
            self.skipTest("this platform has no os.chown")
        with mock.patch("deploy_applier.os.chown",
                        side_effect=lambda p, uid, gid: calls.append((uid, gid))):
            path, _ = self.stage([{"target": "engine/owned.py",
                                   "payload": "new"}])
            self.assertEqual(self.handle(path)["status"], "applied")
        self.assertIn((st.st_uid, st.st_gid), calls,
                      f"owner {st.st_uid}:{st.st_gid} was not re-applied; "
                      f"chown calls: {calls}")
        self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_a_new_file_inherits_the_parent_directory_owner(self):
        """So a deploy never drops a lone root-owned file into a user tree."""
        st = os.stat(self.root / "engine")
        calls = []
        if getattr(os, "chown", None) is None:
            self.skipTest("this platform has no os.chown")
        with mock.patch("deploy_applier.os.chown",
                        side_effect=lambda p, uid, gid: calls.append((uid, gid))):
            path, _ = self.stage([{"target": "engine/brand_new.py",
                                   "payload": "new"}])
            self.assertEqual(self.handle(path)["status"], "applied")
        self.assertIn((st.st_uid, st.st_gid), calls, calls)

    def test_a_refused_chown_does_not_fail_an_already_successful_write(self):
        """The write is the deliverable; ownership is best-effort."""
        if getattr(os, "chown", None) is None:
            self.skipTest("this platform has no os.chown")

        def deny(path, uid, gid):
            raise PermissionError(1, "not permitted to chown")

        with mock.patch("deploy_applier.os.chown", side_effect=deny):
            path, _ = self.stage([{"target": "engine/nochown.py",
                                   "payload": "written\n"}])
            receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "nochown.py").read_text(
            encoding="utf-8"), "written\n")

    def test_no_tmp_file_is_left_behind(self):
        path, _ = self.stage([{"target": "engine/clean.py", "payload": "x"}])
        self.handle(path)
        self.assertEqual(list((self.root / "engine").glob("*.deploy-tmp")), [])

    def test_applied_request_is_moved_out_of_the_outbox(self):
        """The same request must never be applicable twice."""
        path, _ = self.stage([{"target": "engine/once.py", "payload": "x"}])
        self.assertEqual(self.handle(path)["status"], "applied")
        self.assertFalse(path.exists())
        self.assertEqual(pending_requests(self.outbox), [])
        self.assertTrue((self.outbox / "applied" / path.name).exists())

    def test_backups_are_written_before_payloads(self):
        target = self.root / "engine" / "b.py"
        target.write_text("before", encoding="utf-8")
        path, _ = self.stage([{"target": "engine/b.py", "payload": "after"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied")
        backups = list((self.outbox / "backups" / "req-test").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "before")

    def test_a_new_file_is_recorded_as_created_not_backed_up(self):
        path, _ = self.stage([{"target": "engine/newfile.py",
                               "payload": "brand new"}])
        receipt = self.handle(path)
        row = receipt["backups"][0]
        self.assertEqual(row["created"], "true")
        self.assertEqual(row["backup"], "")

    def test_services_are_not_restarted_without_permission(self):
        path, _ = self.stage([{"target": "engine/s.py", "payload": "x"}],
                             services=["market-collector"])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied")
        self.assertIn("restart_skipped", receipt)
        self.assertNotIn("restart", receipt)

    def test_dry_run_passes_gates_and_writes_nothing(self):
        path, _ = self.stage([{"target": "engine/dry.py", "payload": "x"}])
        receipt = self.handle(path, dry_run=True)
        self.assertEqual(receipt["status"], "dry_run")
        self.assertFalse((self.root / "engine" / "dry.py").exists())
        self.assertTrue(path.exists(), "a dry run must not consume the request")
        self.assertEqual(len(receipt["files"]), 1)

    def test_dry_run_still_refuses_an_unsafe_market(self):
        self.write_status(_status_doc(state="blocked"))
        path, _ = self.stage([{"target": "engine/dry2.py", "payload": "x"}])
        receipt = self.handle(path, dry_run=True)
        self.assertEqual(receipt["status"], "refused",
                         "a dry run must not report gates it did not pass")

    def test_receipt_lists_targets_and_a_truncated_hash(self):
        path, _ = self.stage([{"target": "engine/r.py", "payload": "abc"}])
        receipt = self.handle(path)
        row = receipt["files"][0]
        self.assertEqual(row["bytes"], 3)
        self.assertEqual(row["sha256"], sha256_bytes(b"abc")[:16])

    def test_unreadable_request_is_refused_not_crashed_on(self):
        path = self.outbox / f"broken{REQUEST_SUFFIX}"
        path.write_text("{not json", encoding="utf-8")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("not readable JSON", receipt["reason"])


class LockAndCliTest(ApplierTestBase):
    def test_a_held_lock_refuses_a_second_applier(self):
        with ApplierLock(self.outbox):
            with self.assertRaises(Refused) as ctx:
                with ApplierLock(self.outbox):
                    pass
            self.assertIn("another applier", str(ctx.exception))

    def test_a_stale_lock_is_taken_over(self):
        lock = self.outbox / ".applier.lock"
        lock.write_text("{}", encoding="utf-8")
        old = time.time() - 7200
        os.utime(lock, (old, old))
        with ApplierLock(self.outbox) as held:
            self.assertTrue(held.path.exists())

    def test_lock_is_released_after_use(self):
        with ApplierLock(self.outbox):
            pass
        self.assertFalse((self.outbox / ".applier.lock").exists())

    def test_pending_requests_ignores_receipts_and_dotfiles(self):
        (self.outbox / "a.deploy.json").write_text("{}", encoding="utf-8")
        (self.outbox / "a.receipt.json").write_text("{}", encoding="utf-8")
        (self.outbox / ".hidden.deploy.json").write_text("{}",
                                                         encoding="utf-8")
        self.assertEqual([p.name for p in pending_requests(self.outbox)],
                         ["a.deploy.json"])

    def test_cli_dry_run_writes_a_receipt_and_changes_nothing(self):
        path, _ = self.stage([{"target": "engine/cli.py", "payload": "x"}],
                             id="cli-req")
        rc = main(["--root", str(self.root), "--outbox", str(self.outbox),
                   "--dry-run", "--once", "--status-path", str(self.status)])
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "engine" / "cli.py").exists())
        # The receipt name strips the FULL request suffix, not just ".json":
        # "cli-req.deploy.json" -> "cli-req.receipt.json". Deriving it from
        # Path.stem produced "cli-req.deploy.receipt.json" while the agent-side
        # tools looked for the shorter name, so every verdict came back as a
        # generic ok/failed and a refused deploy read as a success.
        receipt = self.outbox / "cli-req.receipt.json"
        self.assertEqual(_receipt_name(path), "cli-req.receipt.json")
        self.assertTrue(receipt.exists(), f"missing receipt for {path.name}")
        self.assertEqual(
            list(self.outbox.glob("cli-req.deploy.receipt.json")), [],
            "the legacy mis-derived receipt name must not reappear")
        doc = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "dry_run")
        self.assertTrue(doc["dry_run"])
        self.assertEqual(pending_requests(self.outbox), [path],
                         "a dry run must leave the request pending")

    def test_cli_reports_rc2_when_a_request_is_refused(self):
        self.write_status(_status_doc(state="busy"))
        self.stage([{"target": "engine/cli2.py", "payload": "x"}],
                   id="cli-refused")
        rc = main(["--root", str(self.root), "--outbox", str(self.outbox),
                   "--policy", str(self.policy_path), "--once",
                   "--status-path", str(self.status)])
        self.assertEqual(rc, 2)
        self.assertFalse((self.root / "engine" / "cli2.py").exists())

    def test_cli_rc3_when_the_root_does_not_exist(self):
        self.assertEqual(
            main(["--root", str(self.root / "nowhere"), "--once"]), 3)

    def test_cli_with_no_pending_requests_is_a_clean_zero(self):
        rc = main(["--root", str(self.root), "--outbox", str(self.outbox),
                   "--once", "--status-path", str(self.status)])
        self.assertEqual(rc, 0)


class OutboxWritabilityTest(ApplierTestBase):
    """The receipt IS the audit trail, so an unwritable outbox must stop
    everything before a payload is written.

    Live host prep 2026-10-30: the outbox was created as an intermediate of
    ``install -d .../outbox/applied`` and GNU install leaves intermediates at
    root:root 755, so the ubuntu-run applier could read requests but not write
    receipts. Nothing caught it: a dry run used a separate writable directory,
    and an empty outbox returns before any write. The first real request would
    have been applied with no receipt and no rollback record.
    """

    def test_a_writable_outbox_passes_and_leaves_no_probe_behind(self):
        check_outbox_writable(self.outbox)
        self.assertEqual(list(self.outbox.glob(".write-probe")), [],
                         "the probe file must be removed")

    def test_an_outbox_that_cannot_be_created_refuses(self):
        """Portable failure: the parent is a FILE, so mkdir cannot succeed.

        Raises DeployError, NOT Refused: a misconfigured installation is a
        different thing from a gate rejecting one request, and the CLI encodes
        that as rc=3 rather than rc=2.
        """
        blocker = self.root / "i-am-a-file"
        blocker.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(DeployError) as ctx:
            check_outbox_writable(blocker / "outbox")
        self.assertNotIsInstance(ctx.exception, Refused,
                                 "an installation fault is not a gate verdict")
        self.assertIn("not usable", str(ctx.exception))

    @unittest.skipIf(os.name == "nt",
                     "Windows ignores POSIX permission bits")
    def test_an_existing_but_unwritable_outbox_refuses(self):
        outbox = self.root / "readonly-outbox"
        outbox.mkdir()
        os.chmod(outbox, 0o500)
        self.addCleanup(lambda: os.chmod(outbox, 0o700))
        with self.assertRaises(DeployError) as ctx:
            check_outbox_writable(outbox)
        msg = str(ctx.exception)
        self.assertIn("not writable by uid", msg)
        self.assertIn("receipt", msg, "the reason must say WHY this matters")
        self.assertIn("chown", msg, "the reason must name the fix")

    def test_the_cli_exits_3_on_an_unwritable_outbox(self):
        blocker = self.root / "file2"
        blocker.write_text("x", encoding="utf-8")
        rc = main(["--root", str(self.root), "--outbox",
                   str(blocker / "outbox"), "--once",
                   "--status-path", str(self.status)])
        self.assertEqual(rc, 3, "a broken installation must not look clean")

    def test_a_unwritable_outbox_blocks_before_any_payload_is_written(self):
        """The ordering guarantee that makes the preflight worth having.

        Writing the file first and discovering the receipt cannot be written
        would leave an applied change with no audit trail -- strictly worse
        than refusing.
        """
        target = self.root / "driver" / "must_not_appear.py"
        target.parent.mkdir()
        path, _ = self.stage([{"target": "driver/must_not_appear.py",
                               "payload": "x"}], id="req-unwritable")
        with mock.patch("deploy_applier.check_outbox_writable",
                        side_effect=DeployError(
                            "outbox is not writable by uid 1")):
            rc = main(["--root", str(self.root), "--outbox",
                       str(self.outbox), "--once",
                       "--status-path", str(self.status)])
        self.assertEqual(rc, 3)
        self.assertFalse(target.exists(),
                         "no payload may be written when no receipt can be")
        self.assertTrue(path.exists(), "the request must stay pending")

    def test_refusal_is_a_deploy_error_not_a_bare_oserror(self):
        """Callers catch DeployError; an escaped OSError would crash the pass."""
        blocker = self.root / "file3"
        blocker.write_text("x", encoding="utf-8")
        with self.assertRaises(DeployError):
            check_outbox_writable(blocker / "outbox")


if __name__ == "__main__":
    unittest.main()
