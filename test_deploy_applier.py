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
    MAX_FAILED_ATTEMPTS,
    ApplierLock,
    DeployError,
    PendingWindow,
    Policy,
    Refused,
    _quarantine_refused,
    _receipt_name,
    check_outbox_writable,
    gate_paths,
    gate_readiness,
    gate_schema,
    handle_request,
    main,
    pending_requests,
    readiness_of,
    run_once,
    sha256_bytes,
    READINESS_KEYS,
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


#: Fake "deployment" for the typed-action tests: a JSON file of instance rows
#: plus three tiny scripts standing in for the deployment's own read, upsert and
#: validate commands. Kept as argv + ``python -c`` on purpose -- that is exactly
#: how a real policy names a command, so the test exercises the same substitution
#: path (and would catch a template that only works with a shell).
_READ_SRC = (
    "import sys;"
    "print(open(sys.argv[1], encoding='utf-8').read())"
)
_UPSERT_SRC = (
    "import sys, json;"
    "p, row = sys.argv[1], json.loads(sys.argv[2]);"
    "doc = json.load(open(p, encoding='utf-8'));"
    "doc['instances'] = [row if i.get('instance_id') == row.get('instance_id')"
    " else i for i in doc['instances']];"
    "open(p, 'w', encoding='utf-8').write(json.dumps(doc));"
    "print('upserted')"
)
#: Succeeds and writes NOTHING -- the shape of a write path that reports success
#: without changing anything, which the re-read check must catch.
_UPSERT_NOOP_SRC = "print('pretended to upsert')"
_VALIDATE_SRC = (
    "import sys;"
    "ok = sys.argv[1] in ('', 'aggressive', 'balanced', 'safe');"
    "print('profile' + (' accepted' if ok else ' rejected'));"
    "sys.exit(0 if ok else 1)"
)


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
        """Write a request document and return its path.

        An entry may carry ``payload`` (contents are known here), or
        ``payload_path`` (read to derive the checksum), or BOTH ``payload_path``
        and ``sha256``/``as_recorded`` -- the last form exists because a real
        request sometimes names a payload the TEST cannot read either: one staged
        through another mount view, or a traversal string. Without it those
        cases, which are the interesting ones, cannot be expressed.
        """
        entries = []
        for entry in files:
            payload = entry.get("payload")
            if payload is None and "sha256" not in entry:
                raw = Path(str(entry["payload_path"])).read_bytes()
                digest = sha256_bytes(raw)
            elif payload is None:
                digest = entry["sha256"]
            else:
                raw = (payload if isinstance(payload, bytes)
                       else payload.encode("utf-8"))
                digest = sha256_bytes(raw)
            row = {"target": entry["target"], "sha256": digest}
            if payload is None:
                row["payload_path"] = str(entry.get("as_recorded")
                                          or entry["payload_path"])
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


class TypedActionTest(ApplierTestBase):
    """A request may ask for a TYPED ACTION instead of a file deploy.

    Why this exists at all: runtime state (an instance's selected risk profile)
    lives in a file that belongs to the RUNNING service, which caches it. The
    applier is forbidden to write ``data/`` -- correctly -- so the only safe way
    to change that state is to ask the service to change it, which is what an
    action is.

    What these tests are really pinning is BOUNDEDNESS. An action must not
    become a general remote-exec: it can only run a command the policy named, on
    one existing instance, setting one of the fields the policy exposed. And it
    must REFUSE when no policy declares any action at all, so shipping this
    machinery changes nothing until a deployment opts in.
    """

    ROW = {"instance_id": "bei-spx", "strategy_key": "s", "instrument": "SPX",
           "params": {}, "risk_profile": None, "contracts": 3, "paused": False}

    def setUp(self):
        super().setUp()
        self.state = self.root / "state.json"
        self.state.write_text(json.dumps({"instances": [dict(self.ROW)]}),
                              encoding="utf-8")
        self.policy = self.action_policy()

    def action_policy(self, **over):
        actions = {
            "container": "",
            "set_instance_field": {
                "allowed_fields": ["risk_profile"],
                "read_argv": [sys.executable, "-c", _READ_SRC, str(self.state)],
                "apply_argv": [sys.executable, "-c", _UPSERT_SRC,
                               str(self.state), "{json}"],
                "validate_argv": [sys.executable, "-c", _VALIDATE_SRC,
                                  "{profile}"],
                "run_timeout_sec": 30,
            },
        }
        actions.update(over.pop("actions", {}) or {})
        return make_policy(actions=actions, **over)

    def stage_action(self, action, **over):
        req = {
            "schema_version": 1,
            "id": over.get("id", "req-act"),
            "purpose": over.get("purpose", "test action"),
            "created_at": over.get("created_at",
                                   datetime.now(timezone.utc).isoformat()),
            "action": action,
        }
        path = self.outbox / f"{req['id']}{REQUEST_SUFFIX}"
        path.write_text(json.dumps(req), encoding="utf-8")
        return path

    def rows(self):
        return json.loads(self.state.read_text(encoding="utf-8"))["instances"]

    # ---- the safe default ---------------------------------------------------
    def test_with_no_actions_declared_an_action_request_is_refused(self):
        """Shipping the machinery must not grant anything by itself."""
        self.policy = make_policy()          # the base policy: no actions block
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("not permitted by this deployment's policy",
                      receipt["reason"])
        self.assertIsNone(self.rows()[0]["risk_profile"])

    # ---- boundedness --------------------------------------------------------
    def test_an_unknown_action_kind_is_refused(self):
        path = self.stage_action({"kind": "run_anything",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("not permitted", receipt["reason"])

    def test_a_field_the_policy_does_not_expose_is_refused(self):
        """The allow-list is the difference between a field and a shell."""
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"contracts": 99}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("does not expose", receipt["reason"])
        self.assertEqual(self.rows()[0]["contracts"], 3)

    def test_a_nested_value_is_refused(self):
        """One field must mean one scalar, not a whole sub-structure."""
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": {"x": 1}}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("must be a scalar", receipt["reason"])

    def test_an_injection_style_instance_id_is_refused(self):
        for bogus in ("../../etc", "bei spx", "a;rm -rf /", ""):
            with self.subTest(instance_id=bogus):
                path = self.stage_action(
                    {"kind": "set_instance_field", "instance_id": bogus,
                     "set": {"risk_profile": "balanced"}},
                    id=f"req-inj-{abs(hash(bogus)) % 10000}")
                receipt = self.handle(path)
                self.assertEqual(receipt["status"], "refused", bogus)

    def test_the_identifier_check_is_what_refuses_it_not_the_lookup(self):
        """Isolate the guard: the id must EXIST and still be refused.

        The obvious version of this test proves nothing, and the mutation run
        showed it: an id like ``../../etc`` is refused anyway because no such
        instance exists, so removing the identifier check left the suite green.
        Seeding a deployment whose own row carries a hostile id removes that
        alibi -- now only the check can refuse it.
        """
        self.state.write_text(json.dumps({"instances": [
            {"instance_id": "../../evil", "risk_profile": None,
             "contracts": 1, "params": {}, "paused": False},
        ]}), encoding="utf-8")
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "../../evil",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("plain identifier", receipt["reason"],
                      "refused for the wrong reason: the identifier check "
                      "must reject it, not the existence lookup")
        self.assertIsNone(self.rows()[0]["risk_profile"])

    def test_an_instance_that_does_not_exist_is_refused_not_created(self):
        """Changing a book is a smaller decision than opening one."""
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "ghost-book",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("does not exist", receipt["reason"])
        self.assertEqual(len(self.rows()), 1,
                         "the action created an instance")

    def test_a_value_the_deployment_rejects_is_refused(self):
        """'balance' is a typo, and must not be installed."""
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balance"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("rejected risk_profile", receipt["reason"])
        self.assertIsNone(self.rows()[0]["risk_profile"])

    # ---- the happy path, and what it must preserve --------------------------
    def test_a_valid_action_is_applied_and_verified_by_re_read(self):
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual(receipt["before"], {"risk_profile": None})
        self.assertEqual(receipt["after"], {"risk_profile": "balanced"})
        self.assertEqual(self.rows()[0]["risk_profile"], "balanced")
        self.assertIn("verified", receipt["verification"])

    def test_the_merge_preserves_every_field_the_request_did_not_name(self):
        """The deployment's write path REPLACES the whole row.

        Sending only the changed field would reset contracts/params/paused to
        their defaults -- the most likely way this feature could lose an
        operator's configuration. Read-modify-write is the fix, and this is the
        assertion that keeps it.
        """
        self.handle(self.stage_action(
            {"kind": "set_instance_field", "instance_id": "bei-spx",
             "set": {"risk_profile": "safe"}}))
        row = self.rows()[0]
        self.assertEqual(row["risk_profile"], "safe")
        for field, value in (("contracts", 3), ("paused", False),
                             ("params", {}), ("instrument", "SPX"),
                             ("strategy_key", "s")):
            self.assertEqual(row[field], value,
                             f"{field} was reset by the upsert")

    def test_a_write_that_did_not_take_is_reported_not_believed(self):
        """An exit code is not evidence; the re-read is."""
        self.policy = self.action_policy(actions={"set_instance_field": {
            "allowed_fields": ["risk_profile"],
            "read_argv": [sys.executable, "-c", _READ_SRC, str(self.state)],
            "apply_argv": [sys.executable, "-c", _UPSERT_NOOP_SRC],
            "validate_argv": [sys.executable, "-c", _VALIDATE_SRC,
                              "{profile}"],
            "run_timeout_sec": 30,
        }})
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertIn("DID NOT TAKE", receipt["verification"])
        self.assertIsNone(self.rows()[0]["risk_profile"])

    # ---- gates shared with a file deploy ------------------------------------
    def test_a_dry_run_changes_nothing(self):
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path, dry_run=True)
        self.assertEqual(receipt["status"], "dry_run")
        self.assertIsNone(self.rows()[0]["risk_profile"])
        self.assertTrue(path.exists(), "a dry run consumed the request")

    def test_the_readiness_gate_still_applies_to_an_action(self):
        """An action changes a live book's sizing; it must not land mid-session."""
        self.write_status(_status_doc(state="blocked"))
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused")
        self.assertTrue(receipt.get("pending_window"),
                        "a busy book must DEFER the action, not discard it")
        self.assertIsNone(self.rows()[0]["risk_profile"])

    def test_an_applied_action_is_consumed_and_receipted(self):
        path = self.stage_action({"kind": "set_instance_field",
                                  "instance_id": "bei-spx",
                                  "set": {"risk_profile": "balanced"}})
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied")
        self.assertFalse(path.exists(), "the request is still pending")
        self.assertTrue((self.outbox / "applied" / path.name).exists())
        self.assertTrue((self.outbox / _receipt_name(path)).exists(),
                        "the verdict must stay where the dashboard reads it")

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
        """A staged payload beside the request is applied and hashed.

        Staged INSIDE the outbox, which is where the sidecar puts it and the only
        place the applier will read from. The previous version of this test wrote
        the payload to ``root/staged.bin`` and recorded that path, i.e. it pinned
        the ability to hand the applier an arbitrary readable file on the host --
        the hole the containment rule exists to close.
        """
        payload = self.outbox / "req-test.0.payload"
        payload.write_bytes(b"BINARY\x00PAYLOAD")
        path, _ = self.stage(
            [{"target": "engine/blob.bin", "payload_path": payload}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "blob.bin").read_bytes(),
                         b"BINARY\x00PAYLOAD")

    def test_a_payload_staged_outside_the_outbox_is_refused(self):
        """The applier reads only what was staged beside the request.

        Targets were always contained to the deployment root; the SOURCE was
        contained to nothing, so a request could name any file the applier
        process could read -- /etc/passwd, a credential, the host's .env -- and
        the applier would copy its bytes into the deployment. Readable by the
        applier is a wide set: it runs as the user that owns production.
        """
        outside = self.root / "host-secret.bin"
        outside.write_bytes(b"NOT FOR DEPLOYMENT")
        path, _ = self.stage(
            [{"target": "engine/blob.bin", "payload_path": outside,
              "sha256": sha256_bytes(b"NOT FOR DEPLOYMENT")}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertIn("not staged beside the request", receipt["reason"])
        self.assertFalse((self.root / "engine" / "blob.bin").exists(),
                         "the applier read a file nobody staged for it")

    def test_a_traversal_in_payload_path_names_no_file_and_leaks_nothing(self):
        """``../../etc/passwd`` is reduced to a NAME and looked up inside.

        Containment is by construction rather than by checking: the resolved path
        is ``outbox / name``, and a name containing no separator cannot leave the
        directory, so there is no traversal left to defend against.
        """
        path, _ = self.stage(
            [{"target": "engine/blob.bin",
              "payload_path": self.outbox / "absent.0.payload",
              "as_recorded": "../../../../../../etc/passwd",
              "sha256": sha256_bytes(b"irrelevant")}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertFalse((self.root / "engine" / "blob.bin").exists())

    def test_a_payload_recorded_through_another_mount_view_still_applies(self):
        """THE LIVE BUG: the two halves of the seam see different paths.

        The sidecar stages through its own mount (a container path such as
        ``/project/app/data/deploy/outbox``) and records that string; the host
        applier reads the SAME directory at whatever path the host uses
        (``/srv/app/data/deploy/outbox`` in one consumer). Resolving the
        recorded string literally looked for the requester's view on a host
        where no such path exists, so every request large enough to be staged
        as a file was refused -- ten of them over a day, zero appliable, and
        the applier's unit permanently FAILED as a result.

        Only the FILENAME survives the crossing, so only the filename is used.
        """
        payload = self.outbox / "req-test.0.payload"
        payload.write_bytes(b"CROSS-MOUNT")
        foreign = f"/project/app/data/deploy/outbox/{payload.name}"
        path, req = self.stage([{"target": "engine/blob.bin",
                                 "payload_path": payload,
                                 "as_recorded": foreign,
                                 "sha256": sha256_bytes(b"CROSS-MOUNT")}])
        self.assertEqual(req["files"][0]["payload_path"], foreign,
                         "the test must record a path the host cannot see")
        self.assertFalse(Path(foreign).exists())
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "blob.bin").read_bytes(),
                         b"CROSS-MOUNT")

    def test_a_windows_pathed_payload_is_reduced_to_its_name(self):
        """``PurePath.name`` is platform-dependent, so splitting is explicit."""
        payload = self.outbox / "req-test.0.payload"
        payload.write_bytes(b"WINDOWS-STAGED")
        foreign = f"C:\\proj\\data\\deploy\\outbox\\{payload.name}"
        path, _ = self.stage([{"target": "engine/blob.bin",
                               "payload_path": payload,
                               "as_recorded": foreign,
                               "sha256": sha256_bytes(b"WINDOWS-STAGED")}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))


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
        self.assertTrue(receipt["readiness"].get("skipped"),
                        receipt["readiness"])

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
        readiness = receipt["readiness"]
        self.assertEqual(readiness["stamp_field"], "checked_at")
        self.assertIsNotNone(readiness["verdict_age_sec"],
                             "the receipt must show the age it measured")
        self.assertLess(readiness["verdict_age_sec"], 60)

    def test_mtime_semantics_are_unchanged_when_no_stamp_field_is_declared(self):
        """Back-compat: an undeclared stamp is not enforced, and says so."""
        policy = self._policy()          # no timestamp_field
        old = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
        self.write_status(_status_doc(state="ready", checked_at=old))
        path, _ = self.stage([{"target": "engine/undeclared.py",
                               "payload": "x"}], id="req-undeclared-stamp")

        receipt = self.handle(path, policy=policy)

        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIsNone(receipt["readiness"]["stamp_field"])
        self.assertIsNone(receipt["readiness"]["verdict_age_sec"],
                          "claiming an age it never measured would be a lie in "
                          "the receipt")
        #: The status document above DOES carry a stamp, but the policy never
        #: named a field to read, so reporting its value would mean the applier
        #: guessed one consumer's field name and then looked like it had
        #: dated the verdict. ``stamp_field: None`` beside ``stamp: None`` says
        #: plainly that freshness was judged by mtime alone.
        self.assertIsNone(receipt["readiness"]["stamp"],
                          "a stamp the policy never asked for must not be "
                          "reported as the verdict's date")

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


class ReceiptSchemaTest(ApplierTestBase):
    """The receipt is an audit trail, so its field names are a contract.

    One name here used to be borrowed from a consumer's domain; renaming it was
    only safe because historical receipts stay readable. Both halves are pinned,
    because each fails in a way that looks like success: a writer still emitting
    the old key keeps every consumer unchanged (silent non-migration), and a
    reader that forgets the old key blanks an audit trail at the upgrade.
    """

    def test_a_new_receipt_uses_the_generic_key_only(self):
        self.write_status(_status_doc())
        path, _ = self.stage([{"target": "engine/schema.py",
                               "payload": "s"}], id="req-schema")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIn("readiness", receipt)
        self.assertNotIn("broker", receipt,
                         "a writer still emitting the retired key would leave "
                         "consumers reading two schemas forever")
        self.assertEqual(receipt["readiness"]["state"], "ready")

    def test_legacy_receipts_are_still_readable(self):
        """A host's outbox holds verdicts from every applier it has run."""
        legacy = {"status": "refused", "reason": "state is 'busy'",
                  "broker": {"gate": "queue", "state": "busy",
                             "age_sec": 3.0}}
        self.assertEqual(readiness_of(legacy), legacy["broker"])
        current = {"status": "applied", "readiness": {"state": "drained"}}
        self.assertEqual(readiness_of(current), {"state": "drained"})
        #: And the three-valued discipline: no verdict is {}, never a guess that
        #: reads as "fine", and one corrupt row cannot break a reader listing
        #: many.
        self.assertEqual(readiness_of({"status": "applied"}), {})
        self.assertEqual(readiness_of(None), {})
        self.assertEqual(readiness_of("not a receipt"), {})
        self.assertEqual(readiness_of({"readiness": "not a dict"}), {})

    def test_the_current_key_wins_when_both_are_present(self):
        """A hand-edited or twice-written receipt must not resurrect the old
        verdict: newest schema decides."""
        both = {"readiness": {"state": "drained"},
                "broker": {"state": "busy"}}
        self.assertEqual(readiness_of(both), {"state": "drained"})
        self.assertEqual(READINESS_KEYS[0], "readiness",
                         "the tuple must stay newest-first or the retired name "
                         "decides every receipt")


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


class GitMirrorTest(ApplierTestBase):
    """An applied change must reach the repository, without ever risking the deploy.

    The mirror is a SEPARATE clone, never the deploy root: a checkout at the root
    would delete agent-applied files that are not committed yet, which is the
    drift this exists to end. These tests use a bare repo as ``origin`` so the
    push is real rather than mocked -- a mocked push would pass just as happily
    with the remote URL wrong.
    """

    def setUp(self):
        super().setUp()
        import subprocess
        self.sh = subprocess.run

        def git(*args, cwd=None):
            proc = self.sh(["git", *args], cwd=str(cwd or self.root),
                           capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0,
                             f"git {' '.join(args)} failed: {proc.stderr}")
            #: Raw stdout, not stripped: these tests compare file CONTENT, and
            #: a stripped trailing newline would make a correct commit look
            #: like it wrote a different file.
            return proc.stdout

        self.git = git
        self.origin = Path(self.tmp.name) / "origin.git"
        self.git("init", "--bare", "--initial-branch=main", str(self.origin))
        seed = Path(self.tmp.name) / "seed"
        self.git("clone", str(self.origin), str(seed))
        (seed / "README.md").write_text("seed\n", encoding="utf-8")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "add", "README.md",
                 cwd=seed)
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "seed",
                 cwd=seed)
        self.git("push", "origin", "main", cwd=seed)

        self.mirror = Path(self.tmp.name) / "mirror"
        self.git("clone", str(self.origin), str(self.mirror))

    def with_git(self, **over):
        block = {"enabled": True, "mirror_dir": str(self.mirror),
                 "remote": "origin", "branch": "main",
                 "user_name": "applier", "user_email": "applier@test",
                 "timeout_sec": 60}
        block.update(over)
        return make_policy(git=block)

    def apply(self, name="engine/thing.py", payload="x = 1\n", policy=None):
        """Stage, apply, and return (receipt, target)."""
        path, _ = self.stage([{"target": name, "payload": payload}],
                             purpose="fix the thing")
        receipt = self.handle(path, policy=policy or self.with_git())
        return receipt, self.root / name

    # ---------------------------------------------------------------- defaults
    def test_git_is_off_unless_the_policy_says_otherwise(self):
        receipt, target = self.apply(policy=self.policy)
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["git"], {"state": "disabled"})
        #: Nothing was committed: the mirror still holds only the seed commit.
        self.assertEqual(
            len(self.git("log", "--oneline", cwd=self.mirror).splitlines()), 1)

    def test_a_missing_mirror_never_fails_the_deploy(self):
        """A deploy must not depend on a repository being reachable."""
        receipt, _t = self.apply(policy=self.with_git(
            mirror_dir=str(Path(self.tmp.name) / "nope")))
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["git"]["state"], "no_mirror")
        self.assertIn("git clone", receipt["git"]["setup"])

    def test_a_directory_that_is_not_a_checkout_is_reported_as_such(self):
        plain = Path(self.tmp.name) / "plain"
        plain.mkdir()
        receipt, _t = self.apply(policy=self.with_git(mirror_dir=str(plain)))
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["git"]["state"], "no_mirror")
        self.assertIn("not a git checkout", receipt["git"]["reason"])

    # ------------------------------------------------------------------ happy
    def test_an_applied_change_is_committed_and_pushed(self):
        receipt, target = self.apply()
        self.assertEqual(receipt["status"], "applied")
        gitinfo = receipt["git"]
        self.assertEqual(gitinfo["state"], "pushed", gitinfo)
        self.assertEqual(gitinfo["files"], ["engine/thing.py"])
        self.assertTrue(gitinfo["sha"])

        #: The mirror holds the content...
        self.assertEqual((self.mirror / "engine" / "thing.py").read_text(
            encoding="utf-8"), "x = 1\n")
        #: ...and so does origin, which is what "pushed" has to mean.
        shown = self.git("show", "main:engine/thing.py", cwd=self.origin)
        self.assertEqual(shown, "x = 1\n")

    def test_the_commit_subject_names_the_request(self):
        receipt, _t = self.apply()
        sha = receipt["git"]["sha"]
        subject = self.git("log", "-1", "--format=%s", sha, cwd=self.mirror)
        self.assertIn("req-test", subject)
        self.assertIn("fix the thing", subject)

    def test_reapplying_identical_content_is_no_change_not_a_failure(self):
        first, _t = self.apply()
        self.assertEqual(first["git"]["state"], "pushed")
        second, _t = self.apply()
        self.assertEqual(second["status"], "applied")
        self.assertEqual(second["git"]["state"], "no_change")
        #: No empty commit was invented.
        self.assertEqual(len(self.git("log", "--format=%s", cwd=self.mirror)
                             .splitlines()), 2)  # seed + one deploy

    # ------------------------------------------------------------ what is staged
    def test_only_the_applied_paths_are_staged(self):
        """A commit must not sweep in unrelated drift in the mirror."""
        (self.mirror / "unrelated.py").write_text("dirty\n", encoding="utf-8")
        receipt, _t = self.apply()
        committed = self.git("show", "--name-only", "--format=",
                             receipt["git"]["sha"], cwd=self.mirror).split()
        self.assertEqual(committed, ["engine/thing.py"])
        #: And it must not be STAGED either. `git add -A` would still produce
        #: the commit above -- the commit's trailing pathspec masks it -- but it
        #: leaves someone else's edit sitting in the index, where the next
        #: commit about a different change would pick it up. So assert the exact
        #: status rather than merely that the filename appears: "?? x" means
        #: untouched, "A  x" means staged.
        porcelain = self.git("status", "--porcelain", cwd=self.mirror)
        self.assertIn("?? unrelated.py", porcelain, porcelain)
        self.assertNotIn("A  unrelated.py", porcelain, porcelain)

    def test_runtime_state_under_data_is_never_committed(self):
        """data/ is the running service's state, not code."""
        path, _ = self.stage([{"target": "data/ledger.json",
                               "payload": '{"a": 1}'}],
                             purpose="touch runtime state")
        receipt = self.handle(path, policy=self.with_git())
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["git"]["state"], "no_change")
        self.assertIn("data/ledger.json", receipt["git"]["skipped"])
        self.assertNotIn("ledger.json",
                         self.git("show", "--name-only", "--format=",
                                  "HEAD", cwd=self.mirror))

    # ------------------------------------------------------------------ failure
    def reject_pushes(self):
        """A remote that accepts fetch but refuses push.

        A nonexistent URL will not do: the sync now FETCHES before committing
        (a commit on a stale base is unpushable), so a broken remote fails at
        the fetch and the push path is never reached. A pre-receive hook keeps
        the fetch working and fails only the push.
        """
        hook = self.origin / "hooks" / "pre-receive"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\necho 'push rejected by test' >&2\nexit 1\n",
                        encoding="utf-8")
        hook.chmod(0o755)

    def test_a_failed_push_keeps_the_commit_and_still_reports_applied(self):
        """The deploy landed and was verified; a push problem must not undo it."""
        self.reject_pushes()
        receipt, _t = self.apply()
        self.assertEqual(receipt["status"], "applied",
                         "a push failure must never overturn a verified deploy")
        gitinfo = receipt["git"]
        self.assertEqual(gitinfo["state"], "committed_not_pushed", gitinfo)
        #: The commit survives -- it is the only copy of the record.
        self.assertEqual(
            self.git("show", "--name-only", "--format=", "HEAD",
                     cwd=self.mirror).split(), ["engine/thing.py"])
        self.assertIn("git push origin", gitinfo["retry"])

    def test_a_push_that_fails_leaves_the_change_retrievable(self):
        """The retry hint must actually work, once the remote accepts pushes."""
        self.reject_pushes()
        self.apply()
        (self.origin / "hooks" / "pre-receive").unlink()
        self.git("push", "origin", "main", cwd=self.mirror)
        self.assertEqual(
            self.git("show", "main:engine/thing.py", cwd=self.origin), "x = 1\n")

    def test_the_applier_never_raises_when_git_is_unusable(self):
        """Bookkeeping must not be able to break the apply path."""
        with mock.patch("deploy_applier.run_command",
                        side_effect=FileNotFoundError("no git")):
            receipt, _t = self.apply()
        self.assertEqual(receipt["status"], "applied")
        #: Whether it is reported as stale_mirror or failed is an implementation
        #: detail; what matters is that the reason says git was missing and the
        #: deploy above still succeeded.
        self.assertIn("git is not installed", receipt["git"]["reason"])
        self.assertFalse((self.mirror / "engine").exists())

    # ------------------------------------------------------------------ wiring
    def test_the_receipt_on_disk_carries_the_git_verdict(self):
        receipt, _t = self.apply()
        on_disk = json.loads(
            (self.outbox / _receipt_name(
                self.outbox / "applied" / f"req-test{REQUEST_SUFFIX}"))
            .read_text(encoding="utf-8"))
        self.assertEqual(on_disk["git"]["state"], "pushed")
        self.assertEqual(on_disk["git"]["sha"], receipt["git"]["sha"])

    def test_policy_to_dict_reports_the_git_configuration(self):
        payload = self.with_git().to_dict()["git"]
        self.assertEqual(payload, {"enabled": True,
                                   "mirror_dir": str(self.mirror),
                                   "remote": "origin", "branch": "main"})

    # -------------------------------------------------------------- subdir map
    def test_a_deploy_root_that_is_a_repo_subdirectory_lands_in_that_subdir(
            self):
        """The deploy root is usually ONE directory inside the repository.

        Found on the live mirror before this ever ran: the mirror is the repo
        root, so copying ``engine/x.py`` to the mirror root creates a NEW
        top-level ``engine/`` and commits a duplicate tree -- silently, because
        git is perfectly happy to add it. Every other test here used matching
        layouts on both sides, so none of them could have caught it.
        """
        repo = Path(self.tmp.name) / "repo"
        self.git("clone", str(self.origin), str(repo))
        (repo / "app" / "engine").mkdir(parents=True)
        (repo / "app" / "README.md").write_text("app\n", encoding="utf-8")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "add", "app",
                 cwd=repo)
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m",
                 "app tree", cwd=repo)
        self.git("push", "origin", "main", cwd=repo)

        receipt, target = self.apply(policy=self.with_git(
            mirror_dir=str(repo), subdir="app"))
        self.assertEqual(receipt["git"]["state"], "pushed", receipt["git"])
        #: The path in the commit is repo-relative...
        self.assertEqual(receipt["git"]["files"], ["app/engine/thing.py"])
        committed = self.git("show", "--name-only", "--format=", "HEAD",
                             cwd=repo).split()
        self.assertEqual(committed, ["app/engine/thing.py"])
        #: ...and no parallel top-level tree was invented.
        self.assertFalse((repo / "engine").exists(),
                         "a duplicate top-level engine/ was created")

    def test_a_subdir_that_is_not_in_the_clone_is_refused(self):
        """Wrong repo or wrong subdir must be reported, not half-applied."""
        receipt, _t = self.apply(policy=self.with_git(subdir="does-not-exist"))
        self.assertEqual(receipt["status"], "applied",
                         "the deploy itself must still succeed")
        self.assertEqual(receipt["git"]["state"], "no_mirror")
        self.assertIn("does not exist", receipt["git"]["reason"])
        self.assertFalse((self.mirror / "does-not-exist").exists())

    def test_a_subdir_cannot_escape_the_clone(self):
        for bad in ("../etc", "/abs/path", "a/../../b"):
            with self.subTest(subdir=bad):
                receipt, _t = self.apply(policy=self.with_git(subdir=bad))
                self.assertEqual(receipt["git"]["state"], "failed",
                                 receipt["git"])
                self.assertIn("subdir", receipt["git"]["reason"])

    # ------------------------------------------------------- stale clone
    def another_commit_on_origin(self, name="other.py"):
        """A human push from somewhere else, i.e. the clone falls behind."""
        work = Path(self.tmp.name) / "someone-else"
        self.git("clone", str(self.origin), str(work))
        (work / name).write_text("theirs\n", encoding="utf-8")
        self.git("-c", "user.name=h", "-c", "user.email=h@h", "add", name,
                 cwd=work)
        self.git("-c", "user.name=h", "-c", "user.email=h@h", "commit", "-m",
                 "a human push", cwd=work)
        self.git("push", "origin", "main", cwd=work)

    def test_a_clone_behind_origin_is_fast_forwarded_before_committing(self):
        """Otherwise the feature dies silently after the first human push.

        A commit made on the stale base is unpushable, the push is rejected,
        the commit is (correctly) kept, and every later attempt repeats the
        same rejection for ever.
        """
        self.another_commit_on_origin()
        receipt, _t = self.apply()
        self.assertEqual(receipt["git"]["state"], "pushed", receipt["git"])
        #: Both the human's commit and the deploy are on origin.
        log = self.git("log", "--format=%s", "main", cwd=self.origin)
        self.assertIn("a human push", log)
        self.assertIn("agent deploy", log)
        self.assertTrue((self.mirror / "other.py").exists(),
                        "the clone was not brought up to date")

    def test_a_diverged_clone_is_reported_not_guessed_at(self):
        """Guessing a merge could discard a human's work; refuse and say so."""
        self.another_commit_on_origin()
        # Local commit in the clone that origin does not have, off the old base.
        (self.mirror / "local-only.py").write_text("mine\n", encoding="utf-8")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "add",
                 "local-only.py", cwd=self.mirror)
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m",
                 "local only", cwd=self.mirror)

        receipt, _t = self.apply()
        self.assertEqual(receipt["status"], "applied",
                         "the deploy must still succeed")
        self.assertEqual(receipt["git"]["state"], "stale_mirror", receipt["git"])
        self.assertIn("DIVERGED", receipt["git"]["reason"])
        #: Nothing was written into the clone and no commit was invented.
        self.assertFalse((self.mirror / "engine" / "thing.py").exists())

    def test_a_modified_tracked_file_in_the_clone_blocks_the_sync(self):
        (self.mirror / "README.md").write_text("edited by hand\n",
                                              encoding="utf-8")
        receipt, _t = self.apply()
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["git"]["state"], "stale_mirror", receipt["git"])
        self.assertIn("modified tracked", receipt["git"]["reason"])

    def test_an_untracked_file_does_not_block_the_sync(self):
        """A fast-forward moves HEAD; it cannot lose an untracked file."""
        (self.mirror / "stray.log").write_text("noise\n", encoding="utf-8")
        receipt, _t = self.apply()
        self.assertEqual(receipt["git"]["state"], "pushed", receipt["git"])


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

    def test_a_failed_verification_still_retries_then_stops(self):
        """Bounded retries: a flaky verifier deserves a second look, not fifty.

        Live 2026-11-02: a request whose verify failed retried every three
        minutes -- each pass backing up production code, writing the payload
        over it, running a ~90 s verification and restoring the original -- and
        only stopped when the request aged past the 15-minute window. Nothing
        counted the attempts, so the loop was bounded by the AGE gate by
        accident.
        """
        target = self.root / "engine" / "retry.py"
        target.write_text("original\n", encoding="utf-8")
        path, _ = self.stage([{"target": "engine/retry.py",
                               "payload": "replacement\n"}],
                             id="req-retry", verify_command="exit 7")
        first = self.handle(path)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["attempts"], 1)
        self.assertNotIn("quarantined", first,
                         "one failure is usually a flaky verifier: retry it")
        self.assertTrue(path.exists(),
                        "the request must stay pending after one failure")
        self.assertEqual(target.read_text(encoding="utf-8"), "original\n")

        second = self.handle(path)
        self.assertEqual(second["attempts"], 2)
        self.assertTrue(path.exists())

        third = self.handle(path)
        self.assertEqual(third["status"], "failed")
        self.assertEqual(third["attempts"], MAX_FAILED_ATTEMPTS)
        self.assertTrue(third.get("quarantined"),
                        "the exhausted request must be marked in its receipt")
        self.assertIn("Giving up after", third["reason"])
        self.assertIn("re-stage", third["reason"])
        self.assertFalse(
            path.exists(),
            "an exhausted request must leave the pending set, or every pass "
            "rewrites and restores the deployment's files again")
        self.assertTrue((self.outbox / "refused" / path.name).is_file(),
                        "nothing is deleted -- re-staging starts by hand")
        self.assertEqual(target.read_text(encoding="utf-8"), "original\n",
                         "the last attempt must still restore the original")

    def test_a_request_that_changes_nothing_is_not_verified_or_retried(self):
        """A no-op must be REPORTED as one, not written, verified and failed.

        Live 2026-11-02: the agent re-proposed a collector fix that was already
        deployed, so both staged files were byte-identical to the live ones. The
        applier rewrote them, ran a ~90 s verification, hit a pre-existing
        failing test, restored the originals and said "failed" -- then retried
        every three minutes until the request aged out. The gate was reporting a
        problem with the patch when the truth was that there was no patch.

        The verify command below is deliberately one that FAILS (exit 7): if the
        no-op short-circuit ever stops running before verification, this test
        goes red instead of silently passing on a green verifier.
        """
        target = self.root / "engine" / "same.py"
        payload = "unchanged content\n"
        #: write_bytes, not write_text: on Windows write_text translates \n to
        #: CRLF, the staged payload is LF, and the fixture would then be testing
        #: a genuine byte difference instead of a no-op.
        target.write_bytes(payload.encode("utf-8"))
        path, _ = self.stage([{"target": "engine/same.py", "payload": payload}],
                             id="req-noop", verify_command="exit 7")

        receipt = self.handle(path)

        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertFalse(receipt.get("changed"),
                         "the receipt must say that nothing changed")
        self.assertIn("already matches the deployment", receipt["reason"])
        self.assertIn("verify_skipped", receipt)
        self.assertNotIn("verify_rc", receipt,
                         "a request with nothing to write must not run a "
                         "verifier against code it did not change")
        #: Consumed, not left pending: otherwise it retries forever, which is
        #: the churn this shortcut exists to prevent.
        self.assertFalse(path.exists())
        self.assertTrue((self.outbox / "applied" / path.name).is_file())
        self.assertFalse((self.outbox / "backups" / "req-noop").exists(),
                         "nothing was written, so nothing needed a backup")
        self.assertEqual(target.read_bytes(), payload.encode("utf-8"))

    def test_identical_files_with_a_restart_request_is_still_a_restart(self):
        """The shortcut must not swallow the one intent it cannot express.

        "No file changed" is not "nothing to do" when the request asks for a
        service restart: restarting IS its purpose. Skipping it would consume
        the request and report success without doing the only thing it wanted.
        """
        target = self.root / "engine" / "same3.py"
        payload = "identical\n"
        target.write_bytes(payload.encode("utf-8"))
        path, _ = self.stage([{"target": "engine/same3.py", "payload": payload}],
                             id="req-restart-only", services=["engine"],
                             verify_command="exit 7")
        receipt = self.handle(path)
        #: It must NOT take the no-op path: with a failing verifier that means
        #: the normal path ran, i.e. the restart was not skipped.
        self.assertEqual(receipt["status"], "failed", receipt.get("reason"))
        self.assertIn("verification failed", receipt["reason"])

    def test_a_partly_identical_request_is_still_applied(self):
        """The short-circuit must require EVERY target to match.

        A request usually carries the fix AND its test; one of them matching the
        deployment already is normal and must not suppress the other.
        """
        same = self.root / "engine" / "same2.py"
        same.write_bytes(b"already here\n")
        changed = self.root / "engine" / "changed2.py"
        changed.write_text("old\n", encoding="utf-8")
        path, _ = self.stage(
            [{"target": "engine/same2.py", "payload": "already here\n"},
             {"target": "engine/changed2.py", "payload": "new\n"}],
            id="req-mixed")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual(changed.read_text(encoding="utf-8"), "new\n")

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

    def test_an_applied_requests_payloads_leave_the_outbox_root(self):
        """The root must mean UNDECIDED, for refusals AND for successes.

        Live 2026-11-01: the REFUSED path moved a request's ``.N.payload``
        files aside, the APPLIED path did not, so eight payloads accumulated in
        the root. An agent tasked with a config change found
        ``req-...0.payload`` (631008 B, an applied copy of engine/collector.py),
        sized it, and concluded the deployed collector was that file -- then
        reported a live feature as never deployed and warned the owner that a
        rebuild would regress production. The deployed file was 620781 B.

        A false "the code you think is running is not running" is worse than
        clutter: the natural response is to redeploy over something that works.

        The payload MUST be staged as a real FILE (``payload_path``), not as
        inline ``content``: an inline request creates no file, so a glob over
        ``<base>.*`` then matches only the RECEIPT and the assertion passes
        while the bug is present. That is exactly how the first version of this
        test was vacuous -- the mutation removing the fix survived it.
        """
        payload = self.outbox / "req-test.0.payload"
        payload.write_bytes(b"staged payload bytes")
        path, _ = self.stage(
            [{"target": "engine/p.py", "payload_path": payload}])
        self.assertTrue(payload.exists(),
                        "the fixture staged no payload file, so this test "
                        "would prove nothing")
        self.assertEqual(self.handle(path)["status"], "applied")

        self.assertFalse(payload.exists(),
                         "the payload is still in the outbox root, where it "
                         "reads as pending work")
        self.assertTrue((self.outbox / "applied" / payload.name).exists(),
                        "the payload was deleted rather than moved: it is part "
                        "of the audit trail")
        self.assertEqual(pending_requests(self.outbox), [])

    def test_the_receipt_stays_where_the_dashboard_reads_it(self):
        """Only the request and its payloads move; the verdict does not.

        The deployment's dashboard reads receipts from the outbox ROOT to show
        what the gate last decided, so moving the receipt with the request would
        blank that view.
        """
        path, _ = self.stage([{"target": "engine/r2.py", "payload": "x"}])
        self.assertEqual(self.handle(path)["status"], "applied")
        self.assertTrue((self.outbox / _receipt_name(path)).exists(),
                        "the receipt left the root and the dashboard can no "
                        "longer show the verdict")

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


class QuarantineTest(ApplierTestBase):
    """A refusal that cannot change must leave the pending set.

    The failure this prevents is not a wrong deploy -- it is an ALARM THAT STOPS
    BEING READ. Every refusal exits 2 (a contract the callers rely on), so a
    request that can never succeed keeps the applier's systemd unit permanently
    FAILED and writes the same lines to its log every pass. Ten such requests sat
    in a live outbox for a day. When the unit is always red, the pass that
    refuses a REAL deploy for a REAL reason is indistinguishable from the noise,
    and the gate's only output channel is spent.

    The distinction is by TYPE, not by matching refusal text: a request whose
    payload, checksum, path or age is wrong is wrong forever, while the readiness
    gate describes a world that changes -- a busy system is the normal state of
    a live one most of the day, and a change proposed during it is deferred,
    not rejected. Getting that backwards in the safe direction would DROP a
    legitimate request with only a receipt to show for it.
    """

    def _pending(self):
        return sorted(p.name for p in pending_requests(self.outbox))

    def test_a_permanent_refusal_leaves_the_pending_set(self):
        path, _ = self.stage([{"target": "engine/a.py", "payload": "x"},
                              {"target": "../escape.py", "payload": "y"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertNotIn(path.name, self._pending(),
                         "it would be refused again on every pass, forever")
        #: The receipt stays where the audit trail and the dashboard expect it.
        self.assertTrue((self.outbox / _receipt_name(path)).is_file())
        self.assertTrue((self.outbox / "refused" / path.name).is_file(),
                        "quarantined, not deleted: moving it back is how a "
                        "request gets reconsidered")

    def test_a_readiness_refusal_stays_pending_until_the_window_opens(self):
        """The asymmetric case: deferring is not rejecting.

        Dropping this one would mean a change proposed while the system is busy
        silently disappears, and the agent that proposed it has no way to learn
        that it must be re-staged.
        """
        self.write_status(_status_doc(state="busy"))
        path, _ = self.stage([{"target": "engine/a.py", "payload": "x"}])
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertTrue(receipt.get("pending_window"),
                        "the receipt must say WHY it is still pending")
        self.assertIn(path.name, self._pending())
        self.assertFalse((self.outbox / "refused" / path.name).exists(),
                         "a window refusal must not be quarantined")
        #: ...and once the world is ready, the SAME request applies.
        self.write_status(_status_doc(state="ready"))
        receipt = self.handle(path, policy=make_policy())
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))

    def test_an_unreadable_request_is_quarantined_with_its_receipt(self):
        path = self.outbox / f"broken{REQUEST_SUFFIX}"
        path.write_text("{not json", encoding="utf-8")
        self.handle(path)
        self.assertNotIn(path.name, self._pending())
        self.assertTrue((self.outbox / _receipt_name(path)).is_file())

    def test_quarantine_takes_the_staged_payloads_with_it(self):
        """Payloads left behind accumulate in the directory every pass scans.

        The payload must share the request's id prefix, because that is how the
        applier knows which staged files belong to which request -- and a
        quarantine that left them behind would grow the scanned directory with
        files no request will ever claim again.
        """
        rid = "req-q"
        payload = self.outbox / f"{rid}.0.payload"
        payload.write_bytes(b"STAGED")
        path, _ = self.stage([{"target": "engine/a.py", "payload_path": payload,
                               "sha256": sha256_bytes(b"WRONG-ON-PURPOSE")}],
                             id=rid)
        self.assertEqual(path.name, f"{rid}{REQUEST_SUFFIX}")
        receipt = self.handle(path)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertIn("checksum", receipt["reason"])
        self.assertTrue((self.outbox / "refused" / payload.name).is_file(),
                        "the payload must travel with its request")
        self.assertFalse(payload.exists())
        self.assertTrue((self.outbox / _receipt_name(path)).is_file(),
                        "the receipt stays at the outbox root, where the audit "
                        "trail and the dashboard read it")

    def test_a_dry_run_moves_nothing(self):
        """A rehearsal that quarantines is not a rehearsal.

        ``--dry-run`` exists to answer "what would happen" without consequences;
        consuming the request would make the real run afterwards report "no
        pending requests", which reads as nothing-was-proposed rather than
        already-consumed. Note the STATUS is still ``refused`` -- a dry run must
        report the gates it would fail, not invent a passing grade.
        """
        path, _ = self.stage([{"target": "../escape.py", "payload": "x"}])
        receipt = self.handle(path, dry_run=True)
        self.assertEqual(receipt["status"], "refused", receipt)
        self.assertTrue(receipt.get("dry_run"), "the receipt must say it was a "
                                                "rehearsal")
        self.assertTrue(path.exists())
        self.assertFalse((self.outbox / "refused").exists())

    def test_a_second_pass_over_a_quarantined_outbox_is_quiet(self):
        """The whole point: run_once stops reporting the same refusal forever."""
        path, _ = self.stage([{"target": "../escape.py", "payload": "x"}])
        self.assertTrue(path.exists())
        first = run_once(self.outbox, self.root, dry_run=False,
                         policy=self.policy)
        second = run_once(self.outbox, self.root, dry_run=False,
                          policy=self.policy)
        self.assertEqual(first, 2, "the refusing pass must still be loud")
        self.assertEqual(second, 0,
                         "the next pass has nothing pending, so it must be "
                         "green rather than repeating yesterday's refusal")

    def test_waiting_for_the_window_is_not_a_failure(self):
        """A deferred request must leave the unit GREEN.

        The readiness window is closed because the system is busy, which on a
        live one is the normal state most of the day -- and the request
        stays queued. Counting that as a failure kept the applier's unit red every
        two minutes for as long as the system stayed busy, which is the alarm fatigue
        the quarantine work was meant to remove, reintroduced from the other side:
        the pass that refuses a REAL deploy then looks exactly like the noise.

        The request must still be there afterwards -- green means "nothing to do
        yet", never "discarded".
        """
        self.write_status(_status_doc(state="busy"))
        path, _ = self.stage([{"target": "engine/a.py", "payload": "x"}])
        rc = run_once(self.outbox, self.root, dry_run=False, policy=self.policy)
        self.assertEqual(rc, 0, "a closed readiness window is not a failure")
        self.assertIn(path.name, self._pending(),
                      "the request must still be waiting, not consumed")


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
        """rc 2 is for a refusal the request can never outgrow.

        A readiness window is deliberately NOT this case: it is closed because
        the system is busy, the request stays queued, and the unit must stay green
        or the alarm means nothing by the time a real refusal arrives. So the
        rc-2 path is exercised with a permanent refusal -- a target that escapes
        the deployment root -- which is refused, quarantined, and reported once.
        """
        self.write_status(_status_doc(state="ready"))
        self.stage([{"target": "../cli-escape.py", "payload": "x"}],
                   id="cli-refused")
        rc = main(["--root", str(self.root), "--outbox", str(self.outbox),
                   "--policy", str(self.policy_path), "--once",
                   "--status-path", str(self.status)])
        self.assertEqual(rc, 2)
        self.assertFalse((self.root.parent / "cli-escape.py").exists())

    def test_cli_stays_green_while_waiting_for_the_window(self):
        """The complement: a deferral must not look like a refusal."""
        self.write_status(_status_doc(state="busy"))
        self.stage([{"target": "engine/cli3.py", "payload": "x"}],
                   id="cli-deferred")
        rc = main(["--root", str(self.root), "--outbox", str(self.outbox),
                   "--policy", str(self.policy_path), "--once",
                   "--status-path", str(self.status)])
        self.assertEqual(rc, 0)
        self.assertFalse((self.root / "engine" / "cli3.py").exists())
        self.assertEqual(len(pending_requests(self.outbox)), 1,
                         "and it is still queued for the next pass")

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


class AdvisoryGateTest(ApplierTestBase):
    """A deployment may ask the gate for EVIDENCE without a VETO.

    Why the switch exists: a gate that blocks is not free. On a paper-trading
    host the owner's stated priority is that every fix lands during the pass
    that found it, and a request deferred until the account happens to be in a
    particular state is a fix that waits for a condition the pass cannot
    schedule -- so it is re-staged next pass, or forgotten. The deployment
    therefore wants the gate's observations kept and its veto dropped.

    What must NOT change is the evidence. These tests pin both halves:

    * the default is still to ENFORCE, so no deployment relaxes itself by
      omission -- only by saying so in its own reviewed policy;
    * when relaxed, the receipt still says what the gate saw and the exact
      sentence it would have refused with, so "applied anyway" is never
      indistinguishable from "the gate was happy".
    """

    def relaxed(self, readiness=None, **over):
        spec = {"status_file": "data/readiness.json", "field": "state",
                "label": "readiness", "safe_states": list(SAFE_STATES),
                "max_age_sec": 900, "enforce": False}
        spec.update(readiness or {})
        return make_policy(readiness=spec, **over)

    def deploy_one(self, policy=None):
        path, _req = self.stage([{"target": "engine/mod.py",
                                  "payload": "NEW = 1\n"}])
        return self.handle(path, policy=policy or self.relaxed())

    # ---- the default has not moved -----------------------------------------
    def test_the_default_is_to_enforce(self):
        """Silence means veto: a relaxation must be written down to happen."""
        self.assertTrue(make_policy().readiness_enforce)
        self.assertFalse(Policy({}).readiness_enforce is False)

    def test_the_spellings_a_json_file_really_uses_are_understood(self):
        """bool("false") is True; a stringly-typed policy must not fail open."""
        for raw, want in ((False, False), ("false", False), ("no", False),
                          (0, False), (True, True), ("true", True), (1, True)):
            with self.subTest(raw=raw):
                self.assertIs(self.relaxed({"enforce": raw}).readiness_enforce,
                              want)

    def test_a_strict_policy_still_defers_a_busy_system(self):
        self.write_status(_status_doc(state="busy"))
        path, _ = self.stage([{"target": "engine/mod.py",
                               "payload": "NEW = 2\n"}])
        receipt = self.handle(path, policy=make_policy())
        self.assertEqual(receipt["status"], "refused")
        self.assertTrue(receipt.get("pending_window"))
        self.assertFalse((self.root / "engine" / "mod.py").exists())

    # ---- relaxed: the deploy lands, the evidence survives ------------------
    def test_an_incoherent_state_is_recorded_and_the_change_still_lands(self):
        self.write_status(_status_doc(state="busy"))
        receipt = self.deploy_one()
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "mod.py")
                         .read_text(encoding="utf-8"), "NEW = 1\n")
        gate = receipt["readiness"]
        self.assertIs(gate["enforced"], False)
        # The observation, not just the omission: which state, and the sentence
        # a strict gate would have used.
        self.assertEqual(gate["state"], "busy")
        self.assertFalse(gate.get("skipped"),
                         "a relaxed gate EVALUATED; it must not read as skipped")
        self.assertIn("busy", gate["would_refuse"])
        self.assertTrue(gate["pending_window"],
                        "a busy system is a closed window, not a malformed one")

    def test_a_stale_verdict_is_recorded_and_the_change_still_lands(self):
        """The 2026-10-31 case: a touched file carrying an hour-old verdict."""
        self.write_status(_status_doc(), stale_sec=3600)
        receipt = self.deploy_one()
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIn("min old", receipt["readiness"]["would_refuse"])
        self.assertGreater(receipt["readiness"]["age_sec"], 900)

    def test_a_missing_status_file_is_recorded_and_the_change_still_lands(self):
        self.status.unlink()
        receipt = self.deploy_one()
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIn("no readiness status file",
                      receipt["readiness"]["would_refuse"])
        self.assertNotIn("state", receipt["readiness"],
                         "there was no state to report, so none may be invented")

    def test_an_unparsable_stamp_is_recorded_and_the_change_still_lands(self):
        self.write_status(_status_doc(checked_at="an hour ago"))
        receipt = self.deploy_one(
            self.relaxed({"timestamp_field": "checked_at"}))
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertIn("no parsable", receipt["readiness"]["would_refuse"])

    def test_a_skipped_gate_is_still_skipped_not_relaxed(self):
        """Relaxing a veto must not make an unevaluated gate look evaluated."""
        no_states = self.relaxed({"safe_states": []})
        gate = gate_readiness(self.root, no_states, status_path=self.status)
        self.assertTrue(gate.get("skipped"))

    # ---- verification: warning, not rollback -------------------------------
    def test_a_failing_verifier_keeps_the_files_when_verification_is_advisory(self):
        path, _ = self.stage([{"target": "engine/mod.py",
                               "payload": "NEW = 3\n"}],
                             verify_command="exit 7")
        receipt = self.handle(path, policy=self.relaxed(verify={"blocking": False}))
        self.assertEqual(receipt["status"], "applied", receipt.get("reason"))
        self.assertEqual((self.root / "engine" / "mod.py")
                         .read_text(encoding="utf-8"), "NEW = 3\n",
                         "an advisory failure must not restore the original")
        self.assertIs(receipt["verify_advisory"], True)
        self.assertEqual(receipt["verify_rc"], 7)
        self.assertIn("KEPT", receipt["verify_warning"])
        self.assertNotIn("quarantined", receipt,
                         "the request succeeded; nothing to give up on")

    def test_a_failing_verifier_still_rolls_back_by_default(self):
        """The safety net is the DEFAULT; only a policy may take it away."""
        path, _ = self.stage([{"target": "engine/mod.py",
                               "payload": "NEW = 4\n"}],
                             verify_command="exit 7")
        receipt = self.handle(path, policy=make_policy())
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse((self.root / "engine" / "mod.py").exists(),
                         "the original file did not exist, so rollback removes "
                         "the new one")

    def test_a_passing_verifier_is_unchanged_under_advisory(self):
        path, _ = self.stage([{"target": "engine/mod.py",
                               "payload": "NEW = 5\n"}],
                             verify_command="exit 0")
        receipt = self.handle(path, policy=self.relaxed(verify={"blocking": False}))
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["verify_rc"], 0)
        self.assertNotIn("verify_advisory", receipt,
                         "advisory only describes a FAILURE that was kept")

    def test_the_policy_report_says_which_gates_can_veto(self):
        """A receipt reader must not have to open the policy to know."""
        self.assertIs(self.relaxed(verify={"blocking": False})
                      .to_dict()["verify"]["blocking"], False)
        self.assertIs(self.relaxed().to_dict()["readiness"]["enforce"], False)
        strict = make_policy().to_dict()
        self.assertIs(strict["readiness"]["enforce"], True)
        self.assertIs(strict["verify"]["blocking"], True)


if __name__ == "__main__":
    unittest.main()
