"""This repo is GENERIC: it holds no one deployment's knowledge. Pinned, not
merely intended.

A consumer ports from here or bundles it at deploy time, and everything
specific to that consumer arrives as CONFIGURATION the mechanism reads: the
policy file, env (``AGENT_*`` plus the provider catalogue's own ``api_key_env``
names), and mounts. The applier's own docstring says it "knows nothing about
brokers, trading sessions" -- this file is what makes that a property instead
of an aspiration.

Two ways that separation leaks, and they fail very differently:

* **BEHAVIOUR** -- the gate hardcoding a consumer's status file, field name or
  safe states. Already pinned in ``test_deploy_applier.py``: ``make_policy``
  drives the readiness gate over ``phase``/``health``/``draining``, vocabulary
  no trading system uses, and every piece of it is a ``Policy`` attribute. Not
  repeated here.
* **NAMES** -- a comment, docstring or test fixture quoting the deployment's own
  directories. This is the one that actually happened: an applier fix arrived
  with the motivating deployment's paths (``/project/<product>/...`` and
  ``/opt/<product>/...``) pasted into its docstring and into the cross-mount
  test, in five places. A name is not a coupling, which is exactly why review
  misses it -- the code keeps working for everyone, and the erosion is
  invisible until a second consumer reads this repo and finds it half-written
  about somebody else's deployment.

The scan matches NAMES, not concepts, and says so rather than implying more:
prose like "an open book is the normal state of a trading system" passes,
because no consumer name appears in it. That prose rides on the published
``gate_broker`` vocabulary, whose extent is pinned by
``published_vocabulary_is_not_extended`` so it stays a known artifact instead
of a growing one.

Ignored paths are out of scope by definition (``sessions/``, ``memories/session/``,
``.env``, the live ``providers.json``/``routes.json``): they cannot reach
another consumer, and the live catalogue is where a consumer's specifics are
SUPPOSED to live locally.

Run:  python test_genericity.py
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from typing import Dict, List, Tuple

import deploy_applier

REPO = Path(__file__).resolve().parent

#: Names of deployments this harness has shipped for. The list is not a
#: blacklist of bad words -- it is the set of separations worth protecting, and
#: it grows by one line when a second consumer ports from here.
#:
#: Test files that DEFINE this list are excluded from the scan, or the guard
#: fails on its own inventory.
DEPLOYMENT_NAMES: Tuple[str, ...] = ("stranglebot", "ibkrai")

#: Scanned text: source, docs and every committed template. Deliberately
#: explicit rather than "everything", so a stray binary or a scratch directory
#: can neither crash the scan nor hide behind it.
SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".example",
                 ".sh", ".toml", ".cfg", ".ini"}
SCAN_NAMES = {".gitignore", "Dockerfile", "LICENSE"}

#: Ignored two ways, because "not source" has two shapes.
#: A component matched ANYWHERE is build or scratch space that never ships.
SKIP_DIR_PARTS = {".git", ".venv", "venv", "__pycache__", "sessions", ".tmp",
                  "out", "workspace", "node_modules"}
#: A PREFIX names a runtime subtree living under a tree that IS tracked --
#: something a single component cannot express without also excluding its
#: siblings.
#:
#: ``memories/session/`` is this harness's own agent-written notes. It is
#: gitignored, never shipped, and by construction full of whatever deployment
#: the agent happened to be monitoring -- so scanning it would fail on
#: legitimate runtime output, and a guard that cries wolf on its own notes is
#: a guard somebody disables rather than obeys. Its siblings,
#: ``memories/repo/GETTING-STARTED.md`` and ``memories/user.md``, ARE tracked
#: seeds and stay scanned.
SKIP_RELATIVE_PREFIXES = (("memories", "session"),)
#: Files that name every consumer on purpose: this guard's own inventory.
SKIP_FILES = {Path(__file__).name}


def find_deployment_names(text: str) -> List[str]:
    """Which known consumer names appear in this text. Case-insensitive.

    A pure function over a string, so the guard can be proven able to FAIL
    without writing anything into the repo.
    """
    low = text.lower()
    return sorted({name for name in DEPLOYMENT_NAMES if name in low})


def scanned_files(root: Path = REPO) -> List[Path]:
    """Every tracked-looking text file under the repo root."""
    out: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_FILES:
            continue
        ancestors = path.relative_to(root).parts[:-1]
        if any(part in SKIP_DIR_PARTS for part in ancestors):
            continue
        if any(ancestors[:len(prefix)] == prefix
               for prefix in SKIP_RELATIVE_PREFIXES):
            continue
        if path.suffix in SCAN_SUFFIXES or path.name in SCAN_NAMES:
            out.append(path)
    return out


def scan(root: Path = REPO) -> Dict[str, List[Tuple[int, List[str]]]]:
    """``{relative file: [(line number, names found), ...]}`` for every hit."""
    hits: Dict[str, List[Tuple[int, List[str]]]] = {}
    for path in scanned_files(root):
        try:
            lines = path.read_text(encoding="utf-8",
                                   errors="replace").splitlines()
        except OSError:
            continue
        found = []
        for number, line in enumerate(lines, start=1):
            names = find_deployment_names(line)
            if names:
                found.append((number, names))
        if found:
            hits[str(path.relative_to(root))] = found
    return hits


class GenericityTest(unittest.TestCase):
    def test_the_scanner_can_fail(self):
        """Prove the guard is not vacuous BEFORE trusting its green.

        A scanner that cannot detect the thing it scans for reports success
        identically to one that can. Both directions, on strings -- nothing is
        written into the repo to prove it.
        """
        self.assertEqual(
            find_deployment_names("see /project/stranglebot/data/deploy"),
            ["stranglebot"])
        self.assertEqual(find_deployment_names("StrangleBot's deploy root"),
                         ["stranglebot"],
                         "a consumer's product name in any casing is a leak")
        self.assertEqual(find_deployment_names("the IBKRAI deployment"),
                         ["ibkrai"])
        self.assertEqual(
            find_deployment_names("resolve by name inside the outbox"), [])

    def test_the_scan_covers_a_tree_it_is_given(self):
        """Coverage proven against a synthetic tree, not against this checkout.

        The first version of this test asserted that ``README.md`` was among the
        scanned files -- true in a checkout, false inside the shipped image,
        which carries only ``*.py`` plus the requirements list. So the guard was
        red in exactly the environment where it is also run as a parity check,
        for a reason that had nothing to do with genericity. Building a tree
        here makes the assertion about the SCANNER rather than about its
        surroundings, and it can state the skip rules instead of implying them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            #: Scanned: a source file and a doc, one clean and one leaking, and
            #: nested -- a scanner that only walks the top level would miss it.
            (root / "app.py").write_text("# clean\n", encoding="utf-8")
            (root / "notes").mkdir()
            (root / "notes" / "README.md").write_text(
                "line one\nsee /opt/stranglebot/data\n", encoding="utf-8")
            #: Ignored trees: runtime artefacts that never reach a consumer.
            for ignored in ("sessions", "memories/session", ".venv/lib",
                            "__pycache__"):
                d = root / ignored
                d.mkdir(parents=True, exist_ok=True)
                (d / "run.md").write_text("stranglebot everywhere\n",
                                          encoding="utf-8")
            #: A TRACKED sibling of an ignored subtree must stay scanned:
            #: excluding ``memories/session`` is not a licence to exclude
            #: ``memories/repo``.
            (root / "memories" / "repo").mkdir(parents=True, exist_ok=True)
            (root / "memories" / "repo" / "GUIDE.md").write_text(
                "a seed mentions ibkrai\n", encoding="utf-8")
            #: The guard's own file names every consumer on purpose.
            (root / Path(__file__).name).write_text(
                'NAMES = ("stranglebot",)\n', encoding="utf-8")
            #: Not a scanned suffix, so not read at all.
            (root / "blob.bin").write_bytes(b"stranglebot")

            hits = scan(root)
            self.assertEqual(
                hits,
                {str(Path("notes") / "README.md"): [(2, ["stranglebot"])],
                 str(Path("memories") / "repo"
                     / "GUIDE.md"): [(1, ["ibkrai"])]},
                "the scan must cover nested text files, must report the line "
                "number, must leave ignored trees and its own inventory file "
                "alone, and must still scan a tracked sibling of an ignored "
                "subtree")

    def test_the_scan_reaches_this_repos_real_sources(self):
        """Limited to files every environment ships: the image carries only
        ``*.py`` plus requirements, so docs and the compose file are asserted
        ONLY when they are actually present -- a guard that fails because it
        was installed instead of cloned would be noise nobody reads."""
        files = {p.name for p in scanned_files()}
        for always_shipped in ("deploy_applier.py", "scheduler.py",
                               "config.py", "test_deploy_applier.py"):
            self.assertIn(always_shipped, files)
        for checkout_only in ("README.md", "docker-compose.yml",
                              str(Path("docs") / "ARCHITECTURE.md")):
            present = (REPO / checkout_only).exists()
            self.assertEqual(present, Path(checkout_only).name in files,
                             f"{checkout_only} exists={present} but "
                             f"scanned={Path(checkout_only).name in files}")

    def test_no_consumer_deployment_name_appears_in_the_generic_sources(self):
        hits = scan()
        self.assertEqual(
            hits, {},
            "the generic repo names a specific deployment:\n"
            + "\n".join(f"  {f}:{line} ({', '.join(names)})"
                        for f, rows in sorted(hits.items())
                        for line, names in rows)
            + "\nA deployment's paths belong in ITS policy/compose/docs. "
              "Rewrite them here as generic placeholders, or move the text to "
              "the consumer's repo; if a new consumer was added, add its name "
              "to DEPLOYMENT_NAMES so this guard protects it too.")


class PublishedVocabularyTest(unittest.TestCase):
    """The readiness gate's trading-flavoured names: acknowledged, and bounded.

    ``deploy_applier.gate_broker`` / the receipt's ``"broker"`` key predate the
    genericity scan and are PUBLISHED: a consumer's suite calls ``gate_broker``
    and asserts on ``receipt["broker"]``. Renaming them is a coordinated
    cross-repo change, not a cleanup to slip in beside one, so this file
    instead keeps them from MULTIPLYING -- one gate name, one receipt key, and
    no consumer's own artefact filenames anywhere in the mechanism.

    Behaviourally the gate is already fully policy-driven (status file, field,
    label, safe states and stamp field all come from ``Policy``).
    ``test_deploy_applier.ReadinessGateTest`` pins that end to end with
    non-trading vocabulary; the two tests below re-prove it where a name scan
    cannot reach -- in the gate's OWN source (a hardcoded field would sit
    beside a policy read elsewhere in the same file) and against a
    queue-shaped deployment whose vocabulary shares nothing with a trading
    one.
    """

    def setUp(self):
        self.applier = (REPO / "deploy_applier.py").read_text(encoding="utf-8")
        #: Same scrub as the applier's own suite: the CLI defaults to the
        #: ambient deploy environment, so inside a sidecar these tests would
        #: otherwise judge a sandbox through production configuration -- and a
        #: verdict that depends on the machine it runs on cannot tell a
        #: regression from its surroundings.
        ambient = [k for k in os.environ
                   if k.startswith("AGENT_DEPLOY_") or k == "AGENT_ROOT"]
        if ambient:
            patcher = mock.patch.dict(os.environ, {})
            patcher.start()
            self.addCleanup(patcher.stop)
            for key in ambient:
                os.environ.pop(key, None)

    def test_the_gate_name_and_receipt_key_exist_exactly_once_each(self):
        self.assertEqual(self.applier.count("def gate_broker"), 1)
        self.assertEqual(self.applier.count('"broker":'), 1,
                         "a second trading-named receipt key would extend the "
                         "published vocabulary instead of leaving it alone")

    def test_no_consumer_artefact_filename_is_known_to_the_mechanism(self):
        """The policy names these; the applier must never spell one."""
        for artefact in ("broker_status.json", "stack_paused.json",
                         "positions.json", "trades.jsonl", "deploy_policy.json"):
            self.assertNotIn(artefact, self.applier,
                             f"{artefact} is one deployment's file, named "
                             f"inside the generic applier")

    def test_the_documented_claim_still_holds(self):
        """The applier advertises knowing nothing about brokers. Pin the claim
        where it is made, not against the whole file."""
        self.assertIn("knows nothing about brokers", self.applier)

    def test_the_gate_reads_a_consumers_vocabulary_from_its_policy(self):
        """Per-FUNCTION, and behaviourally -- never by searching the file.

        A file-level check for ``policy.status_field`` passed while the gate
        itself hardcoded ``data.get("state")``, because that attribute is also
        read further down when building the receipt. "Appears somewhere in the
        file" therefore cannot distinguish one hardcoded site from none -- a
        checker that cannot see the difference certifies nothing. (This test was
        written as exactly that, and mutating the gate proved it: the assertion
        stayed green against a hardcoded field.)
        """
        source = inspect.getsource(deploy_applier.gate_broker)
        for attr in ("status_label", "status_field", "safe_states",
                     "max_status_age_sec"):
            self.assertIn(f"policy.{attr}", source,
                          f"{attr} is not read from the policy inside the "
                          f"gate, so the gate hardcodes consumer vocabulary")
        self.assertNotIn('data.get("state"', source,
                         "a hardcoded status field: a policy naming another "
                         "field would be silently ignored while the gate "
                         "still looked policy-driven")

    def test_a_consumer_with_its_own_vocabulary_is_gated_by_its_own_policy(
            self):
        """The mechanism works for names this repo has never heard of.

        Genericity is not only the absence of a consumer's names; it is that a
        consumer's names arrive as DATA. A queue-shaped deployment (``phase`` /
        ``drained`` / ``queue``) shares no vocabulary with a trading one and
        must gate identically.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            status = root / "data" / "phase.json"
            policy = deploy_applier.Policy(
                {"readiness": {"status_file": "data/phase.json",
                               "field": "phase", "label": "queue",
                               "safe_states": ["drained"],
                               "max_age_sec": 900}},
                name="consumer-policy")
            status.write_text(json.dumps({"phase": "drained"}),
                              encoding="utf-8")
            verdict = deploy_applier.gate_broker(root, policy,
                                                status_path=status)
            self.assertEqual(verdict["gate"], "queue")
            self.assertEqual(verdict["state"], "drained")
            self.assertEqual(verdict["field"], "phase")
            status.write_text(json.dumps({"phase": "busy"}), encoding="utf-8")
            with self.assertRaises(deploy_applier.PendingWindow) as refused:
                deploy_applier.gate_broker(root, policy, status_path=status)
            message = str(refused.exception)
            self.assertIn("busy", message)
            self.assertIn("queue", message,
                          "the refusal must speak the consumer's own label, "
                          "not this repo's vocabulary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
