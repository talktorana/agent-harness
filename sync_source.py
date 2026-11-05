#!/usr/bin/env python3
"""Compare harness source across checkout, host tree, and container image.

Three-way drift detection for a deployment where the harness code is edited on
a workstation, published to a host, and baked into a container image. Any two
of those can disagree, and the disagreement is silent until something breaks.

The tool answers: "is the code I am looking at the code that is running?"

Usage:
    python3 sync_source.py --check              # report only (default)
    python3 sync_source.py --to-host            # publish local → host
    python3 sync_source.py --from-host          # harvest host → local
    python3 sync_source.py --to-host --rebuild  # publish, then rebuild image
    python3 sync_source.py --check --json       # machine-readable

The manifest is derived from the Dockerfile's COPY lines: every *.py file the
image would contain. Config files (providers.json, routes.json, .env) are
excluded by design -- they are deployment-specific and gitignored.

Read-only by default. --to-host and --from-host are the only write modes, and
they are mutually exclusive: a bidirectional sync in one pass would make it
impossible to tell which direction fixed the drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Where the harness checkout lives. Default: this script's directory.
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parent

#: Files that are deployment-specific and should never be synced.
EXCLUDE = {
    "providers.json", "routes.json", ".env", "schedule.json",
    "deploy_policy.json", "mission.md", "model_choice.json",
    "run_settings.json", "schedule_state.json",
}


def manifest_from_dockerfile(root: Path) -> List[str]:
    """The files the image would contain, derived from the Dockerfile.

    An explicit list is a trap: a new .py file works locally and the container
    dies on import because it was missing from the COPY list. Deriving from
    the Dockerfile means the list cannot be wrong about what the image holds.
    """
    dockerfile = root / "Dockerfile"
    if not dockerfile.is_file():
        # Fall back to all .py files if no Dockerfile exists.
        return sorted(p.name for p in root.glob("*.py")
                      if p.name not in EXCLUDE)
    files = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("COPY ") and "*.py" in line:
            files.extend(p.name for p in root.glob("*.py")
                         if p.name not in EXCLUDE)
    return sorted(set(files))


def content_md5(payload: bytes) -> str:
    """MD5 of the CONTENT, with line endings normalised to LF.

    Checkouts on one OS and hosts on another routinely use different line
    endings. Hashing raw bytes reports that as a code difference forever, which
    makes a correct installation look stale and can lead an operator to replace
    good content. Actual source bytes still change the hash.
    """
    return hashlib.md5(payload.replace(b"\r\n", b"\n")).hexdigest()


def md5(path: Path) -> str:
    """Normalized content hash, truncated for the drift table."""
    try:
        return content_md5(path.read_bytes())[:16]
    except OSError:
        return ""


def compare_local(root: Path, manifest: List[str]) -> Dict[str, str]:
    """Hash every manifest file in the local checkout."""
    return {name: md5(root / name) for name in manifest}


def compare_host(host_root: str, manifest: List[str]) -> Dict[str, str]:
    """Hash every manifest file on the host via ssh."""
    result = {}
    for name in manifest:
        path = os.path.join(host_root, name)
        # Normalize line endings on the remote side too, or a correct LF host
        # copy never equals a CRLF checkout on a workstation.
        proc = subprocess.run(
            ["ssh", HOST, f"tr -d '\\r' < {path} 2>/dev/null | md5sum | cut -d' ' -f1"],
            capture_output=True, text=True, timeout=30,
        )
        result[name] = proc.stdout.strip()[:16] if proc.returncode == 0 else ""
    return result


def compare_image(image: str, manifest: List[str]) -> Dict[str, str]:
    """Hash every manifest file inside the container image."""
    result = {}
    for name in manifest:
        proc = subprocess.run(
            ["sudo", "docker", "run", "--rm", "--entrypoint", "sh",
             image, "-c", f"tr -d '\\r' < /harness/{name} 2>/dev/null | md5sum | cut -d' ' -f1"],
            capture_output=True, text=True, timeout=30,
        )
        result[name] = proc.stdout.strip()[:16] if proc.returncode == 0 else ""
    return result


def verdict(local: Dict[str, str], host: Dict[str, str],
            image: Dict[str, str]) -> List[Dict[str, str]]:
    """Classify each file's drift state."""
    rows = []
    for name in sorted(set(local) | set(host) | set(image)):
        l, h, i = local.get(name, ""), host.get(name, ""), image.get(name, "")
        if l == h == i and l:
            state = "CLEAN"
        elif l == h and l and i != l:
            state = "stale_image"
        elif l == i and l and h != l:
            state = "host_behind"
        elif h == i and h and l != h:
            state = "local_behind"
        elif not l:
            state = "missing_local"
        elif not h:
            state = "missing_host"
        elif not i:
            state = "missing_image"
        else:
            state = "THREE_WAY_DRIFT"
        rows.append({"file": name, "local": l, "host": h, "image": i,
                     "state": state})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", default=True,
                        help="Report drift only (default)")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable output")
    parser.add_argument("--to-host", action="store_true",
                        help="Publish local files to the host")
    parser.add_argument("--from-host", action="store_true",
                        help="Harvest host files into the local checkout")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the container image after publishing")
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--host-root", default="/opt/agent-harness")
    parser.add_argument("--host", default=os.environ.get("AGENT_HOST", ""))
    parser.add_argument("--image", default="agent-harness:1.0.0")
    args = parser.parse_args(argv)

    if args.to_host and args.from_host:
        print("ERROR: --to-host and --from-host are mutually exclusive",
              file=sys.stderr)
        return 1
    if not args.host:
        print("ERROR: --host or AGENT_HOST is required", file=sys.stderr)
        return 1

    root = Path(args.local_root)
    manifest = manifest_from_dockerfile(root)
    local = compare_local(root, manifest)
    host = compare_host(args.host_root, manifest) if args.host else {}
    image = compare_image(args.image, manifest)
    rows = verdict(local, host, image)

    if args.json:
        print(json.dumps({"manifest": manifest, "files": rows}, indent=2))
        return 0

    drift = [r for r in rows if r["state"] != "CLEAN"]
    clean = [r for r in rows if r["state"] == "CLEAN"]

    print(f"compared {len(manifest)} source file(s) across checkout / host / image")
    if clean:
        print(f"  CLEAN: {len(clean)} files agree everywhere")
    for state, label in [
        ("stale_image", "image does not match -- rebuild"),
        ("host_behind", "host does not match -- publish with --to-host"),
        ("local_behind", "checkout is behind -- harvest with --from-host"),
        ("missing_local", "absent from checkout"),
        ("missing_host", "absent from host"),
        ("missing_image", "absent from image"),
        ("THREE_WAY_DRIFT", "all three disagree -- resolve manually"),
    ]:
        group = [r for r in drift if r["state"] == state]
        if group:
            print(f"  DRIFT {state} ({len(group)}): "
                  f"{', '.join(r['file'] for r in group[:5])}")
            if len(group) > 5:
                print(f"    ... and {len(group) - 5} more")

    if not drift:
        print("CLEAN: all three agree")
        return 0

    if args.to_host:
        changed = [r for r in drift if r["state"] in
                   ("stale_image", "host_behind", "missing_host")]
        if changed:
            print(f"\npublishing to host: "
                  f"{', '.join(r['file'] for r in changed)}")
            for r in changed:
                src = root / r["file"]
                dst = os.path.join(args.host_root, r["file"])
                subprocess.run(["scp", str(src), f"{args.host}:{dst}"],
                               check=True, timeout=30)
            print("  the host changed; the IMAGE has NOT. "
                  "Run --rebuild or the fix is dead.")
            if args.rebuild:
                print("  rebuilding image...")
                subprocess.run(
                    ["ssh", args.host,
                     f"cd {args.host_root} && sudo docker build -q -t "
                     f"{args.image} ."],
                    check=True, timeout=300,
                )
                print("  image rebuilt")
    elif args.from_host:
        changed = [r for r in drift if r["state"] in
                   ("local_behind", "missing_local")]
        if changed:
            print(f"\nharvesting from host: "
                  f"{', '.join(r['file'] for r in changed)}")
            for r in changed:
                src = os.path.join(args.host_root, r["file"])
                dst = root / r["file"]
                subprocess.run(["scp", f"{args.host}:{src}", str(dst)],
                               check=True, timeout=30)
            print("  harvested. Review with git diff before committing.")
    else:
        print("\nRun --to-host to publish, or --from-host to harvest.")
        return 1 if drift else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
