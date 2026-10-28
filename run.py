"""CLI entry point.

    python run.py "read the collector log and tell me what happened at 15:47"
    python run.py                    # interactive: type a task, get an answer
    python run.py --no-confirm ...   # let write_file apply without asking
"""

import argparse
import shutil
import sys
from pathlib import Path

import config
from agent import run_task


def _utf8_console() -> None:
    """Write to the terminal without dying on a character it cannot encode.

    Windows defaults to cp1252, so an answer containing an arrow or a box-
    drawing character raised UnicodeEncodeError AFTER the task had already
    completed -- the work was done and the result thrown away. Reconfigure
    stdout/stderr to UTF-8, replacing anything still unencodable.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass



def memory_transfer(export_to: str, import_from: str) -> int:
    """Move the memory tree between machines.

    Memory is just markdown files, but on a fresh host there is no editor
    memory tree to discover -- so without this the agent would start each new
    box knowing nothing, which is exactly the regression it exists to prevent.
    """
    src = Path(import_from).expanduser() if import_from else Path(config.MEMORY_ROOT)
    dst = Path(export_to).expanduser() if export_to else Path(config.MEMORY_ROOT)

    if import_from:
        if not src.is_dir():
            print(f"FAIL: no memory tree at {src}")
            return 2
        if dst.exists():
            bak = dst.with_name(dst.name + ".bak")
            if bak.exists():
                shutil.rmtree(bak)
            dst.rename(bak)
            print(f"existing tree kept at {bak}")
        shutil.copytree(src, dst)
        print(f"IMPORTED {src} -> {dst}")
    else:
        if not src.is_dir():
            print(f"FAIL: no memory tree at {src}")
            return 2
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"EXPORTED {src} -> {dst}")

    # Report what actually landed in the destination, not the tree we read
    # from -- after an import the original paths no longer describe reality.
    repo_dir = dst / "repo"
    if repo_dir.is_dir():
        for p in sorted(repo_dir.glob("*.md")):
            try:
                n = len(p.read_text(encoding="utf-8",
                                    errors="replace").splitlines())
            except OSError:
                n = -1
            print(f"  repo/{p.name}: {n} lines")
    else:
        print("  (no repo/ notes in the destination)")
    print("\nCommit the destination, or mount it as a volume in a container, "
          "so it survives the next deploy.")
    return 0


def check_connection() -> int:
    """Verify the endpoint and key BEFORE running a task.

    A 401 at the end of a run is the worst way to learn your key is wrong --
    you have already written a prompt and waited. This asks the API a cheap
    question first and says exactly what is broken.
    """
    from openai import OpenAI

    print(f"base   : {config.BASE_URL}")
    print(f"model  : {config.MODEL}")
    print(f"key    : {'set (' + str(len(config.API_KEY)) + ' chars)' if config.API_KEY else 'MISSING'}"
          f"   [from {config.source_of('AGENT_API_KEY')}]")
    print(f"root   : {config.PROJECT_ROOT}")
    print(f"model from: {config.source_of('AGENT_MODEL')}")
    print(f"root  from: {config.source_of('AGENT_PROJECT_ROOT')}")
    print()

    if not config.API_KEY:
        print("FAIL: AGENT_API_KEY is empty. Put it in agent-harness/.env.")
        return 2

    shadowed = [k for k, v in config.SOURCES.items() if "shadows" in v]
    if shadowed:
        print("WARNING: exported variables are overriding your .env:")
        for k in shadowed:
            print(f"   {k}: {config.SOURCES[k]}")
        print("  An exported value from an old session will win silently. "
              "unset it if that is not what you meant.\n")

    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)

    # 1. What can be reached. NOTE: on this provider /models is NOT an auth
    # gate -- a deliberately bogus key listed all 94 models -- so passing here
    # proves nothing about the key. It only names what is available.
    try:
        models = client.models.list()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not reach {config.BASE_URL} -> "
              f"{type(exc).__name__}: {exc}")
        return 2
    ids = [getattr(m, "id", "") for m in models.data]
    print(f"OK: endpoint reachable, {len(ids)} models listed.")
    print("    (listing models does NOT prove the key -- see below.)")

    # 2. The ONLY thing that proves the key is a real completion.
    print("\nprobing inference with a one-token completion...")
    try:
        probe = client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user",
                       "content": "Reply with the single word OK."}],
            max_tokens=32,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: inference rejected the request -> "
              f"{type(exc).__name__}: {exc}")
        print()
        if "401" in str(exc) or "UNAUTHORIZED" in str(exc):
            print("  The KEY was rejected by " + config.BASE_URL + ".")
            print(f"  It came from: {config.source_of('AGENT_API_KEY')}")
            print("  Likely causes:")
            print("   - a stale AGENT_API_KEY exported in this shell is "
                  "shadowing the one in .env (see the WARNING above)")
            print("   - the key belongs to a different provider than the base URL")
            print("   - the key was copied incompletely or has been revoked")
            print("   - a trailing space or quote got into the value")
            print("  Fix: unset AGENT_API_KEY, or put the right key in "
                  "agent-harness/.env, then re-run --check.")
        else:
            print("  The key may be fine. If this was a model error, check "
                  "the model name below.")
        return 2
    said = (probe.choices[0].message.content or "").strip()
    used = getattr(probe, "model", config.MODEL)
    print(f"OK: the key is accepted. probe replied {said!r} "
          f"(served by {used})")

    # 3. Is the configured model one of the listed ones?
    if config.MODEL in ids:
        print(f"OK: model '{config.MODEL}' is available.")
        return 0

    print(f"\nWARN: model '{config.MODEL}' is NOT in the list.")
    # Show the closest matches so the fix is obvious.
    stem = config.MODEL.split("/")[-1].lower().split("-")[0]
    near = [i for i in ids if stem and stem in i.lower()][:15]
    if near:
        print("      Models matching '" + stem + "':")
        for i in near:
            print("        " + i)
    else:
        print("      First 20 available models:")
        for i in ids[:20]:
            print("        " + i)
    print("\n  Pick one and set AGENT_MODEL (or use --model).")
    return 1


def main() -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(
        description="Local LLM agent harness with shell/file/git tools.")
    parser.add_argument("task", nargs="*",
                        help="The task. If omitted, you are prompted.")
    parser.add_argument("--check", action="store_true",
                        help="Prove the key and model work with a real "
                             "completion, then exit.")
    parser.add_argument("--memory-export", metavar="DIR",
                        help="Copy the memory tree to DIR, then exit. Use this "
                             "to carry what has been learned to another host.")
    parser.add_argument("--memory-import", metavar="DIR",
                        help="Copy a memory tree from DIR over the current one, "
                             "then exit. The existing tree is kept as .bak.")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Apply write_file without asking (use with care).")
    parser.add_argument("--model", help="Override AGENT_MODEL.")
    parser.add_argument("--base-url", help="Override AGENT_BASE_URL.")
    parser.add_argument("--steps", type=int, help="Override AGENT_MAX_STEPS.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the final answer.")
    args = parser.parse_args()

    if args.model:
        config.MODEL = args.model
    if args.base_url:
        config.BASE_URL = args.base_url
    if args.steps:
        config.MAX_STEPS = args.steps
    if args.no_confirm:
        config.CONFIRM_WRITES = False

    if args.check:
        return check_connection()

    if args.memory_export or args.memory_import:
        return memory_transfer(args.memory_export, args.memory_import)

    if not config.API_KEY:
        print("AGENT_API_KEY is not set. Export it or put it in .env before "
              "running. Try `python run.py --check` to verify.")
        return 2

    task = " ".join(args.task).strip()
    if not task:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not task:
        return 1

    print(f"\033[90mmodel={config.MODEL}  base={config.BASE_URL}  "
          f"root={config.PROJECT_ROOT}  max_steps={config.MAX_STEPS}\n"
          f"key from {config.source_of('AGENT_API_KEY')}  "
          f"(run --check to prove it works)\033[0m\n")

    answer = run_task(task, verbose=not args.quiet)
    print("\n" + "=" * 72)
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
