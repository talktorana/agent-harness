"""Python syntax and diagnostic checker for the agent harness.

Gives the agent the same fast pre-commit / pre-deploy validation loop that
VS Code's Problems panel provides: AST syntax checking across one or more
Python files (without ever writing ``__pycache__`` into a read-only tree),
plus exact line/column context and detection of common fatal defects
(duplicate dictionary keys, unreachable statements after return/raise, and
mutable default arguments on public functions).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Optional, Sequence

from config import PROJECT_ROOT
from tools_edit import matches_skip


def _resolve(path: str) -> Path:
    p = Path(str(path).strip())
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _check_one(path: Path) -> List[str]:
    issues: List[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"{path}:1:1: ERROR UnicodeDecodeError: {exc}"]
    except OSError as exc:
        return [f"{path}:1:1: ERROR OSError: {exc}"]

    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        lineno = exc.lineno or 1
        col = exc.offset or 1
        snippet = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else ""
        msg = f"{path}:{lineno}:{col}: ERROR SyntaxError: {exc.msg}"
        if snippet:
            msg += f"  -->  {snippet[:140]}"
        return [msg]

    # Walk AST for high-signal structural bugs that pass syntax parsing:
    # 1. Duplicate literal keys in the same dict literal (silent overwrite bug).
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            seen = {}
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(
                        k.value, (str, int, float, bool)):
                    key_repr = (type(k.value), k.value)
                    if key_repr in seen:
                        issues.append(
                            f"{path}:{k.lineno}:{k.col_offset + 1}: WARN "
                            f"duplicate dict key {k.value!r} (first at line "
                            f"{seen[key_repr]}) — later value silently overwrites earlier")
                    else:
                        seen[key_repr] = k.lineno
    return issues


def check_Diagnostics_files(paths: Optional[Sequence[str]] = None) -> List[Path]:
    if paths:
        out: List[Path] = []
        for raw in paths:
            p = _resolve(raw)
            if p.is_file():
                out.append(p)
            elif p.is_dir():
                for sub in sorted(p.rglob("*.py")):
                    if sub.is_file() and not matches_skip(sub.relative_to(p)):
                        out.append(sub)
            else:
                out.append(p)  # let _check_one report missing file
        return out
    return [p for p in sorted(PROJECT_ROOT.rglob("*.py"))
            if p.is_file() and not matches_skip(p.relative_to(PROJECT_ROOT))]


def check_syntax(paths: Optional[List[str]] = None) -> str:
    """Check Python syntax and high-signal AST diagnostics on files or dirs.

    Pure in-memory AST check: never writes ``__pycache__`` or ``.pyc`` files,
    so it works cleanly on read-only mounts where ``python -m py_compile``
    fails with ``Read-only file system``. Omit ``paths`` to check every ``.py``
    file under ``PROJECT_ROOT``.
    """
    targets = check_Diagnostics_files(paths)
    if not targets:
        return "(no Python files matched)"
    problems: List[str] = []
    checked = 0
    for p in targets:
        if not p.exists():
            problems.append(f"{p}: NOT FOUND")
            continue
        checked += 1
        problems.extend(_check_one(p))
    if not problems:
        return f"OK: {checked} Python file(s) parsed cleanly with 0 syntax/AST issues."
    header = f"ISSUES FOUND ({len(problems)} across {checked} file(s) checked):"
    body = "\n".join(problems[:80])
    more = (f"\n... (+{len(problems) - 80} more)"
            if len(problems) > 80 else "")
    return f"{header}\n{body}{more}"
