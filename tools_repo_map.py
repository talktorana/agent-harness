"""Structural repo map: the codebase's shape without reading all of it.

Copilot answers "where is X?" instantly because the editor indexes the
workspace. A terminal agent has to earn that knowledge, usually by burning
steps on ``ls``/``grep``. This module gives it the same first glance for
free: every class and function signature in the project, bounded to a few
thousand characters, injected into the system prompt at task start and
available on demand via the ``repo_map`` tool.

Python files get a real AST outline (accurate signatures, methods nested
under their class). Everything else gets a conservative regex outline --
wrong-but-harmless beats absent, and the map always says it is a heuristic.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional

from config import PROJECT_ROOT, REPO_MAP_MAX_CHARS
from tools_edit import matches_skip

_PY = ".py"
_REGEX_OUTLINE = re.compile(
    r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+)?"
    r"(class|interface|struct|enum|type|func|function|def|sub)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE)


def _python_outline(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef,)):
            bases = ", ".join(_name(b) for b in node.bases)
            out.append(f"class {node.name}({bases})" if bases
                       else f"class {node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"    {_signature(sub)}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_signature(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    out.append(f"{target.id} = ...")
    return out


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _signature(node) -> str:
    args = []
    a = node.args
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, default in zip(a.args, defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_name(arg.annotation) or '?'}"
        if default is not None:
            text += "=..."
        args.append(text)
    if a.vararg:
        args.append(f"*{a.vararg.arg}")
    if a.kwarg:
        args.append(f"**{a.kwarg.arg}")
    ret = f" -> {_name(node.returns)}" if node.returns is not None else ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(args)}){ret}"


def _regex_outline(source: str) -> List[str]:
    return [f"{kind} {name}" for kind, name in _REGEX_OUTLINE.findall(source)]


def outline_file(path: Path) -> List[str]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if path.suffix == _PY:
        return _python_outline(source)
    return _regex_outline(source)


def build_repo_map(root: Optional[str] = None,
                   max_chars: Optional[int] = None) -> str:
    """A bounded structural overview of the project, newest-relevant first.

    Files are walked in sorted order, skipping VCS/cache/vendor directories.
    The map stops at ``max_chars`` and says so -- a truncated map that admits
    it is truncated beats a silent one that looks complete.
    """
    base = Path(root) if root else PROJECT_ROOT
    limit = int(max_chars or REPO_MAP_MAX_CHARS)
    blocks: List[str] = []
    used = 0
    truncated = False
    for path in sorted(base.rglob("*")):
        if not path.is_file() or matches_skip(path):
            continue
        if path.suffix not in (".py", ".js", ".ts", ".tsx", ".go", ".rs",
                               ".java", ".cs", ".rb", ".php", ".sh", ".ps1"):
            continue
        lines = outline_file(path)
        if not lines:
            continue
        rel = path.relative_to(base).as_posix()
        block = f"## {rel}\n" + "\n".join(lines) + "\n"
        # The FIRST block always lands: a map that says "empty" because one
        # file exceeded the cap would be a lie about the codebase.
        if blocks and used + len(block) > limit:
            truncated = True
            break
        blocks.append(block)
        used += len(block)
    if not blocks:
        return "(repo map empty: no outlineable source files found)"
    note = ("\n... [map truncated at ~%d chars; use glob_files and search "
            "for the rest]" % limit) if truncated else ""
    return "".join(blocks) + note


def repo_map(path: str = ".", max_chars: int = 0) -> str:
    """On-demand structural overview of a subtree (classes, functions,
    signatures). Use it BEFORE reading files to decide which files matter;
    it is far cheaper than reading them all."""
    target = _abs(path)
    if not target.exists():
        return f"NOT FOUND: {target}"
    return build_repo_map(str(target), max_chars or None)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / path)
