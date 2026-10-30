"""Safety rails: classify shell actions, gate the irreversible ones.

Copilot-style agents fail loudly in one place only: a model that was asked to
"fix everything" runs ``rm -rf``, ``git reset --hard`` or ``DROP TABLE`` and
the damage is not recoverable from the transcript. This module turns that
class of accident into a REFUSAL with an explanation, so the model can choose
a reversible route -- and so an unattended run prints the exact command a
human should run later instead of running it.

Three layers, cheapest first:

1. ``classify_command`` -- pure pattern matching, no I/O, unit-testable.
   Returns ``(risk, reason)`` where risk is one of ``ok``, ``caution``,
   ``irreversible``.
2. ``guard_command`` -- the enforcement point ``run_command`` calls. An
   irreversible command needs ``approve=True`` from the model AND either
   ``AGENT_AUTO_APPROVE`` (a trusted unattended run) or an interactive
   confirmation. Anything else is refused and the refusal text tells the
   model what to do instead.
3. ``git_guard`` -- git verbs get their own gate because ``git reset --hard``
   looks innocuous next to ``rm -rf`` and destroys strictly more work.

The pattern lists are deliberately conservative: a false positive costs one
confirmation, a false negative costs the repository.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import config

#: Commands that destroy data with no undo. Matching is on the WHOLE command
#: text (lower-cased) so a pipeline that ends in ``rm -rf`` is caught too.
IRREVERSIBLE_PATTERNS = (
    (r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+", "recursive/forced file deletion"),
    (r"\brm\s+-[a-z]*r", "recursive file deletion"),
    (r"\bshred\b", "secure file destruction"),
    (r"\bmkfs\b", "filesystem format"),
    (r"\bdd\s+if=", "raw device write"),
    (r"\bdrop\s+(table|database|schema|view)\b", "SQL DROP"),
    (r"\btruncate\s+table\b", "SQL TRUNCATE"),
    (r"\bdelete\s+from\b(?![^\n;]*\bwhere\b)", "SQL DELETE without WHERE"),
    (r"\bgit\s+push\b[^\n|;&]*--force", "force push"),
    (r"\bgit\s+push\b[^\n|;&]*-f\b", "force push"),
    (r"\bgit\s+reset\s+--hard", "hard reset discards work"),
    (r"\bgit\s+clean\s+-[a-z]*f", "git clean discards untracked files"),
    (r"\bgit\s+branch\s+-[a-z]*D\b", "force branch delete"),
    (r"\bgit\s+checkout\s+\.\s*$", "checkout . discards all edits"),
    (r"\bgit\s+restore\s+\.\s*$", "restore . discards all edits"),
    (r"\bdocker\s+(rm|rmi)\b[^\n|;&]*(-[a-z]*f|--force|-v)", "docker force/volume removal"),
    (r"\bdocker\s+volume\s+rm\b", "docker volume deletion"),
    (r"\bdocker\s+system\s+prune\b", "docker system prune"),
    (r"\bkubectl\s+delete\b", "kubernetes resource deletion"),
    (r"\bgsutil\s+rm\b|\bgcloud\s+storage\s+rm\b", "cloud storage deletion"),
    (r"\bgcloud\s+projects\s+delete\b", "project deletion"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b", "host shutdown"),
    (r">\s*/dev/(sd|nvme|hd)[a-z]", "write to a raw disk device"),
)

#: Commands that are safe but worth a second look: they change state in ways
#: a monitoring agent should narrate rather than do silently.
CAUTION_PATTERNS = (
    (r"\bgit\s+commit\b", "creates a commit"),
    (r"\bgit\s+push\b", "publishes commits"),
    (r"\bpip\s+install\b|\bnpm\s+install\b|\bapt(-get)?\s+install\b",
     "installs packages"),
    (r"\bdocker\s+(compose\s+)?(up|down|restart|build)\b", "changes running services"),
    (r"\bsystemctl\s+(start|stop|restart|enable|disable)\b", "changes services"),
    (r"\bcrontab\b", "changes scheduled jobs"),
)

_IRREVERSIBLE = [(re.compile(p, re.IGNORECASE), why)
                 for p, why in IRREVERSIBLE_PATTERNS]
_CAUTION = [(re.compile(p, re.IGNORECASE), why)
            for p, why in CAUTION_PATTERNS]


def classify_command(command: str) -> Tuple[str, str]:
    """(risk, reason) for a shell command: 'irreversible' > 'caution' > 'ok'.

    Order matters: ``git push --force`` is both a push (caution) and a force
    push (irreversible); the stronger verdict must win or the gate becomes
    decoration.
    """
    text = (command or "").strip()
    if not text:
        return "ok", ""
    for rx, why in _IRREVERSIBLE:
        if rx.search(text):
            return "irreversible", why
    for rx, why in _CAUTION:
        if rx.search(text):
            return "caution", why
    return "ok", ""


def guard_command(command: str, approve: bool = False,
                  interactive: Optional[bool] = None) -> Tuple[bool, str]:
    """May this command run? Returns (allowed, explanation).

    ``approve`` is the model's explicit intent; it is necessary but not
    sufficient. Unattended runs set AGENT_AUTO_APPROVE only when the operator
    has accepted that the agent may destroy things; interactive runs ask a
    human. Everything else is refused with the reason and the suggested
    reversible alternative, which is what the model needs to recover.
    """
    risk, why = classify_command(command)
    if risk == "ok":
        return True, ""
    if risk == "caution":
        return True, f"[caution: {why}]"
    # irreversible
    if not approve:
        return False, (
            f"REFUSED (irreversible: {why}). This command was not run. If it "
            f"is genuinely required, call run_command again with "
            f"approve=true and a one-line reason, or report the exact command "
            f"for a human to run manually.")
    if interactive is None:
        interactive = not config.AUTO_APPROVE and sys_stdin_is_tty()
    if config.AUTO_APPROVE:
        return True, f"[auto-approved irreversible: {why}]"
    if interactive:
        print(f"\n[IRREVERSIBLE] {why}\n  {command}")
        answer = input("run it anyway? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return True, f"[approved interactively: {why}]"
        return False, f"DECLINED by the operator ({why}). Nothing was run."
    return False, (
        f"REFUSED (irreversible: {why}) and no approval path is available in "
        f"this unattended run (set AGENT_AUTO_APPROVE=1 to allow, or run the "
        f"command manually).")


def sys_stdin_is_tty() -> bool:
    try:
        import sys
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:  # noqa: BLE001
        return False


def git_guard(args: str) -> Tuple[bool, str]:
    """Gate destructive git verbs behind AGENT_ALLOW_DESTRUCTIVE_GIT."""
    risk, why = classify_command("git " + args)
    if risk != "irreversible":
        return True, ""
    if config.ALLOW_DESTRUCTIVE_GIT:
        return True, f"[destructive git allowed by config: {why}]"
    return False, (
        f"REFUSED destructive git ({why}). Set AGENT_ALLOW_DESTRUCTIVE_GIT=1 "
        f"only if losing this work is acceptable, or run it manually.")


def failure_note(path: str, failures: int) -> str:
    """What to say when a file keeps failing: suggest rollback, not retry #4."""
    if failures < config.MAX_FILE_FAILURES:
        return ""
    return (
        f"\n[rollback] {path} has failed {failures} consecutive times. Stop "
        f"editing it and restore the last known-good version: "
        f"git restore -- {path}  (or git checkout HEAD -- {path}). Report the "
        f"failure instead of a fourth attempt.")
