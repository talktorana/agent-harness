"""Live run settings an operator can change from the console: cadence and ceiling.

WHY THIS MODULE EXISTS
----------------------
Two numbers decide how an unattended mission behaves, and both used to be
changeable only by editing a declaration file and recreating the container:

* ``every_min`` -- how often the scheduled mission fires;
* the step ceiling -- how many turns a run may take before it is cut off.

Both are the kind of thing an operator tunes while WATCHING: a pass keeps dying
at the ceiling, or the cadence is too tight for how long passes actually take.
Making that a redeploy means the person who noticed cannot fix it, and the fix
itself needs a recreate -- which kills a run in flight, so the natural instinct
is to hand-roll the recreate and risk exactly that.

So these are read at USE time from a small override file the process can write,
in the same place and for the same reason as the model choice
(``config.MODEL_CHOICE_FILE``): the declarations an operator reviews (compose
env, schedule file) stay read-only, and anything chosen from the console lands
somewhere the console may write and that is read on every job.

The override does NOT rewrite the declaration file. That matters: the schedule
declaration is reviewed configuration (it carries the mission text), and an
agent that can edit its own brief owns its own guardrails. This module changes
behaviour without touching it, and every reader reports WHERE a value came from
so the difference is visible instead of implied.

PRECEDENCE, and why the console wins over the environment
---------------------------------------------------------
console override -> environment -> built-in default. The environment is what a
deployment ships and reviews; the override is what a human just decided while
looking at the page. Shipping defaults are in ``config`` so provenance still
names them; the override is the only layer that can change at runtime.

VALUES ARE VALIDATED, NOT CLAMPED. A ceiling of 0 stops every run on turn one;
an ``every_min`` of 0 fires every poll, which is a token-burning loop with no
mission value. Both are rejected with a message rather than quietly adjusted:
a clamped value looks applied while doing something else, and the operator who
typed 5000 has no way to learn it became 1000.

Run:  python -m unittest test_run_settings
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import config

#: Both files live in the sessions directory -- the one place the process is
#: expected to write, mounted read-write, and wiped by nothing.
SETTINGS_FILE = Path(os.environ.get(
    "AGENT_RUN_SETTINGS_FILE", str(config.SESSIONS_DIR / "run_settings.json")))

#: Built-in floors and ceilings. These are sanity rails, not policy: the
#: deployment's env sets the DEFAULT (``config.MAX_STEPS``,
#: ``EVERY_MIN_DEFAULT``), and this only defines the range a console may ask
#: for. A ceiling below the floor is not a tuning choice, it is a mistake.
MAX_STEPS_MIN = 1
MAX_STEPS_MAX = 4000
EVERY_MIN_MIN = 1
EVERY_MIN_MAX = 60 * 24 * 7          # a week; beyond that, use `at`


class SettingsError(Exception):
    """A setting could not be read or was not acceptable."""


def _read() -> Dict[str, Any]:
    """The override file's contents. A corrupt file is an error, never empty.

    Returning ``{}`` on a parse failure would silently restore the shipped
    default, and an operator who set a ceiling of 200 would watch runs die at
    40 with nothing anywhere saying why. Surfacing it lets the console show the
    problem and the operator delete or repair the file.
    """
    path = SETTINGS_FILE
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SettingsError(
            f"{path} is not readable JSON: {exc}. Delete it to fall back to the "
            f"environment, or save a valid value in the console.") from exc
    if not isinstance(data, dict):
        raise SettingsError(f"{path} must hold a JSON object.")
    return data


def _validate_int(raw: Any, name: str, low: int, high: int) -> int:
    """Parse, range-check and return an integer setting, or raise.

    Booleans are rejected explicitly: ``isinstance(True, int)`` is True in
    Python, so a JSON ``true`` would otherwise arrive as the ceiling value 1
    and every run would stop after a single turn.
    """
    if isinstance(raw, bool):
        raise SettingsError(f"{name} must be an integer, not true/false.")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise SettingsError(f"{name} must not be empty.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"{name} must be a whole number (got {raw!r}).") from exc
    # int("40.5") raises, but int(40.5) does not -- and a float arriving from
    # JSON means somebody typed a decimal into a step count.
    if isinstance(raw, float) and float(raw) != value:
        raise SettingsError(f"{name} must be a whole number (got {raw!r}).")
    if value < low or value > high:
        raise SettingsError(
            f"{name} must be between {low} and {high} (got {value}). A value "
            f"outside that range is not a tuning choice: a ceiling of 0 ends "
            f"every run immediately and an interval of 0 fires on every poll.")
    return value


def _write(payload: Dict[str, Any]) -> None:
    """Persist atomically: a half-written settings file would wedge every run."""
    path = SETTINGS_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".run_settings-",
                                   dir=str(path.parent))
    except OSError as exc:
        raise SettingsError(f"cannot write {path}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except OSError as exc:
        # Reported, not swallowed: a save the console believes succeeded but
        # that never landed is worse than a refusal -- the page says "applied",
        # the value is not in force, and the next run dies at the old ceiling.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise SettingsError(f"cannot write {path}: {exc}") from exc


def effective_max_steps() -> Tuple[int, str]:
    """The ceiling in force, and where it came from.

    Read on every call, so a value saved in the console applies to the NEXT
    run without a recreate. ``source`` is one of ``console`` / ``environment``
    / ``default`` -- the page has to be able to say "the shipped default is in
    force" rather than showing a saved value as if it were live.
    """
    try:
        data = _read()
    except SettingsError:
        raise
    raw = data.get("max_steps")
    if raw is not None:
        return _validate_int(raw, "max_steps", MAX_STEPS_MIN, MAX_STEPS_MAX), "console"
    # ``config.source_of`` is this repo's own provenance record: it distinguishes
    # an exported variable, a specific .env file, and the built-in default, which
    # is exactly the distinction the page needs to make ("your container env sets
    # 40" is a different statement from "nobody set this").
    if config.source_of("AGENT_MAX_STEPS") != "built-in default":
        return config.MAX_STEPS, "environment"
    return config.MAX_STEPS, "default"


def effective_every_min(default: Any = None) -> Tuple[Optional[float], str]:
    """The cadence in force, and where it came from.

    ``default`` is the value declared for the job being reported, so a
    declaration and an override can be shown side by side. Returns ``(None,
    "none")`` when nothing at all specifies an interval (a job that fires on an
    ``at`` schedule rather than an interval).
    """
    data = _read()
    raw = data.get("every_min")
    if raw is not None:
        return _validate_int(raw, "every_min", EVERY_MIN_MIN, EVERY_MIN_MAX), "console"
    if default is not None:
        try:
            return float(default), "schedule"
        except (TypeError, ValueError):
            return None, "schedule"
    return None, "none"


def effective_answer_warn_steps(max_steps: Optional[int] = None) -> int:
    """The wind-down warning distance that goes with a ceiling.

    ``ANSWER_WARN_STEPS`` fires ONCE (the loop compares
    ``remaining == warn``), so the warning window is the run's only chance to
    stop investigating and write its answer. A fixed 3 was chosen for a 40-turn
    budget; scaled to a 200-turn ceiling that is 1.5% of the run -- the model is
    still deep in a tool loop when the one warning arrives and cannot reach an
    answer in time, which is the ``step_limit_with_answer``-with-nothing-useful
    failure this deployment hit repeatedly.

    So it scales with the ceiling, ~13%, and never below the configured
    minimum. It is derived rather than stored: an operator who sets a ceiling
    should not also have to know the warning distance exists, and a stored pair
    goes stale the moment only one of them is changed.

    The floor and the ceiling collide at tiny budgets, and the ceiling wins:
    with 1 turn available, a warning distance of 20 is not "warned early", it
    is never warned at all -- the loop compares ``remaining == warn`` and
    ``remaining`` never reaches 20, so the run is cut off with no warning and
    no answer. Capping below the ceiling guarantees the equality is reachable
    for every valid ceiling, which is the whole point of the warning existing.
    """
    ceiling = max_steps if max_steps is not None else effective_max_steps()[0]
    ceiling = max(1, int(ceiling))
    if ceiling <= 2:
        # No room for "some work, then a warning". A distance of 1 at a ceiling
        # of 1 could only be reached at remaining==1, i.e. before the first turn
        # has run, so the run would be told to stop before it started. Returning
        # 0 is honest: the equality can never fire, the run simply ends at its
        # limit, and the caller's own ``max(1, ...)`` keeps that comparison safe.
        return ceiling - 1
    scaled = int(round(ceiling * config.ANSWER_WARN_STEPS / max(config.MAX_STEPS, 1)))
    warned = max(int(config.ANSWER_WARN_STEPS), scaled)
    # Leave at least one turn of actual work before the warning, so a run is
    # never warned on its first turn for its whole life.
    return min(warned, ceiling - 1)


def save(max_steps: Any = None, every_min: Any = None) -> Dict[str, Any]:
    """Validate and persist one or both settings. Returns what was written.

    Both are validated BEFORE anything is written, so a request carrying one
    good value and one bad one changes nothing rather than half-applying.
    """
    data = _read()
    if max_steps is not None:
        data["max_steps"] = _validate_int(max_steps, "max_steps",
                                         MAX_STEPS_MIN, MAX_STEPS_MAX)
    if every_min is not None:
        data["every_min"] = _validate_int(every_min, "every_min",
                                         EVERY_MIN_MIN, EVERY_MIN_MAX)
    if not data:
        raise SettingsError("nothing to save: pass max_steps and/or every_min.")
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    _write(data)
    return data


def clear(what: Optional[str] = None) -> Dict[str, Any]:
    """Drop one override, or all of them, and return the remaining file.

    Removing a single key (rather than deleting the file) is why ``what`` is a
    name and not a boolean: an operator tuning the ceiling should not also lose
    a cadence they set last week.
    """
    data = _read()
    if what is None:
        data = {}
    elif what in ("max_steps", "every_min"):
        data.pop(what, None)
    else:
        raise SettingsError(f"unknown setting {what!r}; expected max_steps, "
                            f"every_min, or nothing for both.")
    data.pop("saved_at", None)
    if data:
        data["saved_at"] = datetime.now().isoformat(timespec="seconds")
        _write(data)
    else:
        try:
            if SETTINGS_FILE.exists():
                SETTINGS_FILE.unlink()
        except OSError as exc:
            raise SettingsError(f"cannot remove {SETTINGS_FILE}: {exc}") from exc
    return data


def state(every_declared: Any = None) -> Dict[str, Any]:
    """Everything the console needs to render the panel honestly.

    Both the live value and the declared one are included, because seeing
    "override 200 in force, environment ships 40" is the only way to tell that
    a redeploy would silently undo a console choice.

    ``every_declared`` is the cadence the schedule declares. This module owns no
    schedule -- it resolves an override, it cannot know what any job declares --
    so the caller that does (the console, which also serves /schedule) passes
    it in. Without it the panel reported the cadence as unset while the
    scheduler was merrily firing every 45 minutes from the declaration, and a
    page that misreports a live setting is a page nobody trusts for the next
    one.
    """
    out: Dict[str, Any] = {
        "settings_file": str(SETTINGS_FILE),
        "error": None,
        "limits": {"max_steps": [MAX_STEPS_MIN, MAX_STEPS_MAX],
                   "every_min": [EVERY_MIN_MIN, EVERY_MIN_MAX]},
    }
    try:
        steps, steps_source = effective_max_steps()
        every, every_source = effective_every_min(every_declared)
        out.update({
            "max_steps": steps,
            "max_steps_source": steps_source,
            "answer_warn_steps": effective_answer_warn_steps(steps),
            "every_min": every,
            "every_min_source": every_source,
            "declared_max_steps": config.MAX_STEPS,
            "declared_answer_warn_steps": config.ANSWER_WARN_STEPS,
            "declared_every_min": (float(every_declared)
                                   if isinstance(every_declared, (int, float))
                                   and not isinstance(every_declared, bool)
                                   else None),
            "saved_at": _read().get("saved_at"),
        })
    except SettingsError as exc:
        # Reported, never swallowed: the loop is about to run with a ceiling
        # nobody chose, and the page has to say so.
        out["error"] = f"{type(exc).__name__}: {exc}"
        out.update({"max_steps": config.MAX_STEPS, "max_steps_source": "error",
                    "answer_warn_steps": config.ANSWER_WARN_STEPS,
                    "every_min": None, "every_min_source": "error",
                    "declared_max_steps": config.MAX_STEPS,
                    "declared_answer_warn_steps": config.ANSWER_WARN_STEPS,
                    "declared_every_min": None})
    return out
