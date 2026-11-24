"""Generic LLM provider layer: any OpenAI-compatible endpoint, by name.

The harness used to speak to exactly one endpoint (``AGENT_BASE_URL`` +
``AGENT_API_KEY`` + ``AGENT_MODEL``). That is fine for one provider and
useless the moment you want to compare two, or pick a cheap model for a
scheduled pass and a strong one for an interactive session.

This module adds a PROVIDER CATALOGUE, shaped like the config of any
multi-provider gateway but with no gateway in between -- the harness talks
to each endpoint directly:

    {
      "default": {"provider": "example-cloud", "model": "vendor-model-flash"},
      "providers": {
        "provider": {
          "base_url": "https://api.example-cloud.test/v1",
          "api_key_env": "EXAMPLE_CLOUD_API_KEY",     # or "api_key": "sk-..."
          "models": {"vendor-model-flash": "vendor/model-flash-1017"}
        },
        ...
      }
    }

Design rules, learned from running several providers at once:

* KEYS NEVER LIVE IN THE FILE BY DEFAULT. ``api_key_env`` names an environment
  variable; ``api_key`` is accepted for containers that inject secrets as
  files, but the example file ships empty and the catalogue endpoint never
  returns key material.
* ALIAS -> REAL MODEL ID. Providers rename models without notice; the alias
  is what your prompts, scripts and muscle memory use, the real id is what
  the wire sees. One line changes when a provider renames something.
* ENVIRONMENT WINS. ``AGENT_BASE_URL``/``AGENT_API_KEY``/``AGENT_MODEL`` still
  work and override the file, so existing deployments change nothing. An
  explicit ``--providermodel`` (or the web UI's dropdowns) wins over both.
* THE CONSOLE CAN PICK THE DEFAULT. A choice saved from the UI is written to
  ``config.MODEL_CHOICE_FILE`` -- agent-writable by design, because the
  catalogue is mounted read-only. It outranks the catalogue's ``default``
  block, which outranks the environment; see :func:`default_choice`.
* RESOLUTION FAILS LOUDLY AND EARLY. A missing key or unknown alias is a
  configuration error, not a runtime surprise four tool calls in.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

#: The catalogue file, beside the harness. Absent is valid: the environment
#: provider (or the built-in defaults) still works.
PROVIDERS_FILE = Path(os.environ.get(
    "AGENT_PROVIDERS_FILE",
    str(Path(__file__).resolve().parent / "providers.json")))


def split_model_entry(entry: Any) -> Tuple[str, Dict[str, Any]]:
    """Normalise a ``models`` value into (wire_id, capabilities).

    Backward compatible: a plain string is the wire id with no capabilities.
    An object carries the wire id under ``id``/``model``/``model_id`` plus
    optional capability fields (maxInputTokens, maxOutputTokens, toolCalling,
    vision, thinking, supportsReasoningEffort, reasoningEffortFormat) that the
    request builder consumes. This is what lets a deployment declare a model's
    real limits instead of leaving them as documentation.
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        wire = str(entry.get("id") or entry.get("model")
                   or entry.get("model_id") or "").strip()
        if not wire:
            raise ProviderError(
                "a model object must name its wire id via 'id'/'model'.")
        caps = {k: v for k, v in entry.items()
                if k not in ("id", "model", "model_id")}
        return wire, caps
    raise ProviderError(f"unsupported model entry type: {type(entry).__name__}")


@dataclass(frozen=True)
class Resolved:
    """One concrete endpoint choice: where, with what key, which model id."""
    provider: str
    model: str            # the alias the user asked for
    model_id: str         # the real id on the wire
    base_url: str
    api_key: str
    #: Per-model capabilities declared in providers.json (empty for a plain
    #: string alias). Consumed by agent.build_request / compaction.
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class ProviderError(RuntimeError):
    """Configuration problem: unknown provider, or a missing key."""


def _load_file() -> Dict[str, Any]:
    if not PROVIDERS_FILE.is_file():
        return {}
    try:
        data = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderError(f"{PROVIDERS_FILE} is not readable JSON: {exc}")
    if not isinstance(data, dict):
        raise ProviderError(f"{PROVIDERS_FILE} must hold a JSON object.")
    return data


def providers() -> Dict[str, Dict[str, Any]]:
    """The catalogue's provider table, merged with the environment provider.

    The environment provider is ALWAYS present so a bare checkout works with
    nothing but three variables; the file adds named alternatives.
    """
    table: Dict[str, Dict[str, Any]] = {}
    data = _load_file()
    raw = data.get("providers") or {}
    if not isinstance(raw, dict):
        raise ProviderError("'providers' must be an object.")
    for name, pcfg in raw.items():
        if not isinstance(pcfg, dict):
            continue
        table[str(name)] = pcfg
    # The legacy single-endpoint configuration, as provider "env". It exists
    # only when an endpoint is actually configured, and it declares a model
    # ONLY when one is named -- an empty ``AGENT_MODEL`` must not become a
    # blank alias in the catalogue, and the provider still accepts a raw id
    # from the caller (see resolve()).
    if config.BASE_URL:
        table.setdefault("env", {
            "base_url": config.BASE_URL,
            "api_key_env": "AGENT_API_KEY",
            "models": ({config.MODEL: config.MODEL} if config.MODEL else {}),
        })
    return table


def _key_for(pcfg: Dict[str, Any], name: str) -> str:
    """Resolve a provider's key without ever printing it."""
    env_name = str(pcfg.get("api_key_env") or "").strip()
    if env_name:
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
        # Fall through to an inline key only when the env var is unset, so a
        # container that mounts both does not get a silent precedence bug.
    inline = str(pcfg.get("api_key") or "").strip()
    if inline and not inline.startswith(("CHANGE_ME", "your-", "sk-PLACEHOLDER")):
        return inline
    if env_name:
        raise ProviderError(
            f"provider {name!r}: environment variable {env_name} is not set "
            f"(and no usable inline api_key). Put it in the .env beside the "
            f"harness (persistent, gitignored), export it in THIS shell "
            f"(an export in another terminal does not reach this process), "
            f"or set api_key in {PROVIDERS_FILE.name}.")
    raise ProviderError(
        f"provider {name!r}: no api_key_env or api_key configured.")


def _load_choice_file() -> Optional[Tuple[str, str]]:
    """The console-saved choice, or None when nobody has saved one.

    A read failure RAISES instead of falling back. Silently reverting to the
    catalogue default when the override is corrupt is the exact failure this
    module exists to prevent: the console would keep displaying the saved
    model while scheduled missions ran on a different one, and nothing in any
    log would say so.
    """
    path = config.MODEL_CHOICE_FILE
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderError(
            f"{path} is not readable JSON: {exc}. Delete it to fall back to "
            f"{PROVIDERS_FILE.name}, or re-save a model in the console.")
    if not isinstance(data, dict):
        raise ProviderError(f"{path} must hold a JSON object.")
    provider = str(data.get("provider") or "").strip()
    model = str(data.get("model") or "").strip()
    if not provider or not model:
        # Structurally valid but empty: treat as "no choice saved" so a
        # template or a truncating editor does not brick every run.
        return None
    return provider, model


def default_choice() -> Tuple[str, str]:
    """(provider, model) used when the caller does not say.

    Precedence: the console's saved choice, then the catalogue's ``default``
    block, then the environment provider and its model. The saved choice wins
    because it is the only one an operator can change without shell access and
    a redeploy -- which is the entire point of having a dropdown.
    """
    saved = _load_choice_file()
    if saved is not None:
        return saved
    data = _load_file()
    d = data.get("default") or {}
    if isinstance(d, dict) and d.get("provider") and d.get("model"):
        return str(d["provider"]), str(d["model"])
    return "env", config.MODEL


def save_default_choice(provider: str, model: str, *,
                        probe: bool = True) -> Resolved:
    """Persist the console's choice for runs that do not name a model.

    RESOLVES BEFORE WRITING. Persisting an alias that does not exist, or a
    provider whose key is missing, would poison every later run -- including
    the unattended ones nobody is watching. A bad choice is therefore rejected
    here, while the operator is still looking at the page, and the previous
    choice stays in force. The write itself is atomic, so a crash mid-save can
    never leave a half-written file that breaks the next mission.

    AND PROVES THE MODEL CAN COMPLETE A TURN (2026-11-24). Resolving only says
    the alias and key exist. A model was promoted to default that resolved
    cleanly and then stalled on every real mission turn, surviving only by
    fallback. So the default -- the one choice every unattended mission
    inherits -- must also pass :func:`probe_turn`, which sends a real
    tool-enabled request. On failure the choice is REFUSED with the probe's
    own reason and nothing is written. ``probe=False`` skips the network call
    for callers that have already established liveness.

    Raises ``ProviderError`` when the choice is unusable.
    """
    res = resolve(provider, model)      # raises ProviderError when unusable
    if probe:
        verdict = probe_turn(res.provider, res.model)
        if not verdict.get("ok"):
            raise ProviderError(
                f"{res.label()} resolved but cannot complete an agentic turn "
                f"({verdict.get('error')}: {verdict.get('detail', '')}). "
                f"The previous default is unchanged. A model that stalls here "
                f"would time out on every mission turn.")
    path = config.MODEL_CHOICE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps({"provider": res.provider, "model": res.model,
                               "model_id": res.model_id}, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return res


#: Seconds a liveness probe may take before the model is judged stalled.
#:
#: Deliberately FAR below ``config.API_TIMEOUT_SEC`` (120 s by default). A
#: probe exists to reject a model that cannot finish a turn, and waiting the
#: full mission timeout to learn that would tie up the console for two minutes
#: per attempt. A model that needs more than this to answer one tool call is
#: not usable as an unattended default anyway.
PROBE_TIMEOUT_SEC = float(os.environ.get("AGENT_PROBE_TIMEOUT_SEC", "45"))


def probe_turn(provider: str, model: str, *,
               timeout: Optional[float] = None) -> Dict[str, Any]:
    """Can this model COMPLETE one real agentic turn? Report, never raise.

    Resolving is not the same as working. ``resolve`` proves the alias exists
    and a key is present; it says nothing about whether the model answers.
    Measured 2026-11-24: a newly added model resolved cleanly and even replied
    "ok" to a bare 32-token chat with one no-op tool, so it was promoted to
    default. Its FIRST MISSION then stalled on every real turn -- four calls,
    zero completions, each dying at the 120 s timeout -- and the run survived
    only by falling back to the previous model, paying two minutes per turn.
    The difference between the two probes was the REAL request: the full tool
    schema and a real completion budget. So that is what this sends.

    Uses :func:`agent.build_request` -- the one authority for what a mission
    actually transmits -- so the probe can never test a lighter payload than
    the one that matters. ``tool_choice`` is forced to a tool so a model that
    only chats cannot pass by ignoring tools.

    Returns a dict: ``ok`` (bool), ``seconds``, and on failure ``error`` --
    one of ``timeout`` / ``http`` / ``exception`` / ``no_tool_call`` / ``bad_response``.
    Never raises: callers report, and a probe must not be the thing that
    breaks a console page.
    """
    started = time.monotonic()
    out: Dict[str, Any] = {"provider": provider, "model": model}
    try:
        import agent as _agent  # lazy: avoids an import cycle at module load
        res = resolve(provider, model)
        out["model_id"] = res.model_id
        request = _agent.build_request(
            res.model_id,
            [{"role": "user", "content":
              "Call the manage_todo_list tool once to add a single item whose "
              "title is 'liveness probe'. Do not reply in text."}],
            use_tools=True, caps=res.capabilities)
        if not request.get("tools"):
            return {**out, "ok": False, "error": "bad_response",
                    "detail": "model declares toolCalling False; it cannot "
                              "run an agent mission"}
        request["tool_choice"] = "required"
        request["max_tokens"] = min(int(request.get("max_tokens") or 256),
                                    512)
        budget = PROBE_TIMEOUT_SEC if timeout is None else float(timeout)
        #: ``client_for`` builds with the long MISSION timeout on purpose (a
        #: reasoning model may think for minutes). The probe overrides it for
        #: THIS request only, so a stalled model is rejected in seconds.
        client = client_for(res)
        response = client.chat.completions.create(**request, timeout=budget)
        choice = (response.choices or [None])[0]
        message = getattr(choice, "message", None) if choice else None
        calls = list(getattr(message, "tool_calls", None) or []) if message else []
        if not calls:
            return {**out, "ok": False, "error": "no_tool_call",
                    "detail": "the model answered without calling a tool; it "
                              "cannot drive an agent loop"}
        function = getattr(calls[0], "function", None)
        try:
            json.loads(getattr(function, "arguments", None) or "{}")
        except (TypeError, ValueError):
            return {**out, "ok": False, "error": "bad_response",
                    "detail": "the tool call's arguments were not valid JSON"}
        return {**out, "ok": True, "tool": getattr(function, "name", None),
                "seconds": round(time.monotonic() - started, 2)}
    except Exception as exc:  # noqa: BLE001 -- a probe reports, it never raises
        name = type(exc).__name__
        error = ("timeout" if "timeout" in name.lower()
                 or "timed out" in str(exc).lower()
                 else "http" if getattr(exc, "status_code", None)
                 else "exception")
        return {**out, "ok": False, "error": error,
                "seconds": round(time.monotonic() - started, 2),
                "detail": f"{name}: {str(exc)[:240]}"}


def clear_default_choice() -> bool:
    """Forget the saved choice, falling back to the catalogue default.

    Returns True when something was actually cleared, so the caller can tell
    the operator the truth instead of reporting a no-op as success.
    """
    try:
        config.MODEL_CHOICE_FILE.unlink()
        return True
    except FileNotFoundError:
        return False


def choice_source() -> str:
    """Which layer is currently deciding the default: 'console', 'catalogue'
    (its ``default`` block), or 'environment'. Reported to the UI so the page
    can never claim a saved model is in force when it is not."""
    if _load_choice_file() is not None:
        return "console"
    d = _load_file().get("default") or {}
    if isinstance(d, dict) and d.get("provider") and d.get("model"):
        return "catalogue"
    return "environment"


def resolve(provider: Optional[str] = None,
            model: Optional[str] = None) -> Resolved:
    """Turn a (provider, model) request into a concrete endpoint.

    Precedence: explicit arguments > file default > environment provider.
    Raises ProviderError with an actionable message on unknown names or
    missing keys -- never returns a half-configured endpoint.
    """
    table = providers()
    if provider is None or model is None:
        d_provider, d_model = default_choice()
        provider = provider or d_provider
        model = model or d_model
    if provider not in table:
        known = ", ".join(sorted(table)) or "(none)"
        raise ProviderError(
            f"unknown provider {provider!r}; configured: {known}. "
            f"Add it to {PROVIDERS_FILE.name} or use --provider.")
    pcfg = table[provider]
    models = pcfg.get("models") or {}
    if not isinstance(models, dict):
        models = {}
    capabilities: Dict[str, Any] = {}
    if model in models:
        model_id, capabilities = split_model_entry(models[model])
    elif any(split_model_entry(v)[0] == model for v in models.values()):
        # A real id typed straight in is accepted: convenient for one-offs
        # and for providers whose alias table is empty by choice. Recover the
        # capabilities of whichever alias maps to that wire id.
        model_id = model
        for v in models.values():
            wire, caps = split_model_entry(v)
            if wire == model:
                capabilities = caps
                break
    elif provider == "env":
        # The environment provider is the legacy escape hatch: a raw model id
        # on AGENT_BASE_URL must keep working without being listed anywhere.
        # Checked BEFORE the "declares no models" refusal below, because an
        # env endpoint may legitimately declare none -- that is the whole point
        # of a raw-id escape hatch, and the generic config ships no default
        # model to put in the table.
        model_id = model
    elif not models:
        raise ProviderError(f"provider {provider!r} declares no models.")
    else:
        known = ", ".join(sorted(models))[:400]
        raise ProviderError(
            f"provider {provider!r} has no model {model!r}; its models: "
            f"{known}.")
    base_url = str(pcfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ProviderError(f"provider {provider!r} has no base_url.")
    return Resolved(provider=provider, model=model, model_id=model_id,
                    base_url=base_url, api_key=_key_for(pcfg, provider),
                    capabilities=capabilities)


def client_for(res: Resolved):
    """An OpenAI-compatible client for one resolved endpoint.

    Built per call rather than cached: a long-lived scheduler process must
    pick up a rotated key or an edited providers.json on the next task
    without a restart.
    """
    from openai import OpenAI
    return OpenAI(base_url=res.base_url, api_key=res.api_key,
                  timeout=config.API_TIMEOUT_SEC, max_retries=0)


def catalog() -> Dict[str, Any]:
    """What the web UI offers: providers, their model aliases, the default.

    Deliberately key-free -- this is served over HTTP.
    """
    table = providers()
    out: Dict[str, Any] = {"providers": {}, "default": list(default_choice())}
    for name, pcfg in sorted(table.items()):
        models = pcfg.get("models") or {}
        aliases = sorted(models.keys()) if isinstance(models, dict) else []
        out["providers"][name] = {
            "base_url": str(pcfg.get("base_url") or ""),
            "models": aliases,
            # alias -> the provider's real model id on the wire. Served because
            # the UI otherwise shows ONLY the alias, so an operator had no way
            # to confirm which model a selection actually reaches: they looked
            # for the wire id in the dropdown, saw the alias, and concluded the
            # model was missing. Model ids are not secret; keys still never
            # leave the process.
            "model_ids": {a: split_model_entry(models[a])[0] for a in aliases},
            "capabilities": {a: split_model_entry(models[a])[1]
                             for a in aliases},
            "key_configured": _has_key(pcfg),
        }
    # The fallback chain as labels: the UI shows "a → b → c" so the operator
    # can see WHERE a task will land when the primary is busy. Key-free like
    # everything else in this payload -- it is served over HTTP.
    try:
        out["routes"] = [f"{p}/{m}" for p, m in configured_chain()]
    except ProviderError as exc:
        out["routes"] = []
        out["routes_error"] = str(exc)
    return out


def _has_key(pcfg: Dict[str, Any]) -> bool:
    env_name = str(pcfg.get("api_key_env") or "").strip()
    if env_name and os.environ.get(env_name, "").strip():
        return True
    inline = str(pcfg.get("api_key") or "").strip()
    return bool(inline) and not inline.startswith(
        ("CHANGE_ME", "your-", "sk-PLACEHOLDER"))


# --------------------------------------------------------------------- routes
#
# MODEL ROUTING: an ordered chain of (provider, model) pairs. The agent tries
# route 1; if that endpoint is overloaded, rate-limited, down, or simply does
# not have the model, it moves to route 2, and so on -- across PROVIDERS, not
# just model ids, because the failure modes that matter overnight (a whole
# gateway flapping, a quota exhausting) are per-provider, not per-model.
#
# Sources, highest precedence first:
#   1. an explicit (provider, model) handed to Agent()/CLI -- it PREPENDS to
#      the configured chain, so a one-off choice still enjoys the fallbacks;
#   2. AGENT_ROUTES env:  "prov/model-a, prov/model-b, prov2/model-c"
#   3. routes.json beside the harness: {"routes": [{"provider":..,"model":..}]}
#   4. the legacy behaviour: primary model + AGENT_FALLBACK_MODEL.
#
# Routes are resolved LAZILY, one turn at a time (agent.py): a route whose key
# is missing or whose alias is unknown costs a logged skip, not the whole
# chain. That is deliberate -- overnight, a half-configured third route must
# never stop the first two from working.

#: The routing file, beside the harness. Absent is valid.
ROUTES_FILE = Path(os.environ.get(
    "AGENT_ROUTES_FILE",
    str(Path(__file__).resolve().parent / "routes.json")))


def parse_route_spec(raw: str) -> List[Tuple[str, str]]:
    """Parse an AGENT_ROUTES-style string into (provider, model) pairs.

    ``"prov/model-a,prov/model-b"`` -- a bare model with no slash inherits the
    catalogue's default provider, so a one-provider chain stays terse.
    Empty items are ignored; whitespace is forgiven.
    """
    pairs: List[Tuple[str, str]] = []
    default_provider = default_choice()[0]
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            prov, model = item.split("/", 1)
            prov, model = prov.strip(), model.strip()
        else:
            prov, model = default_provider, item
        if prov and model:
            pairs.append((prov, model))
    return pairs


def _routes_from_file() -> List[Tuple[str, str]]:
    if not ROUTES_FILE.is_file():
        return []
    try:
        data = json.loads(ROUTES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderError(f"{ROUTES_FILE} is not readable JSON: {exc}")
    raw = data.get("routes") if isinstance(data, dict) else data
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ProviderError(f"{ROUTES_FILE.name}: 'routes' must be a list.")
    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("model"):
            prov = str(item.get("provider") or default_choice()[0]).strip()
            pairs.append((prov, str(item["model"]).strip()))
        elif isinstance(item, str) and item.strip():
            pairs.extend(parse_route_spec(item))
    return pairs


def configured_chain() -> List[Tuple[str, str]]:
    """The route chain from the environment or routes.json (may be empty)."""
    env_raw = os.environ.get("AGENT_ROUTES", "").strip()
    if env_raw:
        return parse_route_spec(env_raw)
    return _routes_from_file()


def route_reasoning_efforts() -> Dict[Tuple[str, str], str]:
    """Optional per-route reasoning effort from ``routes.json``.

    The route file remains backward compatible: entries without
    ``reasoning_effort`` inherit the run-wide setting. A route that declares it
    uses that effort as its base whenever it is selected, including as a
    fallback. This is deliberately keyed by (provider, alias), not by the wire
    model id, because aliases are the stable user-facing choice.
    """
    if os.environ.get("AGENT_ROUTES", "").strip() or not ROUTES_FILE.is_file():
        return {}
    try:
        data = json.loads(ROUTES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderError(f"{ROUTES_FILE} is not readable JSON: {exc}")
    raw = data.get("routes") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ProviderError(f"{ROUTES_FILE.name}: 'routes' must be a list.")
    result: Dict[Tuple[str, str], str] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("model"):
            continue
        effort = str(item.get("reasoning_effort") or "").strip()
        if effort:
            provider = str(item.get("provider") or default_choice()[0]).strip()
            result[(provider, str(item["model"]).strip())] = effort
    return result


def route_chain(provider: Optional[str] = None,
                model: Optional[str] = None) -> List[Tuple[str, str]]:
    """The full ordered fallback chain for a request.

    The resulting pairs are NOT resolved here -- resolution happens per turn in
    the agent so that one dead route degrades the chain instead of failing the
    task. Duplicates are collapsed, keeping first occurrence (order is the
    whole point of a route list).
    """
    primary: Tuple[str, str]
    if provider or model:
        d_provider, d_model = default_choice()
        primary = (provider or d_provider, model or d_model)
    else:
        primary = default_choice()  # type: ignore[assignment]

    chain: List[Tuple[str, str]] = [primary]
    chain.extend(configured_chain())
    if config.FALLBACK_MODEL:
        # Legacy single-provider fallback: same provider as the primary,
        # appended last so newer route configs win the middle slots.
        chain.append((primary[0], config.FALLBACK_MODEL))

    seen: set = set()
    ordered: List[Tuple[str, str]] = []
    for pair in chain:
        if pair not in seen:
            seen.add(pair)
            ordered.append(pair)
    return ordered
