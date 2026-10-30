"""Generic LLM provider layer: any OpenAI-compatible endpoint, by name.

The harness used to speak to exactly one endpoint (``AGENT_BASE_URL`` +
``AGENT_API_KEY`` + ``AGENT_MODEL``). That is fine for one provider and
useless the moment you want to compare two, or pick a cheap model for a
scheduled pass and a strong one for an interactive session.

This module adds a PROVIDER CATALOGUE, shaped like the config of any
multi-provider gateway but with no gateway in between -- the harness talks
to each endpoint directly:

    {
      "default": {"provider": "openrouter", "model": "provider-v4.1-flash"},
      "providers": {
        "provider": {
          "base_url": "https://openrouter.ai/api/v1",
          "api_key_env": "OPENROUTER_API_KEY",     # or "api_key": "sk-..."
          "models": {"deepseek-v4.1-flash": "providerepseek-v4.1-flash-1017"}
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
* RESOLUTION FAILS LOUDLY AND EARLY. A missing key or unknown alias is a
  configuration error, not a runtime surprise four tool calls in.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import config

#: The catalogue file, beside the harness. Absent is valid: the environment
#: provider (or the built-in defaults) still works.
PROVIDERS_FILE = Path(os.environ.get(
    "AGENT_PROVIDERS_FILE",
    str(Path(__file__).resolve().parent / "providers.json")))


@dataclass(frozen=True)
class Resolved:
    """One concrete endpoint choice: where, with what key, which model id."""
    provider: str
    model: str            # the alias the user asked for
    model_id: str         # the real id on the wire
    base_url: str
    api_key: str

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
        raise providerError("'providers' must be an object.")
    for name, pcfg in raw.items():
        if not isinstance(pcfg, dict):
            continue
        table[str(name)] = pcfg
    # The legacy single-endpoint configuration, as provider "env".
    if config.BASE_URL:
        table.setdefault("env", {
            "base_url": config.BASE_URL,
            "api_key_env": "AGENT_API_KEY",
            "models": {config.MODEL: config.MODEL},
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
            f"(and no usable inline api_key). Export it or edit "
            f"{PROVIDERS_FILE.name}.")
    raise ProviderError(
        f"provider {name!r}: no api_key_env or api_key configured.")


def default_choice() -> Tuple[str, str]:
    """(provider, model) used when the caller does not say: file default,
    else the environment provider and its model."""
    data = _load_file()
    d = data.get("default") or {}
    if isinstance(d, dict) and d.get("provider") and d.get("model"):
        return str(d["provider"]), str(d["model"])
    return "env", config.MODEL


def resolve(provider: Optional[str] = None,
            model: Optional[str] = None) -> Resolved:
    """Turn a (provider, model) request into a concrete endpoint.

    Precedence: explicit arguments > file default > environment provider.
    Raises providerError with an actionable message on unknown names or
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
    if not isinstance(models, dict) or not models:
        raise ProviderError(f"provider {provider!r} declares no models.")
    if model in models:
        model_id = str(models[model])
    elif model in models.values():
        # A real id typed straight in is accepted: convenient for one-offs
        # and for providers whose alias table is empty by choice.
        model_id = model
    elif provider == "env":
        # The environment provider is the legacy escape hatch: a raw model id
        # on AGENT_BASE_URL must keep working without being listed anywhere.
        model_id = model
    else:
        known = ", ".join(sorted(models))[:400]
        raise ProviderError(
            f"provider {provider!r} has no model {model!r}; its models: "
            f"{known}.")
    base_url = str(pcfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ProviderError(f"provider {provider!r} has no base_url.")
    return Resolved(provider=provider, model=model, model_id=model_id,
                    base_url=base_url, api_key=_key_for(pcfg, provider))


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
        out["providers"][name] = {
            "base_url": str(pcfg.get("base_url") or ""),
            "models": sorted(models.keys()) if isinstance(models, dict) else [],
            "key_configured": _has_key(pcfg),
        }
    return out


def _has_key(pcfg: Dict[str, Any]) -> bool:
    env_name = str(pcfg.get("api_key_env") or "").strip()
    if env_name and os.environ.get(env_name, "").strip():
        return True
    inline = str(pcfg.get("api_key") or "").strip()
    return bool(inline) and not inline.startswith(
        ("CHANGE_ME", "your-", "sk-PLACEHOLDER"))
