"""Build a Router from a TOML file.

Keys are never written in the file: each provider names the environment variables that hold them. A variable
that is missing or empty is skipped; a provider left with no keys drops out, and so do its models, from every
chain, with a warning. A role left with no models is not registered. The warnings are on `router.warnings`.
"""
from __future__ import annotations

import logging
import math
import os
import tomllib
from typing import Mapping

from .base import ModelSpec
from .budget import RoleBudget
from .limits import CircuitBreaker, KeyPool
from .router import Router
from .transports import AnthropicTransport, OpenAICompatTransport, Transport

log = logging.getLogger("llm_router")

_PRICE_FIELDS = ("input_per_m", "output_per_m", "cache_read_per_m", "cache_write_per_m", "reasoning_per_m")


def _price(name: str, value, where: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{where}: {name} must be a number >= 0, not {value!r}")
    return float(value)


def _make_transport(name: str, conf: dict) -> Transport:
    kind = conf.get("transport", "openai")
    if kind == "openai":
        return OpenAICompatTransport(
            conf.get("base_url") or None,
            max_tokens_param=conf.get("max_tokens_param", "max_tokens"),
            use_reported_cost=bool(conf.get("use_reported_cost", False)),
        )
    if kind == "anthropic":
        return AnthropicTransport()
    raise ValueError(f"provider {name!r}: unknown transport {kind!r} (use 'openai' or 'anthropic')")


def load_router(path: str, *, env: Mapping[str, str] | None = None,
                transports: Mapping[str, Transport] | None = None, **router_kwargs) -> Router:
    """Read `path` and return a ready Router. `transports` overrides the transport of named providers."""
    with open(path, "rb") as f:
        conf = tomllib.load(f)
    return build_router(conf, env=os.environ if env is None else env, transports=transports or {}, **router_kwargs)


def build_router(conf: dict, *, env: Mapping[str, str], transports: Mapping[str, Transport] | None = None,
                 **router_kwargs) -> Router:
    transports = dict(transports or {})
    warnings: list[str] = []
    settings = conf.get("router", {})

    pools: dict[str, KeyPool] = {}
    chosen_transports: dict[str, Transport] = {}
    for name, p in conf.get("providers", {}).items():
        names = p.get("api_key_env", [])
        names = [names] if isinstance(names, str) else list(names)
        keys = [(var, env[var]) for var in names if env.get(var, "").strip()]
        if not keys:
            warnings.append(f"provider {name!r}: no key in {names or 'api_key_env'}; its models are skipped")
            continue
        for limit in ("rpm", "tpm", "rpd"):
            value = p.get(limit)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"provider {name!r}: {limit} must be an integer >= 1, not {value!r}")
        max_wait = p.get("max_wait", 10.0)
        if not isinstance(max_wait, (int, float)) or isinstance(max_wait, bool) or not 0 <= max_wait < math.inf:
            raise ValueError(f"provider {name!r}: max_wait must be a number of seconds >= 0, not {max_wait!r}")
        pools[name] = KeyPool(name, keys, rpm=p.get("rpm"), tpm=p.get("tpm"), rpd=p.get("rpd"),
                              max_wait=float(max_wait))
        chosen_transports[name] = transports.get(name) or _make_transport(name, p)

    models: dict[str, ModelSpec] = {}
    for key, m in conf.get("models", {}).items():
        where = f"model {key!r}"
        provider = m.get("provider")
        if provider not in conf.get("providers", {}):
            warnings.append(f"{where}: unknown provider {provider!r}; skipped")
            continue
        if provider not in pools:
            continue
        prices = {f: _price(f, m.get(f), where) for f in _PRICE_FIELDS}
        free = m.get("free", False) is True
        if not free and (prices["input_per_m"] is None or prices["output_per_m"] is None):
            raise ValueError(f"{where}: input_per_m and output_per_m are required (the cap reserves each call's "
                             "worst case from them); for a model that really costs nothing, write free = true")
        if free:
            warnings.append(f"{where} is declared free: its calls cost $0 and the cap never limits them")
        models[key] = ModelSpec(
            key=key, provider=provider, model=m.get("model", key),
            input_per_m=prices["input_per_m"] or 0.0, output_per_m=prices["output_per_m"] or 0.0,
            cache_read_per_m=prices["cache_read_per_m"], cache_write_per_m=prices["cache_write_per_m"],
            reasoning_per_m=prices["reasoning_per_m"],
            timeout=m.get("timeout", 60.0), max_output_tokens=m.get("max_output_tokens", 1024),
            output_ceiling=m.get("output_ceiling"),
        )

    def usable(keys: list[str], where: str) -> list[str]:
        kept = [k for k in keys if k in models]
        for k in keys:
            if k not in models:
                warnings.append(f"{where}: model {k!r} is not available; dropped")
        return kept

    fallback = usable(list(settings.get("fallback", [])), "fallback")
    chains: dict[str, list[str]] = {}
    budgets: dict[str, RoleBudget] = {}
    for role, r in conf.get("roles", {}).items():
        chain = usable(list(r.get("chain", [])), f"role {role!r}")
        if not chain and not fallback:
            warnings.append(f"role {role!r}: no model left; not registered")
            continue
        hours = r.get("window_hours", settings.get("window_hours", 24))
        window = None if not hours else float(hours) * 3600
        cap = r.get("budget_usd", settings.get("budget_usd"))
        if cap is None:
            raise ValueError(f"role {role!r}: no budget_usd (set it on the role or under [router])")
        chains[role] = chain
        budgets[role] = RoleBudget(cap, window)

    if not chains:
        raise ValueError("no role could be registered: " + "; ".join(warnings))

    used = {models[k].provider for k in models}
    router = Router(
        models=models, chains=chains, budgets=budgets,
        pools={p: pools[p] for p in used}, transports={p: chosen_transports[p] for p in used},
        fallback=fallback, max_chain=int(settings.get("max_chain", 5)),
        breaker=CircuitBreaker(int(settings.get("breaker_threshold", 3)), float(settings.get("breaker_cooldown", 60))),
        **router_kwargs,
    )
    router.warnings = warnings
    for w in warnings:
        log.warning(w)
    return router
