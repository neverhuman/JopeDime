"""Fail-closed pricing and trace checks for a campaign-local Jnoccio gateway."""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
KILO_MODELS_URL = "https://api.kilo.ai/api/gateway/models"
ALLOWED_PROVIDERS = frozenset({"openrouter", "kilo"})
# Pricing rule v2 (2026-07-20, user-approved): OpenRouter stopped publishing a
# "request" price on free models, so requiring it rejected every OpenRouter
# :free route. prompt+completion must be present and explicitly zero, and every
# OTHER pricing field the provider does publish must also be explicitly zero.
REQUIRED_PRICE_FIELDS = ("prompt", "completion")
PRICING_RULE = "zero-required-prompt-completion; zero-all-present-fields; v2"


class FreeGatewayError(RuntimeError):
    """A route lacks complete proof of zero inference price."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _zero_decimal(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    return number.is_finite() and number == 0


def validate_zero_pricing(pricing: Mapping[str, Any] | None) -> list[str]:
    """Require explicit zero prompt+completion and zero for every published field."""

    if not isinstance(pricing, Mapping):
        return ["pricing is absent or not an object"]
    errors: list[str] = []
    for field in REQUIRED_PRICE_FIELDS:
        if field not in pricing:
            errors.append(f"pricing.{field} is absent")
        elif not _zero_decimal(pricing[field]):
            errors.append(f"pricing.{field} is not explicit numeric zero")
    for field in sorted(pricing.keys()):
        if field in REQUIRED_PRICE_FIELDS:
            continue
        if not _zero_decimal(pricing[field]):
            errors.append(f"pricing.{field} is published and not explicit numeric zero")
    return errors


def route_name_is_free(provider: str, route: str) -> bool:
    provider = provider.casefold()
    route = route.casefold()
    if provider == "openrouter":
        return route == "openrouter/free" or route.endswith(":free")
    if provider == "kilo":
        return route == "kilo-auto/free" or route.endswith(":free") or route.endswith("/free")
    return False


def select_zero_price_routes(provider: str, models: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select only named free routes with complete live zero-price metadata."""

    provider_key = provider.casefold()
    if provider_key not in ALLOWED_PROVIDERS:
        raise FreeGatewayError(f"provider is not allowed: {provider}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for model in models:
        route = str(model.get("id") or "")
        if not route_name_is_free(provider_key, route):
            continue
        errors = validate_zero_pricing(model.get("pricing"))
        compact = {
            "provider": provider_key,
            "route": route,
            "pricing": dict(model.get("pricing") or {}),
            "context_length": model.get("context_length") or model.get("context_window"),
        }
        if errors:
            compact["reasons"] = errors
            rejected.append(compact)
        else:
            compact["pricing_sha256"] = object_hash(compact["pricing"])
            accepted.append(compact)
    accepted.sort(key=lambda row: (row["provider"], row["route"]))
    rejected.sort(key=lambda row: (row["provider"], row["route"]))
    return accepted, rejected


def fetch_json(url: str, *, timeout: float = 20.0) -> tuple[Any, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "jope-zyal-free-gateway/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS provider APIs
        raw = response.read(16 * 1024 * 1024)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def catalog_models(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload.get("models"))
    if not isinstance(payload, list):
        raise FreeGatewayError("provider model catalog does not contain a model list")
    return [row for row in payload if isinstance(row, Mapping)]


def trace_route(trace: Mapping[str, Any]) -> tuple[str, str]:
    provider = str(trace.get("provider") or trace.get("provider_id") or "").casefold()
    route = str(
        trace.get("winner_upstream_model_id")
        or trace.get("upstream_model_id")
        or trace.get("model")
        or trace.get("model_id")
        or ""
    )
    return provider, route


def watchdog_violations(
    traces: Sequence[Mapping[str, Any]], allowlist: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return every trace not exactly covered by the hash-locked allowlist."""

    allowed = {(str(row["provider"]).casefold(), str(row["route"])) for row in allowlist}
    violations: list[dict[str, Any]] = []
    for trace in traces:
        provider, route = trace_route(trace)
        cost = trace.get("cost_usd")
        positive_cost = False
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            positive_cost = math.isfinite(float(cost)) and float(cost) > 0
        if (provider, route) not in allowed or positive_cost:
            violations.append(
                {
                    "request_id": str(trace.get("request_id") or ""),
                    "provider": provider,
                    "route": route,
                    "positive_cost_reported": positive_cost,
                }
            )
    return violations


def is_quota_exhaustion(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).casefold()
    return any(fragment in text for fragment in ("quota", "rate limit", "rate_limit", "429"))

