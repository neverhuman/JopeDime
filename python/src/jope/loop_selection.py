"""Pure grammar and candidate-selection helpers for the closed-loop runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .grammars.spec import GrammarSpec
    from .loop_state import LoopConfig
    from .policy.model_v3 import JopePolicyModelV3


def _model_specs(model: JopePolicyModelV3) -> list[GrammarSpec]:
    # Grammar specs are stored on the model at load time via checkpoint; fall back to repo specs.
    specs = getattr(model, "_grammar_specs", None)
    if specs:
        return specs
    from .grammars import all_grammars

    return all_grammars()


def _select_candidates(pool: list[dict[str, Any]], cfg: LoopConfig) -> list[dict[str, Any]]:
    by_mu = sorted(pool, key=lambda c: c["mu"], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_div: set[tuple[str, str]] = set()
    engines_present = {c["engine"] for c in pool}
    # exploit: top-mu with (engine, head) diversity
    for cand in by_mu:
        if len(selected) >= cfg.cv_top_m:
            break
        div = (cand["engine"], str(cand.get("head_type")))
        if div in seen_div:
            continue
        seen_div.add(div)
        selected.append(cand)
    # >=1 per engine floor
    for engine in engines_present:
        if not any(c["engine"] == engine for c in selected):
            best = next((c for c in by_mu if c["engine"] == engine and c not in selected), None)
            if best is not None:
                selected.append(best)
    # explore: most uncertain among the rest
    rest = [c for c in pool if c not in selected]
    rest.sort(key=lambda c: c["sigma"], reverse=True)
    selected.extend(rest[: cfg.cv_uncertain_m])
    return selected
