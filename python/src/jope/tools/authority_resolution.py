"""Name the winning flywheel authority. Does not unfence, apply v6, or move serving."""
from __future__ import annotations

SCHEMA = "parallel.authority_resolution.v1"
CODEX_TWO_CYCLE = "codex_two_cycle_20260910_1824"
CLOSE_FLYWHEEL = "close_flywheel_p0_p7"
MERGE = "two_cycle_execution_plus_close_flywheel_diagnostics"

# Codex-claimed files from the 18:17–18:18 ownership posts. Do not edit them here.
CODEX_OWNED = (
    "parallel/src/eval_registry.rs",
    "parallel/src/eval_registry/authority.rs",
    "parallel/src/eval_registry/service.rs",
    "parallel/src/teaching_store.rs",
    "parallel/src/corpus.rs",
    "parallel/src/label_admission.rs",
    "parallel/src/gpu_worker.rs",
    "parallel/src/fit_boundary.rs",
    "parallel/src/resident_protocol.rs",
    "parallel/tools/promotion_gate.py",
)


def resolve(user_named="explicit_merge"):
    """Catalog implementation names the merge the catalog itself requires.

    Live execution follows Codex 18:24 (two-cycle). Close-flywheel receipts stay
    diagnostic. Until a later user override: no v6 apply, no frozen-z live
    pointer, no GP unfence.
    """
    if user_named not in ("codex_two_cycle", "close_flywheel", "explicit_merge"):
        raise ValueError("unknown authority name")
    live = CODEX_TWO_CYCLE
    diagnostic = CLOSE_FLYWHEEL
    if user_named == "close_flywheel":
        raise ValueError("close-flywheel is not live execution authority")
    return dict(
        schema=SCHEMA,
        named_by="gap_catalog_implementation",
        user_named=user_named,
        winner=MERGE if user_named == "explicit_merge" else CODEX_TWO_CYCLE,
        live_execution_authority=live,
        diagnostic_authority=diagnostic,
        frozen_z_live_pointer=False,
        blanket_v6_eligibility=False,
        gp_unfence=False,
        apply_contexts_ran=False,
        writes_live_index=False,
        serving_moved=False,
        rules=(
            "preserve current median target for replay/ranking",
            "assign only provenance-resolved new components by frozen 80/20 train/dev-val hash",
            "preserve original memberships",
            "keep ten sealed test families",
        ),
        grok_stops_editing=list(CODEX_OWNED),
    )
