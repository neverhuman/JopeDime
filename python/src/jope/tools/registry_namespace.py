"""Refuse production namespace enable until the remaining release gates exist."""

from __future__ import annotations

GATES = (
    "historical_import_complete",
    "alternate_archives_covered",
    "target_ipc_p99_qualified",
    "public_routes_integrated",
    "target_enospc_device_loss",
    "journal_db_reconciliation",
    "new_gp_identity",
)

CURRENT = dict(
    historical_import_complete=False,
    alternate_archives_covered=False,
    target_ipc_p99_qualified=False,
    public_routes_integrated=False,
    target_enospc_device_loss=False,
    journal_db_reconciliation=False,
    new_gp_identity=False,
    reusable_historical_completions=0,
    verified_terminal_failures=0,
    wired_to_live_gp=False,
    claimed_savings=False,
    python_shadow_is_authority=False,
)


def missing_gates(evidence=None):
    evidence = {**CURRENT, **(evidence or {})}
    return [name for name in GATES if not evidence.get(name)]


def enable_namespace(evidence=None, claimed_savings_745=False):
    if claimed_savings_745:
        raise ValueError("74.5% savings claim is forbidden")
    missing = missing_gates(evidence)
    if missing:
        return False, missing
    raise ValueError("namespace enable is not authorized from this helper")


def savings_claim(hits, completed_evaluations, modeled_wall=0, measured_core=0):
    """Hits must not inflate completed_evaluations. Modeled wall is not savings."""
    if hits and completed_evaluations is not None and hits > completed_evaluations:
        raise ValueError("registry hits must not overload completed_evaluations")
    return dict(
        schema="parallel.registry_savings.v1",
        claimed_745=False,
        achieved=False,
        registry_hit=hits,
        completed_evaluations=completed_evaluations,
        modeled_wall=modeled_wall,
        measured_core_seconds=measured_core,
        unattributed_remainder_is_not_savings=True,
    )


def snapshot(evidence=None):
    evidence = {**CURRENT, **(evidence or {})}
    return dict(
        schema="parallel.registry_namespace_gate.v1",
        enabled=False,
        wired_to_live_gp=False,
        claimed_savings=False,
        python_shadow_is_authority=False,
        reusable_historical_completions=int(evidence.get("reusable_historical_completions") or 0),
        verified_terminal_failures=int(evidence.get("verified_terminal_failures") or 0),
        missing=missing_gates(evidence),
        new_gp_identity=bool(evidence.get("new_gp_identity")),
        legacy_gp_pid=774452,
    )
