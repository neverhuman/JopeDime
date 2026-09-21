"""Stage ledger plus identity join. Directory arrival is not novelty."""

from __future__ import annotations

STAGES = (
    "discovered",
    "retained",
    "provenance-resolved",
    "partition-assigned",
    "fold-ready",
    "prepared",
    "context-published",
    "execution-eligible",
    "observations-admitted",
    "corpus-published",
    "optimizer-exposed",
    "qualified",
)


def join_row(
    component_id,
    aliases=(),
    family=None,
    prepared_path=None,
    context_ids=(),
    stage="discovered",
    already_in_index=False,
    task="regression",
):
    if already_in_index and STAGES.index(stage) < STAGES.index("context-published"):
        stage = "context-published"
    return dict(
        component_id=component_id,
        aliases=sorted(set(aliases) | {component_id}),
        family=family,
        prepared_path=prepared_path,
        context_ids=list(context_ids),
        stage=stage,
        already_in_index=already_in_index,
        # This metadata join has no original admitted-computation receipt.
        # An unindexed arrival is only a candidate for identity qualification.
        counted_as_new=False,
        unindexed_candidate=not already_in_index,
        novelty_evidence="unavailable: original admitted computation evidence not joined",
        task=task,
    )


def build_ledger(rows):
    counts = {stage: 0 for stage in STAGES}
    already = new = 0
    for row in rows:
        counts[row["stage"]] += 1
        already += bool(row["already_in_index"])
        new += bool(row["counted_as_new"])
    return dict(
        schema="parallel.component_stage_ledger.v1",
        stages=list(STAGES),
        n=len(rows),
        already_in_index=already,
        counted_as_new=new,
        unindexed_candidates=sum(row.get("unindexed_candidate", False) for row in rows),
        stage_counts=counts,
        dashboard_must_not_collapse_stages=True,
        rows=rows,
    )


def growth_candidates(transferred_ids, already_in_index, live_ids, sealed_ids, evidenced_unresolved=()):
    blocked = set(already_in_index) | set(live_ids) | set(sealed_ids)
    novel = sorted(set(transferred_ids) - blocked)
    unresolved = sorted(set(evidenced_unresolved) - blocked)
    return dict(
        schema="parallel.growth_candidates.v1",
        n_transferred=len(set(transferred_ids)),
        n_already_in_index=len(set(already_in_index) & set(transferred_ids)),
        n_novel=len(novel),
        n_evidenced_unresolved=len(unresolved),
        already_in_index_ids=sorted(set(already_in_index) & set(transferred_ids)),
        novel_ids=novel,
        unresolved_ids=unresolved,
        auto_admitted=0,
    )


def provenance_receipt(
    component_id,
    object_hashes,
    target=None,
    units=None,
    horizon=None,
    predictors=None,
    entity=None,
    time_col=None,
    prior_split=None,
    reviewer=None,
    policy_version=None,
    source_path=None,
):
    missing = [
        name
        for name, value in (
            ("object_hashes", object_hashes),
            ("target", target),
            ("reviewer", reviewer),
            ("policy_version", policy_version),
        )
        if not value
    ]
    return dict(
        schema="parallel.provenance_receipt.v1",
        component_id=component_id,
        object_hashes=object_hashes or {},
        target=target,
        units=units,
        horizon=horizon,
        predictors=predictors,
        entity=entity,
        time=time_col,
        prior_split_use=prior_split,
        reviewer=reviewer,
        policy_version=policy_version,
        source_path=source_path,
        source_path_is_lead_only=True,
        resolved=not missing,
        missing_fields=missing,
    )
