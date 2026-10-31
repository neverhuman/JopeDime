"""W7 count matching on caller-supplied completion metadata; never admission.

No utilities, models, numerical arrays, network access or training are consumed.
Completion receipt hashes are identity syntax, not authenticated evidence.
"""

import hashlib
import json
from collections import Counter, defaultdict

ARMS = ("current_2greedy_2uniform", "uniform4", "qd_2greedy_1diverse_1uniform")
COMPLETED = ("completed_pass", "completed_gate_fail")
STATUSES = COMPLETED + ("timeout", "error", "unknown")


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def hash_ok(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unexpected/missing metadata fields")


def validate_cell(cell):
    exact(cell, ["held_fold", "seed", "schedule_sha256", "families"])
    if type(cell["held_fold"]) is not int or cell["held_fold"] not in range(5):
        raise ValueError("invalid held fold")
    if type(cell["seed"]) is not int or cell["seed"] not in (701, 702, 703):
        raise ValueError("invalid seed")
    if (
        not hash_ok(cell["schedule_sha256"])
        or not isinstance(cell["families"], list)
        or len(cell["families"]) != 20
    ):
        raise ValueError("exact H104 20-family cell required")
    families, contexts, tables = {}, set(), set()
    for r in cell["families"]:
        exact(r, ["family_id", "context_sha256", "physical_table_sha256"])
        if (
            not isinstance(r["family_id"], str)
            or not r["family_id"]
            or r["family_id"] in families
        ):
            raise ValueError("duplicate/invalid family")
        if not hash_ok(r["context_sha256"]) or not hash_ok(r["physical_table_sha256"]):
            raise ValueError("invalid context/table identity")
        if r["context_sha256"] in contexts or r["physical_table_sha256"] in tables:
            raise ValueError("duplicate context/table across families")
        families[r["family_id"]] = r["context_sha256"]
        contexts.add(r["context_sha256"])
        tables.add(r["physical_table_sha256"])
    return families


def match_cell(cell, records):
    """Select min(count) per family/status without looking at outcomes.

    Require all 240 attempted acquisition records (20 families x 4 x 3 arms)
    before classifying support. Unknown attempts stay excluded, never gate-fail.
    Even CANDIDATE_MATCH requires independent provenance and cost admission.
    """
    families = validate_cell(cell)
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    groups, counts, seen, excluded = defaultdict(list), Counter(), set(), Counter()
    fields = [
        "arm",
        "family_id",
        "context_sha256",
        "program_sha256",
        "status",
        "completion_receipt_sha256",
    ]
    for r in records:
        exact(r, fields)
        arm, family, status = r["arm"], r["family_id"], r["status"]
        if (
            not isinstance(arm, str)
            or arm not in ARMS
            or not isinstance(family, str)
            or family not in families
        ):
            raise ValueError("foreign arm/family")
        if r["context_sha256"] != families[family] or not hash_ok(r["program_sha256"]):
            raise ValueError("foreign context/invalid program identity")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("unrecorded/invalid status")
        pin = r["completion_receipt_sha256"]
        if (status in COMPLETED and not hash_ok(pin)) or (
            pin is not None and not hash_ok(pin)
        ):
            raise ValueError(
                "completed state requires a receipt hash; provenance still needs review"
            )
        key = (arm, family, r["program_sha256"])
        if key in seen:
            raise ValueError(
                "duplicate attempt/program; retries require upstream reconciliation"
            )
        seen.add(key)
        counts[(arm, family)] += 1
        if counts[(arm, family)] > 4:
            raise ValueError("more than four new attempts per arm/family")
        if status in COMPLETED:
            groups[(arm, family, status)].append(r["program_sha256"])
        else:
            excluded[(arm, status)] += 1
    missing = sum(4 - counts[(a, f)] for a in ARMS for f in families)
    strata, support = [], set()
    totals = {
        a: dict(
            matched=0,
            discarded_completed=0,
            excluded=sum(excluded[(a, s)] for s in STATUSES[2:]),
            excluded_by_status={s: excluded[(a, s)] for s in STATUSES[2:]},
            attempted=sum(counts[(a, f)] for f in families),
        )
        for a in ARMS
    }
    for family in sorted(families):
        for status in COMPLETED:
            available = {a: sorted(groups[(a, family, status)]) for a in ARMS}
            n = min(map(len, available.values()))
            if n:
                support.add(family)
            strata.append(
                dict(
                    family_id=family,
                    status=status,
                    matched_per_arm=n,
                    selected={a: available[a][:n] for a in ARMS},
                    available={a: len(available[a]) for a in ARMS},
                )
            )
            for a in ARMS:
                totals[a]["matched"] += n
                totals[a]["discarded_completed"] += len(available[a]) - n
    for a in ARMS:
        assert totals[a]["attempted"] == sum(
            totals[a][k] for k in ("matched", "discarded_completed", "excluded")
        )
    status = (
        "PENDING" if missing else "UNTESTED" if len(support) < 8 else "CANDIDATE_MATCH"
    )
    return dict(
        schema="jope.h111.w7-candidate-matching/v1",
        status=status,
        cell_sha256=digest(cell),
        input_metadata_sha256=digest(
            sorted(
                records, key=lambda r: (r["arm"], r["family_id"], r["program_sha256"])
            )
        ),
        held_fold=cell["held_fold"],
        seed=cell["seed"],
        missing_attempt_records=missing,
        common_support_families=len(support),
        strata=strata,
        arm_counts=totals,
        selection_final=status != "PENDING",
        source_provenance_verified=False,
        replay_schedule_implemented=False,
        cost_update_matching_verified=False,
        execution_admitted=False,
        advance_admitted=False,
        metrics=dict(family_elo=None, val_reg_inner=None, U4=None),
    )


def prepare_cells(plan):
    """Derive 15 cells/45 pending receipt slots from an already verified H104 plan."""
    if (
        plan.get("schema") != "jope.h104.w7-dry-run/v1"
        or plan.get("execution_admitted") is not False
    ):
        raise ValueError("verified H104 preparation plan required")
    schedule = plan["schedule"]
    if len(schedule) != 300:
        raise ValueError("complete five-fold/three-seed H104 schedule required")
    cells, slots = [], []
    for held in range(5):
        for seed in (701, 702, 703):
            rows = [r for r in schedule if r["held_fold"] == held and r["seed"] == seed]
            if sorted(r["round"] for r in rows) != list(range(20)):
                raise ValueError("missing/duplicate context round")
            cell = dict(
                held_fold=held,
                seed=seed,
                schedule_sha256=digest(rows),
                families=[
                    {
                        k: r[k]
                        for k in (
                            "family_id",
                            "context_sha256",
                            "physical_table_sha256",
                        )
                    }
                    for r in rows
                ],
            )
            validate_cell(cell)
            cells.append(cell)
            for arm in ARMS:
                slots.append(
                    dict(
                        slot_id=f"fold{held}/seed{seed}/{arm}",
                        status="pending",
                        matching_receipt_sha256=None,
                        matched_rows=None,
                        discarded_rows=None,
                        excluded_rows=None,
                    )
                )
    return dict(
        schema="jope.h111.w7-matching-preparation/v1",
        h104_plan_canonical_sha256=digest(plan),
        cells=cells,
        common_learner_receipt_panel=slots,
        actual_fits=0,
        actual_optimizer_updates=0,
        execution_admitted=False,
        advance_admitted=False,
        metrics=dict(family_elo=None, val_reg_inner=None, U4=None),
    )
