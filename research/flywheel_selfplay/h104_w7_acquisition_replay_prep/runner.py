#!/usr/bin/env python3
"""W7 metadata-only dry run and feature-only acquisition policy controls.

No fit, outcome reader, model loader, S3 client or execution mode exists here.
Pinned training identity metadata permits preparation, not numerical admission.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

ARMS = ("current_2greedy_2uniform", "uniform4", "qd_2greedy_1diverse_1uniform")
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def hash_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_pinned(path, expected):
    raw = Path(path).read_bytes()
    if not hash_ok(expected) or sha(raw) != expected:
        raise ValueError("byte SHA mismatch")
    return json.loads(raw, object_pairs_hook=unique_object)


def validate_config(c):
    expected = {
        "schema": "jope.h104.w7-config/v1",
        "arms": list(ARMS),
        "seeds": [701, 702, 703],
        "fold_count": 5,
        "fold_seed": 4900,
        "acquisition_rounds_per_fold": 20,
        "contexts_per_round": 1,
        "new_fits_per_context_round": 4,
        "terminal_updates": 40,
        "planned_new_fits": 3600,
        "native_fit_ceiling": 4800,
        "cpu_seconds_ceiling": 14400,
        "gpu_seconds_ceiling": 7200,
        "dope_sha256": DOPE_SHA,
        "execution_admitted": False,
        "v3_import_enabled": False,
    }
    if any(type(c.get(k)) is not type(v) or c.get(k) != v for k, v in expected.items()):
        raise ValueError("fixed W7 contract changed")
    if c.get("expected_contexts") != 1219 or c.get("expected_families") != 205:
        raise ValueError("training census changed")
    if not hash_ok(c.get("fold_manifest_sha256")):
        raise ValueError("fold byte pin required")
    return c


def validate_folds(m, c):
    if (
        m.get("schema") != "jope.w6.family-folds/v1"
        or m.get("fold_seed") != 4900
        or m.get("fold_count") != 5
    ):
        raise ValueError("unexpected inherited H72 folds")
    rows = m.get("assignments")
    if not isinstance(rows, list) or len(rows) != c["expected_contexts"]:
        raise ValueError("incomplete context census")
    fields = {"context_sha256", "family_id", "physical_table_sha256", "fold"}
    contexts, family_fold, table_fold = set(), {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("identity-only fold row required")
        cid, family, table, fold = (
            row[k]
            for k in ("context_sha256", "family_id", "physical_table_sha256", "fold")
        )
        if (
            not hash_ok(cid)
            or not hash_ok(table)
            or not isinstance(family, str)
            or not family
        ):
            raise ValueError("malformed identity")
        if type(fold) is not int or fold not in range(5) or cid in contexts:
            raise ValueError("duplicate context or invalid fold")
        contexts.add(cid)
        if (
            family_fold.setdefault(family, fold) != fold
            or table_fold.setdefault(table, fold) != fold
        ):
            raise ValueError("family/physical-table leakage across folds")
    if len(family_fold) != c["expected_families"]:
        raise ValueError("incomplete family census")
    families = [sorted(f for f in family_fold if family_fold[f] == k) for k in range(5)]
    if m.get("families_by_fold") != families or any(not f for f in families):
        raise ValueError("family partition inconsistent")
    if m.get("context_count") != len(rows) or m.get("family_count") != len(family_fold):
        raise ValueError("declared counts inconsistent")
    return rows


def build_plan(config, manifest, folds=None):
    c = validate_config(config)
    rows = validate_folds(manifest, c)
    folds = list(range(5)) if folds is None else folds
    if (
        not folds
        or any(type(f) is not int or f not in range(5) for f in folds)
        or len(set(folds)) != len(folds)
    ):
        raise ValueError("invalid deterministic fold shard")
    schedule, panel, fold_summaries = [], [], []
    for held in sorted(folds):
        fit, score = (
            [r for r in rows if r["fold"] != held],
            [r for r in rows if r["fold"] == held],
        )
        groups = defaultdict(list)
        for row in fit:
            groups[row["family_id"]].append(row)
        fold_summaries.append(
            {
                "held_fold": held,
                "fit_contexts": len(fit),
                "score_contexts": len(score),
                "fit_families": len(groups),
                "score_families": len({r["family_id"] for r in score}),
                "score_context_ids_sha256": sha(
                    encoded(sorted(r["context_sha256"] for r in score))
                ),
                "fit_context_ids_sha256": sha(
                    encoded(sorted(r["context_sha256"] for r in fit))
                ),
            }
        )
        for seed in c["seeds"]:
            # Twenty distinct fit families, then one context per family; arm-independent.
            order = sorted(
                groups, key=lambda f: sha(f"W7-family/{seed}/{held}/{f}".encode())
            )
            if len(order) < 20:
                raise ValueError("insufficient distinct fit families")
            for round_id, family in enumerate(order[:20]):
                row = min(
                    groups[family],
                    key=lambda r: sha(
                        f"W7-context/{seed}/{held}/{r['context_sha256']}".encode()
                    ),
                )
                schedule.append(
                    {
                        "held_fold": held,
                        "seed": seed,
                        "round": round_id,
                        **{
                            k: row[k]
                            for k in [
                                "context_sha256",
                                "family_id",
                                "physical_table_sha256",
                            ]
                        },
                        "arm_selection_seeds": {
                            arm: int(
                                sha(
                                    f"W7-selection/{seed}/{held}/{round_id}/{arm}".encode()
                                )[:16],
                                16,
                            )
                            for arm in ARMS
                        },
                    }
                )
            for arm in ARMS:
                panel.append(
                    {
                        "slot_id": f"fold{held}/seed{seed}/{arm}",
                        "held_fold": held,
                        "seed": seed,
                        "memory": arm,
                        "status": "pending",
                        "receipt_sha256": None,
                        "matching_receipt_sha256": None,
                        "matched_rows": None,
                        "discarded_rows": None,
                        "excluded_rows": None,
                        "common_support_rows": None,
                    }
                )
    fits = len(schedule) * len(ARMS) * 4
    if fits > c["planned_new_fits"] or fits > c["native_fit_ceiling"]:
        raise ValueError("fit plan exceeds existing envelope")
    return {
        "schema": "jope.h104.w7-dry-run/v1",
        "status": "PREPARATION_ONLY",
        "execution_admitted": False,
        "advance_admitted": False,
        "fold_manifest_sha256": c["fold_manifest_sha256"],
        "folds": fold_summaries,
        "schedule": schedule,
        "planned_context_rounds_per_arm_seed_in_shard": 20 * len(folds),
        "planned_new_fits_in_shard": fits,
        "common_learner_receipt_panel": panel,
        "terminal_updates_per_model": 40,
        "common_support_rule": "Zero matched support is UNTESTED; missing receipts remain pending; no partial panel may advance.",
        "actual_fits": 0,
        "actual_optimizer_updates": 0,
        "actual_campaign_consumption": None,
        "remaining_authorized_budget": None,
        "metrics": {"family_elo": None, "val_reg_inner": None, "U4": None},
    }


def select_four(offers, arm, seed):
    """Testable feature-only rule; probabilities are sequential, not marginal IPS."""
    if arm not in ARMS or len(offers) < 4:
        raise ValueError("known arm and at least four candidates required")
    fields = {"program_sha256", "prior_cost", "disagreement", "descriptor"}
    ids = set()
    for row in offers:
        if set(row) != fields or not hash_ok(row["program_sha256"]):
            raise ValueError("only proposal feature fields allowed; no labels")
        if row["program_sha256"] in ids:
            raise ValueError("duplicate candidate")
        ids.add(row["program_sha256"])
        if (
            any(
                type(row[k]) not in (int, float) or not math.isfinite(row[k])
                for k in ("prior_cost", "disagreement")
            )
            or row["disagreement"] < 0
        ):
            raise ValueError("nonfinite/invalid proposal feature")
        if not isinstance(row["descriptor"], str) or not row["descriptor"]:
            raise ValueError("descriptor required")
    remaining = sorted(offers, key=lambda r: r["program_sha256"])
    rng, result, descriptors = random.Random(seed), [], set()
    for slot in range(4):
        eligible_ids = [r["program_sha256"] for r in remaining]
        if arm != "uniform4" and slot < 2:
            chosen = min(
                remaining, key=lambda r: (r["prior_cost"], r["program_sha256"])
            )
            kind, probability = "greedy", 1.0
        elif arm == ARMS[2] and slot == 2:
            chosen = min(
                remaining,
                key=lambda r: (
                    r["descriptor"] in descriptors,
                    -r["disagreement"],
                    r["program_sha256"],
                ),
            )
            kind, probability = "novel_descriptor_then_disagreement", 1.0
        else:
            chosen = rng.choice(remaining)
            kind, probability = "uniform_remaining", 1 / len(remaining)
        result.append(
            {
                "slot": slot,
                "program_sha256": chosen["program_sha256"],
                "rule": kind,
                "conditional_draw_probability": probability,
                "eligible_ids": eligible_ids,
                "probability_scope": "given prior slots and these fixed features; not marginal inclusion propensity",
            }
        )
        descriptors.add(chosen["descriptor"])
        remaining.remove(chosen)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--fold-manifest", type=Path, required=True)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    c = load_pinned(args.config, args.config_sha256)
    m = load_pinned(args.fold_manifest, c["fold_manifest_sha256"])
    result = build_plan(c, m, args.folds)
    with args.output.open("xb") as out:
        out.write(encoded(result))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "execution_admitted",
                    "planned_new_fits_in_shard",
                    "actual_fits",
                ]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
