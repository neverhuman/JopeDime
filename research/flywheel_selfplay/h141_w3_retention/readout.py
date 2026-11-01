"""H141 conditional retained-cohort raw utility, U4 and fixed-panel Elo."""

from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from elo_tracker import DatasetEvalRecord, compute_family_grouped_bradley_terry_elo
from runner import ARMS, write

ROOT = Path(__file__).resolve().parent


def summarize(root=ROOT):
    run = root / "run"
    budget = json.loads((run / "budget.json").read_text())
    baseline = (
        json.loads((run / "baselines.json").read_text())
        if (run / "baselines.json").exists()
        else {}
    )
    panel = defaultdict(dict)
    for name, rows in baseline.items():
        arm, seed = name.rsplit("-", 1)
        panel[arm][int(seed)] = rows
    cells = []
    for path in sorted(run.glob("*-fold*-seed*.json")):
        if path.name.endswith(".attempt.json"):
            continue
        cell = json.loads(path.read_text())
        panel[cell["arm"]].setdefault(cell["seed"], {}).update(cell["results"])
        cells.append(
            {
                k: v
                for k, v in cell.items()
                if k not in ("results", "losses", "calibrator")
            }
        )
    complete = (
        budget["status"] == "COMPLETE"
        and len(cells) == 90
        and budget["optimizer_updates"] == 3600
    )
    report = {
        "status": "COMPLETE_RETENTION_COHORT_ONLY" if complete else "INCOMPLETE",
        "full_original_panel_advance": False,
        "serving_or_adoption_authorized": False,
        "budget": budget,
        "cells": cells,
        "original_contexts": 1219,
        "original_families": 205,
        "scope": "conditional on recorded, completed retained TRAIN candidates; r882 and teacher exposure biased; no independent confirmation",
        "unknown_contexts_not_filled": True,
        "metrics": {},
        "contrasts": {},
    }
    if not complete:
        write(run / "readout.json", report)
        return report
    with np.load(root / "admitted_features.npz", allow_pickle=False) as z:
        families = dict(zip(map(str, z["cids"]), map(str, z["families"])))
    contrasts = [("c", "b"), ("d", "a"), ("d", "d_scalar"), ("d", "d_shuffled")]
    for metric in ("val_reg_inner", "U4"):
        contenders = list(ARMS) + ["r882", "ridge150", "zero", "random4"]
        if metric == "val_reg_inner":
            contenders.remove("random4")
            contenders.append("best_revealed")
        eligible = sorted(
            cid
            for cid in families
            if all(
                cid in panel[a][s] and panel[a][s][cid][metric] is not None
                for a in contenders
                for s in (701, 702, 703)
            )
        )
        records = []
        scores = defaultdict(lambda: defaultdict(list))
        per_seed = {}
        for cid in eligible:
            value = {
                a: float(np.mean([panel[a][s][cid][metric] for s in (701, 702, 703)]))
                for a in contenders
            }
            records.append(DatasetEvalRecord(cid, families[cid], "regression", value))
            for arm, v in value.items():
                scores[arm][families[cid]].append(v)
        family_scores = {
            a: {f: float(np.mean(v)) for f, v in d.items()} for a, d in scores.items()
        }
        for seed in (701, 702, 703):
            per_seed[str(seed)] = {}
            for arm in contenders:
                values = defaultdict(list)
                for cid in eligible:
                    values[families[cid]].append(panel[arm][seed][cid][metric])
                per_seed[str(seed)][arm] = float(
                    np.mean([np.mean(v) for v in values.values()])
                )
        elo = compute_family_grouped_bradley_terry_elo(
            records, "r882", 1500.0, 1e-4, 400, 0.01, False
        )
        report["metrics"][metric] = {
            "observed_contexts": len(eligible),
            "unknown_contexts": 1219 - len(eligible),
            "observed_families": len({families[cid] for cid in eligible}),
            "conditional_context_means": {
                a: float(np.mean([r.scores_by_contender[a] for r in records]))
                for a in contenders
            },
            "conditional_family_means": {
                a: float(np.mean(list(f.values()))) for a, f in family_scores.items()
            },
            "conditional_family_medians": {
                a: float(np.median(list(f.values()))) for a, f in family_scores.items()
            },
            "per_seed_conditional_family_means": per_seed,
            "family_values": family_scores,
            "Elo_fixed_complete_contender_panel_on_observed_contexts": elo,
            "unknown_context_sha256": [
                cid for cid in families if cid not in set(eligible)
            ],
        }
        if metric == "U4":
            report["metrics"][metric]["conditional_U4_minus_U0_family_means"] = {
                a: float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    panel[a][s][cid]["U4"] - panel[a][s][cid]["U0"]
                                    for cid in eligible
                                    if families[cid] == f
                                    for s in (701, 702, 703)
                                ]
                            )
                            for f in family_scores[a]
                        ]
                    )
                )
                for a in contenders
            }
            report["metrics"][metric]["conditional_U4_minus_random4_family_means"] = {
                a: float(
                    np.mean(
                        [
                            family_scores[a][f] - family_scores["random4"][f]
                            for f in family_scores[a]
                        ]
                    )
                )
                for a in contenders
            }
        for a, b in contrasts:
            names = sorted(set(family_scores[a]) & set(family_scores[b]))
            delta = np.asarray(
                [family_scores[a][f] - family_scores[b][f] for f in names]
            )
            rng = np.random.default_rng(20261031)
            means = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(
                axis=1
            )
            lower = float(np.quantile(means, 0.025))
            gate_delta = []
            gate_key = (
                "gate_failure" if metric == "val_reg_inner" else "U4_gate_failure"
            )
            for f in names:
                cids = [cid for cid in eligible if families[cid] == f]
                gate_delta.append(
                    float(
                        np.mean(
                            [
                                float(panel[a][s][cid][gate_key])
                                - float(panel[b][s][cid][gate_key])
                                for cid in cids
                                for s in (701, 702, 703)
                            ]
                        )
                    )
                )
            harm = int(np.sum(delta < -0.01))
            gate = (
                lower > 0.001
                and float(np.median(delta)) >= 0
                and harm / len(delta) <= 0.1
                and float(np.mean(gate_delta)) <= 0
            )
            report["contrasts"][metric + ":" + a + "-" + b] = {
                "observed_family_count": len(names),
                "conditional_mean_delta": float(delta.mean()),
                "conditional_lower_97_5": lower,
                "conditional_upper_97_5": float(np.quantile(means, 0.975)),
                "median_family_delta": float(np.median(delta)),
                "families_losing_more_than_01": harm,
                "family_mean_gate_failure_delta": float(np.mean(gate_delta)),
                "conditional_gate_passed": gate,
                "full_original_panel_gate_admitted": False,
                "bootstrap_seed": 20261031,
                "bootstrap_draws": 10000,
                "one_sided_alpha": 0.025,
            }
    report["conditional_primary_c_minus_b_pass"] = report["contrasts"][
        "val_reg_inner:c-b"
    ]["conditional_gate_passed"]
    report["conditional_primary_d_minus_a_with_controls_pass"] = all(
        report["contrasts"]["val_reg_inner:" + c]["conditional_gate_passed"]
        for c in ("d-a", "d-d_scalar", "d-d_shuffled")
    )
    report["verdict"] = "NO_ADVANCE_FULL_ORIGINAL_PANEL_NOT_QUALIFIED"
    write(run / "readout.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    result = summarize(parser.parse_args().root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("cells", "metrics", "contrasts")
            },
            indent=2,
        )
    )
