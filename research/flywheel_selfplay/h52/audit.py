#!/usr/bin/env python3
"""Audit per-seed and family-clustered pooled H52 outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (12, 13, 14)
DRAWS = 10_000
MARGIN = 0.001
CONFIGS = {
    "FOCUS-tau0.1": ("FOCUS", "tau0.1", "CONTROL-k16-lr1e-4"),
    "FOCUS-tau0.5": ("FOCUS", "tau0.5", "CONTROL-k16-lr1e-4"),
    "GATE-default": ("GATE", "default", "CONTROL-k16-lr1e-4"),
    "ANCHOR-lambda1": ("ANCHOR", "lambda1", "CONTROL-k16-lr3e-4"),
    "ANCHOR-lambda10": ("ANCHOR", "lambda10", "CONTROL-k16-lr3e-4"),
    "NULL-tau0.1": ("NULL", "tau0.1", "CONTROL-k16-lr1e-4"),
}
CONTROL_CONFIGS = ("CONTROL-k2-lr1e-4", "CONTROL-k16-lr1e-4", "CONTROL-k16-lr3e-4")


def load_cell(root: Path, name: str, seed: int) -> tuple[dict, dict]:
    directory = root / name / f"seed{seed}"
    summaries = list(directory.glob(f"*-seed{seed}.json"))
    summaries = [p for p in summaries if not p.name.endswith("-family-gains.json")]
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one summary for {name} seed{seed}, found {len(summaries)}"
        )
    return json.loads(summaries[0].read_text()), json.loads(
        summaries[0]
        .with_name(summaries[0].name.removesuffix(".json") + "-family-gains.json")
        .read_text()
    )


def sign_test(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return 1.0
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(values: np.ndarray) -> dict:
    positive = int(np.count_nonzero(values > 0))
    negative = int(np.count_nonzero(values < 0))
    nonzero = positive + negative
    draws = np.random.default_rng(12_520 + values.size).integers(
        0, values.size, size=(DRAWS, values.size)
    )
    means = values[draws].mean(axis=1)
    medians = np.median(values[draws], axis=1)
    return {
        "n": int(values.size),
        "family_median": float(np.median(values)),
        "family_mean": float(np.mean(values)),
        "family_bootstrap_lb95": float(np.quantile(medians, 0.025)),
        "family_bootstrap_ub95": float(np.quantile(medians, 0.975)),
        "mean_bootstrap_lb95_diagnostic_only": float(np.quantile(means, 0.025)),
        "mean_bootstrap_ub95_diagnostic_only": float(np.quantile(means, 0.975)),
        "nonzero_count": nonzero,
        "nonzero_fraction": nonzero / values.size,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": int(values.size - nonzero),
        "sign_test_two_sided_p": sign_test(positive, negative),
    }


def clustered_pooled(
    rows: list[dict[str, float]], families: list[str], label: str
) -> dict:
    family_index = {family: i for i, family in enumerate(families)}
    matrix = np.zeros((len(families), len(SEEDS)), dtype=np.float64)
    for seed_index, row in enumerate(rows):
        for family, value in row.items():
            matrix[family_index[family], seed_index] = float(value)
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
    chosen = np.random.default_rng(seed).integers(
        0, len(families), size=(DRAWS, len(families))
    )
    sampled = matrix[chosen].reshape(DRAWS, -1)
    medians = np.median(sampled, axis=1)
    means = np.mean(sampled, axis=1)
    per_family_mean = matrix.mean(axis=1)
    positive = int(np.count_nonzero(per_family_mean > 0))
    negative = int(np.count_nonzero(per_family_mean < 0))
    nonzero = positive + negative
    median = float(np.median(matrix))
    lb = float(np.quantile(medians, 0.025))
    return {
        "family_ids": len(families),
        "seed_rows": len(SEEDS),
        "pooled_rows": int(matrix.size),
        "family_median": median,
        "cluster_bootstrap_lb95": lb,
        "cluster_bootstrap_ub95": float(np.quantile(medians, 0.975)),
        "cluster_mean": float(np.mean(matrix)),
        "cluster_mean_bootstrap_lb95_diagnostic_only": float(np.quantile(means, 0.025)),
        "cluster_mean_bootstrap_ub95_diagnostic_only": float(np.quantile(means, 0.975)),
        "family_nonzero_fraction": nonzero / len(families),
        "family_positive_count": positive,
        "family_negative_count": negative,
        "family_zero_count": len(families) - nonzero,
        "family_sign_test_two_sided_p": sign_test(positive, negative),
        "paired_admitted": median > MARGIN and lb > MARGIN,
        "admission_margin": MARGIN,
        "terminal_dl_val_reads": 0,
        "elo": "N/A (no admitted candidate)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path.home() / "jope-data/h52/runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "schema": "jope.h52.audit/v1",
        "seeds": list(SEEDS),
        "draws": DRAWS,
        "admission": "pooled paired family median > 0.001 and family-cluster bootstrap LB95 > 0.001; ties fail",
        "best_of_seed_selection": False,
        "diagnostics_only": [
            "family nonzero/positive/negative fractions",
            "sign test",
            "paired family mean bootstrap CI",
        ],
        "configurations": {},
        "controls": {},
    }

    control_cache = {}
    live_control = [load_cell(args.runs, "CONTROL-k2-lr1e-4", seed) for seed in SEEDS]
    live_vectors = [entry[1]["family_gains_vs_r882"] for entry in live_control]
    for control_name in CONTROL_CONFIGS:
        entries = [load_cell(args.runs, control_name, seed) for seed in SEEDS]
        control_cache[control_name] = entries
        vectors = [entry[1]["family_gains_vs_r882"] for entry in entries]
        family_ids = sorted(vectors[0])
        vs_live = {}
        for seed, candidate_vector, live_vector in zip(
            SEEDS, vectors, live_vectors, strict=True
        ):
            if set(candidate_vector) != set(live_vector):
                raise RuntimeError(
                    f"{control_name} seed{seed}: live CONTROL family sets differ"
                )
            deltas = np.asarray(
                [
                    float(candidate_vector[fid]) - float(live_vector[fid])
                    for fid in family_ids
                ],
                dtype=np.float64,
            )
            vs_live[str(seed)] = summarize(deltas)
        result["controls"][control_name] = {
            "seeds": {
                str(seed): entry[0] for seed, entry in zip(SEEDS, entries, strict=True)
            },
            "paired_diagnostics_vs_live_sized_control_per_seed": vs_live,
            "distinct_adapter_sha_by_seed": len(
                {tuple(entry[0]["adapter_sha256_by_fold"]) for entry in entries}
            )
            == 3,
            "inner_family_vectors_all_differ": len(
                {
                    hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                    for v in vectors
                }
            )
            == 3,
            "control_seed12_vs_seed13_paired_sanity": summarize(
                np.asarray(
                    [
                        float(vectors[0][family]) - float(vectors[1][family])
                        for family in family_ids
                    ]
                )
            ),
        }

    for label, (arm, config, control_name) in CONFIGS.items():
        candidate_rows, paired_rows, summaries = [], [], {}
        for seed in SEEDS:
            summary, detail = load_cell(args.runs, f"{arm}-{config}", seed)
            control_summary, control_detail = control_cache[control_name][
                SEEDS.index(seed)
            ]
            family = detail["family_gains_vs_r882"]
            control = control_detail["family_gains_vs_r882"]
            if set(family) != set(control):
                raise RuntimeError(f"{label} seed{seed}: paired family sets differ")
            paired = {fid: float(family[fid]) - float(control[fid]) for fid in family}
            recorded = detail["paired_family_gains_vs_control"]
            if recorded is None or any(
                abs(paired[fid] - float(recorded[fid])) > 1e-12 for fid in paired
            ):
                raise RuntimeError(f"{label} seed{seed}: paired vector mismatch")
            if summary["validation_scored"] or summary["validation_payloads_opened"]:
                raise RuntimeError(
                    f"{label} seed{seed}: unexpected terminal validation read"
                )
            candidate_rows.append(family)
            paired_rows.append(paired)
            summaries[str(seed)] = {
                "candidate": summary,
                "paired_diagnostics": summary["paired_diagnostics_diagnostic_only"],
                "candidate_mean_vs_r882": float(
                    np.mean(list(map(float, family.values())))
                ),
                "paired": summarize(
                    np.asarray(list(paired.values()), dtype=np.float64)
                ),
            }
        family_ids = sorted(paired_rows[0])
        if any(set(row) != set(family_ids) for row in paired_rows):
            raise RuntimeError(f"{label}: inconsistent family inventory")
        distinct = len(
            {
                hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                for v in candidate_rows
            }
        ) == len(SEEDS)
        result["configurations"][label] = {
            "arm": arm,
            "config": config,
            "matched_control": control_name,
            "per_seed": summaries,
            "inner_family_vectors_all_differ": distinct,
            "pooled": clustered_pooled(paired_rows, family_ids, label),
            "continuity_vs_r882_pooled_mean": float(
                np.mean([v for row in candidate_rows for v in row.values()])
            ),
            "no_best_of_seed_selection": True,
            "selected_terminal_seed_if_admitted": 12,
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
    print(
        json.dumps(
            {
                "configurations": len(result["configurations"]),
                "paired_admissions": sum(
                    c["pooled"]["paired_admitted"]
                    for c in result["configurations"].values()
                ),
                "terminal_dl_val_reads": 0,
                "out": str(args.out) if args.out else None,
            }
        )
    )


if __name__ == "__main__":
    main()
