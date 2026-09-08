"""Frozen broad critic ablation and training schedule."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import content_sha256


def broad_critic_experiment(
    *,
    paired_seeds: Sequence[int] = (17, 29, 43),
) -> dict[str, Any]:
    seeds = list(paired_seeds)
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("real-only/mixed ablation requires at least two unique paired seeds")
    runs = []
    for seed in seeds:
        runs.extend(
            (
                {
                    "run_id": f"real-only-seed-{seed}",
                    "seed": seed,
                    "model_path": "conditioned",
                    "corpus": "real_only",
                    "ensemble_eligible": False,
                },
                {
                    "run_id": f"mixed-seed-{seed}",
                    "seed": seed,
                    "model_path": "conditioned",
                    "corpus": "real_plus_synthetic",
                    "ensemble_eligible": True,
                },
            )
        )
    runs.append(
        {
            "run_id": f"genome-only-seed-{seeds[0]}",
            "seed": seeds[0],
            "model_path": "genome_only",
            "corpus": "real_plus_synthetic",
            "ensemble_eligible": False,
        }
    )
    plan = {
        "schema": "BroadCriticExperiment.v1",
        "paired_seeds": seeds,
        "ensemble_members": 3,
        "runs": runs,
        "epochs": 300,
        "minimum_early_stop_epoch": 100,
        "patience": 30,
        "checkpoint_retention": {
            "latest": 3,
            "every_n_epochs": 10,
            "retain_every_best": True,
        },
        "pre_online_domain_mix": {"real": 0.60, "synthetic": 0.40, "online": 0.0},
        "online_domain_mix": {"real": 0.50, "synthetic": 0.40, "online": 0.10},
        "loss_weights": {
            "within_dataset_ranking": 0.60,
            "score_huber": 0.20,
            "validity_bce": 0.10,
            "runtime_huber": 0.10,
        },
        "warm_start": {
            "source": "train_only_v3.3_policy",
            "vocabulary_compatible_token_rows_only": True,
            "reset_dataset_adapters": True,
            "reset_prediction_heads": True,
            "forbid_test_rmse_score_head": True,
        },
    }
    plan["experiment_sha256"] = content_sha256(plan)
    return plan
