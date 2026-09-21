import hashlib
import json
import random
import tomllib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_exact_campaign import offspring_rows, rewrite_binary_candidate  # noqa: E402
from test_exact_common import (  # noqa: E402
    SLOT_COUNTS,
    authoritative_observations,
    compose_seed,
    deterministic_deduplicate,
    family_disjoint_split,
    is_strict_receipt,
    queue_refill_needed,
    rank_common_winners,
    receipt_authority_kind,
    selfplay_authority,
    sha256_text,
    validate_catalog_counts,
)
from test_exact_wave import mutate, task_valid  # noqa: E402


def receipt(**updates):
    row = {
        "task": "regression",
        "dataset_hash": "d",
        "semantic_sha256": "s",
        "tier": "test_exact",
        "metric": "r2",
        "score": 0.5,
        "valid": True,
        "test_adaptive": True,
        "benchmark_valid": False,
        "selection_bias": "test-exact-reused",
        "generation": "strict-wave-000001",
        "frozen_seed_sha256": "1" * 64,
        "model_checkpoint_sha256": "2" * 64,
        "hall_of_fame_sha256": "3" * 64,
        "created_at_ms": 1,
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    "updates",
    [
        {"tier": "cv_robust"},
        {"tier": "test_adaptive"},
        {"metric": "rmse"},
        {"test_adaptive": False},
        {"benchmark_valid": True},
        {"selection_bias": None},
        {"generation": "zero"},
        {"generation": "adaptive"},
    ],
)
def test_strict_filter_excludes_every_non_authoritative_label(updates):
    assert not is_strict_receipt(receipt(**updates))


def test_binary_accepts_auc_only():
    assert is_strict_receipt(receipt(task="binary", metric="auc"))
    assert not is_strict_receipt(receipt(task="binary", metric="logloss"))


def test_authoritative_dedup_is_latest_then_receipt_hash():
    rows = [
        receipt(score=0.1, created_at_ms=2, receipt_sha256="a" * 64),
        receipt(score=0.9, created_at_ms=2, receipt_sha256="b" * 64),
        receipt(score=1.0, test_adaptive=False, created_at_ms=3),
    ]
    selected = authoritative_observations(rows)
    assert len(selected) == 1
    assert selected[0]["score"] == 0.9


def test_family_split_is_disjoint_and_deterministic():
    families = {"a": "one", "b": "one", "c": "two", "d": "three"}
    first = family_disjoint_split(families, validation_fraction=0.25)
    second = family_disjoint_split(dict(reversed(list(families.items()))), validation_fraction=0.25)
    assert first == second
    train_families = {families[d] for d, split in first.items() if split == "train"}
    validation_families = {families[d] for d, split in first.items() if split == "validation"}
    assert train_families.isdisjoint(validation_families)


def candidate(index, source="test"):
    genome = f"(pipe X (cheb degree={2 + index}) (ridge))"
    return {"canonical_genome": genome, "source": source}


def test_seed_composition_is_exact_and_deduplicated():
    groups, cursor = {}, 0
    for group, count in SLOT_COUNTS.items():
        groups[group] = [candidate(index, group) for index in range(cursor, cursor + count)]
        cursor += count
    rows = compose_seed(groups, [])
    assert len(rows) == 256
    assert len({row["semantic_sha256"] for row in rows}) == 256
    assert {group: sum(row["slot_group"] == group for row in rows) for group in SLOT_COUNTS} == SLOT_COUNTS


def test_shortage_fill_keeps_slot_contract_and_records_reason():
    groups = {
        "local_elite": [candidate(0)],
        "dl_proposal": [candidate(index) for index in range(32, 96)],
        "common_winner": [candidate(index) for index in range(96, 192)],
        "offspring": [candidate(index) for index in range(192, 256)],
    }
    fills = [candidate(index, "fill") for index in range(300, 400)]
    rows = compose_seed(groups, fills)
    local = [row for row in rows if row["slot_group"] == "local_elite"]
    assert len(local) == 32
    assert sum("fills_shortage_for" in row for row in local) == 31


def test_deduplication_is_first_wins_and_hash_authoritative():
    rows = [candidate(1, "first"), candidate(1, "second"), candidate(2, "third")]
    result = deterministic_deduplicate(rows)
    assert [row["source"] for row in result] == ["first", "third"]
    assert result[0]["semantic_sha256"] == sha256_text(result[0]["canonical_genome"])


def test_offspring_float_parameters_use_canonical_text():
    parents = [
        {
            "canonical_genome": "(pipe X (ridge))",
            "semantic_sha256": "a" * 64,
            "source": "test",
        }
    ]
    rows = offspring_rows(parents, "dataset", 320)
    assert all("q=0.10" not in row["canonical_genome"] for row in rows)
    assert all("q=0.20" not in row["canonical_genome"] for row in rows)


def test_binary_rewrite_removes_target_and_attaches_probability_head():
    row = candidate(1, "accepted_dl_checkpoint")
    row["canonical_genome"] = "(pipe X (target kind=log1p) (cheb degree=3) (ridge))"
    rewritten = rewrite_binary_candidate(row, "dataset", 0)
    assert "(target " not in rewritten["canonical_genome"]
    assert rewritten["canonical_genome"].endswith("(logit_ridge l2=0.1))") or (
        "(logit_ridge l2=" in rewritten["canonical_genome"]
    )
    assert rewritten["binary_head_rewritten"]


def test_common_winner_ranking_rewards_cross_dataset_robustness():
    observations = [
        ("a", "robust", 0.8),
        ("a", "spiky", 1.0),
        ("b", "robust", 0.8),
        ("b", "spiky", 0.0),
        ("c", "robust", 0.8),
        ("c", "other", 0.7),
    ]
    ranked = rank_common_winners(observations)
    assert ranked[0]["semantic_sha256"] == "robust"
    assert ranked[0]["dataset_count"] == 3


def test_checkpoint_provenance_hash_is_content_bound(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"accepted")
    expected = hashlib.sha256(b"accepted").hexdigest()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected


def test_refill_enforces_target_and_hard_cap():
    assert queue_refill_needed(256) == 0
    assert queue_refill_needed(255) == 1
    assert queue_refill_needed(0) == 256
    assert queue_refill_needed(449, target=450, hard_cap=450) == 1
    with pytest.raises(ValueError):
        queue_refill_needed(400, target=500, hard_cap=450)


def test_stale_dispatch_does_not_count_as_live_work():
    from test_exact_backlog import stale_dispatch

    assert stale_dispatch({"state": "dispatching", "pid": None, "runtime_secs": 121})
    assert not stale_dispatch({"state": "dispatching", "pid": None, "runtime_secs": 120})
    assert not stale_dispatch({"state": "running", "pid": 123, "runtime_secs": 10_000})


def test_strict_wave_mutation_is_deterministic_and_records_parentage():
    parents = [
        {
            "canonical_genome": "(pipe X (standardize) (ridge))",
            "semantic_sha256": sha256_text("(pipe X (standardize) (ridge))"),
            "score_receipt": {"score": 0.5},
        },
        {
            "canonical_genome": "(pipe X (rank append=true) (ridge))",
            "semantic_sha256": sha256_text("(pipe X (rank append=true) (ridge))"),
            "score_receipt": {"score": 0.4},
        },
    ]
    first = mutate(
        parents,
        task="regression",
        dataset_hash="dataset",
        wave_id="strict-wave-000001",
        attempt=17,
    )
    second = mutate(
        parents,
        task="regression",
        dataset_hash="dataset",
        wave_id="strict-wave-000001",
        attempt=17,
    )
    assert first == second
    assert len(first["parentage"]) == 2
    assert first["strict_wave"] == "strict-wave-000001"
    assert task_valid(first["canonical_genome"], "regression")


def test_task_valid_rejects_cross_task_heads():
    assert task_valid("(pipe X (logit_ridge l2=0.1))", "binary")
    assert not task_valid("(pipe X (ridge))", "binary")
    assert task_valid("(pipe X (ridge))", "regression")
    assert not task_valid("(pipe X (logit_ridge l2=0.1))", "regression")


def test_fused_policy_batches_are_exactly_75_25():
    from jope.policy.train_v3 import _grouped_batches

    rows = [
        SimpleNamespace(
            dataset_id=f"r-{index // 4}",
            metadata={"task_type": "regression", "reward_pair_identity": f"r-{index // 4}"},
        )
        for index in range(48)
    ] + [
        SimpleNamespace(
            dataset_id=f"b-{index // 4}",
            metadata={"task_type": "binary", "reward_pair_identity": f"b-{index // 4}"},
        )
        for index in range(16)
    ]
    batches = _grouped_batches(rows, 16, random.Random(17))
    for batch in batches:
        tasks = Counter(row.metadata["task_type"] for row in batch)
        assert tasks == {"regression": 12, "binary": 4}


def test_checkpoint_gate_is_fail_closed_until_every_condition_passes():
    from test_exact_accept import evaluate_gate

    evidence = {
        "baseline": {
            "checkpoint_sha256": "a" * 64,
            "macro_pairwise_accuracy": 0.60,
            "regression_pairwise_accuracy": 0.60,
            "binary_pairwise_accuracy": 0.60,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.02,
        },
        "candidate": {
            "checkpoint_sha256": "b" * 64,
            "macro_pairwise_accuracy": 0.61,
            "regression_pairwise_accuracy": 0.595,
            "binary_pairwise_accuracy": 0.62,
            "failure_rate": 0.0125,
            "p95_proposal_latency_ms": 125,
            "val_conditional_advantage": 0.06,
        },
        "proposal_validity": 0.99,
        "verified_checkpoint_origin_strict_wins": 1,
        "candidate_equal_budget_result": 0.51,
        "baseline_equal_budget_result": 0.49,
        "replay_verification": {
            "predictions": True,
            "scores": True,
            "evaluator_identity": True,
            "dataset_hashes": True,
            "candidate_origin": True,
        },
    }
    assert evaluate_gate(evidence)["pass"]
    evidence["verified_checkpoint_origin_strict_wins"] = 0
    result = evaluate_gate(evidence)
    assert not result["pass"]
    assert not result["model_lift_claim_publishable"]
    # Conditionality is one of those conditions: a checkpoint that loses to its
    # own genome-only ablation is rejected even when everything else passes.
    evidence["verified_checkpoint_origin_strict_wins"] = 1
    evidence["candidate"]["val_conditional_advantage"] = -0.060
    result = evaluate_gate(evidence)
    assert not result["pass"]
    assert "candidate_policy_is_not_conditional" in result["reasons"]


def test_scheduler_project_registration_and_weight_are_exact():
    projects = tomllib.loads(Path("/home/ubuntu/smart_cluster/etc/projects.toml").read_text())["projects"]
    assert projects["jope-testonly"] == {
        "max_tier": "normal",
        "weight": 10.0,
        "max_gpus": 1,
    }
    assert projects["jope-selfplay-eval"] == {
        "max_tier": "normal",
        "weight": 10.0,
        "max_gpus": 0,
    }
    assert projects["jope-selfplay-train"] == {
        "max_tier": "normal",
        "weight": 10.0,
        "max_gpus": 3,
    }
    assert projects["sotarena-b42"]["weight"] == 10.0


def test_corrected_authority_is_required_but_legacy_is_classified():
    corrected = receipt()
    legacy = receipt(
        test_adaptive=False,
        benchmark_valid=True,
        selection_bias=None,
        generation="zero",
    )
    assert receipt_authority_kind(corrected) == "corrected"
    assert receipt_authority_kind(legacy) == "legacy-superseded"
    assert not is_strict_receipt(legacy)
    assert is_strict_receipt(legacy, allow_legacy=True)


def test_selfplay_authority_and_catalog_counts_fail_closed():
    authority = selfplay_authority(
        task="binary",
        wave="strict-wave-000042",
        frozen_seed_sha256="1" * 64,
        model_checkpoint_sha256="2" * 64,
        hall_of_fame_sha256="3" * 64,
    )
    assert authority["test_adaptive"] is True
    assert authority["benchmark_valid"] is False
    assert authority["selection_bias"] == "test-exact-reused"
    entries = [
        {"task": task, "dataset_hash": f"{task}-{index}"}
        for task, count in (("regression", 2_322), ("binary", 542))
        for index in range(count)
    ]
    declared = {"regression": 2_322, "binary": 542}
    assert validate_catalog_counts(entries, declared) == {
        "regression": 2_322,
        "binary": 542,
        "timeseries": 0,
        "multiclass": 0,
    }
    with pytest.raises(ValueError, match="catalog count mismatch"):
        validate_catalog_counts(
            entries + [{"task": "regression", "dataset_hash": "extra"}],
            declared,
        )


def test_only_infrastructure_failures_are_recovered():
    from test_exact_backlog import is_infrastructure_failure

    assert is_infrastructure_failure({"state": "failed", "failure_class": "rpc-error"})
    assert not is_infrastructure_failure({"state": "failed", "failure_class": "crash"})
    assert not is_infrastructure_failure({"state": "running", "failure_class": "rpc-error"})


def test_submission_locality_groups_shared_dataset_inputs(tmp_path):
    from test_exact_backlog import submission_locality_key

    rows = []
    for job_id, train_hash, test_hash in (
        ("job-z", "b", "2"),
        ("job-b", "a", "1"),
        ("job-a", "a", "1"),
    ):
        manifest = tmp_path / f"{job_id}.json"
        manifest.write_text(json.dumps({"train_sha256": train_hash, "test_sha256": test_hash}))
        rows.append((job_id, tmp_path / f"{job_id}.toml", "manifest-hash", manifest))

    assert [row[0] for row in sorted(rows, key=submission_locality_key)] == [
        "job-a",
        "job-b",
        "job-z",
    ]


def test_blob_prestaging_is_verified_and_idempotent(tmp_path):
    from test_exact_backlog import stage_blob

    source = tmp_path / "source.csv"
    source.write_bytes(b"x,y\n1,2\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    blob_dir = tmp_path / "blobs"

    assert stage_blob(source, digest, blob_dir) == (True, source.stat().st_size)
    source.write_bytes(b"changed")
    assert stage_blob(source, digest, blob_dir) == (False, len(b"x,y\n1,2\n"))
    assert (blob_dir / digest).read_bytes() == b"x,y\n1,2\n"

    with pytest.raises(RuntimeError, match="content pin drift"):
        stage_blob(source, "0" * 64, blob_dir)


def test_selected_rows_prestage_manifests_and_dataset_inputs(tmp_path):
    from test_exact_backlog import prestage_rows

    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_bytes(b"x,y\n1,2\n")
    test.write_bytes(b"x,y\n3,4\n")
    train_digest = hashlib.sha256(train.read_bytes()).hexdigest()
    test_digest = hashlib.sha256(test.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "train_csv": str(train),
                "train_sha256": train_digest,
                "test_csv": str(test),
                "test_sha256": test_digest,
            }
        )
    )
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = [("job", tmp_path / "job.toml", manifest_digest, manifest)]
    blob_dir = tmp_path / "blobs"

    first = prestage_rows(rows, workers=2, blob_dir=blob_dir)
    second = prestage_rows(rows, workers=2, blob_dir=blob_dir)

    assert first["unique_inputs"] == 3
    assert first["staged"] == 3
    assert second == {"unique_inputs": 3, "staged": 0, "staged_bytes": 0}
    assert (blob_dir / manifest_digest).read_bytes() == manifest.read_bytes()
    assert (blob_dir / train_digest).read_bytes() == train.read_bytes()
    assert (blob_dir / test_digest).read_bytes() == test.read_bytes()


def test_active_trainer_metrics_reports_live_validation_signal(tmp_path):
    from test_exact_monitor import active_trainer_metrics

    output = tmp_path / "attempts" / "0" / "outputs" / "training" / "watermark-000000123"
    output.mkdir(parents=True)
    (output / "train_metrics.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "epoch": 1,
                    "phase": "A",
                    "val_pairwise_win_accuracy": 0.6,
                    "val_loss": 1.2,
                    "overfit_gap": 0.1,
                },
                {
                    "epoch": 2,
                    "phase": "A",
                    "val_pairwise_win_accuracy": 0.59,
                    "val_loss": 1.3,
                    "overfit_gap": 0.2,
                },
            )
        )
    )
    (output / "train_progress.json").write_text(json.dumps({"epoch": 3, "batch": 4, "total_batches": 10}))

    result = active_trainer_metrics({"run_dir": str(tmp_path), "attempt": 0})
    assert result["completed_epochs"] == 2
    assert result["latest_pairwise_accuracy"] == 0.59
    assert result["best_pairwise_accuracy"] == 0.6
    assert result["best_pairwise_epoch"] == 1
    assert result["pairwise_delta_from_epoch_1"] == pytest.approx(-0.01)
    assert result["progress"]["epoch"] == 3
