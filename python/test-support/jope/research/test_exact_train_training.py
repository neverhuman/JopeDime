from __future__ import annotations

from test_exact_train import *

def _train_impl(
    corpus_manifest: Path,
    output: Path,
    *,
    device: str,
    warm: Path,
    recipe: str = "a",
    seed: int = 20260726,
    task: str,
    mode: str,
    packed_manifest: Path | None = None,
    packed_source_train: Path | None = None,
    packed_source_validation: Path | None = None,
    packed_catalog_data: Path | None = None,
    accepted_parent_sha256: str | None = None,
    optional_task_readiness: Path | None = None,
    _prime_source_root,
) -> dict:
    if task not in ENABLED_TASK_METRICS:
        raise ValueError("trainer task is outside the four-task contract")
    if mode not in {"shadow", "production"}:
        raise ValueError("trainer mode must be shadow or production")
    if task not in MANDATORY_TASKS and mode != "shadow":
        raise ValueError("optional-task training is shadow-only")
    readiness = require_optional_task_admission(task, optional_task_readiness)
    if accepted_parent_sha256 is not None and (
        len(accepted_parent_sha256) != 64
        or any(character not in "0123456789abcdef" for character in accepted_parent_sha256)
    ):
        raise ValueError("accepted parent must be a lowercase SHA-256")
    warm_hash = sha256_file(warm)
    if warm == WARM and warm_hash != WARM_SHA:
        raise RuntimeError("accepted warm-start checkpoint hash drift")
    manifest = json.loads(corpus_manifest.read_text())
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(_prime_source_root()))
    import jope.policy.train_v3 as train_v3

    training_rows = None
    packed_artifacts = None
    causal_pair_count = 0
    descendant_receipt_count = 0
    hard_negative_count = 0
    if packed_manifest is not None:
        from test_exact_packed_corpus import load_training_rows

        training_rows, packed_artifacts = load_training_rows(
            packed_manifest,
            task=task,
            source_manifest_path=corpus_manifest,
            source_train=packed_source_train,
            source_validation=packed_source_validation,
            catalog_data=packed_catalog_data,
        )
        if recipe == "c":
            validation = [
                row
                for row in training_rows
                if (row.metadata or {}).get("split") == "validation"
            ]
            train_rows = [
                row
                for row in training_rows
                if (row.metadata or {}).get("split") == "train"
            ]
            newest = sorted(
                train_rows,
                key=lambda row: (
                    -int((row.metadata or {}).get("label_created_at_ms") or 0),
                    str((row.metadata or {}).get("receipt_sha256") or ""),
                ),
            )[:50_000]
            newest_ids = {id(row) for row in newest}
            historical = [row for row in train_rows if id(row) not in newest_ids]
            random.Random(seed).shuffle(historical)
            training_rows = [*newest, *historical[: len(newest)], *validation]
        if recipe == "d":
            causal_pair_count = len(
                {
                    (row.metadata or {}).get("causal_pair_id")
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                    and (row.metadata or {}).get("causal_pair_id")
                }
            )
            if causal_pair_count < 32:
                raise ValueError(
                    "causal-pair recipe d requires at least 32 complete "
                    "accepted-parent pairs"
                )
        if recipe == "e":
            causal_pair_count = len(
                {
                    (row.metadata or {}).get("causal_pair_id")
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "causal_accepted_parent_checkpoint_sha256"
                    )
                    == accepted_parent_sha256
                    and (row.metadata or {}).get("causal_pair_id")
                }
            )
            descendant_receipt_count = len(
                {
                    (row.metadata or {}).get(
                        "descendant_credit_receipt_sha256"
                    )
                    for row in training_rows
                    if (row.metadata or {}).get(
                        "descendant_credit_receipt_sha256"
                    )
                    and not (row.metadata or {}).get(
                        "descendant_censored", False
                    )
                }
            )
            hard_negative_count = sum(
                recipe_e_evidence_lane(
                    row.metadata or {}, accepted_parent_sha256 or ""
                )
                == "validity_instability_hard_negatives"
                for row in training_rows
            )
            if causal_pair_count < 32:
                raise ValueError(
                    "recipe E requires at least 32 complete current-parent pairs"
                )
            if descendant_receipt_count < 1:
                raise ValueError(
                    "recipe E requires completed descendant-credit receipts"
                )
            if hard_negative_count < 1:
                raise ValueError(
                    "recipe E requires validity or instability hard negatives"
                )
    if recipe not in {"a", "b", "c", "d", "e"}:
        raise ValueError("checkpoint recipe must be a, b, c, d, or e")
    if recipe == "d" and (
        mode != "shadow"
        or packed_manifest is None
        or accepted_parent_sha256 is None
    ):
        raise ValueError(
            "causal-pair recipe d is shadow-only and requires packed rows "
            "plus an accepted-parent pin"
        )
    if recipe == "e" and (
        mode != "shadow"
        or packed_manifest is None
        or accepted_parent_sha256 is None
    ):
        raise ValueError(
            "descendant-credit recipe E is shadow-only and requires packed rows "
            "plus an accepted-parent pin"
        )

    regression_fraction = 0.75

    # ---- per-epoch EQUAL-QUOTA draw -------------------------------------
    # The corpus file holds a POOL per dataset; this takes exactly
    # metadata["epoch_quota"] rows out of each pool, re-drawn every epoch from a
    # seed derived from (run seed, epoch). Equal per-epoch quota, different rows
    # each epoch: no epoch over-represents a deep dataset, and across 24 epochs
    # the deep pools are still fully exercised.
    #
    # The draw MUST be identical on every DDP rank — train_v3 shards the batch
    # list by rank and deadlocks on an uneven split — hence a private, fully
    # determined Random and a sorted dataset iteration order.
    epochs_drawn = {"count": 0}

    grouped_batches = make_grouped_batches(
        seed=seed,
        epochs_drawn=epochs_drawn,
        recipe=recipe,
        task=task,
        accepted_parent_sha256=accepted_parent_sha256,
        regression_fraction=regression_fraction,
    )

    original_grouped_batches = train_v3._grouped_batches
    original_precompute = train_v3.precompute_row_artifacts
    original_policy_batch_iterator = train_v3._iter_policy_batches
    train_v3._grouped_batches = grouped_batches
    if packed_artifacts is not None:
        train_v3._iter_policy_batches = four_worker_policy_batch_iterator(
            train_v3, original_policy_batch_iterator
        )
    if packed_artifacts is not None:
        train_v3.precompute_row_artifacts = (
            lambda rows, vocab, persistent_cache=None: {
                id(row): packed_artifacts[id(row)]
                for row in rows
                if id(row) in packed_artifacts
            }
        )

    checkpoint = output / "best.model.pt"
    resume = output / "resume.current.pt"
    resume_existing = resume if resume.is_file() else None
    gpu_monitor = GpuUtilizationMonitor(
        packed_artifacts is not None and str(device).startswith("cuda")
    )
    gpu_monitor.start()
    try:
        summary = train_v3.train_policy_v3_from_jsonl(
            manifest["train"],
            checkpoint,
            # 24 not 60: with the per-dataset caps each epoch now sees ~3.3x more
            # rows, and the selected checkpoint has been epoch 25 (the first
            # phase-B epoch) in every run to date — the remaining 35 epochs only
            # ever moved the selection metric 0.6512 -> 0.6493.
            epochs=24,
            batch_size=256,
            device=device,
            metrics_jsonl=output / "train_metrics.jsonl",
            progress_jsonl=output / "train_progress.json",
            seed=seed,
            # 5e-4 is a FROM-SCRATCH rate. Every run since generation 3 warm-starts
            # from an accepted checkpoint, so the first optimizer steps land on
            # weights that are already near a good basin and 5e-4 spends the early
            # epochs undoing them (a-g7: train loss 3.71 at epoch 1 from a warm
            # start that validates at 0.735 pairwise). 2e-4 for continuation.
            #
            # No warmup or cosine/step decay was added, deliberately:
            #   * The LambdaLR lambdas are hardcoded in train_v3.py:1784-1791 with
            #     no injection point, so any warmup means editing jope-prime —
            #     which is pinned by sha256 as `trainer_source` provenance in this
            #     and every other spec, and is shared with the live comparison run.
            #   * It would be a sixth simultaneous change in a run whose whole
            #     purpose is to measure the five already-validated ones.
            #   * The schedule is not in fact flat: the encoder group drops to
            #     0.1x at the phase A->B boundary (train_v3.py:1945), and
            #     anti_overfit adds plateau-triggered LR halving on top.
            #   * Warmup exists to blunt warm-start shock from a too-large rate;
            #     halving the rate addresses the same failure mode directly.
            lr=2e-4,
            # ONE-WAY DOOR: this moves the optimizer from 4 param groups to 5 (the
            # task/routing/stability heads split out of the reward branch to carry
            # weight decay, train_v3.py:1712-1748) and train_v3.py:1303-1310 refuses
            # to resume an anti-overfit checkpoint without it. It can therefore only
            # ever START a run, never join one -- which is what this is. Turning it
            # on also sets embedding jitter 0.05 / token dropout 0.10 on the model
            # (train_v3.py:1548-1550), EMA 0.999, weight decay 0.05 on the decayed
            # groups (0.01 otherwise), and plateau LR halving at patience 12.
            anti_overfit_experiment=True,
            # Select on CONDITIONALITY, not raw pairwise accuracy. The old metric
            # is fully satisfiable by the genome-only ablation control (which has
            # matched or beaten the conditional model every epoch of every run),
            # so it could never distinguish a policy that uses the dataset
            # embedding from one that ignores it.
            checkpoint_metric="conditional_score",
            extra_rows=None if training_rows is not None else [manifest["validation"]],
            init_from=None if resume_existing else warm,
            resume_from=resume_existing,
            prime_vocab=True,
            routing_classes=14,
            d_model=512,
            n_layers=6,
            n_heads=4,
            ffn_dim=2048,
            amp=True,
            autotune_batch_size=True,
            artifact_cache_path=(
                None if packed_artifacts is not None else output / "artifact_cache.pt"
            ),
            resume_checkpoint=resume,
            # 3, not 12. Macro validation only runs in phase B (train_v3.py:2199 gates on
            # phase == "B"), so with phase_a_epochs=10 of 24 the patience budget is spent
            # entirely inside a 14-epoch window — a patience of 12 can essentially never
            # fire, and no run has ever early-stopped.
            #
            # Measured on run c-r000024 (val_conditional_score, the selection metric):
            #   ep11 0.76710 (best) | 12 bad(1) | 13 0.76725 best, reset | 14 bad(1)
            #   15 bad(2) | 16 bad(3) -> patience 3 stops here
            # and the metric over the remaining 8 epochs went 0.76673 -> 0.76793, i.e. it
            # never recovered past the epoch-13 best by more than measurement noise (the
            # whole 24-epoch spread of val_conditional_score is 0.76519..0.76966 = 0.0045).
            # Meanwhile epoch_sec grows 190s -> 336s across a run, so those 8 epochs are
            # the most expensive ones: 2,533s of 4,547s wall clock, 56% of the generation,
            # bought for less than the noise floor.
            #
            # Bounded on both sides: 3 still tolerates two consecutive down-passes, which
            # is what the epoch 12->13 recovery above actually needed.
            early_stopping_patience=3,
            training_rows=training_rows,
            resume_extra={
                "source_label_watermark": manifest["source_label_watermark"],
                "source_corpus_manifest_sha256": sha256_file(corpus_manifest),
                "accepted_ancestor_sha256": warm_hash,
                "recipe": recipe,
                "seed": seed,
                "recipe_e_batch_mix": (
                    RECIPE_E_BATCH_MIX if recipe == "e" else None
                ),
                "recipe_e_loss_weights": (
                    RECIPE_E_LOSS_WEIGHTS if recipe == "e" else None
                ),
                "recipe_e_model_outputs": (
                    list(RECIPE_E_MODEL_OUTPUTS) if recipe == "e" else None
                ),
                "task_batch_mixture": {
                    "regression": 1.0 if task == "regression" else 0.0,
                    "binary": 1.0 if task == "binary" else 0.0,
                },
                "task": task,
                "trainer_mode": mode,
                "packed_manifest_sha256": (
                    sha256_file(packed_manifest) if packed_manifest else None
                ),
            },
        )
    finally:
        gpu_monitor.stop()
        train_v3._grouped_batches = original_grouped_batches
        train_v3.precompute_row_artifacts = original_precompute
        train_v3._iter_policy_batches = original_policy_batch_iterator
    if checkpoint.is_file():
        checkpoint_hash = sha256_file(checkpoint)
        immutable = output / f"{checkpoint_hash}.model.pt"
        shutil.copy2(checkpoint, immutable)
        summary["checkpoint_sha256"] = checkpoint_hash
        summary["checkpoint_path"] = str(immutable)
    metrics_path = output / "train_metrics.jsonl"
    metrics_rows = (
        [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
        if metrics_path.is_file()
        else []
    )
    phase_a = [
        row for row in metrics_rows
        if row.get("phase") == "A"
        and row.get("val_pairwise_win_accuracy") is not None
    ]
    phase_b = [
        row for row in metrics_rows
        if row.get("phase") == "B"
        and row.get("val_pairwise_win_accuracy") is not None
    ]
    if not phase_b:
        raise RuntimeError("trainer produced no eligible phase-B validation checkpoint")
    selected_epoch = summary.get("selected_checkpoint_epoch")
    selected = next(
        (row for row in phase_b if row.get("epoch") == selected_epoch),
        max(phase_b, key=lambda row: float(row["val_pairwise_win_accuracy"])),
    )
    summary["phase_a_diagnostic_peak"] = (
        None
        if not phase_a
        else max(float(row["val_pairwise_win_accuracy"]) for row in phase_a)
    )
    summary["selected_phase_b_accuracy"] = float(
        selected["val_pairwise_win_accuracy"]
    )
    summary["selected_phase_b_epoch"] = int(selected["epoch"])
    summary["selected_phase_b_task_accuracy"] = {
        task: selected.get(f"val_{task}_pairwise_win_accuracy")
        for task in ("regression", "binary")
    }
    summary["accepted_baseline_sha256"] = warm_hash
    summary["corpus_manifest_sha256"] = sha256_file(corpus_manifest)
    summary["accepted_ancestor_sha256"] = warm_hash
    summary["recipe"] = recipe
    summary["training_recipe"] = (
        {
            "batch_mix": RECIPE_E_BATCH_MIX,
            "loss_weights": RECIPE_E_LOSS_WEIGHTS,
            "model_outputs": list(RECIPE_E_MODEL_OUTPUTS),
            "architecture": {
                "d_model": 512,
                "layers": 6,
                "architecture_expansion": False,
            },
            "equal_rows_per_dataset_per_epoch": EQUAL_QUOTA,
            "complete_parent_pairs": causal_pair_count,
            "completed_descendant_credit_receipts": descendant_receipt_count,
            "hard_negative_rows": hard_negative_count,
            "censored_descendants_have_zero_training_weight": True,
            "learned_risk_ranking_enabled": False,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
        if recipe == "e"
        else {
            "broad_replay_fraction": 0.75,
            "exact_causal_pair_fraction": 0.25,
            "hard_novel_oversampling": True,
            "fixed_anchor_holdout": True,
            "complete_parent_pairs": causal_pair_count,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
        if recipe == "d"
        else {
            "broad_replay_fraction": 1.0,
            "exact_causal_pair_fraction": 0.0,
            "hard_novel_oversampling": False,
            "fixed_anchor_holdout": True,
            "complete_parent_pairs": 0,
            "accepted_parent_checkpoint_sha256": accepted_parent_sha256,
        }
    )
    epoch_throughput = [
        float(row["train_rows"]) / float(row["epoch_sec"])
        for row in metrics_rows
        if isinstance(row.get("train_rows"), (int, float))
        and isinstance(row.get("epoch_sec"), (int, float))
        and float(row["epoch_sec"]) > 0
    ]
    benchmark_throughput = max(
        (
            float(row.get("samples_per_second") or 0.0)
            for row in summary.get("batch_benchmark") or []
            if row.get("finite") is not False
        ),
        default=0.0,
    )
    sustained_throughput = (
        statistics.median(epoch_throughput) if epoch_throughput else 0.0
    )
    summary["packed_loader"] = {
        "deterministic_workers": 4 if packed_artifacts is not None else 0,
        "pinned_memory_prefetch": packed_artifacts is not None,
        "loader_window_batches": 8 if packed_artifacts is not None else 0,
        "sustained_training_rows_per_second": sustained_throughput,
        "batch_microbenchmark_samples_per_second": benchmark_throughput,
        "throughput_fraction_of_microbenchmark": (
            sustained_throughput / benchmark_throughput
            if benchmark_throughput > 0
            else None
        ),
        "meets_seventy_percent_gate": (
            benchmark_throughput > 0
            and sustained_throughput >= 0.70 * benchmark_throughput
        ),
    }
    summary["gpu_utilization"] = gpu_monitor.summary()
    summary["seed"] = seed
    summary["task"] = task
    summary["trainer_mode"] = mode
    summary["task_sampling_weights"] = eight_slot_task_weights(readiness)
    summary["checkpoint_contract"] = (
        "ShadowCheckpoint.v1" if mode == "shadow" else "ProductionCheckpointCandidate.v1"
    )
    summary["promotion_eligible"] = False
    atomic_json(output / "train-result.json", summary)
    return summary
