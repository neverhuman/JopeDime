from __future__ import annotations

from test_exact_train import *


from test_exact_train_corpus_setup import prepare_corpus

def _build_corpus_impl(
    min_rows: int = 1,
    *,
    freeze_watermark: int | None = None,
    corpus_mode: str = "full_consolidation",
    catalog_binding_path: Path | None = None,
) -> dict:
    setup = prepare_corpus(
        min_rows, freeze_watermark, corpus_mode, catalog_binding_path,
        CORPUS_DB, ROOT, SAMPLER_TAG, create_catalog_binding, sha256_file,
        complete_causal_members,
    )
    con = setup['con']
    catalog_binding_path = setup['catalog_binding_path']
    binding = setup['binding']
    binding_sha256 = setup['binding_sha256']
    pointer = setup['pointer']
    catalog = setup['catalog']
    entries = setup['entries']
    total = setup['total']
    watermark = setup['watermark']
    causal_members = setup['causal_members']
    causal_pair_stats = setup['causal_pair_stats']
    legacy_snapshot = setup['legacy_snapshot']
    snapshot = setup['snapshot']
    # The snapshot identity is fully determined by (watermark, mode, sampler),
    # all of which are known here — so the cached-snapshot check MUST happen
    # before the candidate query, not after it. It used to sit below a
    # `.fetchall()` of the whole windowed query, so every idempotent re-freeze
    # paid the full multi-GB scan and sort and then threw the result away.
    if snapshot.exists():
        manifest_path = snapshot / "corpus-manifest.json"
        if not manifest_path.is_file():
            con.close()
            raise RuntimeError(f"incomplete corpus snapshot requires quarantine: {snapshot}")
        manifest = json.loads(manifest_path.read_text())
        con.close()
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="rows_frozen",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
        )
        packed = ensure_packed(
            manifest_path, catalog_binding_path=catalog_binding_path
        )
        packed_manifest_path = (
            ROOT
            / "training"
            / "packed"
            / packed["binding_sha256"]
            / "manifest.json"
        )
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="packed",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
            packed_manifest_path=packed_manifest_path,
        )
        publication = publish_corpus_pointer(
            manifest_path,
            packed_manifest_path,
            watermark,
        )
        if publication["published"]:
            publish_corpus_freeze_state(
                watermark=watermark,
                stage="published",
                manifest_path=manifest_path,
                catalog_binding_path=catalog_binding_path,
                packed_manifest_path=packed_manifest_path,
            )
        return manifest
    projection = ",".join(OBSERVATION_COLUMNS)
    # The ORDER BY ... LIMIT exists ONLY to decide which rows are inside the
    # watermark. When the watermark covers every row in the table it cannot bind
    # (the strict predicate selects a subset of `total` rows and the limit is
    # `total`), so the whole 4.6M-row five-column sort is dead work. Dropping it
    # is output-identical, not an approximation: membership of `frozen` is
    # unchanged, and `dataset_rank` is assigned by its own window ORDER BY
    # (created_at_ms, semantic_sha256) which does not depend on the CTE's order.
    covers_every_row = watermark >= total
    # Four-task freeze: the WHERE clause is derived from ENABLED_TASK_METRICS so
    # the CTE can never silently disagree with the guard or the CLI about which
    # observations are learnable. Each task is pinned to its OWN metric -- an
    # observation whose metric drifted from its task's contract stays excluded.
    # ...and an optional task additionally has to clear the dataset floor before
    # its observations are learnable. This fails CLOSED: a task under the floor
    # is excluded from training while its evaluation continues, so the evidence
    # that clears the floor keeps accumulating. Mandatory tasks are never gated.
    trainable_tasks = trainable_task_metrics(con)
    task_predicate = " OR ".join(
        f"(task='{task}' AND metric='{metric}')"
        for task, metric in sorted(trainable_tasks.items())
    )
    if covers_every_row:
        frozen_cte = f"""
            WITH frozen AS (
              SELECT {projection} FROM observations
              WHERE {task_predicate}
            )
        """
        frozen_params: tuple = ()
    else:
        frozen_cte = f"""
            WITH frozen AS (
              SELECT {projection} FROM observations
              WHERE {task_predicate}
              ORDER BY created_at_ms,receipt_sha256,task,dataset_hash,semantic_sha256
              LIMIT ?
            )
        """
        frozen_params = (watermark,)
    # ONE candidate query for BOTH corpus modes and BOTH splits. The old code had
    # three separate queries and the split-specific cap only existed on the
    # recent_plus_replay path, so full_consolidation quietly handed validation
    # 128 rows per dataset where the other mode handed it 32.
    #
    # ((dataset_rank - 1) * cap) % dataset_total < cap keeps EXACTLY
    # min(dataset_total, cap) rows, evenly spaced across the dataset's history:
    # the residues (r-1)*cap mod total are the multiples of gcd(cap, total), each
    # hit gcd times, and there are cap/gcd of them below cap. It is the SQL twin
    # of even_spread(). The second clause reserves the newest RECENCY_SLOTS rows
    # for recent_plus_replay (and is unreachable when recency=0, since
    # dataset_rank <= dataset_total always).
    recency = RECENCY_SLOTS if corpus_mode == "recent_plus_replay" else 0
    cursor = con.execute(
        frozen_cte
        + """
          , ranked AS (
            SELECT frozen.*,
              ROW_NUMBER() OVER (
                PARTITION BY task,dataset_hash
                ORDER BY created_at_ms,semantic_sha256
              ) AS dataset_rank,
              COUNT(*) OVER (PARTITION BY task,dataset_hash) AS dataset_total
            FROM frozen
          )
          SELECT * FROM ranked
          WHERE ((dataset_rank - 1) * ?) % dataset_total < ?
             OR dataset_rank > dataset_total - ?
          ORDER BY task,dataset_hash,dataset_rank
        """,
        (*frozen_params, CANDIDATE_ROW_CAP, CANDIDATE_ROW_CAP, recency),
    )

    sys.path.insert(0, str(prime_source_root()))
    from jope.prime_label_to_corpus import GenomeEngineResolver
    from jope.schema import TrainingRow

    resolver = GenomeEngineResolver()
    kept, skipped = Counter(), Counter()
    per_dataset_rows: Counter = Counter()
    task_counts: Counter = Counter()
    candidate_label_count = 0

    # --- phase 1: catalog + encodability filter, BEFORE any quota -------------
    # This ordering is the whole point. ~17% of observations resolve to no
    # grammar engine, and the old code capped first and filtered second, so the
    # "cap" it enforced was on candidates while the number that actually reached
    # the model was whatever survived — 18 rows for one dataset and 128 for
    # another out of the same nominal cap of 128.
    #
    # STREAMED, not `.fetchall()`. At the 1.8M watermark the materialised result
    # was ~366k rows; at COUNT(*) it is ~6x that, and the discarded ~17%
    # unencodable and not-in-catalog rows never need to exist in memory at all.
    survivors: dict[tuple[str, str], list] = defaultdict(list)
    for row in cursor:
        candidate_label_count += 1
        task_counts[row["task"]] += 1
        identity = (row["task"], row["dataset_hash"])
        if entries.get(identity) is None:
            skipped["not_in_frozen_catalog"] += 1
            continue
        engine = resolver.resolve(row["canonical_genome"], task_type=row["task"])
        if engine is None:
            skipped["unencodable"] += 1
            continue
        survivors[identity].append((row, engine))
    # Superset-over-enabled, not equality: the mandatory backbone (regression,
    # binary) must be present, unknown tasks must never slip in, and the optional
    # tasks (multiclass, timeseries) join as their first observations arrive.
    # The old equality check would have REJECTED the corpus the moment the first
    # multiclass observation landed -- turning new-task success into a trainer
    # outage.
    present = set(task_counts)
    if not present.issuperset(MANDATORY_TASKS):
        con.close()
        raise RuntimeError(
            f"shared trainer needs the mandatory tasks {sorted(MANDATORY_TASKS)}, "
            f"found {dict(task_counts)}"
        )
    if not present.issubset(ENABLED_TASK_METRICS):
        con.close()
        raise RuntimeError(
            f"corpus contains tasks outside the enabled contract: {dict(task_counts)}"
        )

    blob = Path(pointer["data"]).read_bytes()
    temporary = snapshot.with_name(f".{snapshot.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    handles = {
        "train": (temporary / "corpus.train.jsonl").open("w"),
        "validation": (temporary / "corpus.val.jsonl").open("w"),
    }
    try:
        # --- phase 2: build the per-dataset pool ------------------------------
        # train      -> a pool the per-epoch equal-quota draw samples from
        # validation -> exactly VALIDATION_QUOTA rows, fixed for every epoch, so
        #               the selection metric moves for learning reasons only
        pools: dict[tuple[str, str], list] = {}
        for identity, encodable in survivors.items():
            split = entries[identity]["split"]
            if split == "validation":
                pools[identity] = even_spread(encodable, VALIDATION_QUOTA)
                continue
            if corpus_mode == "recent_plus_replay":
                recent = encodable[-RECENCY_SLOTS:]
                replay = even_spread(
                    encodable[: max(0, len(encodable) - RECENCY_SLOTS)],
                    CANDIDATE_ROW_CAP - len(recent),
                )
                pools[identity] = [*replay, *recent]
            else:
                pools[identity] = even_spread(encodable, CANDIDATE_ROW_CAP)

        # --- phase 3: materialise ---------------------------------------------
        for identity in sorted(pools):
            entry = entries[identity]
            split = entry["split"]
            quota = VALIDATION_QUOTA if split == "validation" else EQUAL_QUOTA
            pool = pools[identity]
            # The dataset embedding is per DATASET, not per row — unpack once.
            offset = int(entry["catalog_index"]) * STRIDE_FLOATS * 4
            values = struct.unpack(
                f"<{STRIDE_FLOATS}f",
                blob[offset : offset + STRIDE_FLOATS * 4],
            )
            embedding = list(values[:1536])
            target_embedding = list(values[1536:])
            for row, engine in pool:
                reward = task_reward(row["task"], row["score"])
                strict_wave = (
                    row["strict_wave"]
                    if "strict_wave" in row.keys() and row["strict_wave"]
                    else "strict-wave-000000"
                )
                metadata = {
                    "provenance": "internal_test_exact_selfplay",
                    "task_type": row["task"],
                    "split": split,
                    "source_family": entry["source_family"],
                    "dataset_hash": row["dataset_hash"],
                    "semantic_plan_hash": row["semantic_sha256"],
                    "score_tier": "test_exact",
                    "test_adaptive": True,
                    "benchmark_valid": False,
                    "selection_bias": "test-exact-reused",
                    "generation": strict_wave,
                    "score_robust": reward,
                    "oof_r2_raw": reward,
                    "reward_pair_identity": (
                        f"test-exact:{row['task']}:{row['dataset_hash']}:{row['metric']}"
                    ),
                    # Left at 1.0 on purpose. train_v3.py:1409-1416 overwrites
                    # seq_weight for every "generated" provenance (ours included)
                    # before the first batch is built, and it reaches only the
                    # imitation loss anyway — see equal_quota_draw's docstring.
                    "seq_weight": 1.0,
                    # Read by the per-epoch sampler in train().
                    "epoch_quota": quota,
                    "quota_pool_size": len(pool),
                    "quota_deficit": max(0, quota - len(pool)),
                    "observation_rank": int(row["dataset_rank"]),
                    "dataset_observation_total": int(row["dataset_total"]),
                    "promotion_eligible": True,
                    "promotion_scope": "internal-selected-test-flywheel-only",
                    "benchmark_promotion_eligible": False,
                    "label_created_at_ms": int(row["created_at_ms"]),
                    "model_checkpoint_sha256": row["model_checkpoint_sha256"],
                    "hall_of_fame_sha256": row["hall_of_fame_sha256"],
                    "receipt_sha256": row["receipt_sha256"],
                }
                causal = causal_members.get(
                    (row["receipt_sha256"], row["semantic_sha256"])
                )
                if causal is not None:
                    metadata.update(causal)
                training_row = TrainingRow(
                    dataset_id=row["dataset_hash"],
                    source="internal_test_exact_selfplay",
                    embedding=embedding,
                    target_embedding=target_embedding,
                    genome=row["canonical_genome"],
                    canonical_genome=row["canonical_genome"],
                    engine=engine,
                    valid=True,
                    metric=row["metric"],
                    score=reward,
                    cost_ms=float(row["runtime_seconds"]) * 1000.0,
                    metadata=metadata,
                )
                handles[split].write(json.dumps(training_row.to_json()) + "\n")
                kept[f"{row['task']}/{split}"] += 1
            per_dataset_rows[(identity[0], split, identity[1])] = len(pool)
    finally:
        for handle in handles.values():
            handle.close()
        con.close()

    # Per-task/split acceptance evidence, recorded in the manifest so a corpus
    # can be audited without re-reading 6 GB of JSONL.
    pool_counts: dict[str, list[int]] = defaultdict(list)
    for (task, split, _), count in per_dataset_rows.items():
        pool_counts[f"{task}/{split}"].append(count)
    quota_stats = {
        key: quota_distribution(
            counts, VALIDATION_QUOTA if key.endswith("/validation") else EQUAL_QUOTA
        )
        for key, counts in sorted(pool_counts.items())
    }

    def epoch_draw_count(pool_size: int, quota: int) -> int:
        if QUOTA_TAIL_REPLICATE and pool_size:
            return quota
        return min(pool_size, quota)

    epoch_draw_stats = {}
    for key, counts in sorted(pool_counts.items()):
        quota = VALIDATION_QUOTA if key.endswith("/validation") else EQUAL_QUOTA
        epoch_draw_stats[key] = quota_distribution(
            [epoch_draw_count(count, quota) for count in counts], quota
        )
    def quarantine(reason: str) -> None:
        rejected = ROOT / "training" / "rejected" / temporary.name.removeprefix(".")
        rejected.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, rejected)
        raise RuntimeError(f"{reason}; partial snapshot moved to {rejected}")

    if not kept["regression/validation"] or not kept["binary/validation"]:
        quarantine(f"both task validation splits must be populated: {dict(kept)}")

    # GUARDRAIL 2 (ops/generalization-plan.md): train and validation per-dataset
    # medians must be within 2x. Asserted on the EPOCH-DRAW medians, and it fails
    # the freeze rather than warning: the sampler defect it catches (train median
    # 6 vs validation median 485) produced an 8-nat "overfit gap" that was read as
    # a model problem for a full day.
    median_scopes = split_median_divergence(epoch_draw_stats)
    diverged = {
        scope: values
        for scope, values in median_scopes.items()
        if values["ratio"] is not None and values["ratio"] > SPLIT_MEDIAN_RATIO_MAX
    }
    if diverged:
        quarantine(
            "train/validation per-dataset epoch-draw medians diverge by more than "
            f"{SPLIT_MEDIAN_RATIO_MAX}x: {diverged}"
        )
    manifest = {
        "schema": "JopeTestOnlyTrainingCorpus.v1",
        "source_db": str(CORPUS_DB),
        "source_db_sha256": sha256_file(CORPUS_DB),
        "source_label_watermark": watermark,
        "corpus_mode": corpus_mode,
        "selected_label_count": sum(kept.values()),
        "candidate_label_count": candidate_label_count,
        "source_label_total": total,
        "watermark_covers_every_row": covers_every_row,
        "sampler": {
            "id": SAMPLER_TAG,
            "equal_quota": EQUAL_QUOTA,
            "validation_quota": VALIDATION_QUOTA,
            "candidate_row_cap": CANDIDATE_ROW_CAP,
            "recency_slots": recency,
            "tail_replicate": QUOTA_TAIL_REPLICATE,
            "pool_selection": "even_spread_over_dataset_history",
            "quota_applied": "post_encodability_filter",
            "epoch_resample": "train split only, per-epoch seed, in train()",
        },
        # Rows per dataset held in the corpus file (the per-epoch draw samples
        # EQUAL_QUOTA of these for train; validation pool == quota, so its pool
        # stats and its epoch-draw stats are identical by construction).
        "quota_stats": quota_stats,
        # Rows per dataset the model actually sees in ONE epoch.
        "epoch_draw_stats": epoch_draw_stats,
        "split_median_divergence": median_scopes,
        "strict_filter": {
            "regression": {"score_tier": "test_exact", "metric": "r2"},
            "binary": {"score_tier": "test_exact", "metric": "auc"},
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "promotion_scope": "internal-selected-test-flywheel-only",
        },
        "counts": dict(kept),
        "skipped": dict(skipped),
        "train": str(snapshot / "corpus.train.jsonl"),
        "train_sha256": sha256_file(temporary / "corpus.train.jsonl"),
        "validation": str(snapshot / "corpus.val.jsonl"),
        "validation_sha256": sha256_file(temporary / "corpus.val.jsonl"),
        "catalog_sha256": pointer["catalog_sha256"],
        "catalog_binding": str(catalog_binding_path),
        "catalog_binding_sha256": binding_sha256,
        "source_family_disjoint": catalog["source_family_disjoint"],
        "causal_pair_receipt_stats": causal_pair_stats,
    }
    # GUARDRAIL 1 (ops/generalization-plan.md): alarm when the corpus dataset
    # count FALLS between freezes. Judged BEFORE the manifest is written, because
    # `test_exact_packed_corpus.verify` binds the packed corpus to the manifest's
    # sha256 — a manifest rewritten after packing invalidates the binding and the
    # trainer then refuses the corpus outright.
    per_split_datasets = {
        key: int(value["datasets"]) for key, value in quota_stats.items()
    }
    history_record = {
        "schema": "JopeCorpusFreezeHistory.v1",
        "at_ms": int(time.time() * 1000),
        "watermark": watermark,
        "source_label_total": total,
        "corpus_mode": corpus_mode,
        "sampler": SAMPLER_TAG,
        "manifest": str(snapshot / "corpus-manifest.json"),
        "datasets_total": sum(per_split_datasets.values()),
        "per_split_datasets": per_split_datasets,
        "per_split_rows": {
            key: int(value["rows"]) for key, value in quota_stats.items()
        },
        "epoch_draw_rows": sum(
            int(value["rows"]) for value in epoch_draw_stats.values()
        ),
        "epoch_draw_medians": {
            key: float(value["median"]) for key, value in epoch_draw_stats.items()
        },
        "split_median_divergence": median_scopes,
        "candidate_label_count": candidate_label_count,
        "selected_label_count": sum(kept.values()),
    }
    guard = publish_freeze_guard(history_record)
    manifest["freeze_guard"] = {
        "regressed": guard["regressed"],
        "previous_best_datasets": guard["previous_best_datasets"],
        "datasets_total": guard["datasets_total"],
    }
    atomic_json(temporary / "corpus-manifest.json", manifest)
    os.replace(temporary, snapshot)
    # Only now is the freeze durable, so only now does it enter the history the
    # next freeze compares itself against.
    append_freeze_history(history_record)
    manifest_path = snapshot / "corpus-manifest.json"
    publish_corpus_freeze_state(
        watermark=watermark,
        stage="rows_frozen",
        manifest_path=manifest_path,
        catalog_binding_path=catalog_binding_path,
    )
    packed = ensure_packed(
        manifest_path,
        catalog_binding_path=catalog_binding_path,
    )
    packed_manifest = (
        ROOT / "training" / "packed" / packed["binding_sha256"] / "manifest.json"
    )
    publish_corpus_freeze_state(
        watermark=watermark,
        stage="packed",
        manifest_path=manifest_path,
        catalog_binding_path=catalog_binding_path,
        packed_manifest_path=packed_manifest,
    )
    publication = publish_corpus_pointer(manifest_path, packed_manifest, watermark)
    if publication["published"]:
        publish_corpus_freeze_state(
            watermark=watermark,
            stage="published",
            manifest_path=manifest_path,
            catalog_binding_path=catalog_binding_path,
            packed_manifest_path=packed_manifest,
        )
    return manifest
