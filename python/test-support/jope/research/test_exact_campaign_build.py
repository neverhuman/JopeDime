from __future__ import annotations

from test_exact_campaign_config import *

from test_exact_campaign_population import *

def build(target_pairs: int, max_shard: int) -> dict:
    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog_meta = json.loads(Path(pointer["catalog"]).read_text())
    hof_pointer = json.loads((ROOT / "hof" / "current.json").read_text())
    hof = json.loads(Path(hof_pointer["hall_of_fame"]).read_text())
    hof_hash = hof_pointer["hall_of_fame_sha256"]
    proposals = load_proposals(PROPOSALS)
    known = database(Path(hof_pointer["known"]), readonly=True)
    ordered = dataset_order(catalog_meta["entries"], hof["observation_counts"])

    campaign_final = ROOT / "campaign" / "generation-zero.sqlite"
    campaign_final.parent.mkdir(parents=True, exist_ok=True)
    if campaign_final.exists():
        raise RuntimeError(f"immutable campaign database already exists: {campaign_final}")
    campaign_path = campaign_final.with_name(
        f".{campaign_final.name}.tmp-{os.getpid()}"
    )
    campaign = database(campaign_path)
    campaign.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE seeds(
          task TEXT,dataset_hash TEXT,slot_group TEXT,slot_index INTEGER,
          semantic_sha256 TEXT,canonical_genome TEXT,source TEXT,
          provenance_json TEXT,PRIMARY KEY(task,dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE evaluations(
          ordinal INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,
          semantic_sha256 TEXT,manifest TEXT
        );
        """
    )
    fresh_by_dataset: dict[tuple[str, str], list[dict]] = {}
    expansions_by_dataset: dict[tuple[str, str], list[dict]] = {}
    for ordinal, entry in enumerate(ordered):
        task, dataset = entry["task"], entry["dataset_hash"]
        known_semantics = {
            row[0]
            for row in known.execute(
                "SELECT semantic_sha256 FROM known WHERE task=? AND dataset_hash=?",
                (task, dataset),
            )
        }
        local = []
        for row in hof["local_elites"][task].get(dataset, []):
            local.append(
                {
                    "canonical_genome": row["canonical_genome"],
                    "semantic_sha256": row["semantic_sha256"],
                    "source": "local_strict_test_elite",
                    "score_receipt": {
                        "observation_id": row["observation_id"],
                        "receipt_ref": row["receipt_ref"],
                        "source": row["source"],
                        "score": row["score"],
                    },
                }
            )
        common = [
            {
                "canonical_genome": row["canonical_genome"],
                "semantic_sha256": row["semantic_sha256"],
                "source": "task_common_winner",
                "common_rank": {
                    key: row[key]
                    for key in (
                        "median_percentile",
                        "lower_quartile_percentile",
                        "win_count",
                        "dataset_count",
                    )
                },
            }
            for row in hof["common_winners"][task]
        ]
        dl = [dict(row) for row in proposals.get((task, dataset), [])]
        if len(dl) < 64:
            raise RuntimeError(f"{task}:{dataset} has only {len(dl)} DL proposals")
        primary_parents = [*local[:32], *dl[:64], *common[:96]]
        for row in primary_parents:
            row.setdefault("semantic_sha256", sha256_text(row["canonical_genome"]))
        offspring = offspring_rows(primary_parents, dataset, 320)
        fills = [*dl[64:], *offspring[64:], *common[96:], *local[32:]]
        seed = compose_seed(
            {
                "local_elite": local,
                "dl_proposal": dl,
                "common_winner": common,
                "offspring": offspring,
            },
            fills,
        )
        frozen_hash = sha256_text(
            json.dumps(
                [
                    {
                        "slot_group": row["slot_group"],
                        "slot_index": row["slot_index"],
                        "semantic_sha256": row["semantic_sha256"],
                    }
                    for row in seed
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        seed = [provenance(row, hof_hash, frozen_hash) for row in seed]
        campaign.executemany(
            "INSERT INTO seeds VALUES(?,?,?,?,?,?,?,?)",
            [
                (
                    task,
                    dataset,
                    row["slot_group"],
                    row["slot_index"],
                    row["semantic_sha256"],
                    row["canonical_genome"],
                    row["source"],
                    json.dumps(row, sort_keys=True),
                )
                for row in seed
            ],
        )
        fresh_by_dataset[(task, dataset)] = [
            row for row in seed if row["semantic_sha256"] not in known_semantics
        ]
        expansion_parents = [*seed[:32], *seed[32:96], *seed[128:]]
        expansion = offspring_rows(expansion_parents, dataset, 512, offset=10_000 + ordinal * 512)
        unique = set(row["semantic_sha256"] for row in seed)
        expansion_fresh = []
        for row in expansion:
            semantic = sha256_text(row["canonical_genome"])
            if semantic in unique or semantic in known_semantics:
                continue
            unique.add(semantic)
            row["semantic_sha256"] = semantic
            row["source"] = "generation_zero_expansion"
            expansion_fresh.append(provenance(row, hof_hash, frozen_hash))
            if len(expansion_fresh) == 256:
                break
        expansions_by_dataset[(task, dataset)] = expansion_fresh
    campaign.commit()

    selected: dict[tuple[str, str], list[dict]] = defaultdict(list)
    selected_count = 0
    for entry in ordered:
        key = (entry["task"], entry["dataset_hash"])
        for row in fresh_by_dataset[key]:
            if selected_count == target_pairs:
                break
            selected[key].append(row)
            selected_count += 1
        if selected_count == target_pairs:
            break
    base_fresh = sum(len(rows) for rows in fresh_by_dataset.values())
    increment = 64
    expansion_round = 0
    while selected_count < target_pairs and expansion_round < 4:
        start = expansion_round * increment
        for entry in ordered:
            key = (entry["task"], entry["dataset_hash"])
            for row in expansions_by_dataset[key][start : start + increment]:
                if selected_count == target_pairs:
                    break
                selected[key].append(row)
                selected_count += 1
            if selected_count == target_pairs:
                break
        expansion_round += 1
    represented = selected_count
    if represented < target_pairs:
        raise RuntimeError(f"only {represented} fresh pairs available after 512 candidates/dataset")

    manifests_final = ROOT / "manifests" / "pending"
    if manifests_final.exists():
        raise RuntimeError(f"immutable manifest set already exists: {manifests_final}")
    manifests = ROOT / "manifests" / f".pending.tmp-{os.getpid()}"
    manifests.mkdir(parents=True, exist_ok=False)
    manifest_count = 0
    evaluation_ordinal = 0
    for entry in ordered:
        key = (entry["task"], entry["dataset_hash"])
        rows = selected.get(key, [])
        for offset in range(0, len(rows), max_shard):
            candidates = rows[offset : offset + max_shard]
            frozen_hash = candidates[0]["frozen_seed_sha256"]
            manifest = {
                "schema": SHARD_SCHEMA,
                "task": entry["task"],
                "dataset_hash": entry["dataset_hash"],
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task=entry["task"],
                    wave="strict-wave-000000",
                    frozen_seed_sha256=frozen_hash,
                    model_checkpoint_sha256=CHECKPOINT_HASH,
                    hall_of_fame_sha256=hof_hash,
                ),
                "candidates": candidates,
            }
            name = f"{manifest_count:05d}-{entry['task'][0]}-{entry['dataset_hash']}-{offset//max_shard:02d}.json"
            atomic_json(manifests / name, manifest)
            for row in candidates:
                campaign.execute(
                    "INSERT INTO evaluations VALUES(?,?,?,?,?)",
                    (
                        evaluation_ordinal,
                        entry["task"],
                        entry["dataset_hash"],
                        row["semantic_sha256"],
                        name,
                    ),
                )
                evaluation_ordinal += 1
            manifest_count += 1
    campaign.commit()
    campaign.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    campaign.close()
    known.close()
    os.replace(campaign_path, campaign_final)
    os.replace(manifests, manifests_final)
    summary = {
        "schema": "JopeTestOnlyGenerationZero.v1",
        "catalog_sha256": pointer["catalog_sha256"],
        "hall_of_fame_sha256": hof_hash,
        "model_checkpoint_sha256": CHECKPOINT_HASH,
        "datasets": len(ordered),
        "seed_candidates": len(ordered) * 256,
        "base_fresh_pairs": base_fresh,
        "represented_fresh_pairs": represented,
        "expansion_rounds": expansion_round,
        "manifests": manifest_count,
        "max_candidates_per_manifest": max_shard,
        "campaign_db": str(campaign_final),
        "campaign_db_sha256": sha256_file(campaign_final),
    }
    atomic_json(ROOT / "campaign" / "generation-zero.json", summary)
    return summary
