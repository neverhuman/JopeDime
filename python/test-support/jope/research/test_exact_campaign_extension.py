from __future__ import annotations

from test_exact_campaign_config import *

from test_exact_campaign_population import dataset_order, provenance

def extend_unselected(max_shard: int) -> dict:
    """Materialize the still-fresh portion of the frozen 256-seed ledger."""
    if not 1 <= max_shard <= 128:
        raise RuntimeError("supplemental shard size must be in 1..128")
    campaign_path = ROOT / "campaign" / "generation-zero.sqlite"
    if not campaign_path.is_file():
        raise RuntimeError("generation-zero campaign must exist before extension")
    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog_meta = json.loads(Path(pointer["catalog"]).read_text())
    hof_pointer = json.loads((ROOT / "hof" / "current.json").read_text())
    hof = json.loads(Path(hof_pointer["hall_of_fame"]).read_text())
    hof_hash = hof_pointer["hall_of_fame_sha256"]
    ordered = dataset_order(catalog_meta["entries"], hof["observation_counts"])

    output_final = ROOT / "campaign" / "generation-zero-supplemental.sqlite"
    manifests_final = ROOT / "manifests" / "supplemental"
    if output_final.exists() or manifests_final.exists():
        raise RuntimeError("immutable supplemental campaign already exists")
    output_temporary = output_final.with_name(f".{output_final.name}.tmp-{os.getpid()}")
    manifests_temporary = ROOT / "manifests" / f".supplemental.tmp-{os.getpid()}"
    manifests_temporary.mkdir(parents=True, exist_ok=False)

    campaign = database(campaign_path, readonly=True)
    known = database(Path(hof_pointer["known"]), readonly=True)
    output = database(output_temporary)
    output.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE evaluations(
          ordinal INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,
          semantic_sha256 TEXT,manifest TEXT
        );
        """
    )
    selected_pairs = {
        (row["task"], row["dataset_hash"], row["semantic_sha256"])
        for row in campaign.execute(
            "SELECT task,dataset_hash,semantic_sha256 FROM evaluations"
        )
    }
    manifest_count = 0
    evaluation_ordinal = 0
    for entry in ordered:
        task, dataset = entry["task"], entry["dataset_hash"]
        known_semantics = {
            row[0]
            for row in known.execute(
                """
                SELECT semantic_sha256 FROM known
                WHERE task=? AND dataset_hash=?
                """,
                (task, dataset),
            )
        }
        candidates = []
        for row in campaign.execute(
            """
            SELECT semantic_sha256,provenance_json FROM seeds
            WHERE task=? AND dataset_hash=?
            ORDER BY CASE slot_group
              WHEN 'local_elite' THEN 0 WHEN 'dl_proposal' THEN 1
              WHEN 'common_winner' THEN 2 ELSE 3 END,slot_index
            """,
            (task, dataset),
        ):
            if (
                (task, dataset, row["semantic_sha256"]) in selected_pairs
                or row["semantic_sha256"] in known_semantics
            ):
                continue
            candidates.append(json.loads(row["provenance_json"]))
        for offset in range(0, len(candidates), max_shard):
            shard = candidates[offset : offset + max_shard]
            frozen_hash = shard[0]["frozen_seed_sha256"]
            manifest = {
                "schema": SHARD_SCHEMA,
                "task": task,
                "dataset_hash": dataset,
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task=task,
                    wave="strict-wave-000000",
                    frozen_seed_sha256=frozen_hash,
                    model_checkpoint_sha256=CHECKPOINT_HASH,
                    hall_of_fame_sha256=hof_hash,
                ),
                "candidates": shard,
            }
            name = (
                f"s-{manifest_count:05d}-{task[0]}-{dataset}-"
                f"{offset // max_shard:02d}.json"
            )
            atomic_json(manifests_temporary / name, manifest)
            output.executemany(
                "INSERT INTO evaluations VALUES(?,?,?,?,?)",
                [
                    (
                        evaluation_ordinal + index,
                        task,
                        dataset,
                        row["semantic_sha256"],
                        name,
                    )
                    for index, row in enumerate(shard)
                ],
            )
            evaluation_ordinal += len(shard)
            manifest_count += 1
    output.commit()
    output.close()
    campaign.close()
    known.close()
    os.replace(output_temporary, output_final)
    os.replace(manifests_temporary, manifests_final)
    summary = {
        "schema": "JopeTestOnlyGenerationZeroSupplemental.v1",
        "catalog_sha256": pointer["catalog_sha256"],
        "hall_of_fame_sha256": hof_hash,
        "model_checkpoint_sha256": CHECKPOINT_HASH,
        "fresh_pairs": evaluation_ordinal,
        "manifests": manifest_count,
        "max_candidates_per_manifest": max_shard,
        "campaign_db": str(output_final),
        "campaign_db_sha256": sha256_file(output_final),
    }
    atomic_json(ROOT / "campaign" / "generation-zero-supplemental.json", summary)
    return summary
