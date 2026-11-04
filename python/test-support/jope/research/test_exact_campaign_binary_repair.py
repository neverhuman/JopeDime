from __future__ import annotations

from test_exact_campaign_config import *

from test_exact_campaign_population import rewrite_binary_candidate

def build_binary_repair(max_shard: int) -> dict:
    """Rebuild every frozen binary seed with an admitted probability head."""
    if not 1 <= max_shard <= 128:
        raise RuntimeError("binary repair shard size must be in 1..128")
    source_path = ROOT / "campaign" / "generation-zero.sqlite"
    output_final = ROOT / "campaign" / "generation-zero-binary-repair.sqlite"
    manifests_final = ROOT / "manifests" / "binary-repair"
    if not source_path.is_file():
        raise RuntimeError("generation-zero campaign must exist before binary repair")
    if output_final.exists() or manifests_final.exists():
        raise RuntimeError("immutable binary repair campaign already exists")

    pointer = json.loads(CATALOG_POINTER.read_text())
    catalog_meta = json.loads(Path(pointer["catalog"]).read_text())
    binary_entries = sorted(
        (entry for entry in catalog_meta["entries"] if entry["task"] == "binary"),
        key=lambda entry: entry["dataset_hash"],
    )
    hof_pointer = json.loads((ROOT / "hof" / "current.json").read_text())
    hof_hash = hof_pointer["hall_of_fame_sha256"]
    source = database(source_path, readonly=True)
    known = database(Path(hof_pointer["known"]), readonly=True)
    output_temporary = output_final.with_name(f".{output_final.name}.tmp-{os.getpid()}")
    manifests_temporary = ROOT / "manifests" / f".binary-repair.tmp-{os.getpid()}"
    manifests_temporary.mkdir(parents=True, exist_ok=False)
    output = database(output_temporary)
    output.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE seeds(
          dataset_hash TEXT,slot_group TEXT,slot_index INTEGER,
          semantic_sha256 TEXT,canonical_genome TEXT,provenance_json TEXT,
          PRIMARY KEY(dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE evaluations(
          ordinal INTEGER PRIMARY KEY,dataset_hash TEXT,
          semantic_sha256 TEXT,manifest TEXT
        );
        """
    )
    manifest_count = 0
    evaluation_ordinal = 0
    known_reused = 0
    for entry in binary_entries:
        dataset = entry["dataset_hash"]
        original = [
            json.loads(row["provenance_json"])
            for row in source.execute(
                """
                SELECT provenance_json FROM seeds
                WHERE task='binary' AND dataset_hash=?
                ORDER BY CASE slot_group
                  WHEN 'local_elite' THEN 0 WHEN 'dl_proposal' THEN 1
                  WHEN 'common_winner' THEN 2 ELSE 3 END,slot_index
                """,
                (dataset,),
            )
        ]
        repaired = []
        seen = set()
        for index, row in enumerate(original):
            candidate = rewrite_binary_candidate(row, dataset, index)
            if candidate["semantic_sha256"] in seen:
                continue
            seen.add(candidate["semantic_sha256"])
            repaired.append(candidate)
        if len(repaired) < 256:
            extra = offspring_rows(repaired, dataset, 2048, offset=2_000_000)
            for row in extra:
                semantic = sha256_text(row["canonical_genome"])
                if semantic in seen:
                    continue
                seen.add(semantic)
                repaired.append(
                    {
                        **row,
                        "semantic_sha256": semantic,
                        "binary_head_rewritten": True,
                        "binary_head": "logit_ridge",
                    }
                )
                if len(repaired) == 256:
                    break
        if len(repaired) < 256:
            raise RuntimeError(f"binary:{dataset} produced only {len(repaired)} repaired seeds")
        repaired = repaired[:256]
        frozen_hash = sha256_text(
            json.dumps(
                [row["semantic_sha256"] for row in repaired],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for index, row in enumerate(repaired):
            row["frozen_seed_sha256"] = frozen_hash
            row["model_checkpoint_sha256"] = CHECKPOINT_HASH
            row["hall_of_fame_sha256"] = hof_hash
            row.setdefault("score_receipt", None)
            row.setdefault("parentage", row.get("parent_semantic_sha256") or [])
            row.setdefault("slot_group", "offspring")
            row.setdefault("slot_index", index)
        output.executemany(
            "INSERT INTO seeds VALUES(?,?,?,?,?,?)",
            [
                (
                    dataset,
                    row["slot_group"],
                    int(row["slot_index"]),
                    row["semantic_sha256"],
                    row["canonical_genome"],
                    json.dumps(row, sort_keys=True),
                )
                for row in repaired
            ],
        )
        known_semantics = {
            row[0]
            for row in known.execute(
                """
                SELECT semantic_sha256 FROM known
                WHERE task='binary' AND dataset_hash=?
                """,
                (dataset,),
            )
        }
        fresh = [
            row for row in repaired if row["semantic_sha256"] not in known_semantics
        ]
        known_reused += len(repaired) - len(fresh)
        for offset in range(0, len(fresh), max_shard):
            shard = fresh[offset : offset + max_shard]
            manifest = {
                "schema": SHARD_SCHEMA,
                "task": "binary",
                "dataset_hash": dataset,
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task="binary",
                    wave="strict-wave-000000",
                    frozen_seed_sha256=frozen_hash,
                    model_checkpoint_sha256=CHECKPOINT_HASH,
                    hall_of_fame_sha256=hof_hash,
                ),
                "candidates": shard,
            }
            name = f"b-{manifest_count:05d}-{dataset}-{offset // max_shard:02d}.json"
            atomic_json(manifests_temporary / name, manifest)
            output.executemany(
                "INSERT INTO evaluations VALUES(?,?,?,?)",
                [
                    (
                        evaluation_ordinal + index,
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
    source.close()
    known.close()
    os.replace(output_temporary, output_final)
    os.replace(manifests_temporary, manifests_final)
    summary = {
        "schema": "JopeTestOnlyGenerationZeroBinaryRepair.v1",
        "catalog_sha256": pointer["catalog_sha256"],
        "hall_of_fame_sha256": hof_hash,
        "model_checkpoint_sha256": CHECKPOINT_HASH,
        "datasets": len(binary_entries),
        "seed_candidates": len(binary_entries) * 256,
        "fresh_pairs": evaluation_ordinal,
        "known_pairs_reused": known_reused,
        "manifests": manifest_count,
        "max_candidates_per_manifest": max_shard,
        "campaign_db": str(output_final),
        "campaign_db_sha256": sha256_file(output_final),
    }
    atomic_json(ROOT / "campaign" / "generation-zero-binary-repair.json", summary)
    return summary
