from __future__ import annotations

from test_exact_campaign_config import *

def freeze_hof(output: Path) -> dict:
    """Materialize deterministic best exact observations and ranking summaries."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    source = database(GENEBANK, readonly=True)
    target = database(temporary)
    target.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE known(
          task TEXT NOT NULL,
          dataset_hash TEXT NOT NULL,
          semantic_sha256 TEXT NOT NULL,
          score REAL NOT NULL,
          canonical_genome TEXT NOT NULL,
          receipt_ref TEXT,
          source TEXT,
          observation_id INTEGER NOT NULL,
          PRIMARY KEY(task,dataset_hash,semantic_sha256)
        ) WITHOUT ROWID;
        CREATE INDEX known_dataset_score
          ON known(task,dataset_hash,score DESC,semantic_sha256);
        """
    )
    rows = source.execute(
        """
        WITH ranked AS (
          SELECT o.task,o.dataset_hash,o.semantic_sha256,o.score,
                 g.canonical_genome,o.receipt_ref,o.source,o.id,
                 ROW_NUMBER() OVER (
                   PARTITION BY o.task,o.dataset_hash,o.semantic_sha256
                   ORDER BY o.score DESC,o.created_at_ms DESC,o.id DESC
                 ) AS authority_rank
          FROM observations o JOIN genomes g USING(semantic_sha256)
          WHERE o.valid=1 AND o.score IS NOT NULL AND o.score_tier='test_exact'
            AND ((o.task='regression' AND o.metric='r2')
              OR (o.task='binary' AND o.metric='auc'))
        )
        SELECT task,dataset_hash,semantic_sha256,score,canonical_genome,
               receipt_ref,source,id
        FROM ranked WHERE authority_rank=1
        """
    )
    inserted = 0
    while True:
        batch = rows.fetchmany(20_000)
        if not batch:
            break
        target.executemany(
            "INSERT INTO known VALUES(?,?,?,?,?,?,?,?)",
            [tuple(row) for row in batch],
        )
        inserted += len(batch)
    target.commit()
    source.close()

    local: dict[str, dict[str, list[dict]]] = {
        "regression": defaultdict(list),
        "binary": defaultdict(list),
    }
    common: dict[str, list[dict]] = {}
    genomes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {"regression": {}, "binary": {}}
    for task in EXPECTED_METRIC:
        observations = []
        current_dataset = None
        dataset_count = 0
        for row in target.execute(
            """
            SELECT dataset_hash,semantic_sha256,score,canonical_genome,
                   receipt_ref,source,observation_id
            FROM known WHERE task=?
              AND lower(source) NOT LIKE '%adaptive%'
              AND lower(source) NOT LIKE 'corpus-eval:%'
            ORDER BY dataset_hash,score DESC,semantic_sha256
            """,
            (task,),
        ):
            dataset = row["dataset_hash"]
            if dataset != current_dataset:
                if current_dataset is not None:
                    counts[task][current_dataset] = dataset_count
                current_dataset, dataset_count = dataset, 0
            dataset_count += 1
            semantic = row["semantic_sha256"]
            genome = row["canonical_genome"]
            genomes[semantic] = genome
            if len(local[task][dataset]) < 64:
                local[task][dataset].append(dict(row))
            observations.append((dataset, semantic, float(row["score"])))
        if current_dataset is not None:
            counts[task][current_dataset] = dataset_count
        ranking = rank_common_winners(observations)
        diversity_seen: defaultdict[tuple[str, ...], int] = defaultdict(int)
        diverse = []
        for row in ranking:
            genome = genomes[row["semantic_sha256"]]
            key = structural_key(genome)
            # Permit repeated structures, but stop one layout from monopolizing
            # the common pool before less-common layouts are represented.
            if diversity_seen[key] >= 8 and len(diverse) < 512:
                continue
            diversity_seen[key] += 1
            diverse.append({**row, "canonical_genome": genome, "structural_key": key})
            if len(diverse) == 1024:
                break
        common[task] = diverse
    target.execute("VACUUM")
    target.close()
    os.replace(temporary, output)
    payload = {
        "schema": "JopeTestOnlyHallOfFame.v1",
        "created_at_ms": int(time.time() * 1000),
        "source_genebank_sha256": sha256_file(GENEBANK),
        "known_pairs_sqlite": str(output),
        "known_pairs_sqlite_sha256": sha256_file(output),
        "known_pairs": inserted,
        "local_elites": {
            task: {dataset: values for dataset, values in sorted(by_dataset.items())}
            for task, by_dataset in local.items()
        },
        "common_winners": common,
        "observation_counts": counts,
    }
    hof_path = output.with_suffix(".hof.json")
    atomic_json(hof_path, payload)
    result = {
        "known": str(output),
        "known_sha256": sha256_file(output),
        "hall_of_fame": str(hof_path),
        "hall_of_fame_sha256": sha256_file(hof_path),
        "known_pairs": inserted,
    }
    atomic_json(ROOT / "hof" / "current.json", result)
    return result
