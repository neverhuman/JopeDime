#!/usr/bin/env python3
"""Freeze HOF state and build 500k fresh strict generation-zero evaluations."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from test_exact_common import (
    EXPECTED_METRIC,
    atomic_json,
    compose_seed,
    interleave_tasks,
    rank_common_winners,
    selfplay_authority,
    sha256_file,
    sha256_text,
    structural_key,
)

GENEBANK = Path("/home/ubuntu/jope-flywheel/genebank/genebank.db")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CATALOG_POINTER = ROOT / "catalog" / "current.json"
PROPOSALS = ROOT / "proposals" / "dl-proposals.jsonl"
CHECKPOINT_HASH = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
GOLDEN42 = Path("/home/ubuntu/golden42/golden42_foundry/data/panel.json")
SHARD_SCHEMA = "JopeFlywheelCorpusEvalShard.v1"
HEAD_PREFIXES = {
    "ridge",
    "elasticnet",
    "gpr",
    "histgbm2",
    "ensemble_ridge_gbm",
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "pls",
    "logit_ridge",
}


def database(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


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


def top_level_stages(genome: str) -> list[str]:
    if not genome.startswith("(pipe X ") or not genome.endswith(")"):
        return []
    body = genome[len("(pipe X ") : -1]
    stages, depth, start = [], 0, None
    for index, char in enumerate(body):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                stages.append(body[start : index + 1])
                start = None
    return stages if depth == 0 else []


def rewrite_binary_candidate(row: dict, dataset: str, index: int) -> dict:
    """Retain a checkpoint-proposed body while attaching a binary-valid head."""
    stages = top_level_stages(row["canonical_genome"])
    if not stages:
        raise RuntimeError("binary proposal is not a pipeline")
    source_head = stages[-1][1:].split(None, 1)[0].rstrip(")")
    body = [
        stage
        for stage in stages[:-1]
        if stage[1:].split(None, 1)[0].rstrip(")") != "target"
    ]
    l2 = ("0.005", "0.02", "0.1", "0.5")[
        int(sha256_text(f"{dataset}:{index}:{row['canonical_genome']}")[:8], 16) % 4
    ]
    genome = "(pipe X " + " ".join([*body, f"(logit_ridge l2={l2})"]) + ")"
    return {
        **row,
        "canonical_genome": genome,
        "semantic_sha256": sha256_text(genome),
        "binary_head_rewritten": True,
        "source_head": source_head,
        "binary_head": "logit_ridge",
    }


def offspring_rows(parents: list[dict], dataset: str, count: int, offset: int = 0) -> list[dict]:
    rows = []
    for index in range(offset, offset + count * 3):
        parent = parents[index % len(parents)]
        genome = parent["canonical_genome"]
        stages = top_level_stages(genome)
        if not stages:
            continue
        head_index = len(stages) - 1
        head_op = stages[-1][1:].split(None, 1)[0].rstrip(")")
        if head_op not in HEAD_PREFIXES:
            continue
        selector = int(sha256_text(f"{dataset}:{index}")[:12], 16)
        mutations = (
            f"(rank append={'true' if selector & 1 else 'false'})",
            f"(cheb degree={2 + selector % 7})",
            f"(pair_prod append=true topk={1 + selector % 20})",
            f"(rand_proj append=true k={8 + selector % 57} seed={selector % 10000})",
            f"(winsor_clip q={(1 + selector % 24) / 100:g})",
            f"(sigmoid append={'true' if selector & 2 else 'false'})",
            f"(quantile_bins bins={3 + selector % 17})",
            "(row_stats)",
        )
        first = mutations[selector % len(mutations)]
        insertions = [first]
        if index % 3 == 0:
            insertions.append(mutations[(selector // 11) % len(mutations)])
        child_stages = [*stages[:head_index], *insertions, stages[head_index]]
        child = "(pipe X " + " ".join(child_stages) + ")"
        rows.append(
            {
                "canonical_genome": child,
                "source": "deterministic_mutation_crossover",
                "parent_semantic_sha256": [parent["semantic_sha256"]],
                "parent_source": parent.get("source"),
                "mutation_index": index,
            }
        )
        if len(rows) == count:
            break
    return rows


def load_proposals(path: Path) -> dict[tuple[str, str], list[dict]]:
    values: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("model_checkpoint_sha256") != CHECKPOINT_HASH:
                raise RuntimeError("DL proposal has an unaccepted checkpoint hash")
            values[(row["task"], row["dataset_hash"])].append(row)
    return values


def dataset_order(catalog: list[dict], counts: dict) -> list[dict]:
    by_key = {(row["task"], row["dataset_hash"]): row for row in catalog}
    golden_ids = json.loads(GOLDEN42.read_text())["dataset_ids"]
    golden = [
        by_key[("regression", dataset)]
        for dataset in golden_ids
        if ("regression", dataset) in by_key
    ]
    used = {("regression", row["dataset_hash"]) for row in golden}
    under = sorted(
        [
            row
            for row in catalog
            if (row["task"], row["dataset_hash"]) not in used
            and int(counts[row["task"]].get(row["dataset_hash"], 0)) < 32
        ],
        key=lambda row: (
            int(counts[row["task"]].get(row["dataset_hash"], 0)),
            row["dataset_hash"],
        ),
    )
    used.update((row["task"], row["dataset_hash"]) for row in under)
    regression = sorted(
        (
            row
            for row in catalog
            if row["task"] == "regression"
            and (row["task"], row["dataset_hash"]) not in used
        ),
        key=lambda row: row["dataset_hash"],
    )
    binary = sorted(
        (
            row
            for row in catalog
            if row["task"] == "binary"
            and (row["task"], row["dataset_hash"]) not in used
        ),
        key=lambda row: row["dataset_hash"],
    )
    return [*golden, *under, *interleave_tasks(regression, binary)]


def provenance(row: dict, hof_hash: str, frozen_hash: str) -> dict:
    value = {
        **row,
        "model_checkpoint_sha256": CHECKPOINT_HASH,
        "hall_of_fame_sha256": hof_hash,
        "frozen_seed_sha256": frozen_hash,
    }
    value.setdefault("score_receipt", None)
    value.setdefault(
        "parentage",
        value.get("parent_semantic_sha256") or [],
    )
    return value


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


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-hof")
    freeze.add_argument("--output", type=Path, default=ROOT / "hof" / "known-exact.sqlite")
    make = commands.add_parser("build")
    make.add_argument("--target-pairs", type=int, default=500_000)
    make.add_argument("--max-shard", type=int, default=100)
    extend = commands.add_parser("extend-unselected")
    extend.add_argument("--max-shard", type=int, default=64)
    repair = commands.add_parser("repair-binary")
    repair.add_argument("--max-shard", type=int, default=64)
    args = parser.parse_args()
    if args.command == "freeze-hof":
        print(json.dumps(freeze_hof(args.output), sort_keys=True))
    elif args.command == "build":
        print(json.dumps(build(args.target_pairs, args.max_shard), sort_keys=True))
    elif args.command == "extend-unselected":
        print(json.dumps(extend_unselected(args.max_shard), sort_keys=True))
    else:
        print(json.dumps(build_binary_repair(args.max_shard), sort_keys=True))


if __name__ == "__main__":
    main()
