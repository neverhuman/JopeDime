from __future__ import annotations

from test_exact_campaign_config import *

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
