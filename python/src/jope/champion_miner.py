"""Champion miner: CPU-parallel search for better-than-champion genomes per dataset.

Runs CONCURRENTLY with GPU training (the CPUs are otherwise idle): for each dataset,
take the current champion (best genome found so far by anyone — corpus, GP, policy, or a
former generation) and race a swarm of grammar-aware MUTANTS against it on the cluster:

    param nudge      resample one param to an adjacent/random grammar bin
    head swap        same-engine, task-legal, generate-legal alternative head
    drop transform   remove one stage (tests whether it actually earns its cost)
    swap transform   replace one stage with a random generatable one
    add transform    append one random cheap stage
    cross-engine     re-express: sample a fresh genome from the OTHER engine's grammar

Improvements become the new champions and are appended as training rows with
``provenance="champion_g<g>"`` and rank_pct computed against the dataset's full pool —
the imitation target the user asked for: data encoding → best algorithm KNOWN, where
"known" keeps rising. ~20 single-core evals run per dataset slice in parallel across
both hosts (the bridges are single-core; the host driver packs ~90/host).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .data.active_v3 import robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .grammars import grammar_for
from .grammars.sampler import sample_genomes_from_spec
from .io import read_jsonl
from .pipeline_text import format_pipeline_actions, parse_pipeline_actions
from .propose import embedding_from_vs, target_summary_from_vs
from .schema import TrainingRow


def build_champions(
    art_dir: Path,
    extra_glob: str | Sequence[str] | None = None,
    *,
    log=None,
) -> dict[str, dict]:
    """Per-dataset best row (genome/engine/score) across corpus + all self/champion files."""
    champs: dict[str, dict] = {}
    globs: list[str] = ["self_rows_g*.jsonl", "self_rows_*_g*.jsonl"]
    if extra_glob is not None:
        if isinstance(extra_glob, str):
            globs.append(extra_glob)
        else:
            globs.extend(str(g) for g in extra_glob)
    sources: list[tuple[str, Path]] = [("active_rows", art_dir / "active_rows.jsonl")]
    for glob in globs:
        sources.extend((glob, path) for path in sorted(art_dir.glob(glob)))
    sources.extend(("champion_rows_g*.jsonl", path) for path in sorted(art_dir.glob("champion_rows_g*.jsonl")))
    seen_paths: set[Path] = set()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "rows": 0})
    for source, path in sources:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not path.exists():
            continue
        counts[source]["files"] += 1
        for r in read_jsonl(path):
            counts[source]["rows"] += 1
            md = r.get("metadata") or {}
            s = md.get("score_robust", r.get("score"))
            try:
                s = float(s)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(s):
                continue
            d = str(r.get("dataset_id"))
            g = r.get("canonical_genome") or r.get("genome")
            e = r.get("engine")
            if not g or not e:
                continue
            cur = champs.get(d)
            if cur is None or s > cur["score"]:
                champs[d] = {"genome": g, "engine": e, "score": s}
    if log is not None:
        log(f"build_champions sources: {json.dumps(counts, sort_keys=True)}")
    return champs


def _mutate_actions(actions: dict, spec, rng: random.Random) -> list[dict]:
    """Grammar-aware neighbors of one genome's action dict."""
    out: list[dict] = []
    head_name = actions["head"]["op"]
    transforms = actions.get("transforms") or []

    def clone() -> dict:
        return json.loads(json.dumps(actions))

    # 1) param nudges (head + transforms): move each binned param to a random other bin
    for loc in ["head"] + list(range(len(transforms))):
        op_name = head_name if loc == "head" else transforms[loc]["op"]
        group = spec.heads if loc == "head" else spec.transforms
        op = group.get(op_name)
        if op is None:
            continue
        for p in op.params:
            if not p.bins or p.kind == "expr":
                continue
            choices = [b for b in p.bins]
            if len(choices) < 2:
                continue
            mut = clone()
            tgt = mut["head"] if loc == "head" else mut["transforms"][loc]
            cur = tgt["params"].get(p.name)
            alt = rng.choice([b for b in choices if b != cur] or choices)
            tgt["params"][p.name] = alt
            out.append(mut)

    # 2) head swaps (generate- and task-legal alternatives)
    for name, op in spec.heads.items():
        if name == head_name or not op.generate or "regression" not in op.tasks:
            continue
        mut = clone()
        mut["head"] = {"op": name, "params": {}}
        out.append(mut)

    # 3) drop one transform
    for i in range(len(transforms)):
        mut = clone()
        del mut["transforms"][i]
        mut["transform_count"] = len(mut["transforms"])
        out.append(mut)

    # 4) swap one transform with a random generatable non-expr one
    swappable = [
        n for n, op in spec.transforms.items()
        if op.generate and not op.takes_expr and "regression" in op.tasks
    ]
    for i in range(len(transforms)):
        if not swappable:
            break
        mut = clone()
        mut["transforms"][i] = {"op": rng.choice(swappable), "params": {}}
        out.append(mut)

    # 5) add one random cheap transform (front or back)
    if swappable and len(transforms) < 5:
        for pos in (0, len(transforms)):
            mut = clone()
            mut["transforms"].insert(pos, {"op": rng.choice(swappable), "params": {}})
            mut["transform_count"] = len(mut["transforms"])
            out.append(mut)
    return out


def mine_slice(
    art_dir: str | Path,
    out_path: str | Path,
    *,
    generation: int,
    datasets: dict[str, dict],  # hash -> manifest entry
    champions: dict[str, dict],
    per_dataset: int = 14,
    cross_engine: int = 4,
    folds: int = 3,
    workers: int = 90,
    seed: int = 0,
    ledger: str | Path | None = None,
    progress_path: str | Path | None = None,  # atomic {raced,total,champions} for the TUI
    gene_feed_path: str | Path | None = None,  # shared gene_events.jsonl (append-atomic)
    log=print,
) -> dict[str, Any]:
    """Mutation-race the champions of `datasets`; write improvement rows to out_path."""
    art_dir = Path(art_dir)
    standardization = json.loads((art_dir / "standardization.json").read_text())
    rng = random.Random(seed * 7919 + generation)
    router = EngineRouter()

    eval_inputs: list[dict] = []
    meta: list[dict] = []
    for h, entry in datasets.items():
        champ = champions.get(h)
        if champ is None:
            continue
        spec = grammar_for(champ["engine"])
        try:
            actions = parse_pipeline_actions(champ["genome"], spec)
        except Exception:
            continue
        muts = _mutate_actions(actions, spec, rng)
        rng.shuffle(muts)
        genomes: list[tuple[str, str]] = []
        seen = {champ["genome"]}
        for m in muts:
            try:
                g = format_pipeline_actions(m)
            except Exception:
                continue
            if g not in seen:
                seen.add(g)
                genomes.append((champ["engine"], g))
            if len(genomes) >= per_dataset:
                break
        # cross-engine fresh samples (the other engine may simply be better here)
        other = "newmath" if champ["engine"] == "featmath" else "featmath"
        try:
            fresh = sample_genomes_from_spec(
                grammar_for(other), cross_engine, seed=rng.randrange(1 << 30), max_cost_class="medium"
            )
            genomes.extend((other, g) for g in fresh[:cross_engine])
        except Exception:
            pass
        for eng, g in genomes:
            eval_inputs.append({
                "engine": eng, "dataset_hash": h, "train_csv": entry["train_csv"],
                "test_csv": entry.get("test_csv"), "genome": g, "folds": folds,
            })
            meta.append({"h": h, "engine": eng, "genome": g})

    log(f"[miner g{generation}] racing {len(eval_inputs)} mutants across {len(datasets)} datasets")
    feed = None
    if gene_feed_path is not None:
        from .gene_feed import GeneFeed

        feed = GeneFeed(gene_feed_path, run_id=f"miner_g{generation}")

    def _write_progress(raced: int, champs: int) -> None:
        if progress_path is None:
            return
        p = Path(progress_path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "generation": generation, "raced": raced, "total": len(eval_inputs),
            "champions": champs, "ts": __import__("time").time(),
        }), encoding="utf-8")
        tmp.replace(p)

    # incremental consumption: live progress + scored events while the GPUs train
    by_idx: dict[int, dict] = {}
    raced = 0
    champs_live = 0
    if ledger is None:
        from .prime_config import load_config

        ledger = load_config().runs / "self_play" / "miner_ledger.jsonl"
    for idx, res in iter_eval_regression_with_ledger(
        eval_inputs, ledger=str(ledger), router=router, workers=workers
    ):
        by_idx[idx] = res
        raced += 1
        if feed is not None:
            m0 = meta[idx]
            cv0 = res.get("cv") if isinstance(res.get("cv"), dict) else {}
            try:
                from .gene_feed import gene_id as _gid

                feed.scored(
                    _gid(m0["engine"], m0["h"], m0["genome"]),
                    engine=m0["engine"], dataset_hash=m0["h"], canonical=m0["genome"],
                    oof_r2=cv0.get("oof_r2"), cache_hit=bool(res.get("cache_hit")),
                )
            except Exception:
                pass
        if raced % 25 == 0 or raced == len(eval_inputs):
            _write_progress(raced, champs_live)

    improvements: list[dict] = []
    new_champs = 0
    for idx, m in enumerate(meta):
        res = by_idx.get(idx) or {}
        cv = res.get("cv") if isinstance(res.get("cv"), dict) else {}
        oof = cv.get("oof_r2")
        fold_r2 = cv.get("fold_r2") if isinstance(cv.get("fold_r2"), list) else None
        if not (isinstance(oof, (int, float)) and math.isfinite(float(oof))):
            continue
        score = robust_score(float(oof), fold_r2)
        if score is None:
            continue
        champ = champions[m["h"]]
        if score <= champ["score"] + 1e-9:
            continue
        champions[m["h"]] = {"genome": m["genome"], "engine": m["engine"], "score": score}
        new_champs += 1
        if feed is not None:
            try:
                from .gene_feed import gene_id as _gid

                feed.selected(
                    _gid(m["engine"], m["h"], m["genome"]),
                    engine=m["engine"], dataset_hash=m["h"], canonical=m["genome"],
                    selection_slot=f"champion_g{generation}",
                )
            except Exception:
                pass
        entry = datasets[m["h"]]
        row = TrainingRow(
            dataset_id=m["h"],
            source="champion_miner",
            embedding=embedding_from_vs(entry["vs_path"], standardization),
            target_embedding=target_summary_from_vs(entry.get("train_target_vs")),
            genome=m["genome"],
            canonical_genome=m["genome"],
            engine=m["engine"],
            score=max(float(oof), -1.0),
            cost_ms=(res.get("timings") or {}).get("total_ms"),
            metadata={
                "provenance": f"champion_g{generation}",
                "generation": generation,
                "task_type": "regression",
                "split": "train",
                "oof_r2_raw": float(oof),
                "score_robust": score,
                "rank_pct": 1.0,  # champion by construction
                "beat_champion_by": round(score - champ["score"], 6) if champ else None,
            },
        )
        improvements.append(row.to_json())

    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in improvements:
            fh.write(json.dumps(r) + "\n")
    _write_progress(len(eval_inputs), new_champs)
    report = {
        "generation": generation,
        "datasets_mined": len(datasets),
        "mutants_raced": len(eval_inputs),
        "new_champions": new_champs,
        "rows_written": len(improvements),
        "out": str(out_path),
    }
    log(f"[miner g{generation}] {json.dumps(report)}")
    return report
