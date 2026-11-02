"""Closed-loop runner: sample -> surrogate-rank -> budgeted real CV -> learn.

Per iteration, for a rolling window of TRAIN datasets: the policy samples K genomes per
(dataset, engine) (engine token forced both ways so both engines stay represented), the
reward ensemble scores them, the top-mu (with engine/head diversity and a >=1-per-engine
floor) plus most-uncertain candidates are CV-scored through the engine-aware ledger
(train.csv ONLY — test.csv stays sealed for evaluate-active), results append to
loop_rows.jsonl as engine-tagged TrainingRows, and the reward branch (optionally the
whole policy via RWR) is refit. Everything is resumable: the ledger dedupes evals, the
state file is written atomically, and corpus appends are idempotent by (dataset, engine,
canonical).

Stop conditions: max_iterations, max_hours, a STOP file in run_dir, patience on the
rolling best robust score, and a surrogate-collapse guard (spearman(surrogate, CV) < 0
twice in a row halts POLICY refits; reward refits continue).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gene_feed import GeneFeed

import torch

from .data.active_v3 import robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .io import read_jsonl
from .loop_selection import _model_specs as _model_specs, _select_candidates as _select_candidates
from .loop_state import (
    LoopConfig as LoopConfig,
    _atomic_write as _atomic_write,
    _load_state as _load_state,
    _patience_exhausted as _patience_exhausted,
)
from .model import _pearson
from .pipeline_text import format_pipeline_actions
from .policy.decode import candidate_risk_reason_v3, sample_sequences
from .policy.losses import PHASE_B_PROFILE, clamp_log_vars, kg_combine
from .policy.model_v3 import JopePolicyModelV3, rows_to_policy_v3_batch
from .policy.tokens import decode_tokens
from .propose import embedding_from_vs, load_any_model, load_standardization, target_summary_from_vs
from .schema import TrainingRow


def run_loop(cfg: LoopConfig, *, router: EngineRouter | None = None) -> dict[str, Any]:
    router = router or EngineRouter()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.run_dir / "loop_state.json"
    metrics_path = cfg.run_dir / "loop_metrics.jsonl"
    loop_rows_path = Path(cfg.art_dir) / "loop_rows.jsonl"
    state = _load_state(state_path)

    model = load_any_model(cfg.checkpoint)
    if not isinstance(model, JopePolicyModelV3):
        raise ValueError("jope loop requires a policy_v3 checkpoint")
    model = model.to(cfg.device)

    split_manifest = json.loads((Path(cfg.art_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    standardization = load_standardization(Path(cfg.art_dir) / "standardization.json")
    cfg_standardization = standardization
    datasets = {
        h: e
        for h, e in sorted((split_manifest.get("datasets") or {}).items())
        if e.get("split", "train") == "train" and e.get("vs_path") and e.get("train_csv")
    }
    roster_keys = list(datasets)
    if not roster_keys:
        raise ValueError("no train datasets in split_manifest")

    corpus_rows = [TrainingRow.from_json(r) for r in read_jsonl(Path(cfg.art_dir) / "active_rows.jsonl")]
    loop_rows: list[TrainingRow] = (
        [TrainingRow.from_json(r) for r in read_jsonl(loop_rows_path)] if loop_rows_path.exists() else []
    )
    known: set[tuple[str, str, str]] = {
        (r.dataset_id, r.engine, r.canonical_genome or r.genome) for r in corpus_rows + loop_rows
    }

    gene_feed = None
    if cfg.gene_feed_path is not None:
        from .gene_feed import GeneFeed

        gene_feed = GeneFeed(cfg.gene_feed_path, run_id=f"loop-{cfg.run_dir.name}")

    started = time.time()
    summary: dict[str, Any] = {}
    try:
        while True:
            iteration = state["iteration"] + 1
            if cfg.max_iterations is not None and iteration > cfg.max_iterations:
                summary["stop_reason"] = "max_iterations"
                break
            if cfg.max_hours is not None and (time.time() - started) / 3600.0 > cfg.max_hours:
                summary["stop_reason"] = "max_hours"
                break
            if (cfg.run_dir / "STOP").exists():
                summary["stop_reason"] = "stop_file"
                break
            if _patience_exhausted(state, cfg):
                summary["stop_reason"] = "patience"
                break

            iter_row = _loop_iteration(
                iteration,
                state,
                cfg,
                model,
                router,
                datasets,
                roster_keys,
                known,
                loop_rows_path,
                loop_rows,
                corpus_rows,
                cfg_standardization,
                gene_feed,
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(iter_row, sort_keys=True) + "\n")
            state["iteration"] = iteration
            _atomic_write(state_path, state)
    finally:
        if gene_feed is not None:
            gene_feed.close()

    summary.update({"iterations": state["iteration"], "loop_rows": len(loop_rows)})
    return summary


def _loop_iteration(
    iteration: int,
    state: dict[str, Any],
    cfg: LoopConfig,
    model: JopePolicyModelV3,
    router: EngineRouter,
    datasets: dict[str, dict[str, Any]],
    roster_keys: list[str],
    known: set[tuple[str, str, str]],
    loop_rows_path: Path,
    loop_rows: list[TrainingRow],
    corpus_rows: list[TrainingRow],
    standardization: dict[str, Any] | None,
    gene_feed: GeneFeed | None = None,
) -> dict[str, Any]:
    iter_started = time.time()
    rng = random.Random(cfg.seed * 100003 + iteration)
    cursor = state.get("roster_cursor", 0)
    roster = [
        roster_keys[(cursor + i) % len(roster_keys)]
        for i in range(min(cfg.datasets_per_iter, len(roster_keys)))
    ]
    state["roster_cursor"] = (cursor + len(roster)) % len(roster_keys)

    per_engine: dict[str, dict[str, float]] = {
        e: {
            "sampled": 0,
            "unique": 0,
            "cv_requested": 0,
            "cv_completed": 0,
            "timeouts": 0,
            "blowups": 0,
            "cache_hits": 0,
        }
        for e in cfg.engines
    }
    eval_inputs: list[dict[str, Any]] = []
    eval_meta: list[dict[str, Any]] = []
    engine_budget = {e: cfg.max_cv_per_engine for e in cfg.engines}

    for dataset_hash in roster:
        entry = datasets[dataset_hash]
        embedding = torch.tensor(embedding_from_vs(entry["vs_path"], standardization), dtype=torch.float32)
        target = torch.tensor(target_summary_from_vs(entry.get("train_target_vs")), dtype=torch.float32)
        candidates: list[dict[str, Any]] = []
        for engine in cfg.engines:
            temperature = cfg.temperature_schedule[iteration % len(cfg.temperature_schedule)]
            seqs = sample_sequences(
                model,
                embedding,
                target,
                n=cfg.samples_per_dataset_engine,
                engine=engine,
                temperature=temperature,
                seed=rng.randrange(1 << 30),
                device=cfg.device,
            )
            per_engine[engine]["sampled"] += len(seqs)
            for seq in seqs:
                decoded_engine, actions = decode_tokens(seq, model.vocab)
                genome = format_pipeline_actions(actions)
                candidates.append(
                    {"engine": decoded_engine, "genome": genome, "actions": actions, "tokens": seq}
                )
        validations = router.validate_many([(c["engine"], c["genome"]) for c in candidates])
        pool: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for cand, meta in zip(candidates, validations):
            if not meta.get("valid"):
                continue
            canonical = str(meta.get("canonical_genome"))
            key = (cand["engine"], canonical)
            if key in seen or (dataset_hash, cand["engine"], canonical) in known:
                continue
            if candidate_risk_reason_v3(cand["engine"], cand["actions"]):
                continue
            seen.add(key)
            cand["canonical"] = canonical
            cand["head_type"] = meta.get("head_type")
            cand["family_bitmap"] = dict(meta.get("family_bitmap") or {})
            pool.append(cand)
            per_engine[cand["engine"]]["unique"] += 1
        if not pool:
            continue
        with torch.no_grad():
            mean, std = model.score_token_sequences(
                embedding, target, [c["tokens"] for c in pool], device=cfg.device
            )
        for i, cand in enumerate(pool):
            cand["mu"] = float(mean[i])
            cand["sigma"] = float(std[i])
            if gene_feed is not None:
                cand["gene_id"] = gene_feed.proposed(
                    engine=cand["engine"],
                    canonical=cand["canonical"],
                    dataset_hash=dataset_hash,
                    head=cand.get("head_type"),
                    families=[k for k, v in (cand.get("family_bitmap") or {}).items() if v],
                    decoder="sample",
                    mu=cand["mu"],
                    sigma=cand["sigma"],
                    iteration=iteration,
                )
        selected = _select_candidates(pool, cfg)
        for cand in selected:
            engine = cand["engine"]
            if engine_budget[engine] <= 0:
                continue
            engine_budget[engine] -= 1
            per_engine[engine]["cv_requested"] += 1
            if gene_feed is not None and cand.get("gene_id"):
                gene_feed.selected(
                    cand["gene_id"],
                    engine=engine,
                    dataset_hash=dataset_hash,
                    canonical=cand["canonical"],
                    selection_slot="loop_cv",
                )
            eval_inputs.append(
                {
                    "engine": engine,
                    "dataset_hash": dataset_hash,
                    "train_csv": entry["train_csv"],
                    "genome": cand["canonical"],
                    "folds": cfg.folds,
                }
            )
            eval_meta.append({**cand, "dataset_hash": dataset_hash})

    fresh_mu: list[float] = []
    fresh_cv: list[float] = []
    new_rows: list[TrainingRow] = []
    for idx, result in iter_eval_regression_with_ledger(
        eval_inputs, ledger=cfg.ledger, router=router, workers=cfg.eval_workers
    ):
        meta = eval_meta[idx]
        engine = meta["engine"]
        stats = per_engine[engine]
        cv = result.get("cv") if isinstance(result.get("cv"), dict) else {}
        oof = cv.get("oof_r2")
        fold_r2 = cv.get("fold_r2") if isinstance(cv.get("fold_r2"), list) else None
        if result.get("cache_hit"):
            stats["cache_hits"] += 1
        row = TrainingRow(
            dataset_id=meta["dataset_hash"],
            source="active_loop",
            embedding=[0.0],
            genome=meta["canonical"],
            canonical_genome=meta["canonical"],
            engine=engine,
            family_bitmap=meta.get("family_bitmap") or {},
            metadata={
                "provenance": "loop",
                "iteration": iteration,
                "surrogate_mu": meta["mu"],
                "surrogate_sigma": meta["sigma"],
                "actions": meta["actions"],
                "split": "train",
            },
        )
        entry = datasets[meta["dataset_hash"]]
        row.embedding = embedding_from_vs(entry["vs_path"], standardization)
        row.target_embedding = target_summary_from_vs(entry.get("train_target_vs"))
        if isinstance(oof, (int, float)) and math.isfinite(float(oof)):
            stats["cv_completed"] += 1
            row.score = max(float(oof), -1.0)
            row.metadata["oof_r2_raw"] = float(oof)
            row.metadata["score_robust"] = robust_score(float(oof), fold_r2)
            row.cost_ms = (result.get("timings") or {}).get("total_ms")
            if float(oof) < -1.0:
                stats["blowups"] += 1
            if not result.get("cache_hit"):
                fresh_mu.append(meta["mu"])
                fresh_cv.append(float(row.metadata["score_robust"]))
            if gene_feed is not None and meta.get("gene_id"):
                gene_feed.scored(
                    meta["gene_id"],
                    engine=engine,
                    dataset_hash=meta["dataset_hash"],
                    canonical=meta["canonical"],
                    head=meta.get("head_type"),
                    oof_r2=float(oof),
                    fold_scores=fold_r2,
                    score_robust=float(row.metadata["score_robust"]),
                    total_ms=row.cost_ms,
                    cache_hit=bool(result.get("cache_hit")),
                    blowup=float(oof) < -1.0,
                )
        else:
            row.score = None
            row.metadata["eval_error"] = (result.get("errors") or ["unknown"])[0]
            row.metadata["score_robust"] = -1.0
            if result.get("timeout"):
                stats["timeouts"] += 1
            if gene_feed is not None and meta.get("gene_id"):
                gene_feed.scored(
                    meta["gene_id"],
                    engine=engine,
                    dataset_hash=meta["dataset_hash"],
                    canonical=meta["canonical"],
                    head=meta.get("head_type"),
                    total_ms=(result.get("timings") or {}).get("total_ms"),
                    timeout=bool(result.get("timeout")),
                    cache_hit=bool(result.get("cache_hit")),
                )
        known.add((row.dataset_id, row.engine, row.canonical_genome or row.genome))
        new_rows.append(row)

    if new_rows:
        with loop_rows_path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row.to_json(), sort_keys=True) + "\n")
        loop_rows.extend(new_rows)

    surrogate_spearman = _spearman(fresh_mu, fresh_cv)
    if surrogate_spearman is not None and surrogate_spearman < 0:
        state["spearman_bad_streak"] = state.get("spearman_bad_streak", 0) + 1
    else:
        state["spearman_bad_streak"] = 0
    if state["spearman_bad_streak"] >= 2:
        state["policy_refits_halted"] = True

    refit_metrics: dict[str, Any] = {}
    if cfg.refit_reward_every and iteration % cfg.refit_reward_every == 0 and new_rows:
        refit_policy = (
            cfg.refit_policy_every > 0
            and iteration % cfg.refit_policy_every == 0
            and not state.get("policy_refits_halted")
        )
        refit_metrics = _refit(model, corpus_rows + loop_rows, cfg, policy=refit_policy)
        payload = model.checkpoint_payload(
            grammar_specs=[spec.to_json() for spec in _model_specs(model)],
            loop_iteration=iteration,
        )
        torch.save(payload, cfg.run_dir / f"model_iter{iteration:04d}.pt")
        torch.save(payload, cfg.checkpoint)

    scored = [r.metadata.get("score_robust") for r in new_rows if r.metadata.get("score_robust") is not None]
    best = max((float(s) for s in scored), default=None)
    window = state.setdefault("best_window", [])
    window.append(best if best is not None else (window[-1] if window else -1.0))
    del window[:-64]

    return {
        "iteration": iteration,
        "roster": len(roster),
        "per_engine": per_engine,
        "eval_requested": len(eval_inputs),
        "new_rows": len(new_rows),
        "corpus_loop_rows": len(loop_rows),
        "surrogate_vs_cv_spearman": surrogate_spearman,
        "policy_refits_halted": bool(state.get("policy_refits_halted")),
        "best_robust_score": best,
        "refit": refit_metrics,
        "iter_sec": time.time() - iter_started,
    }


def _refit(
    model: JopePolicyModelV3,
    rows: list[TrainingRow],
    cfg: LoopConfig,
    *,
    policy: bool,
) -> dict[str, Any]:
    """Short phase-B continuation: reward branch always; full policy (RWR) if requested."""
    from .train import _dataset_grouped_batches

    usable = [r for r in rows if (r.canonical_genome or r.genome)]
    if not usable:
        return {}
    model.train()
    model.set_phase("B")
    params = (
        model.parameters()
        if policy
        else [
            p
            for m in (model.reward_token_emb, model.reward_norm, model.reward_heads, model.stability_head)
            for p in m.parameters()
        ]
    )
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=cfg.refit_lr, weight_decay=0.01)
    rng = random.Random(cfg.seed)
    total = 0.0
    steps = 0
    for _ in range(max(1, cfg.refit_epochs)):
        shuffled = usable[:]
        rng.shuffle(shuffled)
        for batch_rows in _dataset_grouped_batches(shuffled, cfg.refit_batch_size, rng):
            try:
                batch = rows_to_policy_v3_batch(batch_rows, model.vocab, device=cfg.device)
            except Exception:
                continue
            components = model.loss_components(batch, phase="B")
            if not policy:
                components.pop("seq_nll", None)
                components.pop("runtime_huber", None)
            if not components:
                continue
            loss = kg_combine(components, model.loss_log_vars, PHASE_B_PROFILE)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 5.0)
            opt.step()
            clamp_log_vars(model.loss_log_vars)
            total += float(loss.detach().cpu())
            steps += 1
    model.eval()
    return {"refit_loss": total / max(1, steps), "refit_steps": steps, "refit_policy": policy}


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    return _pearson(ranks(a), ranks(b))
