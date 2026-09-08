"""`jope stream` — continuous gene generation with a live event feed ("genes scream").

A thin driver over the loop primitives (`sample_sequences`, `_select_candidates`, the
engine-ledger evaluator, `_refit`). Each cycle:

  1. sample a large batch per (dataset, engine), decode, bridge-validate, dedupe;
  2. surrogate-score the whole pool in one batched forward pass and emit a ``proposed``
     event for EVERY validated candidate — this is the flood the TUI's left pane shows;
  3. select a small set and CV-score it across the cluster, emitting ``scored`` as each
     result lands (the right pane / score column);
  4. append scored rows to ``loop_rows.jsonl`` and, every ``refit_interval_s`` or
     ``refit_min_new_rows``, run a short reward-only refit so scores feed back into the
     surrogate.

Throughput: the proposed stream is GPU-bound and runs at thousands/s; CV is the
bottleneck (~cluster-parallel). Genes proposed but not selected for CV are "dropped" by
design (tracked as ``surrogate_only``) — that IS the backpressure. Stops on a ``STOP``
file, ``max_genes``, or ``max_seconds``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .engines import ENGINES, EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .gene_feed import GeneFeed
from .loop import LoopConfig, _refit, _select_candidates
from .data.active_v3 import robust_score
from .pipeline_text import format_pipeline_actions
from .policy.decode import candidate_risk_reason_v3
from .policy.model_v3 import JopePolicyModelV3
from .policy.tokens import decode_tokens
from .propose import embedding_from_vs, load_any_model, load_standardization, target_summary_from_vs
from .schema import TrainingRow


@dataclass
class StreamConfig:
    art_dir: Path
    checkpoint: Path
    run_dir: Path
    ledger: Path
    engines: tuple[str, ...] = ENGINES
    device: str = "cpu"
    sample_batch: int = 48  # sampled sequences per (dataset, engine) per cycle
    datasets_per_cycle: int = 4
    cv_top_m: int = 3  # exploit selections per cycle
    cv_uncertain_m: int = 1  # explore (most-uncertain) selections per cycle
    folds: int = 3
    temperature: float = 0.9
    eval_workers: int | None = None
    refit_interval_s: float = 30.0
    refit_min_new_rows: int = 64
    refit_lr: float = 1e-4
    refit_epochs: int = 1
    refit_batch_size: int = 128
    max_genes: int | None = None  # stop after N proposed (None = run until STOP)
    max_seconds: float | None = None
    max_cycles: int | None = None  # test hook
    seed: int = 0

    def _loop_cfg(self) -> LoopConfig:
        """Adapter so we can reuse loop._select_candidates / _refit unchanged."""
        return LoopConfig(
            art_dir=self.art_dir,
            checkpoint=self.checkpoint,
            run_dir=self.run_dir,
            ledger=self.ledger,
            engines=self.engines,
            cv_top_m=self.cv_top_m,
            cv_uncertain_m=self.cv_uncertain_m,
            folds=self.folds,
            refit_lr=self.refit_lr,
            refit_epochs=self.refit_epochs,
            refit_batch_size=self.refit_batch_size,
            eval_workers=self.eval_workers,
            seed=self.seed,
            device=self.device,
        )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def run_stream(cfg: StreamConfig, *, router: EngineRouter | None = None) -> dict[str, Any]:
    from .policy.decode import sample_sequences

    router = router or EngineRouter()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    feed_path = cfg.run_dir / "gene_events.jsonl"
    meta_path = cfg.run_dir / "stream_meta.json"
    loop_rows_path = Path(cfg.art_dir) / "loop_rows.jsonl"

    model = load_any_model(cfg.checkpoint)
    if not isinstance(model, JopePolicyModelV3):
        raise ValueError("jope stream requires a policy_v3 checkpoint")
    model = model.to(cfg.device)

    split_manifest = json.loads((Path(cfg.art_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    standardization = load_standardization(Path(cfg.art_dir) / "standardization.json")
    datasets = {
        h: e
        for h, e in sorted((split_manifest.get("datasets") or {}).items())
        if e.get("split", "train") == "train" and e.get("vs_path") and e.get("train_csv")
    }
    roster = list(datasets)
    if not roster:
        raise ValueError("no train datasets in split_manifest")

    known: set[tuple[str, str, str]] = set()
    loop_cfg = cfg._loop_cfg()
    counters = {"proposed": 0, "surrogate_only": 0, "cv_scored": 0, "cycles": 0}
    run_id = f"stream-{cfg.run_dir.name}"

    started = time.time()
    last_refit = started
    refit_count = 0
    rows_since_refit: list[TrainingRow] = []
    cursor = 0
    rng = random.Random(cfg.seed)

    def write_meta(status: str) -> None:
        _atomic_write(
            meta_path,
            {
                "mode": "stream",
                "run_id": run_id,
                "feed": feed_path.name,
                "engines": list(cfg.engines),
                "hosts": list(router.hosts),
                "checkpoint": str(cfg.checkpoint),
                "counters": dict(counters),
                "elapsed_s": round(time.time() - started, 1),
                "status": status,
                "updated_at": time.time(),
            },
        )

    feed = GeneFeed(feed_path, run_id=run_id)
    write_meta("running")
    try:
        while True:
            if (cfg.run_dir / "STOP").exists():
                break
            if cfg.max_cycles is not None and counters["cycles"] >= cfg.max_cycles:
                break
            if cfg.max_genes is not None and counters["proposed"] >= cfg.max_genes:
                break
            if cfg.max_seconds is not None and (time.time() - started) >= cfg.max_seconds:
                break

            # ---- 1) sample + validate + dedupe + surrogate-score one cycle ----
            cycle_ds = [roster[(cursor + i) % len(roster)] for i in range(min(cfg.datasets_per_cycle, len(roster)))]
            cursor = (cursor + len(cycle_ds)) % len(roster)
            eval_inputs: list[dict[str, Any]] = []
            eval_meta: list[dict[str, Any]] = []
            for dataset_hash in cycle_ds:
                entry = datasets[dataset_hash]
                embedding = torch.tensor(embedding_from_vs(entry["vs_path"], standardization), dtype=torch.float32)
                target = torch.tensor(target_summary_from_vs(entry.get("train_target_vs")), dtype=torch.float32)
                candidates: list[dict[str, Any]] = []
                for engine in cfg.engines:
                    seqs = sample_sequences(
                        model, embedding, target, n=cfg.sample_batch, engine=engine,
                        temperature=cfg.temperature, seed=rng.randrange(1 << 30), device=cfg.device,
                    )
                    for seq in seqs:
                        decoded_engine, actions = decode_tokens(seq, model.vocab)
                        candidates.append(
                            {"engine": decoded_engine, "genome": format_pipeline_actions(actions),
                             "actions": actions, "tokens": seq}
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
                    risk = candidate_risk_reason_v3(cand["engine"], cand["actions"])
                    seen.add(key)
                    cand.update(
                        canonical=canonical,
                        head_type=meta.get("head_type"),
                        family_bitmap=dict(meta.get("family_bitmap") or {}),
                        risk=risk,
                    )
                    pool.append(cand)
                if not pool:
                    continue
                with torch.no_grad():
                    mean, std = model.score_token_sequences(
                        embedding, target, [c["tokens"] for c in pool], device=cfg.device
                    )
                    token_ids, pad = _pad(model, [c["tokens"] for c in pool], cfg.device)
                    p_bad = torch.sigmoid(model.instability_logit(token_ids, pad)).cpu()
                for i, cand in enumerate(pool):
                    cand["mu"], cand["sigma"] = float(mean[i]), float(std[i])
                    cand["gene_id"] = feed.proposed(
                        engine=cand["engine"], canonical=cand["canonical"], dataset_hash=dataset_hash,
                        head=cand.get("head_type"),
                        families=[k for k, v in cand["family_bitmap"].items() if v],
                        decoder="sample", mu=cand["mu"], sigma=cand["sigma"], p_bad=float(p_bad[i]),
                        risk=cand.get("risk"),
                    )
                    counters["proposed"] += 1
                # select a few for CV; the rest are surrogate-only (backpressure by design)
                selected = _select_candidates([c for c in pool if not c.get("risk")], loop_cfg)
                selected_ids = {id(c) for c in selected}
                counters["surrogate_only"] += sum(1 for c in pool if id(c) not in selected_ids)
                for cand in selected:
                    feed.selected(cand["gene_id"], engine=cand["engine"], dataset_hash=dataset_hash,
                                  canonical=cand["canonical"], selection_slot="stream_cv")
                    eval_inputs.append(
                        {"engine": cand["engine"], "dataset_hash": dataset_hash,
                         "train_csv": entry["train_csv"], "genome": cand["canonical"], "folds": cfg.folds}
                    )
                    eval_meta.append({**cand, "dataset_hash": dataset_hash})

            # ---- 2) CV the selected across the cluster; emit scored ----
            for idx, result in iter_eval_regression_with_ledger(
                eval_inputs, ledger=cfg.ledger, router=router, workers=cfg.eval_workers
            ):
                meta = eval_meta[idx]
                cv = result.get("cv") if isinstance(result.get("cv"), dict) else {}
                oof = cv.get("oof_r2")
                fold_r2 = cv.get("fold_r2") if isinstance(cv.get("fold_r2"), list) else None
                row = TrainingRow(
                    dataset_id=meta["dataset_hash"], source="active_stream",
                    embedding=embedding_from_vs(datasets[meta["dataset_hash"]]["vs_path"], standardization),
                    target_embedding=target_summary_from_vs(datasets[meta["dataset_hash"]].get("train_target_vs")),
                    genome=meta["canonical"], canonical_genome=meta["canonical"], engine=meta["engine"],
                    family_bitmap=meta.get("family_bitmap") or {},
                    metadata={"provenance": "loop", "surrogate_mu": meta["mu"],
                              "surrogate_sigma": meta["sigma"], "actions": meta["actions"], "split": "train"},
                )
                if isinstance(oof, (int, float)) and math.isfinite(float(oof)):
                    row.score = max(float(oof), -1.0)
                    row.metadata["oof_r2_raw"] = float(oof)
                    row.metadata["score_robust"] = robust_score(float(oof), fold_r2)
                    row.cost_ms = (result.get("timings") or {}).get("total_ms")
                    feed.scored(meta["gene_id"], engine=meta["engine"], dataset_hash=meta["dataset_hash"],
                                canonical=meta["canonical"], head=meta.get("head_type"), oof_r2=float(oof),
                                fold_scores=fold_r2, score_robust=float(row.metadata["score_robust"]),
                                total_ms=row.cost_ms, cache_hit=bool(result.get("cache_hit")),
                                blowup=float(oof) < -1.0)
                else:
                    row.score = None
                    row.metadata["eval_error"] = (result.get("errors") or ["unknown"])[0]
                    row.metadata["score_robust"] = -1.0
                    feed.scored(meta["gene_id"], engine=meta["engine"], dataset_hash=meta["dataset_hash"],
                                canonical=meta["canonical"], head=meta.get("head_type"),
                                timeout=bool(result.get("timeout")), cache_hit=bool(result.get("cache_hit")))
                known.add((row.dataset_id, row.engine, row.canonical_genome or row.genome))
                counters["cv_scored"] += 1
                rows_since_refit.append(row)
                with loop_rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row.to_json(), sort_keys=True) + "\n")

            # ---- 3) periodic reward-only refit so scores feed back ----
            due = (time.time() - last_refit) >= cfg.refit_interval_s or len(rows_since_refit) >= cfg.refit_min_new_rows
            if due and rows_since_refit:
                refit_metrics = _refit(model, rows_since_refit, loop_cfg, policy=False)
                torch.save(
                    model.checkpoint_payload(grammar_specs=[s.to_json() for s in _specs(model)]), cfg.checkpoint
                )
                # surface the reward-refit loss to the TUI's live training panel
                refit_count += 1
                with (cfg.run_dir / "stream_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "epoch": refit_count, "epochs": refit_count, "phase": "refit",
                        "loss": refit_metrics.get("refit_loss"), "refit_rows": len(rows_since_refit),
                    }) + "\n")
                rows_since_refit = []
                last_refit = time.time()

            counters["cycles"] += 1
            write_meta("running")
    finally:
        feed.close()
        write_meta("stopped")

    return {"run_id": run_id, **counters, "elapsed_s": round(time.time() - started, 1)}


def _pad(model: JopePolicyModelV3, token_lists: list[list[int]], device: str):
    from .policy.model_v3 import pad_sequences

    return pad_sequences(token_lists, device=device)


def _specs(model: JopePolicyModelV3):
    specs = getattr(model, "_grammar_specs", None)
    if specs:
        return specs
    from .grammars import all_grammars

    return all_grammars()
