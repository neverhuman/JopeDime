"""Standing Elo benchmark: place the policy's proposals on the 97-algorithm regression ladder.

The headline number is **top-1** — the model's own reward-ranked pick, evaluated on each
held-out dataset's test.csv and Elo-rated against the incumbent field via the quant
ladder's own ``compute_elo`` (utility = −RMSE). ``cv_select`` (pick by OOF among K) and
``best_k`` (oracle test-selection) are DIAGNOSTICS ONLY: they bound how much headroom the
neural selector still has vs the generator. The deployed path is pure one-shot — no data
evals at selection time.

Dataset universe: the jope-validation split (held out from the policy model), with the
subset that is ALSO in the quant ladder's ``real_test`` reported as the "double-holdout"
headline (held out from both the model and the ladder's own selection).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .prime_config import load_config

_PRIME_CONFIG = load_config()
QUANT_MODELS = _PRIME_CONFIG.quant_models_root
LADDER_CACHE = QUANT_MODELS / "tools/artifacts/regression_router_cache_all_real_models_v1"
REGRESSION_ROOT = _PRIME_CONFIG.regression_dataset_root


def _load_ladder():
    lab = np.load(LADDER_CACHE / "labels.npz", allow_pickle=True)
    return {
        "rmse": lab["rmse"],
        "algorithms": [str(a) for a in lab["algorithms"]],
        "hashes": [str(x) for x in lab["dataset_hashes"]],
        "valid": lab["valid_mask"],
    }


def select_bench_datasets(art_dir: str | Path, *, limit: int | None = None) -> dict[str, list[str]]:
    """All jope-validation hashes present in the ladder; `double_holdout` = ∩ quant real_test."""
    manifest = json.loads((Path(art_dir) / "split_manifest.json").read_text())["datasets"]
    val = [h for h, e in manifest.items() if e.get("split") == "validation"]
    ladder = set(_load_ladder()["hashes"])
    split = json.loads((LADDER_CACHE / "split.json").read_text())
    real_test = set(split.get("real_test", []))
    validation = sorted(h for h in val if h in ladder)
    double = [h for h in validation if h in real_test]
    if limit:
        validation = validation[:limit]
        double = [h for h in double if h in validation]
    return {"validation": validation, "double_holdout": double}


def propose_and_eval(
    checkpoint: str | Path,
    hashes: list[str],
    *,
    art_dir: str | Path,
    k: int = 16,
    folds: int = 3,
    seed: int = 17,
    temperature: float = 0.7,
    ledger: str | Path | None = None,
    workers: int = 90,
    selector_calibration: str | Path | dict[str, Any] | None = None,
    selector_reranker: str | Path | dict[str, Any] | None = None,
    samples_per_engine: int | None = None,
    candidate_pool_dir: str | Path | None = None,
    proposal_failure_threshold: float | None = 0.01,
    log=print,
) -> dict[str, list[dict]]:
    """Propose K per dataset with the policy, bridge-evaluate all, return per-dataset rows."""
    from .data.active_v3 import robust_score
    from .engines import EngineRouter
    from .eval_ledger import iter_eval_regression_with_ledger
    from .policy.decode import propose_v3
    from .policy.model_v3 import load_policy_v3
    from .propose import embedding_from_vs, target_summary_from_vs
    from .self_play import _candidate_pool_record, _proposal_failure_exceeded

    std = json.loads((Path(art_dir) / "standardization.json").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_policy_v3(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    router = EngineRouter()
    if ledger is None:
        ledger = _PRIME_CONFIG.artifacts / "elo_bench_ledger.jsonl"

    per_ds: dict[str, list[dict]] = {}
    eval_inputs: list[dict] = []
    candidate_pool_rows: list[dict] = []
    candidate_pool_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    proposal_errors: list[dict[str, Any]] = []
    t0 = time.time()
    for i, h in enumerate(hashes):
        d = REGRESSION_ROOT / h
        emb = embedding_from_vs(str(d / "hyperion3train128dim.vs"), std)
        tgt = target_summary_from_vs(str(d / "train_target.vs"))
        try:
            result = propose_v3(
                emb, target_summary=tgt, model=model, router=router,
                n=k, seed=seed, temperature=temperature, engines=("featmath", "newmath"),
                device=device, samples_per_engine=samples_per_engine,
                selector_calibration=selector_calibration,
                selector_reranker=selector_reranker,
                return_all_candidates=candidate_pool_dir is not None,
            )
            if candidate_pool_dir is not None:
                props, all_candidates = result
                for cand in all_candidates:
                    canonical = cand.get("canonical_genome") or cand.get("genome")
                    engine = str(cand.get("engine") or "")
                    if not canonical or not engine:
                        continue
                    raw = f"elo|{h}|{engine}|{canonical}".encode()
                    cpid = hashlib.sha1(raw).hexdigest()[:16]
                    row = _candidate_pool_record(
                        generation=-1,
                        candidate_pool_id=cpid,
                        policy_checkpoint=str(checkpoint),
                        dataset_hash=h,
                        row=cand,
                    )
                    candidate_pool_rows.append(row)
                    candidate_pool_lookup[(h, engine, str(canonical))] = row
            else:
                props = result
        except Exception as exc:  # noqa: BLE001
            proposal_errors.append({"dataset_hash": h, "error": repr(exc)})
            continue
        if not props:
            proposal_errors.append({"dataset_hash": h, "error": "empty_proposals"})
            continue
        per_ds[h] = []
        for rank, p in enumerate(props):
            g = p.get("canonical_genome") or p.get("genome")
            engine = str(p.get("engine") or "")
            pool_row = candidate_pool_lookup.get((h, engine, str(g)))
            cpid = pool_row.get("candidate_pool_id") if pool_row else None
            if pool_row is not None:
                pool_row["selected_for_eval_rank"] = rank
            per_ds[h].append({
                "rank": rank, "engine": p.get("engine"), "genome": g,
                "reward_mean": p.get("reward_mean", p.get("reward")), "reward_std": p.get("reward_std"),
                "selection_key": p.get("selection_key"), "selection_score": p.get("selection_score"),
                "selector_reranker_score": p.get("selector_reranker_score"),
                "candidate_pool_id": cpid,
            })
            eval_inputs.append({
                "engine": p.get("engine"), "dataset_hash": h,
                "train_csv": str(d / "train.csv"), "test_csv": str(d / "test.csv"),
                "genome": g, "folds": folds, "_h": h, "_rank": rank,
                "_candidate_pool_id": cpid,
            })
        if (i + 1) % 10 == 0 or i + 1 == len(hashes):
            log(f"  proposed {i+1}/{len(hashes)} datasets ({time.time()-t0:.0f}s)")
    if proposal_errors:
        err_path = Path(ledger).with_name("elo_proposal_errors.jsonl")
        with err_path.open("w", encoding="utf-8") as fh:
            for row in proposal_errors:
                fh.write(json.dumps(row) + "\n")
    if _proposal_failure_exceeded(len(proposal_errors), len(hashes), proposal_failure_threshold):
        raise RuntimeError(
            f"elo-bench proposal coverage below threshold: "
            f"failure_rate={len(proposal_errors) / max(1, len(hashes)):.4f} "
            f"threshold={proposal_failure_threshold}"
        )

    log(f"  evaluating {len(eval_inputs)} proposals via bridge...")
    t1 = time.time()
    results = list(iter_eval_regression_with_ledger(
        eval_inputs, ledger=str(ledger), router=router, workers=workers))
    log(f"  evaluated in {time.time()-t1:.0f}s")
    by_idx = {idx: res for idx, res in results}
    pool_by_id = {
        str(row.get("candidate_pool_id")): row
        for row in candidate_pool_rows
        if row.get("candidate_pool_id")
    }
    for idx, inp in enumerate(eval_inputs):
        res = by_idx.get(idx, {}) or {}
        cv = res.get("cv") or {}
        test = (res.get("metrics") or {}).get("test") or {}
        rec = per_ds[inp["_h"]][inp["_rank"]]
        rec["oof_r2"] = cv.get("oof_r2")
        rec["score_robust"] = robust_score(float(cv["oof_r2"]), cv.get("fold_r2")) if _finite(cv.get("oof_r2")) else None
        rec["test_rmse"] = test.get("rmse")
        rec["test_r2"] = test.get("r2")
        rec["valid"] = bool(res.get("valid", False))
        rec["cost_ms"] = (res.get("timings") or {}).get("total_ms")
        pool_row = pool_by_id.get(str(inp.get("_candidate_pool_id") or ""))
        if pool_row is not None:
            pool_row["measured_oof"] = rec.get("oof_r2")
            pool_row["score_robust"] = rec.get("score_robust")
            pool_row["cost_ms"] = rec.get("cost_ms")
    for rows in per_ds.values():
        cand = sorted([row for row in rows if _finite(row.get("test_rmse"))], key=lambda row: row["rank"])[:6]
        withcv = [row for row in cand if _finite(row.get("score_robust")) or _finite(row.get("oof_r2"))]
        if not withcv:
            continue
        deployed = max(withcv, key=_probe_score)
        pool_row = pool_by_id.get(str(deployed.get("candidate_pool_id") or ""))
        if pool_row is not None:
            pool_row["deployed"] = True
    if candidate_pool_dir is not None:
        out = Path(candidate_pool_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "candidate_pools_elo.jsonl").open("w", encoding="utf-8") as fh:
            for row in candidate_pool_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return per_ds


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def rmse_for_mode(props: list[dict], mode: str) -> float | None:
    """top1 = deployed (reward-ranked pick); cv_select/best_k = diagnostics.

    cv_topN (cheap-probe selector): probe ONLY the N best reward-ranked candidates with
    CV and pick the best oof_r2 among them — the deployed cost is N probes/dataset, not K.
    N=K reproduces cv_select; N=1 reproduces top1. The Elo(N) curve says how cheap the
    probe can get while keeping the +104 top1->cv gap.
    """
    ok = [p for p in props if _finite(p.get("test_rmse"))]
    if not ok:
        return None
    if mode == "top1":
        return float(min(ok, key=lambda p: p["rank"])["test_rmse"])
    if mode == "cv_select":
        withcv = [p for p in ok if _finite(p.get("score_robust")) or _finite(p.get("oof_r2"))]
        pick = max(withcv, key=_probe_score) if withcv else min(ok, key=lambda p: p["rank"])
        return float(pick["test_rmse"])
    if mode.startswith("cv_top"):
        n = int(mode[len("cv_top"):])
        cand = sorted(ok, key=lambda p: p["rank"])[:n]              # top-N by reward rank
        withcv = [p for p in cand if _finite(p.get("score_robust")) or _finite(p.get("oof_r2"))]
        pick = max(withcv, key=_probe_score) if withcv else min(cand, key=lambda p: p["rank"])
        return float(pick["test_rmse"])
    return float(min(ok, key=lambda p: p["test_rmse"])["test_rmse"])  # best_k


def _probe_score(row: dict[str, Any]) -> float:
    value = row.get("score_robust")
    if _finite(value):
        return float(value)
    return float(row.get("oof_r2"))


def ladder_elo(per_ds: dict[str, list[dict]], hashes: list[str], mode: str) -> dict[str, Any]:
    """98-player tournament over `hashes`: incumbents + a jope column (utility = −RMSE)."""
    sys.path.insert(0, str(QUANT_MODELS))
    from tools.elo import algorithm_matches, compute_elo  # read-only import

    lad = _load_ladder()
    hidx = {h: i for i, h in enumerate(lad["hashes"])}
    rows = [hidx[h] for h in hashes if h in hidx]
    U = -lad["rmse"].copy()
    jc = np.full((U.shape[0], 1), np.nan)
    jv = np.zeros((U.shape[0], 1), bool)
    per_dataset = []
    for h in hashes:
        if h not in hidx:
            continue
        r = hidx[h]
        jr = rmse_for_mode(per_ds.get(h, []), mode)
        if jr is not None:
            jc[r, 0] = -jr
            jv[r, 0] = True
        inc = lad["rmse"][r]
        m = np.asarray(lad["valid"][r], bool) & np.isfinite(inc)
        inc_valid = inc[m]
        if jr is not None and len(inc_valid):
            beat = int((jr < inc_valid).sum())
            n = len(inc_valid)
            p = min(max(beat / n, 1e-4), 1 - 1e-4)
            per_dataset.append({
                "hash": h, "rmse": jr, "rank": int((inc_valid < jr).sum()) + 1, "field": n,
                "beat": beat, "pctile": round(100 * beat / n, 1),
                "elo_local": round(1000 + 400 * math.log10(p / (1 - p))),
            })
        else:
            per_dataset.append({"hash": h, "rmse": None, "rank": None, "field": int(len(inc_valid)),
                                "beat": 0, "pctile": 0.0, "elo_local": None})
    name = f"jope_{mode}"
    algos = lad["algorithms"] + [name]
    U_aug = np.hstack([U, jc])
    v_aug = np.hstack([lad["valid"], jv])
    matches = algorithm_matches(algos, U_aug, v_aug, dataset_indices=rows)
    ratings = compute_elo(set(algos), matches)
    order = sorted(ratings.items(), key=lambda kv: -kv[1])
    jrank = next(i for i, (kk, _) in enumerate(order) if kk == name) + 1
    # "battles won": datasets where jope's pick beats EVERY incumbent (rank 1 of the field)
    battles_won = sum(1 for p in per_dataset if p.get("rank") == 1)
    battles_top3 = sum(1 for p in per_dataset if p.get("rank") is not None and p["rank"] <= 3)
    return {
        "mode": mode, "elo": round(ratings[name], 1), "field_rank": jrank, "field_size": len(order),
        "battles_won": battles_won, "battles_top3": battles_top3,
        "n_scored": sum(1 for p in per_dataset if p.get("rank") is not None),
        "top_incumbents": [(kk, round(vv, 1)) for kk, vv in order[:5] if kk != name][:5],
        "per_dataset": per_dataset,
    }


def run_elo_bench(
    checkpoint: str | Path,
    art_dir: str | Path,
    *,
    k: int = 16,
    folds: int = 3,
    seed: int = 17,
    temperature: float = 0.7,
    limit: int | None = None,
    subset: str = "validation",
    out_dir: str | Path | None = None,
    ledger: str | Path | None = None,
    workers: int = 90,
    selector_calibration: str | Path | dict[str, Any] | None = None,
    selector_reranker: str | Path | dict[str, Any] | None = None,
    samples_per_engine: int | None = None,
    candidate_pool_dir: str | Path | None = None,
    proposal_failure_threshold: float | None = 0.01,
    log=print,
) -> dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else _PRIME_CONFIG.artifacts / "elo_bench"
    out.mkdir(parents=True, exist_ok=True)
    sets = select_bench_datasets(art_dir, limit=limit)
    hashes = sets[subset] if subset in sets else sets["validation"]
    log(f"elo-bench: {len(hashes)} datasets ({subset}; double_holdout={len(sets['double_holdout'])}) k={k}")
    per_ds = propose_and_eval(
        checkpoint, hashes, art_dir=art_dir, k=k, folds=folds, seed=seed,
        temperature=temperature, ledger=ledger or (out / "ledger.jsonl"), workers=workers, log=log,
        selector_calibration=selector_calibration,
        selector_reranker=selector_reranker,
        samples_per_engine=samples_per_engine,
        candidate_pool_dir=candidate_pool_dir,
        proposal_failure_threshold=proposal_failure_threshold,
    )
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint), "k": k, "folds": folds, "seed": seed,
        "temperature": temperature, "subset": subset, "n_datasets": len(hashes),
        "double_holdout_n": len(sets["double_holdout"]),
        "selector_calibration": str(selector_calibration) if selector_calibration else None,
        "selector_reranker": str(selector_reranker) if selector_reranker else None,
        "samples_per_engine": samples_per_engine,
        "candidate_pool_dir": str(candidate_pool_dir) if candidate_pool_dir else None,
        "modes": {},
    }
    for mode in ("top1", "cv_top4", "cv_top6", "cv_select", "best_k"):
        res = ladder_elo(per_ds, hashes, mode)
        report["modes"][mode] = {kk: vv for kk, vv in res.items() if kk != "per_dataset"}
        if mode == "top1":
            report["per_dataset_top1"] = res["per_dataset"]
        # double-holdout headline for each mode
        dh = set(sets["double_holdout"])
        if dh and len(dh) < len(hashes):
            res_dh = ladder_elo(per_ds, [h for h in hashes if h in dh], mode)
            report["modes"][mode]["double_holdout_elo"] = res_dh["elo"]
            report["modes"][mode]["double_holdout_rank"] = res_dh["field_rank"]
    (out / "per_ds.json").write_text(json.dumps(per_ds))
    (out / "elo_report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps({m: {kk: vv for kk, vv in d.items() if kk != "top_incumbents"}
                    for m, d in report["modes"].items()}, indent=2))
    return report
