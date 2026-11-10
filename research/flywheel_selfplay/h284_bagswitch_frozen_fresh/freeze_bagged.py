"""Full-DL-Train freeze and export of H279 BAG_SWITCH (q = 6) and BAG_SWITCH_NESTED (q* in {5, 6, 7})
plus their paired RERANK8_VETO proposer across seeds 279101-279103 on all 205 DL-Train families (1,219 contexts).

Every model is fit on DL-Train ONLY before any fresh pool or DL-Val pool is read, serialized as LightGBM text
to export_dir, and verified bit-exact on reload across 64 DL-Train contexts."""

import hashlib
import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

import bagged
import h265_core as core
import veto

K = 8
SEEDS = (279101, 279102, 279103)
H278_SEEDS = (278101, 278102, 278103)
SALT = ""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def save_json(path: Path, obj: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=float)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha256_file(path)


def family_folds(train_rows: list[dict], fam_by_cid: dict[str, str], seed: int, n: int = 4) -> dict[int, list[str]]:
    """Deterministic 4-fold family-grouped split of all 205 DL-Train families (same stream as H278 freeze.py)."""
    fams = sorted(set(fam_by_cid.values()))
    rng = np.random.default_rng(core.stream_seed(seed, "h278-full-train-folds"))
    perm = list(rng.permutation(fams))
    f2c: dict[str, list[str]] = {}
    for r in train_rows:
        cid = r["context_sha256"]
        f2c.setdefault(fam_by_cid[cid], []).append(cid)
    out: dict[int, list[str]] = {0: []}
    for i, f in enumerate(perm):
        out.setdefault(1 + (i % n), []).extend(sorted(f2c[f]))
    return {k: sorted(v) for k, v in out.items()}


def verify_reload(model_str: str, path: Path, X_check: np.ndarray, num_iteration: int | None = None) -> dict:
    mem_b = lgb.Booster(model_str=model_str)
    disk_b = lgb.Booster(model_file=str(path))
    if disk_b.num_feature() != 568:
        raise RuntimeError(f"feature count mismatch on {path}: {disk_b.num_feature()}")
    if num_iteration is not None:
        mem_p = np.asarray(mem_b.predict(X_check, num_iteration=int(num_iteration)), dtype=np.float64)
        disk_p = np.asarray(disk_b.predict(X_check, num_iteration=int(num_iteration)), dtype=np.float64)
    else:
        mem_p = np.asarray(mem_b.predict(X_check), dtype=np.float64)
        disk_p = np.asarray(disk_b.predict(X_check), dtype=np.float64)
    if not np.array_equal(mem_p, disk_p):
        raise RuntimeError(f"serialized booster prediction mismatch on {path}")
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "num_feature": int(disk_b.num_feature()),
        "reload_bit_exact_on_64_train_contexts": True,
    }


def fit_full_train_bagged(
    train_rows: list[dict],
    packets: dict[str, dict],
    fam_by_cid: dict[str, str],
    export_dir: Path,
    code_hashes: dict[str, str],
    source_commit: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Fit H279 RERANK8_VETO proposer + B=8 family-bootstrapped rankers on all 205 DL-Train families (1,219 contexts)."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    all_cids = sorted(packets.keys())
    all_fams = sorted(set(fam_by_cid.values()))
    if len(all_cids) != 1219 or len(all_fams) != 205:
        raise RuntimeError(f"unexpected DL-Train size: {len(all_cids)} contexts, {len(all_fams)} families")

    check_cids = all_cids[:64]
    X_check = np.concatenate([packets[c]["X"] for c in check_cids])
    cells = []

    for seed in seeds:
        cids_by_fold = family_folds(train_rows, fam_by_cid, seed, n=4)
        fit_cids = [c for f in (1, 2, 3, 4) for c in cids_by_fold[f]]
        if sorted(fit_cids) != all_cids:
            raise RuntimeError("full-DL-Train 4-fold partition did not cover all 1,219 DL-Train contexts")
        fold_fams = {
            str(f): sorted({fam_by_cid[c] for c in cids_by_fold[f]})
            for f in (1, 2, 3, 4)
        }
        cids_by_fold[0] = [cids_by_fold[1][0]]  # 1 dummy assess cid; fit uses folds 1..4 only

        cell = bagged.run_cell_bagged(
            seed, 0, cids_by_fold, packets, fam_by_cid, K, SALT, store_models=True
        )
        models = cell["models"]
        receipts = cell["receipts"]
        n_est = int(models["n_estimators"])
        strict_margin = models["strict_margin"]
        raw_margin = float(models["raw_margin"])
        q_nested = int(models["q_nested"])

        ranker_path = export_dir / f"ranker-seed{seed}.txt"
        veto_path = export_dir / f"veto-seed{seed}.txt"
        ranker_path.write_text(models["rv_model"])
        veto_path.write_text(models["rv_gate"])
        ranker_info = verify_reload(models["rv_model"], ranker_path, X_check, num_iteration=n_est)
        veto_info = verify_reload(models["rv_gate"], veto_path, X_check, num_iteration=None)

        bag_infos = []
        for b, m_str in enumerate(models["bag_models"]):
            b_path = export_dir / f"bag{b}-seed{seed}.txt"
            b_path.write_text(m_str)
            b_info = verify_reload(m_str, b_path, X_check, num_iteration=n_est)
            b_info["bag_index"] = int(b)
            bag_infos.append(b_info)

        cells.append(
            {
                "seed": int(seed),
                "selected_lr": float(receipts["RERANK8_VETO"]["lr"]),
                "selected_n_estimators": n_est,
                "raw_margin": raw_margin,
                "strict_margin": "inf" if math.isinf(strict_margin) else float(strict_margin),
                "strict_admitted": bool(receipts["RERANK8_VETO"]["strict_admitted"]),
                "selected_tau": float(veto.TAU),
                "n_bags": int(bagged.N_BAGS),
                "q_fixed": int(bagged.Q_FIXED),
                "q_grid": list(bagged.Q_GRID),
                "q_selected": q_nested,
                "q_selection_tier": str(receipts["BAG_SWITCH_NESTED"]["q_selection_tier"]),
                "inner_q_summary": receipts["BAG_SWITCH_NESTED"]["inner_q_summary"],
                "receipts": {
                    k: v for k, v in receipts.items() if k in ("RERANK8_VETO", "BAG_SWITCH", "BAG_SWITCH_NESTED")
                },
                "inner_folds_family_ids": fold_fams,
                "ranker": ranker_info,
                "veto": veto_info,
                "bags": bag_infos,
            }
        )

    lock = {
        "schema": "jope.h284.fit-lock/v1",
        "wave": "H284",
        "arms": ["BAG_SWITCH", "BAG_SWITCH_NESTED", "RERANK8_VETO"],
        "fit_split": "DL-Train",
        "dl_val_reads_before_lock": 0,
        "fresh_pools_read_before_lock": 0,
        "training_family_ids": all_fams,
        "training_context_count": len(all_cids),
        "training_family_count": len(all_fams),
        "k": K,
        "n_bags": int(bagged.N_BAGS),
        "q_fixed": int(bagged.Q_FIXED),
        "q_grid": list(bagged.Q_GRID),
        "tau": float(veto.TAU),
        "seeds": [int(s) for s in seeds],
        "cells": cells,
        "code_sha256": code_hashes,
        "source_commit": source_commit,
    }
    lock_sha = save_json(export_dir / "fit_lock.json", lock)
    lock["fit_lock_sha256"] = lock_sha
    return lock


def parse_margin(val) -> float:
    return math.inf if val == "inf" else float(val)


def score_frozen_bagged_cell(export_dir: Path, cell_info: dict, X: np.ndarray, n_ctx: int, k: int = K) -> dict:
    """Apply one frozen H284 seed's models (ranker, veto, bag0..7) unchanged to feature matrix X (n_ctx * k, 568)."""
    export_dir = Path(export_dir)
    r_path = export_dir / cell_info["ranker"]["file"]
    v_path = export_dir / cell_info["veto"]["file"]
    if sha256_file(r_path) != cell_info["ranker"]["sha256"]:
        raise RuntimeError(f"ranker sha256 mismatch: {r_path}")
    if sha256_file(v_path) != cell_info["veto"]["sha256"]:
        raise RuntimeError(f"veto sha256 mismatch: {v_path}")

    n_est = int(cell_info["selected_n_estimators"])
    strict_margin = parse_margin(cell_info["strict_margin"])
    tau = float(cell_info["selected_tau"])
    q_fixed = int(cell_info["q_fixed"])
    q_nested = int(cell_info["q_selected"])

    ranker = lgb.Booster(model_file=str(r_path))
    gate = lgb.Booster(model_file=str(v_path))
    if ranker.num_feature() != 568 or gate.num_feature() != 568:
        raise RuntimeError("frozen model feature count mismatch")

    p_inv = np.asarray(gate.predict(X), dtype=np.float64).reshape(n_ctx, k)
    raw_scores = np.asarray(ranker.predict(X, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, k)
    masked_scores = veto.mask(raw_scores, p_inv, tau=tau)
    j_ref = [core.select_from_top4_scores(masked_scores[i], strict_margin) for i in range(n_ctx)]

    bag_scores = []
    for b_info in cell_info["bags"]:
        b_path = export_dir / b_info["file"]
        if sha256_file(b_path) != b_info["sha256"]:
            raise RuntimeError(f"bag sha256 mismatch: {b_path}")
        bm = lgb.Booster(model_file=str(b_path))
        if bm.num_feature() != 568:
            raise RuntimeError(f"bag feature count mismatch: {b_path}")
        sb = np.asarray(bm.predict(X, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, k)
        bag_scores.append(sb)

    bag_scores_arr = np.stack(bag_scores, axis=0)
    votes = bagged.vote_counts(bag_scores_arr, j_ref, strict_margin)
    picks_bag = bagged.switch_by_votes(j_ref, votes, q_fixed)
    picks_nested = bagged.switch_by_votes(j_ref, votes, q_nested)

    return {
        "slot0": [0] * n_ctx,
        "RERANK8_VETO": j_ref,
        "BAG_SWITCH": picks_bag,
        "BAG_SWITCH_NESTED": picks_nested,
        "votes": votes.tolist(),
    }


def score_frozen_h278_cell(h278_dir: Path, cell_info: dict, X: np.ndarray, n_ctx: int, k: int = K) -> list[int]:
    """Apply one frozen H278 seed's (278101-278103) RERANK8_VETO model unchanged to feature matrix X."""
    h278_dir = Path(h278_dir)
    r_path = h278_dir / cell_info["ranker"]["file"]
    v_path = h278_dir / cell_info["veto"]["file"]
    if sha256_file(r_path) != cell_info["ranker"]["sha256"]:
        raise RuntimeError(f"H278 ranker sha256 mismatch: {r_path}")
    if sha256_file(v_path) != cell_info["veto"]["sha256"]:
        raise RuntimeError(f"H278 veto sha256 mismatch: {v_path}")

    n_est = int(cell_info["selected_n_estimators"])
    strict_margin = parse_margin(cell_info["strict_margin"])
    tau = float(cell_info["selected_tau"])

    ranker = lgb.Booster(model_file=str(r_path))
    gate = lgb.Booster(model_file=str(v_path))
    p_inv = np.asarray(gate.predict(X), dtype=np.float64).reshape(n_ctx, k)
    raw_scores = np.asarray(ranker.predict(X, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, k)
    masked_scores = veto.mask(raw_scores, p_inv, tau=tau)
    return [core.select_from_top4_scores(masked_scores[i], strict_margin) for i in range(n_ctx)]
