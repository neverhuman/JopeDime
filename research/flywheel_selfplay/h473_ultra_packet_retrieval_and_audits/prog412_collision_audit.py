#!/usr/bin/env python3
"""H473 Part A2 (DV-03 `prog_412` Collision Audit), Diverse 4-Anchor Selector, & Part C `I10` `op_desc`.

Quantifies exact and near (`cosine > 0.9999`) `prog_412` feature collisions across candidate pools
and `champions.json`, measures within-context utility and gate disagreement across colliding pairs,
provides storage-order-invariant 4-archetype selection (`DV-02`), and runs `I10` zero-shot operator
signature (`op_desc`) canonical template selection.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from h256_token_layout_audit import (
    derange_across_families,
    family_unweighted_mean,
    verify_mem_available_gib,
)

# Four canonical 24-D structural prototypes spanning heads (0..4) and regimes (5..9).
ARCHETYPE_PROTOTYPES_4X24: np.ndarray = np.array(
    [
        # 0: ridge_gcv + standardize/projection linear regime
        [0.0, 0.0, 0.0, 0.0, 1.5, 0.1, 0.1, 0.1, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.5, 0.5, 0.0, 0.8, 0.4],
        # 1: histgbm + heavy_tail / monotone_rank regime
        [0.0, 1.5, 0.0, 0.0, 0.0, 1.1, 1.0, 0.1, 0.0, 0.4, 0.8, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.7, 0.7, 0.0, 0.9, 0.8],
        # 2: histgbm + polynomial / Chebyshev / hinge regime
        [0.0, 1.5, 0.0, 0.0, 0.0, 0.4, 0.2, 1.3, 0.6, 0.3, 0.9, 0.9, 0.6, 0.0, 0.0, 0.8, 0.7, 0.0, 0.7, 0.8, 0.8, 1.0, 0.3, 0.8],
        # 3: ksmooth / elasticnet + RFF / oblique interaction regime
        [0.9, 0.0, 1.1, 0.3, 0.0, 0.2, 0.3, 0.3, 1.3, 0.5, 0.0, 0.0, 0.0, 0.7, 0.7, 0.2, 0.3, 0.4, 0.6, 0.7, 0.7, 1.0, 0.4, 0.8],
    ],
    dtype=np.float64,
)
ARCHETYPE_PROTOTYPES_4X24 = ARCHETYPE_PROTOTYPES_4X24 / np.linalg.norm(
    ARCHETYPE_PROTOTYPES_4X24, axis=1, keepdims=True
)


def _row_digests_f32(mat: np.ndarray, quant_decimals: int | None = None) -> list[bytes]:
    arr = np.asarray(mat, dtype=np.float32)
    if quant_decimals is not None:
        arr = np.round(arr, decimals=quant_decimals).astype(np.float32)
    arr = np.where(arr == 0.0, np.float32(0.0), arr)
    le = np.ascontiguousarray(arr, dtype="<f4")
    stride = le.shape[1] * 4
    buf = memoryview(le.tobytes())
    return [
        hashlib.blake2b(buf[i * stride : (i + 1) * stride], digest_size=16).digest()
        for i in range(le.shape[0])
    ]


def _disagreement_stats(u_diffs: list[float], both_pass_diffs: list[float], gate_mismatches: int, n_ctx_hit: int, n_ctx_total: int) -> dict[str, Any]:
    n_pairs = len(u_diffs)
    if n_pairs == 0:
        return {
            "contexts_with_collision": 0,
            "context_fraction": 0.0,
            "pair_count": 0,
            "utility_abs_diff_mean": 0.0,
            "utility_abs_diff_median": 0.0,
            "utility_abs_diff_p90": 0.0,
            "utility_abs_diff_p99": 0.0,
            "utility_abs_diff_max": 0.0,
            "both_gate_pass_pair_count": 0,
            "both_gate_pass_abs_diff_mean": 0.0,
            "both_gate_pass_abs_diff_p90": 0.0,
            "frac_abs_diff_gt_0p01": 0.0,
            "frac_abs_diff_gt_0p05": 0.0,
            "gate_disagreement_count": 0,
            "gate_disagreement_rate": 0.0,
        }
    arr = np.asarray(u_diffs, dtype=np.float64)
    bp = np.asarray(both_pass_diffs, dtype=np.float64) if both_pass_diffs else np.zeros(1, dtype=np.float64)
    return {
        "contexts_with_collision": int(n_ctx_hit),
        "context_fraction": round(float(n_ctx_hit) / max(n_ctx_total, 1), 6),
        "pair_count": int(n_pairs),
        "utility_abs_diff_mean": round(float(np.mean(arr)), 6),
        "utility_abs_diff_median": round(float(np.median(arr)), 6),
        "utility_abs_diff_p90": round(float(np.percentile(arr, 90)), 6),
        "utility_abs_diff_p99": round(float(np.percentile(arr, 99)), 6),
        "utility_abs_diff_max": round(float(np.max(arr)), 6),
        "both_gate_pass_pair_count": int(len(both_pass_diffs)),
        "both_gate_pass_abs_diff_mean": round(float(np.mean(bp)) if both_pass_diffs else 0.0, 6),
        "both_gate_pass_abs_diff_p90": round(float(np.percentile(bp, 90)) if both_pass_diffs else 0.0, 6),
        "frac_abs_diff_gt_0p01": round(float(np.mean(arr > 0.01)), 6),
        "frac_abs_diff_gt_0p05": round(float(np.mean(arr > 0.05)), 6),
        "gate_disagreement_count": int(gate_mismatches),
        "gate_disagreement_rate": round(float(gate_mismatches) / float(n_pairs), 6),
    }


def _summarize_bucket_collisions(pid_to_digest: Mapping[str, bytes]) -> dict[str, Any]:
    buckets: dict[bytes, int] = {}
    for dg in pid_to_digest.values():
        buckets[dg] = buckets.get(dg, 0) + 1
    n_pids = len(pid_to_digest)
    n_distinct_vecs = len(buckets)
    multi = [cnt for cnt in buckets.values() if cnt >= 2]
    colliding_pids = sum(multi)
    colliding_pairs = sum(cnt * (cnt - 1) // 2 for cnt in multi)
    return {
        "distinct_vectors": int(n_distinct_vecs),
        "collision_groups": int(len(multi)),
        "colliding_programs": int(colliding_pids),
        "colliding_program_fraction": round(float(colliding_pids) / max(n_pids, 1), 6),
        "colliding_unordered_pairs": int(colliding_pairs),
        "max_collision_group_size": int(max(multi) if multi else 1),
    }


def audit_prog412_collisions(
    ordered_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    *,
    champions_records: Sequence[Mapping[str, Any]] | None = None,
    near_cos_threshold: float = 0.9999,
) -> dict[str, Any]:
    verify_mem_available_gib(15.0, "dv03_prog412_audit")
    pid_exact_dg: dict[str, bytes] = {}
    pid_q4_dg: dict[str, bytes] = {}
    pid_sem_dg: dict[str, bytes] = {}
    hash_inconsistencies = 0
    total_evals = 0

    ex_u_diffs: list[float] = []
    ex_bp_diffs: list[float] = []
    ex_gate_mis = 0
    ex_ctx_hit = 0

    nr_u_diffs: list[float] = []
    nr_bp_diffs: list[float] = []
    nr_gate_mis = 0
    nr_ctx_hit = 0

    sm_u_diffs: list[float] = []
    sm_bp_diffs: list[float] = []
    sm_gate_mis = 0
    sm_ctx_hit = 0

    champ_want: dict[str, str] = {}
    if champions_records is not None:
        for rec in champions_records:
            if rec.get("dl_split") == "DL-Train":
                champ_want[str(rec["program_sha256"])] = str(rec.get("dsl", ""))
    champ_vecs: dict[str, np.ndarray] = {}

    for cid in ordered_cids:
        pool = pools[cid]
        pids = [str(p) for p in pool["pids"]]
        prog = np.asarray(pool["program"], dtype=np.float32)
        fit = np.asarray(pool["fitness"], dtype=np.float64)
        gate = np.asarray(pool["gate"], dtype=bool)
        util = np.where(gate, -fit, -2.0)
        sem24 = np.asarray(pool["struct_sig"], dtype=np.float32)
        m_len = len(pids)
        total_evals += m_len

        dgs_ex = _row_digests_f32(prog, quant_decimals=None)
        dgs_q4 = _row_digests_f32(prog, quant_decimals=4)
        dgs_sm = _row_digests_f32(sem24, quant_decimals=None)

        local_ex_buckets: dict[bytes, list[int]] = {}
        local_sm_buckets: dict[bytes, list[int]] = {}
        for j, pid in enumerate(pids):
            dg_e = dgs_ex[j]
            prev = pid_exact_dg.get(pid)
            if prev is None:
                pid_exact_dg[pid] = dg_e
                pid_q4_dg[pid] = dgs_q4[j]
                pid_sem_dg[pid] = dgs_sm[j]
            elif prev != dg_e:
                hash_inconsistencies += 1
            if pid in champ_want and pid not in champ_vecs:
                champ_vecs[pid] = prog[j].copy()
            local_ex_buckets.setdefault(dg_e, []).append(j)
            local_sm_buckets.setdefault(dgs_sm[j], []).append(j)

        hit_ex_this = False
        for idxs in local_ex_buckets.values():
            if len(idxs) >= 2:
                for a_pos in range(len(idxs)):
                    ia = idxs[a_pos]
                    for b_pos in range(a_pos + 1, len(idxs)):
                        ib = idxs[b_pos]
                        if pids[ia] == pids[ib]:
                            continue
                        hit_ex_this = True
                        du = abs(float(util[ia] - util[ib]))
                        ex_u_diffs.append(du)
                        if gate[ia] and gate[ib]:
                            ex_bp_diffs.append(du)
                        if gate[ia] != gate[ib]:
                            ex_gate_mis += 1
        if hit_ex_this:
            ex_ctx_hit += 1

        hit_sm_this = False
        for idxs in local_sm_buckets.values():
            if len(idxs) >= 2:
                for a_pos in range(len(idxs)):
                    ia = idxs[a_pos]
                    for b_pos in range(a_pos + 1, len(idxs)):
                        ib = idxs[b_pos]
                        if pids[ia] == pids[ib]:
                            continue
                        hit_sm_this = True
                        du = abs(float(util[ia] - util[ib]))
                        sm_u_diffs.append(du)
                        if gate[ia] and gate[ib]:
                            sm_bp_diffs.append(du)
                        if gate[ia] != gate[ib]:
                            sm_gate_mis += 1
        if hit_sm_this:
            sm_ctx_hit += 1

        # Exact within-pool cosine > near_cos_threshold for non-identical prog_412 pairs
        norms = np.linalg.norm(prog.astype(np.float64), axis=1, keepdims=True)
        unit = prog.astype(np.float64) / np.maximum(norms, 1e-12)
        cos_mat = unit @ unit.T
        rows_i, cols_j = np.where(np.triu(cos_mat > float(near_cos_threshold), k=1))
        hit_nr_this = False
        for ia, ib in zip(rows_i.tolist(), cols_j.tolist()):
            if pids[ia] == pids[ib] or dgs_ex[ia] == dgs_ex[ib]:
                continue
            hit_nr_this = True
            du = abs(float(util[ia] - util[ib]))
            nr_u_diffs.append(du)
            if gate[ia] and gate[ib]:
                nr_bp_diffs.append(du)
            if gate[ia] != gate[ib]:
                nr_gate_mis += 1
        if hit_nr_this:
            nr_ctx_hit += 1

    champ_summary: dict[str, Any] = {"n_champions_matched": len(champ_vecs)}
    if len(champ_vecs) >= 2:
        c_pids = sorted(champ_vecs.keys())
        c_mat = np.stack([champ_vecs[p] for p in c_pids], axis=0)
        c_dgs = _row_digests_f32(c_mat, quant_decimals=None)
        c_map = dict(zip(c_pids, c_dgs, strict=True))
        c_norm = c_mat.astype(np.float64) / np.maximum(
            np.linalg.norm(c_mat.astype(np.float64), axis=1, keepdims=True), 1e-12
        )
        c_cos = c_norm @ c_norm.T
        ci, cj = np.where(np.triu(c_cos > float(near_cos_threshold), k=1))
        near_champ_pairs = sum(1 for a, b in zip(ci.tolist(), cj.tolist()) if c_dgs[a] != c_dgs[b])
        seeded_ops = ("rff_sin", "rff_cos", "rand_proj", "oblique_hinge")
        exact_seed_caused = 0
        exact_champ_pairs = 0
        for a in range(len(c_pids)):
            for b in range(a + 1, len(c_pids)):
                if c_dgs[a] == c_dgs[b]:
                    exact_champ_pairs += 1
                    dsl_a = champ_want.get(c_pids[a], "")
                    dsl_b = champ_want.get(c_pids[b], "")
                    if any(op in dsl_a or op in dsl_b for op in seeded_ops):
                        exact_seed_caused += 1
        champ_summary.update(
            {
                "exact_prog412": _summarize_bucket_collisions(c_map),
                "near_collision_pairs_cos_gt_0p9999": int(near_champ_pairs),
                "exact_pairs_with_seeded_random_ops": int(exact_seed_caused),
                "exact_pairs_total": int(exact_champ_pairs),
            }
        )

    n_ctx = len(ordered_cids)
    exact_global = _summarize_bucket_collisions(pid_exact_dg)
    return {
        "n_contexts_scanned": int(n_ctx),
        "n_total_evaluations": int(total_evals),
        "n_distinct_program_sha256": int(len(pid_exact_dg)),
        "sha256_to_prog412_deterministic": hash_inconsistencies == 0,
        "cross_pool_hash_inconsistencies": int(hash_inconsistencies),
        "global_exact_prog412": exact_global,
        "global_quantized_1e4_prog412": _summarize_bucket_collisions(pid_q4_dg),
        "global_sem24_signature": _summarize_bucket_collisions(pid_sem_dg),
        "within_pool_exact_prog412_disagreement": _disagreement_stats(
            ex_u_diffs, ex_bp_diffs, ex_gate_mis, ex_ctx_hit, n_ctx
        ),
        "within_pool_near_cos_gt_0p9999_disagreement": _disagreement_stats(
            nr_u_diffs, nr_bp_diffs, nr_gate_mis, nr_ctx_hit, n_ctx
        ),
        "within_pool_sem24_exact_disagreement": _disagreement_stats(
            sm_u_diffs, sm_bp_diffs, sm_gate_mis, sm_ctx_hit, n_ctx
        ),
        "dl_train_champions_bank_audit": champ_summary,
        "dv03_structural_sufficiency_verdict": (
            "LOSSY_ALIASING_DETECTED"
            if exact_global["colliding_programs"] > 0 or len(nr_u_diffs) > 0
            else "EXACTLY_BIJECTIVE"
        ),
    }


def select_diverse_structural_anchors(
    sem24: np.ndarray,
    pids: Sequence[str],
    k: int = 4,
) -> np.ndarray:
    """Storage-order-invariant selection of `k` diverse structural archetype anchors."""
    s_mat = np.asarray(sem24, dtype=np.float64)
    m_cands = s_mat.shape[0]
    k_eff = min(int(k), m_cands)
    if k_eff <= 0:
        return np.zeros(0, dtype=np.int64)
    norms = np.maximum(np.linalg.norm(s_mat, axis=1, keepdims=True), 1e-12)
    s_unit = s_mat / norms
    selected: list[int] = []
    chosen_mask = np.zeros(m_cands, dtype=bool)

    for proto_idx in range(k_eff):
        proto = ARCHETYPE_PROTOTYPES_4X24[proto_idx % len(ARCHETYPE_PROTOTYPES_4X24)]
        sims = s_unit @ proto
        if selected:
            chosen_vecs = s_unit[np.asarray(selected, dtype=np.int64)]
            max_sim_prev = np.max(s_unit @ chosen_vecs.T, axis=1)
            scores = sims - 0.35 * max_sim_prev
        else:
            scores = sims
        # Quantize slightly to 1e-9 and break ties strictly by lexicographic program_sha256,
        # ensuring 100% invariance to storage order permutations.
        best_j = min(
            (j for j in range(m_cands) if not chosen_mask[j]),
            key=lambda j: (-round(float(scores[j]), 9), str(pids[j])),
        )
        selected.append(best_j)
        chosen_mask[best_j] = True
    return np.asarray(selected, dtype=np.int64)


def _fit_predict_op_desc(
    x_tr: np.ndarray,
    y_sem_tr: np.ndarray,
    fams_tr: Sequence[str],
    x_ev: np.ndarray,
    seed: int,
) -> np.ndarray:
    n_tr, d_in = x_tr.shape
    if d_in == 0:
        mean_vec = np.mean(y_sem_tr, axis=0, keepdims=True)
        mean_unit = mean_vec / np.maximum(np.linalg.norm(mean_vec, axis=1, keepdims=True), 1e-9)
        return np.repeat(mean_unit.astype(np.float32), x_ev.shape[0], axis=0)

    mu = np.mean(x_tr.astype(np.float64), axis=0, keepdims=True)
    sd = np.where(np.std(x_tr.astype(np.float64), axis=0, keepdims=True) < 1e-6, 1.0, np.std(x_tr.astype(np.float64), axis=0, keepdims=True))
    z_tr = np.clip((x_tr.astype(np.float64) - mu) / sd, -8.0, 8.0)
    z_ev = np.clip((x_ev.astype(np.float64) - mu) / sd, -8.0, 8.0)

    # Inner family holdout (20%) to select Ridge regularization + small seeded MLP residual blend
    uniq_f = sorted(set(str(f) for f in fams_tr))
    rng = np.random.default_rng(int(seed) + 47301)
    perm_f = [uniq_f[i] for i in rng.permutation(len(uniq_f)).tolist()]
    val_f = set(perm_f[: max(1, len(uniq_f) // 5)])
    in_va = np.array([str(f) in val_f for f in fams_tr], dtype=bool)
    in_tr = ~in_va if not np.all(in_va) else np.ones(n_tr, dtype=bool)

    def _ridge(zx_fit: np.ndarray, zy_fit: np.ndarray, zx_out: np.ndarray, reg: float) -> np.ndarray:
        xa = np.hstack([zx_fit, np.ones((zx_fit.shape[0], 1), dtype=np.float64)])
        xo = np.hstack([zx_out, np.ones((zx_out.shape[0], 1), dtype=np.float64)])
        eye = np.eye(xa.shape[1], dtype=np.float64) * reg
        eye[-1, -1] = 1e-4
        w_mat = np.linalg.solve(xa.T @ xa + eye, xa.T @ zy_fit)
        return xo @ w_mat

    best_reg = 100.0
    best_cos = -2.0
    for reg in (5.0, 25.0, 100.0, 400.0, 1500.0):
        pred_va = _ridge(z_tr[in_tr], y_sem_tr[in_tr], z_tr[in_va], reg)
        pred_u = pred_va / np.maximum(np.linalg.norm(pred_va, axis=1, keepdims=True), 1e-9)
        cos_va = float(np.mean(np.sum(pred_u * y_sem_tr[in_va], axis=1)))
        if cos_va > best_cos:
            best_cos = cos_va
            best_reg = reg

    ridge_full = _ridge(z_tr, y_sem_tr, z_ev, best_reg)
    torch.manual_seed(int(seed) + 9100)
    mlp = nn.Sequential(nn.Linear(d_in, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 24))
    opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=2e-2)
    xt_t = torch.from_numpy(z_tr.astype(np.float32))
    yt_t = torch.from_numpy((y_sem_tr - _ridge(z_tr, y_sem_tr, z_tr, best_reg)).astype(np.float32))
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(mlp(xt_t), yt_t)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        res_ev = mlp(torch.from_numpy(z_ev.astype(np.float32))).numpy().astype(np.float64)
    comb = ridge_full + 0.15 * res_ev
    comb_unit = comb / np.maximum(np.linalg.norm(comb, axis=1, keepdims=True), 1e-9)
    return comb_unit.astype(np.float32)


def _score_i10_cell(
    pred_sem24: np.ndarray,
    eval_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    canon_pids: set[str],
    pid_prior_bonus: Mapping[str, float],
    true_sem_eval: np.ndarray,
    true_head_eval: np.ndarray,
) -> dict[str, np.ndarray]:
    n_ev = len(eval_cids)
    util_arr = np.zeros(n_ev, dtype=np.float64)
    gpass_arr = np.zeros(n_ev, dtype=np.float64)
    cos_arr = np.zeros(n_ev, dtype=np.float64)
    head_ok = np.zeros(n_ev, dtype=np.float64)
    pred_heads = np.argmax(pred_sem24[:, 0:5], axis=1)

    for i, cid in enumerate(eval_cids):
        p = pools[cid]
        pids = p["pids"]
        sem = np.asarray(p["struct_sig"], dtype=np.float64)
        q_vec = pred_sem24[i].astype(np.float64)
        cos_arr[i] = float(np.sum(q_vec * true_sem_eval[i]))
        head_ok[i] = float(int(pred_heads[i]) == int(true_head_eval[i]))

        elig = [j for j, pid in enumerate(pids) if pid in canon_pids]
        if not elig:
            elig = list(range(len(pids)))
        sims = sem[elig] @ q_vec
        bonuses = np.array([pid_prior_bonus.get(pids[j], 0.0) for j in elig], dtype=np.float64)
        comb_s = sims + 0.08 * bonuses
        best_local = min(range(len(elig)), key=lambda idx: (-round(float(comb_s[idx]), 9), str(pids[elig[idx]])))
        pick_j = elig[best_local]
        gp = bool(p["gate"][pick_j])
        gpass_arr[i] = 1.0 if gp else 0.0
        util_arr[i] = -float(p["fitness"][pick_j]) if gp else -2.0

    return {"util": util_arr, "gpass": gpass_arr, "cos": cos_arr, "head_ok": head_ok}


def run_i10_zero_shot_archetype_cv(
    train_cids: Sequence[str],
    train_families: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    arm_features_train: Mapping[str, np.ndarray],
    folds_list: Sequence[np.ndarray],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    contrast_fn: Any,
    smoke: bool = False,
    val_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verify_mem_available_gib(15.0, "i10_op_desc")
    n_tr = len(train_cids)
    true_sem = np.zeros((n_tr, 24), dtype=np.float64)
    true_head = np.zeros(n_tr, dtype=np.int64)
    for i, cid in enumerate(train_cids):
        p = pools[cid]
        u = np.where(p["gate"], -p["fitness"], -2.0)
        top4 = np.argsort(-u)[:4]
        w = np.exp((u[top4] - np.max(u[top4])) / 0.15)
        w = w / np.sum(w)
        avg_s = np.sum(p["struct_sig"][top4].astype(np.float64) * w[:, None], axis=0)
        true_sem[i] = avg_s / max(float(np.linalg.norm(avg_s)), 1e-9)
        best_j = int(top4[0])
        true_head[i] = int(np.argmax(p["struct_sig"][best_j, 0:5]))

    arm_names = ("i10_ultra_packet_859", "i10_dope_663", "i10_desc_290", "i10_none_0d", "i10_ultra_packet_shuffled")
    eval_folds = folds_list[:1] if smoke else folds_list
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)
    n_folds = int(np.max(eval_folds[0])) + 1
    denom = 1.0 if smoke else float(len(eval_folds) * len(eval_seeds))

    acc = {a: {k: np.zeros(n_tr, dtype=np.float64) for k in ("util", "gpass", "cos", "head_ok")} for a in arm_names}

    def _build_canon_bank(cids_sub: Sequence[str]) -> tuple[set[str], dict[str, float]]:
        supp: dict[str, int] = {}
        g_sum: dict[str, int] = {}
        z_sum: dict[str, float] = {}
        for cid in cids_sub:
            p = pools[cid]
            u = np.where(p["gate"], -p["fitness"], -2.0)
            mu_u, sd_u = float(np.mean(u)), max(float(np.std(u)), 1e-6)
            seen: set[str] = set()
            for pid, gi, ui in zip(p["pids"], p["gate"], u):
                if pid in seen:
                    continue
                seen.add(pid)
                supp[pid] = supp.get(pid, 0) + 1
                g_sum[pid] = g_sum.get(pid, 0) + (1 if gi else 0)
                z_sum[pid] = z_sum.get(pid, 0.0) + float((ui - mu_u) / sd_u)
        canon = {
            pid
            for pid, cnt in supp.items()
            if cnt >= 3 and (g_sum[pid] / cnt) >= 0.95 and (z_sum[pid] / cnt) > 0.0
        }
        bonus = {pid: z_sum[pid] / (supp[pid] + 5.0) for pid in canon}
        return canon, bonus

    for fold_arr in eval_folds:
        for k in range(1 if smoke else n_folds):
            ho_idx = np.where(fold_arr == k)[0]
            tr_idx = np.where(fold_arr != k)[0]
            tr_cids_k = [train_cids[i] for i in tr_idx.tolist()]
            ho_cids_k = [train_cids[i] for i in ho_idx.tolist()]
            tr_fams_k = [train_families[i] for i in tr_idx.tolist()]
            ho_fams_k = [train_families[i] for i in ho_idx.tolist()]
            canon_k, bonus_k = _build_canon_bank(tr_cids_k)

            for sd in eval_seeds:
                donor_ho = derange_across_families(ho_fams_k, seed=20261120 + 100 * k + int(sd))
                for arm in arm_names:
                    if arm == "i10_ultra_packet_shuffled":
                        base_x = arm_features_train["i10_ultra_packet_859"]
                        x_ev = base_x[ho_idx][donor_ho]
                    else:
                        base_x = arm_features_train[arm]
                        x_ev = base_x[ho_idx]
                    pred_sem = _fit_predict_op_desc(base_x[tr_idx], true_sem[tr_idx], tr_fams_k, x_ev, seed=int(sd))
                    res = _score_i10_cell(
                        pred_sem, ho_cids_k, pools, canon_k, bonus_k, true_sem[ho_idx], true_head[ho_idx]
                    )
                    for m_k in ("util", "gpass", "cos", "head_ok"):
                        acc[arm][m_k][ho_idx] += res[m_k] / denom

    mask_eval = (eval_folds[0] == 0) if smoke else np.ones(n_tr, dtype=bool)
    fams_eval = [train_families[i] for i in np.where(mask_eval)[0].tolist()]
    arms_out: dict[str, Any] = {}
    for arm in arm_names:
        u_v = acc[arm]["util"][mask_eval]
        arms_out[arm] = {
            "family_mean_utility": round(family_unweighted_mean(u_v, fams_eval), 6),
            "ctx_mean_utility": round(float(np.mean(u_v)), 6),
            "gate_pass_rate": round(float(np.mean(acc[arm]["gpass"][mask_eval])), 6),
            "op_desc_cosine_mean": round(float(np.mean(acc[arm]["cos"][mask_eval])), 6),
            "head5_accuracy": round(float(np.mean(acc[arm]["head_ok"][mask_eval])), 6),
        }

    contrasts = {
        "ultra_packet_minus_dope663": contrast_fn(
            acc["i10_ultra_packet_859"]["util"][mask_eval], acc["i10_dope_663"]["util"][mask_eval], fams_eval
        ),
        "ultra_packet_minus_none0d": contrast_fn(
            acc["i10_ultra_packet_859"]["util"][mask_eval], acc["i10_none_0d"]["util"][mask_eval], fams_eval
        ),
        "ultra_packet_minus_shuffled": contrast_fn(
            acc["i10_ultra_packet_859"]["util"][mask_eval], acc["i10_ultra_packet_shuffled"]["util"][mask_eval], fams_eval
        ),
    }

    val_out: dict[str, Any] | None = None
    if val_bundle is not None and not smoke:
        v_cids = val_bundle["val_cids"]
        v_fams = val_bundle["val_families"]
        v_arms_x = val_bundle["arm_features_val"]
        canon_full, bonus_full = _build_canon_bank(train_cids)
        v_true_sem = np.zeros((len(v_cids), 24), dtype=np.float64)
        v_true_head = np.zeros(len(v_cids), dtype=np.int64)
        for i, cid in enumerate(v_cids):
            p = pools[cid]
            u = np.where(p["gate"], -p["fitness"], -2.0)
            b_j = int(np.argmax(u))
            v_true_sem[i] = p["struct_sig"][b_j]
            v_true_head[i] = int(np.argmax(p["struct_sig"][b_j, 0:5]))
        v_acc = {a: {k: np.zeros(len(v_cids), dtype=np.float64) for k in ("util", "gpass", "cos", "head_ok")} for a in arm_names}
        for sd in eval_seeds:
            donor_v = derange_across_families(v_fams, seed=20261199 + int(sd))
            for arm in arm_names:
                bx_tr = arm_features_train["i10_ultra_packet_859" if arm == "i10_ultra_packet_shuffled" else arm]
                bx_ev = v_arms_x["i10_ultra_packet_859"][donor_v] if arm == "i10_ultra_packet_shuffled" else v_arms_x[arm]
                pred_v = _fit_predict_op_desc(bx_tr, true_sem, train_families, bx_ev, seed=int(sd))
                rv = _score_i10_cell(pred_v, v_cids, pools, canon_full, bonus_full, v_true_sem, v_true_head)
                for m_k in ("util", "gpass", "cos", "head_ok"):
                    v_acc[arm][m_k] += rv[m_k] / float(len(eval_seeds))
        val_out = {
            "arms": {
                arm: {
                    "family_mean_utility": round(family_unweighted_mean(v_acc[arm]["util"], v_fams), 6),
                    "ctx_mean_utility": round(float(np.mean(v_acc[arm]["util"])), 6),
                    "gate_pass_rate": round(float(np.mean(v_acc[arm]["gpass"])), 6),
                    "op_desc_cosine_mean": round(float(np.mean(v_acc[arm]["cos"])), 6),
                    "head5_accuracy": round(float(np.mean(v_acc[arm]["head_ok"])), 6),
                }
                for arm in arm_names
            },
            "contrasts": {
                "ultra_packet_minus_dope663": contrast_fn(
                    v_acc["i10_ultra_packet_859"]["util"], v_acc["i10_dope_663"]["util"], v_fams
                ),
                "ultra_packet_minus_none0d": contrast_fn(
                    v_acc["i10_ultra_packet_859"]["util"], v_acc["i10_none_0d"]["util"], v_fams
                ),
                "ultra_packet_minus_shuffled": contrast_fn(
                    v_acc["i10_ultra_packet_859"]["util"], v_acc["i10_ultra_packet_shuffled"]["util"], v_fams
                ),
            },
        }

    return {"arms": arms_out, "contrasts": contrasts, "val91": val_out}
