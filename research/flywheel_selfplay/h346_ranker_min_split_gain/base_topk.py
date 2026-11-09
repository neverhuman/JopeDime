"""K-generic versions of the frozen H265 top-4 pipeline (h265_core.py) for H270. Only K changes: the slice is r882's
K lowest-cost candidates; per-context z-features are computed within the K-slice; the per-pair raw/z cost differences
cover the other K-1 slots; slot position is j / (K-1); the slice spread is (c[K-1] - c[0]) / sd; labels are K-1..0 for
valid candidates by ascending fitness (gate-failed = 0, worst); the LGBMRanker groups and lambdarank truncation level are
K. Fixed context descriptors (the top-8 sem24 centroid and pool rows 0..3) are unchanged. At K = 4 every function here
reproduces h265_core exactly (tested). Selection, the grid and the strict selector are identical (hw_t1830)."""

import math

import numpy as np

import h265_core as core


def topk_relevance(fitness_k, gate_k):
    k = len(fitness_k)
    rel = np.zeros(k, dtype=np.int32)
    valid = [j for j in range(k) if bool(gate_k[j]) and np.isfinite(fitness_k[j])]
    for pos, j in enumerate(sorted(valid, key=lambda j: (float(fitness_k[j]), int(j)))):
        rel[j] = max(k - 1 - pos, 0)
    return rel


def build_topk(rcost, aprior, prog_topk, sem24_full, proj16_full, lm48_z, dope629, k):
    rcost64 = rcost.astype(np.float64)
    topk = np.argsort(rcost64, kind="mergesort")[:k]
    top8 = np.argsort(rcost64, kind="mergesort")[:8]
    r_mean = float(np.mean(rcost64))
    r_sd = max(float(np.std(rcost64)), 1e-12)
    r_z_all = (rcost64 - r_mean) / r_sd
    ck = rcost64[topk]
    ck_mean = float(np.mean(ck))
    ck_sd = max(float(np.std(ck)), 1e-6)
    ck_z = (ck - ck_mean) / ck_sd
    zk = r_z_all[topk]
    sm_tau1 = np.exp(-(zk - np.min(zk)))
    sm_tau1 /= np.sum(sm_tau1)
    sm_tau02 = np.exp(-(zk - np.min(zk)) / 0.2)
    sm_tau02 /= np.sum(sm_tau02)
    sm_ck = np.exp(-(ck_z - np.min(ck_z)))
    sm_ck /= np.sum(sm_ck)
    ap64 = aprior.astype(np.float64)
    ap_z_all = (ap64 - float(np.mean(ap64))) / max(float(np.std(ap64)), 1e-6)
    ctx_top2_zgap = float((ck[1] - ck[0]) / r_sd)
    ctx_spread_z = float((ck[k - 1] - ck[0]) / r_sd)
    sem64 = sem24_full.astype(np.float64)
    sem_z = np.clip(
        (sem64[topk] - sem64.mean(axis=0, keepdims=True))
        / (sem64.std(axis=0, keepdims=True) + 1e-5),
        -5.0,
        5.0,
    ).astype(np.float32)
    proj64 = proj16_full.astype(np.float64)
    proj_z = np.clip(
        (proj64[topk] - proj64.mean(axis=0, keepdims=True))
        / (proj64.std(axis=0, keepdims=True) + 1e-5),
        -5.0,
        5.0,
    ).astype(np.float32)
    proj_unit = proj_z / np.maximum(np.linalg.norm(proj_z, axis=1, keepdims=True), 1e-6)
    sem_topk = sem64[topk]
    centroid = sem64[top8].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
    sem_rev03 = sem64[:4]
    est_slice = prog_topk[:, 24:29]
    est_ids = np.where(
        est_slice.max(axis=1) > 0, np.argmax(est_slice, axis=1), -1
    ).astype(np.int32)
    op_counts = prog_topk[:, :23].sum(axis=1).astype(np.float64)
    d629 = dope629.astype(np.float32)
    d_summary = np.array(
        [
            float(np.mean(d629)),
            float(np.std(d629)),
            float(np.linalg.norm(d629) / math.sqrt(len(d629))),
        ],
        dtype=np.float32,
    )
    ctx_vec76 = np.concatenate(
        [lm48_z.astype(np.float32), d629[:25], d_summary], axis=0
    )
    width = 14 + 2 * (k - 1) + 12 + prog_topk.shape[1] + 24 + 16 + 76
    rows_out = np.zeros((k, width), dtype=np.float32)
    for j in range(k):
        others = [o for o in range(k) if o != j]
        block1 = np.array(
            [
                float(ck[j]),
                float(zk[j]),
                float(ck_z[j]),
                float(j),
                float(j) / float(k - 1),
                1.0 if j == 0 else 0.0,
                float(ck[j] - ck[0]),
                0.0 if j == 0 else float((ck[j] - ck[0]) / r_sd),
                ctx_top2_zgap,
                ctx_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_ck[j]),
                float(ap_z_all[topk[j]]),
                *[float(ck[j] - ck[o]) for o in others],
                *[float((ck[j] - ck[o]) / r_sd) for o in others],
            ],
            dtype=np.float32,
        )
        cos_sem_other = [
            float(np.clip(np.dot(sem_topk[j], sem_topk[o]), -1.0, 1.0)) for o in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit[j], proj_unit[o]), -1.0, 1.0))
            for o in others
        ]
        block2 = np.array(
            [
                float(np.clip(np.dot(sem_topk[j], sem_topk[0]), -1.0, 1.0)),
                float(np.mean(cos_sem_other)),
                float(np.max(cos_sem_other)),
                float(np.clip(np.dot(proj_unit[j], proj_unit[0]), -1.0, 1.0)),
                float(np.mean(cos_proj_other)),
                float(np.max(cos_proj_other)),
                float(np.clip(np.dot(sem_topk[j], centroid), -1.0, 1.0)),
                float(np.mean(np.clip(sem_rev03 @ sem_topk[j], -1.0, 1.0))),
                float(op_counts[j]),
                float(op_counts[j] - op_counts[0]),
                1.0 if est_ids[j] == est_ids[0] else 0.0,
                float(sum(1 for o in range(k) if est_ids[o] == est_ids[j])),
            ],
            dtype=np.float32,
        )
        rows_out[j] = np.concatenate(
            [
                block1,
                block2,
                prog_topk[j].astype(np.float32),
                sem_z[j],
                proj_z[j],
                ctx_vec76,
            ],
            axis=0,
        )
    return topk, rows_out


def load_packets(
    rows, costs_by_cid, aprior_by_cid, proj16_by_cid, ctx_feats_by_cid, k, batch_size=50
):
    """The H265 packet loader with K: features, outcomes and labels of r882's K lowest-cost candidates plus the
    descriptive oracles (best gated cost in the K-slice, in the top-4 and in the whole pool)."""
    packets = {}
    for start in range(0, len(rows), batch_size):
        core.check_cooperative_pause()
        for r in rows[start : start + batch_size]:
            cid = r["context_sha256"]
            with np.load(
                core.INPUT_POOLS_DIR / "pools" / f"{cid}.npz", allow_pickle=False
            ) as z:
                fitness = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(np.bool_)
                prog = z["program"].astype(np.float32)
            sem24 = core.compute_sem24_numpy(prog)
            utility = np.where(gate, -fitness, -2.0)
            rcost = costs_by_cid[cid]
            order = np.argsort(rcost, kind="mergesort")
            topk, x = build_topk(
                rcost,
                aprior_by_cid[cid],
                prog[order[:k]],
                sem24,
                proj16_by_cid[cid],
                ctx_feats_by_cid[cid]["lm48_z"],
                ctx_feats_by_cid[cid]["dope629"],
                k,
            )
            gated = np.where(gate, fitness, 2.0)
            packets[cid] = {
                "family_id": r["family_id"],
                "X": x,
                "utility_k": utility[topk],
                "gate_k": gate[topk],
                "rel_k": topk_relevance(fitness[topk], gate[topk]),
                "r882_u": float(utility[topk[0]]),
                "oracle_top4_u": float(
                    utility[order[:4]][int(np.argmin(gated[order[:4]]))]
                ),
                "oracle_top8_u": float(
                    utility[order[:8]][int(np.argmin(gated[order[:8]]))]
                ),
                "oracle_pool_u": float(utility[int(np.argmin(gated))]),
                "invalid_in_slice": int((~gate[topk]).sum()),
            }
        core.trim_malloc()
    return packets


def fit_ranker(X, y, n_groups, k, lr, random_state):
    import lightgbm as lgb

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        lambdarank_truncation_level=k,
        boosting_type="gbdt",
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        learning_rate=float(lr),
        n_estimators=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=int(random_state),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(X, y, group=np.full(n_groups, k, dtype=np.int32), eval_at=[1, k])
    return model


def shuffle_within_family(rel_by_cid, cids, fam_by_cid, seed, k):
    out, by_fam = {}, {}
    for c in cids:
        by_fam.setdefault(fam_by_cid[c], []).append(c)
    for fam in sorted(by_fam):
        fcids = sorted(by_fam[fam])
        flat = np.concatenate([rel_by_cid[c] for c in fcids], axis=0).copy()
        perm = np.random.default_rng(
            core.stream_seed(seed, "shuf_fam", fam)
        ).permutation(len(flat))
        flat = flat[perm]
        for i, c in enumerate(fcids):
            out[c] = flat[i * k : (i + 1) * k].copy()
    return out


def run_cell(
    seed, assess_fold, arm, shuffle, cids_by_fold, packets, fam_by_cid, k, salt=""
):
    """H265 run_cell_fit_and_select with K: 4-fold inner CV over the FIT folds on the (lr, n_est, margin) grid, the
    strict selector (gain > .001 and LCB95 > 0 over FIT families, else margin = inf = r882), the outer model at lr*
    and the ASSESS picks. `salt` is mixed into every random stream (H270 runs several fold seeds per model seed)."""

    def ss(*parts):
        return core.stream_seed(
            *parts, *([salt] if salt else [])
        )  # salt-free streams equal H265's exactly

    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    rel_map = {c: packets[c]["rel_k"] for c in fit_cids}
    if shuffle:
        rel_map = shuffle_within_family(
            rel_map, fit_cids, fam_by_cid, ss(seed, assess_fold, "shuf"), k
        )
    oof = {(lr, n): {} for lr in core.LR_GRID for n in core.N_EST_GRID}
    for hold in fit_folds:
        core.check_cooperative_pause()
        tr = [c for f in fit_folds if f != hold for c in cids_by_fold[f]]
        va = cids_by_fold[hold]
        X_tr = np.concatenate([packets[c]["X"] for c in tr])
        y_tr = np.concatenate([rel_map[c] for c in tr])
        X_va = np.concatenate([packets[c]["X"] for c in va])
        for lr in core.LR_GRID:
            m = fit_ranker(
                X_tr,
                y_tr,
                len(tr),
                k,
                lr,
                ss(seed, assess_fold, arm, "inner", hold, lr),
            )
            for n in core.N_EST_GRID:
                p = np.asarray(
                    m.booster_.predict(X_va, num_iteration=int(n)), dtype=np.float64
                ).reshape(len(va), k)
                for i, c in enumerate(va):
                    oof[(lr, n)][c] = p[i]
    r882_u = np.array([packets[c]["r882_u"] for c in fit_cids])
    fams = np.array([fam_by_cid[c] for c in fit_cids])
    masks = [fams == f for f in fit_fams]
    r882_fm = np.array([r882_u[m].mean() for m in masks])
    best_gain, best_cfg, best_d = (
        -1e18,
        (core.LR_GRID[0], core.N_EST_GRID[0], 0.0),
        np.zeros(len(fit_fams)),
    )
    for lr in core.LR_GRID:
        for n in core.N_EST_GRID:
            for margin in core.MARGIN_GRID:
                if math.isinf(margin):
                    continue
                u = np.array(
                    [
                        packets[c]["utility_k"][
                            core.select_from_top4_scores(oof[(lr, n)][c], margin)
                        ]
                        for c in fit_cids
                    ]
                )
                d = np.array([u[m].mean() for m in masks]) - r882_fm
                g = float(d.mean())
                if g > best_gain + 1e-12 or (
                    abs(g - best_gain) <= 1e-12
                    and (margin, -n, -lr) > (best_cfg[2], -best_cfg[1], -best_cfg[0])
                ):
                    best_gain, best_cfg, best_d = g, (lr, n, margin), d
    ix = np.random.default_rng(
        ss(core.BOOT_SEED, "inner_select", seed, assess_fold, arm)
    ).integers(0, len(fit_fams), (core.INNER_BOOT_DRAWS, len(fit_fams)))
    lcb, ucb = np.quantile(best_d[ix].mean(axis=1), [0.025, 0.975]).tolist()
    admitted = bool(best_gain > core.STRICT_FIT_MIN_GAIN and lcb > 0.0)
    margin = best_cfg[2] if admitted else math.inf
    model = fit_ranker(
        np.concatenate([packets[c]["X"] for c in fit_cids]),
        np.concatenate([rel_map[c] for c in fit_cids]),
        len(fit_cids),
        k,
        best_cfg[0],
        ss(seed, assess_fold, arm, "outer", best_cfg[0]),
    )
    assess = cids_by_fold[assess_fold]
    sc = np.asarray(
        model.booster_.predict(
            np.concatenate([packets[c]["X"] for c in assess]),
            num_iteration=int(best_cfg[1]),
        ),
        dtype=np.float64,
    ).reshape(len(assess), k)
    picks = [core.select_from_top4_scores(sc[i], margin) for i in range(len(assess))]
    receipt = dict(
        arm=arm,
        k=k,
        lr=float(best_cfg[0]),
        n_estimators=int(best_cfg[1]),
        margin=float(best_cfg[2]),
        inner_fit_family_gain=float(best_gain),
        inner_fit_ci95=[lcb, ucb],
        strict_admitted=admitted,
        strict_margin="inf" if math.isinf(margin) else float(margin),
    )
    return dict(
        receipt=receipt,
        model=model.booster_.model_to_string(),
        n_estimators=int(best_cfg[1]),
        margin=margin,
        assess_picks=picks,
    )
