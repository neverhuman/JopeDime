"""H279 bagged strict-margin switching on top of frozen RERANK8_VETO (topk.py, h265_core.py, veto.py byte-identical).

Arms implemented per cell:
- RERANK8_VETO (REF): byte-identical veto.run_cell_veto (tau = 0.10, K = 8).
- BAG_SWITCH (A): B = 8 family-level bootstrap-bagged rankers (resampling TRAIN families only with replacement) fit at
  the cell's selected (lr*, n_estimators*). Switches from r882 (slot 0) to RERANK8_VETO's pick j_ref only if >= q = 6
  of 8 bagged rankers rank j_ref above slot 0 by the cell's strict H265 margin (bag_score[b, j_ref] - bag_score[b, 0] > margin).
- BAG_SWITCH_NESTED (B): same outer bagged ensemble, with q selected per outer cell by 4-fold inner family-grouped CV
  over Q_GRID = (5, 6, 7) on TRAIN families only (never touching outer ASSESS labels).
- PLACEBO (BAG_SWITCH_SHUF_PLACEBO): BAG_SWITCH (q = 6, B = 8) where both the candidate proposer and all B = 8 bagged
  agreement voters are fit on within-family shuffled relevance labels (topk.shuffle_within_family on TRAIN families only)."""

import math

import numpy as np

import h265_core as core
import topk
import veto

N_BAGS = 8
Q_FIXED = 6
Q_GRID = (5, 6, 7)
HARM_001 = 0.01


def fit_ranker_n(X, y, n_groups, k, lr, n_est, random_state):
    """Exact topk.fit_ranker configuration stopped at n_estimators = int(n_est) (verified bit-identical to
    fitting 200 trees and predicting with num_iteration = int(n_est))."""
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
        n_estimators=int(n_est),
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


def vote_counts(bag_scores, base_picks, margin):
    """Count how many of B bagged score matrices (B, N, K) rank base_picks[i] above slot 0 by > margin."""
    n_ctx = len(base_picks)
    votes = np.zeros(n_ctx, dtype=np.int32)
    if math.isinf(margin):
        return votes
    for i, j in enumerate(base_picks):
        j_int = int(j)
        if j_int != 0:
            votes[i] = int(
                np.sum((bag_scores[:, i, j_int] - bag_scores[:, i, 0]) > margin)
            )
    return votes


def switch_by_votes(base_picks, votes, q):
    """Switch from slot 0 to base_picks[i] iff votes[i] >= q."""
    return [
        int(j) if int(j) != 0 and int(v) >= int(q) else 0
        for j, v in zip(base_picks, votes)
    ]


def select_q_inner(d_ref, d_by_q):
    """Outcome-blind to ASSESS: select q* in Q_GRID = (5, 6, 7) from inner OOF family-delta vectors on TRAIN families.
    Tier 1: q satisfying both inner harm reduction (h_q <= 0.75 * h_ref) and inner gain retention (g_q >= 0.80 * g_ref),
            maximising g_q - 0.001 * h_q (tie-break toward larger q).
    Tier 2: q satisfying inner harm reduction (h_q <= 0.75 * h_ref) and g_q > 0.010, maximising g_q (tie-break larger q).
    Tier 3: fallback over Q_GRID maximising g_q - 0.0015 * h_q (tie-break toward larger q)."""
    g_ref = float(np.mean(d_ref))
    h_ref = int((d_ref < -HARM_001).sum())
    stats = {}
    for q in Q_GRID:
        dq = d_by_q[q]
        stats[q] = dict(
            gain=float(np.mean(dq)),
            harmed_001=int((dq < -HARM_001).sum()),
        )
    tier1 = [
        q
        for q in Q_GRID
        if stats[q]["harmed_001"] <= 0.75 * h_ref + 1e-12
        and stats[q]["gain"] >= 0.80 * g_ref - 1e-12
    ]
    if tier1:
        q_star = max(
            tier1,
            key=lambda q: (
                round(stats[q]["gain"] - 0.001 * stats[q]["harmed_001"], 12),
                q,
            ),
        )
        tier = "tier1_both"
    else:
        tier2 = [
            q
            for q in Q_GRID
            if stats[q]["harmed_001"] <= 0.75 * h_ref + 1e-12
            and stats[q]["gain"] > 0.010
        ]
        if tier2:
            q_star = max(tier2, key=lambda q: (round(stats[q]["gain"], 12), q))
            tier = "tier2_harm"
        else:
            q_star = max(
                Q_GRID,
                key=lambda q: (
                    round(stats[q]["gain"] - 0.0015 * stats[q]["harmed_001"], 12),
                    q,
                ),
            )
            tier = "tier3_tradeoff"
    return int(q_star), tier, dict(ref_gain=g_ref, ref_harmed_001=h_ref, by_q=stats)


def run_cell_bagged(
    seed,
    assess_fold,
    cids_by_fold,
    packets,
    fam_by_cid,
    k,
    salt,
    store_models=False,
):
    """Run RERANK8_VETO (REF), BAG_SWITCH (q=6), BAG_SWITCH_NESTED (q in {5,6,7}), and PLACEBO on one cell."""

    def ss(*parts):
        return core.stream_seed(*parts, *([salt] if salt else []))

    rv = veto.run_cell_veto(
        seed, assess_fold, cids_by_fold, packets, fam_by_cid, k, salt
    )
    best_lr = float(rv["receipt"]["lr"])
    best_n = int(rv["receipt"]["n_estimators"])
    raw_margin = float(rv["receipt"]["margin"])
    strict_margin = rv["margin"]
    admitted = bool(rv["receipt"]["strict_admitted"])

    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    cid_pos = {c: i for i, c in enumerate(fit_cids)}
    f2c = {}
    for c in fit_cids:
        f2c.setdefault(fam_by_cid[c], []).append(c)

    # 1. Inner 4-fold family-grouped CV over Q_GRID = (5, 6, 7) on TRAIN families only
    oof_ref_pick = np.zeros(len(fit_cids), dtype=np.int32)
    oof_votes = np.zeros(len(fit_cids), dtype=np.int32)
    for hold in fit_folds:
        core.check_cooperative_pause()
        tr = [c for f in fit_folds if f != hold for c in cids_by_fold[f]]
        va = cids_by_fold[hold]
        X_tr = np.concatenate([packets[c]["X"] for c in tr])
        y_tr = np.concatenate([packets[c]["rel_k"] for c in tr])
        X_va = np.concatenate([packets[c]["X"] for c in va])
        inv_tr = np.concatenate([~packets[c]["gate_k"] for c in tr])
        g_in = veto.fit_gate(
            X_tr, inv_tr, ss(seed, assess_fold, "RERANK8_VETO", "gate", hold)
        )
        pv = veto.gate_probs(g_in, X_va, len(va), k)
        m_in = fit_ranker_n(
            X_tr,
            y_tr,
            len(tr),
            k,
            best_lr,
            best_n,
            ss(seed, assess_fold, "RERANK8", "inner", hold, best_lr),
        )
        p_in = veto.mask(
            np.asarray(
                m_in.booster_.predict(X_va, num_iteration=best_n), dtype=np.float64
            ).reshape(len(va), k),
            pv,
        )
        j_va = [
            core.select_from_top4_scores(p_in[i], raw_margin) for i in range(len(va))
        ]
        for i, c in enumerate(va):
            oof_ref_pick[cid_pos[c]] = j_va[i]

        tr_f2c = {}
        for c in tr:
            tr_f2c.setdefault(fam_by_cid[c], []).append(c)
        tr_fams = sorted(tr_f2c)
        va_bag_scores = []
        for b in range(N_BAGS):
            rng = np.random.default_rng(
                ss(seed, assess_fold, "BAG_SWITCH", "inner_boot", hold, b)
            )
            bfams = rng.choice(tr_fams, size=len(tr_fams), replace=True)
            bcids = [c for f in bfams for c in tr_f2c[f]]
            Xb = np.concatenate([packets[c]["X"] for c in bcids])
            yb = np.concatenate([packets[c]["rel_k"] for c in bcids])
            mb = fit_ranker_n(
                Xb,
                yb,
                len(bcids),
                k,
                best_lr,
                best_n,
                ss(seed, assess_fold, "BAG_SWITCH", "inner_bag", hold, b),
            )
            sb = np.asarray(
                mb.booster_.predict(X_va, num_iteration=best_n), dtype=np.float64
            ).reshape(len(va), k)
            va_bag_scores.append(sb)
        va_votes = vote_counts(np.stack(va_bag_scores, axis=0), j_va, raw_margin)
        for i, c in enumerate(va):
            oof_votes[cid_pos[c]] = va_votes[i]

    r882_u = np.array([packets[c]["r882_u"] for c in fit_cids])
    fams_arr = np.array([fam_by_cid[c] for c in fit_cids])
    masks = [fams_arr == f for f in fit_fams]
    r882_fm = np.array([r882_u[m].mean() for m in masks])
    u_ref_oof = np.array(
        [packets[c]["utility_k"][j] for c, j in zip(fit_cids, oof_ref_pick)]
    )
    d_ref = np.array([u_ref_oof[m].mean() for m in masks]) - r882_fm
    d_by_q = {}
    for q in Q_GRID:
        jq = switch_by_votes(oof_ref_pick, oof_votes, q)
        uq = np.array([packets[c]["utility_k"][j] for c, j in zip(fit_cids, jq)])
        d_by_q[q] = np.array([uq[m].mean() for m in masks]) - r882_fm
    q_nested, q_tier, inner_q_summary = select_q_inner(d_ref, d_by_q)

    # 2. Outer B = 8 family-bootstrapped rankers on TRAIN families only
    core.check_cooperative_pause()
    assess = cids_by_fold[assess_fold]
    X_a = np.concatenate([packets[c]["X"] for c in assess])
    bag_models = []
    bag_scores = []
    for b in range(N_BAGS):
        rng = np.random.default_rng(
            ss(seed, assess_fold, "BAG_SWITCH", "outer_boot", b)
        )
        bfams = rng.choice(fit_fams, size=len(fit_fams), replace=True)
        bcids = [c for f in bfams for c in f2c[f]]
        Xb = np.concatenate([packets[c]["X"] for c in bcids])
        yb = np.concatenate([packets[c]["rel_k"] for c in bcids])
        mb = fit_ranker_n(
            Xb,
            yb,
            len(bcids),
            k,
            best_lr,
            best_n,
            ss(seed, assess_fold, "BAG_SWITCH", "outer_bag", b),
        )
        if store_models:
            bag_models.append(mb.booster_.model_to_string())
        sb = np.asarray(
            mb.booster_.predict(X_a, num_iteration=best_n), dtype=np.float64
        ).reshape(len(assess), k)
        bag_scores.append(sb)
    bag_scores_arr = np.stack(bag_scores, axis=0)
    j_ref = rv["assess_picks"]
    outer_votes = vote_counts(bag_scores_arr, j_ref, strict_margin)
    picks_bag = switch_by_votes(j_ref, outer_votes, Q_FIXED)
    picks_nested = switch_by_votes(j_ref, outer_votes, q_nested)

    # 3. PLACEBO: train-only within-family label-shuffled proposer + B = 8 shuffled bagged voters at q = 6
    core.check_cooperative_pause()
    rel_orig = {c: packets[c]["rel_k"] for c in fit_cids}
    rel_shuf_main = topk.shuffle_within_family(
        rel_orig, fit_cids, fam_by_cid, ss(seed, assess_fold, "PLACEBO", "shuf_main"), k
    )
    X_fit = np.concatenate([packets[c]["X"] for c in fit_cids])
    y_shuf_main = np.concatenate([rel_shuf_main[c] for c in fit_cids])
    m_shuf = fit_ranker_n(
        X_fit,
        y_shuf_main,
        len(fit_cids),
        k,
        best_lr,
        best_n,
        ss(seed, assess_fold, "PLACEBO", "outer_main"),
    )
    import lightgbm as lgb

    p_inv_a = np.asarray(
        lgb.Booster(model_str=rv["gate"]).predict(X_a), dtype=np.float64
    ).reshape(len(assess), k)
    sc_shuf = veto.mask(
        np.asarray(
            m_shuf.booster_.predict(X_a, num_iteration=best_n), dtype=np.float64
        ).reshape(len(assess), k),
        p_inv_a,
    )
    j_shuf = [
        core.select_from_top4_scores(sc_shuf[i], raw_margin) for i in range(len(assess))
    ]
    shuf_bag_models = []
    shuf_bag_scores = []
    for b in range(N_BAGS):
        rng = np.random.default_rng(ss(seed, assess_fold, "PLACEBO", "outer_boot", b))
        bfams = rng.choice(fit_fams, size=len(fit_fams), replace=True)
        bcids = [c for f in bfams for c in f2c[f]]
        rel_shuf_b = topk.shuffle_within_family(
            rel_orig,
            fit_cids,
            fam_by_cid,
            ss(seed, assess_fold, "PLACEBO", "shuf_bag", b),
            k,
        )
        Xb = np.concatenate([packets[c]["X"] for c in bcids])
        yb = np.concatenate([rel_shuf_b[c] for c in bcids])
        mb_s = fit_ranker_n(
            Xb,
            yb,
            len(bcids),
            k,
            best_lr,
            best_n,
            ss(seed, assess_fold, "PLACEBO", "outer_bag", b),
        )
        if store_models:
            shuf_bag_models.append(mb_s.booster_.model_to_string())
        sb_s = np.asarray(
            mb_s.booster_.predict(X_a, num_iteration=best_n), dtype=np.float64
        ).reshape(len(assess), k)
        shuf_bag_scores.append(sb_s)
    shuf_votes = vote_counts(np.stack(shuf_bag_scores, axis=0), j_shuf, raw_margin)
    picks_placebo = switch_by_votes(j_shuf, shuf_votes, Q_FIXED)

    receipts = {
        "RERANK8_VETO": rv["receipt"],
        "BAG_SWITCH": dict(
            arm="BAG_SWITCH",
            k=k,
            n_bags=N_BAGS,
            q=Q_FIXED,
            lr=best_lr,
            n_estimators=best_n,
            margin=raw_margin,
            strict_admitted=admitted,
            strict_margin="inf" if math.isinf(strict_margin) else float(strict_margin),
            inner_fit_family_gain=float(d_by_q[Q_FIXED].mean()),
            inner_fit_harmed_001=int((d_by_q[Q_FIXED] < -HARM_001).sum()),
        ),
        "BAG_SWITCH_NESTED": dict(
            arm="BAG_SWITCH_NESTED",
            k=k,
            n_bags=N_BAGS,
            q_grid=list(Q_GRID),
            q_selected=q_nested,
            q_selection_tier=q_tier,
            lr=best_lr,
            n_estimators=best_n,
            margin=raw_margin,
            strict_admitted=admitted,
            strict_margin="inf" if math.isinf(strict_margin) else float(strict_margin),
            inner_q_summary=inner_q_summary,
        ),
        "PLACEBO": dict(
            arm="PLACEBO",
            k=k,
            n_bags=N_BAGS,
            q=Q_FIXED,
            lr=best_lr,
            n_estimators=best_n,
            margin=raw_margin,
            strict_admitted=True,
            strict_margin=raw_margin,
        ),
    }
    out = dict(
        receipts=receipts,
        assess_picks={
            "r882": [0] * len(assess),
            "RERANK8_VETO": j_ref,
            "BAG_SWITCH": picks_bag,
            "BAG_SWITCH_NESTED": picks_nested,
            "PLACEBO": picks_placebo,
        },
        outer_votes=outer_votes.tolist(),
        shuf_votes=shuf_votes.tolist(),
    )
    if store_models:
        out["models"] = dict(
            rv_model=rv["model"],
            rv_gate=rv["gate"],
            n_estimators=best_n,
            raw_margin=raw_margin,
            strict_margin=strict_margin,
            q_nested=q_nested,
            bag_models=bag_models,
            shuf_model=m_shuf.booster_.model_to_string(),
            shuf_bag_models=shuf_bag_models,
        )
    return out


def score_dlval_cell(models, X_val, n_ctx, k, core_mod):
    """Score DL-Val91 contexts using a locked cell's outer models (only called AFTER selection.json is fsync-locked)."""
    import lightgbm as lgb

    n_est = int(models["n_estimators"])
    raw_margin = float(models["raw_margin"])
    strict_margin = models["strict_margin"]
    q_nested = int(models["q_nested"])

    p_inv = np.asarray(
        lgb.Booster(model_str=models["rv_gate"]).predict(X_val), dtype=np.float64
    ).reshape(n_ctx, k)
    sc_rv = veto.mask(
        np.asarray(
            lgb.Booster(model_str=models["rv_model"]).predict(
                X_val, num_iteration=n_est
            ),
            dtype=np.float64,
        ).reshape(n_ctx, k),
        p_inv,
    )
    j_ref = [
        core_mod.select_from_top4_scores(sc_rv[i], strict_margin) for i in range(n_ctx)
    ]

    bag_scores = np.stack(
        [
            np.asarray(
                lgb.Booster(model_str=m_str).predict(X_val, num_iteration=n_est),
                dtype=np.float64,
            ).reshape(n_ctx, k)
            for m_str in models["bag_models"]
        ],
        axis=0,
    )
    votes = vote_counts(bag_scores, j_ref, strict_margin)
    picks_bag = switch_by_votes(j_ref, votes, Q_FIXED)
    picks_nested = switch_by_votes(j_ref, votes, q_nested)

    sc_shuf = veto.mask(
        np.asarray(
            lgb.Booster(model_str=models["shuf_model"]).predict(
                X_val, num_iteration=n_est
            ),
            dtype=np.float64,
        ).reshape(n_ctx, k),
        p_inv,
    )
    j_shuf = [
        core_mod.select_from_top4_scores(sc_shuf[i], raw_margin) for i in range(n_ctx)
    ]
    shuf_bag_scores = np.stack(
        [
            np.asarray(
                lgb.Booster(model_str=m_str).predict(X_val, num_iteration=n_est),
                dtype=np.float64,
            ).reshape(n_ctx, k)
            for m_str in models["shuf_bag_models"]
        ],
        axis=0,
    )
    shuf_votes = vote_counts(shuf_bag_scores, j_shuf, raw_margin)
    picks_placebo = switch_by_votes(j_shuf, shuf_votes, Q_FIXED)

    return {
        "r882": [0] * n_ctx,
        "RERANK8_VETO": j_ref,
        "BAG_SWITCH": picks_bag,
        "BAG_SWITCH_NESTED": picks_nested,
        "PLACEBO": picks_placebo,
    }
