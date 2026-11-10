"""RERANK8_VETO for H276: the frozen H272/H270 RERANK8 pipeline (topk.py, unchanged) plus the H273-specified invalid
gate. An LGBMClassifier (H273 README section 2 hyperparameters) is fit on TRAIN candidate rows only, using the same
K = 8 per-candidate feature rows, to predict y_invalid = 1[gate fails]. Any non-baseline slot (k >= 1) with
p_invalid >= tau = 0.10 has its ranker score masked to -inf before the unchanged margin selection, in the inner FIT
selection and on ASSESS / DL-Val91 alike. Inner classifiers are fit on the inner-train folds only (hw_t2030)."""

import math

import numpy as np

import h265_core as core
import topk

TAU = 0.10


def fit_gate(X, invalid, random_state):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=70,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=1,
        random_state=int(random_state),
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
    )
    clf.fit(X, invalid.astype(int))
    return clf


def gate_probs(clf, X, n_ctx, k):
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64).reshape(n_ctx, k)


def mask(scores, p_invalid, tau=TAU):
    out = np.array(scores, dtype=np.float64, copy=True)
    veto = p_invalid >= tau
    veto[:, 0] = False  # the r882 baseline slot is never vetoed
    out[veto] = -np.inf
    return out


def run_cell_veto(seed, assess_fold, cids_by_fold, packets, fam_by_cid, k, salt):
    """topk.run_cell with the H273 veto mask applied before every margin selection (selection grid, strict rule and
    random streams identical to RERANK8, so RERANK8 and RERANK8_VETO differ only by the mask)."""

    def ss(*parts):
        return core.stream_seed(*parts, *([salt] if salt else []))

    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    rel_map = {c: packets[c]["rel_k"] for c in fit_cids}
    oof = {(lr, n): {} for lr in core.LR_GRID for n in core.N_EST_GRID}
    oof_p = {}
    for hold in fit_folds:
        core.check_cooperative_pause()
        tr = [c for f in fit_folds if f != hold for c in cids_by_fold[f]]
        va = cids_by_fold[hold]
        X_tr = np.concatenate([packets[c]["X"] for c in tr])
        y_tr = np.concatenate([rel_map[c] for c in tr])
        X_va = np.concatenate([packets[c]["X"] for c in va])
        inv_tr = np.concatenate([~packets[c]["gate_k"] for c in tr])
        gate = fit_gate(
            X_tr, inv_tr, ss(seed, assess_fold, "RERANK8_VETO", "gate", hold)
        )
        pv = gate_probs(gate, X_va, len(va), k)
        for i, c in enumerate(va):
            oof_p[c] = pv[i]
        for lr in core.LR_GRID:
            # Same ranker stream as RERANK8 (arm label "RERANK8"), so the ranker models are identical to RERANK8's.
            m = topk.fit_ranker(
                X_tr,
                y_tr,
                len(tr),
                k,
                lr,
                ss(seed, assess_fold, "RERANK8", "inner", hold, lr),
            )
            for n in core.N_EST_GRID:
                p = np.asarray(
                    m.booster_.predict(X_va, num_iteration=int(n)), dtype=np.float64
                ).reshape(len(va), k)
                pm = mask(p, pv)
                for i, c in enumerate(va):
                    oof[(lr, n)][c] = pm[i]
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
        ss(core.BOOT_SEED, "inner_select", seed, assess_fold, "RERANK8_VETO")
    ).integers(0, len(fit_fams), (core.INNER_BOOT_DRAWS, len(fit_fams)))
    lcb, ucb = np.quantile(best_d[ix].mean(axis=1), [0.025, 0.975]).tolist()
    admitted = bool(best_gain > core.STRICT_FIT_MIN_GAIN and lcb > 0.0)
    margin = best_cfg[2] if admitted else math.inf
    X_fit = np.concatenate([packets[c]["X"] for c in fit_cids])
    model = topk.fit_ranker(
        X_fit,
        np.concatenate([rel_map[c] for c in fit_cids]),
        len(fit_cids),
        k,
        best_cfg[0],
        ss(seed, assess_fold, "RERANK8", "outer", best_cfg[0]),
    )
    gate = fit_gate(
        X_fit,
        np.concatenate([~packets[c]["gate_k"] for c in fit_cids]),
        ss(seed, assess_fold, "RERANK8_VETO", "gate", "outer"),
    )
    assess = cids_by_fold[assess_fold]
    X_a = np.concatenate([packets[c]["X"] for c in assess])
    sc = mask(
        np.asarray(
            model.booster_.predict(X_a, num_iteration=int(best_cfg[1])),
            dtype=np.float64,
        ).reshape(len(assess), k),
        gate_probs(gate, X_a, len(assess), k),
    )
    picks = [core.select_from_top4_scores(sc[i], margin) for i in range(len(assess))]
    receipt = dict(
        arm="RERANK8_VETO",
        k=k,
        tau=TAU,
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
        gate=gate.booster_.model_to_string(),
        n_estimators=int(best_cfg[1]),
        margin=margin,
        assess_picks=picks,
    )
