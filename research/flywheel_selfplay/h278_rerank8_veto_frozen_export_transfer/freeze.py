"""H278 Part A: fit ONE frozen RERANK8_VETO model per model seed on ALL DL-Train families with the unchanged H276
science (topk.py, h265_core.py, veto.py byte-identical). The hyperparameters come only from the unchanged
family-grouped inner CV over 4 random family folds of DL-Train. freeze_veto is veto.run_cell_veto without an ASSESS
fold: with the same folds and salt it returns the identical receipt, ranker and gate (tested) (hw_t2140)."""

import math

import numpy as np

import h265_core as core
import topk
import veto

SEEDS = (278101, 278102, 278103)


def family_folds(rows, seed, n=4):
    """4 random family-grouped folds of all DL-Train families (stream per model seed); returned as folds 1..n with an
    empty fold 0 so the fold layout matches veto.run_cell_veto's assess_fold = 0 convention."""
    fams = sorted({r["family_id"] for r in rows})
    order = np.random.default_rng(
        core.stream_seed(seed, "h278_family_folds")
    ).permutation(len(fams))
    where = {fams[i]: k + 1 for k, g in enumerate(np.array_split(order, n)) for i in g}
    out = {k: [] for k in range(n + 1)}
    for r in rows:
        out[where[r["family_id"]]].append(r["context_sha256"])
    return out


def freeze_veto(seed, cids_by_fold, packets, fam_by_cid, k=8, salt="h278"):
    """The RERANK8_VETO fit of veto.run_cell_veto (assess_fold 0 is empty) on all FIT folds 1..4."""

    def ss(*parts):
        return core.stream_seed(*parts, *([salt] if salt else []))

    assess_fold = 0
    fit_folds = [f for f in range(5) if f != assess_fold]
    fit_cids = [c for f in fit_folds for c in cids_by_fold[f]]
    fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
    rel_map = {c: packets[c]["rel_k"] for c in fit_cids}
    oof = {(lr, n): {} for lr in core.LR_GRID for n in core.N_EST_GRID}
    for hold in fit_folds:
        core.check_cooperative_pause()
        tr = [c for f in fit_folds if f != hold for c in cids_by_fold[f]]
        va = cids_by_fold[hold]
        X_tr = np.concatenate([packets[c]["X"] for c in tr])
        y_tr = np.concatenate([rel_map[c] for c in tr])
        X_va = np.concatenate([packets[c]["X"] for c in va])
        inv_tr = np.concatenate([~packets[c]["gate_k"] for c in tr])
        gate = veto.fit_gate(
            X_tr, inv_tr, ss(seed, assess_fold, "RERANK8_VETO", "gate", hold)
        )
        pv = veto.gate_probs(gate, X_va, len(va), k)
        for lr in core.LR_GRID:
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
                pm = veto.mask(p, pv)
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
    gate = veto.fit_gate(
        X_fit,
        np.concatenate([~packets[c]["gate_k"] for c in fit_cids]),
        ss(seed, assess_fold, "RERANK8_VETO", "gate", "outer"),
    )
    receipt = dict(
        arm="RERANK8_VETO",
        k=k,
        tau=veto.TAU,
        lr=float(best_cfg[0]),
        n_estimators=int(best_cfg[1]),
        margin=float(best_cfg[2]),
        inner_fit_family_gain=float(best_gain),
        inner_fit_ci95=[lcb, ucb],
        strict_admitted=admitted,
        strict_margin="inf" if math.isinf(margin) else float(margin),
        training_contexts=len(fit_cids),
        training_families=len(fit_fams),
    )
    return dict(
        receipt=receipt,
        ranker=model.booster_.model_to_string(),
        gate=gate.booster_.model_to_string(),
        n_estimators=int(best_cfg[1]),
        margin=margin,
        ranker_obj=model,
        gate_obj=gate,
    )


def frozen_picks(ranker_text, gate_text, X, n_ctx, n_est, margin, k=8):
    """Inference of an exported frozen model: ranker scores at n_est, H273 veto mask (tau 0.10), unchanged margin rule."""
    import lightgbm as lgb

    sc = np.asarray(
        lgb.Booster(model_str=ranker_text).predict(X, num_iteration=int(n_est)),
        dtype=np.float64,
    ).reshape(n_ctx, k)
    p = np.asarray(
        lgb.Booster(model_str=gate_text).predict(X), dtype=np.float64
    ).reshape(n_ctx, k)
    sm = veto.mask(sc, p)
    return [core.select_from_top4_scores(sm[i], margin) for i in range(n_ctx)], sc, p
