"""TRAIN-only cluster weights and nested selection; original H272 modules remain byte-identical."""

import hashlib
import numpy as np
import lightgbm as lgb
from dependencies import core
import gate

TAUS = (0.05, 0.10, 0.15, 0.20)


def weights(cids, fams, cluster, permutation_seed=None):
    """Equal mass per present cluster, per family within cluster, per context within family."""
    families = sorted({fams[c] for c in cids})
    groups = sorted({cluster[f] for f in families})
    sizes = {g: sum(cluster[f] == g for f in families) for g in groups}
    inverse = np.array([1 / sizes[g] for g in groups])
    perm = (
        np.arange(len(groups))
        if permutation_seed is None
        else np.random.default_rng(permutation_seed).permutation(len(groups))
    )
    bygroup = {g: float(inverse[j]) for g, j in zip(groups, perm)}
    counts = {f: sum(fams[c] == f for c in cids) for f in families}
    w = np.array([bygroup[cluster[fams[c]]] / counts[fams[c]] for c in cids])
    w /= w.mean()
    fw = np.array([bygroup[cluster[f]] for f in families])
    fw /= fw.sum()
    return (
        w,
        families,
        fw,
        dict(
            cluster_sizes=sizes,
            cluster_inverse_weights=bygroup,
            permutation=perm.tolist(),
        ),
    )


def fit_ranker(X, y, n_groups, k, lr, random_state, sample_weight):
    # Frozen H272 estimator parameters, with only its fit sample_weight supplied.
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
    model.fit(
        X,
        y,
        group=np.full(n_groups, k, dtype=np.int32),
        eval_at=[1, k],
        sample_weight=sample_weight,
    )
    return model


def picks(scores, margin, probs=None, tau=None):
    j = np.argmax(scores, axis=1)
    j = np.where(scores[np.arange(len(j)), j] - scores[:, 0] > margin, j, 0)
    if probs is not None:
        j = np.where((j != 0) & (probs[np.arange(len(j)), j] > tau), 0, j)
    return j


def choose(scores, utilities, family_names, family_weights, seed, probs=None):
    families = sorted(set(family_names))
    names = np.asarray(family_names)
    averaging = np.stack([(names == f) / sum(names == f) for f in families])
    best_gain = -1e18
    best = None
    best_delta = None
    for lr in core.LR_GRID:
        for trees in core.N_EST_GRID:
            sc = scores[(lr, trees)]
            for margin in core.MARGIN_GRID[:-1]:
                for tau in TAUS if probs is not None else [None]:
                    j = picks(sc, margin, probs, tau)
                    d = averaging @ (utilities[np.arange(len(j)), j] - utilities[:, 0])
                    gain = float(d @ family_weights)
                    tie = (margin, -trees, -lr, -tau if tau is not None else 0)
                    if gain > best_gain + 1e-12 or (
                        abs(gain - best_gain) <= 1e-12
                        and (best is None or tie > best[-1])
                    ):
                        best_gain = gain
                        best = (lr, trees, margin, tau, tie)
                        best_delta = d
    ix = np.random.default_rng(seed).integers(0, len(families), (2000, len(families)))
    boot = (best_delta[ix] * family_weights[ix]).sum(axis=1) / family_weights[ix].sum(
        axis=1
    )
    ci = np.quantile(boot, [0.025, 0.975])
    admit = bool(best_gain > core.STRICT_FIT_MIN_GAIN and ci[0] > 0)
    return dict(
        lr=float(best[0]),
        n_estimators=int(best[1]),
        margin=float(best[2]),
        tau=best[3],
        inner_gain=best_gain,
        inner_ci95=ci.tolist(),
        strict_admitted=admit,
        strict_margin=float(best[2]) if admit else "inf",
    )


def run(seed, folds, packets, fams, cluster, salt, placebo=False):
    ff = [1, 2, 3, 4]
    fitids = [c for f in ff for c in folds[f]]
    pos = {c: i for i, c in enumerate(fitids)}
    n = len(fitids)
    oof = {(lr, t): np.empty((n, 8)) for lr in core.LR_GRID for t in core.N_EST_GRID}
    prob = np.empty((n, 8))
    calls = []
    records = []

    def ss(*parts):
        return core.stream_seed(*parts, salt)

    def fit(cids, lr, rng, label):
        cw, _, _, info = weights(
            cids, fams, cluster, ss(seed, "weights", label) if placebo else None
        )
        X = np.concatenate([packets[c]["X"] for c in cids])
        y = np.concatenate([packets[c]["rel_k"] for c in cids])
        w = np.repeat(cw, 8)
        calls.append(
            dict(
                stage=label,
                cids=cids,
                lr=lr,
                random_state=rng,
                X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                weight_map=info,
            )
        )
        return fit_ranker(X, y, len(cids), 8, lr, rng, w)

    for held in ff:
        core.check_cooperative_pause()
        tr = [c for f in ff if f != held for c in folds[f]]
        va = folds[held]
        ix = [pos[c] for c in va]
        assert not {fams[c] for c in tr} & {fams[c] for c in va + folds[0]}
        Xv = np.concatenate([packets[c]["X"] for c in va])
        for lr in core.LR_GRID:
            m = fit(tr, lr, ss(seed, 0, "RERANK8", "inner", held, lr), f"inner{held}")
            for trees in core.N_EST_GRID:
                oof[(lr, trees)][ix] = m.booster_.predict(
                    Xv, num_iteration=trees, num_threads=1
                ).reshape(len(va), 8)
        if not placebo:
            Xtr = np.concatenate([packets[c]["X"] for c in tr])
            yinv = np.concatenate([~packets[c]["gate_k"] for c in tr])
            g = gate.fit(Xtr, yinv, ss(seed, 0, "H273_GATE", "inner", held))
            prob[ix] = g.booster_.predict(Xv, num_threads=1).reshape(len(va), 8)
            records.append(
                dict(
                    stage=f"inner{held}",
                    cids=tr,
                    gate_y_sha256=hashlib.sha256(yinv.tobytes()).hexdigest(),
                )
            )
    _, family_order, fw, wmap = weights(
        fitids, fams, cluster, ss(seed, "weights", "selector") if placebo else None
    )
    u = np.stack([packets[c]["utility_k"] for c in fitids])
    names = [fams[c] for c in fitids]
    selections = {
        ("PLACEBO" if placebo else "A"): choose(
            oof, u, names, fw, ss(core.BOOT_SEED, "inner_select", seed, 0, "RERANK8")
        )
    }
    if not placebo:
        selections["B"] = choose(
            oof,
            u,
            names,
            fw,
            ss(core.BOOT_SEED, "inner_select", seed, 0, "RERANK8"),
            prob,
        )
    models = {
        lr: fit(fitids, lr, ss(seed, 0, "RERANK8", "outer", lr), "outer").booster_
        for lr in sorted({v["lr"] for v in selections.values()})
    }
    Xa = np.concatenate([packets[c]["X"] for c in folds[0]])
    gate_model = None
    assess_prob = None
    if not placebo:
        g = gate.fit(
            np.concatenate([packets[c]["X"] for c in fitids]),
            np.concatenate([~packets[c]["gate_k"] for c in fitids]),
            ss(seed, 0, "H273_GATE", "outer"),
        )
        gate_model = g.booster_
        assess_prob = gate_model.predict(Xa, num_threads=1).reshape(len(folds[0]), 8)
    out = {}
    for arm, cfg in selections.items():
        sc = (
            models[cfg["lr"]]
            .predict(Xa, num_iteration=cfg["n_estimators"], num_threads=1)
            .reshape(len(folds[0]), 8)
        )
        margin = cfg["strict_margin"] if cfg["strict_admitted"] else np.inf
        out[arm] = dict(
            receipt=cfg,
            model=models[cfg["lr"]].model_to_string(),
            gate=gate_model.model_to_string() if arm == "B" else None,
            assess_picks=picks(
                sc, margin, assess_prob if arm == "B" else None, cfg["tau"]
            ),
        )
    evidence = dict(
        fit_calls=calls,
        gate_calls=records,
        fitids=fitids,
        family_order=family_order,
        selector_weights=fw.tolist(),
        selector_weight_map=wmap,
    )
    arrays = {f"score_{lr}_{t}": v for (lr, t), v in oof.items()}
    if not placebo:
        arrays["invalid_oof"] = prob
    return out, evidence, arrays
