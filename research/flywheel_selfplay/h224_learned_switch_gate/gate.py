"""H224 outcome-free features, nested FIT gate CV and harm-constrained selection."""

import numpy as np

CS = (0.01, 0.1, 1.0, 10.0)


def features(reference, prior, blended, zgap):
    r, p, b = (np.asarray(v, np.float64) for v in (reference, prior, blended))
    if (
        r.ndim != 1
        or len(r) < 2
        or r.shape != p.shape
        or r.shape != b.shape
        or not all(np.isfinite(v).all() for v in (r, p, b))
    ):
        raise ValueError("matching finite cost vectors required")
    a, j = int(r.argmin()), int(b.argmin())
    rank = int(np.flatnonzero(np.argsort(b, kind="stable") == a)[0]) / (len(r) - 1)
    margin = float(np.partition(r, 1)[1] - r[a])
    signal = min(float(zgap(r, b)), 1e6)
    return np.array([signal, margin, rank, p[a] - p[j], len(r)], np.float64), a != j


def weights(families):
    f = np.asarray(families)
    return np.array([1.0 / np.count_nonzero(f == x) for x in f])


def fit(x, y, families, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x, y = np.asarray(x, np.float64), np.asarray(y, int)
    if not len(y):
        return dict(constant=0.0)
    if len(np.unique(y)) == 1:
        return dict(constant=float(y[0]))
    w = weights(families)
    w *= len(w) / w.sum()
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = LogisticRegression(
        C=c, penalty="l2", solver="lbfgs", max_iter=1000, tol=1e-8
    ).fit(scaler.transform(x), y, sample_weight=w)
    if int(model.n_iter_[0]) >= 1000:
        raise RuntimeError("gate optimizer failed to converge")
    return dict(
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=model.coef_[0].tolist(),
        intercept=float(model.intercept_[0]),
        iterations=int(model.n_iter_[0]),
    )


def predict(model, x):
    from scipy.special import expit

    x = np.asarray(x, np.float64)
    if "constant" in model:
        return np.full(len(x), model["constant"])
    return expit(
        ((x - model["mean"]) / model["scale"]) @ np.asarray(model["coef"])
        + model["intercept"]
    )


def shuffled_labels(labels, families, seed):
    y = np.asarray(labels).copy()
    f = np.asarray(families)
    rng = np.random.default_rng(seed)
    for family in sorted(set(f.tolist())):
        ids = np.flatnonzero(f == family)
        y[ids] = rng.permutation(y[ids])
    return y


def train(x, utility_delta, families, groups, switching, seed, shuffled=False):
    """C and threshold use FIT only; threshold uses gate-OOF probabilities, never fitted probabilities."""
    x = np.asarray(x, np.float64)
    delta, fam, group, switch = (
        np.asarray(utility_delta),
        np.asarray(families),
        np.asarray(groups),
        np.asarray(switching, bool),
    )
    if (
        x.shape != (len(delta), 5)
        or not np.isfinite(x).all()
        or not np.isfinite(delta).all()
        or len(set(group.tolist())) != 3
    ):
        raise ValueError("three inner FIT groups and finite five-feature rows required")
    if any(len(set(group[fam == f].tolist())) != 1 for f in set(fam.tolist())):
        raise ValueError("family crosses gate CV groups")
    y = (delta > 0).astype(int)
    if shuffled:
        y[switch] = shuffled_labels(y[switch], fam[switch], seed)
    reports, predictions = [], {}
    for c in CS:
        prob = np.zeros(len(delta))
        for held in sorted(set(group.tolist())):
            train_rows, test_rows = switch & (group != held), switch & (group == held)
            model = fit(x[train_rows], y[train_rows], fam[train_rows], c)
            prob[test_rows] = predict(model, x[test_rows])
        if switch.any():
            q = np.clip(prob[switch], 1e-12, 1 - 1e-12)
            losses = -(y[switch] * np.log(q) + (1 - y[switch]) * np.log1p(-q))
            w = weights(fam[switch])
            loss = float(np.dot(w, losses) / w.sum())
        else:
            loss = 0.0
        reports.append(dict(C=c, oof_log_loss=loss))
        predictions[c] = prob
    chosen = min(reports, key=lambda row: (row["oof_log_loss"], row["C"]))["C"]
    prob = predictions[chosen]
    thresholds = [
        float("inf"),
        *(
            sorted(
                set(
                    np.quantile(
                        prob[switch], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                    ).tolist()
                )
            )
            if switch.any()
            else []
        ),
        float("-inf"),
    ]
    grid = []
    for tau in thresholds:
        opened = switch & (prob > tau)
        gains = np.where(opened, delta, 0.0)
        fmean = np.array([gains[fam == f].mean() for f in sorted(set(fam.tolist()))])
        grid.append(
            dict(
                threshold="closed"
                if tau == float("inf")
                else "open"
                if tau == float("-inf")
                else tau,
                gain=float(fmean.mean()),
                harm=float((fmean < -0.01).mean()),
                adoption=int(opened.sum()),
            )
        )
    feasible = [row for row in grid if row["harm"] <= 0.05]
    best = max(
        feasible, key=lambda row: (row["gain"], threshold_value(row["threshold"]))
    )
    model = fit(x[switch], y[switch], fam[switch], chosen)
    return dict(
        model=model,
        C=chosen,
        threshold=best["threshold"],
        cv=reports,
        grid=grid,
        fit_gain=best["gain"],
        fit_harm=best["harm"],
        fit_switches=int(switch.sum()),
        threshold_uses_gate_oof=True,
        shuffled=shuffled,
    )


def threshold_value(t):
    return float("inf") if t == "closed" else float("-inf") if t == "open" else float(t)


def opens(choice, x, switching):
    return np.asarray(switching, bool) & (
        predict(choice["model"], x) > threshold_value(choice["threshold"])
    )
