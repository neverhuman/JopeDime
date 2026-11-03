"""Exact H276 cell with a TRAIN-only within-family shuffled-ranker placebo and read-only fit receipts."""

import hashlib
import numpy as np
from dependencies import topk, core
import veto


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def run(seed, folds, packets, families, salt, placebo=False):
    fit = [c for f in range(1, 5) for c in folds[f]]
    outer = set(folds[0])
    assert not outer.intersection(fit)
    p = packets
    shuffle_receipt = {}
    if placebo:
        labels = {c: packets[c]["rel_k"] for c in fit}
        shuffled = topk.shuffle_within_family(
            labels, fit, families, core.stream_seed(seed, 0, "shuf", salt), 8
        )
        p = {
            c: dict(v, rel_k=shuffled[c]) if c in shuffled else v
            for c, v in packets.items()
        }
        shuffle_receipt = {c: array_sha(shuffled[c]) for c in fit}
    calls = []
    original_rank = topk.fit_ranker
    original_gate = veto.fit_gate

    def rank(X, y, n, k, lr, random_state):
        calls.append(
            dict(
                kind="ranker",
                X_sha256=array_sha(X),
                y_sha256=array_sha(y),
                contexts=n,
                lr=lr,
                random_state=random_state,
            )
        )
        return original_rank(X, y, n, k, lr, random_state)

    def gate(X, y, random_state):
        calls.append(
            dict(
                kind="gate",
                X_sha256=array_sha(X),
                y_sha256=array_sha(y.astype(int)),
                candidate_rows=len(y),
                random_state=random_state,
            )
        )
        return original_gate(X, y, random_state)

    topk.fit_ranker = rank
    veto.fit_gate = gate
    try:
        result = veto.run_cell_veto(seed, 0, folds, p, families, 8, salt)
    finally:
        topk.fit_ranker = original_rank
        veto.fit_gate = original_gate
    result["fit_receipts"] = calls
    result["shuffled_labels_sha256"] = shuffle_receipt
    return result


def picks(model, gate, X, n_estimators, margin):
    import lightgbm as lgb

    sc = (
        lgb.Booster(model_str=model, params={"num_threads": 1})
        .predict(X, num_iteration=n_estimators, num_threads=1)
        .reshape(-1, 8)
    )
    pr = (
        lgb.Booster(model_str=gate, params={"num_threads": 1})
        .predict(X, num_threads=1)
        .reshape(-1, 8)
    )
    sc = veto.mask(sc, pr)
    return np.asarray([core.select_from_top4_scores(s, margin) for s in sc], int)
