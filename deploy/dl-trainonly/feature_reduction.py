#!/usr/bin/env python3
"""Train-only feature reduction for fixed-width consumers (Jepson ruling relayed by the owner, 2026-10-07 07:17 MT).

Standing rule for every dataset: a0 and dope always see the FULL feature list. A consumer with a fixed input width W gets
a view reduced to W columns, ranked on the outer fold's TRAINING rows only. Two such consumers exist:
  * the S1 GP teacher input (1 <= p <= 512; tools-gen/small_data_qualify.py), the bound that refused every wide dataset;
  * any GPU encoder whose MEASURED width is below p. As of 2026-10-07 none is: hyperion, hyperion_v6, hyperion_v6all,
    hyperion_v7, foundation, tabpfn35 and context2048 all encode p = 2048 and read every column; target reads y only.

Selection reads exactly jtt1.train_view(raw): the inner-fold labels and the outer-train block. The outer-test range is never
decoded, so no outer-test feature value or target can change which columns are kept (test_feature_reduction.LeakGate).

Method: for each inner fold f named by the JTT1 inner-fold labels (a seeded 2-fold split when the labels name fewer than two
folds), fit a CatBoost regressor (XGBoost when catboost is not importable) on the other inner folds. Importance is the mean
absolute SHAP value of each column over the HELD-OUT rows of inner fold f (CatBoost ShapValues / XGBoost pred_contribs),
averaged over inner folds. The ruling admits SHAP on train-row inner validation, and it costs one tree-SHAP pass per fold
where permutation importance would cost p x repeats predictions (measured 135 s wall at p=1,088, n_train=1,600). Columns are
ranked by (-importance, column index), the top k kept, and the kept columns returned in ascending column order. A fixed seed
and one thread for fit, predict and SHAP make every run byte-identical.

Constant inner training targets (#257 review 5443900942) are supported, never excluded. When the training rows of a held-out
inner fold have one target value, no model is fitted: a constant predictor's SHAP values are exactly zero, so that fold
contributes zero importance under either model. Its label goes into `constant_inner_folds`. When EVERY inner fold's training
target is constant, model importance is zero everywhere, and the ranking falls back to each column's population variance over
the outer-train rows (target-free; ties by column index), recorded as `fallback`.

Constant inner training FEATURES (#257 review 5450259534) also contribute zero importance: without a varying column there
is no split to learn, even if targets vary. Such folds are recorded separately as `constant_feature_inner_folds`. When every
fold is uninformative through constant targets or features, the same target-free outer-train variance ranking applies.

reduce_jtt1 copies the kept columns of BOTH blocks positionally into a new JTT1 (target column last, labels unchanged), so a
fixed-width consumer can still score its outer-test rows. Copying makes no decision; the selection never saw those bytes.
"""
import hashlib
import json
import struct

import numpy as np

from dltrain import jtt1

SCHEMA = "jope.dl-feature-reduction/v1"
METHOD = "inner-cv-mean-abs-shap"
SEED = 20261007
TEACHER_WIDTH = 512
CATBOOST = {"iterations": 200, "depth": 6, "learning_rate": 0.1, "loss_function": "RMSE"}
XGBOOST = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist"}


def _model(seed, prefer="catboost"):
    """(fitted-model factory, description). CatBoost first; XGBoost only when catboost is not importable (or asked for)."""
    if prefer == "catboost":
        try:
            import catboost
            from catboost import CatBoostRegressor

            def make():
                return CatBoostRegressor(**CATBOOST, random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
            return make, {"name": "catboost", "version": catboost.__version__, "params": dict(CATBOOST, thread_count=1)}
        except ImportError:
            pass
    import xgboost
    from xgboost import XGBRegressor

    def make():
        return XGBRegressor(**XGBOOST, random_state=seed, n_jobs=1)
    return make, {"name": "xgboost", "version": xgboost.__version__, "params": dict(XGBOOST, n_jobs=1)}


FALLBACK_ALL_CONSTANT = "outer-train population variance (every inner training target constant)"
FALLBACK_ALL_UNINFORMATIVE = "outer-train population variance (every inner training fold has constant targets or features)"


def _folds(inner, seed):
    """(held-out masks, fold names, split description)."""
    names = sorted(set(int(x) for x in inner.tolist()))
    if len(names) >= 2:
        return [inner == f for f in names], names, "jtt1-inner-fold-labels"
    order = np.random.default_rng([seed, 2]).permutation(len(inner))
    half = np.zeros(len(inner), dtype=bool)
    half[order[: len(inner) // 2]] = True
    return [half, ~half], [0, 1], "seeded-2-fold"


def _shap(m, desc, xv, yv):
    """(rows, p) SHAP values of the held-out rows, one thread, expected-value column dropped."""
    if desc["name"] == "catboost":
        from catboost import Pool
        return np.asarray(m.get_feature_importance(Pool(xv, yv), type="ShapValues", thread_count=1))[:, :-1]
    import xgboost
    return np.asarray(m.get_booster().predict(xgboost.DMatrix(xv, nthread=1), pred_contribs=True))[:, :-1]


def importance(X, y, inner, *, seed=SEED, prefer="catboost"):
    """(mean |SHAP| per column over held-out inner rows, model, split, constant-target folds, constant-feature folds).

    A fold with constant training targets or no varying feature contributes a constant predictor's exact SHAP, zero."""
    make, desc = _model(seed, prefer)
    held, names, how = _folds(np.asarray(inner), seed)
    imp = np.zeros(X.shape[1], dtype=np.float64)
    constant = []
    constant_features = []
    for va, name in zip(held, names):
        tr = ~va
        if tr.sum() < 2 or va.sum() < 2:
            raise ValueError("an inner fold needs at least two training and two validation rows")
        if np.all(y[tr] == y[tr][0]):
            constant.append(int(name))
            continue
        xt = X[tr]
        if np.all(xt == xt[0]):
            constant_features.append(int(name))
            continue
        m = make()
        m.fit(xt, y[tr])
        imp += np.abs(_shap(m, desc, X[va], y[va])).mean(axis=0)
    return imp / len(held), desc, how, constant, constant_features


def list_sha256(idx):
    return hashlib.sha256(json.dumps([int(i) for i in idx], separators=(",", ":")).encode()).hexdigest()


def select(X, y, inner, k, *, seed=SEED, prefer="catboost"):
    """(kept column indices ascending, feature_reduction record) for one outer fold's TRAINING rows."""
    p = int(X.shape[1])
    if k < 1:
        raise ValueError("k must be >= 1")
    if p <= k:
        idx = np.arange(p)
        rec = {"method": "identity (p <= k)", "model": None}
    else:
        imp, desc, how, constant, constant_features = importance(X, y, inner, seed=seed, prefer=prefer)
        fallback = None
        if len(constant) + len(constant_features) == len(_folds(np.asarray(inner), seed)[1]):
            imp = np.asarray(X, dtype=np.float64).var(axis=0)
            fallback = FALLBACK_ALL_UNINFORMATIVE if constant_features else FALLBACK_ALL_CONSTANT
        order = sorted(range(p), key=lambda j: (-imp[j], j))
        idx = np.array(sorted(order[:k]), dtype=np.int64)
        rec = {"method": METHOD, "model": desc, "inner_split": how, "constant_inner_folds": constant, "fallback": fallback}
        if constant_features:
            rec["constant_feature_inner_folds"] = constant_features
    return idx, {"schema": SCHEMA, **rec, "k": int(k), "seed": int(seed), "n_features_original": p, "n_train": int(X.shape[0]),
                 "fit_rows": "outer_train_only", "selected_feature_indices": [int(i) for i in idx],
                 "selected_feature_list_sha256": list_sha256(idx)}


def select_for_jtt1(raw, k, *, seed=SEED, prefer="catboost"):
    """Selection from a context JTT1: decodes ONLY jtt1.train_view (labels + outer-train block)."""
    _, inner, X, y = jtt1.train_view(raw)
    idx, rec = select(X, y, inner, k, seed=seed, prefer=prefer)
    rec["train_rows_sha256"] = jtt1.train_rows_sha256(raw)
    return idx, rec


def reduce_jtt1(raw, idx):
    """A JTT1 holding only the kept feature columns of both blocks (target last, inner labels unchanged)."""
    h = jtt1.parse_header(raw)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0 or np.any(np.diff(idx) <= 0) or idx[0] < 0 or idx[-1] >= h["p"]:
        raise ValueError("kept columns must be strictly increasing indices into the feature columns")
    n = h["n_train"] + h["n_test"]
    mv = memoryview(raw)
    labels = bytes(mv[jtt1.HEADER_BYTES:jtt1.HEADER_BYTES + h["n_train"]])
    rows = np.frombuffer(mv[jtt1.HEADER_BYTES + h["n_train"]:], dtype="<f8").reshape(n, h["p"] + 1)
    kept = np.ascontiguousarray(rows[:, np.append(idx, h["p"])], dtype="<f8")
    out = jtt1.MAGIC + struct.pack("<IIIB", h["n_train"], h["n_test"], int(idx.size), h["inner_folds"]) + labels + kept.tobytes()
    jtt1.parse_header(out)
    return out


def teacher_view(raw, k=TEACHER_WIDTH, *, seed=SEED, prefer="catboost"):
    """(teacher JTT1 bytes for a width-k consumer, record). p <= k returns the input unchanged with an identity record."""
    idx, rec = select_for_jtt1(raw, k, seed=seed, prefer=prefer)
    if rec["n_features_original"] <= k:
        return raw, rec
    out = reduce_jtt1(raw, idx)
    rec["reduced_raw_sha256"] = hashlib.sha256(out).hexdigest()
    return out, rec
