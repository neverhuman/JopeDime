"""Train-only per-split preparation and teacher scoring (UPSTREAM-SCALE-01 / S-STAT-09 remedy).

The upstream research_2k data objects were min-max scaled over every sampled row, outer-test rows included, and the
JTT1 inner-fold labels were assigned from those scaled bytes (prepare_fold0.inner_assignments orders tied
exact-feature components by sha256 of the scaled row bytes). Scaling, inner folds, encoder inputs and GP-teacher labels
therefore all depended on the outer-test rows. This module builds every train-side quantity from outer-train rows only:

    per (dataset, outer fold), raw source matrix M (float64, last column = target), outer-train rows R, outer-test rows T:
    1. lo_j = min_R M[:, j], span_j = max_R M[:, j] - lo_j (span 0 -> divisor 1), for every column including the target
    2. every row r of R and T is mapped to (M[r] - lo) / span in float64 (subtract, then divide); T rows may leave [0, 1]
    3. inner 2-fold labels: the upstream policy (exact-feature components, target-mean decile, greedy balance, digest
       tie-break) applied to the train-only-scaled R rows, so the labels are a function of R alone
    4. JTT1 bytes: header, the labels, then the R rows then the T rows, as in prepare_fold0.prepare_one

Teacher scoring keeps the two scores apart: `inner_cv` (2-fold CV inside R, used for q and for selection) and `outer_test`
(fit on all of R, scored on T, stored and reported only). `select_best` receives the inner-CV part alone. Each inner fold
j refits its scaler on its own inner-train rows from the raw values (`inner_fold_fit`), so fold j's validation rows reach
neither fold j's scaler nor its fitted program; both RMSEs are reported in raw target units.

`counterfactual_report` is the regression check the owner asked for: it rebuilds a split with the outer-test rows
perturbed or replaced and lists every train-side field that is not bit-identical. Its negative controls (a scaler fitted
on all rows, a selector that reads the test score) must be reported as differing, which shows the check can fail.

Numpy and the standard library only. No S3, GPU, evaluator or GP-runtime access.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dltrain import jtt1, labels

SCHEMA = "jope.dl-trainonly-split/v1"
INNER_SCHEMA = "jope.inner-cv/exact-feature-component-stratified-2fold/v1"
INNER_POLICY = "exact_feature_components_target_mean_decile_greedy_balance"
SCALE_POLICY = "minmax_outer_train_rows_only_float64_sub_then_div_span0_div1"
MIN_INNER_ROWS = 25
INNER_FOLDS = 2
TRAIN_FIELDS = ("scaler", "train_block", "inner_labels", "jtt1_train_rows", "encoding", "teacher_inner", "selection")
INNER_KEYS = frozenset({"rmse_j", "baseline_rmse_j", "q", "q_grid", "gate_pass"})


class IneligibleSplit(ValueError):
    """The split cannot support group-safe 2-fold inner CV (prepare_fold0.IneligibleView)."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _f64(a) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    if not bool(np.all(np.isfinite(out))):
        raise ValueError("non-finite value in a source matrix")
    return out


# ---------------------------------------------------------------------------------------------- scaling

def fit_minmax(train_block) -> tuple[np.ndarray, np.ndarray]:
    """(lo, span) per column of the outer-train rows only; a constant column gets span 1.0."""
    block = _f64(train_block)
    if block.ndim != 2 or block.shape[0] < 1:
        raise ValueError(f"train block must be 2-d and non-empty, got {block.shape}")
    lo = block.min(axis=0)
    span = block.max(axis=0) - lo
    span = np.where(span == 0.0, 1.0, span)
    if not bool(np.all(np.isfinite(span))):
        raise ValueError("min-max span overflowed")
    return lo, span


def apply_minmax(block, lo, span) -> np.ndarray:
    return np.ascontiguousarray((_f64(block) - lo) / span, dtype="<f8")


# ---------------------------------------------------------------------------------------------- inner folds

def _feature_digest(row) -> str:
    return _sha(np.asarray(row, dtype="<f8").tobytes())


def inner_assignments(train_scaled, min_rows: int = MIN_INNER_ROWS) -> tuple[np.ndarray, dict]:
    """The prepare_fold0 policy on the train-only-scaled outer-train block (rows in outer-train order)."""
    block = _f64(train_scaled)
    components: dict[str, list[int]] = {}
    for i in range(block.shape[0]):
        components.setdefault(_feature_digest(block[i, :-1]), []).append(i)
    if len(components) < 2:
        raise IneligibleSplit("outer training has fewer than two exact-feature components")
    grouped = [(digest, idx, float(np.mean(block[idx, -1]))) for digest, idx in components.items()]
    order = np.argsort(np.asarray([g[2] for g in grouped], dtype=np.float64), kind="mergesort")
    bins = np.empty(len(grouped), dtype=np.int64)
    for rank, gi in enumerate(order):
        bins[gi] = min(9, rank * 10 // len(grouped))
    loads = [0, 0]
    bin_loads = np.zeros((2, 10), dtype=np.int64)
    assignment: dict[str, int] = {}
    for gi, (digest, idx, _mean) in sorted(enumerate(grouped), key=lambda it: (int(bins[it[0]]), -len(it[1][1]), it[1][0])):
        b = int(bins[gi])
        target = min(range(2), key=lambda c: (bin_loads[c, b], loads[c], c))
        assignment[digest] = target
        loads[target] += len(idx)
        bin_loads[target, b] += len(idx)
    lab = np.empty(block.shape[0], dtype=np.uint8)
    for digest, idx, _mean in grouped:
        lab[idx] = assignment[digest]
    counts = np.bincount(lab, minlength=2)
    if counts.min() < min_rows:
        raise IneligibleSplit(f"inner CV load below {min_rows} rows: {counts.tolist()}")
    return lab, {"schema": INNER_SCHEMA, "policy": INNER_POLICY, "basis": "train_only_scaled_outer_train_rows",
                 "loads": counts.tolist(), "target_bin_loads": bin_loads.tolist(),
                 "exact_feature_components": len(components), "assignments_sha256": _sha(lab.tobytes())}


# ---------------------------------------------------------------------------------------------- one split

def jtt1_bytes(train_scaled, test_scaled, inner_labels) -> bytes:
    tr, te = _f64(train_scaled), _f64(test_scaled).reshape(-1, train_scaled.shape[1])
    lab = np.asarray(inner_labels, dtype=np.uint8)
    if lab.shape != (tr.shape[0],):
        raise ValueError("one inner label per outer-train row")
    header = jtt1.MAGIC + struct.pack("<IIIB", tr.shape[0], te.shape[0], tr.shape[1] - 1, INNER_FOLDS)
    raw = header + lab.tobytes() + np.vstack([tr, te]).astype("<f8").tobytes(order="C")
    jtt1.parse_header(raw)
    return raw


def prepare_split(matrix, outer_train: Sequence[int], outer_test: Sequence[int], min_inner_rows: int = MIN_INNER_ROWS,
                  inner_labels=None) -> dict:
    """Build one split from the raw source matrix. Only `outer_train` rows reach the scaler, the inner folds and the
    train block; `outer_test` rows are transformed with the train constants and appended after them. `inner_labels`
    (one u8 per outer-train row) holds the assignment fixed for the inner-validation counterfactual; it must still be
    group-safe on this build's rows."""
    m = _f64(matrix)
    tr_idx = [int(i) for i in outer_train]
    te_idx = [int(i) for i in outer_test]
    if set(tr_idx) & set(te_idx):
        raise ValueError("outer-train and outer-test rows overlap")
    if len(set(tr_idx)) != len(tr_idx) or len(set(te_idx)) != len(te_idx):
        raise ValueError("duplicate row index in a split")
    train_raw = m[tr_idx]
    lo, span = fit_minmax(train_raw)
    train_scaled = apply_minmax(train_raw, lo, span)
    test_scaled = apply_minmax(m[te_idx], lo, span) if te_idx else np.empty((0, m.shape[1]), dtype="<f8")
    if inner_labels is None:
        inner, evidence = inner_assignments(train_scaled, min_inner_rows)
    else:
        inner, evidence = _fixed_inner(train_scaled, inner_labels, min_inner_rows)
    raw = jtt1_bytes(train_scaled, test_scaled, inner)
    return {"schema": SCHEMA, "scale_policy": SCALE_POLICY, "lo": lo, "span": span, "train": train_scaled,
            "test": test_scaled, "train_raw": train_raw, "test_raw": m[te_idx].reshape(-1, m.shape[1]),
            "inner_labels": inner, "inner_cv": evidence, "jtt1": raw, "n_train": len(tr_idx), "n_test": len(te_idx)}


def _fixed_inner(train_scaled, inner_labels, min_rows):
    lab = np.asarray(inner_labels)
    if lab.shape != (train_scaled.shape[0],) or not set(np.unique(lab).tolist()) <= {0, 1}:
        raise ValueError("fixed inner labels need one 0/1 label per outer-train row")
    lab = lab.astype(np.uint8)
    seen: dict[str, int] = {}
    for i in range(train_scaled.shape[0]):
        if seen.setdefault(_feature_digest(train_scaled[i, :-1]), int(lab[i])) != int(lab[i]):
            raise IneligibleSplit("fixed inner labels split an exact-feature component")
    counts = np.bincount(lab, minlength=2)
    if counts.min() < min_rows:
        raise IneligibleSplit(f"inner CV load below {min_rows} rows: {counts.tolist()}")
    return lab, {"schema": INNER_SCHEMA, "policy": "fixed_by_caller", "loads": counts.tolist(),
                 "assignments_sha256": _sha(lab.tobytes())}


def scaler_bytes(split: Mapping[str, Any]) -> bytes:
    return np.asarray(split["lo"], dtype="<f8").tobytes() + np.asarray(split["span"], dtype="<f8").tobytes()


# ---------------------------------------------------------------------------------------------- teacher scores

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _rmse(pred, y) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(math.sqrt(float(np.mean(d * d))))


def inner_fold_fit(split: Mapping[str, Any], j: int, fit_predict: FitPredict, fingerprint: bool = False) -> dict:
    """Inner fold j: the scaler is refitted on fold j's inner-train rows (inner label != j) from the raw source values,
    the program is fitted on them, and fold j's validation rows are only transformed and scored. Predictions are
    mapped back to raw target units, so RMSE and its mean baseline do not depend on any scaler. `fingerprint` adds
    the sha256 of the refitted scaler and of the program's predictions on its own inner-train rows."""
    raw, g = split["train_raw"], np.asarray(split["inner_labels"])
    fit_rows, val_rows = raw[g != j], raw[g == j]
    lo, span = fit_minmax(fit_rows)
    f, v = apply_minmax(fit_rows, lo, span), apply_minmax(val_rows, lo, span)

    def back(p):
        return np.asarray(p, dtype=np.float64) * span[-1] + lo[-1]

    base = float(np.mean(fit_rows[:, -1]))
    out = {"rmse": _rmse(back(fit_predict(f[:, :-1], f[:, -1], v[:, :-1])), val_rows[:, -1]),
           "baseline_rmse": _rmse(np.full(val_rows.shape[0], base), val_rows[:, -1])}
    if fingerprint:
        out["scaler_sha256"] = _sha(np.asarray(lo, "<f8").tobytes() + np.asarray(span, "<f8").tobytes()
                                    + np.float64(base).tobytes())
        out["fit_sha256"] = _sha(np.asarray(back(fit_predict(f[:, :-1], f[:, -1], f[:, :-1])), "<f8").tobytes())
    return out


def inner_cv(split: Mapping[str, Any], fit_predict: FitPredict) -> dict:
    """2-fold CV inside the outer-train rows only, every fold refitted on its own inner-train rows (inner_fold_fit).
    q follows labels.py (PLAN §6.3) on the raw outer-train target."""
    rmse_j, base_j = [], []
    for j in range(INNER_FOLDS):
        r = inner_fold_fit(split, j, fit_predict)
        rmse_j.append(r["rmse"])
        base_j.append(r["baseline_rmse"])
    y = split["train_raw"][:, -1]
    s_c = labels.robust_scale(y)
    eps = labels.epsilon(y, s_c)
    q = labels.clip_q(labels.q_from_detail(rmse_j, base_j, eps))
    gate = 0.5 * sum(r / b if b > 0 else math.inf for r, b in zip(rmse_j, base_j)) <= labels.GATE_RATIO
    return {"rmse_j": rmse_j, "baseline_rmse_j": base_j, "q": q, "q_grid": labels.q_grid(q), "gate_pass": bool(gate)}


def outer_test(split: Mapping[str, Any], fit_predict: FitPredict) -> dict:
    """Fit on every outer-train row, score the outer-test rows in raw target units. Stored and reported; never used
    for selection."""
    tr, te = split["train"], split["test"]
    if te.shape[0] == 0:
        return {"rmse": None, "n_test": 0}
    pred = np.asarray(fit_predict(tr[:, :-1], tr[:, -1], te[:, :-1]), dtype=np.float64) * split["span"][-1] + split["lo"][-1]
    return {"rmse": _rmse(pred, split["test_raw"][:, -1]), "n_test": int(te.shape[0])}


def teacher_record(split: Mapping[str, Any], program_id: str, fit_predict: FitPredict) -> dict:
    return {"program_id": program_id, "inner_cv": inner_cv(split, fit_predict), "outer_test": outer_test(split, fit_predict)}


def select_best(inner_scores: Sequence[tuple[str, Mapping[str, Any]]]) -> str | None:
    """The best program of a split from its inner-CV scores alone: gate-passing first, then the highest q_grid, then the
    smallest program_id. Takes (program_id, inner_cv) pairs, so an outer-test score cannot reach it."""
    best = None
    for pid, s in inner_scores:
        if set(s) != INNER_KEYS:
            raise ValueError(f"select_best takes inner-CV scores only (keys {sorted(INNER_KEYS)}), got {sorted(s)}")
        key = (0 if s["gate_pass"] else 1, -int(s["q_grid"]), str(pid))
        if best is None or key < best[0]:
            best = (key, pid)
    return None if best is None else best[1]


# ---------------------------------------------------------------------------------------------- counterfactual check

def train_side(split: Mapping[str, Any], programs: Mapping[str, FitPredict],
               encoder: Callable[[bytes], bytes] | None = None,
               selector: Callable[[list], str | None] | None = None) -> dict:
    """sha256 of every train-side field of one split. `encoder` maps JTT1 bytes to an encoding (e.g. A0 from the train
    view); `selector` defaults to select_best over the inner-CV scores."""
    records = [teacher_record(split, pid, fn) for pid, fn in sorted(programs.items())]
    if selector is None:
        chosen = select_best([(r["program_id"], r["inner_cv"]) for r in records])
    else:
        chosen = selector(records)
    inner_blob = repr([(r["program_id"], [float(v).hex() for v in r["inner_cv"]["rmse_j"]],
                        [float(v).hex() for v in r["inner_cv"]["baseline_rmse_j"]], float(r["inner_cv"]["q"]).hex(),
                        r["inner_cv"]["q_grid"], r["inner_cv"]["gate_pass"]) for r in records]).encode()
    out = {"scaler": _sha(scaler_bytes(split)),  # the hashes are over exact bytes and float.hex(), never rounded text
           "train_block": _sha(np.asarray(split["train"], dtype="<f8").tobytes()),
           "inner_labels": _sha(np.asarray(split["inner_labels"], dtype=np.uint8).tobytes()),
           "jtt1_train_rows": jtt1.train_rows_sha256(split["jtt1"]),
           "encoding": _sha(encoder(split["jtt1"])) if encoder else None,
           "teacher_inner": _sha(inner_blob),
           "selection": _sha(repr(chosen).encode())}
    out["outer_test"] = _sha(repr([(r["program_id"], r["outer_test"]["rmse"]) for r in records]).encode())
    return out


def counterfactual_report(matrix, outer_train, outer_test, variants: Mapping[str, Any],
                          programs: Mapping[str, FitPredict], encoder=None, selector=None,
                          prepare: Callable[..., dict] = prepare_split) -> dict:
    """Rebuild the split once per variant and list every train-side field that is not bit-identical to the original
    build. A variant is a replacement source matrix (its outer-test rows are every row outside `outer_train`) or a
    (matrix, outer_test_indices) pair; its outer-train rows must equal the original's bit for bit."""
    m = _f64(matrix)
    tr = [int(i) for i in outer_train]
    tr_set = set(tr)
    base = train_side(prepare(m, tr, outer_test), programs, encoder, selector)
    report = {"base": base, "variants": {}, "differing": {}}
    for name, alt in variants.items():
        alt_m, alt_te = alt if isinstance(alt, tuple) else (alt, None)
        a = _f64(alt_m)
        if a.shape[1] != m.shape[1] or a.shape[0] <= max(tr) or a[tr].tobytes() != m[tr].tobytes():
            raise ValueError(f"variant {name} changes an outer-train row")
        te = [i for i in range(a.shape[0]) if i not in tr_set] if alt_te is None else [int(i) for i in alt_te]
        got = train_side(prepare(a, tr, te), programs, encoder, selector)
        diff = [f for f in TRAIN_FIELDS if got[f] != base[f]]
        report["variants"][name] = {"fields": got, "outer_test_changed": got["outer_test"] != base["outer_test"]}
        if diff:
            report["differing"][name] = diff
    report["bit_identical"] = not report["differing"]
    return report
