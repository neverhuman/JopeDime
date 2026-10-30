"""Train-only optimal per-column scaling and categorical encoding.

All statistics, quantile knots, skewness transforms, and categorical vocabularies
are fitted strictly on the training split (D_train) and applied without refitting
to unseen validation/test splits (D_test). Raw column headers and rare category
strings are never retained in public artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

QUANTILE_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
NORMAL_GRID = (-2.32634787, -1.64485363, -1.28155157, -0.67448975, 0.0,
               0.67448975, 1.28155157, 1.64485363, 2.32634787)


def _safe_skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    if x.size < 4:
        return 0.0, 0.0
    m, s = float(np.mean(x)), float(np.std(x))
    if not math.isfinite(s) or s <= 1e-12:
        return 0.0, 0.0
    z = (x - m) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    skew = 0.0 if not math.isfinite(skew) else float(np.clip(skew, -50.0, 50.0))
    kurt = 0.0 if not math.isfinite(kurt) else float(np.clip(kurt, -50.0, 200.0))
    return skew, kurt


@dataclass(frozen=True)
class NumericColumnScaler:
    slot: int
    anonymous_name: str
    transform_kind: str
    missing_rate: float
    center: float
    scale: float
    post_mean: float
    post_std: float
    quantiles: Tuple[float, ...]
    skewness: float
    kurtosis: float
    blend_quantile_weight: float = 0.35

    def _raw_transform(self, vals: np.ndarray) -> np.ndarray:
        if self.transform_kind == "constant":
            return np.zeros_like(vals, dtype=np.float64)
        if self.transform_kind == "binary":
            return np.clip((vals - self.center) / max(self.scale, 1e-9), -6.0, 6.0)
        if self.transform_kind == "signed_log1p_hybrid":
            shifted = vals - self.center
            scaled = np.sign(shifted) * np.log1p(np.abs(shifted) / max(self.scale, 1e-9))
            z_log = (scaled - self.post_mean) / max(self.post_std, 1e-9)
        elif self.transform_kind == "asinh_hybrid":
            shifted = vals - self.center
            scaled = np.arcsinh(shifted / max(self.scale, 1e-9))
            z_log = (scaled - self.post_mean) / max(self.post_std, 1e-9)
        else:
            z_log = (vals - self.post_mean) / max(self.post_std, 1e-9)

        q_arr = np.asarray(self.quantiles, dtype=np.float64)
        if np.all(np.diff(q_arr) > 1e-12):
            z_q = np.interp(vals, q_arr, NORMAL_GRID)
            iqr = max(q_arr[5] - q_arr[3], 1e-9)
            low_mask, high_mask = vals < q_arr[0], vals > q_arr[-1]
            if np.any(low_mask):
                z_q[low_mask] = NORMAL_GRID[0] + (vals[low_mask] - q_arr[0]) / iqr * 1.3489795
            if np.any(high_mask):
                z_q[high_mask] = NORMAL_GRID[-1] + (vals[high_mask] - q_arr[-1]) / iqr * 1.3489795
            w = self.blend_quantile_weight
            z = (1.0 - w) * z_log + w * z_q
        else:
            z = z_log
        return 6.0 * np.tanh(z / 6.0)

    def transform(self, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(col, dtype=np.float64).reshape(-1)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=np.float64)
        if np.any(finite):
            out[finite] = self._raw_transform(arr[finite])
        return out, (~finite).astype(np.float64)

    def inverse_transform(self, z_col: np.ndarray) -> np.ndarray:
        z = np.asarray(z_col, dtype=np.float64).reshape(-1)
        z_unclipped = 6.0 * np.arctanh(np.clip(z / 6.0, -0.999999, 0.999999))
        if self.transform_kind == "constant":
            return np.full_like(z, self.center, dtype=np.float64)
        if self.transform_kind == "binary":
            raw = self.center + z_unclipped * self.scale
            q0, q1 = self.quantiles[0], self.quantiles[-1]
            return np.where(raw >= 0.5 * (q0 + q1), q1, q0)
        if self.transform_kind == "signed_log1p_hybrid":
            scaled = self.post_mean + z_unclipped * self.post_std
            return self.center + np.sign(scaled) * np.expm1(np.abs(scaled)) * self.scale
        if self.transform_kind == "asinh_hybrid":
            scaled = self.post_mean + z_unclipped * self.post_std
            return self.center + np.sinh(scaled) * self.scale
        return self.post_mean + z_unclipped * self.post_std


@dataclass(frozen=True)
class CategoricalColumnEncoder:
    slot: int
    anonymous_name: str
    missing_rate: float
    vocab_tokens: Tuple[str, ...]
    token_to_id: Dict[str, int]
    id_target_Encoding: Tuple[float, ...]
    id_frequencies: Tuple[float, ...]
    post_mean: float
    post_std: float

    def transform(self, col: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(col)
        ids = np.zeros(n, dtype=np.int64)
        enc = np.zeros(n, dtype=np.float64)
        miss = np.zeros(n, dtype=np.float64)
        lut_enc = np.asarray(self.id_target_Encoding, dtype=np.float64)
        for i, val in enumerate(col):
            s = "" if (val is None or (isinstance(val, float) and math.isnan(val))) else str(val).strip()
            if s == "" or s.lower() in ("nan", "null", "none", "?", "na"):
                miss[i], idx = 1.0, 0
            else:
                idx = self.token_to_id.get(s, 0)
            ids[i], enc[i] = idx, lut_enc[idx]
        z = 6.0 * np.tanh(((enc - self.post_mean) / max(self.post_std, 1e-9)) / 6.0)
        return z, ids, miss


@dataclass(frozen=True)
class TargetScaler:
    task: str
    mean: float
    std: float
    median: float
    iqr: float
    skewness: float
    kurtosis: float
    positive_label: Optional[str] = None
    class_prior: float = 0.5
    class_labels: Tuple[float, ...] = ()
    class_priors: Tuple[float, ...] = ()

    def transform(self, y: Sequence[Any]) -> np.ndarray:
        if self.task == "multiclass" and self.class_labels:
            labels = np.asarray(self.class_labels, dtype=np.float64)
            arr = np.array([_parse_float(v) for v in y], dtype=np.float64)
            arr = np.nan_to_num(arr, nan=float(labels[0]))
            idx = np.argmin(np.abs(arr[:, None] - labels[None, :]), axis=1)
            return idx.astype(np.float64)
        if self.task == "classification":
            out = np.zeros(len(y), dtype=np.float64)
            for i, v in enumerate(y):
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    out[i] = 1.0 if float(v) > 0.5 else 0.0
                else:
                    out[i] = 1.0 if str(v).strip() == self.positive_label else 0.0
            return out
        arr = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(arr)
        out = np.zeros_like(arr, dtype=np.float64)
        if np.any(finite):
            out[finite] = 8.0 * np.tanh(((arr[finite] - self.mean) / max(self.std, 1e-9)) / 8.0)
        return out

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        arr = np.asarray(y_scaled, dtype=np.float64).reshape(-1)
        if self.task == "multiclass" and self.class_labels:
            labels = np.asarray(self.class_labels, dtype=np.float64)
            idx = np.clip(np.rint(arr).astype(np.int64), 0, len(labels) - 1)
            return labels[idx]
        if self.task == "classification":
            return (arr >= 0.5).astype(np.float64)
        return arr * self.std + self.mean


@dataclass
class FitOnlyTableTransform:
    """Train-only preprocessor that scales numeric columns and encodes categoricals."""
    task: str
    kinds: Tuple[str, ...]
    numeric_scalers: Dict[int, NumericColumnScaler] = field(default_factory=dict)
    categorical_encoders: Dict[int, CategoricalColumnEncoder] = field(default_factory=dict)
    target_scaler: Optional[TargetScaler] = None
    min_level_count: int = 5
    prior_smoothing: float = 10.0

    @classmethod
    def fit(
        cls,
        X_train: Any,
        y_train: Sequence[Any],
        kinds: Optional[Sequence[str]] = None,
        *,
        task: str = "regression",
        min_level_count: int = 5,
        prior_smoothing: float = 10.0,
    ) -> "FitOnlyTableTransform":
        if isinstance(X_train, np.ndarray) and X_train.dtype.kind in ("f", "i", "u"):
            arr = np.asarray(X_train, dtype=np.float64)
            arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            p = int(arr.shape[1])
            inferred = tuple(kinds) if kinds is not None else ("numeric",) * p
            if len(inferred) != p:
                raise ValueError("scaling_feature_kinds_length_mismatch")
            if all(k == "numeric" for k in inferred):
                target_scaler = _fit_target_scaler(y_train, task=task)
                inst = cls(task=task, kinds=inferred, min_level_count=min_level_count,
                           prior_smoothing=prior_smoothing, target_scaler=target_scaler)
                for j in range(p):
                    inst.numeric_scalers[j] = _fit_numeric_column(j, f"f_{j:03d}", arr[:, j])
                return inst

        rows_list, p = _coerce_feature_rows(X_train)
        inferred = _infer_feature_kinds(rows_list, p) if kinds is None else tuple(kinds)
        if len(inferred) != p:
            raise ValueError("scaling_feature_kinds_length_mismatch")

        target_scaler = _fit_target_scaler(y_train, task=task)
        y_num = target_scaler.transform(y_train)
        y_global_mean = float(np.mean(y_num)) if y_num.size > 0 else 0.0
        inst = cls(task=task, kinds=inferred, min_level_count=min_level_count,
                   prior_smoothing=prior_smoothing, target_scaler=target_scaler)
        for j, kind in enumerate(inferred):
            anon = f"f_{j:03d}"
            if kind == "numeric":
                col_vals = np.array([_parse_float(r[j]) for r in rows_list], dtype=np.float64)
                inst.numeric_scalers[j] = _fit_numeric_column(j, anon, col_vals)
            elif kind == "categorical":
                inst.categorical_encoders[j] = _fit_categorical_column(
                    j, anon, [r[j] for r in rows_list], y_num, y_global_mean,
                    min_level_count=min_level_count, prior_smoothing=prior_smoothing)
            else:
                raise ValueError("scaling_unsupported_column_kind")
        return inst

    def transform_features(self, X: Any, *, include_missing_indicators: bool = False) -> np.ndarray:
        if isinstance(X, np.ndarray) and X.dtype.kind in ("f", "i", "u") and all(k == "numeric" for k in self.kinds):
            arr = np.asarray(X, dtype=np.float64)
            arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
            p = int(arr.shape[1])
            if p != len(self.kinds):
                raise ValueError("scaling_feature_count_mismatch")
            if p == 0:
                return np.zeros((arr.shape[0], 0), dtype=np.float64)
            cols_out, miss_out = [], []
            for j in range(p):
                z, miss = self.numeric_scalers[j].transform(arr[:, j])
                cols_out.append(z)
                if include_missing_indicators and self.numeric_scalers[j].missing_rate > 0.0:
                    miss_out.append(miss - self.numeric_scalers[j].missing_rate)
            return np.column_stack(cols_out + miss_out if include_missing_indicators else cols_out).astype(np.float64)

        rows_list, p = _coerce_feature_rows(X)
        if p != len(self.kinds):
            raise ValueError("scaling_feature_count_mismatch")
        n = len(rows_list)
        cols_out, miss_out = [], []
        for j, kind in enumerate(self.kinds):
            if kind == "numeric":
                raw = np.array([_parse_float(r[j]) for r in rows_list], dtype=np.float64)
                z, miss = self.numeric_scalers[j].transform(raw)
                cols_out.append(z)
                if include_missing_indicators and self.numeric_scalers[j].missing_rate > 0.0:
                    miss_out.append(miss - self.numeric_scalers[j].missing_rate)
            else:
                z, _, miss = self.categorical_encoders[j].transform([r[j] for r in rows_list])
                cols_out.append(z)
                if include_missing_indicators and self.categorical_encoders[j].missing_rate > 0.0:
                    miss_out.append(miss - self.categorical_encoders[j].missing_rate)
        if not cols_out:
            return np.zeros((n, 0), dtype=np.float64)
        return np.column_stack(cols_out + miss_out if include_missing_indicators else cols_out).astype(np.float64)

    def transform_target(self, y: Sequence[Any]) -> np.ndarray:
        if self.target_scaler is None:
            raise ValueError("scaling_target_not_fitted")
        return self.target_scaler.transform(y)

    def transform(
        self, X: Any, y: Optional[Sequence[Any]] = None, *, include_missing_indicators: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_scaled = self.transform_features(X, include_missing_indicators=include_missing_indicators)
        y_scaled = self.transform_target(y) if y is not None else None
        return X_scaled, y_scaled

    def inverse_transform(
        self, X_scaled: np.ndarray, y_scaled: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        arr = np.asarray(X_scaled, dtype=np.float64)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        p = min(arr.shape[1], len(self.kinds))
        cols = [
            self.numeric_scalers[j].inverse_transform(arr[:, j]) if self.kinds[j] == "numeric" else arr[:, j]
            for j in range(p)
        ]
        X_raw = np.column_stack(cols) if cols else np.zeros((arr.shape[0], 0), dtype=np.float64)
        y_raw = self.target_scaler.inverse_transform(y_scaled) if (y_scaled is not None and self.target_scaler is not None) else y_scaled
        return X_raw, y_raw

    def anonymous_metadata(self) -> Dict[str, Any]:
        col_summaries = []
        for j, kind in enumerate(self.kinds):
            if kind == "numeric":
                sc = self.numeric_scalers[j]
                col_summaries.append({
                    "slot": j, "name": sc.anonymous_name, "kind": "numeric", "transform": sc.transform_kind,
                    "missing_rate": round(sc.missing_rate, 6), "skewness": round(sc.skewness, 4),
                    "kurtosis": round(sc.kurtosis, 4),
                })
            else:
                ce = self.categorical_encoders[j]
                col_summaries.append({
                    "slot": j, "name": ce.anonymous_name, "kind": "categorical",
                    "transform": "bayesian_shrunk_frequency_pooled", "missing_rate": round(ce.missing_rate, 6),
                    "retained_cardinality": len(ce.vocab_tokens),
                })
        return {"task": self.task, "feature_count": len(self.kinds), "columns": col_summaries}


def _parse_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "null", "none", "?", "na"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _coerce_feature_rows(X: Any) -> Tuple[List[Sequence[Any]], int]:
    if isinstance(X, np.ndarray):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        return X.tolist(), int(X.shape[1])
    rows = list(X)
    return (rows, len(rows[0])) if rows else ([], 0)


def _infer_feature_kinds(rows: Sequence[Sequence[Any]], p: int) -> Tuple[str, ...]:
    kinds = []
    for j in range(p):
        numeric_ok, non_empty = 0, 0
        for r in rows:
            v = r[j]
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() in ("nan", "null", "none", "?", "na"):
                continue
            non_empty += 1
            try:
                float(s)
                numeric_ok += 1
            except ValueError:
                pass
        kinds.append("numeric" if (non_empty == 0 or numeric_ok == non_empty) else "categorical")
    return tuple(kinds)


def _fit_numeric_column(slot: int, anon_name: str, col: np.ndarray) -> NumericColumnScaler:
    finite = col[np.isfinite(col)]
    miss_rate = float(1.0 - (finite.size / max(col.size, 1)))
    if finite.size == 0:
        return NumericColumnScaler(slot, anon_name, "constant", 1.0, 0.0, 1.0, 0.0, 1.0,
                                   tuple(0.0 for _ in QUANTILE_GRID), 0.0, 0.0)
    uniq = np.unique(finite)
    q_vals = tuple(float(v) for v in np.quantile(finite, QUANTILE_GRID))
    skew, kurt = _safe_skew_kurt(finite)
    if uniq.size <= 1:
        return NumericColumnScaler(slot, anon_name, "constant", miss_rate, float(uniq[0]), 1.0, 0.0, 1.0, q_vals, 0.0, 0.0)
    if uniq.size == 2:
        return NumericColumnScaler(slot, anon_name, "binary", miss_rate, float(np.mean(finite)),
                                   max(float(np.std(finite)), 1e-6), 0.0, 1.0, q_vals, skew, kurt, 0.0)

    med = q_vals[4]
    iqr = max(q_vals[5] - q_vals[3], 0.0)
    span90 = max(q_vals[7] - q_vals[1], 0.0)
    tail_ratio = span90 / max(iqr, 1e-9) if iqr > 1e-9 else 1.0
    mad = float(np.median(np.abs(finite - med)))
    robust_scale = max(iqr / 1.3489795, mad * 1.4826, float(np.std(finite)) * 0.25, 1e-6)

    if abs(skew) > 1.0 or kurt > 4.0 or tail_ratio > 4.2:
        shifted = finite - med
        logged = np.sign(shifted) * np.log1p(np.abs(shifted) / robust_scale)
        return NumericColumnScaler(slot, anon_name, "signed_log1p_hybrid", miss_rate, med, robust_scale,
                                   float(np.mean(logged)), max(float(np.std(logged)), 1e-6), q_vals, skew, kurt, 0.35)

    return NumericColumnScaler(slot, anon_name, "quantile_robust_hybrid", miss_rate, med, robust_scale,
                               float(np.mean(finite)), max(float(np.std(finite)), 1e-6), q_vals, skew, kurt, 0.30)


def _fit_categorical_column(
    slot: int, anon_name: str, col: Sequence[Any], y_num: np.ndarray, y_global_mean: float,
    *, min_level_count: int = 5, prior_smoothing: float = 10.0,
) -> CategoricalColumnEncoder:
    n = max(len(col), 1)
    counts: Dict[str, int] = {}
    y_sums: Dict[str, float] = {}
    miss_count = 0
    cleaned: List[str] = []
    for i, val in enumerate(col):
        s = "" if (val is None or (isinstance(val, float) and math.isnan(val))) else str(val).strip()
        if s == "" or s.lower() in ("nan", "null", "none", "?", "na"):
            miss_count += 1
            cleaned.append("")
        else:
            counts[s] = counts.get(s, 0) + 1
            y_sums[s] = y_sums.get(s, 0.0) + float(y_num[i])
            cleaned.append(s)

    retained = sorted([tok for tok, c in counts.items() if c >= min_level_count], key=lambda t: (-counts[t], t))[:127]
    vocab = ("__other__",) + tuple(retained)
    token_to_id = {tok: idx + 1 for idx, tok in enumerate(retained)}
    id_counts = np.zeros(len(vocab), dtype=np.float64)
    id_y_sums = np.zeros(len(vocab), dtype=np.float64)
    for i, s in enumerate(cleaned):
        idx = token_to_id.get(s, 0)
        id_counts[idx] += 1.0
        id_y_sums[idx] += float(y_num[i])

    shrunk = (id_y_sums + prior_smoothing * y_global_mean) / (id_counts + prior_smoothing)
    freqs = id_counts / float(n)
    combined = shrunk + 0.15 * np.linspace(-0.25, 0.25, len(vocab))
    train_enc = np.array([combined[token_to_id.get(s, 0)] for s in cleaned], dtype=np.float64)
    return CategoricalColumnEncoder(
        slot, anon_name, float(miss_count) / float(n), vocab, token_to_id,
        tuple(float(v) for v in combined), tuple(float(v) for v in freqs),
        float(np.mean(train_enc)), max(float(np.std(train_enc)), 1e-6),
    )


def _fit_target_scaler(y: Sequence[Any], *, task: str) -> TargetScaler:
    if task == "multiclass":
        raw = np.array([_parse_float(v) for v in y], dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return TargetScaler("multiclass", 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, None, 0.5, (0.0, 0.5, 1.0), (0.34, 0.33, 0.33))
        u_vals, u_counts = np.unique(np.round(finite, 6), return_counts=True)
        if u_vals.size > 32:
            order = np.argsort(u_counts)[::-1][:32]
            u_vals = np.sort(u_vals[order])
            u_counts = np.array([np.sum(np.abs(finite - v) <= 1e-5) for v in u_vals], dtype=np.float64)
        priors = u_counts.astype(np.float64) / max(float(np.sum(u_counts)), 1.0)
        return TargetScaler(
            "multiclass", float(np.mean(finite)), max(float(np.std(finite)), 1e-6),
            float(np.median(finite)), 1.0, 0.0, 0.0, None, float(priors[0]),
            tuple(float(v) for v in u_vals), tuple(float(p) for p in priors),
        )
    if task == "classification":
        tokens = [str(v).strip() for v in y]
        uniq = sorted(set(tokens))
        if "1" in uniq or "1.0" in uniq:
            pos = "1" if "1" in uniq else "1.0"
        elif "true" in [u.lower() for u in uniq]:
            pos = [u for u in uniq if u.lower() == "true"][0]
        else:
            pos = uniq[-1] if uniq else "1"
        bin_vals = np.array([1.0 if str(v).strip() == pos else 0.0 for v in y], dtype=np.float64)
        prior = float(np.mean(bin_vals)) if bin_vals.size > 0 else 0.5
        return TargetScaler("classification", prior, max(math.sqrt(prior * (1.0 - prior)), 1e-6),
                            1.0 if prior >= 0.5 else 0.0, 1.0, 0.0, 0.0, pos, prior)
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return TargetScaler("regression", 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    m, s = float(np.mean(finite)), max(float(np.std(finite)), 1e-6)
    q25, q50, q75 = [float(v) for v in np.quantile(finite, (0.25, 0.50, 0.75))]
    skew, kurt = _safe_skew_kurt(finite)
    return TargetScaler("regression", m, s, q50, max(q75 - q25, 1e-6), skew, kurt)
