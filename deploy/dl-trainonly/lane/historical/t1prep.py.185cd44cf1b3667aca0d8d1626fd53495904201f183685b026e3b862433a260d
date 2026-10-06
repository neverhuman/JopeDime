"""T1 preparation inverse and train-only re-fitting (finding T1-PREP-01).

The T1 corpus files are not raw. The acquisition tool (meta.json `acquisition.tool`, e.g.
models.tools.acquire_regression_extra_sources) fitted these steps over every cleaned source row and recorded them in
meta.json:
    feature_encodings[name] = {kind: numeric, impute_value}
                            | {kind: categorical_codes, impute_value, mapping {key: code}}       (explicit __MISSING__ code)
                            | {kind: categorical_codes, impute_value, categories [key, ...]}   (code = list index; missing
                              cells were imputed with the code impute_value, so they look like a real category)
                            | {kind: categorical_codes, mapping, unseen: -1}                   (map fitted on the T1 train
                              split; a category it never saw is coded `unseen`, its value lost: decoded as missing)
    feature_scaling         = {method, min[], max[], constant_columns, range [0, 1]}   (per output_headers[:-1])
    target_scaling          = {min, max}
then printed every value as float32 '%.9g'. research_2k copies those T1 lines verbatim, and v1 also selected features
(audit_target_leakage.py: exact/affine target copies) over train and test rows together.

Per (dataset, outer fold) this module re-fits what can be re-fitted from T1 values on outer-train rows only:
- categorical columns: the T1 value is de-scaled to its code (min + v*(max-min), rounded to the nearest integer and
  checked), the code is mapped back to the source category key, and a NEW code map is fitted on the outer-train keys
  (sorted, numeric order when every key is numeric); missing ('__MISSING__') and categories unseen in outer-train both
  map to one extra code. Outer-test keys never enter the map.
- target-copy feature selection: exact or affine copies of the target are found on outer-train rows only.
- numeric columns and the target keep their T1 values; trainonly.prepare_split re-fits min-max on outer-train rows.

What T1 values cannot undo is counted, never hidden:
- `possibly_imputed`: numeric cells whose de-scaled value equals the all-row impute_value within the float32 printing
  error. A real value equal to the impute value cannot be told apart, so this is an upper bound on imputed cells.
- the float32 rounding grid of every numeric T1 value was set by all-row constants; train-only min-max on T1 values is
  mathematically the raw train-only min-max but not bit-identical to it. Only the original source rows (L7a) remove it.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dltrain import trainonly

SCHEMA = "jope.dl-t1prep/v1"
MISSING_KEY = "__MISSING__"
CODE_TOL = 1e-3            # a de-scaled code must sit this close to an integer, or decoding refuses
F32_PRINT_REL = 2.0 ** -23  # float32 '%.9g' round trip: half an ulp of a value in [0, 1] is at most 2^-24; doubled
AFFINE_REL_RMSE = 1e-10     # audit_target_leakage.py threshold


class T1PrepError(ValueError):
    pass


def _num(key: str) -> float | None:
    try:
        v = float(key)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def layout(meta: Mapping[str, Any], width: int | None = None) -> list[dict]:
    """One spec per feature column of the T1 file, in file order. The file is output_headers wide, or, in some metas,
    feature_columns + target wide (columns dropped after scaling); like dope-kernel, the width decides. Scaling entries
    are indexed by output_headers[:-1] and are looked up by column name."""
    headers = meta.get("output_headers") or []
    scaled = headers[:-1]
    fs = meta.get("feature_scaling") or {}
    lo, hi = fs.get("min") or [], fs.get("max") or []
    if not scaled or len(lo) != len(scaled) or len(hi) != len(scaled):
        raise T1PrepError(f"feature_scaling has {len(lo)}/{len(hi)} entries for {len(scaled)} output columns")
    fc = meta.get("feature_columns") or []
    if width is None or width == len(headers):
        names = scaled
    elif fc and width == len(fc) + 1 and all(n in scaled for n in fc) and len(set(fc)) == len(fc):
        names = list(fc)
    else:
        raise T1PrepError(f"T1 width {width} matches neither output_headers ({len(headers)}) nor feature_columns+target ({len(fc) + 1})")
    pos = {n: i for i, n in enumerate(scaled)}
    if list(fs.get("range") or []) != [0.0, 1.0]:
        raise T1PrepError(f"unexpected feature_scaling range {fs.get('range')}")
    enc = meta.get("feature_encodings") or {}
    specs = []
    for j, name in enumerate(names):
        e = enc.get(name) or {"kind": "numeric"}
        kind = e.get("kind")
        if kind not in ("numeric", "categorical_codes"):
            raise T1PrepError(f"column {name}: unknown encoding kind {kind!r}")
        spec = {"index": j, "name": name, "kind": kind, "lo": float(lo[pos[name]]), "hi": float(hi[pos[name]]),
                "impute_value": e.get("impute_value")}
        if kind == "categorical_codes":
            if ("mapping" in e) == ("categories" in e):
                raise T1PrepError(f"column {name}: categorical encoding needs exactly one of mapping or categories")
            inverse = {}
            if "categories" in e:
                for code, key in enumerate(e["categories"]):
                    inverse[code] = str(key)
                if len(set(inverse.values())) != len(inverse):
                    raise T1PrepError(f"column {name}: duplicate categories")
                imp = e.get("impute_value")
                spec["impute_code"] = int(imp) if isinstance(imp, (int, float)) and float(imp).is_integer() else None
            else:
                for key, code in e["mapping"].items():
                    if not isinstance(code, int) or isinstance(code, bool) or code in inverse:
                        raise T1PrepError(f"column {name}: mapping is not one-to-one over integer codes")
                    inverse[code] = key
                spec["impute_code"] = None
                if "unseen" in e:
                    u = e["unseen"]
                    if not isinstance(u, int) or isinstance(u, bool) or u in inverse:
                        raise T1PrepError(f"column {name}: unseen code must be an integer outside the mapping")
                    spec["unseen_code"] = u
            if sorted(inverse) != list(range(len(inverse))):
                raise T1PrepError(f"column {name}: codes are not contiguous from 0")
            spec["inverse"] = inverse
        specs.append(spec)
    return specs


def _descale(col, spec):
    span = spec["hi"] - spec["lo"]
    return np.asarray(col, dtype=np.float64) * span + spec["lo"] if span != 0 else np.full(len(col), spec["lo"])


def decode_categorical(col, spec) -> list:
    """Source category keys (None for missing) from one categorical T1 column."""
    code_f = _descale(col, spec)
    code = np.rint(code_f)
    bad = np.abs(code_f - code) > CODE_TOL
    if bool(np.any(bad)):
        raise T1PrepError(f"column {spec['name']}: {int(bad.sum())} values are not integer codes after de-scaling")
    out = []
    for c in code.astype(np.int64).tolist():
        if c == spec.get("unseen_code"):
            out.append(None)          # unseen when the T1 map was fitted: the source category is lost
            continue
        if c not in spec["inverse"]:
            raise T1PrepError(f"column {spec['name']}: code {c} is not in the mapping")
        key = spec["inverse"][c]
        out.append(None if key == MISSING_KEY else key)
    return out


def possibly_imputed(col, spec) -> np.ndarray:
    """Numeric cells that may hold the all-row impute value (counting only; an upper bound)."""
    imp = spec.get("impute_value")
    if spec["kind"] != "numeric" or imp is None or not math.isfinite(float(imp)):
        return np.zeros(len(col), dtype=bool)
    span = abs(spec["hi"] - spec["lo"])
    tol = F32_PRINT_REL * max(span, 1e-300) * 2.0
    return np.abs(_descale(col, spec) - float(imp)) <= tol


def fit_category_codes(train_keys: Sequence) -> dict:
    """Train-only code map: the distinct non-missing outer-train keys, numeric order when all are numeric."""
    keys = sorted({k for k in train_keys if k is not None})
    vals = [_num(k) for k in keys]
    if keys and all(v is not None for v in vals):
        keys = [k for _, k in sorted(zip(vals, keys))]
    return {k: i for i, k in enumerate(keys)}


def apply_category_codes(keys: Sequence, codes: Mapping) -> np.ndarray:
    other = float(len(codes))  # missing, and categories unseen in outer-train
    return np.asarray([float(codes[k]) if k is not None and k in codes else other for k in keys], dtype=np.float64)


def target_copy_columns(train_block) -> list[int]:
    """Feature columns that copy the target on the outer-train rows: exact equality, or an affine fit with relative
    RMSE <= 1e-10 (audit_target_leakage.py, here fitted on outer-train rows only)."""
    b = np.asarray(train_block, dtype=np.float64)
    x, y = b[:, :-1], b[:, -1]
    y_sd = float(np.std(y))
    out = []
    for j in range(x.shape[1]):
        v = x[:, j]
        if np.array_equal(v, y):
            out.append(j)
            continue
        v_sd = float(np.std(v))
        if v_sd <= 0 or y_sd <= 0:
            continue
        slope = float(np.cov(v, y, ddof=0)[0, 1] / (v_sd * v_sd))
        intercept = float(np.mean(y) - slope * np.mean(v))
        if float(np.sqrt(np.mean((y - (slope * v + intercept)) ** 2)) / y_sd) <= AFFINE_REL_RMSE:
            out.append(j)
    return out


def refit(matrix, meta: Mapping[str, Any], outer_train: Sequence[int]) -> tuple[np.ndarray, dict]:
    """T1 matrix (columns = output_headers, target last) -> matrix with categorical columns re-coded from outer-train
    rows only and target-copy columns removed (detected on outer-train rows only), plus the strata report."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise T1PrepError("matrix must be 2-d")
    specs = layout(meta, m.shape[1])
    if m.shape[1] != len(specs) + 1:
        raise T1PrepError(f"matrix has {m.shape[1] if m.ndim == 2 else '?'} columns, meta has {len(specs) + 1}")
    tr = [int(i) for i in outer_train]
    tr_set = set(tr)
    out = m.copy()
    cat_report, imputed = {}, np.zeros(m.shape[0], dtype=np.int64)
    for s in specs:
        j = s["index"]
        if s["kind"] == "categorical_codes":
            keys = decode_categorical(m[:, j], s)
            if s.get("impute_code") is not None:   # list format: mode-imputed cells look real; count them as possible
                imputed += np.rint(_descale(m[:, j], s)).astype(np.int64) == s["impute_code"]
            codes = fit_category_codes([keys[i] for i in tr])
            out[:, j] = apply_category_codes(keys, codes)
            unseen = sum(1 for i, k in enumerate(keys) if i not in tr_set and k is not None and k not in codes)
            lost = int(np.sum(np.rint(_descale(m[:, j], s)).astype(np.int64) == s["unseen_code"])) if s.get("unseen_code") is not None else 0
            cat_report[s["name"]] = {
                "train": {"categories": len(codes), "missing": sum(1 for i in tr if keys[i] is None),
                          "map_sha256": hashlib.sha256(repr(sorted(codes.items())).encode()).hexdigest()},
                "test": {"missing": sum(1 for i, k in enumerate(keys) if i not in tr_set and k is None), "unseen": unseen},
                "t1_unseen_cells_all_rows": lost}
        else:
            imputed += possibly_imputed(m[:, j], s)
    drop = target_copy_columns(out[tr])
    keep = [j for j in range(len(specs)) if j not in set(drop)] + [len(specs)]
    test_rows = [i for i in range(m.shape[0]) if i not in tr_set]
    report = {"schema": SCHEMA, "categorical": cat_report, "target_copy_dropped": [specs[j]["name"] for j in drop],
              "kept_columns": keep[:-1],
              "possibly_imputed_cells": {"train": int(imputed[tr].sum()), "test": int(imputed[test_rows].sum())},
              "numeric_rounding_residual": "float32 %.9g grid set by all-row constants (not removable from T1 values)",
              "feature_scaling_method": (meta.get("feature_scaling") or {}).get("method"),
              "downsampling": (meta.get("downsampling") or {}).get("method")}
    return np.ascontiguousarray(out[:, keep]), report


def prepare(matrix, meta: Mapping[str, Any], outer_train: Sequence[int], outer_test: Sequence[int], **kw) -> dict:
    """refit() then trainonly.prepare_split() on the re-fitted columns; the report rides along as split['t1prep']."""
    fitted, report = refit(matrix, meta, outer_train)
    split = trainonly.prepare_split(fitted, outer_train, outer_test, **kw)
    split["t1prep"] = report
    return split
