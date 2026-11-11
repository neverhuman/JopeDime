#!/usr/bin/env python3
"""CW-5 Hybrid (`struct_cols.py`):
Builds strictly context-fitted structural columns `S = [S_keys, S_formulas]` for `[X, S]` ICL inference using
`H430`'s `keys_core.py` and `H433`'s `formula_core.py`.

Zero query target is ever passed or read:
  - Up to `MAX_KEYS = 4` keys from `keys_core.fit_keys(X_ctx, y_ctx)` with LOO `u >= MIN_KEY_U = 0.15`:
      * `enc`: inner 5-fold `nested_encode` on `X_ctx`, full-context `apply_stat` on `X_q` (clipped to fit `[y_min-pad, y_max+pad]`)
      * `log_cnt`: `log1p(cnt_ctx)` on `X_ctx`, `log1p(cnt_q * 0.8)` on `X_q` (matched 4/5 inner-fold count scale!)
  - Up to `MAX_FORMULAS = 6` raw-scale formula survivors from `formula_core.screen(Xf_imp, y_ctx, seed)`:
      * `FC.compute(spec, Xf_imp)` on `X_ctx` and `FC.compute(spec, Xq_imp)` on `X_q`
"""
from __future__ import annotations

import numpy as np

import formula_core as FC
import keys_core as K

MAX_KEYS = 4
MIN_KEY_U = 0.15
MAX_FORMULAS = 6


def build_struct_cols(
    X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, seed: int = 436_001
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns `(S_ctx [n_ctx, d_S], S_q [n_q, d_S], meta)`. Both float32-compatible finite float64 arrays."""
    X_ctx = np.asarray(X_ctx, dtype=np.float64)
    y_ctx = np.asarray(y_ctx, dtype=np.float64)
    X_q = np.asarray(X_q, dtype=np.float64)
    n_c, n_q = X_ctx.shape[0], X_q.shape[0]

    cols_c: list[np.ndarray] = []
    cols_q: list[np.ndarray] = []
    names: list[str] = []

    y_lo, y_hi = float(np.min(y_ctx)), float(np.max(y_ctx))
    y_pad = 0.25 * max(y_hi - y_lo, float(np.std(y_ctx)), 1e-9)

    key_specs = [sp for sp in K.fit_keys(X_ctx, y_ctx) if float(sp["u"]) >= MIN_KEY_U][:MAX_KEYS]
    for idx_k, sp in enumerate(key_specs):
        enc_c, cnt_c = K.nested_encode(sp, X_ctx, y_ctx, seed + 17 * idx_k)
        st = K.make_stat(sp, X_ctx, y_ctx)
        enc_q = K.apply_stat(st, X_q)
        cnt_q = K.apply_count(st, X_q) * ((K.INNER - 1.0) / float(K.INNER))

        enc_c = np.clip(np.where(np.isfinite(enc_c), enc_c, float(np.mean(y_ctx))), y_lo - y_pad, y_hi + y_pad)
        enc_q = np.clip(np.where(np.isfinite(enc_q), enc_q, float(np.mean(y_ctx))), y_lo - y_pad, y_hi + y_pad)
        lc_c = np.log1p(np.maximum(cnt_c, 0.0))
        lc_q = np.log1p(np.maximum(cnt_q, 0.0))
        cols_c.extend([enc_c, lc_c])
        cols_q.extend([enc_q, lc_q])
        k_lbl = "_".join(f"x{c}" for c in sp["cols"])
        names.extend([f"key_enc_{k_lbl}", f"key_cnt_{k_lbl}"])

    Xf_imp, Xq_imp = K._impute(X_ctx, X_q)
    surv, _, finfo = FC.screen(Xf_imp, y_ctx, seed + 503)
    seen_fnames = {FC.name(s["spec"]) for s in surv}
    if key_specs and len(surv) < MAX_FORMULAS:
        y_res = y_ctx - cols_c[0]
        if float(np.std(y_res)) > 1e-6 * max(float(np.std(y_ctx)), 1e-9):
            surv_r, _, _ = FC.screen(Xf_imp, y_res, seed + 709)
            for sr in surv_r:
                fn = FC.name(sr["spec"])
                if fn not in seen_fnames:
                    seen_fnames.add(fn)
                    surv.append(sr)
    for s in surv[:MAX_FORMULAS]:
        dc = FC.compute(s["spec"], Xf_imp)
        dq = FC.compute(s["spec"], Xq_imp)
        if np.ptp(dc) > 1e-12 and np.all(np.isfinite(dc)) and np.all(np.isfinite(dq)):
            cols_c.append(dc)
            cols_q.append(dq)
            names.append(f"form_{FC.name(s['spec'])}")

    if cols_c:
        S_ctx = np.column_stack(cols_c)
        S_q = np.column_stack(cols_q)
    else:
        S_ctx = np.zeros((n_c, 0), dtype=np.float64)
        S_q = np.zeros((n_q, 0), dtype=np.float64)

    meta = {
        "n_keys": len(key_specs),
        "best_key_u": round(float(key_specs[0]["u"]), 4) if key_specs else None,
        "n_formulas": int(len(names) - 2 * len(key_specs)),
        "best_formula": finfo["survivors"][0]["name"] if finfo.get("survivors") else None,
        "d_S": int(S_ctx.shape[1]),
        "col_names": names,
    }
    return S_ctx, S_q, meta
