#!/usr/bin/env python3
"""H450 Context-Only Relational, Lead-Lag Neighbor, Sentinel & Colinear-Difference Engine (`relational_struct_h450.py`).

Strictly fitted on `(X_ctx, y_ctx)` and applied to `X_q` (`0` query target reads; `float64` throughout):
  1. Automatic Sentinel / Point-Mass Detector (`-99.99`, `-999`, `99999`, tail point masses):
     - Emits top binary sentinel flags `is_sentinel_j` and masks sentinel entries to `NaN` in `X_ctx_clean, X_q_clean`
       so continuous differences, linear/kernel projections, and neural scaling are never corrupted.
  2. Exact `float64` Colinear Offset Differences (`x_b - x_a`) & Signed-Log (`sign(x)*log1p(|x|)`):
     - Captures exact durations between epoch-day / Unix-timestamp columns (`corr > 0.97`, `|median| > 25`) and
       stabilizes multi-order-of-magnitude positive/skewed predictors.
  3. Within-Entity & Global 1D Ordered Lead-Lag Neighbor Interpolation (`y_interp`, `y_nearest`, `local_slope`, `log_dt`):
     - Screens `(entity_key K, order_col T)` pairs strictly by leave-one-out (`LOO`) utility on `(X_ctx, y_ctx)`.
     - Uses strict self-excluding LOO bracketing neighbors `(s-1, s+1)` on `X_ctx` and matched full-context bracketing
       neighbors on `X_q` (`0` target leakage!).
  4. Exact Keys (`keys_core`) & Raw Formulas (`formula_core`) via `struct_cols.build_struct_cols` on `X_clean`.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "/home/ubuntu/h448-stage/pub-r1/code" not in sys.path and Path("/home/ubuntu/h448-stage/pub-r1/code").is_dir():
    sys.path.append("/home/ubuntu/h448-stage/pub-r1/code")

import struct_cols as SC  # noqa: E402

KNOWN_SENTINELS = (-99.99, -99.0, -999.0, -999.9, -9999.0, -99999.0, 9999.0, 99999.0, 999999.0)


def _abs_corr_safe(x: np.ndarray, y_centered: np.ndarray, y_norm: float) -> float:
    fin = np.isfinite(x)
    cnt = int(np.sum(fin))
    if cnt < 12 or y_norm <= 1e-12:
        return 0.0
    xf = x[fin] - float(np.mean(x[fin]))
    xn = float(np.linalg.norm(xf))
    if xn <= 1e-12:
        return 0.0
    return abs(float(np.dot(xf, y_centered[fin])) / (xn * y_norm))


def detect_and_mask_sentinels(
    X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray, max_flags: int = 4
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[str]]:
    """Detects extreme point-mass sentinel values strictly on `X_ctx`, masks them to `NaN` in `X_clean`,
    and returns up to `max_flags` binary indicator columns ranked by correlation with `y_ctx`."""
    n_c, p = X_ctx.shape
    n_q = X_q.shape[0]
    Xc = X_ctx.astype(np.float64, copy=True)
    Xq = X_q.astype(np.float64, copy=True)
    yc = y_ctx - float(np.mean(y_ctx))
    yn = float(np.linalg.norm(yc))

    cand_flags = []
    for j in range(p):
        col = Xc[:, j]
        fin = np.isfinite(col)
        n_fin = int(np.sum(fin))
        if n_fin < 20:
            continue
        vals = col[fin]
        u_vals, counts = np.unique(vals, return_counts=True)
        if len(u_vals) < 6:
            continue
        q25, q50, q75 = np.quantile(vals, [0.25, 0.50, 0.75])
        iqr = max(float(q75 - q25), 1e-9)

        # Check extreme low and high atoms
        for idx_atom in (0, len(u_vals) - 1):
            v_atom = float(u_vals[idx_atom])
            cnt = int(counts[idx_atom])
            if cnt < 3 or cnt > int(0.45 * n_fin):
                continue
            is_known = any(abs(v_atom - ks) <= 1e-4 * max(1.0, abs(ks)) for ks in KNOWN_SENTINELS)
            dist_iqr = abs(v_atom - float(q50)) / iqr
            # Check gap to next adjacent unique value
            adj_val = float(u_vals[1] if idx_atom == 0 else u_vals[-2])
            gap_iqr = abs(v_atom - adj_val) / iqr
            if (is_known and dist_iqr > 2.5) or (dist_iqr > 6.0 and gap_iqr > 3.5):
                tol = 1e-7 * max(1.0, abs(v_atom))
                m_c = np.isfinite(Xc[:, j]) & (np.abs(Xc[:, j] - v_atom) <= tol)
                m_q = np.isfinite(Xq[:, j]) & (np.abs(Xq[:, j] - v_atom) <= tol)
                Xc[m_c, j] = np.nan
                Xq[m_q, j] = np.nan
                flag_c = m_c.astype(np.float64)
                flag_q = m_q.astype(np.float64)
                rel = _abs_corr_safe(flag_c, yc, yn)
                cand_flags.append((rel, j, v_atom, flag_c, flag_q))

    cand_flags.sort(key=lambda t: -t[0])
    out_c, out_q, names = [], [], []
    for _, j, v_atom, fc, fq in cand_flags[:max_flags]:
        out_c.append(fc)
        out_q.append(fq)
        names.append(f"sent_x{j}_{v_atom:g}")
    return Xc, Xq, out_c, out_q, names


def build_diffs_and_signed_logs(
    Xc: np.ndarray, y_ctx: np.ndarray, Xq: np.ndarray, max_diffs: int = 3, max_logs: int = 3
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Extracts exact `float64` colinear offset differences `x_b - x_a` (e.g. epoch-day / timestamp durations)
    and signed-log transforms for multi-order-of-magnitude skewed predictors."""
    n_c, p = Xc.shape
    yc = y_ctx - float(np.mean(y_ctx))
    yn = float(np.linalg.norm(yc))
    out_c, out_q, names = [], [], []
    if p == 0 or n_c < 20:
        return out_c, out_q, names

    # 1. Screen offset-heavy continuous columns for colinear differences
    meds = np.zeros(p, dtype=np.float64)
    sds = np.zeros(p, dtype=np.float64)
    valid_cols = []
    for j in range(min(p, 64)):
        fin = np.isfinite(Xc[:, j])
        if int(np.sum(fin)) >= int(0.7 * n_c):
            v = Xc[fin, j]
            meds[j] = float(np.median(v))
            sds[j] = float(np.std(v))
            if sds[j] > 1e-9 and len(np.unique(v)) >= 12:
                valid_cols.append(j)

    diff_cands = []
    offset_cols = [j for j in valid_cols if abs(meds[j]) > 20.0 or sds[j] > 500.0][:24]
    for idx_a in range(len(offset_cols)):
        a = offset_cols[idx_a]
        for idx_b in range(idx_a + 1, len(offset_cols)):
            b = offset_cols[idx_b]
            ratio_sd = min(sds[a], sds[b]) / max(sds[a], sds[b])
            if ratio_sd < 0.25:
                continue
            both = np.isfinite(Xc[:, a]) & np.isfinite(Xc[:, b])
            if int(np.sum(both)) < 20:
                continue
            ca = Xc[both, a] - float(np.mean(Xc[both, a]))
            cb = Xc[both, b] - float(np.mean(Xc[both, b]))
            denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
            if denom <= 1e-12:
                continue
            rho = float(np.dot(ca, cb)) / denom
            if rho >= 0.965:
                dc = Xc[:, b] - Xc[:, a]
                dq = Xq[:, b] - Xq[:, a]
                med_d = float(np.nanmedian(dc)) if np.any(np.isfinite(dc)) else 0.0
                dc_f = np.where(np.isfinite(dc), dc, med_d)
                dq_f = np.where(np.isfinite(dq), dq, med_d)
                if float(np.std(dc_f)) > 1e-9:
                    rel_d = _abs_corr_safe(dc_f, yc, yn)
                    diff_cands.append((rel_d, a, b, dc_f, dq_f))

    diff_cands.sort(key=lambda t: -t[0])
    for _, a, b, dc_f, dq_f in diff_cands[:max_diffs]:
        out_c.append(dc_f)
        out_q.append(dq_f)
        names.append(f"diff64_x{b}_x{a}")

    # 2. Signed-log for heavy-tailed predictors where log-correlation beats linear correlation
    log_cands = []
    for j in valid_cols:
        fin = np.isfinite(Xc[:, j])
        v = Xc[fin, j]
        q50_abs = max(float(np.median(np.abs(v))), 1e-6)
        q99_abs = float(np.quantile(np.abs(v), 0.99))
        if q99_abs / q50_abs < 12.0 and float(np.max(np.abs(v))) < 500.0:
            continue
        med_j = meds[j]
        xc_imp = np.where(fin, Xc[:, j], med_j)
        xq_imp = np.where(np.isfinite(Xq[:, j]), Xq[:, j], med_j)
        lc = np.sign(xc_imp) * np.log1p(np.abs(xc_imp))
        lq = np.sign(xq_imp) * np.log1p(np.abs(xq_imp))
        r_raw = _abs_corr_safe(xc_imp, yc, yn)
        r_log = _abs_corr_safe(lc, yc, yn)
        if r_log > r_raw + 0.015:
            log_cands.append((r_log - r_raw, j, lc, lq))

    log_cands.sort(key=lambda t: -t[0])
    for _, j, lc, lq in log_cands[:max_logs]:
        out_c.append(lc)
        out_q.append(lq)
        names.append(f"slog_x{j}")

    return out_c, out_q, names


def _encode_discrete_entity(col_c: np.ndarray, col_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fin_c = np.isfinite(col_c)
    sc = max(float(np.max(np.abs(col_c[fin_c]))), 1.0) if np.any(fin_c) else 1.0
    kc = np.full(len(col_c), -999999999, dtype=np.int64)
    kq = np.full(len(col_q), -999999999, dtype=np.int64)
    if np.any(fin_c):
        kc[fin_c] = np.round(np.clip(col_c[fin_c] / sc, -1e9, 1e9) * 1e7).astype(np.int64)
    fin_q = np.isfinite(col_q)
    if np.any(fin_q):
        kq[fin_q] = np.round(np.clip(col_q[fin_q] / sc, -1e9, 1e9) * 1e7).astype(np.int64)
    return kc, kq


def _eval_leadlag_pair(
    kc: np.ndarray,
    tc: np.ndarray,
    yc: np.ndarray,
    kq: np.ndarray,
    tq: np.ndarray,
    y_mean: float,
    y_lo: float,
    y_hi: float,
    shrink_m: float = 4.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes strict self-excluding LOO 1D neighbor interpolation along `tc` within groups `kc` on context rows,
    and matched full-context lookup on query rows `(kq, tq)`."""
    n_c = len(yc)
    n_q = len(tq)
    pred_c = np.full(n_c, y_mean, dtype=np.float64)
    ldist_c = np.full(n_c, 4.0, dtype=np.float64)
    pred_q = np.full(n_q, y_mean, dtype=np.float64)
    ldist_q = np.full(n_q, 4.0, dtype=np.float64)

    valid_c = np.isfinite(tc) & (kc != -999999999)
    if int(np.sum(valid_c)) < 16:
        return -1.0, pred_c, ldist_c, pred_q, ldist_q

    idx_c = np.flatnonzero(valid_c)
    # Sort context rows by (kc, tc) once in O(N log N)
    order = np.lexsort((tc[idx_c], kc[idx_c]))
    s_idx = idx_c[order]
    s_k = kc[s_idx]
    s_t = tc[s_idx]
    s_y = yc[s_idx]

    # Partition into entity groups
    splits = np.flatnonzero(np.diff(s_k) != 0) + 1
    starts = np.concatenate([[0], splits])
    ends = np.concatenate([splits, [len(s_idx)]])

    group_tables: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    dts_all = []

    for st, en in zip(starts.tolist(), ends.tolist()):
        m = en - st
        g_code = int(s_k[st])
        t_g = s_t[st:en]
        y_g = s_y[st:en]
        orig_g = s_idx[st:en]
        sum_y = float(np.sum(y_g))

        if m == 1:
            pred_c[orig_g[0]] = y_mean
            group_tables[g_code] = (t_g, y_g, (sum_y + shrink_m * y_mean) / (1.0 + shrink_m), 1.0)
            continue

        diffs = np.diff(t_g)
        pos_diffs = diffs[diffs > 1e-12]
        med_dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
        dts_all.append(med_dt)
        tau = max(3.5 * med_dt, 1e-12)

        # Strict LOO entity mean for each position s in 0..m-1
        loo_gmean = (sum_y - y_g + shrink_m * y_mean) / (float(m - 1) + shrink_m)

        if m == 2:
            dt01 = abs(float(t_g[1] - t_g[0]))
            w = math.exp(-dt01 / tau)
            pred_c[orig_g[0]] = w * y_g[1] + (1.0 - w) * loo_gmean[0]
            pred_c[orig_g[1]] = w * y_g[0] + (1.0 - w) * loo_gmean[1]
            ldist_c[orig_g] = math.log1p(dt01 / max(med_dt, 1e-12))
        else:
            # Vectorized LOO bracketing neighbors (s-1 and s+1) for interior points 1..m-2
            y_interp = np.empty(m, dtype=np.float64)
            dt_near = np.empty(m, dtype=np.float64)

            # Left endpoint s = 0 -> neighbor is s = 1
            y_interp[0] = y_g[1]
            dt_near[0] = abs(float(t_g[1] - t_g[0]))
            # Right endpoint s = m-1 -> neighbor is s = m-2
            y_interp[-1] = y_g[-2]
            dt_near[-1] = abs(float(t_g[-1] - t_g[-2]))

            # Interior points 1..m-2 -> bracketed by s-1 and s+1
            t_l = t_g[:-2]
            t_r = t_g[2:]
            t_mid = t_g[1:-1]
            y_l = y_g[:-2]
            y_r = y_g[2:]
            d_l = np.maximum(t_mid - t_l, 0.0)
            d_r = np.maximum(t_r - t_mid, 0.0)
            span_lr = d_l + d_r
            interp_mid = np.where(span_lr > 1e-12, (d_r * y_l + d_l * y_r) / np.maximum(span_lr, 1e-12), 0.5 * (y_l + y_r))
            y_interp[1:-1] = interp_mid
            dt_near[1:-1] = np.minimum(d_l, d_r)

            w_vec = np.exp(-dt_near / tau)
            pred_c[orig_g] = w_vec * y_interp + (1.0 - w_vec) * loo_gmean
            ldist_c[orig_g] = np.log1p(dt_near / max(med_dt, 1e-12))

        full_gmean = (sum_y + shrink_m * y_mean) / (float(m) + shrink_m)
        # Scale med_dt by (m-1)/m for query lookup to match LOO spacing density
        group_tables[g_code] = (t_g, y_g, full_gmean, med_dt * (float(m - 1) / float(m)))

    pred_c = np.clip(pred_c, y_lo, y_hi)
    rmse_loo = float(np.sqrt(np.mean((pred_c - yc) ** 2)))
    sy = max(float(np.std(yc)), 1e-12)
    u_loo = -math.log(max(rmse_loo / sy, 1e-12))
    if u_loo < 0.06 or n_q == 0:
        return u_loo, pred_c, ldist_c, pred_q, ldist_q

    # Apply matched lookup to query rows `(kq, tq)` using ONLY context `group_tables` (`0` query target reads!)
    valid_q = np.isfinite(tq) & (kq != -999999999)
    idx_q = np.flatnonzero(valid_q)
    if len(idx_q) > 0:
        order_q = np.argsort(kq[idx_q], kind="mergesort")
        sq_idx = idx_q[order_q]
        sq_k = kq[sq_idx]
        sq_t = tq[sq_idx]
        q_splits = np.flatnonzero(np.diff(sq_k) != 0) + 1
        q_starts = np.concatenate([[0], q_splits])
        q_ends = np.concatenate([q_splits, [len(sq_idx)]])

        for qst, qen in zip(q_starts.tolist(), q_ends.tolist()):
            gk = int(sq_k[qst])
            hit = group_tables.get(gk)
            if hit is None:
                continue
            t_g, y_g, full_gmean, med_dt_q = hit
            oq = sq_idx[qst:qen]
            tv = sq_t[qst:qen]
            m = len(t_g)
            if m == 1:
                dt1 = np.abs(tv - t_g[0])
                tau_q = max(3.5 * med_dt_q, 1e-12)
                wq = np.exp(-dt1 / tau_q)
                pred_q[oq] = wq * y_g[0] + (1.0 - wq) * full_gmean
                ldist_q[oq] = np.log1p(dt1 / max(med_dt_q, 1e-12))
                continue

            # Aggregate identical t_g values for clean tie-handling on query lookup
            pos_r = np.searchsorted(t_g, tv, side="left")
            pos_l = pos_r - 1
            has_l = pos_l >= 0
            has_r = pos_r < m
            cl_l = np.clip(pos_l, 0, m - 1)
            cl_r = np.clip(pos_r, 0, m - 1)
            tl = t_g[cl_l]
            tr = t_g[cl_r]
            yl = y_g[cl_l]
            yr = y_g[cl_r]

            dl = np.where(has_l, np.maximum(tv - tl, 0.0), np.inf)
            dr = np.where(has_r, np.maximum(tr - tv, 0.0), np.inf)
            dt_min = np.minimum(dl, dr)

            both = has_l & has_r
            dl_b = np.where(both, dl, 1.0)
            dr_b = np.where(both, dr, 1.0)
            span_b = dl_b + dr_b
            y_int = np.where(
                both,
                np.where(span_b > 1e-12, (dr_b * yl + dl_b * yr) / np.maximum(span_b, 1e-12), 0.5 * (yl + yr)),
                np.where(has_l, yl, yr),
            )
            tau_q = max(3.5 * med_dt_q, 1e-12)
            wq = np.exp(-dt_min / tau_q)
            pred_q[oq] = wq * y_int + (1.0 - wq) * full_gmean
            ldist_q[oq] = np.log1p(dt_min / max(med_dt_q, 1e-12))

    pred_q = np.clip(pred_q, y_lo, y_hi)
    return u_loo, pred_c, ldist_c, pred_q, ldist_q


def build_leadlag_cols(
    Xc: np.ndarray, y_ctx: np.ndarray, Xq: np.ndarray, max_pairs: int = 3
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict]:
    """Screens `(entity_key K, order_col T)` pairs on `(Xc, y_ctx)` and emits top LOO-validated lead-lag columns."""
    n_c, p = Xc.shape
    n_q = Xq.shape[0]
    if n_c < 24 or p == 0:
        return [], [], [], {"leadlag_pairs": 0, "best_leadlag_u": None}

    y_mean = float(np.mean(y_ctx))
    y_lo_raw, y_hi_raw = float(np.min(y_ctx)), float(np.max(y_ctx))
    y_pad = 0.25 * max(y_hi_raw - y_lo_raw, float(np.std(y_ctx)), 1e-9)
    y_lo, y_hi = y_lo_raw - y_pad, y_hi_raw + y_pad

    yc = y_ctx - y_mean
    yn = float(np.linalg.norm(yc))

    # Candidate ordering columns: high unique count continuous columns
    order_cands = []
    entity_cands = []
    for j in range(min(p, 48)):
        fin = np.isfinite(Xc[:, j])
        n_fin = int(np.sum(fin))
        if n_fin < int(0.65 * n_c):
            continue
        u_cnt = len(np.unique(Xc[fin, j]))
        rel = _abs_corr_safe(Xc[:, j], yc, yn)
        if u_cnt >= max(20, int(0.12 * n_c)):
            # Prefer near-monotonic / high-cardinality index or timestamp columns (even when raw corr with y is 0!)
            card_ratio = float(u_cnt) / float(n_fin)
            order_cands.append((card_ratio + 0.5 * rel, j))
        if 2 <= u_cnt <= min(450, max(3, n_c // 2)):
            entity_cands.append((rel, u_cnt, j))

    order_cands.sort(key=lambda t: -t[0])
    top_orders = [j for _, j in order_cands[:3]]
    if not top_orders:
        return [], [], [], {"leadlag_pairs": 0, "best_leadlag_u": None}

    entity_cands.sort(key=lambda t: -t[0])
    top_entities = [j for _, _, j in entity_cands[:3]]

    # Evaluate global ordering (`K = 0`) + top `(entity K, order T)` pairs
    kc_glob = np.zeros(n_c, dtype=np.int64)
    kq_glob = np.zeros(n_q, dtype=np.int64)
    encoded_entities = {ej: _encode_discrete_entity(Xc[:, ej], Xq[:, ej]) for ej in top_entities}

    scored = []
    for tj in top_orders:
        u_g, pc_g, dc_g, pq_g, dq_g = _eval_leadlag_pair(kc_glob, Xc[:, tj], y_ctx, kq_glob, Xq[:, tj], y_mean, y_lo, y_hi)
        if u_g >= 0.06:
            scored.append((u_g, f"glob_t{tj}", pc_g, dc_g, pq_g, dq_g))
        for ej in top_entities:
            if ej == tj:
                continue
            kc_e, kq_e = encoded_entities[ej]
            u_e, pc_e, dc_e, pq_e, dq_e = _eval_leadlag_pair(kc_e, Xc[:, tj], y_ctx, kq_e, Xq[:, tj], y_mean, y_lo, y_hi)
            if u_e >= 0.06:
                scored.append((u_e, f"ent{ej}_t{tj}", pc_e, dc_e, pq_e, dq_e))

    scored.sort(key=lambda r: -r[0])
    out_c, out_q, names = [], [], []
    for u_val, tag, pc, dc, pq, dq in scored[:max_pairs]:
        out_c.extend([pc, dc])
        out_q.extend([pq, dq])
        names.extend([f"ll_y_{tag}", f"ll_dt_{tag}"])

    meta = {
        "leadlag_pairs": len(scored[:max_pairs]),
        "best_leadlag_u": round(float(scored[0][0]), 4) if scored else None,
    }
    return out_c, out_q, names, meta


def build_h450_struct_cols(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_q: np.ndarray,
    seed: int = 450_001,
    fast_mode: bool = False,
    max_total_cols: int = 18,
    include_base_struct: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Builds `(X_ctx_clean, X_q_clean, S_ctx [n_c, d_S], S_q [n_q, d_S], meta)` in `float64` strictly from `(X_ctx, y_ctx) -> X_q`."""
    X_ctx = np.asarray(X_ctx, dtype=np.float64)
    y_ctx = np.asarray(y_ctx, dtype=np.float64)
    X_q = np.asarray(X_q, dtype=np.float64)
    n_c, n_q = X_ctx.shape[0], X_q.shape[0]

    Xc_clean, Xq_clean, sc_sent, sq_sent, n_sent = detect_and_mask_sentinels(
        X_ctx, y_ctx, X_q, max_flags=(2 if fast_mode else 4)
    )
    sc_dl, sq_dl, n_dl = build_diffs_and_signed_logs(
        Xc_clean, y_ctx, Xq_clean, max_diffs=(2 if fast_mode else 3), max_logs=(2 if fast_mode else 3)
    )
    sc_ll, sq_ll, n_ll, ll_meta = build_leadlag_cols(
        Xc_clean, y_ctx, Xq_clean, max_pairs=(2 if fast_mode else 3)
    )

    cols_c = list(sc_ll) + list(sc_sent) + list(sc_dl)
    cols_q = list(sq_ll) + list(sq_sent) + list(sq_dl)
    names = list(n_ll) + list(n_sent) + list(n_dl)

    rem = max(0, max_total_cols - len(cols_c))
    if include_base_struct and rem > 0 and n_c >= 16:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S_base_c, S_base_q, base_meta = SC.build_struct_cols(Xc_clean, y_ctx, Xq_clean, seed=seed)
            if S_base_c.shape[1] > 0:
                take = min(rem, int(S_base_c.shape[1]))
                for k in range(take):
                    cols_c.append(S_base_c[:, k])
                    cols_q.append(S_base_q[:, k])
                    names.append(base_meta["col_names"][k])
        except Exception:
            pass

    if cols_c:
        S_ctx = np.column_stack(cols_c).astype(np.float64, copy=False)
        S_q = np.column_stack(cols_q).astype(np.float64, copy=False)
        S_ctx = np.where(np.isfinite(S_ctx), S_ctx, 0.0)
        S_q = np.where(np.isfinite(S_q), S_q, 0.0)
    else:
        S_ctx = np.zeros((n_c, 0), dtype=np.float64)
        S_q = np.zeros((n_q, 0), dtype=np.float64)

    meta = {
        "d_S": int(S_ctx.shape[1]),
        "col_names": names,
        "n_sentinels": len(n_sent),
        "n_diffs_logs": len(n_dl),
        **ll_meta,
    }
    return Xc_clean, Xq_clean, S_ctx, S_q, meta
