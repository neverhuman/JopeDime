#!/usr/bin/env python3
"""H458 training episodes built by the evaluation's own feature and target pipeline.

`cols_104m_e8_gpresid_h452._prep_full_context_worker` turns one fold `(X_tr, y_tr, X_ev)` into
`[CW key/formula cols | cleaned X | structural cols]` and three target spaces (robust-ridge residual, Yeo-Johnson,
rank-Gauss). No trainer before H458 fed the model those feature blocks or the rank-Gauss space. This module calls the
same functions with the same arguments on a fresh random split of a family pool, so a training episode and an
evaluation fold are the same object apart from which rows are in it. Everything is fitted on context rows only;
query targets are used as loss targets and nowhere else.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import NamedTuple

import numpy as np

import paths_h458 as P

P.setup_paths()

import cols_104m_e8_gpresid_h452 as C  # noqa: E402
from icl_model_mps_h450 import ICLH450MPS  # noqa: E402

REL, RR = C.REL, C.RR
WIDE = int(C.WIDE_GPU)
FID_SLOTS = int(C.FID_SLOTS)
MODES = ("rr", "yj", "rg")

# canonical feature-identity slots (arm "canon"): raw column j -> j, structural col k -> 200 + k, pinned CW col k -> 228 + k
RAW_SLOTS, STRUCT_BASE, PIN_BASE, BLOCK_MAX = 200, 200, 228, 28


def fold_seed(cid: str, rep: int, fk: int) -> int:
    """The evaluation's per-fold seed."""
    return C._sha_seed(f"h452_cv7:{cid}:r{rep}:f{fk}")


class FoldFeatures(NamedTuple):
    X_tr: np.ndarray  # [n_tr, n_pin + n_raw + n_struct]
    X_ev: np.ndarray
    n_pin: int
    n_raw: int
    n_struct: int


class ModeTarget(NamedTuple):
    name: str
    y_ctx: np.ndarray  # context target in the mode's space
    base_ev: np.ndarray  # added back to the prediction at evaluation (zeros unless the ridge base is used)
    warp: object  # fitted warp with `inv_torch`, or None


def eval_fold_features(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, fseed: int) -> FoldFeatures:
    """The column builder's per-fold feature block (same calls, same arguments, same column order)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_tr_cl, x_ev_cl, s_tr, s_ev, _ = REL.build_h450_struct_cols(
            X_tr, y_tr, X_ev, seed=fseed, fast_mode=True, max_total_cols=14, include_base_struct=True
        )
        cw_tr, cw_ev = C.extract_cw1_cw2_features(X_tr, y_tr, X_ev, seed=fseed + 7)
    tr = np.hstack([p for p in (cw_tr, x_tr_cl, s_tr) if p.shape[1] > 0])
    ev = np.hstack([p for p in (cw_ev, x_ev_cl, s_ev) if p.shape[1] > 0])
    return FoldFeatures(tr, ev, int(cw_tr.shape[1]), int(x_tr_cl.shape[1]), int(s_tr.shape[1]))


def mode_targets(ff: FoldFeatures, y_tr: np.ndarray, fseed: int) -> dict[str, ModeTarget]:
    """The three direct target spaces of the evaluation, fitted on context rows only."""
    n_ev = ff.X_ev.shape[0]
    zero = np.zeros(n_ev, dtype=np.float64)
    oof, pq, used, _ = RR.robust_ridge_np(ff.X_tr, y_tr, ff.X_ev, seed=fseed + 11)
    rr = ModeTarget("rr", (y_tr - oof) if used else y_tr, np.asarray(pq, dtype=np.float64) if used else zero, None)
    w_yj = RR.YeoJohnsonWarpF64()
    z_yj = w_yj.fit_transform(y_tr)
    w_rg = C.RankGaussDuanWarpF64()
    z_rg = w_rg.fit_transform(y_tr)
    return {"rr": rr, "yj": ModeTarget("yj", z_yj, zero, w_yj), "rg": ModeTarget("rg", z_rg, zero, w_rg)}


def rank_gauss_forward(warp, y: np.ndarray) -> np.ndarray:
    """Maps raw targets into a fitted rank-Gauss space: the inverse of `RankGaussDuanWarpF64.inv_torch`."""
    yk = np.asarray(warp.y_knots, dtype=np.float64)
    zk = np.asarray(warp.z_knots, dtype=np.float64)
    yu, inv = np.unique(yk, return_inverse=True)  # ties in y share the mean of their z knots
    zu = np.bincount(inv, weights=zk) / np.bincount(inv)
    y = np.asarray(y, dtype=np.float64)
    z = np.interp(y, yu, zu)
    lo, hi = y < yu[0], y > yu[-1]
    if lo.any():
        z[lo] = zk[0] + ((y[lo] - yu[0]) / warp.slope_lo if warp.slope_lo > 0.0 else 0.0)
    if hi.any():
        z[hi] = zk[-1] + ((y[hi] - yu[-1]) / warp.slope_hi if warp.slope_hi > 0.0 else 0.0)
    return np.clip(z, zk[0] - 8.0, zk[-1] + 8.0)


def query_target(mt: ModeTarget, y_ev: np.ndarray) -> np.ndarray:
    """Loss target of the query rows in the mode's space."""
    y_ev = np.asarray(y_ev, dtype=np.float64)
    if mt.name == "rr":
        return y_ev - mt.base_ev
    if mt.name == "yj":
        return np.asarray(mt.warp.transform_any(y_ev), dtype=np.float64)
    return rank_gauss_forward(mt.warp, y_ev)


def random_view(ff: FoldFeatures, tgt_ctx: np.ndarray, rng: np.random.Generator):
    """Arm "rand": one draw from the evaluation's own view family (identity 1 in 4, permuted otherwise)."""
    views = C._build_views(ff.X_tr, tgt_ctx, ff.X_ev, n_views=2, seed=int(rng.integers(0, 2**31 - 1)), keep_first=ff.n_pin)
    vt, ve = views[0] if rng.random() < 0.25 else views[1]
    return vt, ve, None


def canonical_slots(ff: FoldFeatures) -> np.ndarray:
    if ff.n_pin > BLOCK_MAX or ff.n_struct > BLOCK_MAX or ff.n_raw > RAW_SLOTS:
        raise ValueError(f"canonical slot ranges exceeded: pin={ff.n_pin} raw={ff.n_raw} struct={ff.n_struct}")
    return np.concatenate(
        [PIN_BASE + np.arange(ff.n_pin), np.arange(ff.n_raw), STRUCT_BASE + np.arange(ff.n_struct)]
    ).astype(np.int64)


def canonical_view(ff: FoldFeatures, tgt_ctx: np.ndarray):
    """Arm "canon": native column order, one view, slots tied to the column's identity."""
    p = ff.X_tr.shape[1]
    slots = canonical_slots(ff)
    if p <= WIDE:
        return ff.X_tr, ff.X_ev, slots
    # wide table: keep the pinned block and the columns most correlated with the context target, in native order
    x = ff.X_tr[:, ff.n_pin :]
    fin = np.isfinite(x)
    cnt = fin.sum(0)
    xm = np.where(fin, x, 0.0).sum(0) / np.maximum(cnt, 1)
    xc = np.where(fin, x - xm, 0.0)
    yc = np.asarray(tgt_ctx, dtype=np.float64) - float(np.mean(tgt_ctx))
    den = np.sqrt((xc * xc).sum(0)) * max(float(np.linalg.norm(yc)), 1e-12)
    score = np.where((cnt >= 10) & (den > 1e-12), np.abs(xc.T @ yc) / np.maximum(den, 1e-300), 0.0)
    top = np.sort(np.argsort(-score, kind="mergesort")[: WIDE - ff.n_pin]) + ff.n_pin
    keep = np.concatenate([np.arange(ff.n_pin), top]).astype(np.int64)
    return ff.X_tr[:, keep], ff.X_ev[:, keep], slots[keep]


def stratified_sample(y: np.ndarray, n_take: int, rng: np.random.Generator) -> np.ndarray:
    """`n_take` distinct row ids, one random row per target-quantile stratum, in random order."""
    n = len(y)
    if n_take >= n:
        return rng.permutation(n)
    order = np.argsort(np.asarray(y, dtype=np.float64) + 1e-12 * rng.standard_normal(n), kind="mergesort")
    edges = np.linspace(0.0, float(n), n_take + 1)
    pos = np.unique(np.minimum((edges[:-1] + rng.random(n_take) * (edges[1:] - edges[:-1])).astype(np.int64), n - 1))
    if len(pos) < n_take:
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), pos)
        pos = np.concatenate([pos, rng.choice(rest, size=n_take - len(pos), replace=False)])
    return rng.permutation(order[pos])


def split_context_query(y_sel: np.ndarray, n_q: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean query mask over the selected rows; queries are spread over the target quantiles (K-fold style)."""
    mask = np.zeros(len(y_sel), dtype=bool)
    mask[stratified_sample(y_sel, n_q, rng)] = True
    return mask


def plan_sizes(
    member_rows: list[int],
    n_pool: int,
    rng: np.random.Generator,
    q_max: int = 1024,
    ctx_range: tuple[int, int] | None = None,
    ctx_cap_frac: float = 1.0,
) -> tuple[int, int]:
    """Context and query row counts: 70% evaluation-size, 20% at 384-768, 10% at 256-384 (or a forced range)."""
    n_search = int(round(0.7 * int(rng.choice(member_rows))))
    u = float(rng.random())
    if ctx_range is not None:
        n_ctx = int(rng.integers(int(ctx_range[0]), int(ctx_range[1]) + 1))
    elif u < 0.70:
        n_ctx = n_search if rng.random() < 0.4 else int(round(n_search * 6.0 / 7.0))
    elif u < 0.90:
        n_ctx = int(rng.integers(min(384, n_search), min(768, n_search) + 1))
    else:
        n_ctx = int(rng.integers(min(256, n_search), min(384, n_search) + 1))
    n_ctx = max(32, min(n_ctx, n_pool - 32, int(ctx_cap_frac * n_pool)))  # leave-context-out pools are small
    return n_ctx, int(min(q_max, n_pool - n_ctx))


def fit_caps(n_ctx: int, n_q: int, n_feat: int, caps: dict, n_modes: int = len(MODES)) -> tuple[int, int]:
    """`(B, n_q)` inside the token caps: all modes in one batch when at least 128 queries fit, else one mode."""

    def q_fit(b: int) -> int:
        return int(min(n_q, caps["tokens"] // (b * (n_feat + 4)) - n_ctx, caps["row_tokens"] // b - n_ctx))

    q_all = q_fit(n_modes)
    if q_all >= 128:
        return n_modes, q_all
    q_one = q_fit(1)
    return (1, q_one) if q_one >= 32 else (0, 0)


def split_commitment(fam: int, ctx_ids: np.ndarray, q_ids: np.ndarray) -> bytes:
    """16-byte identity of a split: family + the context row set + the query row set (order-free)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(int(fam).to_bytes(4, "little"))
    h.update(np.sort(np.asarray(ctx_ids, dtype=np.int64)).tobytes())
    h.update(b"|")
    h.update(np.sort(np.asarray(q_ids, dtype=np.int64)).tobytes())
    return h.digest()


def draw_split(table, member_rows: list[int], rng: np.random.Generator, q_max: int = 1024, ctx_range=None, ctx_cap_frac: float = 1.0):
    """Fresh stratified downsample and fresh context/query split of a pool table; every row id is used at most once."""
    y = np.asarray(table[:, -1], dtype=np.float64)
    n_ctx, n_q = plan_sizes(member_rows, len(y), rng, q_max=q_max, ctx_range=ctx_range, ctx_cap_frac=ctx_cap_frac)
    ids = stratified_sample(y, n_ctx + n_q, rng) if rng.random() < 0.7 else rng.permutation(len(y))[: n_ctx + n_q]
    qmask = split_context_query(y[ids], n_q, rng)
    return ids[~qmask], ids[qmask]


def build_episode(
    table: np.ndarray,
    member_rows: list[int],
    fam: int,
    rng: np.random.Generator,
    caps: dict,
    arm: str = "rand",
    q_max: int = 1024,
    ctx_range: tuple[int, int] | None = None,
    ctx_cap_frac: float = 1.0,
) -> dict | None:
    """One prepared micro-batch: the same split in up to three target spaces. `None` when it cannot fit the caps."""
    ctx_ids, q_ids = draw_split(table, member_rows, rng, q_max=q_max, ctx_range=ctx_range, ctx_cap_frac=ctx_cap_frac)
    ctx_ids, q_ids = np.sort(ctx_ids), np.sort(q_ids)  # fancy-index the memmap in ascending order
    x_tr = np.array(table[ctx_ids, :-1], dtype=np.float64)
    y_tr = np.array(table[ctx_ids, -1], dtype=np.float64)
    x_ev = np.array(table[q_ids, :-1], dtype=np.float64)
    y_ev = np.array(table[q_ids, -1], dtype=np.float64)
    order_c, order_q = rng.permutation(len(ctx_ids)), rng.permutation(len(q_ids))
    x_tr, y_tr, x_ev, y_ev = x_tr[order_c], y_tr[order_c], x_ev[order_q], y_ev[order_q]
    if float(np.std(y_tr)) <= 1e-12:
        return None
    fseed = int(rng.integers(0, 2**31 - 1))
    ff = eval_fold_features(x_tr, y_tr, x_ev, fseed)
    n_feat = min(ff.X_tr.shape[1], WIDE)
    b, n_q = fit_caps(len(y_tr), len(y_ev), n_feat, caps)
    if b == 0:
        return None
    mts = mode_targets(ff, y_tr, fseed)
    use = MODES if b == len(MODES) else (MODES[int(rng.integers(len(MODES)))],)
    xs, ycs, yqs, slots = [], [], [], []
    for m in use:
        mt = mts[m]
        vt, ve, sl = canonical_view(ff, mt.y_ctx) if arm == "canon" else random_view(ff, mt.y_ctx, rng)
        xs.append(np.vstack([vt, ve[:n_q]]))
        ycs.append(np.asarray(mt.y_ctx, dtype=np.float64))
        yqs.append(query_target(mt, y_ev)[:n_q])
        slots.append(sl)
    ya, yq = np.stack(ycs), np.stack(yqs)
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(yq))):
        return None
    zm, yz, ym, ys = ICLH450MPS.prenorm_cpu_np(np.stack(xs), ya, len(y_tr))
    return {
        "zm": zm,
        "yz": yz,
        "ym": ym,
        "ys": ys,
        "yq": np.ascontiguousarray(yq, dtype=np.float64),
        "slots": np.stack(slots)[:, None, :] if arm == "canon" else None,
        "n_ctx": int(len(y_tr)),
        "n_q": int(n_q),
        "n_feat": int(zm.shape[2]),
        "fam": int(fam),
        "modes": tuple(use),
        "commit": split_commitment(fam, ctx_ids, q_ids[order_q][:n_q]),
    }
