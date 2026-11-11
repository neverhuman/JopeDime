#!/usr/bin/env python3
"""Wave C Core (`w5_online_ast_screener`):
Grammar-Complete Compositional AST + Dataset-Meta-Profile Surrogate & Gen-0 Online Ridge Adapter (`W5`).

Replaces legacy `feat412` (`H0.1-A`), which stripped all `H420` `NEW_STAGE_OPS` (`lang_h420.py:297`) and was blind to
`gpr`, `krr`, `ksmooth`, `histgbm2`, `oblivious`, `xtrees`, `rforest`, and `ensemble` heads:
  1. Encodes the full stock + H420 extended grammar (48-dim `encode_program_ast`: stage categories, all 13 head families,
     kernel types (`rq`, `periodic`, `rbf*linear`), symbolic `map`/`derive` arithmetic operators, log-scaled capacity
     hyperparameters, stage count, depth, smoothness, and cost) with ZERO stripped ops.
  2. Computes a 16-dim strictly SEARCH-only Dataset Meta-Profile `phi(D)` (including 2-way split ridge vs tree probes).
  3. Fits a 5-member Bootstrap-by-Family `HistGradientBoostingRegressor` ensemble predicting within-table
     `0.5 * quantile_rank + 0.5 * clipped_z` utility across `(phi(D), psi(prog), regime x AST interactions)`.
  4. Adapts online per table after `K_obs = 24` Gen-0 seed evaluations via a regularized Ridge residual adapter on AST
     features (`mu_prior + Psi_norm @ w_loc`) + Epistemic UCB diversity acquisition (`ucb_screen`).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

STAGE_SEMANTICS: Dict[str, Tuple[int, float, float]] = {
    # Cat 0: Target warps
    "target": (0, 0.55, 0.10),
    "yeo_johnson_duan": (0, 0.65, 0.15),
    "rank_gauss_duan": (0, 0.55, 0.15),
    # Cat 1: Scaling / clipping / row-norm
    "impute": (1, 0.50, 0.10),
    "standardize": (1, 0.70, 0.10),
    "robust_scale": (1, 0.65, 0.10),
    "winsor_clip": (1, 0.60, 0.12),
    "symlog": (1, 0.75, 0.10),
    "blom": (1, 0.60, 0.15),
    "rank": (1, 0.55, 0.12),
    "sigmoid": (1, 0.70, 0.12),
    "norm_row": (1, 0.68, 0.14),
    "step": (1, 0.30, 0.12),
    # Cat 2: Entity / categorical / bin encodings
    "groupby_agg": (2, 0.40, 0.35),
    "target_encode": (2, 0.35, 0.25),
    "target_bin_encode": (2, 0.35, 0.25),
    "freq_encode": (2, 0.35, 0.15),
    "quantile_bins": (2, 0.40, 0.20),
    # Cat 3: Pairwise / symbolic / basis feature expansions (including H420 map/derive)
    "pair_ratio": (3, 0.65, 0.30),
    "pair_prod": (3, 0.70, 0.28),
    "poly_interact": (3, 0.75, 0.35),
    "poly2": (3, 0.78, 0.35),
    "spline": (3, 0.88, 0.30),
    "ple": (3, 0.72, 0.30),
    "hinge": (3, 0.70, 0.25),
    "oblique_hinge": (3, 0.72, 0.30),
    "cheb": (3, 0.82, 0.28),
    "row_stats": (3, 0.55, 0.20),
    "row_diff_stats": (3, 0.55, 0.22),
    "derive": (3, 0.65, 0.30),
    "map": (3, 0.65, 0.25),
    # Cat 4: Subspace / spectral / feature selection / projections
    "select_corr": (4, 0.55, 0.12),
    "svd_project": (4, 0.80, 0.25),
    "rand_proj": (4, 0.75, 0.18),
    "row_spectral": (4, 0.72, 0.24),
    "fft_mag": (4, 0.74, 0.24),
    "polar": (4, 0.70, 0.20),
    "pls": (4, 0.80, 0.25),
    "pca": (4, 0.80, 0.25),
    # Cat 5: Kernel / landmark feature maps
    "nystrom": (5, 0.85, 0.40),
    "landmark_dist": (5, 0.75, 0.35),
    "rff_sin": (5, 0.82, 0.30),
    "rff_cos": (5, 0.82, 0.30),
    # Cat 6: Linear / penalized heads
    "ridge": (6, 1.00, 0.12),
    "ridge_gcv": (6, 1.00, 0.15),
    "elasticnet": (6, 0.95, 0.20),
    "elastic_net": (6, 0.95, 0.20),
    "huber": (6, 0.95, 0.20),
    "pls_head": (6, 0.90, 0.20),
    # Cat 7: Exact / approximate GP & Kernel heads
    "gpr": (7, 0.92, 0.75),
    "krr": (7, 0.90, 0.55),
    "ksmooth": (7, 0.68, 0.40),
    "knn": (7, 0.55, 0.35),
    # Cat 8: GBDT / Oblivious / Forest heads
    "histgbm": (8, 0.25, 0.55),
    "histgbm2": (8, 0.22, 0.65),
    "oblivious": (8, 0.30, 0.50),
    "xtrees": (8, 0.35, 0.55),
    "rforest": (8, 0.32, 0.60),
    "rf": (8, 0.32, 0.60),
    "et": (8, 0.35, 0.55),
    "gam": (8, 0.65, 0.45),
    # Cat 9: Neural / MLP heads
    "mlp": (9, 0.75, 0.70),
    "tabm": (9, 0.72, 0.65),
    # Cat 10: Compositional combinators
    "pipe": (10, 0.50, 0.05),
    "ensemble": (10, 0.55, 0.80),
    "portfolio": (10, 0.55, 0.85),
    "residual": (10, 0.50, 0.85),
    "route": (10, 0.40, 0.80),
}

HEAD_SLOTS = {
    "ridge_gcv": 15,
    "ridge": 15,
    "elasticnet": 16,
    "elastic_net": 16,
    "huber": 16,
    "gpr": 17,
    "krr": 18,
    "ksmooth": 19,
    "histgbm": 20,
    "histgbm2": 21,
    "oblivious": 22,
    "xtrees": 23,
    "rforest": 24,
    "knn": 25,
    "mlp": 26,
    "ensemble": 27,
    "portfolio": 28,
}

H420_EXT_OPS = frozenset(["norm_row", "row_diff_stats", "row_spectral", "fft_mag", "polar", "step", "map", "derive"])

_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_KV_RE = re.compile(r"^([a-zA-Z0-9_]+)=(.+)$")


def head_family(dsl: str) -> str:
    s = dsl.strip()
    if s.startswith("(portfolio"):
        return "portfolio"
    depth, start, last = 0, None, None
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
            if depth == 2:
                start = i
        elif ch == ")":
            if depth == 2 and start is not None:
                last = s[start + 1 : i]
            depth -= 1
    return re.split(r"[\s)]", last, maxsplit=1)[0] if last else "none"


def encode_program_ast(dsl: str) -> np.ndarray:
    """Encodes ANY GP / H420 S-expression into a 48-dim grammar-complete vector (zero ops stripped!)."""
    toks = _TOKEN_RE.findall(dsl)
    vec = np.zeros(48, dtype=np.float64)
    depth = 0
    max_depth = 0
    n_stages = 0
    n_members = 1
    smooth_vals = []
    cost_sum = 0.0

    i = 0
    while i < len(toks):
        tk = toks[i]
        if tk == "(":
            depth += 1
            max_depth = max(max_depth, depth)
            if i + 1 < len(toks) and toks[i + 1] not in ("(", ")"):
                op = toks[i + 1].lower()
                if op == "pipe" and depth >= 2:
                    n_members += 1
                if op != "pipe":
                    n_stages += 1
                if op in H420_EXT_OPS:
                    vec[40] += 1.0
                if op in STAGE_SEMANTICS:
                    cat, sm, cost = STAGE_SEMANTICS[op]
                    vec[cat] += 1.0
                    if op != "pipe":
                        smooth_vals.append(sm)
                        cost_sum += cost
                else:
                    h = int(hashlib.md5(op.encode()).hexdigest()[:6], 16) % 4
                    vec[11 + h] += 1.0
                    smooth_vals.append(0.5)
                    cost_sum += 0.35
                if op in HEAD_SLOTS:
                    vec[HEAD_SLOTS[op]] += 1.0
        elif tk == ")":
            depth = max(0, depth - 1)
        else:
            m = _KV_RE.match(tk)
            if m:
                k, v = m.group(1).lower(), m.group(2).lower()
                if v in ("true", "false"):
                    if k == "append" and v == "true":
                        vec[29] += 1.0
                else:
                    try:
                        fv = float(v)
                        lv = math.copysign(math.log1p(abs(fv)), fv)
                        if k in ("alpha", "lam", "lambda", "lambda_frac", "l2", "reg"):
                            vec[30] = lv
                        elif k in ("m", "opt_rows", "n_estimators", "iters", "rounds", "trees"):
                            vec[31] = max(vec[31], lv)
                        elif k in ("max_leaves", "leaves", "depth", "k", "topk", "top_k", "active_top_k", "components", "bins"):
                            vec[32] = max(vec[32], lv)
                        elif k in ("lr", "eta", "shrink", "q", "gamma"):
                            vec[33] = lv
                        elif k in ("bags", "min_leaf", "steps"):
                            vec[34] = max(vec[34], lv)
                        elif k in ("row_subsample", "col_subsample", "subsample", "rsm", "max_features"):
                            vec[41] = fv
                    except ValueError:
                        if k == "kernel":
                            if "rq" in v or "matern" in v:
                                vec[42] = 1.0
                            if "linear" in v:
                                vec[43] = 1.0
                            if "periodic" in v:
                                vec[44] = 1.0
                        elif k == "expr":
                            if "/" in v:
                                vec[45] = 1.0
                            if "*" in v or "^" in v:
                                vec[46] = 1.0
                            if "row_" in v:
                                vec[47] = 1.0
        i += 1

    vec[35] = math.log1p(len(toks))
    vec[36] = float(max_depth)
    vec[37] = float(n_stages)
    vec[38] = float(np.mean(smooth_vals)) if smooth_vals else 0.5
    vec[39] = cost_sum / max(1.0, float(max(1, n_members - 1)))
    return vec


def compute_dataset_meta_profile(Xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Computes a 16-dim Dataset Meta-Profile strictly from SEARCH `(Xs, ys)`."""
    n, p = Xs.shape
    M = ~np.isfinite(Xs)
    miss_rate = float(np.mean(M))
    miss_cols = float(np.mean(np.any(M, axis=0)))

    X_imp = np.zeros((n, p), dtype=np.float64)
    cat_cnt = 0
    max_u = min(20, max(2, int(0.05 * n)))
    for j in range(p):
        fin = Xs[~M[:, j], j]
        med = float(np.median(fin)) if len(fin) > 0 else 0.0
        sc = float(np.std(fin)) if len(fin) > 1 else 1.0
        if sc < 1e-8:
            sc = 1.0
        if len(fin) == 0 or len(np.unique(np.round(fin, 5))) <= max_u:
            cat_cnt += 1
        X_imp[:, j] = np.clip((np.where(M[:, j], med, Xs[:, j]) - med) / sc, -8.0, 8.0)

    yc = ys - float(np.mean(ys))
    ysd = max(float(np.std(yc)), 1e-8)
    yn = yc / ysd
    y_skew = float(np.clip(abs(np.mean(yn ** 3)), 0.0, 20.0))
    y_kurt = float(np.clip(np.log1p(max(0.0, float(np.mean(yn ** 4)) - 3.0)), 0.0, 10.0))

    corrs = np.array(
        [abs(float(np.dot(X_imp[:, j] - np.mean(X_imp[:, j]), yn))) / (n * (float(np.std(X_imp[:, j])) + 1e-8)) for j in range(p)]
    )
    max_corr = float(np.max(corrs)) if p > 0 else 0.0
    mean_top5_corr = float(np.mean(np.sort(corrs)[-min(5, p) :])) if p > 0 else 0.0

    ev = np.arange(n) % 2 == 0
    od = ~ev
    A = X_imp[ev].T @ X_imp[ev] + 10.0 * np.eye(p)
    w = np.linalg.solve(A, X_imp[ev].T @ yn[ev])
    p_r = X_imp[od] @ w
    u_ridge = float(-0.5 * math.log(max(float(np.mean((p_r - yn[od]) ** 2)), 1e-4)))

    hgb = HistGradientBoostingRegressor(max_iter=45, max_leaf_nodes=15, min_samples_leaf=max(5, n // 30), random_state=0)
    hgb.fit(X_imp[ev], yn[ev])
    p_t = hgb.predict(X_imp[od])
    u_tree = float(-0.5 * math.log(max(float(np.mean((p_t - yn[od]) ** 2)), 1e-4)))

    return np.array(
        [
            math.log(max(n, 2)),
            math.log(max(p, 1)),
            p / max(n, 1.0),
            miss_rate,
            miss_cols,
            cat_cnt / max(p, 1.0),
            y_skew,
            y_kurt,
            max_corr,
            mean_top5_corr,
            u_ridge,
            u_tree,
            u_tree - u_ridge,
            float(np.std(corrs)) if p > 1 else 0.0,
            1.0 if p > 30 else 0.0,
            1.0 if y_skew > 1.5 else 0.0,
        ],
        dtype=np.float64,
    )


def robust_table_target(u_vals: np.ndarray) -> np.ndarray:
    m = len(u_vals)
    ranks = np.argsort(np.argsort(u_vals)).astype(np.float64) / max(m - 1, 1)
    med_u = float(np.median(u_vals))
    iqr_u = max(float(np.quantile(u_vals, 0.85) - np.quantile(u_vals, 0.15)), 1e-3)
    z_u = np.clip((u_vals - med_u) / iqr_u, -2.5, 3.0) / 3.0
    return 0.50 * ranks + 0.50 * z_u


class GrammarCompleteASTSurrogate:
    """5-Member Bootstrap-by-Family Quantile-Rank + Within-Table Clipped-Z Surrogate with Online AST Ridge Adapter."""

    def __init__(self, n_members: int = 5):
        self.n_members = n_members
        self.models: List[HistGradientBoostingRegressor] = []

    @staticmethod
    def _joint_features(phi_d: np.ndarray, psis: np.ndarray) -> np.ndarray:
        n_c = psis.shape[0]
        phi_rep = np.broadcast_to(phi_d[None, :], (n_c, len(phi_d)))
        i1 = phi_d[12] * (psis[:, 7] + psis[:, 8] - psis[:, 6])
        i2 = phi_d[2] * (psis[:, 6] + psis[:, 4] - psis[:, 8])
        i3 = phi_d[6] * psis[:, 0]
        i4 = phi_d[5] * psis[:, 2]
        i5 = phi_d[0] * psis[:, 39]
        i6 = phi_d[14] * (psis[:, 4] - psis[:, 44])  # wide-table penalty on periodic / reward for subspace
        return np.column_stack([phi_rep, psis, i1, i2, i3, i4, i5, i6])

    def fit(self, records: List[Tuple[str, np.ndarray, List[str], np.ndarray]], seed: int = 42) -> "GrammarCompleteASTSurrogate":
        """records: list of (family_id, phi_d, dsls, u_vals). Bootstraps strictly by family_id!"""
        rng = np.random.default_rng(seed)
        X_blocks, y_blocks, fam_blocks = [], [], []
        fams = sorted({fam for fam, _, _, _ in records})
        fam_to_id = {f: idx for idx, f in enumerate(fams)}
        for fam, phi_d, dsls, u_vals in records:
            if len(dsls) < 6:
                continue
            psis = np.array([encode_program_ast(s) for s in dsls])
            X_blocks.append(self._joint_features(phi_d, psis))
            y_blocks.append(robust_table_target(u_vals))
            fam_blocks.append(np.full(len(dsls), fam_to_id[fam], dtype=np.int64))

        X_mat = np.vstack(X_blocks)
        y_vec = np.concatenate(y_blocks)
        fam_arr = np.concatenate(fam_blocks)
        n_fams = len(fams)

        self.models = []
        for b in range(self.n_members):
            boot_fams = set(rng.choice(n_fams, size=max(1, int(0.82 * n_fams)), replace=True).tolist())
            mask = np.isin(fam_arr, list(boot_fams))
            if int(np.sum(mask)) < 40:
                mask[:] = True
            reg = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.045,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=3.0,
                random_state=int((seed + 101 * b) & 0x7FFFFFFF),
            )
            reg.fit(X_mat[mask], y_vec[mask])
            self.models.append(reg)
        return self

    def score_candidates(self, phi_d: np.ndarray, candidate_dsls: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not candidate_dsls:
            return np.zeros(0), np.zeros(0)
        psis = np.array([encode_program_ast(s) for s in candidate_dsls])
        X_q = self._joint_features(phi_d, psis)
        preds = np.array([m.predict(X_q) for m in self.models])
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    def adapt_online_and_score(
        self,
        phi_d: np.ndarray,
        obs_dsls: List[str],
        obs_u: np.ndarray,
        rem_dsls: List[str],
        ridge_lam: float = 8.0,
        beta_ucb: float = 0.20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adapts the cross-family prior to table D using Gen-0 observed `(obs_dsls, obs_u)` via AST Ridge residual."""
        all_dsls = list(obs_dsls) + list(rem_dsls)
        mu_all, sig_all = self.score_candidates(phi_d, all_dsls)
        n_o = len(obs_dsls)
        mu_o, mu_r = mu_all[:n_o], mu_all[n_o:]
        sig_r = sig_all[n_o:]
        if n_o < 4 or not rem_dsls:
            return mu_r + beta_ucb * sig_r, mu_r

        psis_o = np.array([encode_program_ast(s) for s in obs_dsls])
        psis_r = np.array([encode_program_ast(s) for s in rem_dsls])
        pm = np.mean(psis_o, axis=0)
        psd = np.std(psis_o, axis=0) + 1e-6
        Po = np.clip((psis_o - pm) / psd, -6.0, 6.0)
        Pr = np.clip((psis_r - pm) / psd, -6.0, 6.0)

        y_obs_z = robust_table_target(obs_u)
        res = y_obs_z - mu_o
        A = Po.T @ Po + ridge_lam * np.eye(Po.shape[1])
        w_loc = np.linalg.solve(A, Po.T @ res)
        adapted_mean = mu_r + Pr @ w_loc
        return adapted_mean + beta_ucb * sig_r, adapted_mean


def select_diverse_topk(scores: np.ndarray, dsls: List[str], k_select: int, lam_div: float = 0.03) -> List[int]:
    if len(dsls) <= k_select:
        return list(range(len(dsls)))
    psis = np.array([encode_program_ast(s) for s in dsls])
    psis_u = psis / (np.linalg.norm(psis, axis=1, keepdims=True) + 1e-8)
    chosen: List[int] = []
    for _ in range(k_select):
        if not chosen:
            pick = int(np.argmax(scores))
        else:
            max_sim = np.max(psis_u @ psis_u[chosen].T, axis=1)
            acq = scores - lam_div * max_sim
            acq[chosen] = -1e18
            pick = int(np.argmax(acq))
        chosen.append(pick)
    return chosen


def h01a_head_residual_adapt(
    raw_obs: np.ndarray, obs_dsls: List[str], obs_u: np.ndarray, raw_rem: np.ndarray, rem_dsls: List[str]
) -> np.ndarray:
    """Reproduces stock GP's head-family online residual corrector on top of H0.1-A raw scores."""
    if len(obs_u) < 4 or len(rem_dsls) == 0:
        return raw_rem.copy()
    y_z = robust_table_target(obs_u)
    r_z = robust_table_target(raw_obs)
    resid = y_z - r_z
    heads_o = [head_family(d) for d in obs_dsls]
    head_bias: Dict[str, float] = {}
    for h in set(heads_o):
        vals = [resid[i] for i, hh in enumerate(heads_o) if hh == h]
        head_bias[h] = float(np.sum(vals)) / (len(vals) + 3.0)
    # Normalize raw_rem onto the observed raw distribution scale
    med_r = float(np.median(raw_obs))
    iqr_r = max(float(np.quantile(raw_obs, 0.85) - np.quantile(raw_obs, 0.15)), 1e-3)
    rem_norm = 0.50 * (np.argsort(np.argsort(raw_rem)).astype(np.float64) / max(len(raw_rem) - 1, 1)) + 0.50 * (
        np.clip((raw_rem - med_r) / iqr_r, -2.5, 3.0) / 3.0
    )
    bonuses = np.array([head_bias.get(head_family(d), 0.0) for d in rem_dsls], dtype=np.float64)
    return rem_norm + bonuses
