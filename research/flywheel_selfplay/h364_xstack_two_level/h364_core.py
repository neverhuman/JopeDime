"""H364 XSTACK-2L core (PREREG §2, §4): a second level over the classical library on top of the FROZEN H335 GP stack.

ŷ = x + Σ_j β_j (L_j − x), β ≥ 0, Σβ ≤ c, where x is the locked H335 `AGG.g50.CV3@12` prediction made with its STORED
weights (w_fold0, w_fold1, w_audit; the stacker is never re-solved), so the GP members' relative weights never change.
β minimises mean((x + Dβ − y)²) / mean((x − y)²) + λ‖β‖² (scale-free shrinkage toward GP-only; on two folds the mean
of the two folds' scaled terms), solved exactly by KKT enumeration. Cross-fitting: β⁽⁰⁾ is fit on fold-1 rows (x = P1 @
w_fold0, the weights fit on fold 1) and applied to fold 0 (x = P0 @ w_fold0, the stored cross-fitted prediction);
β⁽¹⁾ the reverse; β⁽ᵃ⁾ is fit on both folds with x = P_k @ w_audit. β⁽ᵃ⁾ is used only if the cross-fitted 2L SEARCH
log-RMSE ratio beats X0's by δ on both folds, and is 0 when X0's cross-fitted SEARCH utility is at least τ.

Runs from a staged code root that holds the H354 files (h354_core installs the H317 guard). SEARCH arrays only reach
`second_level`; AUDIT arrays enter only `audit_utility` (tested by perturbation).
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h354_core as K  # noqa: E402  (installs the H317 guard)

import numpy as np  # noqa: E402

C, E, ST = K.C, K.E, K.ST
WAVE = "H364"
CLASSICAL = K.CLASSICAL  # catboost.v1, lightgbm.v1, xgboost.v1, random_forest.v1 (reference-board configs)
GRID_C = (0.10, 0.20, 0.35, 0.50)
GRID_LAMBDA = (0.0, 0.02, 0.10)
GRID_DELTA = (0.0, 0.005, 0.02)
GRID_TAU = (math.inf, 6.0, 4.0, 3.0)
TOL = 1e-6  # X0 harness check, PREREG §7.1 (stored weights reproduce the stored SEARCH log-ratios and AUDIT utility;
# part A said 1e-9: dev showed float-order noise up to 1.09e-9 at near-perfect contexts, SEARCH u 17-18)
BETA_EPS = 1e-12  # |β| below this is exactly 0 (2L == X0)


@dataclass(frozen=True)
class Design:
    c: float
    lam: float
    delta: float
    tau: float

    def key(self) -> str:
        return f"c{self.c:g}_l{self.lam:g}_d{self.delta:g}_t{self.tau:g}"

    def as_json(self) -> dict:
        d = asdict(self)
        d["tau"] = "inf" if math.isinf(self.tau) else self.tau
        return d

    @staticmethod
    def from_json(d: dict) -> "Design":
        tau = math.inf if d["tau"] == "inf" else float(d["tau"])
        return Design(float(d["c"]), float(d["lam"]), float(d["delta"]), tau)


def grid() -> List[Design]:
    return [Design(c, lam, d, t) for c in GRID_C for lam in GRID_LAMBDA for d in GRID_DELTA for t in GRID_TAU]


def conservative_key(d: Design) -> Tuple[float, float, float, float]:
    """PREREG §4.3 tie-break order: smaller c, then larger λ, then larger δ, then smaller τ."""
    return (d.c, -d.lam, -d.delta, d.tau)


# ----------------------------------------------------------------------------------------------------------------------
# The capped ridge QP: min b'Ab + 2g'b  s.t.  b >= 0, 1'b <= c   (A symmetric PSD, J <= 4), exact by KKT enumeration
# ----------------------------------------------------------------------------------------------------------------------
def qp_capped(A: np.ndarray, g: np.ndarray, c: float) -> np.ndarray:
    J = len(g)
    best, best_f = np.zeros(J), 0.0
    if c <= 0 or J == 0:
        return best

    def f(b):
        return float(b @ A @ b + 2.0 * g @ b)

    for k in range(1, J + 1):
        for S in itertools.combinations(range(J), k):
            S = list(S)
            A_S, g_S = A[np.ix_(S, S)], g[S]
            try:
                free = np.linalg.solve(A_S, -g_S)
                inv1 = np.linalg.solve(A_S, np.ones(k))
            except np.linalg.LinAlgError:
                continue
            cands = [free]
            den = float(inv1.sum())
            if abs(den) > 1e-300:
                mu = (c - float(free.sum())) / den  # b = free + mu * A^{-1} 1 has 1'b = c
                cands.append(free + mu * inv1)
            for bS in cands:
                if not np.all(np.isfinite(bS)) or np.any(bS <= 0) or bS.sum() > c * (1 + 1e-12):
                    continue
                b = np.zeros(J)
                b[S] = bS
                fb = f(b)
                if fb < best_f - 1e-15:
                    best, best_f = b, fb
    return best


def fit_beta(xs: Sequence[np.ndarray], Ls: Sequence[np.ndarray], ys: Sequence[np.ndarray], c: float,
             lam: float) -> np.ndarray:
    """β for one fitting set made of one or two folds (each: X0 prediction x [n], library matrix L [n, J], target y)."""
    J = Ls[0].shape[1]
    A = lam * np.eye(J)
    g = np.zeros(J)
    for x, L, y in zip(xs, Ls, ys):
        r = x - y
        s = float(np.mean(r * r))
        if not s > 0:  # X0 exact on the fitting fold: nothing for a second level to correct
            return np.zeros(J)
        D = L - x[:, None]
        w = 1.0 / (len(y) * s * len(xs))
        A = A + w * (D.T @ D)
        g = g + w * (D.T @ r)
    b = qp_capped(0.5 * (A + A.T), g, c)
    b[np.abs(b) < BETA_EPS] = 0.0
    return b


# ----------------------------------------------------------------------------------------------------------------------
# X0 from its STORED weights
# ----------------------------------------------------------------------------------------------------------------------
def x0_spec(ctx: "K.Context") -> "ST.StackSpec":
    st = ctx.stored["spec"]
    if list(st["members"]) != [m.sha for m in ctx.members]:
        raise AssertionError(f"{ctx.cid[:12]}: stored spec member order")
    return ST.StackSpec("X0", list(st["members"]), [m.dsl for m in ctx.members], [-1] * len(ctx.members),
                        np.asarray(st["w_audit"], np.float64), np.asarray(st["w_fold0"], np.float64),
                        np.asarray(st["w_fold1"], np.float64), dict(st["search"]))


@dataclass
class X0Search:
    x0c: np.ndarray  # fold-0 rows, stored cross-fitted (P0 @ w_fold0; w_fold0 fit on fold 1)
    x1c: np.ndarray  # fold-1 rows, P1 @ w_fold1
    x0f: np.ndarray  # fold-0 rows as a FITTING fold: P0 @ w_fold1 (w_fold1 fit on fold 0)
    x1f: np.ndarray  # fold-1 rows as a fitting fold: P1 @ w_fold0
    x0a: np.ndarray  # P0 @ w_audit (both-fold fit, for β⁽ᵃ⁾)
    x1a: np.ndarray
    lr0: float
    lr1: float

    @property
    def u(self) -> float:
        return -0.5 * (self.lr0 + self.lr1)


def x0_search(ctx: "K.Context", spec: "ST.StackSpec") -> X0Search:
    P0, P1, _, _ = ST._mats(ctx.members)
    fd = ctx.fd
    x0c, x1c = P0 @ spec.w_fold0, P1 @ spec.w_fold1
    return X0Search(x0c, x1c, P0 @ spec.w_fold1, P1 @ spec.w_fold0, P0 @ spec.w_audit, P1 @ spec.w_audit,
                    ST._lr(x0c, fd.y0, fd.b0), ST._lr(x1c, fd.y1, fd.b1))


def x0_audit(ctx: "K.Context", spec: "ST.StackSpec") -> dict:
    out = ST.stack_audit_utility(spec, ctx.audit_by_sha, ctx.ya, ctx.ba, ctx.y_search_mean, return_pred=True)
    if out.get("pred") is None:
        out["pred"] = np.full(len(ctx.ya), ctx.y_search_mean)
    return out


def x0_check(ctx: "K.Context", xs: X0Search, xa: dict) -> Dict[str, float]:
    st = ctx.stored["spec"]["search"]
    return {"dlr": max(abs(xs.lr0 - float(st["lr0"])), abs(xs.lr1 - float(st["lr1"]))),
            "du": abs(float(xa["utility"]) - float(ctx.stored["raw"]["utility"]))}


# ----------------------------------------------------------------------------------------------------------------------
# The second level (SEARCH only) and its one AUDIT score
# ----------------------------------------------------------------------------------------------------------------------
def library_columns(lib_search: Dict[str, "E.SearchResult"]) -> List[str]:
    """Learners that completed with finite SEARCH OOF predictions and pass the S1 gate (as in H357)."""
    return [n for n in CLASSICAL if n in lib_search and lib_search[n].ok and lib_search[n].gate]


def second_level(ctx: "K.Context", xs: X0Search, lib_search: Dict[str, "E.SearchResult"], design: Design) -> dict:
    """β⁽ᵃ⁾ for one context from SEARCH arrays only, with the reason it is zero when it is."""
    cols = library_columns(lib_search)
    fd = ctx.fd
    out = {"cols": cols, "x0_search_u": xs.u, "beta": [0.0] * len(cols), "active": False}
    if not cols:
        return {**out, "reason": "no_library_column"}
    if xs.u >= design.tau:
        return {**out, "reason": "near_perfect_guard"}
    L0 = np.column_stack([lib_search[n].p0 for n in cols])
    L1 = np.column_stack([lib_search[n].p1 for n in cols])
    b0 = fit_beta([xs.x1f], [L1], [fd.y1], design.c, design.lam)  # fit on fold 1 -> applied to fold 0
    b1 = fit_beta([xs.x0f], [L0], [fd.y0], design.c, design.lam)  # fit on fold 0 -> applied to fold 1
    lr0 = ST._lr(xs.x0c + (L0 - xs.x0c[:, None]) @ b0, fd.y0, fd.b0)
    lr1 = ST._lr(xs.x1c + (L1 - xs.x1c[:, None]) @ b1, fd.y1, fd.b1)
    out.update({"lr0_2l": lr0, "lr1_2l": lr1, "lr0_x0": xs.lr0, "lr1_x0": xs.lr1})
    if not (lr0 <= xs.lr0 - design.delta and lr1 <= xs.lr1 - design.delta):
        return {**out, "reason": "acceptance_gate"}
    ba = fit_beta([xs.x0a, xs.x1a], [L0, L1], [fd.y0, fd.y1], design.c, design.lam)
    if not np.any(ba > 0):
        return {**out, "reason": "beta_zero"}
    return {**out, "beta": [float(b) for b in ba], "active": True, "reason": "active"}


def audit_utility(ctx: "K.Context", xa: dict, sl: dict, lib_audit: Dict[str, "E.AuditResult"]) -> dict:
    """The one AUDIT score of a fixed second level. β = 0 gives X0's own stored-weight AUDIT utility exactly."""
    if not sl["active"]:
        return {"utility": float(xa["utility"]), "beta_sum": 0.0, "n_substituted": 0}
    pred = np.asarray(xa["pred"], np.float64).copy()
    x = pred.copy()
    nsub = 0
    for n, b in zip(sl["cols"], sl["beta"]):
        if b <= 0:
            continue
        a = lib_audit[f"lib:{n}"]
        La = a.pa if a.status == "completed" and a.pa is not None else None
        if La is None:  # the H317 rule: a failed AUDIT output takes the SEARCH target mean
            La = np.full(len(ctx.ya), ctx.y_search_mean)
            nsub += 1
        pred = pred + b * (np.asarray(La, np.float64) - x)
    rmse = float(np.sqrt(np.mean((pred - ctx.ya) ** 2)))
    return {"utility": float(C.audit_utility_from_rmse(rmse, ctx.ba)), "beta_sum": float(sum(sl["beta"])),
            "n_substituted": nsub, "rmse": rmse}


def evaluate(ctx: "K.Context", lib_search, lib_audit, designs: Sequence[Design]) -> dict:
    """X0 check + X0 + every design's 2L AUDIT utility for one context (dev and P201d share this path)."""
    spec = x0_spec(ctx)
    xs = x0_search(ctx, spec)
    xa = x0_audit(ctx, spec)
    rec = {"cid": ctx.cid, "family": ctx.family, "x0_check": x0_check(ctx, xs, xa), "x0_u": float(xa["utility"]),
           "x0_search_u": xs.u, "designs": {}}
    for d in designs:
        sl = second_level(ctx, xs, lib_search, d)
        au = audit_utility(ctx, xa, sl, lib_audit)
        rec["designs"][d.key()] = {"u": au["utility"], "active": sl["active"], "reason": sl["reason"],
                                   "beta_sum": au["beta_sum"], "n_substituted": au["n_substituted"],
                                   "beta": dict(zip(sl["cols"], sl["beta"]))}
    return rec


def family_lcb(gain_by_cid: Dict[str, float], fam_of: Dict[str, str]) -> dict:
    """PREREG §4.3: per-family mean of the context gains, then mean − 1.96·sd/√F (unclipped)."""
    by: Dict[str, List[float]] = {}
    for c, g in gain_by_cid.items():
        by.setdefault(fam_of[c], []).append(g)
    v = np.array([np.mean(x) for _, x in sorted(by.items())])
    F = len(v)
    sd = float(np.std(v, ddof=1)) if F > 1 else math.inf
    mean = float(v.mean())
    return {"families": F, "mean": mean, "sd": sd, "lcb": mean - 1.96 * sd / math.sqrt(F),
            "won": int((v > 1e-4).sum()), "tied": int((np.abs(v) <= 1e-4).sum()), "lost": int((v < -1e-4).sum())}


def select(rows: Sequence[dict]) -> dict:
    """rows: {design: Design, lcb: float, ...}. Largest LCB; ties within 1e-6 -> conservative_key."""
    top = max(r["lcb"] for r in rows)
    near = [r for r in rows if r["lcb"] >= top - 1e-6]
    return min(near, key=lambda r: conservative_key(r["design"]))


def lib_from_dir(ctx: "K.Context", lib_dir: Path):
    """The 4 classical learners' SEARCH OOF / AUDIT results (lib_oof_h354.py outputs); LLM members are never read."""
    import json

    ls, la = {}, {}
    for name in CLASSICAL:
        f = Path(lib_dir) / name / f"{ctx.cid}.json"
        r = json.loads(f.read_text()) if f.exists() else {"status": "missing"}

        def arr(k: str, r: dict = r) -> Optional[np.ndarray]:
            return None if r.get(k) is None else np.asarray(r[k], np.float64)

        ls[name] = K.lib_search_result(ctx, name, arr("p0"), arr("p1"), r.get("oof_status", r.get("status", "")),
                                       r.get("error", ""))
        la[f"lib:{name}"] = K.lib_audit_result(ctx, name, arr("pa"), r.get("audit_status", r.get("status", "")))
    return ls, la
