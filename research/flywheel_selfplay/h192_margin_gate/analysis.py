"""H192 strict-margin (confidence-gated) selection between the anchor-only base pick and the r882 full pick, 0 updates.

Fold, partition, utility, family-matrix, Holm, win-matrix and Bradley-Terry helpers are IMPORTED from merged H188
(`research/flywheel_selfplay/h188_kernel_inputs/analysis.py`, hash-pinned in manifest.json). `pooled_select` is the
merged H177/H188 pooled rule with the arm count made a parameter (H177 hard-codes m=4, H188 asserts m=3; H192 is m=1).
All selection is pure numpy over saved per-context cost vectors; no data access or model code here.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "h192_h188_analysis", HERE.parent / "h188_kernel_inputs" / "analysis.py"
)
h188 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h188)

SEEDS = (192101, 192102, 192103)
FOLD_SEEDS = {192101: 192141, 192102: 192142, 192103: 192143}
BOOT_SEED = 192150
QUANTILES = (0.25, 0.50, 0.75, 0.90)
TAU_LABELS = ("TAU0", "Q25", "Q50", "Q75", "Q90")
DRAWS, EPS, MARGIN = 10_000, 1e-12, 0.001

family_folds, partition, context_utility, family_matrix, holm = (
    h188.family_folds,
    h188.partition,
    h188.context_utility,
    h188.family_matrix,
    h188.holm,
)
family_win_matrix, bt_ratings, _percentile_ci = (
    h188.family_win_matrix,
    h188.bt_ratings,
    h188._percentile_ci,
)
zero_update_gap, family_values = h188.zero_update_gap, h188.family_values


# ------------------------------------------------------------------ the gate (per context)


def gate_inputs(full_cost, anchor_cost, fitness, gate):
    """Base pick = argmin anchor-only cost (r882 with the whole adapter re-rank removed); full pick = argmin r882 cost.
    Normalized margin = (full[base] - full[top1]) / std(full over the pool), >= 0; 0 if the picks agree or std == 0."""
    full = np.asarray(full_cost, dtype=np.float64)
    anchor = np.asarray(anchor_cost, dtype=np.float64)
    if (
        full.shape != anchor.shape
        or full.ndim != 1
        or not (np.isfinite(full).all() and np.isfinite(anchor).all())
    ):
        raise ValueError("matching finite cost vectors required")
    base, top1 = int(np.argmin(anchor)), int(np.argmin(full))
    sd = float(full.std())
    margin = 0.0 if base == top1 or sd == 0.0 else float((full[base] - full[top1]) / sd)
    util = context_utility(fitness, gate)
    return dict(
        base=base,
        top1=top1,
        margin=margin,
        u_base=float(util[base]),
        u_top1=float(util[top1]),
    )


def gated_pick(g, tau):
    """tau = 0 is R882 exactly (the full argmin, including its tie-breaking); otherwise override the base pick with the
    full pick only when the normalized margin strictly exceeds tau."""
    if tau == 0.0:
        return g["top1"], g["u_top1"]
    return (g["top1"], g["u_top1"]) if g["margin"] > tau else (g["base"], g["u_base"])


def tau_grid(fit_margins):
    """{0} plus the q25/q50/q75/q90 of the margin over FIT contexts where the two picks disagree (margin > 0)."""
    m = np.asarray([x for x in fit_margins if x > 0.0], dtype=np.float64)
    qs = (
        np.quantile(m, QUANTILES).tolist()
        if len(m)
        else [float("inf")] * len(QUANTILES)
    )
    return [0.0] + [float(q) for q in qs]


def equal_family_mean(rows, values):
    return float(
        family_matrix(
            np.asarray(values, dtype=np.float64), [r["family_id"] for r in rows]
        )[1].mean()
    )


# ------------------------------------------------------------------ stage 1: selection on B only (no A input)


def select_cell(fit_rows, b_rows, gates):
    """tau* = argmax equal-family utility on B over the grid (ties -> the earliest grid entry, i.e. tau = 0 / R882).
    Uses FIT margins for the grid and B utilities for the choice; A is not an input."""
    grid = tau_grid([gates[r["context_sha256"]]["margin"] for r in fit_rows])
    b_util = [
        equal_family_mean(
            b_rows, [gated_pick(gates[r["context_sha256"]], t)[1] for r in b_rows]
        )
        for t in grid
    ]
    best = int(np.argmax(b_util))  # np.argmax returns the first maximum
    return dict(
        grid=grid,
        b_equal_family_utility=b_util,
        tau_star_index=best,
        tau_star=grid[best],
    )


def stage1(train, gates, folds_by_seed):
    """All (seed, outer) selections and the lock payload. Raises if any family would cross FIT/B/A."""
    cells = []
    for seed in SEEDS:
        groups = folds_by_seed[seed]
        for outer in range(5):
            fit, b, a = partition({r["family_id"] for r in train}, outer, groups)
            fit_rows = [r for r in train if r["family_id"] in fit]
            b_rows = [r for r in train if r["family_id"] in b]
            sel = select_cell(fit_rows, b_rows, gates)
            b_panel = np.asarray(
                [
                    gated_pick(gates[r["context_sha256"]], sel["tau_star"])[1]
                    for r in b_rows
                ]
            )
            cells.append(
                dict(
                    seed=seed,
                    outer=outer,
                    **sel,
                    a_families=sorted(a),
                    b_families=sorted(b),
                    b_panel_sha256=hashlib.sha256(b_panel.tobytes()).hexdigest(),
                )
            )
    return dict(
        schema="jope.h192.mgate-lock/v1",
        written_before_any_A_or_DL_Val_scoring=True,
        cells=cells,
    )


# ------------------------------------------------------------------ stage 2: pooled A


def stage2_panels(train, gates, lock):
    """Pooled-A panels [seed, context] for R882 (= tau 0), MGATE (each cell's tau*) and every fixed grid position."""
    index = {r["context_sha256"]: i for i, r in enumerate(train)}
    shape = (len(SEEDS), len(train))
    panels = {name: np.full(shape, np.nan) for name in ("R882", "MGATE", *TAU_LABELS)}
    override = {
        name: [0, 0] for name in TAU_LABELS
    }  # [overrides used, disagreements] over A contexts
    for cell in lock["cells"]:
        si = SEEDS.index(cell["seed"])
        a_fams = set(cell["a_families"])
        for r in train:
            if r["family_id"] not in a_fams:
                continue
            g, i = gates[r["context_sha256"]], index[r["context_sha256"]]
            panels["R882"][si, i] = g["u_top1"]
            panels["MGATE"][si, i] = gated_pick(g, cell["tau_star"])[1]
            for label, tau in zip(TAU_LABELS, cell["grid"], strict=True):
                pick, util = gated_pick(g, tau)
                panels[label][si, i] = util
                if g["base"] != g["top1"]:
                    override[label][1] += 1
                    override[label][0] += int(pick == g["top1"])
    if any(not np.isfinite(p).all() for p in panels.values()):
        raise ValueError("pooled A coverage incomplete")
    return panels, override


def pooled_select(rows, base_values, candidate_panels, seed=BOOT_SEED):
    """Merged H177/H188 pooled rule with m = len(arms): seed-averaged family gains, 10,000 paired family draws; admit
    iff mean gain > 0.001 + 1e-12 AND lower 95% bound > 0 AND Holm(m) p < 0.05. Ties reject."""
    families = [r["family_id"] for r in rows]
    n_fams = len(set(families))
    base = np.asarray(base_values, dtype=np.float64)
    fm_base = np.stack([family_matrix(v, families)[1] for v in base])
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    raw, pvals = {}, {}
    for arm, panel in candidate_panels.items():
        diff = (
            np.stack(
                [
                    family_matrix(v, families)[1]
                    for v in np.asarray(panel, dtype=np.float64)
                ]
            )
            - fm_base
        )
        seed_avg = diff.mean(axis=0)
        boot = seed_avg[ix].mean(axis=1)
        p = float((1 + (boot <= 0.0).sum()) / (DRAWS + 1))
        pvals[arm] = p
        raw[arm] = dict(
            mean_gain=float(seed_avg.mean()),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            p_one_sided=p,
            per_seed_gain=diff.mean(axis=1).tolist(),
            positive_seeds=int((diff.mean(axis=1) > 0).sum()),
        )
    adj = holm(pvals)
    for arm, d in raw.items():
        unadj = bool(d["mean_gain"] > MARGIN + EPS and d["ci95"][0] > 0.0)
        d.update(
            holm_multiplicity=len(raw),
            holm_p=float(adj[arm]),
            admitted=bool(unadj and adj[arm] < 0.05),
        )
    return raw


def median_seed_contrast(rows, panel, base, seed=BOOT_SEED):
    families = [r["family_id"] for r in rows]
    fm_p = np.stack([family_matrix(v, families)[1] for v in panel])
    fm_b = np.stack([family_matrix(v, families)[1] for v in base])
    delta = fm_p - fm_b
    n = delta.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    boot = np.median(
        delta[:, draw].mean(axis=2), axis=0
    )  # axis=0: median over seeds per draw (#1512)
    per_seed = delta.mean(axis=1)
    return dict(
        per_seed_gain=per_seed.tolist(),
        median_gain=float(np.median(per_seed)),
        ci95=_percentile_ci(boot),
        positive_seeds=int((per_seed > 0).sum()),
    )


def elo_pair(rows, panel, base, seed=BOOT_SEED):
    families = [r["family_id"] for r in rows]
    wins = np.stack(
        [
            family_win_matrix(np.stack([base[s], panel[s]], axis=1), families)
            for s in range(len(panel))
        ]
    )
    elo = bt_ratings(wins.sum(axis=1))
    n = wins.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    parts = [
        bt_ratings(np.take(wins, draw[st : st + 100], axis=1).sum(axis=2))
        for st in range(0, DRAWS, 100)
    ]
    med = np.median(np.concatenate(parts, axis=1), axis=0)
    return dict(
        per_seed=elo[:, 1].tolist(),
        median=float(np.median(elo[:, 1])),
        ci95=_percentile_ci(med[:, 1]),
    )


def h178_gate(receipt, h177_analysis_sha256):
    """The H178 calibration receipt must be COMPLETE, bind the current H177 analysis bytes, and show null/harmful
    admission <= 0.05."""
    return bool(
        receipt.get("status") == "COMPLETE"
        and receipt.get("null_and_harmful_admission_le_005") is True
        and receipt.get("h177_analysis_sha256") == h177_analysis_sha256
    )


def verdict(selector, contrast, h178_ok):
    """ADVANCE only if the pooled selector admits MGATE AND the median-of-seeds gain > +0.001 AND its LCB95 > 0 AND the
    H178 receipt gate passes. Ties reject."""
    gates = dict(
        pooled_selector_admits=bool(selector["MGATE"]["admitted"]),
        median_gain_gt_margin=contrast["median_gain"] > MARGIN + EPS,
        median_lcb95_gt_0=contrast["ci95"][0] > 0,
        h178_receipt_gate=bool(h178_ok),
    )
    return dict(verdict="ADVANCE" if all(gates.values()) else "NO_ADVANCE", gates=gates)


def lock_bytes(lock):
    return (json.dumps(lock, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
