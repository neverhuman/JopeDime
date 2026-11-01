"""H198 statistical analysis and Stage-1 (B-only) / Stage-2 (pooled-A) decision rules.

Fold, partition, utility, family-matrix, Holm, win-matrix and Bradley-Terry helpers are imported from merged H188
(`research/flywheel_selfplay/h188_kernel_inputs/analysis.py`, hash-pinned in manifest.json).
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from treatment import GRID, GRID_LABELS

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "h198_h188_analysis", HERE.parent / "h188_kernel_inputs" / "analysis.py"
)
h188 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h188)

SEEDS = (198101, 198102, 198103)
FOLD_SEEDS = {198101: 198141, 198102: 198142, 198103: 198143}
BOOT_SEED = 198150
DRAWS, EPS, MARGIN = 10_000, 1e-12, 0.001
ARMS = ("R882", "TTT_K", "TTT_SHUF", "TTT_L2")

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


def equal_family_mean(rows, values):
    return float(
        family_matrix(
            np.asarray(values, dtype=np.float64), [r["family_id"] for r in rows]
        )[1].mean()
    )


def fold_jaccards(folds_by_seed):
    out = {}
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1 :]:
            g1, g2 = folds_by_seed[s1], folds_by_seed[s2]
            mat = [
                [len(set(x) & set(y)) / len(set(x) | set(y)) for y in g2]
                for x in g1
            ]
            diag = [mat[k][k] for k in range(5)]
            out[f"{s1}-{s2}"] = dict(
                matrix=mat,
                corresponding_fold_jaccard=diag,
                diagonal_mean=float(np.mean(diag)),
                max_entry=float(np.max(mat)),
            )
    return out


def select_cell(b_rows, grid_utilities_by_cid):
    """Select (k*, lr*) on B_j only by maximizing equal-family utility; ties break toward earliest grid entry
    (smallest k, then smallest lr). A utilities are never passed or read."""
    b_cids = [r["context_sha256"] for r in b_rows]
    if any(cid not in grid_utilities_by_cid for cid in b_cids):
        raise ValueError("missing B context in grid_utilities_by_cid")
    b_util = []
    for idx in range(len(GRID)):
        vals = [float(grid_utilities_by_cid[cid][idx]) for cid in b_cids]
        b_util.append(equal_family_mean(b_rows, vals))
    best = int(np.argmax(b_util))
    k_star, lr_star = GRID[best]
    return dict(
        grid_index=best,
        grid_label=GRID_LABELS[best],
        k_star=int(k_star),
        lr_star=float(lr_star),
        b_equal_family_utility=b_util,
    )


def val_modal_grid_index(indices):
    if len(indices) != 5 or any(
        type(i) is not int or i not in range(len(GRID)) for i in indices
    ):
        raise ValueError("five valid grid indices required")
    return int(np.argmax(np.bincount(indices, minlength=len(GRID))))


def stage1_lock(
    train,
    grid_utilities_by_cid,
    folds_by_seed,
    fold_manifest_sha256,
    initial_adapter_sha256,
):
    """Compute all 15 (seed, outer) B-only grid selections and build the lock payload prior to Stage-2 A scoring."""
    all_fams = {r["family_id"] for r in train}
    cells = []
    by_seed_indices = {s: [] for s in SEEDS}
    for seed in SEEDS:
        groups = folds_by_seed[seed]
        for outer in range(5):
            fit, b, a = partition(all_fams, outer, groups)
            b_rows = [r for r in train if r["family_id"] in b]
            b_only_map = {
                r["context_sha256"]: grid_utilities_by_cid[r["context_sha256"]]
                for r in b_rows
            }
            sel = select_cell(b_rows, b_only_map)
            by_seed_indices[seed].append(sel["grid_index"])
            b_selected = np.asarray(
                [
                    float(b_only_map[r["context_sha256"]][sel["grid_index"]])
                    for r in b_rows
                ],
                dtype=np.float64,
            )
            cells.append(
                dict(
                    seed=int(seed),
                    outer=int(outer),
                    **sel,
                    fit_families=sorted(fit),
                    b_families=sorted(b),
                    a_families=sorted(a),
                    b_panel_sha256=hashlib.sha256(b_selected.tobytes()).hexdigest(),
                )
            )
    val_grid = {}
    for seed in SEEDS:
        v_idx = val_modal_grid_index(by_seed_indices[seed])
        val_grid[str(seed)] = dict(
            grid_index=v_idx,
            grid_label=GRID_LABELS[v_idx],
            k_star=int(GRID[v_idx][0]),
            lr_star=float(GRID[v_idx][1]),
        )
    return dict(
        schema="jope.h198.ttt-b-lock/v1",
        wave="H198",
        written_before_any_A_or_treatment_DL_Val_scoring=True,
        fold_manifest_sha256=fold_manifest_sha256,
        adapter_sha256=initial_adapter_sha256,
        grid=[dict(index=i, label=GRID_LABELS[i], k=k, lr=lr) for i, (k, lr) in enumerate(GRID)],
        cells=cells,
        val_grid_by_seed=val_grid,
    )


def lock_bytes(lock):
    return (json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_stage1_lock(lock_path, fold_manifest_sha256, initial_adapter_sha256):
    path = Path(lock_path)
    if not path.is_file():
        raise RuntimeError("Stage-2 A scoring requires on-disk ttt_selection_lock.json")
    raw = path.read_bytes()
    data = json.loads(raw.decode())
    if (
        data.get("schema") != "jope.h198.ttt-b-lock/v1"
        or data.get("wave") != "H198"
        or data.get("written_before_any_A_or_treatment_DL_Val_scoring") is not True
        or data.get("fold_manifest_sha256") != fold_manifest_sha256
        or data.get("adapter_sha256") != initial_adapter_sha256
        or len(data.get("cells", [])) != 15
    ):
        raise RuntimeError("invalid or incomplete ttt_selection_lock.json")
    seen = set()
    for cell in data["cells"]:
        key = (cell["seed"], cell["outer"])
        seen.add(key)
        means = np.asarray(cell["b_equal_family_utility"], dtype=np.float64)
        idx = int(np.argmax(means))
        if (
            means.shape != (len(GRID),)
            or not np.isfinite(means).all()
            or cell["grid_index"] != idx
            or (cell["k_star"], cell["lr_star"]) != GRID[idx]
            or cell["grid_label"] != GRID_LABELS[idx]
        ):
            raise RuntimeError(f"B selection lock integrity failure in cell {key}")
    if seen != {(s, o) for s in SEEDS for o in range(5)}:
        raise ValueError("lock does not cover all 15 (seed, outer) cells")
    return data, hashlib.sha256(raw).hexdigest()


def contrast_stats(rows, arm_panel, base_panel, seed=BOOT_SEED):
    families = [r["family_id"] for r in rows]
    fm_arm = np.stack([family_matrix(v, families)[1] for v in np.asarray(arm_panel, dtype=np.float64)])
    fm_base = np.stack([family_matrix(v, families)[1] for v in np.asarray(base_panel, dtype=np.float64)])
    delta = fm_arm - fm_base
    n = delta.shape[1]
    draw = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    boot = np.median(delta[:, draw].mean(axis=2), axis=0)
    per_seed = delta.mean(axis=1)
    p = float((1 + np.count_nonzero(boot <= 0.0)) / (DRAWS + 1))
    return dict(
        per_seed_gain=per_seed.tolist(),
        median_gain=float(np.median(per_seed)),
        mean_gain=float(delta.mean()),
        ci95=_percentile_ci(boot),
        positive_seeds=int((per_seed > 0.0).sum()),
        p_one_sided=p,
        holm_multiplicity=1,
        holm_p=p,
        harm_rate=float((delta.mean(axis=0) < 0.0).mean()),
        harm_rate_by_seed=(delta < 0.0).mean(axis=1).tolist(),
        harm_below_minus_001=float((delta.mean(axis=0) < -0.01).mean()),
    )


def pooled_select(rows, base_values, candidate_panels, seed=BOOT_SEED):
    """Merged H177/H188/H192 pooled rule with m = len(candidate_panels) (m = 1 for sole primary TTT_K)."""
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
    per_seed = elo[:, 1]
    return dict(
        per_seed=per_seed.tolist(),
        median=float(np.median(per_seed)),
        elo_lift_per_seed=(per_seed - 1500.0).tolist(),
        elo_lift_median=float(np.median(per_seed) - 1500.0),
        ci95=_percentile_ci(med[:, 1]),
    )


def h178_gate(receipt, h177_analysis_sha256):
    return bool(
        receipt.get("status") == "COMPLETE"
        and receipt.get("null_and_harmful_admission_le_005") is True
        and receipt.get("h177_analysis_sha256") == h177_analysis_sha256
    )


def verdict(ttt_k_vs_r882, ttt_shuf_vs_r882, ttt_k_minus_shuf, selector, h178_ok):
    """ADVANCE requires TTT_K vs R882 LCB95 > 0 AND Holm p < 0.05 AND harm <= 10% AND TTT_K > TTT_SHUF, plus H177
    selector admission (median/mean gain > +0.001 + 1e-12) and H178 calibration receipt gate. Strict improvement only;
    ties reject."""
    gates = dict(
        ttt_k_lcb95_gt_0=bool(ttt_k_vs_r882["ci95"][0] > 0.0),
        ttt_k_holm_p_lt_005=bool(ttt_k_vs_r882["holm_p"] < 0.05),
        ttt_k_harm_le_010=bool(ttt_k_vs_r882["harm_rate"] <= 0.10),
        ttt_k_gt_ttt_shuf=bool(
            ttt_k_vs_r882["median_gain"] > ttt_shuf_vs_r882["median_gain"] + EPS
            and ttt_k_minus_shuf["median_gain"] > EPS
        ),
        ttt_k_median_gain_gt_margin=bool(ttt_k_vs_r882["median_gain"] > MARGIN + EPS),
        pooled_selector_admits=bool(selector["TTT_K"]["admitted"]),
        h178_receipt_gate=bool(h178_ok),
    )
    advance = all(gates.values())
    return dict(
        verdict="ADVANCE" if advance else "NO_ADVANCE",
        disposition=(
            "CANDIDATE for flywheel adoption; ENG decides; live loop untouched"
            if advance
            else "reject treatment; keep r882"
        ),
        serving_promotion=False,
        holm_multiplicity=1,
        gates=gates,
    )
