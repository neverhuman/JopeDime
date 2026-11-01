"""H177 pooled cross-fit confirmatory selector (205-family pooled B selection + single prespecified candidate per arm + separate A-family confirmation)."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS = (177101, 177102, 177103)
FOLD_SEED, BOOT_SEED = 177140, 177141
CANDIDATE_STEP = 40
ARMS = ("CTRL", "ACC4", "ACC16", "OC15")
ALL_ARMS = ("r882",) + ARMS
PRESPECIFIED_CANDIDATE_STEP = {arm: CANDIDATE_STEP for arm in ARMS}
DRAWS, EPS = 10000, 1e-12


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(FOLD_SEED).permutation(205)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer):
    groups = folds(rows)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def verify_pooled_b_coverage(rows):
    """Prove that B_k never overlaps FIT_k or A_k and that pooling B_0..B_4 covers all 205 families (1,219 contexts) exactly once."""
    all_fams = sorted({r["family_id"] for r in rows})
    all_cids = sorted({r["context_sha256"] for r in rows})
    if len(all_fams) != 205 or len(all_cids) != len(rows):
        raise ValueError("expected 205 distinct families and unique context IDs")
    b_fam_seen, a_fam_seen = [], []
    b_cid_seen, a_cid_seen = [], []
    b_fold_of_fam, a_fold_of_fam = {}, {}
    for outer in range(5):
        fit, select, assess = partition(rows, outer)
        fit_fams = {r["family_id"] for r in fit}
        b_fams = {r["family_id"] for r in select}
        a_fams = {r["family_id"] for r in assess}
        fit_cids = {r["context_sha256"] for r in fit}
        b_cids = {r["context_sha256"] for r in select}
        a_cids = {r["context_sha256"] for r in assess}
        if len(fit_fams) != 123 or len(b_fams) != 41 or len(a_fams) != 41:
            raise ValueError("invalid 123/41/41 family fold sizes")
        if (fit_fams & b_fams) or (fit_fams & a_fams) or (b_fams & a_fams):
            raise ValueError("family overlap between FIT, B, or A in fold")
        if (fit_cids & b_cids) or (fit_cids & a_cids) or (b_cids & a_cids):
            raise ValueError("context overlap between FIT, B, or A in fold")
        for f in b_fams:
            b_fold_of_fam[f] = outer
        for f in a_fams:
            a_fold_of_fam[f] = outer
        b_fam_seen.extend(sorted(b_fams))
        a_fam_seen.extend(sorted(a_fams))
        b_cid_seen.extend(sorted(b_cids))
        a_cid_seen.extend(sorted(a_cids))
    if (
        len(b_fam_seen) != 205
        or sorted(b_fam_seen) != all_fams
        or len(a_fam_seen) != 205
        or sorted(a_fam_seen) != all_fams
        or len(b_cid_seen) != len(rows)
        or sorted(b_cid_seen) != all_cids
        or len(a_cid_seen) != len(rows)
        or sorted(a_cid_seen) != all_cids
    ):
        raise ValueError("pooled B or A failed exact-once coverage")
    for f in all_fams:
        if b_fold_of_fam[f] != (a_fold_of_fam[f] - 1) % 5:
            raise ValueError("B and A fold rotation mismatch")
    return dict(
        total_families=205,
        total_contexts=len(rows),
        b_exact_once=True,
        a_exact_once=True,
        zero_fit_b_overlap=True,
        zero_b_a_same_fold_overlap=True,
    )


class StageOrderGuard:
    """Enforces that candidate step is fixed before B is read and pooled_b_selection_lock.json is fsynced before any A confirmation scoring."""

    def __init__(self, expected_steps=None):
        self.expected_steps = dict(expected_steps or PRESPECIFIED_CANDIDATE_STEP)
        if any(
            not isinstance(v, int) or v <= 0 for v in self.expected_steps.values()
        ) or set(self.expected_steps) != set(ARMS):
            raise ValueError(
                "exactly one positive integer candidate step per arm required"
            )
        self.b_read_started = False
        self.b_locked_path = None

    def candidate_step_for(self, arm):
        if arm not in self.expected_steps:
            raise ValueError(f"unknown arm: {arm}")
        return self.expected_steps[arm]

    def set_candidate_step(self, arm, step):
        raise RuntimeError(
            "prespecified candidate step is immutable and fixed a priori before B is read"
        )

    def mark_b_read(self):
        self.b_read_started = True

    def mark_b_locked(self, lock_path):
        p = Path(lock_path)
        if not p.is_file():
            raise RuntimeError("pooled_b_selection_lock.json does not exist on disk")
        payload = json.loads(p.read_text())
        if (
            payload.get("wave") != "H177"
            or payload.get("pooled_b_families") != 205
            or set(payload.get("decisions", {})) != set(ARMS)
        ):
            raise RuntimeError("invalid pooled_b_selection_lock.json payload")
        self.b_locked_path = p

    def require_b_locked(self):
        if self.b_locked_path is None or not self.b_locked_path.is_file():
            raise RuntimeError(
                "A-family confirmation forbidden before pooled_b_selection_lock.json is written and fsynced"
            )
        payload = json.loads(self.b_locked_path.read_text())
        if payload.get("pooled_b_families") != 205 or set(
            payload.get("decisions", {})
        ) != set(ARMS):
            raise RuntimeError("corrupted pooled_b_selection_lock.json")
        return payload


def fit_slates(fit_rows, orders, seed, outer):
    """Deterministic 24-candidate training slate (top-4 fitness reveals + 20 random unrevealed) per fit context."""
    slates = {}
    for cid in sorted({r["context_sha256"] for r in fit_rows}):
        order = np.asarray(orders[cid], dtype=np.int64)
        if len(order) < 24:
            raise ValueError("at least 24 candidates required per fit context")
        rng = np.random.default_rng(stream_seed(seed, outer, "fit_slate", cid))
        slates[cid] = np.concatenate(
            [order[:4], rng.choice(order[4:], 20, replace=False)]
        )
    return slates


def family_stratified_plan(fit_rows, seed, outer, excluded_cids=()):
    """Build 40 macro-steps x 16 family-stratified microbatches (8 distinct fit families per microbatch)."""
    excl = set(excluded_cids)
    fam_map = {}
    for r in fit_rows:
        cid = r["context_sha256"]
        if cid in excl:
            raise ValueError("excluded context present in fit_rows")
        fam_map.setdefault(r["family_id"], []).append(cid)
    for f in fam_map:
        fam_map[f] = sorted(set(fam_map[f]))
    fams = sorted(fam_map)
    if len(fams) < 8:
        raise ValueError("at least 8 distinct fit families required")
    plan = []
    for u in range(1, CANDIDATE_STEP + 1):
        stream = stream_seed(seed, outer, "family_stratified_u", u)
        rng = np.random.default_rng(stream)
        fam_seq = []
        while len(fam_seq) < 128:
            perm = [fams[i] for i in rng.permutation(len(fams))]
            if fam_seq:
                tail = (
                    set(fam_seq[-(len(fam_seq) % 8) :]) if len(fam_seq) % 8 else set()
                )
                head_ok = [f for f in perm if f not in tail]
                head_rest = [f for f in perm if f in tail]
                perm = head_ok + head_rest
            fam_seq.extend(perm)
        fam_seq = fam_seq[:128]
        microbatches = []
        for m in range(16):
            mb_fams = fam_seq[m * 8 : (m + 1) * 8]
            if len(set(mb_fams)) != 8:
                raise RuntimeError("microbatch family collision")
            cids = [fam_map[f][int(rng.integers(0, len(fam_map[f])))] for f in mb_fams]
            microbatches.append(dict(microbatch=m, families=mb_fams, cids=cids))
        plan.append(dict(step=u, stream=stream, microbatches=microbatches))
    return plan


def family_matrix(values, families):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(families) or not np.isfinite(values).all():
        raise ValueError("invalid context panel")
    fams = sorted(set(families))
    return fams, np.array(
        [values[np.asarray(families) == f].mean(axis=0) for f in fams]
    )


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def expected_mde80_205fam(mde80_41fam, margin=0.001):
    """Project 205-family pooled-B MDE80 from H175's 41-family single-fold MDE80 via sqrt(41/205) = 1/sqrt(5) SE scaling."""
    if (
        mde80_41fam is None
        or not math.isfinite(float(mde80_41fam))
        or float(mde80_41fam) <= 0.0
    ):
        raise ValueError("positive finite mde80_41fam required")
    mde41 = float(mde80_41fam)
    se_scale = math.sqrt(41.0 / 205.0)
    return float(max(margin + (mde41 - margin) * se_scale, mde41 * se_scale))


def pooled_b_select(rows, r882_values, candidate_panels_by_arm, seed=BOOT_SEED):
    """Confirmatory pooled cross-fit selector over all 205 DL-Train families.

    - Requires exactly ONE prespecified candidate step per arm (shape (3, N_ctx)); multi-step arrays raise ValueError.
    - Averages the 3 seeds within each of the 205 families BEFORE running 10,000 paired family-bootstrap draws.
    - Admits iff equal-family gain > 0.001 + 1e-12 AND lower 95% bound > 0.0 AND Holm-adjusted p < 0.05 across arms; ties reject.
    """
    if not candidate_panels_by_arm:
        raise ValueError("non-empty candidate_panels_by_arm required")
    families = [r["family_id"] for r in rows]
    fams_sorted = sorted(set(families))
    n_fams = len(fams_sorted)
    if n_fams != 205:
        raise ValueError(f"pooled B selector requires 205 families, got {n_fams}")

    base = np.asarray(r882_values, dtype=np.float64)
    if base.ndim == 1:
        if base.shape != (len(rows),):
            raise ValueError("1D r882_values must match len(rows)")
        base_3 = np.broadcast_to(base[None, :], (3, len(rows)))
    elif base.ndim == 2 and base.shape == (3, len(rows)):
        base_3 = base
    else:
        raise ValueError("r882_values must have shape (N_ctx,) or (3, N_ctx)")
    if not np.isfinite(base_3).all():
        raise ValueError("nonfinite r882_values")

    fm_base = np.stack([family_matrix(v, families)[1] for v in base_3])  # (3, 205)
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))

    raw_decisions = {}
    pvals_all = {}
    for arm, panel in candidate_panels_by_arm.items():
        if arm not in PRESPECIFIED_CANDIDATE_STEP:
            raise ValueError(f"unknown arm {arm}; prespecified arm required")
        arr = np.asarray(panel, dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (3, len(rows)):
            raise ValueError(
                f"snapshot search forbidden: arm {arm} must provide a single prespecified candidate panel of shape (3, {len(rows)}), got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite candidate panel for arm {arm}")
        fm_arm = np.stack([family_matrix(v, families)[1] for v in arr])  # (3, 205)
        diff_by_seed_fam = fm_arm - fm_base  # (3, 205)
        per_seed_gain = diff_by_seed_fam.mean(axis=1)  # (3,)
        # Average seeds within each family BEFORE family resampling (205 family values).
        seed_avg_fam_gain = diff_by_seed_fam.mean(axis=0)  # (205,)
        mean_gain = float(seed_avg_fam_gain.mean())
        boot_means = seed_avg_fam_gain[ix].mean(axis=1)  # (DRAWS,)
        ci95 = np.quantile(boot_means, [0.025, 0.975]).tolist()
        p_one_sided = float((1 + (boot_means <= 0.0).sum()) / (DRAWS + 1))
        # Also record median-of-seeds bootstrap CI as a companion diagnostic.
        boot_med_seeds = np.median(diff_by_seed_fam[:, ix].mean(axis=2), axis=0)
        med_seed_ci95 = np.quantile(boot_med_seeds, [0.025, 0.975]).tolist()
        pvals_all[arm] = p_one_sided
        raw_decisions[arm] = dict(
            prespecified_candidate_step=PRESPECIFIED_CANDIDATE_STEP[arm],
            mean_gain=mean_gain,
            ci95=ci95,
            per_seed_gain=per_seed_gain.tolist(),
            median_seed_gain=float(np.median(per_seed_gain)),
            median_seed_ci95=med_seed_ci95,
            positive_seeds=int((per_seed_gain > 0.0).sum()),
            p_one_sided=p_one_sided,
            seed_averaged_family_means=fm_arm.mean(axis=0).tolist(),
        )

    holm_all = holm(pvals_all)
    treat_keys = [a for a in pvals_all if a != "CTRL"]
    holm_treat = holm({a: pvals_all[a] for a in treat_keys}) if treat_keys else {}

    decisions = {}
    for arm, d in raw_decisions.items():
        unadj_ok = bool(d["mean_gain"] > 0.001 + EPS and d["ci95"][0] > 0.0)
        h_all = float(holm_all[arm])
        h_tr = float(holm_treat[arm]) if arm in holm_treat else h_all
        admitted = bool(unadj_ok and h_all < 0.05)
        admitted_treat3 = bool(unadj_ok and h_tr < 0.05)
        cand_step = d["prespecified_candidate_step"]
        decisions[arm] = dict(
            **d,
            holm_p=h_all,
            holm_p_treatments3=h_tr,
            unadjusted_admitted=unadj_ok,
            admitted_treatments3=admitted_treat3,
            admitted=admitted,
            selected_step=cand_step if admitted else 0,
            selected_step_unadjusted=cand_step if unadj_ok else 0,
        )
    return decisions


def bt_ratings(wins, anchor=0, prior=0.01):
    """H148 canonical 400-iteration MM, r882 anchor, pseudo-win prior .01."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2 * prior * (1 - np.eye(n)))
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500 + 400 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > 1e-4, 1.0, np.where(diff < -1e-4, 0.0, 0.5)) * (
        1 - np.eye(values.shape[1])
    )
    return family_matrix(outcomes, families)[1]


def summarize(rows, panel, seed=BOOT_SEED):
    """Summarize a 5-policy panel (r882, CTRL, ACC4, ACC16, OC15) using within-family seed averaging + median-of-seeds companion + H148 BT Elo."""
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), 5) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 5-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # (3, n_fams, 5)
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    contrasts = {}

    def add_contrast(name, col_a, col_b):
        diff = fm[:, :, col_a] - fm[:, :, col_b]  # (3, n_fams)
        per_seed = diff.mean(axis=1)
        # Primary H177 functional: average seeds within family before resampling.
        seed_avg_fam = diff.mean(axis=0)  # (n_fams,)
        boot_avg = seed_avg_fam[ix].mean(axis=1)
        # Companion median-of-seeds functional.
        boot_med = np.median(diff[:, ix].mean(axis=2), axis=0)
        contrasts[name] = dict(
            mean_gain=float(seed_avg_fam.mean()),
            ci95=np.quantile(boot_avg, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (boot_avg <= 0.0).sum()) / (DRAWS + 1)),
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            median_seed_ci95=np.quantile(boot_med, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0.0).sum()),
        )

    for col, arm in enumerate(ARMS, 1):
        add_contrast(arm + "_minus_r882", col, 0)
        if col > 1:
            add_contrast(arm + "_minus_CTRL", col, 1)
    add_contrast("ACC16_minus_ACC4", 3, 2)
    add_contrast("OC15_minus_ACC4", 4, 2)

    adjusted_all4 = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in ARMS}
    )
    adjusted_treat3 = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in ARMS[1:]}
    )
    for arm in ARMS:
        c = contrasts[arm + "_minus_r882"]
        c["holm_p"] = adjusted_all4[arm]
        c["holm_p_treatments3"] = adjusted_treat3.get(arm, adjusted_all4[arm])
        c["advance"] = bool(
            c["mean_gain"] > 0.001 + EPS and c["ci95"][0] > 0.0 and c["holm_p"] < 0.05
        )
    fw = np.stack([family_wins(v, families) for v in values])
    # Seed-averaged family wins for pooled Elo, plus per-seed Elo.
    elo_by_seed = bt_ratings(fw.sum(axis=1))
    elo_seed_avg = bt_ratings(fw.mean(axis=0).sum(axis=0))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = bt_ratings(fw.mean(axis=0)[ei].sum(axis=1))
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        seed_averaged_family_utility=means.mean(axis=0).tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        elo_by_seed=elo_by_seed.tolist(),
        median_elo=np.median(elo_by_seed, axis=0).tolist(),
        seed_averaged_elo=elo_seed_avg.tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )
