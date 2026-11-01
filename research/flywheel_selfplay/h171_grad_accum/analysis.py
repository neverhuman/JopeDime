"""H171 fixed cross-fit protocol and family-stratified gradient accumulation batch plan; model-free inference only."""

import hashlib
import numpy as np

SEEDS = (171101, 171102, 171103)
FOLD_SEED, BOOT_SEED = 171140, 171141
STEPS = (0, 10, 20, 30, 40)
CTRL_EX4_SNAPSHOT_UPDATES = {0: 0, 40: 10, 80: 20, 120: 30, 160: 40}
ARMS = ("CTRL", "ACC4", "ACC16", "CTRL_EX4")
ALL_ARMS = ("r882",) + ARMS
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
    """Build 40 macro-steps x 16 family-stratified microbatches (8 distinct fit families per microbatch).

    - CTRL uses microbatch 0 at each macro-step u=1..40 (40 updates, 40 microbatches).
    - ACC4 accumulates microbatches 0..3 at each macro-step u=1..40 (40 updates, 160 microbatches, 32 distinct families/update).
    - ACC16 accumulates microbatches 0..15 at each macro-step u=1..40 (40 updates, 640 microbatches, covering all fit families/update).
    - CTRL_EX4 steps once on each of microbatches 0..3 across u=1..40 (160 updates, exact same 160 microbatches as ACC4).
    """
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
    for u in range(1, 41):
        stream = stream_seed(seed, outer, "family_stratified_u", u)
        rng = np.random.default_rng(stream)
        # Concatenate random permutations of fams so every 8-context microbatch has 8 distinct families
        # and 16 microbatches (128 draws) cover all fit families (123 in 5-fold cross-fit).
        fam_seq = []
        while len(fam_seq) < 128:
            perm = [fams[i] for i in rng.permutation(len(fams))]
            # Ensure boundary slice of 8 has 8 distinct families when wrapping aroundperm1 -> perm2
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


def choose(rows, panel, seed):
    mat = np.asarray(panel, dtype=np.float64)
    if mat.shape != (len(rows), 5) or not np.isfinite(mat).all():
        raise ValueError("five finite selector snapshots required")
    _, fm = family_matrix(mat, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    # Prespecified numerical ties select the earliest step, including r882.
    best = next(i for i, v in enumerate(means) if v >= means.max() - EPS)
    diff = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(diff[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(diff.mean())
    accepted = gain > 0.001 + EPS and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        selected_step=STEPS[best] if accepted else 0,
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


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


def holm(pvals):
    result, previous = {}, 0.0
    for rank, key in enumerate(sorted(pvals, key=pvals.get)):
        previous = max(previous, min(1.0, pvals[key] * (len(pvals) - rank)))
        result[key] = previous
    return result


def summarize(rows, panel, seed=BOOT_SEED):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), 5) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed 5-policy panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(seed).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)
    contrasts = {}

    def add_contrast(name, col_a, col_b):
        diff = fm[:, :, col_a] - fm[:, :, col_b]
        per_seed = diff.mean(axis=1)
        boot = np.median(diff[:, ix].mean(axis=2), axis=0)
        contrasts[name] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        )

    for col, arm in enumerate(ARMS, 1):
        add_contrast(arm + "_minus_r882", col, 0)
        if col > 1:
            add_contrast(arm + "_minus_CTRL", col, 1)
    # Example-matched mechanism contrast: 4B gradient accumulation (ACC4, 40 updates) vs live B (CTRL_EX4, 160 updates).
    add_contrast("ACC4_minus_CTRL_EX4", 2, 4)
    # Effective-batch dose contrast: 16B (ACC16) vs 4B (ACC4).
    add_contrast("ACC16_minus_ACC4", 3, 2)

    adjusted = holm(
        {arm: contrasts[arm + "_minus_r882"]["p_one_sided"] for arm in ARMS[1:]}
    )
    for arm in ARMS[1:]:
        c = contrasts[arm + "_minus_r882"]
        c["holm_p"] = adjusted[arm]
        c["advance"] = bool(
            c["median_gain"] > 0.001 + EPS and c["ci95"][0] > 0 and adjusted[arm] < 0.05
        )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1))
    ei = np.random.default_rng(seed + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2)), axis=0)
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=means.tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_ci95=np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def arm_identity_audit(
    selections,
    raw_b_panels_by_arm,
    primary_panel,
    secondary_panel,
    adapter_hashes_by_arm,
):
    """Compare matched-context raw B vectors, selected primary/secondary utility vectors, selected/best steps, and adapter hashes separately from arm-seeded bootstrap CI metadata."""
    n_cells = len(selections)
    primary_arr = np.asarray(primary_panel, dtype=np.float64)
    secondary_arr = np.asarray(secondary_panel, dtype=np.float64)
    out = {}
    for ai, a in enumerate(ARMS[1:], 2):
        sel_steps_eq = [d[a]["selected_step"] for d in selections] == [
            d["CTRL"]["selected_step"] for d in selections
        ]
        best_steps_eq = [d[a]["best_step"] for d in selections] == [
            d["CTRL"]["best_step"] for d in selections
        ]
        fam_means_eq = all(
            np.array_equal(
                np.asarray(d[a]["family_means"], dtype=np.float64),
                np.asarray(d["CTRL"]["family_means"], dtype=np.float64),
            )
            for d in selections
        )
        raw_b_eq_cells = sum(
            1
            for b_a, b_c in zip(
                raw_b_panels_by_arm[a], raw_b_panels_by_arm["CTRL"], strict=True
            )
            if np.array_equal(
                np.asarray(b_a, dtype=np.float64), np.asarray(b_c, dtype=np.float64)
            )
        )
        raw_b_all_eq = bool(raw_b_eq_cells == n_cells)
        prim_eq = bool(np.array_equal(primary_arr[:, :, ai], primary_arr[:, :, 1]))
        sec_eq = bool(np.array_equal(secondary_arr[:, :, ai], secondary_arr[:, :, 1]))
        step40_hash_eq_cells = sum(
            1
            for h_a, h_c in zip(
                adapter_hashes_by_arm[a], adapter_hashes_by_arm["CTRL"], strict=True
            )
            if h_a["40"] == h_c["40"]
        )
        sel_hash_eq_cells = sum(
            1
            for d, h_a, h_c in zip(
                selections,
                adapter_hashes_by_arm[a],
                adapter_hashes_by_arm["CTRL"],
                strict=True,
            )
            if h_a[str(d[a]["selected_step"])] == h_c[str(d["CTRL"]["selected_step"])]
        )
        out[a] = dict(
            selected_steps_identical_to_ctrl=bool(sel_steps_eq),
            best_steps_identical_to_ctrl=bool(best_steps_eq),
            family_means_identical_to_ctrl=bool(fam_means_eq),
            raw_b_identical_cells_to_ctrl=int(raw_b_eq_cells),
            raw_b_vectors_identical_to_ctrl=raw_b_all_eq,
            primary_vectors_identical_to_ctrl=prim_eq,
            secondary_vectors_identical_to_ctrl=sec_eq,
            step40_adapter_hashes_identical_to_ctrl=int(step40_hash_eq_cells),
            selected_adapter_hashes_identical_to_ctrl=int(sel_hash_eq_cells),
            selector_identical_to_ctrl=bool(sel_steps_eq and prim_eq and sec_eq),
        )
    return out
