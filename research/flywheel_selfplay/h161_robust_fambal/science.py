"""H161 prospective family cross-fit partitions, equal-family sampler, strict selection, drift matching, and BT Elo."""

import hashlib
import random
import numpy as np

SEEDS = (161101, 161102, 161103)
FOLD_SEED = 161140
BOOT_SEED = 161141
STEPS = (0, 10, 20, 30, 40)
DRAWS = 10000
ARMS = ("CTRL", "CLIP", "FAMBAL", "CLIP_FAMBAL")
TREATMENT_ARMS = ("CLIP", "FAMBAL", "CLIP_FAMBAL")
HOLM_M = 3
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
ACCUM_STEPS = 2
MICROBATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = ACCUM_STEPS * MICROBATCH_SIZE
MARGIN = 0.001
TIE = 1e-12
DRIFT_MATCH_BOUNDS = (0.85, 1.15)


def folds(rows):
    ids = [r["context_sha256"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("unique contexts required")
    if any(
        r.get("dl_split") != "DL-Train"
        or r.get("has_meta_test") is True
        or r.get("source_split") not in ("train", "inner")
        for r in rows
    ):
        raise ValueError("DL-Train inner rows only")
    families = sorted({r["family_id"] for r in rows})
    if len(families) < 5:
        raise ValueError("at least five families required")
    return [
        sorted(g.tolist())
        for g in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), 5
        )
    ]


def partition(rows, outer):
    groups = folds(rows)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = [r for r in rows if r["family_id"] not in a | b]
    select = [r for r in rows if r["family_id"] in b]
    assess = [r for r in rows if r["family_id"] in a]
    assert not a & b and not {r["family_id"] for r in fit} & (a | b)
    return fit, select, assess


def _sample_slate(order, rng):
    k = min(48, 192, len(order))
    rest = order[k:] if len(order) > k else order
    return np.concatenate(
        [
            order[:k],
            rng.choice(rest, 192 - k, replace=len(rest) < 192 - k),
        ]
    )


def batch_plan(fit_ids, orders, seed):
    """H140/H148 paired stream (40 steps, 16 contexts per step) for CTRL and CLIP."""
    if seed not in SEEDS or not fit_ids:
        raise ValueError("registered seed and fit contexts required")
    ids = tuple(sorted(fit_ids))
    sampler = seed * 1_000_003 + 10_007
    plan = []
    for rnd in range(1, 21):
        rng = np.random.RandomState((sampler + rnd * 104729) % 2**32)
        rot = rnd % len(ids)
        rotated = ids[rot:] + ids[:rot]
        for step in range(8):
            stream = int.from_bytes(
                hashlib.sha256(
                    f"flywheel-batch/v1:{sampler}:{rnd}:{step}".encode()
                ).digest()[:8],
                "little",
            )
            cids = random.Random(stream).choices(rotated, k=EFFECTIVE_BATCH_SIZE)
            indexes = [_sample_slate(orders[cid], rng) for cid in cids]
            if step < 2:
                plan.append((stream, cids, indexes))
    return plan


def fambal_batch_plan(fit_rows, orders, seed):
    """Equal-family minibatch plan for FAMBAL and CLIP_FAMBAL: 40 updates, each 16 distinct families accumulated over 2x8 microbatches."""
    if seed not in SEEDS or not fit_rows:
        raise ValueError("registered seed and fit rows required")
    fam_to_cids = {}
    for r in fit_rows:
        fam_to_cids.setdefault(r["family_id"], []).append(r["context_sha256"])
    for f in fam_to_cids:
        fam_to_cids[f] = sorted(fam_to_cids[f])
    fams = sorted(fam_to_cids)
    if len(fams) < EFFECTIVE_BATCH_SIZE:
        raise ValueError("at least 16 fit families required for equal-family sampling")
    sampler = seed * 1_000_003 + 20_011
    queue = []
    epoch = 0
    plan = []
    for rnd in range(1, 21):
        rng = np.random.RandomState((sampler + rnd * 104729) % 2**32)
        for step in range(2):
            stream = int.from_bytes(
                hashlib.sha256(
                    f"flywheel-fambal/v1:{sampler}:{rnd}:{step}".encode()
                ).digest()[:8],
                "little",
            )
            while len(queue) < EFFECTIVE_BATCH_SIZE:
                epoch += 1
                perm = (
                    np.random.default_rng(sampler + epoch * 104729)
                    .permutation(fams)
                    .tolist()
                )
                queue.extend(perm)
            step_fams = []
            rem = []
            for f in queue:
                if len(step_fams) < EFFECTIVE_BATCH_SIZE and f not in step_fams:
                    step_fams.append(f)
                else:
                    rem.append(f)
            queue = rem
            ctx_rng = random.Random(stream)
            cids = [ctx_rng.choice(fam_to_cids[f]) for f in step_fams]
            indexes = [_sample_slate(orders[cid], rng) for cid in cids]
            microbatches = [
                (
                    cids[i : i + MICROBATCH_SIZE],
                    indexes[i : i + MICROBATCH_SIZE],
                )
                for i in range(0, EFFECTIVE_BATCH_SIZE, MICROBATCH_SIZE)
            ]
            plan.append((stream, cids, indexes, step_fams, microbatches))
    return plan


def fold_median_scale(fit_ids, base_losses):
    """Detached fit-fold median r882 per-context loss scale (strictly fit contexts only, floor 1e-6)."""
    if not fit_ids:
        raise ValueError("nonempty fit_ids required")
    vals = np.asarray([base_losses[c] for c in sorted(fit_ids)], dtype=np.float64)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError("finite nonnegative fit losses required")
    return float(max(np.median(vals), 1e-6))


def drift_lr(target_drifts, ctrl_drifts, base_lr=BASE_LR):
    a, c = float(np.median(target_drifts)), float(np.median(ctrl_drifts))
    if not (np.isfinite(a) and np.isfinite(c)) or a <= 0 or c <= 0:
        raise ValueError("positive finite drifts required")
    return base_lr * a / c


def drift_match(target_drifts, matched_drifts, bounds=DRIFT_MATCH_BOUNDS):
    t, m = float(np.median(target_drifts)), float(np.median(matched_drifts))
    if not (np.isfinite(t) and np.isfinite(m)) or t <= 0 or m <= 0:
        raise ValueError("positive finite drifts required")
    return dict(
        target_median=t,
        matched_median=m,
        ratio=m / t,
        bounds=list(bounds),
        valid=bool(bounds[0] * t <= m <= bounds[1] * t),
    )


def family_matrix(values, families):
    order = sorted(set(families))
    arr = np.asarray(values, dtype=np.float64)
    fam_arr = np.asarray(families)
    matrix = np.asarray([arr[fam_arr == family].mean(axis=0) for family in order])
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite complete panel")
    return order, matrix


def bt_ratings(wins, anchor, prior=0.01):
    """Canonical 400-iteration MM Bradley-Terry ratings anchored at r882 = 1500."""
    wins = np.asarray(wins, dtype=np.float64)
    n = wins.shape[-1]
    games = wins + np.swapaxes(wins, -1, -2)
    offdiag = 1.0 - np.eye(n)
    totals = wins.sum(axis=-1) + prior * (n - 1)
    strengths = np.ones(wins.shape[:-1])
    for _ in range(400):
        denom = (
            (games + 2.0 * prior * offdiag)
            / np.maximum(strengths[..., :, None] + strengths[..., None, :], 1e-12)
        ).sum(axis=-1)
        strengths = np.clip(totals / denom, 1e-6, 1e6)
        strengths /= np.maximum(strengths[..., anchor, None], 1e-12)
    return 1500.0 + 400.0 * np.log10(np.maximum(strengths, 1e-9))


def family_wins(values, families, tie_margin=1e-4):
    diff = values[:, :, None] - values[:, None, :]
    outcomes = np.where(diff > tie_margin, 1.0, np.where(diff < -tie_margin, 0.0, 0.5))
    outcomes *= 1.0 - np.eye(values.shape[1])
    return family_matrix(outcomes, families)[1]


def choose(rows, panel, seed):
    """H148 strict selector on selection fold B only: earliest best step; gain > 0.001 + 1e-12 and paired LB > 0."""
    values = np.asarray(panel, dtype=float)
    if values.shape != (len(rows), len(STEPS)):
        raise ValueError("complete five-snapshot selector required")
    _, fm = family_matrix(values, [r["family_id"] for r in rows])
    means = fm.mean(axis=0)
    best = int(np.argmax(means))
    delta = fm[:, best] - fm[:, 0]
    ix = np.random.default_rng(seed).integers(0, len(fm), (DRAWS, len(fm)))
    ci = np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist()
    gain = float(delta.mean())
    accepted = gain > MARGIN + TIE and ci[0] > 0
    return dict(
        best_step=STEPS[best],
        selected_step=STEPS[best] if accepted else 0,
        gain=gain,
        ci95=ci,
        accepted=bool(accepted),
        family_means=means.tolist(),
    )


def selection_seed(outer, seed_index, arm, arms=ARMS):
    return BOOT_SEED + 100 * outer + 10 * seed_index + list(arms).index(arm) + 1


def summarize(rows, panels, invalid_panels, arms=ARMS):
    """panels: [3 seeds, contexts, 1 + len(arms)] with column 0 = r882. Equal-family, shared family bootstrap."""
    values = np.asarray(panels, dtype=float)
    invalids = np.asarray(invalid_panels, dtype=bool)
    cols = ("r882",) + tuple(arms)
    if (
        values.shape != (3, len(rows), len(cols))
        or invalids.shape != values.shape
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete finite three-seed panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([family_matrix(v, families)[1] for v in values])  # [3, F, C]
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    means = fm.mean(axis=1)  # [3, C]
    boot = fm[:, ix].mean(axis=2)  # [3, DRAWS, C]
    contrasts = {}
    pairs = [(f"{a}_minus_r882", a, "r882") for a in arms]
    pairs += [(f"{a}_minus_CTRL", a, "CTRL") for a in arms if a != "CTRL"]
    if "CLIP_FAMBAL" in arms and "CLIP" in arms:
        pairs.append(("CLIP_FAMBAL_minus_CLIP", "CLIP_FAMBAL", "CLIP"))
    if "CLIP_FAMBAL" in arms and "FAMBAL" in arms:
        pairs.append(("CLIP_FAMBAL_minus_FAMBAL", "CLIP_FAMBAL", "FAMBAL"))
    if "CTRL_DRIFT" in arms:
        for a in TREATMENT_ARMS:
            if a in arms:
                pairs.append((f"{a}_minus_CTRL_DRIFT", a, "CTRL_DRIFT"))
    for name, left, right in pairs:
        li, ri = cols.index(left), cols.index(right)
        per_seed = means[:, li] - means[:, ri]
        draws = np.median(boot[:, :, li] - boot[:, :, ri], axis=0)
        mid = sorted(range(3), key=lambda i: (per_seed[i], SEEDS[i]))[1]
        fam_delta_mid = fm[mid, :, li] - fm[mid, :, ri]
        pos = np.maximum(fam_delta_mid, 0.0)
        top3_share = (
            float(np.sort(pos)[-3:].sum() / pos.sum()) if pos.sum() > 0 else 1.0
        )
        contrasts[name] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            p_one_sided=float((1 + (draws <= 0).sum()) / (DRAWS + 1)),
            positive_seeds=int((per_seed > 0).sum()),
            median_seed=SEEDS[mid],
            top3_positive_share=top3_share,
        )
    fw = np.stack([family_wins(v, families) for v in values])
    elo = bt_ratings(fw.sum(axis=1), 0)
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, n, (1000, n))
    elo_boot = np.median(bt_ratings(fw[:, ei].sum(axis=2), 0), axis=0)
    inv_counts = {
        col: [int(invalids[si, :, ci].sum()) for si in range(3)]
        for ci, col in enumerate(cols)
    }
    return dict(
        contexts=len(rows),
        families=n,
        columns=list(cols),
        contrasts=contrasts,
        family_utility_by_seed={
            col: means[:, ci].tolist() for ci, col in enumerate(cols)
        },
        context_utility_by_seed={
            col: values[:, :, ci].mean(axis=1).tolist() for ci, col in enumerate(cols)
        },
        invalid_counts_by_seed=inv_counts,
        invalid_counts_median={
            col: float(np.median(v)) for col, v in inv_counts.items()
        },
        elo_by_seed={col: elo[:, ci].tolist() for ci, col in enumerate(cols)},
        median_elo=dict(zip(cols, np.median(elo, axis=0).tolist())),
        elo_ci95=dict(
            zip(cols, np.quantile(elo_boot, [0.025, 0.975], axis=0).T.tolist())
        ),
        identical_seed_panels=bool(
            np.array_equal(values[0], values[1])
            and np.array_equal(values[1], values[2])
        ),
    )


def decide(primary, nonzero_picks, drift_match_receipt=None, winner_arm=None):
    """Holm step-down (m=3) across CLIP, FAMBAL, CLIP_FAMBAL vs r882; post-hoc CTRL_DRIFT required before ADVANCE."""
    p = {
        a: primary["contrasts"][f"{a}_minus_r882"]["p_one_sided"]
        for a in TREATMENT_ARMS
    }
    out = {}
    adjusted = 0.0
    for rank, arm in enumerate(sorted(TREATMENT_ARMS, key=lambda a: (p[a], a))):
        adjusted = max(adjusted, min(1.0, p[arm] * (HOLM_M - rank)))
        c = primary["contrasts"][f"{arm}_minus_r882"]
        efficacy_gates = dict(
            margin=c["median_gain"] > MARGIN + TIE,
            ci95_lower_gt_0=c["ci95"][0] > 0.0,
            all_three_seeds_positive=c["positive_seeds"] == 3,
            two_of_three_seeds_positive=c["positive_seeds"] >= 2,
            top3_concentration_le_080=c["top3_positive_share"] <= 0.80,
            holm=adjusted < 0.05,
            nonzero_selection=nonzero_picks[arm] > 0,
        )
        # Strict primary efficacy requires median_gain > 0.001 + 1e-12, CI LCB > 0, all 3 seeds > 0, Holm < 0.05, concentration <= 0.80, nonzero_selection > 0.
        efficacy_pass = bool(
            efficacy_gates["margin"]
            and efficacy_gates["ci95_lower_gt_0"]
            and efficacy_gates["all_three_seeds_positive"]
            and efficacy_gates["top3_concentration_le_080"]
            and efficacy_gates["holm"]
            and efficacy_gates["nonzero_selection"]
        )
        drift_gate = False
        if efficacy_pass and drift_match_receipt is not None and winner_arm == arm:
            dm_contrast = primary["contrasts"].get(f"{arm}_minus_CTRL_DRIFT")
            drift_gate = bool(
                drift_match_receipt.get("valid", False)
                and dm_contrast is not None
                and dm_contrast["median_gain"] > 0.0
                and dm_contrast["ci95"][0] > 0.0
            )
        out[arm] = dict(
            holm_adjusted_p=adjusted,
            efficacy_gates=efficacy_gates,
            efficacy_pass=efficacy_pass,
            drift_matched_ctrl_pass=drift_gate if efficacy_pass else None,
            advance=bool(efficacy_pass and drift_gate),
        )
    if all(nonzero_picks.get(a, 0) == 0 for a in ARMS):
        verdict = "RULE_NEUTRAL"
    elif any(out[a]["advance"] for a in TREATMENT_ARMS):
        verdict = "ADVANCE"
    else:
        verdict = "NO_ADVANCE"
    return dict(verdict=verdict, treatment_arms=out)
