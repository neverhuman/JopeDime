"""Prospective H147 statistics: matched-support reveal alignment, 5-fold out-of-fold 205-family bootstrap, intersection Holm m=2 gates, and Bradley-Terry MM Elo."""

import hashlib
import random
import numpy as np

SEEDS = (147101, 147102, 147103)
FOLD_SEED = 147140
BOOT_SEED = 147141
DRAWS = 10000
HOLM_M = 2

ARMS = (
    "SUPPORT_CTRL",
    "STORAGE_REVEAL",
    "STORAGE_MASK",
)
CANDIDATE_ARMS = (
    "STORAGE_REVEAL",
    "STORAGE_MASK",
)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SLATE_SIZE = 192
TOP_K_FITNESS = 48
REVEAL_K = 4


def folds(rows):
    """Split 205 DL-Train families into 5 disjoint 41-family assessment groups using PCG64(147140)."""
    if not rows or len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("unique contexts required")
    if any(
        r["dl_split"] != "DL-Train"
        or r.get("has_meta_test") is True
        or r.get("source_split") not in ("train", "inner")
        for r in rows
    ):
        raise ValueError("DL-Train inner rows only")
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != 205:
        raise ValueError(f"expected 205 DL-Train families, got {len(fams)}")
    return [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(FOLD_SEED).permutation(fams), 5)
    ]


def stable_fitness_order(fitness):
    """Return candidate indices sorted by ascending fitness with ties broken by original storage index."""
    arr = np.asarray(fitness, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < SLATE_SIZE or not np.isfinite(arr).all():
        raise ValueError(
            f"pool fitness must be 1D finite with >= {SLATE_SIZE} entries, got shape {arr.shape}"
        )
    return np.argsort(arr, kind="stable").astype(np.int64)


def build_common_support_slate(order, rng):
    """Construct common 192-candidate support S = A U top48 U uniform_draw and return arm permutations."""
    order_arr = np.asarray(order, dtype=np.int64)
    if order_arr.ndim != 1 or len(order_arr) < SLATE_SIZE:
        raise ValueError(f"pool has fewer than {SLATE_SIZE} entries ({len(order_arr)})")
    anchors_a = np.array([0, 1, 2, 3], dtype=np.int64)
    top48 = order_arr[:TOP_K_FITNESS]
    union_set = set(top48.tolist()) | {0, 1, 2, 3}
    comp = np.asarray(
        [int(i) for i in order_arr if int(i) not in union_set], dtype=np.int64
    )
    n_draw = SLATE_SIZE - len(union_set)
    if len(comp) < n_draw:
        raise ValueError(
            "insufficient complement candidates for without-replacement draw"
        )
    drawn = rng.choice(comp, n_draw, replace=False).astype(np.int64)
    support_set = union_set | set(drawn.tolist())
    if len(support_set) != SLATE_SIZE:
        raise RuntimeError("common support S size != 192")

    # Remaining S ordered in stable fitness order (ties broken by original storage index, preserved in order_arr)
    s_fitness_sorted = np.asarray(
        [int(i) for i in order_arr if int(i) in support_set], dtype=np.int64
    )
    anchors_b = s_fitness_sorted[:REVEAL_K]
    ix_support_ctrl = s_fitness_sorted
    rest_after_a = np.asarray(
        [int(i) for i in s_fitness_sorted if int(i) not in (0, 1, 2, 3)],
        dtype=np.int64,
    )
    ix_storage = np.concatenate([anchors_a, rest_after_a])

    if (
        len(ix_support_ctrl) != SLATE_SIZE
        or len(ix_storage) != SLATE_SIZE
        or set(ix_support_ctrl.tolist()) != set(ix_storage.tolist())
    ):
        raise RuntimeError("common support permutation invariant violated")

    return dict(
        ix_support_ctrl=ix_support_ctrl,
        ix_storage=ix_storage,
        anchors_a=anchors_a,
        anchors_b=anchors_b,
        a_equals_b_set=(set(anchors_a.tolist()) == set(anchors_b.tolist())),
        a_equals_b_ordered=bool(np.array_equal(anchors_a, anchors_b)),
        extra_anchors_added=int(len(union_set) - TOP_K_FITNESS),
    )


def batch_plan(fit_ids, orders, seed, fold):
    """H144/H147 20-round paired stream (8 draws per round, retain first 2) with effective seed = seed + 1_000_000 * fold."""
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    if fold not in range(5):
        raise ValueError(f"fold must be in 0..4, got {fold}")
    if not fit_ids:
        raise ValueError("fit_ids must be non-empty")
    ids = tuple(sorted(fit_ids))
    for cid in ids:
        if cid not in orders or len(orders[cid]) < SLATE_SIZE:
            raise ValueError(f"pool {cid} has fewer than {SLATE_SIZE} entries")

    eff_seed = int(seed) + 1_000_000 * int(fold)
    sampler = eff_seed * 1_000_003 + 10_007
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
            cids = random.Random(stream).choices(rotated, k=16)
            ix_ctrl_list = []
            ix_stor_list = []
            a_eq_b_set = 0
            a_eq_b_ord = 0
            support_changed = 0
            for cid in cids:
                slate = build_common_support_slate(orders[cid], rng)
                ix_ctrl_list.append(slate["ix_support_ctrl"])
                ix_stor_list.append(slate["ix_storage"])
                a_eq_b_set += int(slate["a_equals_b_set"])
                a_eq_b_ord += int(slate["a_equals_b_ordered"])
                support_changed += int(slate["extra_anchors_added"] > 0)
            if step < 2:
                plan.append(
                    dict(
                        round=rnd,
                        round_step=step,
                        stream=stream,
                        cids=cids,
                        ix_support_ctrl=ix_ctrl_list,
                        ix_storage=ix_stor_list,
                        a_equals_b_set_count=a_eq_b_set,
                        a_equals_b_ordered_count=a_eq_b_ord,
                        support_changed_vs_h144_count=support_changed,
                    )
                )
    return plan


def family_matrix(rows, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3, len(rows)) or not np.isfinite(values).all():
        raise ValueError("all three complete finite seed panels required")
    families = sorted({r["family_id"] for r in rows})
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    return families, np.stack(
        [
            values[:, [i for i, r in enumerate(rows) if r["family_id"] == f]].mean(
                axis=1
            )
            for f in families
        ],
        axis=1,
    )


def contrast(rows, values, reference):
    """Equal-family median gain over 3 seeds with 10,000-draw shared family-cluster bootstrap (seed 147141)."""
    fams, delta = family_matrix(
        rows,
        np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64),
    )
    gains = delta.mean(axis=1)
    ix = np.random.default_rng(BOOT_SEED).integers(0, len(fams), (DRAWS, len(fams)))
    boot = np.median(delta[:, ix].mean(axis=2), axis=0)
    # Seed defining the median gain (ties broken by ascending seed).
    mid = sorted(range(3), key=lambda i: (gains[i], SEEDS[i]))[1]
    positive = np.maximum(delta[mid], 0.0)
    share = (
        float(np.sort(positive)[-3:].sum() / positive.sum())
        if positive.sum() > 0.0
        else 1.0
    )
    fam_3seed_mean = delta.mean(axis=0)
    neg_fam_count = int((fam_3seed_mean < 0.0).sum())
    neg_fam_frac = float(neg_fam_count / len(fams))
    wtl_3seed = dict(
        wins=int((fam_3seed_mean > 0.0).sum()),
        ties=int((fam_3seed_mean == 0.0).sum()),
        losses=neg_fam_count,
    )
    wtl_by_seed = [
        dict(
            wins=int((delta[s] > 0.0).sum()),
            ties=int((delta[s] == 0.0).sum()),
            losses=int((delta[s] < 0.0).sum()),
        )
        for s in range(3)
    ]

    score = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))

    def paired_elo(x):
        wins = x.sum(axis=-1)
        n = x.shape[-1]
        return 1500.0 + 400.0 * np.log10((wins + 0.01) / (n - wins + 0.01))

    elo_draws = np.median(paired_elo(score[:, ix]), axis=0)
    return dict(
        families=len(fams),
        per_seed_gain=gains.tolist(),
        median_gain=float(np.median(gains)),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
        seeds_beating=int((gains > 0).sum()),
        median_seed=SEEDS[mid],
        top3_positive_share=share,
        neg_family_count_3seed=neg_fam_count,
        neg_family_fraction_3seed=neg_fam_frac,
        family_wtl_3seed=wtl_3seed,
        family_wtl_by_seed=wtl_by_seed,
        paired_elo=float(np.median(paired_elo(score))),
        paired_elo_ci95=np.quantile(elo_draws, [0.025, 0.975]).tolist(),
    )


def decide(vs_r882, vs_support_ctrl):
    """Intersection Holm m=2 decision over STORAGE_REVEAL and STORAGE_MASK vs both r882 and SUPPORT_CTRL."""
    if set(vs_r882) != set(ARMS) or set(vs_support_ctrl) != set(CANDIDATE_ARMS):
        raise ValueError("complete registered contrasts required for decide")

    p_inter = {
        a: float(max(vs_r882[a]["p_one_sided"], vs_support_ctrl[a]["p_one_sided"]))
        for a in CANDIDATE_ARMS
    }
    holm_adj = {}
    running = 0.0
    for rank, a in enumerate(sorted(CANDIDATE_ARMS, key=lambda x: (p_inter[x], x))):
        running = max(running, min(1.0, p_inter[a] * (HOLM_M - rank)))
        holm_adj[a] = float(running)

    candidates = {}
    for a in CANDIDATE_ARMS:
        r = vs_r882[a]
        c = vs_support_ctrl[a]
        gates = dict(
            margin_both=(r["median_gain"] > 0.001 and c["median_gain"] > 0.001),
            ci_both=(r["ci95"][0] > 0.0 and c["ci95"][0] > 0.0),
            holm_intersection=(holm_adj[a] < 0.05),
            two_seeds_both=(r["seeds_beating"] >= 2 and c["seeds_beating"] >= 2),
            harm_cap_r882=(r["neg_family_fraction_3seed"] <= 0.10),
            concentration_r882=(r["top3_positive_share"] <= 0.80),
        )
        candidates[a] = dict(
            p_vs_r882=float(r["p_one_sided"]),
            p_vs_support_ctrl=float(c["p_one_sided"]),
            p_intersection=p_inter[a],
            holm_adjusted_p=holm_adj[a],
            gates=gates,
            advance=bool(all(gates.values())),
        )

    advancing = [a for a in CANDIDATE_ARMS if candidates[a]["advance"]]
    best_arm = (
        max(
            advancing,
            key=lambda a: (vs_r882[a]["median_gain"], -CANDIDATE_ARMS.index(a)),
        )
        if advancing
        else None
    )
    verdict = "ADVANCE" if advancing else "NO_ADVANCE"
    return dict(
        verdict=verdict,
        best_advancing_arm=best_arm,
        candidates=candidates,
        support_ctrl_vs_r882=dict(
            median_gain=vs_r882["SUPPORT_CTRL"]["median_gain"],
            ci95=vs_r882["SUPPORT_CTRL"]["ci95"],
            p_one_sided=vs_r882["SUPPORT_CTRL"]["p_one_sided"],
            seeds_beating=vs_r882["SUPPORT_CTRL"]["seeds_beating"],
            neg_family_fraction_3seed=vs_r882["SUPPORT_CTRL"][
                "neg_family_fraction_3seed"
            ],
            top3_positive_share=vs_r882["SUPPORT_CTRL"]["top3_positive_share"],
            advance=False,
        ),
    )


def bradley_terry_elo_summary(
    rows,
    r882_values,
    arm_panels,
    elo_mod,
    tie_margin=1e-4,
    prior=0.01,
    n_iters=400,
):
    """Compute pinned multi-contender family-grouped Bradley-Terry Elo (r882=1500) per seed, median, and 10,000-draw family bootstrap CI."""
    r882_arr = np.asarray(r882_values, dtype=np.float64)
    if r882_arr.ndim == 1:
        r882_mat = np.tile(r882_arr, (3, 1))
    else:
        r882_mat = r882_arr
    if r882_mat.shape != (3, len(rows)) or not np.isfinite(r882_mat).all():
        raise ValueError("invalid r882 panel shape for Bradley-Terry Elo")

    contenders = ["r882", *ARMS]
    c_sorted = sorted(contenders)
    c_to_idx = {c: i for i, c in enumerate(c_sorted)}
    anchor_idx = c_to_idx["r882"]
    m_cnt = len(c_sorted)

    panels_by_c = {"r882": r882_mat}
    for arm in ARMS:
        arr = np.asarray(arm_panels[arm], dtype=np.float64)
        if arr.shape != (3, len(rows)) or not np.isfinite(arr).all():
            raise ValueError(f"invalid panel shape for arm {arm}")
        panels_by_c[arm] = arr

    # 1. Exact per-seed calls to pinned elo_tracker.compute_family_grouped_bradley_terry_elo
    per_seed_bt = []
    for s_idx in range(3):
        recs = [
            elo_mod.DatasetEvalRecord(
                dataset_id=r["context_sha256"],
                family_id=r["family_id"],
                task_type="regression",
                scores_by_contender={
                    c: float(panels_by_c[c][s_idx, i]) for c in c_sorted
                },
            )
            for i, r in enumerate(rows)
        ]
        res = elo_mod.compute_family_grouped_bradley_terry_elo(
            recs,
            anchor_contender="r882",
            anchor_elo=1500.0,
            tie_margin=tie_margin,
            n_iters=n_iters,
            prior=prior,
            allow_incomplete_panel=False,
        )
        per_seed_bt.append(res)

    # 2. Precompute per-family W_fam[s, f, a, b] (shape [3, F, M, M]) for exact vectorized 10,000-draw family bootstrap
    fams = sorted({r["family_id"] for r in rows})
    f_cnt = len(fams)
    fam_rows = [[i for i, r in enumerate(rows) if r["family_id"] == f] for f in fams]
    w_fam = np.zeros((3, f_cnt, m_cnt, m_cnt), dtype=np.float64)
    for s_idx in range(3):
        for f_idx, r_indices in enumerate(fam_rows):
            inv_denom = 1.0 / float(len(r_indices))
            for row_i in r_indices:
                for i_a in range(m_cnt):
                    ca = c_sorted[i_a]
                    sa = float(panels_by_c[ca][s_idx, row_i])
                    for i_b in range(i_a + 1, m_cnt):
                        cb = c_sorted[i_b]
                        sb = float(panels_by_c[cb][s_idx, row_i])
                        diff = sa - sb
                        if diff > tie_margin:
                            s_ab = 1.0
                        elif diff < -tie_margin:
                            s_ab = 0.0
                        else:
                            s_ab = 0.5
                        w_fam[s_idx, f_idx, i_a, i_b] += inv_denom * s_ab
                        w_fam[s_idx, f_idx, i_b, i_a] += inv_denom * (1.0 - s_ab)

    def solve_mm_batch(w_batch):
        # w_batch shape: [B, M, M]; n_games is f_cnt for every off-diagonal pair
        bsz = w_batch.shape[0]
        gamma = np.ones((bsz, m_cnt), dtype=np.float64)
        n_ab = float(f_cnt) + 2.0 * prior
        w_sum = np.sum(w_batch, axis=2) + prior * (m_cnt - 1)  # [B, M]
        for _ in range(n_iters):
            new_gamma = np.ones((bsz, m_cnt), dtype=np.float64)
            for a in range(m_cnt):
                denom = np.zeros(bsz, dtype=np.float64)
                for b in range(m_cnt):
                    if a == b:
                        continue
                    denom += n_ab / np.maximum(gamma[:, a] + gamma[:, b], 1e-12)
                new_gamma[:, a] = np.clip(w_sum[:, a] / denom, 1e-6, 1e6)
            new_gamma /= np.maximum(new_gamma[:, anchor_idx : anchor_idx + 1], 1e-12)
            gamma = new_gamma
        return 1500.0 + 400.0 * np.log10(np.maximum(gamma, 1e-9))

    # Verify vectorized MM solver matches elo_tracker.py on all 3 unbootstrapped seed panels
    unboot_ratings = solve_mm_batch(w_fam.sum(axis=1))  # [3, M]
    for s_idx in range(3):
        for c in c_sorted:
            if (
                round(float(unboot_ratings[s_idx, c_to_idx[c]]), 2)
                != per_seed_bt[s_idx]["ratings"][c]
            ):
                raise RuntimeError(
                    "vectorized Bradley-Terry MM solver mismatch with elo_tracker.py"
                )

    ix = np.random.default_rng(BOOT_SEED).integers(0, f_cnt, (DRAWS, f_cnt))
    # Compute bootstrap ratings per seed then median across the 3 seeds for each draw
    seed_boot_ratings = np.zeros((3, DRAWS, m_cnt), dtype=np.float64)
    for s_idx in range(3):
        # Sum over sampled families: w_fam[s_idx, ix] has shape [DRAWS, F, M, M] -> sum(axis=1) is [DRAWS, M, M]
        w_draws = w_fam[s_idx, ix].sum(axis=1)
        seed_boot_ratings[s_idx] = solve_mm_batch(w_draws)
    median_boot_ratings = np.median(seed_boot_ratings, axis=0)  # [DRAWS, M]

    summary = {}
    for c in contenders:
        idx_c = c_to_idx[c]
        seed_vals = [float(unboot_ratings[s, idx_c]) for s in range(3)]
        med_val = float(np.median(seed_vals))
        ci = np.quantile(median_boot_ratings[:, idx_c], [0.025, 0.975]).tolist()
        summary[c] = dict(
            elo_by_seed=[round(x, 2) for x in seed_vals],
            median_elo=round(med_val, 2),
            elo_ci95=[round(float(ci[0]), 2), round(float(ci[1]), 2)],
            pairwise_wtl_vs_r882_by_seed=(
                [per_seed_bt[s]["pairwise_win_tie_loss_vs_anchor"][c] for s in range(3)]
                if c != "r882"
                else None
            ),
        )
    return summary
