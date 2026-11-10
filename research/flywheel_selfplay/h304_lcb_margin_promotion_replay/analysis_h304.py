"""H304 (h304_lcb_margin_promotion_replay): pure-NumPy promotion-rule replay and evaluation on H289 history.npz."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

PREREG_COMMIT = "1c0622736fa77d265777132378674a07d5efcf89"
PREPARED_CLOSURE_VERSION_ID = "dj7fcZrNhjkNwY2s5DkfOah1nfAUObnL"
PREPARED_CLOSURE_SHA256 = "66bc3154f4f2ec49bc9237163712826ec0bda82f32e3ba34c4881ba73b4bd071"
HISTORY_NPZ_SHA256 = "e6214cd5f3f3322a6ef22bfa7b3f7a797e106869c9195cdfe18d217b6ac7f446"
ADMISSIONS_JSONL_SHA256 = "7c01de3b03eeb993d67c84266649a732365afdeaa86c593295b73a9d05b0b868"
EVAL_IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"

SPLIT_SEEDS: tuple[int, int, int] = (30401, 30402, 30403)
BOOT_SEED = 304150
BOOT_DRAWS = 10000
SELECT_LCB_DRAWS = 1000
SELECT_LCB_SEED_BASE = 304200
REPEATED_SPLITS_N = 200
TIE_EPS = 1e-4
MARGIN_001 = 0.001
MARGIN_0035 = 0.0035
MEDIAN_K = 5

RULE_NAMES: tuple[str, ...] = (
    "R0_BASELINE",
    "R1_MARGIN_001",
    "R1_MARGIN_0035",
    "R2_LCB",
    "R3_MEDIAN_OF_K",
)
NON_R0_RULES: tuple[str, ...] = (
    "R1_MARGIN_001",
    "R1_MARGIN_0035",
    "R2_LCB",
    "R3_MEDIAN_OF_K",
)


def split_families_by_hash(unique_families: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Split the 19 dev91 families into SELECT (parity 0) and EVAL (parity 1) halves by
    SHA256('H304-split-v1\\x00' + str(seed) + '\\x00' + family) last-byte parity."""
    select_fams: list[str] = []
    eval_fams: list[str] = []
    for fam in sorted(set(unique_families)):
        digest = hashlib.sha256(
            b"H304-split-v1\x00" + str(int(seed)).encode("utf-8") + b"\x00" + fam.encode("utf-8")
        ).digest()
        if (digest[-1] & 1) == 0:
            select_fams.append(fam)
        else:
            eval_fams.append(fam)
    if not select_fams or not eval_fams:
        raise ValueError(f"Degenerate family split for seed {seed}")
    return select_fams, eval_fams


def generate_repeated_balanced_splits(
    unique_families: list[str],
    n_splits: int = REPEATED_SPLITS_N,
    rng_seed: int = BOOT_SEED,
) -> list[tuple[list[str], list[str]]]:
    """Generate n_splits unique random 9-SELECT / 10-EVAL splits of the 19 families using rng_seed."""
    fams = sorted(set(unique_families))
    n_fams = len(fams)
    n_sel = n_fams // 2  # 9 of 19
    rng = np.random.default_rng(rng_seed)
    seen: set[tuple[str, ...]] = set()
    splits: list[tuple[list[str], list[str]]] = []
    while len(splits) < n_splits:
        perm = rng.permutation(n_fams)
        sel = tuple(sorted(fams[i] for i in perm[:n_sel]))
        if sel in seen:
            continue
        seen.add(sel)
        sel_set = set(sel)
        ev = [f for f in fams if f not in sel_set]
        splits.append((list(sel), ev))
    return splits


def build_family_index_map(context_families: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (sorted_unique_families, ctx_to_fam_idx, fam_weights_matrix (F, C))."""
    uniq_fams = sorted(set(context_families))
    fam_to_idx = {f: i for i, f in enumerate(uniq_fams)}
    ctx_to_fam = np.array([fam_to_idx[f] for f in context_families], dtype=np.int32)
    n_fams = len(uniq_fams)
    n_ctx = len(context_families)
    fam_w = np.zeros((n_fams, n_ctx), dtype=np.float64)
    for fi in range(n_fams):
        mask = ctx_to_fam == fi
        cnt = int(np.sum(mask))
        fam_w[fi, mask] = 1.0 / float(cnt)
    return uniq_fams, ctx_to_fam, fam_w


def make_select_bootstrap_weights(n_select_fams: int, n_draws: int, rng_seed: int) -> np.ndarray:
    """Return (n_draws, n_select_fams) bootstrap weight matrix summing to 1 along axis 1."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n_select_fams, size=(n_draws, n_select_fams))
    counts = np.zeros((n_draws, n_select_fams), dtype=np.float64)
    rows = np.arange(n_draws)[:, None]
    np.add.at(counts, (rows, idx), 1.0)
    return counts / float(n_select_fams)


def compute_pairwise_elo_from_wins_losses(wins: np.ndarray, losses: np.ndarray, anchor: float = 1500.0) -> np.ndarray:
    """Compute 2-player Bradley-Terry Elo vs anchor (1500) with 0.5 continuity correction on decisive pairs."""
    w = np.asarray(wins, dtype=np.float64)
    l = np.asarray(losses, dtype=np.float64)
    decisive = w + l
    # Smooth by +0.5/+0.5 only when decisive > 0 and either w==0 or l==0; if decisive==0 -> anchor
    p = np.where(decisive > 0.0, (w + 0.5) / (decisive + 1.0), 0.5)
    # If w == l, exact 1500.0
    elo = anchor + 400.0 * np.log10(p / (1.0 - p))
    elo = np.where(w == l, anchor, elo)
    return elo


def replay_single_rule(
    rule: str,
    rounds: np.ndarray,
    utils: np.ndarray,
    fam_w: np.ndarray,
    select_fam_indices: np.ndarray,
    eval_fam_indices: np.ndarray,
    select_ctx_mask: np.ndarray,
    eval_ctx_mask: np.ndarray,
    select_boot_w: np.ndarray,
    select_mode: str = "family_equal",
) -> dict[str, Any]:
    """Replay a promotion rule across the round sequence using ONLY select-half utilities,
    then score the resulting incumbent trajectory on eval-half families."""
    n_rounds, n_ctx = utils.shape
    assert len(rounds) == n_rounds

    sel_fw = fam_w[select_fam_indices]  # (F_sel, C)
    eval_fw = fam_w[eval_fam_indices]  # (F_eval, C)

    # Precompute all candidate select-family matrices and select scalars once in C/BLAS
    all_sel_fam = utils @ sel_fw.T  # (n_rounds, F_sel)
    if select_mode == "family_equal":
        all_sel_scalars = np.mean(all_sel_fam, axis=1)  # (n_rounds,)
    elif select_mode == "context_mean":
        all_sel_scalars = np.mean(utils[:, select_ctx_mask], axis=1)  # (n_rounds,)
    else:
        raise ValueError(f"Unknown select_mode: {select_mode}")

    def select_scalar(u_vec: np.ndarray) -> float:
        if select_mode == "family_equal":
            return float(np.mean(sel_fw @ u_vec))
        elif select_mode == "context_mean":
            return float(np.mean(u_vec[select_ctx_mask]))
        raise ValueError(f"Unknown select_mode: {select_mode}")

    def eval_family_vec(u_vec: np.ndarray) -> np.ndarray:
        return eval_fw @ u_vec  # (F_eval,)

    # Initialize at t=0
    accepted_indices: list[int] = [0]
    accepted_rounds: list[int] = [int(rounds[0])]
    accepted_vectors: list[np.ndarray] = [utils[0].copy()]

    inc_vec = utils[0].copy()
    inc_sel_fam = all_sel_fam[0]
    inc_sel_scalar = float(all_sel_scalars[0])
    inc_eval_fam = eval_family_vec(inc_vec)
    inc_eval_scalar = float(np.mean(inc_eval_fam))

    # Track time-averaged incumbent vector via run-length weighting
    sum_inc_vec = np.zeros(n_ctx, dtype=np.float64)
    last_change_t = 0
    promotions: list[dict[str, Any]] = []

    for t in range(1, n_rounds):
        cand_sel_scalar = float(all_sel_scalars[t])
        d_sel = cand_sel_scalar - inc_sel_scalar

        accept = False
        lcb95_val: float | None = None

        # Global tie guard: |d_sel| <= TIE_EPS never accepted under any rule
        if d_sel > TIE_EPS:
            if rule == "R0_BASELINE":
                accept = True
            elif rule == "R1_MARGIN_001":
                accept = bool(d_sel > MARGIN_001)
            elif rule == "R1_MARGIN_0035":
                accept = bool(d_sel > MARGIN_0035)
            elif rule == "R2_LCB":
                d_fam_sel = all_sel_fam[t] - inc_sel_fam  # (F_sel,)
                boot_gains = select_boot_w @ d_fam_sel  # (n_draws,)
                lcb95_val = float(np.quantile(boot_gains, 0.05))
                accept = bool(lcb95_val > 0.0)
            elif rule == "R3_MEDIAN_OF_K":
                accept = True
            else:
                raise ValueError(f"Unknown rule: {rule}")

        if accept:
            sum_inc_vec += float(t - last_change_t) * inc_vec
            last_change_t = t

            prev_sel_scalar = inc_sel_scalar
            prev_eval_scalar = inc_eval_scalar
            cand_vec = utils[t]

            accepted_indices.append(t)
            accepted_rounds.append(int(rounds[t]))
            accepted_vectors.append(cand_vec.copy())

            if rule == "R3_MEDIAN_OF_K":
                k_window = accepted_vectors[-MEDIAN_K:]
                new_inc_vec = np.median(np.stack(k_window, axis=0), axis=0)
                new_sel_fam = sel_fw @ new_inc_vec
                new_sel_scalar = select_scalar(new_inc_vec)
            else:
                new_inc_vec = cand_vec.copy()
                new_sel_fam = all_sel_fam[t]
                new_sel_scalar = cand_sel_scalar

            new_eval_fam = eval_family_vec(new_inc_vec)
            new_eval_scalar = float(np.mean(new_eval_fam))

            step_sel_gain = new_sel_scalar - prev_sel_scalar
            step_eval_gain = new_eval_scalar - prev_eval_scalar

            promotions.append(
                {
                    "step_index": int(t),
                    "round": int(rounds[t]),
                    "cand_minus_inc_select_gain": float(d_sel),
                    "cand_select_lcb95": lcb95_val,
                    "incumbent_step_select_gain": float(step_sel_gain),
                    "incumbent_step_eval_gain": float(step_eval_gain),
                    "eval_improved": bool(step_eval_gain > TIE_EPS),
                    "eval_harmed": bool(step_eval_gain < -TIE_EPS),
                    "shrinkage_select_minus_eval": float(step_sel_gain - step_eval_gain),
                }
            )

            inc_vec = new_inc_vec
            inc_sel_fam = new_sel_fam
            inc_sel_scalar = new_sel_scalar
            inc_eval_fam = new_eval_fam
            inc_eval_scalar = new_eval_scalar

    sum_inc_vec += float(n_rounds - last_change_t) * inc_vec
    time_avg_inc_vec = sum_inc_vec / float(n_rounds)
    init_vec = utils[0]

    init_sel_scalar = select_scalar(init_vec)
    init_eval_fam = eval_family_vec(init_vec)
    init_eval_scalar = float(np.mean(init_eval_fam))

    final_eval_fam = eval_family_vec(inc_vec)
    final_eval_scalar = float(np.mean(final_eval_fam))
    time_avg_eval_fam = eval_family_vec(time_avg_inc_vec)
    time_avg_eval_scalar = float(np.mean(time_avg_eval_fam))

    n_prom = len(promotions)
    n_eval_imp = sum(1 for p in promotions if p["eval_improved"])
    n_eval_harm = sum(1 for p in promotions if p["eval_harmed"])
    precision = float(n_eval_imp / n_prom) if n_prom > 0 else 0.0
    mean_step_shrinkage = (
        float(np.mean([p["shrinkage_select_minus_eval"] for p in promotions]))
        if n_prom > 0
        else 0.0
    )
    net_select_gain_vs_init = float(inc_sel_scalar - init_sel_scalar)
    net_eval_gain_vs_init = float(final_eval_scalar - init_eval_scalar)
    net_shrinkage_vs_init = float(net_select_gain_vs_init - net_eval_gain_vs_init)

    return {
        "rule": rule,
        "select_mode": select_mode,
        "n_promotions": n_prom,
        "n_eval_improved_promotions": n_eval_imp,
        "n_eval_harmed_promotions": n_eval_harm,
        "promotion_precision": precision,
        "accepted_rounds": accepted_rounds,
        "final_incumbent_round": accepted_rounds[-1],
        "init_select_utility": init_sel_scalar,
        "final_select_utility": inc_sel_scalar,
        "net_select_gain_vs_init": net_select_gain_vs_init,
        "init_eval_family_equal_utility": init_eval_scalar,
        "final_eval_family_equal_utility": final_eval_scalar,
        "time_avg_eval_family_equal_utility": time_avg_eval_scalar,
        "net_eval_gain_vs_init": net_eval_gain_vs_init,
        "net_shrinkage_vs_init": net_shrinkage_vs_init,
        "mean_step_shrinkage": mean_step_shrinkage,
        "promotions": promotions,
        "_final_inc_vec": inc_vec,
        "_time_avg_inc_vec": time_avg_inc_vec,
        "_final_eval_fam": final_eval_fam,
        "_time_avg_eval_fam": time_avg_eval_fam,
    }


def evaluate_three_splits(
    rounds: np.ndarray,
    utils: np.ndarray,
    context_families: list[str],
    r3175_vec: np.ndarray,
    split_seeds: tuple[int, int, int] = SPLIT_SEEDS,
    select_mode: str = "family_equal",
    boot_seed: int = BOOT_SEED,
    boot_draws: int = BOOT_DRAWS,
) -> dict[str, Any]:
    """Run all 5 rules across the 3 pre-declared split seeds, compute EVAL-half bootstrap CIs,
    Elo vs r3175, and evaluate the H304 decision rule."""
    uniq_fams, ctx_to_fam, fam_w = build_family_index_map(context_families)
    fam_to_idx = {f: i for i, f in enumerate(uniq_fams)}

    # Shared uniform [0, 1) bootstrap quantiles / multinomial generator per split seed
    rng_boot = np.random.default_rng(boot_seed)

    per_seed_results: dict[str, dict[str, Any]] = {}
    boot_final_diff_vs_r0: dict[str, list[np.ndarray]] = {r: [] for r in RULE_NAMES}
    boot_tavg_diff_vs_r0: dict[str, list[np.ndarray]] = {r: [] for r in RULE_NAMES}
    boot_final_elo_vs_r3175: dict[str, list[np.ndarray]] = {r: [] for r in RULE_NAMES}

    for s_idx, seed in enumerate(split_seeds):
        sel_fams, eval_fams = split_families_by_hash(uniq_fams, seed)
        sel_f_idx = np.array([fam_to_idx[f] for f in sel_fams], dtype=np.int32)
        eval_f_idx = np.array([fam_to_idx[f] for f in eval_fams], dtype=np.int32)
        sel_ctx_mask = np.isin(ctx_to_fam, sel_f_idx)
        eval_ctx_mask = np.isin(ctx_to_fam, eval_f_idx)

        sel_boot_w = make_select_bootstrap_weights(
            len(sel_fams), SELECT_LCB_DRAWS, SELECT_LCB_SEED_BASE + seed
        )

        # Draw whole-family bootstrap weights for EVAL families on this seed
        n_eval = len(eval_fams)
        eval_draw_idx = rng_boot.integers(0, n_eval, size=(boot_draws, n_eval))
        eval_boot_w = np.zeros((boot_draws, n_eval), dtype=np.float64)
        rows = np.arange(boot_draws)[:, None]
        np.add.at(eval_boot_w, (rows, eval_draw_idx), 1.0)
        eval_boot_w /= float(n_eval)

        # Precompute r3175 eval family vector and per-family win/loss counts on EVAL contexts
        eval_fw = fam_w[eval_f_idx]
        r3175_eval_fam = eval_fw @ r3175_vec

        rule_runs: dict[str, dict[str, Any]] = {}
        for rname in RULE_NAMES:
            rule_runs[rname] = replay_single_rule(
                rule=rname,
                rounds=rounds,
                utils=utils,
                fam_w=fam_w,
                select_fam_indices=sel_f_idx,
                eval_fam_indices=eval_f_idx,
                select_ctx_mask=sel_ctx_mask,
                eval_ctx_mask=eval_ctx_mask,
                select_boot_w=sel_boot_w,
                select_mode=select_mode,
            )

        r0_final_fam = rule_runs["R0_BASELINE"]["_final_eval_fam"]
        r0_tavg_fam = rule_runs["R0_BASELINE"]["_time_avg_eval_fam"]

        seed_rules_pub: dict[str, Any] = {}
        for rname in RULE_NAMES:
            rr = rule_runs[rname]
            f_fam = rr["_final_eval_fam"]
            t_fam = rr["_time_avg_eval_fam"]

            diff_final_fam = f_fam - r0_final_fam
            diff_tavg_fam = t_fam - r0_tavg_fam
            diff_vs_r3175_fam = f_fam - r3175_eval_fam

            b_final_diff = eval_boot_w @ diff_final_fam
            b_tavg_diff = eval_boot_w @ diff_tavg_fam
            b_vs_r3175 = eval_boot_w @ diff_vs_r3175_fam

            boot_final_diff_vs_r0[rname].append(b_final_diff)
            boot_tavg_diff_vs_r0[rname].append(b_tavg_diff)

            # Compute MM Bradley-Terry Elo on EVAL contexts vs r3175 (=1500)
            inc_vec = rr["_final_inc_vec"]
            ctx_d = inc_vec - r3175_vec
            fam_wins = np.zeros(n_eval, dtype=np.float64)
            fam_losses = np.zeros(n_eval, dtype=np.float64)
            for k_e, f_i in enumerate(eval_f_idx):
                m_f = ctx_to_fam == f_i
                fam_wins[k_e] = float(np.sum(ctx_d[m_f] > TIE_EPS))
                fam_losses[k_e] = float(np.sum(ctx_d[m_f] < -TIE_EPS))
            elo_point = float(
                compute_pairwise_elo_from_wins_losses(
                    np.array([np.sum(fam_wins)]), np.array([np.sum(fam_losses)])
                )[0]
            )
            b_wins = (eval_boot_w * float(n_eval)) @ fam_wins
            b_losses = (eval_boot_w * float(n_eval)) @ fam_losses
            b_elo = compute_pairwise_elo_from_wins_losses(b_wins, b_losses)
            boot_final_elo_vs_r3175[rname].append(b_elo)

            pub = {k: v for k, v in rr.items() if not k.startswith("_")}
            pub.update(
                {
                    "final_eval_gain_vs_r0": float(np.mean(diff_final_fam)),
                    "final_eval_gain_vs_r0_ci95": np.quantile(b_final_diff, [0.025, 0.975]).tolist(),
                    "final_eval_gain_vs_r0_lcb95_one_sided": float(np.quantile(b_final_diff, 0.05)),
                    "time_avg_eval_gain_vs_r0": float(np.mean(diff_tavg_fam)),
                    "time_avg_eval_gain_vs_r0_ci95": np.quantile(b_tavg_diff, [0.025, 0.975]).tolist(),
                    "time_avg_eval_gain_vs_r0_lcb95_one_sided": float(np.quantile(b_tavg_diff, 0.05)),
                    "final_eval_gain_vs_r3175": float(np.mean(diff_vs_r3175_fam)),
                    "final_eval_gain_vs_r3175_ci95": np.quantile(b_vs_r3175, [0.025, 0.975]).tolist(),
                    "elo_vs_r3175_eval_half": elo_point,
                    "elo_vs_r3175_eval_half_ci95": np.quantile(b_elo, [0.025, 0.975]).tolist(),
                }
            )
            seed_rules_pub[rname] = pub

        per_seed_results[str(seed)] = {
            "seed": int(seed),
            "n_select_families": len(sel_fams),
            "n_eval_families": len(eval_fams),
            "select_families": sel_fams,
            "eval_families": eval_fams,
            "n_select_contexts": int(np.sum(sel_ctx_mask)),
            "n_eval_contexts": int(np.sum(eval_ctx_mask)),
            "rules": seed_rules_pub,
        }

    # Summarize across the 3 split seeds for each rule
    across_seeds_summary: dict[str, Any] = {}
    for rname in RULE_NAMES:
        f_gains = [
            per_seed_results[str(s)]["rules"][rname]["final_eval_gain_vs_r0"]
            for s in split_seeds
        ]
        f_lcbs = [
            per_seed_results[str(s)]["rules"][rname]["final_eval_gain_vs_r0_lcb95_one_sided"]
            for s in split_seeds
        ]
        t_gains = [
            per_seed_results[str(s)]["rules"][rname]["time_avg_eval_gain_vs_r0"]
            for s in split_seeds
        ]
        t_lcbs = [
            per_seed_results[str(s)]["rules"][rname]["time_avg_eval_gain_vs_r0_lcb95_one_sided"]
            for s in split_seeds
        ]
        proms = [
            per_seed_results[str(s)]["rules"][rname]["n_promotions"]
            for s in split_seeds
        ]
        precs = [
            per_seed_results[str(s)]["rules"][rname]["promotion_precision"]
            for s in split_seeds
        ]
        shrinks = [
            per_seed_results[str(s)]["rules"][rname]["net_shrinkage_vs_init"]
            for s in split_seeds
        ]
        elos = [
            per_seed_results[str(s)]["rules"][rname]["elo_vs_r3175_eval_half"]
            for s in split_seeds
        ]

        b_f_mat = np.stack(boot_final_diff_vs_r0[rname], axis=0)  # (3, B)
        b_f_med = np.median(b_f_mat, axis=0)
        b_t_mat = np.stack(boot_tavg_diff_vs_r0[rname], axis=0)
        b_t_med = np.median(b_t_mat, axis=0)
        b_elo_mat = np.stack(boot_final_elo_vs_r3175[rname], axis=0)
        b_elo_med = np.median(b_elo_mat, axis=0)

        across_seeds_summary[rname] = {
            "final_eval_gain_vs_r0_by_seed": f_gains,
            "median_final_eval_gain_vs_r0": float(np.median(f_gains)),
            "pooled_final_eval_gain_vs_r0_ci95": np.quantile(b_f_med, [0.025, 0.975]).tolist(),
            "per_seed_final_lcb95_one_sided": f_lcbs,
            "time_avg_eval_gain_vs_r0_by_seed": t_gains,
            "median_time_avg_eval_gain_vs_r0": float(np.median(t_gains)),
            "pooled_time_avg_eval_gain_vs_r0_ci95": np.quantile(b_t_med, [0.025, 0.975]).tolist(),
            "per_seed_time_avg_lcb95_one_sided": t_lcbs,
            "promotions_by_seed": proms,
            "total_promotions_across_3_seeds": int(sum(proms)),
            "promotion_precision_by_seed": precs,
            "median_promotion_precision": float(np.median(precs)),
            "overall_promotion_precision": (
                float(
                    sum(
                        per_seed_results[str(s)]["rules"][rname]["n_eval_improved_promotions"]
                        for s in split_seeds
                    )
                    / sum(proms)
                )
                if sum(proms) > 0
                else 0.0
            ),
            "net_shrinkage_vs_init_by_seed": shrinks,
            "median_net_shrinkage_vs_init": float(np.median(shrinks)),
            "elo_vs_r3175_by_seed": elos,
            "median_elo_vs_r3175": float(np.median(elos)),
            "pooled_elo_vs_r3175_ci95": np.quantile(b_elo_med, [0.025, 0.975]).tolist(),
        }

    # Select best non-R0 rule on split seed 30401 ONLY, then freeze
    seed0_key = str(split_seeds[0])
    best_non_r0 = max(
        NON_R0_RULES,
        key=lambda r: (
            per_seed_results[seed0_key]["rules"][r]["final_eval_gain_vs_r0"],
            per_seed_results[seed0_key]["rules"][r]["time_avg_eval_gain_vs_r0"],
        ),
    )
    chosen_summary = across_seeds_summary[best_non_r0]
    r0_summary = across_seeds_summary["R0_BASELINE"]

    cond1_median_gt_0001 = bool(chosen_summary["median_final_eval_gain_vs_r0"] > MARGIN_001)
    cond2_lcb95_pos_3of3 = bool(
        all(x > 0.0 for x in chosen_summary["per_seed_final_lcb95_one_sided"])
    )
    cond3_precision_gt_r0 = bool(
        chosen_summary["overall_promotion_precision"] > r0_summary["overall_promotion_precision"]
        and chosen_summary["median_promotion_precision"] >= r0_summary["median_promotion_precision"]
    )
    passes = bool(cond1_median_gt_0001 and cond2_lcb95_pos_3of3 and cond3_precision_gt_r0)

    decision = {
        "best_non_r0_rule_chosen_on_seed_30401": best_non_r0,
        "seed_30401_final_eval_gain_vs_r0": per_seed_results[seed0_key]["rules"][best_non_r0][
            "final_eval_gain_vs_r0"
        ],
        "condition_1_median_eval_gain_gt_0_001": cond1_median_gt_0001,
        "condition_2_lcb95_gt_0_on_3of3_seeds": cond2_lcb95_pos_3of3,
        "condition_3_promotion_precision_gt_r0": cond3_precision_gt_r0,
        "passes_all_criteria": passes,
        "verdict": "RULE_ADVANCES" if passes else "RULE_NOT_ADVANCED",
    }

    return {
        "select_mode": select_mode,
        "split_seeds": list(split_seeds),
        "per_seed": per_seed_results,
        "across_seeds_summary": across_seeds_summary,
        "decision": decision,
    }


def evaluate_repeated_200_splits(
    rounds: np.ndarray,
    utils: np.ndarray,
    context_families: list[str],
    n_splits: int = REPEATED_SPLITS_N,
    rng_seed: int = BOOT_SEED,
    select_mode: str = "family_equal",
) -> dict[str, Any]:
    """Run all 5 rules across 200 random balanced 9-SELECT / 10-EVAL splits (RNG 304150)."""
    uniq_fams, ctx_to_fam, fam_w = build_family_index_map(context_families)
    fam_to_idx = {f: i for i, f in enumerate(uniq_fams)}
    splits = generate_repeated_balanced_splits(uniq_fams, n_splits=n_splits, rng_seed=rng_seed)

    final_gains_vs_r0: dict[str, list[float]] = {r: [] for r in RULE_NAMES}
    tavg_gains_vs_r0: dict[str, list[float]] = {r: [] for r in RULE_NAMES}
    prom_counts: dict[str, list[int]] = {r: [] for r in RULE_NAMES}
    eval_imp_counts: dict[str, list[int]] = {r: [] for r in RULE_NAMES}
    eval_harm_counts: dict[str, list[int]] = {r: [] for r in RULE_NAMES}
    shrinkages: dict[str, list[float]] = {r: [] for r in RULE_NAMES}

    for idx_s, (sel_fams, eval_fams) in enumerate(splits):
        sel_f_idx = np.array([fam_to_idx[f] for f in sel_fams], dtype=np.int32)
        eval_f_idx = np.array([fam_to_idx[f] for f in eval_fams], dtype=np.int32)
        sel_ctx_mask = np.isin(ctx_to_fam, sel_f_idx)
        eval_ctx_mask = np.isin(ctx_to_fam, eval_f_idx)
        sel_boot_w = make_select_bootstrap_weights(
            len(sel_fams), SELECT_LCB_DRAWS, SELECT_LCB_SEED_BASE + 10000 + idx_s
        )

        runs = {
            rname: replay_single_rule(
                rule=rname,
                rounds=rounds,
                utils=utils,
                fam_w=fam_w,
                select_fam_indices=sel_f_idx,
                eval_fam_indices=eval_f_idx,
                select_ctx_mask=sel_ctx_mask,
                eval_ctx_mask=eval_ctx_mask,
                select_boot_w=sel_boot_w,
                select_mode=select_mode,
            )
            for rname in RULE_NAMES
        }
        r0_f = runs["R0_BASELINE"]["final_eval_family_equal_utility"]
        r0_t = runs["R0_BASELINE"]["time_avg_eval_family_equal_utility"]

        for rname in RULE_NAMES:
            rr = runs[rname]
            final_gains_vs_r0[rname].append(rr["final_eval_family_equal_utility"] - r0_f)
            tavg_gains_vs_r0[rname].append(rr["time_avg_eval_family_equal_utility"] - r0_t)
            prom_counts[rname].append(rr["n_promotions"])
            eval_imp_counts[rname].append(rr["n_eval_improved_promotions"])
            eval_harm_counts[rname].append(rr["n_eval_harmed_promotions"])
            shrinkages[rname].append(rr["net_shrinkage_vs_init"])

    out_rules: dict[str, Any] = {}
    for rname in RULE_NAMES:
        fg = np.asarray(final_gains_vs_r0[rname], dtype=np.float64)
        tg = np.asarray(tavg_gains_vs_r0[rname], dtype=np.float64)
        pc = np.asarray(prom_counts[rname], dtype=np.int32)
        ic = np.asarray(eval_imp_counts[rname], dtype=np.int32)
        hc = np.asarray(eval_harm_counts[rname], dtype=np.int32)
        sh = np.asarray(shrinkages[rname], dtype=np.float64)
        tot_p = int(np.sum(pc))
        tot_i = int(np.sum(ic))
        tot_h = int(np.sum(hc))
        out_rules[rname] = {
            "n_splits_with_promotions": int(np.sum(pc > 0)),
            "total_promotions": tot_p,
            "mean_promotions_per_split": float(np.mean(pc)),
            "total_eval_improved_promotions": tot_i,
            "total_eval_harmed_promotions": tot_h,
            "overall_promotion_precision": float(tot_i / tot_p) if tot_p > 0 else 0.0,
            "overall_promotion_harm_rate": float(tot_h / tot_p) if tot_p > 0 else 0.0,
            "mean_final_eval_gain_vs_r0": float(np.mean(fg)),
            "median_final_eval_gain_vs_r0": float(np.median(fg)),
            "final_eval_gain_vs_r0_q025_q975": np.quantile(fg, [0.025, 0.975]).tolist(),
            "splits_final_gain_gt_r0": int(np.sum(fg > TIE_EPS)),
            "splits_final_gain_lt_r0": int(np.sum(fg < -TIE_EPS)),
            "splits_final_gain_tied_r0": int(np.sum(np.abs(fg) <= TIE_EPS)),
            "mean_time_avg_eval_gain_vs_r0": float(np.mean(tg)),
            "median_time_avg_eval_gain_vs_r0": float(np.median(tg)),
            "time_avg_eval_gain_vs_r0_q025_q975": np.quantile(tg, [0.025, 0.975]).tolist(),
            "mean_net_shrinkage_vs_init": float(np.mean(sh)),
            "median_net_shrinkage_vs_init": float(np.median(sh)),
        }

    return {
        "n_splits": n_splits,
        "rng_seed": rng_seed,
        "select_mode": select_mode,
        "rules": out_rules,
    }
