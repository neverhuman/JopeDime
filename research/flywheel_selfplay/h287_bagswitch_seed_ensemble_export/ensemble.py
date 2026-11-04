"""H287 seed-ensemble construction, 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts),
exploratory -1.0 utility clipping (CLIP), and deterministic frozen policy scorer for H288."""

import hashlib
import json
import math
import os
import shutil
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np

import analysis_h287 as stat
import bagged
import h265_core as core
import veto

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

K = 8
H284_SEEDS = (279101, 279102, 279103)
REFIT_SEEDS = (287101, 287102, 287103)
CLIP_FLOOR = -1.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def save_json(path: Path, obj: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=float)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha256_file(path)


def parse_margin(val) -> float:
    return math.inf if val == "inf" else float(val)


def family_folds(
    train_rows: list[dict], fam_by_cid: dict[str, str], seed: int, n: int = 4
) -> dict[int, list[str]]:
    """Deterministic 4-fold family-grouped split of all 205 DL-Train families (same stream as H284/H278)."""
    fams = sorted(set(fam_by_cid.values()))
    rng = np.random.default_rng(core.stream_seed(seed, "h278-full-train-folds"))
    perm = list(rng.permutation(fams))
    f2c: dict[str, list[str]] = {}
    for r in train_rows:
        cid = r["context_sha256"]
        f2c.setdefault(fam_by_cid[cid], []).append(cid)
    out: dict[int, list[str]] = {0: []}
    for i, f in enumerate(perm):
        out.setdefault(1 + (i % n), []).extend(sorted(f2c[f]))
    return {k: sorted(v) for k, v in out.items()}


def clip_packets(packets: dict[str, dict], clip_floor: float = CLIP_FLOOR) -> dict[str, dict]:
    """Return a copy of packets with per-context training utility clipped at clip_floor (-1.0)."""
    out = {}
    for cid, pkt in packets.items():
        p2 = dict(pkt)
        p2["utility_k"] = np.maximum(np.asarray(pkt["utility_k"], dtype=np.float64), float(clip_floor))
        p2["r882_u"] = float(max(float(pkt["r882_u"]), float(clip_floor)))
        out[cid] = p2
    return out


def verify_reload(
    model_str: str, path: Path, X_check: np.ndarray, num_iteration: int | None = None
) -> dict:
    mem_b = lgb.Booster(model_str=model_str)
    disk_b = lgb.Booster(model_file=str(path))
    if disk_b.num_feature() != 568:
        raise RuntimeError(f"feature count mismatch on {path}: {disk_b.num_feature()}")
    if num_iteration is not None:
        mem_p = np.asarray(mem_b.predict(X_check, num_iteration=int(num_iteration)), dtype=np.float64)
        disk_p = np.asarray(disk_b.predict(X_check, num_iteration=int(num_iteration)), dtype=np.float64)
    else:
        mem_p = np.asarray(mem_b.predict(X_check), dtype=np.float64)
        disk_p = np.asarray(disk_b.predict(X_check), dtype=np.float64)
    if not np.array_equal(mem_p, disk_p):
        raise RuntimeError(f"serialized booster prediction mismatch on {path}")
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "num_feature": int(disk_b.num_feature()),
        "reload_bit_exact_on_64_train_contexts": True,
    }


def vote_consensus_pick(member_picks: list[int], min_agree: int) -> int:
    """Switch away from slot 0 to candidate j >= 1 iff >= min_agree of the 3 members switch AND agree on j."""
    non_zero = [int(p) for p in member_picks if int(p) != 0]
    for cand in sorted(set(non_zero)):
        if non_zero.count(cand) >= int(min_agree):
            return int(cand)
    return 0


def aggregate_ensemble_picks(
    member_outputs: list[dict],
    n_ctx: int,
    q_fixed: int = bagged.Q_FIXED,
    tau: float = veto.TAU,
) -> dict:
    """Deterministically aggregate 3 member BAG_SWITCH outputs across n_ctx contexts into:
    - ENS_MEAN: mean veto (< tau) + argmax of mean ranker score among non-vetoed + mean ranker margin > mean(strict_margin)
                + mean bag votes >= q_fixed (i.e. >= 18 of 24 bagged rankers across the 3 seeds).
    - ENS_MEAN_PROPOSER: same mean veto + mean proposer ranker margin without the 24-bag vote gate.
    - ENS_VOTE2: >= 2 of 3 seeds' BAG_SWITCH picks switch away from slot 0 AND agree on the same candidate; else 0.
    - ENS_VOTE3: 3 of 3 seeds' BAG_SWITCH picks switch away from slot 0 AND agree on the same candidate; else 0.
    """
    if len(member_outputs) != 3:
        raise ValueError(f"Expected 3 member outputs, got {len(member_outputs)}")

    p_inv_mean = np.mean([ np.asarray(m["p_inv"], dtype=np.float64) for m in member_outputs ], axis=0)
    raw_mean = np.mean([ np.asarray(m["raw_scores"], dtype=np.float64) for m in member_outputs ], axis=0)
    margins = [parse_margin(m["margin"]) for m in member_outputs]
    margin_mean = math.inf if any(math.isinf(x) for x in margins) else float(np.mean(margins))
    masked_mean = veto.mask(raw_mean, p_inv_mean, tau=tau)

    j_mean_prop = [core.select_from_top4_scores(masked_mean[i], margin_mean) for i in range(n_ctx)]

    total_bag_votes = np.zeros(n_ctx, dtype=np.int32)
    for m in member_outputs:
        bs = np.asarray(m["bag_scores"], dtype=np.float64)  # (8, n_ctx, k)
        mg = parse_margin(m["margin"])
        total_bag_votes += bagged.vote_counts(bs, j_mean_prop, mg)

    min_total_bag_votes = int(len(member_outputs) * int(q_fixed))  # 3 * 6 = 18
    picks_ens_mean = [
        int(j) if int(j) != 0 and int(v) >= min_total_bag_votes else 0
        for j, v in zip(j_mean_prop, total_bag_votes)
    ]

    picks_vote2 = []
    picks_vote3 = []
    for i in range(n_ctx):
        mp = [int(m["picks_bag"][i]) for m in member_outputs]
        picks_vote2.append(vote_consensus_pick(mp, min_agree=2))
        picks_vote3.append(vote_consensus_pick(mp, min_agree=3))

    return {
        "ENS_MEAN": picks_ens_mean,
        "ENS_MEAN_PROPOSER": j_mean_prop,
        "ENS_VOTE2": picks_vote2,
        "ENS_VOTE3": picks_vote3,
        "ens_mean_total_bag_votes": total_bag_votes.tolist(),
        "ens_mean_margin": "inf" if math.isinf(margin_mean) else float(margin_mean),
    }


def fit_and_predict_member_on_fold(
    m_cfg: dict,
    fold_seed: int,
    hold: int,
    tr: list[str],
    va: list[str],
    packets: dict[str, dict],
    fam_by_cid: dict[str, str],
    k: int = K,
) -> dict:
    """Fit one member (RERANK8_VETO proposer + B=8 family-bootstrapped rankers) on 3-of-4 folds `tr`
    and predict on held-out fold `va`. When m_cfg['h284_seed'] == fold_seed, uses the exact H284 inner
    random stream so single-seed BAG_SWITCH is 100% bit-identical to H284."""
    core.check_cooperative_pause()
    s_tag = (
        int(m_cfg["h284_seed"])
        if int(m_cfg["h284_seed"]) == int(fold_seed)
        else core.stream_seed(int(m_cfg["refit_seed"]), int(fold_seed))
    )
    lr = float(m_cfg["lr"])
    n_est = int(m_cfg["n_est"])
    margin = parse_margin(m_cfg["margin"])
    tau = float(m_cfg.get("tau", veto.TAU))
    q_fixed = int(m_cfg.get("q", bagged.Q_FIXED))

    X_tr = np.concatenate([packets[c]["X"] for c in tr])
    y_tr = np.concatenate([packets[c]["rel_k"] for c in tr])
    X_va = np.concatenate([packets[c]["X"] for c in va])
    inv_tr = np.concatenate([~packets[c]["gate_k"] for c in tr])

    g_in = veto.fit_gate(
        X_tr, inv_tr, core.stream_seed(s_tag, 0, "RERANK8_VETO", "gate", hold)
    )
    pv = veto.gate_probs(g_in, X_va, len(va), k)
    m_in = bagged.fit_ranker_n(
        X_tr,
        y_tr,
        len(tr),
        k,
        lr,
        n_est,
        core.stream_seed(s_tag, 0, "RERANK8", "inner", hold, lr),
    )
    raw_sc = np.asarray(
        m_in.booster_.predict(X_va, num_iteration=n_est), dtype=np.float64
    ).reshape(len(va), k)
    masked_sc = veto.mask(raw_sc, pv, tau=tau)
    j_ref = [core.select_from_top4_scores(masked_sc[i], margin) for i in range(len(va))]

    tr_f2c: dict[str, list[str]] = {}
    for c in tr:
        tr_f2c.setdefault(fam_by_cid[c], []).append(c)
    tr_fams = sorted(tr_f2c)
    bag_scores = []
    for b in range(bagged.N_BAGS):
        rng = np.random.default_rng(
            core.stream_seed(s_tag, 0, "BAG_SWITCH", "inner_boot", hold, b)
        )
        bfams = rng.choice(tr_fams, size=len(tr_fams), replace=True)
        bcids = [c for f in bfams for c in tr_f2c[f]]
        Xb = np.concatenate([packets[c]["X"] for c in bcids])
        yb = np.concatenate([packets[c]["rel_k"] for c in bcids])
        mb = bagged.fit_ranker_n(
            Xb,
            yb,
            len(bcids),
            k,
            lr,
            n_est,
            core.stream_seed(s_tag, 0, "BAG_SWITCH", "inner_bag", hold, b),
        )
        sb = np.asarray(
            mb.booster_.predict(X_va, num_iteration=n_est), dtype=np.float64
        ).reshape(len(va), k)
        bag_scores.append(sb)

    bag_scores_arr = np.stack(bag_scores, axis=0)
    votes = bagged.vote_counts(bag_scores_arr, j_ref, margin)
    picks_bag = bagged.switch_by_votes(j_ref, votes, q_fixed)
    return {
        "p_inv": pv,
        "raw_scores": raw_sc,
        "bag_scores": bag_scores_arr,
        "j_ref": j_ref,
        "votes": votes,
        "picks_bag": picks_bag,
        "margin": "inf" if math.isinf(margin) else float(margin),
    }


def select_clip_member_configs(
    train_rows: list[dict],
    packets_clip: dict[str, dict],
    fam_by_cid: dict[str, str],
    seeds: tuple[int, ...] = H284_SEEDS,
    refit_seeds: tuple[int, ...] = REFIT_SEEDS,
) -> list[dict]:
    """Run inner 4-fold hyperparameter selection on DL-Train with per-context training utility clipped at -1.0."""
    cfgs = []
    for idx, seed in enumerate(seeds):
        cbf = family_folds(train_rows, fam_by_cid, seed, n=4)
        cbf[0] = [cbf[1][0]]
        rv = veto.run_cell_veto(seed, 0, cbf, packets_clip, fam_by_cid, K, "")["receipt"]
        cfgs.append(
            {
                "h284_seed": int(seed),
                "refit_seed": int(refit_seeds[idx]),
                "lr": float(rv["lr"]),
                "n_est": int(rv["n_estimators"]),
                "margin": rv["strict_margin"],
                "raw_margin": float(rv["margin"]),
                "strict_admitted": bool(rv["strict_admitted"]),
                "inner_fit_family_gain_clipped": float(rv["inner_fit_family_gain"]),
                "inner_fit_ci95_clipped": rv["inner_fit_ci95"],
                "tau": float(rv["tau"]),
                "q": int(bagged.Q_FIXED),
            }
        )
    return cfgs


def run_inner_cv_dl_train(
    train_rows: list[dict],
    packets: dict[str, dict],
    fam_by_cid: dict[str, str],
    h284_cells: list[dict],
) -> dict:
    """Run 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts) across the 3 H284 fold
    partitions (seeds 279101-279103; refit seeds 287101-287103)."""
    all_cids = [r["context_sha256"] for r in train_rows]
    all_fams = [r["family_id"] for r in train_rows]
    n_ctx = len(all_cids)
    cid_pos = {c: i for i, c in enumerate(all_cids)}

    member_cfgs = [
        {
            "h284_seed": int(c["seed"]),
            "refit_seed": int(REFIT_SEEDS[idx]),
            "lr": float(c["selected_lr"]),
            "n_est": int(c["selected_n_estimators"]),
            "margin": c["strict_margin"],
            "tau": float(c["selected_tau"]),
            "q": int(c["q_fixed"]),
        }
        for idx, c in enumerate(h284_cells)
    ]

    packets_clip = clip_packets(packets, clip_floor=CLIP_FLOOR)
    clip_cfgs = select_clip_member_configs(
        train_rows, packets_clip, fam_by_cid, seeds=H284_SEEDS, refit_seeds=REFIT_SEEDS
    )

    n_seeds = len(H284_SEEDS)
    panel = np.full((n_seeds, n_ctx, len(stat.ALL_ARMS)), np.nan, dtype=np.float64)
    invalid_by_seed = {a: [] for a in stat.ALL_ARMS}
    changed_by_seed = {a: [] for a in stat.ALL_ARMS}
    replicates_meta = []

    for r_idx, fold_seed in enumerate(H284_SEEDS):
        cids_by_fold = family_folds(train_rows, fam_by_cid, fold_seed, n=4)
        # Allocate full-DL-Train OOF buffers for the 3 unclipped members and 3 clipped members on this fold split
        oof_unclip = [
            {
                "p_inv": np.zeros((n_ctx, K), dtype=np.float64),
                "raw_scores": np.zeros((n_ctx, K), dtype=np.float64),
                "bag_scores": np.zeros((bagged.N_BAGS, n_ctx, K), dtype=np.float64),
                "j_ref": [0] * n_ctx,
                "votes": np.zeros(n_ctx, dtype=np.int32),
                "picks_bag": [0] * n_ctx,
                "margin": member_cfgs[m]["margin"],
            }
            for m in range(3)
        ]
        oof_clip = [
            {
                "p_inv": np.zeros((n_ctx, K), dtype=np.float64),
                "raw_scores": np.zeros((n_ctx, K), dtype=np.float64),
                "bag_scores": np.zeros((bagged.N_BAGS, n_ctx, K), dtype=np.float64),
                "j_ref": [0] * n_ctx,
                "votes": np.zeros(n_ctx, dtype=np.int32),
                "picks_bag": [0] * n_ctx,
                "margin": clip_cfgs[m]["margin"],
            }
            for m in range(3)
        ]

        for hold in (1, 2, 3, 4):
            tr = [c for f in (1, 2, 3, 4) if f != hold for c in cids_by_fold[f]]
            va = cids_by_fold[hold]
            for m in range(3):
                res_u = fit_and_predict_member_on_fold(
                    member_cfgs[m], fold_seed, hold, tr, va, packets, fam_by_cid, K
                )
                for i, c in enumerate(va):
                    pos = cid_pos[c]
                    oof_unclip[m]["p_inv"][pos] = res_u["p_inv"][i]
                    oof_unclip[m]["raw_scores"][pos] = res_u["raw_scores"][i]
                    oof_unclip[m]["bag_scores"][:, pos, :] = res_u["bag_scores"][:, i, :]
                    oof_unclip[m]["j_ref"][pos] = int(res_u["j_ref"][i])
                    oof_unclip[m]["votes"][pos] = int(res_u["votes"][i])
                    oof_unclip[m]["picks_bag"][pos] = int(res_u["picks_bag"][i])

                # If clip_cfgs[m] has identical (lr, n_est, margin) to member_cfgs[m], reuse res_u directly!
                if (
                    clip_cfgs[m]["lr"] == member_cfgs[m]["lr"]
                    and clip_cfgs[m]["n_est"] == member_cfgs[m]["n_est"]
                    and clip_cfgs[m]["margin"] == member_cfgs[m]["margin"]
                ):
                    res_c = res_u
                else:
                    res_c = fit_and_predict_member_on_fold(
                        clip_cfgs[m], fold_seed, hold, tr, va, packets, fam_by_cid, K
                    )
                for i, c in enumerate(va):
                    pos = cid_pos[c]
                    oof_clip[m]["p_inv"][pos] = res_c["p_inv"][i]
                    oof_clip[m]["raw_scores"][pos] = res_c["raw_scores"][i]
                    oof_clip[m]["bag_scores"][:, pos, :] = res_c["bag_scores"][:, i, :]
                    oof_clip[m]["j_ref"][pos] = int(res_c["j_ref"][i])
                    oof_clip[m]["votes"][pos] = int(res_c["votes"][i])
                    oof_clip[m]["picks_bag"][pos] = int(res_c["picks_bag"][i])

        ens_u = aggregate_ensemble_picks(oof_unclip, n_ctx, q_fixed=bagged.Q_FIXED, tau=veto.TAU)
        ens_c = aggregate_ensemble_picks(oof_clip, n_ctx, q_fixed=bagged.Q_FIXED, tau=veto.TAU)

        picks_by_arm = {
            "slot0": [0] * n_ctx,
            "RERANK8_VETO": oof_unclip[r_idx]["j_ref"],
            "BAG_SWITCH": oof_unclip[r_idx]["picks_bag"],
            "ENS_MEAN": ens_u["ENS_MEAN"],
            "ENS_MEAN_PROPOSER": ens_u["ENS_MEAN_PROPOSER"],
            "ENS_VOTE2": ens_u["ENS_VOTE2"],
            "ENS_VOTE3": ens_u["ENS_VOTE3"],
            "CLIP": oof_clip[r_idx]["picks_bag"],
            "CLIP_ENS_VOTE2": ens_c["ENS_VOTE2"],
        }

        for ai, arm in enumerate(stat.ALL_ARMS):
            pa = picks_by_arm[arm]
            # Evaluate all arms on true unclipped per-context utility
            panel[r_idx, :, ai] = [packets[c]["utility_k"][j] for c, j in zip(all_cids, pa)]
            inv_cnt = int(
                sum(
                    (not bool(packets[c]["gate_k"][j])) or float(packets[c]["utility_k"][j]) <= -2.0 + 1e-9
                    for c, j in zip(all_cids, pa)
                )
            )
            chg_cnt = int(sum(int(j) != 0 for j in pa))
            invalid_by_seed[arm].append(inv_cnt)
            changed_by_seed[arm].append(chg_cnt)

        replicates_meta.append(
            {
                "replicate_index": int(r_idx),
                "h284_fold_seed": int(fold_seed),
                "refit_seed": int(REFIT_SEEDS[r_idx]),
                "single_seed_config": member_cfgs[r_idx],
                "clip_config": clip_cfgs[r_idx],
            }
        )

    summary = stat.summarize_panel(
        panel,
        all_fams,
        invalid_by_seed,
        changed_by_seed,
        arm_names=stat.ALL_ARMS,
        seed_base=stat.BOOT_SEED,
        label="inner_cv_dl_train_205fams_1219ctxs",
        reused=False,
    )
    return {
        "summary": summary,
        "panel": panel,
        "member_cfgs": member_cfgs,
        "clip_cfgs": clip_cfgs,
        "replicates_meta": replicates_meta,
    }


def score_single_member_from_disk(
    export_dir: Path, cell_info: dict, X: np.ndarray, n_ctx: int, k: int = K
) -> dict:
    """Load one frozen member's LightGBM models (ranker, veto, bag0..7) from export_dir, verify sha256s,
    and return full score and pick dicts on feature matrix X of shape (n_ctx * k, 568) or (n_ctx, k, 568)."""
    export_dir = Path(export_dir)
    X_2d = np.asarray(X, dtype=np.float32).reshape(n_ctx * k, 568)

    r_path = export_dir / cell_info["ranker"]["file"]
    v_path = export_dir / cell_info["veto"]["file"]
    if sha256_file(r_path) != cell_info["ranker"]["sha256"]:
        raise RuntimeError(f"ranker sha256 mismatch: {r_path}")
    if sha256_file(v_path) != cell_info["veto"]["sha256"]:
        raise RuntimeError(f"veto sha256 mismatch: {v_path}")

    n_est = int(cell_info["selected_n_estimators"])
    strict_margin = parse_margin(cell_info["strict_margin"])
    tau = float(cell_info.get("selected_tau", veto.TAU))
    q_fixed = int(cell_info.get("q_fixed", bagged.Q_FIXED))

    ranker = lgb.Booster(model_file=str(r_path))
    gate = lgb.Booster(model_file=str(v_path))
    if ranker.num_feature() != 568 or gate.num_feature() != 568:
        raise RuntimeError("frozen model feature count mismatch")

    p_inv = np.asarray(gate.predict(X_2d), dtype=np.float64).reshape(n_ctx, k)
    raw_scores = np.asarray(ranker.predict(X_2d, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, k)
    masked_scores = veto.mask(raw_scores, p_inv, tau=tau)
    j_ref = [core.select_from_top4_scores(masked_scores[i], strict_margin) for i in range(n_ctx)]

    bag_scores = []
    for b_info in cell_info["bags"]:
        b_path = export_dir / b_info["file"]
        if sha256_file(b_path) != b_info["sha256"]:
            raise RuntimeError(f"bag sha256 mismatch: {b_path}")
        bm = lgb.Booster(model_file=str(b_path))
        if bm.num_feature() != 568:
            raise RuntimeError(f"bag feature count mismatch: {b_path}")
        sb = np.asarray(bm.predict(X_2d, num_iteration=n_est), dtype=np.float64).reshape(n_ctx, k)
        bag_scores.append(sb)

    bag_scores_arr = np.stack(bag_scores, axis=0)
    votes = bagged.vote_counts(bag_scores_arr, j_ref, strict_margin)
    picks_bag = bagged.switch_by_votes(j_ref, votes, q_fixed)
    return {
        "p_inv": p_inv,
        "raw_scores": raw_scores,
        "bag_scores": bag_scores_arr,
        "j_ref": j_ref,
        "votes": votes,
        "picks_bag": picks_bag,
        "margin": "inf" if math.isinf(strict_margin) else float(strict_margin),
    }


def score_all_h287_arms(
    export_dir: Path,
    X: np.ndarray,
    n_ctx: int,
    k: int = K,
    spec: dict | None = None,
) -> dict:
    """Deterministic frozen scorer for all H287 arms on feature matrix X ((n_ctx * k, 568) or (n_ctx, k, 568)).
    Returns deterministic 1D pick lists for ENS_MEAN, ENS_MEAN_PROPOSER, ENS_VOTE2, ENS_VOTE3, CLIP_ENS_VOTE2,
    and per-seed pick lists for BAG_SWITCH, RERANK8_VETO, and CLIP."""
    export_dir = Path(export_dir)
    if spec is None:
        spec = json.loads((export_dir / "ensemble_spec.json").read_text())

    h284_members = [
        score_single_member_from_disk(export_dir, c, X, n_ctx, k)
        for c in spec["h284_bag_switch_members"]
    ]
    ens_u = aggregate_ensemble_picks(
        h284_members,
        n_ctx,
        q_fixed=int(spec.get("q_fixed", bagged.Q_FIXED)),
        tau=float(spec.get("tau", veto.TAU)),
    )

    clip_members = [
        score_single_member_from_disk(export_dir, c, X, n_ctx, k)
        for c in spec["clip_bag_switch_members"]
    ]
    ens_c = aggregate_ensemble_picks(
        clip_members,
        n_ctx,
        q_fixed=int(spec.get("q_fixed", bagged.Q_FIXED)),
        tau=float(spec.get("tau", veto.TAU)),
    )

    return {
        "slot0": [0] * n_ctx,
        "ENS_MEAN": ens_u["ENS_MEAN"],
        "ENS_MEAN_PROPOSER": ens_u["ENS_MEAN_PROPOSER"],
        "ENS_VOTE2": ens_u["ENS_VOTE2"],
        "ENS_VOTE3": ens_u["ENS_VOTE3"],
        "CLIP_ENS_VOTE2": ens_c["ENS_VOTE2"],
        "CLIP_ENS_MEAN": ens_c["ENS_MEAN"],
        "per_seed": {
            "RERANK8_VETO": [m["j_ref"] for m in h284_members],
            "BAG_SWITCH": [m["picks_bag"] for m in h284_members],
            "CLIP": [m["picks_bag"] for m in clip_members],
        },
        "diagnostics": {
            "ens_mean_total_bag_votes": ens_u["ens_mean_total_bag_votes"],
            "ens_mean_margin": ens_u["ens_mean_margin"],
        },
    }


def score_h287_arm(
    export_dir: Path,
    arm: str,
    X: np.ndarray,
    n_ctx: int,
    k: int = K,
    spec: dict | None = None,
    seed_index: int = 0,
) -> list[int]:
    """Score a single named H287 arm on feature matrix X ((n_ctx * k, 568) or (n_ctx, k, 568)).
    For deterministic ensemble arms (ENS_MEAN, ENS_MEAN_PROPOSER, ENS_VOTE2, ENS_VOTE3, CLIP_ENS_VOTE2, CLIP_ENS_MEAN),
    returns the single deterministic pick list of length n_ctx. For per-seed arms (BAG_SWITCH, RERANK8_VETO, CLIP),
    returns seed_index (0, 1, or 2)."""
    all_res = score_all_h287_arms(export_dir, X, n_ctx, k=k, spec=spec)
    if arm in all_res and isinstance(all_res[arm], list):
        return all_res[arm]
    if arm in all_res["per_seed"]:
        return all_res["per_seed"][arm][int(seed_index)]
    raise KeyError(f"Unknown H287 arm: {arm}")


def export_h287_models_and_spec(
    train_rows: list[dict],
    packets: dict[str, dict],
    fam_by_cid: dict[str, str],
    h284_export_dir: Path,
    export_dir: Path,
    code_hashes: dict[str, str],
    source_commit: str,
) -> dict:
    """Freeze and export all H287 models (the 30 frozen H284 models + 30 full-DL-Train CLIP models),
    write ensemble_spec.json and fit_lock.json, and verify 64-context reload bit-exactness."""
    h284_export_dir = Path(h284_export_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    all_cids = sorted(packets.keys())
    all_fams = sorted(set(fam_by_cid.values()))
    if len(all_cids) != 1219 or len(all_fams) != 205:
        raise RuntimeError(f"unexpected DL-Train size: {len(all_cids)} contexts, {len(all_fams)} families")

    check_cids = all_cids[:64]
    X_check = np.concatenate([packets[c]["X"] for c in check_cids])

    h284_em = json.loads((h284_export_dir / "export_manifest.json").read_text())

    # 1. Copy and verify all 30 frozen H284 BAG_SWITCH models (seeds 279101-279103)
    h284_members = []
    for c in h284_em["cells"]:
        seed = int(c["seed"])
        n_est = int(c["selected_n_estimators"])
        r_src = h284_export_dir / c["ranker"]["file"]
        v_src = h284_export_dir / c["veto"]["file"]
        r_dst = export_dir / c["ranker"]["file"]
        v_dst = export_dir / c["veto"]["file"]
        if r_src.resolve() != r_dst.resolve():
            shutil.copy2(r_src, r_dst)
        if v_src.resolve() != v_dst.resolve():
            shutil.copy2(v_src, v_dst)
        if sha256_file(r_dst) != c["ranker"]["sha256"] or sha256_file(v_dst) != c["veto"]["sha256"]:
            raise RuntimeError(f"H284 model sha256 mismatch for seed {seed}")
        r_info = verify_reload(r_dst.read_text(), r_dst, X_check, num_iteration=n_est)
        r_info["h284_S3_VersionId"] = c["ranker"].get("S3_VersionId")
        v_info = verify_reload(v_dst.read_text(), v_dst, X_check, num_iteration=None)
        v_info["h284_S3_VersionId"] = c["veto"].get("S3_VersionId")

        bag_infos = []
        for b in c["bags"]:
            b_src = h284_export_dir / b["file"]
            b_dst = export_dir / b["file"]
            if b_src.resolve() != b_dst.resolve():
                shutil.copy2(b_src, b_dst)
            if sha256_file(b_dst) != b["sha256"]:
                raise RuntimeError(f"H284 bag sha256 mismatch: {b_dst}")
            b_info = verify_reload(b_dst.read_text(), b_dst, X_check, num_iteration=n_est)
            b_info["bag_index"] = int(b["bag_index"])
            b_info["h284_S3_VersionId"] = b.get("S3_VersionId")
            bag_infos.append(b_info)

        h284_members.append(
            {
                "seed": seed,
                "selected_lr": float(c["selected_lr"]),
                "selected_n_estimators": n_est,
                "raw_margin": float(c["raw_margin"]),
                "strict_margin": c["strict_margin"],
                "strict_admitted": bool(c["strict_admitted"]),
                "selected_tau": float(c["selected_tau"]),
                "n_bags": int(c["n_bags"]),
                "q_fixed": int(c["q_fixed"]),
                "q_selected": int(c["q_selected"]),
                "ranker": r_info,
                "veto": v_info,
                "bags": bag_infos,
            }
        )

    # 2. Fit and export full-DL-Train CLIP BAG_SWITCH models (per-context training utility clipped at -1.0)
    packets_clip = clip_packets(packets, clip_floor=CLIP_FLOOR)
    clip_members = []
    for idx, seed in enumerate(H284_SEEDS):
        cids_by_fold = family_folds(train_rows, fam_by_cid, seed, n=4)
        cids_by_fold[0] = [cids_by_fold[1][0]]
        cell = bagged.run_cell_bagged(
            seed, 0, cids_by_fold, packets_clip, fam_by_cid, K, "", store_models=True
        )
        models = cell["models"]
        receipts = cell["receipts"]
        n_est = int(models["n_estimators"])
        strict_margin = models["strict_margin"]
        raw_margin = float(models["raw_margin"])
        q_nested = int(models["q_nested"])

        r_path = export_dir / f"clip-ranker-seed{seed}.txt"
        v_path = export_dir / f"clip-veto-seed{seed}.txt"
        r_path.write_text(models["rv_model"])
        v_path.write_text(models["rv_gate"])
        r_info = verify_reload(models["rv_model"], r_path, X_check, num_iteration=n_est)
        v_info = verify_reload(models["rv_gate"], v_path, X_check, num_iteration=None)

        bag_infos = []
        for b, m_str in enumerate(models["bag_models"]):
            b_path = export_dir / f"clip-bag{b}-seed{seed}.txt"
            b_path.write_text(m_str)
            b_info = verify_reload(m_str, b_path, X_check, num_iteration=n_est)
            b_info["bag_index"] = int(b)
            bag_infos.append(b_info)

        clip_members.append(
            {
                "seed": int(seed),
                "refit_seed": int(REFIT_SEEDS[idx]),
                "clip_floor": float(CLIP_FLOOR),
                "selected_lr": float(receipts["RERANK8_VETO"]["lr"]),
                "selected_n_estimators": n_est,
                "raw_margin": raw_margin,
                "strict_margin": "inf" if math.isinf(strict_margin) else float(strict_margin),
                "strict_admitted": bool(receipts["RERANK8_VETO"]["strict_admitted"]),
                "selected_tau": float(veto.TAU),
                "n_bags": int(bagged.N_BAGS),
                "q_fixed": int(bagged.Q_FIXED),
                "q_selected": q_nested,
                "ranker": r_info,
                "veto": v_info,
                "bags": bag_infos,
            }
        )

    spec = {
        "schema": "jope.h287.ensemble-spec/v1",
        "wave": "H287",
        "slug": "h287_bagswitch_seed_ensemble_export",
        "fit_split": "DL-Train",
        "training_family_count": len(all_fams),
        "training_context_count": len(all_cids),
        "fresh_pools_read": 0,
        "dl_val_reads_before_lock": 0,
        "k": K,
        "tau": float(veto.TAU),
        "n_bags_per_seed": int(bagged.N_BAGS),
        "q_fixed": int(bagged.Q_FIXED),
        "scorer_module": "research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/ensemble.py",
        "scorer_function": "score_h287_arm",
        "batch_scorer_function": "score_all_h287_arms",
        "arms": {
            "ENS_MEAN": {
                "deterministic_single_policy": True,
                "member_group": "h284_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "member_strict_margins": [c["strict_margin"] for c in h284_members],
                "ensemble_strict_margin": float(
                    np.mean([parse_margin(c["strict_margin"]) for c in h284_members])
                ),
                "veto_threshold_tau": float(veto.TAU),
                "min_total_bag_votes_out_of_24": int(3 * bagged.Q_FIXED),
                "aggregation_rule": (
                    "p_inv_mean = mean_{m in 1..3} p_inv_m; mask k >= 1 where p_inv_mean[k] >= 0.10 to -inf; "
                    "s_mean = mean_{m in 1..3} ranker_score_m; j* = argmax_k masked(s_mean)[k]; "
                    "switch from slot 0 to j* iff (s_mean[j*] - s_mean[0]) > mean(strict_margin_m) "
                    "AND sum_{m=1..3} sum_{b=0..7} 1[(bag_{m,b}[j*] - bag_{m,b}[0]) > strict_margin_m] >= 18"
                ),
            },
            "ENS_MEAN_PROPOSER": {
                "deterministic_single_policy": True,
                "member_group": "h284_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "member_strict_margins": [c["strict_margin"] for c in h284_members],
                "ensemble_strict_margin": float(
                    np.mean([parse_margin(c["strict_margin"]) for c in h284_members])
                ),
                "veto_threshold_tau": float(veto.TAU),
                "aggregation_rule": (
                    "p_inv_mean = mean_{m in 1..3} p_inv_m; mask k >= 1 where p_inv_mean[k] >= 0.10 to -inf; "
                    "s_mean = mean_{m in 1..3} ranker_score_m; j* = argmax_k masked(s_mean)[k]; "
                    "switch from slot 0 to j* iff (s_mean[j*] - s_mean[0]) > mean(strict_margin_m)"
                ),
            },
            "ENS_VOTE2": {
                "deterministic_single_policy": True,
                "member_group": "h284_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "member_strict_margins": [c["strict_margin"] for c in h284_members],
                "min_agreeing_seeds": 2,
                "q_fixed_per_seed": int(bagged.Q_FIXED),
                "aggregation_rule": (
                    "Each member m in {279101, 279102, 279103} computes its frozen H284 BAG_SWITCH (q=6) pick p_m in 0..7; "
                    "switch away from slot 0 to candidate j >= 1 iff at least 2 of 3 members have p_m == j; else slot 0."
                ),
            },
            "ENS_VOTE3": {
                "deterministic_single_policy": True,
                "member_group": "h284_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "member_strict_margins": [c["strict_margin"] for c in h284_members],
                "min_agreeing_seeds": 3,
                "q_fixed_per_seed": int(bagged.Q_FIXED),
                "aggregation_rule": (
                    "Each member m in {279101, 279102, 279103} computes its frozen H284 BAG_SWITCH (q=6) pick p_m in 0..7; "
                    "switch away from slot 0 to candidate j >= 1 iff all 3 of 3 members have p_m == j; else slot 0."
                ),
            },
            "CLIP": {
                "deterministic_single_policy": False,
                "exploratory": True,
                "member_group": "clip_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "refit_seeds": list(REFIT_SEEDS),
                "clip_floor": float(CLIP_FLOOR),
                "member_strict_margins": [c["strict_margin"] for c in clip_members],
                "q_fixed_per_seed": int(bagged.Q_FIXED),
                "aggregation_rule": (
                    "3-seed BAG_SWITCH (q=6) refit on DL-Train with per-context training utility clipped at -1.0 "
                    "so -2.0 invalid picks do not dominate inner hyperparameter selection loss."
                ),
            },
            "CLIP_ENS_VOTE2": {
                "deterministic_single_policy": True,
                "exploratory": True,
                "member_group": "clip_bag_switch_members",
                "member_seeds": list(H284_SEEDS),
                "clip_floor": float(CLIP_FLOOR),
                "member_strict_margins": [c["strict_margin"] for c in clip_members],
                "min_agreeing_seeds": 2,
                "q_fixed_per_seed": int(bagged.Q_FIXED),
                "aggregation_rule": (
                    "Majority (>=2 of 3) candidate-agreement ensemble over the 3 CLIP BAG_SWITCH (q=6) policies."
                ),
            },
        },
        "h284_bag_switch_members": h284_members,
        "clip_bag_switch_members": clip_members,
        "code_sha256": code_hashes,
        "source_commit": source_commit,
    }
    spec_sha = save_json(export_dir / "ensemble_spec.json", spec)
    spec["ensemble_spec_sha256"] = spec_sha

    # Verify reload bit-exactness of all ensemble arms on the 64 DL-Train check contexts
    check_run1 = score_all_h287_arms(export_dir, X_check, len(check_cids), k=K, spec=spec)
    check_run2 = score_all_h287_arms(export_dir, X_check, len(check_cids), k=K, spec=None)
    if check_run1 != check_run2:
        raise RuntimeError("64-context ensemble reload bit-exact check failed")

    reload_check_64 = {
        "n_contexts": len(check_cids),
        "context_cids_sha256": sha256_bytes(json.dumps(check_cids).encode()),
        "bit_exact": True,
        "deterministic_arms": {
            arm: {
                "changed_count": int(sum(int(j) != 0 for j in check_run1[arm])),
                "picks_sha256": sha256_bytes(json.dumps(check_run1[arm]).encode()),
                "picks": check_run1[arm],
            }
            for arm in ("ENS_MEAN", "ENS_MEAN_PROPOSER", "ENS_VOTE2", "ENS_VOTE3", "CLIP_ENS_VOTE2", "CLIP_ENS_MEAN")
        },
        "per_seed_arms": {
            arm: [
                {
                    "seed": int(H284_SEEDS[si]),
                    "changed_count": int(sum(int(j) != 0 for j in check_run1["per_seed"][arm][si])),
                    "picks_sha256": sha256_bytes(json.dumps(check_run1["per_seed"][arm][si]).encode()),
                }
                for si in range(3)
            ]
            for arm in ("BAG_SWITCH", "RERANK8_VETO", "CLIP")
        },
    }

    lock = {
        "schema": "jope.h287.fit-lock/v1",
        "wave": "H287",
        "fit_split": "DL-Train",
        "training_family_count": len(all_fams),
        "training_context_count": len(all_cids),
        "fresh_pools_read": 0,
        "dl_val_reads_before_lock": 0,
        "ensemble_spec_sha256": spec_sha,
        "reload_bit_exact_check_on_64_dl_train_contexts": reload_check_64,
        "code_sha256": code_hashes,
        "source_commit": source_commit,
    }
    lock_sha = save_json(export_dir / "fit_lock.json", lock)
    lock["fit_lock_sha256"] = lock_sha
    return {
        "spec": spec,
        "lock": lock,
        "reload_check_64": reload_check_64,
    }
