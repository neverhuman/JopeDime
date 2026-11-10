"""H302 (h302_ensvote2_trainseed_nested_cv): 4-fold outer family CV x 3 root training seeds on DL-Train."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/home/ubuntu/JopeDime")
sys.path.insert(0, str(HERE))
import analysis_h302 as a302  # noqa: E402

H287_EXPORT_COMMIT = "213b30ad52e1ce48927bba960ed609bd98ab4dfe"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SELECTED_JSON_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
PREREG_COMMIT = "2a77787c3df2a0ac9b5ae6bf5684bd862bfca7fc"

PINNED_H287_SHAS = {
    "h265_core.py": "5c17ddad3993b5cebb702d2c58882fa2e54dc15572649c663169416f6965de76",
    "topk.py": "fcd2d100c30d97dfe70fab7b981332f0c8dbffbf8bf7b7badaaef09550983d43",
    "veto.py": "bf9c06ddd97f793b43f8ddbd31488768be3567db3f485f2d9377248e9b878ec0",
    "bagged.py": "3ff0119d9b69367cce0eac5a82ba94fd7e44738ad5fe2a8f43867c2216bcac9a",
    "ensemble.py": "7def710223138e83cdbe0528444d120cd585ebcdd5676c5195f978bfcf7542b8",
    "analysis_h279.py": "763ad41d0e35fbf97dce4ca46deab5557e414d2c93ed46f729cdabfce30752d6",
    "analysis_h287.py": "166b336451f270b31bcc7ad68de72012afba2b6622903f13a793bc0992490930",
    "export_manifest.json": "874693a78d52ebeee2e2c0c38d0f028cd3887a01abec8d0c383128e0b4cf06d7",
    "fit_lock.json": "8a20c58bdb4bfcbadfa5ba927aa0577b630e3174e6dc4a7bfd579175ca08f35f",
}

POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
H276_INPUTS = Path("/home/ubuntu/jope-data/h276/inputs")
H287_EXPORT = Path("/home/ubuntu/jope-data/h287/export")
SCRATCH_DIR = Path("/home/ubuntu/jope_plan/ns/receipts/h302_scratch_models")
MAX_RSS_BYTES = 8 * 1024**3


def install_h302_audit_guard() -> None:
    """Refuse any open of fresh cohorts (v1-v6), DL-Val, DL-Test, sealed, holdout, or neural checkpoints."""

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            s = os.fsdecode(args[0]).lower()
            if any(
                x in s
                for x in (
                    "/sealed",
                    "/holdout",
                    "meta-test",
                    "meta_test",
                    "dl-test",
                    "dl_test",
                    "fresh_pools",
                    "fresh_cohort",
                    "fresh_top8",
                    "fresh-top8",
                    "fresh-build",
                    "r882-secondary-costs",
                    "r3175-gpu-costs",
                )
            ) or s.endswith(".pt"):
                raise RuntimeError(f"Forbidden path open in H302 DL-Train nested-CV wave: {s}")

    sys.addaudithook(audit_hook)


def check_rss_limit() -> int:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss_kb) * 1024
    if rss_bytes > MAX_RSS_BYTES:
        raise RuntimeError(f"Peak RSS {rss_bytes} exceeds 8 GiB cap ({MAX_RSS_BYTES})")
    return rss_bytes


def load_pinned_h287_modules(mod_dir: Path) -> tuple[Any, Any, Any, Any, Any]:
    """Extract and SHA256-verify H287 modules from commit 213b30ad52e1ce48927bba960ed609bd98ab4dfe."""
    mod_dir.mkdir(parents=True, exist_ok=True)
    for fn, exp_sha in PINNED_H287_SHAS.items():
        raw = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "show",
                f"{H287_EXPORT_COMMIT}:research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/{fn}",
            ]
        )
        got_sha = hashlib.sha256(raw).hexdigest()
        if got_sha != exp_sha:
            raise RuntimeError(f"SHA256 mismatch on pinned H287 file {fn}: {got_sha} != {exp_sha}")
        (mod_dir / fn).write_bytes(raw)

    sys.path.insert(0, str(mod_dir))
    import bagged  # type: ignore
    import ensemble as ens  # type: ignore
    import h265_core as core  # type: ignore
    import topk  # type: ignore
    import veto  # type: ignore

    return core, topk, veto, bagged, ens


def fit_and_freeze_member(
    fold_dir: Path,
    outer_fold: int,
    root_seed: int,
    member_idx: int,
    member_seed: int,
    outer_train_rows: list[dict[str, Any]],
    outer_train_packets: dict[str, dict[str, Any]],
    outer_train_fam_by_cid: dict[str, str],
    X_check_64: np.ndarray,
    core: Any,
    veto: Any,
    bagged: Any,
    ens: Any,
) -> dict[str, Any]:
    """Fit one H300/H302 BAG_SWITCH member strictly on outer_train families, serialize its 10 LightGBM models,
    and verify 64-context reload bit-exactness on outer_train before any outer_test read."""
    cids_by_fold = ens.family_folds(outer_train_rows, outer_train_fam_by_cid, member_seed, n=4)
    # Fold-0 plumbing check uses the first context of inner fold 1 as H287/H300_PREREG specify
    cids_by_fold[0] = [cids_by_fold[1][0]]

    rv = veto.run_cell_veto(
        member_seed, 0, cids_by_fold, outer_train_packets, outer_train_fam_by_cid, ens.K, ""
    )
    best_lr = float(rv["receipt"]["lr"])
    best_n = int(rv["receipt"]["n_estimators"])
    raw_margin = float(rv["receipt"]["margin"])
    strict_margin = rv["margin"]
    admitted = bool(rv["receipt"]["strict_admitted"])

    fit_folds = [1, 2, 3, 4]
    fit_cids = [c for f in fit_folds for c in cids_by_fold[f]]
    fit_fams = sorted({outer_train_fam_by_cid[c] for c in fit_cids})
    f2c: dict[str, list[str]] = {}
    for c in fit_cids:
        f2c.setdefault(outer_train_fam_by_cid[c], []).append(c)

    prefix = f"f{outer_fold}-r{root_seed}-m{member_idx}-s{member_seed}"
    r_path = fold_dir / f"{prefix}-ranker.txt"
    v_path = fold_dir / f"{prefix}-veto.txt"
    r_path.write_text(rv["model"])
    v_path.write_text(rv["gate"])
    r_info = ens.verify_reload(rv["model"], r_path, X_check_64, num_iteration=best_n)
    v_info = ens.verify_reload(rv["gate"], v_path, X_check_64, num_iteration=None)

    bag_infos = []
    for b in range(bagged.N_BAGS):
        rng = np.random.default_rng(
            core.stream_seed(member_seed, 0, "BAG_SWITCH", "outer_boot", b)
        )
        bfams = rng.choice(fit_fams, size=len(fit_fams), replace=True)
        bcids = [c for f in bfams for c in f2c[f]]
        Xb = np.concatenate([outer_train_packets[c]["X"] for c in bcids])
        yb = np.concatenate([outer_train_packets[c]["rel_k"] for c in bcids])
        mb = bagged.fit_ranker_n(
            Xb,
            yb,
            len(bcids),
            ens.K,
            best_lr,
            best_n,
            core.stream_seed(member_seed, 0, "BAG_SWITCH", "outer_bag", b),
        )
        m_str = mb.booster_.model_to_string()
        b_path = fold_dir / f"{prefix}-bag{b}.txt"
        b_path.write_text(m_str)
        b_info = ens.verify_reload(m_str, b_path, X_check_64, num_iteration=best_n)
        b_info["bag_index"] = int(b)
        bag_infos.append(b_info)

    return {
        "outer_fold": int(outer_fold),
        "root_seed": int(root_seed),
        "member_index": int(member_idx),
        "seed": int(member_seed),
        "selected_lr": best_lr,
        "selected_n_estimators": best_n,
        "raw_margin": raw_margin,
        "strict_margin": "inf" if math.isinf(strict_margin) else float(strict_margin),
        "strict_admitted": admitted,
        "inner_fit_family_gain": float(rv["receipt"]["inner_fit_family_gain"]),
        "inner_fit_ci95": rv["receipt"]["inner_fit_ci95"],
        "selected_tau": float(veto.TAU),
        "n_bags": int(bagged.N_BAGS),
        "q_fixed": int(bagged.Q_FIXED),
        "ranker": r_info,
        "veto": v_info,
        "bags": bag_infos,
    }


def run_h302() -> dict[str, Any]:
    t0 = time.time()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H302 requires CPU xbabe2 at nice 19")

    install_h302_audit_guard()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["H265_PAUSE_FILE"] = str(SCRATCH_DIR / "h302.pause.json")

    # 1. Verify Step 1 regression receipt
    reg_path = HERE / "regression_receipt.json"
    assert reg_path.exists(), "Step 1 regression_receipt.json must exist before running H302"
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["bit_exact_all_deterministic_and_per_seed_arms"] is True

    mod_dir = SCRATCH_DIR / "pinned_h287_mods"
    core, topk, veto, bagged, ens = load_pinned_h287_modules(mod_dir)
    core.INPUT_POOLS_DIR = POOLS

    # Verify H300 member seeds
    for rseed, exp_tuple in a302.EXPECTED_MEMBER_SEEDS.items():
        got_tuple = tuple(core.stream_seed(rseed, "H300_ENS_VOTE2_MEMBER", m) for m in range(3))
        assert got_tuple == exp_tuple, f"Member seed mismatch for {rseed}: {got_tuple} != {exp_tuple}"

    # 2. Load DL-Train selected.json and top-8 packets (r882 proposal features, unclipped targets)
    sel_bytes = (POOLS / "selected.json").read_bytes()
    assert hashlib.sha256(sel_bytes).hexdigest() == SELECTED_JSON_SHA256
    selected = json.loads(sel_bytes)
    train_rows = selected["train"]
    assert len(train_rows) == 1219
    all_fams = sorted({r["family_id"] for r in train_rows})
    assert len(all_fams) == 205
    if any(r.get("has_meta_test") is True or r["task"] != "regression" for r in train_rows):
        raise RuntimeError("Protected context detected in train_rows")

    all_cids = [r["context_sha256"] for r in train_rows]
    all_cids_fams = [r["family_id"] for r in train_rows]
    cid_to_idx = {c: i for i, c in enumerate(all_cids)}
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train_rows}

    costs = {}
    with np.load(H276_INPUTS / "r882-train-costs.npz") as zt:
        for cid in all_cids:
            costs[cid] = zt[cid].astype(np.float32)

    ids = set(all_cids)
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        cids, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        base, proj = zp["base_prior_cost"], zp["proj16"]
        for i, cid in enumerate(cids):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[cid] = base[lo:hi].astype(np.float32).copy()
                proj16[cid] = proj[lo:hi].astype(np.float32).copy()
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as zc:
        for i, cid in enumerate(str(c) for c in zc["cids"]):
            if cid in ids:
                ctxf[cid] = {
                    "lm48_z": zc["lm48_z"][i].astype(np.float32).copy(),
                    "dope629": zc["dope629"][i].astype(np.float32).copy(),
                }

    P8 = topk.load_packets(train_rows, costs, aprior, proj16, ctxf, ens.K)
    print(f"[H302] Loaded 1,219 DL-Train packets across 205 families ({time.time() - t0:.1f}s)", flush=True)

    # 3. Pre-declared 4-fold outer family split: SHA256("H302-outer-v1" NUL family) last byte mod 4
    fam_to_fold = {f: a302.outer_fold_of_family(f) for f in all_fams}

    # Buffers for out-of-fold predictions across all 1,219 contexts
    n_ctx = len(all_cids)
    oof_picks_ens_vote2 = np.full((3, n_ctx), -1, dtype=np.int32)
    oof_picks_ens_mean = np.full((3, n_ctx), -1, dtype=np.int32)
    oof_picks_ens_vote3 = np.full((3, n_ctx), -1, dtype=np.int32)
    oof_picks_members_bag = np.full((3, 3, n_ctx), -1, dtype=np.int32)
    oof_picks_members_veto = np.full((3, 3, n_ctx), -1, dtype=np.int32)

    outer_fold_receipts = []

    for k in range(4):
        fk_t0 = time.time()
        fold_dir = SCRATCH_DIR / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        outer_train_rows = [r for r in train_rows if fam_to_fold[r["family_id"]] != k]
        outer_test_rows = [r for r in train_rows if fam_to_fold[r["family_id"]] == k]
        outer_train_cids = [r["context_sha256"] for r in outer_train_rows]
        outer_test_cids = [r["context_sha256"] for r in outer_test_rows]
        outer_train_fams = sorted({r["family_id"] for r in outer_train_rows})
        outer_test_fams = sorted({r["family_id"] for r in outer_test_rows})

        assert set(outer_train_fams).isdisjoint(set(outer_test_fams))
        assert set(outer_train_cids).isdisjoint(set(outer_test_cids))

        # Build isolated outer_train dictionaries (outer_test contexts absent)
        outer_train_packets = {c: P8[c] for c in outer_train_cids}
        outer_train_fam_by_cid = {c: fam_by_cid[c] for c in outer_train_cids}

        check_cids_k = sorted(outer_train_cids)[:64]
        X_check_k = np.concatenate([outer_train_packets[c]["X"] for c in check_cids_k])

        lock_path = fold_dir / f"fold_{k}_fit_lock.json"
        if lock_path.exists() and "--refit" not in sys.argv:
            saved_lock = json.loads(lock_path.read_text())
            fold_lock_sha = ens.sha256_file(lock_path)
            members_by_root = {
                rseed: saved_lock["members_by_root_seed"][str(rseed)]
                for rseed in a302.ROOT_TRAINING_SEEDS
            }
            reload_check_64_by_seed = {}
            for rseed in a302.ROOT_TRAINING_SEEDS:
                m_outs_1 = [
                    ens.score_single_member_from_disk(fold_dir, c, X_check_k, len(check_cids_k), ens.K)
                    for c in members_by_root[rseed]
                ]
                agg_1 = ens.aggregate_ensemble_picks(m_outs_1, len(check_cids_k), q_fixed=bagged.Q_FIXED, tau=veto.TAU)
                rec = {
                    "ens_vote2_changed_count_64": int(sum(int(j) != 0 for j in agg_1["ENS_VOTE2"])),
                    "ens_vote2_picks_sha256_64": ens.sha256_bytes(json.dumps(agg_1["ENS_VOTE2"]).encode()),
                    "reload_bit_exact_on_64_outer_train_contexts": True,
                }
                if rec != saved_lock["reload_check_on_64_outer_train_contexts"][str(rseed)]:
                    raise RuntimeError(f"Saved lock 64-context reload mismatch on fold {k} root_seed {rseed}")
                reload_check_64_by_seed[str(rseed)] = rec
        else:
            members_by_root: dict[int, list[dict[str, Any]]] = {rseed: [] for rseed in a302.ROOT_TRAINING_SEEDS}
            for s_idx, rseed in enumerate(a302.ROOT_TRAINING_SEEDS):
                for m_idx in range(3):
                    m_seed = int(a302.EXPECTED_MEMBER_SEEDS[rseed][m_idx])
                    m_info = fit_and_freeze_member(
                        fold_dir=fold_dir,
                        outer_fold=k,
                        root_seed=rseed,
                        member_idx=m_idx,
                        member_seed=m_seed,
                        outer_train_rows=outer_train_rows,
                        outer_train_packets=outer_train_packets,
                        outer_train_fam_by_cid=outer_train_fam_by_cid,
                        X_check_64=X_check_k,
                        core=core,
                        veto=veto,
                        bagged=bagged,
                        ens=ens,
                    )
                    members_by_root[rseed].append(m_info)
                    print(
                        f"[H302] Fold {k} | root {rseed} | member {m_idx} (seed={m_seed}) -> "
                        f"lr={m_info['selected_lr']}, n_est={m_info['selected_n_estimators']}, "
                        f"strict_margin={m_info['strict_margin']}, admitted={m_info['strict_admitted']}",
                        flush=True,
                    )

            # Verify 64-context outer_train reload bit-exactness across all 3 root seeds BEFORE locking or touching outer_test
            reload_check_64_by_seed = {}
            for rseed in a302.ROOT_TRAINING_SEEDS:
                m_outs_1 = [
                    ens.score_single_member_from_disk(fold_dir, c, X_check_k, len(check_cids_k), ens.K)
                    for c in members_by_root[rseed]
                ]
                agg_1 = ens.aggregate_ensemble_picks(m_outs_1, len(check_cids_k), q_fixed=bagged.Q_FIXED, tau=veto.TAU)
                m_outs_2 = [
                    ens.score_single_member_from_disk(fold_dir, c, X_check_k, len(check_cids_k), ens.K)
                    for c in members_by_root[rseed]
                ]
                agg_2 = ens.aggregate_ensemble_picks(m_outs_2, len(check_cids_k), q_fixed=bagged.Q_FIXED, tau=veto.TAU)
                if agg_1 != agg_2:
                    raise RuntimeError(f"64-context outer_train reload check failed on fold {k} root_seed {rseed}")
                reload_check_64_by_seed[str(rseed)] = {
                    "ens_vote2_changed_count_64": int(sum(int(j) != 0 for j in agg_1["ENS_VOTE2"])),
                    "ens_vote2_picks_sha256_64": ens.sha256_bytes(json.dumps(agg_1["ENS_VOTE2"]).encode()),
                    "reload_bit_exact_on_64_outer_train_contexts": True,
                }

            # Freeze and fsync fold_{k}_fit_lock.json BEFORE scoring outer_test
            fold_lock = {
                "schema": "jope.h302.outer-fold-fit-lock/v1",
                "wave": "H302",
                "outer_fold": int(k),
                "n_outer_train_families": len(outer_train_fams),
                "n_outer_train_contexts": len(outer_train_cids),
                "n_outer_test_families": len(outer_test_fams),
                "n_outer_test_contexts": len(outer_test_cids),
                "outer_test_rows_read_before_lock": 0,
                "fresh_pools_read": 0,
                "members_by_root_seed": {str(rseed): members_by_root[rseed] for rseed in a302.ROOT_TRAINING_SEEDS},
                "reload_check_on_64_outer_train_contexts": reload_check_64_by_seed,
            }
            fold_lock_sha = ens.save_json(lock_path, fold_lock)
            fold_lock["fold_fit_lock_sha256"] = fold_lock_sha

        # Now score outer_test_cids ONCE from frozen models on disk
        X_test_k = np.concatenate([P8[c]["X"] for c in outer_test_cids])
        test_indices = [cid_to_idx[c] for c in outer_test_cids]

        for s_idx, rseed in enumerate(a302.ROOT_TRAINING_SEEDS):
            m_test_outs = [
                ens.score_single_member_from_disk(fold_dir, c, X_test_k, len(outer_test_cids), ens.K)
                for c in members_by_root[rseed]
            ]
            agg_test = ens.aggregate_ensemble_picks(
                m_test_outs, len(outer_test_cids), q_fixed=bagged.Q_FIXED, tau=veto.TAU
            )
            for local_i, global_i in enumerate(test_indices):
                oof_picks_ens_vote2[s_idx, global_i] = int(agg_test["ENS_VOTE2"][local_i])
                oof_picks_ens_mean[s_idx, global_i] = int(agg_test["ENS_MEAN"][local_i])
                oof_picks_ens_vote3[s_idx, global_i] = int(agg_test["ENS_VOTE3"][local_i])
                for m_idx in range(3):
                    oof_picks_members_bag[s_idx, m_idx, global_i] = int(m_test_outs[m_idx]["picks_bag"][local_i])
                    oof_picks_members_veto[s_idx, m_idx, global_i] = int(m_test_outs[m_idx]["j_ref"][local_i])

        outer_fold_receipts.append(
            {
                "outer_fold": int(k),
                "elapsed_sec": round(time.time() - fk_t0, 2),
                "n_outer_train_families": len(outer_train_fams),
                "n_outer_train_contexts": len(outer_train_cids),
                "n_outer_test_families": len(outer_test_fams),
                "n_outer_test_contexts": len(outer_test_cids),
                "fold_fit_lock_sha256": fold_lock_sha,
                "reload_check_on_64_outer_train_contexts": reload_check_64_by_seed,
                "member_configs_by_root_seed": {
                    str(rseed): [
                        {
                            "member_index": c["member_index"],
                            "seed": c["seed"],
                            "selected_lr": c["selected_lr"],
                            "selected_n_estimators": c["selected_n_estimators"],
                            "raw_margin": c["raw_margin"],
                            "strict_margin": c["strict_margin"],
                            "strict_admitted": c["strict_admitted"],
                            "inner_fit_family_gain": c["inner_fit_family_gain"],
                            "inner_fit_ci95": c["inner_fit_ci95"],
                            "ranker_sha256": c["ranker"]["sha256"],
                            "veto_sha256": c["veto"]["sha256"],
                        }
                        for c in members_by_root[rseed]
                    ]
                    for rseed in a302.ROOT_TRAINING_SEEDS
                },
            }
        )
        check_rss_limit()
        print(f"[H302] Completed outer fold {k} in {time.time() - fk_t0:.1f}s", flush=True)

    # Verify every context was scored exactly once out-of-fold
    assert not np.any(oof_picks_ens_vote2 < 0)
    assert not np.any(oof_picks_ens_mean < 0)
    assert not np.any(oof_picks_ens_vote3 < 0)

    # 4. Evaluate primary ENS_VOTE2 and secondary/descriptive arms on true unclipped per-context utility & validity
    slot0_u = np.array([float(P8[c]["utility_k"][0]) for c in all_cids], dtype=np.float64)
    slot0_inv = np.array(
        [(not bool(P8[c]["gate_k"][0])) or float(P8[c]["utility_k"][0]) <= -2.0 + 1e-9 for c in all_cids],
        dtype=bool,
    )

    def eval_arm_picks(picks_3xN: np.ndarray) -> dict[str, Any]:
        u_mat = np.zeros((3, n_ctx, 2), dtype=np.float64)
        inv_mat = np.zeros((3, n_ctx, 2), dtype=bool)
        for s in range(3):
            u_mat[s, :, 0] = slot0_u
            inv_mat[s, :, 0] = slot0_inv
            for i, c in enumerate(all_cids):
                j = int(picks_3xN[s, i])
                u_mat[s, i, 1] = float(P8[c]["utility_k"][j])
                inv_mat[s, i, 1] = (not bool(P8[c]["gate_k"][j])) or float(P8[c]["utility_k"][j]) <= -2.0 + 1e-9
        s_dict = a302.summarize_three_seeds(
            u_mat, inv_mat, all_cids_fams, picks_3xN.tolist(), boot_seed=a302.BOOT_SEED
        )
        c_dict = a302.evaluate_h302_criteria(s_dict, inv_mat, len(all_fams))
        s_pub = {k: v for k, v in s_dict.items() if not k.startswith("_")}
        return {"summary": s_pub, "criteria": c_dict}

    primary_ens_vote2 = eval_arm_picks(oof_picks_ens_vote2)
    agreement_and_spread = a302.analyze_policy_pick_agreement_and_seed_spread(
        oof_picks_ens_vote2.tolist(), primary_ens_vote2["summary"]
    )

    secondary_arms = {
        "ENS_MEAN": eval_arm_picks(oof_picks_ens_mean),
        "ENS_VOTE3": eval_arm_picks(oof_picks_ens_vote3),
        "MEMBER_0_BAG_SWITCH": eval_arm_picks(oof_picks_members_bag[:, 0, :]),
        "MEMBER_1_BAG_SWITCH": eval_arm_picks(oof_picks_members_bag[:, 1, :]),
        "MEMBER_2_BAG_SWITCH": eval_arm_picks(oof_picks_members_bag[:, 2, :]),
        "MEMBER_0_RERANK8_VETO": eval_arm_picks(oof_picks_members_veto[:, 0, :]),
        "MEMBER_1_RERANK8_VETO": eval_arm_picks(oof_picks_members_veto[:, 1, :]),
        "MEMBER_2_RERANK8_VETO": eval_arm_picks(oof_picks_members_veto[:, 2, :]),
    }

    # Evaluate H294-style random-valid and random-no-label nulls
    gate_k_by_ctx = [P8[c]["gate_k"] for c in all_cids]
    null_valid_picks = np.zeros((3, n_ctx), dtype=np.int32)
    null_nolabel_picks = np.zeros((3, n_ctx), dtype=np.int32)
    for s, vseed in enumerate(a302.VALID_NULL_SEEDS):
        rng = np.random.default_rng(vseed)
        for i, g in enumerate(gate_k_by_ctx):
            valid_slots = [j for j in range(1, len(g)) if bool(g[j])]
            null_valid_picks[s, i] = int(rng.choice(valid_slots)) if valid_slots else 0
    for s, nseed in enumerate(a302.NOLABEL_NULL_SEEDS):
        null_nolabel_picks[s] = np.random.default_rng(nseed).integers(0, ens.K, size=n_ctx)

    nulls = {
        "NULL_RANDOM_VALID_SLOT": eval_arm_picks(null_valid_picks),
        "NULL_RANDOM_SLOT_NO_LABELS": eval_arm_picks(null_nolabel_picks),
    }

    peak_rss = check_rss_limit()
    verdict = primary_ens_vote2["criteria"]["verdict"]

    full_payload = {
        "wave_id": "H302",
        "slug": "h302_ensvote2_trainseed_nested_cv",
        "owner": "rd_argon",
        "status": "COMPLETE",
        "verdict": verdict,
        "prereg_commit": PREREG_COMMIT,
        "h287_export_commit": H287_EXPORT_COMMIT,
        "selected_json_sha256": SELECTED_JSON_SHA256,
        "dope_sha256": DOPE_SHA256,
        "dope_untouched": True,
        "optimizer_updates": 0,
        "gpu_used": False,
        "fresh_pools_read": 0,
        "outer_test_rows_read_before_fold_lock": 0,
        "elapsed_sec": round(time.time() - t0, 2),
        "root_training_seeds": list(a302.ROOT_TRAINING_SEEDS),
        "member_seeds_by_root": {str(k): list(v) for k, v in a302.EXPECTED_MEMBER_SEEDS.items()},
        "shared_bootstrap_seed": a302.BOOT_SEED,
        "regression_step1": reg,
        "primary_ENS_VOTE2": primary_ens_vote2,
        "policy_pick_agreement_and_seed_spread": agreement_and_spread,
        "secondary_arms": secondary_arms,
        "nulls": nulls,
        "outer_fold_receipts": outer_fold_receipts,
        "resources": {
            "host": "xbabe2",
            "nice": 19,
            "cores": list(range(8)),
            "peak_rss_bytes": peak_rss,
            "max_rss_limit_bytes": MAX_RSS_BYTES,
        },
    }

    # Save full payload into SCRATCH_DIR for S3 archive
    (SCRATCH_DIR / "metrics_full_h302.json").write_text(json.dumps(full_payload, indent=2) + "\n")

    # Build compact metrics.json (< 50 KB)
    def compact_primary_arm(arm_obj: dict[str, Any]) -> dict[str, Any]:
        s2 = {k: v for k, v in arm_obj["summary"].items() if k != "harmed_families_by_seed"}
        return {"summary": s2, "criteria": arm_obj["criteria"]}

    def compact_secondary_arm(arm_obj: dict[str, Any]) -> dict[str, Any]:
        s = arm_obj["summary"]
        c = arm_obj["criteria"]
        return {
            "summary": {
                "per_seed_family_mean_gain": s["per_seed_family_mean_gain"],
                "median_gain_across_seeds": s["median_gain_across_seeds"],
                "pooled_ci95": s["pooled_ci95"],
                "per_seed_lcb_one_sided_95": s["per_seed_lcb_one_sided"]["0.05"],
                "secondary_family_median_gain": s["secondary_family_median_gain"]["per_seed_family_median_gain"],
                "harmed_share_by_seed": s["harmed_share_by_seed"],
                "invalid_by_seed": s["invalid_by_seed"],
                "changed_picks_by_seed": s["changed_picks_by_seed"],
                "elo_by_seed": s["elo_by_seed"],
                "median_elo_across_seeds": s["median_elo_across_seeds"],
            },
            "criteria": {
                "passes_h302_primary": c["passes_h302_primary"],
                "verdict": c["verdict"],
                "condition_1_median_gain_gt_0_001": c["condition_1_median_gain_gt_0_001"],
                "condition_2_lcb95_gt_0_on_3of3_seeds": c["condition_2_lcb95_gt_0_on_3of3_seeds"],
                "condition_3_harmed_share_le_35pct_each_seed": c["condition_3_harmed_share_le_35pct_each_seed"],
                "condition_4_invalid_noninferior_each_seed": c["condition_4_invalid_noninferior_each_seed"],
            },
        }

    compact_payload = dict(full_payload)
    compact_payload["regression_step1"] = {
        k: v for k, v in reg.items() if k != "model_checks"
    }
    compact_payload["primary_ENS_VOTE2"] = compact_primary_arm(primary_ens_vote2)
    compact_payload["secondary_arms"] = {
        k: compact_secondary_arm(v) for k, v in secondary_arms.items()
    }
    compact_payload["nulls"] = {
        k: compact_secondary_arm(v) for k, v in nulls.items()
    }
    compact_payload["outer_fold_receipts"] = [
        {
            "outer_fold": r["outer_fold"],
            "n_outer_train_families": r["n_outer_train_families"],
            "n_outer_train_contexts": r["n_outer_train_contexts"],
            "n_outer_test_families": r["n_outer_test_families"],
            "n_outer_test_contexts": r["n_outer_test_contexts"],
            "fold_fit_lock_sha256": r["fold_fit_lock_sha256"],
            "reload_check_on_64_outer_train_contexts": r["reload_check_on_64_outer_train_contexts"],
            "member_configs_summary_by_root_seed": {
                rseed: [
                    {
                        "seed": m["seed"],
                        "lr": m["selected_lr"],
                        "n_est": m["selected_n_estimators"],
                        "strict_margin": m["strict_margin"],
                        "admitted": m["strict_admitted"],
                    }
                    for m in mlist
                ]
                for rseed, mlist in r["member_configs_by_root_seed"].items()
            },
        }
        for r in outer_fold_receipts
    ]

    out_path = HERE / "metrics.json"
    out_path.write_text(json.dumps(compact_payload, indent=2, allow_nan=False) + "\n")
    assert out_path.stat().st_size < 50 * 1024, f"metrics.json size {out_path.stat().st_size} >= 50KB"
    return compact_payload


if __name__ == "__main__":
    res = run_h302()
    print(
        json.dumps(
            {
                "verdict": res["verdict"],
                "elapsed_sec": res["elapsed_sec"],
                "primary_ENS_VOTE2_summary": {
                    k: v
                    for k, v in res["primary_ENS_VOTE2"]["summary"].items()
                    if k != "harmed_families_by_seed"
                },
                "primary_ENS_VOTE2_criteria": res["primary_ENS_VOTE2"]["criteria"],
                "policy_pick_agreement_and_seed_spread": res["policy_pick_agreement_and_seed_spread"],
            },
            indent=2,
        )
    )
