#!/usr/bin/env python3
"""Wave H458 Track G: Three-Tier (`Tier A -> Tier B -> Tier C`) Winner's-Curse-Free Multi-Generation
`Train Hyper-CV` Neural-GP Evolution Engine + Shared `K=5` Canonical Feature-Stratum Helper
(`ultra_gp_cycle_h458.py`).

Strict zero-leakage & Track S / Track D interoperability contract:
  1. Zero `y_audit` Bytes Read Anywhere (`assert "ya" not in zb.files`):
     - Inference and `10 -> 50 -> 100 -> 250 -> 500 -> 1000` generation GP evolution use STRICTLY `SEARCH` `(Xs, ys)`
       and unlabeled `Xa` query features (`0` `ya` bytes ever loaded).
  2. Fixes `cl_h458` Pitfall `5a`:
     - Defaults all 4 output/warm/slot/colblock roots into `/home/ubuntu/jope-data/h458-flywheel/ultra_gp_cycle2_d91/`
       and refuses overwriting `/home/ubuntu/jope-data/h452-flywheel/ultra_gp_d91`.
     - Hard-asserts `qhat.ok == True` via `PretrainedQHatV4Prior(..., require_ok=True)`.
  3. Three-Tier Disjoint `Train Hyper-CV` (`M = 1` Untouched Stack OOF):
     - Tier A (`Search`, `rep=0,1`): `QHatV4` (`overfit_gap` penalized) 5-island evolution anchored on the strongest
       Stage 1-3 `104M` / `L+M` valley-soup `2xCV-7` OOF residuals (`Island 4 resid104m`).
     - Tier B (`Vaulted Horizon Ratchet`, unseen `rep=77`): Evaluates only the top-`3` `u_cons` finalists at each
       horizon `g in snapshots` on an unseen 7-fold partition (`rep=77`), advancing the horizon champion ONLY when
       `u_vault(g) >= u_vault(g_prev) + ratchet_margin` (monotonically non-decreasing horizon ratchet!).
     - Tier C (`Untouched Stack OOF`, unseen `rep=101,102`, `M = 1`): Cross-fits ONLY the single locked Tier-B winning
       AST (`M = 1`) on two completely untouched 7-fold partitions (`rep=101, 102`) + full-`Xs` refit, eliminating
       the `1000`-generation `M ~ 3000` selection optimism (`+0.04..+0.09 u`) that previously fooled L2 stackers!
  4. Track S Interface (i) `_load_col_block` + Sidecar JSON & Shared `K=5` Feature-Stratum Spec (`h458-k5-strata-v1`).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
for _cand in reversed((
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
)):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)
if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))

import h335_core as K  # noqa: E402
from bagged_neural_seed_h452 import BaggedNeuralEncoderH452  # noqa: E402
from hyper_cv_test_gp_harvester_h458 import compute_fold_utility_spread  # noqa: E402
from train_qhat_v4_generalizability_h458 import PretrainedQHatV4Prior  # noqa: E402
from ultra_gp_evolve_1000g_h453 import (  # noqa: E402
    TwoTierSlotWorkerRunner,
    _parse_indices,
    evolve_single_context_1000g,
)
from ultra_gp_evolve_h452 import (  # noqa: E402
    SUPERSET_BIN,
    SUPERSET_SHA,
    _sha256_file,
    fold_log_utility,
    make_stratified_kfold,
    pack_jtt1_bytes,
)

C, G = K.C, K.G

K5_STRATA_KEY = b"h458-k5-strata-v1"
CANON_NAN_U64 = np.uint64(0x7FF8000000000000)


def canonicalize_feature_matrix_f64(X: np.ndarray) -> np.ndarray:
    """Canonicalizes feature matrix `X` (`<f8`, C-order, `-0.0 -> +0.0`, every `NaN -> 0x7ff8000000000000`, `0` labels)."""
    Xc = np.ascontiguousarray(X, dtype="<f8").copy()
    if Xc.ndim == 1:
        Xc = Xc[None, :]
    Xc[Xc == 0.0] = 0.0
    u = Xc.view("<u8")
    u[np.isnan(Xc)] = CANON_NAN_U64
    return Xc


def make_k5_feature_byte_strata(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Shared Track D / Track S / Track G `K=5` deterministic feature-byte hash stratum assignment (`0` label bytes!).

    Exact spec (`cl_h458` 21:45Z + `cl_mtc` S6b + `agy_apex_w10` C-AGY-1):
      - `canon(X_row)`: `<f8`, C-order, features only; `-0.0 -> +0.0`; every `NaN -> 0x7ff8000000000000`.
      - `stratum = int.from_bytes(blake2b(canon(X_row), digest_size=8, key=b"h458-k5-strata-v1").digest(), "little") % k`
      - Guarantees identical-`X`/divergent-`y` twins land in the SAME fold AND `AUDIT` rows route without reading labels.
    """
    k_eff = max(2, int(k))
    Xc = canonicalize_feature_matrix_f64(X)
    u = Xc.view("<u8")
    out = np.empty(len(Xc), dtype=np.int64)
    for i in range(len(Xc)):
        d = hashlib.blake2b(u[i].tobytes(), digest_size=8, key=K5_STRATA_KEY).digest()
        out[i] = int(int.from_bytes(d, "little") % k_eff)
    return out


def resolve_best_104m_anchor(
    cid: str,
    n_s: int,
    n_a: int,
    anchor_dirs: Sequence[Path],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Resolves the strongest available `104M` / `L+M` valley-soup `(q, bag, ref, source_tag)` for `Island 4`."""
    preferred_keys = (
        ("q_icl_e8_all_cv7", "bag_icl_e8_all_cv7", "ref_icl_e8_all_cv7"),
        ("q_icl_e8_gp_resid_cv7", "bag_icl_e8_gp_resid_cv7", "ref_icl_e8_gp_resid_cv7"),
        ("q_traj_opt_e8", "bag_traj_opt_e8", "ref_traj_opt_e8"),
    )
    for d in anchor_dirs:
        p = d / f"{cid}.npz"
        if not p.is_file():
            continue
        try:
            with np.load(p, allow_pickle=False) as z:
                assert "ya" not in z.files
                if "ok" in z.files and not bool(z["ok"]):
                    continue
                for qk, bk, rk in preferred_keys:
                    if qk in z.files and bk in z.files and rk in z.files:
                        q = np.asarray(z[qk], dtype=np.float64)
                        b = np.asarray(z[bk], dtype=np.float64)
                        r = np.asarray(z[rk], dtype=np.float64)
                        if (
                            len(q) == n_s
                            and len(b) == n_a
                            and len(r) == n_a
                            and np.all(np.isfinite(q))
                            and np.all(np.isfinite(b))
                        ):
                            r_clean = r if np.all(np.isfinite(r)) else b.copy()
                            return q, b, r_clean, f"{d.name}:{qk}"
        except (OSError, ValueError, KeyError):
            continue
    return None


def crossfit_single_ast_on_kfold(
    dsl: str,
    target_mode: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab_k: np.ndarray,
    anchor_104m: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    runner: TwoTierSlotWorkerRunner,
    include_full_refit: bool = False,
    timeout_s: float = 7.5,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float, float]]:
    """Cross-fits a single AST `(dsl, target_mode)` on `lab_k` (`0` `ya` bytes!) returning `(oof_s, bag_a, ref_a, u_k, spread)`."""
    n_s, width = Xs.shape
    n_a = Xa.shape[0]
    X_joint = np.vstack([Xs, Xa])
    y_dummy_a = np.linspace(-1.0, 1.0, n_a, dtype=np.float64)
    ones_a = np.ones(n_a, dtype=np.uint8)

    if target_mode == "resid104m":
        if anchor_104m is None:
            return None
        q_anc, bag_anc, ref_anc = anchor_104m
        y_fit = ys - q_anc
    else:
        q_anc = bag_anc = ref_anc = None
        y_fit = ys

    cand = G.Candidate.from_dsl(dsl, "tier_eval")
    if "(gpr" in cand.dsl:
        cand = G.cap_gpr(cand, width)

    uniq_f = sorted(int(x) for x in np.unique(lab_k))
    eff_k = len(uniq_f)
    if eff_k < 2:
        return None

    tasks: List[Tuple[bytes, G.Candidate]] = []
    va_masks: List[np.ndarray] = []
    for fk in uniq_f:
        va_m = lab_k == fk
        va_masks.append(va_m)
        fu8 = np.concatenate([va_m.astype(np.uint8), ones_a])
        tasks.append((pack_jtt1_bytes(X_joint, np.concatenate([y_fit, y_dummy_a]), fu8), cand))
    if include_full_refit:
        fu8_pa = np.concatenate([np.zeros(n_s, dtype=np.uint8), ones_a])
        tasks.append((pack_jtt1_bytes(X_joint, np.concatenate([y_fit, y_dummy_a]), fu8_pa), cand))

    gots = runner.map_s1(tasks, timeout_s=timeout_s)
    oof_s = np.empty(n_s, dtype=np.float64)
    bag_a = np.zeros(n_a, dtype=np.float64)

    for va_m, (got, _) in zip(va_masks, gots[:eff_k]):
        n_va = int(np.sum(va_m))
        if got is None or got[1] is None or len(got[1]) != (n_va + n_a):
            return None
        preds = np.asarray(got[1], dtype=np.float64)
        if not np.all(np.isfinite(preds)):
            return None
        oof_s[va_m] = preds[:n_va]
        bag_a += preds[n_va:] / float(eff_k)

    if target_mode == "resid104m" and q_anc is not None and bag_anc is not None:
        oof_s = q_anc + oof_s
        bag_a = bag_anc + bag_a

    y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
    span = max(y_hi - y_lo, float(np.std(ys)), 1e-12)
    lo_g, hi_g = y_lo - 1.5 * span, y_hi + 1.5 * span
    oof_s = np.clip(oof_s, lo_g, hi_g)
    bag_a = np.clip(bag_a, lo_g, hi_g)

    ref_a = None
    if include_full_refit:
        got_pa, _ = gots[eff_k]
        if got_pa is not None and got_pa[1] is not None and len(got_pa[1]) == n_a:
            pa_raw = np.asarray(got_pa[1], dtype=np.float64)
            if np.all(np.isfinite(pa_raw)):
                if target_mode == "resid104m" and ref_anc is not None:
                    pa_raw = ref_anc + pa_raw
                ref_a = np.clip(pa_raw, lo_g, hi_g)
        if ref_a is None:
            ref_a = bag_a.copy()

    u_k = float(fold_log_utility(oof_s, ys, lab_k))
    spread = compute_fold_utility_spread(oof_s, ys, lab_k)
    if not math.isfinite(u_k):
        return None
    return oof_s, bag_a, ref_a, u_k, spread


def run_three_tier_hyper_cv_for_context(
    cid: str,
    family_id: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    lab5: np.ndarray,
    anchor_pack: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    warm_root: Optional[Path],
    out_root: Path,
    colblock_root: Path,
    slot_dir: Path,
    encoder: BaggedNeuralEncoderH452,
    qhat: PretrainedQHatV4Prior,
    n_pop: int,
    start_gen: int,
    n_gens: int,
    snapshots: Tuple[int, ...],
    eval_per_gen: int,
    screen_max_rows: int,
    timeout_cv2: float,
    timeout_crossfit: float,
    stagnation_gens: int,
    ratchet_margin: float,
    seed: int,
    inner_slots: int,
    rayon_threads: int,
) -> Dict:
    """Runs Tier A (`evolve_single_context_1000g`) -> Tier B (`rep=77` Vault Ratchet) -> Tier C (`rep=101,102`, `M=1`)."""
    assert qhat.ok, "PretrainedQHatV4Prior must be loaded (qhat.ok == True)!"
    t0 = time.time()
    n_s, width = Xs.shape
    anc_tup = anchor_pack[:3] if anchor_pack is not None else None
    anc_src = anchor_pack[3] if anchor_pack is not None else "none"

    # Tier A: Multi-generation `QHatV4`-guided 5-island evolution on `rep=0,1`
    tier_a_res = evolve_single_context_1000g(
        cid=cid,
        family_id=family_id,
        Xs=Xs,
        Xa=Xa,
        ys=ys,
        lab5=lab5,
        anchor_104m=anc_tup,
        warm_root=warm_root,
        out_root=out_root,
        slot_dir=slot_dir,
        encoder=encoder,
        qhat=qhat,
        n_pop=n_pop,
        start_gen=start_gen,
        n_gens=n_gens,
        snapshots=snapshots,
        top_export=6,
        eval_per_gen=eval_per_gen,
        screen_max_rows=screen_max_rows,
        timeout_cv2=timeout_cv2,
        timeout_crossfit=timeout_crossfit,
        stagnation_gens=stagnation_gens,
        seed=seed,
        inner_slots=inner_slots,
        rayon_threads=rayon_threads,
    )

    # Disjoint `Train Hyper-CV` fold partitions (`0` `ya` bytes!):
    #   - Tier B Vault: `rep=77` (unseen by Tier A search!)
    #   - Tier C Stack OOF: `rep=101` and `rep=102` (unseen by BOTH Tier A and Tier B; evaluates ONLY M=1 locked AST!)
    lab7_vault = make_stratified_kfold(cid, ys, k=7, rep=77)
    lab7_stack0 = make_stratified_kfold(cid, ys, k=7, rep=101)
    lab7_stack1 = make_stratified_kfold(cid, ys, k=7, rep=102)

    # Encode real 44-dim `DOPE+LM12` `ctx_z` on `(Xs, ys)` for `QHatV4` overfit-gap penalty
    enc_view = encoder.encode_bag(Xs, ys, use_full_dope=True)
    ctx_z = qhat.encode_ctx_z(enc_view)

    runner = TwoTierSlotWorkerRunner(slot_dir / "tier_bc", n_slots=max(1, inner_slots), rayon_threads=rayon_threads)
    locked_champ: Optional[Dict] = None
    horizon_ledger: List[Dict] = []

    try:
        for g_snap in sorted(snapshots):
            snap_p = out_root / f"g{g_snap}" / f"{cid}.npz"
            if not snap_p.is_file():
                continue
            with np.load(snap_p, allow_pickle=False) as zs:
                assert "ya" not in zs.files, "Audit labels forbidden in snapshot!"
                dsls_7 = [str(x) for x in zs["dsls"].tolist()]
                tmodes_7 = [str(x) for x in zs["target_modes"].tolist()] if "target_modes" in zs.files else ["raw"] * len(dsls_7)
                tm_map = {dsls_7[j]: (tmodes_7[j] if j < len(tmodes_7) else "raw") for j in range(len(dsls_7))}
                if "dsls_cons" in zs.files and len(zs["dsls_cons"]) > 0:
                    cands_b = [(str(d), tm_map.get(str(d), "raw")) for d in zs["dsls_cons"].tolist()[:3]]
                else:
                    cands_b = [(dsls_7[j], tm_map[dsls_7[j]]) for j in range(min(3, len(dsls_7)))]

            best_b_for_gen: Optional[Dict] = None
            for dsl_b, tm_b in cands_b:
                got_b = crossfit_single_ast_on_kfold(
                    dsl_b,
                    tm_b,
                    Xs,
                    Xa,
                    ys,
                    lab7_vault,
                    anc_tup,
                    runner,
                    include_full_refit=False,
                    timeout_s=timeout_crossfit,
                )
                if got_b is None:
                    continue
                _oof_b, _bag_b, _ref_b, u_vault, spread_b = got_b
                _qp, _sp, gap_p = qhat.predict_components(ctx_z, [dsl_b])
                u_vault_adj = u_vault - 0.15 * float(gap_p[0]) - 0.08 * spread_b
                if best_b_for_gen is None or u_vault_adj > best_b_for_gen["u_vault_adj"]:
                    best_b_for_gen = {
                        "dsl": dsl_b,
                        "target_mode": tm_b,
                        "u_vault": u_vault,
                        "u_vault_adj": u_vault_adj,
                        "vault_spread": spread_b,
                        "pred_overfit_gap": float(gap_p[0]),
                        "horizon_gen": g_snap,
                    }

            # Monotonic Tier-B horizon ratchet: only advance when unseen `u_vault_adj` improves by >= `ratchet_margin`!
            ratcheted = False
            if best_b_for_gen is not None:
                if locked_champ is None or best_b_for_gen["u_vault_adj"] >= locked_champ["u_vault_adj"] + float(ratchet_margin):
                    locked_champ = best_b_for_gen
                    ratcheted = True

            if locked_champ is None:
                continue

            # Tier C: Untouched `2xCV-7` (`rep=101, 102`) cross-fit of ONLY the single (`M = 1`) locked AST!
            if ratcheted or ("tier_c_pack" not in locked_champ):
                got_c0 = crossfit_single_ast_on_kfold(
                    locked_champ["dsl"],
                    locked_champ["target_mode"],
                    Xs,
                    Xa,
                    ys,
                    lab7_stack0,
                    anc_tup,
                    runner,
                    include_full_refit=True,
                    timeout_s=timeout_crossfit,
                )
                got_c1 = crossfit_single_ast_on_kfold(
                    locked_champ["dsl"],
                    locked_champ["target_mode"],
                    Xs,
                    Xa,
                    ys,
                    lab7_stack1,
                    anc_tup,
                    runner,
                    include_full_refit=False,
                    timeout_s=timeout_crossfit,
                )
                if got_c0 is not None and got_c1 is not None:
                    oof0, bag0, ref0, u_c0, _sp0 = got_c0
                    oof1, bag1, _ref1, u_c1, _sp1 = got_c1
                    q_tierc = 0.5 * (oof0 + oof1)
                    bag_tierc = 0.5 * (bag0 + bag1)
                    ref_tierc = ref0 if ref0 is not None else bag_tierc.copy()
                    u_tierc = 0.5 * (
                        fold_log_utility(q_tierc, ys, lab7_stack0) + fold_log_utility(q_tierc, ys, lab7_stack1)
                    )
                    locked_champ["tier_c_pack"] = (q_tierc, bag_tierc, ref_tierc, u_tierc, u_c0, u_c1)

            if "tier_c_pack" in locked_champ:
                q_tc, bag_tc, ref_tc, u_tc, u_c0, u_c1 = locked_champ["tier_c_pack"]
                prog_sha = hashlib.sha256(locked_champ["dsl"].encode("utf-8")).hexdigest()
                cb_dir = colblock_root / f"colblocks_g{g_snap}"
                cb_dir.mkdir(parents=True, exist_ok=True)
                cb_dst = cb_dir / f"{cid}.npz"
                cb_tmp = slot_dir / f"cb_g{g_snap}.tmp.npz"
                np.savez_compressed(
                    cb_tmp,
                    ok=np.array(True),
                    q_ugp_tierc=q_tc.astype(np.float64),
                    bag_ugp_tierc=bag_tc.astype(np.float64),
                    ref_ugp_tierc=ref_tc.astype(np.float64),
                    dsl=np.array(locked_champ["dsl"]),
                    target_mode=np.array(locked_champ["target_mode"]),
                    program_sha256=np.array(prog_sha),
                    u_vault=np.array(locked_champ["u_vault"], dtype=np.float64),
                    u_tierc_2xcv7=np.array(u_tc, dtype=np.float64),
                    horizon_gen=np.array(int(g_snap), dtype=np.int32),
                    ratcheted_from_gen=np.array(int(locked_champ["horizon_gen"]), dtype=np.int32),
                )
                os.replace(cb_tmp, cb_dst)
                horizon_ledger.append(
                    {
                        "horizon_gen": int(g_snap),
                        "active_born_gen": int(locked_champ["horizon_gen"]),
                        "ratcheted": bool(ratcheted),
                        "dsl": locked_champ["dsl"],
                        "target_mode": locked_champ["target_mode"],
                        "program_sha256": prog_sha,
                        "u_vault": round(float(locked_champ["u_vault"]), 6),
                        "u_tierc_2xcv7": round(float(u_tc), 6),
                        "u_tierc_r0": round(float(u_c0), 6),
                        "u_tierc_r1": round(float(u_c1), 6),
                    }
                )
    finally:
        runner.close()

    return {
        "cid": cid,
        "family_id": family_id,
        "anchor_source": anc_src,
        "tier_a_final_u_cv7": tier_a_res.get("final_u_cv7"),
        "tier_b_monotonic_vault": all(
            horizon_ledger[i]["u_vault"] >= horizon_ledger[i - 1]["u_vault"] - 1e-9
            for i in range(1, len(horizon_ledger))
        ),
        "horizons": horizon_ledger,
        "wall_s": round(time.time() - t0, 2),
    }


def write_colblock_sidecars(colblock_root: Path, snapshots: Sequence[int], results: Sequence[Dict]) -> List[str]:
    """Writes Track S Interface (i) `sidecar.json` inside each `colblocks_g{g}/` directory."""
    written: List[str] = []
    for g_snap in sorted(snapshots):
        cb_dir = colblock_root / f"colblocks_g{g_snap}"
        if not cb_dir.is_dir():
            continue
        prog_by_cid: Dict[str, Dict] = {}
        for r in results:
            for h in r.get("horizons", []):
                if int(h["horizon_gen"]) == int(g_snap):
                    prog_by_cid[str(r["cid"])] = {
                        "program_sha256": h["program_sha256"],
                        "dsl": h["dsl"],
                        "target_mode": h["target_mode"],
                        "u_vault": h["u_vault"],
                        "u_tierc_2xcv7": h["u_tierc_2xcv7"],
                    }
        npz_cnt = len(list(cb_dir.glob("*.npz")))
        sidecar = {
            "schema": "jope.h458.colblock_sidecar/v1",
            "track": "H458_TrackG",
            "closure": "fair-search-cv",
            "cross_fitted": True,
            "audit_labels_read": 0,
            "selection_candidates_crossfitted_in_tier_c": 1,
            "col_names": ["ugp_tierc"],
            "horizon_gen": int(g_snap),
            "contexts_present": npz_cnt,
            "programs_by_cid": prog_by_cid,
            "utc": time.strftime("%FT%TZ", time.gmtime()),
        }
        sc_p = cb_dir / "sidecar.json"
        tmp_p = cb_dir / ".sidecar.tmp.json"
        tmp_p.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_p, sc_p)
        written.append(str(sc_p))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
    ap.add_argument(
        "--anchor-104m-dirs",
        default=(
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_lh455v1_mh455v12_50_50,"
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_lw2_mh455_v2_65_35,"
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_m_h455_v12_avg,"
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/cv7_hyb_l_x1_h455_v1_soup50,"
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/traj_e8_w2aplus,"
            "/home/ubuntu/jope-data/h452-flywheel/cols_e8_d91/traj_e8_v1"
        ),
    )
    ap.add_argument("--warm-root", default="/home/ubuntu/jope-data/h458-flywheel/ultra_gp_cycle2_d91/warm")
    ap.add_argument("--out-root", default="/home/ubuntu/jope-data/h458-flywheel/ultra_gp_cycle2_d91/snapshots")
    ap.add_argument("--colblock-root", default="/home/ubuntu/jope-data/h458-flywheel/ultra_gp_cycle2_d91")
    ap.add_argument("--slots-root", default="/home/ubuntu/jope-data/h458-flywheel/slots/ultra_gp_cycle2")
    ap.add_argument("--qhat-ckpt", default="/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt")
    ap.add_argument("--out-receipt", default="/home/ubuntu/jope_plan/evidence/H458/trackG/ULTRA_GP_CYCLE2_SUMMARY.json")
    ap.add_argument("--indices", default="0-90")
    ap.add_argument("--stride-mod", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--inner-slots", type=int, default=3)
    ap.add_argument("--rayon-threads", type=int, default=1)
    ap.add_argument("--n-pop", type=int, default=44)
    ap.add_argument("--start-gen", type=int, default=1)
    ap.add_argument("--n-gens", type=int, default=1000)
    ap.add_argument("--snapshots", default="10,50,100,250,500,1000")
    ap.add_argument("--eval-per-gen", type=int, default=3)
    ap.add_argument("--screen-max-rows", type=int, default=560)
    ap.add_argument("--timeout-cv2", type=float, default=1.6)
    ap.add_argument("--timeout-crossfit", type=float, default=7.5)
    ap.add_argument("--stagnation-gens", type=int, default=40)
    ap.add_argument("--ratchet-margin", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=458301)
    a = ap.parse_args()

    # Hard guard against cl_h458 pitfall 5a (never overwrite h452-flywheel/ultra_gp_d91!)
    for p_str in (a.out_root, a.colblock_root, a.slots_root):
        if "h452-flywheel/ultra_gp_d91" in str(p_str):
            raise ValueError(f"Refusing to write into legacy h452-flywheel/ultra_gp_d91: {p_str}")

    try:
        os.nice(19)
    except OSError:
        pass

    C.resume_stage_v()
    assert _sha256_file(SUPERSET_BIN) == SUPERSET_SHA, f"Rust superset sha mismatch: {SUPERSET_BIN}"

    qhat = PretrainedQHatV4Prior(Path(a.qhat_ckpt), require_ok=True)
    assert qhat.ok, f"QHat failed to load: {a.qhat_ckpt}"

    snaps = tuple(sorted(int(x.strip()) for x in a.snapshots.split(",") if x.strip()))
    out_root = Path(a.out_root)
    colblock_root = Path(a.colblock_root)
    warm_root = Path(a.warm_root) if (a.warm_root and Path(a.warm_root).is_dir()) else None
    slots_root = Path(a.slots_root)
    out_root.mkdir(parents=True, exist_ok=True)
    colblock_root.mkdir(parents=True, exist_ok=True)
    slots_root.mkdir(parents=True, exist_ok=True)

    anchor_dirs = [Path(s.strip()) for s in a.anchor_104m_dirs.split(",") if s.strip()]

    dev_rows = C.dev_rows_for_stage_v()
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in dev_rows}
    cids_sorted = sorted(fam_by_cid)

    bank_by_cid: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(a.bank, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden in search bank!"
        b_cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(b_cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            bank_by_cid[c] = (
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            so += n * w
            ao += na * f
            lo += n

    base_indices = [idx for idx in _parse_indices(a.indices) if 0 <= idx < len(cids_sorted)]
    size_ranked = sorted(
        base_indices,
        key=lambda idx: -(bank_by_cid[cids_sorted[idx]][0].shape[0] * math.sqrt(max(1, bank_by_cid[cids_sorted[idx]][0].shape[1]))),
    )
    if a.stride_mod:
        rem_s, mod_s = a.stride_mod.split("/", 1)
        rem_i, mod_i = int(rem_s), int(mod_s)
        target_indices = [idx for pos, idx in enumerate(size_ranked) if (pos % mod_i) == rem_i]
    else:
        target_indices = size_ranked

    encoder = BaggedNeuralEncoderH452(dope_threads=2)

    def _run_one(idx: int) -> Dict:
        c = cids_sorted[idx]
        Xs, Xa, ys, lab5 = bank_by_cid[c]
        anc_pack = resolve_best_104m_anchor(c, len(ys), len(Xa), anchor_dirs)
        slot_d = slots_root / f"ctx_{idx:03d}"
        res = run_three_tier_hyper_cv_for_context(
            cid=c,
            family_id=fam_by_cid[c],
            Xs=Xs,
            Xa=Xa,
            ys=ys,
            lab5=lab5,
            anchor_pack=anc_pack,
            warm_root=warm_root,
            out_root=out_root,
            colblock_root=colblock_root,
            slot_dir=slot_d,
            encoder=encoder,
            qhat=qhat,
            n_pop=a.n_pop,
            start_gen=a.start_gen,
            n_gens=a.n_gens,
            snapshots=snaps,
            eval_per_gen=a.eval_per_gen,
            screen_max_rows=a.screen_max_rows,
            timeout_cv2=a.timeout_cv2,
            timeout_crossfit=a.timeout_crossfit,
            stagnation_gens=a.stagnation_gens,
            ratchet_margin=a.ratchet_margin,
            seed=a.seed,
            inner_slots=a.inner_slots,
            rayon_threads=a.rayon_threads,
        )
        print(json.dumps(res, sort_keys=True), flush=True)
        return res

    t_all = time.time()
    results: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = [ex.submit(_run_one, idx) for idx in target_indices]
        for f in cf.as_completed(futs):
            results.append(f.result())

    dope_ok, dope_sha = encoder.verify_dope_unchanged()
    assert dope_ok, f"DOPE weights changed: {dope_sha}"

    sidecars = write_colblock_sidecars(colblock_root, snaps, results)
    summary = {
        "wave": "H458_TrackG",
        "done_contexts": len(results),
        "qhat_loaded": qhat.ok,
        "qhat_version": qhat.version,
        "dope_unchanged": dope_ok,
        "dope_sha256": dope_sha,
        "all_monotonic_vault": all(bool(r.get("tier_b_monotonic_vault", True)) for r in results),
        "sidecars_written": sidecars,
        "total_wall_s": round(time.time() - t_all, 2),
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "contexts": results,
    }
    out_rec = Path(a.out_receipt)
    out_rec.parent.mkdir(parents=True, exist_ok=True)
    tmp_rec = out_rec.with_suffix(".tmp.json")
    tmp_rec.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_rec, out_rec)
    print(json.dumps({k: v for k, v in summary.items() if k != "contexts"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
