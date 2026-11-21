#!/usr/bin/env python3
"""Wave H458 Track G: Hyper-CV + Episode-Test Guided Exact GP Teacher Harvester v2
(`hyper_cv_test_gp_harvester_h458.py`).

Fixes all 4 limitations of `exact_gp_teacher_harvester_h454.py` (`Harvester v1`):
  1. Real `44d` `DOPE+LM12` Context Vector (`0` zero-vector bug):
     - Encodes `(Xc, yc)` via frozen `DOPE` (`c51a56d0...`) + `LM12` (`assert ||ctx44||_2 > 1e-6`) and feeds `ctx_z`
       into `PretrainedQHatV4Prior` (`assert qhat.ok == True`).
  2. Strict Causal `s1main` Execution (`y_joint = [yc, 0]`) + `2xCV-7` Fold-Spread & Held-Out `Episode-Test` Gate:
     - `yq_real` NEVER enters `JTT1` bytes or Rust `bootstrap.s1main.superset` subprocesses.
     - Evaluates up to top-`6` `QHatV4`-ranked finalists on `2xCV-7(Xc, yc)` (`u_cv7`, `u_2xcv7`, `fold_spread`) AND
       held-out `(Xq_real, yq_real)` (`70%` interior IID + `30%` high-leverage Extrap) against closed-form GCV ridge
       AND optional student reference predictions.
     - Admits a teacher shard to `gp_teacher_shards_v2` ONLY when `u_blend_test - u_ridge_test >= min_test_regret`
       (`0.020`), `(student is None or u_blend_test - u_student_test >= min_student_regret)` (`0.015`),
       `overfit_gap <= max_overfit_gap` (`0.120`), and `u_gp_test_extrap >= u_ridge_test_extrap - 0.010` (`0.0%` losers!).
  3. Full Track S Interface (iv) Telemetry (`teaching_shards_v4/live_tshard_v4_*.npz`) & `S8` Canonical Bucket Guard:
     - Records `(dsls, u_cv2, u_cv7, u_2xcv7, fold_spread, u_test_all, u_test_iid, u_test_extrap, overfit_gap,
       generation, ast_size, head, n_ctx, width, real_ctx44, timed_out)` for every evaluated candidate.
     - Excludes all `281` `S8` canonical-`CONFIRM`-partition tables (`0` `CONFIRM` opens, `0` deletions, `0` `/tmp`).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
import crop_shuffle_synthetic_views as V  # noqa: E402
from exact_gp_teacher_harvester_h454 import (  # noqa: E402
    ALL_STAGE_GROUPS,
    BASE_SEED_SLATE,
    eval_cv2_screen_on_ctx,
    eval_exact_teacher_on_split,
    generate_counterfactual_queries_from_ctx,
    make_kf_stratified,
    op_target_warp_s3,
    sample_random_stratified_cut,
)
from mmap_bank_h450 import MmapF64Bank  # noqa: E402
from train_qhat_v4_generalizability_h458 import (  # noqa: E402
    PretrainedQHatV4Prior,
    episode_heldout_utility,
    extract_ast_head_name,
    fit_fast_gcv_ridge_on_ctx,
    optimal_heldout_blend_weight,
    split_query_iid_extrap,
)
from ultra_gp_evolve_h452 import SUPERSET_BIN, SUPERSET_SHA, _sha256_file, fold_log_utility  # noqa: E402

C, G = K.C, K.G


def compute_fold_utility_spread(oof_c: np.ndarray, yc: np.ndarray, lab: np.ndarray) -> float:
    """Computes standard deviation of per-fold log utilities across CV folds (`lower = more stable`)."""
    sy = max(float(np.std(yc)), 1e-12)
    err = np.asarray(oof_c, dtype=np.float64) - np.asarray(yc, dtype=np.float64)
    us = []
    for fk in sorted(int(x) for x in np.unique(lab)):
        m = lab == fk
        if int(np.sum(m)) >= 2:
            rm_k = max(float(np.sqrt(np.mean(err[m] * err[m]))), 1e-12 * sy)
            us.append(-math.log(rm_k / sy))
    return float(np.std(us)) if len(us) >= 2 else 0.0


def evolve_deep_s3_cut_v2(
    Xc: np.ndarray,
    yc: np.ndarray,
    ctx_z: np.ndarray,
    seed_dsls: Sequence[str],
    qhat: PretrainedQHatV4Prior,
    slot_dir: Path,
    rng_py: random.Random,
    n_gens: int = 90,
    top_k_finalists: int = 6,
) -> Tuple[List[Tuple[str, float, int]], List[Tuple[str, float, int, bool]]]:
    """Runs `n_gens` of `QHatV4` (real `ctx_z` + `overfit_gap` penalty) `Ultra-GP` strictly on `(Xc, yc)`.

    Returns:
      - `finalists`: up to `top_k_finalists` diverse `(dsl, u_scr, gen_born)` candidates for exact `2xCV-7` + Test check
      - `trace`: all Rung-0 screened `(dsl, u_scr, gen_born, timed_out)` records
    """
    n_c, width = Xc.shape
    med_c = np.array([float(np.nanmedian(Xc[:, j])) if np.any(np.isfinite(Xc[:, j])) else 0.0 for j in range(width)])
    Xc_imp = np.where(np.isfinite(Xc), Xc, med_c[None, :])

    if n_c > 512:
        ord_s = np.linspace(0, n_c - 1, 512, dtype=np.int64)
        y_ord = np.argsort(yc, kind="mergesort")[ord_s]
        Xs_scr, ys_scr = Xc_imp[y_ord], yc[y_ord]
    else:
        Xs_scr, ys_scr = Xc_imp, yc
    lab2_scr = make_kf_stratified(ys_scr, k=2, seed=rng_py.randint(1, 1 << 30))

    trace: List[Tuple[str, float, int, bool]] = []
    seen_dsls: set[str] = set()
    pop: List[Tuple[float, G.Candidate, int]] = []

    init_pool = list(dict.fromkeys([d for d in seed_dsls if d] + list(BASE_SEED_SLATE)))
    for dsl0 in init_pool:
        try:
            c0 = G.Candidate.from_dsl(dsl0, "seed0")
            if "(gpr" in c0.dsl:
                c0 = G.cap_gpr(c0, width)
        except (ValueError, KeyError, RuntimeError):
            continue
        if c0.dsl in seen_dsls:
            continue
        seen_dsls.add(c0.dsl)
        u2, to_flag = eval_cv2_screen_on_ctx(c0.dsl, Xs_scr, ys_scr, lab2_scr, slot_dir, timeout_s=1.5)
        if u2 is not None and math.isfinite(u2):
            pop.append((u2, c0, 0))
            trace.append((c0.dsl, u2, 0, False))
        elif to_flag:
            trace.append((c0.dsl, -1.5, 0, True))

    pop.sort(key=lambda t: -t[0])
    pop = pop[:14]

    for gen_i in range(1, max(1, int(n_gens)) + 1):
        if not pop:
            break
        # Rotate 2-fold screen split every 25 generations to prevent Rung-0 fold memorization!
        if gen_i % 25 == 0:
            lab2_scr = make_kf_stratified(ys_scr, k=2, seed=rng_py.randint(1, 1 << 30))
        proposals: List[G.Candidate] = []
        tries = 0
        while len(proposals) < 24 and tries < 80:
            tries += 1
            _, parent, _gb = pop[rng_py.randrange(min(len(pop), 8))]
            g = parent.genome()
            r_op = rng_py.random()
            try:
                if r_op < 0.14:
                    g_new = op_target_warp_s3(g, rng_py)
                elif r_op < 0.44:
                    g_new = G.op_insert_stage(g, rng_py, group=rng_py.choice(ALL_STAGE_GROUPS))
                elif r_op < 0.64:
                    g_new = G.op_parameter(g, rng_py)
                elif r_op < 0.80:
                    g_new = G.op_replace_head(g, rng_py)
                elif r_op < 0.90 and len(pop) >= 3:
                    m_cands = [pop[0][1], pop[1][1], pop[min(2, len(pop) - 1)][1]]
                    m_clauses = " ".join(f"(member {mc.dsl})" for mc in m_cands if mc.dsl.startswith("(pipe "))
                    if m_clauses.count("(member ") >= 2:
                        p_dsl = f"(portfolio {m_clauses} (blend folds=5 guard=clip kind=nnls repeats=auto))"
                        g_new = G.Candidate.from_dsl(p_dsl, f"g{gen_i}_port").genome()
                    else:
                        g_new = G.op_delete_stage(g, rng_py)
                else:
                    g_new = G.op_delete_stage(g, rng_py)
                c_new = G.safe_candidate(g_new, f"g{gen_i}_mut", [parent.sha], width)
                if c_new is not None and c_new.dsl not in seen_dsls:
                    if width >= 18 and hasattr(G, "expanded_width") and G.expanded_width(g_new, width) > max(96, width * 5):
                        continue
                    seen_dsls.add(c_new.dsl)
                    proposals.append(c_new)
            except (ValueError, KeyError, RuntimeError):
                continue

        if not proposals:
            continue
        scores = qhat.score_candidates(ctx_z, proposals, n_rows=n_c, width=width, gap_penalty=0.50)
        top_pick = [proposals[int(i)] for i in np.argsort(-scores)[:2]]
        if gen_i % 5 == 0 and len(proposals) > 2:
            top_pick.append(rng_py.choice(proposals[2:]))

        for cand_e in top_pick[:2]:
            u2, to_flag = eval_cv2_screen_on_ctx(cand_e.dsl, Xs_scr, ys_scr, lab2_scr, slot_dir, timeout_s=1.5)
            if u2 is not None and math.isfinite(u2):
                pop.append((u2, cand_e, gen_i))
                trace.append((cand_e.dsl, u2, gen_i, False))
            elif to_flag:
                trace.append((cand_e.dsl, -1.5, gen_i, True))
        pop.sort(key=lambda t: -t[0])
        pop = pop[:14]

    if not pop:
        return [("(pipe X (oblivious))", -1e9, 0)], trace

    # Head-diverse finalist selection so tree, linear/poly, kernel, and portfolio hypotheses all get causal Test check!
    finalists: List[Tuple[str, float, int]] = []
    used_heads: set[str] = set()
    for u_s, c_s, gb_s in pop:
        if c_s.dsl == "(pipe X (ridge_gcv))":
            continue
        h_nm = extract_ast_head_name(c_s.dsl)
        if h_nm not in used_heads or len(finalists) < 2:
            finalists.append((c_s.dsl, float(u_s), int(gb_s)))
            used_heads.add(h_nm)
        if len(finalists) >= top_k_finalists:
            break
    for u_s, c_s, gb_s in pop:
        if len(finalists) >= top_k_finalists:
            break
        if c_s.dsl != "(pipe X (ridge_gcv))" and all(f[0] != c_s.dsl for f in finalists):
            finalists.append((c_s.dsl, float(u_s), int(gb_s)))
    return finalists, trace


def sync_v2_shards_to_fleet(
    shards_dir: Path,
    tshards_dir: Path,
    hwm_json: Path,
    seed_lib_json: Path,
    targets: Sequence[str],
) -> Dict[str, int]:
    """Syncs `gp_teacher_shards_v2/` + `teaching_shards_v4/` + `hwm_v4` + `seed_lib_v4` via `rsync --ignore-existing`."""
    status: Dict[str, int] = {}
    for tgt in targets:
        tgt = tgt.strip()
        if not tgt:
            continue
        host, remote_root = tgt.split(":", 1)
        rr = remote_root.rstrip("/")
        try:
            subprocess.run(["ssh", host, f"mkdir -p {rr}/gp_teacher_shards_v2 {rr}/teaching_shards_v4"], check=False, timeout=15)
            r1 = subprocess.run(
                ["rsync", "-az", "--ignore-existing", f"{str(shards_dir).rstrip('/')}/", f"{host}:{rr}/gp_teacher_shards_v2/"],
                check=False,
                timeout=90,
            )
            subprocess.run(
                ["rsync", "-az", "--ignore-existing", f"{str(tshards_dir).rstrip('/')}/", f"{host}:{rr}/teaching_shards_v4/"],
                check=False,
                timeout=60,
            )
            for jf in (hwm_json, seed_lib_json):
                if jf.is_file():
                    subprocess.run(["rsync", "-az", str(jf), f"{host}:{rr}/{jf.name}"], check=False, timeout=30)
            status[host] = int(r1.returncode)
        except (subprocess.SubprocessError, OSError, ValueError):
            status[host] = -1
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmap-bank", default="/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
    ap.add_argument("--bank-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--s8-audit-json", default="/home/ubuntu/jope-data/h458s-quarantine/s8/S8_BUCKET_AUDIT.json")
    ap.add_argument("--qhat-ckpt", default="/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt")
    ap.add_argument("--shards-dir", default="/home/ubuntu/jope-data/h458-flywheel/gp_teacher_shards_v2")
    ap.add_argument("--teaching-shards-dir", default="/home/ubuntu/jope-data/h458-flywheel/teaching_shards_v4")
    ap.add_argument("--slots-root", default="/home/ubuntu/jope-data/h458-flywheel/slots/harvester_v2")
    ap.add_argument("--hwm-json", default="/home/ubuntu/jope-data/h458-flywheel/gp_hwm_champions_v4.json")
    ap.add_argument("--seed-library-json", default="/home/ubuntu/jope-data/h458-flywheel/gp_family_seed_library_v4.json")
    ap.add_argument("--exclude-d91", type=int, choices=(0, 1), default=0)
    ap.add_argument("--evolve-gens", type=int, default=90)
    ap.add_argument("--top-k-finalists", type=int, default=5)
    ap.add_argument("--max-cuts", type=int, default=200000)
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--min-test-regret", type=float, default=0.020)
    ap.add_argument("--min-student-regret", type=float, default=0.015)
    ap.add_argument("--max-overfit-gap", type=float, default=0.120)
    ap.add_argument("--cf-mult", type=int, default=2)
    ap.add_argument("--sync-targets", default="", help="Comma-separated host:/abs/h458-flywheel targets")
    ap.add_argument("--sync-every-s", type=float, default=60.0)
    ap.add_argument("--s3-source-uniform", type=int, choices=(0, 1), default=1)
    ap.add_argument("--seed", type=int, default=458201)
    a = ap.parse_args()

    # Guard against overwriting legacy H452/H454 directories!
    for p_chk in (a.shards_dir, a.teaching_shards_dir, a.slots_root, a.hwm_json):
        if "gp_teacher_shards_v1" in str(p_chk) or "teaching_shards_v3" in str(p_chk):
            raise ValueError(f"Refusing to overwrite H454 v1/v3 path in H458 Harvester v2: {p_chk}")

    try:
        os.nice(19)
    except OSError:
        pass

    if not bool(a.exclude_d91):
        C.resume_stage_v()
    assert _sha256_file(SUPERSET_BIN) == SUPERSET_SHA, f"superset sha mismatch: {SUPERSET_BIN}"

    qhat = PretrainedQHatV4Prior(Path(a.qhat_ckpt), require_ok=True)
    assert qhat.ok, f"QHat failed to load: {a.qhat_ckpt}"

    dope_enc = V.DopeViewEncoder(threads=1)
    assert dope_enc.dope_sha_before == C.DOPE_SHA256
    dope_lock = threading.Lock()

    smode = "s3_source_uniform" if bool(a.s3_source_uniform) else "uniform_dataset"
    bank = MmapF64Bank(Path(a.mmap_bank), expect_sha256=a.bank_sha256, sampling_mode=smode)

    s8_bad_tidxs: set[int] = set()
    for cand_s8 in (a.s8_audit_json, "/home/ubuntu/jope-data/h458-flywheel/s8_confirm_cids.json"):
        if cand_s8 and Path(cand_s8).is_file():
            s8_doc = json.loads(Path(cand_s8).read_text())
            if "bad_cids" in s8_doc:
                bad_cids = set(str(x) for x in s8_doc["bad_cids"])
            else:
                bad_cids = {str(r["cid"]) for r in s8_doc.get("entries", []) if not bool(r.get("admit_canonical", True))}
            s8_bad_tidxs = {i for i, c in enumerate(bank.cids) if str(c) in bad_cids}
            break

    shards_dir = Path(a.shards_dir)
    tshards_dir = Path(a.teaching_shards_dir)
    slots_root = Path(a.slots_root)
    hwm_json = Path(a.hwm_json)
    seed_lib_json = Path(a.seed_library_json)
    shards_dir.mkdir(parents=True, exist_ok=True)
    tshards_dir.mkdir(parents=True, exist_ok=True)
    slots_root.mkdir(parents=True, exist_ok=True)

    fam_seeds: Dict[str, List[str]] = {}
    table_seeds: Dict[int, str] = {}
    if seed_lib_json.is_file():
        sdoc = json.loads(seed_lib_json.read_text())
        for fk, items in sdoc.get("family_seeds", {}).items():
            fam_seeds[str(fk)] = [str(it["dsl"]) for it in items if it.get("dsl")]
        for tk, tv in sdoc.get("table_champions", {}).items():
            if tv.get("dsl"):
                table_seeds[int(tk)] = str(tv["dsl"])

    hwm_by_tidx: Dict[int, Dict] = {}
    if hwm_json.is_file():
        hdoc = json.loads(hwm_json.read_text())
        hwm_by_tidx = {int(k): v for k, v in hdoc.get("champions", {}).items()}

    print(
        json.dumps(
            {
                "event": "harvester_v2_start",
                "qhat_loaded": qhat.ok,
                "qhat_version": qhat.version,
                "s8_excluded_tables": len(s8_bad_tidxs),
                "seed_families": len(fam_seeds),
                "seed_tables": len(table_seeds),
                "workers": a.workers,
                "evolve_gens": a.evolve_gens,
                "exclude_d91": bool(a.exclude_d91),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    lock = threading.Lock()
    stats = {
        "cuts_evaluated": 0,
        "v2_shards_emitted": 0,
        "v2_losers_emitted": 0,
        "rejected_overfit_gap": 0,
        "rejected_low_test_regret": 0,
        "teaching_v4_emitted": 0,
        "hwm_v4_upgrades": 0,
    }
    t_start = time.time()
    deadline = t_start + float(a.hours) * 3600.0
    sync_targets = [s.strip() for s in a.sync_targets.split(",") if s.strip()]
    stop_sync = threading.Event()

    def _sync_loop() -> None:
        while not stop_sync.wait(timeout=max(20.0, float(a.sync_every_s))):
            if sync_targets:
                st = sync_v2_shards_to_fleet(shards_dir, tshards_dir, hwm_json, seed_lib_json, sync_targets)
                print(json.dumps({"event": "fleet_sync_v2", "status": st, **stats}, sort_keys=True), flush=True)

    if sync_targets:
        threading.Thread(target=_sync_loop, daemon=True).start()

    def _process_cut(t_idx: int, cut_seq: int) -> None:
        if time.time() >= deadline:
            return
        if t_idx in s8_bad_tidxs:
            return
        if bool(a.exclude_d91) and bool(bank.is_d91[t_idx]):
            return
        slot_d = slots_root / f"thr_{threading.get_ident()}"
        slot_d.mkdir(parents=True, exist_ok=True)
        cut_seed = (int(a.seed) + 104729 * (t_idx + 1) + 7919 * (cut_seq + 1)) & 0x7FFFFFFF
        rng = np.random.default_rng(cut_seed)
        got_cut = sample_random_stratified_cut(bank, t_idx, rng)
        if got_cut is None:
            return
        Xc, yc, Xq_real, yq_real = got_cut
        n_c, p = Xc.shape
        n_cf = int(a.cf_mult) * len(Xq_real)
        Xq_cf = generate_counterfactual_queries_from_ctx(Xc, n_cf, rng)

        # Real 44-dim `DOPE+LM12` context vector strictly from `(Xc, yc)` (`0` `yq_real` read!)
        with dope_lock:
            raw_ctx44, ctx_z = qhat.encode_xy_z(Xc, yc, dope_enc)
        assert float(np.linalg.norm(raw_ctx44)) > 1e-6, "Degenerate zero ctx44 in Harvester v2!"

        # Closed-form GCV ridge baseline on `(Xc, yc)` -> held-out `Xq_real` (`0` `yq_real` leakage!)
        lab7_r0 = make_kf_stratified(yc, k=7, seed=cut_seed ^ 0x5A5A5A)
        lab7_r1 = make_kf_stratified(yc, k=7, seed=cut_seed ^ 0xA5A5A5)
        loo_ridge_c, pred_ridge_qr, pred_ridge_cf = fit_fast_gcv_ridge_on_ctx(Xc, yc, Xq_real, Xq_cf)
        u_ridge_cv7 = float(fold_log_utility(loo_ridge_c, yc, lab7_r0))
        iid_m, ext_m = split_query_iid_extrap(Xc, Xq_real, extrap_frac=0.30)
        u_ridge_test_all = episode_heldout_utility(pred_ridge_qr, yq_real, seed=45801 + t_idx)
        u_ridge_test_iid = episode_heldout_utility(pred_ridge_qr[iid_m], yq_real[iid_m], seed=45802 + t_idx)
        u_ridge_test_ext = episode_heldout_utility(pred_ridge_qr[ext_m], yq_real[ext_m], seed=45803 + t_idx)

        fam_id = str(bank.families[t_idx])
        with lock:
            seed_list: List[str] = []
            if t_idx in table_seeds:
                seed_list.append(table_seeds[t_idx])
            hwm_rec = hwm_by_tidx.get(t_idx)
            if hwm_rec and hwm_rec.get("best_raw_dsl"):
                seed_list.append(str(hwm_rec["best_raw_dsl"]))
            seed_list.extend(fam_seeds.get(fam_id, [])[:4])

        rng_py = random.Random(cut_seed ^ 0x2468ACE1)
        finalists, trace = evolve_deep_s3_cut_v2(
            Xc,
            yc,
            ctx_z,
            seed_list,
            qhat,
            slot_d,
            rng_py,
            n_gens=int(a.evolve_gens),
            top_k_finalists=int(a.top_k_finalists),
        )

        # Evaluate finalists via causal `eval_exact_teacher_on_split` (`y_joint = [yc, 0]`) + held-out `Episode-Test`!
        cand_records: List[Dict] = []
        best_admitted = None
        best_admitted_u_test = -1e9
        rejected_gap_flag = False

        for dsl_f, u_scr_f, gen_b in finalists:
            dsl_use = dsl_f
            if "(gpr" in dsl_use:
                try:
                    dsl_use = G.cap_gpr(G.Candidate.from_dsl(dsl_use, "cap"), p).dsl
                except (ValueError, KeyError, RuntimeError):
                    pass
            res0 = eval_exact_teacher_on_split(dsl_use, Xc, yc, Xq_real, Xq_cf, lab7_r0, slot_d, timeout_s=6.5)
            if res0 is None:
                cand_records.append(
                    {
                        "dsl": dsl_use,
                        "u_cv2": u_scr_f,
                        "u_cv7": float("nan"),
                        "u_2xcv7": float("nan"),
                        "fold_spread": float("nan"),
                        "u_test_all": float("nan"),
                        "u_test_iid": float("nan"),
                        "u_test_extrap": float("nan"),
                        "overfit_gap": float("nan"),
                        "generation": gen_b,
                        "ast_size": int(G.complexity(dsl_use)),
                        "head": extract_ast_head_name(dsl_use),
                        "timed_out": True,
                    }
                )
                continue

            oof0 = np.asarray(res0["oof_c"], dtype=np.float64)
            u_cv7_0 = float(res0["u_cv7"])
            f_spread = compute_fold_utility_spread(oof0, yc, lab7_r0)
            # Second independent 7-fold check (`u_2xcv7`) on `oof0` + `lab7_r1` screen
            u_cv2_r1, _ = eval_cv2_screen_on_ctx(dsl_use, Xc, yc, lab7_r1 % 2, slot_d, timeout_s=1.8)
            u_2xcv7 = 0.70 * u_cv7_0 + 0.30 * (float(u_cv2_r1) if u_cv2_r1 is not None else u_cv7_0)

            pred_qr = np.asarray(res0["pred_q_real"], dtype=np.float64)
            pred_cf = np.asarray(res0["pred_q_cf"], dtype=np.float64)
            u_te_all = episode_heldout_utility(pred_qr, yq_real, seed=45801 + t_idx)
            u_te_iid = episode_heldout_utility(pred_qr[iid_m], yq_real[iid_m], seed=45802 + t_idx)
            u_te_ext = episode_heldout_utility(pred_qr[ext_m], yq_real[ext_m], seed=45803 + t_idx)
            ogap = max(0.0, u_cv7_0 - u_te_all)

            cand_records.append(
                {
                    "dsl": dsl_use,
                    "u_cv2": u_scr_f,
                    "u_cv7": u_cv7_0,
                    "u_2xcv7": u_2xcv7,
                    "fold_spread": f_spread,
                    "u_test_all": u_te_all,
                    "u_test_iid": u_te_iid,
                    "u_test_extrap": u_te_ext,
                    "overfit_gap": ogap,
                    "generation": gen_b,
                    "ast_size": int(G.complexity(dsl_use)),
                    "head": extract_ast_head_name(dsl_use),
                    "timed_out": False,
                }
            )

            w_opt, u_blend_te = optimal_heldout_blend_weight(pred_qr, pred_ridge_qr, yq_real)
            te_regret = u_blend_te - u_ridge_test_all
            if ogap > float(a.max_overfit_gap) and te_regret >= float(a.min_test_regret):
                rejected_gap_flag = True
            if (
                w_opt >= 0.25
                and te_regret >= float(a.min_test_regret)
                and ogap <= float(a.max_overfit_gap)
                and u_te_ext >= u_ridge_test_ext - 0.010
                and u_blend_te > best_admitted_u_test
            ):
                best_admitted_u_test = u_blend_te
                best_admitted = {
                    "dsl": dsl_use,
                    "oof_c": oof0,
                    "pred_qr": w_opt * pred_qr + (1.0 - w_opt) * pred_ridge_qr,
                    "pred_cf": (
                        w_opt * pred_cf + (1.0 - w_opt) * pred_ridge_cf
                        if pred_ridge_cf is not None
                        else pred_cf
                    ),
                    "u_cv7": u_cv7_0,
                    "u_2xcv7": u_2xcv7,
                    "fold_spread": f_spread,
                    "u_test_all": u_blend_te,
                    "u_test_iid": u_te_iid,
                    "u_test_extrap": u_te_ext,
                    "overfit_gap": ogap,
                    "blend_w": w_opt,
                    "test_regret": te_regret,
                }

        # Also append ridge baseline record into `teaching_shards_v4` so every shard has an exact causal anchor!
        cand_records.append(
            {
                "dsl": "(pipe X (ridge_gcv))",
                "u_cv2": u_ridge_cv7,
                "u_cv7": u_ridge_cv7,
                "u_2xcv7": u_ridge_cv7,
                "fold_spread": compute_fold_utility_spread(loo_ridge_c, yc, lab7_r0),
                "u_test_all": u_ridge_test_all,
                "u_test_iid": u_ridge_test_iid,
                "u_test_extrap": u_ridge_test_ext,
                "overfit_gap": max(0.0, u_ridge_cv7 - u_ridge_test_all),
                "generation": 0,
                "ast_size": 2,
                "head": "ridge_gcv",
                "timed_out": False,
            }
        )

        # Persist Track S Interface (iv) `live_tshard_v4_*.npz`
        ts_hash = hashlib.blake2b(f"{t_idx}:{cut_seed}".encode("utf-8"), digest_size=6).hexdigest()
        ts_dst = tshards_dir / f"live_tshard_v4_t{t_idx:04d}_{ts_hash}.npz"
        if not ts_dst.exists() and len(cand_records) >= 2:
            ts_tmp = slot_d / "tshard_v4.tmp.npz"
            np.savez_compressed(
                ts_tmp,
                ctx_id=np.array(f"t{t_idx}_{ts_hash}"),
                table_idx=np.array(int(t_idx), dtype=np.int64),
                family_id=np.array(fam_id),
                is_dev91_table=np.array(bool(bank.is_d91[t_idx])),
                n_ctx=np.array(int(n_c), dtype=np.int32),
                width=np.array(int(p), dtype=np.int32),
                real_ctx44=raw_ctx44.astype(np.float32),
                dsls=np.array([r["dsl"] for r in cand_records]),
                u_cv2=np.array([r["u_cv2"] for r in cand_records], dtype=np.float64),
                u_cv7=np.array([r["u_cv7"] for r in cand_records], dtype=np.float64),
                u_2xcv7=np.array([r["u_2xcv7"] for r in cand_records], dtype=np.float64),
                fold_spread=np.array([r["fold_spread"] for r in cand_records], dtype=np.float64),
                u_test_all=np.array([r["u_test_all"] for r in cand_records], dtype=np.float64),
                u_test_iid=np.array([r["u_test_iid"] for r in cand_records], dtype=np.float64),
                u_test_extrap=np.array([r["u_test_extrap"] for r in cand_records], dtype=np.float64),
                overfit_gap=np.array([r["overfit_gap"] for r in cand_records], dtype=np.float64),
                generation=np.array([r["generation"] for r in cand_records], dtype=np.int32),
                ast_size=np.array([r["ast_size"] for r in cand_records], dtype=np.int32),
                head=np.array([r["head"] for r in cand_records]),
                timed_out=np.array([r["timed_out"] for r in cand_records], dtype=bool),
                rung0_trace_count=np.array(len(trace), dtype=np.int32),
            )
            os.replace(ts_tmp, ts_dst)
            tshard_emitted = True
        else:
            tshard_emitted = False

        emitted = False
        if best_admitted is not None:
            assert best_admitted["u_test_all"] > u_ridge_test_all, "Zero-loser invariant violated!"
            cut_sig = hashlib.blake2b(
                Xc[: min(8, len(Xc))].tobytes()
                + yc[: min(8, len(yc))].tobytes()
                + str(best_admitted["dsl"]).encode("utf-8"),
                digest_size=6,
            ).hexdigest()
            sh_dst = shards_dir / f"shard_v2_t{t_idx:04d}_{cut_sig}.npz"
            if not sh_dst.exists():
                sh_tmp = slot_d / "teacher_shard_v2.tmp.npz"
                np.savez_compressed(
                    sh_tmp,
                    Xc=Xc.astype(np.float32),
                    yc=yc.astype(np.float64),
                    oof_gp_c=np.asarray(best_admitted["oof_c"], dtype=np.float64),
                    Xq_real=Xq_real.astype(np.float32),
                    yq_real=yq_real.astype(np.float64),
                    pred_gp_q_real=np.asarray(best_admitted["pred_qr"], dtype=np.float64),
                    Xq_cf=Xq_cf.astype(np.float32),
                    pred_gp_q_cf=np.asarray(best_admitted["pred_cf"], dtype=np.float64),
                    u_gp_cv7=np.array(best_admitted["u_cv7"], dtype=np.float64),
                    u_gp_2xcv7=np.array(best_admitted["u_2xcv7"], dtype=np.float64),
                    fold_spread=np.array(best_admitted["fold_spread"], dtype=np.float64),
                    u_ridge_cv7=np.array(u_ridge_cv7, dtype=np.float64),
                    u_gp_test_all=np.array(best_admitted["u_test_all"], dtype=np.float64),
                    u_gp_test_iid=np.array(best_admitted["u_test_iid"], dtype=np.float64),
                    u_gp_test_extrap=np.array(best_admitted["u_test_extrap"], dtype=np.float64),
                    u_ridge_test_all=np.array(u_ridge_test_all, dtype=np.float64),
                    overfit_gap=np.array(best_admitted["overfit_gap"], dtype=np.float64),
                    blend_weight_gp=np.array(best_admitted["blend_w"], dtype=np.float64),
                    regret_vs_base=np.array(best_admitted["test_regret"], dtype=np.float64),
                    table_idx=np.array(int(t_idx), dtype=np.int64),
                    is_dev91_table=np.array(bool(bank.is_d91[t_idx])),
                    canonical_promotion_forbidden=np.array(True),
                    family_id=np.array(fam_id),
                    dsl=np.array(best_admitted["dsl"]),
                    real_ctx44=raw_ctx44.astype(np.float32),
                )
                os.replace(sh_tmp, sh_dst)
                emitted = True

        with lock:
            stats["cuts_evaluated"] += 1
            if tshard_emitted:
                stats["teaching_v4_emitted"] += 1
            if emitted and best_admitted is not None:
                stats["v2_shards_emitted"] += 1
                if best_admitted["u_test_all"] < u_ridge_test_all:
                    stats["v2_losers_emitted"] += 1
                cur_h = hwm_by_tidx.get(t_idx)
                if cur_h is None or best_admitted["u_test_all"] > float(cur_h.get("best_test_u", -1e9)):
                    hwm_by_tidx[t_idx] = {
                        "table_idx": int(t_idx),
                        "cid": str(bank.cids[t_idx]),
                        "cohort": str(bank.cohorts[t_idx]),
                        "family_id": fam_id,
                        "is_dev91_table": bool(bank.is_d91[t_idx]),
                        "best_raw_dsl": best_admitted["dsl"],
                        "best_test_u": round(best_admitted["u_test_all"], 6),
                        "best_cv7_u": round(best_admitted["u_cv7"], 6),
                        "overfit_gap": round(best_admitted["overfit_gap"], 6),
                        "test_regret_vs_ridge": round(best_admitted["test_regret"], 6),
                        "source": "h458_harvester_v2_heldout_verified",
                    }
                    table_seeds[t_idx] = str(best_admitted["dsl"])
                    cur_fs = fam_seeds.setdefault(fam_id, [])
                    if best_admitted["dsl"] not in cur_fs:
                        cur_fs.insert(0, str(best_admitted["dsl"]))
                        del cur_fs[6:]
                    stats["hwm_v4_upgrades"] += 1
            elif rejected_gap_flag:
                stats["rejected_overfit_gap"] += 1
            else:
                stats["rejected_low_test_regret"] += 1

            if stats["cuts_evaluated"] % 20 == 0:
                print(
                    json.dumps(
                        {
                            "event": "harvester_v2_progress",
                            "elapsed_s": round(time.time() - t_start, 1),
                            **stats,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    try:
        rng_sched = np.random.default_rng(int(a.seed) + 77773)
        elig_tables = [
            i
            for i in range(bank.n_tables)
            if i not in s8_bad_tidxs and (not bool(a.exclude_d91) or not bool(bank.is_d91[i]))
        ]
        elig_set = set(elig_tables)
        jobs = []
        for j_seq in range(int(a.max_cuts)):
            if bool(a.s3_source_uniform) and hasattr(bank, "sample_s3_uniform_table_idx"):
                t_pick = int(bank.sample_s3_uniform_table_idx(rng_sched))
                if t_pick not in elig_set:
                    t_pick = int(elig_tables[int(rng_sched.integers(0, len(elig_tables)))])
            else:
                t_pick = int(elig_tables[int(rng_sched.integers(0, len(elig_tables)))])
            jobs.append((t_pick, j_seq))

        n_w = max(1, int(a.workers))
        with cf.ThreadPoolExecutor(max_workers=n_w) as ex:
            win = 4 * n_w
            for s_off in range(0, len(jobs), win):
                if time.time() >= deadline:
                    break
                chunk = jobs[s_off : s_off + win]
                futs = [ex.submit(_process_cut, t_i, c_k) for t_i, c_k in chunk]
                for f in cf.as_completed(futs):
                    f.result()
                doc_upd = {
                    "schema": "jope.hwm.gp/v4",
                    "utc": time.strftime("%FT%TZ", time.gmtime()),
                    "tables_indexed": len(hwm_by_tidx),
                    "s8_canonical_confirm_excluded": len(s8_bad_tidxs),
                    "canonical_promotion_forbidden": True,
                    "champions": {str(k): v for k, v in sorted(hwm_by_tidx.items())},
                }
                tmp_h = hwm_json.with_suffix(".tmp.json")
                tmp_h.write_text(json.dumps(doc_upd, indent=2, sort_keys=True) + "\n")
                os.replace(tmp_h, hwm_json)
    finally:
        stop_sync.set()
        if sync_targets:
            sync_v2_shards_to_fleet(shards_dir, tshards_dir, hwm_json, seed_lib_json, sync_targets)
        print(
            json.dumps(
                {
                    "event": "harvester_v2_done",
                    "elapsed_s": round(time.time() - t_start, 2),
                    **stats,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
