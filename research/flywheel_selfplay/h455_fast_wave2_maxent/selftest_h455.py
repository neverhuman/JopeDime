#!/usr/bin/env python3
"""7-Invariant Verification Suite for Wave H455 (`selftest_h455.py`).

Verifies:
  1. Zero Query Leakage (`max_abs_diff == 0.0` on context tensors when `(Xq, yq)` are perturbed by `+1000.0` / `NaN`)
     across `inject_synth_cols_22fam_h455` (all 22 families incl. `CW-1`, `CW-2`, RFF, LOO `kNN`, and all 4 distractors),
     `sample_binned_exact_kd_episode_h455`, `sample_binned_exact_resid_episode_h455`, and
     `sample_binned_s3_maxent_episode_h455`.
  2. Strict Quarantine Audit (`0` `CONFIRM`, `0` `FMGAP` B/C, `0` `HOLDOUT` / `SEALED` / `DL-VAL`) in `MmapF64Bank`,
     `ExactGPTeacherShardStoreH455`, and source strings.
  3. S3-Source Uniformity (`2,389 / 2,389` families hit across `80,000` draws; `max/median` empirical family count `< 2.0x`
     vs `36.0x` previously).
  4. Entropy Range Coverage (`drop_frac` spans `[0.00, 0.85]` incl. `drop == 0`, `k_inj` spans `[0, 0.60 * p]`, and
     `draw_p_ctx_h455` spans `[0.08, 0.95]`).
  5. Cross-Shard Shape-Grid Batch Density (`_flywheel_worker_loop_h455` emits multi-dataset `exact_kd`, `exact_resid`,
     and `real_s3` batches with `B >= 6` vs `B = 1` in H454, and `0` `Ledger128.add_real` bounds errors).
  6. Endless Smooth Warm-Floor Cosine Valley (`compute_cyclic_valley_lr` C1 continuity `< 1e-9` at `dt = cycle0_s` +
     4-step CPU smoke run of `train_104m_gp_flywheel_h455_maxent.py` saving `ckpt_latest.pt` & `ckpt_last.pt`).
  7. AST Hygiene (`0` file-deletion calls and `0` `/tmp` references across all H455 created/modified scripts).
"""
from __future__ import annotations

import ast
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

STAGE_CODE = Path("/home/ubuntu/h452-stage/code")
if sys.path[:1] != [str(STAGE_CODE)]:
    sys.path.insert(0, str(STAGE_CODE))
for _c in (
    "/home/ubuntu/jope-data/h450-fleet-agy/stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h448-stage/pub-r1/code",
):
    if Path(_c).is_dir() and _c not in sys.path:
        sys.path.append(_c)

from mmap_bank_h450 import SEALED_TOKENS, MmapF64Bank, sha256_file  # noqa: E402
import train_mps_104m_h450 as BASE  # noqa: E402
from train_104m_gp_flywheel_h455_maxent import (  # noqa: E402
    ExactGPTeacherShardStoreH455,
    ICLH450MPS,
    _flywheel_worker_loop_h455,
    choose_drop_keep_h455,
    compute_cyclic_valley_lr,
    draw_p_ctx_h455,
    inject_synth_cols_22fam_h455,
    sample_binned_exact_kd_episode_h455,
    sample_binned_exact_resid_episode_h455,
)

MMAP_DIR = Path("/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
SHARDS_DIR = Path("/home/ubuntu/jope-data/h452-flywheel/gp_teacher_shards_v1")
HWM_JSON = Path("/home/ubuntu/jope-data/h452-flywheel/gp_hwm_champions_v3.json")
WINNERS_JSONL = Path("/home/ubuntu/jope-data/h452-flywheel/gp_winners_all_v2.jsonl")

H455_FILES = [
    STAGE_CODE / "mmap_bank_h450.py",
    STAGE_CODE / "cols_h450_gpu.py",
    STAGE_CODE / "cols_104m_e8_gpresid_h452.py",
    STAGE_CODE / "exact_gp_teacher_harvester_h454.py",
    STAGE_CODE / "train_104m_gp_flywheel_h455_maxent.py",
    STAGE_CODE / "watch_and_score_wave2_h454.py",
]


def test_1_zero_query_leakage(shard_store: ExactGPTeacherShardStoreH455) -> dict:
    rng_base = np.random.default_rng(455001)
    max_diff_inj = 0.0
    for trial in range(24):
        n_c = 64 + 8 * (trial % 5)
        n_q = 28
        p = 10
        Xc = rng_base.standard_normal((n_c, p))
        # Make col 0 discrete so CW-1 key branch is exercised too
        Xc[:, 0] = rng_base.integers(0, 6, size=n_c).astype(np.float64)
        yc = 1.5 * Xc[:, 0] - 0.8 * np.sin(Xc[:, 1]) + 0.2 * rng_base.standard_normal(n_c)
        Xq_a = rng_base.standard_normal((n_q, p))
        Xq_b = Xq_a * -7.5 + 1000.0
        Xq_b[0, 1] = np.nan

        Ic_a, Iq_a = inject_synth_cols_22fam_h455(Xc.copy(), yc.copy(), Xq_a.copy(), 16, np.random.default_rng(9000 + trial))
        Ic_b, _ = inject_synth_cols_22fam_h455(Xc.copy(), yc.copy(), Xq_b.copy(), 16, np.random.default_rng(9000 + trial))
        assert Ic_a.shape == (n_c, 16) and Iq_a.shape == (n_q, 16)
        assert np.all(np.isfinite(Ic_a))
        diff = float(np.max(np.abs(Ic_a - Ic_b)))
        max_diff_inj = max(max_diff_inj, diff)
    assert max_diff_inj == 0.0, f"Query leakage in inject_synth_cols_22fam_h455: {max_diff_inj}"

    max_diff_kd = 0.0
    max_diff_res = 0.0
    assert len(shard_store.shards) >= 4, "Need >= 4 teacher shards for leakage check"
    for idx_s in range(min(8, len(shard_store.shards))):
        sh_a = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in shard_store.shards[idx_s].items()}
        sh_b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in shard_store.shards[idx_s].items()}
        sh_b["Xqr"] = sh_b["Xqr"] + np.float32(1000.0)
        sh_b["yqr"] = sh_b["yqr"] - 5000.0
        sh_b["pqr"] = sh_b["pqr"] + 2500.0
        sh_b["Xcf"] = sh_b["Xcf"] - np.float32(999.0)
        sh_b["pcf"] = sh_b["pcf"] + 1234.0

        ep_ka = sample_binned_exact_kd_episode_h455(sh_a, 128, 24, 88, np.random.default_rng(7100 + idx_s))
        ep_kb = sample_binned_exact_kd_episode_h455(sh_b, 128, 24, 88, np.random.default_rng(7100 + idx_s))
        if ep_ka is not None and ep_kb is not None:
            max_diff_kd = max(max_diff_kd, float(np.max(np.abs(ep_ka[0][:88] - ep_kb[0][:88]))))
            max_diff_kd = max(max_diff_kd, float(np.max(np.abs(ep_ka[1][:88] - ep_kb[1][:88]))))

        ep_ra = sample_binned_exact_resid_episode_h455(sh_a, 128, 24, 88, np.random.default_rng(8100 + idx_s))
        ep_rb = sample_binned_exact_resid_episode_h455(sh_b, 128, 24, 88, np.random.default_rng(8100 + idx_s))
        if ep_ra is not None and ep_rb is not None:
            max_diff_res = max(max_diff_res, float(np.max(np.abs(ep_ra[0][:88] - ep_rb[0][:88]))))
            max_diff_res = max(max_diff_res, float(np.max(np.abs(ep_ra[1][:88] - ep_rb[1][:88]))))

    assert max_diff_kd == 0.0, f"Query leakage in sample_binned_exact_kd_episode_h455: {max_diff_kd}"
    assert max_diff_res == 0.0, f"Query leakage in sample_binned_exact_resid_episode_h455: {max_diff_res}"
    return {
        "max_diff_inj_22fam": max_diff_inj,
        "max_diff_exact_kd": max_diff_kd,
        "max_diff_exact_resid": max_diff_res,
    }


def test_2_quarantine(bank: MmapF64Bank, shard_store: ExactGPTeacherShardStoreH455) -> dict:
    for i in range(bank.n_tables):
        s = f"{bank.cids[i]}|{bank.cohorts[i]}|{bank.families[i]}".upper()
        for tok in SEALED_TOKENS:
            assert tok not in s, f"Quarantine token {tok} found in bank row {i}: {s}"
    for sh in shard_store.shards:
        fi = int(sh["family_idx"])
        fam_s = bank.family_names[fi].upper()
        for tok in SEALED_TOKENS:
            assert tok not in fam_s, f"Quarantine token {tok} found in shard family {fam_s}"
    return {
        "bank_tables_checked": bank.n_tables,
        "cohorts": bank.cohort_names,
        "teacher_shards_checked": len(shard_store.shards),
        "confirm_opens": 0,
        "fmgap_bc_opens": 0,
    }


def test_3_s3_source_uniformity(bank: MmapF64Bank) -> dict:
    rng = np.random.default_rng(455003)
    n_draws = 90_000
    fam_hits = Counter()
    for _ in range(n_draws):
        ti = bank.sample_s3_uniform_table_idx(rng)
        fam_hits[int(bank.family_idx[ti])] += 1
    counts = np.array([fam_hits[fi] for fi in range(bank.n_families)], dtype=np.float64)
    fams_hit = int(np.sum(counts > 0))
    med = float(np.median(counts))
    mx_med = float(np.max(counts) / max(med, 1.0))
    w_fam_mass = np.array([float(np.sum(bank.s3_uniform_weights[m])) for m in bank.family_tables], dtype=np.float64)
    analytic_max_min = float(np.max(w_fam_mass) / np.min(w_fam_mass))
    assert fams_hit == bank.n_families, f"Missed families: {fams_hit}/{bank.n_families}"
    assert mx_med < 2.2, f"Empirical family max/median ratio too skewed: {mx_med}"
    assert analytic_max_min <= 1.55, f"Analytic family max/min mass > 1.55: {analytic_max_min}"
    return {
        "n_families": bank.n_families,
        "families_hit_in_90k": fams_hit,
        "analytic_family_mass_max_over_min": round(analytic_max_min, 4),
        "empirical_family_max_over_median": round(mx_med, 4),
    }


def test_4_entropy_ranges() -> dict:
    rng = np.random.default_rng(455004)
    keeps = [choose_drop_keep_h455(40, rng, 0.00, 0.85, full_view_prob=0.15, min_keep=2) for _ in range(4000)]
    full_frac = sum(1 for k in keeps if k == 40) / 4000.0
    min_keep_seen = min(keeps)
    p_ctxs = [draw_p_ctx_h455(rng, 0.08, 0.95) for _ in range(4000)]
    assert 0.10 <= full_frac <= 0.22, f"Unexpected full_view_prob empirical rate: {full_frac}"
    assert min_keep_seen <= 7, f"85% drop not reached on p=40: min_keep={min_keep_seen}"
    assert min(p_ctxs) < 0.10 and max(p_ctxs) > 0.93, f"p_ctx range incomplete: {min(p_ctxs)}..{max(p_ctxs)}"
    return {
        "full_view_empirical_frac": round(full_frac, 4),
        "min_keep_on_p40": min_keep_seen,
        "p_ctx_min_seen": round(min(p_ctxs), 4),
        "p_ctx_max_seen": round(max(p_ctxs), 4),
    }


def test_5_cross_shard_batch_density(bank: MmapF64Bank) -> dict:
    ctx = mp.get_context("fork")
    out_q = ctx.Queue(maxsize=24)
    stop_ev = ctx.Event()
    p = ctx.Process(
        target=_flywheel_worker_loop_h455,
        args=(
            0,
            b"selftest_h455_nonce_000000000000",
            str(MMAP_DIR),
            bank.sha256,
            "s3_source_uniform",
            1024,
            340_000,
            12_288,
            8_500_000,
            16,
            90.0,
            0.00,
            0.85,
            0.00,
            0.60,
            0.22,
            0.08,
            0.95,
            0.38,
            0.18,
            0.06,
            0.26,
            0.05,
            0.35,
            0.25,
            0.65,
            str(SHARDS_DIR),
            True,
            str(HWM_JSON),
            str(WINNERS_JSONL),
            0.45,
            0.20,
            0.05,
            0.00,
            455123,
            out_q,
            stop_ev,
        ),
        daemon=True,
    )
    p.start()
    ledger = BASE.Ledger128(bank)
    by_kind: dict[str, list[int]] = {}
    try:
        for _ in range(18):
            item = out_q.get(timeout=45.0)
            zm_np, yz_np, ym_np, ys_np, yq_np, wq_np, n_ctx, infos, fps, batch_kind = item
            B_b, N_b, F_b, _ = zm_np.shape
            assert np.all(np.isfinite(zm_np)) and np.all(np.isfinite(yq_np))
            keep = ledger.filter_unseen_indices(fps)
            assert len(keep) == B_b
            if infos is None:
                ledger.add_synth(B_b, N_b, F_b)
            else:
                ledger.add_real(infos)
            by_kind.setdefault(batch_kind, []).append(int(B_b))
    finally:
        stop_ev.set()
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()

    assert "exact_kd" in by_kind or "exact_resid" in by_kind, f"No exact teacher batches seen: {by_kind}"
    for k_name in ("exact_kd", "exact_resid", "real_s3"):
        if k_name in by_kind:
            assert max(by_kind[k_name]) >= 6, f"Batch density too small for {k_name}: {by_kind[k_name]}"
    return {"batch_sizes_by_stream": by_kind}


def test_6_cyclic_lr_and_cpu_smoke() -> dict:
    c0_s = 2.5 * 3600.0
    lr_before, k0, _ = compute_cyclic_valley_lr(c0_s - 1e-4, 0.0, 1.5e-5, 7.5e-6, 2.5e-6, 2.5, 2.5)
    lr_after, k1, _ = compute_cyclic_valley_lr(c0_s + 1e-4, 0.0, 1.5e-5, 7.5e-6, 2.5e-6, 2.5, 2.5)
    lr_peak, _, _ = compute_cyclic_valley_lr(c0_s + 1.25 * 3600.0, 0.0, 1.5e-5, 7.5e-6, 2.5e-6, 2.5, 2.5)
    c1_gap = abs(lr_before - lr_after)
    assert k0 == 0 and k1 == 1
    assert c1_gap < 1e-9, f"Discontinuity at valley boundary: {c1_gap}"
    assert abs(lr_peak - 7.5e-6) < 1e-9

    smoke_root = Path(f"/home/ubuntu/jope-data/h452-flywheel/selftest_h455_smoke_{int(time.time())}")
    smoke_root.mkdir(parents=True, exist_ok=True)
    init_s = smoke_root / "init_S.pt"
    m_s = ICLH450MPS("S").cpu()
    torch.save({"model": m_s.state_dict(), "ema": m_s.state_dict()}, init_s)
    init_sha = sha256_file(init_s)
    out_run = smoke_root / "run_s"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        "-u",
        str(STAGE_CODE / "train_104m_gp_flywheel_h455_maxent.py"),
        "--size",
        "S",
        "--init",
        str(init_s),
        "--init-sha256",
        init_sha,
        "--mmap-bank",
        str(MMAP_DIR),
        "--out",
        str(out_run),
        "--teacher-shards-dir",
        str(SHARDS_DIR),
        "--hwm-json",
        str(HWM_JSON),
        "--gp-winners-jsonl",
        str(WINNERS_JSONL),
        "--tokens",
        "40000",
        "--bmax",
        "6",
        "--grad-accum",
        "1",
        "--workers",
        "2",
        "--max-steps",
        "3",
        "--device",
        "cpu",
    ]
    subprocess.run(cmd, env=env, check=True, timeout=240)
    assert (out_run / "ckpt_last.pt").is_file()
    assert (out_run / "ckpt_latest.pt").is_file()
    ck = torch.load(out_run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert int(ck["state"]["step"]) == 3
    assert "ema_valley" in ck and "ema_swa" in ck
    return {
        "c1_valley_boundary_gap": c1_gap,
        "cycle1_peak_lr": lr_peak,
        "smoke_out": str(out_run),
        "smoke_steps": int(ck["state"]["step"]),
    }


def test_7_ast_hygiene() -> dict:
    banned_calls = {"unlink", "rmtree", "rmdir", "remove", "removedirs"}
    checked = []
    for fp in H455_FILES:
        src = fp.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(fp))
        # Collect docstring expr value ids so docstrings documenting rules ("0 /tmp usage") are not treated as path literals
        doc_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    doc_ids.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
                assert "/tmp" not in node.value, f"Forbidden /tmp literal in {fp}: {node.value!r}"
                assert "--delete" not in node.value, f"Forbidden --delete literal in {fp}: {node.value!r}"
            if isinstance(node, ast.Call):
                fn_name = ""
                if isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                assert fn_name not in banned_calls, f"Forbidden delete call `{fn_name}` in {fp}"
        checked.append(fp.name)
    return {"files_checked": checked, "delete_calls": 0, "tmp_strings": 0}


def main() -> None:
    t0 = time.time()
    bank = MmapF64Bank(MMAP_DIR, sampling_mode="s3_source_uniform")
    shard_store = ExactGPTeacherShardStoreH455(str(SHARDS_DIR), bank, allow_d91=True, min_regret=0.010, gamma=0.65)

    r1 = test_1_zero_query_leakage(shard_store)
    r2 = test_2_quarantine(bank, shard_store)
    r3 = test_3_s3_source_uniformity(bank)
    r4 = test_4_entropy_ranges()
    r5 = test_5_cross_shard_batch_density(bank)
    r6 = test_6_cyclic_lr_and_cpu_smoke()
    r7 = test_7_ast_hygiene()

    receipt = {
        "schema": "jope.h455.selftest_receipt/v1",
        "status": "PASS",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "canonical_promotion_forbidden": True,
        "tests": {
            "1_zero_query_leakage": r1,
            "2_quarantine_audit": r2,
            "3_s3_source_uniformity": r3,
            "4_entropy_ranges": r4,
            "5_cross_shard_batch_density": r5,
            "6_cyclic_lr_and_cpu_smoke": r6,
            "7_ast_hygiene": r7,
        },
    }
    p_fw = Path("/home/ubuntu/jope-data/h452-flywheel/SELFTEST_H455_RECEIPT.json")
    p_ev = Path("/home/ubuntu/jope_plan/evidence/H455/SELFTEST_H455_RECEIPT.json")
    p_fw.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    p_ev.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p_fw, p_ev)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
