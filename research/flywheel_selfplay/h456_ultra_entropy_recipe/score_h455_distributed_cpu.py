#!/usr/bin/env python3
"""Distributed `xbabe0` + `xbabe2` Memory-Capped MKL CPU Scorer (`score_h455_distributed_cpu.py`).

Hard safety invariants:
  - `xbabe0`: Pure CPU (`CUDA_VISIBLE_DEVICES=""`, zero GPU/driver touch), `24` 3-thread MKL shards inside
    `systemd-run --user --scope -p MemoryHigh=48G -p MemoryMax=58G choom -n 500 --`.
  - `xbabe2`: Dynamic reverse-depth tail-stealing `.venv` MKL workers (`2` workers for `L` = `~3.2 GiB` RSS;
    `4` workers for `M` = `~3.2 GiB` RSS) inside `systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=14G
    choom -n 500 --`, `taskset -c 32-63,80-127` (leaving `0-31` for `fleet_M_x2_h455_maxent` and `64-79` clear
    for `cl_fwriter`), with per-fold `91/91` instant abort so workers never linger past `91/91`.
  - Zero trainer stops, zero `CONFIRM` / `FMGAP B/C` / `HOLDOUT` / `SEALED` opens, zero deletions, zero temp-root paths.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

FLYWHEEL_ROOT = Path("/home/ubuntu/jope-data/h452-flywheel")
STAGE_CODE = Path("/home/ubuntu/h452-stage/code")
BANK_D91 = Path("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
EVIDENCE_H455 = Path("/home/ubuntu/jope_plan/evidence/H455")

CV7_HYBRID_COLS = (
    "icl_e8_cw12_robridge_cv7",
    "icl_e8_cw12_yj_cv7",
    "icl_e8_rankgauss_duan_cv7",
    "icl_e8_gp_resid_cv7",
)


def dist_file_sha256(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dist_count_npz(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(1 for p in out_dir.glob("*.npz") if p.is_file() and not p.name.startswith("."))


def dist_mem_available_gib() -> float:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemAvailable:"):
            kb = float(ln.split()[1])
            return round(kb / (1024.0 * 1024.0), 2)
    return -1.0


def build_slim_soup50_ckpt(src_ckpt: Path, dst_slim: Path, expected_size: str) -> dict:
    dst_slim.parent.mkdir(parents=True, exist_ok=True)
    src_sha = dist_file_sha256(src_ckpt)
    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    sz = str(ck.get("size", expected_size)).strip().upper()
    st = dict(ck.get("state", {}))
    if not dst_slim.is_file():
        cd = ck.get("ema_cooldown")
        sw = ck.get("ema_swa")
        if cd is not None and sw is not None:
            soup = {k: (0.5 * cd[k].float() + 0.5 * sw[k].float()).to(dtype=cd[k].dtype) for k in cd}
        else:
            base = ck.get("ema_cooldown") or ck.get("ema_valley") or ck.get("ema_swa") or ck.get("ema") or ck["model"]
            soup = {k: v for k, v in base.items()}
        slim_obj = {"ema_soup50": soup, "size": sz, "state": st, "source_sha256": src_sha}
        tmp_p = dst_slim.parent / f".{dst_slim.name}.{os.getpid()}.tmp"
        torch.save(slim_obj, tmp_p)
        os.replace(tmp_p, dst_slim)
    del ck
    return {
        "source_path": str(src_ckpt),
        "source_sha256": src_sha,
        "slim_path": str(dst_slim),
        "slim_sha256": dist_file_sha256(dst_slim),
        "size": sz,
        "state": st,
    }


def sync_stage_inputs_to_xbabe0(extra_dirs: list[Path]) -> None:
    remote_mkdirs = [
        "/home/ubuntu/h452-stage/code",
        "/home/ubuntu/jope-data/h452-flywheel/gp_anchors_p0",
        "/home/ubuntu/jope-data/h452-flywheel/gp_anchors_oof_cache_v1",
        "/home/ubuntu/jope-data/h452-flywheel/ckpts_h455_slim_eval",
        "/home/ubuntu/jope-data/h452-flywheel/logs",
        "/home/ubuntu/jope_plan/evidence/H418/pulled",
        "/home/ubuntu/jope_plan/evidence/H448/pulled",
    ] + [str(d) for d in extra_dirs]
    subprocess.run(
        ["ssh", "xbabe0", "mkdir -p " + " ".join(remote_mkdirs)],
        check=True,
        timeout=30,
    )
    file_pairs = [
        (str(STAGE_CODE / "cols_104m_e8_gpresid_h452.py"), "xbabe0:/home/ubuntu/h452-stage/code/cols_104m_e8_gpresid_h452.py"),
        (str(STAGE_CODE / "run_local_shard_pool.py"), "xbabe0:/home/ubuntu/h452-stage/code/run_local_shard_pool.py"),
        (str(STAGE_CODE / "steal_distributed_tails_cpu.py"), "xbabe0:/home/ubuntu/h452-stage/code/steal_distributed_tails_cpu.py"),
        (str(FLYWHEEL_ROOT / "ckpts_h455_slim_eval") + "/", "xbabe0:/home/ubuntu/jope-data/h452-flywheel/ckpts_h455_slim_eval/"),
    ]
    for d in extra_dirs:
        d.mkdir(parents=True, exist_ok=True)
        file_pairs.append((str(d) + "/", f"xbabe0:{d}/"))
    for src, dst in file_pairs:
        subprocess.run(["rsync", "-az", src, dst], check=True, timeout=300)


def load_d91_shapes_for_lpt() -> list[tuple[int, str, int, int]]:
    with np.load(BANK_D91, allow_pickle=False) as zb:
        assert "ya" not in zb.files
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        out = []
        for i, cid in enumerate(cids):
            ntot = int(ss[i, 0]) + int(sa[i, 0])
            f_raw = int(sa[i, 1])
            out.append((i, cid, ntot, f_raw))
        return out


def assign_lpt_buckets(
    pending_items: list[tuple[int, float]],
    n_xbabe0: int,
) -> list[list[int]]:
    """Assigns `(ctx_idx, cost)` items via LPT min-heap across `n_xbabe0` primary `xbabe0` slots."""
    sorted_items = sorted(pending_items, key=lambda x: -x[1])
    buckets: list[list[int]] = [[] for _ in range(n_xbabe0)]
    heap: list[tuple[float, int]] = [(0.0, s) for s in range(n_xbabe0)]
    heapq.heapify(heap)
    for idx, cost in sorted_items:
        load, slot_i = heapq.heappop(heap)
        buckets[slot_i].append(idx)
        heapq.heappush(heap, (load + cost, slot_i))
    return [sorted(b) for b in buckets if b]


def pull_stage_dirs_from_xbabe0(out_dirs: list[Path]) -> None:
    for od in out_dirs:
        subprocess.run(["rsync", "-az", f"xbabe0:{od}/", f"{od}/"], check=False, timeout=90)


def run_distributed_stage(
    stage_name: str,
    multi_specs: str,
    out_dirs: list[Path],
    n_xbabe0: int,
    n_xbabe2_stealers: int,
) -> float:
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)
    if all(dist_count_npz(od) >= 91 for od in out_dirs):
        return 0.0

    shapes = load_d91_shapes_for_lpt()
    pending: list[tuple[int, float]] = []
    for idx, cid, ntot, f_raw in shapes:
        missing = sum(1 for od in out_dirs if not (od / f"{cid}.npz").is_file())
        if missing == 0:
            continue
        cost = float(missing) * float(ntot) * float(min(64, f_raw + 22))
        pending.append((idx, cost))

    sync_stage_inputs_to_xbabe0(out_dirs)
    xb0_buckets = assign_lpt_buckets(pending, n_xbabe0=n_xbabe0)

    cfg_dir = FLYWHEEL_ROOT / "logs" / f"dist_cfg_{stage_name}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_xb0 = cfg_dir / "xbabe0_pool.json"
    claims_dir = cfg_dir / "tail_claims_v1"

    cfg_xb0.write_text(
        json.dumps(
            {
                "python_bin": "/usr/bin/python3",
                "threads": 3,
                "affinity": "",
                "multi_specs": multi_specs,
                "shards": [",".join(str(x) for x in b) for b in xb0_buckets],
                "stagger_s": 0.45,
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(["ssh", "xbabe0", f"mkdir -p {cfg_dir}"], check=True, timeout=20)
    subprocess.run(["rsync", "-az", str(cfg_xb0), f"xbabe0:{cfg_xb0}"], check=True, timeout=30)

    print(
        json.dumps(
            {
                "event": "stage_start",
                "stage": stage_name,
                "pending_contexts": len(pending),
                "xbabe0_shards": len(xb0_buckets),
                "xbabe2_tail_stealers": int(n_xbabe2_stealers),
                "xbabe2_mem_avail_gib": dist_mem_available_gib(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    t0 = time.time()
    procs: list[tuple[str, subprocess.Popen]] = []
    if xb0_buckets:
        r_cmd = (
            "CUDA_VISIBLE_DEVICES='' systemd-run --user --scope -p MemoryHigh=48G -p MemoryMax=58G "
            f"choom -n 500 -- /usr/bin/python3 -u /home/ubuntu/h452-stage/code/run_local_shard_pool.py --spec-json {cfg_xb0}"
        )
        procs.append(("xbabe0", subprocess.Popen(["ssh", "xbabe0", r_cmd])))

    if n_xbabe2_stealers > 0 and xb0_buckets:
        l_cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "MemoryHigh=10G",
            "-p",
            "MemoryMax=14G",
            "choom",
            "-n",
            "500",
            "--",
            "/home/ubuntu/.venv/bin/python3",
            "-u",
            str(STAGE_CODE / "steal_distributed_tails_cpu.py"),
            "--stage-cfg",
            str(cfg_xb0),
            "--claims-dir",
            str(claims_dir),
            "--affinity",
            "32-63,80-127",
            "--workers",
            str(n_xbabe2_stealers),
            "--threads",
            "4",
            "--remote-host",
            "xbabe0",
        ]
        env_l = os.environ.copy()
        env_l["CUDA_VISIBLE_DEVICES"] = ""
        procs.append(("xbabe2_stealer", subprocess.Popen(l_cmd, env=env_l)))

    while True:
        time.sleep(20.0)
        pull_stage_dirs_from_xbabe0(out_dirs)
        alive = [name for name, p in procs if p.poll() is None]
        counts = {od.name: dist_count_npz(od) for od in out_dirs}
        print(
            json.dumps(
                {
                    "event": "stage_progress",
                    "stage": stage_name,
                    "alive_hosts": alive,
                    "counts": counts,
                    "xbabe2_mem_avail_gib": dist_mem_available_gib(),
                    "elapsed_s": round(time.time() - t0, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not alive:
            break

    pull_stage_dirs_from_xbabe0(out_dirs)
    bad = [(name, p.returncode) for name, p in procs if p.returncode != 0]
    if bad:
        raise SystemExit(f"stage {stage_name} failed on host(s): {bad}")
    for od in out_dirs:
        if dist_count_npz(od) < 91:
            raise SystemExit(f"stage {stage_name} incomplete directory {od}: {dist_count_npz(od)}/91")
    return round(time.time() - t0, 2)


def build_weighted_prediction_soup(weighted_dirs: list[tuple[Path, float]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dist_count_npz(out_dir) >= 91:
        return 91
    first_dir = weighted_dirs[0][0]
    first_files = sorted(p.name for p in first_dir.glob("*.npz") if not p.name.startswith("."))
    if len(first_files) < 91:
        raise SystemExit(f"expected 91 .npz files in {first_dir}, found {len(first_files)}")
    weights = np.array([float(w) for _, w in weighted_dirs], dtype=np.float64)
    weights /= float(np.sum(weights))
    built = 0
    for fname in first_files:
        loaded = [np.load(sd / fname, allow_pickle=False) for sd, _ in weighted_dirs]
        out_arrays: dict[str, np.ndarray] = {"ok": np.array(True)}
        for col in CV7_HYBRID_COLS:
            for pref in ("q_", "bag_", "ref_"):
                k = f"{pref}{col}"
                mat = np.column_stack([np.asarray(z[k], dtype=np.float64) for z in loaded])
                out_arrays[k] = mat @ weights
        for z in loaded:
            z.close()
        tmp_p = out_dir / f".{fname}.{os.getpid()}.tmp.npz"
        np.savez(tmp_p, **out_arrays)
        os.replace(tmp_p, out_dir / fname)
        built += 1
    return built


def extract_h455_matrix_metrics(matrix_path: Path) -> dict:
    doc = json.loads(matrix_path.read_text())
    tbl = {row["arm"]: row for row in doc.get("summary_table", [])}
    pw = doc.get("paired_comparisons", {})

    def _pick(arm: str) -> dict:
        r = tbl.get(arm, {})
        ci = r.get("ci95", [None, None])
        return {
            "elo": r.get("elo"),
            "ci95": ci,
            "elo_lcb95": ci[0] if isinstance(ci, list) and len(ci) >= 2 else None,
            "elo_ucb95": ci[1] if isinstance(ci, list) and len(ci) >= 2 else None,
            "rank_of_34": r.get("rank_of_34"),
            "family_mean_u": r.get("family_mean_u"),
            "vs_tabpfn_v3p5": r.get("vs_tabpfn_v3p5"),
            "vs_kumo_large": r.get("vs_kumo_large"),
            "vs_tabfm_ens": r.get("vs_tabfm_ens"),
        }

    post_or_joint_rows = [
        r
        for r in doc.get("summary_table", [])
        if ("post_h454" in r.get("arm", "") or "joint_pre_post" in r.get("arm", ""))
    ]
    post_only_rows = [r for r in doc.get("summary_table", []) if "post_h454" in r.get("arm", "")]
    best_arm_row = max(post_or_joint_rows, key=lambda x: float(x.get("elo", -1e9)))
    best_post_only_row = max(post_only_rows, key=lambda x: float(x.get("elo", -1e9)))
    b_ci = best_arm_row.get("ci95", [None, None])
    return {
        "matrix_json": str(matrix_path),
        "matrix_sha256": dist_file_sha256(matrix_path),
        "best_post_arm": best_arm_row.get("arm"),
        "best_post_elo": best_arm_row.get("elo"),
        "best_post_lcb95": b_ci[0] if isinstance(b_ci, list) and len(b_ci) >= 2 else None,
        "best_post_ucb95": b_ci[1] if isinstance(b_ci, list) and len(b_ci) >= 2 else None,
        "best_post_rank_of_34": best_arm_row.get("rank_of_34"),
        "best_post_utility": best_arm_row.get("family_mean_u"),
        "best_post_only_arm": best_post_only_row.get("arm"),
        "best_post_only_elo": best_post_only_row.get("elo"),
        "arms": {
            "pure_104m_cv7_post": _pick("pure_104m_cv7_post_h454"),
            "hyb_104m_only_cv7_post": _pick("hyb_104m_only_cv7_post_h454"),
            "hyb_104m_gpresid_cv7_post": _pick("hyb_104m_gpresid_cv7_post_h454"),
            "neural_104m_all_post": _pick("neural_104m_all_post_h454"),
            "neural_104m_all_pre": _pick("neural_104m_all_pre_h454"),
            "neural104m_ugp_g1000_cv7_post": _pick("neural104m_ugp_g1000_cv7_post_h454"),
            "neural104m_ugp_g1000_2xcv7_post": _pick("neural104m_ugp_g1000_2xcv7_post_h454"),
            "neural104m_ugp_g1000_cons_post": _pick("neural104m_ugp_g1000_cons_post_h454"),
            "stack_pure104m_no_ugp_post": _pick("stack_pure104m_no_ugp_post_h454"),
            "stack_pure104m_ugp_g10_cv7_post": _pick("stack_pure104m_ugp_g10_cv7_post_h454"),
            "stack_no_ugp_post": _pick("stack_no_ugp_post_h454"),
            "stack_ugp_g1000_cv7_post": _pick("stack_ugp_g1000_cv7_post_h454"),
            "stack_ugp_g1000_2xcv7_post": _pick("stack_ugp_g1000_2xcv7_post_h454"),
            "stack_ugp_g1000_cons_post": _pick("stack_ugp_g1000_cons_post_h454"),
            "stack_guarded_hyb_ugp_g1000_cons_post": _pick("stack_guarded_hyb_ugp_g1000_cons_post_h454"),
            "stack_no_ugp_joint": _pick("stack_no_ugp_joint_pre_post"),
            "stack_ugp_g1000_2xcv7_joint": _pick("stack_ugp_g1000_2xcv7_joint_pre_post"),
            "stack_ugp_g1000_cons_joint": _pick("stack_ugp_g1000_cons_joint_pre_post"),
            "stack_guarded_hyb_ugp_g1000_cons_joint": _pick("stack_guarded_hyb_ugp_g1000_cons_joint_pre_post"),
        },
        "paired_comparisons": pw,
    }


def score_jobs_memory_capped(jobs: list[tuple[str, Path, Path]], traj_post_dir: Path, n_boot: int = 2000) -> dict[str, dict]:
    scorer = STAGE_CODE / "score_h454_paired_flywheel_matrix.py"
    env_sc = os.environ.copy()
    env_sc["CUDA_VISIBLE_DEVICES"] = ""
    results: dict[str, dict] = {}
    for label, post_dir, out_json in jobs:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        reused = False
        ts0 = time.time()
        if out_json.is_file() and out_json.stat().st_size > 10000:
            try:
                existing_doc = json.loads(out_json.read_text())
                if existing_doc.get("summary_table"):
                    reused = True
            except (OSError, json.JSONDecodeError):
                reused = False
        if not reused:
            subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    "--scope",
                    "-p",
                    "MemoryHigh=10G",
                    "-p",
                    "MemoryMax=14G",
                    "choom",
                    "-n",
                    "500",
                    "--",
                    "taskset",
                    "-c",
                    "32-63,80-127",
                    "nice",
                    "-n",
                    "19",
                    sys.executable,
                    str(scorer),
                    "--cv7-post-dir",
                    str(post_dir),
                    "--traj-post-dir",
                    str(traj_post_dir),
                    "--procs",
                    "28",
                    "--n-boot",
                    str(n_boot),
                    "--out-json",
                    str(out_json),
                ],
                env=env_sc,
                check=True,
            )
        rec = extract_h455_matrix_metrics(out_json)
        rec["score_elapsed_s"] = 0.0 if reused else round(time.time() - ts0, 2)
        rec["reused_existing_json"] = reused
        results[label] = rec
        print(
            json.dumps(
                {
                    "event": "matrix_scored",
                    "label": label,
                    "reused": reused,
                    "best_post_arm": rec["best_post_arm"],
                    "best_post_elo": rec["best_post_elo"],
                    "neural_104m_all_post_elo": rec["arms"]["neural_104m_all_post"]["elo"],
                    "stack_2xcv7_elo": rec["arms"]["stack_ugp_g1000_2xcv7_post"]["elo"],
                    "stack_cons_post_elo": rec["arms"]["stack_ugp_g1000_cons_post"]["elo"],
                    "stack_cons_joint_elo": rec["arms"]["stack_ugp_g1000_cons_joint"]["elo"],
                    "stack_no_ugp_elo": rec["arms"]["stack_no_ugp_post"]["elo"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xbabe0-shards", type=int, default=24)
    ap.add_argument("--xbabe2-stealers-m", type=int, default=4)
    ap.add_argument("--xbabe2-stealers-l", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--stage1-only", type=int, default=0)
    a = ap.parse_args()

    EVIDENCE_H455.mkdir(parents=True, exist_ok=True)
    cols_root = FLYWHEEL_ROOT / "cols_e8_d91"
    traj_post_dir = cols_root / "traj_e8_w2aplus"
    slim_dir = FLYWHEEL_ROOT / "ckpts_h455_slim_eval"

    m_v1_src = FLYWHEEL_ROOT / "ckpts_h455_m_valleys" / "fleet_M_x2_h455_valley_001.pt"
    m_v2_src = FLYWHEEL_ROOT / "ckpts_h455_m_valleys" / "fleet_M_x2_h455_valley_002.pt"
    l_v1_src = FLYWHEEL_ROOT / "ckpts_h455_ckpt_valley_001" / "fleet_L_x1_h455_ckpt_valley_001.pt"

    m_v1_slim = slim_dir / "m_x2_v1_soup50_slim.pt"
    m_v2_slim = slim_dir / "m_x2_v2_soup50_slim.pt"
    l_v1_slim = slim_dir / "l_x1_v1_soup50_slim.pt"

    meta_m1 = build_slim_soup50_ckpt(m_v1_src, m_v1_slim, "M")
    meta_m2 = build_slim_soup50_ckpt(m_v2_src, m_v2_slim, "M")
    meta_l1 = build_slim_soup50_ckpt(l_v1_src, l_v1_slim, "L")

    # =========================================================================
    # STAGE 1: fleet_M_x2_h455_maxent valleys 001 & 002 + L+M cross-scale soups
    # =========================================================================
    d_m_v1 = cols_root / "cv7_hyb_m_h455_v1_soup50"
    d_m_v2 = cols_root / "cv7_hyb_m_h455_v2_soup50"
    multi_m = ";".join(
        [
            f"m_v1_soup50|ema_soup50|M:{m_v1_slim}|{d_m_v1}",
            f"m_v2_soup50|ema_soup50|M:{m_v2_slim}|{d_m_v2}",
        ]
    )
    elapsed_m = run_distributed_stage(
        stage_name="stage1_m_valleys",
        multi_specs=multi_m,
        out_dirs=[d_m_v1, d_m_v2],
        n_xbabe0=int(a.xbabe0_shards),
        n_xbabe2_stealers=int(a.xbabe2_stealers_m),
    )

    d_m_v12 = cols_root / "cv7_hyb_m_h455_v12_avg"
    build_weighted_prediction_soup([(d_m_v1, 0.50), (d_m_v2, 0.50)], d_m_v12)

    w2_base = cols_root / "cv7_hyb_w2_cooled_soup_avg"
    m_jobs: list[tuple[str, Path, Path]] = [
        ("m_x2_h455_valley001_soup50", d_m_v1, EVIDENCE_H455 / "H455_M_X2_VALLEY001_SOUP50_MATRIX.json"),
        ("m_x2_h455_valley002_soup50", d_m_v2, EVIDENCE_H455 / "H455_M_X2_VALLEY002_SOUP50_MATRIX.json"),
        ("m_x2_h455_valley001_002_avg", d_m_v12, EVIDENCE_H455 / "H455_M_X2_VALLEY001_002_AVG_MATRIX.json"),
    ]
    if w2_base.is_dir() and dist_count_npz(w2_base) >= 91:
        d_lm_80_20 = cols_root / "cv7_hyb_lw2_mh455_v2_80_20"
        d_lm_65_35 = cols_root / "cv7_hyb_lw2_mh455_v2_65_35"
        build_weighted_prediction_soup([(w2_base, 0.80), (d_m_v2, 0.20)], d_lm_80_20)
        build_weighted_prediction_soup([(w2_base, 0.65), (d_m_v2, 0.35)], d_lm_65_35)
        m_jobs.extend(
            [
                ("lw2_cooled_plus_m_h455_v2_80_20", d_lm_80_20, EVIDENCE_H455 / "H455_LW2_MH455_V2_80_20_MATRIX.json"),
                ("lw2_cooled_plus_m_h455_v2_65_35", d_lm_65_35, EVIDENCE_H455 / "H455_LW2_MH455_V2_65_35_MATRIX.json"),
            ]
        )

    m_res = score_jobs_memory_capped(m_jobs, traj_post_dir=traj_post_dir, n_boot=int(a.n_boot))
    best_m_only = max((k for k in m_res if k.startswith("m_x2_h455_")), key=lambda k: float(m_res[k]["best_post_elo"] or -1e9))
    best_m_all = max(m_res.keys(), key=lambda k: float(m_res[k]["best_post_elo"] or -1e9))
    m_summary_p = EVIDENCE_H455 / "H455_M_X2_VALLEY_PROGRESSION_SUMMARY.json"
    if not (m_summary_p.is_file() and m_summary_p.stat().st_size > 2000):
        m_summary = {
            "schema": "jope.h455.m_x2_valley_progression/v1",
            "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
            "in_corpus_dev91_disclosure": True,
            "canonical_promotion_forbidden": True,
            "prior_wave2_cooled_soup_best_elo": 1859.63,
            "delta_vs_prior_best_elo": round(float(m_res[best_m_all]["best_post_elo"]) - 1859.63, 2),
            "quarantine": {"confirm_opens": 0, "fmgap_bc_opens": 0, "holdout_opens": 0, "sealed_opens": 0},
            "eval_hosts": {
                "xbabe0_shards": int(a.xbabe0_shards),
                "xbabe2_tail_stealers": int(a.xbabe2_stealers_m),
                "device": "cpu",
            },
            "stage1_eval_elapsed_s": elapsed_m,
            "checkpoints": {"valley_001": meta_m1, "valley_002": meta_m2},
            "best_standalone_m_label": best_m_only,
            "best_standalone_m_arm": m_res[best_m_only]["best_post_arm"],
            "best_standalone_m_elo": m_res[best_m_only]["best_post_elo"],
            "best_overall_label": best_m_all,
            "best_overall_arm": m_res[best_m_all]["best_post_arm"],
            "best_overall_elo": m_res[best_m_all]["best_post_elo"],
            "best_overall_lcb95": m_res[best_m_all]["best_post_lcb95"],
            "best_overall_ucb95": m_res[best_m_all]["best_post_ucb95"],
            "best_overall_utility": m_res[best_m_all]["best_post_utility"],
            "waves": m_res,
        }
        m_tmp = m_summary_p.parent / f".{m_summary_p.name}.{os.getpid()}.tmp"
        m_tmp.write_text(json.dumps(m_summary, indent=2, sort_keys=True) + "\n")
        os.replace(m_tmp, m_summary_p)
    print(f"STAGE1_SUMMARY_WRITTEN: {m_summary_p}", flush=True)
    if int(a.stage1_only) == 1:
        return

    # =========================================================================
    # STAGE 2: fleet_L_x1_h455_maxent_v2 valley 001 + L/M multi-scale soups
    # =========================================================================
    d_l_v1 = cols_root / "cv7_hyb_l_x1_h455_v1_soup50"
    multi_l = f"l_v1_soup50|ema_soup50|L:{l_v1_slim}|{d_l_v1}"
    elapsed_l = run_distributed_stage(
        stage_name="stage2_l_valley001",
        multi_specs=multi_l,
        out_dirs=[d_l_v1],
        n_xbabe0=int(a.xbabe0_shards),
        n_xbabe2_stealers=int(a.xbabe2_stealers_l),
    )

    l_jobs: list[tuple[str, Path, Path]] = [
        ("l_x1_h455_v2_valley001_soup50", d_l_v1, EVIDENCE_H455 / "H455_L_X1_V2_VALLEY001_SOUP50_MATRIX.json"),
    ]
    w2a_dir = cols_root / "cv7_hyb_w2a_soup50"
    if w2a_dir.is_dir() and dist_count_npz(w2a_dir) >= 91:
        d_l_w2a = cols_root / "cv7_hyb_lh455v1_w2a_50_50"
        build_weighted_prediction_soup([(d_l_v1, 0.50), (w2a_dir, 0.50)], d_l_w2a)
        l_jobs.append(("lh455v1_plus_w2a_50_50", d_l_w2a, EVIDENCE_H455 / "H455_LH455V1_W2A_50_50_MATRIX.json"))

    d_l_mv12 = cols_root / "cv7_hyb_lh455v1_mh455v12_50_50"
    build_weighted_prediction_soup([(d_l_v1, 0.50), (d_m_v12, 0.50)], d_l_mv12)
    l_jobs.append(("lh455v1_plus_mh455v12_50_50", d_l_mv12, EVIDENCE_H455 / "H455_LH455V1_MH455V12_50_50_MATRIX.json"))

    if w2_base.is_dir() and dist_count_npz(w2_base) >= 91:
        d_lw2_l_50 = cols_root / "cv7_hyb_lw2_lh455v1_50_50"
        d_lw2_l_65 = cols_root / "cv7_hyb_lw2_lh455v1_65_35"
        d_tri = cols_root / "cv7_hyb_tri_lw2_lh455v1_mh455v12_45_30_25"
        build_weighted_prediction_soup([(w2_base, 0.50), (d_l_v1, 0.50)], d_lw2_l_50)
        build_weighted_prediction_soup([(w2_base, 0.65), (d_l_v1, 0.35)], d_lw2_l_65)
        build_weighted_prediction_soup([(w2_base, 0.45), (d_l_v1, 0.30), (d_m_v12, 0.25)], d_tri)
        l_jobs.extend(
            [
                ("lw2_cooled_plus_lh455v1_50_50", d_lw2_l_50, EVIDENCE_H455 / "H455_LW2_LH455V1_50_50_MATRIX.json"),
                ("lw2_cooled_plus_lh455v1_65_35", d_lw2_l_65, EVIDENCE_H455 / "H455_LW2_LH455V1_65_35_MATRIX.json"),
                ("tri_lw2_lh455v1_mh455v12_45_30_25", d_tri, EVIDENCE_H455 / "H455_TRI_LW2_LH455V1_MH455V12_MATRIX.json"),
            ]
        )

    l_res = score_jobs_memory_capped(l_jobs, traj_post_dir=traj_post_dir, n_boot=int(a.n_boot))
    best_l_all = max(l_res.keys(), key=lambda k: float(l_res[k]["best_post_elo"] or -1e9))
    l_summary_p = EVIDENCE_H455 / "H455_L_X1_V2_VALLEY001_SUMMARY.json"
    l_summary = {
        "schema": "jope.h455.l_x1_v2_valley001_progression/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "in_corpus_dev91_disclosure": True,
        "canonical_promotion_forbidden": True,
        "quarantine": {"confirm_opens": 0, "fmgap_bc_opens": 0, "holdout_opens": 0, "sealed_opens": 0},
        "eval_hosts": {
            "xbabe0_shards": int(a.xbabe0_shards),
            "xbabe2_tail_stealers": int(a.xbabe2_stealers_l),
            "device": "cpu",
        },
        "stage2_eval_elapsed_s": elapsed_l,
        "checkpoint": meta_l1,
        "standalone_elo": l_res["l_x1_h455_v2_valley001_soup50"]["best_post_elo"],
        "best_overall_label": best_l_all,
        "best_overall_arm": l_res[best_l_all]["best_post_arm"],
        "best_overall_elo": l_res[best_l_all]["best_post_elo"],
        "best_overall_lcb95": l_res[best_l_all]["best_post_lcb95"],
        "best_overall_ucb95": l_res[best_l_all]["best_post_ucb95"],
        "best_overall_utility": l_res[best_l_all]["best_post_utility"],
        "waves": l_res,
    }
    l_summary_p.write_text(json.dumps(l_summary, indent=2, sort_keys=True) + "\n")
    print(f"STAGE2_SUMMARY_WRITTEN: {l_summary_p}", flush=True)


if __name__ == "__main__":
    main()
