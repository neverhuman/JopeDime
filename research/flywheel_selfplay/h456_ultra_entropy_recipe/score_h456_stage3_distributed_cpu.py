#!/usr/bin/env python3
"""Stage-3 Pure-CPU Multi-Valley Distributed Scorer (`score_h456_stage3_distributed_cpu.py`).

Scores genuine unscored cyclical cosine-valley checkpoints strictly read-only on CPU (`CUDA_VISIBLE_DEVICES=""`):
  - `fleet_M_x2_h455_maxent` `ckpt_valley_003.pt` (`step = 40,525` on `xbabe2`, trainer left 100% untouched)
  - `fleet_L_x1_h455_maxent_v2` `ckpt_valley_004.pt` (`step = 108,437` pulled read-only from `xbabe1`)
  - 5 Stage-3 in-corpus `dev91` BT-Elo matrices (`n_boot = 2000` on `xbabe2` OpenBLAS64):
      1. `m_x2_h455_valley003_soup50`
      2. `m_x2_h455_valley001_002_003_avg` (3-valley `M` soup vs `1865.36` Stage-1 high)
      3. `l_x1_h455_valley004_soup50`
      4. `l_x1_h455_valley001_004_avg`
      5. `lh455_v1v4_plus_mh455_v1v2v3_50_50`

Hard safety invariants:
  - Waits for Stage 2 (`H455_L_X1_V2_VALLEY001_SUMMARY.json` and optional Stage-2 PID) before launching shards.
  - Zero GPU/driver touch, zero trainer restarts/modifications, zero `CONFIRM` / `FMGAP B/C` / `HOLDOUT` / `SEALED`
    opens (`in_corpus_dev91_disclosure = True`, `canonical_promotion_forbidden = True`), zero deletions, zero temp-root paths.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE_CODE = Path("/home/ubuntu/h452-stage/code")
for _cand in (HERE, STAGE_CODE):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from score_h455_distributed_cpu import (  # noqa: E402
    FLYWHEEL_ROOT,
    assign_lpt_buckets,
    build_slim_soup50_ckpt,
    build_weighted_prediction_soup,
    dist_count_npz,
    dist_mem_available_gib,
    extract_h455_matrix_metrics,
    load_d91_shapes_for_lpt,
    pull_stage_dirs_from_xbabe0,
    sync_stage_inputs_to_xbabe0,
)

EVIDENCE_H455 = Path("/home/ubuntu/jope_plan/evidence/H455")
EVIDENCE_H456 = Path("/home/ubuntu/jope_plan/evidence/H456")
STAGE2_SUMMARY_PATH = EVIDENCE_H455 / "H455_L_X1_V2_VALLEY001_SUMMARY.json"


def stage3_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_stage2_completion(stage2_pid: int, summary_path: Path, poll_s: float = 20.0) -> None:
    t0 = time.time()
    while True:
        pid_running = stage3_pid_alive(stage2_pid)
        summary_ready = summary_path.is_file() and summary_path.stat().st_size > 2000
        if (not pid_running) and summary_ready:
            print(
                json.dumps(
                    {
                        "event": "stage3_gate_open",
                        "stage2_summary": str(summary_path),
                        "waited_s": round(time.time() - t0, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        print(
            json.dumps(
                {
                    "event": "stage3_waiting_for_stage2",
                    "stage2_pid": stage2_pid,
                    "stage2_pid_alive": pid_running,
                    "stage2_summary_ready": summary_ready,
                    "xbabe2_mem_avail_gib": dist_mem_available_gib(),
                    "waited_s": round(time.time() - t0, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_s)


def pull_readonly_remote_ckpt(remote_spec: str, local_dst: Path) -> Path:
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    if local_dst.is_file() and local_dst.stat().st_size > 100_000_000:
        return local_dst
    tmp_dst = local_dst.parent / f".{local_dst.name}.{os.getpid()}.part"
    subprocess.run(["rsync", "-az", remote_spec, str(tmp_dst)], check=True, timeout=600)
    os.replace(tmp_dst, local_dst)
    return local_dst


def build_stage3_spec_tuple(
    m_v3_slim: Path,
    d_m_v3: Path,
    l_v4_slim: Path,
    d_l_v4: Path,
) -> tuple[str, list[Path]]:
    multi_s3 = ";".join(
        [
            f"m_v3_soup50|ema_soup50|M:{m_v3_slim}|{d_m_v3}",
            f"l_v4_soup50|ema_soup50|L:{l_v4_slim}|{d_l_v4}",
        ]
    )
    return multi_s3, [d_m_v3, d_l_v4]


def run_stage3_dual_stealer_eval(
    stage_name: str,
    multi_specs: str,
    out_dirs: list[Path],
    n_xbabe0: int,
    n_tail_stealers: int,
    n_cost_stealers: int,
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

    t0 = time.time()
    procs: list[tuple[str, subprocess.Popen]] = []
    env_cpu = os.environ.copy()
    env_cpu["CUDA_VISIBLE_DEVICES"] = ""

    if xb0_buckets:
        r_cmd = (
            "CUDA_VISIBLE_DEVICES='' systemd-run --user --scope -p MemoryHigh=48G -p MemoryMax=58G "
            f"choom -n 500 -- /usr/bin/python3 -u /home/ubuntu/h452-stage/code/run_local_shard_pool.py --spec-json {cfg_xb0}"
        )
        procs.append(("xbabe0", subprocess.Popen(["ssh", "xbabe0", r_cmd])))

    stealer_script = str(STAGE_CODE / "steal_distributed_tails_cpu.py")
    if n_tail_stealers > 0 and xb0_buckets:
        procs.append(
            (
                "xbabe2_shard_tail",
                subprocess.Popen(
                    [
                        "systemd-run",
                        "--user",
                        "--scope",
                        "-p",
                        "MemoryHigh=10G",
                        "-p",
                        "MemoryMax=14G",
                        "-p",
                        "MemorySwapMax=0",
                        "choom",
                        "-n",
                        "800",
                        "--",
                        "/home/ubuntu/.venv/bin/python3",
                        "-u",
                        stealer_script,
                        "--stage-cfg",
                        str(cfg_xb0),
                        "--claims-dir",
                        str(claims_dir),
                        "--order",
                        "shard_tail",
                        "--worker-offset",
                        "0",
                        "--affinity",
                        "32-63,80-127",
                        "--workers",
                        str(n_tail_stealers),
                        "--threads",
                        "4",
                        "--remote-host",
                        "xbabe0",
                    ],
                    env=env_cpu,
                ),
            )
        )

    if n_cost_stealers > 0 and xb0_buckets:
        procs.append(
            (
                "xbabe2_cost_desc",
                subprocess.Popen(
                    [
                        "systemd-run",
                        "--user",
                        "--scope",
                        "-p",
                        "MemoryHigh=12G",
                        "-p",
                        "MemoryMax=16G",
                        "-p",
                        "MemorySwapMax=0",
                        "choom",
                        "-n",
                        "800",
                        "--",
                        "/home/ubuntu/.venv/bin/python3",
                        "-u",
                        stealer_script,
                        "--stage-cfg",
                        str(cfg_xb0),
                        "--claims-dir",
                        str(claims_dir),
                        "--order",
                        "cost_desc",
                        "--worker-offset",
                        "10",
                        "--affinity",
                        "32-63,80-127",
                        "--workers",
                        str(n_cost_stealers),
                        "--threads",
                        "4",
                        "--remote-host",
                        "xbabe0",
                    ],
                    env=env_cpu,
                ),
            )
        )

    while True:
        time.sleep(20.0)
        pull_stage_dirs_from_xbabe0(out_dirs)
        alive = [name for name, p in procs if p.poll() is None]
        counts = {od.name: dist_count_npz(od) for od in out_dirs}
        print(
            json.dumps(
                {
                    "event": "stage3_progress",
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


def score_stage3_matrices_parallel(
    jobs: list[tuple[str, Path, Path]],
    traj_post_dir: Path,
    n_boot: int = 2000,
    max_parallel: int = 2,
) -> dict[str, dict]:
    scorer = STAGE_CODE / "score_h454_paired_flywheel_matrix.py"
    env_sc = os.environ.copy()
    env_sc["CUDA_VISIBLE_DEVICES"] = ""

    def _run_one(item: tuple[str, Path, Path]) -> tuple[str, dict]:
        label, post_dir, out_json = item
        out_json.parent.mkdir(parents=True, exist_ok=True)
        reused = False
        ts0 = time.time()
        if out_json.is_file() and out_json.stat().st_size > 10000:
            try:
                if json.loads(out_json.read_text()).get("summary_table"):
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
                    "-p",
                    "MemorySwapMax=0",
                    "choom",
                    "-n",
                    "800",
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
                    "16",
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
        print(
            json.dumps(
                {
                    "event": "stage3_matrix_scored",
                    "label": label,
                    "reused": reused,
                    "best_post_arm": rec["best_post_arm"],
                    "best_post_elo": rec["best_post_elo"],
                    "stack_cons_post_elo": rec["arms"]["stack_ugp_g1000_cons_post"]["elo"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return label, rec

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_parallel))) as pool:
        for lbl, rec in pool.map(_run_one, jobs):
            results[lbl] = rec
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait-stage2", type=int, default=1)
    ap.add_argument("--stage2-pid", type=int, default=4038184)
    ap.add_argument("--xbabe0-shards", type=int, default=24)
    ap.add_argument("--xbabe2-tail-stealers", type=int, default=2)
    ap.add_argument("--xbabe2-cost-stealers", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    EVIDENCE_H456.mkdir(parents=True, exist_ok=True)
    if int(a.wait_stage2) == 1:
        wait_for_stage2_completion(int(a.stage2_pid), STAGE2_SUMMARY_PATH)

    cols_root = FLYWHEEL_ROOT / "cols_e8_d91"
    traj_post_dir = cols_root / "traj_e8_w2aplus"
    slim_dir = FLYWHEEL_ROOT / "ckpts_h455_slim_eval"
    pulled_dir = FLYWHEEL_ROOT / "ckpts_h456_pulled_valleys"

    # Strictly read-only access to M valley_003 on xbabe2 and L valley_004 on xbabe1
    m_v3_readonly = FLYWHEEL_ROOT / "runs" / "fleet_M_x2_h455_maxent" / "ckpt_valley_003.pt"
    l_v4_pulled = pull_readonly_remote_ckpt(
        "xbabe1:/home/ubuntu/jope-data/h452-flywheel/runs/fleet_L_x1_h455_maxent_v2/ckpt_valley_004.pt",
        pulled_dir / "fleet_L_x1_h455_v2_ckpt_valley_004.pt",
    )

    m_v3_slim = slim_dir / "m_x2_v3_soup50_slim.pt"
    l_v4_slim = slim_dir / "l_x1_v4_soup50_slim.pt"
    meta_m3 = build_slim_soup50_ckpt(m_v3_readonly, m_v3_slim, "M")
    meta_l4 = build_slim_soup50_ckpt(l_v4_pulled, l_v4_slim, "L")

    d_m_v1 = cols_root / "cv7_hyb_m_h455_v1_soup50"
    d_m_v2 = cols_root / "cv7_hyb_m_h455_v2_soup50"
    d_l_v1 = cols_root / "cv7_hyb_l_x1_h455_v1_soup50"
    d_m_v3 = cols_root / "cv7_hyb_m_h455_v3_soup50"
    d_l_v4 = cols_root / "cv7_hyb_l_x1_h455_v4_soup50"

    multi_s3, s3_dirs = build_stage3_spec_tuple(m_v3_slim, d_m_v3, l_v4_slim, d_l_v4)
    elapsed_s3 = run_stage3_dual_stealer_eval(
        stage_name="stage3_m_v3_and_l_v4",
        multi_specs=multi_s3,
        out_dirs=s3_dirs,
        n_xbabe0=int(a.xbabe0_shards),
        n_tail_stealers=int(a.xbabe2_tail_stealers),
        n_cost_stealers=int(a.xbabe2_cost_stealers),
    )

    d_m_v123 = cols_root / "cv7_hyb_m_h455_v123_avg"
    d_l_v14 = cols_root / "cv7_hyb_l_x1_h455_v14_avg"
    d_lm_multi = cols_root / "cv7_hyb_lh455v14_mh455v123_50_50"

    build_weighted_prediction_soup([(d_m_v1, 1.0), (d_m_v2, 1.0), (d_m_v3, 1.0)], d_m_v123)
    build_weighted_prediction_soup([(d_l_v1, 0.50), (d_l_v4, 0.50)], d_l_v14)
    build_weighted_prediction_soup([(d_l_v14, 0.50), (d_m_v123, 0.50)], d_lm_multi)

    s3_jobs: list[tuple[str, Path, Path]] = [
        ("m_x2_h455_valley003_soup50", d_m_v3, EVIDENCE_H456 / "H456_M_X2_VALLEY003_SOUP50_MATRIX.json"),
        ("m_x2_h455_valley001_002_003_avg", d_m_v123, EVIDENCE_H456 / "H456_M_X2_VALLEY001_002_003_AVG_MATRIX.json"),
        ("l_x1_h455_valley004_soup50", d_l_v4, EVIDENCE_H456 / "H456_L_X1_V2_VALLEY004_SOUP50_MATRIX.json"),
        ("l_x1_h455_valley001_004_avg", d_l_v14, EVIDENCE_H456 / "H456_L_X1_V2_VALLEY001_004_AVG_MATRIX.json"),
        ("lh455_v1v4_plus_mh455_v1v2v3_50_50", d_lm_multi, EVIDENCE_H456 / "H456_LH455V14_MH455V123_50_50_MATRIX.json"),
    ]

    s3_res = score_stage3_matrices_parallel(s3_jobs, traj_post_dir=traj_post_dir, n_boot=int(a.n_boot), max_parallel=2)
    best_s3 = max(s3_res.keys(), key=lambda k: float(s3_res[k]["best_post_elo"] or -1e9))
    summary_p = EVIDENCE_H456 / "H456_STAGE3_VALLEY_PROGRESSION_SUMMARY.json"
    summary_doc = {
        "schema": "jope.h456.stage3_valley_progression/v1",
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
        "in_corpus_dev91_disclosure": True,
        "canonical_promotion_forbidden": True,
        "prior_stage1_best_elo": 1865.36,
        "delta_vs_stage1_best_elo": round(float(s3_res[best_s3]["best_post_elo"]) - 1865.36, 2),
        "quarantine": {"confirm_opens": 0, "fmgap_bc_opens": 0, "holdout_opens": 0, "sealed_opens": 0},
        "eval_hosts": {
            "xbabe0_shards": int(a.xbabe0_shards),
            "xbabe2_tail_stealers": int(a.xbabe2_tail_stealers),
            "xbabe2_cost_stealers": int(a.xbabe2_cost_stealers),
            "device": "cpu",
            "fleet_m_x2_touched": False,
        },
        "stage3_eval_elapsed_s": elapsed_s3,
        "checkpoints": {"m_valley_003": meta_m3, "l_valley_004": meta_l4},
        "best_overall_label": best_s3,
        "best_overall_arm": s3_res[best_s3]["best_post_arm"],
        "best_overall_elo": s3_res[best_s3]["best_post_elo"],
        "best_overall_lcb95": s3_res[best_s3]["best_post_lcb95"],
        "best_overall_ucb95": s3_res[best_s3]["best_post_ucb95"],
        "best_overall_utility": s3_res[best_s3]["best_post_utility"],
        "waves": s3_res,
    }
    tmp_s3 = summary_p.parent / f".{summary_p.name}.{os.getpid()}.tmp"
    tmp_s3.write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_s3, summary_p)
    print(f"STAGE3_SUMMARY_WRITTEN: {summary_p}", flush=True)


if __name__ == "__main__":
    main()
