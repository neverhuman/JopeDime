#!/usr/bin/env python3
"""Event-Driven Multi-Scale (`L:` / `M:`) Valley Watcher & CPU `dev91` Scorer (`watch_and_score_h455_valleys.py`).

Upgrades over `watch_and_score_wave2_h454.py`:
  1. Auto-detects checkpoint model size (`L:` vs `M:`) from `ck["size"]` instead of hardcoding `L:`.
  2. Watches both local `fleet_M_x2_h455_maxent` (`xbabe2`) and remote `fleet_L_x1_h455_maxent_v2` (`xbabe1`)
     for new `ckpt_valley_NNN.pt` files (`0` trainer stops, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`).
  3. Scores standalone `ema_soup50` + cross-scale `Wave-2 Cooled L + H455 Valley` soups on `91/91` in-corpus
     `dev91` (`canonical_promotion_forbidden = true`, `0` CONFIRM / FMGAP / HOLDOUT / SEALED opens).
"""
from __future__ import annotations

import argparse
import hashlib
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
GP_ANCHORS = FLYWHEEL_ROOT / "gp_anchors_p0"
GP_OOF_CACHE = FLYWHEEL_ROOT / "gp_anchors_oof_cache_v1"
EVIDENCE_H455 = Path("/home/ubuntu/jope_plan/evidence/H455")


def compute_digest_sha256(path_obj: Path) -> str:
    with path_obj.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def count_completed_npz(dir_path: Path) -> int:
    return sum(1 for p in dir_path.glob("*.npz") if p.is_file() and not p.name.startswith("."))


def detect_ckpt_size_prefix(ckpt_path: Path) -> str:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sz = str(ck.get("size", "L")).strip().upper()
    if sz not in ("S", "M", "L"):
        sz = "L"
    return sz


def run_cpu_sharded_cv7_eval(
    tag: str,
    ckpt_path: Path,
    ckpt_key: str,
    n_shards: int = 16,
    cpu_affinity: str = "48-127",
) -> Path:
    sz = detect_ckpt_size_prefix(ckpt_path)
    out_dir = FLYWHEEL_ROOT / "cols_e8_d91" / f"cv7_hyb_{tag}_{ckpt_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if count_completed_npz(out_dir) >= 91:
        return out_dir

    multi_spec = f"{tag}|{ckpt_key}|{sz}:{ckpt_path}|{out_dir}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    procs: list[subprocess.Popen] = []
    base = 91 // max(1, n_shards)
    rem = 91 % max(1, n_shards)
    cur = 0
    for si in range(max(1, n_shards)):
        span = base + (1 if si < rem else 0)
        if span <= 0:
            continue
        i0, i1 = cur, cur + span - 1
        cur += span
        cmd = [
            "taskset",
            "-c",
            cpu_affinity,
            "nice",
            "-n",
            "19",
            sys.executable,
            "-u",
            str(STAGE_CODE / "cols_104m_e8_gpresid_h452.py"),
            "--cmd",
            "eval-cv7",
            "--bank",
            str(BANK_D91),
            "--multi-specs",
            multi_spec,
            "--prep-workers",
            "0",
            "--gp-oof-cache-dir",
            str(GP_OOF_CACHE),
            "--gp-anchor-dir",
            str(GP_ANCHORS),
            "--indices",
            f"{i0}-{i1}",
            "--device",
            "cpu",
        ]
        procs.append(subprocess.Popen(cmd, env=env))

    for pr in procs:
        pr.wait()
    if count_completed_npz(out_dir) < 91:
        raise RuntimeError(f"incomplete 91/91 eval for {tag}: {count_completed_npz(out_dir)}")
    return out_dir


def score_post_dir_matrix(post_dir: Path, out_json: Path, cpu_affinity: str = "48-127") -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    traj_dir = FLYWHEEL_ROOT / "cols_e8_d91" / "traj_e8_w2aplus"
    subprocess.run(
        [
            "taskset",
            "-c",
            cpu_affinity,
            "nice",
            "-n",
            "19",
            sys.executable,
            str(STAGE_CODE / "score_h454_paired_flywheel_matrix.py"),
            "--cv7-post-dir",
            str(post_dir),
            "--traj-post-dir",
            str(traj_dir),
            "--procs",
            "24",
            "--n-boot",
            "2000",
            "--out-json",
            str(out_json),
        ],
        env=env,
        check=True,
    )
    doc = json.loads(out_json.read_text())
    best_row = max(
        (r for r in doc.get("summary_table", []) if "post_h454" in r.get("arm", "")),
        key=lambda x: float(x.get("elo", -1e9)),
    )
    return {
        "matrix_json": str(out_json),
        "best_arm": best_row.get("arm"),
        "best_elo": best_row.get("elo"),
        "ci95": best_row.get("ci95"),
        "sha256_matrix": compute_digest_sha256(out_json),
    }


def sync_remote_xbabe1_valleys(dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        [
            "ssh",
            "xbabe1",
            "ls -1 /home/ubuntu/jope-data/h452-flywheel/runs/fleet_L_x1_h455_maxent_v2/ckpt_valley_[0-9][0-9][0-9].pt 2>/dev/null || true",
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    if rc.returncode != 0:
        return []
    synced: list[Path] = []
    for line in rc.stdout.splitlines():
        rem = line.strip()
        if not rem:
            continue
        loc = dest_dir / f"l_x1_v2_{Path(rem).name}"
        if not loc.is_file():
            subprocess.run(["rsync", "-az", f"xbabe1:{rem}", str(loc)], check=True, timeout=240)
        if loc.is_file():
            synced.append(loc)
    return synced


def discover_local_m_valleys() -> list[Path]:
    m_dir = FLYWHEEL_ROOT / "runs" / "fleet_M_x2_h455_maxent"
    if not m_dir.is_dir():
        return []
    return sorted(p for p in m_dir.glob("ckpt_valley_[0-9][0-9][0-9].pt") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-json", default=str(EVIDENCE_H455 / "WATCH_H455_VALLEYS_STATE.json"))
    ap.add_argument("--poll-s", type=float, default=45.0)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--cpu-affinity", default="48-127")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    state_p = Path(args.state_json)
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_p.read_text()) if state_p.is_file() else {"scored": {}}

    snap_root = FLYWHEEL_ROOT / "ckpts_h455_watched"
    while True:
        candidates: list[tuple[str, Path]] = []
        for mp in discover_local_m_valleys():
            candidates.append((f"m_x2_h455_{mp.stem}", mp))
        for lp in sync_remote_xbabe1_valleys(snap_root):
            candidates.append((lp.stem, lp))

        for tag, ck_p in candidates:
            if tag in state["scored"]:
                continue
            out_dir = run_cpu_sharded_cv7_eval(
                tag=tag,
                ckpt_path=ck_p,
                ckpt_key="ema_soup50",
                n_shards=int(args.shards),
                cpu_affinity=str(args.cpu_affinity),
            )
            mat_p = EVIDENCE_H455 / f"AUTO_{tag.upper()}_SOUP50_MATRIX.json"
            res = score_post_dir_matrix(out_dir, mat_p, cpu_affinity=str(args.cpu_affinity))
            res["ckpt_sha256"] = compute_digest_sha256(ck_p)
            res["in_corpus_dev91_disclosure"] = True
            res["canonical_promotion_forbidden"] = True
            state["scored"][tag] = res
            state_p.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"event": "valley_scored", "tag": tag, **res}, sort_keys=True), flush=True)

        if args.once:
            break
        time.sleep(max(10.0, float(args.poll_s)))


if __name__ == "__main__":
    main()
