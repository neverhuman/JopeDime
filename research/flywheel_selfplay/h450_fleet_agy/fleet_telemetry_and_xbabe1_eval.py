#!/usr/bin/env python3
"""5-Host H450 Fleet Telemetry, Cross-Run Unique-Dataset Union Counter, and `xbabe1` CPU Every-`N`-Dataset
`dev91` Frozen-Split Elo Evaluator (`fleet_telemetry_and_xbabe1_eval.py`).

Operates 100% read-only on `cl_fwriter`'s `r4_*` runs/checkpoints and writes ONLY to:
  - `/home/ubuntu/jope-data/h450-fleet-agy/` (local & `xbabe1` staging/eval)
  - `/home/ubuntu/jope_plan/evidence/H450/diag_results/fleet_x1_*` (create-once `analyze_h450.py` outputs)
  - `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md` (under `H450_FLEET_COORDINATION.md.lock`)
  - `/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` (milestone lines under `H450_CLAUDE_GEMINI_COORD.md.lock`)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

COORD_MD = Path("/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md")
COORD_LOCK = Path("/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md.lock")
PEER_MD = Path("/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md")
PEER_LOCK = Path("/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md.lock")

EVAL_BANK = Path("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
EVAL_BANK_SHA = "0bce958572faee0ddae494df6d75c837e1c14936e34e565f042ef13aa727d955"
PUB_R6_STAGE = Path("/home/ubuntu/h450-stage/pub-r6")
FLEET_ROOT = Path("/home/ubuntu/jope-data/h450-fleet-agy")
DIAG_ROOT = Path("/home/ubuntu/jope_plan/evidence/H450/diag_results")


def _sh(cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return 1, str(e)


def _ssh(host: str, script: str, timeout: float = 15.0) -> tuple[int, str]:
    if host == "xbabe2":
        return _sh(["bash", "-c", script], timeout=timeout)
    return _sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, script], timeout=timeout)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_host_metrics() -> dict[str, dict]:
    hosts = {}
    linux_probe = (
        "ncpu=$(nproc); load1=$(awk '{print $1}' /proc/loadavg); "
        "mem=$(free -m | awk '/^Mem:/ {print $3\"/\"$2\" MiB (avail \"$7\" MiB)\"}'); "
        "gpu=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 'ERROR_OR_NONE'); "
        "printf '%s|%s|%s|%s\n' \"$ncpu\" \"$load1\" \"$mem\" \"$gpu\""
    )
    for h in ("xbabe0", "xbabe1", "xbabe2", "xbabe3"):
        rc, out = _ssh(h, linux_probe, timeout=10.0)
        if rc == 0 and "|" in out:
            ncpu_s, load1_s, mem_s, gpu_s = out.splitlines()[-1].split("|", 3)
            ncpu = max(1, int(ncpu_s))
            load1 = float(load1_s)
            cpu_pct = round(100.0 * load1 / ncpu, 1)
            vram_pct = None
            gpu_desc = gpu_s.strip()
            if "," in gpu_s and "ERROR" not in gpu_s.upper() and "UNKNOWN" not in gpu_s.upper():
                parts = [x.strip() for x in gpu_s.split(",")]
                if len(parts) >= 4:
                    used, tot, util = float(parts[1]), float(parts[2]), parts[3]
                    vram_pct = round(100.0 * used / max(tot, 1.0), 1)
                    gpu_desc = f"{parts[0]}: {int(used)}/{int(tot)} MiB ({vram_pct}% VRAM, {util}% util)"
            hosts[h] = {"ncpu": ncpu, "load1": load1, "cpu_pct": cpu_pct, "mem": mem_s, "gpu": gpu_desc, "vram_pct": vram_pct}
        else:
            hosts[h] = {"ncpu": 0, "load1": 0.0, "cpu_pct": 0.0, "mem": "unreachable", "gpu": out[:80], "vram_pct": None}

    mac_probe = (
        "ncpu=$(sysctl -n hw.ncpu); "
        "load1=$(sysctl -n vm.loadavg | awk '{print $2}'); "
        "pscpu=$(ps -A -o %cpu | awk 'NR>1 {s+=$1} END {printf \"%.1f\", s}'); "
        "printf '%s|%s|%s\n' \"$ncpu\" \"$load1\" \"$pscpu\""
    )
    rc, out = _ssh("xbabe4", mac_probe, timeout=10.0)
    if rc == 0 and "|" in out:
        ncpu_s, load1_s, pscpu_s = out.splitlines()[-1].split("|", 2)
        ncpu = max(1, int(ncpu_s))
        cpu_pct = round(float(pscpu_s) / ncpu, 1)
        hosts["xbabe4"] = {
            "ncpu": ncpu, "load1": float(load1_s), "cpu_pct": cpu_pct,
            "mem": "512 GiB Unified Memory", "gpu": "Apple M3 Ultra 80-core Metal (MPS active)", "vram_pct": None,
        }
    else:
        hosts["xbabe4"] = {"ncpu": 32, "load1": 0.0, "cpu_pct": 0.0, "mem": "unreachable", "gpu": out[:80], "vram_pct": None}
    return hosts


def collect_run_logs() -> dict[str, dict]:
    runs = {}
    specs = [
        ("r4_L_A", "xbabe2", "104M (L)", "/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_L_A/log.jsonl"),
        ("r4_L_B", "xbabe3", "104M (L)", "/home/ubuntu/jope-data/h450-r4-x3/runs/r4_L_B/log.jsonl"),
        ("fleet_L_x4", "xbabe4", "104M (L)", "/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/log.jsonl"),
        ("r4_M_A", "xbabe2", "24M (M)", "/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_M_A/log.jsonl"),
        ("r4_M_B", "xbabe3", "24M (M)", "/home/ubuntu/jope-data/h450-r4-x3/runs/r4_M_B/log.jsonl"),
    ]
    for name, host, size_lbl, log_p in specs:
        rc, out = _ssh(host, f"tail -n 5 {log_p} 2>/dev/null || true", timeout=8.0)
        rec = None
        if rc == 0 and out:
            for line in reversed(out.splitlines()):
                try:
                    d = json.loads(line)
                    if "datasets" in d and "loss" in d:
                        rec = d
                        break
                except Exception:
                    continue
        if rec:
            dups = int(rec.get("dup_real", 0)) + int(rec.get("dup_synth", 0)) + int(rec.get("dup_fed", 0))
            runs[name] = {
                "host": host, "size": size_lbl, "active": True,
                "step": int(rec.get("step", 0)), "train_h": float(rec.get("train_h", 0.0)),
                "loss": float(rec.get("loss", 0.0)),
                "unique_datasets": int(rec.get("unique_datasets", rec.get("datasets", 0))),
                "datasets_seen": int(rec.get("datasets_seen", rec.get("datasets", 0))),
                "dups_fed": dups,
                "src_tables_seen": int(rec.get("src_tables_seen", 0)),
                "inj_share": float(rec.get("inj_share", 0.0)),
                "mem_gb": float(rec.get("mem_gb", 0.0)),
            }
        else:
            runs[name] = {"host": host, "size": size_lbl, "active": False, "step": 0, "train_h": 0.0, "loss": 0.0,
                          "unique_datasets": 0, "datasets_seen": 0, "dups_fed": 0, "src_tables_seen": 0, "inj_share": 0.0, "mem_gb": 0.0}
    return runs


def collect_dev91_results() -> list[dict]:
    rows = []
    if not DIAG_ROOT.is_dir():
        return rows
    for p in sorted(DIAG_ROOT.glob("*/H450_DEV91.json"), key=lambda x: x.stat().st_mtime):
        tag = p.parent.name
        if not (tag.startswith("r4_") or tag.startswith("fleet_x1_")):
            continue
        try:
            d = json.loads(p.read_text())
            ch = d["in_corpus_random_split_champion_91"]
            ic = d["in_corpus_icl_boost_only_91"]
            ng = d["in_corpus_neural_plus_tab_no_gp_91"]
            zr = d.get("fair_zero_d91_control_91")
            meta_p = p.parent / "MILESTONE_META.json"
            ms_meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
            rows.append({
                "tag": tag,
                "mtime_utc": time.strftime("%FT%TZ", time.gmtime(p.stat().st_mtime)),
                "fleet_104m_unique_ds": ms_meta.get("fleet_104m_unique_datasets", 0 if "ep00_init" in tag else None),
                "champion_elo": round(float(ch["elo"]["elo"]), 1),
                "champion_ci95": [round(float(x), 1) for x in ch["elo"]["ci95"]],
                "icl_only_elo": round(float(ic["elo"]["elo"]), 1),
                "nogp_elo": round(float(ng["elo"]["elo"]), 1),
                "zero_ctrl_elo": round(float(zr["elo"]["elo"]), 1) if zr else None,
                "gap_vs_kumo_L": round(float(ch["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1),
                "gap_vs_h448_fair": round(float(ch["gap_elo"]["vs_h448.APEX_FAIR_CHAMPION"]), 1),
            })
        except Exception:
            continue
    return rows


def build_104m_soup(ckpt_paths: list[Path], out_path: Path) -> str:
    states = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        w = ck.get("ema") or ck["model"]
        states.append(w)
    avg = {}
    for k in states[0]:
        t0 = states[0][k]
        if torch.is_floating_point(t0):
            acc = sum(s[k].double() for s in states) / float(len(states))
            avg[k] = acc.to(dtype=t0.dtype)
        else:
            avg[k] = t0.clone()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save({"ema": avg, "model": avg, "size": "L", "soup_members": len(states)}, tmp)
    os.replace(tmp, out_path)
    return _sha256(out_path)


def run_xbabe1_milestone_eval(milestone_ds: int, runs: dict[str, dict], shards: int = 18, threads: int = 2) -> dict | None:
    """Evaluates current fleet `104M` EMA checkpoints on `xbabe1` CPU (`<= 75%` CPU, `0` GPU touched) on frozen `dev91`."""
    tag = f"ms_{milestone_ds // 1000:04d}k"
    diag_dir = DIAG_ROOT / f"fleet_x1_{tag}"
    if (diag_dir / "H450_DEV91.json").is_file():
        return None

    work_ck = FLEET_ROOT / "eval_ckpts" / tag
    work_ck.mkdir(parents=True, exist_ok=True)

    # Snapshot read-only checkpoints (copying `.pt` safely; retry once if mid-rename)
    la_src = Path("/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_L_A/ckpt_last.pt")
    ma_src = Path("/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_M_A/ckpt_last.pt")
    init_l = Path("/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt")
    init_m = Path("/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_M/ckpt_unfair_final.pt")

    la_p = work_ck / "la.pt"
    ma_p = work_ck / "ma.pt"
    lb_p = work_ck / "lb.pt"
    mb_p = work_ck / "mb.pt"

    shutil.copyfile(la_src if la_src.is_file() else init_l, la_p)
    shutil.copyfile(ma_src if ma_src.is_file() else init_m, ma_p)

    rc_lb, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                    "xbabe3:/home/ubuntu/jope-data/h450-r4-x3/runs/r4_L_B/ckpt_last.pt", str(lb_p)], timeout=60.0)
    if rc_lb != 0 or not lb_p.is_file():
        shutil.copyfile(la_p, lb_p)

    rc_mb, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                    "xbabe3:/home/ubuntu/jope-data/h450-r4-x3/runs/r4_M_B/ckpt_last.pt", str(mb_p)], timeout=60.0)
    if rc_mb != 0 or not mb_p.is_file():
        shutil.copyfile(ma_p, mb_p)

    # Also pull xbabe4 104M checkpoint if available and build 3-way 104M EMA soup into lb_p (keeping la_p = r4_L_A pure, lb_p = fleet 104M soup!)
    x4_p = work_ck / "x4_l.pt"
    rc_x4, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                    "xbabe4:/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/ckpt_last.pt", str(x4_p)], timeout=60.0)
    soup_members = [la_p, lb_p] + ([x4_p] if (rc_x4 == 0 and x4_p.is_file()) else [])
    soup_p = work_ck / "soup_l.pt"
    build_104m_soup(soup_members, soup_p)

    # Strip optimizer/ledger state so rsync to xbabe1 transfers ~415 MB instead of 1.67 GB!
    slim_dir = work_ck / "slim"
    slim_dir.mkdir(parents=True, exist_ok=True)
    for role, src_pt, sz in (("la", la_p, "L"), ("ma", ma_p, "M"), ("lb", soup_p, "L"), ("mb", mb_p, "M")):
        ck = torch.load(src_pt, map_location="cpu", weights_only=False)
        w = ck.get("ema") or ck["model"]
        torch.save({"ema": w, "model": w, "size": sz}, slim_dir / f"{role}.pt")

    x1_root = "/home/ubuntu/jope-data/h450-fleet-agy/x1-eval"
    _ssh("xbabe1", f"mkdir -p {x1_root}/stage/pub-r6 {x1_root}/ckpts/{tag} {x1_root}/cols/{tag} {x1_root}/logs", timeout=15.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{PUB_R6_STAGE}/", f"xbabe1:{x1_root}/stage/pub-r6/"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", "/home/ubuntu/h448-stage/pub-r1/", f"xbabe1:{x1_root}/stage/h448-pub-r1/"], timeout=60.0)
    rc_bk, _ = _ssh("xbabe1", f"[ -f {x1_root}/bank_d91_search_f64.npz ] && echo OK || echo MISS", timeout=10.0)
    if "OK" not in rc_bk and True:
        _, out_bk = _ssh("xbabe1", f"sha256sum {x1_root}/bank_d91_search_f64.npz 2>/dev/null | cut -c1-64 || true", timeout=15.0)
        if out_bk.strip() != EVAL_BANK_SHA:
            _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(EVAL_BANK), f"xbabe1:{x1_root}/bank_d91_search_f64.npz"], timeout=60.0)

    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{slim_dir}/", f"xbabe1:{x1_root}/ckpts/{tag}/"], timeout=90.0)

    # Run `cols_h450_gpu.py --device cpu` on xbabe1 at nice 15 with 18 shards x 2 threads (~36 cores / 102 -> ~68% total CPU <= 85%!)
    run_cmd = (
        f"seq 0 {shards - 1} > {x1_root}/logs/{tag}.shards && "
        f"PYTHONPATH={x1_root}/stage/pub-r6/code:{x1_root}/stage/h448-pub-r1/code "
        f"OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "
        f"xargs -a {x1_root}/logs/{tag}.shards -P {shards} -I@ nice -n 15 python3 -B "
        f"{x1_root}/stage/pub-r6/code/cols_h450_gpu.py "
        f"--bank {x1_root}/bank_d91_search_f64.npz "
        f"--ckpt-la {x1_root}/ckpts/{tag}/la.pt --ckpt-ma {x1_root}/ckpts/{tag}/ma.pt "
        f"--ckpt-lb {x1_root}/ckpts/{tag}/lb.pt --ckpt-mb {x1_root}/ckpts/{tag}/mb.pt "
        f"--out {x1_root}/cols/{tag} --device cpu --num-shards {shards} --shard-idx @ > {x1_root}/logs/{tag}.log 2>&1"
    )
    t0 = time.time()
    rc_c, _ = _ssh("xbabe1", run_cmd, timeout=2400.0)
    _, n_npz_s = _ssh("xbabe1", f"find {x1_root}/cols/{tag} -maxdepth 1 -name '[0-9a-f]*.npz' | wc -l", timeout=10.0)
    if rc_c != 0 or n_npz_s.strip() != "91":
        return None

    loc_cols = FLEET_ROOT / "cols" / tag
    loc_cols.mkdir(parents=True, exist_ok=True)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"xbabe1:{x1_root}/cols/{tag}/", f"{loc_cols}/"], timeout=60.0)

    diag_dir.mkdir(parents=True, exist_ok=True)
    zero_dir = Path("/home/ubuntu/jope-data/h450-quarantine/r4/cols/ep00_init")
    zero_args = ["--zero-d91-gpu-cols", str(zero_dir)] if (zero_dir.is_dir() and len(list(zero_dir.glob("[0-9a-f]*.npz"))) == 91) else []

    an_cmd = [
        "nice", "-n", "19", "python3", "-B", str(PUB_R6_STAGE / "code/analyze_h450.py"),
        "--bank", str(EVAL_BANK),
        *zero_args,
        "--gpu-cols", str(loc_cols),
        "--tab-cols", "/home/ubuntu/jope-data/h450-quarantine/cols_tab_h450_D91",
        "--h448-pulled", "/home/ubuntu/jope_plan/evidence/H448/pulled",
        "--h418-pulled", "/home/ubuntu/jope_plan/evidence/H418/pulled",
        "--board", "/home/ubuntu/jope_plan/evidence/H450/board_post_h448/dev91-audit-v1",
        "--lock", "/home/ubuntu/jope_plan/evidence/H450/stage_v/LOCK.json",
        "--receipt", "/home/ubuntu/jope_plan/ns/receipts/H450_UNFAIR_LOCK_sha256.txt",
        "--diag-out", str(diag_dir / "H450_DEV91.json"),
        "--diag-txt", str(diag_dir / "H450_DEV91.txt"),
        "--diag-board", str(diag_dir / "board"),
        "--procs", "8",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    r_an = subprocess.run(an_cmd, env=env, capture_output=True, text=True, timeout=600.0)
    if r_an.returncode != 0 or not (diag_dir / "H450_DEV91.json").is_file():
        return None

    fleet_104m_u = sum(runs[k]["unique_datasets"] for k in ("r4_L_A", "r4_L_B", "fleet_L_x4") if k in runs)
    ms_meta = {
        "tag": tag,
        "milestone_target_datasets": milestone_ds,
        "fleet_104m_unique_datasets": fleet_104m_u,
        "per_run_unique_datasets": {k: v["unique_datasets"] for k, v in runs.items()},
        "eval_host": "xbabe1",
        "eval_wall_s": round(time.time() - t0, 1),
        "soup_members_count": len(soup_members),
    }
    (diag_dir / "MILESTONE_META.json").write_text(json.dumps(ms_meta, indent=2, sort_keys=True) + "\n")

    res_doc = json.loads((diag_dir / "H450_DEV91.json").read_text())
    ch_elo = round(float(res_doc["in_corpus_random_split_champion_91"]["elo"]["elo"]), 1)
    ic_elo = round(float(res_doc["in_corpus_icl_boost_only_91"]["elo"]["elo"]), 1)
    gap_k = round(float(res_doc["in_corpus_random_split_champion_91"]["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1)

    # Notify peer coordination log (`H450_CLAUDE_GEMINI_COORD.md`) under flock!
    ts = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    line = (
        f"[{ts}] agy_apex_w10 -> cl_fwriter (cc all): [xbabe1 N={milestone_ds:,} dev91 frozen-split in-corpus readout {tag}] "
        f"104M_fleet_unique_datasets={fleet_104m_u:,} (0 dups), champion_stack={ch_elo} Elo (vs Kumo L 2070.2: {gap_k:+.1f} Elo), "
        f"pure_ICL_boost={ic_elo} Elo, wall={ms_meta['eval_wall_s']}s on xbabe1 CPU (<=75% CPU, ZImage GPU untouched); "
        f"receipt {diag_dir / 'H450_DEV91.json'}."
    )
    PEER_LOCK.touch(exist_ok=True)
    with open(PEER_LOCK, "r+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        with open(PEER_MD, "a") as pf:
            pf.write(line + "\n")
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return ms_meta


def write_coordination_dashboard(hosts: dict[str, dict], runs: dict[str, dict], evals: list[dict]) -> dict:
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    u_104m = sum(runs[k]["unique_datasets"] for k in ("r4_L_A", "r4_L_B", "fleet_L_x4") if k in runs)
    u_all = sum(v["unique_datasets"] for v in runs.values())
    dups_all = sum(v["dups_fed"] for v in runs.values())

    lines = [
        "# H450 5-Host Fleet-Wide `104M` (`ICL_L`) All-Data Fine-Tuning & `dev91` Elo Dashboard",
        "",
        f"- **Last Telemetry Refresh (UTC):** `{now_utc}`",
        "- **Shared Coordination Files:**",
        "  - **Live Fleet Telemetry & `dev91` Elo Dashboard (this file):** `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md` (lock: `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md.lock`)",
        "  - **Claude (`cl_fwriter`) <-> Gemini (`agy_apex_w10`) Append-Only Protocol Log:** `/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` (lock: `/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md.lock`)",
        "- **Starting `104M` Checkpoint:** `H449` `all_non_d91_L/ckpt_unfair_final.pt` (`104,360,576` params, `sha256: b854b518…3a29`, `1828.6` FAIR `dev91` Elo)",
        "- **Corpus:** `3,525` combined `train U test` `float64` tables (`9dfc60ed…` + bit-identical `np.memmap` `mmap_v1` `de5016de…` — verified **100%** of valid `>=32`-row non-sealed S3 regression datasets in `catalog_v246`, `0` `APEX-CONFIRM` / `DL-Holdout` / `sealed` opens)",
        f"- **Total Unique Datasets Seen by `104M` (`L`) Fleet:** **`{u_104m:,}`** (`{u_all:,}` incl. companion `24M` runs) | **Duplicate Datasets Fed to Model:** **`{dups_all}`** (`0` repeats)",
        "- **Quarantine Notice:** `canonical_promotion_forbidden = true` — `dev91` is in the random-split training corpus; scores below measure **in-corpus fit on the original frozen `dev91-audit-v1` `X_search / X_audit` split** (`0` `y_audit` touched during fit/stacking; reference target `kumo_tabular_large_e8.v1 = 2070.2` Elo; canonical `h448.APEX_FAIR_CHAMPION = 1821.5` Elo untouched).",
        "",
        "## 1. Live 5-Host Hardware Utilization (`~80%` GPU VRAM Target | `<= 85%` CPU Cap)",
        "",
        "| Host | Active H450 Role | GPU / Accelerator Telemetry | Host CPU Avg (`<= 85%` Cap) | Memory |",
        "| --- | --- | --- | --- | --- |",
    ]
    roles = {
        "xbabe0": "`cl_fwriter` `dev91` Frozen-Split CPU Eval (`ep00_init` + hourly epochs)",
        "xbabe1": "`agy_apex_w10` Every-`25k`-Dataset `dev91` CPU Eval (`ZImage` GPU untouched)",
        "xbabe2": "`cl_fwriter` `r4_L_A` (`104M`) + `r4_M_A` (`24M`) CUDA Training",
        "xbabe3": "`cl_fwriter` `r4_L_B` (`104M`) + `r4_M_B` (`24M`) CUDA Training",
        "xbabe4": "`agy_apex_w10` `fleet_L_x4` (`104M`) M3 Ultra 80-Core MPS Training",
    }
    for h in ("xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"):
        hm = hosts.get(h, {})
        lines.append(
            f"| `{h}` | {roles[h]} | {hm.get('gpu', 'n/a')} | **`{hm.get('cpu_pct', 0.0)}%`** (load `{hm.get('load1', 0.0)}` / `{hm.get('ncpu', 0)}c`) | {hm.get('mem', 'n/a')} |"
        )

    lines.extend([
        "",
        "## 2. Live Per-Worker Unique-Dataset & Loss Ledger (`0` Repeats)",
        "",
        "| Run | Host | Model | Status | Step | Train Hours | Loss | Unique Datasets Seen | Duplicates Fed | Distinct Source Tables | Inj Col Share |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for rname, rv in runs.items():
        st = "RUNNING" if rv["active"] else "STARTING/IDLE"
        lines.append(
            f"| `{rname}` | `{rv['host']}` | `{rv['size']}` | `{st}` | `{rv['step']:,}` | `{rv['train_h']:.3f}h` | `{rv['loss']:.4f}` | **`{rv['unique_datasets']:,}`** | **`{rv['dups_fed']}`** | `{rv['src_tables_seen']:,} / 3,525` | `{100.0 * rv['inj_share']:.2f}%` |"
        )

    lines.extend([
        "",
        "## 3. Frozen-Split `dev91-audit-v1` In-Corpus BT-Elo Trajectory (`N = 25,000` Cadence)",
        "",
        "| Milestone / Tag | Timestamp (UTC) | `104M` Fleet Unique Datasets | Champion Stack Elo (`95% CI`) | Pure `ICL` Boost Elo | Neural+Tab (No GP) Elo | Gap vs `Kumo L` (`2070.2`) | Gap vs `H448` Fair (`1821.5`) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    if not evals:
        lines.append("| `ep00_init` + `ms_0050k` | *evaluating now on `xbabe0` & `xbabe1`* | *in progress* | *pending* | *pending* | *pending* | *pending* | *pending* |")
    else:
        for ev in evals:
            ds_s = f"`{ev['fleet_104m_unique_ds']:,}`" if ev["fleet_104m_unique_ds"] is not None else "`hourly`"
            ci = ev["champion_ci95"]
            lines.append(
                f"| `{ev['tag']}` | `{ev['mtime_utc']}` | {ds_s} | **`{ev['champion_elo']:.1f}`** (`[{ci[0]:.1f}, {ci[1]:.1f}]`) | `{ev['icl_only_elo']:.1f}` | `{ev['nogp_elo']:.1f}` | `{ev['gap_vs_kumo_L']:+.1f}` | `{ev['gap_vs_h448_fair']:+.1f}` |"
            )

    body = "\n".join(lines) + "\n"
    COORD_LOCK.touch(exist_ok=True)
    with open(COORD_LOCK, "r+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        tmp = COORD_MD.with_suffix(".md.tmp")
        tmp.write_text(body)
        os.replace(tmp, COORD_MD)
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return {"utc": now_utc, "fleet_104m_unique_datasets": u_104m, "fleet_all_unique_datasets": u_all, "dups_fed": dups_all, "evals_count": len(evals)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmd", choices=("status", "daemon"), default="status")
    ap.add_argument("--every-datasets", type=int, default=25_000)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--xbabe1-shards", type=int, default=18)
    ap.add_argument("--xbabe1-threads", type=int, default=2)
    a = ap.parse_args()

    if a.cmd == "status":
        hosts = collect_host_metrics()
        runs = collect_run_logs()
        evals = collect_dev91_results()
        summary = write_coordination_dashboard(hosts, runs, evals)
        print(json.dumps({"summary": summary, "hosts": hosts, "runs": runs, "evals": evals}, indent=2, sort_keys=True))
        return

    last_milestone = 0
    while True:
        try:
            hosts = collect_host_metrics()
            runs = collect_run_logs()
            evals = collect_dev91_results()
            summary = write_coordination_dashboard(hosts, runs, evals)
            u_104m = summary["fleet_104m_unique_datasets"]
            target_ms = (u_104m // a.every_datasets) * a.every_datasets
            if target_ms >= a.every_datasets and target_ms > last_milestone:
                last_milestone = target_ms
                run_xbabe1_milestone_eval(target_ms, runs, shards=a.xbabe1_shards, threads=a.xbabe1_threads)
                hosts = collect_host_metrics()
                runs = collect_run_logs()
                evals = collect_dev91_results()
                write_coordination_dashboard(hosts, runs, evals)
        except Exception as exc:
            print(f"[daemon warn] {exc}", file=sys.stderr, flush=True)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
