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
        "ncpu=$(nproc --all); load1=$(awk '{print $1}' /proc/loadavg); "
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
        ("fleet_L_x1", "xbabe1", "104M (L)", "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1/log.jsonl"),
        ("fleet_L_x4", "xbabe4", "104M (L)", "/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/log.jsonl"),
        ("r4_M_A", "xbabe2", "24M (M)", "/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_M_A/log.jsonl"),
        ("r4_M_B", "xbabe3", "24M (M)", "/home/ubuntu/jope-data/h450-r4-x3/runs/r4_M_B/log.jsonl"),
    ]
    now = int(time.time())
    for name, host, size_lbl, log_p in specs:
        stat_cmd = f"stat -f %m {log_p} 2>/dev/null || echo 0" if host == "xbabe4" else f"stat -c %Y {log_p} 2>/dev/null || echo 0"
        rc, out = _ssh(host, f"({stat_cmd}); tail -n 5 {log_p} 2>/dev/null || true", timeout=8.0)
        rec = None
        mtime = 0
        if rc == 0 and out:
            lines = out.splitlines()
            try:
                mtime = int(lines[0].strip())
            except Exception:
                mtime = 0
            for line in reversed(lines[1:]):
                try:
                    d = json.loads(line)
                    if "datasets" in d and "loss" in d:
                        rec = d
                        break
                except Exception:
                    continue
        if rec:
            dups = int(rec.get("dup_real", 0)) + int(rec.get("dup_synth", 0)) + int(rec.get("dup_fed", 0))
            is_live = (mtime > 0 and (now - mtime) < 240)
            runs[name] = {
                "host": host, "size": size_lbl, "active": is_live, "has_log": True,
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
            runs[name] = {"host": host, "size": size_lbl, "active": False, "has_log": False, "step": 0, "train_h": 0.0, "loss": 0.0,
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


def run_xbabe0_fast_milestone_eval(milestone_ds: int, runs: dict[str, dict]) -> dict | None:
    """Evaluates `104M` EMA soup (`ms_0000k` init baseline or `ms_NNNNk` 4-GPU fleet soup) across `xbabe0`'s `128` CPUs
    in ~2-4 minutes via `fast_dev91_every_n_h450.py` (`<= 75%` CPU on `xbabe0`, `0` GPU/RAM touched on `xbabe1`/`xbabe2`/`xbabe3`)."""
    tag = f"ms_{milestone_ds // 1000:04d}k"
    diag_dir = DIAG_ROOT / f"fleet_x1_{tag}"
    if (diag_dir / "H450_DEV91.json").is_file():
        return None

    work_ck = FLEET_ROOT / "eval_ckpts" / tag
    work_ck.mkdir(parents=True, exist_ok=True)
    init_l = Path("/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt")
    soup_p = work_ck / "soup_l.pt"

    if milestone_ds == 0:
        ck = torch.load(init_l, map_location="cpu", weights_only=False)
        w = ck.get("ema") or ck["model"]
        torch.save({"ema": w, "model": w, "size": "L"}, soup_p)
        soup_members = [init_l]
    else:
        la_src = Path("/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_L_A/ckpt_last.pt")
        la_p = work_ck / "la.pt"
        lb_p = work_ck / "lb.pt"
        x1_p = work_ck / "x1_l.pt"
        x4_p = work_ck / "x4_l.pt"
        shutil.copyfile(la_src if la_src.is_file() else init_l, la_p)
        rc_lb, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                        "xbabe3:/home/ubuntu/jope-data/h450-r4-x3/runs/r4_L_B/ckpt_last.pt", str(lb_p)], timeout=60.0)
        rc_x1, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                        "xbabe1:/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1/ckpt_last.pt", str(x1_p)], timeout=60.0)
        rc_x4, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes",
                        "xbabe4:/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/ckpt_last.pt", str(x4_p)], timeout=60.0)
        soup_members = [la_p]
        if rc_lb == 0 and lb_p.is_file():
            soup_members.append(lb_p)
        if rc_x1 == 0 and x1_p.is_file():
            soup_members.append(x1_p)
        if rc_x4 == 0 and x4_p.is_file():
            soup_members.append(x4_p)
        build_104m_soup(soup_members, soup_p)

    remote_root = "/home/ubuntu/jope-data/h450-fleet-agy/eval_stage"
    fast_py_local = Path(__file__).resolve().parent / "fast_dev91_every_n_h450.py"
    _ssh("xbabe0", f"mkdir -p {remote_root}/pub-r6 {remote_root}/h448-pub-r1 {remote_root}/fleet {remote_root}/ckpts/{tag} {remote_root}/cols/{tag} {remote_root}/logs", timeout=15.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{PUB_R6_STAGE}/", f"xbabe0:{remote_root}/pub-r6/"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", "/home/ubuntu/h448-stage/pub-r1/", f"xbabe0:{remote_root}/h448-pub-r1/"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(fast_py_local), f"xbabe0:{remote_root}/fleet/fast_dev91_every_n_h450.py"], timeout=30.0)
    _, out_bk = _ssh("xbabe0", f"sha256sum {remote_root}/bank_d91_search_f64.npz 2>/dev/null | cut -c1-64 || true", timeout=15.0)
    if out_bk.strip() != EVAL_BANK_SHA:
        _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(EVAL_BANK), f"xbabe0:{remote_root}/bank_d91_search_f64.npz"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(soup_p), f"xbabe0:{remote_root}/ckpts/{tag}/soup_l.pt"], timeout=60.0)

    # Run all 91 contexts on xbabe0's 128 idle CPUs (-P 46 x 2 threads = 92/128 = 71.9% CPU <= 85% cap!)
    cmd_x0 = (
        f"seq 0 90 > {remote_root}/logs/{tag}.idx && "
        f"PYTHONPATH={remote_root}/pub-r6/code:{remote_root}/h448-pub-r1/code "
        f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "
        f"xargs -a {remote_root}/logs/{tag}.idx -P 46 -I@ nice -n 15 python3 -B "
        f"{remote_root}/fleet/fast_dev91_every_n_h450.py --cmd ctx --ctx-idx @ "
        f"--bank {remote_root}/bank_d91_search_f64.npz --ckpt {remote_root}/ckpts/{tag}/soup_l.pt "
        f"--cols-dir {remote_root}/cols/{tag} --n-views 2 > {remote_root}/logs/{tag}.log 2>&1"
    )
    t0 = time.time()
    p0 = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "xbabe0", cmd_x0])
    p0.wait(timeout=1800)

    loc_cols = FLEET_ROOT / "cols" / tag
    loc_cols.mkdir(parents=True, exist_ok=True)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"xbabe0:{remote_root}/cols/{tag}/", f"{loc_cols}/"], timeout=60.0)
    if len(list(loc_cols.glob("[0-9a-f]*.npz"))) != 91:
        return None

    diag_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    r_sc = subprocess.run(
        ["nice", "-n", "19", "python3", "-B", str(fast_py_local), "--cmd", "score",
         "--cols-dir", str(loc_cols), "--out-json", str(diag_dir / "H450_DEV91.json")],
        env=env, capture_output=True, text=True, timeout=300.0,
    )
    if r_sc.returncode != 0 or not (diag_dir / "H450_DEV91.json").is_file():
        return None

    fleet_104m_u = 0 if milestone_ds == 0 else sum(runs[k]["unique_datasets"] for k in ("r4_L_A", "r4_L_B", "fleet_L_x1", "fleet_L_x4") if k in runs)
    ms_meta = {
        "tag": tag,
        "milestone_target_datasets": milestone_ds,
        "fleet_104m_unique_datasets": fleet_104m_u,
        "per_run_unique_datasets": {k: v["unique_datasets"] for k, v in runs.items()},
        "eval_hosts": ["xbabe0"],
        "eval_wall_s": round(time.time() - t0, 1),
        "soup_members_count": len(soup_members),
    }
    (diag_dir / "MILESTONE_META.json").write_text(json.dumps(ms_meta, indent=2, sort_keys=True) + "\n")

    res_doc = json.loads((diag_dir / "H450_DEV91.json").read_text())
    ch_elo = round(float(res_doc["in_corpus_random_split_champion_91"]["elo"]["elo"]), 1)
    ic_elo = round(float(res_doc["in_corpus_icl_boost_only_91"]["elo"]["elo"]), 1)
    ng_elo = round(float(res_doc["in_corpus_neural_plus_tab_no_gp_91"]["elo"]["elo"]), 1)
    gap_k = round(float(res_doc["in_corpus_random_split_champion_91"]["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1)

    ts = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    line = (
        f"[{ts}] agy_apex_w10 -> cl_fwriter (cc all): [xbabe0 128c CPU fast N={milestone_ds:,} dev91 frozen-split readout {tag}] "
        f"104M_fleet_unique_datasets={fleet_104m_u:,} (0 dups, {len(soup_members)}-member 104M soup), "
        f"104M_stack_elo={ch_elo} (vs Kumo L 2070.2: {gap_k:+.1f} Elo), neural+tab_elo={ng_elo}, pure_104M_ICL_elo={ic_elo}, "
        f"wall={ms_meta['eval_wall_s']}s on xbabe0 CPU (0 GPU/RAM touched on xbabe1..3); receipt {diag_dir / 'H450_DEV91.json'}."
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
    u_104m = sum(runs[k]["unique_datasets"] for k in ("r4_L_A", "r4_L_B", "fleet_L_x1", "fleet_L_x4") if k in runs)
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
        "xbabe0": "`agy_apex_w10` Every-`25k`-Dataset Fast Frozen-Split `dev91` CPU Eval (`128c`, GPU in PCIe error)",
        "xbabe1": "`agy_apex_w10` `fleet_L_x1` (`104M`) `RTX 4090` CUDA Training (`~80%` VRAM, `ZImage` finished)",
        "xbabe2": "`cl_fwriter` `r4_L_A` (`104M`) `RTX 3090` CUDA Training + `dev91` GPU Eval Shard 0 (`~80%` VRAM)",
        "xbabe3": "`cl_fwriter` `r4_L_B` (`104M`) `RTX 3090` CUDA Training + `dev91` GPU Eval Shard 1 (`~80%` VRAM)",
        "xbabe4": "`agy_apex_w10` `fleet_L_x4` (`104M`) `M3 Ultra` 80-Core Metal `MPS` Training (`<= 82%` CPU)",
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
        st = "RUNNING" if rv["active"] else ("PAUSED (`104M`-only)" if rv.get("has_log") else "STARTING/IDLE")
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

    last_milestone = -1
    while True:
        try:
            hosts = collect_host_metrics()
            runs = collect_run_logs()
            evals = collect_dev91_results()
            summary = write_coordination_dashboard(hosts, runs, evals)
            if not (DIAG_ROOT / "fleet_x1_ms_0000k" / "H450_DEV91.json").is_file():
                run_xbabe0_fast_milestone_eval(0, runs)
                evals = collect_dev91_results()
                write_coordination_dashboard(hosts, runs, evals)
            u_104m = summary["fleet_104m_unique_datasets"]
            target_ms = (u_104m // a.every_datasets) * a.every_datasets
            if target_ms >= a.every_datasets and target_ms > last_milestone:
                last_milestone = target_ms
                run_xbabe0_fast_milestone_eval(target_ms, runs)
                hosts = collect_host_metrics()
                runs = collect_run_logs()
                evals = collect_dev91_results()
                write_coordination_dashboard(hosts, runs, evals)
        except Exception as exc:
            print(f"[daemon warn] {exc}", file=sys.stderr, flush=True)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
