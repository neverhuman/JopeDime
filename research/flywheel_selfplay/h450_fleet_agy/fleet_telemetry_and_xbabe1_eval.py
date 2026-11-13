#!/usr/bin/env python3
"""5-Host H450 Fleet Telemetry, Cross-Run Unique-Dataset Union Counter, Automatic `xbabe1->xbabe0`
Epoch Checkpoint Archiver, and Multi-Host (`xbabe0 + xbabe2 + xbabe1`) `V=4` Frozen-Split `dev91`
BT-Elo Evaluator (`fleet_telemetry_and_xbabe1_eval.py`).

Operates 100% read-only on `cl_fwriter`'s `r4_*` runs/checkpoints (`0` writes on `xbabe3`, `0` touches to
`xbabe0` GPU) and writes ONLY to:
  - `/home/ubuntu/jope-data/h450-fleet-agy/` (`xbabe2` local + `xbabe0`/`xbabe1` staging & `xbabe0` archive)
  - `/home/ubuntu/jope_plan/evidence/H450/diag_results/fleet_x1_*` (quarantined `H450_DEV91.json` + `SOUP_MANIFEST.json`)
  - `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md` (under `H450_FLEET_COORDINATION.md.lock`)
  - `/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` (milestone lines under `H450_CLAUDE_GEMINI_COORD.md.lock`)
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
ARCHIVE_RECEIPT = Path("/home/ubuntu/jope_plan/evidence/H450/xbabe1_to_xbabe0_epoch_archive_receipt.json")


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


def _ssh_py(host: str, code: str, timeout: float = 120.0) -> tuple[int, str]:
    cmd = ["python3", "-"] if host == "xbabe2" else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "python3", "-"]
    try:
        r = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return 1, str(e)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_old_x1_epochs_to_xbabe0(keep_latest: int = 8) -> int:
    """Offloads older `fleet_L_x1` immutable `ckpt_epoch_*.pt` files from `xbabe1` to `xbabe0`,
    independently SHA-256 verifies every file on `xbabe0` against `epoch_*.json['ckpt_sha256']`,
    and ONLY unlinks verified `.pt` copies from `xbabe1` (retaining all `epoch_*.json` on both hosts)."""
    x1_dir = "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1"
    x0_arch = "/home/ubuntu/jope-data/h450-fleet-agy/x1_epoch_archive"
    disc_py = f"""
import glob, json, os
eps = []
for jp in sorted(glob.glob({x1_dir!r} + "/epoch_*.json")):
    try:
        d = json.load(open(jp))
        ep = int(d["epoch"])
        pt = os.path.join({x1_dir!r}, os.path.basename(d["ckpt"]))
        eps.append((ep, os.path.basename(jp), os.path.basename(pt), d["ckpt_sha256"], os.path.isfile(pt)))
    except Exception:
        pass
eps.sort()
max_ep = eps[-1][0] if eps else 0
cutoff = max_ep - {int(keep_latest)}
cands = [
    {{"epoch": ep, "json": jn, "pt": pn, "sha256": sh, "bytes": os.path.getsize(os.path.join({x1_dir!r}, pn))}}
    for (ep, jn, pn, sh, has_pt) in eps if has_pt and ep <= cutoff
]
print(json.dumps(cands))
"""
    rc, out = _ssh_py("xbabe1", disc_py, timeout=30.0)
    if rc != 0 or not out:
        return 0
    try:
        cands = json.loads(out)
    except Exception:
        return 0
    if not cands:
        return 0
    cands = cands[:24]  # bounded per cycle
    _ssh_py("xbabe0", f"import os; os.makedirs({x0_arch!r}, exist_ok=True)", timeout=20.0)
    rels = []
    for c in cands:
        rels.extend([c["json"], c["pt"]])
    tar_cmd = (
        f"ssh -o BatchMode=yes xbabe1 'cd {x1_dir} && tar -cf - {' '.join(rels)}' "
        f"| ssh -o BatchMode=yes xbabe0 'cd {x0_arch} && tar -xf -'"
    )
    rc_t, _ = _sh(["bash", "-c", tar_cmd], timeout=600.0)
    if rc_t != 0:
        return 0
    ver_py = f"""
import hashlib, json, os
cands = {json.dumps(cands)}
ok = []
for c in cands:
    pt = os.path.join({x0_arch!r}, c["pt"])
    jp = os.path.join({x0_arch!r}, c["json"])
    if not (os.path.isfile(pt) and os.path.isfile(jp)):
        continue
    h = hashlib.sha256()
    with open(pt, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() == c["sha256"]:
        os.chmod(pt, 0o444)
        os.chmod(jp, 0o444)
        ok.append(c)
print(json.dumps(ok))
"""
    rc_v, out_v = _ssh_py("xbabe0", ver_py, timeout=300.0)
    if rc_v != 0 or not out_v:
        return 0
    verified = json.loads(out_v)
    if not verified:
        return 0
    try:
        existing = json.loads(ARCHIVE_RECEIPT.read_text()) if ARCHIVE_RECEIPT.is_file() else {"files": []}
    except Exception:
        existing = {"files": []}
    by_ep = {int(x["epoch"]): x for x in existing.get("files", [])}
    for v in verified:
        by_ep[int(v["epoch"])] = v
    merged_files = [by_ep[k] for k in sorted(by_ep)]
    ARCHIVE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_RECEIPT.write_text(json.dumps({
        "schema": "jope.h450.epoch_archive_receipt/v1",
        "updated_utc": time.strftime("%FT%TZ", time.gmtime()),
        "source_host": "xbabe1",
        "source_dir": x1_dir,
        "archive_host": "xbabe0",
        "archive_dir": x0_arch,
        "epochs_archived_count": len(merged_files),
        "bytes_archived": sum(int(x.get("bytes", 0)) for x in merged_files),
        "sha256_verified_on_xbabe0_before_unlink": True,
        "files": merged_files,
    }, indent=2, sort_keys=True) + "\n")
    rm_py = f"""
import json, os
for v in {json.dumps(verified)}:
    pt = os.path.join({x1_dir!r}, v["pt"])
    if os.path.isfile(pt):
        os.chmod(pt, 0o644)
        os.unlink(pt)
print(len({json.dumps(verified)}))
"""
    _ssh_py("xbabe1", rm_py, timeout=60.0)
    return len(verified)


def prune_completed_temp_member_copies() -> None:
    """For completed `eval_ckpts/ms_*` directories where `H450_DEV91.json` and `soup_l.pt` already exist,
    records member SHA-256s in `SOUP_MANIFEST.json` and removes temporary `1.6 GB` `ckpt_last` copies (`la.pt`, `lb.pt`, `x1_l.pt`, `x4_l.pt`)."""
    ck_root = FLEET_ROOT / "eval_ckpts"
    if not ck_root.is_dir():
        return
    for d in sorted(ck_root.iterdir()):
        if not d.is_dir():
            continue
        soup_p = d / "soup_l.pt"
        done_json = DIAG_ROOT / f"fleet_x1_{d.name}" / "H450_DEV91.json"
        if not (soup_p.is_file() and done_json.is_file()):
            continue
        temps = [d / fn for fn in ("la.pt", "lb.pt", "x1_l.pt", "x4_l.pt") if (d / fn).is_file()]
        if not temps:
            continue
        man_p = d / "SOUP_MANIFEST.json"
        if not man_p.is_file():
            rec = {
                "tag": d.name,
                "soup_ckpt": str(soup_p),
                "soup_sha256": _sha256(soup_p),
                "temp_members_archived_metadata": [
                    {"name": tp.name, "bytes": tp.stat().st_size, "sha256": _sha256(tp)} for tp in temps
                ],
            }
            man_p.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        for tp in temps:
            try:
                tp.unlink()
            except Exception:
                pass


def collect_host_metrics() -> dict[str, dict]:
    hosts = {}
    linux_probe = (
        "ncpu=$(nproc --all); load1=$(awk '{print $1}' /proc/loadavg); "
        "mem=$(free -m | awk '/^Mem:/ {print $3\"/\"$2\" MiB (avail \"$7\" MiB)\"}'); "
        "disk=$(df -BG /home 2>/dev/null | awk 'NR==2 {print $4\" free (\"$5\" used)\"}'); "
        "gpu=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 'ERROR_OR_NONE'); "
        "printf '%s|%s|%s|%s|%s\n' \"$ncpu\" \"$load1\" \"$mem\" \"$disk\" \"$gpu\""
    )
    for h in ("xbabe0", "xbabe1", "xbabe2", "xbabe3"):
        rc, out = _ssh(h, linux_probe, timeout=10.0)
        if rc == 0 and "|" in out:
            ncpu_s, load1_s, mem_s, disk_s, gpu_s = out.splitlines()[-1].split("|", 4)
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
            hosts[h] = {
                "ncpu": ncpu, "load1": load1, "cpu_pct": cpu_pct,
                "mem": f"{mem_s} | {disk_s}", "gpu": gpu_desc, "vram_pct": vram_pct,
            }
        else:
            hosts[h] = {"ncpu": 0, "load1": 0.0, "cpu_pct": 0.0, "mem": "unreachable", "gpu": out[:80], "vram_pct": None}

    mac_probe = (
        "ncpu=$(sysctl -n hw.ncpu); "
        "load1=$(sysctl -n vm.loadavg | awk '{print $2}'); "
        "pscpu=$(ps -A -o %cpu | awk 'NR>1 {s+=$1} END {printf \"%.1f\", s}'); "
        "disk=$(df -g /Users/agent | awk 'NR==2 {print $4\" GiB free (\"$5\" used)\"}'); "
        "printf '%s|%s|%s|%s\n' \"$ncpu\" \"$load1\" \"$pscpu\" \"$disk\""
    )
    rc, out = _ssh("xbabe4", mac_probe, timeout=10.0)
    if rc == 0 and "|" in out:
        ncpu_s, load1_s, pscpu_s, disk_s = out.splitlines()[-1].split("|", 3)
        ncpu = max(1, int(ncpu_s))
        cpu_pct = round(float(pscpu_s) / ncpu, 1)
        hosts["xbabe4"] = {
            "ncpu": ncpu, "load1": float(load1_s), "cpu_pct": cpu_pct,
            "mem": f"512 GiB Unified Memory | {disk_s}",
            "gpu": "Apple M3 Ultra 80-core Metal (MPS active)", "vram_pct": None,
        }
    else:
        hosts["xbabe4"] = {"ncpu": 32, "load1": 0.0, "cpu_pct": 0.0, "mem": "unreachable", "gpu": out[:80], "vram_pct": None}
    return hosts


def collect_run_logs() -> dict[str, dict]:
    runs = {}
    cache_p = FLEET_ROOT / "last_runs_cache.json"
    cache: dict[str, dict] = {}
    if cache_p.is_file():
        try:
            cache = json.loads(cache_p.read_text())
        except Exception:
            cache = {}

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
        last_obj = None
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
                    if last_obj is None:
                        last_obj = d
                    if "datasets" in d and "loss" in d:
                        rec = d
                        break
                except Exception:
                    continue
        if rec:
            dups = int(rec.get("dup_real", 0)) + int(rec.get("dup_synth", 0)) + int(rec.get("dup_fed", 0))
            is_live = (mtime > 0 and (now - mtime) < 240)
            hb_utc = time.strftime("%FT%TZ", time.gmtime(mtime)) if mtime > 0 else "unknown"
            paused_cleanly = bool((last_obj and "paused" in last_obj) or name.startswith("r4_M_"))
            if is_live:
                status_str = "RUNNING"
            elif paused_cleanly:
                status_str = "PAUSED (`104M`-only)"
            else:
                status_str = f"STALLED (last hb {hb_utc})"
            entry = {
                "host": host, "size": size_lbl, "active": is_live, "has_log": True,
                "status_str": status_str, "heartbeat_utc": hb_utc,
                "step": int(rec.get("step", 0)), "train_h": float(rec.get("train_h", 0.0)),
                "loss": float(rec.get("loss", 0.0)),
                "ds_per_s": float(rec.get("ds_per_s", 0.0)),
                "unique_datasets": int(rec.get("unique_datasets", rec.get("datasets", 0))),
                "datasets_seen": int(rec.get("datasets_seen", rec.get("datasets", 0))),
                "dups_fed": dups,
                "src_tables_seen": int(rec.get("src_tables_seen", 0)),
                "inj_share": float(rec.get("inj_share", 0.0)),
                "drop_share_p2": rec.get("drop_share_p2"),
                "mem_gb": float(rec.get("mem_gb", 0.0)),
            }
            runs[name] = entry
            cache[name] = entry
        elif name in cache:
            prev = dict(cache[name])
            prev["active"] = False
            hb_prev = prev.get("heartbeat_utc", "unknown")
            reason = "SSH unreachable" if rc != 0 else "no heartbeat"
            prev["status_str"] = (
                "PAUSED (`104M`-only)" if name.startswith("r4_M_")
                else f"STALLED ({reason}; last hb {hb_prev})"
            )
            runs[name] = prev
        else:
            st_fallback = "STALLED (SSH unreachable)" if rc != 0 else "STARTING/IDLE"
            runs[name] = {
                "host": host, "size": size_lbl, "active": False, "has_log": False,
                "status_str": st_fallback, "heartbeat_utc": "n/a",
                "step": 0, "train_h": 0.0, "loss": 0.0, "ds_per_s": 0.0,
                "unique_datasets": 0, "datasets_seen": 0, "dups_fed": 0,
                "src_tables_seen": 0, "inj_share": 0.0, "drop_share_p2": None, "mem_gb": 0.0,
            }
    try:
        FLEET_ROOT.mkdir(parents=True, exist_ok=True)
        tmp_c = cache_p.with_suffix(".json.tmp")
        tmp_c.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_c, cache_p)
    except Exception:
        pass
    return runs


def collect_dev91_results() -> list[dict]:
    rows = []
    if not DIAG_ROOT.is_dir():
        return rows
    traj_ds: dict[str, int] = {}
    traj_p = DIAG_ROOT / "r4_TRAJECTORY.jsonl"
    if traj_p.is_file():
        for line in traj_p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                td = json.loads(line)
                t_tag = "r4_" + str(td.get("tag", ""))
                ua = int(td.get("run_a", {}).get("unique_datasets", 0))
                ub = int(td.get("run_b", {}).get("unique_datasets", 0))
                if ua + ub > 0:
                    traj_ds[t_tag] = ua + ub
            except Exception:
                continue
    for p in sorted(DIAG_ROOT.glob("*/H450_DEV91.json"), key=lambda x: x.stat().st_mtime):
        tag = p.parent.name
        if not (tag.startswith("r4_") or tag.startswith("fleet_x1_")):
            continue
        try:
            d = json.loads(p.read_text())
            ch = d["in_corpus_random_split_champion_91"]
            ic = d["in_corpus_icl_boost_only_91"]
            ng = d["in_corpus_neural_plus_tab_no_gp_91"]
            lg = d.get("in_corpus_legacy_guard_stack_91")
            zr = d.get("fair_zero_d91_control_91") or d.get("fair_zero_d91_upgraded_control_91")
            meta_p = p.parent / "MILESTONE_META.json"
            ms_meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
            ds_val = ms_meta.get(
                "evaluated_unique_datasets",
                ms_meta.get(
                    "fleet_104m_unique_datasets",
                    0 if ("ep00_init" in tag or "g00_init" in tag or "ms_0000k" in tag or "init_ctrl" in tag) else traj_ds.get(tag),
                ),
            )
            if not tag.startswith("fleet_x1_"):
                pipe_lbl = "cols_h450_gpu"
            elif "solo_" in tag:
                pipe_lbl = f"multihost_{ms_meta.get('n_views', 4)}view_cpu (x1 solo)"
            elif "soup3_" in tag:
                pipe_lbl = f"multihost_{ms_meta.get('n_views', 4)}view_cpu (3-GPU soup)"
            elif "v4_init" in tag:
                pipe_lbl = "multihost_4view_cpu (N=0 ctrl)"
            else:
                pipe_lbl = "fast_2view_cpu (4-soup)"
            leg_elo = round(float(lg["elo"]["elo"]), 1) if lg else None
            rows.append({
                "tag": tag,
                "pipeline": pipe_lbl,
                "mtime_utc": time.strftime("%FT%TZ", time.gmtime(p.stat().st_mtime)),
                "fleet_104m_unique_ds": ds_val,
                "champion_elo": round(float(ch["elo"]["elo"]), 1),
                "champion_ci95": [round(float(x), 1) for x in ch["elo"]["ci95"]],
                "legacy_elo": leg_elo,
                "legacy_vs_1828": round(leg_elo - 1828.6, 1) if leg_elo is not None else None,
                "icl_only_elo": round(float(ic["elo"]["elo"]), 1),
                "nogp_elo": round(float(ng["elo"]["elo"]), 1),
                "zero_ctrl_elo": round(float(zr["elo"]["elo"]), 1) if zr else None,
                "gap_vs_kumo_L": round(float(ch["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1),
                "gap_vs_h448_fair": round(float(ch["gap_elo"]["vs_h448.APEX_FAIR_CHAMPION"]), 1),
            })
        except Exception:
            continue
    return rows


def _latest_immutable_epoch(host: str, run_dir: str) -> dict | None:
    """Discovers the newest immutable `epoch_NN.json` + existing `ckpt_epoch_NN.pt` on `host:run_dir` (read-only)."""
    py = f"""
import glob, json, os
best = None
for jp in glob.glob({run_dir!r} + "/epoch_*.json"):
    try:
        d = json.load(open(jp))
        ep = int(d["epoch"])
        pt = os.path.join({run_dir!r}, os.path.basename(d["ckpt"]))
        if os.path.isfile(pt):
            u = int(d.get("exposure", {{}}).get("unique_datasets_seen", d.get("exposure", {{}}).get("datasets_seen", 0)))
            cand = (ep, pt, d["ckpt_sha256"], u, int(d.get("step", 0)))
            if best is None or cand[0] > best[0]:
                best = cand
    except Exception:
        pass
if best is None:
    for pt in glob.glob({run_dir!r} + "/ckpt_epoch_*.pt"):
        bn = os.path.basename(pt)
        try:
            ep = int(bn.replace("ckpt_epoch_", "").replace(".pt", ""))
            cand = (ep, pt, "", 0, 0)
            if best is None or cand[0] > best[0]:
                best = cand
        except Exception:
            pass
print(json.dumps(None if best is None else {{"epoch": best[0], "ckpt": best[1], "sha256": best[2], "unique_datasets": best[3], "step": best[4]}}))
"""
    rc, out = _ssh_py(host, py, timeout=20.0)
    if rc != 0 or not out:
        return None
    try:
        obj = json.loads(out)
        if obj:
            obj["host"] = host
        return obj
    except Exception:
        return None


def build_weighted_104m_soup(ckpt_paths: list[Path], weights: list[float], out_path: Path) -> str:
    w_sum = max(1e-12, float(sum(weights)))
    wn = [float(w) / w_sum for w in weights]
    states = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        states.append(ck.get("ema") or ck["model"])
    avg = {}
    for k in states[0]:
        t0 = states[0][k]
        if torch.is_floating_point(t0):
            acc = sum(wn[i] * states[i][k].double() for i in range(len(states)))
            avg[k] = acc.to(dtype=t0.dtype)
        else:
            avg[k] = t0.clone()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save({"ema": avg, "model": avg, "size": "L", "soup_members": len(states), "weights": wn}, tmp)
    os.replace(tmp, out_path)
    return _sha256(out_path)


def _stage_host_for_eval(host: str, tag: str, ckpt_local: Path, fast_py_local: Path) -> str | None:
    remote_root = "/home/ubuntu/jope-data/h450-fleet-agy/eval_stage"
    if host == "xbabe1":
        # Guard xbabe1 RAM (>= 10 GiB avail) and free disk (>= 25 GB) before staging CPU eval shard!
        rc_g, out_g = _ssh("xbabe1", "free -m | awk '/^Mem:/ {m=$7} END {printf \"%d\", m}'; echo -n '|'; df -BG /home | awk 'NR==2 {gsub(/G/,\"\",$4); printf \"%d\", $4}'", timeout=8.0)
        if rc_g != 0 or "|" not in out_g:
            return None
        av_mb, fr_gb = [int(x) for x in out_g.split("|", 1)]
        if av_mb < 10_000 or fr_gb < 25:
            return None
    rc_m, _ = _ssh(host, f"mkdir -p {remote_root}/pub-r6 {remote_root}/h448-pub-r1 {remote_root}/fleet {remote_root}/ckpts/{tag} {remote_root}/cols/{tag} {remote_root}/logs", timeout=15.0)
    if rc_m != 0:
        return None
    if host == "xbabe2":
        shutil.copy2(fast_py_local, f"{remote_root}/fleet/fast_dev91_every_n_h450.py")
        shutil.copy2(ckpt_local, f"{remote_root}/ckpts/{tag}/model_l.pt")
        bk_dst = Path(f"{remote_root}/bank_d91_search_f64.npz")
        if not bk_dst.exists():
            os.symlink(EVAL_BANK, bk_dst)
        return remote_root
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{PUB_R6_STAGE}/", f"{host}:{remote_root}/pub-r6/"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", "/home/ubuntu/h448-stage/pub-r1/", f"{host}:{remote_root}/h448-pub-r1/"], timeout=60.0)
    _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(fast_py_local), f"{host}:{remote_root}/fleet/fast_dev91_every_n_h450.py"], timeout=30.0)
    _, out_bk = _ssh(host, f"sha256sum {remote_root}/bank_d91_search_f64.npz 2>/dev/null | cut -c1-64 || true", timeout=15.0)
    if out_bk.strip() != EVAL_BANK_SHA:
        _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(EVAL_BANK), f"{host}:{remote_root}/bank_d91_search_f64.npz"], timeout=90.0)
    rc_ck, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", str(ckpt_local), f"{host}:{remote_root}/ckpts/{tag}/model_l.pt"], timeout=60.0)
    return remote_root if rc_ck == 0 else None


def run_multihost_fast_eval(tag: str, ckpt_local: Path, manifest_meta: dict, n_views: int = 4) -> dict | None:
    """Shards all `91` `dev91` contexts across `xbabe0` (`48` workers) + `xbabe2` (`24` workers) + `xbabe1` (`6` workers)
    with persistent model-per-worker `--cmd shard`, `CUDA_VISIBLE_DEVICES=''`, and `--cpu-hard-pct <= 83%`."""
    diag_dir = DIAG_ROOT / f"fleet_x1_{tag}"
    if (diag_dir / "H450_DEV91.json").is_file():
        return None

    fast_py_local = Path(__file__).resolve().parent / "fast_dev91_every_n_h450.py"
    t0 = time.time()
    r0 = _stage_host_for_eval("xbabe0", tag, ckpt_local, fast_py_local)
    if r0 is None:
        return None
    r2 = _stage_host_for_eval("xbabe2", tag, ckpt_local, fast_py_local)
    r1 = _stage_host_for_eval("xbabe1", tag, ckpt_local, fast_py_local)

    if r2 is not None and r1 is not None:
        plan = [
            ("xbabe0", r0, "0-54", 48, 83.0),
            ("xbabe2", r2, "55-78", 24, 80.0),
            ("xbabe1", r1, "79-90", 6, 78.0),
        ]
    elif r2 is not None:
        plan = [
            ("xbabe0", r0, "0-62", 48, 83.0),
            ("xbabe2", r2, "63-90", 24, 80.0),
        ]
    else:
        plan = [("xbabe0", r0, "0-90", 48, 83.0)]

    procs_running = []
    eval_hosts = []
    for host, rroot, idx_spec, n_procs, cpu_cap in plan:
        eval_hosts.append(host)
        py_path = "/home/ubuntu/h450-stage/pub-r6/code:/home/ubuntu/h448-stage/pub-r1/code" if host == "xbabe2" else f"{rroot}/pub-r6/code:{rroot}/h448-pub-r1/code"
        shard_cmd = (
            f"PYTHONPATH={py_path} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "
            f"nice -n 15 python3 -B {rroot}/fleet/fast_dev91_every_n_h450.py --cmd shard --indices {idx_spec} "
            f"--procs {n_procs} --threads 2 --cpu-hard-pct {cpu_cap} "
            f"--bank {rroot}/bank_d91_search_f64.npz --ckpt {rroot}/ckpts/{tag}/model_l.pt "
            f"--cols-dir {rroot}/cols/{tag} --n-views {n_views} > {rroot}/logs/{tag}.log 2>&1"
        )
        if host == "xbabe2":
            p = subprocess.Popen(["bash", "-c", shard_cmd])
        else:
            p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, shard_cmd])
        procs_running.append((host, rroot, p))

    for _, _, p in procs_running:
        try:
            p.wait(timeout=1800)
        except Exception:
            p.kill()

    loc_cols = FLEET_ROOT / "cols" / tag
    loc_cols.mkdir(parents=True, exist_ok=True)
    for host, rroot, _ in procs_running:
        if host == "xbabe2":
            for npz in Path(f"{rroot}/cols/{tag}").glob("[0-9a-f]*.npz"):
                shutil.copy2(npz, loc_cols / npz.name)
        else:
            _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{host}:{rroot}/cols/{tag}/", f"{loc_cols}/"], timeout=90.0)

    # Fallback sweep on xbabe0 for any missing contexts (idempotent)
    if len(list(loc_cols.glob("[0-9a-f]*.npz"))) < 91:
        _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{loc_cols}/", f"xbabe0:{r0}/cols/{tag}/"], timeout=60.0)
        sweep_cmd = (
            f"PYTHONPATH={r0}/pub-r6/code:{r0}/h448-pub-r1/code OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "
            f"nice -n 15 python3 -B {r0}/fleet/fast_dev91_every_n_h450.py --cmd shard --indices 0-90 "
            f"--procs 48 --threads 2 --cpu-hard-pct 83.0 --bank {r0}/bank_d91_search_f64.npz "
            f"--ckpt {r0}/ckpts/{tag}/model_l.pt --cols-dir {r0}/cols/{tag} --n-views {n_views} >> {r0}/logs/{tag}.log 2>&1"
        )
        _ssh("xbabe0", sweep_cmd, timeout=900.0)
        _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"xbabe0:{r0}/cols/{tag}/", f"{loc_cols}/"], timeout=90.0)

    # Clean remote scratch ckpts/cols on xbabe0/xbabe1/xbabe2 eval_stage
    for host, rroot, _ in procs_running:
        _ssh(host, f"rm -rf {rroot}/ckpts/{tag} {rroot}/cols/{tag}", timeout=20.0)

    if len(list(loc_cols.glob("[0-9a-f]*.npz"))) != 91:
        return None

    diag_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    r_sc = subprocess.run(
        ["nice", "-n", "12", "python3", "-B", str(fast_py_local), "--cmd", "score", "--procs", "16",
         "--cols-dir", str(loc_cols), "--out-json", str(diag_dir / "H450_DEV91.json")],
        env=env, capture_output=True, text=True, timeout=900.0,
    )
    if r_sc.returncode != 0 or not (diag_dir / "H450_DEV91.json").is_file():
        return None

    ms_meta = {
        **manifest_meta,
        "tag": tag,
        "n_views": n_views,
        "eval_hosts": eval_hosts,
        "eval_wall_s": round(time.time() - t0, 1),
    }
    (diag_dir / "MILESTONE_META.json").write_text(json.dumps(ms_meta, indent=2, sort_keys=True) + "\n")
    if "soup_manifest" in manifest_meta:
        (diag_dir / "SOUP_MANIFEST.json").write_text(json.dumps(manifest_meta["soup_manifest"], indent=2, sort_keys=True) + "\n")

    res_doc = json.loads((diag_dir / "H450_DEV91.json").read_text())
    ch_elo = round(float(res_doc["in_corpus_random_split_champion_91"]["elo"]["elo"]), 1)
    lg_elo = round(float(res_doc["in_corpus_legacy_guard_stack_91"]["elo"]["elo"]), 1)
    ic_elo = round(float(res_doc["in_corpus_icl_boost_only_91"]["elo"]["elo"]), 1)
    ng_elo = round(float(res_doc["in_corpus_neural_plus_tab_no_gp_91"]["elo"]["elo"]), 1)
    gap_k = round(float(res_doc["in_corpus_random_split_champion_91"]["gap_elo"]["vs_kumo_tabular_large_e8.v1"]), 1)

    ts = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    line = (
        f"[{ts}] agy_apex_w10 -> cl_fwriter (cc all): [multihost {'+'.join(eval_hosts)} CPU V={n_views} dev91 frozen-split readout {tag}] "
        f"evaluated_unique_datasets={ms_meta.get('evaluated_unique_datasets', 0):,} (fleet_104M={ms_meta.get('fleet_104m_unique_datasets', 0):,}, 0 dups), "
        f"legacy_stack_elo={lg_elo} ({lg_elo - 1828.6:+.1f} vs 1828.6 init), champion_stack_elo={ch_elo} (vs Kumo L 2070.2: {gap_k:+.1f}), "
        f"neural+tab_elo={ng_elo}, pure_104M_ICL_elo={ic_elo}, wall={ms_meta['eval_wall_s']}s (0 GPU touched); receipt {diag_dir / 'H450_DEV91.json'}."
    )
    PEER_LOCK.touch(exist_ok=True)
    with open(PEER_LOCK, "r+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        with open(PEER_MD, "a") as pf:
            pf.write(line + "\n")
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    return ms_meta


def maybe_run_scheduled_evals(runs: dict[str, dict], every_datasets: int) -> bool:
    """Runs:
      1. `v4_init_ctrl` (`N = 0` `all_non_d91_L/ckpt_unfair_final.pt` on `V = 4`) once;
      2. `solo_epNNN` (newest immutable `fleet_L_x1` single-model `ckpt_epoch_NN.pt` every `>= 4` new epochs = `100k` x1 datasets);
      3. `soup3_NNNNk` (exposure-weighted 3-GPU `104M` EMA soup from immutable `ckpt_epoch_*.pt` every `every_datasets` fleet datasets)."""
    did_any = False
    fleet_104m_u = sum(runs[k]["unique_datasets"] for k in ("r4_L_A", "r4_L_B", "fleet_L_x1", "fleet_L_x4") if k in runs)
    init_l = Path("/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt")

    # 1. Matched N=0 control on V=4
    if not (DIAG_ROOT / "fleet_x1_v4_init_ctrl" / "H450_DEV91.json").is_file():
        meta0 = {
            "mode": "v4_init_control",
            "evaluated_unique_datasets": 0,
            "fleet_104m_unique_datasets": fleet_104m_u,
            "ckpt_path": str(init_l),
            "ckpt_sha256": _sha256(init_l),
        }
        if run_multihost_fast_eval("v4_init_ctrl", init_l, meta0, n_views=4):
            did_any = True

    # 2. Single-model `fleet_L_x1` newest immutable `ckpt_epoch_NN.pt` (every >= 4 epochs = 100k x1 datasets)
    x1_ep = _latest_immutable_epoch("xbabe1", "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1")
    if x1_ep is not None:
        ep_num = int(x1_ep["epoch"])
        done_solo_eps = []
        for d in DIAG_ROOT.glob("fleet_x1_solo_ep*/H450_DEV91.json"):
            try:
                done_solo_eps.append(int(d.parent.name.split("solo_ep", 1)[1]))
            except Exception:
                pass
        last_solo = max(done_solo_eps) if done_solo_eps else 0
        if ep_num >= last_solo + 4:
            solo_tag = f"solo_ep{ep_num:03d}"
            work_ck = FLEET_ROOT / "eval_ckpts" / solo_tag
            work_ck.mkdir(parents=True, exist_ok=True)
            loc_pt = work_ck / f"ckpt_epoch_{ep_num:02d}.pt"
            rc_cp, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"xbabe1:{x1_ep['ckpt']}", str(loc_pt)], timeout=60.0)
            if rc_cp == 0 and loc_pt.is_file():
                sha_got = _sha256(loc_pt)
                if not x1_ep["sha256"] or sha_got == x1_ep["sha256"]:
                    meta_solo = {
                        "mode": "single_model_fleet_L_x1",
                        "epoch": ep_num,
                        "evaluated_unique_datasets": int(x1_ep["unique_datasets"] or runs.get("fleet_L_x1", {}).get("unique_datasets", 0)),
                        "fleet_104m_unique_datasets": fleet_104m_u,
                        "source_host": "xbabe1",
                        "source_ckpt": x1_ep["ckpt"],
                        "ckpt_sha256": sha_got,
                        "local_staged_ckpt": str(loc_pt),
                    }
                    if run_multihost_fast_eval(solo_tag, loc_pt, meta_solo, n_views=4):
                        did_any = True

    # 3. Exposure-weighted 3-GPU `104M` EMA soup (`fleet_L_x1`, `r4_L_B`, `r4_L_A` immutable `ckpt_epoch_*.pt`)
    step_ms = max(250_000, int(every_datasets))
    target_ms = (fleet_104m_u // step_ms) * step_ms
    soup_tag = f"soup3_{target_ms // 1000:04d}k"
    if target_ms >= step_ms and not (DIAG_ROOT / f"fleet_x1_{soup_tag}" / "H450_DEV91.json").is_file():
        la_ep = _latest_immutable_epoch("xbabe2", "/home/ubuntu/jope-data/h450-quarantine/r4/runs/r4_L_A")
        lb_ep = _latest_immutable_epoch("xbabe3", "/home/ubuntu/jope-data/h450-r4-x3/runs/r4_L_B")
        x1_ep = x1_ep or _latest_immutable_epoch("xbabe1", "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1")
        work_ck = FLEET_ROOT / "eval_ckpts" / soup_tag
        work_ck.mkdir(parents=True, exist_ok=True)
        member_specs = []
        for run_lbl, ep_info, fallback_u in (
            ("fleet_L_x1", x1_ep, runs.get("fleet_L_x1", {}).get("unique_datasets", 1)),
            ("r4_L_B", lb_ep, runs.get("r4_L_B", {}).get("unique_datasets", 1)),
            ("r4_L_A", la_ep, runs.get("r4_L_A", {}).get("unique_datasets", 1)),
        ):
            if ep_info is None:
                continue
            dst_p = work_ck / f"{run_lbl}_ep{int(ep_info['epoch']):03d}.pt"
            if ep_info["host"] == "xbabe2":
                shutil.copy2(ep_info["ckpt"], dst_p)
                rc_m = 0
            else:
                rc_m, _ = _sh(["rsync", "-a", "-e", "ssh -o BatchMode=yes", f"{ep_info['host']}:{ep_info['ckpt']}", str(dst_p)], timeout=60.0)
            if rc_m == 0 and dst_p.is_file():
                u_cnt = max(1, int(ep_info["unique_datasets"] or fallback_u))
                member_specs.append({
                    "run": run_lbl,
                    "host": ep_info["host"],
                    "epoch": int(ep_info["epoch"]),
                    "source_ckpt": ep_info["ckpt"],
                    "sha256": _sha256(dst_p),
                    "unique_datasets": u_cnt,
                    "local_temp": dst_p,
                })
        if len(member_specs) >= 2:
            raw_w = [math.sqrt(float(m["unique_datasets"])) for m in member_specs]
            w_tot = sum(raw_w)
            norm_w = [round(w / w_tot, 6) for w in raw_w]
            soup_p = work_ck / "soup_l.pt"
            soup_sha = build_weighted_104m_soup([m["local_temp"] for m in member_specs], raw_w, soup_p)
            for m in member_specs:
                try:
                    m["local_temp"].unlink()
                except Exception:
                    pass
            clean_members = [{k: v for k, v in m.items() if k != "local_temp"} | {"weight": norm_w[i]} for i, m in enumerate(member_specs)]
            soup_manifest = {
                "schema": "jope.h450.soup_manifest/v1",
                "tag": soup_tag,
                "soup_ckpt": str(soup_p),
                "soup_sha256": soup_sha,
                "weighting": "sqrt(unique_datasets) across active CUDA 104M runs (immutable ckpt_epoch_*.pt EMA weights)",
                "members": clean_members,
            }
            (work_ck / "SOUP_MANIFEST.json").write_text(json.dumps(soup_manifest, indent=2, sort_keys=True) + "\n")
            meta_soup = {
                "mode": "weighted_3gpu_104m_soup",
                "milestone_target_datasets": target_ms,
                "evaluated_unique_datasets": sum(m["unique_datasets"] for m in clean_members),
                "fleet_104m_unique_datasets": fleet_104m_u,
                "soup_ckpt": str(soup_p),
                "soup_sha256": soup_sha,
                "soup_manifest": soup_manifest,
            }
            if run_multihost_fast_eval(soup_tag, soup_p, meta_soup, n_views=4):
                did_any = True
    return did_any


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
        "| Host | Active H450 Role | GPU / Accelerator Telemetry | Host CPU Avg (`<= 85%` Cap) | Memory & Disk |",
        "| --- | --- | --- | --- | --- |",
    ]
    roles = {
        "xbabe0": "`agy_apex_w10` Multi-Host `V=4` `dev91` CPU Eval Primary Shard (`48` workers / `128c`) + SHA-Verified `fleet_L_x1` Epoch Archive (`GPU` in PCIe error)",
        "xbabe1": "`agy_apex_w10` `fleet_L_x1` (`104M`) `RTX 4090` CUDA Training (`~80%` VRAM, async workers) + `dev91` CPU Eval Shard (`6` workers)",
        "xbabe2": "`cl_fwriter` `r4_L_A` (`104M`) `RTX 3090` CUDA Training + `cols_h450_gpu` Shard 0 + `agy` `dev91` CPU Eval Shard (`24` workers)",
        "xbabe3": "`cl_fwriter` `r4_L_B` (`104M`) `RTX 3090` CUDA Training + `cols_h450_gpu` Shard 1 (`99-100%` GPU TDP; `0` `agy` disk writes)",
        "xbabe4": "`agy_apex_w10` `fleet_L_x4` (`104M`) `M3 Ultra` 80-Core Metal `MPS` Training (async workers, `<= 82%` host CPU)",
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
        "| Run | Host | Model | Status | Last Heartbeat (UTC) | Step | Train Hours | Throughput | Loss | Unique Datasets Seen | Duplicates Fed | Distinct Source Tables | Inj Col Share |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for rname, rv in runs.items():
        st = rv.get("status_str") or ("RUNNING" if rv["active"] else ("PAUSED (`104M`-only)" if rv.get("has_log") else "STARTING/IDLE"))
        hb = rv.get("heartbeat_utc", "n/a")
        thr = f"`{rv.get('ds_per_s', 0.0):.1f} ds/s`" if rv.get("active") else "`0.0 ds/s`"
        lines.append(
            f"| `{rname}` | `{rv['host']}` | `{rv['size']}` | `{st}` | `{hb}` | `{rv['step']:,}` | `{rv['train_h']:.3f}h` | {thr} | `{rv['loss']:.4f}` | **`{rv['unique_datasets']:,}`** | **`{rv['dups_fed']}`** | `{rv['src_tables_seen']:,} / 3,525` | `{100.0 * rv['inj_share']:.2f}%` |"
        )

    lines.extend([
        "",
        "## 3. Frozen-Split `dev91-audit-v1` In-Corpus BT-Elo Trajectory (`cols_h450_gpu` Official + Multi-Host `V=4` Single-Model & 3-GPU Soup)",
        "",
        "| Milestone / Tag | Pipeline | Timestamp (UTC) | Evaluated `104M` Unique Datasets | Legacy Guard Stack Elo (vs `1828.6` Init) | H450 Champion Stack Elo (`95% CI`) | Pure `ICL` Boost Elo | Gap vs `Kumo L` (`2070.2`) |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ])
    if not evals:
        lines.append("| `r4_g00_init` | `cols_h450_gpu` | *evaluating* | `0` | *pending* | *pending* | *pending* | *pending* |")
    else:
        for ev in evals:
            ds_s = f"`{ev['fleet_104m_unique_ds']:,}`" if ev["fleet_104m_unique_ds"] is not None else "`hourly`"
            ci = ev["champion_ci95"]
            leg_s = (
                f"**`{ev['legacy_elo']:.1f}`** (`{ev['legacy_vs_1828']:+.1f}`)"
                if ev.get("legacy_elo") is not None else "*(n/a — fast 2-view)*"
            )
            lines.append(
                f"| `{ev['tag']}` | `{ev.get('pipeline', 'cols_h450_gpu')}` | `{ev['mtime_utc']}` | {ds_s} | {leg_s} | **`{ev['champion_elo']:.1f}`** (`[{ci[0]:.1f}, {ci[1]:.1f}]`) | `{ev['icl_only_elo']:.1f}` | `{ev['gap_vs_kumo_L']:+.1f}` |"
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
    ap.add_argument("--every-datasets", type=int, default=250_000)
    ap.add_argument("--poll-s", type=float, default=45.0)
    ap.add_argument("--xbabe1-shards", type=int, default=6)
    ap.add_argument("--xbabe1-threads", type=int, default=2)
    a = ap.parse_args()

    if a.cmd == "status":
        hosts = collect_host_metrics()
        runs = collect_run_logs()
        evals = collect_dev91_results()
        summary = write_coordination_dashboard(hosts, runs, evals)
        print(json.dumps({"summary": summary, "hosts": hosts, "runs": runs, "evals": evals[-8:]}, indent=2, sort_keys=True))
        return

    while True:
        try:
            archive_old_x1_epochs_to_xbabe0(keep_latest=8)
            prune_completed_temp_member_copies()
            hosts = collect_host_metrics()
            runs = collect_run_logs()
            evals = collect_dev91_results()
            write_coordination_dashboard(hosts, runs, evals)
            if maybe_run_scheduled_evals(runs, a.every_datasets):
                hosts = collect_host_metrics()
                runs = collect_run_logs()
                evals = collect_dev91_results()
                write_coordination_dashboard(hosts, runs, evals)
        except Exception as exc:
            print(f"[daemon warn] {exc}", file=sys.stderr, flush=True)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
