#!/usr/bin/env python3
"""Event-Driven Cooled Wave-2 (`H454`) & Cosine-Valley (`H455`) Checkpoint Watcher & Scorer
(`watch_and_score_wave2_h454.py`).

Runs continuously as a lightweight background daemon (`0` `/schedule` timers, `0` trainer interruptions,
`0` `CONFIRM` / `FMGAP B/C` / `HOLDOUT` / `SEALED` opens, `0` deletions, `0` `/tmp` paths):
  1. Polls `xbabe1` `fleet_L_x1/log.jsonl` and `fleet_L_x1_h455_maxent/log.jsonl` + `xbabe4` `fleet_L_x4/log.jsonl`.
  2. Automatically snapshots, pins (`~/ops/ckpt_pruner/pins.txt`), extracts `91/91` `2xCV-7` columns (`ema_soup50`
     and `ema_cooldown`), scores against the 33-model `dev91` board (`score_h454_paired_flywheel_matrix.py`), and
     appends `flock`-serialized receipts to `/home/ubuntu/JopeDime/AGENT_CHAT.md` whenever:
       - `fleet_L_x1` crosses `lr <= 5.0e-6` (`w2b_lr5e6`)
       - `fleet_L_x1` crosses `lr <= 3.25e-6` (`w2c_floor3e6`)
       - `fleet_L_x1_h455_maxent` saves a new cosine-valley checkpoint `ckpt_valley_NNN.pt` (`h455_valley_NNN`).
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

STAGE_CODE = Path("/home/ubuntu/h452-stage/code")
FLYWHEEL_ROOT = Path("/home/ubuntu/jope-data/h452-flywheel")
PINS_FILE = Path("/home/ubuntu/ops/ckpt_pruner/pins.txt")
AGENT_CHAT = Path("/home/ubuntu/JopeDime/AGENT_CHAT.md")
AGENT_LOCK = Path("/home/ubuntu/JopeDime/AGENT_CHAT.md.lock")
BANK_D91 = Path("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
GP_ANCHORS = FLYWHEEL_ROOT / "gp_anchors_p0"
GP_OOF_CACHE = FLYWHEEL_ROOT / "gp_anchors_oof_cache_v1"


def sha256_file(p: Path) -> str:
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin_paths(paths: list[Path], note: str) -> None:
    PINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = set(PINS_FILE.read_text().splitlines()) if PINS_FILE.is_file() else set()
    lines = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in existing:
            lines.append(rp)
    if lines:
        stamp = time.strftime("%FT%TZ", time.gmtime())
        with open(PINS_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"# {stamp} {note}\n")
            for ln in lines:
                fh.write(ln + "\n")


def append_agent_chat(msg: str) -> None:
    if not AGENT_CHAT.is_file():
        return
    with open(AGENT_LOCK, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            with open(AGENT_CHAT, "a", encoding="utf-8") as cf:
                cf.write("\n" + msg.strip() + "\n")
                cf.flush()
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def remote_tail_json(host: str, remote_log: str) -> dict | None:
    cp = subprocess.run(
        ["ssh", host, f"tail -n 3 {remote_log} 2>/dev/null"],
        capture_output=True,
        text=True,
        timeout=25,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    for ln in reversed(cp.stdout.strip().splitlines()):
        try:
            rec = json.loads(ln)
            if isinstance(rec, dict) and ("lr" in rec or "valley_idx" in rec):
                return rec
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def remote_list_valleys(host: str, remote_run_dir: str) -> list[str]:
    cp = subprocess.run(
        ["ssh", host, f"ls -1 {remote_run_dir}/ckpt_valley_[0-9][0-9][0-9].pt 2>/dev/null || true"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if cp.returncode != 0:
        return []
    return [ln.strip() for ln in cp.stdout.splitlines() if ln.strip().endswith(".pt")]


def snapshot_slim_remote(host: str, python_bin: str, src_ckpt: str, dst_local: Path) -> str:
    dst_local.parent.mkdir(parents=True, exist_ok=True)
    remote_slim = f"{src_ckpt}.{dst_local.stem}.slim.pt"
    py_cmd = (
        "import torch; "
        f"ck = torch.load({src_ckpt!r}, map_location='cpu', weights_only=False); "
        "keep = {k: ck[k] for k in ('ema_cooldown','ema_valley','ema_swa','ema','model','state','size','sampler_cfg') if k in ck}; "
        f"torch.save(keep, {remote_slim!r})"
    )
    subprocess.run(["ssh", host, f"{python_bin} -c {json.dumps(py_cmd)}"], check=True, timeout=180)
    subprocess.run(["rsync", "-az", f"{host}:{remote_slim}", str(dst_local)], check=True, timeout=300)
    return sha256_file(dst_local)


def xbabe1_free_vram_mib() -> int:
    cp = subprocess.run(
        ["ssh", "xbabe1", "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return 0
    try:
        return int(cp.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return 0


def _count_npz(out_dir: Path) -> int:
    return sum(1 for p in out_dir.glob("*.npz") if p.is_file() and not p.name.startswith("."))


def extract_cv7_multi_91(
    tag: str,
    key_modes: tuple[str, ...],
    ckpts_local: list[Path],
) -> dict[str, Path]:
    out_dirs = {km: FLYWHEEL_ROOT / "cols_e8_d91" / f"cv7_hyb_{tag}_{km}" for km in key_modes}
    for od in out_dirs.values():
        od.mkdir(parents=True, exist_ok=True)
    if all(_count_npz(od) >= 91 for od in out_dirs.values()):
        return out_dirs

    free_mib = xbabe1_free_vram_mib()
    if free_mib >= 2400 and ckpts_local:
        r_dir = "/home/ubuntu/jope-data/h452-flywheel/ckpts_auto_watch"
        r_dirs_str = " ".join(str(od) for od in out_dirs.values())
        subprocess.run(["ssh", "xbabe1", f"mkdir -p {r_dir} {GP_OOF_CACHE} {r_dirs_str}"], check=True, timeout=25)
        for lp in ckpts_local:
            subprocess.run(["rsync", "-az", str(lp), f"xbabe1:{r_dir}/{lp.name}"], check=True, timeout=180)
        r_spec = ",".join(f"L:{r_dir}/{lp.name}" for lp in ckpts_local)
        multi_arg = ";".join(f"{km}|{km}|{r_spec}|{out_dirs[km]}" for km in key_modes)
        cmd_gpu = (
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
            f"timeout 1800s python3 -u /home/ubuntu/h452-stage/code/cols_104m_e8_gpresid_h452.py --cmd eval-cv7 "
            f"--bank {BANK_D91} --multi-specs {json.dumps(multi_arg)} "
            f"--prep-workers 8 --gp-oof-cache-dir {GP_OOF_CACHE} "
            f"--gp-anchor-dir {GP_ANCHORS} --indices 0-90 --device cuda"
        )
        rc = subprocess.run(["ssh", "xbabe1", cmd_gpu], timeout=1900)
        if rc.returncode == 0:
            for od in out_dirs.values():
                subprocess.run(["rsync", "-az", f"xbabe1:{od}/", f"{od}/"], check=True, timeout=180)
            if all(_count_npz(od) >= 91 for od in out_dirs.values()):
                return out_dirs

    # Safe CPU fallback on xbabe2 across 7 shards (`CUDA_VISIBLE_DEVICES=""`, `taskset -c 0-47,80-119`, `nice -n 19`)
    l_spec = ",".join(f"L:{p}" for p in ckpts_local)
    multi_local = ";".join(f"{km}|{km}|{l_spec}|{out_dirs[km]}" for km in key_modes)
    ranges = [(0, 12), (13, 25), (26, 38), (39, 51), (52, 64), (65, 77), (78, 90)]
    procs = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    for i0, i1 in ranges:
        cmd = [
            "taskset",
            "-c",
            "0-47,80-119",
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
            multi_local,
            "--prep-workers",
            "2",
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
    for km, od in out_dirs.items():
        got = _count_npz(od)
        if got < 91:
            raise RuntimeError(f"incomplete eval-cv7 extraction in {od}: {got}/91")
    return out_dirs


def score_matrix(cv7_dir: Path, traj_post_dir: Path, out_flywheel_json: Path, out_evidence_json: Path) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        "taskset",
        "-c",
        "0-47,80-119",
        "nice",
        "-n",
        "19",
        sys.executable,
        "-u",
        str(STAGE_CODE / "score_h454_paired_flywheel_matrix.py"),
        "--cv7-post-dir",
        str(cv7_dir),
        "--traj-post-dir",
        str(traj_post_dir),
        "--out-json",
        str(out_flywheel_json),
        "--procs",
        "28",
        "--n-boot",
        "2000",
    ]
    subprocess.run(cmd, env=env, check=True, timeout=1800)
    out_evidence_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_flywheel_json, out_evidence_json)
    return json.loads(out_flywheel_json.read_text())


def run_milestone(
    tag: str,
    ckpts_local: list[Path],
    ckpt_shas: dict[str, str],
    traj_post_dir: Path,
    ev_subdir: str,
) -> dict:
    pin_paths(ckpts_local, f"watch_wave2_{tag}")
    key_modes = ("ema_soup50", "ema_cooldown")
    out_dirs = extract_cv7_multi_91(tag, key_modes, ckpts_local)
    results = {}
    for key_mode in key_modes:
        sub_tag = f"{tag}_{key_mode}"
        cv7_dir = out_dirs[key_mode]
        fw_json = FLYWHEEL_ROOT / f"{sub_tag.upper()}_MATRIX.json"
        ev_json = Path(f"/home/ubuntu/jope_plan/evidence/{ev_subdir}/{sub_tag.upper()}_MATRIX.json")
        doc = score_matrix(cv7_dir, traj_post_dir, fw_json, ev_json)
        top = doc["summary_table"][0] if doc.get("summary_table") else {}
        results[key_mode] = {
            "top_arm": top.get("arm"),
            "elo": top.get("elo"),
            "family_mean_u": top.get("family_mean_u"),
            "evidence_json": str(ev_json),
        }

    stamp = time.strftime("%FT%TZ", time.gmtime())
    append_agent_chat(
        f"### [{stamp}] AUTO-SCORED MILESTONE `{tag}` (`watch_and_score_wave2_h454.py`, `canonical_promotion_forbidden=true`)\n"
        f"- Checkpoints: `{json.dumps(ckpt_shas)}`\n"
        f"- Results (`dev91` 33-model board, `0` CONFIRM / `0` FMGAP B/C): `{json.dumps(results)}`\n"
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll-s", type=float, default=45.0)
    ap.add_argument("--state-json", default=str(FLYWHEEL_ROOT / "logs" / "watch_wave2_state.json"))
    ap.add_argument("--traj-post-dir", default=str(FLYWHEEL_ROOT / "cols_e8_d91" / "traj_e8_w2aplus"))
    ap.add_argument("--once", action="store_true", help="Run a single non-destructive poll cycle and exit")
    a = ap.parse_args()

    state_p = Path(a.state_json)
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_p.read_text()) if state_p.is_file() else {"completed": {}}

    while True:
        try:
            traj_dir = Path(a.traj_post_dir)
            if not traj_dir.is_dir():
                traj_dir = FLYWHEEL_ROOT / "cols_e8_d91" / "traj_e8_w2a"
            if not traj_dir.is_dir():
                traj_dir = FLYWHEEL_ROOT / "cols_e8_d91" / "traj_e8_v1"

            rec_x1 = remote_tail_json("xbabe1", "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1/log.jsonl")
            rec_x4 = remote_tail_json("xbabe4", "/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/log.jsonl")
            if rec_x1 and "lr" in rec_x1:
                lr_x1 = float(rec_x1["lr"])
                for m_name, thr in (("h454_w2b_lr5e6", 5.0e-6), ("h454_w2c_floor3e6", 3.25e-6)):
                    if lr_x1 <= thr and m_name not in state["completed"]:
                        ck_dir = FLYWHEEL_ROOT / f"ckpts_{m_name}"
                        p_x1 = ck_dir / f"fleet_L_x1_{m_name}.pt"
                        p_x4 = ck_dir / f"fleet_L_x4_{m_name}.pt"
                        s1 = snapshot_slim_remote(
                            "xbabe1",
                            "python3",
                            "/home/ubuntu/jope-data/h450-fleet-agy/runs/fleet_L_x1/ckpt_last.pt",
                            p_x1,
                        )
                        cks = [p_x1]
                        shas = {str(p_x1): s1, "lr_x1": str(lr_x1)}
                        if rec_x4:
                            s4 = snapshot_slim_remote(
                                "xbabe4",
                                "/Users/agent/apex-venv/bin/python3",
                                "/Users/agent/jope-data/h450-fleet-agy/runs/fleet_L_x4/ckpt_last.pt",
                                p_x4,
                            )
                            cks.append(p_x4)
                            shas[str(p_x4)] = s4
                        res = run_milestone(m_name, cks, shas, traj_dir, "H454")
                        state["completed"][m_name] = {"at": time.strftime("%FT%TZ", time.gmtime()), "results": res}
                        state_p.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

            valleys = (
                remote_list_valleys("xbabe1", "/home/ubuntu/jope-data/h452-flywheel/runs/fleet_L_x1_h455_maxent_v2")
                + remote_list_valleys("xbabe1", "/home/ubuntu/jope-data/h452-flywheel/runs/fleet_L_x1_h455_maxent")
            )
            for r_vp in valleys:
                v_stem = Path(r_vp).stem  # ckpt_valley_001
                m_name = f"h455_{v_stem}"
                if m_name not in state["completed"]:
                    ck_dir = FLYWHEEL_ROOT / f"ckpts_{m_name}"
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    p_v = ck_dir / f"fleet_L_x1_h455_{v_stem}.pt"
                    subprocess.run(["rsync", "-az", f"xbabe1:{r_vp}", str(p_v)], check=True, timeout=300)
                    sv = sha256_file(p_v)
                    cks = [p_v]
                    shas = {str(p_v): sv}
                    w2a_x4 = FLYWHEEL_ROOT / "ckpts_wave2a_h454" / "fleet_L_x4_w2a.pt"
                    if w2a_x4.is_file():
                        cks.append(w2a_x4)
                        shas[str(w2a_x4)] = sha256_file(w2a_x4)
                    res = run_milestone(m_name, cks, shas, traj_dir, "H455")
                    state["completed"][m_name] = {"at": time.strftime("%FT%TZ", time.gmtime()), "results": res}
                    state_p.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

            print(
                json.dumps(
                    {
                        "ts": time.strftime("%FT%TZ", time.gmtime()),
                        "lr_x1": rec_x1.get("lr") if rec_x1 else None,
                        "step_x1": rec_x1.get("step") if rec_x1 else None,
                        "lr_x4": rec_x4.get("lr") if rec_x4 else None,
                        "h455_valleys": len(valleys),
                        "completed": sorted(state["completed"].keys()),
                    }
                ),
                flush=True,
            )
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            print(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "watch_err": repr(exc)}), flush=True)

        if a.once:
            break
        time.sleep(max(10.0, float(a.poll_s)))


if __name__ == "__main__":
    main()
