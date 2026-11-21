#!/usr/bin/env python3
"""CI-first, memory/PSI-guarded queue launcher for H459 Track S9 Arm A5 `P201B-80` (`xbabe3` & `xbabe1`).

Enforces:
  1. Instant `SIGSTOP` / `SIGCONT` yield whenever a GitHub Actions job (`Runner.Worker`) is active (when `--pause-on-ci 1`),
     or `MemAvailable < --min-mem-gib`, or memory PSI `full avg10 > 10.0%` (`PREREG_S9.md` §6 + `cl_mtc` 01:59Z).
  2. Periodic `rsync` of completed `{cid}.npz`, `{cid}.meta.json`, `SETTINGS.json`, `SIDECAR.json`, and `RECEIPT_*.json`
     back to `xbabe2` so `cl_mtc` can score blocks directly on `xbabe2` and remote disk stays `< 30 MB`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

DRIVER = "/home/ubuntu/h452-stage/code/boosted_residual_gp_h459.py"
BANK = "/home/ubuntu/jope-data/h430/bank_P201B.npz"
HYB_ANCHOR_DIR = "/home/ubuntu/jope-data/h335-cl/h447/pub-r3/cols_r4e8_base_P201B"
DEFAULT_OUT_ROOT = Path("/home/ubuntu/jope-data/h459-s9/r2")
RECEIPT_ROOT = Path("/home/ubuntu/jope_plan/evidence/H459/runs_r2")


def _mem_avail_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return 64.0


def _psi_full_avg10(kind: str) -> float:
    p = Path(f"/proc/pressure/{kind}")
    if not p.is_file():
        return 0.0
    for line in p.read_text().splitlines():
        if line.startswith("full "):
            m = re.search(r"avg10=([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return 0.0


def _ci_active() -> bool:
    rc = subprocess.run(
        ["pgrep", "-f", "Runner.Worker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    return rc == 0


def _sync_to_xbabe2(out_root: Path) -> None:
    for src, dst in [
        (f"{out_root}/", f"xbabe2:{out_root}/"),
        (f"{RECEIPT_ROOT}/", f"xbabe2:{RECEIPT_ROOT}/"),
    ]:
        subprocess.run(
            ["rsync", "-az", "--exclude=slots/", src, dst],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
            check=False,
        )


def build_jobs(
    out_root: Path,
    workers: int,
    inner_slots: int,
    stride_mod: str,
    slots_root: Path,
    receipt_suffix: str,
) -> List[Tuple[str, List[str]]]:
    base_common = [
        "/usr/bin/python3",
        "-u",
        DRIVER,
        "--bank",
        BANK,
        "--cid-sha-top-n",
        "80",
        "--arm-mode",
        "paired",
        "--dom-mult",
        "2.0",
        "--prior",
        "dev91-free",
        "--enforce-wall-cap",
        "0",
        "--workers",
        str(workers),
        "--inner-slots",
        str(inner_slots),
        "--rayon-threads",
        "1",
        "--slots-root",
        str(slots_root),
    ]
    if stride_mod:
        base_common.extend(["--stride-mod", stride_mod])

    specs = [
        (
            "p201b80_medium_prod_pure",
            ["--effort", "medium", "--budget-mode", "product", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
        ),
        (
            "p201b80_medium_prod_hyb",
            [
                "--effort", "medium", "--budget-mode", "product", "--budget-unit", "candidates",
                "--anchor-dir", HYB_ANCHOR_DIR, "--anchor-col", "r4_e8", "--closure", "fair",
            ],
        ),
        (
            "p201b80_high_prod_pure",
            ["--effort", "high", "--budget-mode", "product", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
        ),
        (
            "p201b80_high_prod_hyb",
            [
                "--effort", "high", "--budget-mode", "product", "--budget-unit", "candidates",
                "--anchor-dir", HYB_ANCHOR_DIR, "--anchor-col", "r4_e8", "--closure", "fair",
            ],
        ),
        (
            "p201b80_medium_cycle_pure",
            ["--effort", "medium", "--budget-mode", "cycle", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
        ),
        (
            "p201b80_high_cycle_pure",
            ["--effort", "high", "--budget-mode", "cycle", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
        ),
    ]
    out: List[Tuple[str, List[str]]] = []
    sfx = f"_{receipt_suffix}" if receipt_suffix else ""
    for tag, extra in specs:
        cmd = (
            base_common
            + extra
            + [
                "--colblock-dir",
                str(out_root / tag),
                "--out-receipt",
                str(RECEIPT_ROOT / f"RECEIPT_{tag}{sfx}.json"),
            ]
        )
        out.append((tag, cmd))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--inner-slots", type=int, default=3)
    ap.add_argument("--stride-mod", default="")
    ap.add_argument("--slots-root", default=str(DEFAULT_OUT_ROOT / "slots" / "worker_pool"))
    ap.add_argument("--receipt-suffix", default="")
    ap.add_argument("--pause-on-ci", type=int, default=1)
    ap.add_argument("--min-mem-gib", type=float, default=16.0)
    a = ap.parse_args()

    if socket.gethostname().strip() == "xbabe2":
        raise RuntimeError(
            "Refusing to run run_h459_p201b_xbabe3_queue.py on xbabe2: ssh alias 'xbabe2' on xbabe2 points to .89 (M4)."
        )

    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ssh", "xbabe2", f"mkdir -p {out_root} {RECEIPT_ROOT}"], check=False, timeout=15)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"

    active_proc: Optional[subprocess.Popen] = None
    paused = False
    stop_mon = threading.Event()

    def _monitor() -> None:
        nonlocal paused
        last_sync = time.monotonic()
        while not stop_mon.wait(15.0):
            p = active_proc
            if p is not None and p.poll() is None:
                ci_on = bool(a.pause_on_ci) and _ci_active()
                mem_gb = _mem_avail_gib()
                psi_mem = _psi_full_avg10("memory")
                should_pause = ci_on or (mem_gb < float(a.min_mem_gib)) or (psi_mem > 10.0)
                if should_pause and not paused:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGSTOP)
                        paused = True
                        print(
                            json.dumps(
                                {"event": "paused_for_ci_or_pressure", "ci": ci_on, "mem_gib": round(mem_gb, 2), "psi_mem": psi_mem}
                            ),
                            flush=True,
                        )
                    except ProcessLookupError:
                        pass
                elif not should_pause and paused:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGCONT)
                        paused = False
                        print(json.dumps({"event": "resumed_after_ci_or_pressure", "mem_gib": round(mem_gb, 2)}), flush=True)
                    except ProcessLookupError:
                        pass
            if time.monotonic() - last_sync >= 45.0:
                _sync_to_xbabe2(out_root)
                last_sync = time.monotonic()

    mon_t = threading.Thread(target=_monitor, daemon=True)
    mon_t.start()

    try:
        jobs = build_jobs(
            out_root=out_root,
            workers=int(a.workers),
            inner_slots=int(a.inner_slots),
            stride_mod=a.stride_mod,
            slots_root=Path(a.slots_root),
            receipt_suffix=a.receipt_suffix,
        )
        for tag, cmd in jobs:
            t0 = time.monotonic()
            print(
                json.dumps(
                    {
                        "event": "job_start",
                        "tag": tag,
                        "stride_mod": a.stride_mod,
                        "workers": a.workers,
                        "utc": time.strftime("%FT%TZ", time.gmtime()),
                    }
                ),
                flush=True,
            )
            active_proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            rc = active_proc.wait()
            paused = False
            active_proc = None
            _sync_to_xbabe2(out_root)
            print(
                json.dumps(
                    {
                        "event": "job_done",
                        "tag": tag,
                        "stride_mod": a.stride_mod,
                        "rc": rc,
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "utc": time.strftime("%FT%TZ", time.gmtime()),
                    }
                ),
                flush=True,
            )
            if rc != 0:
                return rc
    finally:
        stop_mon.set()
        _sync_to_xbabe2(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
