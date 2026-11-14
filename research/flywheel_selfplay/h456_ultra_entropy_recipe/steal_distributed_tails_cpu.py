#!/usr/bin/env python3
"""Non-Destructive Reverse-Depth Tail-Stealing MKL CPU Pool (`steal_distributed_tails_cpu.py`).

Designed to run on `xbabe2` inside:
  CUDA_VISIBLE_DEVICES="" systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=14G \
    choom -n 500 -- taskset -c 32-63,80-127 nice -n 19 /home/ubuntu/.venv/bin/python3 -u ...

Key properties:
  - Prioritizes tail positions of `xbabe0_pool.json` buckets (`depth max-1 .. 0`) so `xbabe0`
    never wastes in-flight `b[0]` work and straggler multi-context buckets are eliminated first.
  - Persistent worker processes load slim checkpoints once (`~0.75 GiB` RSS/worker for `M`,
    `~1.5 GiB` RSS/worker for `L` -> `--workers 2` for `L` stays `~3.2 GiB`, well below `MemoryHigh=10G`).
  - Passes a per-fold `abort_check` into `_eval_prepped_context_gpu` so workers abort redundant
    in-flight folds within seconds once `{cid}.npz` lands or all output directories hit `91/91`.
  - Claims 1 context at a time via POSIX-atomic `os.mkdir(claims_dir / ...)` (zero deletions) and
    immediately pushes each finished `.npz` to `xbabe0:` via `rsync -az --ignore-existing`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BANK_D91 = Path("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
GP_OOF_CACHE_DIR = Path("/home/ubuntu/jope-data/h452-flywheel/gp_anchors_oof_cache_v1")
GP_ANCHOR_DIR = Path("/home/ubuntu/jope-data/h452-flywheel/gp_anchors_p0")
P418_DIR = Path("/home/ubuntu/jope_plan/evidence/H418/pulled")
P448_DIR = Path("/home/ubuntu/jope_plan/evidence/H448/pulled")


def steal_count_npz(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(1 for p in out_dir.glob("*.npz") if p.is_file() and not p.name.startswith("."))


def steal_mem_available_gib() -> float:
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return round(float(ln.split()[1]) / (1024.0 * 1024.0), 2)
    except OSError:
        pass
    return -1.0


def steal_self_cgroup_mem_gib() -> float:
    try:
        for ln in Path("/proc/self/cgroup").read_text().splitlines():
            parts = ln.split(":", 2)
            if len(parts) == 3 and parts[2].strip():
                cg_cur = Path("/sys/fs/cgroup") / parts[2].strip().lstrip("/") / "memory.current"
                if cg_cur.is_file():
                    return round(float(cg_cur.read_text().strip()) / (1024.0 ** 3), 2)
    except OSError:
        pass
    return -1.0


def steal_parse_out_dirs(multi_specs: str) -> list[Path]:
    out: list[Path] = []
    for entry in multi_specs.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("|")
        if len(parts) != 4:
            raise ValueError(f"bad multi_specs entry: {entry}")
        out.append(Path(parts[3].strip()))
    return out


def build_reverse_depth_priority(shards_csv: list[str], total_contexts: int = 91) -> list[int]:
    buckets: list[list[int]] = []
    for s in shards_csv:
        items = [int(x.strip()) for x in s.split(",") if x.strip()]
        if items:
            buckets.append(items)
    buckets.sort(key=lambda b: -len(b))
    max_depth = max((len(b) for b in buckets), default=0)
    priority: list[int] = []
    seen: set[int] = set()
    for depth in range(max_depth - 1, -1, -1):
        for b in buckets:
            if len(b) > depth:
                idx = b[depth]
                if idx not in seen:
                    seen.add(idx)
                    priority.append(idx)
    for idx in range(total_contexts):
        if idx not in seen:
            seen.add(idx)
            priority.append(idx)
    return priority


def run_tail_steal_worker(a: argparse.Namespace) -> None:
    sys.path.extend(
        [
            "/home/ubuntu/.local/lib/python3.12/site-packages",
            "/usr/lib/python3/dist-packages",
            "/home/ubuntu/h452-stage/code",
        ]
    )
    import numpy as np
    import torch
    import cols_104m_e8_gpresid_h452 as C

    torch.set_num_threads(int(a.threads))
    gc.enable()

    cfg = json.loads(Path(a.stage_cfg).read_text())
    multi_specs = str(cfg["multi_specs"])
    priority_indices = [int(x) for x in json.loads((Path(a.claims_dir) / "priority_order.json").read_text())]
    claims_dir = Path(a.claims_dir)
    dev = torch.device("cpu")

    specs = C._parse_multi_specs(multi_specs, "", "auto", "", dev)
    out_dirs = [sp["out_dir"] for sp in specs]

    with np.load(BANK_D91, allow_pickle=False) as zb:
        assert "ya" not in zb.files, "audit labels forbidden in search bank!"
        cids = [str(x) for x in zb["cids"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        ctx_data: dict[int, tuple] = {}
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            if not all((od / f"{c}.npz").is_file() for od in out_dirs):
                S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
                Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
                ctx_data[i] = (
                    c,
                    np.array(S_mat[:, :-1], np.float64, copy=True),
                    Xa,
                    np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                    np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
                )
            so += n * w
            ao += na * f
            lo += n
    gc.collect()

    print(
        json.dumps(
            {
                "event": "worker_ready",
                "worker_id": a.worker_id,
                "pid": os.getpid(),
                "threads": torch.get_num_threads(),
                "loaded_pending_ctxs": len(ctx_data),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    completed_by_worker = 0
    for idx in priority_indices:
        if all(steal_count_npz(od) >= 91 for od in out_dirs):
            break
        cid = cids[idx]
        if all((od / f"{cid}.npz").is_file() for od in out_dirs):
            ctx_data.pop(idx, None)
            continue

        claim_path = claims_dir / f"ctx_{idx:03d}_{cid}"
        try:
            os.mkdir(claim_path)
        except FileExistsError:
            ctx_data.pop(idx, None)
            continue

        entry = ctx_data.pop(idx, None)
        if entry is None:
            continue
        c, Xs, Xa, y, lab = entry
        if all((od / f"{c}.npz").is_file() for od in out_dirs):
            continue

        t_ctx = time.time()
        work_item = (
            idx,
            c,
            Xs,
            Xa,
            y,
            lab,
            GP_ANCHOR_DIR,
            P418_DIR,
            P448_DIR,
            GP_OOF_CACHE_DIR,
            4,
            2,
            7,
        )
        prepped = C._prep_full_context_worker(work_item)
        specs_written: list[str] = []
        aborted_early = False
        for sp in specs:
            dst = sp["out_dir"] / f"{c}.npz"
            if dst.is_file():
                continue

            def _should_abort(dst_p: Path = dst) -> bool:
                return dst_p.is_file() or all(steal_count_npz(od) >= 91 for od in out_dirs)

            arrays = C._eval_prepped_context_gpu(prepped, sp["models_w"], dev, abort_check=_should_abort)
            if arrays is None:
                aborted_early = True
                continue
            tmp = sp["out_dir"] / f".{c}.steal.{os.getpid()}.{sp['tag']}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, dst)
            specs_written.append(sp["tag"])
            if a.remote_host:
                subprocess.run(
                    ["rsync", "-az", "--ignore-existing", str(dst), f"{a.remote_host}:{dst}"],
                    check=False,
                    timeout=60,
                )

        del prepped, Xs, Xa, y, lab, entry
        gc.collect()
        if specs_written:
            completed_by_worker += 1
        rec = {
            "event": "worker_ctx_done",
            "worker_id": a.worker_id,
            "idx": idx,
            "cid": c,
            "specs_written": specs_written,
            "aborted_early": aborted_early,
            "ctx_elapsed_s": round(time.time() - t_ctx, 1),
            "worker_completed_total": completed_by_worker,
        }
        (claim_path / "done.json").write_text(json.dumps(rec, sort_keys=True) + "\n")
        print(json.dumps(rec, sort_keys=True), flush=True)
        if all(steal_count_npz(od) >= 91 for od in out_dirs):
            break

    print(
        json.dumps(
            {
                "event": "worker_exit",
                "worker_id": a.worker_id,
                "worker_completed_total": completed_by_worker,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_tail_steal_coordinator(a: argparse.Namespace) -> None:
    stage_cfg_p = Path(a.stage_cfg)
    cfg = json.loads(stage_cfg_p.read_text())
    multi_specs = str(cfg["multi_specs"])
    out_dirs = steal_parse_out_dirs(multi_specs)
    for od in out_dirs:
        od.mkdir(parents=True, exist_ok=True)

    claims_dir = Path(a.claims_dir) if a.claims_dir else (stage_cfg_p.parent / "tail_claims_v1")
    claims_dir.mkdir(parents=True, exist_ok=True)

    priority = build_reverse_depth_priority(cfg.get("shards", []), total_contexts=91)
    (claims_dir / "priority_order.json").write_text(json.dumps(priority) + "\n")

    print(
        json.dumps(
            {
                "event": "tail_steal_start",
                "stage_cfg": str(stage_cfg_p),
                "claims_dir": str(claims_dir),
                "workers": int(a.workers),
                "threads_per_worker": int(a.threads),
                "first_15_priority_indices": priority[:15],
                "initial_counts": {od.name: steal_count_npz(od) for od in out_dirs},
                "xbabe2_mem_avail_gib": steal_mem_available_gib(),
                "cgroup_mem_gib": steal_self_cgroup_mem_gib(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["MKL_NUM_THREADS"] = str(a.threads)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    procs: list[tuple[int, subprocess.Popen]] = []
    for wid in range(int(a.workers)):
        cmd: list[str] = []
        if a.affinity.strip():
            cmd.extend(["taskset", "-c", a.affinity.strip()])
        cmd.extend(
            [
                "nice",
                "-n",
                "19",
                str(a.python_bin),
                "-u",
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--worker-id",
                str(wid),
                "--stage-cfg",
                str(stage_cfg_p),
                "--claims-dir",
                str(claims_dir),
                "--threads",
                str(a.threads),
                "--remote-host",
                str(a.remote_host),
            ]
        )
        procs.append((wid, subprocess.Popen(cmd, env=env)))
        time.sleep(0.6)

    t0 = time.time()
    all_91_since: float | None = None
    terminated_cleanly_at_91 = False
    while True:
        time.sleep(15.0)
        alive = [wid for wid, p in procs if p.poll() is None]
        counts = {od.name: steal_count_npz(od) for od in out_dirs}
        claimed_dirs = sum(1 for p in claims_dir.iterdir() if p.is_dir() and p.name.startswith("ctx_"))
        done_claims = sum(1 for p in claims_dir.glob("ctx_*/done.json") if p.is_file())
        print(
            json.dumps(
                {
                    "event": "tail_steal_progress",
                    "alive_workers": alive,
                    "claimed": claimed_dirs,
                    "stealer_done": done_claims,
                    "counts": counts,
                    "xbabe2_mem_avail_gib": steal_mem_available_gib(),
                    "cgroup_mem_gib": steal_self_cgroup_mem_gib(),
                    "elapsed_s": round(time.time() - t0, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not alive:
            break
        if all(c >= 91 for c in counts.values()):
            if all_91_since is None:
                all_91_since = time.time()
            elif time.time() - all_91_since >= 25.0:
                terminated_cleanly_at_91 = True
                for _, p in procs:
                    if p.poll() is None:
                        p.terminate()
                for _, p in procs:
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                break

    bad = [
        (wid, p.returncode)
        for wid, p in procs
        if p.returncode not in (0, None) and not (terminated_cleanly_at_91 and p.returncode in (-15, -9, 143, 137))
    ]
    print(
        json.dumps(
            {
                "event": "tail_steal_workers_finished",
                "bad_workers": bad,
                "terminated_cleanly_at_91": terminated_cleanly_at_91,
                "final_counts": {od.name: steal_count_npz(od) for od in out_dirs},
                "xbabe2_mem_avail_gib": steal_mem_available_gib(),
                "cgroup_mem_gib": steal_self_cgroup_mem_gib(),
                "elapsed_s": round(time.time() - t0, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if bad:
        raise SystemExit(f"tail-stealing worker failure(s): {bad}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("coordinator", "worker"), default="coordinator")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--stage-cfg", required=True, help="Path to stage xbabe0_pool.json")
    ap.add_argument("--claims-dir", default="", help="Directory for POSIX-atomic mkdir claims")
    ap.add_argument("--python-bin", default="/home/ubuntu/.venv/bin/python3")
    ap.add_argument("--affinity", default="32-63,80-127")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--remote-host", default="xbabe0")
    a = ap.parse_args()

    if a.mode == "worker":
        run_tail_steal_worker(a)
    else:
        run_tail_steal_coordinator(a)


if __name__ == "__main__":
    main()
