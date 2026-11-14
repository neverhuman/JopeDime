#!/usr/bin/env python3
"""H458 trainer: evaluation-matched fine-tuning of an ICL checkpoint on dev91 family pools.

One run fine-tunes one checkpoint on one or more family pools (`family_pool_h458`). Every micro-batch is a fresh
stratified downsample and a fresh context/query split of a pool, built by the evaluation's own feature pipeline
(`episodes_h458`). Progress is counted in query exposures per unique pool row; the learning-rate schedule, the
milestone checkpoints and the stop rule are all functions of that count.

Differences from the H455 trainer that matter operationally:
  * batch size follows the token caps down to one episode (no `max(4, B)` floor);
  * an out-of-memory error is logged at once; the item is trimmed and retried, the caps shrink once per item built
    under them, a fully failed item is followed by a growing wait, and the run aborts after 8 such items in a row;
  * checkpoints are saved on exposure milestones, never on wall-clock time, and never without an optimizer step;
  * the ledger counts repeated splits for real (family + context row set + query row set);
  * data workers may run on other hosts (`--remote-workers host:n`): the same `worker_loop`, the same per-worker
    item sequence, streamed back over ssh, so the CPU cost of episode building can leave the GPU host.

In-corpus by construction (`canonical_promotion_forbidden = true`). Reads no evaluation bank and no sealed cohort.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import queue  # noqa: E402
import shlex  # noqa: E402
import signal  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import paths_h458 as P  # noqa: E402

P.setup_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import episodes_h458 as E  # noqa: E402
from family_pool_h458 import FamilyPool  # noqa: E402

C = E.C
ICLH450MPS = E.ICLH450MPS
EPISODES_PER_STEP = len(E.MODES)
STALE_SKIP = "built under older caps, no longer fits"
DEFAULTS = {
    "arm": "rand",
    "size": "M",
    "exposures_target": 2000,
    "save_at_exposures": [250, 500, 1000, 1500, 2000],
    "lr_peak": 5e-5,
    "lr_floor": 5e-6,
    "warmup_steps": 300,
    "hold_frac": 0.6,
    "weight_decay": 0.0,
    "clip": 1.0,
    "ema": 0.999,
    "caps": {"tokens": 240000, "row_tokens": 12000},
    "q_max": 1024,
    "checkpoint_row_blocks": True,
    "fault_inject_oom_rows": 0,  # selftest only: a batch with more rows than this fails like a CUDA out-of-memory error
}


def lr_at(step: int, frac: float, cfg: dict) -> float:
    """Linear warmup over steps, hold at the peak until `hold_frac` of the exposure budget, cosine to the floor."""
    warm = min(1.0, (step + 1) / float(max(1, cfg["warmup_steps"])))
    hold = float(cfg["hold_frac"])
    if frac <= hold:
        base = float(cfg["lr_peak"])
    else:
        u = min(1.0, (frac - hold) / max(1e-9, 1.0 - hold))
        base = cfg["lr_floor"] + (cfg["lr_peak"] - cfg["lr_floor"]) * 0.5 * (1.0 + math.cos(math.pi * u))
    return warm * base


def item_rng(nonce: bytes, worker: int, seq: int) -> np.random.Generator:
    d = hashlib.blake2b(nonce + worker.to_bytes(4, "little") + seq.to_bytes(8, "little"), digest_size=32).digest()
    return np.random.default_rng(np.frombuffer(d, dtype=np.uint64))


def pool_tables(pool: FamilyPool, cfg: dict) -> dict[int, np.ndarray]:
    """The training table of every configured family.

    `exclude_stratum: k` drops one feature-hash stratum (cross-fitted "skill" specialists); `exclude_cids` drops
    every row of the named contexts (leave-context-out).
    """
    excl = list(cfg.get("exclude_cids") or [])
    out = {}
    for f in (pool.resolve(x) for x in cfg["families"]):
        mine = [c for c in excl if c in pool.meta["member_cids"][f]]
        if cfg.get("exclude_stratum") is not None:
            out[f] = pool.table_without_stratum(f, int(cfg["exclude_stratum"]))
        else:
            out[f] = pool.table_excluding(f, P.MMAP_V1, mine) if mine else pool.table(f)
    return out


def ctx_range_at(frac: float, cfg: dict):
    """Optional context-size phases, e.g. `[{"until": 0.3, "ctx": [48, 100]}]` for a tiny-context-first arm."""
    for ph in cfg.get("ctx_phases") or []:
        if frac < float(ph["until"]):
            return (int(ph["ctx"][0]), int(ph["ctx"][1]))
    return None


def worker_loop(w, cfg, pool_dir, pool_sha, nonce, start_seq, cap_scale, progress_v, out_q, stop_ev) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.nice(5)
    except OSError:
        pass
    torch.set_num_threads(1)
    pool = FamilyPool(Path(pool_dir), expect_sha256=pool_sha)
    tables = pool_tables(pool, cfg)
    fams = list(tables)
    wts = np.asarray(cfg.get("family_weights") or [1.0] * len(fams), dtype=np.float64)
    wts = wts / wts.sum()
    seq = int(start_seq)
    while not stop_ev.is_set():
        seq += 1
        rng = item_rng(nonce, w, seq)
        fam = int(fams[int(rng.choice(len(fams), p=wts))])
        scale = float(cap_scale.value)
        caps = {k: int(v * scale) for k, v in cfg["caps"].items()}
        try:
            item = E.build_episode(
                tables[fam], pool.member_rows(fam), fam, rng, caps, cfg["arm"], cfg["q_max"],
                ctx_range=ctx_range_at(float(progress_v.value), cfg),
                ctx_cap_frac=float(cfg.get("ctx_cap_frac", 1.0)),
            )
            if item is None:
                item = {"skip": "does not fit caps or degenerate target"}
        except Exception as exc:  # a bad episode must not kill the worker; the main loop counts and reports it
            item = {"skip": f"{type(exc).__name__}: {exc}"[:240]}
        item["seq"] = seq
        item["scale"] = scale
        while not stop_ev.is_set():
            try:
                out_q.put(item, timeout=1.0)
                break
            except queue.Full:
                continue


FRAME = struct.Struct("<IQ")  # worker index, payload bytes


def serve_main() -> None:
    """Remote side of `--remote-workers`: data workers on this host, items written to stdout as framed pickles.

    Items leave in round-robin order over the workers, which is the order the trainer consumes them in. Control lines
    ("cap_scale progress") arrive on stdin; end of input means the trainer is gone and everything stops.
    """
    ap = argparse.ArgumentParser(description=serve_main.__doc__)
    ap.add_argument("--serve-workers", type=int, required=True)
    ap.add_argument("--serve-base", type=int, required=True, help="index of the first worker served here")
    ap.add_argument("--serve-spec-b64", required=True)
    ap.add_argument("--pool", default=str(P.POOL_DIR))
    a = ap.parse_args()
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # nothing a library prints may reach the item stream
    spec = json.loads(base64.b64decode(a.serve_spec_b64))
    cfg = spec["cfg"]
    if cfg.get("exclude_cids"):
        raise SystemExit("remote workers do not support exclude_cids (the source bank stays on the training host)")
    pool = FamilyPool(Path(a.pool), expect_sha256=spec["pool_sha256"], verify_data=True)
    pool_tables(pool, cfg)
    if P.verify_deps(strict=True) != spec["deps"]:
        raise SystemExit("dependency bytes differ from the training host")
    ctx = mp.get_context("fork")
    halt = ctx.Event()
    cap_scale = ctx.Value("d", float(spec["cap_scale"]))
    progress_v = ctx.Value("d", float(spec["progress"]))
    nonce = bytes.fromhex(spec["nonce"])
    n = int(a.serve_workers)
    lanes = [ctx.Queue(maxsize=3) for _ in range(n)]
    kids = [
        ctx.Process(
            target=worker_loop,
            args=(a.serve_base + i, cfg, str(pool.dir), pool.sha256, nonce, int(spec["start_seqs"][i]), cap_scale, progress_v, lanes[i], halt),
            daemon=True,
        )
        for i in range(n)
    ]
    for kid in kids:
        kid.start()

    def control() -> None:
        for line in sys.stdin:
            parts = line.split()
            if len(parts) == 2:
                cap_scale.value, progress_v.value = float(parts[0]), float(parts[1])
        halt.set()

    threading.Thread(target=control, daemon=True).start()
    i = 0
    try:
        while not halt.is_set():
            try:
                item = lanes[i % n].get(timeout=2.0)
            except queue.Empty:
                if not kids[i % n].is_alive():
                    raise SystemExit(f"remote worker {a.serve_base + i % n} died") from None
                continue
            blob = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            out.write(FRAME.pack(a.serve_base + i % n, len(blob)))
            out.write(blob)
            out.flush()
            i += 1
    except BrokenPipeError:
        pass
    finally:
        halt.set()
        for lane in lanes:
            lane.cancel_join_thread()
        for kid in kids:
            kid.terminate()


class RemoteWorkers:
    """Data workers on another host. Their items arrive over ssh and land in one queue per worker."""

    def __init__(self, host: str, n: int, base: int, code: str, pool_dir: str, spec: dict, err_path: Path):
        self.host, self.n, self.base = host, int(n), int(base)
        blob = base64.b64encode(json.dumps(spec, sort_keys=True).encode()).decode()
        remote = (
            f"cd {shlex.quote(code)} && exec env JOPE_REF_CODE= PYTHONDONTWRITEBYTECODE=1 nice -n 19 /usr/bin/python3 -u -B train_h458.py"
            f" --serve-workers {self.n} --serve-base {self.base} --pool {shlex.quote(pool_dir)} --serve-spec-b64 {blob}"
        )
        self.err = open(err_path, "ab")
        # host "local" runs the same command on this host without ssh (selftest of the stream protocol)
        argv = ["bash", "-c", remote] if host == "local" else ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4", host, remote]
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.err,
        )
        self.queues = [queue.Queue(maxsize=3) for _ in range(self.n)]
        self.dead = ""
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        f = self.proc.stdout
        try:
            while True:
                head = f.read(FRAME.size)
                if len(head) < FRAME.size:
                    break
                w, size = FRAME.unpack(head)
                blob = f.read(size)
                if len(blob) < size or not (self.base <= w < self.base + self.n):
                    break
                self.queues[w - self.base].put(pickle.loads(blob))
        except (OSError, EOFError, ValueError, pickle.UnpicklingError, struct.error) as exc:
            self.dead = f"{type(exc).__name__}: {exc}"[:200]
        finally:  # whatever ended the stream, the main loop sees it through `is_alive` and aborts the run
            self.dead = self.dead or "item stream ended"

    def is_alive(self) -> bool:
        return not self.dead and self.thread.is_alive() and self.proc.poll() is None

    def update(self, scale: float, progress: float) -> None:
        try:
            self.proc.stdin.write(f"{scale:.6f} {progress:.6f}\n".encode())
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        for step in (self.proc.stdin.close, self.proc.terminate, self.err.close):
            try:
                step()
            except (OSError, ValueError):
                pass


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", required=True, help="checkpoint to fine-tune, or 'random' (selftest only)")
    ap.add_argument("--init-key", default="auto")
    ap.add_argument("--init-sha256", default="")
    ap.add_argument("--pool", default=str(P.POOL_DIR))
    ap.add_argument("--pool-sha256", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cuda-vram-frac", type=float, default=0.13)
    ap.add_argument("--workers", type=int, default=8, help="data workers on this host")
    ap.add_argument("--remote-workers", default="", help="host:n[,host:n] data workers on other hosts, streamed over ssh")
    ap.add_argument("--remote-code", default=str(P.HERE), help="path of this staged code on the remote hosts")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--log-every-s", type=float, default=30.0)
    ap.add_argument("--no-step-abort-s", type=float, default=900.0)
    ap.add_argument("--last-every-s", type=float, default=600.0, help="time-based ckpt_last, besides the milestone saves")
    a = ap.parse_args()

    cfg = {**DEFAULTS, **json.loads(Path(a.config).read_text())}
    cfg["caps"] = {**DEFAULTS["caps"], **cfg.get("caps", {})}
    if cfg["arm"] not in ("rand", "canon"):
        raise SystemExit("arm must be rand or canon")
    out_dir = P.guard_write_path(Path(a.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    last_p = out_dir / "ckpt_last.pt"
    if last_p.exists() and not a.resume:
        raise SystemExit(f"{last_p} exists: pass --resume or use a new --out")
    if a.resume and not last_p.exists():
        raise SystemExit("--resume needs ckpt_last.pt")
    pool = FamilyPool(Path(a.pool), expect_sha256=a.pool_sha256)
    rows = {f: int(t.shape[0]) for f, t in pool_tables(pool, cfg).items()}
    fams = list(rows)
    deps = P.verify_deps(strict=True)
    remote_spec = [(h, int(n)) for h, n in (part.split(":") for part in a.remote_workers.split(",") if part)]
    n_workers = int(a.workers) + sum(n for _h, n in remote_spec)
    if n_workers < 1:
        raise SystemExit("no data workers")

    if a.resume:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        state = ck["state"]
        if state["config_sha256"] != hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest():
            raise SystemExit("--resume with a different config")
    else:
        ck = None
        state = {
            "step": 0,
            "items": 0,
            "accum_eps": 0,
            "exposures": {str(f): 0 for f in fams},
            "saved": [],
            "seq": [0] * n_workers,
            "nonce": os.urandom(32).hex(),
            "cap_scale": 1.0,
            "oom": 0,
            "oom_streak": 0,
            "skips": 0,
            "skip_reasons": {},
            "nonfinite_grads": 0,
            "ctx_hist": {},
            "dup_splits": 0,
            "train_s": 0.0,
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "pool_sha256": pool.sha256,
        }
    for _k, _v in (("skip_reasons", {}), ("nonfinite_grads", 0), ("ctx_hist", {})):
        state.setdefault(_k, _v)
    if len(state["seq"]) != n_workers:
        raise SystemExit("the total number of data workers must match the run being resumed")
    ledger: set[bytes] = set(ck["ledger"]) if ck else set()

    # workers first, CUDA second (a forked child must not inherit an initialised CUDA context)
    ctx = mp.get_context("fork")
    stop_ev = ctx.Event()
    cap_scale = ctx.Value("d", float(state["cap_scale"]))
    progress_v = ctx.Value("d", 0.0)
    nonce = bytes.fromhex(state["nonce"])
    qs = [ctx.Queue(maxsize=3) for _ in range(a.workers)]
    procs = [
        ctx.Process(
            target=worker_loop,
            args=(w, cfg, str(pool.dir), pool.sha256, nonce, state["seq"][w], cap_scale, progress_v, qs[w], stop_ev),
            daemon=True,
        )
        for w in range(a.workers)
    ]
    for p in procs:
        p.start()
    remotes: list[RemoteWorkers] = []
    alive = [p.is_alive for p in procs]
    base = int(a.workers)
    for host, n in remote_spec:
        spec = {
            "cfg": cfg,
            "pool_sha256": pool.sha256,
            "deps": deps,
            "nonce": state["nonce"],
            "start_seqs": state["seq"][base : base + n],
            "cap_scale": float(state["cap_scale"]),
            "progress": 0.0,
        }
        r = RemoteWorkers(host, n, base, a.remote_code, a.pool, spec, P.guard_write_path(out_dir / f"remote_{host}.stderr"))
        remotes.append(r)
        qs.extend(r.queues)
        alive.extend([r.is_alive] * n)
        base += n

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(float(a.cuda_vram_frac))
    torch.manual_seed(int.from_bytes(nonce[:4], "little"))
    model = ICLH450MPS(cfg["size"], checkpoint_row_blocks=bool(cfg["checkpoint_row_blocks"])).to(dev)
    ema = ICLH450MPS(cfg["size"]).to(dev).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    init_sha = ""
    if ck is not None:
        model.load_state_dict(ck["model"], strict=True)
        ema.load_state_dict(ck["ema"], strict=True)
    elif a.init != "random":
        init_sha = P.digest_file(Path(a.init))
        if a.init_sha256 and init_sha != a.init_sha256:
            raise SystemExit("init checkpoint sha256 mismatch")
        src = torch.load(a.init, map_location="cpu", weights_only=False)
        if a.init_key not in ("", "auto", "ema_soup50") and src.get(a.init_key) is None:
            raise SystemExit(f"init checkpoint has no key {a.init_key!r}")
        sd = C._extract_state_dict(src, ckpt_key=a.init_key)
        model.load_state_dict(sd, strict=True)
        ema.load_state_dict(sd, strict=True)
        del src, sd
    else:
        ema.load_state_dict(model.state_dict(), strict=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr_peak"], betas=(0.9, 0.98), eps=1e-8, weight_decay=cfg["weight_decay"]
    )
    if ck is not None:
        opt.load_state_dict(ck["opt"])
    del ck
    model.train()

    stop = {"flag": False}

    def _on_sig(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    def per_row() -> dict[int, float]:
        return {f: state["exposures"][str(f)] / float(rows[f]) for f in fams}

    def progress() -> float:
        tgt = float(cfg["exposures_target"])
        return float(np.mean([min(1.0, v / tgt) for v in per_row().values()]))

    def cpu_sd(m) -> dict:
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}

    def save_last() -> None:
        state["cap_scale"] = float(cap_scale.value)
        atomic_save(
            {"model": cpu_sd(model), "ema": cpu_sd(ema), "opt": opt.state_dict(), "state": state, "ledger": list(ledger), "cfg": cfg},
            last_p,
        )

    def save_milestone(tag: int) -> None:
        sd = cpu_sd(ema)
        plain = {k: v for k, v in state.items() if k != "nonce"}
        atomic_save({"size": cfg["size"], "ema": sd, "model": sd, "state": plain, "cfg": cfg, **P.DISCLOSURE}, out_dir / f"spec_e{tag:05d}.pt")

    def abort(reason: str) -> None:
        save_last()
        (out_dir / "ABORTED.json").write_text(json.dumps({"reason": reason, "step": state["step"], "items": state["items"]}) + "\n")
        stop_ev.set()
        raise SystemExit(3)

    log_f = open(out_dir / "log.jsonl", "a", buffering=1)

    def log(rec: dict) -> None:
        log_f.write(json.dumps({"utc": time.strftime("%H:%M:%S", time.gmtime()), **rec}) + "\n")

    log(
        {
            "event": "start",
            "resume": bool(a.resume),
            "workers": {"local": int(a.workers), "remote": a.remote_workers},
            "device": str(dev),
            "params": sum(p_.numel() for p_ in model.parameters()),
            "families": {pool.short(f): rows[f] for f in fams},
            "init": a.init,
            "init_sha256": init_sha,
            "init_key": a.init_key,
            "pool_sha256": pool.sha256,
            "cfg": cfg,
            "deps": deps,
            **P.DISCLOSURE,
        }
    )

    t_run = t_log = t_step = t_last = time.time()
    t_ctl = 0.0
    pend: list[tuple[bytes, str, int]] = []  # (split commitment, family, exposures) of the open accumulation group
    fault_rows = int(cfg.get("fault_inject_oom_rows", 0))
    saved_step = int(state["step"])
    s0 = float(state["train_s"])
    run = {"loss": {m: [0.0, 0] for m in E.MODES}, "eps": 0, "qrows": 0, "wait": 0.0, "items": 0}
    opt.zero_grad(set_to_none=True)
    k = int(state["items"])
    try:
        while not stop["flag"]:
            frac = progress()
            progress_v.value = frac
            if frac >= 1.0 or (a.max_items and state["items"] >= a.max_items):
                break
            w = k % n_workers
            t_w = time.time()
            if remotes and t_w - t_ctl > 2.0:
                t_ctl = t_w
                for r in remotes:
                    r.update(float(cap_scale.value), frac)
            item = None
            while not stop["flag"]:
                try:
                    item = qs[w].get(timeout=2.0)
                    break
                except queue.Empty:
                    if not alive[w]():
                        abort(f"worker {w} died" + "".join(f" ({r.host}: {r.dead})" for r in remotes if r.dead))
                    if time.time() - t_w > 600.0:
                        abort(f"worker {w} produced nothing for 600 s")
            if item is None:
                break
            run["wait"] += time.time() - t_w
            k += 1
            state["items"] = k
            state["seq"][w] = int(item["seq"])
            run["items"] += 1
            if "skip" in item:
                state["skips"] += 1
                why = str(item["skip"])[:60]
                state["skip_reasons"][why] = state["skip_reasons"].get(why, 0) + 1
                if state["skips"] in (1, 10, 100) or state["skips"] % 1000 == 0:
                    log({"event": "skip", "n": state["skips"], "why": item["skip"]})
            elif item["commit"] in ledger or any(item["commit"] == c for c, _f, _e in pend):
                state["dup_splits"] += 1
            else:
                b, n_q, n_ctx = int(item["zm"].shape[0]), int(item["n_q"]), int(item["n_ctx"])
                trained = oom_here = False
                keep = np.arange(b)  # the modes of this item still in the batch
                tries = 8
                if float(item.get("scale", 1.0)) > float(cap_scale.value) + 1e-9:
                    # built under caps that have been lowered since: size it to the caps in force now
                    caps_now = {key: int(v * float(cap_scale.value)) for key, v in cfg["caps"].items()}
                    b_fit, q_fit = E.fit_caps(n_ctx, n_q, int(item["n_feat"]), caps_now, n_modes=b)
                    if b_fit == 0:
                        tries = 0
                        state["skips"] += 1
                        state["skip_reasons"][STALE_SKIP] = state["skip_reasons"].get(STALE_SKIP, 0) + 1
                    else:
                        if b_fit < b:
                            keep, b = keep[[k % b]], 1
                        n_q = min(n_q, int(q_fit))
                # A queued item was sized under the caps in force when it was built. If it does not fit, trim it here
                # (query rows are independent, modes are independent) and retry, instead of failing on stale shapes.
                for attempt in range(tries):
                    in_backward = False
                    try:
                        if fault_rows and b * (n_ctx + n_q) > fault_rows:
                            raise RuntimeError("CUDA out of memory (injected by fault_inject_oom_rows)")
                        zm = torch.from_numpy(item["zm"][keep, : n_ctx + n_q]).to(dev)
                        yz = torch.from_numpy(item["yz"][keep]).to(dev)
                        ym = torch.from_numpy(item["ym"][keep]).to(dev)
                        ys = torch.from_numpy(item["ys"][keep]).to(dev)
                        yq = torch.from_numpy(item["yq"][keep, :n_q]).to(dev)
                        if item["slots"] is None:
                            slots = torch.rand(b, E.FID_SLOTS, device=dev).argsort(dim=-1)[:, None, : zm.shape[2]]
                        else:
                            slots = torch.from_numpy(item["slots"][keep]).to(dev)
                        logits, ym_o, ys_o = model(None, None, n_ctx, slots=slots, amp=(dev.type == "cuda"), prenorm=(zm, yz, ym, ys))
                        per_ep = model.loss(logits, yq, ym_o, ys_o).mean(dim=1)
                        loss = per_ep.mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite loss")
                        in_backward = True
                        (loss * (b / float(EPISODES_PER_STEP))).backward()
                        trained = True
                        break
                    except (RuntimeError, FloatingPointError) as exc:
                        is_oom = "out of memory" in str(exc).lower()
                        if not is_oom and not isinstance(exc, FloatingPointError):
                            raise
                        if in_backward:
                            # a failed backward may have left partial gradients: drop the whole open group. A failure
                            # before it has touched no gradient, so the episodes already accumulated stay.
                            opt.zero_grad(set_to_none=True)
                            state["accum_eps"] = 0
                            pend.clear()
                        if not is_oom:
                            state["skips"] += 1
                            log({"event": "nonfinite", "step": state["step"]})
                            break
                        state["oom"] += 1
                        oom_here = True
                        del exc
                        logits = per_ep = loss = zm = yq = None
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        if float(item.get("scale", 1.0)) <= float(cap_scale.value) + 1e-9:
                            # only an item built under the caps now in force shrinks them; a stale, larger item is
                            # trimmed here and says nothing about the current caps
                            cap_scale.value = float(cap_scale.value) * 0.85
                        log({"event": "oom", "n": state["oom"], "attempt": attempt, "cap_scale": round(cap_scale.value, 4), "shape": [b, n_ctx + n_q, int(item["n_feat"])]})
                        # context rows dominate the memory, so after one halving of the queries drop to one mode
                        if b > 1 and n_q <= 512:
                            keep, b = keep[[state["oom"] % b]], 1
                        elif n_q > 64:
                            n_q = max(64, n_q // 2)
                        else:
                            break
                if not trained and oom_here:
                    state["oom_streak"] += 1
                    if state["oom_streak"] >= 8 or cap_scale.value < 0.2:
                        abort("out of memory on 8 items in a row after trimming, or caps below 20%")
                    # a shared card can be full for a moment (another process's burst): wait before the next item,
                    # about 150 s in all before the abort above
                    time.sleep(0.0 if fault_rows else min(30.0, 2.0 ** state["oom_streak"]))
                if trained:
                    pend.append((item["commit"], str(item["fam"]), b * n_q))
                    state["oom_streak"] = 0
                    state["accum_eps"] += b
                    hb = "<256" if n_ctx < 256 else "256-383" if n_ctx < 384 else "384-767" if n_ctx < 768 else "768-999" if n_ctx < 1000 else "1000+"
                    state["ctx_hist"][hb] = state["ctx_hist"].get(hb, 0) + 1
                    run["eps"] += b
                    run["qrows"] += b * n_q
                    for m_name, v in zip([item["modes"][int(j)] for j in keep], per_ep.detach().float().cpu().tolist()):
                        run["loss"][m_name][0] += v
                        run["loss"][m_name][1] += 1
                if trained and state["accum_eps"] >= EPISODES_PER_STEP:
                    cur_lr = lr_at(state["step"], frac, cfg)
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
                    if not bool(torch.isfinite(grad_norm)):
                        # a finite loss can still carry non-finite bf16 gradients; stepping would poison the weights,
                        # the EMA and every later checkpoint
                        opt.zero_grad(set_to_none=True)
                        state["accum_eps"] = 0
                        pend.clear()
                        state["nonfinite_grads"] += 1
                        log({"event": "nonfinite_grad", "n": state["nonfinite_grads"], "step": state["step"]})
                        if state["nonfinite_grads"] >= 50:
                            abort("non-finite gradients 50 times")
                    else:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        state["step"] += 1
                        state["accum_eps"] = 0
                        for c_, f_, e_ in pend:
                            ledger.add(c_)
                            state["exposures"][f_] += e_
                        pend.clear()
                        t_step = time.time()
                        d = min(float(cfg["ema"]), (1.0 + state["step"]) / (10.0 + state["step"]))
                        with torch.no_grad():
                            for pe, pm in zip(ema.parameters(), model.parameters()):
                                pe.lerp_(pm, 1.0 - d)
                        done = min(per_row().values())
                        for tag in cfg["save_at_exposures"]:
                            if done >= tag and tag not in state["saved"]:
                                state["saved"].append(tag)
                                save_milestone(int(tag))
                                save_last()
                                log({"event": "milestone", "exposures_per_row": tag, "step": state["step"]})
            now = time.time()
            state["train_s"] = s0 + (now - t_run)
            if now - t_step > a.no_step_abort_s:
                abort(f"no optimizer step for {int(now - t_step)} s")
            if now - t_last >= a.last_every_s and state["step"] > saved_step:
                save_last()
                saved_step, t_last = int(state["step"]), now
            if now - t_log >= a.log_every_s:
                dt = max(1e-3, now - t_log)
                log(
                    {
                        "step": state["step"],
                        "items": state["items"],
                        "exposures_per_row": {pool.short(f): round(v, 1) for f, v in per_row().items()},
                        "progress": round(frac, 4),
                        "loss": {m: round(s / n, 5) for m, (s, n) in run["loss"].items() if n},
                        "lr": lr_at(state["step"], frac, cfg),
                        "eps_per_s": round(run["eps"] / dt, 2),
                        "qrows_per_s": round(run["qrows"] / dt, 1),
                        "queue_wait_frac": round(run["wait"] / dt, 3),
                        "oom": state["oom"],
                        "cap_scale": round(float(cap_scale.value), 4),
                        "skips": state["skips"],
                        "skip_reasons": state["skip_reasons"],
                        "nonfinite_grads": state["nonfinite_grads"],
                        "ctx_hist": state["ctx_hist"],
                        "dup_splits": state["dup_splits"],
                        "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0.0,
                    }
                )
                run = {"loss": {m: [0.0, 0] for m in E.MODES}, "eps": 0, "qrows": 0, "wait": 0.0, "items": 0}
                t_log = now
    finally:
        stop_ev.set()
        for r in remotes:
            r.close()
        for q_ in qs[: a.workers]:
            q_.cancel_join_thread()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=3.0)
    state["train_s"] = s0 + (time.time() - t_run)
    if state["step"] > 0:
        save_last()
    done = progress() >= 1.0
    log({"event": "final" if done else "paused", "step": state["step"], "items": state["items"], "exposures_per_row": {pool.short(f): round(v, 1) for f, v in per_row().items()}, "dup_splits": state["dup_splits"], "oom": state["oom"], "train_s": round(state["train_s"], 1)})
    log_f.close()
    (out_dir / ("FINAL.json" if done else "PAUSED.json")).write_text(json.dumps({"step": state["step"], "saved": state["saved"], **P.DISCLOSURE}) + "\n")


if __name__ == "__main__":
    if any(arg.startswith("--serve-workers") for arg in sys.argv[1:]):
        serve_main()
    else:
        main()
    sys.exit(0)
