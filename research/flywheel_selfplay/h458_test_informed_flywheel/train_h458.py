#!/usr/bin/env python3
"""H458 trainer: evaluation-matched fine-tuning of an ICL checkpoint on dev91 family pools.

One run fine-tunes one checkpoint on one or more family pools (`family_pool_h458`). Every micro-batch is a fresh
stratified downsample and a fresh context/query split of a pool, built by the evaluation's own feature pipeline
(`episodes_h458`). Progress is counted in query exposures per unique pool row; the learning-rate schedule, the
milestone checkpoints and the stop rule are all functions of that count.

Differences from the H455 trainer that matter operationally:
  * batch size follows the token caps down to one episode (no `max(4, B)` floor);
  * an out-of-memory error is logged at once, shrinks the caps, and aborts the run after 8 in a row;
  * checkpoints are saved on exposure milestones, never on wall-clock time, and never without an optimizer step;
  * the ledger counts repeated splits for real (family + context row set + query row set).

In-corpus by construction (`canonical_promotion_forbidden = true`). Reads no evaluation bank and no sealed cohort.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
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
        while not stop_ev.is_set():
            try:
                out_q.put(item, timeout=1.0)
                break
            except queue.Full:
                continue


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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--log-every-s", type=float, default=30.0)
    ap.add_argument("--no-step-abort-s", type=float, default=900.0)
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
            "seq": [0] * a.workers,
            "nonce": os.urandom(32).hex(),
            "cap_scale": 1.0,
            "oom": 0,
            "oom_streak": 0,
            "skips": 0,
            "dup_splits": 0,
            "train_s": 0.0,
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "pool_sha256": pool.sha256,
        }
    if len(state["seq"]) != a.workers:
        raise SystemExit("--workers must match the run being resumed")
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

    t_run = t_log = t_step = time.time()
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
            w = k % a.workers
            t_w = time.time()
            item = None
            while not stop["flag"]:
                try:
                    item = qs[w].get(timeout=2.0)
                    break
                except queue.Empty:
                    if not procs[w].is_alive():
                        abort(f"worker {w} died")
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
                if state["skips"] in (1, 10, 100) or state["skips"] % 1000 == 0:
                    log({"event": "skip", "n": state["skips"], "why": item["skip"]})
            elif item["commit"] in ledger:
                state["dup_splits"] += 1
            else:
                b, n_q, n_ctx = int(item["zm"].shape[0]), int(item["n_q"]), int(item["n_ctx"])
                trained = False
                try:
                    zm = torch.from_numpy(item["zm"]).to(dev)
                    yz = torch.from_numpy(item["yz"]).to(dev)
                    ym = torch.from_numpy(item["ym"]).to(dev)
                    ys = torch.from_numpy(item["ys"]).to(dev)
                    yq = torch.from_numpy(item["yq"]).to(dev)
                    if item["slots"] is None:
                        slots = torch.rand(b, E.FID_SLOTS, device=dev).argsort(dim=-1)[:, None, : zm.shape[2]]
                    else:
                        slots = torch.from_numpy(item["slots"]).to(dev)
                    logits, ym_o, ys_o = model(None, None, n_ctx, slots=slots, amp=(dev.type == "cuda"), prenorm=(zm, yz, ym, ys))
                    per_ep = model.loss(logits, yq, ym_o, ys_o).mean(dim=1)
                    loss = per_ep.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    (loss * (b / float(EPISODES_PER_STEP))).backward()
                    trained = True
                except (RuntimeError, FloatingPointError) as exc:
                    is_oom = "out of memory" in str(exc).lower()
                    if not is_oom and not isinstance(exc, FloatingPointError):
                        raise
                    opt.zero_grad(set_to_none=True)
                    state["accum_eps"] = 0
                    if is_oom:
                        state["oom"] += 1
                        state["oom_streak"] += 1
                        cap_scale.value = float(cap_scale.value) * 0.85
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        log({"event": "oom", "n": state["oom"], "streak": state["oom_streak"], "cap_scale": round(cap_scale.value, 4), "shape": [b, n_ctx + n_q, int(item["n_feat"])]})
                        if state["oom_streak"] >= 8 or cap_scale.value < 0.35:
                            abort("out of memory 8 times in a row or caps below 35%")
                    else:
                        state["skips"] += 1
                        log({"event": "nonfinite", "step": state["step"]})
                if trained:
                    ledger.add(item["commit"])
                    state["oom_streak"] = 0
                    state["accum_eps"] += b
                    state["exposures"][str(item["fam"])] += b * n_q
                    run["eps"] += b
                    run["qrows"] += b * n_q
                    for m_name, v in zip(item["modes"], per_ep.detach().float().cpu().tolist()):
                        run["loss"][m_name][0] += v
                        run["loss"][m_name][1] += 1
                if trained and state["accum_eps"] >= EPISODES_PER_STEP:
                    cur_lr = lr_at(state["step"], frac, cfg)
                    for g in opt.param_groups:
                        g["lr"] = cur_lr
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    state["step"] += 1
                    state["accum_eps"] = 0
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
                        "dup_splits": state["dup_splits"],
                        "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else 0.0,
                    }
                )
                run = {"loss": {m: [0.0, 0] for m in E.MODES}, "eps": 0, "qrows": 0, "wait": 0.0, "items": 0}
                t_log = now
    finally:
        stop_ev.set()
        for q_ in qs:
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
    main()
    sys.exit(0)
