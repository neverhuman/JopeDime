#!/usr/bin/env python3
"""H450 pub-r14 parity check (cl_fwriter): with the pub-r14 flags off, the episode workers must emit exactly the items
of the pub-r12 trainer under the same pub-r12 flags. Both workers run with one nonce and one worker id on one loaded
bank; items are compared byte for byte (X, y, n_ctx, fingerprints and all 8 info fields).

  parity_r14_h450.py --old /home/ubuntu/h450-stage/pub-r13/code/train_aggressive_random_h450.py --bank BANK \
      --bank-sha256 HEX --feature-drop-every-batch 1 --synth-inj-every-batch 1 --items 400 --out PARITY.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path

import train_aggressive_random_h450 as T

R12_KEYS = ("feature_drop_every", "feature_drop_max_frac", "synth_inj_every")


def _collect(mod, nonce, bank, a, q, ev) -> None:
    mod._episode_worker_loop(0, nonce, bank, a.tokens, a.row_tokens, a.bmax, q, ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--bank-sha256", required=True)
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=480_000)
    ap.add_argument("--row-tokens", type=int, default=8_192)
    ap.add_argument("--bmax", type=int, default=16)
    ap.add_argument("--feature-drop-every-batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--feature-drop-max-frac", type=float, default=0.35)
    ap.add_argument("--synth-inj-every-batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    spec = importlib.util.spec_from_file_location("train_old_r12", a.old)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    if not hasattr(old, "AUG") or set(old.AUG) != set(R12_KEYS):
        raise SystemExit("--old is not a pub-r12 trainer")
    r12 = {"feature_drop_every": bool(a.feature_drop_every_batch), "feature_drop_max_frac": float(a.feature_drop_max_frac),
           "synth_inj_every": bool(a.synth_inj_every_batch)}
    old.AUG.update(r12)
    T.AUG.update(r12)            # the pub-r14 keys keep their defaults (off, inj_max_frac 0.10)
    if any(T.AUG[k] for k in ("rows_random", "target_random", "feat_affine_noise", "uniform_by_dataset")) \
            or T.AUG["inj_max_frac"] != T.INJ_MAX_FRAC:
        raise SystemExit("pub-r14 flags must be off for the parity check")
    bank = T.CombinedF64Bank(Path(a.bank), expect_sha256=a.bank_sha256)
    nonce = hashlib.blake2b(b"h450-pub-r14-parity", digest_size=32).digest()
    ctx = mp.get_context("fork")
    qs, evs, procs = [], [], []
    for mod in (old, T):
        q, ev = ctx.Queue(maxsize=8), ctx.Event()
        pr = ctx.Process(target=_collect, args=(mod, nonce, bank, a, q, ev), daemon=True)
        pr.start()
        qs.append(q); evs.append(ev); procs.append(pr)
    t0 = time.time()
    n_eps, n_batched, mism = 0, 0, []
    for i in range(a.items):
        xo, yo, co, io, fo = qs[0].get(timeout=300)
        xn, yn, cn, inn, fn = qs[1].get(timeout=300)
        same = (xo.shape == xn.shape and xo.tobytes() == xn.tobytes() and yo.tobytes() == yn.tobytes() and co == cn
                and fo == fn and [tuple(t) for t in io] == [tuple(t) for t in inn])
        if not same:
            mism.append(i)
        n_eps += len(io)
        n_batched += int(len(io) > 1)
    for ev in evs:
        ev.set()
    for pr in procs:
        pr.terminate()
    doc = {"schema": "jope.h450.parity_r14/v1", "old": a.old, "old_sha256": hashlib.sha256(Path(a.old).read_bytes()).hexdigest(),
           "new_sha256": hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest(), "bank_sha256": bank.sha256,
           "aug_r12": r12, "items": a.items, "episodes": n_eps, "batched_items": n_batched, "mismatched_items": len(mism),
           "first_mismatches": mism[:10], "wall_s": round(time.time() - t0, 2), "PASS": not mism}
    with open(out, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(json.dumps(doc, sort_keys=True))
    raise SystemExit(0 if not mism else 1)


if __name__ == "__main__":
    main()
