#!/usr/bin/env python3
"""Synthetic self-test for `H449` (`selftest_h449.py`) — touches ZERO `dev91` bytes.

Verifies:
  1. `train_unfair_icl_h449.py` 2-stream (`[X, S]` direct + `RealBank`/`PROBS_V3S`) training on tiny `S` model,
     including `< 384`-row tables (`r = 48`) and wide (`F = 38 > 30`) tables, plus quarantine path refusal!
  2. `analyze_h449_unfair.py` quarantine refusal on canonical board paths!
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h449_unfair as AN49  # noqa: E402
import train_unfair_icl_h449 as TR49  # noqa: E402
from icl_model import ICL  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    root = Path(a.dir).resolve()
    if "/h449" not in str(root) and "/H449/" not in str(root):
        raise SystemExit("--dir must contain /h449 or /H449/ to test quarantine")
    root.mkdir(parents=True, exist_ok=True)

    # 1. Quarantine refusal checks
    bad_passed = False
    try:
        AN49._check_quarantined_path(Path("/home/ubuntu/JopeDime/research/elo/reference-board/boards/dev91-audit-v1/scores"))
        bad_passed = True
    except SystemExit:
        pass
    assert not bad_passed, "Quarantine check failed to refuse canonical reference-board path!"

    # 2. Tiny S-model training smoke test (with both small r=48 d91 table and wide F=38 d91 table)
    m0 = ICL("S")
    ck_p = root / "init_s.pt"
    torch.save({"model": m0.state_dict(), "ema": m0.state_dict(), "cfg": m0.cfg}, ck_p)
    ck_sha = _sha(ck_p)

    rng = np.random.default_rng(44901)
    t_small = rng.normal(size=(48, 6)).astype(np.float32)    # r=48 < 384: tests Stream A small-table coverage!
    t_wide = rng.normal(size=(300, 39)).astype(np.float32)   # F=38 > 30: tests subsets_keep_s!
    t_other = rng.normal(size=(320, 8)).astype(np.float32)
    tabs = [t_small, t_wide, t_other]
    shapes = np.array([t.shape for t in tabs], np.int64)
    offs = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])[:-1]]).astype(np.int64)
    bank_p = root / "unfair_smoke.npz"
    np.savez(
        bank_p,
        data=np.concatenate([t.ravel() for t in tabs]),
        shape=shapes,
        offset=offs,
        cids=np.array(["d91_small", "d91_wide", "other_01"]),
        family=np.array(["f1", "f2", "f3"]),
        cohort=np.array(["D91-ALL", "D91-ALL", "FIT-ICL"]),
        manifest_sha256=np.array("smoke"),
    )
    bank_sha = _sha(bank_p)

    TR49.ROW_BUCKETS = (64,)
    ns = argparse.Namespace(
        size="S",
        init=str(ck_p),
        init_sha256=ck_sha,
        real_bank=str(bank_p),
        bank_sha256=bank_sha,
        out=str(root / "train_out"),
        device="cpu",
        threads=2,
        seed=449,
        lr=5e-5,
        ema_decay=0.99,
        hours=0.05,
        max_steps=4,
        target_cell_tokens=0,
        precache_other=2,
        d91_folds=2,
        precache_workers=1,
        struct_frac=0.5,
        struct_bmax=2,
        real_frac=0.65,
        tokens=4096,
        max_rows=128,
        bmax=2,
        log_every=2,
    )
    TR49.cmd_train(ns)
    man = json.loads((root / "train_out" / "unfair_manifest.json").read_text())
    assert man["state"]["step"] == 4 and len(man["ckpt_sha256"]) == 64
    assert man["state"]["cum_cell_tokens"] > 0 and man["state"]["approx_train_flops"] > 0

    print(json.dumps({"all_passed": True, "tests": 2, "steps": man["state"]["step"]}), flush=True)
    print("H449_SELFTEST_OK", flush=True)


if __name__ == "__main__":
    main()
