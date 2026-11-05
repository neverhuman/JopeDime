#!/usr/bin/env python3
"""H381 E2 (PREREG §3): a deeper screener grown from the stored fold-clean S0 (H317 H0.1-A) models.

S3 = scale S with 3 layers (d 32, 4 heads, cand_hidden 228, ffn x4, LayerNorm/GELU), registered at runtime. Growth
loads an S0 fold model into S3 (strict=False; only ctx_blocks.{1,2}.* / cross_blocks.{1,2}.* may be missing) and zeroes
the weight AND bias of att.o and ffn.w2 in every new block, so each new block is an exact identity at init; the grown
model's predictions are checked equal to S0's (max |delta| == 0) before any step. The stored normalizer is kept.

Arms (opt1 = AdamW + cosine, 3,000 steps, A525 subset, 2 seeds x 5 folds, S0's (seed, fold) run seeds):
  S0cont   S0 weights, 3,000 further steps, every parameter at 8e-5
  S3sc     fresh S3, standard opt1 (normalizer fit as in H317)
  S3fz     grown S3, S0 parameters locked (requires_grad False), new blocks at 8e-4
  S3ft     grown S3, new blocks at 8e-4, S0 parameters at 8e-5
  S3ftB16  S3ft on the B16 view subset (existing synthetic row-crop / column / distractor views)

Runs through the pinned rapid_hypothesis_lab harness (run_arm: fold split, true targets, predictions, receipts) with
these runtime patches only: curriculum_trainer.train_one (wrapper: S0 lookup + fold-cleanliness + frozen check),
multiscale_axial_bridge.build_model, curriculum_trainer.fit_normalizer (no-op for warm starts) and
curriculum_trainer.build_optimizers. Patches are installed per arm and restored after it.

  oof     --arms A,B --folds 0,1,2 --s0-dir D --out-root R --threads 2 --confirm-true-targets   (GPU host)
  decide  --s0-dir D --arms-root R --out F                                                       (one host, all folds)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

H317 = Path(os.environ.get("H381_H317_CODE",
                           "/home/ubuntu/jope-data/h317-cl/code/h317_island10_multiscale_curriculum"))
PINS = {  # = h335_core.H317_PINS for the neural import closure (H317 closeout 97a1ffce)
    "common_h317.py": "1242bae25379dc1adb7cbaf253fb7066dcf783b579258f98302b4a3c31c1c3d4",
    "crop_shuffle_synthetic_views.py": "eb7dee6bbfc1924b96baf2dc24d60b7fb704e1ec22c29e93f3795c754449bedb",
    "rapid_hypothesis_lab.py": "31161becd922ecd25784f1f08b16b445c81f8f0a1d3940f06f41e2fd6a805e25",
    "multiscale_axial_bridge.py": "f99726e42cfec84875d15cd5ae202e2674b8638315c2ddb5afb583900d13b248",
    "curriculum_trainer.py": "5453c8e25424bdc64ab55e84bcafbc570fc60aa22e72a059de6523d7585ae9c8",
    "neural_data.py": "a453dc090db7643a10331cd4c3a0986533cfc48826a65bc02d61e6a08e480f75",
    "trainonly_loader.py": "eef47f4f9ce4248edbfd90f0a1b4e2946e99013dda6e5c5c779cef94843fc3c1",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_bad = sorted(n for n, s in PINS.items() if _sha(H317 / n) != s)
if _bad:
    raise RuntimeError(f"H381: pinned H317 module changed: {_bad}")
sys.path.insert(0, str(H317))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common_h317 as C  # noqa: E402
import curriculum_trainer as T  # noqa: E402
import multiscale_axial_bridge as MB  # noqa: E402
import rapid_hypothesis_lab as RHL  # noqa: E402

MB.SCALES.setdefault("S3", MB.ScaleConfig("S3", d=32, n_heads=4, n_layers=3, cand_hidden=228, ffn_mult=4.0))
NEW_PREFIX = ("ctx_blocks.1.", "ctx_blocks.2.", "cross_blocks.1.", "cross_blocks.2.")
ZERO_SUFFIX = ("att.o.weight", "att.o.bias", "ffn.w2.weight", "ffn.w2.bias")
LR_NEW, LR_OLD = 8e-4, 8e-5
BUDGET = RHL.Budget(steps=3000)  # = the H0.1 budget (slate 128, top_k 48, batch 16, 2 seeds)
ARMS = {
    "S0cont": (RHL.ArmConfig("H381-S0cont", "S", "opt1", "A4", "A525"), "cont"),
    "S3sc": (RHL.ArmConfig("H381-S3sc", "S3", "opt1", "A4", "A525"), "scratch"),
    "S3fz": (RHL.ArmConfig("H381-S3fz", "S3", "opt1", "A4", "A525"), "frozen"),
    "S3ft": (RHL.ArmConfig("H381-S3ft", "S3", "opt1", "A4", "A525"), "ft"),
    "S3ftB16": (RHL.ArmConfig("H381-S3ftB16", "S3", "opt1", "B16", "A525"), "ft"),
}
RHL.PLANNED["H381-E2"] = {"baseline": "S0", "challengers": tuple(ARMS), "m": len(ARMS)}
RS = {C.stream_seed("H317-neural", s, j) % (2 ** 31): (s, j) for s in C.TRAIN_SEEDS[:2] for j in range(C.N_FOLDS)}

_ORIG = {"train_one": T.train_one, "build_model": MB.build_model, "fit_normalizer": T.fit_normalizer,
         "build_optimizers": T.build_optimizers}
STATE: dict = {"mode": None, "s0_dir": None, "meta": None, "tt": None, "device": None, "ck": None, "train_idx": None}


def _is_new(name: str) -> bool:
    return name.startswith(NEW_PREFIX)


def _s0_ckpt(seed_run: int) -> dict:
    s, j = RS[seed_run]
    d = Path(STATE["s0_dir"]) / f"seed{s}" / f"fold{j}"
    conf = json.loads((d / "config.json").read_text())
    if _sha(d / "model.pt") != conf["model_sha256"]:
        raise AssertionError(f"S0 seed{s}/fold{j}: model.pt sha != its config")
    ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
    if ck["scale"] != "S" or ck["seed"] != s or ck["fold"] != j:
        raise AssertionError("S0 checkpoint identity")
    ck["_sha256"] = conf["model_sha256"]
    return ck


def grown_build_model(scale: str):
    """Called by train_one right after torch.manual_seed(run seed): fresh model (seeded init), S0 weights loaded,
    new-block output projections zeroed, S0-equality checked on 4 held-in contexts."""
    ck, mode, dev = STATE["ck"], STATE["mode"], STATE["device"]
    m = _ORIG["build_model"](scale)
    sd = ck["state_dict"]
    res = m.load_state_dict(sd, strict=(scale == "S"))
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected or any(not _is_new(k) for k in missing):
        raise AssertionError(f"growth: unexpected {unexpected[:4]} / non-new missing {missing[:4]}")
    with torch.no_grad():
        for name, p in m.named_parameters():
            if _is_new(name) and name.endswith(ZERO_SUFFIX):
                p.zero_()
    m = m.to(dev)
    s0 = _ORIG["build_model"]("S")
    s0.load_state_dict(sd, strict=True)
    s0 = s0.to(dev)
    probe = np.asarray(STATE["train_idx"][:4], np.int64)
    a = T.predict_contexts(m, STATE["tt"], probe, dev)
    b = T.predict_contexts(s0, STATE["tt"], probe, dev)
    diff = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    if diff != 0.0:
        raise AssertionError(f"grown model != S0 at init (max |delta| {diff})")
    STATE["init_check"] = {"probe_contexts": len(probe), "max_abs_delta": diff, "missing_new_keys": len(missing)}
    if mode == "frozen":
        for name, p in m.named_parameters():
            p.requires_grad_(_is_new(name))
    del s0
    return m


def grown_optimizers(model, r, lr_mult: float = 1.0):
    mode = STATE["mode"]
    new = [p for n, p in model.named_parameters() if _is_new(n) and p.requires_grad]
    old = [p for n, p in model.named_parameters() if not _is_new(n) and p.requires_grad]
    if mode == "cont":
        groups = [{"params": old, "lr": LR_OLD * lr_mult}]
    elif mode == "frozen":
        groups = [{"params": new, "lr": LR_NEW * lr_mult}]
    elif mode == "ft":
        groups = [{"params": new, "lr": LR_NEW * lr_mult}, {"params": old, "lr": LR_OLD * lr_mult}]
    else:
        raise ValueError(mode)
    return [torch.optim.AdamW(groups, weight_decay=r.wd, betas=(0.9, 0.999))]


def grown_train_one(tt, train_idx, spec, seed, device, max_steps=None, log_every=0):
    mode = STATE["mode"]
    if mode == "scratch":
        return _ORIG["train_one"](tt, train_idx, spec, seed, device, max_steps, log_every)
    ck = _s0_ckpt(int(seed))
    meta = STATE["meta"]
    tr_cids = sorted({meta["cids"][i] for i in train_idx})
    if tr_cids != list(ck["train_cids"]):
        raise AssertionError("S0 training contexts != this run's training contexts (fold-cleanliness)")
    STATE.update(ck=ck, train_idx=np.asarray(train_idx), device=device)
    out, stats = _ORIG["train_one"](tt, train_idx, spec, seed, device, max_steps, log_every)
    extra = {"mode": mode, "s0_model_sha256": ck["_sha256"], "lr_new": LR_NEW, "lr_old": LR_OLD,
             "init_check": STATE.get("init_check")}
    if mode == "frozen":
        sd = out.state_dict()
        moved = [k for k, v in ck["state_dict"].items() if not torch.equal(sd[k].detach().cpu(), v)]
        if moved:
            raise AssertionError(f"frozen arm moved S0 tensors: {moved[:4]}")
        extra["frozen_s0_tensors_unchanged"] = len(ck["state_dict"])
    stats = {**stats, "h381": extra}
    STATE["ck"] = None
    return out, stats


def _install(mode: str) -> None:
    STATE["mode"] = mode
    T.train_one = grown_train_one
    if mode != "scratch":
        MB.build_model = grown_build_model
        T.fit_normalizer = lambda *a, **k: None
        T.build_optimizers = grown_optimizers


def _restore() -> None:
    T.train_one = _ORIG["train_one"]
    MB.build_model = _ORIG["build_model"]
    T.fit_normalizer = _ORIG["fit_normalizer"]
    T.build_optimizers = _ORIG["build_optimizers"]
    STATE["mode"] = None


def cmd_oof(a) -> None:
    res, dev, dope_before, tt, meta = RHL._true_target_setup(a)
    STATE.update(meta=meta, tt=tt, s0_dir=a.s0_dir)
    folds = [int(x) for x in a.folds.split(",")]
    t0 = time.time()
    done = []
    for name in a.arms.split(","):
        cfg, mode = ARMS[name]
        _install(mode)
        try:
            RHL.run_arm(cfg, BUDGET, tt, meta, dev, Path(a.out_root), folds=folds, keep_models=True)
        finally:
            _restore()
        done.append(name)
        print(json.dumps({"arm": name, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    after = RHL.dope_hashes()
    C.write_json(Path(a.out_root) / f"oof_receipt_h381_{os.uname().nodename}_{int(t0)}.json",
                 {"wave": "H381", "arms": done, "folds": folds, "budget": BUDGET.__dict__, "s0_dir": a.s0_dir,
                  "resources": res, "dope_before": dope_before, "dope_after": after,
                  "dope_unchanged": after == dope_before and after["composite_sha256"] == C.DOPE_SHA256,
                  "elapsed_s": time.time() - t0, "guard": C.guard_report(), "host": os.uname().nodename})


def cmd_decide(a) -> None:
    C.install_guard()
    fam_of = {r["context_sha256"]: r["family_id"] for r in C.train_rows()}
    dirs = {"S0": Path(a.s0_dir), **{n: Path(a.arms_root) / ARMS[n][0].arm_id for n in ARMS}}
    validation = RHL.validate_arm_dirs(dirs)
    per = {k: RHL.arm_context_utilities(v) for k, v in dirs.items()}
    rep = RHL.decide(per, "H381-E2", fam_of)
    rep["validation"] = validation
    base = C.family_equal_mean(per["S0"], fam_of)
    desc = {n: C.family_equal_mean(per[n], fam_of) - base for n in ARMS}
    if a.l_dir:  # stored H317 H0.2-L (descriptive, not rerun)
        lu = RHL.arm_context_utilities(Path(a.l_dir))
        desc["H0.2-L(stored)"] = C.family_equal_mean(lu, fam_of) - base
    rep["family_mean_gain_vs_S0"] = desc
    iso = {"layers_beyond_more_training(S3ft-S0cont)": ("S3ft", "S0cont"),
           "lock_vs_finetune(S3fz-S3ft)": ("S3fz", "S3ft"), "warm_start(S3ft-S3sc)": ("S3ft", "S3sc"),
           "variety_views(S3ftB16-S3ft)": ("S3ftB16", "S3ft")}
    rep["isolating"] = {k: C.robustness_summary(per[x], per[y], fam_of, C.BOOT_SEED_GATE_H0)
                        for k, (x, y) in iso.items()}
    best = max(ARMS, key=lambda n: desc[n])
    rep["best_arm"] = {"arm": best, "gain": desc[best], "gp_in_loop_trigger": bool(desc[best] >= 0.01)}
    rep["guard"] = C.guard_report()
    C.write_json(Path(a.out), rep)
    print(json.dumps({"winner": rep["winner"], "gain_vs_S0": desc, "best_arm": rep["best_arm"],
                      "gates": {n: rep["challengers"][n]["gate"]["pass"] for n in ARMS}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("oof")
    p.add_argument("--arms", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--s0-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--confirm-true-targets", action="store_true")
    p = sub.add_parser("decide")
    p.add_argument("--s0-dir", required=True)
    p.add_argument("--arms-root", required=True)
    p.add_argument("--l-dir", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"oof": cmd_oof, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    main()
