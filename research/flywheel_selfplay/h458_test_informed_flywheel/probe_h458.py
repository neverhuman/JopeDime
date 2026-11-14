#!/usr/bin/env python3
"""H458 probe: SEARCH-only out-of-fold utility of an ICL checkpoint under the evaluation's own fold pipeline.

For every probed benchmark context this reproduces repeat 0 of the column builder's CV-7 on the SEARCH rows
(same folds, same per-fold seeds, same feature block, same three direct target spaces, same four views and slot draw
through `_batched_icl_predict`) and reports `u = -ln(RMSE_oof / std(y_search))` per target mode. It never opens an
audit label, and it does not need the audit features either: a query row's prediction depends only on the context.

For a specialist whose pool contains the context's rows this number is recall, by design (closure class "recall");
for a leave-context-out specialist the same number is skill on rows the weights never saw.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import paths_h458 as P  # noqa: E402

P.setup_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import episodes_h458 as E  # noqa: E402
import family_pool_h458 as FP  # noqa: E402

C = E.C
SEED_OFFSET = {"rr": 101, "yj": 202, "rg": 303}


def load_search_bank(bank_p: Path) -> dict[str, dict]:
    """SEARCH rows per context. The bank holds no audit labels; the audit features in it are not read."""
    out: dict[str, dict] = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        if "ya" in zb.files:
            raise SystemExit("audit labels found in the search bank")
        cids = [str(x) for x in zb["cids"].tolist()]
        fam = [str(x) for x in zb["family"].tolist()]
        ss = zb["s_shape"]
        search, y64 = zb["search"], zb["y64"]
        so = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            mat = np.array(search[so : so + n * w], dtype=np.float64).reshape(n, w)
            out[c] = {"family": fam[i], "Xs": np.ascontiguousarray(mat[:, :-1]), "y": np.array(y64[lo : lo + n], dtype=np.float64)}
            so += n * w
            lo += n
    return out


def fold_features_cached(cache_dir: Path | None, cid: str, fk: int, x_tr, y_tr, x_ev, fseed: int) -> E.FoldFeatures:
    fp = cache_dir / f"{cid}_r0_f{fk}.npz" if cache_dir is not None else None
    if fp is not None and fp.is_file():
        with np.load(fp, allow_pickle=False) as z:
            return E.FoldFeatures(np.array(z["X_tr"]), np.array(z["X_ev"]), int(z["n_pin"]), int(z["n_raw"]), int(z["n_struct"]))
    ff = E.eval_fold_features(x_tr, y_tr, x_ev, fseed)
    if fp is not None:
        tmp = fp.with_name(fp.name + ".part.npz")
        np.savez(tmp, X_tr=ff.X_tr, X_ev=ff.X_ev, n_pin=ff.n_pin, n_raw=ff.n_raw, n_struct=ff.n_struct)
        os.replace(tmp, fp)
    return ff


@torch.no_grad()
def predict_canon(model, vt, ve, y_ctx, slots, dev, inv_fn) -> np.ndarray:
    x = torch.from_numpy(np.vstack([vt, ve])[None]).to(device=dev, dtype=torch.float64)
    yc = torch.from_numpy(np.asarray(y_ctx, dtype=np.float64)[None]).to(device=dev)
    logits, ym, ys = model(x, yc, len(y_ctx), slots=torch.from_numpy(slots).to(dev), amp=(dev.type == "cuda"))
    out = model.mean_f64(logits, ym, ys) if inv_fn is None else model.mean_warped(logits, ym, ys, inv_fn)
    return out[0].detach().cpu().numpy()


def probe_context(
    model, dev, cid: str, xs: np.ndarray, y: np.ndarray, arm: str, cache_dir: Path | None, max_folds: int = 0, only_stratum: int | None = None
) -> dict:
    n_s = len(y)
    sy = max(float(np.std(y)), 1e-12)
    span = max(float(np.max(y) - np.min(y)), sy)
    lo_g, hi_g = float(np.min(y)) - 1.5 * span, float(np.max(y)) + 1.5 * span
    eff_k = 7 if n_s >= 35 else 5
    lab = C.make_stratified_kfold(cid, y, k=eff_k, rep=0)
    oof = {m: np.full(n_s, np.nan) for m in E.MODES}
    for fk in range(eff_k if not max_folds else min(eff_k, max_folds)):
        va = lab == fk
        tr = ~va
        fseed = E.fold_seed(cid, 0, fk)
        ff = fold_features_cached(cache_dir, cid, fk, xs[tr], y[tr], xs[va], fseed)
        mts = E.mode_targets(ff, y[tr], fseed)
        for m in E.MODES:
            mt = mts[m]
            inv = None if mt.warp is None else mt.warp.inv_torch
            if arm == "canon":
                vt, ve, slots = E.canonical_view(ff, mt.y_ctx)
                pr = predict_canon(model, vt, ve, mt.y_ctx, slots, dev, inv)
            else:
                views = C._build_views(ff.X_tr, mt.y_ctx, ff.X_ev, n_views=4, seed=fseed + SEED_OFFSET[m], keep_first=ff.n_pin)
                # one view per call: same seed, hence the same slot draw and the same mean as the batched call, at a
                # quarter of the memory
                pr = np.mean(
                    [C._batched_icl_predict([(model, 1.0)], [v], mt.y_ctx, dev, seed=fseed + SEED_OFFSET[m], inv_fn=inv) for v in views],
                    axis=0,
                )
            oof[m][va] = np.clip(mt.base_ev + pr, lo_g, hi_g)
    done = np.isfinite(oof["rr"])
    if only_stratum is not None:  # cross-fitted specialist: score only the rows its weights never trained on
        done &= FP.feature_stratum(xs) == int(only_stratum)
    res = {"n_search": int(n_s), "rows_scored": int(done.sum()), "sy": sy}
    for m in E.MODES:
        res[f"sse_{m}"] = float(np.sum((oof[m][done] - y[done]) ** 2))
        res[f"u_{m}"] = float(-math.log(max(float(np.sqrt(res[f"sse_{m}"] / max(1, int(done.sum())))) / sy, 1e-12)))
    blend = np.mean([oof[m][done] for m in E.MODES], axis=0)
    res["sse_blend"] = float(np.sum((blend - y[done]) ** 2))
    res["u_blend"] = float(-math.log(max(float(np.sqrt(res["sse_blend"] / max(1, int(done.sum())))) / sy, 1e-12)))
    res["u_best_mode"] = max(res[f"u_{m}"] for m in E.MODES)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-key", default="ema")
    ap.add_argument("--size", default="M")
    ap.add_argument("--arm", choices=("rand", "canon"), default="rand")
    ap.add_argument("--families", default="", help="comma-separated 8-hex family ids")
    ap.add_argument("--cids", default="", help="comma-separated context ids (overrides --families)")
    ap.add_argument("--bank", default=str(P.EVAL_BANK))
    ap.add_argument("--cache-dir", default=str(P.QUARANTINE / "probe_cache"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cuda-vram-frac", type=float, default=0.06, help="hard cap on this process's share of the GPU")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--only-stratum", type=int, default=-1, help="score only rows of this feature-hash stratum (cross-fitted specialists)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.set_num_threads(max(1, a.threads))
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(a.cuda_vram_frac))
    bank = load_search_bank(Path(a.bank))
    if a.cids:
        cids = [c for c in a.cids.split(",") if c]
    else:
        want = [f for f in a.families.split(",") if f]
        cids = sorted(c for c, d in bank.items() if any(d["family"].startswith(w) or d["family"][4:12] == w for w in want))
    if not cids:
        raise SystemExit("no contexts selected")
    cache_dir = P.guard_write_path(Path(a.cache_dir)) if a.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if a.ckpt_key not in ("auto", "ema_soup50") and ck.get(a.ckpt_key) is None:
        raise SystemExit(f"checkpoint has no key {a.ckpt_key!r}")
    del ck
    model = C.load_icl_ckpt(Path(a.ckpt), a.size, dev, ckpt_key=a.ckpt_key)
    t0 = time.time()
    per_ctx = {}
    for cid in cids:
        d = bank[cid]
        strat = a.only_stratum if a.only_stratum >= 0 else None
        per_ctx[cid] = {"family": d["family"][4:12], **probe_context(model, dev, cid, d["Xs"], d["y"], a.arm, cache_dir, a.max_folds, strat)}
    fams = sorted({v["family"] for v in per_ctx.values()})
    keys = [f"u_{m}" for m in E.MODES] + ["u_blend", "u_best_mode"]
    by_fam = {f: {k: float(np.mean([v[k] for v in per_ctx.values() if v["family"] == f])) for k in keys} for f in fams}
    doc = {
        "schema": "jope.h458.probe/v1",
        "label": a.label,
        "ckpt": str(a.ckpt),
        "ckpt_sha256": P.digest_file(Path(a.ckpt)),
        "ckpt_key": a.ckpt_key,
        "arm": a.arm,
        "only_stratum": a.only_stratum,
        "reads_audit_labels": False,
        "protocol": "search-only CV-7 repeat 0, evaluation fold pipeline, direct modes rr/yj/rg",
        "elapsed_s": round(time.time() - t0, 1),
        "by_family": by_fam,
        "per_context": per_ctx,
        **P.DISCLOSURE,
    }
    if a.out:
        out_p = P.guard_write_path(Path(a.out))
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"label": a.label, "arm": a.arm, "by_family": {f: {k: round(v, 4) for k, v in d.items()} for f, d in by_fam.items()}, "elapsed_s": doc["elapsed_s"]}))


if __name__ == "__main__":
    main()
