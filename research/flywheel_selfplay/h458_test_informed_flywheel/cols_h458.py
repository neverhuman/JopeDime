#!/usr/bin/env python3
"""H458 scored column builder for specialist checkpoints.

Runs the evaluation column builder of `cols_104m_e8_gpresid_h452` (its CPU prep and its GPU evaluation: 2 x CV-7 on
SEARCH rows plus the full-SEARCH refit) for H458 checkpoints and writes one npz per context in the `_load_col_block`
layout (`ok`, `q_*`, `bag_*`, `ref_*`, `wall_s`) plus one `SIDECAR.json` per block directory, the format Track S's
scorer reads. No evaluation code is copied; two things are overridden for the duration of a call and restored
afterwards:

  * the GP anchor is switched off. H458 specialists are not trained in the GP-residual space, so a block carries the
    three direct columns only (`COLUMNS`); the builder's fourth column would repeat the ridge column and is not
    written;
  * arm "canon": a specialist trained with canonical feature-identity slots is scored with the same slots and the
    same single native-order view (`episodes_h458.canonical_view`) instead of the builder's random slot draw.

Closure classes (written to the sidecar, read by the stack):
  recall       one checkpoint trained on the whole family pool; its numbers are recall of labels it has seen. A
               recall specialist is worse than its init on rows outside its pool (measured: -0.09 to -0.15 u with
               canonical slots), so this closure is refused unless every SEARCH and AUDIT feature row of the
               context sits in the checkpoint's pool;
  skill-xfit5  five checkpoints, the k-th trained without feature stratum k; every SEARCH row and every AUDIT row is
               scored by the checkpoint that never trained on its stratum;
  fair         a checkpoint that never trained on dev91 rows.

It reads SEARCH rows and AUDIT features. It never reads an audit label. One process, one context at a time: the
overrides are module-level and are not safe to use from several threads.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import paths_h458 as P  # noqa: E402

P.setup_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import episodes_h458 as E  # noqa: E402
import family_pool_h458 as FP  # noqa: E402
import probe_h458 as PR  # noqa: E402

C = E.C
COLUMNS = ("icl_e8_cw12_robridge_cv7", "icl_e8_cw12_yj_cv7", "icl_e8_rankgauss_duan_cv7")
UNUSED_COLUMN = "icl_e8_gp_resid_cv7"
P418 = Path("/home/ubuntu/jope_plan/evidence/H418/pulled")
P448 = Path("/home/ubuntu/jope_plan/evidence/H448/pulled")
N_VIEWS, N_REPEATS, K_FOLDS = 4, 2, 7
XFIT_K = 5
_STATE = {"n_raw": None}


class SlotView(np.ndarray):
    """A context-feature view that carries the canonical slot of each of its columns."""

    h458_slots: np.ndarray | None = None

    def __array_finalize__(self, obj) -> None:
        self.h458_slots = getattr(obj, "h458_slots", None)


def _no_gp_anchor(cid, y, lab, na, *_a, **_k):
    zero_s, zero_a = np.zeros(len(y), dtype=np.float64), np.zeros(int(na), dtype=np.float64)
    return zero_s, zero_a, zero_a.copy(), False


def _views_canon(X_tr, y_tr, X_ev, n_views, seed, keep_first=0):
    n_pin, n_raw = int(keep_first), int(_STATE["n_raw"])
    ff = E.FoldFeatures(X_tr, X_ev, n_pin, n_raw, int(X_tr.shape[1]) - n_pin - n_raw)
    if ff.n_struct < 0:
        raise ValueError("column roles do not add up; the GP anchor must be off for the canonical arm")
    vt, ve, slots = E.canonical_view(ff, y_tr)
    vt = np.asarray(vt).view(SlotView)
    vt.h458_slots = slots
    return [(vt, ve)]


def _predict_canon(models_w, views, y_tr, dev, seed, inv_fn=None) -> np.ndarray:
    vt, ve = views[0]
    if ve.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    acc, w_sum = np.zeros(ve.shape[0], dtype=np.float64), 0.0
    for mdl, w in models_w:
        acc += float(w) * PR.predict_canon(mdl, np.asarray(vt), ve, y_tr, vt.h458_slots, dev, inv_fn)
        w_sum += float(w)
    return acc / max(w_sum, 1e-12)


@contextlib.contextmanager
def builder_overrides(arm: str, n_raw: int):
    """Applies the H458 overrides to the column-builder module and restores it afterwards."""
    names = ("_load_gp_anchor_for_ctx", "_build_views", "_batched_icl_predict")
    saved = {n: getattr(C, n) for n in names}
    if arm not in ("rand", "canon"):
        raise ValueError(f"unknown arm {arm!r}")
    C._load_gp_anchor_for_ctx = _no_gp_anchor
    if arm == "canon":
        _STATE["n_raw"] = int(n_raw)
        C._build_views = _views_canon
        C._batched_icl_predict = _predict_canon
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(C, n, f)
        _STATE["n_raw"] = None


def build_context(cid: str, xs, xa, y, lab, model_sets, dev, arm: str, n_repeats: int = N_REPEATS) -> list[dict[str, np.ndarray]]:
    """All columns of one context, once per entry of `model_sets` (each a `[(model, weight), ...]` list).

    The CPU prep (folds, features, targets, views) is shared by the model sets, as in the builder's own multi-spec run.
    Scored blocks use the default two repeats; the selftest passes one to halve its cost.
    """
    with builder_overrides(arm, int(np.asarray(xs).shape[1])):
        prepped = C._prep_full_context_worker((0, cid, xs, xa, y, lab, None, P418, P448, None, N_VIEWS, int(n_repeats), K_FOLDS))
        return [C._eval_prepped_context_gpu(prepped, mw, dev) for mw in model_sets]


def crossfit_merge(per_k: list[dict[str, np.ndarray]], s_search: np.ndarray, s_audit: np.ndarray) -> dict[str, np.ndarray]:
    """Takes each row's prediction from the checkpoint that never trained on the row's feature stratum."""
    out: dict[str, np.ndarray] = {"ok": np.array(all(bool(a["ok"]) for a in per_k))}
    for key in per_k[0]:
        if key == "ok":
            continue
        strata = np.asarray(s_search if key.startswith("q_") else s_audit, dtype=np.int64)
        table = np.stack([np.asarray(a[key], dtype=np.float64) for a in per_k])
        if table.shape[1] != len(strata) or int(strata.max(initial=0)) >= len(per_k):
            raise ValueError(f"{key}: strata do not match the columns")
        out[key] = table[strata, np.arange(table.shape[1])]
    return out


def pack_block(arrays: dict[str, np.ndarray], wall_s: float) -> dict[str, np.ndarray]:
    """The npz of one context: the three direct columns only, in the `_load_col_block` layout."""
    out: dict[str, np.ndarray] = {"ok": np.array(bool(arrays["ok"]))}
    for col in COLUMNS:
        for pre in ("q_", "bag_", "ref_"):
            out[pre + col] = np.asarray(arrays[pre + col], dtype=np.float64)
    out["wall_s"] = np.array(float(wall_s))
    return out


def verify_block_file(fp: Path, n_search: int, n_audit: int) -> None:
    """What the stack's loader and the scorer's closure check require of a block file."""
    with np.load(fp, allow_pickle=False) as z:
        extra = [k for k in z.files if k not in ("ok", "wall_s") and not k.startswith(("q_", "bag_", "ref_"))]
        if extra or not bool(z["ok"]):
            raise SystemExit(f"{fp}: not ok, or keys outside the block layout: {extra}")
        for col in COLUMNS:
            q, bag, ref = z["q_" + col], z["bag_" + col], z["ref_" + col]
            if q.shape != (n_search,) or bag.shape != (n_audit,) or ref.shape != (n_audit,):
                raise SystemExit(f"{fp}: {col} has the wrong shape")
            if not (np.isfinite(q).all() and np.isfinite(bag).all() and np.isfinite(ref).all()):
                raise SystemExit(f"{fp}: {col} has non-finite values")


def producer() -> str:
    src = P.HERE / "source_commit"
    return "cl_h458 cols_h458.py @ " + (src.read_text().strip() if src.is_file() else "unstaged working copy")


def update_sidecar(out_dir: Path, fixed: dict, call: dict, coverage: dict) -> dict:
    """`SIDECAR.json` of a block directory (the layout Track S's scorer reads), merged over the calls that fill it.

    A directory holds one closure class, one slot arm and one pool; a call that disagrees is refused.
    """
    fp = out_dir / "SIDECAR.json"
    if fp.is_file():
        doc = json.loads(fp.read_text())
        clash = {k: (doc.get(k), v) for k, v in fixed.items() if doc.get(k) != v}
        if clash:
            raise SystemExit(f"{fp}: this block directory was started with different settings: {clash}")
    else:
        doc = {
            "names": list(COLUMNS),
            **fixed,
            "producer": producer(),
            "notes": "H458 Track D specialist block; GP-residual column not produced; reads no audit label",
            "strata_spec": "blake2b(canon(X_row), digest_size=8, key=b'h458-k5-strata-v1') mod 5" if fixed["cross_fitted"] else None,
            "reads_audit_labels": False,
            "weights_by_cid": {},
            "pool_coverage_search_audit": {},
            "calls": [],
            **P.DISCLOSURE,
        }
    doc["weights_by_cid"].update({cid: call["ckpt_sha256"] for cid in call["cids"]})
    doc["pool_coverage_search_audit"].update(coverage)
    doc["calls"].append(call)
    doc["contexts"] = sorted(doc["weights_by_cid"])
    doc["weights_sha256"] = hashlib.sha256(json.dumps(doc["weights_by_cid"], sort_keys=True).encode()).hexdigest()
    tmp = out_dir / f".SIDECAR.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, fp)
    return doc


def load_bank(bank_p: Path) -> dict[str, dict]:
    """SEARCH rows, SEARCH labels, the stored fold labels and AUDIT features per context. No audit label exists here."""
    out: dict[str, dict] = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        if "ya" in zb.files:
            raise SystemExit("audit labels found in the search bank")
        cids = [str(v) for v in zb["cids"].tolist()]
        fam = [str(v) for v in zb["family"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            mat = np.array(zb["search"][so : so + n * w], dtype=np.float64).reshape(n, w)
            out[c] = {
                "family": fam[i],
                "Xs": np.ascontiguousarray(mat[:, :-1]),
                "Xa": np.array(zb["audit"][ao : ao + na * f], dtype=np.float64).reshape(na, f),
                "y": np.array(zb["y64"][lo : lo + n], dtype=np.float64),
                "lab": np.array(zb["lab"][lo : lo + n], dtype=np.int64),
            }
            so += n * w
            ao += na * f
            lo += n
    return out


def require_pool_coverage(pool: FP.FamilyPool, fam8: str, cid: str, xs: np.ndarray, xa: np.ndarray) -> list[float]:
    """Closure "recall" on a context is legitimate only when the pool holds every one of its rows."""
    cov = [pool.coverage(fam8, xs), pool.coverage(fam8, xa)]
    if min(cov) < 1.0:
        raise SystemExit(f"{cid}: closure 'recall' refused, pool coverage search {cov[0]:.4f} audit {cov[1]:.4f} (must be 1.0)")
    return cov


def check_checkpoint(path: Path, key: str, closure: str, arm: str, stratum: int | None, pool_sha: str) -> dict:
    """Refuses a checkpoint whose recorded training config contradicts the closure class, the slot arm or the pool."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if key not in ("auto", "ema_soup50") and ck.get(key) is None:
        raise SystemExit(f"{path}: no key {key!r}")
    cfg = ck.get("cfg") if isinstance(ck.get("cfg"), dict) else None
    trained_pool = (ck.get("state") or {}).get("pool_sha256") if isinstance(ck.get("state"), dict) else None
    del ck
    if closure != "fair" and trained_pool != pool_sha:
        raise SystemExit(f"{path}: trained on pool {str(trained_pool)[:16]}, scoring against pool {pool_sha[:16]}")
    if closure == "fair":
        if cfg is not None and cfg.get("families"):
            raise SystemExit(f"{path}: an H458 specialist cannot be scored as closure 'fair'")
        return {}
    if cfg is None:
        raise SystemExit(f"{path}: closure {closure!r} needs an H458 checkpoint with its training config")
    if cfg.get("arm") != arm:
        raise SystemExit(f"{path}: trained with arm {cfg.get('arm')!r}, asked to score with {arm!r}")
    left_out = cfg.get("exclude_stratum")
    if closure == "recall" and (left_out is not None or cfg.get("exclude_cids")):
        raise SystemExit(f"{path}: trained with rows left out; that is not closure 'recall'")
    if closure == "skill-xfit5" and (left_out is None or int(left_out) != int(stratum)):
        raise SystemExit(f"{path}: trained without stratum {left_out!r}, expected {stratum}")
    return {"families": cfg.get("families"), "exclude_stratum": left_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closure", choices=("recall", "skill-xfit5", "fair"), required=True)
    ap.add_argument("--ckpt", default="", help="one checkpoint (closure recall or fair)")
    ap.add_argument("--xfit-ckpts", default="", help="five comma-separated checkpoints; the k-th was trained without stratum k")
    ap.add_argument("--ckpt-key", default="ema")
    ap.add_argument("--size", default="M")
    ap.add_argument("--arm", choices=("rand", "canon"), required=True)
    ap.add_argument("--families", default="", help="comma-separated 8-hex family ids scored with these checkpoints")
    ap.add_argument("--cids", default="")
    ap.add_argument("--out-dir", required=True, help="block directory; one npz per context, one sidecar line per call")
    ap.add_argument("--bank", default=str(P.EVAL_BANK))
    ap.add_argument("--pool", default=str(P.POOL_DIR), help="the family pool the checkpoints were trained on")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cuda-vram-frac", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    xfit = a.closure == "skill-xfit5"
    paths = [Path(p) for p in (a.xfit_ckpts if xfit else a.ckpt).split(",") if p]
    if len(paths) != (XFIT_K if xfit else 1) or bool(a.xfit_ckpts) != xfit or bool(a.ckpt) == xfit:
        raise SystemExit("closure skill-xfit5 takes --xfit-ckpts with five checkpoints; the other closures take --ckpt")
    torch.set_num_threads(max(1, a.threads))
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(a.cuda_vram_frac))
    out_dir = P.guard_write_path(Path(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_bank(Path(a.bank))
    want = [f for f in a.families.split(",") if f]
    cids = [c for c in a.cids.split(",") if c] or sorted(c for c, d in bank.items() if d["family"][4:12] in want)
    if not cids:
        raise SystemExit("no contexts selected")
    pool = FP.FamilyPool(Path(a.pool), verify_data=True)
    trained = [check_checkpoint(p, a.ckpt_key, a.closure, a.arm, k if xfit else None, pool.sha256) for k, p in enumerate(paths)]
    coverage: dict[str, list[float]] = {}
    for cid in cids:
        fam8 = bank[cid]["family"][4:12]
        for t in trained:
            if t.get("families") and fam8 not in [str(f)[:8] for f in t["families"]]:
                raise SystemExit(f"{cid}: family {fam8} is not what the checkpoint was trained on ({t['families']})")
        if a.closure == "recall":
            coverage[cid] = require_pool_coverage(pool, fam8, cid, bank[cid]["Xs"], bank[cid]["Xa"])
        elif a.closure == "skill-xfit5":
            coverage[cid] = [pool.coverage(fam8, bank[cid]["Xs"]), pool.coverage(fam8, bank[cid]["Xa"])]
    shas = [P.digest_file(p) for p in paths]
    models = [C.load_icl_ckpt(p, a.size, dev, ckpt_key=a.ckpt_key) for p in paths]
    t0, done = time.time(), []
    for cid in cids:
        dst = out_dir / f"{cid}.npz"
        if dst.exists():
            raise SystemExit(f"{dst} exists: a block directory is written once per context")
        d = bank[cid]
        t_c = time.time()
        per_model = build_context(cid, d["Xs"], d["Xa"], d["y"], d["lab"], [[(m, 1.0)] for m in models], dev, a.arm)
        arrays = crossfit_merge(per_model, FP.feature_stratum(d["Xs"], XFIT_K), FP.feature_stratum(d["Xa"], XFIT_K)) if xfit else per_model[0]
        tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
        np.savez(tmp, **pack_block(arrays, time.time() - t_c))
        verify_block_file(tmp, len(d["y"]), d["Xa"].shape[0])
        os.replace(tmp, dst)
        done.append(cid)
    update_sidecar(
        out_dir,
        {"closure": a.closure, "cross_fitted": xfit, "arm": a.arm, "size": a.size, "ckpt_key": a.ckpt_key, "pool_sha256": pool.sha256},
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cids": done,
            "ckpts": [str(p) for p in paths],
            "ckpt_sha256": shas,
            "elapsed_s": round(time.time() - t0, 1),
        },
        coverage,
    )
    print(json.dumps({"written": len(done), "arm": a.arm, "closure": a.closure, "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
