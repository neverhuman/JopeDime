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
  fair         a checkpoint that never trained on dev91 rows: its sha256 must be on the allowlist
               (`FAIR_ALLOWLIST`) and it must carry no in-corpus marker;
  fresh-probe  the fresh-row probe board of Track S (`PROBE_BOARD.json` beside the bank): SEARCH is the evaluation
               bank's, the "audit" rows are rows that sit in no bank and no pool. A specialist (role "specialist")
               and its pre-fine-tune checkpoint (role "init") are scored on the same contexts; the difference is
               skill on rows no model saw. Refused unless every SEARCH row and no probe row is in the family pool.
               A probe bank is refused for every other closure, so a probe block can never pass as a board block.
               `--merge-probe init=DIR,canon=DIR` joins such blocks into the one comparison block Track S's probe
               scorer reads: columns `<tag>_rr`, `<tag>_yj`, `<tag>_rg` and their mean `<tag>_mean3`.

It reads SEARCH rows and AUDIT features. It never reads an audit label. One process, one context at a time: the
overrides are module-level and are not safe to use from several threads.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"  # forced: the ridge base differs in its last bits under threaded BLAS

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
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
CLOSURES = ("recall", "skill-xfit5", "fair", "fresh-probe")
PROBE_MARK = "PROBE_BOARD.json"
FAIR_ALLOWLIST = P.EVIDENCE / "trackD" / "FAIR_CKPT_ALLOWLIST.json"
SHORT = {"icl_e8_cw12_robridge_cv7": "rr", "icl_e8_cw12_yj_cv7": "yj", "icl_e8_rankgauss_duan_cv7": "rg"}
# `quarantined_unfair_diagnostic` is not listed: in H449 it marks other boards' test rows, and the dev91-free H449 checkpoints carry it
IN_CORPUS_MARKS = ("in_corpus_dev91_disclosure", "quarantined_in_corpus", "quarantined_in_corpus_random_split_diagnostic")
KNOWN_PROBE_BANKS = P.KNOWN_PROBE_BANKS  # probe banks by content (one registry for every H458 entry point)
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


def _struct_cols_same_width(inner):
    """The canonical slots map raw column j to slot j, so the builder's cleaning step must keep every raw column."""

    def checked(x_ctx, y_ctx, x_q, *args, **kwargs):
        out = inner(x_ctx, y_ctx, x_q, *args, **kwargs)
        n_raw = int(_STATE["n_raw"])
        if out[0].shape[1] != n_raw or out[1].shape[1] != n_raw or np.asarray(x_ctx).shape[1] != n_raw:
            raise ValueError(f"cleaned width {out[0].shape[1]}/{out[1].shape[1]} differs from the context's {n_raw} raw columns")
        return out

    return checked


@contextlib.contextmanager
def builder_overrides(arm: str, n_raw: int):
    """Applies the H458 overrides to the column-builder module and restores it afterwards."""
    names = ("_load_gp_anchor_for_ctx", "_build_views", "_batched_icl_predict")
    saved = {n: getattr(C, n) for n in names}
    saved_struct = C.REL.build_h450_struct_cols
    if arm not in ("rand", "canon"):
        raise ValueError(f"unknown arm {arm!r}")
    C._load_gp_anchor_for_ctx = _no_gp_anchor
    if arm == "canon":
        _STATE["n_raw"] = int(n_raw)
        C._build_views = _views_canon
        C._batched_icl_predict = _predict_canon
        C.REL.build_h450_struct_cols = _struct_cols_same_width(saved_struct)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(C, n, f)
        C.REL.build_h450_struct_cols = saved_struct
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
        probe = fixed.get("closure") == "fresh-probe"
        doc = {
            "names": list(COLUMNS),
            **fixed,
            "producer": producer(),
            "notes": ("H458 Track D fresh-row probe block: the 'audit' rows are probe rows outside every bank and pool; not a board block, never stacked"
                      if probe else "H458 Track D specialist block; GP-residual column not produced; reads no audit label"),
            "strata_spec": "blake2b(canon(X_row), digest_size=8, key=b'h458-k5-strata-v1') mod 5" if fixed["cross_fitted"] else None,
            "reads_audit_labels": False,
            "weights_by_cid": {},
            "pool_coverage_search_audit": {},
            "calls": [],
            **P.DISCLOSURE,
        }
    doc["weights_by_cid"].update({cid: call["ckpt_sha256"] for cid in call["cids"]})
    doc["pool_coverage_search_audit"].update(coverage)
    last = doc["calls"][-1] if doc["calls"] else {}
    if call.get("call_id") is not None and last.get("call_id") == call["call_id"]:  # the same run, one more context
        last["cids"] = list(last["cids"]) + list(call["cids"])
        last["utc"], last["elapsed_s"] = call["utc"], call["elapsed_s"]
    else:
        doc["calls"].append(call)
    doc["contexts"] = sorted(doc["weights_by_cid"])
    doc["weights_sha256"] = hashlib.sha256(json.dumps(doc["weights_by_cid"], sort_keys=True).encode()).hexdigest()
    tmp = out_dir / f".SIDECAR.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, fp)
    return doc


def require_dir_consistent(out_dir: Path) -> None:
    """A block directory holds exactly the contexts its sidecar lists (a crash between the two leaves at most one orphan)."""
    have = sorted(f.stem for f in out_dir.glob("*.npz") if not f.name.startswith("."))
    fp = out_dir / "SIDECAR.json"
    listed = sorted(json.loads(fp.read_text()).get("contexts", [])) if fp.is_file() else []
    if have != listed:
        odd = sorted(set(have) ^ set(listed))
        raise SystemExit(f"{out_dir}: block files and SIDECAR.json disagree on {len(odd)} context(s), first {odd[:3]}; move the unlisted file aside and rerun")


def probe_board_of(bank_p: Path) -> dict | None:
    """The fresh-row probe board a bank file belongs to (Track S writes `PROBE_BOARD.json` beside it), or None."""
    mark = bank_p.parent / PROBE_MARK
    sha = P.digest_file(bank_p)
    if not mark.is_file():
        if sha in KNOWN_PROBE_BANKS:
            raise SystemExit(f"{bank_p}: the bank of probe board {KNOWN_PROBE_BANKS[sha]}, away from its board directory; use it in place")
        return None
    sums = bank_p.parent / "SHA256SUMS"
    listed = {ln.split()[0] for ln in sums.read_text().splitlines() if ln.strip()} if sums.is_file() else set()
    if sha not in listed:
        raise SystemExit(f"{bank_p}: sits beside {PROBE_MARK} but its sha256 is not in that board's SHA256SUMS")
    return {"probe_bank_sha256": sha, "probe_board_sha256": P.digest_file(mark)}


def merge_probe_blocks(sources: dict[str, Path], out_dir: Path, want: list[str]) -> dict:
    """One comparison block for the probe scorer from several fresh-probe blocks of the same probe bank.

    Every source keeps its three direct columns under `<tag>_<mode>` and gains their equal-weight mean `<tag>_mean3`.
    Only contexts that every source holds are written. No label is read: this renames and averages predictions.
    """
    docs: dict[str, dict] = {}
    for tag, src in sources.items():
        if not re.fullmatch(r"[a-z][a-z0-9]{0,11}", tag):
            raise SystemExit(f"merge tag {tag!r}: lower-case letters and digits only")
        require_dir_consistent(src)
        docs[tag] = json.loads((src / "SIDECAR.json").read_text())
        if docs[tag].get("closure") != "fresh-probe" or list(docs[tag].get("names", [])) != list(COLUMNS):
            raise SystemExit(f"{src}: not a fresh-probe block of this builder")
    banks = {d.get("probe_bank_sha256") for d in docs.values()}
    boards = {d.get("probe_board_sha256") for d in docs.values()}
    if len(sources) < 2 or len(banks) != 1 or len(boards) != 1 or None in banks:
        raise SystemExit("merge needs two or more blocks built on the same probe board")
    common = set.intersection(*(set(d["contexts"]) for d in docs.values()))
    cids = sorted(c for c in common if not want or c in want)
    if not cids or (want and len(cids) != len(set(want))):
        raise SystemExit(f"merge: {len(cids)} context(s) are in every source, {len(set(want))} were asked for")
    if any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir}: a merged block is written into an empty directory")
    for cid in cids:
        out: dict[str, np.ndarray] = {"ok": np.array(True)}
        wall = 0.0
        for tag, src in sources.items():
            with np.load(src / f"{cid}.npz", allow_pickle=False) as z:
                if not bool(z["ok"]):
                    raise SystemExit(f"{src}/{cid}.npz: not ok")
                wall += float(z["wall_s"])
                for pre in ("q_", "bag_", "ref_"):
                    cols = [np.asarray(z[pre + c], dtype=np.float64) for c in COLUMNS]
                    for c, v in zip(COLUMNS, cols):
                        out[f"{pre}{tag}_{SHORT[c]}"] = v
                    out[f"{pre}{tag}_mean3"] = np.mean(np.stack(cols), axis=0)
        out["wall_s"] = np.array(wall)
        tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
        np.savez(tmp, **out)
        os.replace(tmp, out_dir / f"{cid}.npz")
    bank_sha = next(iter(banks))
    doc = {
        "names": [f"{t}_{m}" for t in sources for m in (*SHORT.values(), "mean3")],
        "closure": "fresh-probe",
        "bank_sha256": bank_sha,
        "probe_bank_sha256": bank_sha,
        "probe_board_sha256": next(iter(boards)),
        "contexts": cids,
        "dropped_contexts": {t: sorted(set(docs[t]["contexts"]) - set(cids)) for t in sources},
        "merged_from": {t: {"dir": str(src), "sidecar_sha256": P.digest_file(src / "SIDECAR.json"), "probe_role": docs[t].get("probe_role"),
                            "arm": docs[t].get("arm"), "weights_sha256": docs[t].get("weights_sha256")} for t, src in sources.items()},
        "producer": producer(),
        "notes": "H458 Track D fresh-row probe comparison block: <tag>_mean3 is the equal-weight mean of a source's three direct columns; not a board block",
        "reads_audit_labels": False,
        **P.DISCLOSURE,
    }
    tmp = out_dir / f".SIDECAR.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "SIDECAR.json")
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


def require_probe_rows_fresh(pool: FP.FamilyPool, fam8: str, cid: str, xs: np.ndarray, xa: np.ndarray) -> list[float]:
    """Closure "fresh-probe": the context is the board's own (every SEARCH row in the pool) and no probe row is in the pool."""
    width = int(np.asarray(pool.table(fam8)).shape[1]) - 1
    if xs.shape[1] != width or xa.shape[1] != width:
        raise SystemExit(f"{cid}: closure 'fresh-probe' refused, widths search {xs.shape[1]} probe {xa.shape[1]} pool {width}")
    cov = [pool.coverage(fam8, xs), pool.coverage(fam8, xa)]
    if cov[0] < 1.0 or cov[1] > 0.0:
        raise SystemExit(f"{cid}: closure 'fresh-probe' refused, pool coverage search {cov[0]:.4f} (must be 1.0) probe {cov[1]:.4f} (must be 0.0)")
    return cov


def fair_allowlist() -> dict[str, str]:
    """sha256 -> note for the checkpoints that may be scored as closure "fair"; empty until someone registers one."""
    if not FAIR_ALLOWLIST.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(FAIR_ALLOWLIST.read_text()).get("sha256", {}).items()}


def check_checkpoint(path: Path, key: str, closure: str, arm: str, stratum: int | None, pool_sha: str, role: str = "specialist") -> dict:
    """Refuses a checkpoint whose recorded training config contradicts the closure class, the slot arm or the pool."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if key not in ("auto", "ema_soup50") and ck.get(key) is None:
        raise SystemExit(f"{path}: no key {key!r}")
    cfg = ck.get("cfg") if isinstance(ck.get("cfg"), dict) else None
    trained_pool = (ck.get("state") or {}).get("pool_sha256") if isinstance(ck.get("state"), dict) else None
    marks = [m for m in IN_CORPUS_MARKS if ck.get(m) or (cfg or {}).get(m)]
    del ck
    is_specialist = cfg is not None and bool(cfg.get("families"))
    if closure == "fair":
        if is_specialist:
            raise SystemExit(f"{path}: an H458 specialist cannot be scored as closure 'fair'")
        if marks:
            raise SystemExit(f"{path}: carries in-corpus marker(s) {marks}; not closure 'fair'")
        sha = P.digest_file(path)
        if sha not in fair_allowlist():
            raise SystemExit(f"{path}: sha256 {sha[:16]} is not on the fair allowlist {FAIR_ALLOWLIST}")
        return {}
    if closure == "fresh-probe" and role == "init":
        if is_specialist:
            raise SystemExit(f"{path}: role 'init' is the pre-fine-tune checkpoint, and this one is an H458 specialist")
        if arm != "rand":
            raise SystemExit("role 'init' is scored with arm 'rand': a checkpoint that was never trained with canonical slots has no canonical path")
        return {}
    if trained_pool != pool_sha:
        raise SystemExit(f"{path}: trained on pool {str(trained_pool)[:16]}, scoring against pool {pool_sha[:16]}")
    if cfg is None:
        raise SystemExit(f"{path}: closure {closure!r} needs an H458 checkpoint with its training config")
    if cfg.get("arm") != arm:
        raise SystemExit(f"{path}: trained with arm {cfg.get('arm')!r}, asked to score with {arm!r}")
    left_out = cfg.get("exclude_stratum")
    if closure in ("recall", "fresh-probe") and (left_out is not None or cfg.get("exclude_cids")):
        raise SystemExit(f"{path}: trained with rows left out; that is not closure {closure!r}")
    if closure == "skill-xfit5" and (left_out is None or int(left_out) != int(stratum)):
        raise SystemExit(f"{path}: trained without stratum {left_out!r}, expected {stratum}")
    return {"families": cfg.get("families"), "exclude_stratum": left_out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--closure", choices=CLOSURES, required=True)
    ap.add_argument("--probe-role", choices=("specialist", "init"), default="specialist", help="closure fresh-probe: the specialist, or its pre-fine-tune checkpoint")
    ap.add_argument("--ckpt", default="", help="one checkpoint (every closure but skill-xfit5)")
    ap.add_argument("--xfit-ckpts", default="", help="five comma-separated checkpoints; the k-th was trained without stratum k")
    ap.add_argument("--ckpt-key", default="ema")
    ap.add_argument("--size", default="M")
    ap.add_argument("--arm", choices=("", "rand", "canon"), default="")
    ap.add_argument("--merge-probe", default="", help="closure fresh-probe: tag=DIR,tag=DIR; writes the comparison block and stops")
    ap.add_argument("--families", default="", help="comma-separated 8-hex family ids scored with these checkpoints")
    ap.add_argument("--cids", default="")
    ap.add_argument("--out-dir", required=True, help="block directory; one npz per context, one sidecar line per call")
    ap.add_argument("--bank", default=str(P.EVAL_BANK))
    ap.add_argument("--pool", default=str(P.POOL_DIR), help="the family pool the checkpoints were trained on")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cuda-vram-frac", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    if a.merge_probe:
        if a.closure != "fresh-probe" or a.ckpt or a.xfit_ckpts:
            raise SystemExit("--merge-probe goes with closure 'fresh-probe' and takes no checkpoint")
        pairs = [kv.split("=", 1) for kv in a.merge_probe.split(",") if kv]
        if any(len(kv) != 2 for kv in pairs) or len({k for k, _ in pairs}) != len(pairs):
            raise SystemExit("--merge-probe takes tag=DIR pairs with distinct tags")
        out_dir = P.guard_write_path(Path(a.out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = merge_probe_blocks({k: Path(v) for k, v in pairs}, out_dir, [c for c in a.cids.split(",") if c])
        print(json.dumps({"merged": len(doc["contexts"]), "names": doc["names"], "out": str(out_dir)}))
        return
    if a.arm not in ("rand", "canon"):
        raise SystemExit("--arm rand or canon is required")
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
    require_dir_consistent(out_dir)
    fresh = a.closure == "fresh-probe"
    board = probe_board_of(Path(a.bank))
    if fresh != (board is not None):
        raise SystemExit("closure 'fresh-probe' takes the probe bank (PROBE_BOARD.json beside it), and a probe bank takes no other closure")
    if a.probe_role != "specialist" and not fresh:
        raise SystemExit("--probe-role belongs to closure 'fresh-probe'")
    bank = load_bank(Path(a.bank))
    want = [f for f in a.families.split(",") if f]
    cids = [c for c in a.cids.split(",") if c] or sorted(c for c, d in bank.items() if d["family"][4:12] in want)
    if not cids:
        raise SystemExit("no contexts selected")
    pool = FP.FamilyPool(Path(a.pool), verify_data=True)
    trained = [check_checkpoint(p, a.ckpt_key, a.closure, a.arm, k if xfit else None, pool.sha256, a.probe_role) for k, p in enumerate(paths)]
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
        elif fresh:
            coverage[cid] = require_probe_rows_fresh(pool, fam8, cid, bank[cid]["Xs"], bank[cid]["Xa"])
    shas = [P.digest_file(p) for p in paths]
    models = [C.load_icl_ckpt(p, a.size, dev, ckpt_key=a.ckpt_key) for p in paths]
    fixed = {"closure": a.closure, "cross_fitted": xfit, "arm": a.arm, "size": a.size, "ckpt_key": a.ckpt_key, "pool_sha256": pool.sha256}
    if fresh:  # keys that exist only on probe blocks, so earlier board sidecars stay comparable
        fixed.update({"probe_role": a.probe_role, "bank_sha256": board["probe_bank_sha256"], **board})  # bank_sha256: the key the probe scorer checks
    call_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
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
        update_sidecar(  # per context: a stopped run leaves a directory whose sidecar lists what is in it
            out_dir,
            fixed,
            {"call_id": call_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cids": [cid], "ckpts": [str(p) for p in paths],
             "ckpt_sha256": shas, "elapsed_s": round(time.time() - t0, 1)},
            {cid: coverage[cid]} if cid in coverage else {},
        )
    print(json.dumps({"written": len(done), "arm": a.arm, "closure": a.closure, "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
