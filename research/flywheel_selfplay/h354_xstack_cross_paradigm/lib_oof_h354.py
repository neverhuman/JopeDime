#!/usr/bin/env python3
"""H354 library runner: SEARCH out-of-fold (and, where not stored, SEARCH -> AUDIT) predictions of the fixed library.

Runs inside sandbox_h354.sh, which exposes only the H354 input root (read-only), the output root (read-write), the
reference-board harness (read-only) and the frontier binary (read-only). It imports nothing from H317/H335 and opens
no file outside --inputs.

Inputs (written by `phase_d_h354.py prepare` / Phase C prepare, every file sha256-listed in inputs/MANIFEST.sha256):
  search/<cid>.jtt1   H335's native SEARCH envelope (fold byte 0 / 1 = SEARCH fold 0 / 1)
  audit/<cid>.jtt1    the AUDIT envelope (fold 0 = all SEARCH rows, fold 1 = AUDIT rows; nq = 0)
  stored/<model>/<cid>.json   (Phase D only) the reference-board record + its stored AUDIT predictions
  frontier_sha.json   (Phase D only) the stored H335 frontier AUDIT prediction sha256 per (cid, entrant)

Per (learner, cid), out/<learner>/<cid>.json:
  p0 = predictions for SEARCH fold-0 rows from a fit on SEARCH fold-1 rows; p1 = the mirror (fold-0 fit -> fold 1);
  pa = AUDIT predictions from a fit on all SEARCH rows: classical learners in Phase D read the STORED reference-board
       predictions (no re-run); the LLM regressors re-run the pinned binary and must reproduce the stored sha256.
Usage: lib_oof_h354.py --inputs DIR --out DIR --jobs N [--learners a,b] [--audit fit|stored] [--limit K]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CLASSICAL = ("catboost.v1", "lightgbm.v1", "xgboost.v1", "random_forest.v1")
LLM = ("Opus5p5_Extra", "Gemini4_argon_high", "GPT6p1_Sol_Max", "GPT6_Astra_Max")
RUNNER_VERSION = "h354-lib-oof-v1"
LLM_CPU_LIMIT_S = 300  # the H335 frontier runner limit
LLM_WALL_LIMIT_S = 900
REFBOARD_TOOLS = Path(os.environ.get("H354_REFBOARD_TOOLS", ""))
FRONTIER_BIN = Path(os.environ.get("H354_FRONTIER_BIN", ""))
FRONTIER_BIN_SHA256 = "b359807c289a7fbd8304218d8c29b31d758d552e0b1f8a2e82a748686cc22d43"


def read_env(path: Path):
    raw = path.read_bytes()
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if raw[:4] != b"JTT1" or len(raw) != 17 + nt + 8 * (nt + nq) * (width + 1):
        raise ValueError(f"bad envelope {path}")
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17).copy()
    mat = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1).copy()
    return folds, mat[:, :width], mat[:, width], hashlib.sha256(raw).hexdigest()


def env_bytes(X, y, folds) -> bytes:
    mat = np.column_stack([X, y])
    return (b"JTT1" + struct.pack("<III", len(y), 0, X.shape[1]) + bytes([2]) + np.asarray(folds, np.uint8).tobytes()
            + np.ascontiguousarray(mat, dtype="<f8").tobytes())


def preds_sha(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<f8").tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Classical: the reference-board harness's own fit_predict (same registry config, threads, seed)
# ----------------------------------------------------------------------------------------------------------------------
def _refboard():
    sys.path.insert(0, str(REFBOARD_TOOLS))
    import score_reference as R  # noqa: E402
    return R


def classical_job(model: str, cid: str, inputs: Path, audit_mode: str) -> dict:
    R = _refboard()
    spec = R.REGISTRY["models"][model]
    R.check_versions(spec)
    folds, X, y, sha_s = read_env(inputs / "search" / f"{cid}.jtt1")
    out = {"cid": cid, "learner": model, "runner": RUNNER_VERSION, "search_env_sha256": sha_s,
           "harness_sha256": hashlib.sha256((REFBOARD_TOOLS / "score_reference.py").read_bytes()).hexdigest()}
    t0, c0 = time.time(), time.process_time()
    try:
        f0, f1 = folds == 0, folds == 1
        p0 = np.asarray(R.fit_predict(spec, X[f1], y[f1], X[f0]), np.float64)
        p1 = np.asarray(R.fit_predict(spec, X[f0], y[f0], X[f1]), np.float64)
        if p0.shape != (f0.sum(),) or p1.shape != (f1.sum(),) or not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            raise ValueError("non-finite or mis-shaped OOF predictions")
        out.update({"oof_status": "completed", "p0": p0.tolist(), "p1": p1.tolist()})
    except Exception as e:  # noqa: BLE001  (a failed learner is excluded on SEARCH; recorded, never retried)
        out.update({"oof_status": "failed", "p0": None, "p1": None, "error": f"{type(e).__name__}: {e}"[:500]})
    out["oof_wall_s"], out["oof_cpu_s"] = time.time() - t0, time.process_time() - c0
    if audit_mode == "stored":
        st = json.loads((inputs / "stored" / model / f"{cid}.json").read_text())
        if st["status"] == "completed":
            pa = np.asarray(st["pa"], np.float64)
            if preds_sha(pa) != st["preds_sha256"]:
                raise RuntimeError(f"{model} {cid}: stored AUDIT predictions fail their sha256")
            out.update({"audit_status": "completed", "pa": pa.tolist(), "audit_source": "stored reference-board record",
                        "audit_preds_sha256": st["preds_sha256"]})
        else:
            out.update({"audit_status": st["status"], "pa": None, "audit_source": "stored reference-board record"})
    else:
        fa, Xa, ya, sha_a = read_env(inputs / "audit" / f"{cid}.jtt1")
        try:
            pa = np.asarray(R.fit_predict(spec, Xa[fa == 0], ya[fa == 0], Xa[fa == 1]), np.float64)
            if pa.shape != ((fa == 1).sum(),) or not np.isfinite(pa).all():
                raise ValueError("non-finite or mis-shaped AUDIT predictions")
            out.update({"audit_status": "completed", "pa": pa.tolist(), "audit_preds_sha256": preds_sha(pa),
                        "audit_env_sha256": sha_a, "audit_source": "fit on all SEARCH rows"})
        except Exception as e:  # noqa: BLE001
            out.update({"audit_status": "failed", "pa": None, "error_audit": f"{type(e).__name__}: {e}"[:500]})
    out["status"] = "completed" if out["oof_status"] == "completed" else "failed"
    return out


# ----------------------------------------------------------------------------------------------------------------------
# LLM Rust regressors: the pinned H335 frontier binary (fit on fold-0 rows, predict fold-1 rows; --dump-preds)
# ----------------------------------------------------------------------------------------------------------------------
def _limit():
    resource.setrlimit(resource.RLIMIT_CPU, (LLM_CPU_LIMIT_S, LLM_CPU_LIMIT_S + 5))


def run_binary(env: bytes, entrant: str, tmp: Path, tag: str):
    ep, pp = tmp / f"{tag}.jtt1", tmp / f"{tag}.preds"
    ep.write_bytes(env)
    t0 = time.time()
    cp = subprocess.run([str(FRONTIER_BIN), str(ep), entrant, "-", "--dump-preds", str(pp)], capture_output=True,
                        text=True, timeout=LLM_WALL_LIMIT_S, preexec_fn=_limit)
    rec = json.loads(cp.stdout) if cp.returncode == 0 and cp.stdout.strip() else {"status": "failed",
                                                                                   "error": cp.stderr[-300:]}
    preds = np.frombuffer(pp.read_bytes(), dtype="<f8").copy() if rec.get("status") == "completed" and pp.exists() \
        else None
    return rec, preds, time.time() - t0


def llm_job(entrant: str, cid: str, inputs: Path, audit_mode: str, work: Path) -> dict:
    folds, X, y, sha_s = read_env(inputs / "search" / f"{cid}.jtt1")
    out = {"cid": cid, "learner": entrant, "runner": RUNNER_VERSION, "search_env_sha256": sha_s,
           "binary_sha256": FRONTIER_BIN_SHA256}
    # The exact envelopes and raw prediction dumps the binary saw are kept (no-delete rule): work/<entrant>/<cid>/
    tmp = work / entrant / cid
    tmp.mkdir(parents=True, exist_ok=True)
    # p0: fit on SEARCH fold 1 (byte 0), predict SEARCH fold 0 (byte 1); rows keep their order
    r0, p0, w0 = run_binary(env_bytes(X, y, np.where(folds == 1, 0, 1)), entrant, tmp, "p0")
    r1, p1, w1 = run_binary(env_bytes(X, y, folds), entrant, tmp, "p1")
    ok = p0 is not None and p1 is not None and len(p0) == (folds == 0).sum() and len(p1) == (folds == 1).sum()
    out.update({"oof_status": "completed" if ok else "failed", "p0": p0.tolist() if ok else None,
                "p1": p1.tolist() if ok else None, "oof_wall_s": w0 + w1,
                "error": None if ok else f"{r0.get('error')} | {r1.get('error')}"})
    fa, Xa, ya, sha_a = read_env(inputs / "audit" / f"{cid}.jtt1")
    ra, pa, _ = run_binary(env_bytes(Xa, ya, fa), entrant, tmp, "pa")
    aok = pa is not None and len(pa) == (fa == 1).sum()
    out.update({"audit_status": "completed" if aok else ra.get("status", "failed"),
                "pa": pa.tolist() if aok else None, "audit_preds_sha256": preds_sha(pa) if aok else None,
                "audit_env_sha256": sha_a})
    if audit_mode == "stored":
        want = json.loads((inputs / "frontier_sha.json").read_text())[cid][entrant]
        if out["audit_preds_sha256"] != want:
            raise RuntimeError(f"{entrant} {cid}: AUDIT re-run {out['audit_preds_sha256']} != stored {want}")
        out["audit_source"] = "re-run; sha256 equals the stored H335 frontier record"
    out["status"] = out["oof_status"]
    return out


def job(learner: str, cid: str, inputs: str, out_dir: str, audit_mode: str) -> str:
    inputs, out_dir = Path(inputs), Path(out_dir)
    f = out_dir / learner / f"{cid}.json"
    if f.exists():
        return "skip"
    if learner in CLASSICAL:
        r = classical_job(learner, cid, inputs, audit_mode)
    else:
        r = llm_job(learner, cid, inputs, audit_mode, out_dir / "_work")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(r) + "\n")
    os.replace(tmp, f)
    return r["status"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--learners", default=",".join(CLASSICAL + LLM))
    ap.add_argument("--audit", choices=("stored", "fit"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if hashlib.sha256(FRONTIER_BIN.read_bytes()).hexdigest() != FRONTIER_BIN_SHA256:
        raise SystemExit("frontier binary sha256 mismatch")
    inputs = Path(a.inputs)
    subprocess.run(["sha256sum", "--quiet", "-c", "MANIFEST.sha256"], cwd=inputs, check=True)
    cids = sorted(p.stem for p in (inputs / "search").glob("*.jtt1"))
    if a.limit:
        cids = cids[: a.limit]
    learners = [x for x in a.learners.split(",") if x]
    # classical jobs are 8-thread fits: interleave so a few run with many single-thread LLM jobs
    todo = [(lr, c) for c in cids for lr in learners]
    t0 = time.time()
    counts: dict = {}
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(job, lr, c, str(inputs), a.out, a.audit): (lr, c) for lr, c in todo}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            st = fu.result()  # a sha mismatch raises here and stops the run
            counts[st] = counts.get(st, 0) + 1
            if i % 100 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(todo)} {counts} {time.time() - t0:.0f}s", flush=True)
    print(f"LIB_OOF_DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
