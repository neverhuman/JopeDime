#!/usr/bin/env python3
"""Score ONE reference model on ONE context of a reference board, once, and save score + predictions.

The rule this tool exists for: a reference model is scored on a board exactly once per registry id. Later work reads
the stored scores (scores/<model_id>.jsonl) and predictions (preds/<model_id>/<cid>.f64); it never re-fits the model.

Protocol (PROTOCOL.md; identical to every H335 AUDIT entrant and to frontier_audit_h335.rs):
  fit on fold 0 (SEARCH rows) only, predict fold 1 (AUDIT rows) in envelope order;
  u = -ln(RMSE_audit / RMSE_base), RMSE_base = RMSE of mean(y_SEARCH) on the AUDIT rows; ratio clamped to >= 1e-12;
  degenerate baseline, exception, non-finite prediction or timeout -> status != completed and u = 0.

  score_reference.py one BOARD_ROOT MODEL_ID CID     (inside sandbox_refboard.sh; appends one JSON line, flock)
  score_reference.py check MODEL_ID                  (versions + weights sha256 the registry pins; non-zero exit on drift;
                                                      run once per model before a board run; `one` checks versions only)

BOARD_ROOT holds BOARD.json and envs/<cid>.jtt1 (read-only) and scores/, preds/ (written). Stdlib + numpy + the model's
own library. Prints one status line, never data.
"""
import fcntl
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REGISTRY = json.loads((HERE.parent / "models.json").read_text())
HARNESS_VERSION = "refboard-score-v1"


def read_jtt1(p: Path):
    raw = p.read_bytes()
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError(f"bad JTT1 header: {p.name}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0 or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise ValueError(f"not an AUDIT envelope: {p.name}")
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
    if not np.isin(folds, (0, 1)).all():
        raise ValueError(f"fold byte outside {{0,1}}: {p.name}")
    data = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1)
    return hashlib.sha256(raw).hexdigest(), folds.copy(), data[:, :width].copy(), data[:, width].copy()


def check_versions(spec: dict, weights: bool = False) -> dict:
    got = {}
    for pkg, want in spec.get("versions", {}).items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            have = None
        got[pkg] = have
        if have != want:
            raise RuntimeError(f"version drift for {pkg}: registry {want}, installed {have}; register a new model id")
    for path, want in (spec.get("weights_sha256", {}) if weights else {}).items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != want:
            raise RuntimeError(f"weights sha drift for {path}")
    return got


# --------------------------------------------------------------------------------------------------------------------
# One fit_predict per registry "kind". Every config value comes from models.json, so the registry is the record.
# --------------------------------------------------------------------------------------------------------------------
def fit_predict(spec: dict, Xs, ys, Xa):
    kind, cfg, T = spec["kind"], dict(spec.get("config", {})), int(spec.get("threads", 8))
    seed = int(spec.get("seed", 0))
    if kind == "sklearn_ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV())
    elif kind == "sklearn_random_forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_jobs=T, random_state=seed, **cfg)
    elif kind == "xgboost":
        import xgboost as x
        m = x.XGBRegressor(n_jobs=T, random_state=seed, **cfg)
    elif kind == "lightgbm":
        import lightgbm as lg
        m = lg.LGBMRegressor(n_jobs=T, random_state=seed, verbose=-1, **cfg)
    elif kind == "catboost":
        import catboost as c
        m = c.CatBoostRegressor(thread_count=T, random_seed=seed, verbose=0, allow_writing_files=False, **cfg)
    elif kind == "autogluon":
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        cols = [f"f{i}" for i in range(Xs.shape[1])]
        tr = pd.DataFrame(Xs, columns=cols)
        tr["y"] = ys
        # model files go to the sandbox's private /tmp tmpfs, discarded when the sandbox exits
        pr = TabularPredictor("y", problem_type="regression", eval_metric="rmse", path=f"/tmp/ag_{os.getpid()}",
                              verbosity=0).fit(tr, num_cpus=T, num_gpus=0, **cfg)
        return pr.predict(pd.DataFrame(Xa, columns=cols)).to_numpy(dtype=np.float64)
    elif kind == "tabpfn":
        import torch
        from tabpfn import TabPFNRegressor
        torch.set_num_threads(T)
        m = TabPFNRegressor(random_state=seed, device=spec.get("device", "cpu"), ignore_pretraining_limits=True, **cfg)
    elif kind in ("tabfm", "tabfm_ensemble"):
        import torch
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
        torch.set_num_threads(T)
        model = tabfm_v1_0_0_pytorch.load(model_type="regression", device=spec.get("device", "cpu"))
        m = TabFMRegressor.ensemble(model=model) if kind == "tabfm_ensemble" else TabFMRegressor(model=model)
    elif kind == "kumo_tabular":
        return kumo_fit_predict(cfg, T, Xs, ys, Xa, spec.get("device", "cpu"))
    else:
        raise ValueError(f"unknown kind {kind}")
    m.fit(Xs, ys)
    return np.asarray(m.predict(Xa), dtype=np.float64)


def kumo_fit_predict(cfg, T, Xs, ys, Xa, dev):
    """KumoTabular (NVIDIA structured-data-models): the panel_v1 kumo_run.py regression path, unchanged."""
    import pandas as pd
    import sdm
    import torch
    torch.set_num_threads(T)
    model = sdm.models.KumoTabular(task="regression", size=cfg["size"], device=dev).eval()
    cols = [f"f{i}" for i in range(Xs.shape[1])]
    df = pd.DataFrame(np.vstack([Xs, Xa]), columns=cols)
    df["target"] = np.concatenate([ys, np.resize(ys, len(Xa))]).astype(float)
    st = sdm.infer_stypes(df, overrides={"target": "numerical"})
    tt = sdm.TableTensor.from_pandas(df=df, stypes=st, device=dev)
    n = len(ys)
    with torch.inference_mode():
        o = model(x_context=tt[:n].drop_columns("target"), y_context=tt[:n, "target"],
                  x_query=tt[n:].drop_columns("target"), num_estimators=int(cfg["num_estimators"]))
    v = getattr(o, "numerical", o)
    p = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return p.reshape(len(Xa), -1).mean(axis=1).astype(np.float64)  # mean of the predicted quantiles


def one(board_root: Path, model_id: str, cid: str) -> None:
    spec = REGISTRY["models"][model_id]
    board = json.loads((board_root / "BOARD.json").read_text())
    pinned = {c["cid"]: c["envelope_sha256"] for c in board["contexts"]}
    env_sha, folds, X, y = read_jtt1(board_root / "envs" / f"{cid}.jtt1")
    if pinned.get(cid) != env_sha:
        raise SystemExit(f"envelope {cid[:12]} does not match BOARD.json")
    s, a = folds == 0, folds == 1
    rec = {"board": board["board"], "board_sha256": board["board_sha256"], "cid": cid, "envelope_sha256": env_sha,
           "model_id": model_id, "harness_version": HARNESS_VERSION,
           "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), "spec": spec,
           "n_search": int(s.sum()), "n_audit": int(a.sum()), "width": int(X.shape[1]),
           "status": "failed", "utility": 0.0, "host": platform.node(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0, r0 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    try:
        rec["versions"] = check_versions(spec)
        base = float(np.sqrt(np.mean((y[a] - y[s].mean()) ** 2)))
        rec["base_rmse_audit"] = base
        if not math.isfinite(base) or base <= 1e-12:
            raise ValueError("degenerate baseline")
        p = fit_predict(spec, X[s], y[s], X[a])
        if p.shape != (int(a.sum()),):
            raise ValueError(f"prediction shape {p.shape}")
        pdir = board_root / "preds" / model_id
        pdir.mkdir(parents=True, exist_ok=True)
        pb = p.astype("<f8").tobytes()
        (pdir / f"{cid}.f64").write_bytes(pb)
        rec["preds_sha256"] = hashlib.sha256(pb).hexdigest()
        rmse = float(np.sqrt(np.mean((y[a] - p) ** 2)))
        ratio = max(rmse / max(base, 1e-12), 1e-12)
        u = -math.log(ratio)
        rec.update(rmse_audit=rmse, ratio=ratio)
        if not (math.isfinite(rmse) and math.isfinite(u)):
            raise ValueError("non-finite rmse/ratio/utility")
        rec.update(utility=u, status="completed")
    except Exception as e:  # recorded as a failure with u = 0, the same rule as every H335 entrant
        rec["error"] = repr(e)[:400]
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["cpu_s"] = round((r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime), 3)
    rec["max_rss_mb"] = round(r1.ru_maxrss / 1024, 1)
    out = board_root / "scores" / f"{model_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps({"model_id": model_id, "cid": cid[:12], "status": rec["status"], "wall_s": rec["wall_s"]}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "one":
        one(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        print(json.dumps(check_versions(REGISTRY["models"][sys.argv[2]], weights=True)))
    else:
        raise SystemExit(__doc__)
