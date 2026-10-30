#!/usr/bin/env python3
"""CPU-only H09 archive-selection experiment; see PROTOCOL.md before any fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 53009
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SOURCE = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
NORMALIZERS = (
    "step2_lm_mean_1x256",
    "step2_lm_std_1x256",
    "step2_svd_basis_256x48",
    "lm80_mu",
    "lm80_sd",
    "mu_lm48",
    "sd_lm48",
    "mu_bb20",
    "sd_bb20",
    "mu_dm24",
    "sd_dm24",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    with Path(path).open("x") as out:
        out.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def roster(packet, catalog):
    """Metadata-only admission. No payload is opened until all roles pass."""
    manifest = read(packet / "source_manifest.json")
    if (
        manifest.get("source_manifest_sha256") != SOURCE
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
        or manifest.get("excluded_dl_holdout_count") != 93
    ):
        raise ValueError("unqualified source manifest")
    selected = read(packet / "selected.json")
    cat = read(catalog)
    by_id = {r["id"]: r for r in cat["datasets"]}
    excluded = set(cat["inventory"]["sealed_name_matches_excluded"])
    forbidden_families = {
        ds["family_key"]
        for ds in by_id.values()
        if ds.get("family_key")
        and (ds["id"] in excluded or "beyondarena" in json.dumps(ds).lower())
    }
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]

    rows = []
    for role, split in [("train", "DL-Train"), ("dev", "DL-Val")]:
        for original in selected[role]:
            r = dict(original)
            ds = by_id.get(r["dataset_id"])
            if (
                not ds
                or ds["id"] in excluded
                or "beyondarena" in json.dumps(ds).lower()
                or ds.get("task") != "regression"
                or r.get("task") != "regression"
                or r.get("dl_split") != split
                or r.get("source_split") not in ("train", "inner")
                or not ds.get("family_key")
                or ds.get("family_key") in forbidden_families
                or not r.get("family_id")
                or not ds["canonical_sha256"].startswith(r["dataset_id"])
            ):
                raise ValueError("inadmissible context metadata")
            a, b = "packet:" + r["family_id"], "catalog:" + ds["family_key"]
            ra, rb = root(a), root(b)
            parents[max(ra, rb)] = min(ra, rb)
            r["role"] = role
            r["component_key"] = a
            r["catalog_family"] = ds["family_key"]
            r["canonical_sha256"] = ds["canonical_sha256"]
            r["s3_manifest_key"] = ds["s3_objects"]["manifest.json"]["key"]
            rows.append(r)
    ids = [r["context_sha256"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate context")
    roles = {}
    for r in rows:
        r["component"] = root(r.pop("component_key"))
        roles.setdefault(r["component"], set()).add(r["role"])
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError("connected family crosses train/validation")
    families = sorted({r["component"] for r in rows if r["role"] == "train"})
    if len(families) < 5:
        raise ValueError("need at least five training components")
    np.random.default_rng(SEED).shuffle(families)
    fold = {f: i % 5 for i, f in enumerate(families)}
    for r in rows:
        r["group"] = (
            (
                "tree"
                if fold[r["component"]] < 3
                else "calibrate"
                if fold[r["component"]] == 3
                else "select"
            )
            if r["role"] == "train"
            else "validate"
        )
    return rows


def dope_snapshot(path):
    """Same tensor/parameter/normalizer hashing as dope_freeze_guard; no model forward."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    tensors = state["state_dict"]
    params, buffers = hashlib.sha256(), hashlib.sha256()
    keys = sorted(k for k in tensors if k.startswith("dope_model."))
    if len(keys) != 56:
        raise ValueError(
            "unexpected DOPE architecture; cannot classify parameter tensors"
        )

    def tensor_hash(tensor):
        arr = tensor.detach().cpu().contiguous().numpy()
        h = hashlib.sha256(f"{arr.dtype.str}:{list(arr.shape)}:".encode())
        h.update(arr.tobytes(order="C"))
        return h.hexdigest()

    for key in keys:
        params.update(f"{key}={tensor_hash(tensors[key])}\n".encode())
    for key in NORMALIZERS:
        buffers.update(f"normalizer.{key}={tensor_hash(tensors[key])}\n".encode())
    composite = hashlib.sha256(
        f"params:{params.hexdigest()}|buffers:{buffers.hexdigest()}".encode()
    ).hexdigest()
    if composite != DOPE or state["dope_sha256"] != DOPE:
        raise ValueError("DOPE checksum mismatch")
    return {"composite": composite, "checkpoint_sha256": digest(path)}


def context_features(raw):
    n, outer, p = struct.unpack_from("<III", raw, 4)
    if raw[:4] != b"JTT1" or outer or raw[16] != 2 or n < 2 or p < 1:
        raise ValueError("requires a train-only two-fold JTT1 table")
    if len(raw) != 17 + n + n * (p + 1) * 8:
        raise ValueError("invalid JTT1 size")
    a = np.frombuffer(raw, dtype="<f8", offset=17 + n).reshape(n, p + 1).copy()
    missing = float(np.mean(~np.isfinite(a[:, :-1])))
    a[~np.isfinite(a)] = np.nan
    med = np.nanmedian(a, axis=0)
    med = np.nan_to_num(med)
    a = np.where(np.isfinite(a), a, med)
    x, y = a[:, :-1], a[:, -1]
    z = x - x.mean(0)
    sd = x.std(0)
    z /= np.maximum(sd, 1e-8)
    zy = (y - y.mean()) / max(float(y.std()), 1e-8)
    f = [
        np.log1p(n),
        np.log1p(p),
        missing,
        np.log1p(sd).mean(),
        np.log1p(sd).std(),
        np.abs(np.mean(z**3, axis=0)).mean(),
        np.abs(np.mean(z * zy[:, None], axis=0)).mean(),
        np.log1p(y.std()),
    ]
    if not np.isfinite(f).all():
        raise ValueError("nonfinite context summaries")
    return np.asarray(f, dtype=np.float64)


def load_context(packet, row):
    cid = row["context_sha256"]
    if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
        raise ValueError("invalid context path")
    pool = packet / "pools" / (cid + ".npz")
    table = packet / "tables" / (cid + ".jtt1")
    if (
        digest(pool) != row["pool_sha256"]
        or digest(table) != row["train_only_jtt1_sha256"]
    ):
        raise ValueError("input digest mismatch")
    with np.load(pool, allow_pickle=False) as data:
        x, y = data["program"].astype(np.float64), data["fitness"].astype(np.float64)
        pid = data["program_sha256"].astype("U64")
    if (
        x.shape != (len(y), 412)
        or not len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid candidate matrix")
    if len(np.unique(pid)) != len(pid):
        raise ValueError("duplicate program identity")
    order = np.argsort(pid, kind="stable")
    return {
        "row": row,
        "x": x[order],
        "y": y[order],
        "pids": pid[order],
        "context": context_features(table.read_bytes()),
    }


def design(data, sample=False):
    xs, cs, ys, weights = [], [], [], []
    family_counts = {}
    for d in data:
        f = d["row"]["component"]
        family_counts[f] = family_counts.get(f, 0) + 1
    for d in data:
        n = min(64, len(d["y"])) if sample else len(d["y"])
        xs.append(d["x"][:n])
        cs.append(np.broadcast_to(d["context"], (n, 8)))
        y = d["y"][:n]
        ys.append(y - y.mean())
        weights.extend([1 / (family_counts[d["row"]["component"]] * n)] * n)
    w = np.asarray(weights)
    return np.concatenate(xs), np.concatenate(cs), np.concatenate(ys), w / w.mean()


def residual_features(x, c, sx, sc, projection):
    z = sx.transform(x)
    # Full 412 main effects survive; projection is only the additional interaction block.
    interactions = ((z @ projection)[:, :, None] * sc.transform(c)[:, None, :]).reshape(
        len(x), -1
    )
    return np.concatenate([z, interactions], axis=1)


def choices(data, tree, residual=None):
    result = []
    for d in data:
        c = np.broadcast_to(d["context"], (len(d["y"]), 8))
        prediction = tree.predict(np.concatenate([d["x"], c], axis=1))
        if residual is not None:
            model, sx, sc, projection = residual
            prediction += model.predict(
                residual_features(d["x"], c, sx, sc, projection)
            )
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite predictions")
        i = int(np.argmin(prediction))
        result.append(
            {
                "context": d["row"]["context_sha256"],
                "component": d["row"]["component"],
                "program": str(d["pids"][i]),
                "fitness": float(d["y"][i]),
            }
        )
    return result


def family_mean(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["component"], []).append(r["fitness"])
    return {f: float(np.mean(v)) for f, v in groups.items()}


def interval(values, draws=20000):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("need finite paired family effects")
    rng = np.random.default_rng(SEED + 1)
    means = np.mean(rng.choice(v, size=(draws, len(v)), replace=True), axis=1)
    return np.quantile(means, [0.0125, 0.9875]).tolist()


def plan(args):
    rows = roster(args.input, args.catalog)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        n: digest(args.input / n) for n in ["selected.json", "source_manifest.json"]
    }
    protocol = {
        "source_sha": digest(HERE / "run.py"),
        "protocol_sha": digest(HERE / "PROTOCOL.md"),
        "catalog_sha": digest(args.catalog),
        "input_metadata": metadata,
        "rows": rows,
        "dope_before": dope_snapshot(args.checkpoint),
        "created_unix": time.time(),
    }
    write_once(args.output / "plan.json", protocol)
    print(
        "PLAN FROZEN",
        {
            g: sum(r["group"] == g for r in rows)
            for g in ["tree", "calibrate", "select", "validate"]
        },
        flush=True,
    )


def bound_plan(args):
    p = read(args.output / "plan.json")
    if (
        p["source_sha"] != digest(HERE / "run.py")
        or p["protocol_sha"] != digest(HERE / "PROTOCOL.md")
        or p["catalog_sha"] != digest(args.catalog)
        or p["rows"] != roster(args.input, args.catalog)
        or p["dope_before"] != dope_snapshot(args.checkpoint)
        or any(digest(args.input / n) != h for n, h in p["input_metadata"].items())
    ):
        raise ValueError("frozen plan/input identity changed")
    return p


def train(args):
    p = bound_plan(args)
    if (args.output / "train-start.json").exists():
        raise ValueError("one training attempt per frozen plan")
    write_once(args.output / "train-start.json", {"unix": time.time()})
    groups = {
        g: [load_context(args.input, r) for r in p["rows"] if r["group"] == g]
        for g in ["tree", "calibrate", "select"]
    }
    x, c, y, w = design(groups["tree"], sample=True)
    tree = HistGradientBoostingRegressor(
        max_iter=128,
        max_leaf_nodes=31,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=SEED,
    )
    tree.fit(np.concatenate([x, c], axis=1), y, sample_weight=w)
    print("TREE FIT", len(y), flush=True)
    x, c, y, w = design(groups["calibrate"], sample=True)
    target = y - tree.predict(np.concatenate([x, c], axis=1))
    sx, sc = (
        StandardScaler().fit(x, sample_weight=w),
        StandardScaler().fit(c, sample_weight=w),
    )
    projection = np.random.default_rng(SEED).normal(size=(412, 16)) / np.sqrt(412)
    z = residual_features(x, c, sx, sc, projection)
    candidates = []
    for alpha in [1, 10, 100]:
        model = Ridge(alpha=alpha).fit(z, target, sample_weight=w)
        residual = (model, sx, sc, projection)
        value = float(
            np.mean(
                list(family_mean(choices(groups["select"], tree, residual)).values())
            )
        )
        candidates.append((value, alpha, residual))
        print("TRAIN SELECTION", alpha, value, flush=True)
    best = min(candidates, key=lambda t: (t[0], t[1]))
    joblib.dump((tree, best[2]), args.output / "models.joblib")
    write_once(
        args.output / "trained.json",
        {
            "plan_sha": digest(args.output / "plan.json"),
            "models_sha": digest(args.output / "models.joblib"),
            "alpha": best[1],
            "selection": [{"alpha": a, "mean_fitness": v} for v, a, _ in candidates],
            "dope_after": dope_snapshot(args.checkpoint),
            "scientific_native_fits": 0,
            "dope_updates": 0,
            "val_payloads_opened": 0,
        },
    )


def score(args):
    p = bound_plan(args)
    trained = read(args.output / "trained.json")
    if (
        trained["plan_sha"] != digest(args.output / "plan.json")
        or trained["models_sha"] != digest(args.output / "models.joblib")
        or trained["dope_after"] != p["dope_before"]
    ):
        raise ValueError("training identity changed")
    write_once(args.output / "score-start.json", {"unix": time.time()})
    tree, residual = joblib.load(args.output / "models.joblib")
    data = [load_context(args.input, r) for r in p["rows"] if r["group"] == "validate"]
    base, corrected = choices(data, tree), choices(data, tree, residual)
    b, c = family_mean(base), family_mean(corrected)
    differences = {f: b[f] - c[f] for f in sorted(b)}
    ci = interval(list(differences.values()))
    result = {
        "scope": "DEVELOPMENT_ONLY archived regression candidate selection; no new GP fits",
        "train_receipt_sha": digest(args.output / "trained.json"),
        "n_contexts": len(data),
        "n_components": len(b),
        "baseline_mean_fitness": float(np.mean(list(b.values()))),
        "residual_mean_fitness": float(np.mean(list(c.values()))),
        "lift": float(np.mean(list(differences.values()))),
        "paired_97_5_ci": ci,
        "per_component_lift": differences,
        "baseline_choices": base,
        "residual_choices": corrected,
        "dope_after": dope_snapshot(args.checkpoint),
        "terminal_val_evaluations": 1,
        "regression_elo": None,
        "elo_status": "HWV-01 handoff required"
        if ci[0] > 0
        else "no positive-CI handoff",
        "binary_elo": None,
        "holdout_sealed_payloads_opened": 0,
    }
    if result["dope_after"] != p["dope_before"]:
        raise ValueError("checkpoint/DOPE changed")
    write_once(args.output / "results.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "lift",
                    "paired_97_5_ci",
                    "n_contexts",
                    "n_components",
                    "elo_status",
                ]
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["plan", "train", "score"])
    ap.add_argument("--input", type=Path, default=Path.home() / "jope-data/h49/input")
    ap.add_argument(
        "--catalog",
        type=Path,
        default=Path.home() / "jope_plan/master/source/S3_DATASET_CATALOG.json",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "jope-data/h49/ckpt/round-785.pt",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
