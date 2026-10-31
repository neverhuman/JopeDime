#!/usr/bin/env python3
"""H124 fixed-lambda GBM shrink blend, CPU-only fit and paired readout."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import struct
import subprocess
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = Path.home() / "jope-data/h50/input"
CACHE = Path.home() / "jope-data/h58/cache"
R882 = Path.home() / "jope-data/h51/ckpt/round-882.pt"
SEEDS = (64, 65, 66)
FOLD_SEED = 12400
SAMPLE_SEED = 12400
LAMBDA_PRIMARY = 0.5
LAMBDA_SECONDARY = 1.0
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 124000
ELO_BOOTSTRAP_DRAWS = 1_000
ELO_BOOTSTRAP_SEED = 124001
H124_PREREG_COMMIT = "0967d1214a52aaafa35aaff51816afc143ded2fc"
H119_SOURCE_COMMIT = "db2862d677a6f402cfc4d075911d70d22c26d666"
H119_RUNNER_SHA = "8ed175fa240c3769f1fcef81e23c6128736bf539f7e771358da84f4dc7278ab7"
BASE = "a1b08cbfabe77410a7c323762146c85a50d7f16f"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
PINS = {
    "python/src/jope/north_star/dope_freeze_guard.py": "07de5bc34f181e8ef14c3e5cdc20716716b2cc7dd418792f82f3db1d0d576f9c",
    "python/src/jope/north_star/elo_tracker.py": "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835",
}
FILES = {
    INPUT
    / "source_manifest.json": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
    INPUT
    / "selected.json": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    INPUT
    / "cpu_exact_fused.py": "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab",
    CACHE
    / "cache_manifest.json": "66b1df1374242580d482b7d6f9b72b6fe8f65830f90f658ca662a698aa229897",
    CACHE
    / "context_features.npz": "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf",
    CACHE
    / "pool_priors.npz": "6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe",
    R882: "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    Path(
        "/home/ubuntu/JopeDimeMicro/python/model.py"
    ): "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a",
    Path(
        "/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py"
    ): "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a",
    Path(
        "/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py"
    ): "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
}
MODEL = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".new")
    with temp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def source_module(path, name):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO)
    if hashlib.sha256(source).hexdigest() != PINS[path]:
        raise RuntimeError("pinned canonical helper mismatch")
    module = types.ModuleType(name)
    module.__file__ = str(REPO / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def folds(rows):
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise ValueError("duplicate context")
    if any(
        r.get("dl_split") != "DL-Train"
        or r.get("source_split") not in ("train", "inner")
        or r.get("task") != "regression"
        for r in rows
    ):
        raise ValueError("folds require DL-Train regression metadata")
    families = sorted({r["family_id"] for r in rows})
    groups = [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), 5
        )
    ]
    mapping = {f: k for k, group in enumerate(groups) for f in group}
    return groups, {r["context_sha256"]: mapping[r["family_id"]] for r in rows}


def endpoints(cost, utility):
    cost, utility = np.asarray(cost), np.asarray(utility)
    if (
        cost.ndim != 1
        or cost.shape != utility.shape
        or len(cost) < 8
        or not np.isfinite(cost).all()
        or not np.isfinite(utility).all()
    ):
        raise ValueError("finite complete cost/utility pool required")
    top = 4 + np.argsort(cost[4:])[:4]
    return {
        "u4": float(max(np.max(utility[:4]), np.max(utility[top]))),
        "legacy": float(utility[np.argmin(cost)]),
    }


def blend_cost(r882_cost, gbm_cost, lam=LAMBDA_PRIMARY):
    if lam not in (LAMBDA_PRIMARY, LAMBDA_SECONDARY):
        raise ValueError("H124 lambda is fixed to 0.5 primary or 1.0 descriptive")
    reference, prior = np.asarray(r882_cost), np.asarray(gbm_cost)
    if (
        reference.shape != prior.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(prior).all()
    ):
        raise ValueError("finite, shape-matched H124 score vectors required")
    # Negating both model costs gives higher-is-better utilities; linear blending
    # in cost space is exactly the registered utility-space interpolation.
    return (reference + lam * (prior - reference)).astype(np.float32)


def calibration(pred, fitness, gate):
    mask = np.asarray(gate, bool) & np.isfinite(fitness) & np.isfinite(pred)
    x, y = (
        np.asarray(pred)[mask].astype(np.float64),
        np.asarray(fitness)[mask].astype(np.float64),
    )
    if len(x) < 10 or np.std(x) <= 1e-12:
        return 0.0, 1.0
    b = max(
        -float(np.mean((x - x.mean()) * (y - y.mean()))) / max(float(np.var(x)), 1e-12),
        0.05,
    )
    return float(y.mean() + b * x.mean()), b


def headroom(root):
    while os.getloadavg()[0] > 85:
        print("WAIT load", os.getloadavg()[0], flush=True)
        time.sleep(30)
    mem = {
        p[0].rstrip(":"): int(p[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if len(p := line.split()) >= 2 and p[1].isdigit()
    }
    if mem["MemAvailable"] < 16 * 1024**2:
        raise RuntimeError("MemAvailable below16GiB")
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss_kib >= 6 * 1024 * 1024:
        raise RuntimeError(f"worker RSS limit reached: {rss_kib} KiB")
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) >= 2_000_000_000:
        raise RuntimeError("own scratch reached2GB")
    if shutil.disk_usage(root).free < 2_000_000_000:
        raise RuntimeError("insufficient disk headroom")


def load_data(root):
    for path, expected in FILES.items():
        if sha(path) != expected:
            raise RuntimeError(f"input SHA mismatch: {path}")
    selected = json.loads((INPUT / "selected.json").read_text())
    manifest = json.loads((INPUT / "source_manifest.json").read_text())
    if (
        manifest["excluded_payloads_opened"] != 0
        or manifest["outer_rows_delivered"] != 0
    ):
        raise RuntimeError("input closure violated")
    rows = selected["train"]
    groups, mapping = folds(rows)
    if (
        len(rows) != 1219
        or len({r["family_id"] for r in rows}) != 205
        or any(len(x) != 41 for x in groups)
        or len(selected["dev"]) != 91
    ):
        raise RuntimeError("full roster required")
    if any(
        r.get("has_meta_test") is True
        or r.get("origin") not in (None, "real")
        or r.get("dl_split") != "DL-Train"
        for r in rows
    ):
        raise RuntimeError("H124 only admits real DL-Train rows")
    if {x["family_id"] for x in rows} & {x["family_id"] for x in selected["dev"]}:
        raise RuntimeError("train/secondary family alias")
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as z:
        cids = [str(x) for x in z["cids"]]
        arrays = {k: z[k] for k in ["tokens_10x32", "mask_10", "lm48_z", "dope629"]}
        context = {
            cid: {k: array[i] for k, array in arrays.items()}
            for i, cid in enumerate(cids)
        }
    if set(cids) != {r["context_sha256"] for r in rows + selected["dev"]}:
        raise RuntimeError("cache closure differs")
    # Primary phase loads only DL-Train payloads; DL-Val stays unopened until sealed.
    pools = load_pools(rows)
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = {
        r["context_sha256"]: 4
        + rng.choice(len(pools[r["context_sha256"]]["u"]) - 4, 192, replace=False)
        for r in rows
    }
    write(
        root / "split.json",
        {
            "seed": FOLD_SEED,
            "groups": groups,
            "fold_by_cid": mapping,
            "rows": rows,
            "sampling_sha256": hashlib.sha256(
                b"".join(
                    samples[r["context_sha256"]].astype("<i8").tobytes() for r in rows
                )
            ).hexdigest(),
            "sampling_seed": SAMPLE_SEED,
        },
    )
    return selected, context, pools, samples, mapping


def load_pools(rows):
    pools = {}
    for row in rows:
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("source_split") not in ("train", "inner")
            or row.get("task") != "regression"
        ):
            raise RuntimeError("protected or unsupported row")
        cid = row["context_sha256"]
        table = (INPUT / "tables" / f"{cid}.jtt1").read_bytes()
        if (
            hashlib.sha256(table).hexdigest() != row["train_only_jtt1_sha256"]
            or table[:4] != b"JTT1"
            or struct.unpack_from("<I", table, 8)[0] != 0
            or table[16] != 2
        ):
            raise RuntimeError("train-only table identity/header mismatch")
        path = INPUT / "pools" / f"{cid}.npz"
        if sha(path) != row["pool_sha256"]:
            raise RuntimeError("pool identity mismatch")
        with np.load(path, allow_pickle=False) as z:
            fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
            pools[cid] = {
                "program": z["program"].astype(np.float32),
                "fitness": fit,
                "gate": gate,
                "u": np.where(gate, -fit, -2.0),
            }
    return pools


def fit_one(task, root, rows, context, pools, samples, mapping):
    from sklearn.ensemble import HistGradientBoostingRegressor

    seed, excluded = task
    if (
        seed not in SEEDS
        or len(excluded) not in (1, 2)
        or tuple(sorted(set(excluded))) != excluded
        or any(k not in range(5) for k in excluded)
    ):
        raise ValueError("unregistered fit")
    name = f"seed{seed}-exclude{''.join(map(str, excluded)) or 'none'}"
    out = root / (name + ".npz")
    if out.exists():
        raise RuntimeError("fit already exists; no implicit retry/resume")
    headroom(root)
    t0 = time.monotonic()
    fit_rows = [
        r
        for k in range(5)
        if k not in excluded
        for r in rows
        if mapping[r["context_sha256"]] == k
    ]
    parts, targets = [], []
    for row in fit_rows:
        cid = row["context_sha256"]
        p, index = pools[cid], samples[cid]
        cf = np.concatenate([context[cid]["lm48_z"], context[cid]["dope629"]]).astype(
            np.float32
        )
        parts.append(
            np.concatenate(
                [p["program"][index], np.repeat(cf[None], 192, axis=0)], axis=1
            )
        )
        targets.append((p["u"][index] - p["u"].mean()).astype(np.float32))
    x, y = np.concatenate(parts), np.concatenate(targets)
    del parts, targets
    model = HistGradientBoostingRegressor(**MODEL, random_state=seed).fit(x, y)
    del x, y
    predictions = {}
    target_rows = [r for r in rows if mapping[r["context_sha256"]] in excluded]
    for row in target_rows:
        cid = row["context_sha256"]
        cf = np.concatenate([context[cid]["lm48_z"], context[cid]["dope629"]]).astype(
            np.float32
        )
        xp = pools[cid]["program"]
        predictions[cid] = model.predict(
            np.concatenate([xp, np.repeat(cf[None], len(xp), axis=0)], axis=1)
        ).astype(np.float32)
    np.savez_compressed(out, **predictions)
    receipt = {
        "seed": seed,
        "excluded_folds": list(excluded),
        "fit_contexts": len(fit_rows),
        "fit_rows": len(fit_rows) * 192,
        "features": 1089,
        "iterations": int(model.n_iter_),
        "model_parameters": model.get_params(),
        "prediction_sha256": sha(out),
        "predicted_contexts": len(predictions),
        "wall_seconds": time.monotonic() - t0,
    }
    write(out.with_suffix(".json"), receipt)
    print("FIT", name, receipt["wall_seconds"], flush=True)
    return receipt


def tree_packets(root, rows, pools, baseline):
    _, mapping = folds(rows)
    packets, calibration_receipts = {}, {}
    for seed in SEEDS:
        packets[str(seed)] = {}
        for k in range(5):
            pred_fit, fit, gates = [], [], []
            for j in range(5):
                if j == k:
                    continue
                pair = tuple(sorted((k, j)))
                with np.load(
                    root / f"seed{seed}-exclude{pair[0]}{pair[1]}.npz",
                    allow_pickle=False,
                ) as z:
                    for row in rows:
                        cid = row["context_sha256"]
                        if mapping[cid] == j:
                            pred_fit.append(z[cid][4:])
                            fit.append(pools[cid]["fitness"][4:])
                            gates.append(pools[cid]["gate"][4:])
            a, b = calibration(
                np.concatenate(pred_fit), np.concatenate(fit), np.concatenate(gates)
            )
            calibration_receipts[f"{seed}/{k}"] = [a, b]
            with np.load(root / f"seed{seed}-exclude{k}.npz", allow_pickle=False) as z:
                for row in rows:
                    cid = row["context_sha256"]
                    if mapping[cid] == k:
                        gbm_cost = (a - b * z[cid]).astype(np.float32)
                        blended_cost = blend_cost(
                            baseline[cid], gbm_cost, LAMBDA_PRIMARY
                        )
                        packets[str(seed)][cid] = {
                            "blend": endpoints(blended_cost, pools[cid]["u"]),
                            "pure_gbm": endpoints(gbm_cost, pools[cid]["u"]),
                            "r882": endpoints(baseline[cid], pools[cid]["u"]),
                        }
    return packets, calibration_receipts


def protocol():
    return {
        "wave": "H124",
        "fold_seed": FOLD_SEED,
        "sampling_seed": SAMPLE_SEED,
        "seeds": list(SEEDS),
        "folds": 5,
        "families_per_fold": 41,
        "gbm_model": dict(MODEL),
        "fit_count": 45,
        "samples_per_context": 192,
        "primary_lambda": LAMBDA_PRIMARY,
        "secondary_lambda": LAMBDA_SECONDARY,
        "primary_endpoint": "H119 Track-2 U4",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "elo_bootstrap_draws": ELO_BOOTSTRAP_DRAWS,
        "elo_bootstrap_seed": ELO_BOOTSTRAP_SEED,
        "checkpoint_sha256": FILES[R882],
        "dope_sha256": DOPE,
        "h119_source_commit": H119_SOURCE_COMMIT,
        "h119_runner_sha256": H119_RUNNER_SHA,
    }


def protocol_sha256():
    raw = json.dumps(protocol(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def family_values(rows, packets, method):
    cids = [row["context_sha256"] for row in rows]
    families = sorted({row["family_id"] for row in rows})
    if len(cids) != len(set(cids)) or len(families) != 205:
        raise ValueError(
            "H124 needs the unique 1,219-context/205-family DL-Train roster"
        )
    if set(packets) != {str(seed) for seed in SEEDS}:
        raise ValueError("H124 requires exactly seeds 64, 65, 66")
    values = np.empty((len(SEEDS), len(families)), dtype=np.float64)
    family_index = {family: index for index, family in enumerate(families)}
    for seed_index, seed in enumerate(SEEDS):
        if set(packets[str(seed)]) != set(cids):
            raise ValueError("H124 requires every DL-Train context once per seed")
        grouped = defaultdict(list)
        for row in rows:
            cell = packets[str(seed)][row["context_sha256"]]
            value = cell[method]["u4"]
            if not math.isfinite(value):
                raise ValueError("nonfinite H124 U4 utility")
            grouped[row["family_id"]].append(float(value))
        for family, index in family_index.items():
            if not grouped[family]:
                raise ValueError("empty H124 family")
            values[seed_index, index] = float(np.mean(grouped[family]))
    return families, values


def elo_h124(rows, packets, method, rng_seed=ELO_BOOTSTRAP_SEED):
    """Family-weighted two-contender Bradley-Terry Elo anchored at r882=1500."""
    helper = source_module("python/src/jope/north_star/elo_tracker.py", "h124_elo")
    records = []
    by_family = defaultdict(list)
    for row in rows:
        cid, family = row["context_sha256"], row["family_id"]
        candidate_score = float(
            np.mean([packets[str(seed)][cid][method]["u4"] for seed in SEEDS])
        )
        reference_score = float(
            np.mean([packets[str(seed)][cid]["r882"]["u4"] for seed in SEEDS])
        )
        records.append(
            helper.DatasetEvalRecord(
                dataset_id=cid,
                family_id=family,
                task_type="regression",
                scores_by_contender={
                    "H124": candidate_score,
                    "R882_CONTINUITY": reference_score,
                },
            )
        )
        delta = candidate_score - reference_score
        by_family[family].append(1.0 if delta > 1e-4 else 0.0 if delta < -1e-4 else 0.5)
    point = helper.compute_family_grouped_bradley_terry_elo(
        records, "R882_CONTINUITY", prior=0.01
    )
    per_seed = {}
    for seed in SEEDS:
        seed_records = [
            helper.DatasetEvalRecord(
                dataset_id=row["context_sha256"],
                family_id=row["family_id"],
                task_type="regression",
                scores_by_contender={
                    "H124": packets[str(seed)][row["context_sha256"]][method]["u4"],
                    "R882_CONTINUITY": packets[str(seed)][row["context_sha256"]][
                        "r882"
                    ]["u4"],
                },
            )
            for row in rows
        ]
        per_seed[str(seed)] = helper.compute_family_grouped_bradley_terry_elo(
            seed_records, "R882_CONTINUITY", prior=0.01
        )["ratings"]
    family_weights = np.asarray(
        [np.mean(by_family[family]) for family in sorted(by_family)], dtype=np.float64
    )
    samples = np.random.default_rng(rng_seed).integers(
        0, len(family_weights), size=(ELO_BOOTSTRAP_DRAWS, len(family_weights))
    )
    wins = family_weights[samples].sum(axis=1) + 0.01
    losses = len(family_weights) + 0.02 - wins
    gamma = np.ones((ELO_BOOTSTRAP_DRAWS, 2))
    for _ in range(400):
        denominator = (len(family_weights) + 0.02) / gamma.sum(axis=1)
        updated = np.stack([wins / denominator, losses / denominator], axis=1).clip(
            1e-6, 1e6
        )
        gamma = updated / updated[:, 1:2]
    rating_draws = (1500 + 400 * np.log10(gamma[:, 0].clip(1e-9))).round(2)
    return {
        "ratings": point["ratings"],
        "candidate_ci95": np.quantile(rating_draws, [0.025, 0.975]).round(2).tolist(),
        "reference_elo": 1500,
        "per_seed_ratings": per_seed,
        "bootstrap_draws": ELO_BOOTSTRAP_DRAWS,
        "bootstrap_seed": rng_seed,
    }


def h124_summary(rows, packets):
    families, reference = family_values(rows, packets, "r882")
    _, blend = family_values(rows, packets, "blend")
    _, pure_gbm = family_values(rows, packets, "pure_gbm")
    primary_gain = blend - reference
    secondary_gain = pure_gbm - reference
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(families), size=(BOOTSTRAP_DRAWS, len(families))
    )

    def summary(gain):
        per_seed = gain.mean(axis=1)
        boot = np.median(gain[:, indices].mean(axis=2), axis=0)
        harms = gain.mean(axis=0) < -0.01
        median = float(np.median(per_seed))
        lcb = float(np.quantile(boot, 0.05))
        return {
            "per_seed_mean_gain": per_seed.tolist(),
            "median_seed_gain": median,
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": lcb,
            "harm_count": int(harms.sum()),
            "family_count": len(families),
            "harm_fraction": float(harms.mean()),
            "harm_threshold": -0.01,
            "family_mean_gain_by_seed": gain.tolist(),
            "elo": None,
            "admitted": bool(
                median > 0 and lcb > 0.001 + 1e-12 and harms.mean() <= 0.10
            ),
        }

    primary = summary(primary_gain)
    primary["elo"] = elo_h124(rows, packets, "blend")
    secondary = summary(secondary_gain)
    secondary["admitted"] = None
    secondary["admission_assessed"] = False
    secondary["elo"] = elo_h124(rows, packets, "pure_gbm")
    return {
        "complete": True,
        "families": len(families),
        "contexts": len(rows),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_lambda": LAMBDA_PRIMARY,
        "primary_vs_r882": primary,
        "secondary_lambda": LAMBDA_SECONDARY,
        "secondary_pure_gbm_vs_r882": secondary,
    }


def baseline(root, rows, context, pools, *, net=None):
    import torch

    guard = source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h119_guard"
    )
    torch.set_num_threads(4)
    if net is None:
        copied = root / "r882-copy.pt"
        if copied.exists():
            raise RuntimeError("checkpoint copy already exists")
        shutil.copyfile(R882, copied)
        if sha(copied) != FILES[R882]:
            raise RuntimeError("checkpoint copy mismatch")
        spec = importlib.util.spec_from_file_location(
            "h119_fused", INPUT / "cpu_exact_fused.py"
        )
        fused = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused)
        net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
        checkpoint = torch.load(copied, map_location="cpu", weights_only=False)
        if checkpoint.get("dope_sha256") != DOPE:
            raise RuntimeError("checkpoint DOPE identity mismatch")
        unmatched = net.load_state_dict(checkpoint["state_dict"], strict=False)
        if (
            any(
                n.startswith(("adapter_10tok.", "dope_model."))
                for n in unmatched.missing_keys
            )
            or unmatched.unexpected_keys
        ):
            raise RuntimeError("checkpoint did not restore exact model")
        write(
            root / "checkpoint-load.json",
            {
                "missing": unmatched.missing_keys,
                "unexpected": unmatched.unexpected_keys,
                "checkpoint_sha256": sha(copied),
            },
        )
    for p in net.parameters():
        p.requires_grad_(False)
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    before = guard.snapshot_dope_weights(net).composite_sha256
    if before != DOPE:
        raise RuntimeError("DOPE hash mismatch before")
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as z:
        positions = {str(c): i for i, c in enumerate(z["cids"])}
        offsets, projection = z["offsets"], z["proj16"]
    predictions = {}
    with torch.no_grad():
        for row in rows:
            cid = row["context_sha256"]
            p = pools[cid]
            n, idx = len(p["u"]), positions[cid]
            proj = projection[int(offsets[idx]) : int(offsets[idx + 1])]
            result = net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((1, 584)),
                prog_412=torch.from_numpy(p["program"])[None],
                proj_16=torch.from_numpy(proj)[None],
                base_prior_cost=torch.zeros((1, n)),
                scale=torch.tensor([0.4208]),
                sos_prior=torch.zeros((1, n, 2)),
                mm_tokens_10x32=torch.from_numpy(context[cid]["tokens_10x32"])[None],
                mm_mask_10=torch.from_numpy(context[cid]["mask_10"])[None].bool(),
                revealed_indices=torch.arange(4)[None],
                revealed_raw_f=torch.as_tensor(p["fitness"][:4], dtype=torch.float32)[
                    None
                ],
                revealed_gate=torch.as_tensor(p["gate"][:4], dtype=torch.float32)[None],
            )
            predictions[cid] = result["total_cost"][0].numpy().astype(np.float64)
    after = guard.snapshot_dope_weights(net).composite_sha256
    if after != DOPE or before != after or sha(R882) != FILES[R882]:
        raise RuntimeError("DOPE/checkpoint changed")
    label = "train" if rows[0]["dl_split"] == "DL-Train" else "secondary"
    np.savez_compressed(root / f"r882-{label}-costs.npz", **predictions)
    write(
        root / f"dope-{label}.json",
        {
            "before": before,
            "after": after,
            "device": "cpu",
            "optimizer_updates": 0,
            "CUDA_initialized": torch.cuda.is_initialized(),
        },
    )
    return predictions, net


def preflight(root_parent: Path) -> dict:
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice < 15:
        raise RuntimeError(f"H124 requires nice >= 15, got {nice}")
    threads = {}
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        raw = os.environ.get(key)
        if raw is None or not raw.isdigit() or not 1 <= int(raw) <= 4:
            raise RuntimeError(f"H124 requires explicit {key} in 1..4")
        threads[key] = int(raw)
    while os.getloadavg()[0] > 85:
        print("WAIT load", os.getloadavg()[0], flush=True)
        time.sleep(30)
    mem_available = next(
        int(part[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if (part := line.split()) and part[0] == "MemAvailable:"
    )
    if mem_available < 16 * 1024**2:
        raise RuntimeError("MemAvailable below 16 GiB")
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss_kib >= 6 * 1024 * 1024:
        raise RuntimeError(f"worker RSS limit reached: {rss_kib} KiB")
    usage = shutil.disk_usage(
        root_parent if root_parent.exists() else root_parent.parent
    )
    if usage.free < 2_000_000_000:
        raise RuntimeError("less than 2 GB disk headroom")
    return {
        "load1": os.getloadavg()[0],
        "mem_available_kib": mem_available,
        "process_max_rss_kib": rss_kib,
        "nice": nice,
        "threads": threads,
        "filesystem_free_bytes": usage.free,
        "scratch_cap_bytes": 2_000_000_000,
    }


def verify_prereg_committed(source_commit: str) -> str:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", H124_PREREG_COMMIT, source_commit],
        cwd=REPO,
        check=True,
    )
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{H124_PREREG_COMMIT}:research/flywheel_selfplay/h124_gbm_shrink_blend/PREREG.md",
        ],
        cwd=REPO,
    )
    if expected != (HERE / "PREREG.md").read_bytes():
        raise RuntimeError("H124 PREREG bytes changed after preregistration commit")
    return hashlib.sha256(expected).hexdigest()


def scratch_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def cpu_dry_run() -> dict:
    """Synthetic protocol controls: no H50/H58/H51 reads or estimator fits."""
    rows = [
        {
            "context_sha256": f"{index:064x}",
            "family_id": f"synthetic-family-{index:03d}",
            "dl_split": "DL-Train",
            "source_split": "train",
            "task": "regression",
        }
        for index in range(205)
    ]
    groups, mapping = folds(rows)
    if [len(group) for group in groups] != [41] * 5 or set(mapping) != {
        row["context_sha256"] for row in rows
    }:
        raise RuntimeError("H124 synthetic family split failed")
    r882_cost = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    gbm_cost = np.asarray([0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 2.0, 1.0])
    blend_cost = r882_cost + LAMBDA_PRIMARY * (gbm_cost - r882_cost)
    pure_cost = r882_cost + LAMBDA_SECONDARY * (gbm_cost - r882_cost)
    if not np.array_equal(pure_cost, gbm_cost) or blend_cost[4] != 2.5:
        raise RuntimeError("H124 fixed-lambda score blend failed")
    return {
        "schema": "jope.h124.cpu-dry-run/v1",
        "status": "CPU_DRY_RUN_PASS",
        "protocol_sha256": protocol_sha256(),
        "synthetic_families": 205,
        "synthetic_folds": [len(group) for group in groups],
        "seeds": list(SEEDS),
        "fit_count_planned": 45,
        "data_reads": 0,
        "checkpoint_reads": 0,
        "estimator_fits": 0,
        "outcome_cells": 0,
        "GPU_used": False,
        "qualification_lock_used": False,
        "FIFO_ticket_created": False,
    }


def run_h124(root: Path) -> dict:
    root = Path(root)
    expected_parent = Path.home() / "jope-data/h124"
    if (
        root.exists()
        or root.is_symlink()
        or root.parent != expected_parent
        or not root.name.startswith("run-")
    ):
        raise RuntimeError("H124 requires a fresh run-* root under ~/jope-data/h124")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    prereg_sha = verify_prereg_committed(source_commit)
    launch = preflight(expected_parent)
    expected_parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    t0, cpu0 = time.monotonic(), time.process_time()
    run_sha = sha(HERE / "runner.py")
    write(
        root / "execution-identity.json",
        {
            "source_commit": source_commit,
            "prereg_commit": H124_PREREG_COMMIT,
            "prereg_sha256": prereg_sha,
            "runner_sha256": run_sha,
            "command": sys.argv,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "launch_preflight": launch,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        },
    )
    headroom(root)
    selected, context, pools, samples, mapping = load_data(root)
    rows = selected["train"]
    if len(rows) != 1219 or len({row["family_id"] for row in rows}) != 205:
        raise RuntimeError("H124 full pinned DL-Train roster required")
    headroom(root)
    r882, net = baseline(root, rows, context, pools)
    tasks = [(seed, (fold,)) for seed in SEEDS for fold in range(5)] + [
        (seed, pair) for seed in SEEDS for pair in itertools.combinations(range(5), 2)
    ]
    receipts = []
    # One worker keeps peak RSS below the wave's strict 6 GiB limit.
    with ThreadPoolExecutor(max_workers=1) as executor:
        for receipt in executor.map(
            lambda task: fit_one(task, root, rows, context, pools, samples, mapping),
            tasks,
        ):
            receipts.append(receipt)
            headroom(root)
    if len(receipts) != 45:
        raise RuntimeError("H124 requires 45 fit receipts")
    packets, calibration_receipts = tree_packets(root, rows, pools, r882)
    result = h124_summary(rows, packets)
    import sklearn
    import torch

    dope_receipt = json.loads((root / "dope-train.json").read_text())
    if (
        dope_receipt["before"] != DOPE
        or dope_receipt["after"] != DOPE
        or torch.cuda.is_initialized()
        or sha(HERE / "runner.py") != run_sha
    ):
        raise RuntimeError("H124 frozen DOPE/source/CPU execution guard failed")
    if any(receipt["seed"] not in SEEDS for receipt in receipts):
        raise RuntimeError("H124 unregistered seed in fit receipt")
    metrics = {
        "schema": "jope.h124.gbm-shrink-blend/v1",
        "status": "COMPLETE",
        "verdict": "ADMIT_LAMBDA_0_5"
        if result["primary_vs_r882"]["admitted"]
        else "NO_ADVANCE",
        "protocol": protocol(),
        "protocol_sha256": protocol_sha256(),
        "source_commit": source_commit,
        "prereg_commit": H124_PREREG_COMMIT,
        "prereg_sha256": prereg_sha,
        "runner_sha256": run_sha,
        "h119_source_commit": H119_SOURCE_COMMIT,
        "h119_runner_sha256": H119_RUNNER_SHA,
        "input_sha256": {str(path): expected for path, expected in FILES.items()},
        "primary": result["primary_vs_r882"],
        "secondary_report_only": result["secondary_pure_gbm_vs_r882"],
        "family_ids": sorted({row["family_id"] for row in rows}),
        "calibration": calibration_receipts,
        "fit_receipts": receipts,
        "runtime": {
            "wall_seconds": time.monotonic() - t0,
            "process_cpu_seconds": time.process_time() - cpu0,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scratch_bytes": scratch_bytes(root),
        },
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "read_boundary": {
            "dataset_source": "real S3 H50 cached input",
            "rows_opened": 1219,
            "dl_val_payloads_opened": 0,
            "holdout_payloads_opened": 0,
            "sealed_payloads_opened": 0,
            "meta_test_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "S3_requests": 0,
        },
        "DOPE_sha256_before": dope_receipt["before"],
        "DOPE_sha256_after": dope_receipt["after"],
        "optimizer_updates": 0,
        "GPU_used": False,
        "qualification_lock_used": False,
        "FIFO_ticket_created": False,
    }
    headroom(root)
    if metrics["runtime"]["scratch_bytes"] >= 2_000_000_000:
        raise RuntimeError("H124 scratch exceeded 2 GB")
    metrics_path = root / "metrics.json"
    write(metrics_path, metrics)
    if metrics_path.stat().st_size >= 100_000:
        raise RuntimeError("H124 metrics.json exceeds 100 KB")
    print(
        "COMPLETE",
        metrics["verdict"],
        json.dumps(metrics["runtime"], sort_keys=True),
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--cpu-dry-run", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.cpu_dry_run:
        print(json.dumps(cpu_dry_run(), sort_keys=True, allow_nan=False))
        return
    if args.root is None:
        parser.error("--root is required with --run")
    result = run_h124(args.root)
    print(
        json.dumps(
            {"verdict": result["verdict"], "metrics": str(args.root / "metrics.json")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
