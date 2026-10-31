#!/usr/bin/env python3
"""H119: fixed H85 trees, fresh family cross-fitting, CPU r882 and paired readout."""

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
SEEDS = (51, 52, 53)
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
        for x in np.array_split(np.random.default_rng(11900).permutation(families), 5)
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


def family_vectors(rows, packets):
    cids = [r["context_sha256"] for r in rows]
    families = sorted({r["family_id"] for r in rows})
    if len(set(cids)) != len(cids) or set(packets) != set(map(str, SEEDS)):
        raise ValueError("complete unique three-seed panel required")
    result = {}
    for endpoint in ("u4", "legacy"):
        seed_vectors = []
        for seed in SEEDS:
            cells = packets[str(seed)]
            if set(cells) != set(cids):
                raise ValueError("full context panel required")
            by_family = defaultdict(list)
            for row in rows:
                item = cells[row["context_sha256"]]
                if not all(math.isfinite(item[a][endpoint]) for a in ("GBM", "r882")):
                    raise ValueError("nonfinite utility")
                by_family[row["family_id"]].append(
                    item["GBM"][endpoint] - item["r882"][endpoint]
                )
            seed_vectors.append([float(np.mean(by_family[f])) for f in families])
        result[endpoint] = np.array(seed_vectors)
    return families, result


def readout(rows, packets):
    families, delta = family_vectors(rows, packets)
    ix = np.random.default_rng(119000).integers(
        0, len(families), (10000, len(families))
    )
    boots = {k: np.median(v[:, ix].mean(axis=2), axis=0) for k, v in delta.items()}
    margins = {"u4": 0.001, "legacy": -0.01}
    tails = {k: float((1 + np.sum(v <= margins[k])) / 10001) for k, v in boots.items()}
    order = sorted(boots, key=lambda k: (tails[k], ("u4", "legacy").index(k)))
    continuing, adjusted = True, 0.0
    stats = {}
    for rank, k in enumerate(order):
        alpha = 0.05 / (2 - rank)
        bound = float(np.quantile(boots[k], alpha))
        adjusted = max(adjusted, min(1.0, tails[k] * (2 - rank)))
        continuing = bool(
            continuing and tails[k] <= alpha and bound > margins[k] + 1e-12
        )
        d = delta[k]
        harmed = d.mean(axis=0) < -0.01
        stats[k] = {
            "per_seed_family_mean_delta": d.mean(axis=1).tolist(),
            "median_across_seed_means": float(np.median(d.mean(axis=1))),
            "family_mean_delta": float(d.mean()),
            "median_family_delta": float(np.median(d.mean(axis=0))),
            "ci95": np.quantile(boots[k], [0.025, 0.975]).tolist(),
            "lcb95": float(np.quantile(boots[k], 0.05)),
            "holm_lcb": bound,
            "holm_alpha": alpha,
            "holm_rank": rank + 1,
            "holm_tail_area": adjusted,
            "raw_tail_area": tails[k],
            "holm_pass": continuing,
            "margin": margins[k],
            "harm_count": int(harmed.sum()),
            "family_count": len(families),
            "harm_fraction": float(harmed.mean()),
            "per_seed_harm_fraction": (d < -0.01).mean(axis=1).tolist(),
            "any_negative_family_fraction": float((d.mean(axis=0) < 0).mean()),
            "family_wtl": [
                int((d.mean(axis=0) > 1e-12).sum()),
                int((np.abs(d.mean(axis=0)) <= 1e-12).sum()),
                int((d.mean(axis=0) < -1e-12).sum()),
            ],
            "identical_seed_delta_groups": equality_groups(d),
        }
    for endpoint, stat in stats.items():
        stat["per_seed_contender_family_means"] = {}
        for name in ("GBM", "r882"):
            values = []
            for seed in SEEDS:
                by_family = defaultdict(list)
                for row in rows:
                    by_family[row["family_id"]].append(
                        packets[str(seed)][row["context_sha256"]][name][endpoint]
                    )
                values.append(float(np.mean([np.mean(x) for x in by_family.values()])))
            stat["per_seed_contender_family_means"][name] = values
        stat["seed_context_utility_groups"] = equality_groups(
            np.array(
                [
                    [
                        packets[str(s)][r["context_sha256"]]["GBM"][endpoint]
                        for r in rows
                    ]
                    for s in SEEDS
                ]
            )
        )
    admitted = bool(
        stats["u4"]["median_across_seed_means"] > 1e-12
        and all(stats[k]["holm_pass"] for k in stats)
        and stats["u4"]["harm_fraction"] <= 0.1
    )
    return {
        "endpoints": stats,
        "admitted": admitted,
        "families": len(families),
        "contexts": len(rows),
        "bootstrap_draws": 10000,
        "bootstrap_seed": 119000,
        "complete": True,
    }


def equality_groups(array):
    groups = []
    for i, seed in enumerate(SEEDS):
        for group in groups:
            if np.array_equal(array[i], array[SEEDS.index(group[0])]):
                group.append(seed)
                break
        else:
            groups.append([seed])
    return groups


def elo(rows, packets, endpoint, rng_seed, draws=1000):
    """Canonical two-contender family weighting and MM, including its .01 prior."""
    helper = source_module("python/src/jope/north_star/elo_tracker.py", "h119_elo")
    by_fam = defaultdict(list)
    deltas = []
    records = []
    for row in rows:
        cid, fam = row["context_sha256"], row["family_id"]
        scores = {
            name: float(np.mean([packets[str(s)][cid][key][endpoint] for s in SEEDS]))
            for name, key in (("GBM_PROG412_CTX", "GBM"), ("R882_CONTINUITY", "r882"))
        }
        records.append(
            helper.DatasetEvalRecord(
                dataset_id=cid,
                family_id=fam,
                task_type="regression",
                scores_by_contender=scores,
            )
        )
        d = scores["GBM_PROG412_CTX"] - scores["R882_CONTINUITY"]
        by_fam[fam].append(1.0 if d > 1e-4 else 0.0 if d < -1e-4 else 0.5)
        deltas.append(d)
    result = helper.compute_family_grouped_bradley_terry_elo(
        records, "R882_CONTINUITY", prior=0.01
    )
    result["per_seed_ratings"] = {}
    for seed in SEEDS:
        seed_records = [
            helper.DatasetEvalRecord(
                dataset_id=r["context_sha256"],
                family_id=r["family_id"],
                task_type="regression",
                scores_by_contender={
                    "GBM_PROG412_CTX": packets[str(seed)][r["context_sha256"]]["GBM"][
                        endpoint
                    ],
                    "R882_CONTINUITY": packets[str(seed)][r["context_sha256"]]["r882"][
                        endpoint
                    ],
                },
            )
            for r in rows
        ]
        result["per_seed_ratings"][str(seed)] = (
            helper.compute_family_grouped_bradley_terry_elo(
                seed_records, "R882_CONTINUITY", prior=0.01
            )["ratings"]
        )
    # Exactly the same two-player MM solver, vectorized over bootstrap draws.
    weights = np.array([np.mean(by_fam[f]) for f in sorted(by_fam)])
    ix = np.random.default_rng(rng_seed).integers(
        0, len(weights), (draws, len(weights))
    )
    wins = weights[ix].sum(axis=1) + 0.01
    losses = len(weights) + 0.02 - wins
    gamma = np.ones((draws, 2))
    for _ in range(400):
        denom = (len(weights) + 0.02) / gamma.sum(axis=1)
        updated = np.stack([wins / denom, losses / denom], axis=1).clip(1e-6, 1e6)
        gamma = updated / updated[:, 1:2]
    ratings = (1500 + 400 * np.log10(gamma[:, 0].clip(1e-9))).round(2)
    result["elo_ci95"] = {
        "GBM_PROG412_CTX": np.quantile(ratings, [0.025, 0.975]).round(2).tolist(),
        "R882_CONTINUITY": [1500.0, 1500.0],
    }
    result["bootstrap_draws"] = draws
    result["bootstrap_seed"] = rng_seed
    return result


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
        or any(len(x) != 41 for x in groups)
        or len(selected["dev"]) != 91
    ):
        raise RuntimeError("full roster required")
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
    rng = np.random.default_rng(700)
    samples = {
        r["context_sha256"]: 4
        + rng.choice(len(pools[r["context_sha256"]]["u"]) - 4, 192, replace=False)
        for r in rows
    }
    write(
        root / "split.json",
        {
            "seed": 11900,
            "groups": groups,
            "fold_by_cid": mapping,
            "rows": rows,
            "sampling_sha256": hashlib.sha256(
                b"".join(
                    samples[r["context_sha256"]].astype("<i8").tobytes() for r in rows
                )
            ).hexdigest(),
        },
    )
    return selected, context, pools, samples, mapping


def load_pools(rows):
    pools = {}
    for row in rows:
        if (
            row.get("dl_split") not in ("DL-Train", "DL-Val")
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


def tree_packets(root, rows, pools, baseline, *, secondary=False):
    if secondary:
        raise RuntimeError("secondary uses explicit full-fit path")
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
                        packets[str(seed)][cid] = {
                            "GBM": endpoints(
                                (a - b * z[cid]).astype(np.float32), pools[cid]["u"]
                            ),
                            "r882": endpoints(baseline[cid], pools[cid]["u"]),
                        }
    return packets, calibration_receipts


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = args.root
    if (
        root.exists()
        or root.is_symlink()
        or root.parent.resolve() != (Path.home() / "jope-data/h119").resolve()
    ):
        raise RuntimeError("fresh owned H119 root required")
    root.mkdir(parents=True)
    for filename in ("runner.py", "PREREG.md"):
        source = subprocess.check_output(
            [
                "git",
                "show",
                args.source_commit
                + ":research/flywheel_selfplay/h119_gbm_ctx_confirm/"
                + filename,
            ],
            cwd=REPO,
        )
        if source != (HERE / filename).read_bytes():
            raise RuntimeError("execution bytes do not match frozen source commit")
    write(
        root / "execution-identity.json",
        {
            "source_commit": args.source_commit,
            "runner_sha256": sha(HERE / "runner.py"),
            "prereg_sha256": sha(HERE / "PREREG.md"),
            "command": sys.argv,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "environment": {
                k: os.environ.get(k)
                for k in [
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                ]
            },
        },
    )
    t0, cpu0 = time.monotonic(), time.process_time()
    headroom(root)
    selected, context, pools, samples, mapping = load_data(root)
    rows = selected["train"]
    r882, net = baseline(root, rows, context, pools)
    tasks = [(s, (k,)) for s in SEEDS for k in range(5)] + [
        (s, pair) for s in SEEDS for pair in itertools.combinations(range(5), 2)
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        receipts = list(
            executor.map(
                lambda task: fit_one(
                    task, root, rows, context, pools, samples, mapping
                ),
                tasks,
            )
        )
    packets, cal = tree_packets(root, rows, pools, r882)
    primary = readout(rows, packets)
    primary["elo"] = {
        k: elo(rows, packets, k, 119001 + i) for i, k in enumerate(("u4", "legacy"))
    }
    write(root / "primary-vectors.json", {"rows": rows, "packets": packets})
    write(root / "primary-metrics.json", primary)
    # Explicit sealing boundary before secondary labels/payloads.
    write(
        root / "primary-sealed.json",
        {
            "vectors_sha256": sha(root / "primary-vectors.json"),
            "metrics_sha256": sha(root / "primary-metrics.json"),
            "DL_Val_payloads_opened": 0,
        },
    )
    secondary_rows = selected["dev"]
    secondary_pools = load_pools(secondary_rows)
    secondary_reference, _ = baseline(
        root, secondary_rows, context, secondary_pools, net=net
    )
    secondary_packets, full_receipts = secondary_fit(
        root,
        rows,
        secondary_rows,
        context,
        pools,
        secondary_pools,
        samples,
        mapping,
        secondary_reference,
    )
    write(
        root / "secondary-vectors.json",
        {"rows": secondary_rows, "packets": secondary_packets},
    )
    secondary = readout(secondary_rows, secondary_packets)
    secondary["admission_assessed"] = False
    secondary.pop("admitted")
    secondary["elo"] = {
        k: elo(secondary_rows, secondary_packets, k, 119001 + i)
        for i, k in enumerate(("u4", "legacy"))
    }
    import sklearn
    import torch

    if (
        torch.cuda.is_initialized()
        or sha(HERE / "runner.py")
        != json.loads((root / "execution-identity.json").read_text())["runner_sha256"]
    ):
        raise RuntimeError("CUDA initialized or executing source changed")
    metrics = {
        "schema": "jope.h119.confirmation/v1",
        "verdict": "CONFIRMED_CONDITIONAL_DEVELOPMENT"
        if primary["admitted"]
        else "NO_ADVANCE",
        "primary": primary,
        "secondary_report_only": secondary,
        "fit_receipts": receipts + full_receipts,
        "calibration": cal,
        "runtime": {
            "wall_seconds": time.monotonic() - t0,
            "process_cpu_seconds": time.process_time() - cpu0,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "inputs": {str(p): h for p, h in FILES.items()},
        "source_sha256": sha(HERE / "runner.py"),
        "source_commit": args.source_commit,
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "prereg_sha256": sha(HERE / "PREREG.md"),
        "GPU_used": False,
        "qualification_lock_used": False,
    }
    write(root / "metrics.json", metrics)
    print("COMPLETE", metrics["verdict"], metrics["runtime"], flush=True)


def secondary_fit(
    root, train_rows, dev_rows, context, pools, dev_pools, samples, mapping, reference
):
    from sklearn.ensemble import HistGradientBoostingRegressor

    packets, receipts = {}, []
    ordered = [
        r for k in range(5) for r in train_rows if mapping[r["context_sha256"]] == k
    ]
    x = np.concatenate(
        [
            np.concatenate(
                [
                    pools[r["context_sha256"]]["program"][samples[r["context_sha256"]]],
                    np.repeat(
                        np.concatenate(
                            [
                                context[r["context_sha256"]]["lm48_z"],
                                context[r["context_sha256"]]["dope629"],
                            ]
                        )[None],
                        192,
                        axis=0,
                    ),
                ],
                axis=1,
            )
            for r in ordered
        ]
    )
    y = np.concatenate(
        [
            (
                pools[r["context_sha256"]]["u"][samples[r["context_sha256"]]]
                - pools[r["context_sha256"]]["u"].mean()
            ).astype(np.float32)
            for r in ordered
        ]
    )
    for seed in SEEDS:
        headroom(root)
        t0 = time.monotonic()
        model = HistGradientBoostingRegressor(**MODEL, random_state=seed).fit(x, y)
        # Global OOF calibration uses training outcomes only.
        pp, ff, gg = [], [], []
        for k in range(5):
            with np.load(root / f"seed{seed}-exclude{k}.npz", allow_pickle=False) as z:
                for row in train_rows:
                    cid = row["context_sha256"]
                    if mapping[cid] == k:
                        pp.append(z[cid][4:])
                        ff.append(pools[cid]["fitness"][4:])
                        gg.append(pools[cid]["gate"][4:])
        a, b = calibration(np.concatenate(pp), np.concatenate(ff), np.concatenate(gg))
        packets[str(seed)] = {}
        predictions = {}
        for row in dev_rows:
            cid = row["context_sha256"]
            xp = dev_pools[cid]["program"]
            cf = np.concatenate([context[cid]["lm48_z"], context[cid]["dope629"]])
            pred = model.predict(
                np.concatenate([xp, np.repeat(cf[None], len(xp), axis=0)], axis=1)
            ).astype(np.float32)
            predictions[cid] = pred
            packets[str(seed)][cid] = {
                "GBM": endpoints(
                    (a - b * pred).astype(np.float32), dev_pools[cid]["u"]
                ),
                "r882": endpoints(reference[cid], dev_pools[cid]["u"]),
            }
        out = root / f"secondary-seed{seed}.npz"
        np.savez_compressed(out, **predictions)
        receipts.append(
            {
                "seed": seed,
                "secondary_only": True,
                "fit_contexts": len(train_rows),
                "predicted_contexts": len(dev_rows),
                "features": 1089,
                "iterations": int(model.n_iter_),
                "model_parameters": model.get_params(),
                "calibration": [a, b],
                "prediction_sha256": sha(out),
                "wall_seconds": time.monotonic() - t0,
            }
        )
    return packets, receipts


if __name__ == "__main__":
    main()
