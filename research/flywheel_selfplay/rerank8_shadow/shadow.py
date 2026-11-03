"""Read-only, outcome-blind RERANK8_CR_VETO shadow inference; never trains or promotes."""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from features import (
    build_topk_feature_matrix_for_context,
    compute_sem24_numpy,
    select_from_topk_scores_with_veto,
)

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
H273 = "29fdbd8be15e2d2aa4ddf7b33e96f637f11dbd85"
MAX_BYTES = 99_000_000


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    used = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    if used + len(raw) >= MAX_BYTES:
        raise RuntimeError("shadow output cap reached; no deletion or writer action")
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def validate_dev_rows(rows):
    for r in rows:
        if r["dl_split"] != "DL-Val" or r["source_split"] not in ("train", "inner"):
            raise ValueError("shadow accepts approved DL-Val train/inner folds only")


def prepare(args):
    """Copy Dev-only features to a small packet, independently of model availability."""
    out = args.output.resolve()
    assert not out.is_relative_to(args.run.resolve())
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads(args.h273_manifest.read_text())["input_checksums_sha256"]
    pool_input = args.pool_input
    assert sha(pool_input / "selected.json") == pins["selected_json"]
    assert sha(args.cache / "context_features.npz") == pins["context_features_npz"]
    assert sha(args.cache / "pool_priors.npz") == pins["pool_priors_npz"]
    selected = json.loads((pool_input / "selected.json").read_text())
    live_selected = json.loads((args.live_input / "selected.json").read_text())
    dev = selected["dev"]
    assert len(dev) == 91 and len({r["family_id"] for r in dev}) == 19
    assert {r["context_sha256"] for r in dev} == {
        r["context_sha256"] for r in live_selected["dev"]
    }
    train_families = {r["family_id"] for r in selected["train"]}
    assert not train_families.intersection(r["family_id"] for r in dev)
    validate_dev_rows(dev)
    with np.load(args.costs, allow_pickle=False) as z:
        assert set(z.files) == {r["context_sha256"] for r in dev}
        costs = {k: z[k].copy() for k in z.files}
    with np.load(args.cache / "context_features.npz", allow_pickle=False) as z:
        cids = z["cids"].tolist()
        lm = z["lm48_z"]
        dope = z["dope629"]
        contexts = {
            r["context_sha256"]: (
                lm[cids.index(r["context_sha256"])].copy(),
                dope[cids.index(r["context_sha256"])].copy(),
            )
            for r in dev
        }
    with np.load(args.cache / "pool_priors.npz", allow_pickle=False) as z:
        cids = z["cids"].tolist()
        offsets = z["offsets"]
        prior = z["base_prior_cost"]
        proj = z["proj16"]
        priors = {}
        for r in dev:
            cid = r["context_sha256"]
            j = cids.index(cid)
            lo, hi = offsets[j : j + 2]
            priors[cid] = (prior[lo:hi].copy(), proj[lo:hi].copy())
    X, pids, u8, contexts_meta = [], [], [], []
    for r in dev:
        cid = r["context_sha256"]
        path = pool_input / "pools" / (cid + ".npz")
        assert sha(path) == r["pool_sha256"]
        live_path = args.live_input / "pools" / path.name
        with (
            np.load(path, allow_pickle=False) as z,
            np.load(live_path, allow_pickle=False) as live,
        ):
            # Same immutable Dev pool, not a substituted corpus or stale row order.
            for name in ["program", "program_sha256", "fitness", "gate"]:
                assert np.array_equal(z[name], live[name]), (cid, name)
            prog = z["program"].astype(np.float32)
            fitness = z["fitness"].astype(np.float64)
            gate = z["gate"].astype(bool)
            ids = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in z["program_sha256"]
            ]
        rc = costs[cid]
        assert len(rc) == len(prog) and np.isfinite(rc).all()
        indices, features = build_topk_feature_matrix_for_context(
            rc,
            priors[cid][0],
            prog,
            compute_sem24_numpy(prog),
            priors[cid][1],
            contexts[cid][0],
            contexts[cid][1],
        )
        assert features.shape == (8, 568) and np.isfinite(features).all()
        X.append(features)
        pids.append([ids[int(i)] for i in indices])
        u8.append(np.where(gate[indices], -fitness[indices], -2.0))
        contexts_meta.append(
            {"cid": cid, "family_id": r["family_id"], "pool_sha256": r["pool_sha256"]}
        )
    packet = out / "dev-packet.npz"
    np.savez_compressed(
        packet, X=np.asarray(X), pids=np.asarray(pids), utility=np.asarray(u8)
    )
    meta = {
        "schema": "jope.rerank8-shadow-packet/v1",
        "contexts": contexts_meta,
        "live_dev_order": [r["context_sha256"] for r in live_selected["dev"]],
        "training_family_ids": sorted(train_families),
        "DOPE_sha256": DOPE,
        "source_H273": H273,
        "packet_sha256": sha(packet),
        "outcome_blind_feature_builder": True,
        "live_Dev_pools_exactly_equal": True,
        "input_sha256": {
            str(p): sha(p)
            for p in [
                args.costs,
                args.h273_manifest,
                Path(__file__).with_name("features.py"),
            ]
        },
    }
    write(out / "packet.json", meta)
    return meta


def metrics(inc, shadow, families, seed=2030):
    inc, shadow = np.asarray(inc, float), np.asarray(shadow, float)
    assert (
        inc.shape == shadow.shape
        and np.isfinite(inc).all()
        and np.isfinite(shadow).all()
    )
    groups = [np.flatnonzero(np.asarray(families) == f) for f in sorted(set(families))]
    # H273 comparison: mean context gain per family, median across families.
    delta = np.asarray([np.mean(shadow[g] - inc[g]) for g in groups])
    wins = np.where(shadow > inc + 1e-9, 1.0, np.where(shadow < inc - 1e-9, 0.0, 0.5))
    credits = np.asarray([wins[g].mean() for g in groups])

    def elo(p):
        # Same 1e-3 symmetric pseudocount as H273's two-contender MM fit.
        p = (p * len(groups) + 0.001) / (len(groups) + 0.002)
        return 400 * np.log10(p / (1 - p))

    draws = np.random.default_rng(seed).integers(0, len(groups), (10000, len(groups)))
    return {
        "family_median_delta": float(np.median(delta)),
        "family_median_delta_ci95": np.quantile(
            np.median(delta[draws], axis=1), [0.025, 0.975]
        ).tolist(),
        "Elo_lift": float(elo(credits.mean())),
        "incumbent_Elo": 1500.0,
        "shadow_Elo": 1500.0 + float(elo(credits.mean())),
        "Elo_lift_ci95": np.quantile(
            elo(credits[draws].mean(axis=1)), [0.025, 0.975]
        ).tolist(),
        "contexts": len(inc),
        "families": len(groups),
        "bootstrap_seed": seed,
        "bootstrap_draws": 10000,
        "scope": "repeated development shadow; no independent test claim",
    }


def load_models(path, meta, fit_lock):
    if not path.exists():
        raise FileNotFoundError("H273_FROZEN_RANKER_AND_VETO_WEIGHTS_NOT_EXPORTED")
    m = json.loads(path.read_text())
    assert m["schema"] == "jope.h273.frozen-shadow-models/v1"
    assert m["H273_source_sha"] == H273 and m["arm"] == "RERANK8_CR_VETO"
    assert m["fit_split"] == "DL-Train" and m["dl_val_reads_before_lock"] == 0
    assert m["fit_selections_lock_sha256"] == sha(fit_lock)
    families = set(m["training_family_ids"])
    assert families and families <= set(meta["training_family_ids"])
    assert not families.intersection(r["family_id"] for r in meta["contexts"])
    expected = {
        (f, s, k)
        for f in [273140, 273141, 273142]
        for s in [273101, 273102, 273103]
        for k in range(5)
    }
    assert {
        (c["fold_seed"], c["seed"], c["assess_fold"]) for c in m["cells"]
    } == expected
    assert len(m["cells"]) == 45
    locks = {
        (r["fold_seed"], r["seed"], r["assess_fold"]): r
        for r in json.loads(fit_lock.read_text())["selections"]
        if r["arm"] == "RERANK8_CR_VETO"
    }
    import lightgbm as lgb

    models = []
    for c in m["cells"]:
        lock = locks[c["fold_seed"], c["seed"], c["assess_fold"]]
        for key in ["selected_n_estimators", "strict_margin", "selected_tau"]:
            assert c[key] == lock[key]
        assert c["selected_tau"] == 0.10
        boosters = []
        for role in ["ranker", "veto"]:
            rec = c[role]
            p = (path.parent / rec["file"]).resolve()
            assert p.is_relative_to(path.parent.resolve())
            assert not (path.parent / rec["file"]).is_symlink()
            assert rec["S3_VersionId"] and rec["S3_key"].startswith(
                "experiments/rd_breakthrough_20261030/H273/"
            )
            assert sha(p) == rec["sha256"]
            b = lgb.Booster(model_file=str(p))
            assert b.num_feature() == 568
            boosters.append(b)
        models.append((c, *boosters))
    return models


def readout(round_path, output, model_manifest, fit_lock):
    meta = json.loads((output / "packet.json").read_text())
    assert sha(output / "dev-packet.npz") == meta["packet_sha256"]
    row = json.loads(round_path.read_text())
    assert (
        row["dope_sha256"] == DOPE
        and row["dope_checksum_match"]
        and row["trainable_params"] == 29211
    )
    result = {
        "schema": "jope.rerank8-shadow-round/v1",
        "round": row["round"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_round_sha256": sha(round_path),
        "incumbent_best_round": row["adapter_admission"]["best_round"],
        "DOPE_sha256": DOPE,
        "mode": "READ_ONLY_SHADOW",
        "promotion_wiring": False,
        "fit_calls": 0,
        "weight_updates": 0,
    }
    try:
        models = load_models(model_manifest, meta, fit_lock)
    except FileNotFoundError as e:
        result.update(
            status="BLOCKED_MISSING_FROZEN_MODEL",
            reason=str(e),
            contexts=91,
            families=19,
            family_median_delta=None,
            Elo_lift=None,
            picks=None,
            scored_contexts=0,
        )
    else:
        with np.load(output / "dev-packet.npz", allow_pickle=False) as z:
            X = z["X"]
            ids = z["pids"]
            utilities = z["utility"]
        order = meta["live_dev_order"]
        values = row["dev_utility"]["per_context"]
        inc_ids = row["dev_utility"]["selected_programs"]
        assert len(order) == len(values) == len(inc_ids) == 91
        lookup = {c: (u, p) for c, u, p in zip(order, values, inc_ids)}
        inc = np.asarray([lookup[r["cid"]][0] for r in meta["contexts"]])
        fam = [r["family_id"] for r in meta["contexts"]]
        cells = []
        for c, ranker, veto in models:
            preds = ranker.predict(
                X.reshape(-1, 568),
                num_iteration=c["selected_n_estimators"],
                num_threads=4,
            ).reshape(91, 8)
            probs = veto.predict(X.reshape(-1, 568), num_threads=4).reshape(91, 8)
            assert (
                np.isfinite(preds).all()
                and np.isfinite(probs).all()
                and ((probs >= 0) & (probs <= 1)).all()
            )
            margin = (
                math.inf if c["strict_margin"] == "inf" else float(c["strict_margin"])
            )
            picks = np.asarray(
                [
                    select_from_topk_scores_with_veto(s, p, 0.10, margin)
                    for s, p in zip(preds, probs)
                ]
            )
            shadow = utilities[np.arange(91), picks]
            cells.append(
                {
                    "fold_seed": c["fold_seed"],
                    "seed": c["seed"],
                    "assess_fold": c["assess_fold"],
                    **metrics(inc, shadow, fam),
                    "picks": [
                        {
                            "cid": r["cid"],
                            "incumbent_pid": lookup[r["cid"]][1],
                            "shadow_pid": str(ids[i, picks[i]]),
                            "slot": int(picks[i]),
                        }
                        for i, r in enumerate(meta["contexts"])
                    ],
                }
            )
        # Every frozen cell is reported; no post-outcome winning-cell selection.
        result.update(
            status="SCORED",
            cells=cells,
            scored_contexts=91,
            aggregate_policy="all45 frozen H273 cells; no outcome-based selection",
        )
    write(output / f"round-{row['round']}.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    for x in ["pool-input", "live-input", "cache", "costs", "h273-manifest"]:
        a.add_argument("--" + x, type=Path, required=True)
    a = sub.add_parser("watch")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--models", type=Path, required=True)
    a.add_argument("--fit-lock", type=Path, required=True)
    a.add_argument("--writer-pid", type=int, required=True)
    a.add_argument("--first-round", type=int, required=True)
    a.add_argument("--max-rounds", type=int, default=40)
    a.add_argument("--hours", type=float, default=6)
    args = p.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args)))
        return
    assert 0 < args.max_rounds <= 40 and 0 < args.hours <= 6
    assert not args.output.resolve().is_relative_to(args.run.resolve())
    import psutil

    proc = psutil.Process(args.writer_pid)
    started = proc.create_time()
    end = time.time() + args.hours * 3600
    seen = set()
    while time.time() < end and len(seen) < args.max_rounds:
        own = psutil.Process()
        if own.num_threads() > 4 or own.memory_info().rss > 2 * 2**30:
            raise RuntimeError("shadow resource ceiling; no writer action")
        if (
            not proc.is_running()
            or proc.create_time() != started
            or proc.status() == psutil.STATUS_ZOMBIE
        ):
            break
        for path in sorted(
            (
                p
                for p in args.run.glob("round-*.json")
                if p.stem.removeprefix("round-").isdigit()
            ),
            key=lambda x: int(x.stem.split("-")[1]),
        ):
            n = int(path.stem.split("-")[1])
            if n < args.first_round or n in seen:
                continue
            r = readout(path, args.output, args.models, args.fit_lock)
            seen.add(n)
            print(json.dumps({k: v for k, v in r.items() if k != "cells"}), flush=True)
            if len(seen) >= args.max_rounds:
                break
        time.sleep(15)
    write(
        args.output / "WATCH_STOP.json",
        {
            "rounds": sorted(seen),
            "reason": "bound_or_writer_exit",
            "deadline_epoch": end,
        },
    )


if __name__ == "__main__":
    main()
