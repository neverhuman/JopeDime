"""H268: H265 feature/model/selector code, three frozen training-label arms."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import h265_frozen as h
import common_h252 as common

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = (268101, 268102, 268103)
FOLD_SEEDS = (268140, 268141, 268142)
BOOT_SEED = 268150
ARMS = ("CONTROL", "RERANK4", "CTXNORM", "CTXNORM_SHUF")
CUTS = np.array([-1.0, -0.25, 0.25, 1.0])


def ctxnorm_relevance(cost, gate):
    """Valid finite candidates define median/MAD; equality goes to upper grade."""
    cost = np.asarray(cost, dtype=np.float64)
    valid = np.asarray(gate, dtype=bool) & np.isfinite(cost)
    z = np.full(cost.shape, -2.0, dtype=np.float64)
    if valid.any():
        median = np.median(cost[valid])
        scale = max(1.4826 * np.median(np.abs(cost[valid] - median)), 1e-3)
        z[valid] = np.clip((median - cost[valid]) / scale, -2, 2)
    return np.searchsorted(CUTS, z, side="right").astype(np.int32)


def configure(out):
    h.INPUTS_DIR = ROOT / "inputs"
    h.R882_CKPT = h.INPUTS_DIR / "r882/round-882.pt"
    h.R882_BEST_DEV_JSON = h.INPUTS_DIR / "r882/best_dev_checkpoint.json"
    h.R882_TRAIN_COSTS_NPZ = h.INPUTS_DIR / "r882-train-costs.npz"
    h.R882_VAL_COSTS_NPZ = h.INPUTS_DIR / "r882-secondary-costs.npz"
    h.SNAPSHOT_DIR = HERE
    h.BOOT_SEED = BOOT_SEED
    os.environ["H265_PAUSE_FILE"] = str(out.with_suffix(".pause.json"))


def load_packets(rows, costs):
    ids = {r["context_sha256"] for r in rows}
    pri, proj, ctx = {}, {}, {}
    with np.load(h.POOL_PRIORS_NPZ, allow_pickle=False) as z:
        offsets, a, b = z["offsets"], z["base_prior_cost"], z["proj16"]
        for i, cid in enumerate(z["cids"].astype(str)):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                pri[cid], proj[cid] = a[lo:hi].copy(), b[lo:hi].copy()
    with np.load(h.CONTEXT_FEATURES_NPZ, allow_pickle=False) as z:
        lm, dope = z["lm48_z"], z["dope629"]
        for i, cid in enumerate(z["cids"].astype(str)):
            if cid in ids:
                ctx[cid] = dict(lm48_z=lm[i].copy(), dope629=dope[i].copy())
    packets = h.load_cohort_top4_packets(rows, costs, pri, proj, ctx)
    h.trim_malloc()
    return packets


def save_packets(path, rows, packets):
    cids = [r["context_sha256"] for r in rows]
    np.savez_compressed(
        path,
        cids=np.array(cids),
        **{
            k: np.stack([packets[c][k] for c in cids])
            for k in [
                "X_top4",
                "top4_indices",
                "fitness_4",
                "gate_4",
                "utility_4",
                "rel_4",
            ]
        },
    )


def fit_cell(out, key, seed, fold, shuf, folds, packets, fams):
    """Observe calls to unmodified H265 code; preserve complete selector replay evidence."""
    predictions, fits = [], []
    original_fit, original_predict = h.fit_lgbm_ranker_200, h.predict_lgbm_ranker

    def fit(X, y, n_groups, lr, random_state):
        h.check_cooperative_pause()
        fits.append(
            dict(
                X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                n_groups=n_groups,
                lr=lr,
                random_state=random_state,
            )
        )
        return original_fit(X, y, n_groups, lr, random_state)

    def predict(model, X, num_iteration):
        h.check_cooperative_pause()
        result = original_predict(model, X, num_iteration)
        predictions.append(result.copy())
        return result

    h.fit_lgbm_ranker_200, h.predict_lgbm_ranker = fit, predict
    try:
        # Identical seed namespace across arms isolates labels; original RERANK4 key retained.
        result = h.run_cell_fit_and_select(
            seed, fold, "RERANK4", shuf, folds, packets, fams
        )
    finally:
        h.fit_lgbm_ranker_200, h.predict_lgbm_ranker = original_fit, original_predict
    np.savez_compressed(
        out / "cells" / (key + ".npz"),
        **{f"prediction_{i:02d}": v for i, v in enumerate(predictions)},
    )
    result["receipt"].update(key=key, actual_arm=key.split("_")[-1], fit_calls=fits)
    return result


def contrasts(rows, panel):
    n = len({r["family_id"] for r in rows})
    boot = np.random.default_rng(BOOT_SEED).integers(0, n, size=(10000, n))
    result = {}
    for cand, base in [
        ("CTXNORM", "CONTROL"),
        ("CTXNORM", "RERANK4"),
        ("CTXNORM", "CTXNORM_SHUF"),
        ("RERANK4", "CONTROL"),
        ("CTXNORM_SHUF", "CONTROL"),
    ]:
        result[cand + " - " + base] = h.paired_family_contrast(
            rows, panel, list(ARMS), cand, base, boot
        )
    result["elo"] = h.compute_chunked_mm_bt_elo(
        common, rows, panel, list(ARMS), "CONTROL", boot
    )
    result["invalid_minus2_context_counts_by_seed"] = {
        a: [int(np.sum(panel[k, :, i] == -2.0)) for k in range(3)]
        for i, a in enumerate(ARMS)
    }
    return result


def apply_holm(primary, secondary):
    """H265 conservative simultaneous 97.5% intervals plus Holm adjusted one-sided p."""
    rows = [primary, secondary]
    order = sorted(range(2), key=lambda i: rows[i]["raw_p_one_sided"])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (2 - rank) * rows[i]["raw_p_one_sided"]))
        rows[i]["holm_p_one_sided"] = previous
        rows[i]["holm_lcb95"] = rows[i]["ci975_holm2"][0]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir()
    (out / "cells").mkdir()
    (out / "models").mkdir()
    start = time.time()
    configure(out)
    launch = json.loads(args.launch.read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert launch["source_commit"] == os.environ["H268_SOURCE"]
    for name, digest in manifest["source_sha256"].items():
        assert h.sha256_file(HERE / name) == digest, name
    for name, digest in manifest["input_sha256"].items():
        assert h.sha256_file(ROOT / name) == digest, name
    train, val = common.load_cohorts()
    # Verify every allowed pool's identity; no protected cohort is opened.
    for i, r in enumerate(train + val):
        h.check_cooperative_pause()
        assert (
            h.sha256_file(h.INPUT_POOLS_DIR / "pools" / (r["context_sha256"] + ".npz"))
            == r["pool_sha256"]
        )
    from baseline import native_regression

    costs, baseline = native_regression(out)
    h.fsync_write_json(
        out / "progress.json",
        dict(stage="Native91_PASS", matches=91, epoch=time.time()),
    )
    packets = load_packets(train, costs)
    save_packets(out / "train_packets.npz", train, packets)
    cids = [r["context_sha256"] for r in train]
    positions = {c: i for i, c in enumerate(cids)}
    fams = {r["context_sha256"]: r["family_id"] for r in train}
    reference = {c: packets[c]["rel_4"].copy() for c in cids}
    normalized = {
        c: ctxnorm_relevance(packets[c]["fitness_4"], packets[c]["gate_4"])
        for c in cids
    }
    folds_manifest = json.loads((HERE / "fold_manifest.json").read_text())
    panel = np.full((3, 3, len(train), 4), np.nan, dtype=np.float64)
    panel[:, :, :, 0] = np.array([packets[c]["r882_u"] for c in cids])
    selections = []
    for rep, fs in enumerate(FOLD_SEEDS):
        folds = {
            int(f["fold"]): f["context_ids"] for f in folds_manifest["repeats"][str(fs)]
        }
        for si, seed in enumerate(SEEDS):
            for fold in range(5):
                for ai, arm in enumerate(ARMS[1:], 1):
                    h.check_cooperative_pause()
                    for c in cids:
                        packets[c]["rel_4"] = (
                            reference if arm == "RERANK4" else normalized
                        )[c]
                    key = f"{fs}_{seed}_{fold}_{arm}"
                    result = fit_cell(
                        out,
                        key,
                        seed,
                        fold,
                        arm == "CTXNORM_SHUF",
                        folds,
                        packets,
                        fams,
                    )
                    receipt = result["receipt"]
                    receipt.update(actual_arm=arm, fold_seed=fs)
                    for c, pred in result["assess_preds"].items():
                        panel[rep, si, positions[c], ai] = packets[c]["utility_4"][
                            pred["j_strict"]
                        ]
                    lr, trees, margin = result["best_unc_cfg"]
                    # All selected outer models retained; only repeat 268140 used on DL-Val.
                    model_path = out / "models" / (key + ".txt")
                    result["outer_models_by_lr"][lr].booster_.save_model(
                        str(model_path)
                    )
                    receipt["model_sha256"] = h.sha256_file(model_path)
                    h.fsync_write_json(out / "cells" / (key + ".json"), receipt)
                    selections.append(receipt)
                    h.fsync_write_json(
                        out / "progress.json",
                        dict(
                            stage="FIT",
                            cells_complete=len(selections),
                            cells_total=135,
                            epoch=time.time(),
                            elapsed=time.time() - start,
                        ),
                    )
                    del result
                    h.trim_malloc()
    assert np.isfinite(panel).all() and len(selections) == 135
    lock = dict(
        source_commit=launch["source_commit"],
        selections=selections,
        trained_val_reads_before_lock=0,
        epoch=time.time(),
        cell_count=135,
    )
    h.fsync_write_json(out / "fit_selections_lock.json", lock)
    locksha = h.sha256_file(out / "fit_selections_lock.json")
    np.savez_compressed(
        out / "primary_panel.npz",
        utilities_by_repeat=panel,
        utilities=panel.mean(axis=0),
        cids=np.array(cids),
    )
    packets.clear()
    h.trim_malloc()
    vp = load_packets(val, costs)
    save_packets(out / "val_packets.npz", val, vp)
    vc = [r["context_sha256"] for r in val]
    X = np.concatenate([vp[c]["X_top4"] for c in vc])
    utilities = np.stack([vp[c]["utility_4"] for c in vc])
    vpanel = np.zeros((3, 5, len(val), 4), dtype=np.float64)
    vpanel[:, :, :, 0] = utilities[:, 0]
    vchoices = []
    for receipt in selections:
        if receipt["fold_seed"] != FOLD_SEEDS[0]:
            continue
        h.check_cooperative_pause()
        key = receipt["key"]
        model = lgb.Booster(
            model_file=str(out / "models" / (key + ".txt")), params={"num_threads": 1}
        )
        scores = model.predict(
            X, num_iteration=receipt["unconstrained_best_n_estimators"], num_threads=1
        ).reshape(len(vc), 4)
        margin = receipt["strict_selected_margin"]
        margin = math.inf if margin == "inf" else margin
        pick = np.array([h.select_from_top4_scores(sc, margin) for sc in scores])
        si = SEEDS.index(receipt["seed"])
        ai = ARMS.index(receipt["actual_arm"])
        fold = receipt["assess_fold"]
        vpanel[si, fold, :, ai] = utilities[np.arange(len(vc)), pick]
        np.savez_compressed(
            out / "cells" / (key + "-val.npz"), scores=scores, choices=pick
        )
        vchoices.append(dict(key=key, choices=pick.tolist()))
        del model
    np.savez_compressed(
        out / "secondary_panel.npz",
        utilities_by_fold=vpanel,
        utilities=vpanel.mean(axis=1),
        cids=np.array(vc),
    )
    h.fsync_write_json(
        out / "progress.json",
        dict(stage="INFERENCE", cells_complete=135, epoch=time.time()),
    )
    primary = contrasts(train, panel.mean(axis=0))
    secondary = contrasts(val, vpanel.mean(axis=1))
    # Full per-family/per-seed harms and raw invalid counts remain in durable outputs.
    harm_tables = {}
    for cohort, rows, values in [
        ("primary", train, panel.mean(axis=0)),
        ("secondary", val, vpanel.mean(axis=1)),
    ]:
        famnames = np.array([row["family_id"] for row in rows])
        harm_tables[cohort] = [
            dict(
                family_id=f,
                per_seed_delta=(
                    values[:, famnames == f, 2] - values[:, famnames == f, 0]
                )
                .mean(axis=1)
                .tolist(),
            )
            for f in sorted(set(famnames))
        ]
    h.fsync_write_json(out / "family-harm-table.json", harm_tables)
    raw_invalid = dict(
        primary={
            a: (panel[:, :, :, ai] == -2.0).sum(axis=2).tolist()
            for ai, a in enumerate(ARMS)
        },
        secondary={
            a: (vpanel[:, :, :, ai] == -2.0).sum(axis=2).tolist()
            for ai, a in enumerate(ARMS)
        },
    )
    h.fsync_write_json(out / "raw-invalid-counts.json", raw_invalid)
    pc, sc = apply_holm(primary["CTXNORM - CONTROL"], secondary["CTXNORM - CONTROL"])
    placebo = primary["CTXNORM - CTXNORM_SHUF"]
    gates = dict(
        primary_margin=pc["median_family_gain"] > 0.002,
        primary_holm_lcb=pc["holm_lcb95"] > 0,
        primary_three_positive=pc["positive_seeds"] == 3,
        placebo_positive=placebo["median_family_gain"] > 0 and placebo["ci95"][0] > 0,
        secondary_three_positive=sc["positive_seeds"] == 3,
    )
    assert h.sha256_file(h.R882_CKPT) == h.R882_EXPECTED_SHA256
    frozen_before = baseline["dope_sha256"]
    import torch

    ckpt = torch.load(h.R882_CKPT, map_location="cpu", weights_only=False)
    assert ckpt["dope_sha256"] == frozen_before == h.DOPE_EXPECTED_SHA256
    del ckpt
    full = dict(
        schema="jope.h268.metrics/v1",
        status="COMPLETE",
        source_commit=launch["source_commit"],
        verdict="ADVANCE" if all(gates.values()) else "NO_ADVANCE",
        gates=gates,
        baseline=baseline,
        primary=primary,
        secondary=secondary,
        selections=selections,
        fold_seeds=list(FOLD_SEEDS),
        model_seeds=list(SEEDS),
        neural_updates=0,
        cell_count=135,
        fit_selections_lock_sha256=locksha,
        admitted_cells={
            a: sum(r["actual_arm"] == a and r["strict_admitted"] for r in selections)
            for a in ARMS[1:]
        },
        input_checkpoint_unchanged=True,
        dope_unchanged=True,
        trained_val_reads_before_global_lock=0,
        elapsed_seconds=time.time() - start,
        estimand="Mean OOF utilities over three fold partitions; DL-Val mean utility over five models in foldseed268140; not a single deployable checkpoint",
        support_disclosure="No new reranker test-time reveals; frozen cached r882 scores retain native support conditioning",
        completed_epoch=time.time(),
    )
    # Never enforce compact packaging size here: preserve full outputs before S3 and packaging.
    h.fsync_write_json(out / "metrics-full.json", full)
    h.fsync_write_json(
        out / "progress.json",
        dict(stage="COMPLETE", cells_complete=135, epoch=time.time()),
    )


if __name__ == "__main__":
    main()
