"""H271: fresh-seed H268 confirmation with unconditional diagnostic panels."""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
from dependencies import h, common, prev, CLOSURE, ROOT as CUSTODY_ROOT

HERE = Path(__file__).resolve().parent
ROOT = CUSTODY_ROOT
SEEDS = (271101, 271102, 271103)
FOLD_SEEDS = (271140, 271141, 271142)
BOOT_SEED = 271150
FIT_ARMS = ("RERANK4", "CTXNORM", "CTXNORM_SHUF")
ARMS = ("CONTROL",) + FIT_ARMS + tuple(a + "_UNC" for a in FIT_ARMS)


ctxnorm_relevance = prev.ctxnorm_relevance


def configure(out):
    prev.ROOT = ROOT
    prev.HERE = CLOSURE
    prev.BOOT_SEED = BOOT_SEED
    prev.SEEDS = SEEDS
    prev.FOLD_SEEDS = FOLD_SEEDS
    prev.ARMS = ARMS
    prev.configure(out)


load_packets = prev.load_packets


save_packets = prev.save_packets


fit_cell = prev.fit_cell


def contrasts(rows, panel):
    result = prev.contrasts(rows, panel)
    n = len({row["family_id"] for row in rows})
    boot = np.random.default_rng(BOOT_SEED).integers(0, n, size=(10000, n))
    for cand, base in [
        ("RERANK4_UNC", "CONTROL"),
        ("CTXNORM_UNC", "CONTROL"),
        ("CTXNORM_SHUF_UNC", "CONTROL"),
        ("CTXNORM_UNC", "CTXNORM_SHUF_UNC"),
        ("CTXNORM_UNC", "RERANK4_UNC"),
    ]:
        result[cand + " - " + base] = h.paired_family_contrast(
            rows, panel, list(ARMS), cand, base, boot
        )
    return result


apply_holm = prev.apply_holm


def confirmation_gates(primary, secondary):
    pc = primary["CTXNORM - CONTROL"]
    sc = secondary["CTXNORM - CONTROL"]
    placebo = primary["CTXNORM - CTXNORM_SHUF"]
    unc_placebo = primary["CTXNORM_UNC - CTXNORM_SHUF_UNC"]
    return dict(
        primary_margin=pc["median_family_gain"] > 0.002,
        primary_holm_lcb=pc["holm_lcb95"] > 0,
        primary_three_positive=pc["positive_seeds"] == 3,
        placebo_positive=placebo["median_family_gain"] > 0 and placebo["ci95"][0] > 0,
        secondary_three_positive=sc["positive_seeds"] == 3,
        unconditional_placebo_positive=unc_placebo["median_family_gain"] > 0,
        unconditional_placebo_three_positive=unc_placebo["positive_seeds"] == 3,
    )


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
    assert launch["source_commit"] == os.environ["H271_SOURCE"]
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
    folds_manifest = json.loads((ROOT / "inputs/fold_manifest.json").read_text())
    panel = np.full((3, 3, len(train), len(ARMS)), np.nan, dtype=np.float64)
    panel[:, :, :, 0] = np.array([packets[c]["r882_u"] for c in cids])
    selections = []
    for rep, fs in enumerate(FOLD_SEEDS):
        folds = {
            int(f["fold"]): f["context_ids"] for f in folds_manifest["repeats"][str(fs)]
        }
        for si, seed in enumerate(SEEDS):
            for fold in range(5):
                for ai, arm in enumerate(FIT_ARMS, 1):
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
                        panel[rep, si, positions[c], ARMS.index(arm + "_UNC")] = (
                            packets[c]["utility_4"][pred["j_unc"]]
                        )
                    lr, trees, margin = result["best_unc_cfg"]
                    # All selected outer models retained; only repeat 271140 used on DL-Val.
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
    vpanel = np.full((3, 5, len(val), len(ARMS)), np.nan, dtype=np.float64)
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
        unc_pick = np.array(
            [
                h.select_from_top4_scores(sc, receipt["unconstrained_best_margin"])
                for sc in scores
            ]
        )
        vpanel[si, fold, :, ARMS.index(receipt["actual_arm"] + "_UNC")] = utilities[
            np.arange(len(vc)), unc_pick
        ]
        np.savez_compressed(
            out / "cells" / (key + "-val.npz"),
            scores=scores,
            choices=pick,
            unconditional_choices=unc_pick,
        )
        vchoices.append(dict(key=key, choices=pick.tolist()))
        del model
    assert np.isfinite(vpanel).all()
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
    apply_holm(primary["CTXNORM - CONTROL"], secondary["CTXNORM - CONTROL"])
    gates = confirmation_gates(primary, secondary)
    assert h.sha256_file(h.R882_CKPT) == h.R882_EXPECTED_SHA256
    frozen_before = baseline["dope_sha256"]
    import torch

    ckpt = torch.load(h.R882_CKPT, map_location="cpu", weights_only=False)
    assert ckpt["dope_sha256"] == frozen_before == h.DOPE_EXPECTED_SHA256
    del ckpt
    full = dict(
        schema="jope.h271.metrics/v1",
        status="COMPLETE",
        source_commit=launch["source_commit"],
        verdict="ADVANCE" if all(gates.values()) else "NO_ADVANCE",
        confirmation_status="CONFIRMED" if all(gates.values()) else "NOT_CONFIRMED",
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
            for a in FIT_ARMS
        },
        input_checkpoint_unchanged=True,
        dope_unchanged=True,
        trained_val_reads_before_global_lock=0,
        elapsed_seconds=time.time() - start,
        estimand="Mean OOF utilities over three fold partitions; DL-Val mean utility over five models in foldseed271140; not a single deployable checkpoint",
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
