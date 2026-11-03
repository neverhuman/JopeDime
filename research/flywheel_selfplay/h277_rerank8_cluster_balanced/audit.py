"""Saved-model/choice, FIT closure/weight, nested-grid and panel/statistic replay; no refitting."""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from dependencies import ROOT, clusters, core
import runner
import weighted
import analysis_h277 as stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    out = ap.parse_args().output
    m = json.loads((out / "metrics-full.json").read_text())
    lock = json.loads((out / "fit_selections_lock.json").read_text())
    sel = json.loads((ROOT / "inputs/selected.json").read_text())
    rows = sel["train"]
    val = sel["dev"]
    parts = clusters.partitions_from_manifest(
        json.loads((ROOT / "inputs/fold_manifest.json").read_text())
    )
    with np.load(out / "train_packets.npz", allow_pickle=False) as z:
        ids = z["cids"].astype(str).tolist()
        X = z["X"]
        u = z["utility_k"]
        labels = z["rel_k"]
        gates = z["gate_k"]
    pos = {c: i for i, c in enumerate(ids)}
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    panel = np.full((3, 4, 1219, 5), np.nan)
    panel[:, :, :, 0] = u[:, 0]
    fits = selectors = models = 0
    for pinned in lock["cells"]:
        file = out / "cells" / (pinned["key"] + ".json")
        assert runner.sha(file) == pinned["sha256"]
        cell = json.loads(file.read_text())
        name, seed, held = cell["partition"], cell["seed"], cell["held"]
        folds = {int(k): v for k, v in cell["folds"].items()}
        assert folds == clusters.cluster_out_folds(rows, parts[name], held, seed)
        cl = {f: i for i, g in enumerate(parts[name]) for f in g}
        salt = f"{name}-{held}"
        assess = cell["assess_ids"]
        assert assess == folds[0]
        for evname, placebo in [
            ("weighted_evidence", False),
            ("placebo_evidence", True),
        ]:
            ev = cell[evname]
            fitids = ev["fitids"]
            assert fitids == [c for f in [1, 2, 3, 4] for c in folds[f]]
            for call in ev["fit_calls"]:
                cids = call["cids"]
                stage = call["stage"]
                tr = [
                    c
                    for f in [1, 2, 3, 4]
                    if stage == "outer" or f != int(stage[-1])
                    for c in folds[f]
                ]
                assert cids == tr and not {fam[c] for c in cids} & {
                    fam[c] for c in assess
                }
                families = sorted({fam[c] for c in cids})
                groups = sorted({cl[f] for f in families})
                sizes = {g: sum(cl[f] == g for f in families) for g in groups}
                iv = np.array([1 / sizes[g] for g in groups])
                perm = (
                    np.random.default_rng(
                        core.stream_seed(seed, "weights", stage, salt)
                    ).permutation(len(groups))
                    if placebo
                    else np.arange(len(groups))
                )
                by = {g: iv[j] for g, j in zip(groups, perm)}
                counts = {f: sum(fam[c] == f for c in cids) for f in families}
                w = np.array([by[cl[fam[c]]] / counts[fam[c]] for c in cids])
                w /= w.mean()
                w = np.repeat(w, 8)
                assert (
                    call["X_sha256"]
                    == hashlib.sha256(
                        np.concatenate([X[pos[c]] for c in cids]).tobytes()
                    ).hexdigest()
                )
                assert (
                    call["y_sha256"]
                    == hashlib.sha256(
                        np.concatenate([labels[pos[c]] for c in cids]).tobytes()
                    ).hexdigest()
                )
                assert call["weight_sha256"] == hashlib.sha256(w.tobytes()).hexdigest()
                fits += 1
            for gatecall in ev["gate_calls"]:
                tr = gatecall["cids"]
                assert not {fam[c] for c in tr} & {
                    fam[c] for c in folds[0] + folds[int(gatecall["stage"][-1])]
                }
                assert (
                    gatecall["gate_y_sha256"]
                    == hashlib.sha256(
                        np.concatenate([~gates[pos[c]] for c in tr]).tobytes()
                    ).hexdigest()
                )
            with np.load(
                out / "cells" / (cell["key"] + "-inner.npz"), allow_pickle=False
            ) as z:
                prefix = "placebo_" if placebo else ""
                scores = {
                    (lr, t): z[f"{prefix}score_{lr}_{t}"]
                    for lr in core.LR_GRID
                    for t in core.N_EST_GRID
                }
                probs = None if placebo else z["invalid_oof"]
                for arm in ["PLACEBO"] if placebo else ["A", "B"]:
                    # Recompute the declared entire finite grid from stored inner-OOF predictions, no model fitting.
                    cfg = weighted.choose(
                        scores,
                        np.stack([u[pos[c]] for c in fitids]),
                        [fam[c] for c in fitids],
                        np.array(ev["selector_weights"]),
                        core.stream_seed(
                            stat.BOOT, "inner_select", seed, 0, "RERANK8", salt
                        ),
                        probs if arm == "B" else None,
                    )
                    assert cfg == cell["policies"][arm]["config"]
                    selectors += 1
        for arm, rec in cell["policies"].items():
            model = out / "models" / rec["model"]
            assert runner.sha(model) == rec["model_sha256"]
            cfg = rec["config"]
            sc = (
                lgb.Booster(model_file=str(model), params={"num_threads": 1})
                .predict(
                    X[[pos[c] for c in assess]].reshape(-1, 568),
                    num_iteration=cfg["n_estimators"],
                    num_threads=1,
                )
                .reshape(-1, 8)
            )
            j = np.argmax(sc, axis=1)
            margin = np.inf if cfg["strict_margin"] == "inf" else cfg["strict_margin"]
            j = np.where(sc[np.arange(len(j)), j] - sc[:, 0] > margin, j, 0)
            if rec["gate"]:
                path = out / "models" / rec["gate"]
                assert runner.sha(path) == rec["gate_sha256"]
                pv = (
                    lgb.Booster(model_file=str(path), params={"num_threads": 1})
                    .predict(
                        X[[pos[c] for c in assess]].reshape(-1, 568), num_threads=1
                    )
                    .reshape(-1, 8)
                )
                j = np.where((j != 0) & (pv[np.arange(len(j)), j] > cfg["tau"]), 0, j)
            np.testing.assert_array_equal(j, rec["picks"])
            ix = [pos[c] for c in assess]
            panel[
                stat.SEEDS.index(seed),
                stat.PARTITIONS.index(name),
                ix,
                stat.ARMS.index(arm),
            ] = u[ix, j]
            models += 1
    with np.load(out / "primary_panel.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(panel, z["utilities"])
    with np.load(out / "val_packets.npz", allow_pickle=False) as z:
        VX = z["X"]
        VU = z["utility_k"]
    vp = np.full((3, 5, 91, 5), np.nan)
    vp[:, :, :, 0] = VU[:, 0]
    for rec in json.loads((out / "val-picks.json").read_text()):
        obj = dict(
            receipt=rec["receipt"],
            model=(out / "models" / rec["model"]).read_text(),
            gate=(out / "models" / rec["gate"]).read_text() if rec["gate"] else None,
        )
        j = runner.score(obj, VX.reshape(-1, 568))
        np.testing.assert_array_equal(j, rec["picks"])
        vp[
            stat.SEEDS.index(rec["seed"]), rec["held"], :, stat.ARMS.index(rec["arm"])
        ] = VU[np.arange(91), j]
        models += 1
    with np.load(out / "secondary_panel.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(vp, z["utilities"])
    prim, sec, raw = stat.summarize(rows, panel, val, vp, parts)
    assert (
        prim == m["primary"]
        and sec == m["secondary"]
        and raw == m["raw_invalid_counts"]
    )
    assert stat.decide(prim, sec, raw) == m["decision"]
    assert (
        runner.sha(out / "fit_selections_lock.json") == m["fit_selections_lock_sha256"]
    )
    assert lock["count"] == 60 and lock["trained_val_reads_before_lock"] == 0
    runner.save(
        out / "audit.json",
        dict(
            status="PASS",
            source_commit=m["source_commit"],
            fit_weight_input_receipts=fits,
            weighted_selector_grids=selectors,
            outer_and_val_model_replays=models,
            panels_exact=True,
            statistics_recomputed=True,
            statistics_independent_implementation=False,
            neural_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
