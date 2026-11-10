"""Replay saved H274 selections, model scores and utilities independently of training."""

from pathlib import Path
import argparse
import json
import math
import hashlib
import numpy as np
import lightgbm as lgb
from dependencies import h
import runner as r


def replay_selector(scores, utilities, family_ids, seed, fold):
    unique = sorted(set(family_ids))
    masks = [np.asarray(family_ids) == f for f in unique]
    baseline = np.array([utilities[m, 0].mean() for m in masks])
    best = -1e18
    cfg = (0.03, 50, 0.0)
    for lr in h.LR_GRID:
        for trees in h.N_EST_GRID:
            sc = scores[(lr, trees)]
            j = np.argmax(sc, axis=1)
            gap = sc[np.arange(len(sc)), j] - sc[:, 0]
            for margin in h.MARGIN_GRID[:-1]:
                ix = np.where(gap > margin, j, 0)
                u = utilities[np.arange(len(sc)), ix]
                delta = np.array([u[m].mean() for m in masks]) - baseline
                gain = float(delta.mean())
                if gain > best + 1e-12 or (
                    abs(gain - best) <= 1e-12
                    and (margin, -trees, -lr) > (cfg[2], -cfg[1], -cfg[0])
                ):
                    best = gain
                    cfg = (lr, trees, margin)
                    bestdelta = delta
    boot = np.random.default_rng(
        h.stream_seed(r.BOOT_SEED, "inner_select", seed, fold, "RERANK4")
    ).integers(0, len(unique), (2000, len(unique)))
    ci = np.quantile(bestdelta[boot].mean(axis=1), [0.025, 0.975])
    admitted = best > 0.001 + 1e-12 and ci[0] > 0
    return cfg, best, ci, bool(admitted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    metrics = json.loads((out / "metrics-full.json").read_text())
    lock = json.loads((out / "fit_selections_lock.json").read_text())
    folds = json.loads((r.ROOT / "inputs/fold_manifest.json").read_text())["repeats"]
    selected = json.loads((r.ROOT / "inputs/selected.json").read_text())
    train, val = selected["train"], selected["dev"]
    fams = {x["context_sha256"]: x["family_id"] for x in train}
    source = json.loads((r.HERE / "manifest.json").read_text())
    for name, digest in source["source_sha256"].items():
        assert h.sha256_file(r.HERE / name) == digest
    for name, digest in source["input_sha256"].items():
        assert h.sha256_file(r.ROOT / name) == digest
    with np.load(out / "train_packets.npz", allow_pickle=False) as z:
        ids = z["cids"].astype(str).tolist()
        X = z["X_top4"]
        cost = z["fitness_4"]
        gate = z["gate_4"]
        u = z["utility_4"]
        rel = z["rel_4"]
    pos = {c: i for i, c in enumerate(ids)}
    reference = {c: rel[i] for i, c in enumerate(ids)}
    normalized = {c: r.ctxnorm_relevance(cost[i], gate[i]) for i, c in enumerate(ids)}
    assert all(
        np.array_equal(
            reference[c], h.compute_true_top4_relevance(cost[pos[c]], gate[pos[c]])
        )
        for c in ids
    )
    panel = np.full((3, 3, len(ids), len(r.ARMS)), np.nan)
    panel[:, :, :, 0] = u[:, 0]
    fits_checked = choices_checked = 0
    for rec in lock["selections"]:
        fs, seed, fold, arm = (
            rec["fold_seed"],
            rec["seed"],
            rec["assess_fold"],
            rec["actual_arm"],
        )
        key = rec["key"]
        groups = {x["fold"]: x["context_ids"] for x in folds[str(fs)]}
        ff = [i for i in range(5) if i != fold]
        fitids = [c for f in ff for c in groups[f]]
        fitpos = {c: i for i, c in enumerate(fitids)}
        labels = reference if arm == "RERANK4" else normalized
        if arm == "CTXNORM_SHUF":
            labels = h.shuffle_labels_within_family(
                labels, fitids, fams, h.stream_seed(seed, fold, "shuf")
            )
        donors = {c: c for c in fitids}
        if arm == "CTXNORM_FSHUF":
            for family in sorted({fams[c] for c in fitids}):
                recipients = [c for c in fitids if fams[c] == family]
                permutation = np.random.default_rng(
                    h.stream_seed(seed, fold, "fshuf", family)
                ).permutation(len(recipients))
                donors.update(
                    {c: recipients[int(j)] for c, j in zip(recipients, permutation)}
                )
            assert donors == rec["feature_donors"]
            assert rec["feature_fixed_contexts"] == sum(
                c == d for c, d in donors.items()
            )
        arrays = np.load(out / "cells" / (key + ".npz"), allow_pickle=False)
        predno = fitno = 0
        scores = {
            (lr, n): np.empty((len(fitids), 4))
            for lr in h.LR_GRID
            for n in h.N_EST_GRID
        }

        def check_fit(cids, lr, rng):
            nonlocal fitno, fits_checked
            expected = rec["fit_calls"][fitno]
            assert set(donors[c] for c in cids) == set(cids)
            xx = np.concatenate([X[pos[donors[c]]] for c in cids])
            yy = np.concatenate([labels[c] for c in cids])
            assert expected == dict(
                X_sha256=hashlib.sha256(xx.tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(yy.tobytes()).hexdigest(),
                n_groups=len(cids),
                lr=lr,
                random_state=rng,
            )
            fitno += 1
            fits_checked += 1

        for inner in ff:
            tr = [c for f in ff if f != inner for c in groups[f]]
            assert not {fams[c] for c in tr} & {
                fams[c] for c in groups[inner] + groups[fold]
            }
            for lr in h.LR_GRID:
                check_fit(
                    tr, lr, h.stream_seed(seed, fold, "RERANK4", "inner", inner, lr)
                )
                for trees in h.N_EST_GRID:
                    pred = arrays[f"prediction_{predno:02d}"].reshape(-1, 4)
                    predno += 1
                    scores[(lr, trees)][[fitpos[c] for c in groups[inner]]] = pred
        cfg, gain, ci, admit = replay_selector(
            scores, u[[pos[c] for c in fitids]], [fams[c] for c in fitids], seed, fold
        )
        assert cfg == (
            rec["unconstrained_best_lr"],
            rec["unconstrained_best_n_estimators"],
            rec["unconstrained_best_margin"],
        )
        assert abs(gain - rec["inner_fit_family_gain"]) < 1e-14 and np.allclose(
            ci, rec["inner_fit_ci95"], rtol=0, atol=1e-14
        )
        assert admit == rec["strict_admitted"] and rec["strict_selected_margin"] == (
            cfg[2] if admit else "inf"
        )
        for lr in sorted({cfg[0], 0.05}):
            check_fit(fitids, lr, h.stream_seed(seed, fold, "RERANK4", "outer", lr))
        assert fitno == len(rec["fit_calls"])
        path = out / "models" / (key + ".txt")
        assert h.sha256_file(path) == rec["model_sha256"]
        model = lgb.Booster(model_file=str(path), params={"num_threads": 1})
        ix = [pos[c] for c in groups[fold]]
        scores = model.predict(
            X[ix].reshape(-1, 560), num_iteration=cfg[1], num_threads=1
        ).reshape(-1, 4)
        np.testing.assert_allclose(
            scores, arrays["prediction_36"].reshape(-1, 4), rtol=0, atol=1e-12
        )
        pick = np.argmax(scores, axis=1)
        margin = cfg[2] if admit else math.inf
        pick = np.where(
            scores[np.arange(len(ix)), pick] - scores[:, 0] > margin, pick, 0
        )
        panel[r.FOLD_SEEDS.index(fs), r.SEEDS.index(seed), ix, r.ARMS.index(arm)] = u[
            ix, pick
        ]
        unc = np.argmax(scores, axis=1)
        unc = np.where(scores[np.arange(len(ix)), unc] - scores[:, 0] > cfg[2], unc, 0)
        panel[
            r.FOLD_SEEDS.index(fs), r.SEEDS.index(seed), ix, r.ARMS.index(arm + "_UNC")
        ] = u[ix, unc]
        names = np.array([fams[c] for c in groups[fold]])
        delta = u[ix, unc] - u[ix, 0]
        outer_gain = float(
            np.mean([delta[names == f].mean() for f in sorted(set(names))])
        )
        assert abs(outer_gain - rec["outer_unc_family_gain"]) < 1e-14
        choices_checked += 2 * len(ix)
        arrays.close()
        del model
    with np.load(out / "primary_panel.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(panel, z["utilities_by_repeat"])
        np.testing.assert_array_equal(panel.mean(axis=0), z["utilities"])
    with np.load(out / "val_packets.npz", allow_pickle=False) as z:
        VX = z["X_top4"]
        VU = z["utility_4"]
    vpanel = np.full((3, 5, 91, len(r.ARMS)), np.nan)
    vpanel[:, :, :, 0] = VU[:, 0]
    for rec in lock["selections"]:
        if rec["fold_seed"] != r.FOLD_SEEDS[0]:
            continue
        key = rec["key"]
        model = lgb.Booster(
            model_file=str(out / "models" / (key + ".txt")), params={"num_threads": 1}
        )
        score = model.predict(
            VX.reshape(-1, 560),
            num_iteration=rec["unconstrained_best_n_estimators"],
            num_threads=1,
        ).reshape(91, 4)
        j = np.argmax(score, axis=1)
        m = rec["strict_selected_margin"]
        m = math.inf if m == "inf" else m
        pick = np.where(score[np.arange(91), j] - score[:, 0] > m, j, 0)
        with np.load(out / "cells" / (key + "-val.npz"), allow_pickle=False) as z:
            np.testing.assert_allclose(score, z["scores"], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(pick, z["choices"])
            unc = np.argmax(score, axis=1)
            unc = np.where(
                score[np.arange(91), unc] - score[:, 0]
                > rec["unconstrained_best_margin"],
                unc,
                0,
            )
            np.testing.assert_array_equal(unc, z["unconditional_choices"])
        vpanel[
            r.SEEDS.index(rec["seed"]),
            rec["assess_fold"],
            :,
            r.ARMS.index(rec["actual_arm"]),
        ] = VU[np.arange(91), pick]
        vpanel[
            r.SEEDS.index(rec["seed"]),
            rec["assess_fold"],
            :,
            r.ARMS.index(rec["actual_arm"] + "_UNC"),
        ] = VU[np.arange(91), unc]
        choices_checked += 2 * 91
        del model
    with np.load(out / "secondary_panel.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(vpanel, z["utilities_by_fold"])
        np.testing.assert_array_equal(vpanel.mean(axis=1), z["utilities"])
    for cohort, rows, values in [
        ("primary", train, panel.mean(axis=0)),
        ("secondary", val, vpanel.mean(axis=1)),
    ]:
        families = sorted({x["family_id"] for x in rows})
        names = np.array([x["family_id"] for x in rows])
        delta = values[:, :, 2] - values[:, :, 0]
        gains = np.array([delta[:, names == f].mean(axis=1) for f in families]).mean(
            axis=0
        )
        np.testing.assert_allclose(
            gains,
            metrics[cohort]["CTXNORM - CONTROL"]["per_seed_family_gains"],
            rtol=0,
            atol=1e-14,
        )
    assert (
        h.sha256_file(out / "fit_selections_lock.json")
        == metrics["fit_selections_lock_sha256"]
    )
    assert lock["cell_count"] == 180 and lock["trained_val_reads_before_lock"] == 0
    assert metrics["verdict"] == (
        "ADVANCE" if all(metrics["gates"].values()) else "NO_ADVANCE"
    )
    assert metrics["gates"] == r.confirmation_gates(
        metrics["primary"],
        metrics["secondary"],
        json.loads((out / "raw-invalid-counts.json").read_text()),
    )
    assert metrics["confirmation_status"] == (
        "CONFIRMED" if all(metrics["gates"].values()) else "NOT_CONFIRMED"
    )
    result = dict(
        status="PASS",
        source_commit=metrics["source_commit"],
        cells=180,
        fit_input_seed_checks=fits_checked,
        choice_checks=choices_checked,
        selector_replay="180 complete grids and 2000-draw CIs",
        models_replayed=240,
        input_hashes_unchanged=True,
        neural_updates=0,
    )
    h.fsync_write_json(out / "audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
