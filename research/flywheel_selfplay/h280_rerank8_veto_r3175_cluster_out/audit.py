"""Saved-model/input/fold replay; frozen statistics recomputed (not an independent estimator)."""

import argparse
import json
import numpy as np
from dependencies import ROOT, core, topk, clusters
from pool_build import save, sha
import cell
import analysis_h280 as stat


def load(path):
    with np.load(path) as z:
        return {
            str(c): {k: z[k][i].copy() for k in ["X", "utility_k", "gate_k", "rel_k"]}
            for i, c in enumerate(z["cids"])
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from pathlib import Path

    out = Path(args.output)
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    rows = selected["train"]
    val = selected["dev"]
    P = load(out / "train_packets.npz")
    V = load(out / "val_packets.npz")
    ids = [r["context_sha256"] for r in rows]
    pos = {c: i for i, c in enumerate(ids)}
    families = {r["context_sha256"]: r["family_id"] for r in rows}
    parts = clusters.partitions_from_manifest(
        json.loads((ROOT / "source/folds.json").read_text())
    )
    models = fit_count = 0
    with np.load(out / "primary_panel.npz") as z:
        panel = z["utilities"].copy()
    with np.load(out / "secondary_panel.npz") as z:
        vp = z["utilities"].copy()
    lock = json.loads((out / "fit_selections_lock.json").read_text())
    assert len(lock["cells"]) == 60
    for locked in lock["cells"]:
        path = out / "cells" / (locked["key"] + ".json")
        assert sha(path) == locked["sha256"]
        rec = json.loads(path.read_text())
        folds = {int(k): v for k, v in rec["folds"].items()}
        assert folds == clusters.cluster_out_folds(
            rows, parts[rec["partition"]], rec["held"], rec["seed"]
        )
        fit = [c for f in range(1, 5) for c in folds[f]]
        for arm, x in rec["policies"].items():
            labels = {c: P[c]["rel_k"] for c in fit}
            if arm == "PLACEBO":
                labels = topk.shuffle_within_family(
                    labels,
                    fit,
                    families,
                    core.stream_seed(
                        rec["seed"], 0, "shuf", f"{rec['partition']}-{rec['held']}"
                    ),
                    8,
                )
                assert {c: cell.array_sha(y) for c, y in labels.items()} == x[
                    "shuffled_labels_sha256"
                ]
            expected = []
            for hold in [1, 2, 3, 4, "outer"]:
                tr = [c for k in range(1, 5) if k != hold for c in folds[k]]
                X = np.concatenate([P[c]["X"] for c in tr])
                inv = np.concatenate([~P[c]["gate_k"] for c in tr])
                y = np.concatenate([labels[c] for c in tr])
                xh = cell.array_sha(X)
                if hold != "outer":
                    expected.append(("gate", xh, cell.array_sha(inv.astype(int))))
                    expected.extend(
                        [("ranker", xh, cell.array_sha(y)) for lr in core.LR_GRID]
                    )
                else:
                    expected.extend(
                        [
                            ("ranker", xh, cell.array_sha(y)),
                            ("gate", xh, cell.array_sha(inv.astype(int))),
                        ]
                    )
            assert [
                (q["kind"], q["X_sha256"], q["y_sha256"]) for q in x["fit_receipts"]
            ] == expected
            fit_count += len(expected)
            for role in ["model", "gate"]:
                assert sha(out / "models" / x[role]) == x[role + "_sha256"]
            cfg = x["config"]
            margin = np.inf if cfg["strict_margin"] == "inf" else cfg["strict_margin"]
            X = np.concatenate([P[c]["X"] for c in rec["assess_ids"]])
            j = cell.picks(
                (out / "models" / x["model"]).read_text(),
                (out / "models" / x["gate"]).read_text(),
                X,
                cfg["n_estimators"],
                margin,
            )
            assert j.tolist() == x["picks"]
            si = stat.SEEDS.index(rec["seed"])
            pi = stat.PARTITIONS.index(rec["partition"])
            ai = stat.ARMS.index(arm)
            indices = [pos[c] for c in rec["assess_ids"]]
            assert np.array_equal(
                panel[si, pi, indices, ai],
                [P[c]["utility_k"][k] for c, k in zip(rec["assess_ids"], j)],
            )
            models += 1
    vids = [r["context_sha256"] for r in val]
    X = np.concatenate([V[c]["X"] for c in vids])
    u = np.stack([V[c]["utility_k"] for c in vids])
    for rec in json.loads((out / "val-picks.json").read_text()):
        cfg = rec["receipt"]
        margin = np.inf if cfg["strict_margin"] == "inf" else cfg["strict_margin"]
        j = cell.picks(
            (out / "models" / rec["model"]).read_text(),
            (out / "models" / rec["gate"]).read_text(),
            X,
            cfg["n_estimators"],
            margin,
        )
        assert j.tolist() == rec["picks"]
        assert np.array_equal(
            vp[
                stat.SEEDS.index(rec["seed"]),
                rec["held"],
                :,
                stat.ARMS.index(rec["arm"]),
            ],
            u[np.arange(len(vids)), j],
        )
        models += 1
    from runner import transfer

    for rows_, packets_, name, values in [
        (rows, P, "train", panel),
        (val, V, "val", vp),
    ]:
        got, recs = transfer(packets_, rows_)
        assert recs == json.loads((out / (name + "-transfer-picks.json")).read_text())
        assert np.array_equal(
            values[:, :, :, 1],
            np.broadcast_to(got[:, None, :], values[:, :, :, 1].shape),
        )
        models += 3
    primary, secondary, raw = stat.summarize(rows, panel, val, vp, parts)
    m = json.loads((out / "metrics-full.json").read_text())
    assert (
        primary == m["primary"]
        and secondary == m["secondary"]
        and raw == m["raw_invalid_counts"]
        and stat.decide(primary, secondary, raw) == m["decision"]
    )
    assert (
        m["checkpoint_before"] == m["checkpoint_after"]
        and m["dope_before"] == m["dope_after"]
    )
    receipt = dict(
        status="PASS",
        fit_input_receipts=fit_count,
        outer_and_val_and_transfer_model_replays=models,
        panels_exact=True,
        folds_exact=True,
        train_only_shuffles=True,
        statistic_recomputation="same frozen estimator; not independent",
        inner_selector="exact byte-pinned H276 implementation; unit regression and input closures, not full grid re-fit",
        source_commit=m["source_commit"],
    )
    save(out / "audit.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
