"""One H280 leave-cluster-out attempt, after published prereg and all regressions."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from dependencies import ROOT, topk, core, clusters
import data
import cell
import analysis_h280 as stat
from pool_build import sha, save

HERE = Path(__file__).resolve().parent


def packet_save(path, rows, P):
    ids = [r["context_sha256"] for r in rows]
    np.savez_compressed(
        path,
        cids=np.array(ids),
        **{
            k: np.stack([P[c][k] for c in ids])
            for k in ["X", "utility_k", "gate_k", "rel_k"]
        },
    )


def packets(rows, split):
    filename = (
        "r882-train-costs.npz" if split == "train" else "r882-secondary-costs.npz"
    )
    with np.load(ROOT / "prep" / split / filename) as z:
        cost = {
            r["context_sha256"]: z[r["context_sha256"]].astype(np.float32) for r in rows
        }
    ap, pj, cf = data.priors_for(rows)
    P = topk.load_packets(rows, cost, ap, pj, cf, 8)
    with np.load(ROOT / "prep" / split / "features.npz") as z:
        assert list(z["cids"]) == [r["context_sha256"] for r in rows]
        assert np.array_equal(
            z["X"], np.stack([P[r["context_sha256"]]["X"] for r in rows])
        )
    return P, cost


def sensitivity(rows, cost, split):
    """CPU primary; report stored picks when aligned, plus every CPU runner-up near tie."""
    stored = json.loads((ROOT / "inputs/round-3175.json").read_text())[
        "candidate_train_utility" if split == "train" else "dev_utility"
    ]
    pids = stored["selected_programs"]
    utilities = stored["per_context"]
    aligned = len(pids) == len(rows)
    records = []
    cpu_u = []
    stored_u = []
    alt_u = []
    for i, r in enumerate(rows):
        c = r["context_sha256"]
        rc = cost[c]
        order = np.argsort(rc, kind="mergesort")
        j = int(order[0])
        gap = float(rc[order[1]] - rc[j])
        with np.load(data.POOLS / "pools" / f"{c}.npz") as z:
            ids = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in z["program_sha256"]
            ]
            u = np.where(z["gate"], -z["fitness"].astype(np.float64), -2.0)
        sj = ids.index(pids[i]) if aligned and pids[i] in ids else None
        if sj is None:
            aligned = False
        if sj is not None and float(u[sj]) != float(utilities[i]):
            aligned = False
        cpu_u.append(float(u[j]))
        stored_u.append(float(u[sj]) if sj is not None else None)
        alt = int(order[1]) if gap < 1e-4 else j
        alt_u.append(float(u[alt]))
        records.append(
            dict(
                cid=c,
                cpu_index=j,
                stored_index=sj,
                cpu_top2_gap=gap,
                stored_vs_cpu_cost_gap=float(rc[sj] - rc[j])
                if sj is not None
                else None,
                near_tie=bool(
                    gap < 1e-4
                    or (
                        sj is not None and sj != j and abs(float(rc[sj] - rc[j])) < 1e-4
                    )
                ),
                cpu_utility=float(u[j]),
                stored_utility=float(u[sj]) if sj is not None else None,
                near_tie_runnerup_utility=float(u[alt]),
            )
        )
    return (
        dict(
            primary="CPU_REPRODUCED_PICK",
            stored_alignment_verified=aligned,
            stored_status="AVAILABLE"
            if aligned
            else "UNAVAILABLE_ORDER_OR_POOL_MISMATCH",
            records=records,
        ),
        np.asarray(cpu_u),
        np.asarray(stored_u, float) if aligned else None,
        np.asarray(alt_u),
    )


def transfer(P, rows):
    manifest = json.loads((ROOT / "export/export_manifest.json").read_text())
    ids = [r["context_sha256"] for r in rows]
    X = np.concatenate([P[c]["X"] for c in ids])
    u = np.stack([P[c]["utility_k"] for c in ids])
    values = []
    recs = []
    for c in manifest["cells"]:
        for role in ["ranker", "veto"]:
            assert sha(ROOT / "export" / c[role]["file"]) == c[role]["sha256"]
        margin = np.inf if c["strict_margin"] == "inf" else c["strict_margin"]
        j = cell.picks(
            (ROOT / "export" / c["ranker"]["file"]).read_text(),
            (ROOT / "export" / c["veto"]["file"]).read_text(),
            X,
            c["selected_n_estimators"],
            margin,
        )
        values.append(u[np.arange(len(ids)), j])
        recs.append(dict(seed=c["seed"], picks=j.tolist()))
    return np.array(values), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir()
    (out / "cells").mkdir()
    (out / "models").mkdir()
    started = time.time()
    os.environ["H265_PAUSE_FILE"] = str(out.with_suffix(".pause.json"))
    sys.addaudithook(data.audit_hook)
    core.INPUT_POOLS_DIR = data.POOLS
    m = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    assert launch["source_commit"] == os.environ["H280_SOURCE"]
    for n, d in m["source_sha256"].items():
        assert sha(HERE / n) == d, n
    for n, d in m["input_sha256"].items():
        assert sha(ROOT / n) == d, n
    for n, d in m["shared_input_sha256"].items():
        assert sha(n) == d, n
    reg = json.loads((ROOT / "regression.json").read_text())
    assert reg["status"] == "PASS"
    save(out / "baseline.json", reg)
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    selected_fam = set(json.loads((HERE / "cohort.json").read_text())["families"])
    train = [r for r in selected["train"] if r["family_id"] in selected_fam]
    val = selected["dev"]
    assert len(train) == 1219 and len(selected_fam) == 205 and len(val) == 91
    assert all(
        r["task"] == "regression"
        and r["source_split"] in ["train", "inner"]
        and not r.get("has_meta_test")
        for r in train + val
    )
    assert not selected_fam.intersection(r["family_id"] for r in val)
    before = {n: sha(ROOT / "inputs" / n) for n in ["round-882.pt", "round-3175.pt"]}
    dope = data.dope_of(ROOT / "inputs/round-3175.pt")
    assert dope == m["dope_composite_sha256"]
    P, cost = packets(train, "train")
    packet_save(out / "train_packets.npz", train, P)
    sense, cu, su, au = sensitivity(train, cost, "train")
    save(out / "train-baseline-sensitivity.json", sense)
    parts = clusters.partitions_from_manifest(
        json.loads((HERE / "folds.json").read_text())
    )
    ids = [r["context_sha256"] for r in train]
    pos = {c: i for i, c in enumerate(ids)}
    families = {r["context_sha256"]: r["family_id"] for r in train}
    panel = np.full((3, 4, len(train), 4), np.nan)
    panel[:, :, :, 0] = [P[c]["r882_u"] for c in ids]
    trans, trans_recs = transfer(P, train)
    panel[:, :, :, 1] = trans[:, None, :]
    save(out / "train-transfer-picks.json", trans_recs)
    receipts = []
    validation = []
    for pi, name in enumerate(stat.PARTITIONS):
        for si, seed in enumerate(stat.SEEDS):
            for held in range(5):
                core.check_cooperative_pause()
                folds = clusters.cluster_out_folds(train, parts[name], held, seed)
                salt = f"{name}-{held}"
                key = f"{name}_{seed}_{held}"
                assess = folds[0]
                indices = [pos[c] for c in assess]
                rec = dict(
                    key=key,
                    partition=name,
                    seed=seed,
                    held=held,
                    folds={str(f): cs for f, cs in folds.items()},
                    assess_ids=assess,
                    policies={},
                )
                for arm in ["A", "PLACEBO"]:
                    result = cell.run(
                        seed, folds, P, families, salt, placebo=arm == "PLACEBO"
                    )
                    j = np.asarray(result["assess_picks"], int)
                    panel[si, pi, indices, stat.ARMS.index(arm)] = [
                        P[c]["utility_k"][k] for c, k in zip(assess, j)
                    ]
                    model = out / "models" / f"{key}-{arm}.txt"
                    gate = out / "models" / f"{key}-{arm}-gate.txt"
                    model.write_text(result["model"])
                    gate.write_text(result["gate"])
                    rec["policies"][arm] = dict(
                        config=result["receipt"],
                        picks=j.tolist(),
                        model=model.name,
                        model_sha256=sha(model),
                        gate=gate.name,
                        gate_sha256=sha(gate),
                        fit_receipts=result["fit_receipts"],
                        shuffled_labels_sha256=result["shuffled_labels_sha256"],
                    )
                    if pi == 0:
                        validation.append(
                            dict(
                                seed=seed,
                                held=held,
                                arm=arm,
                                model=model.name,
                                gate=gate.name,
                                receipt=result["receipt"],
                            )
                        )
                save(out / "cells" / f"{key}.json", rec)
                receipts.append(
                    dict(key=key, sha256=sha(out / "cells" / f"{key}.json"))
                )
                save(
                    out / "progress.json",
                    dict(
                        stage="FIT",
                        cells_complete=len(receipts),
                        cells_total=60,
                        fitted_policies=len(receipts) * 2,
                        epoch=time.time(),
                        elapsed=time.time() - started,
                    ),
                )
                core.trim_malloc()
    assert np.isfinite(panel).all()
    save(
        out / "fit_selections_lock.json",
        dict(
            source_commit=launch["source_commit"],
            cells=receipts,
            count=60,
            fitted_policies=120,
            trained_val_reads_before_lock=0,
            epoch=time.time(),
        ),
    )
    np.savez_compressed(out / "primary_panel.npz", utilities=panel, cids=np.array(ids))
    del P
    core.trim_malloc()
    V, vcost = packets(val, "val")
    packet_save(out / "val_packets.npz", val, V)
    vsense, vcu, vsu, vau = sensitivity(val, vcost, "val")
    save(out / "val-baseline-sensitivity.json", vsense)
    vids = [r["context_sha256"] for r in val]
    X = np.concatenate([V[c]["X"] for c in vids])
    u = np.stack([V[c]["utility_k"] for c in vids])
    vp = np.full((3, 5, 91, 4), np.nan)
    vp[:, :, :, 0] = u[:, 0]
    vt, vtr = transfer(V, val)
    vp[:, :, :, 1] = vt[:, None, :]
    save(out / "val-transfer-picks.json", vtr)
    vr = []
    for rec in validation:
        cfg = rec["receipt"]
        margin = np.inf if cfg["strict_margin"] == "inf" else cfg["strict_margin"]
        j = cell.picks(
            (out / "models" / rec["model"]).read_text(),
            (out / "models" / rec["gate"]).read_text(),
            X,
            cfg["n_estimators"],
            margin,
        )
        vp[
            stat.SEEDS.index(rec["seed"]), rec["held"], :, stat.ARMS.index(rec["arm"])
        ] = u[np.arange(91), j]
        vr.append(dict(**rec, picks=j.tolist()))
    assert np.isfinite(vp).all()
    np.savez_compressed(out / "secondary_panel.npz", utilities=vp, cids=np.array(vids))
    save(out / "val-picks.json", vr)
    primary, secondary, raw = stat.summarize(train, panel, val, vp, parts)
    decision = stat.decide(primary, secondary, raw)
    # Baseline sensitivity is descriptive only; it cannot rescue a failed primary gate.
    sense_stats = {}
    for label, bu, bv in [("stored", su, vsu), ("near_tie_runnerup", au, vau)]:
        if bu is None or bv is None:
            sense_stats[label] = {
                "status": "UNAVAILABLE",
                "reason": "stored context order/pool match could not be verified",
            }
            continue
        pp = panel.copy()
        vv = vp.copy()
        pp[:, :, :, 0] = bu
        vv[:, :, :, 0] = bv
        sp, sv, sr = stat.summarize(train, pp, val, vv, parts)
        sense_stats[label] = dict(
            status="DESCRIPTIVE",
            primary={
                n: {a: sp[n][a] for a in ["A", "PLACEBO", "REF_TRANSFER"]}
                for n in stat.PARTITIONS
            },
            secondary={a: sv[a] for a in ["A", "PLACEBO", "REF_TRANSFER"]},
        )
    save(out / "sensitivity-metrics.json", sense_stats)
    after = {n: sha(ROOT / "inputs" / n) for n in before}
    assert before == after and data.dope_of(ROOT / "inputs/round-3175.pt") == dope
    full = dict(
        schema="jope.h280.metrics/v1",
        status="COMPLETE",
        source_commit=launch["source_commit"],
        verdict=decision["verdict"],
        decision=decision,
        primary=primary,
        secondary=secondary,
        raw_invalid_counts=raw,
        baseline=reg,
        cell_count=60,
        fitted_policies=120,
        neural_updates=0,
        checkpoint_before=before,
        checkpoint_after=after,
        dope_before=dope,
        dope_after=dope,
        fit_selections_lock_sha256=sha(out / "fit_selections_lock.json"),
        trained_val_reads_before_global_lock=0,
        elapsed_seconds=time.time() - started,
        model_seeds=list(stat.SEEDS),
        partitions=list(stat.PARTITIONS),
        cluster_manifest_sha256=sha(HERE / "folds.json"),
        sensitivity=dict(
            file="sensitivity-metrics.json",
            sha256=sha(out / "sensitivity-metrics.json"),
            train_stored_alignment_verified=sense["stored_alignment_verified"],
            val_stored_alignment_verified=vsense["stored_alignment_verified"],
            train_near_ties=sum(r["near_tie"] for r in sense["records"]),
            val_near_ties=sum(r["near_tie"] for r in vsense["records"]),
        ),
        baseline_sensitivity={
            label: dict(
                status=v["status"],
                primary={
                    n: {
                        k: v["primary"][n]["A"][k]
                        for k in [
                            "per_seed_family_gains",
                            "median_family_gain",
                            "ci95",
                            "positive_seeds",
                        ]
                    }
                    for n in stat.PARTITIONS
                },
                secondary={
                    k: v["secondary"]["A"][k]
                    for k in [
                        "per_seed_family_gains",
                        "median_family_gain",
                        "ci95",
                        "positive_seeds",
                    ]
                },
            )
            if v["status"] == "DESCRIPTIVE"
            else v
            for label, v in sense_stats.items()
        },
        transfer="IN_SAMPLE_DESCRIPTIVE_ONLY; frozen H278 full-DL-Train models include assessed families; never gated",
    )
    save(out / "metrics-full.json", full)
    save(
        out / "progress.json",
        dict(stage="COMPLETE", cells_complete=60, epoch=time.time()),
    )


if __name__ == "__main__":
    main()
