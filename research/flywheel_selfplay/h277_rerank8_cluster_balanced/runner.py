import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from dependencies import ROOT, topk, core, clusters
import data
import weighted
import analysis_h277 as stat

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def packets_save(path, rows, P):
    ids = [r["context_sha256"] for r in rows]
    np.savez_compressed(
        path,
        cids=np.array(ids),
        **{
            k: np.stack([P[c][k] for c in ids])
            for k in ["X", "utility_k", "gate_k", "rel_k"]
        },
    )


def score(result, X):
    cfg = result["receipt"]
    trees = cfg["n_estimators"]
    margin = cfg.get("strict_margin")
    margin = np.inf if margin == "inf" else float(margin)
    scores = (
        lgb.Booster(model_str=result["model"], params={"num_threads": 1})
        .predict(X, num_iteration=trees, num_threads=1)
        .reshape(-1, 8)
    )
    probs = (
        None
        if not result.get("gate")
        else lgb.Booster(model_str=result["gate"], params={"num_threads": 1})
        .predict(X, num_threads=1)
        .reshape(-1, 8)
    )
    return weighted.picks(scores, margin, probs, cfg.get("tau"))


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
    core.BOOT_SEED = stat.BOOT
    core.INPUT_POOLS_DIR = data.POOLS
    sys.addaudithook(data.audit_hook)
    manifest = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    assert launch["source_commit"] == os.environ["H277_SOURCE"]
    for name, digest in manifest["source_sha256"].items():
        assert sha(HERE / name) == digest, name
    for name, digest in manifest["input_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for path, digest in manifest["shared_input_sha256"].items():
        assert sha(path) == digest, path
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    train, val = selected["train"], selected["dev"]
    assert len(train) == 1219 and len(val) == 91
    assert len({r["family_id"] for r in train}) == 205 and not {
        r["family_id"] for r in train
    } & {r["family_id"] for r in val}
    assert all(
        r["source_split"] in ["train", "inner"]
        and not r.get("has_meta_test")
        and r["task"] == "regression"
        for r in train + val
    )
    ckpt = ROOT / "inputs/r882/round-882.pt"
    before = sha(ckpt)
    dope = data.dope_of(ckpt)
    assert dope == manifest["dope_composite_sha256"]
    for row in train + val:
        core.check_cooperative_pause()
        assert (
            sha(data.POOLS / "pools" / (row["context_sha256"] + ".npz"))
            == row["pool_sha256"]
        )
    costs = {}
    with (
        np.load(ROOT / "inputs/r882-train-costs.npz", allow_pickle=False) as tr,
        np.load(ROOT / "inputs/r882-secondary-costs.npz", allow_pickle=False) as va,
    ):
        for row in train + val:
            c = row["context_sha256"]
            costs[c] = (tr[c] if c in tr.files else va[c]).astype(np.float32)
    baseline = data.native91(
        val,
        costs,
        json.loads((ROOT / "inputs/r882/best_dev_checkpoint.json").read_text()),
    )
    save(out / "baseline.json", baseline)
    save(
        out / "progress.json",
        dict(stage="Native91_PASS", matches=91, epoch=time.time()),
    )
    ap_, pj, cf = data.priors_for(train)
    P = topk.load_packets(train, costs, ap_, pj, cf, 8)
    del ap_, pj, cf
    core.trim_malloc()
    packets_save(out / "train_packets.npz", train, P)
    cm = json.loads((ROOT / "inputs/fold_manifest.json").read_text())
    parts = clusters.partitions_from_manifest(cm)
    assert tuple(parts) == stat.PARTITIONS
    ids = [r["context_sha256"] for r in train]
    pos = {c: i for i, c in enumerate(ids)}
    fams = {r["context_sha256"]: r["family_id"] for r in train}
    panel = np.full((3, 4, 1219, 5), np.nan)
    panel[:, :, :, 0] = np.array([P[c]["r882_u"] for c in ids])
    receipts = []
    validation = []
    for pi, name in enumerate(stat.PARTITIONS):
        groups = parts[name]
        cluster = {f: i for i, g in enumerate(groups) for f in g}
        for si, seed in enumerate(stat.SEEDS):
            for held in range(5):
                core.check_cooperative_pause()
                folds = clusters.cluster_out_folds(train, groups, held, seed)
                salt = f"{name}-{held}"
                key = f"{name}_{seed}_{held}"
                assess = folds[0]
                indices = [pos[c] for c in assess]
                ref = topk.run_cell(
                    seed, 0, "RERANK8", False, folds, P, fams, 8, salt=salt
                )
                ab, evidence, arrays = weighted.run(seed, folds, P, fams, cluster, salt)
                pl, pevidence, parrays = weighted.run(
                    seed, folds, P, fams, cluster, salt, placebo=True
                )
                result = {"REF": ref, **ab, **pl}
                cell = dict(
                    key=key,
                    partition=name,
                    seed=seed,
                    held=held,
                    folds={str(f): cs for f, cs in folds.items()},
                    assess_ids=assess,
                    policies={},
                    weighted_evidence=evidence,
                    placebo_evidence=pevidence,
                )
                np.savez_compressed(
                    out / "cells" / (key + "-inner.npz"),
                    **arrays,
                    **{"placebo_" + k: v for k, v in parrays.items()},
                )
                for arm in stat.ARMS[1:]:
                    obj = result[arm]
                    picks = np.asarray(obj["assess_picks"], dtype=int)
                    assert len(picks) == len(assess)
                    panel[si, pi, indices, stat.ARMS.index(arm)] = [
                        P[c]["utility_k"][j] for c, j in zip(assess, picks)
                    ]
                    model = out / "models" / (key + "-" + arm + ".txt")
                    model.write_text(obj["model"])
                    gate_path = None
                    if obj.get("gate"):
                        gate_path = out / "models" / (key + "-" + arm + "-gate.txt")
                        gate_path.write_text(obj["gate"])
                    cell["policies"][arm] = dict(
                        config=obj["receipt"],
                        picks=picks.tolist(),
                        model=model.name,
                        model_sha256=sha(model),
                        gate=gate_path.name if gate_path else None,
                        gate_sha256=sha(gate_path) if gate_path else None,
                    )
                    if name == stat.PARTITIONS[0]:
                        validation.append(
                            dict(
                                seed=seed,
                                held=held,
                                arm=arm,
                                model=model.name,
                                gate=gate_path.name if gate_path else None,
                                receipt=obj["receipt"],
                            )
                        )
                save(out / "cells" / (key + ".json"), cell)
                receipts.append(
                    dict(key=key, sha256=sha(out / "cells" / (key + ".json")))
                )
                save(
                    out / "progress.json",
                    dict(
                        stage="FIT",
                        cells_complete=len(receipts),
                        cells_total=60,
                        fitted_policies=len(receipts) * 4,
                        epoch=time.time(),
                        elapsed=time.time() - started,
                    ),
                )
                del result, ref, ab, pl
                core.trim_malloc()
    assert np.isfinite(panel).all()
    save(
        out / "fit_selections_lock.json",
        dict(
            source_commit=launch["source_commit"],
            cells=receipts,
            count=60,
            fitted_policies=240,
            trained_val_reads_before_lock=0,
            epoch=time.time(),
        ),
    )
    np.savez_compressed(out / "primary_panel.npz", utilities=panel, cids=np.array(ids))
    del P
    core.trim_malloc()
    ap_, pj, cf = data.priors_for(val)
    V = topk.load_packets(val, costs, ap_, pj, cf, 8)
    del ap_, pj, cf
    packets_save(out / "val_packets.npz", val, V)
    vids = [r["context_sha256"] for r in val]
    X = np.concatenate([V[c]["X"] for c in vids])
    u = np.stack([V[c]["utility_k"] for c in vids])
    vp = np.full((3, 5, 91, 5), np.nan)
    vp[:, :, :, 0] = u[:, 0]
    val_receipts = []
    for rec in validation:
        core.check_cooperative_pause()
        obj = dict(
            receipt=rec["receipt"],
            model=(out / "models" / rec["model"]).read_text(),
            gate=(out / "models" / rec["gate"]).read_text() if rec["gate"] else None,
        )
        j = score(obj, X)
        vp[
            stat.SEEDS.index(rec["seed"]), rec["held"], :, stat.ARMS.index(rec["arm"])
        ] = u[np.arange(91), j]
        val_receipts.append(dict(**rec, picks=j.tolist()))
    assert np.isfinite(vp).all()
    np.savez_compressed(out / "secondary_panel.npz", utilities=vp, cids=np.array(vids))
    save(out / "val-picks.json", val_receipts)
    primary, secondary, raw = stat.summarize(train, panel, val, vp, parts)
    decision = stat.decide(primary, secondary, raw)
    assert sha(ckpt) == before and data.dope_of(ckpt) == dope
    full = dict(
        schema="jope.h277.metrics/v1",
        status="COMPLETE",
        source_commit=launch["source_commit"],
        verdict=decision["verdict"],
        decision=decision,
        primary=primary,
        secondary=secondary,
        raw_invalid_counts=raw,
        baseline=baseline,
        cell_count=60,
        fitted_policies=240,
        neural_updates=0,
        r882_sha_before=before,
        r882_sha_after=sha(ckpt),
        dope_before=dope,
        dope_after=data.dope_of(ckpt),
        fit_selections_lock_sha256=sha(out / "fit_selections_lock.json"),
        trained_val_reads_before_global_lock=0,
        elapsed_seconds=time.time() - started,
        model_seeds=list(stat.SEEDS),
        partitions=list(stat.PARTITIONS),
        cluster_manifest_sha256=sha(ROOT / "inputs/fold_manifest.json"),
    )
    save(out / "metrics-full.json", full)
    save(
        out / "progress.json",
        dict(stage="COMPLETE", cells_complete=60, epoch=time.time()),
    )


if __name__ == "__main__":
    main()
