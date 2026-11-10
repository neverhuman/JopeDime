"""One CPU-only H272 wave: fresh-split confirmation of the H270 RERANK8 re-ranker (r882 top-8, zero-update, single-pick)
with the H270 execution source unchanged (topk.py and h265_core.py byte-identical; only seeds change), its placebo and the
RERANK4 reference on a full 45-cell cross, descriptive depth arms RERANK12/RERANK16 and a descriptive leave-family-
cluster-out panel; DL-Val91 is secondary (hw_t1910)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import analysis as stat

import topk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = Path("/home/ubuntu/jope-data/h272/inputs")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853
OUT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=float)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha(path)


def verify(pins):
    for path, digest in pins["inputs"].items():
        if sha(path) != digest:
            raise RuntimeError("input pin mismatch: " + path)
    for path, digest in pins["parent_sources"].items():
        if sha(REPO / path) != digest:
            raise RuntimeError("parent pin mismatch: " + path)
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or launch.get("publication_ref") != "v1/next"
        or not launch.get("published_ancestor_verified")
        or len(str(launch.get("review_receipt_sha256", ""))) != 64
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external published/reviewed launch binding")


def audit_hook(event, args):
    """Refuse any open of a sealed, holdout or meta-test path."""
    if event == "open" and isinstance(args[0], (str, bytes)):
        s = os.fsdecode(args[0]).lower()
        if any(
            x in s
            for x in (
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
            )
        ):
            raise RuntimeError("protected path: " + s)


def dope_of(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    value = str(ckpt.get("dope_sha256", ""))
    del ckpt
    return value


def load_core():
    import importlib.util

    spec = importlib.util.spec_from_file_location("h265_core", HERE / "h265_core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["h265_core"] = core
    spec.loader.exec_module(core)
    core.INPUT_POOLS_DIR = POOLS  # same shared read-only pools as H265
    return core


def native91(val, costs, ref):
    ref_map = {
        c["cid"]: float(u)
        for c, u in zip(
            ref["evaluation_contract"]["contexts"], ref["utility"]["per_context"]
        )
    }
    single, top4, exact = [], [], 0
    for r in val:
        cid = r["context_sha256"]
        with np.load(POOLS / "pools" / f"{cid}.npz", allow_pickle=False) as z:
            fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        u = np.where(gate, -fit, -2.0)
        rc = costs[cid]
        u0 = float(u[int(np.argmin(rc))])
        single.append(u0)
        exact += u0 == ref_map[cid]
        t4 = np.argsort(rc, kind="mergesort")[:4]
        top4.append(float(u[int(t4[int(np.argmin(np.where(gate[t4], fit[t4], 2.0)))])]))
    fams = np.array([r["family_id"] for r in val])
    t4a = np.asarray(top4)
    fam_top4 = float(np.mean([t4a[fams == f].mean() for f in sorted(set(fams))]))
    if exact != 91 or float(np.mean(single)) != NATIVE91:
        raise RuntimeError("Native91 regression")
    if abs(float(np.mean(top4)) - TOP4_CTX) > 1e-12 or abs(fam_top4 - TOP4_FAM) > 1e-12:
        raise RuntimeError("TOP4_PICK@r882 regression")
    return dict(
        native91=float(np.mean(single)),
        matches=exact,
        top4_ctx=float(np.mean(top4)),
        top4_fam=fam_top4,
    )


def priors_for(rows):
    ids = {r["context_sha256"] for r in rows}
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        cids, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        base, proj = zp["base_prior_cost"], zp["proj16"]
        for i, cid in enumerate(cids):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[cid] = base[lo:hi].astype(np.float32).copy()
                proj16[cid] = proj[lo:hi].astype(np.float32).copy()
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as zc:
        for i, cid in enumerate(str(c) for c in zc["cids"]):
            if cid in ids:
                ctxf[cid] = {
                    "lm48_z": zc["lm48_z"][i].astype(np.float32).copy(),
                    "dope629": zc["dope629"][i].astype(np.float32).copy(),
                }
    return aprior, proj16, ctxf


def predict_picks(model_text, X, n_ctx, k, n_est, margin, core):
    import lightgbm as lgb

    sc = np.asarray(
        lgb.Booster(model_str=model_text).predict(X, num_iteration=int(n_est)),
        dtype=np.float64,
    ).reshape(n_ctx, k)
    return [core.select_from_top4_scores(sc[i], margin) for i in range(n_ctx)]


def oracle_summary(rows, packets):
    fams = np.array([r["family_id"] for r in rows])
    out = {}
    for key in ("r882_u", "oracle_top4_u", "oracle_top8_u", "oracle_pool_u"):
        u = np.array([packets[r["context_sha256"]][key] for r in rows])
        out[key] = dict(
            context_mean=float(u.mean()),
            family_mean=float(
                np.mean([u[fams == f].mean() for f in sorted(set(fams))])
            ),
        )
    out["invalid_in_top8_slice"] = int(
        sum(packets[r["context_sha256"]]["invalid_in_slice"] for r in rows)
    )
    return out


DEPTH_DEADLINE = 1793740500.0  # 2026-11-03 21:15Z: no new depth cell starts after this (descriptive arms only)
REGRESSION_CELL = "F270140-S270101-K0"  # H270 cell reproduced exactly before compute


def load_train_context(train, costs, ks):
    ap_, pj, cf = priors_for(train)
    packets = {k: topk.load_packets(train, costs, ap_, pj, cf, k) for k in ks}
    lm48 = {c: cf[c]["lm48_z"] for c in cf}
    return packets, lm48


def regression(core, train, costs, out):
    """Reproduce one H270 cell (RERANK8 at H270 seeds) exactly before any H272 compute (reads only H270-published TRAIN
    outcomes)."""
    ref = json.loads((INPUTS / f"h270-{REGRESSION_CELL}.json").read_text())
    packets, _ = load_train_context(train, costs, [8])
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    cbf = stat.cids_by_fold(train, 270140)
    if [c for c in cbf[0]] != ref["assess_ids"]:
        raise RuntimeError("regression fold identity")
    got = topk.run_cell(
        270101, 0, "RERANK8", False, cbf, packets[8], fam, 8, salt=270140
    )
    same = (
        got["receipt"] == ref["receipts"]["RERANK8"]
        and got["assess_picks"] == ref["assess_picks"]["RERANK8"]
    )
    if not same:
        raise RuntimeError("H270 regression cell differs")
    return dict(
        cell=REGRESSION_CELL,
        receipt_equal=True,
        picks_equal=True,
        n_assess=len(ref["assess_ids"]),
    )


def run_arm(arm, k, shuffle, s, f, kfold, cbf, packets, fam, salt):
    return topk.run_cell(s, kfold, arm, shuffle, cbf, packets[k], fam, k, salt=salt)


def score(arm_picks, assess, pk):
    u = [pk[c]["utility_k"][j] for c, j in zip(assess, arm_picks)]
    inv = int(sum(not pk[c]["gate_k"][j] for c, j in zip(assess, arm_picks)))
    return u, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    os.environ["H265_PAUSE_FILE"] = str(args.output.with_suffix(".pause.json"))
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        args.smoke,
    )
    if args.resume_from is not None:
        raise RuntimeError(
            "H272 has no resume path: a failed attempt closes INCOMPLETE"
        )
    sys.addaudithook(audit_hook)
    core = load_core()
    started = time.monotonic()
    r882_sha_before = sha(pins["checkpoint"])
    dope_before = dope_of(pins["checkpoint"])
    if dope_before != pins["dope_composite_sha256"]:
        raise RuntimeError("DOPE identity before")
    selected = json.loads((POOLS / "selected.json").read_text())
    train, val = selected["train"], selected["dev"]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
    ):
        raise RuntimeError("cohort size")
    if any(
        r.get("has_meta_test") is True or r["task"] != "regression" for r in train + val
    ):
        raise RuntimeError("protected context")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("family overlap")
    if any(int(r["n_candidates"]) < max(stat.DEPTH_KS) for r in train + val):
        raise RuntimeError("fewer than 16 candidates")
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    if {str(f): stat.folds(train, f) for f in stat.FOLD_SEEDS} != fm[
        "families_by_fold_seed"
    ]:
        raise RuntimeError("fold identity")
    costs = {}
    with (
        np.load(INPUTS / "r882-train-costs.npz") as zt,
        np.load(INPUTS / "r882-secondary-costs.npz") as zv,
    ):
        for r in train + val:
            cid = r["context_sha256"]
            costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
    base = native91(val, costs, json.loads(Path(pins["reference"]).read_text()))
    save(out / "baseline.json", dict(status="PASS", **base))
    print("BASELINE PASS", base, flush=True)
    if args.smoke:
        reg = regression(core, train, costs, out)
        save(
            out / "smoke.json",
            dict(
                status="PASS", optimizer_updates=0, baseline=base, h270_regression=reg
            ),
        )
        print("REGRESSION PASS", reg, flush=True)
        return
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    ks = (4, 8) + stat.DEPTH_KS
    P, lm48 = load_train_context(train, costs, ks)
    held_out = stat.cluster_holdout(train, lm48)
    if held_out != fm["tertiary_held_out_families"]:
        raise RuntimeError("tertiary holdout identity")
    print("TRAIN PACKETS", {k: len(v) for k, v in P.items()}, flush=True)
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    primary = np.full((3, 3, 1219, 4), np.nan)
    depth = np.full((3, 3, 1219, 4), np.nan)  # r882, RERANK8, RERANK12, RERANK16
    invalid = {
        a: 0
        for a in ("r882", "RERANK4", "RERANK8", "RERANK8_SHUF", "RERANK12", "RERANK16")
    }
    cells, val_models = [], {}
    gated = (
        ("RERANK4", 4, False, None),
        ("RERANK8", 8, False, "f"),
        ("RERANK8_SHUF", 8, True, "f"),
    )
    for fi, f in enumerate(stat.FOLD_SEEDS):  # Phase A: confirmatory arms, full cross
        cbf = stat.cids_by_fold(train, f)
        for si, s in enumerate(stat.SEEDS):
            for k in range(5):
                core.check_cooperative_pause()
                assess, idx = cbf[k], [positions[c] for c in cbf[k]]
                u0, inv0 = score([0] * len(assess), assess, P[8])
                primary[si, fi, idx, 0] = depth[si, fi, idx, 0] = u0
                invalid["r882"] += inv0
                cell = dict(
                    cell=f"F{f}-S{s}-K{k}",
                    fold_seed=f,
                    seed=s,
                    assess_fold=k,
                    receipts={},
                    invalid={"r882": inv0},
                    changed={},
                    deep_picks=0,
                )
                for ai, (arm, kk, shuf, salt) in enumerate(gated, 1):
                    r = run_arm(arm, kk, shuf, s, f, k, cbf, P, fam, f if salt else "")
                    u, inv = score(r["assess_picks"], assess, P[kk])
                    primary[si, fi, idx, ai] = u
                    if arm == "RERANK8":
                        depth[si, fi, idx, 1] = u
                        cell["deep_picks"] = int(sum(j >= 4 for j in r["assess_picks"]))
                    invalid[arm] += inv
                    cell["receipts"][arm], cell["invalid"][arm] = r["receipt"], inv
                    cell["changed"][arm] = int(sum(j != 0 for j in r["assess_picks"]))
                    if f == stat.SECONDARY_FOLD_SEED:
                        val_models[(s, k, arm)] = (
                            r["model"],
                            kk,
                            r["n_estimators"],
                            r["margin"],
                        )
                save(out / (cell["cell"] + ".json"), cell)
                cells.append(cell)
                print(
                    "CELL A",
                    cell["cell"],
                    {
                        a: cell["receipts"][a]["strict_admitted"]
                        for a in cell["receipts"]
                    },
                    flush=True,
                )
    lock_a = save(out / "selection_A.json", dict(cells=cells))
    np.savez_compressed(out / "primary_panel.npz", primary=primary)
    print("PHASE A LOCKED", lock_a, flush=True)
    depth_cells, truncated = [], False
    for fi, f in enumerate(
        stat.FOLD_SEEDS
    ):  # Phase B: descriptive depth arms, full cross unless the depth deadline hits
        cbf = stat.cids_by_fold(train, f)
        for si, s in enumerate(stat.SEEDS):
            for k in range(5):
                if time.time() > DEPTH_DEADLINE:
                    truncated = True
                    break
                core.check_cooperative_pause()
                assess, idx = cbf[k], [positions[c] for c in cbf[k]]
                dc = dict(
                    cell=f"F{f}-S{s}-K{k}", receipts={}, invalid={}, deep_picks={}
                )
                for ai, kk in enumerate(stat.DEPTH_KS, 2):
                    arm = f"RERANK{kk}"
                    r = run_arm(arm, kk, False, s, f, k, cbf, P, fam, f)
                    u, inv = score(r["assess_picks"], assess, P[kk])
                    depth[si, fi, idx, ai] = u
                    invalid[arm] += inv
                    dc["receipts"][arm], dc["invalid"][arm] = r["receipt"], inv
                    dc["deep_picks"][arm] = int(sum(j >= 8 for j in r["assess_picks"]))
                    if f == stat.SECONDARY_FOLD_SEED:
                        val_models[(s, k, arm)] = (
                            r["model"],
                            kk,
                            r["n_estimators"],
                            r["margin"],
                        )
                save(out / ("depth-" + dc["cell"] + ".json"), dc)
                depth_cells.append(dc)
                print(
                    "CELL B",
                    dc["cell"],
                    {a: dc["receipts"][a]["strict_admitted"] for a in dc["receipts"]},
                    flush=True,
                )
    lock_b = save(
        out / "selection_B.json", dict(cells=depth_cells, truncated=truncated)
    )
    np.savez_compressed(out / "depth_panel.npz", depth=depth)
    print("PHASE B LOCKED", lock_b, "truncated", truncated, flush=True)
    cbf = stat.cluster_folds(
        train, held_out, stat.SECONDARY_FOLD_SEED
    )  # TERTIARY: leave-family-cluster-out
    out_cids = cbf[0]
    tert = np.full((3, 1, len(out_cids), 4), np.nan)
    tert_cells = []
    for si, s in enumerate(stat.SEEDS):
        tert[si, 0, :, 0], _ = score([0] * len(out_cids), out_cids, P[8])
        tc = dict(seed=s, receipts={}, invalid={})
        for ai, (arm, kk, shuf, salt) in enumerate(gated, 1):
            r = run_arm(
                arm, kk, shuf, s, "cluster", 0, cbf, P, fam, "cluster" if salt else ""
            )
            tert[si, 0, :, ai], tc["invalid"][arm] = score(
                r["assess_picks"], out_cids, P[kk]
            )
            tc["receipts"][arm] = r["receipt"]
        tert_cells.append(tc)
    lock_c = save(out / "selection_C.json", dict(cells=tert_cells, held_out=held_out))
    print(
        "TERTIARY LOCKED", lock_c, flush=True
    )  # All selections are durable before ANY DL-Val read.
    train_oracles = oracle_summary(train, P[8])
    del P
    core.trim_malloc()
    ap_, pj, cf = priors_for(val)
    V = {k: topk.load_packets(val, costs, ap_, pj, cf, k) for k in ks}
    vids = [r["context_sha256"] for r in val]
    XV = {k: np.concatenate([V[k][c]["X"] for c in vids]) for k in ks}
    secondary = np.full((3, 5, 91, 4), np.nan)
    sec_depth = np.full((3, 5, 91, 4), np.nan)
    invalid_val = {a: 0 for a in invalid}
    for si, s in enumerate(stat.SEEDS):
        for k in range(5):
            u0, _ = score([0] * 91, vids, V[8])
            secondary[si, k, :, 0] = sec_depth[si, k, :, 0] = u0
            for ai, arm in enumerate(stat.ARMS, 1):
                text, kk, n_est, margin = val_models[(s, k, arm)]
                u, inv = score(
                    predict_picks(text, XV[kk], 91, kk, n_est, margin, core),
                    vids,
                    V[kk],
                )
                secondary[si, k, :, ai] = u
                invalid_val[arm] += inv
                if arm == "RERANK8":
                    sec_depth[si, k, :, 1] = u
            for ai, kk in enumerate(stat.DEPTH_KS, 2):
                key = (s, k, f"RERANK{kk}")
                if key in val_models:
                    text, kk2, n_est, margin = val_models[key]
                    u, inv = score(
                        predict_picks(text, XV[kk], 91, kk, n_est, margin, core),
                        vids,
                        V[kk],
                    )
                    sec_depth[si, k, :, ai] = u
                    invalid_val[key[2]] += inv
    np.savez_compressed(
        out / "secondary_panels.npz",
        secondary=secondary,
        sec_depth=sec_depth,
        tertiary=tert,
    )
    val_oracles = oracle_summary(val, V[8])
    rows_by = {r["context_sha256"]: r for r in train}
    result = dict(
        wave="H272", status="COMPUTED", elapsed_seconds=time.monotonic() - started
    )
    try:  # Packaging only: errors are recorded, never raised.
        prim = stat.summarize(train, primary)
        sec = stat.summarize(val, secondary.mean(axis=1)[:, None])
        decision = stat.decide(prim, sec, invalid["RERANK8"], invalid["r882"])
        result.update(
            status="COMPLETE",
            verdict="CONFIRMED" if decision["advance"] else "NOT_CONFIRMED",
            decision=decision,
            primary_train_oof=prim,
            secondary_dlval=sec,
            tertiary_cluster_holdout=stat.summarize(
                [rows_by[c] for c in out_cids], tert
            ),
            tertiary_held_out_families=held_out,
            invalid_picks=dict(train_assess=invalid, dlval_15cells=invalid_val),
            oracles=dict(train=train_oracles, dlval=val_oracles),
            cells=cells,
            depth_cells=[
                {
                    "cell": d["cell"],
                    "admitted": {
                        a: d["receipts"][a]["strict_admitted"] for a in d["receipts"]
                    },
                    "invalid": d["invalid"],
                    "deep_picks": d["deep_picks"],
                }
                for d in depth_cells
            ],
            depth_truncated=truncated,
            tertiary_cells=tert_cells,
            cliff_context=dict(
                index=stat.CLIFF_CONTEXT,
                per_arm_by_seed=secondary.mean(axis=1)[
                    :, stat.CLIFF_CONTEXT, :
                ].tolist(),
            ),
        )
        if not truncated:
            result["depth_train_oof"] = stat.summarize(
                train, depth, arms=stat.DEPTH_ARMS
            )
            result["depth_dlval"] = stat.summarize(
                val, sec_depth.mean(axis=1)[:, None], arms=stat.DEPTH_ARMS
            )
    except Exception as exc:
        result["packaging_error"] = repr(exc)
    result.update(
        arms=["r882", *stat.ARMS],
        depth_arms=list(stat.DEPTH_ARMS),
        seeds=stat.SEEDS,
        fold_seeds=stat.FOLD_SEEDS,
        optimizer_updates=0,
        baseline=base,
        selection_sha256=dict(A=lock_a, B=lock_b, C=lock_c),
        dlval_reuse_note="DL-Val91 has been read by many waves (H241-H271); DL-Train OOF on fresh fold seeds is the decision endpoint.",
        source_commit=launch["source_commit"],
        manifest_sha256=sha(HERE / "manifest.json"),
        serving_promotion=False,
        r882_sha_before=r882_sha_before,
        r882_sha_after=sha(pins["checkpoint"]),
        dope_before=dope_before,
        dope_after=dope_of(pins["checkpoint"]),
    )
    save(out / "metrics.json", result)
    print("COMPLETE", result.get("verdict", result["status"]), flush=True)


if __name__ == "__main__":
    main()
