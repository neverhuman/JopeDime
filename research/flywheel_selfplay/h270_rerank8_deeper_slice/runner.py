"""One CPU-only H270 wave: zero-update re-ranking of a DEEPER slice of r882's own pool (top-8, RERANK8) with the frozen
H265 pipeline where only K changes (topk.py; K = 4 reproduces h265_core exactly), vs the H265 RERANK4 reference and
r882 single-pick, with a within-family label-shuffled placebo (RERANK8_SHUF); 0 test-time reveals. Primary: DL-Train
repeated nested family CV, full 45-cell cross (3 fold seeds x 5 folds x 3 model seeds); secondary: DL-Val91 (hw_t1830)."""

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
INPUTS = Path("/home/ubuntu/jope-data/h270/inputs")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    global OUT
    os.environ["H265_PAUSE_FILE"] = str(args.output.with_suffix(".pause.json"))
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = OUT = args.output.resolve()
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
            "H270 has no resume path: a failed attempt closes INCOMPLETE"
        )
    sys.addaudithook(audit_hook)
    core = load_core()
    sys.modules["h265_core"] = core
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
    if any(int(r["n_candidates"]) < stat.K_DEEP for r in train + val):
        raise RuntimeError(
            "fewer than 8 candidates"
        )  # never true: pools hold 1,151-1,152 candidates
    if {str(f): stat.folds(train, f) for f in stat.FOLD_SEEDS} != json.loads(
        (HERE / "fold_manifest.json").read_text()
    )["families_by_fold_seed"]:
        raise RuntimeError("fold identity")
    costs = {}
    with (
        np.load(INPUTS / "r882-train-costs.npz") as zt,
        np.load(INPUTS / "r882-secondary-costs.npz") as zv,
    ):
        for r in train + val:
            cid = r["context_sha256"]
            costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
    ref = json.loads(Path(pins["reference"]).read_text())
    base = native91(val, costs, ref)
    save(out / "baseline.json", dict(status="PASS", **base))
    print("BASELINE PASS", base, flush=True)
    if args.smoke:
        rng = np.random.default_rng(
            0
        )  # Synthetic K=8 exercise only; no study outcome is read.
        toy_cost = rng.normal(size=40).astype(np.float32)
        _, X = topk.build_topk(
            toy_cost,
            rng.normal(size=40),
            rng.random((8, 412)).astype(np.float32),
            rng.normal(size=(40, 24)),
            rng.normal(size=(40, 16)),
            rng.normal(size=48),
            rng.normal(size=629),
            8,
        )
        Xs = rng.normal(size=(64 * 8, X.shape[1])).astype(np.float32)
        m = topk.fit_ranker(Xs, rng.integers(0, 8, 64 * 8), 64, 8, 0.05, 1)
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                train_contexts_read=0,
                k8_feature_width=int(X.shape[1]),
                synthetic_fit_trees=int(m.booster_.num_trees()),
                baseline=base,
            ),
        )
        return
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train}
    ap_, pj, cf = priors_for(train)
    P4 = topk.load_packets(train, costs, ap_, pj, cf, 4)
    P8 = topk.load_packets(train, costs, ap_, pj, cf, stat.K_DEEP)
    del ap_, pj, cf
    core.trim_malloc()
    print("TRAIN PACKETS", len(P4), len(P8), flush=True)
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    primary = np.full((3, 3, 1219, len(stat.ALL_ARMS)), np.nan)
    cells, val_models, invalid_train = [], {}, {a: 0 for a in stat.ALL_ARMS}
    for fi, f in enumerate(stat.FOLD_SEEDS):
        cbf = stat.cids_by_fold(train, f)
        for si, s in enumerate(stat.SEEDS):
            for k in range(5):
                core.check_cooperative_pause()
                r4 = topk.run_cell(s, k, "RERANK4", False, cbf, P4, fam_by_cid, 4)
                r8 = topk.run_cell(
                    s, k, "RERANK8", False, cbf, P8, fam_by_cid, stat.K_DEEP, salt=f
                )
                r8s = topk.run_cell(
                    s, k, "RERANK8_SHUF", True, cbf, P8, fam_by_cid, stat.K_DEEP, salt=f
                )
                assess = cbf[k]
                picks = {
                    "r882": [0] * len(assess),
                    "RERANK4": r4["assess_picks"],
                    "RERANK8": r8["assess_picks"],
                    "RERANK8_SHUF": r8s["assess_picks"],
                }
                pk = {"r882": P8, "RERANK4": P4, "RERANK8": P8, "RERANK8_SHUF": P8}
                idx = [positions[c] for c in assess]
                inv = {}
                for ai, arm in enumerate(stat.ALL_ARMS):
                    primary[si, fi, idx, ai] = [
                        pk[arm][c]["utility_k"][j] for c, j in zip(assess, picks[arm])
                    ]
                    inv[arm] = int(
                        sum(
                            not pk[arm][c]["gate_k"][j]
                            for c, j in zip(assess, picks[arm])
                        )
                    )
                    invalid_train[arm] += inv[arm]
                name = f"F{f}-S{s}-K{k}"
                cell = dict(
                    cell=name,
                    fold_seed=f,
                    seed=s,
                    assess_fold=k,
                    receipts={
                        "RERANK4": r4["receipt"],
                        "RERANK8": r8["receipt"],
                        "RERANK8_SHUF": r8s["receipt"],
                    },
                    assess_invalid=inv,
                    assess_changed={
                        a: int(sum(j != 0 for j in picks[a])) for a in stat.ALL_ARMS
                    },
                    assess_deep_picks=int(sum(j >= 4 for j in picks["RERANK8"])),
                    assess_ids=assess,
                    assess_picks=picks,
                )
                save(out / (name + ".json"), cell)
                cells.append(
                    {
                        key: v
                        for key, v in cell.items()
                        if key not in ("assess_ids", "assess_picks")
                    }
                )
                if f == stat.SECONDARY_FOLD_SEED:
                    val_models[(s, k)] = {
                        arm: (r["model"], r["n_estimators"], r["margin"])
                        for arm, r in (
                            ("RERANK4", r4),
                            ("RERANK8", r8),
                            ("RERANK8_SHUF", r8s),
                        )
                    }
                print(
                    "CELL COMPLETE",
                    name,
                    {a: cell["receipts"][a]["strict_admitted"] for a in stat.ARMS},
                    "deep_picks",
                    cell["assess_deep_picks"],
                    flush=True,
                )
    lock = save(out / "selection.json", dict(cells=cells))
    print(
        "SELECTIONS LOCKED", lock, flush=True
    )  # Durable before ANY DL-Val feature, pool or label read.
    np.savez_compressed(out / "primary_panel.npz", primary=primary)
    train_oracles = oracle_summary(train, P8)
    del P4, P8
    core.trim_malloc()
    ap_, pj, cf = priors_for(val)
    V4 = topk.load_packets(val, costs, ap_, pj, cf, 4)
    V8 = topk.load_packets(val, costs, ap_, pj, cf, stat.K_DEEP)
    vids = [r["context_sha256"] for r in val]
    X4, X8 = (
        np.concatenate([V4[c]["X"] for c in vids]),
        np.concatenate([V8[c]["X"] for c in vids]),
    )
    secondary = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
    val_detail, invalid_val = [], {a: 0 for a in stat.ALL_ARMS}
    for (s, k), m in sorted(val_models.items()):
        si = stat.SEEDS.index(s)
        picks = {"r882": [0] * 91}
        picks["RERANK4"] = predict_picks(
            *m["RERANK4"][:1], X4, 91, 4, m["RERANK4"][1], m["RERANK4"][2], core
        )
        for arm in ("RERANK8", "RERANK8_SHUF"):
            picks[arm] = predict_picks(
                m[arm][0], X8, 91, stat.K_DEEP, m[arm][1], m[arm][2], core
            )
        pk = {"r882": V8, "RERANK4": V4, "RERANK8": V8, "RERANK8_SHUF": V8}
        for ai, arm in enumerate(stat.ALL_ARMS):
            secondary[si, k, :, ai] = [
                pk[arm][c]["utility_k"][j] for c, j in zip(vids, picks[arm])
            ]
            invalid_val[arm] += int(
                sum(not pk[arm][c]["gate_k"][j] for c, j in zip(vids, picks[arm]))
            )
        val_detail.append(dict(seed=s, assess_fold=k, picks=picks))
    save(out / "val_detail.json", val_detail)
    np.savez_compressed(out / "secondary_panel.npz", secondary=secondary)
    val_oracles = oracle_summary(val, V8)
    # Packaging only from here: no assertion can discard the computed panels.
    result = dict(
        wave="H270", status="COMPUTED", elapsed_seconds=time.monotonic() - started
    )
    try:
        prim = stat.summarize(train, primary)
        sec = stat.summarize(val, secondary.mean(axis=1)[:, None])
        decision = stat.decide(
            prim, sec, invalid_train["RERANK8"], invalid_train["r882"]
        )
        result.update(
            status="COMPLETE",
            verdict="ADVANCE" if decision["advance"] else "NO_ADVANCE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            fold_seeds=stat.FOLD_SEEDS,
            k_deep=stat.K_DEEP,
            optimizer_updates=0,
            baseline=base,
            decision=decision,
            primary_train_oof=prim,
            secondary_dlval=sec,
            invalid_picks=dict(train_assess=invalid_train, dlval_15cells=invalid_val),
            oracles=dict(train=train_oracles, dlval=val_oracles),
            cells=cells,
            deep_pick_share=float(
                sum(c["assess_deep_picks"] for c in cells) / (45 * 1219 / 5)
            ),
            cliff_context=dict(
                index=stat.CLIFF_CONTEXT,
                context_sha256=vids[stat.CLIFF_CONTEXT],
                per_arm_by_seed=secondary.mean(axis=1)[
                    :, stat.CLIFF_CONTEXT, :
                ].tolist(),
            ),
            dlval_reuse_note="DL-Val91 has been read by many waves (H241-H269); DL-Train OOF on fresh fold seeds is the decision endpoint.",
            selection_sha256=lock,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
        )
    except (
        Exception
    ) as exc:  # recorded, never raised: the panels above are the durable result
        result["packaging_error"] = repr(exc)
    result["r882_sha_before"], result["r882_sha_after"] = (
        r882_sha_before,
        sha(pins["checkpoint"]),
    )
    result["dope_before"], result["dope_after"] = (
        dope_before,
        dope_of(pins["checkpoint"]),
    )
    save(out / "metrics.json", result)
    print("COMPLETE", result.get("verdict", result["status"]), flush=True)


if __name__ == "__main__":
    main()
