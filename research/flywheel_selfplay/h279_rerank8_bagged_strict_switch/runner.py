"""One CPU-only H279 wave: leave-family-CLUSTER-out test of bagged strict-margin switching (BAG_SWITCH q=6 and
BAG_SWITCH_NESTED q in {5,6,7}) on top of frozen H276 RERANK8_VETO science (topk.py, h265_core.py, veto.py byte-identical),
evaluated on the pinned H276 outcome-blind family clusterings (hw_t2140)."""

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
import bagged
import topk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = Path("/home/ubuntu/jope-data/h276/inputs")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853
REGRESSION_CELL = "F272140-S272101-K0"


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
    core.INPUT_POOLS_DIR = POOLS
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


def descriptors(rows, costs, ap_, pj, cf):
    """Outcome-blind per-context descriptors: context (lm48_z, first 25 dope629 dims) and pool (r882 cost mean, std,
    top-1/top-2 gap, top-8 mean, base-prior mean/std, proj16 mean). No fitness, gate or utility."""
    ctx, pool = {}, {}
    for r in rows:
        c = r["context_sha256"]
        ctx[c] = np.concatenate([cf[c]["lm48_z"], cf[c]["dope629"][:25]]).astype(
            np.float64
        )
        rc = np.sort(costs[c].astype(np.float64))
        ap = ap_[c].astype(np.float64)
        pool[c] = np.concatenate(
            [
                [
                    rc.mean(),
                    rc.std(),
                    rc[1] - rc[0],
                    rc[:8].mean(),
                    ap.mean(),
                    ap.std(),
                ],
                pj[c].astype(np.float64).mean(axis=0),
            ]
        )
    return ctx, pool


def regression(core, train, costs):
    ref = json.loads((INPUTS / f"h272-{REGRESSION_CELL}.json").read_text())
    ap_, pj, cf = priors_for(train)
    P8 = topk.load_packets(train, costs, ap_, pj, cf, 8)
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    import analysis_h272 as a272

    cbf = a272.cids_by_fold(train, 272140)
    got = topk.run_cell(272101, 0, "RERANK8", False, cbf, P8, fam, 8, salt=272140)
    inv = int(sum(not P8[c]["gate_k"][j] for c, j in zip(cbf[0], got["assess_picks"])))
    chg = int(sum(j != 0 for j in got["assess_picks"]))
    deep = int(sum(j >= 4 for j in got["assess_picks"]))
    ok = (
        got["receipt"] == ref["receipts"]["RERANK8"]
        and inv == ref["invalid"]["RERANK8"]
        and chg == ref["changed"]["RERANK8"]
        and deep == ref["deep_picks"]
    )
    if not ok:
        raise RuntimeError("H272 regression cell differs")
    return dict(
        cell=REGRESSION_CELL,
        receipt_equal=True,
        invalid=inv,
        changed=chg,
        deep_picks=deep,
    )


def family_contrasts(rows, panel, seed):
    """DL-Val91 (supporting): per-seed equal-family gains vs r882 and vs RERANK8_VETO, family bootstrap 10k."""
    fams = [r["family_id"] for r in rows]
    fam_ids = sorted(set(fams))
    fm = np.stack([stat.family_matrix(v, fams)[1] for v in panel])  # (3, F, A)
    ix = np.random.default_rng(seed).integers(0, fm.shape[1], (stat.DRAWS, fm.shape[1]))
    out = {}
    for col, arm in enumerate(stat.ARMS, 1):
        for label in stat.REFERENCES[arm]:
            d = fm[:, :, col] - fm[:, :, stat.ALL_ARMS.index(label)]
            boot = np.median(d[:, ix].mean(axis=2), axis=0)
            harmed_001_by_seed = [
                int((row < -stat.HARM_GATE - stat.EPS).sum()) for row in d
            ]
            out[arm + "_minus_" + label] = dict(
                per_seed_gain=d.mean(axis=1).tolist(),
                median_gain=float(np.median(d.mean(axis=1))),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                positive_seeds=int((d.mean(axis=1) > 0).sum()),
                harmed_001_by_seed=harmed_001_by_seed,
                harmed_001_median=float(np.median(harmed_001_by_seed)),
                harm_families_by_seed=[
                    [f for f, x in zip(fam_ids, row) if x < -stat.HARM] for row in d
                ],
            )
    return dict(contrasts=out, median_elo=stat.elo(panel, fams))


def compact_metrics(full):
    """Build the <50 KB Git metrics.json summary from metrics_full.json."""
    out = {k: v for k, v in full.items() if k not in ("cells",)}
    if "primary_cluster_out" in out:
        prim = dict(out["primary_cluster_out"])
        contrasts = {}
        for k, v in prim.get("contrasts", {}).items():
            contrasts[k] = {
                ck: cv
                for ck, cv in v.items()
                if ck not in ("harm_families_by_seed", "harmed_001_families_by_seed")
            }
        prim["contrasts"] = contrasts
        out["primary_cluster_out"] = prim
    if "secondary_dlval" in out:
        sec = dict(out["secondary_dlval"])
        sc = {}
        for k, v in sec.get("contrasts", {}).items():
            sc[k] = {ck: cv for ck, cv in v.items() if ck != "harm_families_by_seed"}
        sec["contrasts"] = sc
        out["secondary_dlval"] = sec
    return out


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
        [
            "git",
            "--git-dir=/home/ubuntu/JopeDime/.git",
            "rev-parse",
            "refs/remotes/origin/v1/next",
        ],
        text=True,
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
            "H279 has no resume path: a failed attempt closes INCOMPLETE"
        )
    sys.addaudithook(audit_hook)
    core = load_core()
    started = time.monotonic()
    r882_sha_before, dope_before = sha(pins["checkpoint"]), dope_of(pins["checkpoint"])
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
    ap_, pj, cf = priors_for(train)
    ctx_d, pool_d = descriptors(train, costs, ap_, pj, cf)
    fams, F = stat.family_descriptors(train, ctx_d, pool_d)
    parts = stat.clusterings(fams, F)
    cm = json.loads((HERE / "fold_manifest.json").read_text())
    if parts != stat.partitions_from_manifest(cm):
        raise RuntimeError("cluster manifest identity")
    if args.smoke:
        reg = regression(core, train, costs)
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                baseline=base,
                h272_regression=reg,
                cluster_sizes={k: [len(g) for g in v] for k, v in parts.items()},
            ),
        )
        print("REGRESSION PASS", reg, flush=True)
        return

    fam = {r["context_sha256"]: r["family_id"] for r in train}
    P8 = topk.load_packets(train, costs, ap_, pj, cf, 8)
    print("TRAIN PACKETS", len(P8), flush=True)
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    primary = np.full((3, len(stat.PARTITIONS), 1219, len(stat.ALL_ARMS)), np.nan)
    invalid = {a: 0 for a in stat.ALL_ARMS}
    changed = {a: 0 for a in stat.ALL_ARMS}
    q_selected_counts = {str(q): 0 for q in bagged.Q_GRID}
    q_tier_counts = {}
    cells, val_models = [], {}
    for pi, pname in enumerate(stat.PARTITIONS):
        clusters = parts[pname]
        for si, s in enumerate(stat.SEEDS):
            for held in range(stat.N_CLUSTERS):
                core.check_cooperative_pause()
                cbf = stat.cluster_out_folds(train, clusters, held, s)
                salt = f"{pname}-{held}"
                assess = cbf[0]
                idx = [positions[c] for c in assess]
                is_sec = pname == stat.SECONDARY_PARTITION
                cell_out = bagged.run_cell_bagged(
                    s, 0, cbf, P8, fam, 8, salt=salt, store_models=is_sec
                )
                picks = cell_out["assess_picks"]
                receipts = cell_out["receipts"]
                q_sel = int(receipts["BAG_SWITCH_NESTED"]["q_selected"])
                q_tier = str(receipts["BAG_SWITCH_NESTED"]["q_selection_tier"])
                q_selected_counts[str(q_sel)] = q_selected_counts.get(str(q_sel), 0) + 1
                q_tier_counts[q_tier] = q_tier_counts.get(q_tier, 0) + 1
                cell = dict(
                    cell=f"{pname}-S{s}-C{held}",
                    partition=pname,
                    seed=s,
                    held_cluster=held,
                    n_families=len(clusters[held]),
                    n_contexts=len(assess),
                    receipts=receipts,
                    invalid={},
                    changed={},
                    assess_ids=assess,
                    assess_picks=picks,
                    outer_votes=cell_out["outer_votes"],
                    shuf_votes=cell_out["shuf_votes"],
                )
                for ai, a in enumerate(stat.ALL_ARMS):
                    primary[si, pi, idx, ai] = [
                        P8[c]["utility_k"][j] for c, j in zip(assess, picks[a])
                    ]
                    cell["invalid"][a] = int(
                        sum(not P8[c]["gate_k"][j] for c, j in zip(assess, picks[a]))
                    )
                    cell["changed"][a] = int(sum(j != 0 for j in picks[a]))
                    invalid[a] += cell["invalid"][a]
                    changed[a] += cell["changed"][a]
                save(out / (cell["cell"] + ".json"), cell)
                cells.append(
                    {
                        k: v
                        for k, v in cell.items()
                        if k
                        not in (
                            "assess_ids",
                            "assess_picks",
                            "outer_votes",
                            "shuf_votes",
                        )
                    }
                )
                if is_sec:
                    val_models[(s, held)] = cell_out["models"]
                print(
                    "CELL",
                    cell["cell"],
                    dict(
                        admitted=receipts["RERANK8_VETO"]["strict_admitted"],
                        q_nested=q_sel,
                        q_tier=q_tier,
                        chg_ref=cell["changed"]["RERANK8_VETO"],
                        chg_bag6=cell["changed"]["BAG_SWITCH"],
                        chg_nested=cell["changed"]["BAG_SWITCH_NESTED"],
                        chg_placebo=cell["changed"]["PLACEBO"],
                    ),
                    flush=True,
                )
    lock = save(out / "selection.json", dict(cells=cells, partitions=parts))
    np.savez_compressed(out / "primary_panel.npz", primary=primary)
    print(
        "SELECTIONS LOCKED", lock, flush=True
    )  # durable before ANY DL-Val feature, pool or label read
    del P8
    core.trim_malloc()
    vap, vpj, vcf = priors_for(val)
    V8 = topk.load_packets(val, costs, vap, vpj, vcf, 8)
    vids = [r["context_sha256"] for r in val]
    XV8 = np.concatenate([V8[c]["X"] for c in vids])
    secondary = np.full((3, stat.N_CLUSTERS, 91, len(stat.ALL_ARMS)), np.nan)
    invalid_val = {a: 0 for a in stat.ALL_ARMS}
    for (s, held), m in sorted(val_models.items()):
        si = stat.SEEDS.index(s)
        picks = bagged.score_dlval_cell(m, XV8, 91, 8, core)
        for ai, a in enumerate(stat.ALL_ARMS):
            secondary[si, held, :, ai] = [
                V8[c]["utility_k"][j] for c, j in zip(vids, picks[a])
            ]
            invalid_val[a] += int(
                sum(not V8[c]["gate_k"][j] for c, j in zip(vids, picks[a]))
            )
    np.savez_compressed(out / "secondary_panel.npz", secondary=secondary)
    result = dict(
        wave="H279", status="COMPUTED", elapsed_seconds=time.monotonic() - started
    )
    try:  # packaging only: errors are recorded, never raised
        prim = stat.summarize(train, primary, parts)
        sec_dlval = family_contrasts(val, secondary.mean(axis=1), stat.BOOT_SEED)
        decision = stat.decide(prim, invalid, stat.PARTITIONS, sec_dlval)
        result.update(
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            primary_cluster_out=prim,
            secondary_dlval=sec_dlval,
            invalid_picks=dict(
                train_cluster_out=invalid, dlval_5_models_x_3_seeds=invalid_val
            ),
            changed_picks_train=changed,
            q_selected_counts=q_selected_counts,
            q_tier_counts=q_tier_counts,
            cluster_sizes={k: [len(g) for g in v] for k, v in parts.items()},
            cells=cells,
            watch_family=dict(
                primary_train={
                    a: [
                        [f for f in seed_list if f.startswith(stat.WATCH_FAMILY_PREFIX)]
                        for seed_list in prim["contrasts"][a + "_minus_r882"][
                            "harm_families_by_seed"
                        ]
                    ]
                    for a in stat.ARMS
                },
                secondary_dlval={
                    a: [
                        [f for f in seed_list if f.startswith(stat.WATCH_FAMILY_PREFIX)]
                        for seed_list in sec_dlval["contrasts"][a + "_minus_r882"][
                            "harm_families_by_seed"
                        ]
                    ]
                    for a in stat.ARMS
                },
            ),
        )
    except Exception as exc:
        result["packaging_error"] = repr(exc)
    result.update(
        arms=list(stat.ALL_ARMS),
        seeds=stat.SEEDS,
        partitions=list(stat.PARTITIONS),
        optimizer_updates=0,
        baseline=base,
        selection_sha256=lock,
        source_commit=launch["source_commit"],
        manifest_sha256=sha(HERE / "manifest.json"),
        serving_promotion=False,
        dlval_reuse_note="DL-Val91 is supporting only (heavily reused, H241-H278).",
        r882_sha_before=r882_sha_before,
        r882_sha_after=sha(pins["checkpoint"]),
        dope_before=dope_before,
        dope_after=dope_of(pins["checkpoint"]),
    )
    save(out / "metrics_full.json", result)
    save(out / "metrics.json", compact_metrics(result))
    print("COMPLETE", result.get("verdict", result["status"]), flush=True)


if __name__ == "__main__":
    main()
