"""One CPU-only H267 wave: zero-update family-bootstrap bagged (B = 25) median-rank LGBMRanker re-ranker of r882's top-4
(BAG_MED), its within-family label-shuffled placebo (BAG_SHUF) and the H265 single-model reference (RERANK4), against
r882 single-pick, 0 test-time reveals. Features, labels, the LGBMRanker recipe and grid, the strict inner selector and
the margin rule are the frozen H265 code (h265_core.py, byte-identical to v1/next h265 runner.py), imported unchanged.
Primary: DL-Train repeated nested family CV (fold seed per model seed); secondary: DL-Val91 (hw_t1805)."""

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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = Path("/home/ubuntu/jope-data/h267/inputs")
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


def packets_for(core, rows, costs):
    """The exact H265 packet builder (560 features, labels, utilities) for the given rows only."""
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
    out = core.load_cohort_top4_packets(rows, costs, aprior, proj16, ctxf)
    core.trim_malloc()
    return out


def stack(packets, cids):
    return np.concatenate([packets[c]["X_top4"] for c in cids], axis=0)


def fit_bags(core, packets, fit_cids, fam_by_cid, labels, lr, seed, k, arm):
    """B family-bootstrap LGBMRankers at (lr*, 200 trees); returns model strings and each bag's family set."""
    by_fam = {}
    for c in fit_cids:
        by_fam.setdefault(fam_by_cid[c], []).append(c)
    models, bag_sets = [], []
    for b in range(stat.N_BAGS):
        core.check_cooperative_pause()
        fams_b = stat.bag_families(by_fam, seed, k, b)
        cids_b = [c for f in fams_b for c in sorted(by_fam[f])]
        model = core.fit_lgbm_ranker_200(
            stack(packets, cids_b),
            np.concatenate([labels[c] for c in cids_b]),
            len(cids_b),
            lr=lr,
            random_state=stat.stream_seed(seed, k, arm, "bag", b),
        )
        models.append(model.booster_.model_to_string())
        bag_sets.append(set(fams_b))
        del model
    core.trim_malloc()
    return models, bag_sets


def bag_ranks(models, X, n_ctx, n_est):
    import lightgbm as lgb

    out = []
    for text in models:
        booster = lgb.Booster(model_str=text)
        scores = np.asarray(
            booster.predict(X, num_iteration=int(n_est)), dtype=np.float64
        ).reshape(n_ctx, 4)
        out.append(stat.within_context_ranks(scores))
    return np.stack(out)  # (bags, contexts, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    global OUT
    os.environ["H265_PAUSE_FILE"] = str(
        args.output.with_suffix(".pause.json")
    )  # core's cooperative pause hook
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
            "H267 has no resume path: a failed attempt closes INCOMPLETE"
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
    if {str(s): stat.folds(train, s) for s in stat.SEEDS} != json.loads(
        (HERE / "fold_manifest.json").read_text()
    )["families_by_seed"]:
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
    base = native91(
        val, costs, ref
    )  # Regression first: Native91 91/91 bit-exact and TOP4_PICK@r882.
    save(out / "baseline.json", dict(status="PASS", **base))
    print("BASELINE PASS", base, flush=True)
    if args.smoke:
        rng = np.random.default_rng(
            0
        )  # Synthetic bag/median exercise only; no study outcome is read.
        toy = {
            f"t{i}": {
                "X_top4": rng.normal(size=(4, 560)).astype(np.float32),
                "rel_4": rng.permutation(4),
            }
            for i in range(64)
        }
        fam = {c: f"f{i % 16}" for i, c in enumerate(toy)}
        labels = {c: toy[c]["rel_4"] for c in toy}
        models, sets = fit_bags(
            core, toy, sorted(toy), fam, labels, 0.05, 0, 0, "SMOKE"
        )
        ranks = bag_ranks(models[:3], stack(toy, sorted(toy)[:8]), 8, 50)
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                train_contexts_read=0,
                synthetic_bags=len(models),
                rank_shape=list(ranks.shape),
                baseline=base,
                lightgbm=importlib.metadata.version("lightgbm"),
            ),
        )
        return
    try:
        fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train}
        packets = packets_for(core, train, costs)
        print("TRAIN PACKETS", len(packets), flush=True)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        primary = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        cells, val_models = [], {}
        for si, seed in enumerate(stat.SEEDS):
            cbf = stat.cids_by_fold(train, seed)
            for k in range(5):
                core.check_cooperative_pause()
                ref_cell = core.run_cell_fit_and_select(
                    seed, k, "RERANK4", False, cbf, packets, fam_by_cid
                )
                lr, n_est = (
                    float(ref_cell["best_unc_cfg"][0]),
                    int(ref_cell["best_unc_cfg"][1]),
                )
                fit_cids = [c for f in range(5) if f != k for c in cbf[f]]
                assess = cbf[k]
                fit_fams = sorted({fam_by_cid[c] for c in fit_cids})
                real = {c: packets[c]["rel_4"] for c in fit_cids}
                shuf = core.shuffle_labels_within_family(
                    real, fit_cids, fam_by_cid, stat.stream_seed(seed, k, "BAG_SHUF")
                )
                X_fit, X_assess = stack(packets, fit_cids), stack(packets, assess)
                cell = dict(
                    cell=f"S{seed}-F{k}",
                    seed=seed,
                    fold_seed=stat.FOLD_SEEDS[seed],
                    assess_fold=k,
                    rerank4=ref_cell["receipt"],
                    bag_lr=lr,
                    bag_n_estimators=n_est,
                    assess_ids=assess,
                )
                picks = {
                    "r882": [0] * len(assess),
                    "RERANK4": [
                        ref_cell["assess_preds"][c]["j_strict"] for c in assess
                    ],
                }
                val_models[cell["cell"]] = dict(
                    rerank_model=ref_cell["outer_models_by_lr"][
                        lr
                    ].booster_.model_to_string(),
                    rerank_n_est=n_est,
                    rerank_margin=ref_cell["strict_margin"],
                )
                for arm, labels in (("BAG_MED", real), ("BAG_SHUF", shuf)):
                    models, sets = fit_bags(
                        core, packets, fit_cids, fam_by_cid, labels, lr, seed, k, arm
                    )
                    fit_ranks = bag_ranks(models, X_fit, len(fit_cids), n_est)
                    oob_u = []
                    for i, c in enumerate(fit_cids):
                        oob = [
                            b
                            for b in range(stat.N_BAGS)
                            if fam_by_cid[c] not in sets[b]
                        ]
                        oob_u.append(
                            packets[c]["utility_4"][
                                stat.median_rank_pick(fit_ranks[oob, i])
                            ]
                        )
                    oob_u = np.asarray(oob_u)
                    r882_u = np.array([packets[c]["r882_u"] for c in fit_cids])
                    fams = np.array([fam_by_cid[c] for c in fit_cids])
                    deltas = np.array(
                        [
                            oob_u[fams == f].mean() - r882_u[fams == f].mean()
                            for f in fit_fams
                        ]
                    )
                    adm = stat.admit(deltas, seed, k, arm)
                    a_ranks = bag_ranks(models, X_assess, len(assess), n_est)
                    picks[arm] = [
                        stat.median_rank_pick(a_ranks[:, i]) if adm["admitted"] else 0
                        for i in range(len(assess))
                    ]
                    cell[arm.lower() + "_oob_admission"] = adm
                    val_models[cell["cell"]][arm] = dict(
                        models=models, admitted=adm["admitted"]
                    )
                    del fit_ranks, a_ranks
                idx = [positions[c] for c in assess]
                for ai, arm in enumerate(stat.ALL_ARMS):
                    primary[si, idx, ai] = [
                        packets[c]["utility_4"][j] for c, j in zip(assess, picks[arm])
                    ]
                cell["assess_picks"] = picks
                cell["assess_invalid"] = {
                    a: int(
                        sum(
                            not packets[c]["gate_4"][j]
                            for c, j in zip(assess, picks[a])
                        )
                    )
                    for a in stat.ALL_ARMS
                }
                cell["assess_changed"] = {
                    a: int(sum(j != 0 for j in picks[a])) for a in stat.ALL_ARMS
                }
                save(out / (cell["cell"] + ".json"), cell)
                cells.append(cell)
                del ref_cell
                core.trim_malloc()
                print(
                    "CELL COMPLETE",
                    cell["cell"],
                    "rerank_admitted",
                    cell["rerank4"]["strict_admitted"],
                    "bag_med",
                    cell["bag_med_oob_admission"]["admitted"],
                    "bag_shuf",
                    cell["bag_shuf_oob_admission"]["admitted"],
                    flush=True,
                )
        lock = save(
            out / "selection.json",
            dict(
                cells=[
                    {
                        k2: v
                        for k2, v in c.items()
                        if k2 not in ("assess_ids", "assess_picks")
                    }
                    for c in cells
                ]
            ),
        )
        print(
            "SELECTIONS LOCKED", lock, flush=True
        )  # Durable before ANY DL-Val feature, pool or label read.
        del packets
        core.trim_malloc()
        vpk = packets_for(core, val, costs)
        vids = [r["context_sha256"] for r in val]
        Xv = stack(vpk, vids)
        secondary = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        val_detail = []
        import lightgbm as lgb

        for c in cells:
            si, k, m = (
                stat.SEEDS.index(c["seed"]),
                c["assess_fold"],
                val_models[c["cell"]],
            )
            sc = np.asarray(
                lgb.Booster(model_str=m["rerank_model"]).predict(
                    Xv, num_iteration=m["rerank_n_est"]
                )
            ).reshape(91, 4)
            picks = {
                "r882": [0] * 91,
                "RERANK4": [
                    core.select_from_top4_scores(sc[i], m["rerank_margin"])
                    for i in range(91)
                ],
            }
            for arm in ("BAG_MED", "BAG_SHUF"):
                if m[arm]["admitted"]:
                    ranks = bag_ranks(m[arm]["models"], Xv, 91, c["bag_n_estimators"])
                    picks[arm] = [stat.median_rank_pick(ranks[:, i]) for i in range(91)]
                else:
                    picks[arm] = [0] * 91
            for ai, arm in enumerate(stat.ALL_ARMS):
                secondary[si, k, :, ai] = [
                    vpk[v]["utility_4"][j] for v, j in zip(vids, picks[arm])
                ]
            val_detail.append(
                dict(
                    cell=c["cell"],
                    picks=picks,
                    invalid={
                        a: int(
                            sum(not vpk[v]["gate_4"][j] for v, j in zip(vids, picks[a]))
                        )
                        for a in stat.ALL_ARMS
                    },
                    changed={
                        a: int(sum(j != 0 for j in picks[a])) for a in stat.ALL_ARMS
                    },
                )
            )
        save(out / "val_detail.json", val_detail)
        if (
            not np.isfinite(primary).all()
            or not np.isfinite(secondary).all()
            or len(cells) != 15
        ):
            raise RuntimeError("incomplete wave")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        val_panel = secondary.mean(axis=1)
        prim, sec = stat.summarize(train, primary), stat.summarize(val, val_panel)
        decision = stat.decide(prim, sec)
        r882_sha_after = sha(pins["checkpoint"])
        metrics = dict(
            wave="H267",
            status="COMPLETE",
            verdict="ADVANCE" if decision["advance"] else "NO_ADVANCE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            fold_seeds=stat.FOLD_SEEDS,
            n_bags=stat.N_BAGS,
            optimizer_updates=0,
            baseline=base,
            decision=decision,
            primary_train_oof=prim,
            secondary_dlval=sec,
            cells=[
                {
                    k2: v
                    for k2, v in c.items()
                    if k2 not in ("assess_ids", "assess_picks")
                }
                for c in cells
            ],
            val_invalid={d["cell"]: d["invalid"] for d in val_detail},
            val_changed={d["cell"]: d["changed"] for d in val_detail},
            cliff_context=dict(
                index=stat.CLIFF_CONTEXT,
                context_sha256=vids[stat.CLIFF_CONTEXT],
                per_arm_by_seed=val_panel[:, stat.CLIFF_CONTEXT, :].tolist(),
            ),
            r882_sha_before=r882_sha_before,
            r882_sha_after=r882_sha_after,
            dope_before=dope_before,
            dope_after=dope_of(pins["checkpoint"]),
            selection_sha256=lock,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            elapsed_seconds=time.monotonic() - started,
        )
        if r882_sha_after != r882_sha_before or metrics["dope_after"] != dope_before:
            raise RuntimeError("r882 or DOPE changed")
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(status="INCOMPLETE", optimizer_updates=0, error=repr(exc)),
        )
        raise


if __name__ == "__main__":
    main()
