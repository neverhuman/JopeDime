"""H294 evaluation (xbabe2 CPU, hw_t0158) on fresh EVEN cohort v5 (358 families), GPU-exact r3175 costs.

PRIMARY: ENS_VOTE2 from rd_argon's frozen H287 export (213b30ad), scored with ensemble.score_all_h287_arms.
SECONDARY/descriptive (cannot flip the verdict):
  ENS_VOTE2+HARMVETO: rd_argon's frozen H291 harm-veto models (fit lock at d8bc373b80; files fetched by S3 VersionId from
    H291's s3_manifest.json, sha256 enforced against fit_lock.json), applied exactly as H291's _score_frozen_arm does:
    base picks = harm_veto.compute_vote_picks(member picks, 2); features = harm_veto.build_context_harm_features_from_arrays;
    a proposed switch is kept iff p_harm < the locked tau of that seed (291101-291103). Regression first: the published
    H291 Part B seed gains on H286 EVEN are reproduced bit-exactly;
  CLIP and BAG_SWITCH (H287 per-seed picks; H287's BAG_SWITCH equals H284's, verified in H288).
Nulls: random valid slot (294101-294103), label-free random slot (294201-294203).
Regression first: H281's r3175 GPU panel (bit-exact) and H292's ENS_VOTE2 cell (regress_h292.py, before v5).
Pooled H288 + H292 + H294 is descriptive only (every arm and null re-scored on H288's and H292's published packets)."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H278 = HERE.parent / "h278_rerank8_veto_frozen_export_transfer"
H281 = HERE.parent / "h281_r3175_gpu_rescore_p2"
H283 = HERE.parent / "h283_rerank8_veto_fresh_cohort_gpu"
H287 = HERE.parent / "h287_bagswitch_seed_ensemble_export"
H288 = HERE.parent / "h288_fresh_v3_odd_powered_ensemble"
H291 = HERE.parent / "h291_harm_veto_abstention"
H291_SEEDS = (291101, 291102, 291103)
H291_TARGET_SEED_GAINS = [
    0.0023750521973936573,
    0.02714368976363186,
    0.03312056683143756,
]
WATCH = "srcfile:rdataset"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fetch_h291(edir):
    """The ENS_VOTE2_HARMVETO harm-veto models (3 seeds) by S3 VersionId from H291's s3_manifest, sha256 vs fit_lock."""
    edir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((H291 / "fit_lock.json").read_text())
    objs = {
        o["local_file"]: o
        for o in json.loads((H291 / "s3_manifest.json").read_text())["objects"]
    }
    models = {}
    for m in lock["exported_harm_veto_models"]:
        if m["arm"] != "ENS_VOTE2_HARMVETO" or m["role"] != "harm_veto_full_dl_train":
            continue
        o = objs[m["file"]]
        path = edir / m["file"]
        if not path.exists():
            subprocess.run(
                [
                    "aws",
                    "s3api",
                    "get-object",
                    "--bucket",
                    o["s3_bucket"],
                    "--key",
                    o["s3_key"],
                    "--version-id",
                    o["version_id"],
                    str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if sha(path) != m["sha256"] or o["sha256"] != m["sha256"]:
            raise RuntimeError("H291 model sha256 mismatch: " + m["file"])
        models[int(m["seed"])] = dict(path=path, tau=float(m["locked_tau"]))
    if sorted(models) != list(H291_SEEDS):
        raise RuntimeError("H291 ENS_VOTE2_HARMVETO models incomplete")
    return models


def harmveto_picks(ctx, X3, n):
    """H291 _score_frozen_arm for ENS_VOTE2_HARMVETO (harm-veto branch), on top-8 features X3 (n, 8, 568)."""
    import lightgbm as lgb

    ens, hv, spec, edir = ctx["ens"], ctx["hv"], ctx["h287_spec"], ctx["h287_dir"]
    X2 = X3.reshape(n * 8, 568)
    um = [
        ens.score_single_member_from_disk(edir, c, X2, n, 8)
        for c in spec["h284_bag_switch_members"]
    ]
    cm = [
        ens.score_single_member_from_disk(edir, c, X2, n, 8)
        for c in spec["clip_bag_switch_members"]
    ]

    def stack(ms):
        return (
            np.stack([m["raw_scores"] for m in ms]),
            np.stack([m["bag_scores"] for m in ms]),
            np.stack([m["p_inv"] for m in ms]),
            np.asarray([m["picks_bag"] for m in ms], dtype=np.int32),
            np.asarray([m["j_ref"] for m in ms], dtype=np.int32),
        )

    u_raw, u_bag, u_pinv, u_pbag, u_jref = stack(um)
    c_raw, c_bag, c_pinv, c_pbag, c_jref = stack(cm)
    ens2 = hv.compute_vote_picks(u_pbag, min_agree=2)
    F = hv.build_context_harm_features_from_arrays(
        X_all=X3, c_raw_r=c_raw, c_bag_r=c_bag, c_pinv_r=c_pinv, c_pbag_r=c_pbag, c_jref_r=c_jref,
        u_raw_r=u_raw, u_bag_r=u_bag, u_pinv_r=u_pinv, u_pbag_r=u_pbag, u_jref_r=u_jref, cand_picks=ens2, use_clip=False,
    )  # fmt: skip
    picks = []
    for seed in H291_SEEDS:
        m = ctx["h291_models"][seed]
        p = lgb.Booster(model_file=str(m["path"])).predict(F)
        picks.append(
            [
                int(ens2[i]) if ens2[i] != 0 and float(p[i]) < m["tau"] else 0
                for i in range(n)
            ]
        )
    return picks, [int(j) for j in ens2]


def run_arms(ctx, V, rows):
    stat, ev, ens = ctx["stat"], ctx["ev"], ctx["ens"]
    ids = [r["context_sha256"] for r in rows]
    X3 = np.stack([V[c]["X"] for c in ids]).astype(np.float32)
    X = X3.reshape(len(ids) * 8, 568)
    out = {}
    r = ens.score_all_h287_arms(ctx["h287_dir"], X, len(ids), 8, ctx["h287_spec"])
    ev2 = [[int(j) for j in r["ENS_VOTE2"]]] * 3
    out["ENS_VOTE2"] = (*ev.to_panel(V, ids, ev2), ev2, stat.DETERMINISTIC_BOOT_SEEDS)
    for arm in ("CLIP", "BAG_SWITCH"):
        pk = [[int(j) for j in p] for p in r["per_seed"][arm]]
        out[arm] = (*ev.to_panel(V, ids, pk), pk, (stat.BOOT_SEED,) * 3)
    hvp, ens2_h291 = harmveto_picks(ctx, X3, len(ids))
    out["ENS_VOTE2+HARMVETO"] = (*ev.to_panel(V, ids, hvp), hvp, (stat.BOOT_SEED,) * 3)
    ctx.setdefault("h291_vote2_equals_h287_vote2", []).append(ens2_h291 == ev2[0])
    vp = [
        stat.random_valid_slot_picks([V[c]["gate_k"] for c in ids], s)
        for s in stat.VALID_NULL_SEEDS
    ]
    out["NULL_RANDOM_VALID_SLOT"] = (
        *ev.to_panel(V, ids, vp),
        vp,
        (stat.BOOT_SEED,) * 3,
    )
    npk = [
        stat.random_slot_no_label_picks(len(ids), s) for s in stat.NOLABEL_NULL_SEEDS
    ]
    out["NULL_RANDOM_SLOT_NO_LABELS"] = (
        *ev.to_panel(V, ids, npk),
        npk,
        (stat.BOOT_SEED,) * 3,
    )
    return out


def judge(stat, arms, families):
    n = len(set(families))
    sums = {
        k: stat.summarize(u, inv, families, pk, bs)
        for k, (u, inv, pk, bs) in arms.items()
    }
    crit = {k: stat.criteria(sums[k], arms[k][1], n) for k in sums}
    nulls = [crit["NULL_RANDOM_VALID_SLOT"], crit["NULL_RANDOM_SLOT_NO_LABELS"]]
    report = {}
    for k, s in sums.items():
        hs = [set(x) for x in s["harmed_families_by_seed"]]
        always = sorted(set.intersection(*hs))
        groups = {}
        for f in always:
            groups[stat.source_group(f)] = groups.get(stat.source_group(f), 0) + 1
        report[k] = dict(
            summary=stat.public(s),
            criteria=crit[k],
            passes_rule=stat.passes(crit[k]),
            verdict=None if k.startswith("NULL_") else stat.verdict(n, crit[k], nulls),
            always_harmed=dict(
                count=len(always),
                by_source_group=groups,
                watch_rdataset=sum(
                    g.startswith(WATCH) for g in map(stat.source_group, always)
                ),
                families=always,
            ),
        )
    return report


def main():
    ap = argparse.ArgumentParser()
    for a in (
        "fresh-dir",
        "gpu-dir",
        "h287-export-manifest",
        "h281-costs",
        "h281-panel",
        "h286-fresh",
        "h286-costs",
        "h288-fresh",
        "h288-costs",
        "h292-fresh",
        "h292-costs",
        "regression-h292",
        "out",
    ):
        ap.add_argument("--" + a, type=Path, required=True)
    args = ap.parse_args()
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    if (
        int(
            next(
                line.split()[1]
                for line in open("/proc/meminfo")
                if line.startswith("MemAvailable")
            )
        )
        < 10 * 1024 * 1024
    ):
        raise RuntimeError("MemAvailable < 10 GiB: abort")
    out = args.out.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    for root, table in (
        (H278, "h278_sources"),
        (H281, "h281_sources"),
        (H283, "h283_sources"),
        (H287, "h287_sources"),
        (H288, "h288_sources"),
        (H291, "h291_sources"),
        (HERE.parent / "h292_fresh_v4_odd_confirm_ensvote2_clip", "h292_sources"),
    ):
        for name, digest in pins[table].items():
            if sha(root / name) != digest:
                raise RuntimeError("source pin mismatch: " + name)
    if sha(args.h287_export_manifest) != pins["h287_export_manifest_sha256"]:
        raise RuntimeError("H287 export manifest hash")
    if not json.loads(args.regression_h292.read_text())["bit_exact"]:
        raise RuntimeError("H292 cell regression must pass first")
    rec = json.loads((args.gpu_dir / "rescore.json").read_text())
    if rec.get("outcome") != "A_EXACT" or rec["fresh"].get("status") != "COMPLETE":
        raise RuntimeError("fresh scoring opens only after an exact r3175 regression")
    if sha(args.gpu_dir / "fresh-r3175-gpu-costs.npz") != rec["fresh"]["costs_sha256"]:
        raise RuntimeError("fresh costs hash")
    sys.path.insert(0, str(H278))
    import freeze
    import runner as rt
    import topk

    sys.addaudithook(rt.audit_hook)
    rt.load_core()
    p2 = load("h281_p2", H281 / "p2.py")
    e283 = load("h283_evaluate", H283 / "evaluate.py")
    ev = load("h288_evaluate", H288 / "evaluate.py")
    stat = load("h294_analysis", HERE / "analysis.py")
    h287_dir = Path("/home/ubuntu/jope-data/h294/export_h287")
    h287_spec = ev.fetch_h287(
        json.loads(args.h287_export_manifest.read_text()), h287_dir
    )
    sys.path.insert(0, str(H287))
    ens = load("h287_ensemble", H287 / "ensemble.py")
    sys.path.insert(0, str(H291))
    hv = load("h291_harm_veto", H291 / "harm_veto.py")
    ctx = dict(
        stat=stat,
        ev=ev,
        ens=ens,
        hv=hv,
        h287_dir=h287_dir,
        h287_spec=h287_spec,
        h291_models=fetch_h291(Path("/home/ubuntu/jope-data/h294/export_h291")),
    )

    # Regression: H281's r3175 GPU panel on DL-Val91, bit-exact.
    h278_cells = p2.fetch_models(
        json.loads(
            (Path.home() / "jope_plan/evidence/H278/export_manifest.json").read_text()
        ),
        Path("/home/ubuntu/jope-data/h294/export_h278"),
    )
    topk.core.INPUT_POOLS_DIR = rt.POOLS
    val = json.loads((rt.POOLS / "selected.json").read_text())["dev"]
    vap, vpj, vcf = rt.priors_for(val)
    with np.load(args.h281_costs) as z:
        c81 = {
            r["context_sha256"]: z[r["context_sha256"]].astype(np.float32) for r in val
        }
    u81, inv81, _ = p2.panel(
        rt, freeze, val, topk.load_packets(val, c81, vap, vpj, vcf, 8), h278_cells
    )
    with np.load(args.h281_panel) as z:
        regression = bool(
            np.array_equal(u81, z["utility"]) and np.array_equal(inv81, z["invalid"])
        )
    if not regression:
        raise RuntimeError("H281 panel regression failed")

    def packets(fresh_dir, costs_path):
        topk.core.INPUT_POOLS_DIR = fresh_dir
        rows = json.loads((fresh_dir / "selected.json").read_text())["fresh"]
        ids = [r["context_sha256"] for r in rows]
        fap, fpj, fcf = e283.priors_from(fresh_dir, rows)
        with np.load(costs_path) as z:
            costs = {c: z[c].astype(np.float32) for c in ids}
        return rows, ids, costs, topk.load_packets(rows, costs, fap, fpj, fcf, 8)

    # Regression: the published H291 Part B ENS_VOTE2+HARMVETO seed gains on H286 EVEN, bit-exact.
    r6, i6, _, V6 = packets(args.h286_fresh, args.h286_costs)
    hvp6, _ = harmveto_picks(
        ctx, np.stack([V6[c]["X"] for c in i6]).astype(np.float32), len(i6)
    )
    u6, _ = ev.to_panel(V6, i6, hvp6)
    h291_gains = [float(np.mean(u6[s, :, 1] - u6[s, :, 0])) for s in range(3)]
    h291_regression = h291_gains == H291_TARGET_SEED_GAINS
    if not h291_regression:
        raise RuntimeError(f"H291 HARMVETO regression failed: {h291_gains}")

    rows, ids, costs, V = packets(
        args.fresh_dir, args.gpu_dir / "fresh-r3175-gpu-costs.npz"
    )
    families = [r["family_id"] for r in rows]
    slot0_mean = float(np.mean([V[c]["utility_k"][0] for c in ids]))
    if abs(slot0_mean - rec["fresh"]["slot0_mean"]) > 1e-12:
        raise RuntimeError("slot 0 differs from the GPU argmin utilities")
    arms = run_arms(ctx, V, rows)
    np.savez_compressed(
        out / "panels.npz",
        **{f"{k}_utility": v[0] for k, v in arms.items()},
        **{f"{k}_invalid": v[1] for k, v in arms.items()},
    )
    (out / "picks.json").write_text(
        json.dumps({k: [list(map(int, p)) for p in v[2]] for k, v in arms.items()})
        + "\n"
    )
    np.savez_compressed(
        out / "fresh_top8_packets.npz",
        cids=np.asarray(ids),
        X=np.stack([V[c]["X"] for c in ids]),
        costs_top8=np.stack([np.sort(costs[c], kind="mergesort")[:8] for c in ids]),
        order_top8=np.stack([np.argsort(costs[c], kind="mergesort")[:8] for c in ids]),
    )
    result = dict(
        wave="H294",
        step="evaluate",
        regression_h281_panel_exact=regression,
        regression_h292_cell_exact=True,
        regression_h291_harmveto_exact=h291_regression,
        fresh_contexts=len(ids),
        fresh_families=len(set(families)),
        slot0_fresh_utility_mean=slot0_mean,
        oracle_top8_mean=float(np.mean([V[c]["oracle_top8_u"] for c in ids])),
    )
    try:
        report = judge(stat, arms, families)
        result.update(verdict=report[stat.PRIMARY]["verdict"], arms=report)
        result["h291_vote2_equals_h287_vote2"] = ctx["h291_vote2_equals_h287_vote2"][-1]
    except Exception as error:  # packaging only: panels.npz is the result
        result["packaging_error"] = repr(error)
    try:
        parts = [
            (packets(args.h288_fresh, args.h288_costs)),
            (packets(args.h292_fresh, args.h292_costs)),
        ]
        famsP, armsP = list(families), {k: [v] for k, v in arms.items()}
        for rr, _, _, VV in parts:
            ff = [r["family_id"] for r in rr]
            if set(ff) & set(famsP):
                raise RuntimeError("pooled families overlap")
            a = run_arms(ctx, VV, rr)
            famsP = ff + famsP
            for k in armsP:
                armsP[k].insert(0, a[k])
        pooled = {
            k: (
                np.concatenate([x[0] for x in v], axis=1),
                np.concatenate([x[1] for x in v], axis=1),
                [sum((list(x[2][s]) for x in v), []) for s in range(3)],
                v[0][3],
            )
            for k, v in armsP.items()
        }
        prep = judge(stat, pooled, famsP)
        result["pooled_h288_h292_h294"] = dict(
            label="pooled, H288 and H292 portions already seen (descriptive; H288 ODD v3 + H292 ODD v4 + H294 EVEN v5)",
            families=len(set(famsP)),
            verdicts={k: v["verdict"] for k, v in prep.items()},
            arms={
                k: dict(
                    summary={
                        kk: vv
                        for kk, vv in v["summary"].items()
                        if kk != "harmed_families_by_seed"
                    },
                    criteria=v["criteria"],
                    passes_rule=v["passes_rule"],
                )
                for k, v in prep.items()
            },
        )
    except Exception as error:
        result["pooled_error"] = repr(error)
    (out / "evaluate.json").write_text(
        json.dumps(result, indent=1, allow_nan=False) + "\n"
    )
    print("H294", result.get("verdict"), flush=True)


if __name__ == "__main__":
    main()
