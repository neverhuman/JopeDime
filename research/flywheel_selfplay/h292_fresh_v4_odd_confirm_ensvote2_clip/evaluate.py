"""H292 evaluation (xbabe2 CPU, hw_t0125) on fresh cohort v4 (295 ODD families), GPU-exact r3175 costs.

PRIMARIES (Holm, 2 hypotheses): ENS_VOTE2 and CLIP from rd_argon's frozen H287 export (commit 213b30ad), scored with
ensemble.score_all_h287_arms. SECONDARY/descriptive: BAG_SWITCH (H284 export, H284's scorer), ENS_MEAN and ENS_VOTE3
(H287), RERANK8_VETO (H278). Every file is fetched by S3 VersionId with sha256 enforced (H288's fetch helpers,
unchanged). Nulls: random valid slot (292101-292103) and label-free random slot (292201-292203).
Regressions first: H281's r3175 GPU panel (bit-exact) and H288's ENS_VOTE2 cell (regress_h288.py, run before v4).
Pooled H288 + H292 (600 ODD families) is descriptive only: every arm and null re-scored on H288's published packets."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H278 = HERE.parent / "h278_rerank8_veto_frozen_export_transfer"
H281 = HERE.parent / "h281_r3175_gpu_rescore_p2"
H283 = HERE.parent / "h283_rerank8_veto_fresh_cohort_gpu"
H284 = HERE.parent / "h284_bagswitch_frozen_fresh"
H287 = HERE.parent / "h287_bagswitch_seed_ensemble_export"
H288 = HERE.parent / "h288_fresh_v3_odd_powered_ensemble"
WATCH = "srcfile:rdataset"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_arms(ctx, V, rows):
    stat, ev, p2, fb, rt, freeze, ens = (
        ctx[k] for k in ("stat", "ev", "p2", "fb", "rt", "freeze", "ens")
    )
    ids = [r["context_sha256"] for r in rows]
    X = np.concatenate([V[c]["X"] for c in ids])
    out = {}
    r = ens.score_all_h287_arms(ctx["h287_dir"], X, len(ids), 8, ctx["h287_spec"])
    for arm in ("ENS_VOTE2", "ENS_MEAN", "ENS_VOTE3"):
        picks = [[int(j) for j in r[arm]]] * 3
        out[arm] = (*ev.to_panel(V, ids, picks), picks, stat.DETERMINISTIC_BOOT_SEEDS)
    clip = [[int(j) for j in p] for p in r["per_seed"]["CLIP"]]
    out["CLIP"] = (*ev.to_panel(V, ids, clip), clip, (stat.BOOT_SEED,) * 3)
    bag = []
    for cell in ctx["h284_manifest"]["cells"]:
        bag.append(
            [
                int(j)
                for j in fb.score_frozen_bagged_cell(
                    ctx["h284_dir"], cell, X, len(ids), 8
                )["BAG_SWITCH"]
            ]
        )
    out["BAG_SWITCH"] = (*ev.to_panel(V, ids, bag), bag, (stat.BOOT_SEED,) * 3)
    ctx.setdefault("clip_equals_bag_switch_by_seed", []).append(
        [a == b for a, b in zip(clip, bag)]
    )
    u, inv, pk = p2.panel(rt, freeze, rows, V, ctx["h278_cells"])
    out["RERANK8_VETO"] = (u, inv, [p["picks"] for p in pk], (stat.BOOT_SEED,) * 3)
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
    sums = {
        k: stat.summarize(u, inv, families, pk, bs)
        for k, (u, inv, pk, bs) in arms.items()
    }
    nulls = {
        k: sums[k] for k in ("NULL_RANDOM_VALID_SLOT", "NULL_RANDOM_SLOT_NO_LABELS")
    }
    verdicts, procedure = stat.holm(sums, nulls, len(set(families)))
    report = {}
    for k, s in sums.items():
        entry = dict(
            summary=stat.public(s), unadjusted_criteria=stat.criteria(s, 0.025)
        )
        hs = [set(x) for x in s["harmed_families_by_seed"]]
        always = sorted(set.intersection(*hs))
        groups = {}
        for f in always:
            groups[stat.source_group(f)] = groups.get(stat.source_group(f), 0) + 1
        entry["always_harmed"] = dict(
            count=len(always),
            by_source_group=groups,
            watch_rdataset=sum(stat.source_group(f).startswith(WATCH) for f in always),
            families=always,
        )
        report[k] = entry
    return report, verdicts, procedure


def main():
    ap = argparse.ArgumentParser()
    for a in (
        "fresh-dir",
        "gpu-dir",
        "h278-export-manifest",
        "h284-export-manifest",
        "h287-export-manifest",
        "h281-costs",
        "h281-panel",
        "h288-fresh",
        "h288-costs",
        "regression-h288",
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
        (H284, "h284_sources"),
        (H287, "h287_sources"),
        (H288, "h288_sources"),
    ):
        for name, digest in pins[table].items():
            if sha(root / name) != digest:
                raise RuntimeError("source pin mismatch: " + name)
    for path, key in (
        (args.h278_export_manifest, "export_manifest_sha256"),
        (args.h284_export_manifest, "h284_export_manifest_sha256"),
        (args.h287_export_manifest, "h287_export_manifest_sha256"),
    ):
        if sha(path) != pins[key]:
            raise RuntimeError("export manifest hash: " + key)
    if not json.loads(args.regression_h288.read_text())["bit_exact"]:
        raise RuntimeError("H288 cell regression must pass first")
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
    stat = load("h292_analysis", HERE / "analysis.py")
    sys.path.insert(0, str(H284))
    fb = load("h284_freeze_bagged", H284 / "freeze_bagged.py")
    h278_cells = p2.fetch_models(
        json.loads(args.h278_export_manifest.read_text()),
        Path("/home/ubuntu/jope-data/h292/export_h278"),
    )
    h284_manifest = json.loads(args.h284_export_manifest.read_text())
    h284_dir = Path("/home/ubuntu/jope-data/h292/export_h284")
    ev.fetch_h284(h284_manifest, h284_dir)
    h287_dir = Path("/home/ubuntu/jope-data/h292/export_h287")
    h287_spec = ev.fetch_h287(
        json.loads(args.h287_export_manifest.read_text()), h287_dir
    )
    sys.path.insert(0, str(H287))
    ens = load("h287_ensemble", H287 / "ensemble.py")
    ctx = dict(
        stat=stat,
        ev=ev,
        p2=p2,
        fb=fb,
        rt=rt,
        freeze=freeze,
        ens=ens,
        h278_cells=h278_cells,
        h284_manifest=h284_manifest,
        h284_dir=h284_dir,
        h287_dir=h287_dir,
        h287_spec=h287_spec,
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
        wave="H292",
        step="evaluate",
        regression_h281_panel_exact=regression,
        regression_h288_cell_exact=True,
        fresh_contexts=len(ids),
        fresh_families=len(set(families)),
        slot0_fresh_utility_mean=slot0_mean,
        oracle_top8_mean=float(np.mean([V[c]["oracle_top8_u"] for c in ids])),
    )
    try:
        report, verdicts, procedure = judge(stat, arms, families)
        result.update(verdicts=verdicts, holm=procedure, arms=report)
        result["clip_equals_bag_switch_by_seed"] = ctx[
            "clip_equals_bag_switch_by_seed"
        ][0]
    except Exception as error:  # packaging only: panels.npz is the result
        result["packaging_error"] = repr(error)
    try:
        r8, _, _, V8 = packets(args.h288_fresh, args.h288_costs)
        fam8 = [r["family_id"] for r in r8]
        if set(fam8) & set(families):
            raise RuntimeError("H288 and H292 families overlap")
        arms8 = run_arms(ctx, V8, r8)
        pooled = {k: (np.concatenate([arms8[k][0], arms[k][0]], axis=1), np.concatenate([arms8[k][1], arms[k][1]], axis=1),
                      [list(a) + list(b) for a, b in zip(arms8[k][2], arms[k][2])], arms[k][3]) for k in arms}  # fmt: skip
        prep, pver, pproc = judge(stat, pooled, fam8 + families)
        result["pooled_h288_h292"] = dict(
            label="pooled, H288 portion already seen (descriptive; 600 ODD families; every arm and null re-scored on H288's published packets)",
            families=len(set(fam8 + families)),
            holm_verdicts=pver,
            arms={
                k: {
                    kk: vv
                    for kk, vv in v["summary"].items()
                    if kk != "harmed_families_by_seed"
                }
                for k, v in prep.items()
            },
        )
    except Exception as error:
        result["pooled_error"] = repr(error)
    (out / "evaluate.json").write_text(
        json.dumps(result, indent=1, allow_nan=False) + "\n"
    )
    print("H292", result.get("verdicts"), flush=True)


if __name__ == "__main__":
    main()
