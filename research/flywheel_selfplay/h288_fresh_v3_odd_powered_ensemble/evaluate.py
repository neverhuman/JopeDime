"""H288 evaluation (xbabe2 CPU, hw_t0005) on fresh cohort v3 (300 ODD families), GPU-exact r3175 costs.

Arms (all frozen, no refit, every file fetched by S3 VersionId with sha256 enforced):
  PRIMARY    BAG_SWITCH, BAG_SWITCH_NESTED (rd_argon's H284 export, seeds 279101-279103; H284's own scorer);
  SECONDARY  RERANK8_VETO (H278 export, seeds 278101-278103; H278 freeze.frozen_picks via H281's panel);
  H287 (rd_argon; export on origin/v1/next at 213b30ad, 00:47:46Z, before the prereg): ENS_MEAN, ENS_VOTE2, ENS_VOTE3
  (deterministic single policies: "3 seeds" = bootstrap seeds 288150/288151/288152 on the same picks) and CLIP (3 refit
  seeds), scored with rd_argon's own ensemble.score_all_h287_arms.
Nulls: SEEDED_RANDOM_VALID_SLOT (288101-288103) and SEEDED_RANDOM_SLOT_NO_LABELS (288201-288203); both must fail.
H278's topk builds every top-8 packet. Regression first: H281's r3175 GPU panel, bit-exact. Pooled (descriptive): the
same arms re-scored on H286's EVEN v2 contexts (its RERANK8_VETO panel must be reproduced exactly) plus H288."""

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
H284 = HERE.parent / "h284_bagswitch_frozen_fresh"
H287 = HERE.parent / "h287_bagswitch_seed_ensemble_export"
H287_ARMS_DET = ("ENS_MEAN", "ENS_VOTE2", "ENS_VOTE3")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fetch_h284(manifest, edir):
    edir.mkdir(parents=True, exist_ok=True)
    for cell in manifest["cells"]:
        for info in [cell["ranker"], cell["veto"], *cell["bags"]]:
            path = edir / info["file"]
            if not path.exists():
                subprocess.run(
                    ["aws", "s3api", "get-object", "--bucket", info["S3_bucket"], "--key", info["S3_key"],
                     "--version-id", info["S3_VersionId"], str(path)],
                    check=True, stdout=subprocess.DEVNULL,
                )  # fmt: skip
            if sha(path) != info["sha256"]:
                raise RuntimeError("H284 model sha256 mismatch: " + info["file"])


def get(info, path):
    if not path.exists():
        subprocess.run(
            ["aws", "s3api", "get-object", "--bucket", info["S3_bucket"], "--key", info["S3_key"],
             "--version-id", info["S3_VersionId"], str(path)],
            check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
    if sha(path) != info["sha256"]:
        raise RuntimeError("H287 file sha256 mismatch: " + path.name)


def fetch_h287(manifest, edir):
    """The ensemble spec and every member file (ranker, veto, 8 bags; 6 members) by S3 key + VersionId, sha256 enforced."""
    edir.mkdir(parents=True, exist_ok=True)
    get(manifest["ensemble_spec"], edir / manifest["ensemble_spec"]["file"])
    spec = json.loads((edir / manifest["ensemble_spec"]["file"]).read_text())
    for m in spec["h284_bag_switch_members"] + spec["clip_bag_switch_members"]:
        for info in [m["ranker"], m["veto"], *m["bags"]]:
            get(info, edir / info["file"])
    return spec


def to_panel(V, ids, picks):
    u = np.zeros((3, len(ids), 2))
    inv = np.zeros((3, len(ids), 2), dtype=bool)
    for s, pk in enumerate(picks):
        for i, (c, j) in enumerate(zip(ids, pk)):
            u[s, i] = [V[c]["utility_k"][0], V[c]["utility_k"][int(j)]]
            inv[s, i] = [not V[c]["gate_k"][0], not V[c]["gate_k"][int(j)]]
    return u, inv


def run_arms(ctx, V, rows):
    """All arms and both nulls on one set of top-8 packets -> {name: (u, inv, picks, boot_seeds)}."""
    stat, p2, fb, rt, freeze = (
        ctx["stat"],
        ctx["p2"],
        ctx["fb"],
        ctx["rt"],
        ctx["freeze"],
    )
    ids = [r["context_sha256"] for r in rows]
    out = {}
    u, inv, pk = p2.panel(rt, freeze, rows, V, ctx["h278_cells"])
    out["RERANK8_VETO"] = (u, inv, [p["picks"] for p in pk], (stat.BOOT_SEED,) * 3)
    X = np.concatenate([V[c]["X"] for c in ids])
    bag = {"BAG_SWITCH": [], "BAG_SWITCH_NESTED": []}
    for cell in ctx["h284_manifest"]["cells"]:
        res = fb.score_frozen_bagged_cell(ctx["h284_dir"], cell, X, len(ids), 8)
        for arm in bag:
            bag[arm].append([int(j) for j in res[arm]])
    for arm, picks in bag.items():
        out[arm] = (*to_panel(V, ids, picks), picks, (stat.BOOT_SEED,) * 3)
    if ctx.get("ens") is not None:
        r = ctx["ens"].score_all_h287_arms(
            ctx["h287_dir"], X, len(ids), 8, ctx["h287_spec"]
        )
        for arm in H287_ARMS_DET:
            picks = [[int(j) for j in r[arm]]] * 3
            out[arm] = (*to_panel(V, ids, picks), picks, stat.DETERMINISTIC_BOOT_SEEDS)
        clip = [[int(j) for j in p] for p in r["per_seed"]["CLIP"]]
        out["CLIP"] = (*to_panel(V, ids, clip), clip, (stat.BOOT_SEED,) * 3)
        ctx.setdefault("h287_bag_switch_equals_h284", []).append(
            [[int(j) for j in p] for p in r["per_seed"]["BAG_SWITCH"]]
            == bag["BAG_SWITCH"]
        )
    vp = [
        stat.random_valid_slot_picks([V[c]["gate_k"] for c in ids], s)
        for s in stat.VALID_NULL_SEEDS
    ]
    out["NULL_RANDOM_VALID_SLOT"] = (*to_panel(V, ids, vp), vp, (stat.BOOT_SEED,) * 3)
    npk = [
        stat.random_slot_no_label_picks(len(ids), s) for s in stat.NOLABEL_NULL_SEEDS
    ]
    out["NULL_RANDOM_SLOT_NO_LABELS"] = (
        *to_panel(V, ids, npk),
        npk,
        (stat.BOOT_SEED,) * 3,
    )
    return out


def judge(stat, arms, families, oracle):
    sums = {
        k: stat.summarize(u, inv, families, pk, bs)
        for k, (u, inv, pk, bs) in arms.items()
    }
    nulls = [sums["NULL_RANDOM_VALID_SLOT"], sums["NULL_RANDOM_SLOT_NO_LABELS"]]
    res = {}
    for k, s in sums.items():
        entry = dict(summary=stat.public(s))
        if not k.startswith("NULL_"):
            entry["verdict"] = stat.verdict(len(set(families)), s, nulls)
            if oracle is not None:
                entry["secondary"] = stat.secondary(s, arms[k][0], families, oracle)
        res[k] = entry
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument("--h278-export-manifest", type=Path, required=True)
    ap.add_argument("--h284-export-manifest", type=Path, required=True)
    ap.add_argument("--h287-export-manifest", type=Path, required=True)
    ap.add_argument("--h281-costs", type=Path, required=True)
    ap.add_argument("--h281-panel", type=Path, required=True)
    ap.add_argument("--h286-fresh", type=Path, required=True)
    ap.add_argument("--h286-costs", type=Path, required=True)
    ap.add_argument("--h286-panels", type=Path, required=True)
    ap.add_argument("--h286-always-harmed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = args.out.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    for root, table in (
        (H278, "h278_sources"),
        (H281, "h281_sources"),
        (H283, "h283_sources"),
        (H284, "h284_sources"),
        (H287, "h287_sources"),
    ):
        for name, digest in pins[table].items():
            if sha(root / name) != digest:
                raise RuntimeError("source pin mismatch: " + name)
    if sha(args.h278_export_manifest) != pins["export_manifest_sha256"]:
        raise RuntimeError("H278 export manifest hash")
    if sha(args.h284_export_manifest) != pins["h284_export_manifest_sha256"]:
        raise RuntimeError("H284 export manifest hash")
    if sha(args.h287_export_manifest) != pins["h287_export_manifest_sha256"]:
        raise RuntimeError("H287 export manifest hash")
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
    stat = load("h288_analysis", HERE / "analysis.py")
    sys.path.insert(0, str(H284))
    fb = load("h284_freeze_bagged", H284 / "freeze_bagged.py")
    h278_cells = p2.fetch_models(
        json.loads(args.h278_export_manifest.read_text()),
        Path("/home/ubuntu/jope-data/h288/export_h278"),
    )
    h284_manifest = json.loads(args.h284_export_manifest.read_text())
    h284_dir = Path("/home/ubuntu/jope-data/h288/export_h284")
    fetch_h284(h284_manifest, h284_dir)
    ctx = dict(
        stat=stat,
        p2=p2,
        fb=fb,
        rt=rt,
        freeze=freeze,
        h278_cells=h278_cells,
        h284_manifest=h284_manifest,
        h284_dir=h284_dir,
    )
    ctx["h287_dir"] = Path("/home/ubuntu/jope-data/h288/export_h287")
    ctx["h287_spec"] = fetch_h287(
        json.loads(args.h287_export_manifest.read_text()), ctx["h287_dir"]
    )
    sys.path.insert(0, str(H287))
    ctx["ens"] = load("h287_ensemble", H287 / "ensemble.py")

    # Regression: H281's r3175 GPU panel on DL-Val91, bit-exact.
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
    oracle = [V[c]["oracle_top8_u"] for c in ids]
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
        wave="H288",
        step="evaluate",
        regression_h281_panel_exact=regression,
        fresh_contexts=len(ids),
        fresh_families=len(set(families)),
        slot0_fresh_utility_mean=slot0_mean,
        oracle_top8_mean=float(np.mean(oracle)),
    )
    try:
        res = judge(stat, arms, families, oracle)
        result["arms"] = res
        result["primary_verdicts"] = {
            k: res[k]["verdict"] for k in ("BAG_SWITCH", "BAG_SWITCH_NESTED")
        }
        result["secondary_replication_verdict"] = res["RERANK8_VETO"]["verdict"]
        result["h287_verdicts"] = {
            k: res[k]["verdict"] for k in (*H287_ARMS_DET, "CLIP") if k in res
        }
        result["h287_bag_switch_equals_h284"] = ctx.get("h287_bag_switch_equals_h284")
        h286_always = set(json.loads(args.h286_always_harmed.read_text()))

        def kind(f):
            k = f.removeprefix("fresh:")
            return (
                k.split(":")[0] + ":" + k.split(":")[1][:8]
                if ":" in k
                else "dedupe_fingerprint"
            )

        overlap = {}
        for arm in ("BAG_SWITCH", "BAG_SWITCH_NESTED", "RERANK8_VETO"):
            hs = [set(x) for x in res[arm]["summary"]["harmed_families_by_seed"]]
            always = sorted(set.intersection(*hs))
            overlap[arm] = dict(
                always_harmed=len(always),
                by_source_type={
                    t: sum(kind(f) == t for f in always)
                    for t in sorted({kind(f) for f in always})
                },
                h286_always_by_source_type={
                    t: sum(kind(f) == t for f in h286_always)
                    for t in sorted({kind(f) for f in h286_always})
                },
            )
        result["harmed_overlap_with_h286"] = dict(
            label="descriptive; H286 and H288 families are disjoint, so overlap is by source type",
            arms=overlap,
        )
    except Exception as error:  # packaging only: panels.npz is the result
        result["packaging_error"] = repr(error)
    try:
        r6, i6, _, V6 = packets(args.h286_fresh, args.h286_costs)
        fam6 = [r["family_id"] for r in r6]
        arms6 = run_arms(ctx, V6, r6)
        with np.load(args.h286_panels) as z:
            reg6 = bool(
                np.array_equal(arms6["RERANK8_VETO"][0], z["frozen_utility"])
                and np.array_equal(arms6["RERANK8_VETO"][1], z["frozen_invalid"])
            )
        if set(fam6) & set(families):
            raise RuntimeError("H286 and H288 families overlap")
        pooled = {k: (np.concatenate([arms6[k][0], arms[k][0]], axis=1), np.concatenate([arms6[k][1], arms[k][1]], axis=1),
                      [list(a) + list(b) for a, b in zip(arms6[k][2], arms[k][2])], arms[k][3]) for k in arms}  # fmt: skip
        pres = judge(stat, pooled, fam6 + families, None)
        result["pooled_h286_h288"] = dict(
            label="pooled, H286 portion already seen (descriptive; every arm and null re-scored on H286's packets)",
            h286_veto_panel_reproduced=reg6,
            families=len(set(fam6 + families)),
            verdicts={k: v.get("verdict") for k, v in pres.items()},
            arms={
                k: {
                    kk: vv
                    for kk, vv in v["summary"].items()
                    if kk != "harmed_families_by_seed"
                }
                for k, v in pres.items()
            },
        )
    except Exception as error:
        result["pooled_error"] = repr(error)
    (out / "evaluate.json").write_text(
        json.dumps(result, indent=1, allow_nan=False) + "\n"
    )
    print(
        "H288",
        result.get("primary_verdicts"),
        result.get("secondary_replication_verdict"),
        flush=True,
    )


if __name__ == "__main__":
    main()
