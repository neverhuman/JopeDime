"""H286 evaluation (xbabe2 CPU, hw_t2325). Pre-declared arms: RERANK8_VETO (H278 frozen; the primary hypothesis), and, because
rd_argon's H284 export manifest was on origin/v1/next at prereg time, BAG_SWITCH and BAG_SWITCH_NESTED (H284 frozen,
scored with H284's own score_frozen_bagged_cell; every file fetched by S3 VersionId, sha256 enforced). Each arm gets the
same rule and its own verdict. Otherwise H283's evaluation unchanged in method (null seeds 286101-286103, bootstrap
seed 286150, 80-family floor), plus the pre-declared SECONDARY measures, the POOLED H283+H286 descriptive readout
("pooled, H283 portion already seen") and the DESCRIPTIVE r782/r785 distinct-pool readouts. H283 docstring follows.

H283 evaluation (xbabe2 CPU, hw_t2305): the three frozen H278 RERANK8_VETO models (no refit; S3 VersionId, sha256
enforced) and the SEEDED_RANDOM_VALID_SLOT null against r3175's own slot-0 pick on the fresh cohort's top-8 pools.

H278 code is imported unchanged (sha-pinned): topk.load_packets builds each top-8 packet from the fresh GPU costs with
the fresh context features built by build_fresh.py (H58's own functions); freeze.frozen_picks applies each model.
H281's fetch_models and panel are reused unchanged. Regression first: H281's r3175 GPU panel, bit-exact."""

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
H284 = HERE.parent / "h284_bagswitch_frozen_fresh"
SEEDS_NULL = (286101, 286102, 286103)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def priors_from(root, rows):
    """H278 runner.priors_for, reading the same two cache files from the fresh build directory."""
    ids = {r["context_sha256"] for r in rows}
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(root / "pool_priors.npz", allow_pickle=False) as zp:
        cids, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        for i, cid in enumerate(cids):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[cid] = zp["base_prior_cost"][lo:hi].astype(np.float32).copy()
                proj16[cid] = zp["proj16"][lo:hi].astype(np.float32).copy()
    with np.load(root / "context_features.npz", allow_pickle=False) as zc:
        for i, cid in enumerate(str(c) for c in zc["cids"]):
            if cid in ids:
                ctxf[cid] = dict(
                    lm48_z=zc["lm48_z"][i].astype(np.float32).copy(),
                    dope629=zc["dope629"][i].astype(np.float32).copy(),
                )
    return aprior, proj16, ctxf


def bag_arms(args, pins, rt, V, ids, families, stat, null_summary):
    """BAG_SWITCH and BAG_SWITCH_NESTED from H284's frozen export, applied unchanged to the same top-8 features."""
    import subprocess

    if sha(args.h284_export_manifest) != pins["h284_export_manifest_sha256"]:
        raise RuntimeError("H284 export manifest hash")
    man = json.loads(args.h284_export_manifest.read_text())
    edir = Path("/home/ubuntu/jope-data/h286/export_h284")
    edir.mkdir(parents=True, exist_ok=True)
    for cell in man["cells"]:
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
    sys.path.insert(0, str(H284))
    fb = load("h284_freeze_bagged", H284 / "freeze_bagged.py")
    X = np.concatenate([V[c]["X"] for c in ids])
    out = {}
    per = {"BAG_SWITCH": [], "BAG_SWITCH_NESTED": []}
    for cell in man["cells"]:
        res = fb.score_frozen_bagged_cell(edir, cell, X, len(ids), 8)
        for arm in per:
            per[arm].append(res[arm])
    for arm, picks in per.items():
        u = np.zeros((3, len(ids), 2))
        inv = np.zeros((3, len(ids), 2), dtype=bool)
        for s_, pk in enumerate(picks):
            for i, (c, j) in enumerate(zip(ids, pk)):
                u[s_, i] = [V[c]["utility_k"][0], V[c]["utility_k"][j]]
                inv[s_, i] = [not V[c]["gate_k"][0], not V[c]["gate_k"][j]]
        sm = stat.summarize(u, inv, families, [list(map(int, pk)) for pk in picks])
        out[arm] = dict(
            seeds=[c["seed"] for c in man["cells"]],
            summary=sm,
            verdict=stat.verdict(len(set(families)), sm, null_summary),
            secondary=stat.secondary(u, families, [V[c]["oracle_top8_u"] for c in ids]),
            picks=[list(map(int, pk)) for pk in picks],
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument("--export-manifest", type=Path, required=True)
    ap.add_argument("--h281-costs", type=Path, required=True)
    ap.add_argument("--h281-panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--h283-panels", type=Path, required=True)
    ap.add_argument("--h284-export-manifest", type=Path, required=True)
    ap.add_argument("--h283-fresh", type=Path, required=True)
    ap.add_argument(
        "--descriptive",
        nargs="*",
        default=[],
        help="name=gpu_attempt_dir for r782/r785",
    )
    args = ap.parse_args()
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = args.out.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    for root, table in ((H278, "h278_sources"), (H281, "h281_sources")):
        for name, digest in pins[table].items():
            if sha(root / name) != digest:
                raise RuntimeError("source pin mismatch: " + name)
    if sha(args.export_manifest) != pins["export_manifest_sha256"]:
        raise RuntimeError("export manifest hash")
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
    stat = load("h286_analysis", HERE / "analysis.py")
    cells = p2.fetch_models(
        json.loads(args.export_manifest.read_text()),
        Path("/home/ubuntu/jope-data/h286/export"),
    )

    # Regression: H281's r3175 GPU panel on DL-Val91, bit-exact (H50 pools, H58 cache).
    topk.core.INPUT_POOLS_DIR = rt.POOLS
    val = json.loads((rt.POOLS / "selected.json").read_text())["dev"]
    vap, vpj, vcf = rt.priors_for(val)
    with np.load(args.h281_costs) as z:
        c81 = {
            r["context_sha256"]: z[r["context_sha256"]].astype(np.float32) for r in val
        }
    u81, inv81, _ = p2.panel(
        rt, freeze, val, topk.load_packets(val, c81, vap, vpj, vcf, 8), cells
    )
    with np.load(args.h281_panel) as z:
        regression = bool(
            np.array_equal(u81, z["utility"]) and np.array_equal(inv81, z["invalid"])
        )
    if not regression:
        raise RuntimeError("H281 panel regression failed")

    # Fresh cohort.
    topk.core.INPUT_POOLS_DIR = args.fresh_dir
    rows = json.loads((args.fresh_dir / "selected.json").read_text())["fresh"]
    ids = [r["context_sha256"] for r in rows]
    families = [r["family_id"] for r in rows]
    fap, fpj, fcf = priors_from(args.fresh_dir, rows)
    with np.load(args.gpu_dir / "fresh-r3175-gpu-costs.npz") as z:
        costs = {c: z[c].astype(np.float32) for c in ids}
    V = topk.load_packets(rows, costs, fap, fpj, fcf, 8)
    slot0_mean = float(np.mean([V[c]["utility_k"][0] for c in ids]))
    if abs(slot0_mean - rec["fresh"]["slot0_mean"]) > 1e-12:
        raise RuntimeError("slot 0 differs from the GPU argmin utilities")
    u, inv, picks = p2.panel(rt, freeze, rows, V, cells)
    nu = np.zeros_like(u)
    ninv = np.zeros_like(inv)
    npicks = []
    for s, seed in enumerate(SEEDS_NULL):
        pk = stat.random_valid_slot_picks([V[c]["gate_k"] for c in ids], seed)
        npicks.append(pk)
        for i, (c, j) in enumerate(zip(ids, pk)):
            nu[s, i] = [V[c]["utility_k"][0], V[c]["utility_k"][j]]
            ninv[s, i] = [not V[c]["gate_k"][0], not V[c]["gate_k"][j]]
    np.savez_compressed(
        out / "fresh_panels.npz",
        frozen_utility=u,
        frozen_invalid=inv,
        null_utility=nu,
        null_invalid=ninv,
    )
    np.savez_compressed(
        out / "fresh_top8_packets.npz",
        cids=np.asarray(ids),
        X=np.stack([V[c]["X"] for c in ids]),
        costs_top8=np.stack([np.sort(costs[c], kind="mergesort")[:8] for c in ids]),
        order_top8=np.stack([np.argsort(costs[c], kind="mergesort")[:8] for c in ids]),
    )
    (out / "fresh_picks.json").write_text(
        json.dumps(dict(frozen=picks, null=npicks)) + "\n"
    )
    result = dict(
        wave="H286",
        step="evaluate",
        regression_h281_panel_exact=regression,
        fresh_contexts=len(ids),
        fresh_families=len(set(families)),
        slot0_fresh_utility_mean=slot0_mean,
        oracle_top8_mean=float(np.mean([V[c]["oracle_top8_u"] for c in ids])),
    )
    try:
        fz = stat.summarize(u, inv, families, [p["picks"] for p in picks])
        nl = stat.summarize(nu, ninv, families, npicks)
        result.update(
            frozen=fz, null=nl, verdict=stat.verdict(len(set(families)), fz, nl)
        )
        try:
            result["arms"] = bag_arms(args, pins, rt, V, ids, families, stat, nl)
        except Exception as error:  # an arm failure is recorded; it cannot affect the RERANK8_VETO verdict
            result["arms_error"] = repr(error)
        result["secondary"] = stat.secondary(
            u, families, [V[c]["oracle_top8_u"] for c in ids]
        )
        with np.load(args.h283_panels) as z83:
            rows83 = json.loads(args.h283_fresh.read_text())["fresh"]
            fam83 = [r["family_id"] for r in rows83]
            pu = np.concatenate([z83["frozen_utility"], u], axis=1)
            pinv = np.concatenate([z83["frozen_invalid"], inv], axis=1)
            nu2 = np.concatenate([z83["null_utility"], nu], axis=1)
            ninv2 = np.concatenate([z83["null_invalid"], ninv], axis=1)
        if set(fam83) & set(families):
            raise RuntimeError("H283 and H286 families overlap")
        pf = stat.summarize(pu, pinv, fam83 + families, [[0] * pu.shape[1]] * 3)
        pn = stat.summarize(nu2, ninv2, fam83 + families, [[0] * pu.shape[1]] * 3)
        result["pooled_h283_h286"] = dict(
            label="pooled, H283 portion already seen (descriptive; H283 null seeds 283101-3, H286 null seeds 286101-3)",
            families=len(set(fam83 + families)),
            frozen=pf,
            null=pn,
            rule_verdict=stat.verdict(len(set(fam83 + families)), pf, pn),
        )
        result["descriptive_distinct_pools"] = {}
        for spec in args.descriptive:
            name, gdir = spec.split("=", 1)
            gdir = Path(gdir)
            grec = (
                json.loads((gdir / "rescore.json").read_text())
                if (gdir / "rescore.json").exists()
                else {}
            )
            if (
                grec.get("outcome") != "A_EXACT"
                or grec.get("fresh", {}).get("status") != "COMPLETE"
            ):
                result["descriptive_distinct_pools"][name] = dict(
                    status="NOT_AVAILABLE", outcome=grec.get("outcome")
                )
                continue
            with np.load(gdir / "fresh-r3175-gpu-costs.npz") as z:
                dc = {c: z[c].astype(np.float32) for c in ids}
            Vd = topk.load_packets(rows, dc, fap, fpj, fcf, 8)
            ud, invd, pkd = p2.panel(rt, freeze, rows, Vd, cells)
            np.savez_compressed(
                out / f"descriptive_{name}_panel.npz", utility=ud, invalid=invd
            )
            result["descriptive_distinct_pools"][name] = dict(
                label="DESCRIPTIVE (not part of the verdict): frozen VETO vs this checkpoint's own slot 0",
                dl_val91_regression=dict(
                    gpu_mean=grec["gpu_mean"], stored_mean=grec["stored_mean"]
                ),
                slot0_mean=float(np.mean([Vd[c]["utility_k"][0] for c in ids])),
                frozen=stat.summarize(ud, invd, families, [pp["picks"] for pp in pkd]),
            )
    except Exception as error:  # packaging only: fresh_panels.npz is the result
        result.update(packaging_error=repr(error))
    (out / "evaluate.json").write_text(
        json.dumps(result, indent=1, allow_nan=False) + "\n"
    )
    print("H286", result.get("verdict"), flush=True)


if __name__ == "__main__":
    main()
