"""H282 step 2 (xbabe2 CPU, hw_t2235): apply the three frozen H278 RERANK8_VETO models (no refit) to the DL-Val91
top-8 pools of the retained checkpoints whose GPU re-score reproduced their stored utility exactly.

H278's code is imported unchanged (sha-pinned): topk.load_packets builds each top-8 packet from the checkpoint's GPU
costs, freeze.frozen_picks applies each frozen model. H281's p2.py supplies fetch_models (S3 VersionId + export-manifest
sha256) and panel, unchanged. Regression first: H281's r3175 GPU panel must be reproduced bit-exactly."""

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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpu-dir",
        type=Path,
        required=True,
        help="attempt dir with r<N>/rescore.json + costs",
    )
    ap.add_argument(
        "--stored-dir", type=Path, required=True, help="round-<N>.json copies"
    )
    ap.add_argument("--export-manifest", type=Path, required=True)
    ap.add_argument("--h281-costs", type=Path, required=True)
    ap.add_argument("--h281-panel", type=Path, required=True)
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
    for root, table in ((H278, "h278_sources"), (H281, "h281_sources")):
        for name, digest in pins[table].items():
            if sha(root / name) != digest:
                raise RuntimeError("source pin mismatch: " + name)
    if sha(args.export_manifest) != pins["export_manifest_sha256"]:
        raise RuntimeError("export manifest hash")
    manifest = json.loads(args.export_manifest.read_text())
    sys.path.insert(0, str(H278))
    import freeze
    import runner as rt
    import topk

    sys.addaudithook(rt.audit_hook)
    rt.load_core()
    p2 = load("h281_p2", H281 / "p2.py")
    stat = load("h282_analysis", HERE / "analysis.py")
    val = json.loads((rt.POOLS / "selected.json").read_text())["dev"]
    vids = [r["context_sha256"] for r in val]
    families = [r["family_id"] for r in val]
    cells = p2.fetch_models(manifest, Path("/home/ubuntu/jope-data/h282/export"))
    vap, vpj, vcf = rt.priors_for(val)

    def costs_of(path):
        with np.load(path) as z:
            return {c: z[c].astype(np.float32) for c in vids}

    # Regression: H281's r3175 GPU panel, bit-exact.
    u81, inv81, _ = p2.panel(
        rt,
        freeze,
        val,
        topk.load_packets(val, costs_of(args.h281_costs), vap, vpj, vcf, 8),
        cells,
    )
    with np.load(args.h281_panel) as z:
        regression = bool(
            np.array_equal(u81, z["utility"]) and np.array_equal(inv81, z["invalid"])
        )
    if not regression:
        raise RuntimeError("H281 panel regression failed")
    included, excluded, grid, panels, picks_all = [], [], {}, {}, {}
    for n in pins["eligible_rounds"]:
        d = args.gpu_dir / f"r{n}"
        rec = (
            json.loads((d / "rescore.json").read_text())
            if (d / "rescore.json").exists()
            else {}
        )
        if rec.get("status") != "COMPLETE" or rec.get("outcome") != "A_EXACT":
            excluded.append(
                dict(
                    round=n,
                    status=rec.get("status", "NOT_RUN"),
                    outcome=rec.get("outcome"),
                    abs_mean_residual=rec.get("abs_mean_residual"),
                    per_context_equal=rec.get("per_context_equal"),
                    picks_equal=rec.get("picks_equal"),
                )
            )
            continue
        if sha(d / "r3175-gpu-costs.npz") != rec["costs_sha256"]:
            raise RuntimeError(f"r{n} costs hash")
        stored = json.loads((args.stored_dir / f"round-{n}.json").read_text())[
            "dev_utility"
        ]
        V = topk.load_packets(
            val, costs_of(d / "r3175-gpu-costs.npz"), vap, vpj, vcf, 8
        )
        slot0 = int(
            sum(
                float(V[c]["utility_k"][0]) == s
                for c, s in zip(vids, stored["per_context"])
            )
        )
        if slot0 != 91:
            raise RuntimeError(f"r{n} slot 0 differs from stored utilities")
        u, inv, picks = p2.panel(rt, freeze, val, V, cells)
        key = f"r{n}"
        included.append(n)
        panels[key] = (u, inv)
        picks_all[key] = picks
        grid[key] = [
            stat.cell(u[s], inv[s], families, picks[s]["picks"]) for s in range(3)
        ]
    np.savez_compressed(
        out / "panels.npz",
        **{k + "_utility": v[0] for k, v in panels.items()},
        **{k + "_invalid": v[1] for k, v in panels.items()},
    )
    (out / "picks.json").write_text(json.dumps(picks_all) + "\n")
    result = dict(
        wave="H282",
        step="transfer",
        regression_h281_panel_exact=regression,
        included=included,
        excluded=excluded,
    )
    try:
        prim = stat.primary(grid)
        result.update(
            primary=prim,
            verdict=prim["verdict"],
            per_checkpoint={
                k: dict(seeds=[stat.public(c) for c in v], **stat.checkpoint_summary(v))
                for k, v in grid.items()
            },
            secondary=dict(
                plateau=stat.PLATEAU,
                reranked_context_mean_median_over_seeds={
                    k: float(np.median([c["context_mean"][1] for c in v]))
                    for k, v in grid.items()
                },
                slot0_context_mean={
                    k: v[0]["context_mean"][0] for k, v in grid.items()
                },
            ),
        )
        if grid:
            best = max(
                result["secondary"]["reranked_context_mean_median_over_seeds"].items(),
                key=lambda kv: kv[1],
            )
            result["secondary"]["best_reranked"] = dict(
                checkpoint=best[0], mean=best[1], minus_plateau=best[1] - stat.PLATEAU
            )
    except Exception as error:  # packaging only: panels.npz is the result
        result.update(packaging_error=repr(error))
    (out / "transfer.json").write_text(
        json.dumps(result, indent=1, allow_nan=False) + "\n"
    )
    print(
        "H282",
        result.get("verdict"),
        "included",
        included,
        "excluded",
        [e["round"] for e in excluded],
        flush=True,
    )


if __name__ == "__main__":
    main()
