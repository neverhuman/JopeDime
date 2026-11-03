"""H281 step 2 (xbabe2 CPU): H278's preregistered P2, unchanged, on the GPU-reproduced r3175 top-8 pool (hw_t2205).

Runs only after gpu_rescore.py reports Outcome A (exact). Imports the H278 study code byte-identical from origin/v1/next
(runner.py, topk.py, h265_core.py, veto.py, freeze.py, analysis.py; sha256 pinned in manifest.json), downloads the
three frozen RERANK8_VETO models from the H278 S3 export by VersionId (sha256 checked against export_manifest.json; no
refit), rebuilds the r3175 top-8 packet from the GPU costs with H278's own loader, and applies H278's P2 rule:
ADOPT_SHADOW_PASS iff gain > 0 on 3/3 seeds, pooled family-bootstrap (10k, seed 278150) LCB95 > 0 and invalid picks not
above r3175 (ties never wins); else NOT_TRANSFERRED.

Regression first: the same code on H278's CPU r3175 costs must reproduce H278's saved exploratory P2 panel exactly."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H278 = HERE.parent / "h278_rerank8_veto_frozen_export_transfer"
BUCKET = "veox-jopedime-learning-use2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_models(manifest, dest):
    """Download each frozen ranker/veto by S3 key + VersionId; enforce the export manifest's sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for c in manifest["cells"]:
        files = {}
        for part in ("ranker", "veto"):
            m = c[part]
            path = dest / m["file"]
            if not path.exists():
                subprocess.run(
                    [
                        "aws",
                        "s3api",
                        "get-object",
                        "--bucket",
                        BUCKET,
                        "--key",
                        m["S3_key"],
                        "--version-id",
                        m["S3_VersionId"],
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            if sha(path) != m["sha256"]:
                raise RuntimeError("exported model sha256 mismatch: " + m["file"])
            files[part] = path.read_text()
        cells.append(
            dict(
                seed=c["seed"],
                n_est=c["selected_n_estimators"],
                margin=float("inf")
                if c["strict_margin"] == "inf"
                else float(c["strict_margin"]),
                **files,
            )
        )
    return cells


def panel(rt, freeze, val, V, cells):
    vids = [r["context_sha256"] for r in val]
    X = np.concatenate([V[c]["X"] for c in vids])
    u = np.zeros((3, len(vids), 2))
    inv = np.zeros((3, len(vids), 2), dtype=bool)
    picks_out = []
    for si, c in enumerate(cells):
        picks, _, _ = freeze.frozen_picks(
            c["ranker"], c["veto"], X, len(vids), c["n_est"], c["margin"]
        )
        for i, (cid, j) in enumerate(zip(vids, picks)):
            u[si, i] = [V[cid]["utility_k"][0], V[cid]["utility_k"][j]]
            inv[si, i] = [not V[cid]["gate_k"][0], not V[cid]["gate_k"][j]]
        picks_out.append(
            dict(
                seed=c["seed"],
                picks=[int(j) for j in picks],
                changed=int(sum(j != 0 for j in picks)),
            )
        )
    return u, inv, picks_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-costs", type=Path, required=True)
    ap.add_argument("--rescore", type=Path, required=True)
    ap.add_argument("--export-manifest", type=Path, required=True)
    ap.add_argument("--h278-panel", type=Path, required=True)
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
    for name, digest in pins["h278_sources"].items():
        if sha(H278 / name) != digest:
            raise RuntimeError("H278 source pin mismatch: " + name)
    rescore = json.loads(args.rescore.read_text())
    if rescore.get("status") != "COMPLETE" or rescore.get("outcome") != "A_EXACT":
        raise RuntimeError("P2 opens only on Outcome A (exact GPU re-score)")
    if sha(args.gpu_costs) != rescore["costs_sha256"]:
        raise RuntimeError("GPU costs hash")
    manifest = json.loads(args.export_manifest.read_text())
    if sha(args.export_manifest) != pins["export_manifest_sha256"]:
        raise RuntimeError("export manifest hash")
    sys.path.insert(0, str(H278))
    import analysis as stat
    import freeze
    import runner as rt
    import topk

    sys.addaudithook(rt.audit_hook)
    rt.load_core()
    selected = json.loads((rt.POOLS / "selected.json").read_text())
    val = selected["dev"]
    vids = [r["context_sha256"] for r in val]
    cells = fetch_models(manifest, Path("/home/ubuntu/jope-data/h281/export"))
    vap, vpj, vcf = rt.priors_for(val)
    # Regression: H278's own CPU r3175 costs must reproduce H278's saved exploratory P2 panel bit-exactly.
    with np.load(rt.INPUTS / "r3175-secondary-costs.npz") as z3:
        cpu = {c: z3[c].astype(np.float32) for c in vids}
    u_cpu, inv_cpu, _ = panel(
        rt, freeze, val, topk.load_packets(val, cpu, vap, vpj, vcf, 8), cells
    )
    with np.load(args.h278_panel) as zp:
        regression = bool(
            np.array_equal(u_cpu, zp["P2_r3175_utility"])
            and np.array_equal(inv_cpu, zp["P2_r3175_invalid"])
        )
    if not regression:
        raise RuntimeError("H278 P2 panel regression failed")
    with np.load(args.gpu_costs) as zg:
        gpu = {c: zg[c].astype(np.float32) for c in vids}
    V = topk.load_packets(val, gpu, vap, vpj, vcf, 8)
    top8_cpu = {c: np.argsort(cpu[c], kind="mergesort")[:8] for c in vids}
    top8_gpu = {c: np.argsort(gpu[c], kind="mergesort")[:8] for c in vids}
    pool_diff = [
        dict(
            index=i,
            cid=c,
            cpu_top8=top8_cpu[c].tolist(),
            gpu_top8=top8_gpu[c].tolist(),
        )
        for i, c in enumerate(vids)
        if not np.array_equal(top8_cpu[c], top8_gpu[c])
    ]
    incumbent = [float(V[c]["utility_k"][0]) for c in vids]
    stored = json.loads((rt.INPUTS / "r3175" / "round-3175.json").read_text())[
        "dev_utility"
    ]
    slot0_equals_stored = int(
        sum(a == b for a, b in zip(incumbent, stored["per_context"]))
    )
    if slot0_equals_stored != len(vids):
        raise RuntimeError("slot 0 differs from the stored r3175 utilities")
    u, inv, picks = panel(rt, freeze, val, V, cells)
    np.savez_compressed(out / "p2_panel.npz", utility=u, invalid=inv)
    (out / "p2_picks.json").write_text(json.dumps(picks) + "\n")
    result = dict(
        wave="H281",
        step="p2",
        regression_h278_p2_panel_exact=regression,
        slot0_equals_stored=slot0_equals_stored,
        top8_pool_diff_vs_cpu=pool_diff,
        max_abs_cost_diff_gpu_vs_cpu=float(
            max(np.max(np.abs(gpu[c] - cpu[c])) for c in vids)
        ),
        picks_changed=[p["changed"] for p in picks],
    )
    try:
        s = stat.summarize(val, u, inv)
        result.update(
            p2=s,
            adopt_gate=stat.adopt_gate(s),
            verdict="ADOPT_SHADOW_PASS" if stat.adopt_gate(s) else "NOT_TRANSFERRED",
        )
    except Exception as error:  # packaging only: the saved panel is the result
        result.update(packaging_error=repr(error))
    (out / "p2.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print("P2", result.get("verdict"), flush=True)


if __name__ == "__main__":
    main()
