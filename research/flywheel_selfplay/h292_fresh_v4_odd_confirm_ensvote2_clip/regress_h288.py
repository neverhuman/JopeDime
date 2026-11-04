"""H292 step 1b (xbabe2 CPU, hw_t0125): reproduce one published H288 cohort-v3 cell bit-exactly BEFORE cohort v4 is
touched: ENS_VOTE2 on bootstrap seed 288150 (family-mean gain +0.08537299387889595, CI95 [0.03955..., 0.13829...],
W/T/L 60/221/19). Inputs are H288's PUBLISHED artifacts, fetched from S3 by VersionId with sha256 enforced
(fresh_pools_H288.json: fresh-build.tar.gz and fresh-r3175-gpu-costs.npz); the frozen H287 export is fetched the same
way. Code is H288's evaluate/analysis and the exporters' scorers, unchanged."""

import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H278 = HERE.parent / "h278_rerank8_veto_frozen_export_transfer"
H283 = HERE.parent / "h283_rerank8_veto_fresh_cohort_gpu"
H287 = HERE.parent / "h287_bagswitch_seed_ensemble_export"
H288 = HERE.parent / "h288_fresh_v3_odd_powered_ensemble"
POOLS = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H288.json")
WORK = Path("/home/ubuntu/jope-data/h292/h288_published")
TARGET = dict(
    gain=0.08537299387889595,
    ci95=[0.039552121289789376, 0.13829149681075167],
    wtl=[60, 221, 19],
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    WORK.mkdir(parents=True, exist_ok=True)
    ev = load("h288_evaluate", H288 / "evaluate.py")
    objs = {
        o["s3_uri"].split("/")[-1]: o for o in json.loads(POOLS.read_text())["objects"]
    }
    for name in ("fresh-build.tar.gz", "fresh-r3175-gpu-costs.npz"):
        o = objs[name]
        path = WORK / name
        if not path.exists():
            bucket, key = o["s3_uri"][5:].split("/", 1)
            subprocess.run(
                [
                    "aws",
                    "s3api",
                    "get-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--version-id",
                    o["version_id"],
                    str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if ev.sha(path) != o["sha256"]:
            raise RuntimeError("published artifact sha256 mismatch: " + name)
    if not (WORK / "fresh" / "selected.json").exists():
        with tarfile.open(WORK / "fresh-build.tar.gz") as tf:
            tf.extractall(WORK, filter="data")
    sys.path.insert(0, str(H278))
    import runner as rt
    import topk

    sys.addaudithook(rt.audit_hook)
    rt.load_core()
    e283 = load("h283_evaluate", H283 / "evaluate.py")
    stat = load("h288_analysis", H288 / "analysis.py")
    spec = ev.fetch_h287(
        json.loads((H287 / "export_manifest.json").read_text()),
        Path("/home/ubuntu/jope-data/h292/export_h287"),
    )
    sys.path.insert(0, str(H287))
    ens = load("h287_ensemble", H287 / "ensemble.py")
    fresh = WORK / "fresh"
    topk.core.INPUT_POOLS_DIR = fresh
    rows = json.loads((fresh / "selected.json").read_text())["fresh"]
    ids = [r["context_sha256"] for r in rows]
    fap, fpj, fcf = e283.priors_from(fresh, rows)
    with np.load(WORK / "fresh-r3175-gpu-costs.npz") as z:
        costs = {c: z[c].astype(np.float32) for c in ids}
    V = topk.load_packets(rows, costs, fap, fpj, fcf, 8)
    X = np.concatenate([V[c]["X"] for c in ids])
    r = ens.score_all_h287_arms(
        Path("/home/ubuntu/jope-data/h292/export_h287"), X, len(ids), 8, spec
    )
    picks = [[int(j) for j in r["ENS_VOTE2"]]] * 3
    u, inv = ev.to_panel(V, ids, picks)
    s = stat.summarize(
        u, inv, [r_["family_id"] for r_ in rows], picks, stat.DETERMINISTIC_BOOT_SEEDS
    )
    got = dict(
        gain=s["per_seed_family_mean_gain"][0],
        ci95=s["per_seed_ci95"][0],
        wtl=s["wtl_by_seed"][0],
    )
    exact = got == TARGET
    out = dict(
        step="H292 regression: H288 ENS_VOTE2 cell (boot seed 288150) from published artifacts",
        target=TARGET,
        got=got,
        bit_exact=exact,
    )
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print("REGRESSION", "PASS" if exact else "FAIL", json.dumps(got))
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
