"""Pre-prereg FIT-only label-scale calibration and sampler hashes; no model inference or utility scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import runner as run
import analysis as stat
from treatment import failure_ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or sorted(os.sched_getaffinity(0)) != [48, 49, 50, 51]
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("host/CPU/nice")
    run.PAUSE = args.output.with_suffix(".pause.json")
    args.output.mkdir()
    pins = json.loads((run.HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    assert launch["manifest_sha256"] == run.sha(run.HERE / "manifest.json") and launch[
        "calibrate_sha256"
    ] == run.sha(__file__)
    run.verify(pins)
    rt = run.runtime(pins)
    train, _ = run.cohorts(rt)
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(args.launch.resolve())}
        | {str(f.resolve()) for f in run.HERE.iterdir() if f.is_file()}
        | {str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train}
    )
    audit = rt.Audit(allowed, args.output)
    sys.addaudithook(audit.check)
    spreads, orders, failures, counts = {}, {}, {}, {}
    for row in train:
        run.cooperate()
        cid = row["context_sha256"]
        path = rt.INPUT / "pools" / (cid + ".npz")
        if run.sha(path) != row["pool_sha256"]:
            raise RuntimeError("pool hash")
        with np.load(path, allow_pickle=False) as pool:
            fitness = pool["fitness"].astype(np.float64)
            gate = pool["gate"].astype(bool)
        ceiling = failure_ceiling(fitness, gate)
        corrected = np.where(gate, fitness, ceiling)
        spreads[cid] = float(np.std(corrected, ddof=0))
        orders[cid] = np.argsort(fitness)
        failures[cid] = int((~gate).sum())
        counts[cid] = len(gate)
    spreadfile = args.output / "fit-context-label-std.json"
    run.save(spreadfile, spreads)
    result = dict(
        wave="H250",
        status="PASS",
        optimizer_updates=0,
        model_forward_calls=0,
        utility_outcomes_scored=0,
        DLVal_scored=0,
        fold_manifest_sha256=run.sha(run.HERE / "fold_manifest.json"),
        spread_sha256=run.sha(spreadfile),
        checkpoint_sha256=run.sha(pins["checkpoint"]),
        parent_commit=pins["parent_commit"],
        calibration_source_sha256=run.sha(__file__),
        by_outer={},
        by_cell={},
        definition="epsilon=.1*median across FIT contexts of float64 full-pool population std after replacing invalid-gate costs with worst valid full-context cost+1. No valid clipping. No model forward or utility scoring.",
    )
    for outer in range(5):
        fit, _, _ = stat.partition(train, outer, stat.SEEDS[0])
        ids = sorted(r["context_sha256"] for r in fit)
        median = float(np.median([spreads[c] for c in ids]))
        epsilon = 0.1 * median
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("positive finite FIT epsilon required")
        result["by_outer"][str(outer)] = dict(
            epsilon=epsilon,
            median_label_std=median,
            fit_contexts=len(ids),
            fit_families=len({r["family_id"] for r in fit}),
            fit_ids_sha256=hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            failure_labels_replaced=sum(failures[c] for c in ids),
            candidate_labels=sum(counts[c] for c in ids),
            all_contexts_have_nonfailure=True,
        )
        for seed in stat.SEEDS:
            result["by_cell"][f"A{outer}-{seed}"] = dict(
                batch_plan_sha256_by_arm={
                    arm: stat.plan_digest(
                        stat.arm_batch_plan(fit, orders, seed, outer, arm)
                    )
                    for arm in stat.ARMS
                }
            )
    result["audit"] = audit.receipt()
    run.save(args.output / "calibration.json", result)
    compact = {k: v for k, v in result.items() if k != "audit"}
    compact["audit_summary"] = dict(
        protected_reads=result["audit"]["protected_reads"],
        refused=result["audit"]["refused"],
        opened_count=len(result["audit"]["opened"]),
    )
    run.save(args.output / "normalization.json", compact)
    print(
        json.dumps(
            dict(
                status="PASS",
                epsilons={k: v["epsilon"] for k, v in result["by_outer"].items()},
                zero_updates=True,
                model_forward_calls=0,
                no_assessment_scoring=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
