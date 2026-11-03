"""Pre-prereg FIT-only frozen-r882 score-spread calibration; zero updates and no utility/assessment scoring."""

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
    torch.use_deterministic_algorithms(True)
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
    assert launch["manifest_sha256"] == run.sha(run.HERE / "manifest.json")
    assert launch["calibrate_sha256"] == run.sha(__file__)
    run.verify(pins)
    rt = run.runtime(pins)
    train, val = run.cohorts(rt)
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(args.launch.resolve())}
        | {str(rt.INPUT / "pools" / (row["context_sha256"] + ".npz")) for row in train}
    )
    audit = rt.Audit(allowed, args.output)
    sys.addaudithook(audit.check)
    net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
    initial = rt.tensor_hash({n: net.state_dict()[n] for n in names})
    engine = run.Engine(net, rt, names, frozen, guard, fused, train + val)
    engine.load(train)
    groups = {}
    for row in train:
        cid = row["context_sha256"]
        groups.setdefault(len(engine.pools[cid]["fitness"]), []).append(cid)
    spreads = {}
    with torch.no_grad():
        for k, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                run.cooperate()
                batch = cids[start : start + 8]
                kwargs, tok, mask, _, _ = engine.batch(
                    batch, [np.arange(k) for _ in batch]
                )
                anchor = engine.anchored(kwargs)
                cost = (
                    engine.forward(kwargs, tok, mask, anchor).numpy().astype(np.float64)
                )
                if not np.isfinite(cost).all():
                    raise RuntimeError("nonfinite r882 scores")
                for cid, values in zip(batch, cost, strict=True):
                    spreads[cid] = float(np.std(values, ddof=0))
    engine.check()
    assert rt.tensor_hash({n: net.state_dict()[n] for n in names}) == initial
    spreadfile = args.output / "fit-context-score-std.json"
    run.save(spreadfile, spreads)
    result = dict(
        wave="H248",
        status="PASS",
        optimizer_updates=0,
        utility_outcomes_scored=0,
        DLVal_scored=0,
        fold_manifest_sha256=run.sha(run.HERE / "fold_manifest.json"),
        spread_sha256=run.sha(spreadfile),
        checkpoint_sha256=run.sha(pins["checkpoint"]),
        parent_commit=pins["parent_commit"],
        calibration_source_sha256=run.sha(__file__),
        by_outer={},
        definition="Median across FIT contexts of float64 population std (ddof=0) of all full-pool frozen-r882 predicted costs, first four native-index reveals; no outcome selection; same partition for all seeds.",
        audit=audit.receipt(),
        dope_before=rt.DOPE,
        dope_after=guard.snapshot_dope_weights(net).composite_sha256,
        initial_adapter_sha256=initial,
    )
    for outer in range(5):
        fit, select, assess = stat.partition(train, outer, stat.SEEDS[0])
        ids = sorted(r["context_sha256"] for r in fit)
        T = float(np.median([spreads[c] for c in ids]))
        assert np.isfinite(T) and T > 0
        result["by_outer"][str(outer)] = dict(
            temperature=T,
            fit_contexts=len(ids),
            fit_families=len({r["family_id"] for r in fit}),
            fit_ids_sha256=hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            failure_labels_replaced=sum(
                int((~engine.pools[c]["gate"]).sum()) for c in ids
            ),
            candidate_labels=sum(len(engine.pools[c]["gate"]) for c in ids),
            all_contexts_have_nonfailure=all(
                np.isfinite(
                    failure_ceiling(engine.pools[c]["fitness"], engine.pools[c]["gate"])
                )
                for c in ids
            ),
        )
    run.save(args.output / "calibration.json", result)
    compact = {k: v for k, v in result.items() if k != "audit"}
    compact["audit_summary"] = dict(
        protected_reads=result["audit"]["protected_reads"],
        refused=result["audit"]["refused"],
        opened_count=len(result["audit"]["opened"]),
    )
    run.save(args.output / "temperatures.json", compact)
    print(
        json.dumps(
            dict(
                status="PASS",
                temperatures={
                    k: v["temperature"] for k, v in result["by_outer"].items()
                },
                zero_updates=True,
                no_assessment_scoring=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
