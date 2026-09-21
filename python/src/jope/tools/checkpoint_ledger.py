#!/usr/bin/env python3
"""Checkpoint ledger: which retained checkpoints are REAL checkpoints, from the mirrored receipts only (read-only, local).

A real checkpoint is a block-end checkpoint (step % block_steps == 0 on the learning lineage) whose identity (sha256 of its
state.json, which binds model/optimizer/config digests) is carried by a frozen-panel monitor receipt (`binding.checkpoint_sha256`)
and, where the mirrored journal window covers it, by a diagnostic verdict. Intermediate checkpoints (step % block_steps != 0)
are recovery points. Orphans are block-end checkpoints without a monitor receipt. Nothing is deleted or written on the node;
the pruner change (`training.rs:328`) is Codex's and rides the next trainer rebuild under the F6 bridge.

  checkpoint_ledger.py --mirror parallel/recovery/eight-a100-20260912 --output parallel/recovery/eight-a100-20260912/review/checkpoint-policy-v1/ledger-<utc>.json
"""

import argparse
import json
import re
from pathlib import Path

import promotion_gate
from resume_expired_cycle import read, require

SCHEMA = "parallel.checkpoint_ledger.v1"
BLOCK_STEPS = 3000
STEP = re.compile(r"learning/(cycle-\d+)/train-2/checkpoints/step-(\d+)/state\.json$")


def ledger(mirror, run="runs/eight-a100-recovery-20260912", block_steps=BLOCK_STEPS, verdicts=None):
    """Classify every mirrored checkpoint state.json; verdicts (sha -> event) may come from a bounded journal read-out."""
    mirror = Path(mirror)
    manifest = read(mirror / "learning-receipts/MANIFEST.json")
    require(manifest.get("schema") == "parallel.learning_receipt_mirror.v1", "not a learning receipt mirror")
    monitors = {}
    for rel, entry in manifest["files"].items():
        if "/learning/monitoring/" in rel and rel.endswith(".json") and not entry.get("stub"):
            receipt = read(mirror / "learning-receipts" / entry["mirrored"])
            bound = (receipt.get("binding") or {}).get("checkpoint_sha256")
            if receipt.get("schema") == "parallel.frozen_monitoring.v1" and bound:
                monitors.setdefault(bound, []).append(
                    dict(
                        path=rel,
                        sha256=entry["sha256"],
                        ranking_bt=receipt.get("ranking_bt"),
                        sequence_nll=receipt.get("sequence_nll"),
                        evaluated_step=receipt.get("evaluated_step"),
                    )
                )
    rows = []
    for rel, entry in manifest["files"].items():
        match = STEP.search(rel)
        if not match or not rel.startswith(run + "/"):
            continue
        cycle, step = match.group(1), int(match.group(2))
        identity = (
            entry.get("full_sha256") or entry["sha256"]
        )  # a stubbed state.json records the full digest separately
        fields = (
            (read(mirror / "learning-receipts" / entry["mirrored"]).get("fields") or {})
            if entry.get("stub")
            else {
                k: v
                for k, v in read(mirror / "learning-receipts" / entry["mirrored"]).items()
                if k in ("model_sha256", "optimizer_sha256", "config_sha256", "step")
            }
        )
        kind = "block_end" if step % block_steps == 0 else "intermediate"
        monitored = monitors.get(identity, [])
        verdict = (verdicts or {}).get(identity)
        rows.append(
            dict(
                cycle=cycle,
                step=step,
                kind=kind,
                state_sha256=identity,
                model_sha256=fields.get("model_sha256"),
                optimizer_sha256=fields.get("optimizer_sha256"),
                monitor_receipts=monitored,
                verdict=verdict,
                real=kind == "block_end" and bool(monitored),
                evidence_gap=None
                if kind != "block_end" or monitored
                else "block-end checkpoint without a frozen-panel monitor receipt",
            )
        )
    rows.sort(key=lambda r: (r["cycle"], r["step"]))
    orphans = [
        dict(cycle=r["cycle"], step=r["step"], state_sha256=r["state_sha256"])
        for r in rows
        if r["kind"] == "block_end" and not r["monitor_receipts"]
    ]
    counts = dict(
        total=len(rows),
        block_end=sum(r["kind"] == "block_end" for r in rows),
        intermediate=sum(r["kind"] == "intermediate" for r in rows),
        real=sum(r["real"] for r in rows),
        orphans=len(orphans),
        monitor_receipts=sum(len(v) for v in monitors.values()),
        monitored_identities=len(monitors),
    )
    return dict(
        schema=SCHEMA,
        run=run,
        block_steps=block_steps,
        manifest_sha256=manifest.get("sha256"),
        counts=counts,
        orphans=orphans,
        rows=rows,
        policy="block-end checkpoints are retained forever (rejected ones included, per F11); intermediates are recovery points for the last 3 blocks; nothing referenced by state, candidates, initialization parents, best.json or a monitor receipt is ever deleted; the pruner predicate change rides the next trainer rebuild under F6",
        final_tests_opened=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mirror", type=Path, default=Path("/home/ubuntu/jope-dime/parallel/recovery/eight-a100-20260912")
    )
    parser.add_argument("--run", default="runs/eight-a100-recovery-20260912")
    parser.add_argument(
        "--verdicts",
        type=Path,
        help="optional json {checkpoint_sha256: diagnostic_event} from a journal read-out",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = ledger(args.mirror, args.run, verdicts=read(args.verdicts) if args.verdicts else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["receipt_sha256"] = promotion_gate.immutable(args.output, result)
    print(
        json.dumps(
            dict(
                counts=result["counts"],
                orphans=result["orphans"][:10],
                receipt=str(args.output) if args.output else None,
                receipt_sha256=result.get("receipt_sha256"),
            ),
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    main()
