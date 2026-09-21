#!/usr/bin/env python3
"""Snapshot compact dashboard inputs, never candidate ledgers, datasets or weights."""

import argparse
import hashlib
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
files = set()


def add(path):
    path = a.root / path
    if path.is_file():
        files.add(path)


for name in [
    "status.json",
    "hardware.json",
    "gp-service.json",
    "neural-service.json",
    "encoding-service.json",
    "encoding-pipeline.json",
    "cycle-progress.json",
    "campaign-progress.json",
    "cohort64-bootstrap-readiness.json",
    "teacher-progress.json",
    "corpus-progress.json",
    "runtime-readiness.json",
    "report-progress.json",
    "proposal-library.json",
    "pure-proposal-library.json",
    "proposal-library-admission.json",
    "auxiliary-training.json",
    "gpu-wait.json",
    "events.log",
]:
    add(Path("runs") / name)
status = json.loads((a.root / "runs/status.json").read_text())
for key in ["contexts_index", "family_registry"]:
    if status.get(key):
        add(status[key])
for name in status.get("prepared_job_manifests", []):
    add(name)
add("runs/parity/hyperion-reviewed4-recovery-parity.json")
cycles = {p.parent for p in (a.root / "runs").glob("*/cycle.json")}
campaign = a.root / "runs/campaign-progress.json"
if campaign.exists():
    current = json.loads(campaign.read_text()).get("current_cycle")
    if current:
        current = a.root / current
        cycles.add(current)
        completed = current.parent / "completed-cycles.json"
        add(completed)
        if completed.exists():
            for row in json.loads(completed.read_text()).get("cycles", []):
                cycles.add(a.root / row["cycle"])
for cycle in cycles:
    for name in [
        "cycle.json",
        "complete.json",
        "comparisons.json",
        "report-v1/report.json",
        "predictor-exports.json",
    ]:
        add(cycle / name)
    for block in ["train-1", "train-2"]:
        for name in ["latest.json", "complete.json", "initialization.json", "metrics.jsonl"]:
            add(cycle / block / name)
for latest in (a.root / "runs").glob("*/latest.json"):
    if (latest.parent / "invalidated.json").exists():
        continue
    for name in ["latest.json", "complete.json", "initialization.json", "metrics.jsonl"]:
        add(latest.parent / name)
policy = json.loads((a.root / "runs/neural-service.json").read_text())
if policy.get("run"):
    for name in ["latest.json", "complete.json", "initialization.json", "metrics.jsonl"]:
        add(Path(policy["run"]) / name)
rows = []
for path in sorted(files):
    rel = path.relative_to(a.root)
    dest = a.output / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = path.read_bytes()
    if path.name == "events.log":
        data = data[-65536:]
    dest.write_bytes(data)
    rows.append(dict(path=str(rel), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
(a.output / "snapshot.json").write_text(
    json.dumps(
        dict(
            schema="parallel.dashboard_snapshot.v1",
            timestamp=int(time.time()),
            files=rows,
            copied_dataset_bytes=0,
            copied_model_bytes=0,
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(dict(files=len(rows), bytes=sum(x["bytes"] for x in rows), output=str(a.output))))
