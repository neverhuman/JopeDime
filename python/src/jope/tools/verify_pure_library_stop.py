#!/usr/bin/env python3
"""Verify the stopped v53 proposal library and summarize its full hardware interval."""

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import opportunistic_proposals_v2 as lib

root = Path.cwd()
os.sched_setaffinity(0, {60})
unit = "jope-parallel-pure-proposal-library-v53.service"
properties = dict(
    line.split("=", 1)
    for line in subprocess.check_output(
        ["systemctl", "show", unit, "--property=ActiveState,SubState,ExecMainStatus,MainPID,CPUUsageNSec"]
    )
    .decode()
    .splitlines()
)
assert (
    properties["ActiveState"] == "inactive"
    and properties["ExecMainStatus"] == "0"
    and properties["MainPID"] == "0"
)
spec = lib.read(root / "manifests/opportunistic-proposals-v53.json")
output = root / spec["output"]
stop = lib.read(output / "wrapper-complete.json")
index = lib.read(output / "library-index.json")
assert stop["schema"] == "parallel.pure_proposal_library_stop.v1"
assert index["schema"] == "parallel.pure_proposal_library_index.v1"
assert (
    stop["gp_jobs_admitted"]
    == stop["gp_evaluations"]
    == index["gp_jobs_admitted"]
    == index["gp_evaluations"]
    == 0
)
contexts = {(c["dataset"], c["cut"]): c for c in lib.read(root / spec["corpus"] / "index.json")["contexts"]}
binding = lib.read(output / "binding.json")
pools, chunks = lib.restore_library(root, output, spec["checkpoint_sha256"], binding, contexts)
assert len(pools) == stop["saved_seed_pools"] and chunks == stop["completed_chunks"]
complete_pools = sum(c["pools"] for c in chunks)
assert not list(output.glob("chunk-*/neural/*/cut-*/search/search.json"))
# Late, complete generations may exist after the final one-second scan. Verify
# and expose them separately; never silently edit the frozen library index.
unindexed = []
for path in output.glob("chunk-*/neural/*/cut-*/generation.json"):
    if str(path.parent) not in pools:
        unindexed.append(
            lib.validate_pool(
                root,
                path.parent,
                spec["checkpoint_sha256"],
                lib.digest(path.parents[3] / "champion-snapshot.json"),
                contexts,
            )
        )
gp = lib.read(root / "runs/gp-service.json")
first = lib.read(root / "runs/acceptance/pure-proposal-library-first-chunk-v53.json")
receipt = dict(
    schema="parallel.pure_proposal_library_stop_acceptance.v1",
    timestamp=int(time.time()),
    passed=True,
    verifier_sha256=lib.digest(Path(__file__)),
    reason=stop["reason"],
    indexed_verified_pools=len(pools),
    complete_chunk_pools=complete_pools,
    indexed_partial_pools=len(pools) - complete_pools,
    unindexed_verified_partial_pools=len(unindexed),
    saved_ranked_program_slots=64 * len(pools),
    completed_chunks=len(chunks),
    teacher_overlaps=sum(p["teacher_overlaps"] for p in pools.values()),
    gp_jobs_admitted=0,
    gp_evaluations=0,
    encodings_recomputed=0,
    pure_prediction_verified=True,
    gp_population_training_only_end_to_end=False,
    stop_sha256=lib.digest(output / "wrapper-complete.json"),
    index_sha256=lib.digest(output / "library-index.json"),
    spec_sha256=lib.digest(root / "manifests/opportunistic-proposals-v53.json"),
    service=properties,
    current_gp_active=gp["active"],
    current_gp_failed_views=gp["failed_views"],
)
lib.write(root / "runs/acceptance/pure-proposal-library-stop-v53.json", receipt)
journal = [
    json.loads(line)
    for line in subprocess.check_output(["sudo", "journalctl", "-u", unit, "-o", "json", "--no-pager"])
    .decode()
    .splitlines()
]
started = min(
    int(x["__REALTIME_TIMESTAMP"]) / 1e6 for x in journal if x.get("MESSAGE", "").startswith("Started ")
)
ended = max(
    int(x["__REALTIME_TIMESTAMP"]) / 1e6
    for x in journal
    if "Deactivated successfully." in x.get("MESSAGE", "")
)
samples = []
for path in (root / "runs/telemetry").glob("hardware-day-*.jsonl"):
    for line in path.open():
        row = json.loads(line)
        if started <= row["timestamp"] <= ended:
            samples.append(row)
samples.sort(key=lambda x: x["timestamp"])
assert samples and len({x["timestamp"] for x in samples}) == len(samples)
sample_path = root / "runs/acceptance/pure-proposal-library-hardware-v53.jsonl"
sample_path.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in samples))
utilization = dict(
    schema="parallel.pure_proposal_library_utilization.v1",
    timestamp=int(time.time()),
    interval_start=started,
    interval_end=ended,
    interval_seconds=ended - started,
    interval_scope="Entire systemd service including startup, chunk transitions, verification and cutoff",
    hardware_samples=len(samples),
    maximum_sample_gap_seconds=max(b["timestamp"] - a["timestamp"] for a, b in zip(samples, samples[1:])),
    mean_cpu_activity_percent=statistics.mean(s["cpu"] for s in samples),
    min_available_ram_gib=min(s["available_gb"] for s in samples),
    min_free_disk_gib=min(s["disk_free_gb"] for s in samples),
    hardware_errors=sum(len(s["errors"]) for s in samples),
    gpu_saturation_claim=False,
    gpus=[
        dict(
            rank=i,
            mean_utilization_percent=statistics.mean(s["gpus"][i]["utilization"] for s in samples),
            max_temperature_c=max(s["gpus"][i]["temperature"] for s in samples),
            mean_power_w=statistics.mean(s["gpus"][i]["watts"] for s in samples),
            peak_vram_mib=max(s["gpus"][i]["memory_mb"] for s in samples),
        )
        for i in range(8)
    ],
    indexed_verified_pools=len(pools),
    complete_chunk_pools=complete_pools,
    indexed_partial_pools=len(pools) - complete_pools,
    service_cpu_seconds=max(int(x["CPU_USAGE_NSEC"]) for x in journal if "CPU_USAGE_NSEC" in x) / 1e9,
    gp_concurrency_evidence=dict(
        workers=first["concurrent_gp_workers"],
        timestamp=first["timestamp"],
        source="runs/acceptance/pure-proposal-library-first-chunk-v53.json",
    ),
    gp_jobs_admitted=0,
    gp_evaluations=0,
    encodings_recomputed=0,
    samples_sha256=lib.digest(sample_path),
)
lib.write(root / "runs/acceptance/pure-proposal-library-utilization-v53.json", utilization)
lib.write(root / "runs/acceptance/pure-proposal-library-journal-v53.json", journal)
print(json.dumps({"receipt": receipt, "utilization": utilization}, indent=2))
