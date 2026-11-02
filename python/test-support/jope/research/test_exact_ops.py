#!/usr/bin/env python3
"""SCQ operational helpers for the dedicated strict test-only campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from test_exact_common import (
    ALLOWED_LIVE_STATES,
    atomic_json,
    prime_source_root,
    sha256_file,
    sha256_text,
)

SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SCQ_EVENTS = Path("/home/ubuntu/smart_cluster/var/events.jsonl")
PROJECTS = Path("/home/ubuntu/smart_cluster/etc/projects.toml")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CHECKPOINT = Path(
    "/home/ubuntu/jope-flywheel/training/round1/checkpoints/"
    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt"
)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise RuntimeError(
            f"{' '.join(command)} exited {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[:2000]}"
        )
    return result


def scq_json(*args: str) -> dict:
    return json.loads(run([str(SCQ), "--json", *args]).stdout)


def jobs() -> list[dict]:
    value = scq_json("ls")
    rows = value.get("Jobs", {}).get("jobs")
    return rows if isinstance(rows, list) else []


def snapshot(path: Path) -> dict:
    current_jobs = jobs()
    value = {
        "schema": "JopeTestOnlyScqSnapshot.v1",
        "created_at_ms": int(time.time() * 1000),
        "projects_toml_sha256": sha256_file(PROJECTS),
        "projects_toml": PROJECTS.read_text(),
        "jobs": [
            {
                "id": row.get("id"),
                "state": row.get("state"),
                "node": row.get("node"),
                "project": row.get("project"),
                "gpus_per_task": row.get("gpus_per_task"),
            }
            for row in current_jobs
            if row.get("id")
        ],
        "nodes": scq_json("nodes"),
        "priority": run([str(SCQ), "priority", "ls"]).stdout,
    }
    atomic_json(path, value)
    return value


def verify_survival(before: Path) -> dict:
    prior = json.loads(before.read_text())
    current = {row["id"]: row for row in jobs() if row.get("id")}
    missing_live = {
        row["id"]
        for row in prior["jobs"]
        if row["id"] not in current and row.get("state") in ALLOWED_LIVE_STATES
    }
    protected_missing = sorted(
        row["id"]
        for row in prior["jobs"]
        if row["id"] in missing_live
        and (
            row.get("node") in {"xbabe1", "xbabe3"}
            or "ratchet" in row["id"].lower()
        )
    )
    cancelled = []
    snapshot_time = datetime.fromtimestamp(
        int(prior["created_at_ms"]) / 1_000, tz=timezone.utc
    ).isoformat()
    if SCQ_EVENTS.is_file():
        for line in SCQ_EVENTS.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("kind") == "cancel"
                and event.get("job") in missing_live
                and str(event.get("ts") or "") >= snapshot_time
            ):
                cancelled.append(event["job"])
    archived_terminal = sorted(
        row["id"]
        for row in prior["jobs"]
        if row["id"] not in current and row.get("state") not in ALLOWED_LIVE_STATES
    )
    ratchet = [
        row
        for row in current.values()
        if "ratchet" in row["id"] and row.get("state") in ALLOWED_LIVE_STATES
    ]
    external = scq_json("nodes").get("Nodes", {}).get("gpus", [])
    xbabe1 = [gpu for gpu in external if gpu.get("node") == "xbabe1"]
    result = {
        "schema": "JopeTestOnlyRestartVerification.v1",
        "checked_at_ms": int(time.time() * 1000),
        "preexisting_jobs": len(prior["jobs"]),
        "missing_after_rollout": len(missing_live),
        "cancelled_preexisting_job_ids": sorted(set(cancelled)),
        "protected_workloads_missing": protected_missing,
        "archived_terminal_jobs": len(archived_terminal),
        "ratchet_live": len(ratchet),
        "xbabe1_external_busy": bool(
            xbabe1 and xbabe1[0].get("status") == "external-busy"
        ),
        "priority": run([str(SCQ), "priority", "ls"]).stdout,
    }
    atomic_json(before.with_name("post-restart-verification.json"), result)
    if cancelled:
        raise RuntimeError(
            f"{len(set(cancelled))} preexisting jobs were cancelled during rollout"
        )
    if protected_missing:
        raise RuntimeError("protected SCQ workloads disappeared during rollout")
    if not ratchet or not result["xbabe1_external_busy"]:
        raise RuntimeError("ratchet or xbabe1 external workload is not healthy")
    return result


def toml(value: object) -> str:
    return json.dumps(str(value))


def input_block(name: str, source: Path, target: str) -> list[str]:
    return [
        "[[input]]",
        f"name = {toml(name)}",
        f"source = {toml(source)}",
        f"target = {toml(target)}",
        f"sha256 = {toml(sha256_file(source))}",
        "",
    ]


def render_checkpoint_proposal_spec(
    path: Path,
    *,
    checkpoint: Path,
    pointer_path: Path | None = None,
    split: str | None = None,
    cohort: Path | None = None,
    seed: int = 20260726,
    proposals_per_dataset: int = 64,
    project: str = "jope-selfplay-train",
) -> dict:
    pointer_path = pointer_path or ROOT / "catalog" / "current.json"
    pointer = json.loads(pointer_path.read_text())
    catalog = Path(pointer["catalog"])
    data = Path(pointer["data"])
    script = Path(__file__).with_name("test_exact_dl_generate.py")
    common = Path(__file__).with_name("test_exact_common.py")
    prime_source = prime_source_root()
    checkpoint_hash = sha256_file(checkpoint)
    cohort_hash = sha256_file(cohort) if cohort is not None else None
    split_name = split or "all"
    # The generator's own identity is part of the job identity.
    #
    # Without this the job id was a pure function of (checkpoint, cohort, seed),
    # and `_seed_next_strict_wave` treats `failed`/`cancelled` as terminal
    # (test_exact_controller.py, "micro-cohort proposal job failed"). A single
    # generator defect therefore parked the phase FOREVER: the controller
    # re-rendered the identical id, scq answered "already exists" with the dead
    # record, and the handler raised again -- 848 consecutive times on
    # 2026-07-28, tournament stuck at phase_index 9, corpus watermark frozen at
    # 1,805,437. Fixing the generator could not clear it because the id did not
    # move. It also removes a second sticky failure: scq pins `[[input]] sha256`
    # at submit and rejects an in-place requeue whose input changed
    # (`error[input-hash-mismatch]`, the live trainer-watchdog alarm), which a
    # content-addressed id turns into a clean new submission instead.
    generator_hash = sha256_text(sha256_file(script) + sha256_file(common))
    job_id = (
        f"jope-selfplay-propose-{checkpoint_hash[:12]}-"
        f"{cohort_hash[:12] if cohort_hash else split_name}-{seed}-cpu4-p2"
        f"-g{generator_hash[:8]}"
    )
    output = "dl-proposals.jsonl"
    lines = [
        "[job]",
        f"id = {toml(job_id)}",
        f'project = "{project}"',
        'tier = "normal"',
        'command = "/usr/bin/python3"',
        (
            'args = ["{input:script}", "--catalog-pointer", "{input:pointer}", '
            '"--catalog", "{input:catalog}", "--data", "{input:data}", '
            '"--checkpoint", "{input:checkpoint}", '
            f'"--output", "{{output:proposals}}/{output}", "--device", "cuda", '
            f'"--seed", "{seed}", "--proposals-per-dataset", '
            f'"{proposals_per_dataset}", "--batch-contexts", "64"'
            + (f', "--split", "{split}"' if split else "")
            + (', "--cohort", "{input:cohort}"' if cohort is not None else "")
            + "]"
        ),
        "",
        "[resource]",
        "gpus_per_task = 1",
        "cpus_reserved = 4",
        "omp_threads = 4",
        # MEASURED, not guessed. scq `peak_ram` for this exact job class:
        #   949-1052 MB (micro-cohort 128 datasets and full-catalog runs alike;
        #   the generator streams one batch of contexts at a time and the only
        #   unbounded structure is the cross-dataset backfill pool).
        # scq adds MEM_SAFETY_MB = 8 GiB on top of the reservation
        # (scq-controller/src/state.rs:185, free_ram at :2425-2427), so
        # ram_mb = 49152 demanded 56 GiB free and was placeable on at most one
        # node; it stalled the propose lane. 12288 demands 20 GiB, fits every
        # GPU node, and still leaves a ~11x margin over the measured peak.
        "ram_mb = 12288",
        "gang_size = 1",
        'placement = "single_node"',
        "",
        # NO `[dependencies]` BLOCK -- deliberate, and load-bearing.
        #
        # This used to carry `hostname = [{ command = "/usr/bin/grep", args =
        # ["-qxE", "xbabe2|xbabe3", "/etc/hostname"], ... }]`, which read as a
        # placement restriction but was a SERDE NO-OP: `hostname` is not a field
        # of scq's DependencySpec (scq-core/src/deps.rs:17-31) and there is no
        # `deny_unknown_fields`, so serde silently discarded it,
        # `DependencySpec::is_empty()` became true, and `prepare_dependencies`
        # ran `rec.dependency_allowed_nodes.clear()`
        # (scq-controller/src/state.rs:3038-3040). Placement was never
        # constrained. VERIFIED: `scq check` accepts the spec with and without
        # the key, identically ("check ok" both ways).
        #
        # Removing it is therefore a byte-level no-op to the scheduler, and it
        # stops the source lying about what constrains placement.
        #
        # It is NOT replaced by `[[dependencies.smoke]]` here, even though that
        # is the only form scq enforces. Declaring ANY dependency changes submit
        # semantics: scq probes only nodes that ALREADY have the requested
        # GPU/cores/RAM free (scq-controller/src/state.rs:3092-3140) and REJECTS
        # the submit outright when none do, instead of queueing. MEASURED
        # 2026-07-28 23:3xZ with all three GPUs allocated: a spec identical to
        # this one plus a smoke probe that PASSES on all three nodes was refused
        # with `error[dependency-unsatisfied]: dependencies cannot be verified on
        # any eligible node`. The trainer watchdog survives this only because it
        # retries every 5 s (and still logged 44 consecutive `cycle_failed`).
        # This propose lane has no such retry, so a smoke block would convert a
        # queued job into a parked tournament phase.
        #
        # See src/spec_policy.rs (the reservations) and src/spec_drift.rs (the
        # test that fails when two renderers of this family disagree).
        "",
        "[env]",
        f'PYTHONPATH = "{prime_source}:{script.parent}"',
        'OMP_NUM_THREADS = "4"',
        "",
        "[preempt]",
        "preemptible = true",
        'checkpoint = "none"',
        'resume = "rerun"',
        "",
        "[limits]",
        "max_runtime_hours = 24",
        "max_restarts = 1",
        "",
    ]
    for item in (
        ("script", script, "scripts/test_exact_dl_generate.py"),
        ("common", common, "scripts/test_exact_common.py"),
        ("pointer", pointer_path, "catalog/current.json"),
        ("catalog", catalog, f"catalog/{catalog.name}"),
        ("data", data, f"catalog/{data.name}"),
        ("checkpoint", checkpoint, f"checkpoint/{checkpoint.name}"),
        *((("cohort", cohort, f"cohort/{cohort.name}"),) if cohort is not None else ()),
    ):
        lines.extend(input_block(*item))
    lines.extend(
        [
            "[[output]]",
            'name = "proposals"',
            'source = "proposals"',
            "directory = true",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    result = {
        "job_id": job_id,
        "spec": str(path),
        "checkpoint_sha256": checkpoint_hash,
        "split": split,
        "cohort_sha256": cohort_hash,
        "seed": seed,
        "proposals_per_dataset": proposals_per_dataset,
    }
    return result


def render_dl_spec(path: Path) -> dict:
    result = render_checkpoint_proposal_spec(
        path,
        checkpoint=CHECKPOINT,
        project="jope-testonly",
    )
    atomic_json(ROOT / "ops" / "dl-job.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--output", type=Path, default=ROOT / "ops" / "pre-launch.json")
    verify = commands.add_parser("verify-survival")
    verify.add_argument("--before", type=Path, default=ROOT / "ops" / "pre-launch.json")
    spec = commands.add_parser("render-dl-spec")
    spec.add_argument("--output", type=Path, default=ROOT / "specs" / "dl-gen0.toml")
    args = parser.parse_args()
    if args.command == "snapshot":
        result = snapshot(args.output)
        print(json.dumps({"snapshot": str(args.output), "jobs": len(result["jobs"])}, sort_keys=True))
    elif args.command == "verify-survival":
        print(json.dumps(verify_survival(args.before), sort_keys=True))
    else:
        print(json.dumps(render_dl_spec(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
