#!/usr/bin/env python3
"""Admit complete saved proposal chunks only at the protected first-cycle boundary."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path

from experiment_runtime import require

SCHEMA = "parallel.deferred_gp_admission.v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inside(root, value):
    value = Path(value)
    require((not value.is_absolute() and ".." not in value.parts), "unsafe relative path")
    result = (root / value).resolve()
    require((result.is_relative_to(root)), "path escapes workspace")
    return result


def lease(path):
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def phase_gate(root, cycle, total, deadline, boundary=None):
    if time.time() >= deadline:
        return "closed", "fixed campaign deadline reached"
    campaign = read(root / "runs/campaign-progress.json")
    if boundary is not None:
        require(
            (cycle.name == "cycle-0000" and boundary == cycle.parent / "cycle-0001"), "unsupported boundary"
        )
        if Path(campaign["current_cycle"]).resolve() != boundary:
            return "closed", "campaign is outside the explicitly bound next-cycle boundary"
        if not (cycle / "complete.json").exists():
            return "closed", "original comparison cycle is not complete"
        if (boundary / "complete.json").exists() or (boundary / "search-jobs.json").exists():
            return "closed", "next cycle has already reached GP admission"
        if all((boundary / stage / "complete.json").exists() for stage in ["train-1", "neural", "shuffled"]):
            return "closed", "next cycle has no mandatory GPU stage remaining"
        return "ready", "original comparisons complete; next-cycle GPU stage remains before GP admission"
    if Path(campaign["current_cycle"]).resolve() != cycle:
        return "closed", "campaign advanced beyond the allowed cycle"
    if (cycle / "complete.json").exists() or (cycle / "train-2/complete.json").exists():
        return "closed", "second training block already complete"
    progress = read(root / "runs/cycle-progress.json")
    count = progress["neural_seeded_gp"]
    require((count["total"] == total), "comparison scope changed")
    if count["completed"] != total:
        return "waiting", "current comparisons are unfinished"
    return "ready", "current comparisons complete; mandatory train2 remains"


def preverify(root, spec_path, contexts_path, executable, output, total, boundary=None):
    pins = {}

    def pin(path, expected=None):
        actual = digest(path)
        require((expected is None or actual == expected), f"hash changed: {path}")
        pins[str(path.relative_to(root))] = actual
        return actual

    def get(path, expected=None):
        pin(path, expected)
        return read(path)

    spec_sha = pin(spec_path)
    spec = read(spec_path)
    require(
        (spec["schema"] == "parallel.opportunistic_proposals_spec.v1"),
        'Invalid artifact: spec["schema"] == "parallel.opportunistic_proposals_spec.v1"',
    )
    paths = {
        name: inside(root, spec[name]) for name in ["cycle", "corpus", "checkpoint", "output", "executable"]
    }
    cycle, library = paths["cycle"], paths["output"]
    require((cycle.name == "cycle-0000"), 'Invalid artifact: cycle.name == "cycle-0000"')
    require(
        (library.is_relative_to(root / "runs/gp-reviewed3")),
        'Invalid artifact: library.is_relative_to(root / "runs/gp-reviewed3")',
    )
    binding = get(library / "binding.json")
    require((binding["spec_sha256"] == spec_sha), 'Invalid artifact: binding["spec_sha256"] == spec_sha')
    wrapper_source = root / "tools/opportunistic_proposals.py"
    pin(wrapper_source, binding["script_sha256"])
    module_spec = importlib.util.spec_from_file_location("frozen_proposal_library", wrapper_source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    index = get(library / "library-index.json")
    stopped = get(library / "wrapper-complete.json")
    require(
        (index["schema"] == "parallel.proposal_library_index.v1"),
        'Invalid artifact: index["schema"] == "parallel.proposal_library_index.v1"',
    )
    require(
        (stopped["schema"] == "parallel.proposal_library_stop.v1"),
        'Invalid artifact: stopped["schema"] == "parallel.proposal_library_stop.v1"',
    )
    for obj in [index, stopped]:
        require(
            (all(obj[k] == value for k, value in binding.items())),
            "Invalid artifact: all(obj[k] == value for k, value in binding.items())",
        )
        require(
            (obj["gp_jobs_admitted"] == 0 and obj["gp_evaluations"] == 0),
            'Invalid artifact: obj["gp_jobs_admitted"] == 0 and obj["gp_evaluations"] == 0',
        )
    require(
        (stopped["completed_chunks"] == index["chunks"]),
        'Invalid artifact: stopped["completed_chunks"] == index["chunks"]',
    )
    require((index["chunks"]), "no complete proposal chunks")
    corpus = get(paths["corpus"] / "index.json", spec["corpus_sha256"])
    checkpoint = get(paths["checkpoint"] / "state.json", spec["checkpoint_sha256"])
    pin(paths["checkpoint"] / "model.ot", checkpoint["model_sha256"])
    pin(paths["executable"], spec["executable_sha256"])
    vocabulary = paths["corpus"] / "vocabulary.json"
    pin(vocabulary, checkpoint["vocabulary_sha256"])
    require(
        (checkpoint["corpus_sha256"] == spec["corpus_sha256"]),
        'Invalid artifact: checkpoint["corpus_sha256"] == spec["corpus_sha256"]',
    )
    schedule = get(cycle.parent / "schedule.json")
    require(
        (schedule["initial_corpus_sha256"] == spec["corpus_sha256"]),
        'Invalid artifact: schedule["initial_corpus_sha256"] == spec["corpus_sha256"]',
    )
    require(
        (schedule["executable_sha256"] == spec["executable_sha256"]),
        'Invalid artifact: schedule["executable_sha256"] == spec["executable_sha256"]',
    )
    contexts = get(contexts_path, schedule["contexts_sha256"])
    require(
        (contexts["schema"] == "parallel.context_index.v1" and contexts["dimensions"] == 2048),
        'Invalid artifact: contexts["schema"] == "parallel.context_index.v1" and contexts["dimensions"] == 2048',
    )
    amendment_path = root / "runs/acceptance/proposal-training-input-scope-v44.json"
    amendment = get(amendment_path)
    require(
        (
            amendment["runtime"] == "v44"
            and amendment["candidate_pool_independent_of_heldout_selected_teacher_identities"] is False
        ),
        'Invalid artifact: amendment["runtime"] == "v44" and amendment["candidate_pool_independent_of_heldout_selected_teacher_identities"] is False',
    )
    build = get(root / "runs/parity/runtime-v44-build.json")
    cpu = next(row for row in build["builds"] if row["kind"] == "cpu")
    require(
        (executable == inside(root, cpu["path"])), 'Invalid artifact: executable == inside(root, cpu["path"])'
    )
    pin(executable, cpu["sha256"])
    original = get(cycle / "search-jobs.json")
    require((len(original["jobs"]) == total), 'Invalid artifact: len(original["jobs"]) == total')
    for job in original["jobs"]:
        result = get(inside(root, job["output"]) / "search.json")
        require(
            (result["schema"] == "parallel.search.v1" and result["final_tests_opened"] is False),
            'Invalid artifact: result["schema"] == "parallel.search.v1" and result["final_tests_opened"] is False',
        )
        require(
            (result["seeds_sha256"] == job["seeds_sha256"]),
            'Invalid artifact: result["seeds_sha256"] == job["seeds_sha256"]',
        )
        require(
            (result["vocabulary_sha256"] == checkpoint["vocabulary_sha256"]),
            'Invalid artifact: result["vocabulary_sha256"] == checkpoint["vocabulary_sha256"]',
        )
        require(
            (result["completed_evaluations"] > 0 and math.isfinite(result["wall_seconds"])),
            'Invalid artifact: result["completed_evaluations"] > 0 and math.isfinite(result["wall_seconds"])',
        )
    lookup = {(row["dataset"], row["cut"]): row for row in contexts["encodings"]}
    require(
        (len(lookup) == len(contexts["encodings"])),
        'Invalid artifact: len(lookup) == len(contexts["encodings"])',
    )
    pool_index = {row["path"]: row for row in index["pools"]}
    jobs, outputs, selected_pools, prepared_cache = [], set(), [], {}
    for chunk in index["chunks"]:
        manifest = inside(root, chunk["jobs"])
        require((manifest.is_relative_to(library)), "Invalid artifact: manifest.is_relative_to(library)")
        job_list = get(manifest, chunk["jobs_sha256"])
        destination = manifest.parent
        chunk_root = destination.parent
        snapshot = get(chunk_root / "champion-snapshot.json", chunk["snapshot_sha256"])
        require(
            (snapshot["request"]["checkpoint_sha256"] == spec["checkpoint_sha256"]),
            'Invalid artifact: snapshot["request"]["checkpoint_sha256"] == spec["checkpoint_sha256"]',
        )
        require(
            (snapshot["request"]["corpus_sha256"] == spec["corpus_sha256"]),
            'Invalid artifact: snapshot["request"]["corpus_sha256"] == spec["corpus_sha256"]',
        )
        require(
            (
                snapshot["request"]["cut_start"] == chunk["cut_start"]
                and snapshot["request"]["cuts"] == chunk["cuts"]
            ),
            'Invalid artifact: snapshot["request"]["cut_start"] == chunk["cut_start"] and snapshot["request"]["cuts"] == chunk["cuts"]',
        )
        run = get(destination / "run.json")
        done = get(destination / "complete.json")
        require(
            (run["checkpoint_sha256"] == done["checkpoint_sha256"] == spec["checkpoint_sha256"]),
            'Invalid artifact: run["checkpoint_sha256"] == done["checkpoint_sha256"] == spec["checkpoint_sha256"]',
        )
        require(
            (run["corpus_sha256"] == spec["corpus_sha256"] and not run["shuffled"]),
            'Invalid artifact: run["corpus_sha256"] == spec["corpus_sha256"] and not run["shuffled"]',
        )
        require(
            (run["champion_snapshot_sha256"] == chunk["snapshot_sha256"]),
            'Invalid artifact: run["champion_snapshot_sha256"] == chunk["snapshot_sha256"]',
        )
        require(
            (run.get("cut_start", 0) == chunk["cut_start"] and run["cuts"] == chunk["cuts"]),
            'Invalid artifact: run.get("cut_start",0) == chunk["cut_start"] and run["cuts"] == chunk["cuts"]',
        )
        require(
            (done["world"] == 8 and done["jobs"] == len(job_list["jobs"]) == chunk["pools"]),
            'Invalid artifact: done["world"] == 8 and done["jobs"] == len(job_list["jobs"]) == chunk["pools"]',
        )
        require(
            (not done["final_tests_opened"] and not done["shuffled_context"]),
            'Invalid artifact: not done["final_tests_opened"] and not done["shuffled_context"]',
        )
        wanted = {
            (row["dataset"], row["cut"])
            for row in corpus["contexts"]
            if row["partition"] != "meta_test"
            and chunk["cut_start"] <= row["cut"] < chunk["cut_start"] + chunk["cuts"]
        }
        found = set()
        for job in job_list["jobs"]:
            path = inside(root, job["output"])
            require(
                (path.is_relative_to(destination) and path.name == "search"),
                'Invalid artifact: path.is_relative_to(destination) and path.name == "search"',
            )
            require((str(path) not in outputs), "Invalid artifact: str(path) not in outputs")
            outputs.add(str(path))
            pool = path.parent
            actual = module.validate_pool(root, pool, spec["checkpoint_sha256"], chunk["snapshot_sha256"])
            require(
                (actual == pool_index[str(pool.relative_to(root))]),
                "Invalid artifact: actual == pool_index[str(pool.relative_to(root))]",
            )
            require(
                (inside(root, job["seeds"]) == pool / "seeds.json"),
                'Invalid artifact: inside(root, job["seeds"]) == pool / "seeds.json"',
            )
            for name in ["generation.json", "seeds.json", "ranking.json", "best-predicted.json"]:
                pin(pool / name)
            require(
                (digest(pool / "seeds.json") == job["seeds_sha256"]),
                'Invalid artifact: digest(pool / "seeds.json") == job["seeds_sha256"]',
            )
            seeds = read(pool / "seeds.json")
            require(
                (seeds["prepared_sha256"] == job["prepared_sha256"] and seeds["cut"] == job["cut"]),
                'Invalid artifact: seeds["prepared_sha256"] == job["prepared_sha256"] and seeds["cut"] == job["cut"]',
            )
            pp = inside(root, job["prepared"]) / "prepared.json"
            if pp not in prepared_cache:
                prepared_cache[pp] = get(pp, job["prepared_sha256"])
            prepared = prepared_cache[pp]
            require(
                (pins[str(pp.relative_to(root))] == job["prepared_sha256"]),
                'Invalid artifact: pins[str(pp.relative_to(root))] == job["prepared_sha256"]',
            )
            require(
                (prepared["meta_partition"] != "meta_test"),
                'Invalid artifact: prepared["meta_partition"] != "meta_test"',
            )
            cut = next(c for c in prepared["cuts"] if c["index"] == job["cut"])
            require(
                (actual["dataset"] == prepared["dataset"] and actual["cut"] == job["cut"]),
                'Invalid artifact: actual["dataset"] == prepared["dataset"] and actual["cut"] == job["cut"]',
            )
            key = (prepared["dataset"], job["cut"])
            require(
                (key not in found and key in wanted), "Invalid artifact: key not in found and key in wanted"
            )
            found.add(key)
            row = lookup[key]
            require(
                (row["prepared_sha256"] == job["prepared_sha256"]),
                'Invalid artifact: row["prepared_sha256"] == job["prepared_sha256"]',
            )
            require(
                (row["family"] == prepared["family"] and row["partition"] == prepared["meta_partition"]),
                'Invalid artifact: row["family"] == prepared["family"] and row["partition"] == prepared["meta_partition"]',
            )
            payload = inside(root, row["path"])
            pin(payload, row["sha256"])
            data = payload.read_bytes()
            require(
                (len(data) == 8192 and all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", data))),
                'Invalid artifact: len(data)==8192 and all(math.isfinite(v[0]) for v in struct.iter_unpack("<f",data))',
            )
            metadata = get(payload.with_suffix(".json"))
            require(
                (metadata["output_sha256"] == row["sha256"] == seeds["context_sha256"]),
                'Invalid artifact: metadata["output_sha256"] == row["sha256"] == seeds["context_sha256"]',
            )
            require(
                (metadata["encoder_spec_sha256"] == contexts["assembly_sha256"]),
                'Invalid artifact: metadata["encoder_spec_sha256"] == contexts["assembly_sha256"]',
            )
            require(
                (metadata["prepared_sha256"] == job["prepared_sha256"]),
                'Invalid artifact: metadata["prepared_sha256"] == job["prepared_sha256"]',
            )
            require(
                (metadata["training_sha256"] == cut["training_sha256"] == seeds["training_sha256"]),
                'Invalid artifact: metadata["training_sha256"] == cut["training_sha256"] == seeds["training_sha256"]',
            )
            require(
                (metadata["preprocessing_sha256"] == cut["preprocessing_sha256"]),
                'Invalid artifact: metadata["preprocessing_sha256"] == cut["preprocessing_sha256"]',
            )
            require(
                (
                    metadata["selected_training_rows_sha256"]
                    == hashlib.sha256(
                        json.dumps(cut["train_indices"], separators=(",", ":")).encode()
                    ).hexdigest()
                ),
                'Invalid artifact: metadata["selected_training_rows_sha256"] == hashlib.sha256(json.dumps(cut["train_indices"],separators=(",",":")).encode()).hexdigest()',
            )
            require(
                (metadata["dataset"] == prepared["dataset"] and metadata["family"] == prepared["family"]),
                'Invalid artifact: metadata["dataset"] == prepared["dataset"] and metadata["family"] == prepared["family"]',
            )
            require(
                (metadata["partition"] == prepared["meta_partition"] and metadata["cut"] == job["cut"]),
                'Invalid artifact: metadata["partition"] == prepared["meta_partition"] and metadata["cut"] == job["cut"]',
            )
            jobs.append(job)
            selected_pools.append(actual)
        require((found == wanted), "Invalid artifact: found == wanted")
    combined = {
        "schema": "parallel.gp_jobs.v1",
        "qualification": True,
        "jobs": jobs,
        "vocabulary": str(vocabulary.relative_to(root)),
    }
    write(output / "combined-jobs.json", combined)
    pin(output / "combined-jobs.json")
    plan = {
        "schema": "parallel.deferred_gp_admission_plan.v1",
        "pins": pins,
        "requested_jobs": len(jobs),
        "complete_chunks": len(index["chunks"]),
        "preserved_partial_pools": len(index["pools"]) - len(jobs),
        "cycle": str(cycle.relative_to(root)),
        "library": str(library.relative_to(root)),
        "contexts": str(contexts_path.relative_to(root)),
        "executable": str(executable.relative_to(root)),
        "deadline": schedule["deadline"],
        "expected_comparisons": total,
        "qualification": True,
        "proposal_scope_amendment": str(amendment_path.relative_to(root)),
        "end_to_end_training_only_prediction_established": False,
    }
    if boundary is not None:
        get(cycle / "complete.json")
        plan["boundary_cycle"] = str(boundary.relative_to(root))
    write(output / "verification.json", plan)
    return plan


def present(root, jobs):
    path = root / "runs/gp-priority-jobs.json"
    if not path.exists():
        return 0
    old = {job["output"]: job for job in read(path)["jobs"]}
    count = 0
    for job in jobs:
        if job["output"] in old:
            require((old[job["output"]] == job), "existing queue identity changed")
            count += 1
    return count


def invoke(command, remaining, log):
    with log.open("ab") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining)


def protected_admit(root, output, plan, execute=invoke):
    """Caller owns the stopped wrapper lock; this scope alone holds the GPU phase barrier."""
    barrier = lease(root / "runs/encoding-service.lock")
    if barrier is None:
        return None
    begun = time.monotonic()
    try:
        cycle = inside(root, plan["cycle"])
        boundary = inside(root, plan["boundary_cycle"]) if plan.get("boundary_cycle") else None
        phase, reason = phase_gate(root, cycle, plan["expected_comparisons"], plan["deadline"], boundary)
        if phase != "ready":
            return {"outcome": "skipped", "reason": reason, "newly_admitted_jobs": 0}
        if plan["deadline"] - time.time() <= 32:
            return {
                "outcome": "skipped",
                "reason": "insufficient time before fixed deadline",
                "newly_admitted_jobs": 0,
            }
        expires = begun + 30
        for name, expected in plan["pins"].items():
            require((time.monotonic() < expires), "phase barrier verification exceeded30 seconds")
            require(
                (digest(inside(root, name)) == expected), "verified input changed before admission: " + name
            )
        phase, reason = phase_gate(root, cycle, plan["expected_comparisons"], plan["deadline"], boundary)
        if phase != "ready":
            return {"outcome": "skipped", "reason": reason, "newly_admitted_jobs": 0}
        jobs_path = output / "combined-jobs.json"
        jobs = read(jobs_path)["jobs"]
        before = present(root, jobs)
        error = None
        try:
            if before < len(jobs):
                execute(
                    [
                        str(inside(root, plan["executable"])),
                        "--root",
                        str(root),
                        "enqueue",
                        "--jobs",
                        str(jobs_path),
                        "--contexts",
                        str(inside(root, plan["contexts"])),
                    ],
                    max(0.001, expires - time.monotonic()),
                    output / "enqueue.log",
                )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            error = str(exc)
        after = present(root, jobs)
        require((after in [before, len(jobs)]), "queue transaction was not atomic")
        if after != len(jobs):
            raise RuntimeError(error or "enqueue returned without admitting the verified manifest")
        receipt = root / "runs/queue-receipts" / (digest(jobs_path) + ".json")
        if receipt.exists():
            saved = read(receipt)
            require(
                (saved["jobs_sha256"] == digest(jobs_path)),
                'Invalid artifact: saved["jobs_sha256"] == digest(jobs_path)',
            )
            require(
                (saved["contexts_sha256"] == plan["pins"][plan["contexts"]]),
                'Invalid artifact: saved["contexts_sha256"] == plan["pins"][plan["contexts"]]',
            )
            require(
                (saved["verified_contexts"] == len(jobs)),
                'Invalid artifact: saved["verified_contexts"] == len(jobs)',
            )
        return {
            "outcome": "admitted",
            "reason": "complete library chunks admitted before the remaining mandatory GPU stage",
            "requested_jobs": len(jobs),
            "newly_admitted_jobs": after - before,
            "jobs_already_present": before,
            "queue_receipt": str(receipt.relative_to(root)) if receipt.exists() else None,
            "commit_confirmed_from_queue": True,
            "cli_error": error,
            "barrier_seconds": time.monotonic() - begun,
        }
    finally:
        barrier.close()


def run(root, spec_path, contexts, executable, output, expected, boundary=None):
    spec = read(spec_path)
    cycle, library = inside(root, spec["cycle"]), inside(root, spec["output"])
    schedule = read(cycle.parent / "schedule.json")
    output.mkdir(parents=True, exist_ok=True)
    guard = lease(output / "watcher.lock")
    require((guard is not None), "admission watcher is already running")
    identity = {
        "spec_sha256": digest(spec_path),
        "contexts_sha256": digest(contexts),
        "executable_sha256": digest(executable),
        "script_sha256": digest(Path(__file__)),
        "schedule_sha256": digest(cycle.parent / "schedule.json"),
    }
    if boundary is not None:
        require(
            (boundary == cycle.parent / "cycle-0001"),
            "only the explicit first next-cycle boundary is allowed",
        )
        identity["boundary_cycle"] = str(boundary.relative_to(root))
    binding_path = output / "binding.json"
    if binding_path.exists():
        require((read(binding_path) == identity), "watcher resume identity changed")
    else:
        write(binding_path, identity)

    def publish(phase, reason, verified=0, new=0, terminal=False):
        write(
            root / "runs/proposal-library-admission.json",
            {
                "schema": "parallel.proposal_library_admission_progress.v1",
                "timestamp": int(time.time()),
                "phase": phase,
                "reason": reason,
                "verified_jobs": verified,
                "requested_jobs": verified,
                "newly_admitted_jobs": new,
                "output": str(output.relative_to(root)),
                "receipt": str((output / "complete.json").relative_to(root)) if terminal else None,
                "complete": terminal,
                "status": "complete" if terminal else "active",
            },
        )

    if (output / "complete.json").exists():
        final = read(output / "complete.json")
        require((final["binding"] == identity), 'Invalid artifact: final["binding"]==identity')
        publish(
            final["outcome"],
            final["reason"],
            final.get("requested_jobs", 0),
            final.get("newly_admitted_jobs", 0),
            True,
        )
        return
    # Recover a committed queue transaction after a watcher interruption without
    # calling enqueue again or changing priority order in a later cycle.
    manifest = output / "combined-jobs.json"
    if manifest.exists() and (output / "verification.json").exists():
        plan = read(output / "verification.json")
        require(
            (plan["pins"][str(manifest.relative_to(root))] == digest(manifest)),
            'Invalid artifact: plan["pins"][str(manifest.relative_to(root))] == digest(manifest)',
        )
        jobs = read(manifest)["jobs"]
        if jobs and present(root, jobs) == len(jobs):
            receipt = root / "runs/queue-receipts" / (digest(manifest) + ".json")
            saved = read(receipt) if receipt.exists() else None
            if saved is not None:
                require(
                    (
                        saved["jobs_sha256"] == digest(manifest)
                        and saved["contexts_sha256"] == identity["contexts_sha256"]
                    ),
                    'Invalid artifact: saved["jobs_sha256"] == digest(manifest) and saved["contexts_sha256"] == identity["contexts_sha256"]',
                )
            result = {
                "schema": SCHEMA,
                "outcome": "admitted",
                "reason": "recovered already committed admission without reordering queue",
                "timestamp": int(time.time()),
                "binding": identity,
                "requested_jobs": len(jobs),
                "newly_admitted_jobs": 0,
                "prior_admission_delta": saved.get("new_jobs") if saved else None,
                "jobs_sha256": digest(manifest),
                "contexts_sha256": identity["contexts_sha256"],
                "plan_sha256": digest(output / "verification.json"),
                "commit_recovered": True,
                "qualification": True,
                "final_tests_opened": False,
                "gpu_computation": False,
                "runtime_or_services_changed": False,
            }
            write(output / "complete.json", result)
            publish("admitted", result["reason"], len(jobs), 0, True)
            return
    plan = None
    while True:
        phase, reason = phase_gate(root, cycle, expected, schedule["deadline"], boundary)
        if phase == "closed":
            result = {"outcome": "skipped", "reason": reason, "newly_admitted_jobs": 0}
            break
        if phase != "ready" or not (library / "wrapper-complete.json").exists():
            publish("waiting", reason if phase != "ready" else "proposal wrapper has not durably stopped")
            time.sleep(2)
            continue
        stopped = lease(library / "wrapper.lock")
        if stopped is None:
            publish("waiting", "proposal wrapper still owns its lock")
            time.sleep(2)
            continue
        try:
            publish("verifying", "hash-checking complete chunks outside the GPU phase barrier")
            if plan is None:
                plan = preverify(root, spec_path, contexts, executable, output, expected, boundary)
            publish(
                "ready", "complete chunks verified; seeking bounded phase barrier", plan["requested_jobs"]
            )
            result = protected_admit(root, output, plan)
            heartbeat = 0.0
            while result is None:
                phase, reason = phase_gate(root, cycle, expected, schedule["deadline"], boundary)
                if phase != "ready":
                    result = {"outcome": "skipped", "reason": reason, "newly_admitted_jobs": 0}
                    break
                if time.monotonic() >= heartbeat:
                    publish(
                        "waiting",
                        "GPU lease is busy; verified plan retained without admission",
                        plan["requested_jobs"],
                    )
                    heartbeat = time.monotonic() + 1
                time.sleep(0.025)
                result = protected_admit(root, output, plan)
            result.setdefault("requested_jobs", plan["requested_jobs"])
            result["plan_sha256"] = digest(output / "verification.json")
            result["jobs_sha256"] = digest(output / "combined-jobs.json")
            result["contexts_sha256"] = identity["contexts_sha256"]
            result["proposal_scope_amendment"] = plan["proposal_scope_amendment"]
            result["end_to_end_training_only_prediction_established"] = False
            break
        finally:
            stopped.close()
    result.update(
        schema=SCHEMA,
        timestamp=int(time.time()),
        binding=identity,
        qualification=True,
        final_tests_opened=False,
        gpu_computation=False,
        runtime_or_services_changed=False,
    )
    write(output / "complete.json", result)
    publish(
        result["outcome"],
        result["reason"],
        result.get("requested_jobs", 0),
        result.get("newly_admitted_jobs", 0),
        True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["root", "spec", "contexts", "executable", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-comparisons", type=int, default=336)
    parser.add_argument("--boundary-cycle", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        run(
            root,
            args.spec.resolve(),
            args.contexts.resolve(),
            args.executable.resolve(),
            args.output.resolve(),
            args.expected_comparisons,
            args.boundary_cycle.resolve() if args.boundary_cycle else None,
        )
    except Exception as error:
        write(
            args.output / "failure.json",
            {"timestamp": int(time.time()), "failure": repr(error), "admission_commit_must_be_checked": True},
        )
        raise
