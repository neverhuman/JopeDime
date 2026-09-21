#!/usr/bin/env python3
"""Stage hash-bound proposal pools while the fixed campaign waits for GP.

No queue writer or evaluator is invoked. Rust owns the all-GPU lease and ranks;
this wrapper owns their deadline and stops only its own process descendants.
"""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from proposal_pool_v2 import ASSEMBLY_PROTOCOL, PURE_PROTOCOL, verify_generation

PHASE = "GP optimizes cut targets for the next teaching corpus"


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


def gate(root, cycle, now, stop_fraction=0.9):
    try:
        campaign = read(root / "runs/campaign-progress.json")
        progress = read(root / "runs/cycle-progress.json")
        gp = read(root / "runs/gp-service.json")
        if Path(campaign["current_cycle"]).resolve() != cycle.resolve():
            return False, "campaign moved to another cycle"
        if now >= campaign["deadline"]:
            return False, "fixed campaign deadline reached"
        if progress.get("complete") or progress["phase"] != PHASE:
            return False, "campaign left the GP waiting phase"
        if not 0 <= now - progress["timestamp"] <= 5:
            return False, "main cycle progress is stale"
        if not 0 <= now - gp["timestamp"] <= 5 or gp["phase"] != "running":
            return False, "GP service progress is stale or unavailable"
        count = progress["neural_seeded_gp"]
        if count["total"] <= 0 or count["completed"] / count["total"] >= stop_fraction:
            return False, "GP reached the proposal-library admission cutoff"
        auxiliary = read(cycle / "auxiliary/complete.json")
        if auxiliary.get("schema") != "parallel.auxiliary_training_complete.v1":
            return False, "auxiliary learning is not durably complete"
        return True, "GP remains active; auxiliary learning is complete"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, "required campaign status unavailable: " + str(error)


def descendants(pid):
    parents = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            # comm can contain spaces/parentheses; fields after the final ')'.
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(directory.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        found = {child for child, parent in parents.items() if parent in owned}
        if found <= owned:
            return owned - {pid}
        owned |= found


PIDFD_AVAILABLE = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")


def _stop_owned_by_pid(process):
    """Fallback without pidfds (interpreters built against a sysroot older than glibc 2.36, e.g.
    python-build-standalone): freeze the controller by pid, kill its descendants by pid, then the
    controller. Same order as the pidfd path; without pidfds a reused pid could in principle be
    signalled, which is why the pidfd path is preferred whenever the interpreter offers it."""
    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    children = list(descendants(process.pid))
    for pid in children:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    for pid in children:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def stop_owned(process):
    """Freeze the spawning controller, then kill pidfd-bound owned descendants."""
    if process.poll() is not None:
        return
    if not PIDFD_AVAILABLE:
        _stop_owned_by_pid(process)
        return
    controller = os.pidfd_open(process.pid)
    children = []
    try:
        signal.pidfd_send_signal(controller, signal.SIGSTOP)
        for pid in descendants(process.pid):
            try:
                children.append((pid, os.pidfd_open(pid)))
            except ProcessLookupError:
                pass
        for _, fd in children:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
        signal.pidfd_send_signal(controller, signal.SIGKILL)
        process.wait(timeout=5)
        for pid, _ in children:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
    finally:
        os.close(controller)
        for _, fd in children:
            os.close(fd)


@lru_cache(maxsize=256)
def prepared_cuts(root, path, expected_sha):
    path = (root / path / "prepared.json").resolve()
    assert path.is_relative_to(root) and digest(path) == expected_sha
    prepared = read(path)
    return {cut["index"]: cut for cut in prepared["cuts"]}


def validate_pool(root, path, checkpoint_sha, snapshot_sha, contexts):
    generation, seeds, best = [
        read(path / name) for name in ("generation.json", "seeds.json", "pure-best-predicted.json")
    ]
    assert generation["schema"] == "parallel.proposal_generation.v2"
    assert generation["champion_snapshot_sha256"] == snapshot_sha
    assert generation["checkpoint_sha256"] == checkpoint_sha
    assert generation["context_mode"] == "matching_training_cut"
    key = (best["dataset"], seeds["cut"])
    row = contexts[key]
    assert row["partition"] != "meta_test"
    assert path.name == f"cut-{key[1]}" and path.parent.name == key[0]
    assert seeds["prepared_sha256"] == row["prepared_sha256"] and seeds["context_sha256"] == row["sha256"]
    context = (root / row["path"]).resolve()
    assert (
        context.is_relative_to(root) and context.stat().st_size == 8192 and digest(context) == row["sha256"]
    )
    cut = prepared_cuts(root, row["prepared"], row["prepared_sha256"])[key[1]]
    first = dict(
        dataset=key[0],
        cut=key[1],
        prepared_sha256=row["prepared_sha256"],
        training_sha256=cut["training_sha256"],
        vocabulary_sha256=best["vocabulary_sha256"],
    )
    job = dict(cut=key[1], prepared_sha256=row["prepared_sha256"], seeds_sha256=digest(path / "seeds.json"))
    counts, scope = verify_generation(path, generation, seeds, job, first, read)
    assert scope == "pure_prediction_then_adaptive_gp_assembly"
    return {
        "path": str(path.relative_to(root)),
        "seeds_sha256": job["seeds_sha256"],
        "generation_sha256": digest(path / "generation.json"),
        "dataset": key[0],
        "cut": key[1],
        "prepared": row["prepared"],
        "prepared_sha256": row["prepared_sha256"],
        "pure_ranking_sha256": generation["pure_ranking_sha256"],
        "pure_best_predicted_sha256": generation["pure_best_predicted_sha256"],
        "gp_ranking_sha256": generation["gp_ranking_sha256"],
        "gp_population_assembly_sha256": generation["gp_population_assembly_sha256"],
        "teacher_overlaps": counts["teacher_overlaps"],
        "overlap_refills": counts["overlap_refills"],
    }


def restore_library(root, output, checkpoint_sha, binding, contexts):
    index_path = output / "library-index.json"
    if not index_path.exists():
        return {}, []
    index = read(index_path)
    assert index["schema"] == "parallel.pure_proposal_library_index.v1"
    assert all(index[k] == v for k, v in binding.items())
    pools = {}
    for previous in index["pools"]:
        path = (root / previous["path"]).resolve()
        assert path.is_relative_to(output)
        snapshot_sha = digest(path.parents[2] / "champion-snapshot.json")
        actual = validate_pool(root, path, checkpoint_sha, snapshot_sha, contexts)
        assert actual == previous
        pools[str(path)] = actual
    for chunk in index["chunks"]:
        path = (root / chunk["jobs"]).resolve()
        assert path.is_relative_to(output) and digest(path) == chunk["jobs_sha256"]
        assert (path.parent / "complete.json").exists()
    return pools, index["chunks"]


def run(root, spec_path):
    spec = read(spec_path)
    assert spec["schema"] == "parallel.opportunistic_proposals_spec.v2"
    assert 1 <= spec["chunk_cuts"] <= 8 and 1 <= spec["block_seconds"] <= 60
    assert 1 <= spec["service_seconds"] <= 1200
    paths = {}
    for name in ["cycle", "corpus", "checkpoint", "output", "executable"]:
        value = Path(spec[name])
        assert not value.is_absolute() and ".." not in value.parts
        paths[name] = (root / value).resolve()
        assert paths[name].is_relative_to(root)
    for name, filename in [("corpus", "index.json"), ("checkpoint", "state.json"), ("executable", None)]:
        path = paths[name] / filename if filename else paths[name]
        assert digest(path) == spec[name + "_sha256"], name + " identity changed"
    state = read(paths["checkpoint"] / "state.json")
    assert state["corpus_sha256"] == spec["corpus_sha256"]
    assert digest(paths["checkpoint"] / "model.ot") == state["model_sha256"]
    assert digest(paths["corpus"] / "vocabulary.json") == state["vocabulary_sha256"]
    output = paths["output"]
    assert output.is_relative_to(root / "runs/gp-reviewed3")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "wrapper.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validator_sha = digest(Path(__file__).with_name("proposal_pool_v2.py"))
    assert validator_sha == spec["pool_validator_sha256"]
    assert (
        spec["pure_prediction_protocol"] == PURE_PROTOCOL
        and spec["gp_assembly_protocol"] == ASSEMBLY_PROTOCOL
    )
    binding = {
        "spec_sha256": digest(spec_path),
        "script_sha256": digest(Path(__file__)),
        "pool_validator_sha256": validator_sha,
        "executable_sha256": spec["executable_sha256"],
    }
    if (output / "binding.json").exists():
        assert read(output / "binding.json") == binding
    else:
        write(output / "binding.json", binding)
    contexts = read(paths["corpus"] / "index.json")["contexts"]
    contexts_by_key = {(c["dataset"], c["cut"]): c for c in contexts}
    assert len(contexts_by_key) == len(contexts)
    starts = range(spec["cut_start"], spec["cut_end"], spec["chunk_cuts"])
    chunks = [(start, min(spec["chunk_cuts"], spec["cut_end"] - start)) for start in starts]
    expected = {
        (c["dataset"], c["cut"])
        for c in contexts
        if c["partition"] != "meta_test" and spec["cut_start"] <= c["cut"] < spec["cut_end"]
    }
    # Restore verified saved work before any admission gate; a restart after
    # the campaign cutoff must never publish zero progress over saved pools.
    pools, completed = restore_library(root, output, spec["checkpoint_sha256"], binding, contexts_by_key)
    begin = time.monotonic()
    stop = [False]
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0  # Linux child subreaper.

    def publish(phase, reason, chunk=None):
        write(
            root / "runs/pure-proposal-library.json",
            {
                "schema": "parallel.pure_proposal_library_progress.v1",
                "timestamp": int(time.time()),
                "phase": phase,
                "reason": reason,
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "corpus_sha256": spec["corpus_sha256"],
                "output": str(output.relative_to(root)),
                "active_chunk": chunk,
                "saved_seed_pools": len(pools),
                "planned_seed_pools": len(expected),
                "saved_ranked_programs": 64 * len(pools),
                "completed_chunks": len(completed),
                "gpu_world": 8,
                "gp_jobs_admitted": 0,
                "gp_evaluations": 0,
                "pure_prediction_protocol": PURE_PROTOCOL,
                "gp_assembly_protocol": ASSEMBLY_PROTOCOL,
                "pure_prediction_verified": bool(pools),
                "gp_population_training_only_end_to_end": False,
                "teacher_overlaps": sum(p["teacher_overlaps"] for p in pools.values()),
                "overlap_refills": sum(p["overlap_refills"] for p in pools.values()),
                "qualification_only": True,
                "encodings_recomputed": 0,
            },
        )

    reason = "all requested proposal pools saved"
    for start, cuts in chunks:
        if any(c["cut_start"] == start and c["cuts"] == cuts for c in completed):
            continue
        chunk_root = output / f"chunk-{start:04}"
        destination = chunk_root / "neural"
        wanted = {
            (c["dataset"], c["cut"])
            for c in contexts
            if c["partition"] != "meta_test" and start <= c["cut"] < start + cuts
        }
        if not wanted:
            continue
        allowed, reason = gate(root, paths["cycle"], int(time.time()))
        if not allowed or stop[0] or time.monotonic() - begin >= spec["service_seconds"]:
            reason = reason if not allowed else "proposal service time or stop limit reached"
            break
        if destination.exists() and not (destination / "complete.json").exists():
            reason = "saved partial chunk preserved; explicit resume review required"
            break
        if not (destination / "complete.json").exists():
            lease = (root / "runs/encoding-service.lock").open("a+")
            while True:
                allowed, reason = gate(root, paths["cycle"], int(time.time()))
                if not allowed or stop[0] or time.monotonic() - begin >= spec["service_seconds"]:
                    break
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lease, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    publish("waiting", "shared GPU lease belongs to the campaign or qualification")
                    time.sleep(1)
            lease.close()
            if not allowed or stop[0] or time.monotonic() - begin >= spec["service_seconds"]:
                break
            chunk_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(paths["executable"]),
                "--root",
                str(root),
                "propose",
                "--corpus",
                str(paths["corpus"]),
                "--checkpoint",
                str(paths["checkpoint"]),
                "--output",
                str(destination),
                "--cut-start",
                str(start),
                "--cuts",
                str(cuts),
            ]
            with (chunk_root / "controller.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        allowed, reason = gate(root, paths["cycle"], int(time.time()))
                        elapsed = time.monotonic() - started
                        if (
                            not allowed
                            or stop[0]
                            or elapsed >= spec["block_seconds"]
                            or time.monotonic() - begin >= spec["service_seconds"]
                        ):
                            reason = (
                                reason if not allowed else "owned proposal block time or stop limit reached"
                            )
                            stop_owned(process)
                            break
                        snapshot_path = chunk_root / "champion-snapshot.json"
                        if snapshot_path.exists():
                            snapshot_sha = digest(snapshot_path)
                            for path in destination.glob("*/cut-*/generation.json"):
                                key = str(path.parent)
                                if key not in pools:
                                    pool = validate_pool(
                                        root,
                                        path.parent,
                                        spec["checkpoint_sha256"],
                                        snapshot_sha,
                                        contexts_by_key,
                                    )
                                    assert (pool["dataset"], pool["cut"]) in wanted
                                    pools[key] = pool
                        publish(
                            "running",
                            "saving verified pure predictions and GP seed pools; GP admission remains zero",
                            {
                                "cut_start": start,
                                "cuts": cuts,
                                "expected_pools": len(wanted),
                                "elapsed_seconds": elapsed,
                            },
                        )
                        time.sleep(1)
                finally:
                    stop_owned(process)
                if process.returncode != 0:
                    if allowed and not stop[0] and time.monotonic() - started < spec["block_seconds"]:
                        reason = f"proposal controller failed with status {process.returncode}; inspect its saved log"
                    write(
                        chunk_root / "stopped.json",
                        {
                            "reason": reason,
                            "returncode": process.returncode,
                            "timestamp": int(time.time()),
                            "saved_artifacts_preserved": True,
                        },
                    )
                    break
        snapshot_sha = digest(chunk_root / "champion-snapshot.json")
        current = []
        for path in destination.glob("*/cut-*/generation.json"):
            pool = validate_pool(root, path.parent, spec["checkpoint_sha256"], snapshot_sha, contexts_by_key)
            pools[str(path.parent)] = pool
            current.append(pool)
        assert {(p["dataset"], p["cut"]) for p in current} == wanted
        jobs = read(destination / "jobs.json")
        assert len(jobs["jobs"]) == len(wanted)
        assert len({(j["prepared"], j["cut"]) for j in jobs["jobs"]}) == len(wanted)
        for job in jobs["jobs"]:
            seed_path = (root / job["seeds"]).resolve()
            assert seed_path.is_relative_to(destination)
            pool = pools[str(seed_path.parent)]
            assert digest(seed_path) == job["seeds_sha256"] == pool["seeds_sha256"]
            assert job["prepared"] == pool["prepared"] and job["prepared_sha256"] == pool["prepared_sha256"]
            assert (
                job["cut"] == pool["cut"] and (root / job["output"]).resolve() == seed_path.parent / "search"
            )
        completed.append(
            {
                "cut_start": start,
                "cuts": cuts,
                "pools": len(current),
                "jobs": str((destination / "jobs.json").relative_to(root)),
                "jobs_sha256": digest(destination / "jobs.json"),
                "snapshot_sha256": snapshot_sha,
            }
        )
        write(
            output / "library-index.json",
            {
                "schema": "parallel.pure_proposal_library_index.v1",
                **binding,
                "chunks": completed,
                "pools": list(pools.values()),
                "gp_jobs_admitted": 0,
                "gp_evaluations": 0,
            },
        )
        publish("running", "chunk verified; checking campaign before next GPU block")
    else:
        reason = "all requested proposal pools saved"
    complete = len(pools) == len(expected)
    write(
        output / "library-index.json",
        {
            "schema": "parallel.pure_proposal_library_index.v1",
            **binding,
            "chunks": completed,
            "pools": list(pools.values()),
            "gp_jobs_admitted": 0,
            "gp_evaluations": 0,
        },
    )
    publish("completed" if complete else "stopped", reason)
    write(
        output / "wrapper-complete.json",
        {
            "schema": "parallel.pure_proposal_library_stop.v1",
            **binding,
            "timestamp": int(time.time()),
            "reason": reason,
            "all_requested_saved": complete,
            "saved_seed_pools": len(pools),
            "completed_chunks": completed,
            "gp_jobs_admitted": 0,
            "gp_evaluations": 0,
            "encodings_recomputed": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.spec.resolve())
