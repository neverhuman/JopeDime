#!/usr/bin/env python3
"""Stage hash-bound proposal pools while the fixed campaign waits for GP.

No queue writer or evaluator is invoked. Rust owns the all-GPU lease and ranks;
this wrapper owns their deadline and stops only its own process descendants.
"""
import argparse
from experiment_runtime import require
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


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


def stop_owned(process):
    """Freeze the spawning controller, then kill pidfd-bound owned descendants."""
    if process.poll() is not None:
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


def validate_pool(root, path, checkpoint_sha, snapshot_sha):
    generation = read(path / "generation.json")
    for key, name in [("seeds_sha256", "seeds.json"), ("ranking_sha256", "ranking.json"),
                      ("best_predicted_sha256", "best-predicted.json")]:
        if generation[key] != digest(path / name):
            raise ValueError("saved proposal hash changed: " + str(path / name))
    seeds, ranking, best = [read(path / name) for name in ("seeds.json", "ranking.json", "best-predicted.json")]
    require((generation["champion_snapshot_sha256"] == snapshot_sha), 'Invalid artifact: generation["champion_snapshot_sha256"] == snapshot_sha')
    require((generation["checkpoint_sha256"] == checkpoint_sha == seeds["checkpoint_sha256"]), 'Invalid artifact: generation["checkpoint_sha256"] == checkpoint_sha == seeds["checkpoint_sha256"]')
    require((ranking["bindings"]["checkpoint_sha256"] == checkpoint_sha), 'Invalid artifact: ranking["bindings"]["checkpoint_sha256"] == checkpoint_sha')
    require((ranking["bindings"]["conditioning_context_sha256"] == seeds["context_sha256"]), 'Invalid artifact: ranking["bindings"]["conditioning_context_sha256"] == seeds["context_sha256"]')
    require((ranking["training_context_only"] and not ranking["heldout_labels_read"] and not ranking["gp_scores_read"]), 'Invalid artifact: ranking["training_context_only"] and not ranking["heldout_labels_read"] and not ranking["gp_scores_read"]')
    require((len(seeds["candidates"]) == 80 and ranking["chosen_count"] == 64), 'Invalid artifact: len(seeds["candidates"]) == 80 and ranking["chosen_count"] == 64')
    require(([c["genome"] for c in seeds["candidates"][:64]] == [x["program"] for x in ranking["ranked"]]), 'Invalid artifact: [c["genome"] for c in seeds["candidates"][:64]] == [x["program"] for x in ranking["ranked"]]')
    require((best["prediction"] == ranking["ranked"][0] and not best["final_tests_opened"]), 'Invalid artifact: best["prediction"] == ranking["ranked"][0] and not best["final_tests_opened"]')
    require((best["training_context_sha256"] == seeds["context_sha256"]), 'Invalid artifact: best["training_context_sha256"] == seeds["context_sha256"]')
    return {"path": str(path.relative_to(root)), "seeds_sha256": generation["seeds_sha256"],
            "generation_sha256": digest(path / "generation.json"), "dataset": best["dataset"],
            "cut": seeds["cut"], "prepared_sha256": seeds["prepared_sha256"]}


def restore_library(root, output, checkpoint_sha, binding):
    index_path = output / "library-index.json"
    if not index_path.exists():
        return {}, []
    index = read(index_path)
    require((index["schema"] == "parallel.proposal_library_index.v1"), 'Invalid artifact: index["schema"] == "parallel.proposal_library_index.v1"')
    require((all(index[k] == v for k, v in binding.items())), 'Invalid artifact: all(index[k] == v for k, v in binding.items())')
    pools = {}
    for previous in index["pools"]:
        path = (root / previous["path"]).resolve()
        require((path.is_relative_to(output)), 'Invalid artifact: path.is_relative_to(output)')
        snapshot_sha = digest(path.parents[2] / "champion-snapshot.json")
        actual = validate_pool(root, path, checkpoint_sha, snapshot_sha)
        require((actual == previous), 'Invalid artifact: actual == previous')
        pools[str(path)] = actual
    for chunk in index["chunks"]:
        path = (root / chunk["jobs"]).resolve()
        require((path.is_relative_to(output) and digest(path) == chunk["jobs_sha256"]), 'Invalid artifact: path.is_relative_to(output) and digest(path) == chunk["jobs_sha256"]')
        require(((path.parent / "complete.json").exists()), 'Invalid artifact: (path.parent / "complete.json").exists()')
    return pools, index["chunks"]


def run(root, spec_path):
    spec = read(spec_path)
    require((spec["schema"] == "parallel.opportunistic_proposals_spec.v1"), 'Invalid artifact: spec["schema"] == "parallel.opportunistic_proposals_spec.v1"')
    require((1 <= spec["chunk_cuts"] <= 8 and 1 <= spec["block_seconds"] <= 60), 'Invalid artifact: 1 <= spec["chunk_cuts"] <= 8 and 1 <= spec["block_seconds"] <= 60')
    require((1 <= spec["service_seconds"] <= 1200), 'Invalid artifact: 1 <= spec["service_seconds"] <= 1200')
    paths = {}
    for name in ["cycle", "corpus", "checkpoint", "output", "executable"]:
        value = Path(spec[name])
        require((not value.is_absolute() and ".." not in value.parts), 'Invalid artifact: not value.is_absolute() and ".." not in value.parts')
        paths[name] = (root / value).resolve()
        require((paths[name].is_relative_to(root)), 'Invalid artifact: paths[name].is_relative_to(root)')
    for name, filename in [("corpus", "index.json"), ("checkpoint", "state.json"), ("executable", None)]:
        path = paths[name] / filename if filename else paths[name]
        require((digest(path) == spec[name + "_sha256"]), name + " identity changed")
    state = read(paths["checkpoint"] / "state.json")
    require((state["corpus_sha256"] == spec["corpus_sha256"]), 'Invalid artifact: state["corpus_sha256"] == spec["corpus_sha256"]')
    require((digest(paths["checkpoint"] / "model.ot") == state["model_sha256"]), 'Invalid artifact: digest(paths["checkpoint"] / "model.ot") == state["model_sha256"]')
    require((digest(paths["corpus"] / "vocabulary.json") == state["vocabulary_sha256"]), 'Invalid artifact: digest(paths["corpus"] / "vocabulary.json") == state["vocabulary_sha256"]')
    output = paths["output"]
    require((output.is_relative_to(root / "runs/gp-reviewed3")), 'Invalid artifact: output.is_relative_to(root / "runs/gp-reviewed3")')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "wrapper.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {"spec_sha256": digest(spec_path), "script_sha256": digest(Path(__file__))}
    if (output / "binding.json").exists():
        require((read(output / "binding.json") == binding), 'Invalid artifact: read(output / "binding.json") == binding')
    else:
        write(output / "binding.json", binding)
    contexts = read(paths["corpus"] / "index.json")["contexts"]
    starts = range(spec["cut_start"], spec["cut_end"], spec["chunk_cuts"])
    chunks = [(start, min(spec["chunk_cuts"], spec["cut_end"] - start)) for start in starts]
    expected = {(c["dataset"], c["cut"]) for c in contexts if c["partition"] != "meta_test"
                and spec["cut_start"] <= c["cut"] < spec["cut_end"]}
    # Restore verified saved work before any admission gate; a restart after
    # the campaign cutoff must never publish zero progress over saved pools.
    pools, completed = restore_library(root, output, spec["checkpoint_sha256"], binding)
    begin = time.monotonic()
    stop = [False]
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))
    require((ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0), 'Invalid artifact: ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0')  # Linux child subreaper.
    def publish(phase, reason, chunk=None):
        write(root / "runs/proposal-library.json", {"schema": "parallel.proposal_library_progress.v1",
              "timestamp": int(time.time()), "phase": phase, "reason": reason,
              "checkpoint_sha256": spec["checkpoint_sha256"], "corpus_sha256": spec["corpus_sha256"],
              "output": str(output.relative_to(root)), "active_chunk": chunk,
              "saved_seed_pools": len(pools), "planned_seed_pools": len(expected),
              "saved_ranked_programs": 64 * len(pools), "completed_chunks": len(completed),
              "gpu_world": 8, "gp_jobs_admitted": 0, "gp_evaluations": 0,
              "qualification_only": True, "encodings_recomputed": 0})
    reason = "all requested proposal pools saved"
    for start, cuts in chunks:
        if any(c["cut_start"] == start and c["cuts"] == cuts for c in completed):
            continue
        chunk_root = output / f"chunk-{start:04}"
        destination = chunk_root / "neural"
        wanted = {(c["dataset"], c["cut"]) for c in contexts if c["partition"] != "meta_test" and start <= c["cut"] < start + cuts}
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
            command = [str(paths["executable"]), "--root", str(root), "propose", "--corpus", str(paths["corpus"]),
                       "--checkpoint", str(paths["checkpoint"]), "--output", str(destination),
                       "--cut-start", str(start), "--cuts", str(cuts)]
            with (chunk_root / "controller.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        allowed, reason = gate(root, paths["cycle"], int(time.time()))
                        elapsed = time.monotonic() - started
                        if not allowed or stop[0] or elapsed >= spec["block_seconds"] or time.monotonic() - begin >= spec["service_seconds"]:
                            reason = reason if not allowed else "owned proposal block time or stop limit reached"
                            stop_owned(process)
                            break
                        snapshot_path = chunk_root / "champion-snapshot.json"
                        if snapshot_path.exists():
                            snapshot_sha = digest(snapshot_path)
                            for path in destination.glob("*/cut-*/generation.json"):
                                key = str(path.parent)
                                if key not in pools:
                                    pool = validate_pool(root, path.parent, spec["checkpoint_sha256"], snapshot_sha)
                                    require(((pool["dataset"], pool["cut"]) in wanted), 'Invalid artifact: (pool["dataset"], pool["cut"]) in wanted')
                                    pools[key] = pool
                        publish("running", "saving ranked neural seed pools; GP admission remains zero",
                                {"cut_start": start, "cuts": cuts, "expected_pools": len(wanted), "elapsed_seconds": elapsed})
                        time.sleep(1)
                finally:
                    stop_owned(process)
                if process.returncode != 0:
                    if allowed and not stop[0] and time.monotonic() - started < spec["block_seconds"]:
                        reason = f"proposal controller failed with status {process.returncode}; inspect its saved log"
                    write(chunk_root / "stopped.json", {"reason": reason, "returncode": process.returncode,
                          "timestamp": int(time.time()), "saved_artifacts_preserved": True})
                    break
        snapshot_sha = digest(chunk_root / "champion-snapshot.json")
        current = []
        for path in destination.glob("*/cut-*/generation.json"):
            pool = validate_pool(root, path.parent, spec["checkpoint_sha256"], snapshot_sha)
            pools[str(path.parent)] = pool
            current.append(pool)
        require(({(p["dataset"], p["cut"]) for p in current} == wanted), 'Invalid artifact: {(p["dataset"], p["cut"]) for p in current} == wanted')
        jobs = read(destination / "jobs.json")
        require((len(jobs["jobs"]) == len(wanted)), 'Invalid artifact: len(jobs["jobs"]) == len(wanted)')
        for job in jobs["jobs"]:
            require((digest(root / job["seeds"]) == job["seeds_sha256"]), 'Invalid artifact: digest(root / job["seeds"]) == job["seeds_sha256"]')
        completed.append({"cut_start": start, "cuts": cuts, "pools": len(current),
                          "jobs": str((destination / "jobs.json").relative_to(root)),
                          "jobs_sha256": digest(destination / "jobs.json"), "snapshot_sha256": snapshot_sha})
        write(output / "library-index.json", {"schema": "parallel.proposal_library_index.v1", **binding,
              "chunks": completed, "pools": list(pools.values()), "gp_jobs_admitted": 0, "gp_evaluations": 0})
        publish("running", "chunk verified; checking campaign before next GPU block")
    else:
        reason = "all requested proposal pools saved"
    complete = len(pools) == len(expected)
    write(output / "library-index.json", {"schema": "parallel.proposal_library_index.v1", **binding,
          "chunks": completed, "pools": list(pools.values()), "gp_jobs_admitted": 0, "gp_evaluations": 0})
    publish("completed" if complete else "stopped", reason)
    write(output / "wrapper-complete.json", {"schema": "parallel.proposal_library_stop.v1", **binding,
          "timestamp": int(time.time()), "reason": reason, "all_requested_saved": complete,
          "saved_seed_pools": len(pools), "completed_chunks": completed, "gp_jobs_admitted": 0,
          "gp_evaluations": 0, "encodings_recomputed": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.spec.resolve())
