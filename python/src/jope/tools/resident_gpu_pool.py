#!/usr/bin/env python3
"""Keep bounded CUDA ranks alive and release them for explicit GPU handoffs."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import stat
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import experiment_runtime as runtime
from resume_expired_cycle import atomic, read, require, sha


def suspended(root, now, owner=None):
    pause = root / "runs/continuous-control.json"
    if pause.is_file() and read(pause).get("paused", False):
        return True
    path = root / "runs/qualification-control.json"
    if path.is_file():
        control = read(path)
        if control.get("active") and control.get("requested_at", now + 1) <= now < control.get("deadline", 0):
            return owner is None or control.get("output") != owner
    return owner is not None


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(
            pid=pid,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            start_ticks=int(fields[19]),
        )
    except (OSError, ValueError, IndexError):
        return None


def worker_matches(state, identity, executable):
    worker = state.get("worker", {})
    try:
        return bool(
            identity
            and state.get("pid") == identity["pid"]
            and worker.get("process") == identity
            and worker.get("executable_sha256") == executable
            and process_identity(identity["pid"]) == identity
            and sha(Path(f"/proc/{identity['pid']}/exe")) == executable
            and process_identity(identity["pid"]) == identity
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def idle_worker(state, pid, identity=None, executable=None):
    # Legacy PID-only readiness cannot authorize a handoff or a new dispatch.
    return (
        identity is not None
        and identity.get("pid") == pid
        and worker_matches(state, identity, executable)
        and state.get("status") in ("idle", "warming")
    )


@contextmanager
def custody_lock(path, create=False):
    with runtime.parent_fd(path) as (parent, name):
        flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
        with os.fdopen(os.open(name, flags, 0o600, dir_fd=parent), "r+b") as handle:
            metadata = os.fstat(handle.fileno())
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(
                stat.S_ISREG(metadata.st_mode)
                and (metadata.st_dev, metadata.st_ino) == (current.st_dev, current.st_ino),
                "resident lock identity changed",
            )
            yield handle


def rank_observation(pool, rank, identity, executable):
    """Observe custody without connecting: a probe connection itself kills old workers."""
    path = runtime.checked_path(pool / f"rank-{rank}.json")
    state = read(path)
    require(
        state.get("rank") == rank and worker_matches(state, identity, executable),
        "resident worker process or loaded executable changed",
    )
    endpoint = runtime.checked_path(pool / f"rank-{rank}.sock").lstat()
    worker = state["worker"]
    require(
        stat.S_ISSOCK(endpoint.st_mode)
        and endpoint.st_uid == os.geteuid()
        and endpoint.st_dev == worker.get("socket_device")
        and endpoint.st_ino == worker.get("socket_inode"),
        "resident socket identity changed",
    )
    with custody_lock(pool / f"rank-{rank}.lock") as handle:
        lock_identity = os.fstat(handle.fileno())
        descriptors = []
        for descriptor in Path(f"/proc/{identity['pid']}/fd").iterdir():
            try:
                descriptors.append(descriptor.stat())
            except FileNotFoundError:
                pass
        require(
            any((d.st_dev, d.st_ino) == (lock_identity.st_dev, lock_identity.st_ino) for d in descriptors),
            "resident worker no longer owns the rank lock descriptor",
        )
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("resident rank no longer retains execution custody")
    require(state.get("status") in ("idle", "busy", "warming"), "resident status unavailable")
    # Native state timestamps mark transitions, not heartbeats. Freshness here
    # comes from a new live process/socket/lock observation on every poll.
    stamp = state.get("timestamp")
    require(
        type(stamp) in (int, float) and 0 <= stamp <= time.time() + 1,
        "resident transition timestamp absent or in the future",
    )
    require(process_identity(identity["pid"]) == identity, "resident process disappeared during observation")
    return dict(
        state_sha256=sha(path),
        process=identity,
        status=state["status"],
        socket_device=endpoint.st_dev,
        socket_inode=endpoint.st_ino,
        transition_timestamp=stamp,
        observed_at=time.time(),
    )


def native_request_id(request):
    return hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def reconcile_tasks(root, pool, executable):
    """Read native tombstones; this neither dispatches nor erases uncertain work."""
    paths = {}
    for name in ("dispatches", "tasks"):
        parent = runtime.checked_path(pool / name)
        if parent.exists():
            require(parent.is_dir(), "resident evidence directory replaced")
            for path in parent.iterdir():
                runtime.checked_path(path)
                require(
                    len(path.name) == 64 and all(c in "0123456789abcdef" for c in path.name),
                    "resident evidence has an invalid identity",
                )
                paths.setdefault(path.name, {})[name] = path
    records = []
    for identity, artifacts in sorted(paths.items()):
        result = dict(request_sha256=identity, status="unknown", automatic_retry_permitted=False)
        try:
            dispatch = artifacts.get("dispatches")
            require(dispatch is not None and dispatch.is_file(), "original dispatch tombstone absent")
            intent = read(dispatch)
            require(
                intent.get("request_sha256") == identity
                and isinstance(intent.get("client"), dict)
                and set(intent["client"]) == {"pid", "boot_id", "start_ticks"},
                "original dispatch identity invalid",
            )
            result["dispatch_sha256"] = sha(dispatch)
            directory = artifacts.get("tasks")
            require(
                directory is not None and directory.is_dir(), "transfer outcome unknown; acceptance absent"
            )
            request_path = runtime.checked_path(directory / "request.json")
            request = read(request_path)
            require(
                native_request_id(request) == identity
                and request.get("root") == str(root)
                and request.get("executable_sha256") == executable,
                "accepted request identity changed",
            )
            result["request"] = request
            acceptance_path = runtime.checked_path(directory / "accepted.json")
            accepted = read(acceptance_path)
            require(
                accepted.get("schema") == "parallel.resident_task_acceptance.v2"
                and accepted.get("request_sha256") == identity
                and accepted.get("worker", {}).get("executable_sha256") == executable,
                "original acceptance identity invalid",
            )
            result["accepted_sha256"] = sha(acceptance_path)
            result["worker"] = accepted["worker"]
            terminal = [
                runtime.checked_path(directory / name)
                for name in ("complete.json", "failure.json")
                if (directory / name).exists()
            ]
            require(len(terminal) == 1, "accepted task has missing or conflicting terminal evidence")
            completion = read(terminal[0])
            success = terminal[0].name == "complete.json"
            require(
                completion.get("schema") == "parallel.resident_task_completion.v2"
                and completion.get("request_sha256") == identity
                and completion.get("accepted_sha256") == result["accepted_sha256"]
                and completion.get("success") is success,
                "original terminal binding invalid",
            )
            if success:
                task = request["task"]
                kind = task["kind"]
                rank = request["rank"]
                require(type(rank) is int and 0 <= rank < 8, "original rank invalid")
                if kind == "Train":
                    output = Path(task["launch"]) / f"complete-{rank}.json"
                elif kind == "Proposal":
                    output = Path(task["output"]) / f"complete-{rank}.json"
                elif kind == "Monitor":
                    output = Path(task["output"])
                else:
                    raise RuntimeError("unsupported original resident task")
                output = runtime.inside(root, output)
                require(sha(output) == completion.get("receipt_sha256"), "committed original output changed")
                result["output"] = str(output)
                result["output_sha256"] = sha(output)
            else:
                require(
                    completion.get("receipt_sha256") is None and isinstance(completion.get("error"), str),
                    "failed original task published successful output evidence",
                )
            result.update(
                status="committed_success" if success else "committed_failure",
                terminal_sha256=sha(terminal[0]),
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            result["reason"] = str(error)
        records.append(result)
    return dict(
        schema="parallel.resident_reconciliation.v1",
        observed_at=time.time(),
        records=records,
        unknown=sum(r["status"] == "unknown" for r in records),
        scope="retained_native_evidence_only",
        execution_authorized=False,
    )


def reconcile_startup(pool, world):
    for rank in range(world):
        path = runtime.checked_path(pool / f"rank-{rank}.json")
        if path.is_file():
            previous = read(path).get("worker", {}).get("process")
            require(
                not isinstance(previous, dict) or process_identity(previous.get("pid")) != previous,
                f"resident rank {rank} has a surviving worker; reconcile before restart",
            )
        with custody_lock(pool / f"rank-{rank}.lock", create=True) as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"resident rank {rank} still has execution custody; reconcile surviving worker before restart"
                ) from error


def verify_pool_ready(pool, world, executable, supervisor):
    state = read(runtime.checked_path(pool / "pool.json"))
    require(
        state.get("schema") == "parallel.resident_gpu_pool.v3"
        and state.get("phase") == "running"
        and state.get("world") == world
        and state.get("executable_sha256") == executable,
        "resident pool is not observed ready",
    )
    require(
        state.get("supervisor") == supervisor
        and process_identity(supervisor["pid"]) == supervisor
        and 0 <= time.time() - state.get("timestamp", 0) <= 5,
        "resident pool readiness is stale or its supervisor changed",
    )
    owner_path = runtime.inside(pool, Path(state.get("ownership", "missing")))
    require(
        owner_path.parent.parent == pool / "supervisors"
        and owner_path.name == "start.json"
        and sha(owner_path) == state.get("ownership_sha256"),
        "resident supervisor ownership binding changed",
    )
    owner = read(owner_path)
    require(
        owner.get("schema") == "parallel.resident_supervisor_ownership.v1"
        and owner.get("supervisor") == supervisor
        and owner.get("worker_executable_sha256") == executable
        and owner.get("supervisor_executable_sha256") == sha(Path(f"/proc/{supervisor['pid']}/exe")),
        "resident supervisor loaded executable or original process changed",
    )
    sources = owner.get("sources", {})
    require(
        sources.get(str(Path(__file__).absolute())) == sha(Path(__file__))
        and all(sha(runtime.checked_path(Path(path))) == digest for path, digest in sources.items()),
        "resident supervisor source binding changed",
    )
    for rank in range(world):
        identity = state.get("processes", {}).get(str(rank))
        require(identity is not None, "resident readiness omits rank")
        rank_observation(pool, rank, identity, executable)
    return state


def run(root, binary, pool, world, owner=None):
    root = runtime.checked_path(root)
    binary = runtime.checked_path(binary)
    pool = runtime.checked_path(pool)
    pool.mkdir(parents=True, exist_ok=True)
    require(
        pool.is_relative_to(root / "runs") and pool != root / "runs",
        "worker pool must be private to the learning workspace",
    )
    require(world in (2, 4, 8), "unsupported worker pool size")
    expected = sha(binary)
    supervisor = process_identity(os.getpid())
    task_lease_path = root / "runs/encoding-service.lock"
    with custody_lock(task_lease_path) as handle:
        lease_metadata = os.fstat(handle.fileno())
        lease_identity = (lease_metadata.st_dev, lease_metadata.st_ino)
    # These are observed source bindings, not a qualified Python/runtime closure.
    sources = {
        str(Path(module.__file__).absolute()): sha(Path(module.__file__))
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and Path(module.__file__).suffix == ".py"
        and Path(module.__file__).absolute().parent == Path(__file__).absolute().parent
    }
    sources[str(Path(__file__).absolute())] = sha(Path(__file__))
    cache_source = Path(__file__).with_name("shared_teaching_cache.py")
    sources[str(cache_source.absolute())] = sha(cache_source)
    ownership = pool / "supervisors" / uuid.uuid4().hex
    ownership.mkdir(parents=True)
    binding = dict(
        schema="parallel.resident_supervisor_ownership.v1",
        supervisor=supervisor,
        supervisor_executable_sha256=sha(Path("/proc/self/exe")),
        worker_executable_sha256=expected,
        sources=sources,
        task_lease_device=lease_identity[0],
        task_lease_inode=lease_identity[1],
        created_at=time.time(),
        runtime_closure_qualified=False,
    )
    atomic(ownership / "start.json", binding)
    lock_context = custody_lock(pool / "supervisor.lock", create=True)
    lock = lock_context.__enter__()
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_metadata = os.fstat(lock.fileno())
    children = {}
    retired = []
    stopping = False
    failure = None
    cache_log = (pool / f"teaching-cache-{time.time_ns()}.log").open("xb")
    cache = None
    cache_command = [
        sys.executable,
        "-B",
        str(Path(__file__).with_name("shared_teaching_cache.py")),
        "--root",
        str(root),
        "--pool",
        str(pool),
    ]

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        reconcile_startup(pool, world)
        reconciliation = reconcile_tasks(root, pool, expected)
        atomic(ownership / "startup-reconciliation.json", reconciliation)
        require(
            reconciliation["unknown"] == 0, "resident original task requires reconciliation before restart"
        )
        while not stopping:
            current_lock = (pool / "supervisor.lock").lstat()
            require(
                (current_lock.st_dev, current_lock.st_ino) == (lock_metadata.st_dev, lock_metadata.st_ino),
                "resident supervisor lock was replaced",
            )
            require(
                all(sha(Path(path)) == digest for path, digest in sources.items()),
                "resident supervisor dependency changed",
            )
            handoff = suspended(root, time.time(), owner)
            if cache is None:
                cache = runtime.popen(cache_command, stdout=cache_log, stderr=cache_log)
            else:
                # Surviving ranks may still mmap every object from this epoch.
                # Recreating only the cache would reset accounting underneath
                # those live borrowers. Cleanup stops/reaps ranks and reconciles.
                require(
                    cache.poll() is None,
                    "sealed cache epoch exited; stop all borrowers and reconcile before a new pool epoch",
                )
            ready = True
            observations = {}
            errors = {}
            for rank in range(world):
                if rank in children and children[rank][0].poll() is not None:
                    child, log, identity = children.pop(rank)
                    log.close()
                    retired.append((child, identity))
                    runtime.append(
                        pool / "journal.jsonl",
                        dict(
                            rank=rank,
                            event="worker_exited",
                            process=identity,
                            exit_code=child.returncode,
                            timestamp=time.time(),
                            ownership_sha256=sha(ownership / "start.json"),
                        ),
                    )
                    reconciliation = reconcile_tasks(root, pool, expected)
                    atomic(ownership / f"exit-{rank}-{time.time_ns()}.json", reconciliation)
                    require(
                        reconciliation["unknown"] == 0,
                        "resident original task requires reconciliation after worker exit",
                    )
                if handoff and rank in children:
                    path = pool / f"rank-{rank}.json"
                    state = read(path) if path.is_file() else {}
                    if idle_worker(state, children[rank][0].pid, children[rank][2], expected):
                        # Native acceptance must already own this lease. Holding it
                        # excludes a task racing the earlier idle status observation.
                        try:
                            with custody_lock(task_lease_path) as task_lease:
                                current = os.fstat(task_lease.fileno())
                                require(
                                    (current.st_dev, current.st_ino) == lease_identity,
                                    "resident task lease identity changed",
                                )
                                fcntl.flock(task_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                rank_observation(pool, rank, children[rank][2], expected)
                                child, log, identity = children.pop(rank)
                                runtime.stop([child])
                                log.close()
                                retired.append((child, identity))
                                runtime.append(
                                    pool / "journal.jsonl",
                                    dict(
                                        rank=rank,
                                        event="worker_drained",
                                        process=identity,
                                        exit_code=child.returncode,
                                        timestamp=time.time(),
                                        ownership_sha256=sha(ownership / "start.json"),
                                    ),
                                )
                        except (BlockingIOError, FileNotFoundError):
                            ready = False
                if not handoff and rank not in children:
                    require(sha(binary) == expected, "worker executable changed")
                    cpu = rank * 2 if rank < 4 else 62 + (rank - 4) * 2
                    log = (pool / f"rank-{rank}-{time.time_ns()}.log").open("xb")
                    child = runtime.popen(
                        [
                            "taskset",
                            "-c",
                            f"{cpu},{cpu + 1}",
                            str(binary),
                            "--root",
                            str(root),
                            "gpu-worker",
                            "--pool",
                            str(pool),
                            "--rank",
                            str(rank),
                        ],
                        stdout=log,
                        stderr=log,
                        env=dict(
                            os.environ,
                            JOPE_TEACHING_CACHE=str(pool / "teaching.sock"),
                            OMP_NUM_THREADS="2",
                            RAYON_NUM_THREADS="2",
                            OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                        ),
                    )
                    identity = process_identity(child.pid)
                    children[rank] = (child, log, identity)
                    require(identity is not None, "resident worker process vanished during launch")
                    runtime.append(
                        pool / "journal.jsonl",
                        dict(
                            rank=rank,
                            event="worker_started",
                            process=identity,
                            timestamp=time.time(),
                            ownership_sha256=sha(ownership / "start.json"),
                        ),
                    )
                if rank in children:
                    try:
                        observation = rank_observation(pool, rank, children[rank][2], expected)
                        observations[str(rank)] = observation
                        ready = ready and observation["status"] in ("idle", "busy")
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                        ready = False
                        errors[str(rank)] = str(error)
                else:
                    ready = False
            atomic(
                pool / "pool.json",
                dict(
                    schema="parallel.resident_gpu_pool.v3",
                    timestamp=time.time(),
                    phase="suspended"
                    if handoff and not children
                    else "draining"
                    if handoff
                    else "running"
                    if ready
                    else "initializing",
                    executable_sha256=expected,
                    pids={str(r): c.pid for r, (c, _, _) in children.items()},
                    processes={str(r): identity for r, (_, _, identity) in children.items()},
                    world=world,
                    supervisor=supervisor,
                    ownership_sha256=sha(ownership / "start.json"),
                    ownership=str(ownership / "start.json"),
                    observations=observations,
                    errors=errors,
                    scope="observed_worker_custody_only",
                ),
            )
            time.sleep(1)
    except BaseException as error:
        failure = str(error)
        raise
    finally:
        try:
            runtime.stop([child for child, _, _ in children.values()] + ([cache] if cache else []))
            owned = [(child, identity) for child, _, identity in children.values()] + retired
            remaining = [
                identity
                for _, identity in owned
                if identity and process_identity(identity["pid"]) == identity
            ]
            reconciliation = reconcile_tasks(root, pool, expected)
            cleanup = dict(
                schema="parallel.resident_supervisor_cleanup.v1",
                supervisor=supervisor,
                ownership_sha256=sha(ownership / "start.json"),
                timestamp=time.time(),
                remaining_owned_processes=remaining,
                reconciliation=reconciliation,
                failure=failure,
                rank_exits={str(rank): child.returncode for rank, (child, _, _) in children.items()},
            )
            atomic(ownership / "complete.json", cleanup)
            atomic(
                pool / "pool.json",
                dict(
                    schema="parallel.resident_gpu_pool.v3",
                    phase="reconciliation_required"
                    if failure or reconciliation["unknown"] or remaining
                    else "stopped",
                    timestamp=time.time(),
                    supervisor=supervisor,
                    cleanup_sha256=sha(ownership / "complete.json"),
                ),
            )
        finally:
            cache_log.close()
            for _, log, _ in children.values():
                log.close()
            lock_context.__exit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "binary", "pool"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--qualification-owner")
    a = p.parse_args()
    run(a.root.absolute(), a.binary.absolute(), a.pool.absolute(), a.world, a.qualification_owner)
