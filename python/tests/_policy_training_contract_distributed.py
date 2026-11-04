"""Distributed launch and worker cleanup tests for the policy training contract."""

from __future__ import annotations

import pytest
import torch

from jope.policy import ddp


@pytest.mark.parametrize(
    "updates",
    [
        {"WORLD_SIZE": "0"},
        {"WORLD_SIZE": "-1"},
        {"WORLD_SIZE": "bad"},
        {"RANK": None},
        {"RANK": "-1"},
        {"RANK": "2"},
        {"LOCAL_RANK": None},
        {"LOCAL_RANK": "-1"},
        {"MASTER_ADDR": None},
        {"MASTER_ADDR": " "},
        {"MASTER_PORT": None},
        {"MASTER_PORT": "0"},
        {"MASTER_PORT": "65536"},
    ],
)
def test_distributed_environment_refused_before_collectives(monkeypatch, updates):
    for key, value in {
        "WORLD_SIZE": "2",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        **updates,
    }.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: pytest.fail("invalid topology reached rendezvous"),
    )
    with pytest.raises(ValueError):
        ddp.init_distributed("cpu")


@pytest.mark.parametrize("world_size", [2, 3])
def test_real_distributed_cpu_updates_and_shards(tmp_path, world_size):
    """Exercise env:// startup and actual collectives, with a serial gradient reference."""
    import os
    import socket
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent("""
        import torch
        from jope.policy import ddp
        ctx = ddp.init_distributed("cpu", timeout_seconds=20)
        try:
            torch.set_num_threads(1)
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(ctx.rank + 1.)
            ddp.wrap_ddp(model, ctx)
            assert model.weight.item() == 1.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            x = torch.tensor([[float(ctx.rank + 1)]])
            model(x).square().mean().backward()
            ddp.all_reduce_gradients(model, ctx)
            optimizer.step()
            expected = 1. - 0.2 * sum(i*i for i in range(1, ctx.world_size+1)) / ctx.world_size
            assert abs(model.weight.item() - expected) < 1e-6
            assert model.weight.item() != 1.
            # Uneven shards retain every example at this helper boundary. The
            # trainer's equal-step truncation remains a separate limitation.
            shards = ddp.shard_for_rank(list(range(7)), ctx)
            totals = ddp.aggregate_scalars({"count": len(shards), "sum": sum(shards)}, ctx)
            assert totals == {"count": 7., "sum": 21.}
            assert ddp.all_ranks_true(True, ctx)
            assert not ddp.all_ranks_true(ctx.rank != 0, ctx)
            # A rank-local unused parameter contributes zero, without mismatched collectives.
            optimizer.zero_grad(set_to_none=True)
            if ctx.rank == 0:
                model.weight.sum().backward()
            ddp.all_reduce_gradients(model, ctx)
            assert abs(model.weight.grad.item() - 1. / ctx.world_size) < 1e-6
            ddp.barrier(ctx)
            print(f"rank={ctx.rank} updates=1 weight={model.weight.item()} OK", flush=True)
        finally:
            ddp.shutdown_distributed(ctx)
    """)
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    # Own every worker directly rather than relying on torchrun's detached
    # grandchildren. These are the same env:// values supplied by torchrun.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    children = []
    try:
        for rank in range(world_size):
            rank_env = {
                **env,
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
            children.append(
                subprocess.Popen(
                    [sys.executable, str(worker)],
                    env=rank_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )
        outputs = _finish_distributed_workers(children, timeout=90)
        for rank, (child, output) in enumerate(zip(children, outputs)):
            assert child.returncode == 0, output
            assert f"rank={rank} updates=1" in output
    finally:
        _stop_distributed_workers(children)


def test_distributed_existing_group_mismatch_refused(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="existing process group"):
        ddp.init_distributed("cpu")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_distributed_timeout_must_be_bounded(timeout):
    with pytest.raises(ValueError, match="timeout"):
        ddp.init_distributed("cpu", timeout_seconds=timeout)


def test_distributed_startup_passes_explicit_topology_and_timeout(monkeypatch):
    from datetime import timedelta

    for key, value in {
        "WORLD_SIZE": "3",
        "RANK": "2",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs))
    ctx = ddp.init_distributed("cpu", timeout_seconds=7)
    assert (ctx.rank, ctx.world_size, ctx.local_rank, ctx.device) == (2, 3, 0, "cpu")
    assert calls == [
        {
            "backend": "gloo",
            "init_method": "env://",
            "rank": 2,
            "world_size": 3,
            "timeout": timedelta(seconds=7),
        }
    ]


@pytest.mark.parametrize("requested,local_rank,count", [("cuda", "1", 1), ("cuda:1", "0", 2)])
def test_distributed_cuda_mapping_refused_before_device_selection(monkeypatch, requested, local_rank, count):
    for key, value in {
        "WORLD_SIZE": "2",
        "RANK": "0",
        "LOCAL_RANK": local_rank,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("invalid CUDA mapping selected"))
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        ddp.init_distributed(requested)


def _stop_distributed_workers(children):
    import os
    import signal

    # The test owns each worker session, including descendants if its leader
    # exits first. No torchrun launcher can orphan independently detached ranks.
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children:
        child.communicate(timeout=5)


def _finish_distributed_workers(children, *, timeout):
    import time

    deadline = time.monotonic() + timeout
    return [child.communicate(timeout=max(0.01, deadline - time.monotonic()))[0] for child in children]


@pytest.mark.parametrize("parent_exits", [False, True])
def test_distributed_probe_timeout_reaps_owned_worker_group(tmp_path, parent_exits):
    import subprocess
    import sys
    import time

    import psutil

    pid_file = tmp_path / "worker.pid"
    # Worker descendants inherit the directly owned session. Retain the pipe
    # after the leader exits to prove cleanup does not depend on a live leader.
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
        # jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
        "Path(sys.argv[1]).write_text(str(p.pid)); " + ("sys.exit(0)" if parent_exits else "time.sleep(300)")
    )
    children = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(pid_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    ]
    start = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _finish_distributed_workers(children, timeout=2)
    finally:
        _stop_distributed_workers(children)
    assert time.monotonic() - start < 15
    assert children[0].poll() is not None
    worker_pid = int(pid_file.read_text())
    try:
        assert psutil.Process(worker_pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        pass
