"""Synthetic CPU fixtures; no real dataset admission or model-quality claim."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from jope.policy import ddp, train_v3
from jope.policy.model_v3 import TASK_LANE, load_policy_v3
from jope.schema import TrainingRow


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_classification_only_corpus_selects_classification_lane(task):
    tasks = train_v3._training_task_types([SimpleNamespace(metadata={"task_type": task})])
    assert any(TASK_LANE[value] == "binary" for value in tasks)
    with pytest.raises(ValueError, match="unsupported training tasks"):
        train_v3._training_task_types([SimpleNamespace(metadata={"task_type": "unknown"})])


@pytest.mark.parametrize("world_size", ["1", "2"])
def test_cuda_request_never_silently_becomes_cpu(monkeypatch, world_size):
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        ddp.init_distributed("cuda")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert ddp.init_distributed("cpu").device == "cpu"


@pytest.mark.parametrize("options", [{"epochs": 0}, {"batch_size": 0}, {"lr": 0.0}])
def test_invalid_training_budget_rejected_before_distributed_init(monkeypatch, options):
    monkeypatch.setattr(ddp, "init_distributed", lambda _: pytest.fail("device init before validation"))
    with pytest.raises(ValueError):
        train_v3.train_policy_v3_from_jsonl("nonexistent-synthetic-corpus", **options)


class FakeScaler:
    def __init__(self, skipped):
        self.scale = 16.0
        self.skipped = skipped

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        if not self.skipped:
            optimizer.step()

    def update(self):
        if self.skipped:
            self.scale /= 2


def test_amp_skipped_step_is_not_counted_as_an_update():
    calls = []
    optimizer = SimpleNamespace(step=lambda: calls.append("update"))
    assert not train_v3._step_optimizer(optimizer, FakeScaler(skipped=True))
    assert calls == []
    assert train_v3._step_optimizer(optimizer, FakeScaler(skipped=False))
    assert calls == ["update"]


def synthetic_rows(task="regression"):
    head = {
        "regression": "ridge",
        "binary": "logit_ridge",
        "multiclass": "multiclass_ridge",
    }[task]
    return [
        TrainingRow(
            dataset_id=f"synthetic-{split}",
            source="synthetic-training-contract-fixture",
            embedding=[float(index % 3)] * 512,
            target_embedding=[1.0, 0.5] + [0.0] * 510,
            genome=f"(pipe X ({head}))",
            metric="synthetic-loss",
            score=float(index),
            metadata={"split": split, "task_type": task, "provenance": "corpus"},
        )
        for split in ("train", "development")
        for index in range(2)
    ]


@pytest.fixture
def small_trainer(monkeypatch):
    # Exercise the real optimizer/checkpoint path; validation here is not a quality
    # benchmark, so avoid unrelated engine proposal/evaluation work in this fixture.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    yield {
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
        "d_model": 8,
        "n_layers": 1,
        "n_heads": 2,
        "ffn_dim": 16,
        "greedy_eval_datasets": 0,
        "checkpoint_state": "current",
        "phase_a_fraction": 0.0,
    }
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_real_synthetic_update_reload_and_continuation(tmp_path, small_trainer, task):
    checkpoint = tmp_path / "policy.pt"
    resume = tmp_path / "resume.pt"
    options = {**small_trainer, "training_rows": synthetic_rows(task)}
    first = train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        checkpoint,
        resume_checkpoint=resume,
        **options,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded = load_policy_v3(payload)
    train_v3.verify_training_proof(loaded, payload["training_proof"])
    assert payload["fused_task_heads"] is (task != "regression")
    assert first["training_proof"]["optimizer_updates_this_run"] > 0
    assert not first["training_proof"]["optimizer_state_resumed"]
    first_resume = torch.load(resume, map_location="cpu", weights_only=False)
    first_steps = {
        key: float(state["step"])
        for key, state in first_resume["optimizer"]["state"].items()
        if "step" in state
    }
    assert first_steps and min(first_steps.values()) > 0
    second = train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        checkpoint,
        resume_from=resume,
        resume_checkpoint=resume,
        **options,
    )
    assert second["global_epoch"] == first["global_epoch"] + 1
    assert second["training_proof"]["optimizer_state_resumed"]
    assert (
        second["training_proof"]["initial_parameter_sha256"] == payload["training_proof"]["parameter_sha256"]
    )
    assert second["training_proof"]["optimizer_updates_total"] == (
        first["training_proof"]["optimizer_updates_total"]
        + second["training_proof"]["optimizer_updates_this_run"]
    )
    resumed = torch.load(resume, map_location="cpu", weights_only=False)
    assert all(float(resumed["optimizer"]["state"][key]["step"]) > step for key, step in first_steps.items())
    train_v3.verify_training_proof(load_policy_v3(resumed), resumed["training_proof"])
    assert resumed["training_proof"]["parameter_sha256"] != payload["training_proof"]["parameter_sha256"]

    contract = resumed["training_contract"]
    for path in (
        ("tasks",),
        ("model_metadata", "target_block_rms"),
        ("model_metadata", "action_vocab", "version"),
        ("grammar_specs",),
    ):
        changed = deepcopy(contract)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = "incompatible"
        with pytest.raises(ValueError, match="contract differs"):
            train_v3._validate_resume_contract(resumed, changed)
    with pytest.raises(ValueError, match="lacks a training contract"):
        train_v3._validate_resume_contract({}, contract)
    with torch.no_grad():
        next(loaded.parameters()).add_(1)
    with pytest.raises(ValueError, match="parameter hash"):
        train_v3.verify_training_proof(loaded, payload["training_proof"])


@pytest.mark.parametrize("failure", ["empty", "nonfinite_loss", "unchanged"])
def test_zero_update_or_unchanged_run_cannot_publish_success(tmp_path, small_trainer, monkeypatch, failure):
    if failure == "empty":
        monkeypatch.setattr(train_v3, "_grouped_batches", lambda *args: [])
    elif failure == "nonfinite_loss":
        monkeypatch.setattr(train_v3, "kg_combine", lambda *args: torch.tensor(float("nan")))
    else:
        monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: None)
    checkpoint = tmp_path / "must-not-exist.pt"
    with pytest.raises(
        RuntimeError,
        match="no training batches|zero optimizer updates|parameters unchanged",
    ):
        train_v3.train_policy_v3_from_jsonl(
            tmp_path / "synthetic.jsonl",
            checkpoint,
            training_rows=synthetic_rows(),
            **small_trainer,
        )
    assert not checkpoint.exists()


@pytest.mark.parametrize("corruption", ["incompatible", "missing_optimizer", "empty_optimizer"])
def test_strict_resume_refuses_optimizer_fallback_before_updates(
    tmp_path, small_trainer, monkeypatch, corruption
):
    resume = tmp_path / "resume.pt"
    options = {**small_trainer, "training_rows": synthetic_rows()}
    train_v3.train_policy_v3_from_jsonl(tmp_path / "synthetic.jsonl", resume_checkpoint=resume, **options)
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if corruption == "incompatible":
        payload["optimizer_state_compatible_with_model"] = False
    elif corruption == "missing_optimizer":
        payload.pop("optimizer")
    else:
        payload["optimizer"]["state"] = {}
    torch.save(payload, resume)
    monkeypatch.setattr(
        torch.optim.AdamW, "step", lambda *args, **kwargs: pytest.fail("update before strict admission")
    )
    with pytest.raises(ValueError, match="strict resume.*optimizer"):
        train_v3.train_policy_v3_from_jsonl(
            tmp_path / "synthetic.jsonl", resume_from=resume, strict_resume=True, **options
        )


def test_interrupted_update_resume_matches_uninterrupted_state(tmp_path, small_trainer, monkeypatch):
    rows = []
    for group in range(3):
        for row in deepcopy(synthetic_rows()):
            row.dataset_id += f"-group-{group}"
            rows.append(row)
    options = {**small_trainer, "epochs": 2, "training_rows": rows, "strict_resume": True}
    uninterrupted = tmp_path / "uninterrupted.pt"
    interrupted = tmp_path / "interrupted.pt"
    train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl", resume_checkpoint=uninterrupted, **options
    )
    original = train_v3._atomic_torch_save

    def interrupt_after_commit(payload, path):
        original(payload, path)
        if payload.get("strict_progress") and payload["training_proof"]["optimizer_updates_total"] == 2:
            raise InterruptedError("fixture interruption after committed update")

    with monkeypatch.context() as patcher:
        patcher.setattr(train_v3, "_atomic_torch_save", interrupt_after_commit)
        with pytest.raises(InterruptedError, match="committed update"):
            train_v3.train_policy_v3_from_jsonl(
                tmp_path / "synthetic.jsonl", resume_checkpoint=interrupted, **options
            )
    partial = torch.load(interrupted, map_location="cpu", weights_only=False)
    assert partial["strict_progress"]["sampler"]["next_batch"] == 2
    train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl", resume_from=interrupted, resume_checkpoint=interrupted, **options
    )
    full = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    resumed = torch.load(interrupted, map_location="cpu", weights_only=False)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        elif hasattr(left, "shape"):
            assert (left == right).all()
        else:
            assert left == right

    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "sampler", "global_epoch"):
        equal(full[key], resumed[key])
    assert (
        full["training_proof"]["optimizer_updates_total"]
        == resumed["training_proof"]["optimizer_updates_total"]
    )


def test_amp_overflow_loop_skips_then_updates(tmp_path, small_trainer, monkeypatch):
    from contextlib import nullcontext

    class LoopScaler:
        value = 65536.0
        reductions = 0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.value

        def step(self, optimizer):
            optimizer.step()

        def update(self, new_scale=None):
            if new_scale is not None:
                self.value = new_scale
                self.reductions += 1

        def state_dict(self):
            return {"scale": self.value}

    scaler = LoopScaler()
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: scaler)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = 0

    def overflowing_once(parameters, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor(float("inf"))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", overflowing_once)
    rows = synthetic_rows()
    extra = synthetic_rows()[:2]
    for row in extra:
        row.dataset_id = "synthetic-second-train"
    result = train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        tmp_path / "policy.pt",
        training_rows=rows + extra,
        **small_trainer,
    )
    assert scaler.reductions == 1
    assert result["training_proof"]["amp_skipped_steps_this_run"] == 1
    assert result["training_proof"]["optimizer_updates_this_run"] > 0


@pytest.mark.parametrize("drift", [{"lr": 0.003}, {"seed": 99}, {"amp": True}])
def test_resume_execution_drift_refuses(tmp_path, small_trainer, drift):
    resume = tmp_path / "resume.pt"
    train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        tmp_path / "policy.pt",
        resume_checkpoint=resume,
        training_rows=synthetic_rows(),
        **small_trainer,
    )
    with pytest.raises(
        ValueError,
        match="training task, vocabulary, grammar or context contract differs",
    ):
        train_v3.train_policy_v3_from_jsonl(
            tmp_path / "synthetic.jsonl",
            tmp_path / "next.pt",
            resume_from=resume,
            training_rows=synthetic_rows(),
            **{**small_trainer, **drift},
        )


@pytest.mark.parametrize("ema", [False, True])
def test_historical_publication_explicitly_marks_inherited_state(tmp_path, small_trainer, ema):
    options = {
        **small_trainer,
        "checkpoint_state": "best",
        "anti_overfit_experiment": ema,
    }
    checkpoint = tmp_path / "policy.pt"
    resume = tmp_path / "resume.pt"
    train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        checkpoint,
        resume_checkpoint=resume,
        training_rows=synthetic_rows(),
        **options,
    )
    result = train_v3.train_policy_v3_from_jsonl(
        tmp_path / "synthetic.jsonl",
        checkpoint,
        resume_checkpoint=resume,
        resume_from=resume,
        training_rows=synthetic_rows(),
        **options,
    )
    proof = result["training_proof"]
    assert proof["published_inherited"]
    assert not proof["published_changed_from_initial"]
    assert proof["trained_parameter_sha256"] != proof["initial_parameter_sha256"]
    assert proof["parameter_sha256"] == proof["initial_parameter_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_v3.verify_training_proof(load_policy_v3(payload), proof)


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
        assert psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
