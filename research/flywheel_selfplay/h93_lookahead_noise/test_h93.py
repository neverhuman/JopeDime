"""CPU fixtures for H93 optimizer recipes, guards, pairing, and family readouts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r = load("test_h93_runner", HERE / "runner.py")
w = load("test_h93_wave", HERE / "run_wave.py")


def test_fresh_seeds_share_the_actual_h76_optimizer_and_sampler():
    assert r.SEEDS == (30, 31, 32)
    assert not set(r.SEEDS) & set(range(12, 30))
    assert r.optimizer_step is r.science.optimizer_step
    assert r.batch_plan is r.science.batch_plan
    pools = {str(i): {"fitness": np.arange(200)} for i in range(10)}

    def sampler(pool, order, size):
        return np.random.choice(len(order), size, replace=False)

    a, b = [r.batch_plan(pools, sampler, seed) for seed in (24, 25)]
    assert len(a) == len(b) == 160
    assert a[0].seed != b[0].seed
    replay = r.batch_plan(pools, sampler, 24)
    assert all(
        x.context_ids == y.context_ids
        and all(
            np.array_equal(i, j)
            for i, j in zip(x.candidate_indices, y.candidate_indices, strict=True)
        )
        for x, y in zip(a, replay, strict=True)
    )


def test_fixed_update_plans_keep_two_updates_and_eb4_accumulates_four_batches():
    batches = [
        SimpleNamespace(
            round_index=round_index, step_index=step, seed=round_index * 10 + step
        )
        for round_index in range(1, 21)
        for step in range(8)
    ]
    for arm in ("CONTROL", "LA", "GN"):
        plan = r.make_update_plan(batches, arm)
        assert len(plan) == 40 and all(len(update) == 1 for update in plan)
        assert all(update[0].step_index < 2 for update in plan)
    eb4 = r.make_update_plan(batches, "EB4")
    assert len(eb4) == 40 and all(len(update) == 4 for update in eb4)
    assert [batch.seed for update in eb4[:2] for batch in update] == list(range(10, 18))


def test_lookahead_syncs_every_five_fast_steps_and_carries_slow_weights():
    parameter = torch.tensor([0.0])
    state = r.LookaheadState([parameter])
    for _ in range(4):
        parameter.add_(1.0)
        assert not state.after_fast_step()
    parameter.add_(1.0)
    assert state.after_fast_step()
    assert parameter.item() == pytest.approx(2.5)
    for _ in range(5):
        parameter.add_(1.0)
        synced = state.after_fast_step()
    assert synced and state.steps == 10
    assert parameter.item() == pytest.approx(5.0)


def test_annealed_gradient_noise_is_seed_keyed_and_has_preregistered_variance():
    assert r.noise_variance(0) == pytest.approx(0.01)
    assert r.noise_variance(39) == pytest.approx(0.01 / 40**0.55)
    first = torch.randn((8,), generator=r.noise_generator(30))
    replay = torch.randn((8,), generator=r.noise_generator(30))
    other = torch.randn((8,), generator=r.noise_generator(31))
    assert torch.equal(first, replay) and not torch.equal(first, other)


def test_accumulated_update_averages_microbatch_losses_before_one_clipped_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    batches = [1.0, 3.0, 5.0, 7.0]
    loss, norm, synced = r.accumulated_step(
        [parameter],
        optimizer,
        batches,
        lambda target: (parameter - target).square().mean(),
    )
    assert loss == pytest.approx(14.0)
    assert norm == pytest.approx(6.0)
    assert not synced and parameter.item() == pytest.approx(1.1)


def test_gradient_noise_is_added_before_clipping_with_the_fixed_stream():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    generator = r.noise_generator(30)
    expected = torch.randn((1,), generator=r.noise_generator(30)).item() * 0.1
    r.accumulated_step(
        [parameter],
        optimizer,
        [None],
        lambda _: parameter.sum() * 0.0,
        generator=generator,
        update_index=0,
    )
    assert parameter.grad.item() == pytest.approx(expected)


def test_top4_overlap_is_context_paired_and_normalized():
    candidate = {"a": [4, 5, 6, 7], "b": [4, 5, 6, 7]}
    reference = {"a": [4, 5, 8, 9], "b": [8, 9, 10, 11]}
    assert r.top4_overlap(candidate, reference) == pytest.approx(0.25)
    with pytest.raises(RuntimeError, match="roster"):
        r.top4_overlap(candidate, {"a": [4, 5, 6, 7]})


def test_reference_restores_frozen_adapter_and_preserves_other_weights():
    net = torch.nn.Module()
    net.adapter_10tok = torch.nn.Linear(1, 29211, bias=False)
    net.other = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    net.adapter_10tok.weight.requires_grad_(False)
    state = {"adapter_10tok.weight": torch.ones_like(net.adapter_10tok.weight)}
    r.restore_adapter(net, state)
    assert torch.equal(net.adapter_10tok.weight, state["adapter_10tok.weight"])
    assert not net.adapter_10tok.weight.requires_grad and net.other.item() == 7
    with pytest.raises(RuntimeError):
        r.restore_adapter(net, {"other": torch.ones(29211)})


def test_scoring_marker_is_durable_monotonic_and_token_bound(tmp_path):
    phase = tmp_path / "attempt.phase.json"
    r.atomic_json(phase, {"attempt_token": "right", "outcome_started": False})
    args = SimpleNamespace(phase=phase, attempt_token="right")
    r.mark_phase(args, "training")
    assert json.loads(phase.read_text())["outcome_started"] is False
    r.mark_phase(args, "scoring", outcome_started=True)
    assert json.loads(phase.read_text())["outcome_started"] is True
    with pytest.raises(RuntimeError, match="clear"):
        r.mark_phase(args, "training")
    with pytest.raises(RuntimeError, match="token"):
        r.mark_phase(
            SimpleNamespace(phase=phase, attempt_token="wrong"),
            "scoring",
            outcome_started=True,
        )


def attempt_record(attempt=1):
    return {
        "arm": "LA",
        "seed": 30,
        "attempt": attempt,
        "error_kind": "host_load",
        "attempt_token": "x",
    }


def test_only_two_pre_scoring_load_retries_are_eligible():
    phase = {"arm": "LA", "seed": 30, "attempt_token": "x", "outcome_started": False}
    assert w.retryable(attempt_record(1), phase, result_exists=False)
    assert w.retryable(attempt_record(2), phase, result_exists=False)
    assert not w.retryable(attempt_record(3), phase, result_exists=False)
    assert not w.retryable(attempt_record(), phase, result_exists=True)
    for bad in (
        {},
        dict(phase, outcome_started=True),
        dict(phase, outcome_started=None),
        dict(phase, attempt_token="other"),
    ):
        assert not w.retryable(attempt_record(), bad, result_exists=False)
    assert not w.retryable(
        dict(attempt_record(), error_kind="fatal"), phase, result_exists=False
    )


def test_execute_never_exceeds_three_attempts_or_resets_void_charges(tmp_path):
    ledger = {"attempts": [], "in_progress": None}
    calls = []

    def attempt(root, current, arm, seed, timing, index):
        calls.append(index)
        row = {"wall_seconds": 10, "wait_seconds": 30}
        current["attempts"].append(row)
        return ({"complete": True} if index == 3 else None), row

    result, _ = w.execute(tmp_path, ledger, "LA", 30, attempt_fn=attempt)
    assert result["complete"] and calls == [1, 2, 3]
    assert w.remaining(ledger) == 2670

    def always_void(root, current, arm, seed, timing, index):
        return None, {}

    with pytest.raises(RuntimeError, match="two pre-outcome"):
        w.execute(tmp_path, ledger, "LA", 30, attempt_fn=always_void)
    with pytest.raises(RuntimeError, match="unaccounted"):
        w.remaining(dict(ledger, in_progress={"arm": "LA"}))


def test_load_gate_resets_and_requires_two_checks_30_seconds_apart(tmp_path):
    loads = iter([90, 80, 85.1, 85, 84])
    now = [0]

    def sleep(seconds):
        assert seconds == 30
        now[0] += seconds

    wait = w.cpu_gate(
        tmp_path / "status.log",
        load_fn=lambda: (next(loads), 0, 0),
        sleep_fn=sleep,
        clock_fn=lambda: now[0],
    )
    assert wait == 120
    now[0] = 0
    with pytest.raises(RuntimeError, match="30 minutes"):
        w.cpu_gate(
            tmp_path / "status.log",
            load_fn=lambda: (90, 0, 0),
            sleep_fn=sleep,
            clock_fn=lambda: now[0],
            max_wait=60,
        )


def test_lease_released_before_phase_read_and_retry_decision(tmp_path, monkeypatch):
    events = []

    class Lease:
        def fileno(self):
            return 99

        def close(self):
            events.append("close")

    lease = Lease()
    monkeypatch.setattr(w, "cpu_gate", lambda status: 30)
    monkeypatch.setattr(w.fair, "acquire_lock", lambda *a, **k: (lease, 1, 24000, {}))
    monkeypatch.setattr(w.fcntl, "flock", lambda *a: events.append("unlock"))

    def abort(command, log, timeout, held, record):
        assert held is lease
        events.append("child")
        raise w.HostLoadAbort(90)

    monkeypatch.setattr(w, "run_child", abort)
    ledger = {"attempts": [], "in_progress": None}
    result, row = w.run_attempt(tmp_path, ledger, "LA", 30, False, 1)
    assert result is None and row["outcome_started"] is False
    assert events == ["child", "unlock", "close"]
    assert row["wall_seconds"] > 0 and len(ledger["attempts"]) == 1
    assert json.loads((tmp_path / "budget.json").read_text())["in_progress"] is None


def fixture_cells(gain):
    cells = []
    for arm in r.ARMS:
        for seed in r.SEEDS:
            value = 0 if arm == "CONTROL" else gain
            dev = {
                "mean": value,
                "per_family": {str(i): value for i in range(19)},
                "per_context": {str(i): value for i in range(91)},
            }
            cell = {
                "wave": "H93",
                "arm": arm,
                "seed": seed,
                "timing_only": False,
                "updates": 40,
                "trainable_count": 29211,
                "dope_before": r.h59.DOPE_SHA,
                "dope_after": r.h59.DOPE_SHA,
                "checkpoint_sha256": r.h59.R882_SHA,
                "manifest_sha256": r.h59.MANIFEST_SHA,
                "selected_sha256": r.SELECTED_SHA,
                "initial_adapter_sha256": "initial",
                "final_adapter_sha256": "final",
                "adapter_l2_from_r882": 0.1,
                "dev_top4_indices": {str(i): [4, 5, 6, 7] for i in range(91)},
                "source_commit": "source",
                "prereg_sha256": "prereg",
                "folds": [],
                "train": dev,
                "dev": dev,
            }
            if arm == "CONTROL":
                cell["r882_dev"] = {"per_family": {str(i): 0.1 for i in range(19)}}
                cell["r882_top4_indices"] = {
                    str(i): ([4, 5, 6, 7] if i % 2 == 0 else [8, 9, 10, 11])
                    for i in range(91)
                }
            cells.append(cell)
    return cells


@pytest.mark.parametrize(
    "gain,admitted", [(0, False), (0.001546, False), (0.002, True), (-0.002, False)]
)
def test_admission_is_strict_and_has_both_requested_elo_references(gain, admitted):
    result = r.summarize(fixture_cells(gain), draws=100)
    assert result["arms"]["LA"]["admitted"] is admitted
    assert result["arms"]["LA"]["seed_pseudoreplication"]["identical_utility_flag"]
    assert result["arms"]["LA"]["seed_pseudoreplication"]["scalar_utility_groups"] == [
        [30, 31, 32]
    ]
    assert result["arms"]["LA"]["top4_overlap_with_r882_median"] == pytest.approx(
        46 / 91
    )
    assert result["arms"]["LA"]["adapter_l2_from_r882_median"] == pytest.approx(0.1)
    assert result["arms"]["CONTROL"]["elo_vs_control"]["median_elo"] == 1500
    assert len(result["arms"]["LA"]["elo_vs_control"]["per_seed"]) == 3
    assert len(result["arms"]["LA"]["elo_vs_r882"]["per_seed"]) == 3
    json.dumps(result, allow_nan=False)


def test_elo_seed_median_is_not_pooled_elo():
    gains = np.asarray([[1 if i < n else -1 for i in range(19)] for n in (2, 10, 12)])
    sample = np.random.default_rng(93000).integers(0, 19, size=(100, 19))
    value = r.elo_summary(gains, sample)
    assert value["median_elo"] == pytest.approx(1500 + 400 * np.log10(10 / 9))
    assert value["pooled_elo"] < 1500 < value["median_elo"]


def test_missing_cells_and_mixed_source_or_old_seed_fail():
    cells = fixture_cells(0.01)
    with pytest.raises(RuntimeError):
        r.summarize(cells[:-1])
    cells[-1]["source_commit"] = "other"
    with pytest.raises(RuntimeError):
        r.summarize(cells)
    cells = fixture_cells(0.01)
    cells[-1]["seed"] = 12
    with pytest.raises(RuntimeError):
        r.summarize(cells)


def test_load_guard_after_scoring_is_fatal_not_retryable(tmp_path, monkeypatch):
    class Lease:
        def fileno(self):
            return 99

        def close(self):
            pass

    monkeypatch.setattr(w, "cpu_gate", lambda status: 30)
    monkeypatch.setattr(w.fair, "acquire_lock", lambda *a, **k: (Lease(), 1, 24000, {}))
    monkeypatch.setattr(w.fcntl, "flock", lambda *a: None)

    def abort(command, log, timeout, lease, record):
        phase = Path(command[command.index("--phase") + 1])
        data = json.loads(phase.read_text())
        data["outcome_started"] = True
        r.atomic_json(phase, data)
        raise w.HostLoadAbort(90)

    monkeypatch.setattr(w, "run_child", abort)
    ledger = {"attempts": [], "in_progress": None}
    with pytest.raises(w.HostLoadAbort):
        w.run_attempt(tmp_path, ledger, "LA", 30, False, 1)
    assert ledger["attempts"][0]["outcome_started"] is True


def test_h93_run_root_cannot_read_h76_h82_or_reset_budget_elsewhere(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    expected = tmp_path / "jope-data/h93/h93-wave"
    w.validate_run_root(expected)
    with pytest.raises(RuntimeError, match="H76/H82"):
        w.validate_run_root(tmp_path / "jope-data/h76/h76-wave")
    with pytest.raises(RuntimeError, match="H76/H82"):
        w.validate_run_root(tmp_path / "jope-data/h82/h82-wave")
    with pytest.raises(RuntimeError, match="alternate ledger"):
        w.validate_run_root(tmp_path / "another-run")
    expected.parent.mkdir(parents=True)
    old = tmp_path / "jope-data/h82/h82-wave"
    old.mkdir(parents=True)
    expected.symlink_to(old, target_is_directory=True)
    with pytest.raises(RuntimeError, match="H76/H82"):
        w.validate_run_root(expected)


def test_launch_barrier_requires_h80_closeout_and_live_h90_ticket(tmp_path):
    closeout = tmp_path / "H80_closeout.json"
    queue = tmp_path / "queue"
    queue.mkdir()
    alive = {123}
    with pytest.raises(RuntimeError, match="H80 closeout"):
        w.validate_launch_barrier(closeout, queue, is_alive_fn=lambda pid: pid in alive)
    closeout.write_text(
        json.dumps({"schema": "jope.h80.closeout/v1", "wave": "H80", "complete": True})
    )
    with pytest.raises(RuntimeError, match="H90 holds"):
        w.validate_launch_barrier(closeout, queue, is_alive_fn=lambda pid: pid in alive)
    (queue / "1-123-H90").write_text("epoch_ns=1\npid=123\nwave=H90\n")
    assert (
        w.validate_launch_barrier(
            closeout, queue, is_alive_fn=lambda pid: pid in alive
        )["h90_ticket"]
        == "1-123-H90"
    )


def test_cpu_dry_run_reads_no_dev_and_creates_no_run_root(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    root = tmp_path / "jope-data/h93/h93-wave"
    monkeypatch.setattr(
        sys, "argv", ["run_wave.py", "--cpu-dry-run", "--root", str(root)]
    )
    w.main()
    result = json.loads(capsys.readouterr().out)
    assert result["wave"] == "H93" and result["reads_dev"] is False
    assert result["created_fifo_ticket"] is False and not root.exists()
