"""CPU fixtures for H90 retry boundaries, fresh pairing, and family Elo/readout."""

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


r = load("test_h90_runner", HERE / "runner.py")
w = load("test_h90_wave", HERE / "run_wave.py")


def test_fresh_seeds_share_the_actual_h76_optimizer_and_sampler():
    assert r.SEEDS == (27, 28, 29)
    assert not set(r.SEEDS) & set(range(12, 27))
    assert r.optimizer_step is r.science.optimizer_step
    assert r.batch_plan is r.science.batch_plan
    pools = {str(i): {"fitness": np.arange(200)} for i in range(10)}

    def sampler(pool, order, size):
        return np.random.choice(len(order), size, replace=False)

    a, b = [r.batch_plan(pools, sampler, seed) for seed in (27, 28)]
    assert len(a) == len(b) == 160
    assert a[0].seed != b[0].seed
    replay = r.batch_plan(pools, sampler, 27)
    assert all(
        x.context_ids == y.context_ids
        and all(
            np.array_equal(i, j)
            for i, j in zip(x.candidate_indices, y.candidate_indices, strict=True)
        )
        for x, y in zip(a, replay, strict=True)
    )


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
        "arm": "LS",
        "seed": 27,
        "attempt": attempt,
        "error_kind": "host_load",
        "attempt_token": "x",
    }


def test_only_two_pre_scoring_load_retries_are_eligible():
    phase = {"arm": "LS", "seed": 27, "attempt_token": "x", "outcome_started": False}
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

    result, _ = w.execute(tmp_path, ledger, "LS", 27, attempt_fn=attempt)
    assert result["complete"] and calls == [1, 2, 3]
    assert w.remaining(ledger) == 2670

    def always_void(root, current, arm, seed, timing, index):
        return None, {}

    with pytest.raises(RuntimeError, match="two pre-outcome"):
        w.execute(tmp_path, ledger, "LS", 27, attempt_fn=always_void)
    with pytest.raises(RuntimeError, match="unaccounted"):
        w.remaining(dict(ledger, in_progress={"arm": "LS"}))


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
    monkeypatch.setattr(w, "require_h80_closed", lambda root: {})
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
    result, row = w.run_attempt(tmp_path, ledger, "LS", 27, False, 1)
    assert result is None and row["outcome_started"] is False
    assert events == ["child", "unlock", "close"]
    assert row["wall_seconds"] > 0 and len(ledger["attempts"]) == 1
    assert json.loads((tmp_path / "budget.json").read_text())["in_progress"] is None


def fixture_cells(gain):
    cells = []
    for arm in r.ARMS:
        for seed in r.SEEDS:
            value = 0 if arm == "CONTROL" else gain
            dev = {"mean": value, "per_family": {str(i): value for i in range(19)}}
            cell = {
                "wave": "H90",
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
                "source_commit": "source",
                "prereg_sha256": "prereg",
                "folds": [],
                "train": dev,
                "dev": dev,
            }
            cell["recipe_sha256"] = r.recipes.recipe_hash(arm)
            cell["adapter_l2_from_r882"] = 0.1
            cell["dev"]["per_context"] = {f"{i:064x}": value for i in range(91)}
            cell["dev"]["top4"] = {
                f"{i:064x}": ["p4", "p5", "p6", "p7"] for i in range(91)
            }
            if arm == "CONTROL":
                cell["r882_dev"] = {
                    "per_family": {str(i): 0.1 for i in range(19)},
                    "top4": dict(cell["dev"]["top4"]),
                }
            cells.append(cell)
    return cells


@pytest.mark.parametrize(
    "gain,admitted", [(0, False), (0.001546, False), (0.002, True), (-0.002, False)]
)
def test_admission_is_strict_and_has_both_requested_elo_references(gain, admitted):
    result = r.summarize(fixture_cells(gain), draws=100)
    assert result["arms"]["LS"]["admitted"] is admitted
    assert result["arms"]["CONTROL"]["elo_vs_control"]["median_elo"] == 1500
    assert len(result["arms"]["LS"]["elo_vs_control"]["per_seed"]) == 3
    assert len(result["arms"]["LS"]["elo_vs_r882"]["per_seed"]) == 3
    json.dumps(result, allow_nan=False)


def test_elo_seed_median_is_not_pooled_elo():
    gains = np.asarray([[1 if i < n else -1 for i in range(19)] for n in (2, 10, 12)])
    sample = np.random.default_rng(90000).integers(0, 19, size=(100, 19))
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
    monkeypatch.setattr(w, "require_h80_closed", lambda root: {})

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
        w.run_attempt(tmp_path, ledger, "LS", 27, False, 1)
    assert ledger["attempts"][0]["outcome_started"] is True


def test_h90_run_root_cannot_read_h76_or_reset_budget_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    expected = tmp_path / "jope-data/h90/h90-wave"
    w.validate_run_root(expected)
    with pytest.raises(RuntimeError, match="H76 outcome"):
        w.validate_run_root(tmp_path / "jope-data/h76/h76-wave")
    with pytest.raises(RuntimeError, match="alternate ledger"):
        w.validate_run_root(tmp_path / "another-run")
    expected.parent.mkdir(parents=True)
    old = tmp_path / "jope-data/h76/h76-wave"
    old.mkdir(parents=True)
    expected.symlink_to(old, target_is_directory=True)
    with pytest.raises(RuntimeError, match="H76 outcome"):
        w.validate_run_root(expected)


def test_family_mass_and_actual_training_plan_are_fixed_and_repeatable():
    pools = {
        f"{i:064x}": {
            "fitness": np.arange(256),
            "rec": {"family_id": "a" if i == 0 else "b"},
        }
        for i in range(4)
    }
    weights = r.recipes.family_weights(pools)
    assert weights[f"{0:064x}"] == 1
    assert sum(weights[f"{i:064x}"] for i in (1, 2, 3)) == pytest.approx(1)

    def sampler(pool, order, count):
        return np.random.choice(order, count, replace=False)

    plans = {arm: r.training_plan(pools, sampler, 27, arm) for arm in r.ARMS}
    assert all(
        len(plan) == 40 and sum(len(b.context_ids) for b in plan) == 640
        for plan in plans.values()
    )
    for arm in ("LS", "WCR"):
        assert all(
            a.context_ids == b.context_ids
            and all(
                np.array_equal(x, y)
                for x, y in zip(a.candidate_indices, b.candidate_indices, strict=True)
            )
            for a, b in zip(plans["CONTROL"], plans[arm], strict=True)
        )
    replay = r.training_plan(pools, sampler, 27, "FBGA")
    assert all(
        a.context_ids == b.context_ids
        and all(
            np.array_equal(x, y)
            for x, y in zip(a.candidate_indices, b.candidate_indices, strict=True)
        )
        for a, b in zip(plans["FBGA"], replay, strict=True)
    )
    assert any(
        a.context_ids != b.context_ids
        for a, b in zip(plans["CONTROL"], replay, strict=True)
    )
    with pytest.raises(ValueError):
        r.recipes.family_weights({"x": {"fitness": np.arange(256)}})


def test_smoothing_preserves_context_means_and_original_targets():
    original = torch.arange(384, dtype=torch.float64).reshape(2, 192)
    before = original.clone()
    smooth = r.recipes.smooth_targets(original)
    assert torch.equal(original, before)
    assert torch.allclose(smooth.mean(1), original.mean(1))
    assert torch.allclose(
        smooth - smooth.mean(1, keepdim=True),
        0.9 * (original - original.mean(1, keepdim=True)),
    )
    assert torch.equal(r.recipes.smooth_targets(torch.ones(2, 192)), torch.ones(2, 192))
    with pytest.raises(ValueError):
        r.recipes.smooth_targets(torch.ones(2, 191))


def test_microbatch_accumulation_matches_one_batch_and_one_decay():
    import copy

    torch.manual_seed(27)
    full = torch.nn.Linear(2, 192, dtype=torch.float64)
    micro = copy.deepcopy(full)
    x = torch.randn(16, 2, dtype=torch.float64)
    targets = torch.randn(16, 192, dtype=torch.float64)
    left = torch.optim.AdamW(full.parameters(), lr=1e-4, weight_decay=1e-4)
    right = torch.optim.AdamW(micro.parameters(), lr=1e-4, weight_decay=1e-4)
    batch = r.h59.TrainingBatch(
        1,
        0,
        27,
        tuple(str(i) for i in range(16)),
        tuple(np.arange(192) for _ in range(16)),
    )
    r.optimizer_step(
        full.named_parameters(),
        left,
        lambda: r.h59.loss_by_arm("CONTROL", full(x), targets),
        "CONTROL",
    )
    calls = []

    def predict(cids, indexes):
        calls.append(len(cids))
        return micro(x[[int(c) for c in cids]])

    r.recipes.accumulated_step(
        micro.named_parameters(),
        right,
        batch,
        predict,
        lambda b, device: targets[[int(c) for c in b.context_ids]],
        r.h59.loss_by_arm,
    )
    assert calls == [8, 8]
    for a, b in zip(full.parameters(), micro.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    assert all(value["step"].item() == 1 for value in right.state.values())


def test_fixed_restart_lr_and_adapter_distance():
    lr = r.recipes.learning_rate
    assert lr(1) == 1e-4 and lr(2) == 2e-4
    assert lr(10) == lr(20) == pytest.approx(1e-5)
    assert lr(11) == 2e-4
    assert all(lr(i) >= lr(i + 1) for i in range(2, 10))
    assert all(lr(i) >= lr(i + 1) for i in range(11, 20))
    for index in (0, 21):
        with pytest.raises(ValueError):
            lr(index)
    assert (
        r.recipes.adapter_l2({"x": torch.zeros(2)}, {"x": torch.tensor([3.0, 4.0])})
        == 5
    )


def test_top4_excludes_anchors_and_multisets_handle_duplicates():
    pools = {"c": {"pids": ["anchor"] * 4 + ["a", "a", "b", "c", "d"]}}
    scores = {"c": np.asarray([-999.0] * 4 + [1.0, 1.0, 2.0, 3.0, 4.0])}
    chosen = r.recipes.top_four(pools, scores)
    assert chosen == {"c": ["a", "a", "b", "c"]}
    assert r.recipes.overlap(chosen, chosen) == 1
    assert r.recipes.overlap(chosen, {"c": ["a", "b", "d", "e"]}) == 0.5
    with pytest.raises(ValueError):
        r.recipes.top_four(pools, {"c": np.full(9, np.nan)})


def test_identical_utilities_flag_pseudoreplication_not_three_confirmations():
    metrics = r.summarize(fixture_cells(0.002), draws=100)
    for arm in r.ARMS:
        diag = metrics["arms"][arm]["seed_pseudoreplication"]
        assert diag["identical_utility_flag"]
        assert diag["context_utility_groups"] == [[27, 28, 29]]
        assert diag["independent_seed_replicates_claimed"] == 0
        assert metrics["arms"][arm]["diversity"]["top4_overlap_mean"] == 1
    assert metrics["arms"]["LS"][
        "admitted"
    ]  # gate unchanged; no independent-seed claim


def test_h80_active_or_unproven_never_creates_a_ticket(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("FIFO acquisition before H80 closeout")

    monkeypatch.setattr(w.fair, "acquire_lock", forbidden)
    with pytest.raises(w.H80Pending):
        w.run_attempt(
            tmp_path, {"attempts": [], "in_progress": None}, "CONTROL", 27, False, 1
        )
    proof = tmp_path / "H80_t0030.md"
    proof.write_text("H80 CLOSED; owner closeout evidence")
    monkeypatch.setattr(w, "H80_RECEIPT", proof)
    (tmp_path / "h80-closeout.json").write_text(
        json.dumps(
            {
                "schema": "jope.h90.h80-closeout/v1",
                "wave": "H80",
                "closed_out": True,
                "coordinator_pid": 123,
                "evidence_path": str(proof),
                "evidence_sha256": r.h59.sha256(proof),
            }
        )
    )
    queue = tmp_path / "queue"
    queue.mkdir()
    assert w.require_h80_closed(tmp_path, queue_dir=queue, alive_fn=lambda pid: False)[
        "closed_out"
    ]
    with pytest.raises(w.H80Pending):
        w.require_h80_closed(tmp_path, queue_dir=queue, alive_fn=lambda pid: True)
    (queue / "1-123-H80").write_text("H80")
    with pytest.raises(w.H80Pending):
        w.require_h80_closed(tmp_path, queue_dir=queue, alive_fn=lambda pid: False)
    (queue / "1-123-H80").unlink()
    proof.write_text("changed evidence")
    with pytest.raises(w.H80Pending):
        w.require_h80_closed(tmp_path, queue_dir=queue, alive_fn=lambda pid: False)


def test_h80_wait_stays_outside_fifo(tmp_path):
    calls = []

    def check(root):
        calls.append(root)
        if len(calls) < 3:
            raise w.H80Pending("not closed")
        return {"closed_out": True}

    sleeps = []
    assert w.wait_h80_closed(tmp_path, sleep_fn=sleeps.append, check_fn=check)[
        "closed_out"
    ]
    assert sleeps == [30, 30] and not (tmp_path / "qualification.queue").exists()


@pytest.mark.parametrize("charge", [-1, float("nan"), float("inf"), None, True])
@pytest.mark.parametrize("field", ["wall_seconds", "wait_seconds"])
def test_unknown_or_invalid_budget_charge_cannot_reset_cap(charge, field):
    attempt = {"wall_seconds": 1.0, "wait_seconds": 1.0, field: charge}
    with pytest.raises(RuntimeError, match="budget charge"):
        w.remaining({"attempts": [attempt], "in_progress": None})


def test_dry_run_opens_train_payloads_only_and_never_builds_model(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    train, dev = "a" * 64, "b" * 64
    (tmp_path / "pools").mkdir()
    (tmp_path / "tables").mkdir()
    header = (
        b"JTT1"
        + (1).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + b"\x02"
    )
    (tmp_path / "tables" / f"{train}.jtt1").write_bytes(header)
    np.savez(
        tmp_path / "pools" / f"{train}.npz",
        fitness=np.arange(256),
        program_sha256=np.arange(256),
    )
    row = {
        "context_sha256": train,
        "family_id": "f",
        "task": "regression",
        "dl_split": "DL-Train",
        "source_split": "train",
        "pool_sha256": "pool",
        "train_only_jtt1_sha256": "table",
    }
    hashes = {
        "source_manifest.json": r.h59.MANIFEST_SHA,
        "selected.json": r.SELECTED_SHA,
        "cpu_exact_fused.py": r.FUSED_SHA,
        "checkpoint.pt": r.h59.R882_SHA,
        f"{train}.npz": "pool",
        f"{train}.jtt1": "table",
        "PREREG.md": "prereg",
    }
    monkeypatch.setattr(r.h59, "sha256", lambda path: hashes[Path(path).name])
    monkeypatch.setattr(
        r.h55, "input_metadata", lambda *_: ({}, [row], [{"context_sha256": dev}])
    )
    monkeypatch.setattr(r.h55, "family_folds", lambda *_: [{"f"}])
    sampler = SimpleNamespace(
        cached_training_indices=lambda pool, order, count: np.arange(count)
    )
    monkeypatch.setattr(
        r,
        "load",
        lambda name, path: sampler if "sampler" in name else SimpleNamespace(),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run built a model or read dev payloads")

    monkeypatch.setattr(r.h55, "build_model", forbidden)
    original_load = np.load
    opened = []

    def train_only(path, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name == f"{train}.npz"
        return original_load(path, **kwargs)

    monkeypatch.setattr(np, "load", train_only)
    result = r.dry_run(tmp_path, tmp_path / "checkpoint.pt", tmp_path / "dry-run.json")
    assert opened == [f"{train}.npz"]
    assert result["dev_payloads_opened"] == result["optimizer_updates"] == 0
    assert result["GPU_started"] is False and len(result["plans"]) == 6
