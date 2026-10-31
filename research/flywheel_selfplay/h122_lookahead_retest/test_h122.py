"""Synthetic CPU controls for H122 split, scratch, completeness and admission gates."""

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h122_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def _rows():
    return [
        {
            "context_sha256": f"{i:064x}",
            "family_id": f"family-{i:03d}",
            "task": "regression",
            "dl_split": "DL-Train",
            "source_split": "train",
            "origin": "real",
        }
        for i in range(205)
    ]


def test_fresh_family_assignment_is_deterministic_and_disjoint():
    first = r.family_assignment(_rows())
    second = r.family_assignment(_rows())
    assert first == second
    assert first["seed"] == 122042
    assert [len(fold) for fold in first["folds"]] == [41] * 5
    assert len(first["train_families"]) == 164
    assert len(first["assessment_families"]) == 41
    assert not set(first["train_families"]) & set(first["assessment_families"])


def test_family_assignment_rejects_protected_or_duplicate_metadata():
    protected = _rows()
    protected[0]["has_meta_test"] = True
    with pytest.raises(ValueError, match="real regression DL-Train"):
        r.family_assignment(protected)
    duplicate = _rows()
    duplicate[1]["context_sha256"] = duplicate[0]["context_sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        r.family_assignment(duplicate)


def test_attempt_scratch_is_unique_and_refuses_reuse(tmp_path):
    control = r.attempt_scratch(tmp_path, "CONTROL", 48, 1)
    lookahead = r.attempt_scratch(tmp_path, "LA", 48, 1)
    retry = r.attempt_scratch(tmp_path, "LA", 48, 2)
    other_seed = r.attempt_scratch(tmp_path, "LA", 49, 1)
    assert len({control, lookahead, retry, other_seed}) == 4
    assert lookahead == tmp_path / "scratch/LA-seed48-attempt1"
    lookahead.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        r.attempt_scratch(tmp_path, "LA", 48, 1)
    for args in (("GN", 48, 1), ("LA", 30, 1), ("LA", 48, True)):
        with pytest.raises(ValueError):
            r.attempt_scratch(tmp_path, *args)


def test_preflight_failure_happens_before_checkpoint_copy_or_scratch_creation(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "round-882.pt"
    checkpoint.write_bytes(b"synthetic only")
    monkeypatch.setattr(r, "R882_COPY", checkpoint)
    monkeypatch.setattr(r.core, "assert_lease", lambda _fd: None)

    def blocked():
        raise RuntimeError("launch disk gate")

    called = []
    monkeypatch.setattr(r.h61wave, "safe_preflight", blocked)
    monkeypatch.setattr(r.h59, "stage_r882_copy", lambda *_args: called.append(True))
    with pytest.raises(RuntimeError, match="launch disk gate"):
        r.stage_r882_for_attempt(tmp_path / "run", "LA", 48, 1, 9)
    assert called == []
    assert not (tmp_path / "run/scratch").exists()
    assert checkpoint.read_bytes() == b"synthetic only"


def test_readonly_legacy_copy_does_not_block_fresh_attempt_stage(tmp_path, monkeypatch):
    root = tmp_path / "run"
    legacy = root / "scratch/r882_adapter_copy.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"preserve old staged copy")
    legacy.chmod(0o444)
    previous_mode = stat.S_IMODE(legacy.stat().st_mode)

    checkpoint = tmp_path / "round-882.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for unit test")
    monkeypatch.setattr(r.h59, "R882_SHA", r.sha256(checkpoint))

    scratch = r.attempt_scratch(root, "LA", 48, 2)
    staged = r.h59.stage_r882_copy(checkpoint, scratch)
    assert staged == scratch / "r882_adapter_copy.pt"
    assert staged.read_bytes() == checkpoint.read_bytes()
    assert legacy.read_bytes() == b"preserve old staged copy"
    assert stat.S_IMODE(legacy.stat().st_mode) == previous_mode == 0o444


def _synthetic_cell(arm, seed, la_utility):
    split = r.family_assignment(_rows())
    families = split["assessment_families"]
    reference = {family: 0.5 for family in families}
    control = {family: 0.55 for family in families}
    assessment = (
        control if arm == "CONTROL" else {family: la_utility for family in families}
    )

    def report(values):
        mapping = {f"{i:064x}": family for i, family in enumerate(families)}
        return {
            "eval_mode": r.h61.EVAL_MODE_UNREV,
            "per_family": values,
            "per_context": {cid: values[family] for cid, family in mapping.items()},
            "family_by_context": mapping,
            "chosen_indices": {cid: 4 for cid in mapping},
        }

    return {
        "wave": "H122",
        "arm": arm,
        "seed": seed,
        "fold_seed": r.FOLD_SEED,
        "protocol_sha256": r.protocol_sha256(),
        "updates": r.UPDATES,
        "timing_only": False,
        "attempt": 1,
        "attempt_token": f"fixture-{arm}-{seed}",
        "training": {
            "optimizer_updates": 40,
            "lookahead_sync_steps": list(range(5, 41, 5)) if arm == "LA" else [],
        },
        "trainable_count": 29211,
        "trainable_names": ["adapter_10tok.fixture"],
        "prereg_sha256": r.sha256(HERE / "PREREG.md"),
        "manifest_sha256": r.MANIFEST_SHA,
        "selected_sha256": r.SELECTED_SHA,
        "fused_sha256": r.FUSED_SHA,
        "source_hashes": dict(r.SOURCE_PINS),
        "checkpoint_sha256": r.R882_SHA,
        "dope_before": r.DOPE_SHA,
        "dope_after": r.DOPE_SHA,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_commit": "1" * 40,
        "split": split,
        "split_sha256": split["sha256"],
        "initial_adapter_sha256": "2" * 64,
        "final_adapter_sha256": "3" * 64,
        "adapter_l2": 0.1,
        "assessment": report(assessment),
        "r882_assessment": report(reference),
    }


def test_summary_requires_complete_panel_and_uses_strict_primary_gates():
    with pytest.raises(RuntimeError, match="six unique"):
        r.summarize([])
    cells = [_synthetic_cell(arm, seed, 0.502) for seed in r.SEEDS for arm in r.ARMS]
    result = r.summarize(cells)
    assert result["complete"] is True
    assert result["primary_vs_r882"]["admitted"] is True
    assert result["primary_vs_r882"]["median_seed_gain"] > 0
    assert result["secondary_vs_control"]["admission_gate"] is False

    tied = [_synthetic_cell(arm, seed, 0.501) for seed in r.SEEDS for arm in r.ARMS]
    assert r.summarize(tied)["primary_vs_r882"]["admitted"] is False


def test_cpu_dry_run_is_data_free_and_reports_no_execution():
    result = r.cpu_dry_run()
    assert result["status"] == "CPU_DRY_RUN_PASS"
    assert result["data_reads"] == result["checkpoint_reads"] == 0
    assert result["optimizer_updates"] == result["outcome_cells_run"] == 0
    assert not result["GPU_started"] and not result["FIFO_ticket_created"]
    assert set(json.loads(json.dumps(result))["synthetic_updates_per_arm"]) == set(
        r.ARMS
    )


@pytest.mark.parametrize("arm", r.ARMS)
def test_worker_update_path_executes_40_cpu_updates_and_freezes_other_parameters(arm):
    torch = r.torch
    torch.set_num_threads(1)
    net = torch.nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.fill_(0.1)
        net.bias.fill_(0.2)
    net.bias.requires_grad_(False)
    initial, frozen = net.weight.detach().clone(), net.bias.detach().clone()
    losses = []
    guards = []

    def loss_for(_batch):
        loss = net(torch.tensor([[1.0, 2.0]])).square().mean()
        losses.append(float(loss.detach()))
        return loss

    def guard(optimizer):
        assert optimizer.param_groups[0]["params"] == [net.weight]
        guards.append(True)

    result = r.execute_updates(
        net, [net.weight], [(None,)] * 40, arm, loss_for, optimizer_guard=guard
    )
    assert result["optimizer_updates"] == len(losses) == 40
    assert guards == [True] and losses[-1] < losses[0]
    assert not torch.equal(initial, net.weight) and torch.equal(frozen, net.bias)
    assert net.bias.grad is None
    assert result["lookahead_sync_steps"] == (
        list(range(5, 41, 5)) if arm == "LA" else []
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "prereg_sha256",
        "manifest_sha256",
        "selected_sha256",
        "fused_sha256",
        "source_hashes",
        "trainable_names",
        "training",
        "split",
        "attempt_token",
        "final_adapter_sha256",
        "adapter_l2",
    ],
)
def test_summary_refuses_missing_worker_evidence(field):
    cells = [_synthetic_cell(arm, seed, 0.6) for seed in r.SEEDS for arm in r.ARMS]
    cells[-1].pop(field)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        r.summarize(cells)


def test_official_unrevealed_argmin_uses_equal_family_weight_not_context_weight():
    np = r.np
    pools = {
        str(i): {
            "rec": {"family_id": family},
            "fitness": np.array([1.0, 1.0, 1.0, 1.0, -utility, -utility]),
            "gate": np.ones(6, dtype=bool),
        }
        for i, (family, utility) in enumerate([("a", 0.2), ("a", 0.2), ("b", 0.8)])
    }
    predictions = {cid: np.array([-100.0, 0.0, 0.0, 0.0, 1.0, 0.0]) for cid in pools}
    result = r.h61.evaluate_pool_predictions(
        pools, pools, predictions, eval_mode=r.h61.EVAL_MODE_UNREV
    )
    assert set(result["chosen_indices"].values()) == {5}
    assert result["mean"] == pytest.approx(0.5)
    assert result["context_mean"] == pytest.approx(0.4)


def test_summary_refuses_forged_family_mean_and_cross_seed_roster():
    cells = [_synthetic_cell(arm, seed, 0.6) for seed in r.SEEDS for arm in r.ARMS]
    family = next(iter(cells[-1]["assessment"]["per_family"]))
    cells[-1]["assessment"]["per_family"][family] = 0.9
    with pytest.raises(RuntimeError, match="does not match contexts"):
        r.summarize(cells)
    cells[-1] = _synthetic_cell("LA", 50, 0.6)
    cells[-1]["assessment"]["chosen_indices"]["0" * 64] = 0
    with pytest.raises(RuntimeError, match="official assessment"):
        r.summarize(cells)


@pytest.fixture
def wave(monkeypatch):
    w = r.load("h122_test_coordinator", HERE / "run_wave.py")
    monkeypatch.setattr(w.r.h59, "check_writer_disk_budget", lambda: None)
    monkeypatch.setattr(w.fair, "safe_preflight", lambda: {"fixture": True})
    monkeypatch.setattr(w, "cpu_gate", lambda _deadline: 0.0)
    monkeypatch.setattr(w.os, "setpriority", lambda *_args: None)
    return w


def _identity(w):
    return {
        "source_commit": "1" * 40,
        "protocol_sha256": r.protocol_sha256(),
        "prereg_sha256": r.sha256(HERE / "PREREG.md"),
        "runner_sha256": r.sha256(HERE / "runner.py"),
        "coordinator_sha256": r.sha256(HERE / "run_wave.py"),
    }


def _closeouts(w, tmp_path):
    paths = {}
    for predecessor in w.PREDECESSORS:
        path = tmp_path / f"{predecessor}.json"
        path.write_text(
            json.dumps(
                {
                    "wave": predecessor,
                    "closed": True,
                    "qualification_lock_released": True,
                    "in_progress": None,
                }
            )
        )
        paths[predecessor] = path
    return paths


def _fake_executor(w, tmp_path, monkeypatch):
    calls = []

    def acquire(*_args, **_kwargs):
        return (
            (tmp_path / "private-fixture-lease").open("a+"),
            0.0,
            0,
            {"fixture": True},
        )

    def child(command, _log, _timeout, _lease, _record):
        args = dict(zip(command[2::2], command[3::2], strict=True))
        cell = _synthetic_cell(args["--arm"], int(args["--seed"]), 0.6)
        cell["attempt_token"] = args["--attempt-token"]
        Path(args["--out"]).write_text(json.dumps(cell))
        phase = json.loads(Path(args["--phase"]).read_text())
        phase.update(stage="complete", outcome_started=True)
        Path(args["--phase"]).write_text(json.dumps(phase))
        calls.append((cell["arm"], cell["seed"]))

    monkeypatch.setattr(w.fair, "acquire_lock", acquire)
    monkeypatch.setattr(w, "run_child", child)
    return calls


def test_coordinator_executes_six_reservations_collects_and_refuses_reset(
    wave, tmp_path, monkeypatch
):
    calls = _fake_executor(wave, tmp_path, monkeypatch)
    root = tmp_path / "run"
    predecessors, identity = _closeouts(wave, tmp_path), _identity(wave)
    metrics = wave.execute(root, predecessors, identity)
    assert calls == list(wave.CELLS) and metrics["complete"]
    assert wave.collect(root, identity) == metrics
    budget = json.loads((root / "budget.json").read_text())
    assert all(x["wall_seconds"] > 0 for x in budget["attempts"])
    assert budget["in_progress"] is None
    with pytest.raises(FileExistsError):
        wave.execute(root, predecessors, identity)
    assert len(calls) == 6


def test_coordinator_blocks_predecessors_before_root_or_fifo(
    wave, tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("shared FIFO called before predecessor admission")

    monkeypatch.setattr(wave.fair, "acquire_lock", forbidden)
    with pytest.raises(RuntimeError, match="all five"):
        wave.execute(tmp_path / "run", {}, _identity(wave))
    assert not (tmp_path / "run").exists()
    paths = _closeouts(wave, tmp_path)
    paths["H121"].write_text(json.dumps({"wave": "H121", "closed": False}))
    with pytest.raises(RuntimeError, match="not closed"):
        wave.execute(tmp_path / "run", paths, _identity(wave))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "fault", ["unknown", "cap", "wait_nan", "failed", "phase", "receipt", "missing"]
)
def test_collector_refuses_unknown_over_budget_failed_or_tampered_panel(
    wave, tmp_path, monkeypatch, fault
):
    _fake_executor(wave, tmp_path, monkeypatch)
    root, identity = tmp_path / "run", _identity(wave)
    wave.execute(root, _closeouts(wave, tmp_path), identity)
    ledger_path = root / "budget.json"
    ledger = json.loads(ledger_path.read_text())
    if fault == "unknown":
        ledger["in_progress"] = {"arm": "LA"}
    elif fault == "cap":
        ledger["attempts"][0]["wall_seconds"] = 2701.0
    elif fault == "wait_nan":
        ledger["resource_wait_seconds"] = float("nan")
    elif fault == "failed":
        ledger["attempts"][0]["error"] = "worker failed"
    elif fault == "missing":
        ledger["attempts"].pop()
    else:
        suffix = ".phase.json" if fault == "phase" else ".json"
        with (root / f"CONTROL-seed48-attempt1{suffix}").open("a") as stream:
            stream.write(" ")
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(RuntimeError):
        wave.collect(root, identity)


def test_failed_worker_retains_charge_and_never_runs_another_cell(
    wave, tmp_path, monkeypatch
):
    calls = _fake_executor(wave, tmp_path, monkeypatch)

    def fail(*_args):
        raise RuntimeError("fixture failure before scoring")

    monkeypatch.setattr(wave, "run_child", fail)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="fixture failure"):
        wave.execute(root, _closeouts(wave, tmp_path), _identity(wave))
    ledger = json.loads((root / "budget.json").read_text())
    assert len(ledger["attempts"]) == 1 and ledger["attempts"][0]["wall_seconds"] > 0
    assert ledger["attempts"][0]["error"] and not calls
    assert (root / "incomplete.json").exists() and not (root / "metrics.json").exists()


def test_worker_cli_is_reachable_but_refuses_absent_lease_before_payloads(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "runner.py"),
            "--arm",
            "LA",
            "--seed",
            "48",
            "--out",
            str(tmp_path / "cell.json"),
            "--phase",
            str(tmp_path / "cell.phase.json"),
            "--attempt",
            "1",
            "--attempt-token",
            "fixture",
            "--lock-fd",
            "99999",
            "--parent-pid",
            str(__import__("os").getpid()),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "Bad file descriptor" in result.stderr
    assert (
        "unrecognized arguments" not in result.stderr
        and not (tmp_path / "cell.json").exists()
    )


def test_phase_rejects_other_cell_and_reversed_boundary(tmp_path, monkeypatch):
    phase = tmp_path / "cell.phase.json"
    args = SimpleNamespace(
        phase=phase, arm="LA", seed=48, attempt=1, attempt_token="fixture"
    )
    monkeypatch.setattr(r.h59, "check_writer_disk_budget", lambda: None)
    phase.write_text(
        json.dumps(
            {
                "arm": "CONTROL",
                "seed": 48,
                "attempt": 1,
                "attempt_token": "fixture",
                "stage": "launch",
            }
        )
    )
    with pytest.raises(RuntimeError, match="boundary"):
        r.mark_phase(args, "loading")
