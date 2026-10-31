"""CPU-only contract tests for H99 preparation; no packet or checkpoint reads."""
from __future__ import annotations

import sys
from pathlib import Path
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402
import coordinator  # noqa: E402
import build_metrics  # noqa: E402


def test_fixed_paired_cells_and_h80_budget():
    plan = runner.execution_plan()
    assert plan["arms"] == ["NULL", "ESH-l2", "ESH-l4"]
    assert plan["lambda_by_arm"] == {"ESH-l2": 2.0, "ESH-l4": 4.0}
    assert plan["paired_seeds"] == [36, 37, 38]
    assert len(plan["candidate_cells"]) == 6
    assert {key: plan["null_cell"][key] for key in ("arm", "seed", "generations")} == {
        "arm": "NULL",
        "seed": 36,
        "generations": 16,
    }
    assert all(cell["generations"] == 16 for cell in plan["candidate_cells"])
    assert all(cell["scratch_pattern"].endswith("-attemptN/") for cell in plan["candidate_cells"])
    assert plan["null_cell"]["scratch_pattern"] == "scratch/NULL-seed36-attemptN/"
    assert plan["fold_seed"] != 4900
    assert plan["outcome_reads"] == plan["checkpoint_reads"] == 0
    assert plan["gpu_used"] is False and plan["gpu_ticket_created"] is False
    assert plan["timing_cells_run"] == plan["optimizer_updates"] == 0


def test_seed_stream_is_fixed_and_rejects_unregistered_cells():
    assert runner.seed_stream(36, 1) == 36 * 1_000_003 + 97_409 + 1_009
    assert runner.seed_stream(36, 1) == runner.seed_stream(36, 1)
    for seed, generation in ((35, 1), (39, 1), (36, 0), (36, 17)):
        try:
            runner.seed_stream(seed, generation)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unregistered stream seed={seed}, generation={generation}")


def test_cpu_dry_run_reuses_h80_math_without_data_or_training():
    result = runner.cpu_dry_run()
    assert result["status"] == "cpu_dry_run_pass"
    assert result["shared_h80_math"]["perturbations"] == 32
    assert result["shared_h80_math"]["antithetic_pairs"] == 16
    assert result["shared_h80_math"]["gradient_finite"] is True
    assert result["shared_h80_math"]["synthetic_lambda4_fitness"] < result["shared_h80_math"]["synthetic_lambda2_fitness"]
    assert result["outcome_reads"] == result["checkpoint_reads"] == 0
    assert result["optimizer_updates"] == result["timing_cells_run"] == 0
    assert not result["gpu_used"] and not result["gpu_ticket_created"]


def test_fresh_fold_is_family_grouped_and_train_only():
    class FixtureH48:
        @staticmethod
        def family_folds(rows, *, count, seed):
            assert count == 5 and seed == 9900
            families = sorted({row["family_id"] for row in rows})
            return [set(families[i::count]) for i in range(count)]

    rows = [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"f{i}"}
        for i in range(10)
        for _context in range(2)
    ]
    assignment = runner.fresh_family_assignment(rows, h48=FixtureH48)
    assert assignment["seed"] == 9900
    assert len(assignment["fold_a_families"]) == 8
    assert len(assignment["fold_b_families"]) == 2
    assert not set(assignment["fold_a_families"]) & set(assignment["fold_b_families"])
    assert len(assignment["fold_assignment_sha256"]) == 64


def test_fresh_fold_rejects_protected_and_non_train_rows():
    for row in (
        {"dl_split": "DL-Val", "task": "regression", "family_id": "f0"},
        {"dl_split": "DL-Train", "task": "regression", "family_id": "f0", "has_meta_test": True},
        {"dl_split": "DL-Train", "task": "classification", "family_id": "f0"},
    ):
        try:
            runner.fresh_family_assignment([row], h48=object())
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted forbidden H99 fold row: {row}")


def test_launch_resource_limits_are_strict_at_boundaries():
    coordinator.check_resource_snapshot(
        load1=85.0,
        mem_available_kib=16 * 1024 * 1024,
        disk_used_percent=93.999,
    )
    for snapshot in (
        {"load1": 85.01, "mem_available_kib": 16 * 1024 * 1024, "disk_used_percent": 90.0},
        {"load1": 1.0, "mem_available_kib": 16 * 1024 * 1024 - 1, "disk_used_percent": 90.0},
        {"load1": 1.0, "mem_available_kib": 16 * 1024 * 1024, "disk_used_percent": 94.0},
    ):
        try:
            coordinator.check_resource_snapshot(**snapshot)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"accepted unsafe launch snapshot: {snapshot}")


def test_wall_deadline_interrupts_overlong_cell_action():
    try:
        with coordinator._wall_deadline(0.02):
            time.sleep(0.2)
    except TimeoutError as exc:
        assert "9,000-second" in str(exc)
    else:
        raise AssertionError("H99 wall deadline failed to interrupt an overlong action")


def test_h99_acquire_adapter_uses_shared_fifo_and_h99_wave_name(tmp_path):
    calls = []
    markers = []

    class Handle:
        def close(self):
            self.closed = True

    class FakeWave:
        @staticmethod
        def acquire_lock(status_path, wait_total, **kwargs):
            marker = next((tmp_path / "queue").glob("H99-owner-*.json"))
            markers.append(json.loads(marker.read_text()))
            calls.append((status_path, wait_total, kwargs))
            return Handle(), 3.0, 8192, {"load1": 4.0}

    queue_dir = tmp_path / "queue"
    result = coordinator.acquire_h99_fifo(
        tmp_path / "H99_STATUS.log", wait_total=2.0, wave=FakeWave,
        owner="luna_v1_res03", eta_utc="2026-10-31T16:00:00Z", queue_dir=queue_dir,
    )
    assert isinstance(result[0], coordinator.H99Lease)
    status_path, wait_total, kwargs = calls[0]
    assert status_path == tmp_path / "H99_STATUS.log"
    assert wait_total == 2.0
    assert kwargs["wave_name"] == "H99"
    assert kwargs["after_release"] is True
    assert callable(kwargs["preflight_fn"])
    assert markers[0]["owner"] == "luna_v1_res03"
    assert markers[0]["eta_utc"] == "2026-10-31T16:00:00Z"
    assert list(queue_dir.glob("H99-owner-*.json"))
    coordinator.release_lock(result[0], flock_fn=lambda *_args: None)
    assert list(queue_dir.glob("H99-owner-*.json")) == []


def test_cpu_gate_requires_two_consecutive_safe_load_samples():
    samples = iter((90.0, 70.0, 80.0))
    sleeps = []
    waited = coordinator.wait_for_cpu_gate(
        load_fn=lambda: (next(samples), 0.0, 0.0),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )
    assert sleeps == [30.0, 30.0]
    assert waited >= 0.0


def test_read_only_r882_source_stages_to_separate_attempts_without_mutation(tmp_path, monkeypatch):
    source = tmp_path / "r882-readonly.pt"
    source.write_bytes(b"synthetic r882 source fixture")
    source.chmod(0o444)
    original_mode = source.stat().st_mode & 0o777
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(coordinator, "load_h80_wave", lambda: type("H80", (), {"runner_mod": type("R", (), {"R882_SHA": digest, "sha256": staticmethod(lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())})})())

    first_dir = coordinator.attempt_scratch_path(tmp_path / "wave", "ESH-l2", 36, 1)
    second_dir = coordinator.attempt_scratch_path(tmp_path / "wave", "ESH-l2", 36, 2)
    first = coordinator.stage_r882_copy(source, first_dir)
    second = coordinator.stage_r882_copy(source, second_dir)
    assert first != second
    assert first.read_bytes() == second.read_bytes() == source.read_bytes()
    assert source.stat().st_mode & 0o777 == original_mode == 0o444
    assert first.stat().st_mode & 0o222
    assert second.stat().st_mode & 0o222


def test_fifo_coordinator_charges_all_seven_cells_and_releases_each_lease(tmp_path, monkeypatch):
    checkpoint = tmp_path / "r882.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    monkeypatch.setattr(coordinator, "stage_r882_copy", lambda _source, scratch: (scratch.mkdir(parents=True), scratch / "copy.pt")[1])
    leases = []
    released = []

    def acquire(status_path, *, wait_total):
        handle = object()
        leases.append((status_path, wait_total, handle))
        return handle, 0.0, 8192, {"load1": 1.0}

    cells = [*runner.execution_plan()["candidate_cells"], runner.execution_plan()["null_cell"]]
    result = coordinator.run_fifo_cells(
        cells,
        run_root=tmp_path / "run",
        checkpoint=checkpoint,
        worker=lambda cell, _staged, _scratch, remaining, _attempt: {"arm": cell["arm"], "remaining": remaining},
        acquire_fn=acquire,
        release_fn=lambda handle: released.append(handle),
        cpu_gate_fn=lambda: 0.0,
        setup_fn=lambda: None,
        clock_fn=iter(range(14)).__next__,
    )
    assert len(result["attempts"]) == 7
    assert len(leases) == len(released) == 7
    assert [Path(item["scratch"]).name for item in result["attempts"]] == [
        f"{cell['arm']}-seed{cell['seed']}-attempt1" for cell in cells
    ]
    assert sum(item["wall_seconds"] for item in result["attempts"]) == 7.0


def test_cpu_fake_wave_end_to_end_produces_preregistered_metrics_json(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source-r882.pt"
    checkpoint.write_bytes(b"fake pinned checkpoint")
    def stage(_source, scratch):
        scratch.mkdir(parents=True)
        staged = scratch / "r882-copy.pt"
        staged.write_bytes(b"fake staged checkpoint")
        return staged
    monkeypatch.setattr(coordinator, "stage_r882_copy", stage)
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    leases = []
    released = []
    clock_values = iter(range(20))

    def acquire(status_path, *, wait_total):
        handle = object()
        leases.append((status_path, wait_total, handle))
        return handle, 0.0, 8192, {"cpu": "fake-safe"}

    def fake_worker(cell, _staged, scratch, _remaining, attempt):
        out = tmp_path / "run" / "outcomes" / "cells" / cell["arm"] / f"seed{cell['seed']}" / f"attempt{attempt}"
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({"arm": cell["arm"], "seed": cell["seed"], "status": "complete"}))
        return {"scratch": str(scratch), "fake": True}

    def fake_finalize(_staged, _scratch, _remaining, _attempt):
        candidates = []
        for cell in cells:
            if cell["arm"] == "NULL":
                continue
            gains = {
                f"family-{family:02d}": 0.02 - 0.0005 * family + (0.0001 if cell["seed"] == 37 else 0.0)
                for family in range(20)
            }
            candidates.append({"arm": cell["arm"], "seed": cell["seed"], "family_gain_vs_r882": gains, "fold_b_scoring_passes": 1})
        fold_b = {
            "schema": "jope.h99.fold-b/v1", "status": "complete", "fold_seed": 9900,
            "fold_assignment_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "fold_b_family_count": 20, "candidate_count": 6, "fold_b_read_count": 6,
            "null_cell_complete": True, "null_fold_b_scored": False,
            "candidates": candidates,
        }
        out = tmp_path / "run" / "outcomes" / "fold_b.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fold_b))
        return {"fake": True}

    budget = coordinator.run_fifo_cells(
        cells, run_root=tmp_path / "run", checkpoint=checkpoint, worker=fake_worker,
        finalize_worker=fake_finalize, acquire_fn=acquire,
        release_fn=lambda handle: released.append(handle), cpu_gate_fn=lambda: None,
        setup_fn=lambda: None, clock_fn=clock_values.__next__,
    )
    assert len(budget["attempts"]) == 8  # seven cells plus one metered Fold-B scoring action
    assert len(leases) == len(released) == 8
    assert len({item["scratch"] for item in budget["attempts"]}) == 8
    fold_b = json.loads((tmp_path / "run/outcomes/fold_b.json").read_text())
    metrics = build_metrics.build_metrics(fold_b)
    output = tmp_path / "run/outcomes/admission.json"
    output.write_text(json.dumps(metrics, sort_keys=True, allow_nan=False))
    decoded = json.loads(output.read_text())
    assert decoded["schema"] == "jope.h99.admission-metrics/v1"
    assert decoded["bootstrap_draws"] == 10_000
    assert decoded["arms"]["ESH-l2"]["mean_lcb95_one_sided"] > 0.001
    assert decoded["arms"]["ESH-l2"]["elo_vs_r882_median_across_seeds"] > 1500.0
    assert decoded["null_diagnostic"]["admitted"] is False


def test_metrics_reject_missing_duplicate_and_inexact_six_candidate_panels():
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    rows = [
        {"arm": cell["arm"], "seed": cell["seed"], "family_gain_vs_r882": {"f0": 0.01, "f1": 0.02}, "fold_b_scoring_passes": 1}
        for cell in cells
        if cell["arm"] != "NULL"
    ]
    base = {
        "schema": "jope.h99.fold-b/v1", "status": "complete", "fold_seed": 9900,
        "fold_assignment_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
        "fold_b_family_count": 2, "candidate_count": 6, "fold_b_read_count": 6,
        "null_cell_complete": True, "null_fold_b_scored": False, "candidates": rows,
    }
    for invalid in (dict(base, candidates=rows[:-1]), dict(base, candidates=rows + [rows[0]])):
        try:
            build_metrics.build_metrics(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted incomplete or duplicate H99 Fold-B panel")


def test_exact_fold_b_ties_never_admit():
    plan = runner.execution_plan()
    rows = [
        {"arm": cell["arm"], "seed": cell["seed"], "family_gain_vs_r882": {"f0": 0.0, "f1": 0.0}, "fold_b_scoring_passes": 1}
        for cell in plan["candidate_cells"]
    ]
    fold_b = {
        "schema": "jope.h99.fold-b/v1", "status": "complete", "fold_seed": 9900,
        "fold_assignment_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
        "fold_b_family_count": 2, "candidate_count": 6, "fold_b_read_count": 6,
        "null_cell_complete": True, "null_fold_b_scored": False, "candidates": rows,
    }
    metrics = build_metrics.build_metrics(fold_b)
    assert metrics["arms"]["ESH-l2"]["is_exact_tie"] is True
    assert metrics["arms"]["ESH-l2"]["admitted"] is False


def test_resume_retries_only_void_cell_as_next_attempt_and_preserves_budget(tmp_path, monkeypatch):
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    target = next(cell for cell in cells if cell["arm"] == "ESH-l4" and cell["seed"] == 38)
    run_root = tmp_path / "run"
    run_root.mkdir()
    attempts = []
    for cell in cells:
        status = "void-or-unknown" if cell == target else "complete"
        attempts.append({
            **cell, "attempt": 1,
            "scratch": str(coordinator.attempt_scratch_path(run_root, cell["arm"], cell["seed"], 1)),
            "wall_seconds": 2.0, "wait_seconds": 3.0, "status": status,
        })
    budget_path = run_root / "budget.json"
    coordinator.write_budget(budget_path, {
        "schema": "jope.h99.gpu-budget/v1", "cap_seconds": coordinator.CAP_SECONDS,
        "attempts": attempts, "in_progress": None,
    })
    monkeypatch.setattr(coordinator, "stage_r882_copy", lambda _source, scratch: (scratch.mkdir(parents=True), scratch / "copy.pt")[1])
    calls = []
    leases = []
    released = []
    clock = iter((10.0, 17.0))

    def acquire(status_path, *, wait_total):
        handle = object()
        leases.append((status_path, wait_total, handle))
        return handle, 4.0, 8192, {"cpu": "synthetic-safe"}

    result = coordinator.run_fifo_cells(
        cells, run_root=run_root, checkpoint=tmp_path / "checkpoint",
        worker=lambda cell, _staged, scratch, remaining, attempt: calls.append((cell, scratch, remaining, attempt)) or {"ok": True},
        acquire_fn=acquire, release_fn=lambda handle: released.append(handle),
        cpu_gate_fn=lambda: None, setup_fn=lambda: None, clock_fn=clock.__next__,
    )
    assert len(calls) == len(leases) == len(released) == 1
    cell, scratch, remaining, attempt = calls[0]
    assert cell == target
    assert attempt == 2
    assert scratch == coordinator.attempt_scratch_path(run_root, "ESH-l4", 38, 2)
    assert remaining == coordinator.CAP_SECONDS - sum(x["wall_seconds"] for x in attempts)
    assert leases[0][1] == sum(x["wait_seconds"] for x in attempts)
    assert result["attempts"][:-1] == attempts
    assert result["attempts"][-1]["status"] == "complete"
    assert result["attempts"][-1]["attempt"] == 2
    assert coordinator.completed_attempt(run_root, "ESH-l4", 38) == 2


def test_resume_uses_actual_ledger_for_void_and_unstarted_fixed_cells(tmp_path, monkeypatch):
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    run_root = tmp_path / "run"
    run_root.mkdir()
    failed = next(cell for cell in cells if cell["arm"] == "ESH-l4" and cell["seed"] == 38)
    attempts = []
    for cell in cells[:-1]:
        attempts.append({
            **cell, "attempt": 1,
            "scratch": str(coordinator.attempt_scratch_path(run_root, cell["arm"], cell["seed"], 1)),
            "wall_seconds": 2.0, "wait_seconds": 0.0,
            "status": "void-or-unknown" if cell == failed else "complete",
        })
    coordinator.write_budget(run_root / "budget.json", {
        "schema": "jope.h99.gpu-budget/v1", "cap_seconds": coordinator.CAP_SECONDS,
        "attempts": attempts, "in_progress": None,
    })
    monkeypatch.setattr(coordinator, "stage_r882_copy", lambda _source, scratch: (scratch.mkdir(parents=True), scratch / "copy.pt")[1])
    calls = []
    clock = iter((0.0, 1.0, 2.0, 3.0))
    coordinator.run_fifo_cells(
        cells, run_root=run_root, checkpoint=tmp_path / "checkpoint",
        worker=lambda cell, _staged, scratch, _remaining, attempt: calls.append((cell, scratch, attempt)) or {},
        acquire_fn=lambda _status, *, wait_total: (object(), 0.0, 8192, {}),
        release_fn=lambda _handle: None, cpu_gate_fn=lambda: None,
        setup_fn=lambda: None, clock_fn=clock.__next__,
    )
    assert [(cell["arm"], cell["seed"], attempt) for cell, _scratch, attempt in calls] == [
        ("ESH-l4", 38, 2), ("NULL", 36, 1),
    ]
    assert calls[0][1] == coordinator.attempt_scratch_path(run_root, "ESH-l4", 38, 2)
    assert calls[1][1] == coordinator.attempt_scratch_path(run_root, "NULL", 36, 1)


def test_resume_rejects_retries_after_a_completed_cell(tmp_path):
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    cell = cells[0]
    attempts = []
    for attempt, status in ((1, "complete"), (2, "void-or-unknown")):
        attempts.append({
            **cell, "attempt": attempt,
            "scratch": str(coordinator.attempt_scratch_path(tmp_path, cell["arm"], cell["seed"], attempt)),
            "wall_seconds": 1.0, "status": status,
        })
    try:
        coordinator._cell_histories({
            "schema": "jope.h99.gpu-budget/v1", "cap_seconds": coordinator.CAP_SECONDS,
            "attempts": attempts, "in_progress": None,
        }, cells)
    except RuntimeError as exc:
        assert "retried after completion" in str(exc)
    else:
        raise AssertionError("accepted a retry after a completed H99 cell")


def test_resume_never_repeats_void_or_unknown_fold_b_scoring(tmp_path):
    plan = runner.execution_plan()
    cells = [*plan["candidate_cells"], plan["null_cell"]]
    attempts = [
        {
            **cell, "attempt": 1,
            "scratch": str(coordinator.attempt_scratch_path(tmp_path, cell["arm"], cell["seed"], 1)),
            "wall_seconds": 1.0, "status": "complete",
        }
        for cell in cells
    ]
    attempts.append({
        "action": "fold_b_scoring", "attempt": 1, "scratch": str(tmp_path / "scratch/fold-b-scoring-attempt1"),
        "wall_seconds": 1.0, "wait_seconds": 0.0, "status": "void-or-unknown",
    })
    run_root = tmp_path / "run"
    coordinator.write_budget(run_root / "budget.json", {
        "schema": "jope.h99.gpu-budget/v1", "cap_seconds": coordinator.CAP_SECONDS,
        "attempts": attempts, "in_progress": None,
    })
    calls = []
    try:
        coordinator.run_fifo_cells(
            cells, run_root=run_root, checkpoint=tmp_path / "checkpoint",
            worker=lambda *_args: calls.append("cell"),
            finalize_worker=lambda *_args: calls.append("protected fold"),
            acquire_fn=lambda *_args, **_kwargs: calls.append("FIFO acquire"),
            setup_fn=lambda: None,
        )
    except RuntimeError as exc:
        assert "refusing a duplicate protected-fold read" in str(exc)
    else:
        raise AssertionError("retried a void-or-unknown H99 Fold-B scoring action")
    assert calls == []
