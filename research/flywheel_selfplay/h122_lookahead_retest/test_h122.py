"""Synthetic CPU controls for H122 split, scratch, completeness and admission gates."""

import importlib.util
import json
from pathlib import Path
import stat
import sys

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
    return {
        "wave": "H122",
        "arm": arm,
        "seed": seed,
        "fold_seed": r.FOLD_SEED,
        "protocol_sha256": r.protocol_sha256(),
        "updates": r.UPDATES,
        "checkpoint_sha256": r.R882_SHA,
        "dope_before": r.DOPE_SHA,
        "dope_after": r.DOPE_SHA,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "source_commit": "synthetic-source",
        "split_sha256": split["sha256"],
        "initial_adapter_sha256": "synthetic-initial-adapter",
        "assessment": {"per_family": assessment},
        "r882_assessment": {"per_family": reference},
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
