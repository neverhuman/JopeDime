"""Synthetic CPU-only controls for the H109 preparation contract."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h109_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def synthetic_rows(n=25):
    return [
        {"dl_split": "DL-Train", "task": "regression", "family_id": f"f{i}"}
        for i in range(n)
    ]


def test_fold_is_deterministic_exact_and_family_disjoint():
    first = r.fold_receipt(synthetic_rows())
    second = r.fold_receipt(synthetic_rows())
    assert first == second
    assert len(first["folds"]) == 5
    assert set(first["fold_a_families"]).isdisjoint(first["fold_b_families"])
    assert set(first["fold_a_families"]) | set(first["fold_b_families"]) == {
        row["family_id"] for row in synthetic_rows()
    }


@pytest.mark.parametrize(
    "row",
    [
        {"dl_split": "DL-Val", "task": "regression", "family_id": "f"},
        {"dl_split": "DL-Train", "task": "classification", "family_id": "f"},
        {
            "dl_split": "DL-Train",
            "task": "regression",
            "family_id": "f",
            "has_meta_test": True,
        },
    ],
)
def test_fold_rejects_out_of_scope_or_invalid_rows(row):
    with pytest.raises(ValueError):
        r.family_folds([row] + synthetic_rows())


def test_per_cell_attempt_scratch_is_unique_and_non_reusable(tmp_path):
    cells = [r.scratch_path(tmp_path, arm, seed) for arm in r.ARMS for seed in r.SEEDS]
    assert len(cells) == 9 and len(set(cells)) == 9
    assert r.scratch_path(tmp_path, "FBAL", 39, 2) != r.scratch_path(
        tmp_path, "FBAL", 39, 1
    )
    assert r.require_fresh_scratch(tmp_path, "FBAL", 39) == cells[3]
    cells[3].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        r.require_fresh_scratch(tmp_path, "FBAL", 39)


def test_sampler_plans_match_budget_and_family_balancing_is_explicit():
    roster = {"small": ["s1"], "large": [f"l{i}" for i in range(9)]}
    control = r.sample_microbatches(roster, "CONTROL", 39)
    balanced = r.sample_microbatches(roster, "FBAL", 39)
    assert control == r.sample_microbatches(roster, "CONTROL", 39)
    assert len(control) == len(balanced) == r.UPDATES * r.MICROBATCHES
    assert all(len(batch) == r.MICROBATCH_SIZE for batch in control + balanced)
    assert control != balanced


def test_read_only_source_copy_is_isolated_and_hash_checked(tmp_path):
    source = tmp_path / "readonly-source.pt"
    source.write_bytes(b"synthetic checkpoint, not model data")
    source.chmod(0o444)
    digest = r._sha256(source)
    scratch = r.require_fresh_scratch(tmp_path, "CONTROL", 39)
    staged = r.stage_r882_copy(source, scratch, expected_sha256=digest)
    assert staged.read_bytes() == source.read_bytes()
    assert not staged.samefile(source)
    assert source.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        r.stage_r882_copy(source, scratch, expected_sha256=digest)


def test_accumulation_makes_one_mean_gradient_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    losses = [parameter.square().sum() for _ in range(r.MICROBATCHES)]
    value = r.accumulate_step([parameter], optimizer, losses)
    assert value == pytest.approx(1.0)
    assert parameter.item() == pytest.approx(0.9)  # global norm clip is fixed at 1.0


def test_schedule_contract_and_cpu_dry_run(tmp_path):
    assert len({r.learning_rate("CONTROL", i) for i in range(r.UPDATES)}) == 1
    assert len({r.learning_rate("FBAL", i) for i in range(r.UPDATES)}) == 1
    rates = [r.learning_rate("FBAL+WCR", i) for i in range(r.UPDATES)]
    assert rates[0] == pytest.approx(r.BASE_LR / 2)
    assert rates[1] == pytest.approx(r.BASE_LR)
    assert rates[2] == pytest.approx(
        r.BASE_LR
    )  # first cosine cycle begins after warmup
    assert rates[3] < rates[2]
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([0.0]))], lr=2.0)
    assert r.set_update_learning_rate(optimizer, "FBAL+WCR", 0) == pytest.approx(
        r.BASE_LR / 2
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(r.BASE_LR / 2)
    receipt = r.cpu_dry_run(tmp_path)
    assert receipt["status"] == "cpu_dry_run_pass"
    assert receipt["optimizer_updates"] == 0
    assert receipt["gpu_ticket_created"] is False


def test_readout_synthetic_bootstrap_elo_and_admission(tmp_path):
    receipt_root = tmp_path / "dry-run"
    receipt_root.mkdir()
    receipt = r.readout_cpu_dry_run(receipt_root)
    assert receipt["status"] == "READOUT_DRY_RUN_PASS"
    assert receipt["synthetic_cells"] == len(r.ARMS) * len(r.SEEDS)
    assert receipt["real_outcome_files_read"] == 0
    assert receipt["synthetic_receipts_emitted"] == 9
    assert receipt["synthetic_receipts_read"] == 9
    assert receipt["synthetic_fb_admitted"] is True
    assert receipt["synthetic_control_admitted"] is False
    assert receipt["GPU_used"] is False
    assert receipt["qualification_lock_used"] is False
    assert receipt["FIFO_ticket_created"] is False
    selected = {f"{arm}/seed{seed}": 1 for arm in r.ARMS for seed in r.SEEDS}
    hashes = {
        key: receipt["receipt_sha256"][
            f"{key.split('/')[0].replace('+', 'plus')}-seed{key.split('/')[1][4:]}-attempt1.json"
        ]
        for key in selected
    }
    emitted = r.read_outcome_receipts(receipt_root, selected, hashes, synthetic=True)
    assert len(emitted) == 9
    assert emitted[0]["receipt_schema_version"] == "jope.h109.synthetic-cell-receipt/v1"
    receipt_path = receipt_root / "cell-receipts" / "CONTROL-seed39-attempt1.json"
    receipt_path.write_text(receipt_path.read_text() + " ")
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        r.read_outcome_receipts(receipt_root, selected, hashes, synthetic=True)


def _synthetic_outcome_cells():
    families = [f"fixture-{index}" for index in range(8)]
    baseline = {family: 0.0 for family in families}
    cells = []
    for arm in r.ARMS:
        for seed in r.SEEDS:
            cells.append(
                {
                    "schema": "jope.h109.cell-outcome/v1",
                    "wave": "H109",
                    "arm": arm,
                    "seed": seed,
                    "attempt": 1,
                    "cell_status": "COMPLETE",
                    "trained_parameter_count": r.ADAPTER_PARAMS,
                    "synthetic_fixture": True,
                    "execution_provenance": {
                        "gpu_used": False,
                        "qualification_lock_held": False,
                        "owner_authorized": False,
                        "owner_authorization_ref": None,
                        "qualification_lock_path": None,
                        "fifo_ticket_id": None,
                        "gpu_device": "synthetic",
                    },
                    "receipt_schema_version": "jope.h109.synthetic-cell-receipt/v1",
                    "synthetic_receipt": True,
                    "prereg_sha256": r.prereg_sha256(),
                    "fold_seed": r.FOLD_SEED,
                    "source_commit": "fixture-source",
                    "fold_assignment_sha256": "fixture-fold",
                    "checkpoint_sha256": r.R882_SHA256,
                    "dope_before": r.DOPE_SHA256,
                    "dope_after": r.DOPE_SHA256,
                    "optimizer_updates": r.UPDATES,
                    "dl_val_reads": 0,
                    "sealed_holdout_meta_test_reads": 0,
                    "assessment": {"utility_by_family": dict(baseline)},
                    "r882_utility_by_family": dict(baseline),
                }
            )
    return cells


def test_readout_fails_closed_on_missing_cells_or_wrong_prereg():
    cells = _synthetic_outcome_cells()
    with pytest.raises(RuntimeError, match="all nine"):
        r.summarize_readout(cells[:-1])
    cells[0]["prereg_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity, completion, or boundary"):
        r.summarize_readout(cells, synthetic=True)


def test_readout_exact_margin_ties_do_not_advance():
    cells = _synthetic_outcome_cells()
    for cell in cells:
        if cell["arm"] == "FBAL":
            cell["assessment"]["utility_by_family"] = {
                family: 0.001 for family in cell["r882_utility_by_family"]
            }
    metrics = r.summarize_readout(cells, synthetic=True)
    assert metrics["arms"]["FBAL"]["family_mean_gain"] == pytest.approx(0.001)
    assert metrics["arms"]["FBAL"]["holm"]["passes"] is False
    assert metrics["arms"]["FBAL"]["admitted"] is False


def test_outcome_receipt_emission_is_durable_and_attempt_scoped(tmp_path):
    cell = _synthetic_outcome_cells()[0]
    first = r.emit_outcome_receipt(cell, tmp_path, synthetic=True)
    first_path = Path(first["path"])
    assert first_path.is_file()
    assert r._sha256(first_path) == first["sha256"]
    emitted = json.loads(first_path.read_text())
    assert emitted["receipt_schema_version"] == "jope.h109.synthetic-cell-receipt/v1"
    with pytest.raises(FileExistsError, match="preserve it"):
        r.emit_outcome_receipt(cell, tmp_path, synthetic=True)
    retry = dict(cell, attempt=2)
    second = r.emit_outcome_receipt(retry, tmp_path, synthetic=True)
    assert Path(second["path"]) != first_path
    assert Path(second["path"]).is_file()


def test_real_emission_rejects_synthetic_cells_and_non_h109_root(tmp_path):
    cell = _synthetic_outcome_cells()[0]
    with pytest.raises(ValueError, match="synthetic fixture"):
        r.emit_outcome_receipt(cell, tmp_path)
    with pytest.raises(ValueError, match=r"run-\* H109 data root"):
        r._safe_output_directory(tmp_path, synthetic=False)


def test_real_cell_receipt_requires_owner_lock_and_fifo_provenance():
    cell = _synthetic_outcome_cells()[0]
    cell["synthetic_fixture"] = False
    cell["source_commit"] = "a" * 40
    cell["fold_assignment_sha256"] = "b" * 64
    cell["runner_sha256"] = r._sha256(Path(r.__file__))
    cell["candidate_adapter_sha256"] = "c" * 64
    cell["execution_provenance"] = {}
    with pytest.raises(ValueError, match="GPU, lock, owner, and FIFO evidence"):
        r._validate_cell_receipt(cell, synthetic=False)
