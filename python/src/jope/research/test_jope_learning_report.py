from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("jope_learning_report.py")
SPEC = importlib.util.spec_from_file_location("jope_learning_report", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def observation(index: int, candidate: float = 0.8, accepted: float = 0.7) -> dict:
    return {
        "context_sha256": f"{index + 1:064x}",
        "source_family_sha256": f"{index + 1000:064x}",
        "candidate_score": candidate,
        "accepted_score": accepted,
        "gp_score": 0.6,
        "candidate_conditional_advantage": 0.03,
        "accepted_conditional_advantage": 0.01,
        "valid": True,
        "replay_verified": True,
        "candidate_failed": False,
        "accepted_failed": False,
        "candidate_latency_ms": 10,
        "accepted_latency_ms": 10,
        "strict_win": True,
        "equal_budget_win": True,
        "slices": {
            "origin": "real",
            "size": "small",
            "missingness": "none",
            "imbalance": "balanced",
            "target_shape": "scalar",
        },
    }


def evaluation(checkpoint: str, panel: str, run: int, *, accepted: bool = True, lower: float = 0.01) -> dict:
    return {
        "schema": "TaskPromotionEvaluation.v3",
        "task": "binary",
        "accepted_parent_checkpoint_sha256": "b" * 64,
        "candidate_checkpoint_sha256": checkpoint,
        "panel_consumption_sha256": panel,
        "run_index": run,
        "metrics": {
            "contexts": 128,
            "confidence": {
                "resamples": 10000,
                "seed": run,
                "candidate_vs_accepted_lower": lower,
                "candidate_vs_gp_lower": 0.01,
                "conditional_advantage_lower": 0.01,
            },
            "maximum_slice_regression": 0.0,
            "validity": 1.0,
            "replay_verification": 1.0,
            "failure_rate_increase": 0.0,
            "latency_ratio": 1.0,
            "strict_wins": 128,
            "equal_budget_wins": 128,
        },
        "accepted": accepted,
        "rejection_reasons": [] if accepted else ["parent regression"],
        "evaluated_at_ms": 1,
    }


def install_run(root: Path, checkpoint: str, run: int, *, accepted: bool = True, lower: float = 0.01) -> None:
    panel = hashlib.sha256(f"panel-{run}".encode()).hexdigest()
    disposition = "accepted" if accepted else "rejected"
    write(
        root / f"training/task-promotion-evaluations/{disposition}/run-{run}.v3.json",
        evaluation(checkpoint, panel, run, accepted=accepted, lower=lower),
    )
    rows = [observation(run * 128 + index) for index in range(128)]
    write(
        root / f"training/task-promotion-evaluations/work/run-{run}/prefix-128.observations.v3.json",
        {
            "schema": "TaskPromotionObservationSet.v3",
            "panel_consumption_sha256": panel,
            "observations": rows,
        },
    )


def test_learning_report_requires_three_native_passes_and_pointer_cas(tmp_path):
    checkpoint = "a" * 64
    for run in range(3):
        install_run(tmp_path, checkpoint, run)
    report = module.build_report(tmp_path, checkpoint, "binary", 10)
    assert report["schema"] == "JopeLearningReport.v2"
    assert report["disposition"] == "inconclusive"
    write(
        tmp_path / f"training/task-promotions/binary-{checkpoint}.v1.json",
        {
            "schema": "TaskCheckpointPromotion.v2",
            "task": "binary",
            "child_checkpoint_sha256": checkpoint,
            "independent_single_use_panel_acceptances": 3,
            "external_pass_receipt_ids": ["x", "y", "z"],
        },
    )
    write(
        tmp_path / "training/accepted-by-task.json",
        {"tasks": {"binary": {"checkpoint_sha256": checkpoint}}},
    )
    report = module.build_report(tmp_path, checkpoint, "binary", 10)
    assert report["disposition"] == "improved"
    assert report["metrics"]["panel_coverage"] == 3
    assert report["metrics"]["source_family_count"] == 384
    assert report["baseline_diagnosis"]["rate"] == 0.175


def test_complete_negative_parent_bound_is_regressed_but_gp_only_is_inconclusive(tmp_path):
    checkpoint = "c" * 64
    install_run(tmp_path, checkpoint, 0, accepted=False, lower=-0.01)
    report = module.build_report(tmp_path, checkpoint, "binary", 10)
    assert report["disposition"] == "regressed"
    evaluation_path = tmp_path / "training/task-promotion-evaluations/rejected/run-0.v3.json"
    value = json.loads(evaluation_path.read_text())
    value["metrics"]["confidence"]["candidate_vs_accepted_lower"] = 0.003
    value["metrics"]["confidence"]["candidate_vs_gp_lower"] = -0.01
    write(evaluation_path, value)
    assert module.build_report(tmp_path, checkpoint, "binary", 10)["disposition"] == "inconclusive"


def test_metrics_are_bounded_before_difference():
    assert module.bound_metric("regression", 1e300) == 1.0
    assert module.bound_metric("regression", -1e300) == -1.0
    assert module.bound_metric("timeseries", -0.25) == 0.75
    assert module.bound_metric("timeseries", -100.0) == -1.0


def test_learning_report_is_immutable(tmp_path):
    report = module.build_report(tmp_path, "d" * 64, "regression", 10)
    path = tmp_path / "report.json"
    first = module.publish(path, report)
    assert module.publish(path, report) == first
    report["disposition"] = "regressed"
    try:
        module.publish(path, report)
    except ValueError:
        pass
    else:
        raise AssertionError("mutable report overwrite was accepted")
