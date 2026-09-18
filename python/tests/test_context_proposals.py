from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jope.research

SCRIPT_DIR = Path(jope.research.__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "test_exact_context_proposals",
    SCRIPT_DIR / "test_exact_context_proposals.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(character: str) -> str:
    return character * 64


def key(reservoir: int = 64) -> dict:
    return {
        "task": "regression",
        "accepted_checkpoint_sha256": digest("a"),
        "context_sha256": digest("b"),
        "seed": 7,
        "decoding_policy_sha256": digest("c"),
        "reservoir_size": reservoir,
    }


def raw_row(character: str, score: float, head: str = "ridge") -> dict:
    return {
        "semantic_sha256": digest(character),
        "canonical_genome": f"genome-{character}",
        "decoder": "nucleus",
        "decoder_seed": 7,
        "tokens": [1, 2, 3],
        "_head_op": head,
        "reward_mean": score,
        "reward_uncertainty": 0.2,
        "reward_lcb": score - 0.1,
        "instability_probability": 0.1,
        "selection_score": score - 0.1,
        "candidate_risk_reason": None,
    }


def test_cache_identity_excludes_catalog_and_reuses_artifact(tmp_path) -> None:
    first = MODULE.cache_path(tmp_path, key())
    catalog_after_append = digest("f")
    second = MODULE.cache_path(tmp_path, key())
    assert catalog_after_append not in str(second)
    assert first == second

    selected = [
        raw_row(character, 1.0 - index / 10)
        for index, character in enumerate("12345678")
    ]
    artifact = MODULE.proposal_artifact(key(), 64, selected, selected)
    assert MODULE.atomic_create_json(first, artifact)
    assert not MODULE.atomic_create_json(first, artifact)
    assert MODULE.load_artifact(first, key())["proposals"] == artifact["proposals"]


def test_slots_are_logged_before_decode_and_missing_slots_escalate(tmp_path) -> None:
    log = MODULE.write_slot_log(tmp_path, key())
    value = json.loads(log.read_text())
    assert len(value["slots"]) == 8
    assert {row["status"] for row in value["slots"]} == {"pending_decode"}

    rows = [raw_row("a", 1.0), raw_row("b", 0.9, "gpr")]
    retry = MODULE.proposal_artifact(key(128), 128, rows, rows)
    assert [row["status"] for row in retry["slots"][2:]] == [
        "pending_retry"
    ] * 6
    permanent = MODULE.proposal_artifact(key(256), 256, rows, rows)
    assert [row["status"] for row in permanent["slots"][2:]] == [
        "permanent_failure"
    ] * 6
    assert permanent["rejection_counts"]["unavailable"] == 6
    assert set(permanent["rejection_counts"]) == set(MODULE.REJECTION_REASONS)


def test_calibration_fallback_removes_learned_risk_from_ranking() -> None:
    bad = {
        "realized_error_correlation_lower_bound_95": -0.01,
        "interval_coverage": 0.8,
        "target_coverage": 0.95,
    }
    selection_policy = {"learned_risk": {"enabled": True}}
    policy = MODULE.decoding_policy(
        bad, digest("d"), digest("e"), selection_policy, digest("f")
    )
    assert policy["proposal_selection_policy_sha256"] == digest("f")
    assert not policy["learned_uncertainty_and_risk_enabled"]
    assert policy["selector_calibration_sha256"] is None
    assert policy["uncertainty_fallback"] == "empirical_reward_mad"
    assert (
        policy["validity_repair"]
        == "dataset-local-deterministic-featmath-grammar-v1"
    )

    rows = [raw_row("a", 0.2), raw_row("b", 0.8)]
    rows[1]["instability_probability"] = 0.99
    MODULE.empirical_dispersion_rank(rows)
    assert rows[1]["selection_score"] > rows[0]["selection_score"]


def test_calibration_must_pass_both_confidence_and_coverage() -> None:
    qualified = {
        "realized_error_correlation_lower_bound_95": 0.0,
        "interval_coverage": 0.91,
        "target_coverage": 0.95,
    }
    assert MODULE.calibration_qualifies(qualified)
    qualified["interval_coverage"] = 0.899
    assert not MODULE.calibration_qualifies(qualified)


def test_validity_repair_source_is_preserved_in_aggregate_rows() -> None:
    repair = raw_row("a", 1.0, "logistic")
    repair["source"] = "grammar_validity_repair"
    artifact = MODULE.proposal_artifact(key(), 64, [repair], [repair])
    row = MODULE.aggregate_row(
        artifact["proposals"][0],
        task="binary",
        dataset_hash="dataset",
        context_sha256=digest("b"),
        catalog_sha256=digest("c"),
        checkpoint_sha256=digest("d"),
    )
    assert row["source"] == "grammar_validity_repair"
    assert row["origin"] == "grammar_validity_repair"
    assert row["native_origin"] is False


def test_validity_repair_uses_the_strict_task_head_allowlist() -> None:
    assert MODULE.head_valid_for_task("logistic", "binary")
    assert not MODULE.head_valid_for_task("ridge", "binary")
    assert MODULE.head_valid_for_task("ridge", "regression")
    assert not MODULE.head_valid_for_task("logistic", "regression")
