from collections import Counter
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from test_exact_train import (  # noqa: E402
    RECIPE_E_BATCH_MIX,
    RECIPE_E_LOSS_WEIGHTS,
    RECIPE_E_MODEL_OUTPUTS,
    recipe_e_batch_lane_counts,
    recipe_e_evidence_lane,
    recipe_e_training_trigger,
)
from test_exact_packed_corpus import _write_recipe_e_evidence_index  # noqa: E402


def test_recipe_e_mix_is_exact_over_five_batches():
    observed = Counter()
    for batch_index in range(5):
        counts = recipe_e_batch_lane_counts(256, batch_index)
        assert sum(counts.values()) == 256
        assert counts["exact_current_parent_pairs"] % 2 == 0
        observed.update(counts)
    assert observed == {
        lane: int(fraction * 5 * 256)
        for lane, fraction in RECIPE_E_BATCH_MIX.items()
    }


def test_recipe_e_trigger_and_fixed_objective_contract():
    assert recipe_e_training_trigger(50_000, 0)
    assert recipe_e_training_trigger(0, 5_000)
    assert not recipe_e_training_trigger(49_999, 4_999)
    assert sum(RECIPE_E_LOSS_WEIGHTS.values()) == pytest.approx(1.0)
    assert {
        "descendant_value_h1",
        "descendant_value_h2",
        "descendant_value_h4",
        "descendant_value_h8",
        "mutation_operator_distribution",
        "parent_conditioned_structured_edit_distribution",
    }.issubset(RECIPE_E_MODEL_OUTPUTS)


def test_recipe_e_censored_rows_are_not_descendant_labels():
    parent = "a" * 64
    assert (
        recipe_e_evidence_lane(
            {
                "descendant_credit_receipt_sha256": "b" * 64,
                "descendant_horizon": 4,
                "descendant_censored": True,
            },
            parent,
        )
        == "censored_descendant"
    )
    assert (
        recipe_e_evidence_lane(
            {
                "descendant_credit_receipt_sha256": "b" * 64,
                "descendant_horizon": 4,
                "descendant_censored": False,
            },
            parent,
        )
        == "completed_descendant_trajectories"
    )


def test_recipe_e_current_parent_pair_and_hard_negative_are_disjoint():
    parent = "a" * 64
    metadata = {
        "causal_pair_id": "pair",
        "causal_lane": "native_dl_direct",
        "causal_accepted_parent_checkpoint_sha256": parent,
        "validity_hard_negative": True,
    }
    assert recipe_e_evidence_lane(metadata, parent) == "exact_current_parent_pairs"
    assert (
        recipe_e_evidence_lane(
            {"validity_hard_negative": True},
            parent,
        )
        == "validity_instability_hard_negatives"
    )


def test_recipe_e_packed_sidecar_is_deterministic_and_counts_unique_receipts(
    tmp_path,
):
    receipt = "b" * 64
    members = [
        {
            "ordinal": 0,
            "task": "binary",
            "lane": "completed_descendant_trajectories",
            "descendant_credit_receipt_sha256": receipt,
            "descendant_horizon": horizon,
        }
        for horizon in (1, 2, 4, 8)
    ]
    members.append(
        {
            "ordinal": 4,
            "task": "binary",
            "lane": "validity_instability_hard_negatives",
        }
    )
    path = tmp_path / "recipe_e_evidence.jsonl"
    stats = _write_recipe_e_evidence_index(path, members)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == members
    assert stats["rows"] == 5
    assert stats["task_counts"]["binary"] == {
        "completed_descendant_trajectories": 4,
        "validity_instability_hard_negatives": 1,
        "completed_descendant_receipts": 1,
    }
    assert stats["censored_descendants_excluded"] is True
