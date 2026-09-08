from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

import validate_unified_v3 as validator  # noqa: E402


def graphs() -> tuple[dict, dict]:
    return (
        json.loads(validator.V3_GRAPH.read_text()),
        json.loads(validator.V2_GRAPH.read_text()),
    )


def test_unified_graph_and_append_only_genesis_validate():
    graph, v2 = graphs()
    validator.validate_graph(graph, v2)
    validator.validate_ledger(graph)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda graph: graph["migration"]["old_to_canonical_milestone"].pop("P-2"),
            "not complete",
        ),
        (
            lambda graph: graph["frozen_capability_policy"]["scq80"].__setitem__(
                "benchmark_valid", True
            ),
            "SCQ80",
        ),
        (
            lambda graph: graph["milestones"][9].__setitem__("optional", False),
            "optional release lanes",
        ),
    ],
)
def test_unified_graph_mutations_fail_closed(mutation, message):
    graph, v2 = graphs()
    changed = copy.deepcopy(graph)
    mutation(changed)
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_graph(changed, v2)
