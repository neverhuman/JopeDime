"""Direct API controls for the QLT-03 research record reader."""

import importlib.util
import json
from pathlib import Path

_READER_PATH = Path(__file__).parents[1] / "src/jope/north_star/record_reader.py"
_SPEC = importlib.util.spec_from_file_location("qlt03_record_reader", _READER_PATH)
reader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reader)


def _program(index):
    return {
        "program_sha256": f"{index:064x}",
        "operator_family": "synthetic_control",
        "feat412": [0.0] * 412,
        "proj16": [0.0] * 16,
    }


def _record(*, evaluate_all, complete):
    proposals = [_program(index) for index in range(5)]
    dispatched = list(range(5))
    if not complete:
        dispatched = dispatched[:-1]
    return {
        "schema": reader.VERSION,
        "identity": {
            "run_id": "synthetic-control",
            "task": "regression",
            "evidence": "synthetic_control",
            "partition": "train",
            "source_commit": "a" * 40,
            "corpus_manifest_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "family_sha256": "d" * 64,
            "problem_sha256": "e" * 64,
            "bundle_sha256": "f" * 64,
            "conditioning_sha256": "0" * 64,
            "evaluator_sha256": "1" * 64,
            "dope_weights_sha256": reader.DOPE_SHA256,
        },
        "round": 0,
        "seed": 29,
        "tokens": [[0.0] * 32 for _ in range(10)],
        "mask": [True] * 10,
        "revealed": [],
        "proposals": proposals,
        "source_indices": list(range(5)),
        "policy": {
            "kind": "heuristic",
            "explore": False,
            "slate_costs": None,
            "admitted": list(range(5)),
        },
        "evaluate_all_proposals": evaluate_all,
        "dispatched": dispatched,
        "capture_complete": complete,
        "outcomes": [
            {
                "proposal_index": index,
                "wall_ms": 1,
                "measurement": {
                    "status": "success",
                    "surrogate": 0.0,
                    "objective": 0.0,
                    "gate": True,
                },
            }
            for index in dispatched
        ],
    }


def _write(path, record):
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_reader_api_accepts_complete_all_proposal_capture(tmp_path):
    path = tmp_path / "all.jsonl"
    _write(path, _record(evaluate_all=True, complete=True))

    rows = list(reader.read_records(path, allow_synthetic=True, require_evaluate_all=True))

    assert len(rows) == 1
    assert rows[0]["dispatched"] == list(range(5))


def test_reader_api_rejects_complete_ordinary_capture_for_all_proposal_consumer(tmp_path):
    path = tmp_path / "ordinary.jsonl"
    _write(path, _record(evaluate_all=False, complete=True))

    try:
        list(reader.read_records(path, allow_synthetic=True, require_evaluate_all=True))
    except ValueError as error:
        assert "complete all-proposal capture required" in str(error)
    else:
        raise AssertionError("ordinary capture passed the all-proposal reader guard")


def test_reader_api_rejects_partial_all_proposal_capture_even_for_inspection(tmp_path):
    path = tmp_path / "partial-all.jsonl"
    _write(path, _record(evaluate_all=True, complete=False))

    try:
        list(
            reader.read_records(
                path,
                allow_synthetic=True,
                allow_incomplete=True,
                require_evaluate_all=True,
            )
        )
    except ValueError as error:
        assert "complete all-proposal capture required" in str(error)
    else:
        raise AssertionError("partial all-proposal capture passed the reader guard")
