"""Explicit synthetic recipe-boundary fixtures; no fit/model/training work."""

import hashlib
import json
from copy import deepcopy

import pytest

from jope.grammars import all_grammars
from jope.grammars.vocab import build_active_vocab_v3
from jope.policy.training_corpus_contract import validate_teaching_recipe


def fixture(tmp_path, genome=" (pipe X (ridge)) "):
    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()

    executable = "(pipe X (input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256) (ridge lambda=1.0))"
    recipe = {
        "original_proposal": genome,
        "original_bytes_sha256": sha(genome),
        "executable_canonical": executable,
        "executable_sha256": sha(executable),
    }
    candidate = {"genome": genome, "recipe": recipe, "program_sha256": sha(executable)}
    data = json.dumps({"candidate": candidate}).encode()
    path = tmp_path / "recipe.json"
    path.write_bytes(data)
    ref = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
    row = {
        "genome": genome,
        "dataset_id": "a" * 64,
        "engine": "featmath",
        "metadata": {
            "task_type": "regression",
            "teaching_recipe": ref,
            "executable_program_sha256": sha(executable),
            "original_proposal_sha256": sha(genome),
        },
    }
    qualification = {
        "record_sha256": ref["sha256"],
        "program_sha256": sha(executable),
        "comparison": {"task": "regression", "dataset_hash": "a" * 64},
    }
    return row, qualification, build_active_vocab_v3([], all_grammars())


def test_exact_original_is_preserved(tmp_path):
    args = fixture(tmp_path)
    before = deepcopy(args[0])
    validate_teaching_recipe(*args)
    assert args[0] == before


@pytest.mark.parametrize("genome", ["(pipe X (elasticnet alpha=1.23456789))", "(pipe X (ridge surprise=1))"])
def test_oov_or_dropped_parameter_is_refused(tmp_path, genome):
    with pytest.raises(ValueError):
        validate_teaching_recipe(*fixture(tmp_path, genome))


@pytest.mark.parametrize("mutation", ["original", "executable", "record", "task", "canonical", "legacy"])
def test_binding_and_frozen_policy_are_required(tmp_path, mutation):
    row, q, vocabulary = fixture(tmp_path)
    if mutation == "original":
        row["genome"] += " "
    elif mutation == "executable":
        q["program_sha256"] = "b" * 64
    elif mutation == "record":
        q["record_sha256"] = "b" * 64
    elif mutation == "task":
        row["metadata"]["task_type"] = "binary"
    elif mutation == "canonical":
        row["canonical_genome"] = "(pipe X (ridge))"
    else:
        del vocabulary["engines"]["featmath"]["heads"]["ridge"]["tasks"]
    with pytest.raises(ValueError):
        validate_teaching_recipe(row, q, vocabulary)


def frozen_table():
    from dataclasses import asdict

    from jope.policy.tokens import build_action_vocab

    value = asdict(build_action_vocab(all_grammars()))
    value.pop("op_id_by_name")
    value["schema"] = "parallel.action_vocab.v1"
    return json.loads(json.dumps(value))


def test_full_action_table_preserves_every_token_id(tmp_path):
    from jope.pipeline_text import parse_pipeline_actions
    from jope.policy.tokens import decode_tokens, encode_actions
    from jope.policy.training_corpus_contract import load_frozen_action_vocab

    value = frozen_table()
    specs, vocab = load_frozen_action_vocab(value)
    assert vocab.tokens == value["tokens"]
    assert vocab.version == value["version"]
    assert {str(k): v.token_id for k, v in vocab.ops.items()} == {
        k: v["token_id"] for k, v in value["ops"].items()
    }
    spec = next(s for s in specs if s.engine == "featmath")
    actions = parse_pipeline_actions("(pipe X (standardize) (ridge))", spec)
    tokens = encode_actions("featmath", actions, vocab, task="regression")
    engine, decoded = decode_tokens(tokens, vocab)
    assert encode_actions(engine, decoded, vocab, task="regression") == tokens
    row, qualification, _ = fixture(tmp_path)
    validate_teaching_recipe(row, qualification, value)


@pytest.mark.parametrize("mutation", ["token", "index", "task", "slot", "operator", "expression"])
def test_full_action_table_rejects_changed_bindings(mutation):
    from jope.policy.training_corpus_contract import load_frozen_action_vocab

    value = frozen_table()
    if mutation == "token":
        value["tokens"][0] = "wrong"
    elif mutation == "index":
        value["index"]["BOS"] = 2
    elif mutation == "task":
        value["task_ids"]["regression"] = 0
    elif mutation == "slot":
        op = next(op for op in value["ops"].values() if any(s["val_ids"] for s in op["slots"]))
        next(slot for slot in op["slots"] if slot["val_ids"])["val_ids"][0] = 0
    elif mutation == "operator":
        next(iter(value["ops"].values()))["token_id"] = 0
    else:
        value["expr"]["const_values"][0] = 12345.0
    with pytest.raises(ValueError):
        load_frozen_action_vocab(value)
