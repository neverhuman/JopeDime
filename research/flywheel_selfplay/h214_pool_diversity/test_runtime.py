"""Synthetic adapter integration: no model, source cohort or native worker is opened."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.flywheel_selfplay.h214_pool_diversity import runner as r


def program(index):
    dsl = f"(pipe X (ridge alpha={index}))"
    return dict(dsl=dsl, program_sha256=hashlib.sha256(dsl.encode()).hexdigest())


def test_bank_is_train_only_family_disjoint_and_parent_ties_use_sha():
    one = dict(
        program(1), dl_split="DL-Train", family_id="train", landmark_summary=[0] * 12
    )
    two = dict(
        program(2), dl_split="DL-Train", family_id="target", landmark_summary=[0] * 12
    )
    assert r.family_bank([one, two], "target") == [one]
    assert r.retrieval_parent([one, two], np.zeros(48))["program_sha256"] == min(
        one["program_sha256"], two["program_sha256"]
    )
    with pytest.raises(ValueError):
        r.family_bank([dict(one, dl_split="DL-Val")], "target")


def test_complete_adapter_evaluates_only_selected_pools_and_passes_context_explicitly(
    tmp_path,
):
    cid = "a" * 64
    raw = b"synthetic train-only fixture"
    raw_sha = hashlib.sha256(raw).hexdigest()
    source = tmp_path / "input"
    (source / "tables").mkdir(parents=True)
    (source / "tables" / (cid + ".jtt1")).write_bytes(raw)
    output = tmp_path / "output"
    output.mkdir()
    programs = [program(i) for i in range(65)]
    calls = []

    def propose(**kwargs):
        assert kwargs["target_family_id"] == "val-family"
        assert all(p["family_id"] != "val-family" for p in kwargs["champion_bank"])
        return programs[1 : 1 + kwargs["n_proposals"]]

    def features(dsl):
        index = int(dsl.split("alpha=")[1].split(")")[0])
        x = np.zeros(16)
        x[:2] = [np.cos(index * 2 * np.pi / 64), np.sin(index * 2 * np.pi / 64)]
        return np.zeros(412), x

    def evaluate(table, items, **kwargs):
        assert kwargs == dict(run_outer_test=False, max_workers=4, timeout_s=25.0)
        calls.append(items)
        return [
            dict(
                program_sha256=p["program_sha256"],
                status="completed",
                fitness=0.25,
                cv_gate_passed=True,
            )
            for p in items
        ]

    def predict(net, h124, context_id, context_row, pool, projection, items, generator):
        assert context_id == cid and "context_sha256" not in context_row
        return np.zeros(4 + len(items))

    def validate(data, view, identity, transform):
        assert (
            data == raw
            and view == raw
            and identity == raw_sha
            and transform(raw) == raw
        )

    h212 = SimpleNamespace(
        INPUT=source, validate_train_cv_envelope=validate, predict_dynamic=predict
    )
    generator = SimpleNamespace(
        propose_flywheel_candidates=propose,
        featurize_dsl_412=features,
        evaluate_gp_batch_rust=evaluate,
        EVAL_SCRATCH_ROOT=output / "rust_eval",
    )
    champion = dict(
        programs[0],
        family_id="train-family",
        dl_split="DL-Train",
        landmark_summary=[0] * 12,
    )
    runtime = dict(
        h212=h212,
        generator=generator,
        envelope=SimpleNamespace(training_cv_envelope=lambda x: x),
        context={cid: dict(lm48_z=np.zeros(48))},
        champions=[champion],
        net=None,
        h124=None,
        pools={cid: None},
        projection=None,
    )
    record = r.measure(
        runtime,
        dict(
            context_sha256=cid, family_id="val-family", train_only_jtt1_sha256=raw_sha
        ),
        214101,
        output,
    )
    assert len(calls) == 1 and 16 <= len(calls[0]) <= 31
    assert len(calls[0]) < 64
    for arm in ["NATIVE16", "FPS16"]:
        assert len(record["arms"][arm]["ordered_program_sha256"]) == 16
        assert record["arms"][arm]["selected_index"] == 0
        assert record["arms"][arm]["utility"] == -0.25
    saved = json.loads((output / "panels.jsonl").read_text())
    assert saved["context_sha256"] == cid and saved["evaluated_union"] == len(calls[0])
    assert (source / "tables" / (cid + ".jtt1")).read_bytes() == raw


def test_incomplete_native_result_is_not_relabelled_as_gate_failure():
    programs = [program(i) for i in range(16)]
    pools = {"NATIVE16": programs, "FPS16": programs}
    results = [
        dict(
            program_sha256=p["program_sha256"],
            status="completed",
            fitness=1.0,
            cv_gate_passed=True,
        )
        for p in programs
    ]
    with pytest.raises(ValueError):
        r.score_panels(pools, programs, results[:-1], np.zeros(16))
    results[0]["status"] = "timeout"
    with pytest.raises(ValueError):
        r.score_panels(pools, programs, results, np.zeros(16))
