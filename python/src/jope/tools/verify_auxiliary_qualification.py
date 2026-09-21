#!/usr/bin/env python3
"""Bind the real two-block auxiliary run to its frozen parent and rank receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root, output = a.root.resolve(), a.output.resolve()
    bound = {}

    def read(path):
        bound[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())

    report = read(output / "qualification.json")
    assert report["passed"] and report["world"] == 8 and report["optimizer_steps"] == 20
    assert report["gp_heartbeat_advanced"] and report["final_tests_opened"] is False
    spec = read(output / "spec.json")
    complete = read(output / "complete.json")
    assert complete["spec_sha256"] == digest(output / "spec.json")
    assert complete["completed_optimizer_steps"] == 20
    parent = spec["frozen_proposal_checkpoint"]
    parent_state = read(root / parent["checkpoint"] / "state.json")
    assert digest(root / parent["checkpoint"] / "state.json") == parent["state_sha256"]
    assert parent["state_sha256"] == report["frozen_proposal_checkpoint_sha256"]
    panel = spec["panel_sha256"]
    launch_results = []
    for launch in sorted((output / "train").glob("launch-*")):
        receipt = read(launch / "controller.json")
        assert receipt["passed"]
        ranks = [read(launch / f"complete-{rank}.json") for rank in range(8)]
        assert len({row["parameter_sha256"] for row in ranks}) == 1
        assert len({row["steps"] for row in ranks}) == 1
        launch_results.append(
            dict(
                step=ranks[0]["steps"],
                resumed=receipt["resumed"],
                seconds=receipt["seconds"],
                parameter_sha256=ranks[0]["parameter_sha256"],
            )
        )
    launch_results.sort(key=lambda row: row["step"])
    assert [(row["step"], row["resumed"]) for row in launch_results] == [(10, False), (20, True)]
    for step in [10, 20]:
        cp = output / f"train/checkpoints/step-{step}"
        state = read(cp / "state.json")
        opt = read(cp / "optimizer.json")
        assert state["step"] == step and opt["step"] == step
        assert state["initialization"]["state_sha256"] == parent["state_sha256"]
        assert state["validation"]["evaluated_step"] == step
        assert state["validation"]["panel_sha256"] == panel
        assert state["corpus_sha256"] == spec["corpus_sha256"]
        assert state["sampler_id"] == spec["sampler_id"]
        for name, key in [
            ("model.ot", "model_sha256"),
            ("optimizer.ot", "optimizer_sha256"),
            ("optimizer.json", "optimizer_metadata_sha256"),
            ("config.json", "config_sha256"),
        ]:
            assert digest(cp / name) == state[key]
            bound[str((cp / name).relative_to(root))] = state[key]
    assert parent_state["validation"]["panel_sha256"] == panel
    selected = complete["selection"]
    selected_state = read(root / selected["checkpoint"] / "state.json")
    assert digest(root / selected["checkpoint"] / "state.json") == selected["state_sha256"]
    assert selected_state["validation"]["sequence_nll"] == selected["validation_nll"]
    assert selected["validation_nll"] <= parent["validation_nll"]
    result = dict(
        schema="parallel.auxiliary_receipt_verification.v1",
        passed=True,
        launches=launch_results,
        selected=selected,
        validation_panel_sha256=panel,
        exact_optimizer_continuation=True,
        frozen_proposal_checkpoint_unchanged=True,
        inputs=bound,
        qualification=True,
        final_tests_opened=False,
    )
    path = output / "verified.json"
    assert not path.exists(), "Verification output exists"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}))


if __name__ == "__main__":
    main()
