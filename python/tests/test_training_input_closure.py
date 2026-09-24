"""Projected metadata fixtures; no sealed payload or model is loaded."""

import copy
import json
import tempfile
from pathlib import Path

import pytest

from jope.policy import training_campaign as campaign
from jope.policy import training_input_closure as closure


@pytest.fixture
def state():
    root = Path(__file__).resolve().parents[2] / "target" / "training-closure-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


@pytest.fixture
def projected(state):
    inputs = campaign.synthetic_inputs(state, ["regression"])
    root = Path(inputs["corpus"]["path"]).parent
    rows = [json.loads(line) for line in Path(inputs["corpus"]["path"]).read_text().splitlines()]
    manifest = {
        "schema": "jope.training-input-closure/v1",
        "root": str(root),
        "excluded_roles": sorted(closure.EXCLUDED),
        "families": {r["metadata"]["source_family"]: r["metadata"]["split"] for r in rows},
        "contexts": [r["metadata"]["context_id"] for r in rows],
        "artifacts": [{"artifact": ref, "role": closure.INPUT_ROLES[name]} for name, ref in inputs.items()],
    }
    campaign.write(root / "input-closure.json", manifest)
    inputs["input_closure"] = campaign.artifact(root / "input-closure.json")
    return inputs, manifest


def rebind(inputs, manifest):
    path = Path(inputs["input_closure"]["path"])
    campaign.write(path, manifest)
    inputs["input_closure"] = campaign.artifact(path)


def test_projected_closure_is_invariant_to_excluded_artifact_mutation(projected, state):
    inputs, manifest = projected
    outside = state / "excluded-fixture.json"
    outside.write_text('{"fixture_label":1}')
    before = closure.admit(inputs)
    outside.write_text('{"fixture_label":999}')
    assert closure.admit(inputs) == before == manifest


def test_unlisted_reachable_artifact_is_refused_before_open(projected, monkeypatch):
    inputs, manifest = projected
    forbidden = str(Path(manifest["root"]).parent / "must-not-open.json")
    encoder = Path(inputs["encoder"]["path"])
    campaign.write(encoder, {"nested": {"path": forbidden, "sha256": "a" * 64}})
    inputs["encoder"] = campaign.artifact(encoder)
    for item in manifest["artifacts"]:
        if item["role"] == "encoder-identity":
            item["artifact"] = inputs["encoder"]
    rebind(inputs, manifest)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert str(path) != forbidden, "opened an unlisted input"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="unlisted transitive artifact"):
        closure.admit(inputs)


@pytest.mark.parametrize("corruption", ["role", "family", "extra-file", "outside-root", "hash"])
def test_projection_rejects_forbidden_or_changed_closure(projected, corruption):
    inputs, original = projected
    manifest = copy.deepcopy(original)
    if corruption == "role":
        manifest["artifacts"][0]["role"] = "sealed-meta-test"
    elif corruption == "family":
        manifest["families"][next(iter(manifest["families"]))] = "outside-selection-evaluation"
    elif corruption == "extra-file":
        (Path(manifest["root"]) / "unlisted.json").write_text("{}")
    elif corruption == "outside-root":
        manifest["artifacts"][0]["artifact"] = {"path": "/must-not-read", "sha256": "a" * 64}
    else:
        Path(inputs["corpus"]["path"]).write_text("{}\n")
    rebind(inputs, manifest)
    with pytest.raises(ValueError):
        closure.admit(inputs)
