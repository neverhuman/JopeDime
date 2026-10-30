"""Arithmetic and refusal controls only; these fixtures are not scientific data."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "h27_analysis", Path(__file__).with_name("analyze.py")
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture():
    contexts, pairs = [], []
    # Nine contexts in one family, one in another: families still have equal weight.
    for i in range(10):
        c = dict(
            context_sha256=str(i),
            family_id="a" if i < 9 else "b",
            dl_split="DL-Val",
            slate_sha256="slate",
            evaluator_sha256="eval",
            cut_id="cut",
            vocabulary_sha256="vocab",
        )
        contexts.append(c)
        for seed in analysis.SEEDS:
            pairs.append(
                dict(
                    c,
                    seed=seed,
                    with_fingerprint=dict(status="ok", fitness=0 if i < 9 else 2),
                    without_fingerprint=dict(status="ok", fitness=1),
                )
            )
    return dict(
        dope_loaded_sha256=analysis.DOPE,
        seeds=list(analysis.SEEDS),
        train_families=["train"],
        validation_contexts=contexts,
    ), pairs


def test_whole_family_weighting_and_deterministic_interval():
    manifest, pairs = fixture()
    result = analysis.analyze(manifest, pairs, draws=1000)
    assert result["elo"] == 0  # Context weighting would produce +381.7 Elo.
    assert result["ci975"] == [-math.inf, math.inf]
    assert result["verdict"] == "REJECT"
    assert result == analysis.analyze(manifest, list(reversed(pairs)), draws=1000)
    assert analysis.json_safe(result)["ci975"] == ["-Infinity", "+Infinity"]


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "unknown", "nonfinite", "slate", "role", "family", "dope"],
)
def test_invalid_or_leaking_pairs_refused(corruption):
    manifest, pairs = copy.deepcopy(fixture())
    if corruption == "missing":
        pairs.pop()
    elif corruption == "duplicate":
        pairs.append(pairs[0])
    elif corruption == "unknown":
        pairs[0]["with_fingerprint"]["status"] = "unknown"
    elif corruption == "nonfinite":
        pairs[0]["with_fingerprint"]["fitness"] = math.nan
    elif corruption == "slate":
        pairs[0]["slate_sha256"] = "other"
    elif corruption == "role":
        manifest["validation_contexts"][0]["dl_split"] = "DL-Holdout"
    elif corruption == "family":
        manifest["train_families"].append("a")
    else:
        manifest["dope_loaded_sha256"] = "other"
    with pytest.raises(ValueError):
        analysis.analyze(manifest, pairs, draws=1000)


def test_explicit_failures_and_exact_ties():
    assert analysis.outcome(None, 0) == 0
    assert analysis.outcome(0, None) == 1
    assert analysis.outcome(None, None) == 0.5
    assert analysis.outcome(1, 1) == 0.5
    assert analysis.elo(0.5) == 0
    assert analysis.elo(0.75) == pytest.approx(190.84850188786498)
