# ruff: noqa: F811
# Pytest fixtures imported from the existing contract suite.
"""Real tiny CPU siblings and original-deadline refusal; no live quality claim."""

import json
from pathlib import Path

import pytest
from test_policy_training_contract import small_trainer  # noqa: F401
from test_training_campaign import plan, state  # noqa: F401

from jope.policy import training_campaign as c
from jope.policy import training_comparison as comparison


def test_sampler_exact_exposure_and_no_fresh_control():
    schedules = comparison.schedule([0, 1], [2, 3], 4, 2, 7)
    assert sum(i >= 2 for b in schedules["fresh_replay"] for i in b) == 2
    assert all(i < 2 for b in schedules["replay_only"] for i in b)
    assert schedules == comparison.schedule([0, 1], [2, 3], 4, 2, 7)
    with pytest.raises(ValueError, match="exact 25/75"):
        comparison.schedule([0], [1], 1, 2, 7)


def test_actual_siblings_share_parent_and_cutoff(state, small_trainer, monkeypatch):
    plan(state, task="regression")

    def launch(arguments, env, attempt, deadline):
        started = int(c.time.time() * 1000)
        if "jope.policy.training_comparison" in arguments:
            name = arguments[arguments.index("--branch") + 1]
            checked = c.checked

            def guarded(ref, *args, **kwargs):
                path = Path(ref["path"])
                assert (
                    path != state / "fresh.json"
                    and not path.is_relative_to(state / "fresh")
                    and not path.is_relative_to(state / "comparison/fresh_replay")
                ), "replay worker opened fresh data"
                return checked(ref, *args, **kwargs)

            with monkeypatch.context() as guard:
                if name == "replay_only":
                    guard.setattr(c, "checked", guarded)
                comparison.worker(state, name, attempt)
        else:
            c.worker(state, "regression", attempt)
        c.write(
            attempt / "outcome.json",
            {
                "status": "completed",
                "exit_code": 0,
                "started_unix_ms": started,
                "ended_unix_ms": int(c.time.time() * 1000),
                "deadline_unix_ms": deadline,
                "wall_seconds": 0.1,
            },
            immutable=True,
        )
        return "completed"

    monkeypatch.setattr(c, "launch_worker", launch)
    assert c.run(state)["success"]
    head = (state / "regression/head.json").read_bytes()
    fresh = c.synthetic_inputs(state / "fresh", ["regression"])
    rows = [json.loads(line) for line in c.checked(fresh["corpus"]).read_text().splitlines()]
    contexts = c.read(c.checked(fresh["contexts"]))
    contexts["contexts"] = {key + "-fresh": value for key, value in contexts["contexts"].items()}
    for row in rows:
        row["metadata"]["context_id"] += "-fresh"
        row["score"] += 1
    Path(fresh["corpus"]["path"]).write_bytes(b"".join(c.encoded(r) + b"\n" for r in rows))
    c.write(fresh["contexts"]["path"], contexts)
    for key in ("corpus", "contexts"):
        fresh[key] = c.artifact(fresh[key]["path"])
    manifest = state / "fresh.json"
    c.write(manifest, {"schema": "jope.experimental-training-inputs/v1", **fresh})
    frozen = comparison.prepare(state, "regression", manifest, 4)
    with monkeypatch.context() as expired:
        expired.setattr(c.time, "time", lambda: frozen["category_window"]["deadline_unix_ms"] / 1000 + 1)
        exhausted = comparison.run(state)
        assert exhausted["status"] == "inconclusive" and not exhausted["success"]
        assert not list((state / "comparison").glob("*/attempt-*"))
    result = comparison.run(state)
    assert result["success"] and not result["serving_promotion"]
    assert result["window"] == frozen["window"]
    assert result["category_window"] == frozen["category_window"]
    for name, branch in result["branches"].items():
        assert branch["training_proof"]["optimizer_updates_this_run"] == 4
        assert branch["parent_optimizer_sha256"] == frozen["parent_optimizer_sha256"]
        assert branch["fresh_examples"] == (2 if name == "fresh_replay" else 0)
        assert len(branch["examples"]) == 8 and branch["nonpadding_tokens"] > 0
    import torch

    snapshots = {
        name: torch.load(c.checked(branch["resume"]), weights_only=False)["derived_supervision"]
        for name, branch in result["branches"].items()
    }
    assert snapshots["fresh_replay"][: len(snapshots["replay_only"])] == snapshots["replay_only"]
    corrupted = dict(result["branches"]["fresh_replay"])
    corrupted["fresh_examples"] += 1
    with pytest.raises(ValueError, match="consumption receipt differs"):
        comparison.verify_sibling(frozen, "fresh_replay", corrupted, c.sha(state / "comparison.json"))
    assert (state / "regression/head.json").read_bytes() == head
    assert comparison.run(state)["branches"] == result["branches"]
    # Completed work remains verifiable after expiry; no second allocation opens.
    monkeypatch.setattr(c.time, "time", lambda: frozen["category_window"]["deadline_unix_ms"] / 1000 + 1)
    assert comparison.run(state)["success"]


def test_comparison_cannot_open_a_window(state):
    plan(state, task="regression")
    with pytest.raises(FileNotFoundError):
        comparison.prepare(state, "regression", state / "absent", 4)
    assert not (state / "window.json").exists()
