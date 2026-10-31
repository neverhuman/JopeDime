import argparse
import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dev_incumbent import (
    DevIncumbent,
    add_admission_arguments,
    validate_admission_options,
)


@pytest.fixture(autouse=True)
def retention_fixture_publisher(request, tmp_path, monkeypatch):
    if "retention" not in request.node.name:
        return
    import shutil

    import dev_incumbent

    remote = tmp_path / "fixture-remote-store"
    remote.mkdir()

    def publish(path, prefix, profile="fixture"):
        destination = remote / Path(path).name
        shutil.copyfile(path, destination)
        assert destination.read_bytes() == Path(path).read_bytes()
        return {"uri": "fixture://" + destination.name, "readback_verified": True}

    monkeypatch.setattr(dev_incumbent, "publish_checkpoint_archive", publish)


def retention_fixture(tmp_path, count=6, best=1):
    root = tmp_path / "run"
    root.mkdir()
    (root / "rejected").mkdir()
    for n in range(1, count + 1):
        (root / f"round-{n:02d}.pt").write_bytes(f"root-{n}".encode())
        (root / "rejected" / f"round-{n:02d}.pt").write_bytes(f"rejected-{n}".encode())
        (root / f"round-{n:02d}.json").write_text(json.dumps({"round": n}))
        (root / f"round-{n:02d}-scores.jsonl").write_text("{}\n")
        directory = root / f"rust_eval_round{n}"
        directory.mkdir()
        (directory / "input.jtt1").write_bytes(b"training-only-fixture")
        (directory / "eval_s1.json").write_text("{}")
        (directory / "unknown.bin").write_bytes(b"retain unknown evidence")
    (root / "best_dev_checkpoint.json").write_text(
        json.dumps({"best_checkpoint": str(root / f"round-{best:02d}.pt")})
    )
    for name in ["summary.jsonl", "promotions.jsonl", "adapter_admissions.jsonl"]:
        (root / name).write_text("{}\n")
    return root, tmp_path / "archives"


@pytest.mark.parametrize(("best", "resume"), [(1, 2), (2, 1), (1, 1), (6, 1), (1, 6)])
def test_retention_archives_then_prunes_and_preserves_best_resume_recent_receipts(
    tmp_path, best, resume
):
    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path, best=best)
    receipts = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.suffix in (".json", ".jsonl")
    }
    result = retain_checkpoint_payloads(
        root, archives, [f"round-{resume:02d}.pt"], keep_last=2, minimum_free_bytes=0
    )
    assert result["pruned_files"] == result["restored_verified_files"] > 0
    assert Path(result["archive"]).is_file()
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["files"] and all(
        not p.endswith((".json", ".jsonl")) for p in manifest["files"]
    )
    for n in {best, resume, 5, 6}:
        assert (root / f"round-{n:02d}.pt").read_bytes() == f"root-{n}".encode()
    assert not (root / "rejected/round-05.pt").exists()
    assert not (root / "rejected/round-06.pt").exists()
    assert not (root / "rejected/round-03.pt").exists()
    assert not (root / "rust_eval_round3/input.jtt1").exists()
    assert (root / "rust_eval_round3/unknown.bin").exists()
    for rel, body in receipts.items():
        assert (root / rel).read_bytes() == body


def test_retention_default_keeps_twenty_and_retains_unqualified_payloads(tmp_path):
    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path, count=25)
    (root / "round-999.pt").write_bytes(b"incomplete; no round receipt")
    (root / "round-03.pt.new").write_bytes(b"partial checkpoint")
    retain_checkpoint_payloads(
        root, archives, [root / "round-02.pt"], minimum_free_bytes=0
    )
    for n in [1, 2, *range(6, 26)]:
        assert (root / f"round-{n:02d}.pt").exists()
    assert not (root / "round-03.pt").exists()
    assert (root / "round-999.pt").exists() and (root / "round-03.pt.new").exists()


def test_retention_resolves_cwd_relative_canonical_best_pointer(tmp_path):
    import os

    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    (root / "best_dev_checkpoint.json").write_text(
        json.dumps({"best_checkpoint": os.path.relpath(root / "round-01.pt")})
    )
    retain_checkpoint_payloads(
        root, archives, [root / "round-02.pt"], keep_last=2, minimum_free_bytes=0
    )
    assert (root / "round-01.pt").exists() and (root / "round-02.pt").exists()


def test_retention_actual_runner_prelude_preserves_resume_and_operator_pin(tmp_path):
    import ast
    import fcntl
    import shutil
    import subprocess
    import tarfile
    import time
    from types import SimpleNamespace

    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    module = ast.parse((Path(__file__).parent / "run_fused_closed_loop.py").read_text())
    main = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    loop = next(
        n
        for n in main.body
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "rnd"
    )
    prelude = []
    for statement in loop.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "plan" for t in statement.targets
        ):
            break
        prelude.append(statement)
    args = SimpleNamespace(
        output=root,
        campaign_prior_output=[],
        resume=root / "round-02.pt",
        best_checkpoint=None,
        checkpoint_archive_dir=archives,
        checkpoint_keep_last=2,
        retention_s3_prefix="s3://veox-jopedime-learning-use2/experiments/fixture/",
        retain_checkpoint=[Path("round-03.pt")],
    )

    def fixture_retention(*positional, **kwargs):
        # Preserve the actual archive/prune path with a tiny fixture disk budget.
        return retain_checkpoint_payloads(*positional, **kwargs, minimum_free_bytes=0)

    with (tmp_path / "test-qualification.lock").open("a") as lock:
        env = {
            "args": args,
            "rnd": 7,
            "start_round": 7,
            "reporter": None,
            "fcntl": fcntl,
            "lock": lock,
            "time": time,
            "Path": Path,
            "json": json,
            "shutil": shutil,
            "subprocess": subprocess,
            "tarfile": tarfile,
            "last_retention_attempt": None,
            "incumbent": SimpleNamespace(best_checkpoint=str(root / "round-01.pt")),
            "margin_policy": {},
            "retain_checkpoint_payloads": fixture_retention,
        }
        exec(  # noqa: S102 -- execute only the checked-in runner prelude against CPU fixtures.
            compile(
                ast.Module(body=prelude, type_ignores=[]),
                "actual-runner-prelude",
                "exec",
            ),
            env,
        )
    assert env["retention"]["pruned_files"] > 0
    for n in [1, 2, 3, 5, 6]:
        assert (root / f"round-{n:02d}.pt").exists()
    assert not (root / "round-04.pt").exists()


@pytest.mark.parametrize("where", ["run", "prior", "symlink"])
def test_retention_rejects_archive_inside_any_guarded_root(tmp_path, where):
    from dev_incumbent import retain_checkpoint_payloads

    root, _ = retention_fixture(tmp_path)
    prior = tmp_path / "prior"
    prior.mkdir()
    if where == "symlink":
        target = tmp_path / "archive-link"
        target.symlink_to(root, target_is_directory=True)
    else:
        target = (root if where == "run" else prior) / "archive"
    with pytest.raises(ValueError, match="outside"):
        retain_checkpoint_payloads(
            root,
            target,
            [root / "round-02.pt"],
            guarded_roots=[prior],
            minimum_free_bytes=0,
        )
    assert (root / "round-03.pt").exists()


def test_retention_rejects_missing_resume_and_symlink_escape(tmp_path):
    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    with pytest.raises(ValueError, match="protected checkpoint"):
        retain_checkpoint_payloads(
            root, archives, [root / "missing.pt"], minimum_free_bytes=0
        )
    outside = tmp_path / "foreign.pt"
    outside.write_bytes(b"do not touch")
    (root / "round-03.pt").unlink()
    (root / "round-03.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        retain_checkpoint_payloads(
            root, archives, [root / "round-02.pt"], keep_last=2, minimum_free_bytes=0
        )
    assert outside.read_bytes() == b"do not touch"


def test_retention_corrupt_archive_never_prunes_source(tmp_path, monkeypatch):
    import tarfile

    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(
        b"broken"
    )  # Same length as root-3; header/size alone cannot catch it.
    original = tarfile.TarFile.add

    def corrupt(tar, name, **kwargs):
        return original(
            tar, bad if Path(name) == root / "round-03.pt" else name, **kwargs
        )

    monkeypatch.setattr(tarfile.TarFile, "add", corrupt)
    with pytest.raises(ValueError, match="hash mismatch"):
        retain_checkpoint_payloads(
            root, archives, [root / "round-02.pt"], keep_last=2, minimum_free_bytes=0
        )
    assert (root / "round-03.pt").read_bytes() == b"root-3"
    assert (root / "rejected/round-03.pt").exists()


def test_retention_pointer_change_at_delete_boundary_aborts(tmp_path, monkeypatch):
    import subprocess

    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)

    def move_pointer(command, **kwargs):
        (root / "best_dev_checkpoint.json").write_text(
            json.dumps({"best_checkpoint": str(root / "round-03.pt")})
        )
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", move_pointer)
    with pytest.raises(ValueError, match="pointer changed"):
        retain_checkpoint_payloads(
            root, archives, [root / "round-02.pt"], keep_last=2, minimum_free_bytes=0
        )
    for n in range(1, 7):
        assert (root / f"round-{n:02d}.pt").exists()


def test_retention_active_reader_blocks_pruning(tmp_path):
    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    with (root / "round-03.pt").open("rb"), pytest.raises(ValueError, match="consumer"):
        retain_checkpoint_payloads(
            root,
            archives,
            [root / "round-02.pt"],
            keep_last=2,
            minimum_free_bytes=0,
        )
    assert (root / "round-03.pt").exists()


def test_retention_free_space_refusal_preserves_sources(tmp_path, monkeypatch):
    import shutil

    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(100, 100, 0)
    )
    with pytest.raises(ValueError, match="headroom"):
        retain_checkpoint_payloads(
            root, archives, [root / "round-02.pt"], keep_last=2, minimum_free_bytes=1
        )
    assert (root / "round-03.pt").exists() and not archives.exists()


def test_retention_s3_failure_cannot_delete_or_rotate(tmp_path, monkeypatch):
    import dev_incumbent

    root, archives = retention_fixture(tmp_path)
    before = {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }

    def fail(*args, **kwargs):
        raise RuntimeError("independent S3 readback failed")

    monkeypatch.setattr(dev_incumbent, "publish_checkpoint_archive", fail)
    with pytest.raises(RuntimeError, match="S3"):
        dev_incumbent.retain_checkpoint_payloads(
            root,
            archives,
            [root / "round-02.pt"],
            keep_last=2,
            minimum_free_bytes=0,
            rotate_bytes=1,
        )
    assert before == {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_retention_t533_keeps_accepted_lineage_and_rotates_only_verified_logs(tmp_path):
    from dev_incumbent import retain_checkpoint_payloads

    root, archives = retention_fixture(tmp_path)
    for n in range(1, 7):
        (root / f"round-{n:02d}.json").write_text(
            json.dumps({"round": n, "adapter_admission": {"accepted": n in (1, 3)}})
        )
    (root / "closed.log").write_bytes(b"log receipt")
    result = retain_checkpoint_payloads(
        root,
        archives,
        [root / "round-02.pt"],
        keep_last=2,
        minimum_free_bytes=0,
        scored_round=6,
        rotate_bytes=1,
    )
    for n in (1, 2, 3, 6):
        assert (root / f"round-{n:02d}.pt").exists()
    assert not (root / "round-05.pt").exists()
    assert not list((root / "rejected").glob("*.pt"))
    assert (root / "summary.jsonl").read_bytes() == b"" and (
        root / "closed.log"
    ).read_bytes() == b""
    assert (root / "promotions.jsonl").read_text() == "{}\n"
    assert (root / "round-03-scores.jsonl").exists()
    assert result["s3_archive"] and result["s3_manifest"]


def setup():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    opt.zero_grad()
    return model, opt


def test_worse_restores_both_weights_and_optimizer():
    model, opt = setup()
    saved = copy.deepcopy(model.state_dict())
    saved_opt = copy.deepcopy(opt.state_dict())
    gate = DevIncumbent(model, opt, {"mean": 2.2678}, "fixed-dev", "round-40.pt", 40)
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    event = gate.decide(model, opt, {"mean": 1.148}, "fixed-dev", "round-41.pt", 41)
    assert not event["accepted"] and event["best_checkpoint"] == "round-40.pt"
    for name, v in model.state_dict().items():
        assert torch.equal(v, saved[name])
    for key, state in opt.state_dict()["state"].items():
        for name, v in state.items():
            assert v == saved_opt["state"][key][name]


def test_improvement_kept_and_later_regression_restores_it():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with torch.no_grad():
        model.weight.add_(1)
    assert gate.decide(model, opt, {"mean": 3}, "fixed-dev", "best.pt", 2)["accepted"]
    best = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert not gate.decide(model, opt, {"mean": 2.5}, "fixed-dev", "bad.pt", 3)[
        "accepted"
    ]
    assert torch.equal(model.weight, best) and gate.best_checkpoint == "best.pt"


def test_tie_advances_training_without_lowering_dev_utility():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    best = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    event = gate.decide(model, opt, {"mean": 2}, "fixed-dev", "tie.pt", 2)
    assert event["accepted"] and event["action"] == "accept_tie"
    assert not torch.equal(model.weight, best)
    assert json.dumps(event, separators=(",", ":")) == (
        '{"round":2,"evaluation_identity":"fixed-dev","candidate_utility":2.0,'
        '"previous_best_utility":2.0,"accepted":true,"action":"accept_tie",'
        '"best_utility":2.0,"best_round":2,"best_checkpoint":"tie.pt"}'
    )


def test_unchanged_cli_defaults_to_legacy_admission_policy():
    parser = add_admission_arguments(argparse.ArgumentParser())
    args = parser.parse_args([])
    assert args.strict_adapter_admission is False
    assert args.adapter_min_gain is None
    assert validate_admission_options(
        args.strict_adapter_admission, args.adapter_min_gain
    ) is None


@pytest.mark.parametrize(
    ("candidate", "accepted"),
    [
        (2.0, False),
        (2.125, False),
        (2.25, False),
        (math.nextafter(2.25, math.inf), True),
    ],
)
def test_opt_in_strict_margin_boundary_and_adamw_restore(candidate, accepted):
    model, opt = setup()
    before_model = copy.deepcopy(model.state_dict())
    before_opt = copy.deepcopy(opt.state_dict())
    gate = DevIncumbent(
        model,
        opt,
        {"mean": 2},
        "fixed-dev",
        "old.pt",
        1,
        strict_admission=True,
        min_gain=0.25,
    )
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    proposed_model = copy.deepcopy(model.state_dict())
    proposed_opt = copy.deepcopy(opt.state_dict())
    event = gate.decide(model, opt, {"mean": candidate}, "fixed-dev", "new.pt", 2)

    assert event["accepted"] is accepted
    assert event["admission_policy"] == "strict-margin/v1"
    assert event["min_gain"] == 0.25
    assert event["action"] == (
        "accept_improvement" if accepted else "restore_best_model_and_optimizer"
    )
    expected_model, expected_opt = (
        (proposed_model, proposed_opt) if accepted else (before_model, before_opt)
    )
    assert all(
        torch.equal(value, expected_model[name])
        for name, value in model.state_dict().items()
    )
    actual_opt = opt.state_dict()
    assert actual_opt["param_groups"] == expected_opt["param_groups"]
    for key, state in actual_opt["state"].items():
        for name, value in state.items():
            assert torch.equal(value, expected_opt["state"][key][name])
    assert gate.best_checkpoint == ("new.pt" if accepted else "old.pt")
    assert gate.best_round == (2 if accepted else 1)


def test_strict_admission_requires_explicit_valid_margin():
    with pytest.raises(ValueError, match="requires --adapter-min-gain"):
        validate_admission_options(True, None)
    with pytest.raises(ValueError, match="requires --strict-adapter-admission"):
        validate_admission_options(False, 0.0)
    for margin in (-1, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            validate_admission_options(True, margin)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_invalid_score_fails_closed(score):
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with pytest.raises(ValueError):
        gate.decide(model, opt, {"mean": score}, "fixed-dev", "bad.pt", 2)


def test_identity_change_refused():
    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    with pytest.raises(ValueError):
        gate.decide(model, opt, {"mean": 3}, "different-dev", "bad.pt", 2)


def test_round_step_batch_stream_advances_and_replays_without_best_round():
    from dev_incumbent import batch_seed

    seeds = [batch_seed(42, rnd, step) for rnd in range(926, 973) for step in range(2)]
    assert len(set(seeds)) == 94
    assert seeds == [
        batch_seed(42, rnd, step) for rnd in range(926, 973) for step in range(2)
    ]
    assert batch_seed(43, 926, 0) != seeds[0]


def test_rejected_parameter_bytes_never_reach_evaluator_again_after_resume():
    import hashlib

    from dev_incumbent import CandidateNovelty, RejectedCandidateError

    model, opt = setup()
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "old.pt", 1)
    novelty = CandidateNovelty("fixed-dev", [])
    with torch.no_grad():
        model.weight.add_(1)
    candidate = copy.deepcopy(model.state_dict())
    parameter_sha = hashlib.sha256(model.weight.detach().numpy().tobytes()).hexdigest()
    calls = []

    def evaluate():
        calls.append(parameter_sha)
        return {"mean": 1}

    utility = novelty.evaluate(parameter_sha, evaluate)
    decision = gate.decide(model, opt, utility, "fixed-dev", "rejected.pt", 2)
    decision["candidate_parameter_sha256"] = parameter_sha
    novelty.record(decision)
    # Restore/rebuild the exact rejected candidate; journal-only resume must refuse it.
    model.load_state_dict(candidate)
    resumed = CandidateNovelty("fixed-dev", [decision])
    with pytest.raises(RejectedCandidateError, match="first rejected round 2"):
        resumed.evaluate(parameter_sha, evaluate)
    assert calls == [parameter_sha]


def test_legacy_rejection_records_and_identity_are_respected():
    from dev_incumbent import CandidateNovelty, RejectedCandidateError

    legacy = {
        "candidate_parameter_sha256": "a" * 64,
        "adapter_admission": {
            "evaluation_identity": "fixed-dev",
            "accepted": False,
            "round": 2,
        },
    }
    later = {
        **legacy["adapter_admission"],
        "round": 3,
        "candidate_parameter_sha256": "a" * 64,
    }
    novelty = CandidateNovelty("fixed-dev", [later, legacy])
    with pytest.raises(RejectedCandidateError, match="first rejected round 2"):
        novelty.evaluate(
            "a" * 64, lambda: pytest.fail("must not evaluate rejected bytes")
        )
    assert CandidateNovelty("other-dev", [legacy]).evaluate("a" * 64, lambda: 7) == 7
    novelty.record(
        {
            "evaluation_identity": "fixed-dev",
            "accepted": True,
            "round": 3,
            "candidate_parameter_sha256": "b" * 64,
        }
    )
    assert novelty.evaluate("b" * 64, lambda: 8) == 8


def test_repeated_rejections_never_mutate_saved_optimizer_state():
    model, opt = setup()
    original = copy.deepcopy(opt.state_dict())
    gate = DevIncumbent(model, opt, {"mean": 2}, "fixed-dev", "best.pt", 1)
    for rnd in range(2, 5):
        opt.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        opt.step()
        gate.decide(model, opt, {"mean": 1}, "fixed-dev", f"bad-{rnd}.pt", rnd)
        for key, state in original["state"].items():
            for name, value in state.items():
                assert torch.equal(opt.state_dict()["state"][key][name], value)
                assert torch.equal(gate.optimizer_state["state"][key][name], value)
        assert opt.param_groups[0]["lr"] == original["param_groups"][0]["lr"]


def test_rejection_escape_changes_lr_steps_seed_and_enables_gp_after_k():
    from dev_incumbent import proposal_plan

    history = [
        {
            "round": n,
            "adapter_admission": {"accepted": False},
            "candidate_dev_utility": {"mean": 1, "selected_programs": ["same"]},
        }
        for n in range(1, 4)
    ]
    plan = proposal_plan(42, 4, history)
    assert plan["steps"] == 1 and plan["lr_factor"] == 0.25
    assert plan["same_candidate_streak"] == 3 and plan["gp_exploration"]
    assert plan["gp_max_evaluated_per_context"] == 4
    history.append(
        {
            "round": 4,
            "adapter_admission": {"accepted": False},
            "candidate_dev_utility": {"mean": 1, "selected_programs": ["same"]},
            "proposal_plan": plan,
        }
    )
    retry = proposal_plan(42, 5, history)
    assert (
        retry["lr_factor"] < plan["lr_factor"]
        and retry["round_seed"] != plan["round_seed"]
    )


def test_acceptance_resets_retry_but_round_seed_keeps_changing():
    from dev_incumbent import proposal_plan

    history = [{"round": 3, "adapter_admission": {"accepted": True}}]
    a = proposal_plan(42, 4, history)
    b = proposal_plan(42, 5, history)
    assert a["retry_level"] == 0 and a["steps"] == 2 and not a["gp_exploration"]
    assert a["round_seed"] != b["round_seed"]


def test_stalled_gp_selection_is_reproducible_and_reserves_exploration():
    from dev_incumbent import gp_selection_indices

    costs = list(range(16))
    a = gp_selection_indices(costs, 42, True)
    assert a[:2] == [0, 1] and len(a) == len(set(a)) == 4
    assert a == gp_selection_indices(costs, 42, True)
    assert a != gp_selection_indices(costs, 43, True)
    assert gp_selection_indices(costs, 42, False) == [0, 1, 2, 3]


def test_margin_projection_preserves_ranking_and_allows_real_progress():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([3.5e-5, 0.0]))
    frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0)
    for _ in range(3):
        opt.zero_grad()
        loss = p[0] - p[1]
        loss.backward()
        opt.step()
        assert p[0] < 0  # reproduces raw winner flip before projection
        report = project_score_margins([p], [("near_tie", lambda: p[0], 1.75e-5)])
        assert report["satisfied"] and p[0] >= 1.75e-5
    assert p[1] > 0.002 and frozen.item() == 7
    assert int(opt.state[p]["step"]) == 3


def test_projection_does_not_move_a_satisfied_model():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    before = p.detach().clone()
    report = project_score_margins([p], [("safe", lambda: p[0], 0.01)])
    assert report["satisfied"] and not report["corrections"]
    assert torch.equal(p, before)


def test_projection_refuses_frozen_parameters():
    from dev_incumbent import project_score_margins

    p = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)
    with pytest.raises(ValueError):
        project_score_margins([p], [])


def gp_drought_history(count):
    return [
        {
            "round": n,
            "adapter_admission": {"accepted": True},
            "candidate_dev_utility": {"mean": 2.267816},
            "promotions": 0,
            "GP": {"evaluated": 64},
        }
        for n in range(1, count + 1)
    ]


def test_gp_drought_explores_despite_accepted_adapter_ties():
    from dev_incumbent import proposal_plan

    plan = proposal_plan(42, 19, gp_drought_history(18))
    assert plan["gp_exploration"]
    assert plan["steps"] == 2 and plan["lr_factor"] == 1
    assert plan["gp_generated_per_context"] == 16
    assert plan["gp_max_evaluated_per_context"] == 4


@pytest.mark.parametrize(
    ("count", "explore"),
    [
        (17, False),
        (18, True),
        (19, True),
        (20, True),
        (21, False),
        (23, False),
        (24, True),
    ],
)
def test_gp_drought_bursts_and_cooldown_are_bounded(count, explore):
    from dev_incumbent import proposal_plan

    history = gp_drought_history(count)
    for index, row in enumerate(history):
        row["proposal_plan"] = proposal_plan(42, row["round"], history[:index])
    plan = proposal_plan(42, count + 1, history)
    assert plan["gp_exploration"] is explore
    assert plan["gp_drought_policy"]["completed_zero_promotion_rounds"] == count
    assert plan["gp_max_evaluated_per_context"] == 4
    assert plan["gp_generated_per_context"] == (16 if explore else 8)


@pytest.mark.parametrize(
    "interruption", ["promotion", "missing", "no_evaluations", "gap"]
)
def test_gp_drought_resets_on_promotion_or_unknown_outcome(interruption):
    from dev_incumbent import proposal_plan

    history = gp_drought_history(18)
    if interruption == "promotion":
        history[-1]["promotions"] = 1
    elif interruption == "missing":
        del history[-1]["promotions"]
    elif interruption == "no_evaluations":
        history[-1]["GP"]["evaluated"] = 0
    else:
        history.pop()
    plan = proposal_plan(42, 19, history)
    assert not plan["gp_exploration"]
    assert plan["gp_drought_policy"]["completed_zero_promotion_rounds"] == 0


def test_first_gp_policy_installation_does_not_invent_past_exploration_rounds():
    from dev_incumbent import proposal_plan

    plan = proposal_plan(42, 24, gp_drought_history(23))
    assert plan["gp_exploration"] and plan["gp_drought_policy"]["phase"] == 0


def test_consecutive_restored_candidates_have_distinct_parameter_shas():
    import copy
    import hashlib

    import torch
    from dev_incumbent import warm_restart_adapter

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    baseline = copy.deepcopy(model.state_dict())
    optimizer = copy.deepcopy(opt.state_dict())
    candidates = []
    for rnd in (8, 9):
        model.load_state_dict(baseline)
        opt.load_state_dict(copy.deepcopy(optimizer))
        warm_restart_adapter(model, opt, 104729 * rnd, rnd)
        opt.zero_grad()
        model(torch.ones((4, 3))).square().mean().backward()
        opt.step()
        candidates.append(
            hashlib.sha256(
                b"".join(
                    p.detach().numpy().tobytes()
                    for p in model.parameters()
                    if p.requires_grad
                )
            ).hexdigest()
        )
        for name in ("0.weight", "0.bias"):
            assert torch.equal(model.state_dict()[name], baseline[name])
    assert candidates[0] != candidates[1]


@pytest.mark.parametrize("upload_fails", [False, True])
def test_retention_actual_after_scoring_hook_archives_each_round(
    tmp_path, upload_fails
):
    import ast
    import subprocess
    import tarfile
    from types import SimpleNamespace

    import dev_incumbent
    from automatic_promotion import write_tree_shard

    root, archives = retention_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    module = ast.parse((Path(__file__).parent / "run_fused_closed_loop.py").read_text())
    main = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    loop = next(
        n
        for n in main.body
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "rnd"
    )
    hook = next(
        n
        for n in loop.body
        if isinstance(n, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "write_tree_shard"
            for call in ast.walk(n)
        )
    )
    calls = []

    def publish(path, prefix):
        calls.append("tree-upload")
        if upload_fails:
            raise RuntimeError("injected tree upload failure")
        return dev_incumbent.publish_checkpoint_archive(path, prefix)

    def retain(*args, **kwargs):
        calls.append(("cleanup", kwargs["scored_round"]))
        return dev_incumbent.retain_checkpoint_payloads(
            *args, **kwargs, minimum_free_bytes=0
        )

    env = {
        "args": SimpleNamespace(
            output=root,
            campaign_prior_output=[],
            resume=root / "round-02.pt",
            best_checkpoint=None,
            checkpoint_archive_dir=archives,
            checkpoint_keep_last=2,
            retain_checkpoint=[],
            retention_s3_prefix="s3://veox-jopedime-learning-use2/experiments/fixture/",
        ),
        "rnd": 6,
        "tree_records": [{"dsl": "(add 1 2)", "fitness": 1.0, "pool_accepted": False}],
        "incumbent": SimpleNamespace(best_checkpoint=str(root / "round-01.pt")),
        "margin_policy": {},
        "Path": Path,
        "write_tree_shard": write_tree_shard,
        "publish_checkpoint_archive": publish,
        "retain_checkpoint_payloads": retain,
        "subprocess": subprocess,
        "tarfile": tarfile,
    }
    exec(  # noqa: S102 -- checked-in maintenance hook against CPU fixtures.
        compile(
            ast.Module(body=[hook], type_ignores=[]), "actual-post-scoring-hook", "exec"
        ),
        env,
    )
    if upload_fails:
        assert calls == ["tree-upload"]
        assert env["cleanup"]["pruned_files"] is None
        assert all(Path(p).read_bytes() == body for p, body in before.items())
    else:
        assert calls == ["tree-upload", ("cleanup", 6)]
        assert env["cleanup"]["pruned_files"] > 0
        for n in [1, 2, 5, 6]:
            assert (root / f"round-{n:02d}.pt").exists()
        assert not list((root / "rejected").glob("*.pt"))


def test_restart_scales_do_not_repeat_the_four_round_cycle():
    import copy

    from dev_incumbent import warm_restart_adapter

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    baseline = copy.deepcopy(model.state_dict())
    scales = []
    for rnd in range(1341, 1373):
        model.load_state_dict(baseline)
        result = warm_restart_adapter(model, opt, 42 + rnd * 104729, rnd)
        assert 2.5e-5 <= result["lr"] <= 3e-4
        assert 1e-5 <= result["sigma"] <= 3e-4
        scales.append((result["lr"], result["sigma"]))
    assert len({x[0] for x in scales}) == 32
    assert len({x[1] for x in scales}) == 32
    assert all(a != b for a, b in zip(scales, scales[4:]))


def test_restart_replays_exactly_preserves_adam_and_global_rng():
    import copy

    from dev_incumbent import warm_restart_adapter

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model(torch.ones((2, 4))).square().mean().backward()
    opt.step()
    baseline = copy.deepcopy(model.state_dict())
    optimizer = copy.deepcopy(opt.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    outputs = []
    for seed, rnd in [(731, 1361), (731, 1361), (732, 1361), (731, 1365)]:
        model.load_state_dict(baseline)
        opt.load_state_dict(copy.deepcopy(optimizer))
        result = warm_restart_adapter(model, opt, seed, rnd)
        for name in ["0.weight", "0.bias"]:
            assert torch.equal(model.state_dict()[name], baseline[name])
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                assert torch.equal(opt.state_dict()["state"][key][name], value)
        outputs.append(
            (
                result,
                torch.cat(
                    [
                        p.detach().flatten()
                        for p in model.parameters()
                        if p.requires_grad
                    ]
                ).clone(),
            )
        )
    assert outputs[0][0] == outputs[1][0] and torch.equal(outputs[0][1], outputs[1][1])
    assert (
        outputs[0][0]["lr"] != outputs[2][0]["lr"]
        and outputs[0][0]["sigma"] != outputs[2][0]["sigma"]
    )
    assert (
        outputs[0][0]["lr"] != outputs[3][0]["lr"]
        and outputs[0][0]["sigma"] != outputs[3][0]["sigma"]
    )
    assert torch.equal(torch.random.get_rng_state(), global_rng)


@pytest.mark.parametrize("digest", [b"\0" * 32, b"\xff" * 32])
def test_restart_scale_endpoints_respect_previous_bounds(monkeypatch, digest):
    import hashlib

    from dev_incumbent import warm_restart_adapter

    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    class FixedHash:
        def digest(self):
            return digest

    monkeypatch.setattr(hashlib, "sha256", lambda *args: FixedHash())
    result = warm_restart_adapter(model, optimizer, 42, 1361)
    assert 1e-5 <= result["sigma"] <= 3e-4
    assert 2.5e-5 <= result["lr"] <= 3e-4
