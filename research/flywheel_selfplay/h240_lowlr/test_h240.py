"""Data-free H240 optimizer doses, horizon labels, paired sampler and boundary controls."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import analysis as stat
import monitor as mon
import runner
from treatment import (
    ARMS,
    LEARNING_RATES,
    UPDATES,
    arm_loss,
    make_optimizer,
    settings,
    validate_launch,
    validate_resume,
)

HERE = Path(__file__).resolve().parent


def rows():
    return [dict(context_sha256=f"c{i:03}", family_id=f"f{i:03}") for i in range(205)]


@pytest.mark.parametrize("arm", ARMS)
def test_exact_adamw_recipe_with_only_lr_and_horizon_changed(arm):
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    inactive = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    opt = make_optimizer(arm, [p, inactive])
    ref = torch.optim.AdamW(
        [q], lr=LEARNING_RATES[arm], betas=(0.9, 0.999), weight_decay=1e-4
    )
    assert type(opt) is torch.optim.AdamW and opt.param_groups[0]["eps"] == 1e-8
    for step in range(1, UPDATES[arm] + 1):
        p.grad = torch.tensor([step, -step / 2], dtype=p.dtype)
        q.grad = p.grad.clone()
        inactive.grad = None
        torch.nn.utils.clip_grad_norm_([p, inactive], 1.0)
        torch.nn.utils.clip_grad_norm_([q], 1.0)
        opt.step()
        ref.step()
        assert torch.equal(p, q)
        assert settings(arm, step) == (LEARNING_RATES[arm], 0.9, 0.999, 1e-4)
    assert (
        torch.equal(inactive, torch.tensor([3.0], dtype=p.dtype))
        and inactive not in opt.state
    )
    with pytest.raises(ValueError):
        settings(arm, UPDATES[arm] + 1)
    with pytest.raises(ValueError):
        settings(arm, 0)


def test_all_arms_use_identical_ctrl_loss_and_constant_schedules():
    sentinel = object()
    table = json.loads((HERE / "schedules.json").read_text())
    assert stat.ARM_UPDATES == UPDATES == dict(CONTROL=40, LR3E5=40, LR1E5_120=120)
    assert 15 * sum(UPDATES.values()) == 3000
    for arm in ARMS:
        assert arm_loss(arm, lambda *args: sentinel, None, None, None) is sentinel
        assert table["candidate_steps"][arm] == list(stat.ARM_STEPS[arm])
        assert len(table["arms"][arm]) == UPDATES[arm]
        for step, row in enumerate(table["arms"][arm], 1):
            assert row == dict(
                step=step,
                lr=LEARNING_RATES[arm],
                beta1=0.9,
                beta2=0.999,
                weight_decay=1e-4,
            )


def test_120_batch_stream_preserves_exact_paired_first40_and_extends_sampler():
    ids = [r["context_sha256"] for r in rows()[:123]]
    orders = {c: np.arange(48) for c in ids}
    short = stat.batch_plan(ids, orders, 240101)
    long = stat.batch_plan(list(reversed(ids)), orders, 240101, updates=120)
    assert len(short) == 40 and len(long) == 120
    for i, (stream, cids, indexes) in enumerate(long):
        assert stream == stat.stream_seed(240101, i + 1)
        assert len(cids) == 8 and set(cids) <= set(ids)
        for ix in indexes:
            assert len(set(ix)) == 24 and np.array_equal(ix[:4], np.arange(4))
        if i < 40:
            a, b, c = short[i]
            assert (
                a == stream
                and b == cids
                and all(np.array_equal(x, y) for x, y in zip(c, indexes))
            )


def test_arm_specific_long_selector_returns_actual_step30_not10():
    source = rows()[:41]
    panel = np.zeros((41, 5))
    panel[:, 1:] = 0.005
    assert stat.choose(source, panel, 1)["selected_step"] == 10
    assert (
        stat.choose(source, panel, 1, steps=stat.ARM_STEPS["LR1E5_120"])[
            "selected_step"
        ]
        == 30
    )
    panel[:] = 0
    assert (
        stat.choose(source, panel, 1, steps=stat.ARM_STEPS["LR1E5_120"])[
            "selected_step"
        ]
        == 0
    )
    for invalid in [(0, 30, 30, 90, 120), (10, 20, 30, 40, 50), (0, 30, 20, 90, 120)]:
        with pytest.raises(ValueError):
            stat.choose(source, panel, 1, steps=invalid)


def test_runner_uses_per_arm_budget_candidates_and_preserves_native_smoke_order():
    s = (HERE / "runner.py").read_text()
    assert (
        "batch_cache[: stat.ARM_UPDATES[arm]]" in s
        or "batch_cache[:stat.ARM_UPDATES[arm]]" in s
    )
    assert "if step in stat.ARM_STEPS[arm]" in s
    assert "steps=stat.ARM_STEPS[arm]" in s
    assert s.index("baseline, costs = engine.evaluate(val, native=True") < s.index(
        "for arm in stat.ARMS:"
    )
    assert s.index('out / "smoke.json"') < s.index("engine.load(train)")
    assert "optimizer_mode" not in s and "1.0, error_if_nonfinite=True" in s


def sample(utc=0.0, available=30 * 2**30, load5=100.0):
    return dict(
        utc=utc,
        available=available,
        load5=load5,
        rss=4 * 2**30 - 1,
        threads=4,
        free=300_000_000_000,
    )


def test_cooperative_sustained_memory_guard_and_instant_floor():
    g = mon.Guard()
    assert mon.admission_refusals(sample()) == []
    assert mon.admission_refusals(sample(available=30 * 2**30 - 1))
    assert mon.admission_refusals(sample(load5=100.1))
    assert g.update(sample(0, 19 * 2**30)) == ([], True)
    assert g.update(sample(1, 11 * 2**30)) == ([], True)
    assert g.update(sample(30.99, 11 * 2**30)) == ([], True)
    assert "MemAvailable<12GiB sustained30s" in g.update(sample(31, 11 * 2**30))[0]
    g = mon.Guard()
    assert g.update(sample(0, 11 * 2**30))[0] == []
    assert g.update(sample(29, 12 * 2**30))[0] == []
    assert (
        g.update(sample(30, 11 * 2**30))[0] == []
    )  # recovery resets the sustained clock
    assert "MemAvailable<6GiB" in g.update(sample(30.1, 6 * 2**30 - 1))[0]
    g = mon.Guard()
    g.update(sample(0, 19 * 2**30))
    assert g.update(sample(1, 25 * 2**30))[1]
    assert g.update(sample(30.9, 25 * 2**30))[1]
    assert not g.update(sample(31, 25 * 2**30))[1]
    g = mon.Guard()
    g.update(sample(0, 19 * 2**30))
    g.update(sample(1, 25 * 2**30))
    g.update(sample(20, 25 * 2**30 - 1))
    assert g.update(sample(50, 25 * 2**30))[1]
    assert not g.update(sample(80, 25 * 2**30))[1]
    assert mon.COMPUTE_STOP == 1793691000.0


def test_load_hysteresis_and_other_hard_limits_unchanged():
    g = mon.Guard()
    assert not g.update(sample(load5=108))[1]
    assert g.update(sample(1, load5=108.1))[1]
    assert g.update(sample(2, load5=100.1))[1]
    assert not g.update(sample(3, load5=100))[1]
    for key, value in dict(
        rss=4 * 2**30, threads=5, free=299_999_999_999, utc=mon.COMPUTE_STOP
    ).items():
        assert mon.violations(dict(sample(), **{key: value}))


def test_resume_is_bounded_and_binds_source_protocol_and_directory(tmp_path):
    prior = tmp_path / "attempt-001"
    output = tmp_path / "attempt-002"
    previous = dict(
        resume_number=0,
        source_commit="s",
        manifest_sha256="m",
        fold_manifest_sha256="f",
        prereg_sha256="p",
    )
    launch = dict(previous, resume_number=1, resume_from=str(prior.resolve()))
    validate_resume(previous, launch, prior, output)
    validate_resume(
        dict(previous, resume_number=1), dict(launch, resume_number=2), prior, output
    )
    for changes in [
        dict(resume_number=True),
        dict(resume_number=3),
        dict(resume_number=0),
        dict(source_commit="x"),
        dict(manifest_sha256="x"),
        dict(fold_manifest_sha256="x"),
        dict(prereg_sha256="x"),
        dict(resume_from="elsewhere"),
    ]:
        with pytest.raises(RuntimeError):
            validate_resume(previous, dict(launch, **changes), prior, output)


def test_cooperate_waits_without_signals_and_logs_actual_safe_point(
    tmp_path, monkeypatch
):
    import threading
    import time

    marker = tmp_path / "pause.json"
    marker.write_text("{}")
    events = tmp_path / "events.jsonl"
    monkeypatch.setattr(runner, "PAUSE", marker)
    monkeypatch.setattr(runner, "COOP_EVENTS", events)
    thread = threading.Thread(target=lambda: (time.sleep(0.02), marker.unlink()))
    thread.start()
    runner.cooperate()
    thread.join()
    assert [json.loads(line)["action"] for line in events.read_text().splitlines()] == [
        "worker_paused_at_safe_point",
        "worker_resumed_same_state",
    ]
    assert "SIGSTOP" not in (HERE / "monitor.py").read_text().replace(
        "No SIGSTOP/SIGCONT is used.", ""
    )


def test_immutable_runtime_releases_namespace_and_launch_is_exact_head_bound():
    import sys

    rt = runner.runtime(json.loads((HERE / "manifest.json").read_text()))
    assert rt.Audit.__name__ == "Audit" and rt.DOPE.startswith("c51a56d0")
    assert not [n for n in sys.modules if n == "research" or n.startswith("research.")]
    launch = dict(
        manifest_sha256="m",
        source_commit="h",
        prereg_sha256="p",
        fold_manifest_sha256="f",
        review_id=123,
        prereg_pr=456,
        merged_source_verified=True,
    )
    validate_launch(launch, "h", "m", "p", "f")
    for key, value in dict(
        source_commit="x",
        manifest_sha256="x",
        prereg_sha256="x",
        fold_manifest_sha256="x",
        review_id=True,
        prereg_pr=0,
        merged_source_verified=False,
    ).items():
        with pytest.raises(RuntimeError):
            validate_launch(
                dict(launch, **{key: value}), "h", "m", "p", "f", smoke=True
            )


def test_family_folds_fit_exclusion_and_exact_assessment_cover():
    source = rows()
    cover = []
    for outer in range(5):
        fit, select, assess = stat.partition(source, outer, stat.SEEDS[0])
        groups = [{r["family_id"] for r in rs} for rs in [fit, select, assess]]
        assert [len(g) for g in groups] == [123, 41, 41]
        assert not (
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
        cover += list(groups[2])
    assert len(cover) == len(set(cover)) == 205
    fit, _, _ = stat.partition(source, 0, stat.SEEDS[0])
    assert (
        len({json.dumps(stat.folds(source, s)) for s in stat.SEEDS}) == 3
    )  # one fresh partition per seed
    ids = [r["context_sha256"] for r in fit]
    orders = {c: np.arange(48) for c in ids}
    first = stat.batch_plan(ids, orders, 240101)
    other = stat.batch_plan(list(reversed(ids)), orders, 240101)
    assert len(first) == 40
    for (s, c, ix), (s2, c2, ix2) in zip(first, other):
        assert s == s2 and c == c2 and len(c) == 8 and set(c) <= set(ids)
        for a, b in zip(ix, ix2):
            assert np.array_equal(a, b) and len(set(a)) == 24


def test_strict_selector_reachability_ties_and_no_second_best_fallback():
    source = rows()[:41]
    panel = np.zeros((41, 5))
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:, 1] = 0.005
    panel[:, 2] = 0.005 + 5e-13
    assert stat.choose(source, panel, 1)["selected_step"] == 10
    panel[:, 1:] = 0.001
    assert stat.choose(source, panel, 1)["selected_step"] == 0
    panel[:] = 0
    panel[:, 1] = np.r_[-np.ones(20), np.ones(21)]
    panel[:, 2] = 0.005
    pick = stat.choose(source, panel, 1)
    assert (
        pick["best_step"] == 10 and pick["ci95"][0] < 0 and pick["selected_step"] == 0
    )


def test_median_family_bootstrap_holm_and_canonical_elo():
    source = [{"family_id": "big"}] * 9 + [{"family_id": "small"}]
    panel = np.zeros((3, 10, 4))
    panel[:, :9, 1:] = 1.0
    panel[:, 9, 1:] = -1.0
    s = stat.summarize(source, panel)
    assert s["contrasts"]["LR3E5_minus_r882"]["median_gain"] == 0.0
    assert np.allclose(s["median_elo"], 1500.0)
    panel[:] = 0
    panel[:, :, 2] = 0.005
    s = stat.summarize(source, panel)
    assert s["contrasts"]["LR3E5_minus_r882"]["advance"]
    assert s["contrasts"]["LR3E5_minus_r882"]["ci95"][0] > 0
    assert (
        "LR3E5_minus_CONTROL" in s["contrasts"]
        and "CONTROL_minus_r882" in s["contrasts"]
    )
    # hw_t2120 margin: a +0.002 gain is a tie with the margin, not an advance
    panel[:, :, 2] = 0.002
    assert not stat.summarize(source, panel)["contrasts"]["LR3E5_minus_r882"]["advance"]
    panel[:, :, 2] = 0.0021
    assert stat.summarize(source, panel)["contrasts"]["LR3E5_minus_r882"]["advance"]
    assert stat.ADVANCE_MARGIN == 0.002
    assert s["median_elo"][2] > 1500.0 and s["median_elo"][0] == 1500.0
    assert stat.holm({"a": 0.01, "b": 0.04, "c": 0.02}) == {
        "a": 0.03,
        "c": 0.04,
        "b": 0.04,
    }


def test_exact_utility_duplicates_not_equal_means_and_signed_zero():
    assert stat.duplicate_groups({"a": [1, -1], "b": [-1, 1]}) == []
    assert stat.duplicate_groups({"a": [-0.0, 1.0], "b": [0.0, 1.0]}) == [["a", "b"]]
    assert stat.duplicate_groups({"a": "hash1", "b": "hash1", "c": "hash2"}) == [
        ["a", "b"]
    ]
    with pytest.raises(ValueError):
        stat.duplicate_groups({"bad": [np.nan]})


def test_pseudoreplication_discloses_identical_predictions_despite_distinct_adapters():
    panel = np.zeros((3, 2, 4))
    cells = []
    for outer in range(5):
        for seed in stat.SEEDS:
            cells.append(
                dict(
                    outer_fold=outer,
                    seed=seed,
                    selector_panels={
                        a: [dict(values=[0.0, 0.0]) for _ in stat.STEPS]
                        for a in stat.ARMS
                    },
                    adapter_sha256={
                        a: {
                            "0": "r882",
                            str(stat.ARM_UPDATES[a]): f"{outer}:{seed}:{a}",
                        }
                        for a in stat.ARMS
                    },
                    decisions={a: {"selected_step": 0} for a in stat.ARMS},
                )
            )
    report = stat.pseudoreplication(panel, panel, panel, panel, cells)
    assert len(report["strict_OOF_identical"][0]) == 12  # 3 seeds x (r882 + 3 arms)
    assert (
        report["terminal_adapter_identical"] == []
        and report["all_arms_candidate_index_panels_identical_to_CONTROL"]
    )
    cells[0]["selector_panels"]["LR3E5"][1]["values"] = [1.0, 0.0]
    assert not stat.pseudoreplication(panel, panel, panel, panel, cells)[
        "all_arms_candidate_index_panels_identical_to_CONTROL"
    ]


def test_completed_checkpoints_skip_training_and_use_arm_specific_budgets():
    source = (HERE / "runner.py").read_text()
    assert (
        "if (outer, seed) in recovered:" in source
        or "if (outer,seed) in recovered:" in source
    )
    assert (
        'len(cell["traces"][a]) != stat.ARM_UPDATES[a]' in source
        or "len(cell['traces'][a])!=stat.ARM_UPDATES[a]" in source
    )
    assert "for step in stat.ARM_STEPS[arm][1:]" in source
    assert (
        source.index("if (outer, seed) in recovered:")
        < source.index("selbase, _ = engine.evaluate(select)")
        if "if (outer, seed) in recovered:" in source
        else True
    )
    assert "physical_new_optimizer_updates" in source
    assert "positive_seeds" in (HERE / "analysis.py").read_text()


def test_completed_cell_json_is_atomic_on_serialization_failure(tmp_path):
    target = tmp_path / "A0-240101.json"
    runner.save(target, {"complete": True})
    original = target.read_bytes()
    with pytest.raises(ValueError):
        runner.save(target, {"invalid": float("nan")})
    assert target.read_bytes() == original


def test_actual_resume_loader_copies_completed_cells_and_rejects_tampering(tmp_path):
    import ast
    from types import SimpleNamespace

    # Execute the actual recovery branch without constructing a model or reading study data.
    tree = ast.parse((HERE / "runner.py").read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    branch = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "args.resume_from"
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "validate_resume"
            for c in ast.walk(n)
        )
    )
    code = compile(
        ast.Module(body=branch.body, type_ignores=[]), "actual_resume_branch", "exec"
    )
    prior, output = tmp_path / "attempt-001", tmp_path / "attempt-002"
    prior.mkdir()
    output.mkdir()
    previous = dict(
        resume_number=0,
        source_commit="s",
        manifest_sha256="m",
        fold_manifest_sha256="f",
        prereg_sha256="p",
    )
    prior.with_suffix(".launch.json").write_text(json.dumps(previous))
    launch = dict(previous, resume_number=1, resume_from=str(prior.resolve()))
    rt = runner.runtime(json.loads((HERE / "manifest.json").read_text()))
    selection = prior / "A0-240101-selection.json"
    selection.write_text('{"locked":true}')
    cell = dict(
        outer_fold=0,
        seed=240101,
        traces={a: [0] * stat.ARM_UPDATES[a] for a in ARMS},
        selection_sha256=runner.sha(selection),
        adapter_sha256={a: {} for a in ARMS},
    )
    for arm in ARMS:
        for step in stat.ARM_STEPS[arm][1:]:
            tensor = torch.tensor([float(step)], dtype=torch.float32)
            np.savez_compressed(
                prior / f"A0-240101-{arm}-step{step}.npz", weight=tensor.numpy()
            )
            cell["adapter_sha256"][arm][str(step)] = rt.tensor_hash({"weight": tensor})
    (prior / "A0-240101.json").write_text(json.dumps(cell))
    env = dict(
        runner.__dict__,
        args=SimpleNamespace(resume_from=prior),
        out=output,
        launch=launch,
        rt=rt,
        recovered={},
    )
    exec(code, env)
    assert set(env["recovered"]) == {(0, 240101)}
    assert len(list(output.iterdir())) == 14
    assert all(
        p.read_bytes() == (prior / p.name).read_bytes() for p in output.iterdir()
    )
    bad_output = tmp_path / "attempt-003"
    bad_output.mkdir()
    selection.write_text('{"tampered":true}')
    with pytest.raises(RuntimeError, match="resume selection hash"):
        exec(code, dict(env, out=bad_output, recovered={}))
    assert not list(bad_output.iterdir())


def test_launcher_only_resumes_memory_stops_with_same_source_and_bound_count(tmp_path):
    import launcher

    prior = tmp_path / "attempt-001"
    prior.mkdir()
    previous = dict(
        source_commit="s",
        manifest_sha256="m",
        fold_manifest_sha256="f",
        prereg_sha256="p",
        resume_number=0,
    )
    prior.with_suffix(".launch.json").write_text(json.dumps(previous))
    terminal = prior.with_suffix(".resource_summary.json")
    terminal.write_text(
        json.dumps(
            dict(status="INCOMPLETE", reasons=["MemAvailable<12GiB sustained30s"])
        )
    )
    output = tmp_path / "attempt-002"
    path = output.with_suffix(".launch.json")
    launch = dict(
        previous,
        output=str(output),
        resume_number=1,
        resume_from=str(prior),
        merged_source_verified=True,
    )
    assert launcher.check_launch(launch, path) == (output, 1)
    for reasons in [[], ["aggregate RSS>=4GiB"], ["absolute compute stop 07:30Z"]]:
        terminal.write_text(json.dumps(dict(status="INCOMPLETE", reasons=reasons)))
        with pytest.raises(RuntimeError):
            launcher.check_launch(launch, path)
    terminal.write_text(
        json.dumps(dict(status="INCOMPLETE", reasons=["MemAvailable<6GiB"]))
    )
    with pytest.raises(RuntimeError):
        launcher.check_launch(dict(launch, source_commit="different"), path)
    with pytest.raises(RuntimeError):
        launcher.check_launch(dict(launch, resume_number=3), path)
    assert launcher.STOP == mon.COMPUTE_STOP
