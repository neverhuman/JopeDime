"""Data-free acceptance for H253's objective, control, FIT sampler, estimator and resource guards."""

from pathlib import Path
import json
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402
import treatment as tx  # noqa: E402
import monitor  # noqa: E402
import launcher  # noqa: E402


def ctrl(pred, fitness, gate):
    target = torch.where(gate, fitness.clamp(-4, 2), torch.full_like(fitness, 2))
    return (
        F.smooth_l1_loss(pred, target, reduction="none").mean(1)
        + 0.1
        * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)
    ).mean()


def fixture():
    rng = np.random.default_rng(19)
    return (
        torch.tensor(rng.normal(size=(8, 24)), dtype=torch.float64, requires_grad=True),
        torch.tensor(rng.normal(size=(8, 24)), dtype=torch.float64),
        torch.tensor(rng.uniform(size=(8, 24)) > 0.15),
        ["f0", "f0", "f1", "f1", "f2", "f2", "f3", "f3"],
    )


def rows():
    return [
        dict(family_id=f"f{i:03}", context_sha256=f"c{i:03}-{j}")
        for i in range(205)
        for j in range(1 + i % 3)
    ]


def test_candidate_loss_preserves_h170_kl_sum():
    pred, fitness, gate, _ = fixture()
    assert torch.allclose(
        tx.candidate_losses(pred, fitness, gate).mean(),
        ctrl(pred, fitness, gate),
        rtol=1e-14,
        atol=1e-14,
    )
    a = torch.autograd.grad(
        tx.candidate_losses(pred, fitness, gate).mean(), pred, retain_graph=True
    )[0]
    b = torch.autograd.grad(ctrl(pred, fitness, gate), pred)[0]
    assert torch.allclose(a, b, rtol=1e-13, atol=1e-14)


def test_control_is_exact_native_loss_and_gradient():
    p, f, g, families = fixture()
    result, diag = tx.arm_loss("CONTROL", ctrl, p, f, g, families, 1234)
    baseline = ctrl(p, f, g)
    assert torch.equal(result, baseline) and diag["beta"] == 0
    assert torch.equal(
        torch.autograd.grad(result, p, retain_graph=True)[0],
        torch.autograd.grad(baseline, p)[0],
    )


@pytest.mark.parametrize("arm,mult", [("VREX_B1", 1), ("VREX_B10", 10)])
def test_vrex_is_population_variance_with_live_gradient(arm, mult):
    p, f, g, families = fixture()
    objective, diag = tx.arm_loss(arm, ctrl, p, f, g, families, 2.5)
    contexts = tx.candidate_losses(p, f, g).mean(1)
    risks = contexts.reshape(4, 2).mean(1)
    expected = ctrl(p, f, g) + mult * 2.5 * ((risks - risks.mean()) ** 2).mean()
    assert torch.allclose(objective, expected, atol=1e-14, rtol=1e-14)
    assert diag["beta"] == mult * 2.5
    assert not torch.equal(
        torch.autograd.grad(objective, p, retain_graph=True)[0],
        torch.autograd.grad(ctrl(p, f, g), p)[0],
    )


def test_family_order_does_not_change_objective():
    p, f, g, families = fixture()
    ix = [7, 0, 4, 2, 5, 1, 6, 3]
    a, _ = tx.arm_loss("VREX_B1", ctrl, p, f, g, families, 3)
    b, _ = tx.arm_loss(
        "VREX_B1", ctrl, p[ix], f[ix], g[ix], [families[i] for i in ix], 3
    )
    assert torch.allclose(a, b)


def test_equal_family_risks_have_zero_penalty():
    p = torch.zeros(8, 24, requires_grad=True)
    f = torch.ones_like(p)
    g = torch.ones_like(p, dtype=torch.bool)
    objective, diag = tx.arm_loss(
        "VREX_B10", ctrl, p, f, g, ["a", "a", "b", "b", "c", "c", "d", "d"], 100
    )
    assert diag["family_variance"] == 0 and torch.equal(objective, ctrl(p, f, g))


@pytest.mark.parametrize(
    "families",
    [
        ["a"] * 8,
        ["a", "b", "c", "d", "e", "f", "g", "h"],
        ["a", "a", "a", "b", "c", "c", "d", "d"],
    ],
)
def test_invalid_family_structure_is_rejected(families):
    p, f, g, _ = fixture()
    with pytest.raises(ValueError):
        tx.arm_loss("VREX_B1", ctrl, p, f, g, families, 1)


def test_scale_uses_detached_median_and_fixed_zero_floor():
    assert tx.beta_scale([1, 2, 100])["beta_scale"] == 0.5
    assert tx.beta_scale([0, 0, 1])["beta_scale"] == 1e12
    for values in [[], [float("nan")], [float("inf")], [-1]]:
        with pytest.raises(ValueError):
            tx.beta_scale(values)


def test_control_adamw_matches_torch_and_excludes_foreign_tensor():
    a = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    b = torch.nn.Parameter(a.detach().clone())
    foreign = torch.nn.Parameter(torch.tensor([3.0]))
    ours = tx.make_optimizer("CONTROL", [a])
    reference = torch.optim.AdamW([b], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    for step in range(40):
        a.grad = b.grad = torch.tensor([0.3, -0.2])
        ours.step()
        reference.step()
    assert torch.equal(a, b) and foreign.item() == 3
    for arm in tx.ARMS:
        opt = tx.make_optimizer(arm, [a])
        assert {id(p) for group in opt.param_groups for p in group["params"]} == {id(a)}
        assert tx.settings(arm, 40)["lr"] == 1e-4


def test_grouped_folds_and_rotating_roles_are_disjoint():
    roster = rows()
    groups = stat.folds(roster, 253101)
    assert [len(x) for x in groups] == [41] * 5 and groups == stat.folds(roster, 253103)
    for outer in range(5):
        fit, select, assess = stat.partition(roster, outer, 253101)
        famsets = [{r["family_id"] for r in role} for role in [fit, select, assess]]
        assert [len(x) for x in famsets] == [123, 41, 41]
        assert not (
            famsets[0] & famsets[1]
            or famsets[0] & famsets[2]
            or famsets[1] & famsets[2]
        )


def test_sampler_is_paired_reproducible_and_fit_only():
    fit, select, assess = stat.partition(rows(), 2, 253101)
    orders = {r["context_sha256"]: np.arange(32) for r in fit}
    plan = stat.structured_plan(fit, orders, 253101, 2)
    again = stat.structured_plan(list(reversed(fit)), orders, 253101, 2)

    def encode(x):
        return json.dumps(
            [
                (stream, cids, [ix.tolist() for ix in indexes], families)
                for stream, cids, indexes, families in x
            ]
        )

    assert encode(plan) == encode(again) and len(plan) == 40
    assert encode(plan) != encode(stat.structured_plan(fit, orders, 253102, 2))
    allowed = {r["context_sha256"] for r in fit}
    excluded = {r["context_sha256"] for r in select + assess}
    for _, cids, indexes, families in plan:
        assert (
            len(cids) == 8
            and len(set(families)) == 4
            and all(families.count(f) == 2 for f in set(families))
        )
        assert set(cids) <= allowed and not (set(cids) & excluded)
        for ix in indexes:
            assert (
                len(ix) == 24 and len(set(ix)) == 24 and ix[:4].tolist() == [0, 1, 2, 3]
            )


def test_strict_selector_retains_incumbent_on_ties_or_small_gain():
    roster = [dict(family_id=f"f{i}") for i in range(6)]
    for gain in [0, 0.0005, 0.001]:
        panel = np.zeros((6, 5))
        panel[:, 1:] = gain
        assert stat.choose(roster, panel, 12)["selected_step"] == 0
    panel = np.zeros((6, 5))
    panel[:, 2:] = 0.003
    assert stat.choose(roster, panel, 12)["selected_step"] == 20


def test_dlval_context_weighting_and_family_cluster_estimator():
    roster = [dict(family_id="one")] + [dict(family_id="many")] * 3
    panel = np.zeros((3, 4, 4))
    panel[:, 0, 2] = 0.04
    weighted = stat.summarize(roster, panel, context_weighted=True)
    balanced = stat.summarize(roster, panel)
    assert weighted["contrasts"]["VREX_B1_minus_r882"]["median_gain"] == pytest.approx(
        0.01
    )
    assert balanced["contrasts"]["VREX_B1_minus_r882"]["median_gain"] == pytest.approx(
        0.02
    )
    assert weighted["contexts"] == 4 and weighted["families"] == 2
    assert weighted["family_utility_by_seed"] == balanced["family_utility_by_seed"]
    assert weighted["primary_utility_by_seed"][0][2] == pytest.approx(0.01)
    assert weighted["contrasts"]["VREX_B1_minus_CONTROL"][
        "median_gain"
    ] == pytest.approx(0.01)


def test_holm_two_treatments_and_no_variance_equivalence_claim():
    assert stat.holm(dict(VREX_B1=0.01, VREX_B10=0.04)) == dict(
        VREX_B1=0.02, VREX_B10=0.04
    )
    roster = [dict(family_id=f"f{i}") for i in range(3)]
    r = stat.summarize(roster, np.zeros((3, 3, 4)), context_weighted=True)
    for arm in stat.ARMS[1:]:
        c = r["contrasts"][arm + "_minus_r882"]
        assert c["ci95"] == [0, 0] and c["holm_p"] == 1 and not c["advance"]


def sample(utc=0, available=40 * 2**30, load5=90):
    return dict(
        utc=utc,
        available=available,
        load5=load5,
        rss=1000000,
        free=400_000_000_000,
        threads=2,
    )


def test_memory_pause_recovery_requires_full_thirty_seconds():
    guard = monitor.Guard()
    assert guard.update(sample(0, 19 * 2**30))[1]
    assert guard.update(sample(1, 25 * 2**30))[1]
    assert guard.update(sample(30, 25 * 2**30))[1]
    assert not guard.update(sample(31, 25 * 2**30))[1]


def test_memory_hard_stop_sustained_and_instant():
    guard = monitor.Guard()
    assert not guard.update(sample(0, 11 * 2**30))[0]
    assert not guard.update(sample(29, 11 * 2**30))[0]
    assert "MemAvailable<12GiB sustained30s" in guard.update(sample(30, 11 * 2**30))[0]
    assert "MemAvailable<6GiB" in monitor.Guard().update(sample(0, 5 * 2**30))[0]


def test_load_pause_hysteresis_and_absolute_deadline():
    guard = monitor.Guard()
    assert guard.update(sample(0, load5=109))[1]
    assert guard.update(sample(1, load5=101))[1]
    assert not guard.update(sample(2, load5=100))[1]
    assert monitor.violations(sample(monitor.COMPUTE_STOP)) == [
        "absolute compute stop 15:30Z"
    ]
    assert monitor.COMPUTE_STOP == launcher.STOP == 1793719800


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("rss", 4 * 2**30, "aggregate RSS>=4GiB"),
        ("free", 299_999_999_999, "disk free<300GB"),
        ("threads", 5, "aggregate threads>4"),
    ],
)
def test_resource_caps_are_fail_closed(field, value, reason):
    s = sample()
    s[field] = value
    assert reason in monitor.violations(s)


def test_launch_rejects_resume_and_reused_output(tmp_path):
    output = tmp_path / "run-001"
    launch = dict(output=str(output), resume_number=0, merged_source_verified=True)
    assert launcher.check_launch(launch, output.with_suffix(".launch.json")) == (
        output,
        0,
    )
    with pytest.raises(RuntimeError):
        launcher.check_launch(
            dict(launch, resume_number=1), output.with_suffix(".launch.json")
        )
    output.mkdir()
    with pytest.raises(RuntimeError):
        launcher.check_launch(launch, output.with_suffix(".launch.json"))


def test_launch_requires_exact_merged_source_and_review():
    launch = dict(
        manifest_sha256="m",
        source_commit="a" * 40,
        merged_source_verified=True,
        review_id=12,
        prereg_pr=2100,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    tx.validate_launch(launch, "a" * 40, "m", "p", "f")
    with pytest.raises(RuntimeError):
        tx.validate_launch(launch, "b" * 40, "m", "p", "f")
    with pytest.raises(RuntimeError):
        tx.validate_launch(dict(launch, review_id=None), "a" * 40, "m", "p", "f")
