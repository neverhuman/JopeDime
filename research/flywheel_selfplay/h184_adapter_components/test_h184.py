"""H184 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import analysis as stat
import treatment

HERE = Path(__file__).resolve().parent
LIVE = Path("/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py")
_spec = importlib.util.spec_from_file_location("h184_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


class FakeAdapter(nn.Module):
    """The attributes compute_adapter_delta_from_static_terms reads, with the live shapes and init rules."""

    def __init__(self, d_token=32):
        super().__init__()
        self.d_token = d_token
        self.null_kernel_w0_raw = nn.Parameter(torch.tensor(-0.5))
        self.sem_kernel_scale_raw = nn.Parameter(torch.tensor(1.2))
        self.emb_kernel_scale_raw = nn.Parameter(torch.tensor(0.8))
        self.traj_kernel_blend = nn.Parameter(torch.tensor(0.25))
        self.hist_token_proj = nn.Sequential(
            nn.Linear(24 + d_token + 4, d_token), nn.LayerNorm(d_token), nn.SiLU()
        )
        self.null_hist_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.05)
        self.traj_cross_attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=4, dropout=0.0, batch_first=True
        )
        self.traj_delta_head = nn.Sequential(
            nn.Linear(d_token * 2 + 3, 24), nn.SiLU(), nn.Linear(24, 1)
        )
        self.traj_attn_scale = nn.Parameter(torch.tensor(0.10))


def synthetic(b=3, m=24, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, m, 32, generator=g)
    sem = torch.nn.functional.normalize(torch.randn(b, m, 24, generator=g), dim=-1)
    adj = 0.1 * torch.randn(b, m, generator=g)
    static = dict(
        eff_prior=torch.randn(b, m, generator=g) + adj,
        mm_prior_adj=adj,
        sem24=sem,
        cand_q=q,
        cand_q_norm=torch.nn.functional.normalize(q, p=2, dim=-1, eps=1e-8),
    )
    rev = torch.arange(k).reshape(1, -1).expand(b, -1)
    raw = torch.randn(b, k, generator=g)
    gate = (torch.rand(b, k, generator=g) > 0.3).float()
    return static, torch.full((b,), 0.4208), rev, raw, gate


def components(ad, seed=0):
    static, scale, rev, raw, gate = synthetic(seed=seed)
    with torch.no_grad():
        return treatment.adapter_components(ad, static, scale, rev, raw, gate), (
            static,
            scale,
            rev,
            raw,
            gate,
        )


def test_registered_constants_and_arms():
    assert (
        stat.SEEDS == (184101, 184102, 184103)
        and stat.FOLD_SEED == 184140
        and stat.BOOT_SEED == 184141
    )
    assert stat.ARMS == (
        "R882",
        "ZERO_ADAPTER",
        "RAND0",
        "RAND0_NO_PRIOR",
        "RAND0_NO_KERNEL",
        "RAND0_NO_ATTN",
    )
    assert (
        len(stat.ARMS) <= 6
        and stat.CANDIDATES == stat.ARMS[2:]
        and treatment.COMPONENTS == ("PRIOR", "KERNEL", "ATTN")
    )


def test_five_fold_family_crossfit_is_disjoint_and_assesses_each_family_once():
    families = {f"f{i:03}" for i in range(205)}
    groups = stat.family_folds(families)
    assessed = []
    for outer in range(5):
        fit, select, assess = stat.partition(families, outer, groups)
        assert (
            (len(fit), len(select), len(assess)) == (123, 41, 41)
            and not fit & assess
            and not select & assess
        )
        assessed += list(assess)
    assert len(assessed) == len(set(assessed)) == 205


@pytest.mark.skipif(
    not LIVE.exists(),
    reason="pinned JopeDimeMicro model source not present (CI); the run gates this",
)
def test_components_sum_to_the_live_delta_bit_for_bit():
    import sys

    sys.path.insert(
        0, str(LIVE.parent)
    )  # the pinned module imports its sibling `model`
    try:
        spec = importlib.util.spec_from_file_location("h184_live_model", LIVE)
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
    finally:
        sys.path.remove(str(LIVE.parent))
    torch.manual_seed(1)
    ad = FakeAdapter().eval()
    parts, (static, scale, rev, raw, gate) = components(ad)
    with torch.no_grad():
        delta = (
            live.FlywheelBlockFormerAdapterNet.compute_adapter_delta_from_static_terms(
                ad, static, scale, rev, raw, gate
            )
        )
    assert treatment.assert_component_sum(parts, delta)


def test_component_sum_refusal_and_ablation_semantics():
    torch.manual_seed(2)
    parts, _ = components(FakeAdapter().eval())
    total = parts["PRIOR"] + parts["KERNEL"] + parts["ATTN"]
    with pytest.raises(RuntimeError):
        treatment.assert_component_sum(parts, total + 1e-6)
    anchor = torch.randn_like(total)
    assert torch.allclose(
        treatment.ablated_scores(anchor, parts, "KERNEL"),
        anchor + parts["PRIOR"] + parts["ATTN"],
    )
    assert torch.allclose(
        treatment.ablated_scores(anchor, parts, "PRIOR"),
        anchor + parts["KERNEL"] + parts["ATTN"],
    )
    assert torch.allclose(
        treatment.ablated_scores(anchor, parts, "ATTN"),
        anchor + parts["PRIOR"] + parts["KERNEL"],
    )
    with pytest.raises(ValueError):
        treatment.ablated_scores(anchor, parts, "NOPE")
    with pytest.raises(ValueError):
        static, scale, rev, raw, gate = synthetic()
        treatment.adapter_components(
            FakeAdapter(), static, scale, rev[:, :0], raw[:, :0], gate[:, :0]
        )


def test_attention_term_starts_small_but_nonzero_and_kernel_term_is_live():
    torch.manual_seed(3)
    ad = FakeAdapter().eval()
    nn.init.normal_(ad.traj_delta_head[-1].weight, mean=0.0, std=1e-3)  # live init rule
    nn.init.zeros_(ad.traj_delta_head[-1].bias)
    parts, _ = components(ad)
    assert (
        parts["ATTN"].abs().max() > 0
        and parts["ATTN"].abs().max() < parts["KERNEL"].abs().max()
    )


def panels(n=41, gains=None, noise=0.0, seed=0, offsets=None):
    rng = np.random.default_rng(seed)
    p = np.zeros((3, n, len(stat.ARMS)))
    for j, arm in enumerate(stat.ARMS):
        e = rng.normal(0.0, noise, size=(3, n)) if noise else np.zeros((3, n))
        e -= e.mean(axis=1, keepdims=True)
        off = np.asarray((offsets or {}).get(arm, [0.0, 0.0, 0.0]))[:, None]
        p[:, :, j] = (gains or {}).get(arm, 0.0) + (e if arm != "R882" else 0.0) + off
    return [f"f{i:03}" for i in range(n)], p


def run(families, p):
    minus2 = {a: [0, 0, 0] for a in stat.ARMS}
    agree = {a: [1.0, 1.0, 1.0] for a in stat.ARMS}
    return stat.summarize(families, p, p, families, minus2, agree)


def test_identical_arms_give_elo_1500_and_no_carrier():
    s = run(*panels())
    assert all(
        v["median_over_seeds"] == pytest.approx(1500.0)
        for v in s["elo_vs_r882"].values()
    )
    assert (
        s["structure_verdict"] == "NO_SINGLE_CARRIER" and not s["any_advance_candidate"]
    )
    assert all(c["label"] == "NEUTRAL" for c in s["components"].values())


def test_attribution_labels_holm_and_share():
    # RAND0 lifts +0.06 over ZERO; dropping the kernel loses it all; dropping attention loses nothing; dropping the
    # prior *helps* (RAND0 - NO_PRIOR < 0).
    gains = {
        "ZERO_ADAPTER": -0.06,
        "RAND0": 0.0,
        "RAND0_NO_KERNEL": -0.06,
        "RAND0_NO_ATTN": 0.0,
        "RAND0_NO_PRIOR": 0.02,
    }
    s = run(*panels(gains=gains))
    comps = s["components"]
    assert comps["KERNEL"]["label"] == "CARRIES_LIFT" and comps["KERNEL"][
        "share_of_rand0_minus_zero"
    ] == pytest.approx(1.0)
    assert comps["ATTN"]["label"] == "NEUTRAL" and comps["PRIOR"]["label"] == "HURTS"
    assert s["structure_verdict"] == "CARRIERS:KERNEL"
    assert stat.holm({"a": 0.01, "b": 0.02, "c": 0.5}) == pytest.approx(
        {"a": 0.03, "b": 0.04, "c": 0.5}
    )


def test_paired_bootstrap_reduces_over_seeds_axis0_not_draws():
    # Per-seed RAND0 - NO_KERNEL means are all slightly positive but family variance is large: axis=0 keeps the family
    # uncertainty (CI spans 0 -> NEUTRAL), whereas reducing over draws would collapse to three positive seed medians.
    fam, p = panels(
        205,
        noise=0.035,
        seed=184999,
        offsets={"RAND0_NO_KERNEL": [-0.0018, -0.0021, -0.0038]},
    )
    s = run(fam, p)
    lo, hi = s["attribution"]["KERNEL"]["family_bootstrap_ci95"]
    assert lo < 0.0 < hi and s["components"]["KERNEL"]["label"] == "NEUTRAL"


def test_advance_gates_strict_with_margin_tie():
    s = run(*panels(gains={"RAND0": 0.02}))
    assert s["advance_candidates"]["RAND0"]["advance_candidate"]
    s = run(*panels(gains={"RAND0": 0.001}))  # exactly the margin is a tie: rejected
    assert (
        not s["advance_candidates"]["RAND0"]["gates"]["margin"]
        and not s["any_advance_candidate"]
    )


def test_launch_review_or_waiver_and_monitor_guards():
    base = dict(
        manifest_sha256="m",
        merged_source_verified=True,
        source_commit="h",
        prereg_pr=1,
        prereg_sha256="p",
        fold_manifest_sha256="f",
    )
    treatment.validate_launch(dict(base, review_id=5), "h", "m", "p", "f")
    with pytest.raises(RuntimeError):
        treatment.validate_launch(base, "h", "m", "p", "f")
    ok = dict(
        load=20.0,
        cpu=30.0,
        available=70 * 2**30,
        free=500e9,
        disk_used_fraction=0.93,
        rss=1e9,
        threads=3,
    )
    assert monitor.violations(ok) == []
    assert monitor.violations(dict(ok, disk_used_fraction=0.938)) == [
        "disk user-visible used>=93.8%"
    ]
    assert monitor.violations(dict(ok, threads=5)) == ["aggregate threads>4"]
