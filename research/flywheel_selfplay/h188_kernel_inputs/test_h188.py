"""H188 design/statistics controls; all fixtures are synthetic and data-free (no checkpoint, corpus or model)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import analysis as stat
import treatment

HERE = Path(__file__).resolve().parent
LIVE = Path("/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py")
_spec = importlib.util.spec_from_file_location("h188_test_monitor", HERE / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


class FakeAdapter(nn.Module):
    """Attributes read by the live kernel/delta code, with the live shapes and init rules (as in H184)."""

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


class FakeAnchor(nn.Module):
    def encode_candidates(self, ctx_feat, prog_412, proj_16, sos_prior):
        b, n, _ = prog_412.shape
        g = torch.Generator().manual_seed(7)
        w_emb, w_sig = (
            torch.randn(412, 48, generator=g),
            torch.randn(412, 24, generator=g),
        )
        return torch.zeros(b, 48), prog_412 @ w_emb, prog_412 @ w_sig, torch.zeros(b, n)


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
    rev = (
        torch.full((b,), 0.4208),
        torch.arange(k).reshape(1, -1).expand(b, -1),
        torch.randn(b, k, generator=g),
        (torch.rand(b, k, generator=g) > 0.3).float(),
    )
    return static, rev


def test_registered_constants_and_arms():
    assert (
        stat.SEEDS == (188101, 188102, 188103)
        and stat.FOLD_SEED == 188140
        and stat.BOOT_SEED == 188141
    )
    assert (
        stat.ARMS == ("R882", "KIN_ANCHOR", "KIN_RANKRES", "KIN_CLIPRES")
        and stat.CANDIDATES == stat.ARMS[1:]
    )


def test_seed_reshuffled_folds_cover_each_family_once_in_b_and_once_in_a():
    families = [f"f{i:03}" for i in range(205)]
    per_seed = [stat.seed_folds(families, s) for s in stat.SEEDS]
    assert per_seed[0] != per_seed[1] != per_seed[2] and per_seed[0] == stat.seed_folds(
        families, stat.SEEDS[0]
    )
    for groups in per_seed:
        b_cov, a_cov = [], []
        for outer in range(5):
            fit, b, a = stat.partition(set(families), outer, groups)
            assert not fit & (a | b) and not a & b
            b_cov += list(b)
            a_cov += list(a)
        assert sorted(b_cov) == sorted(a_cov) == sorted(families)


def test_identity_kernel_replay_equals_live_kernel_term_bit_for_bit():
    torch.manual_seed(1)
    ad = FakeAdapter().eval()
    static, rev = synthetic()
    with torch.no_grad():
        parts = treatment.adapter_components(ad, static, *rev)
        ident = treatment.kernel_term(
            ad, static["sem24"], static["cand_q_norm"], static["eff_prior"], *rev
        )
    assert torch.equal(ident, parts["KERNEL"])
    if LIVE.exists():
        sys.path.insert(0, str(LIVE.parent))
        try:
            spec = importlib.util.spec_from_file_location("h188_live_model", LIVE)
            live = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(live)
        finally:
            sys.path.remove(str(LIVE.parent))
        with torch.no_grad():
            delta = live.FlywheelBlockFormerAdapterNet.compute_adapter_delta_from_static_terms(
                ad, static, *rev
            )
        assert torch.equal(parts["PRIOR"] + ident + parts["ATTN"], delta)


def test_rank_residuals_keep_location_scale_and_order_and_average_ties():
    r = torch.tensor(
        [[0.1, -3.0, 0.5, 9.0], [1.0, 1.0, 2.0, 0.0], [2.0, 2.0, 2.0, 2.0]]
    )
    out = treatment.rank_residuals(r)
    assert torch.allclose(out[:2].mean(-1), r[:2].mean(-1), atol=1e-6)
    assert torch.allclose(
        out[:2].std(-1, unbiased=False), r[:2].std(-1, unbiased=False), atol=1e-5
    )
    assert torch.equal(torch.argsort(out[0]), torch.argsort(r[0]))
    assert out[1, 0] == out[1, 1] and torch.equal(
        out[2], r[2]
    )  # ties averaged; constant row unchanged
    assert (out[0, 3] - out[0, 2]) == pytest.approx(
        float(out[0, 2] - out[0, 0]), rel=1e-5
    )  # equal rank gaps


def test_huber_clip_residuals_clip_invalid_outlier_and_keep_inliers():
    r = torch.tensor([[0.0, 0.2, -0.2, 5.9], [1.0, 1.0, 1.0, 1.0]])
    out = treatment.huber_clip_residuals(r)
    med = r[0].median()
    bound = treatment.HUBER_C * treatment.MAD_TO_SD * (r[0] - med).abs().median()
    assert out[0, 3] == pytest.approx(float(med + bound)) and out[0, 3] < r[0, 3]
    assert torch.equal(out[0, :3], r[0, :3]) and torch.equal(out[1], r[1])


def test_variant_scores_replace_only_the_kernel_term_and_anchor_features_are_unit_norm():
    torch.manual_seed(2)
    ad = FakeAdapter().eval()
    static, rev = synthetic()
    with torch.no_grad():
        parts = treatment.adapter_components(ad, static, *rev)
        anchor = torch.randn_like(parts["PRIOR"])
        same = treatment.variant_scores(anchor, parts, parts["KERNEL"])
        assert torch.equal(
            same, (anchor + (parts["PRIOR"] + parts["KERNEL"] + parts["ATTN"])).float()
        )
        prog = torch.randn(3, 24, 412)
        sig, emb = treatment.anchor_features(
            FakeAnchor(),
            torch.zeros(3, 584),
            prog,
            torch.zeros(3, 24, 16),
            torch.zeros(3, 24, 2),
        )
        assert sig.shape == (3, 24, 24) and torch.allclose(
            sig.norm(dim=-1), torch.ones(3, 24), atol=1e-5
        )
        assert torch.allclose(emb.norm(dim=-1), torch.ones(3, 24), atol=1e-5)
        v = treatment.kernel_term(ad, sig, emb, static["eff_prior"], *rev)
        assert v.shape == parts["KERNEL"].shape and not torch.equal(v, parts["KERNEL"])


def rows(n=205, per=2):
    return [
        {"context_sha256": f"c{f}-{i}", "family_id": f"f{f:03}"}
        for f in range(n)
        for i in range(per)
    ]


def panel(r, gains=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([1.0 + 0.01 * (int(x["family_id"][1:]) % 7) for x in r])
    cols = [base]
    for arm in stat.CANDIDATES:
        e = rng.normal(0.0, noise, size=len(r)) if noise else 0.0
        cols.append(base + (gains or {}).get(arm, 0.0) + e)
    return np.stack([np.stack(cols, axis=1)] * 3)


def test_pooled_b_select_m3_strict_rule_and_tie():
    r = rows()
    p = panel(r, {"KIN_ANCHOR": 0.01, "KIN_RANKRES": 0.001})
    d = stat.pooled_b_select_m3(
        r, p[:, :, 0], {a: p[:, :, j + 1] for j, a in enumerate(stat.CANDIDATES)}
    )
    assert d["KIN_ANCHOR"]["admitted"] and d["KIN_ANCHOR"]["holm_multiplicity"] == 3
    assert (
        not d["KIN_RANKRES"]["admitted"] and not d["KIN_CLIPRES"]["admitted"]
    )  # exactly +0.001 ties; 0 rejects
    with pytest.raises(ValueError):
        stat.pooled_b_select_m3(r, p[:, :, 0], {"KIN_ANCHOR": p[:, :, 1]})


def test_verdict_requires_b_admission_and_a_confirmation():
    r = rows()
    p = panel(r, {"KIN_ANCHOR": 0.01, "KIN_CLIPRES": 0.01})
    b = stat.pooled_b_select_m3(
        r, p[:, :, 0], {a: p[:, :, j + 1] for j, a in enumerate(stat.CANDIDATES)}
    )
    m2 = {a: [0, 0, 0] for a in stat.ARMS}
    ag = {a: [1.0, 1.0, 1.0] for a in stat.ARMS}
    s = stat.summarize_a(r, p, m2, ag)
    v = stat.verdict(b, s)
    assert (
        v["verdict"] == "ADVANCE:KIN_ANCHOR+KIN_CLIPRES" and s["identical_seed_panels"]
    )
    b["KIN_CLIPRES"]["admitted"] = False
    assert stat.verdict(b, s)["verdict"] == "ADVANCE:KIN_ANCHOR"
    flat = stat.summarize_a(r, panel(r), m2, ag)
    assert stat.verdict(b, flat)["verdict"] == "NO_ADVANCE"
    assert all(
        v["median_over_seeds"] == pytest.approx(1500.0)
        for v in flat["elo_vs_r882"].values()
    )


def test_pooled_a_bootstrap_reduces_over_seeds_axis0():
    r = rows()
    p = panel(r, noise=0.2, seed=188999)
    p[:, :, 1] += np.array([0.002, 0.0015, 0.001])[:, None]
    m2 = {a: [0, 0, 0] for a in stat.ARMS}
    ag = {a: [1.0, 1.0, 1.0] for a in stat.ARMS}
    s = stat.summarize_a(r, p, m2, ag)
    lo, hi = s["contrasts"]["KIN_ANCHOR_minus_R882"]["family_bootstrap_ci95"]
    assert (
        lo < 0.0 < hi and s["contrasts"]["KIN_ANCHOR_minus_R882"]["positive_seeds"] == 3
    )


def test_launch_and_monitor_guards():
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
