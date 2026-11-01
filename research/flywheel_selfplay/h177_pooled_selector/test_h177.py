"""Data-free unit controls for H177: pooled cross-fit confirmatory selector (205-family pooled B + single prespecified candidate + separate A confirmation)."""

import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

import analysis as stat
import monitor
from treatment import (
    CANDIDATE_STEP,
    PARAMETERS,
    TENSORS,
    accumulate_microbatches,
    adapter_forward_static,
    apply_optimizer_settings,
    compute_fit_family_weights,
    flatten_grad_with_none_as_zero,
    grad_norms,
    oc15_cycle,
    precompute_static_slate,
    validate_h175_power_anchor,
)

HERE = Path(__file__).resolve().parent


def _synthetic_1219_rows():
    """Construct a data-free 1,219-context / 205-family cohort matching the exact fold_manifest.json families."""
    fold_fams = json.loads((HERE / "fold_manifest.json").read_text())["families"]
    all_fams = [f for grp in fold_fams for f in grp]
    assert len(all_fams) == 205 and len(set(all_fams)) == 205
    rows = []
    # 205 families * 5 = 1025 contexts, plus 194 extra contexts across the first 194 families = 1,219 contexts.
    cid_idx = 0
    for fi, fam in enumerate(all_fams):
        count = 6 if fi < 194 else 5
        for _ in range(count):
            cid_idx += 1
            rows.append(
                dict(
                    context_sha256=f"cid-{cid_idx:04d}",
                    family_id=fam,
                    dl_split="DL-Train",
                    task="regression",
                )
            )
    assert len(rows) == 1219
    return rows


def test_pooled_b_never_overlaps_fold_fit_and_covers_all_205_families_once():
    rows = _synthetic_1219_rows()
    proof = stat.verify_pooled_b_coverage(rows)
    assert proof == dict(
        total_families=205,
        total_contexts=1219,
        b_exact_once=True,
        a_exact_once=True,
        zero_fit_b_overlap=True,
        zero_b_a_same_fold_overlap=True,
    )
    # Also verify fold_manifest.json matches stat.folds(rows)
    manifest = json.loads((HERE / "fold_manifest.json").read_text())
    assert manifest["fold_seed"] == 177140
    assert manifest["boot_seed"] == 177141
    assert manifest["seeds"] == [177101, 177102, 177103]
    assert stat.folds(rows) == manifest["families"]


def test_candidate_fixed_before_b_read_and_snapshot_search_forbidden(tmp_path):
    rows = _synthetic_1219_rows()
    guard = stat.StageOrderGuard()
    for arm in stat.ARMS:
        assert guard.candidate_step_for(arm) == CANDIDATE_STEP == 40
    with pytest.raises(RuntimeError, match="immutable"):
        guard.set_candidate_step("ACC4", 20)

    # Attempting A confirmation before pooled_b_selection_lock.json is written must fail.
    with pytest.raises(RuntimeError, match="A-family confirmation forbidden"):
        guard.require_b_locked()

    # Passing a multi-step candidate array (e.g. shape (3, 1219, 4) for steps 10/20/30/40) must fail.
    r882 = np.zeros(len(rows), dtype=np.float64)
    multi_step = np.zeros((3, len(rows), 4), dtype=np.float64)
    with pytest.raises(ValueError, match="snapshot search forbidden"):
        stat.pooled_b_select(rows, r882, {"ACC4": multi_step})

    # Writing a valid pooled_b_selection_lock.json unlocks Stage 2 A confirmation.
    single_step = {arm: np.zeros((3, len(rows)), dtype=np.float64) for arm in stat.ARMS}
    decisions = stat.pooled_b_select(rows, r882, single_step)
    lock_file = tmp_path / "pooled_b_selection_lock.json"
    lock_file.write_text(
        json.dumps(
            dict(
                wave="H177",
                pooled_b_families=205,
                decisions=decisions,
            )
        )
    )
    guard.mark_b_locked(lock_file)
    locked_payload = guard.require_b_locked()
    assert locked_payload["pooled_b_families"] == 205


def test_exact_null_identical_candidate_admits_zero_times_and_ties_reject():
    rows = _synthetic_1219_rows()
    rng = np.random.default_rng(177901)
    r882 = rng.normal(1.2, 0.3, size=len(rows))
    identical_panels = {
        arm: np.broadcast_to(r882[None, :], (3, len(rows))).copy() for arm in stat.ARMS
    }
    admitted_count = 0
    for trial in range(5):
        decisions = stat.pooled_b_select(
            rows, r882, identical_panels, seed=stat.BOOT_SEED + trial
        )
        for arm in stat.ARMS:
            d = decisions[arm]
            assert d["mean_gain"] == 0.0
            assert d["ci95"] == [0.0, 0.0]
            assert d["admitted"] is False
            assert d["unadjusted_admitted"] is False
            assert d["selected_step"] == 0
            if d["admitted"] or d["unadjusted_admitted"]:
                admitted_count += 1
    assert admitted_count == 0

    # Exact boundary tie at +0.001 (and +0.001 + 0.5e-12) must reject even with zero variance (ci95 > 0).
    tie_panels = {
        "CTRL": np.broadcast_to((r882 + 0.001)[None, :], (3, len(rows))).copy(),
        "ACC4": np.broadcast_to(
            (r882 + 0.001 + 0.5e-12)[None, :], (3, len(rows))
        ).copy(),
        "ACC16": np.broadcast_to(r882[None, :], (3, len(rows))).copy(),
        "OC15": np.broadcast_to(r882[None, :], (3, len(rows))).copy(),
    }
    tie_decisions = stat.pooled_b_select(rows, r882, tie_panels)
    assert tie_decisions["CTRL"]["ci95"][0] > 0.0
    assert tie_decisions["CTRL"]["admitted"] is False
    assert tie_decisions["CTRL"]["selected_step"] == 0
    assert tie_decisions["ACC4"]["admitted"] is False
    assert tie_decisions["ACC4"]["selected_step"] == 0


def test_synthetic_plus_001_shift_admits_and_within_family_seed_averaging():
    rows = _synthetic_1219_rows()
    rng = np.random.default_rng(177902)
    r882 = rng.normal(1.15, 0.25, size=len(rows))
    # Synthetic +0.01 shift with modest seed noise admits cleanly.
    shift_panel = np.stack(
        [r882 + 0.01 + rng.normal(0.0, 0.002, size=len(rows)) for _ in range(3)]
    )
    # Construct a zero-mean cancelling seed perturbation (+0.05 on seed 0, -0.05 on seed 1, 0.0 on seed 2)
    # to prove that seeds are averaged WITHIN each family before resampling.
    cancelling_panel = np.stack([r882 + 0.05, r882 - 0.05, r882.copy()])
    panels = {
        "CTRL": np.broadcast_to(r882[None, :], (3, len(rows))).copy(),
        "ACC4": shift_panel,
        "ACC16": cancelling_panel,
        "OC15": np.broadcast_to(r882[None, :], (3, len(rows))).copy(),
    }
    decisions = stat.pooled_b_select(rows, r882, panels, seed=stat.BOOT_SEED)
    acc4 = decisions["ACC4"]
    assert acc4["mean_gain"] > 0.008
    assert acc4["ci95"][0] > 0.005
    assert acc4["holm_p"] < 0.01
    assert acc4["admitted"] is True
    assert acc4["selected_step"] == 40

    # ACC16 has +0.05 on seed 0 and -0.05 on seed 1, so within-family seed average is identically 0.0.
    acc16 = decisions["ACC16"]
    assert abs(acc16["mean_gain"]) < 1e-15
    assert np.allclose(acc16["ci95"], [0.0, 0.0], atol=1e-15)
    assert acc16["admitted"] is False
    assert acc16["selected_step"] == 0


def test_h175_power_anchor_gate_mde80_scaling_and_h178_reusable_decision_object():
    anchor = json.loads((HERE / "h175_power_anchor.json").read_text())
    # Verify both context (+0.0158/+0.0228) and family-clustered (+0.0640/+0.0947) MDE80s are pinned.
    mde41 = anchor["mde80_single_fold_41_family"]
    mde205 = anchor["expected_pooled_205_family_mde80"]
    assert math.isclose(
        mde41["context"]["24_slate"], 0.015834306207683856, rel_tol=1e-9
    )
    assert math.isclose(
        mde41["context"]["full_slate"], 0.02280187032608344, rel_tol=1e-9
    )
    assert math.isclose(
        mde41["family_clustered"]["24_slate"], 0.06395704693361139, rel_tol=1e-9
    )
    assert math.isclose(
        mde41["family_clustered"]["full_slate"], 0.09467790954714358, rel_tol=1e-9
    )
    assert math.isclose(
        mde205["family_clustered"]["24_slate_raw_div_sqrt5"],
        0.06395704693361139 / math.sqrt(5.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        mde205["family_clustered"]["full_slate_raw_div_sqrt5"],
        0.09467790954714358 / math.sqrt(5.0),
        rel_tol=1e-12,
    )
    assert anchor["h178_calibration_gate"] == stat.H178_CALIBRATION_GATE

    # Committed anchor is now READY_TO_LAUNCH with H175 closeout b0261812e125c8743370b7801b75d27e7f27977f (MIXED).
    assert anchor["status"] == "READY_TO_LAUNCH"
    assert anchor["h175_closeout_commit"] == "b0261812e125c8743370b7801b75d27e7f27977f"
    assert anchor["h175_verdict"] == "MIXED"
    validate_h175_power_anchor(anchor, smoke=True)
    validate_h175_power_anchor(anchor, smoke=False)

    # Pending closeout must block non-smoke runs.
    pending = dict(anchor, status="PENDING_H175_CLOSEOUT")
    with pytest.raises(RuntimeError, match="blocked until H175 closes"):
        validate_h175_power_anchor(pending, smoke=False)

    # CEILING_LIKELY must hard-refuse science run.
    ceiling = dict(anchor, status="READY_TO_LAUNCH", h175_verdict="CEILING_LIKELY")
    with pytest.raises(RuntimeError, match="CEILING_LIKELY"):
        validate_h175_power_anchor(ceiling, smoke=False)

    # Missing H178 calibration gate or missing family_clustered MDE80 must fail.
    bad_gate = dict(anchor, h178_calibration_gate="missing")
    with pytest.raises(RuntimeError, match="H178 calibration gate"):
        validate_h175_power_anchor(bad_gate, smoke=True)

    # Valid MIXED anchor with 40-hex closeout commit passes.
    ready = dict(
        anchor,
        status="READY_TO_LAUNCH",
        h175_verdict="MIXED",
        h175_closeout_commit="a" * 40,
    )
    validate_h175_power_anchor(ready, smoke=False)

    # Verify H177_DECISION_RULE is importable as one object and reusable unchanged by H178 on single or custom panels.
    rows = _synthetic_1219_rows()
    rng = np.random.default_rng(177909)
    r882 = rng.normal(1.15, 0.25, size=len(rows))
    null_panel = np.broadcast_to(r882[None, :], (3, len(rows))).copy()
    plus_panel = np.stack([r882 + 0.015 for _ in range(3)])
    single_null = stat.H177_DECISION_RULE(rows, r882, null_panel)
    assert single_null["admitted"] is False and single_null["selected_step"] == 0
    single_plus = stat.H177_DECISION_RULE.decide_single_panel(rows, r882, plus_panel)
    assert single_plus["admitted"] is True and single_plus["selected_step"] == 40
    custom_bank = stat.H177_DECISION_RULE.select(
        rows, r882, {"H175_NULL": null_panel, "H175_ORACLE_005": plus_panel}
    )
    assert custom_bank["H175_NULL"]["admitted"] is False
    assert custom_bank["H175_ORACLE_005"]["admitted"] is True


def test_recipe_arms_family_stratified_batches_oc15_and_none_grad_handling():
    rows = _synthetic_1219_rows()
    fit, select, assess = stat.partition(rows, 0)
    excl = {r["context_sha256"] for r in select + assess}
    ordered_cids, weights, n_fams = compute_fit_family_weights(fit, excl)
    assert n_fams == 123
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-12)

    plan = stat.family_stratified_plan(fit, 177101, 0, excl)
    assert len(plan) == 40
    for step_entry in plan:
        mbs = step_entry["microbatches"]
        assert len(mbs) == 16
        # CTRL / OC15 microbatch 0 has 8 distinct fit families
        assert len(set(mbs[0]["families"])) == 8
        # ACC4 microbatches 0..3 have 32 distinct fit families
        acc4_fams = [f for mb in mbs[:4] for f in mb["families"]]
        assert len(set(acc4_fams)) == 32
        # ACC16 microbatches 0..15 cover all 123 fit families
        acc16_fams = [f for mb in mbs for f in mb["families"]]
        assert len(set(acc16_fams)) == 123

    # Verify OC15 schedule endpoints
    lr1, b1_1 = oc15_cycle(1)
    lr12, b1_12 = oc15_cycle(12)
    lr40, b1_40 = oc15_cycle(40)
    assert math.isclose(lr1, 1e-5, rel_tol=1e-12) and math.isclose(b1_1, 0.95)
    assert math.isclose(lr12, 1.5e-4, rel_tol=1e-12) and math.isclose(b1_12, 0.85)
    assert math.isclose(lr40, 1e-6, rel_tol=1e-12) and math.isclose(b1_40, 0.95)

    # Verify 47-tensor (41 active + 6 None-as-zero = 29,211 params) gradient flattening & accumulation
    sizes = [600] * 41 + [768] * 5 + [771]
    assert sum(sizes) == PARAMETERS and len(sizes) == TENSORS
    params = [nn.Parameter(torch.ones(s, dtype=torch.float32)) for s in sizes]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    lr_t, b1_t = apply_optimizer_settings(opt, "OC15", 12)
    assert math.isclose(lr_t, 1.5e-4) and math.isclose(b1_t, 0.85)

    def make_mb_loss(scale):
        return lambda: sum((p * scale).sum() for p in params[:41])

    mean_loss = accumulate_microbatches(
        params,
        [make_mb_loss(1.0), make_mb_loss(2.0), make_mb_loss(3.0), make_mb_loss(4.0)],
    )
    assert mean_loss > 0.0
    flat_vec, act_n, none_n = flatten_grad_with_none_as_zero(params)
    assert flat_vec.numel() == PARAMETERS and act_n == 41 and none_n == 6
    vec, ginfo = grad_norms(params, max_norm=1.0)
    assert vec.numel() == PARAMETERS
    assert ginfo["active_grad_tensors"] == 41
    assert ginfo["none_grad_tensors"] == 6
    assert torch.all(vec[41 * 600 :] == 0.0)


class _ToyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_modalities = 10
        self.register_buffer("prog_mean", torch.zeros(412))
        self.register_buffer("prog_std", torch.ones(412))
        self.modality_gate_logits = nn.Parameter(torch.zeros(10))
        self.modality_type_emb = nn.Parameter(torch.zeros(10, 32))
        self.modality_norm = nn.LayerNorm(32)
        self.prog_412_bottleneck = nn.Linear(412, 16)
        self.cand_query_proj = nn.Linear(24 + 24 + 16 + 2, 32)
        self.mm_cross_attn = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True)
        self.ctx_bilinear_u = nn.Linear(32, 16)
        self.prog_bilinear_v = nn.Linear(24, 16)
        self.mm_prior_out = nn.Linear(32 + 16 + 2, 1)
        self.mm_prior_scale = nn.Parameter(torch.tensor(0.1))
        self.extra_scale = nn.Parameter(torch.tensor(0.2))
        self.unused = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(6)])

    def compute_adapter_delta_from_static_terms(
        self, static_terms, scale, revealed_indices, revealed_raw_f, revealed_gate
    ):
        return static_terms["mm_prior_adj"] + self.extra_scale * (
            static_terms["cand_q"].mean(dim=-1)
            + static_terms["ctx_pooled"].mean(dim=-1, keepdim=True)
        )


def test_static_forward_and_perturbed_gradient_parity_and_resource_guards():
    torch.manual_seed(177903)
    ad = _ToyAdapter().eval()
    net = type("Net", (), {"adapter_10tok": ad})()
    b, k = 4, 24
    prog = torch.randn(b, k, 412)
    sos = torch.randn(b, k, 2)
    tok = torch.randn(b, 10, 32)
    mask = torch.ones(b, 10, dtype=torch.bool)
    scale = torch.full((b,), 0.4208)
    rev = torch.arange(4).unsqueeze(0).expand(b, -1)
    raw = torch.randn(b, 4)
    gate = torch.ones(b, 4)
    anchor_total = torch.randn(b, k)
    anchor_prior = torch.randn(b, k)

    def sem24_fn(p):
        return p[:, :, :24]

    def transform_fn(p, m, s):
        return (p - m) / s

    static_slate = precompute_static_slate(
        ad, prog, sos, mask, sem24_fn=sem24_fn, transform_prog_fn=transform_fn
    )
    out1 = adapter_forward_static(
        net, tok, static_slate, scale, rev, raw, gate, anchor_total, anchor_prior
    )
    out2 = adapter_forward_static(
        net, tok, static_slate, scale, rev, raw, gate, anchor_total, anchor_prior
    )
    assert torch.equal(out1, out2)

    # Verify summarize() and canonical H148 BT Elo anchor at 1500.0
    rows = _synthetic_1219_rows()
    base = np.ones((3, len(rows), 1), dtype=np.float64)
    panel = np.concatenate([base, base, base + 0.02, base + 0.01, base], axis=2)
    summary = stat.summarize(rows, panel)
    assert math.isclose(summary["seed_averaged_elo"][0], 1500.0, abs_tol=1e-9)
    assert summary["seed_averaged_elo"][2] > 1500.0
    assert summary["contrasts"]["ACC4_minus_r882"]["advance"] is True
    assert summary["contrasts"]["CTRL_minus_r882"]["advance"] is False

    # Verify monitor resource guards
    ok_sample = dict(
        load=10.0,
        cpu=20.0,
        available=32 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.80,
        rss=2_000_000_000,
        threads=2,
    )
    assert monitor.violations(ok_sample) == []
    bad_sample = dict(ok_sample, rss=6_500_000_000, threads=5)
    v = monitor.violations(bad_sample)
    assert "RSS>=6GB" in v and "aggregate threads>4" in v
