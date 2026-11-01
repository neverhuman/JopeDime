"""Data-free unit controls for H171: family-stratified gradient accumulation, None-as-zero gradient vectors, and cosine diagnostics."""

import numpy as np
import pytest
import torch

import analysis as stat
import monitor
from treatment import (
    ACCUM_STEPS,
    DIAG_STEPS,
    PARAMETERS,
    TENSORS,
    TOTAL_UPDATES,
    accumulate_microbatches,
    adapter_forward,
    adapter_forward_static,
    adapter_l2_drift,
    compute_fit_family_weights,
    flatten_grad_with_none_as_zero,
    grad_cosine,
    grad_norms,
    precompute_static_slate,
)


def _synthetic_train_rows():
    rows = []
    idx = 0
    for f in range(205):
        fam = f"fam_{f:03d}"
        # 1 to 10 contexts per family, totaling 1219 contexts
        count = 6 if f < 189 else (5 if f < 204 else 10)
        for _ in range(count):
            rows.append(
                dict(
                    context_sha256=f"cid_{idx:04d}",
                    family_id=fam,
                    dl_split="DL-Train",
                    task="regression",
                    has_meta_test=False,
                    source_split="train",
                )
            )
            idx += 1
    assert len(rows) == 1219
    return rows


def test_family_folds_and_disjoint_partition():
    rows = _synthetic_train_rows()
    groups = stat.folds(rows)
    assert len(groups) == 5
    assert all(len(g) == 41 for g in groups)
    assert len({f for g in groups for f in g}) == 205
    for outer in range(5):
        fit, select, assess = stat.partition(rows, outer)
        f_fit = {r["family_id"] for r in fit}
        f_sel = {r["family_id"] for r in select}
        f_ass = {r["family_id"] for r in assess}
        assert len(f_fit) == 123 and len(f_sel) == 41 and len(f_ass) == 41
        assert not (f_fit & f_sel) and not (f_fit & f_ass) and not (f_sel & f_ass)
        assert len(fit) + len(select) + len(assess) == 1219


def test_family_stratified_plan_and_fit_weights():
    rows = _synthetic_train_rows()
    fit, select, assess = stat.partition(rows, 0)
    excluded = {r["context_sha256"] for r in select + assess}
    ordered_cids, weights, n_fams = compute_fit_family_weights(fit, excluded)
    assert n_fams == 123
    assert len(ordered_cids) == len(fit)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert float(weights.min()) > 0.0

    with pytest.raises(ValueError, match="disjoint"):
        compute_fit_family_weights(fit, {fit[0]["context_sha256"]})

    plan = stat.family_stratified_plan(fit, stat.SEEDS[0], 0, excluded)
    assert len(plan) == 40
    fit_fams = {r["family_id"] for r in fit}
    fit_cids = set(ordered_cids)
    for step_idx, entry in enumerate(plan, 1):
        assert entry["step"] == step_idx
        mbs = entry["microbatches"]
        assert len(mbs) == 16
        # Every single 8-context microbatch has 8 distinct families and fit-only cids
        for mb in mbs:
            assert len(mb["cids"]) == 8
            assert len(set(mb["families"])) == 8
            assert set(mb["cids"]) <= fit_cids
            assert not (set(mb["cids"]) & excluded)
        # ACC4 (first 4 microbatches = 32 contexts) spans 32 distinct families
        acc4_fams = [f for mb in mbs[:4] for f in mb["families"]]
        assert len(set(acc4_fams)) == 32
        # ACC16 (all 16 microbatches = 128 contexts) covers all 123 fit families
        acc16_fams = {f for mb in mbs for f in mb["families"]}
        assert acc16_fams == fit_fams

    # Verify CTRL_EX4 example-matched snapshot mapping (0/40/80/120/160 -> 0/10/20/30/40)
    assert stat.CTRL_EX4_SNAPSHOT_UPDATES == {0: 0, 40: 10, 80: 20, 120: 30, 160: 40}
    assert ACCUM_STEPS == {"CTRL": 1, "ACC4": 4, "ACC16": 16, "CTRL_EX4": 1}
    assert TOTAL_UPDATES == {"CTRL": 40, "ACC4": 40, "ACC16": 40, "CTRL_EX4": 160}
    assert DIAG_STEPS["CTRL"] == tuple(range(1, 41))
    assert DIAG_STEPS["ACC4"] == tuple(range(1, 41))
    assert DIAG_STEPS["ACC16"] == tuple(range(1, 41))
    assert len(DIAG_STEPS["CTRL_EX4"]) == 41
    assert DIAG_STEPS["CTRL_EX4"][0] == 1 and DIAG_STEPS["CTRL_EX4"][-1] == 160


def test_none_grad_as_zero_and_grad_norms():
    # Construct 47 parameter tensors totaling exactly 29,211 parameters (41 active, 6 None)
    sizes = [600] * 46 + [29211 - 46 * 600]
    assert len(sizes) == TENSORS and sum(sizes) == PARAMETERS
    params = [torch.nn.Parameter(torch.zeros(s, dtype=torch.float32)) for s in sizes]
    none_indices = {3, 11, 19, 27, 35, 46}
    for i, p in enumerate(params):
        if i not in none_indices:
            p.grad = torch.full_like(p, 0.05)

    vec, act_n, none_n = flatten_grad_with_none_as_zero(params)
    assert vec.shape == (PARAMETERS,)
    assert act_n == 41 and none_n == 6
    # Verify the 6 None-grad tensors are zero-filled at their exact slices
    offset = 0
    for i, s in enumerate(sizes):
        chunk = vec[offset : offset + s]
        if i in none_indices:
            assert float(torch.abs(chunk).max()) == 0.0
        else:
            assert torch.allclose(chunk, torch.full_like(chunk, 0.05))
        offset += s

    pre_vec, info = grad_norms(params, max_norm=1.0)
    assert torch.equal(pre_vec, vec)
    assert info["active_grad_tensors"] == 41
    assert info["none_grad_tensors"] == 6
    assert info["raw_grad_l2"] > 1.0
    assert abs(info["clipped_grad_l2"] - 1.0) < 1e-5

    for p in params:
        p.grad = None
    with pytest.raises(RuntimeError, match="all parameter gradients are None"):
        flatten_grad_with_none_as_zero(params)


def test_gradient_accumulation_and_cosine_monotonicity():
    torch.manual_seed(171101)
    p_active = torch.nn.Parameter(torch.zeros(200, dtype=torch.float32))
    p_none = torch.nn.Parameter(torch.zeros(50, dtype=torch.float32))
    params = [p_active, p_none]

    true_dir = torch.randn(200, dtype=torch.float32)
    true_dir = true_dir / torch.linalg.vector_norm(true_dir)
    g_full = torch.cat([true_dir.double(), torch.zeros(50, dtype=torch.float64)])

    cos_by_m = {1: [], 4: [], 16: []}
    gen = torch.Generator(device="cpu").manual_seed(171999)
    for _ in range(40):
        noises = [
            true_dir + 2.0 * torch.randn(200, generator=gen, dtype=torch.float32)
            for _ in range(16)
        ]
        for m in (1, 4, 16):
            fns = [lambda g=noises[k]: torch.dot(p_active, g) for k in range(m)]
            mean_loss = accumulate_microbatches(params, fns)
            assert np.isfinite(mean_loss)
            vec, act_n, none_n = flatten_grad_with_none_as_zero(
                params, expected_numel=250
            )
            assert act_n == 1 and none_n == 1
            expected_grad = torch.stack(noises[:m], dim=0).mean(dim=0).double()
            assert torch.allclose(vec[:200], expected_grad, atol=1e-6)
            assert float(torch.abs(vec[200:]).max()) == 0.0
            cos_by_m[m].append(grad_cosine(vec, g_full))

    med1 = float(np.median(cos_by_m[1]))
    med4 = float(np.median(cos_by_m[4]))
    med16 = float(np.median(cos_by_m[16]))
    assert med16 > med4 > med1 > 0.0


def test_selector_and_summary_contrasts():
    rows = [dict(context_sha256=f"c{i}", family_id=f"f{i // 2}") for i in range(82)]
    base = np.ones((82, 5), dtype=np.float64)
    # Tie rejects to step 0 (r882)
    d_tie = stat.choose(rows, base, 171141)
    assert (
        d_tie["selected_step"] == 0
        and d_tie["best_step"] == 0
        and not d_tie["accepted"]
    )

    # Positive gain > 0.001 with positive LCB95 selects step 20
    pos = base.copy()
    pos[:, 2] += 0.02
    d_pos = stat.choose(rows, pos, 171141)
    assert (
        d_pos["selected_step"] == 20 and d_pos["best_step"] == 20 and d_pos["accepted"]
    )

    # Summarize 3 seeds x 82 contexts x 5 policies (r882, CTRL, ACC4, ACC16, CTRL_EX4)
    panel = np.ones((3, 82, 5), dtype=np.float64)
    panel[:, :, 2] += 0.015  # ACC4
    panel[:, :, 3] += 0.025  # ACC16
    summary = stat.summarize(rows, panel)
    assert "ACC4_minus_r882" in summary["contrasts"]
    assert "ACC16_minus_r882" in summary["contrasts"]
    assert "CTRL_EX4_minus_r882" in summary["contrasts"]
    assert "ACC4_minus_CTRL_EX4" in summary["contrasts"]
    assert "ACC16_minus_ACC4" in summary["contrasts"]
    assert summary["contrasts"]["ACC16_minus_r882"]["advance"] is True
    assert summary["contrasts"]["CTRL_EX4_minus_r882"]["advance"] is False

    # Regression control for arm_identity_audit: identical B/primary/secondary utility vectors
    # with different arm-seeded bootstrap CIs must still be reported as selector_identical_to_ctrl=True.
    rng = np.random.default_rng(171888)
    b_mat = np.ones((82, 5), dtype=np.float64)
    b_mat[:, 1] += 0.005 + 0.002 * rng.standard_normal(82)
    cell_decisions = {
        arm: stat.choose(
            rows,
            b_mat,
            stat.stream_seed(stat.BOOT_SEED, 0, stat.SEEDS[0], arm),
        )
        for arm in stat.ARMS
    }
    assert cell_decisions["CTRL"]["selected_step"] == 10
    assert cell_decisions["ACC4"]["selected_step"] == 10
    # Arm-seeded bootstrap CIs differ even on identical B vectors:
    assert cell_decisions["CTRL"]["ci95"] != cell_decisions["ACC4"]["ci95"]
    selections = [cell_decisions]
    raw_b_by_arm = {
        "CTRL": [b_mat],
        "ACC4": [b_mat.copy()],
        "ACC16": [b_mat + 0.01],
        "CTRL_EX4": [b_mat.copy()],
    }
    prim = np.ones((3, 82, 5), dtype=np.float64)
    prim[:, :, 3] += 0.02  # ACC16 differs on primary; ACC4 and CTRL_EX4 match CTRL
    sec = np.ones((3, 19, 5), dtype=np.float64)
    hashes_by_arm = {
        a: [{str(s): f"{a}_s{s}" if s > 0 else "r882" for s in stat.STEPS}]
        for a in stat.ARMS
    }
    id_audit = stat.arm_identity_audit(
        selections, raw_b_by_arm, prim, sec, hashes_by_arm
    )
    assert id_audit["ACC4"]["selected_steps_identical_to_ctrl"] is True
    assert id_audit["ACC4"]["raw_b_vectors_identical_to_ctrl"] is True
    assert id_audit["ACC4"]["primary_vectors_identical_to_ctrl"] is True
    assert id_audit["ACC4"]["secondary_vectors_identical_to_ctrl"] is True
    assert id_audit["ACC4"]["selector_identical_to_ctrl"] is True
    assert id_audit["ACC4"]["step40_adapter_hashes_identical_to_ctrl"] == 0
    assert id_audit["ACC16"]["raw_b_vectors_identical_to_ctrl"] is False
    assert id_audit["ACC16"]["primary_vectors_identical_to_ctrl"] is False
    assert id_audit["ACC16"]["selector_identical_to_ctrl"] is False


def test_adapter_l2_drift_and_monitor_guards():
    s0 = {"w": torch.zeros(4), "b": torch.ones(3)}
    s1 = {"w": torch.full((4,), 3.0), "b": torch.full((3,), 1.0)}
    assert abs(adapter_l2_drift(s0, s1, ("w", "b")) - 6.0) < 1e-12

    ok = dict(
        load=10.0,
        cpu=20.0,
        available=32 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.80,
        rss=2_000_000_000,
        threads=3,
    )
    assert monitor.violations(ok) == []
    for k, bad_val in [
        ("load", 86.0),
        ("cpu", 86.0),
        ("available", 15 * 2**30),
        ("free", 39_000_000_000),
        ("disk_used_fraction", 0.95),
        ("rss", 6_000_000_000),
        ("threads", 5),
    ]:
        bad = dict(ok, **{k: bad_val})
        assert len(monitor.violations(bad)) == 1


class _SyntheticAdapter10Tok(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_modalities = 10
        self.register_buffer("prog_mean", torch.zeros(1, 1, 412))
        self.register_buffer("prog_std", torch.ones(1, 1, 412))
        self.modality_gate_logits = torch.nn.Parameter(torch.zeros(10))
        self.modality_type_emb = torch.nn.Parameter(torch.randn(10, 32) * 0.02)
        self.modality_norm = torch.nn.LayerNorm(32)
        self.prog_412_bottleneck = torch.nn.Linear(412, 16)
        self.cand_query_proj = torch.nn.Linear(24 + 24 + 16 + 3, 32)
        self.mm_cross_attn = torch.nn.MultiheadAttention(
            embed_dim=32, num_heads=4, dropout=0.0, batch_first=True
        )
        self.ctx_bilinear_u = torch.nn.Linear(32, 16)
        self.prog_bilinear_v = torch.nn.Linear(24, 16)
        self.mm_prior_out = torch.nn.Linear(32 + 16 + 3, 1)
        self.mm_prior_scale = torch.nn.Parameter(torch.tensor(0.25))
        self.delta_proj = torch.nn.Linear(32, 1)
        # 6 inactive parameters (analogous to UCB/prior tensors inactive under enable_ucb=False)
        self.unused_params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(4)) for _ in range(6)]
        )

    @staticmethod
    def _sem24(prog):
        return prog[..., :24] * 0.5 - prog[..., 24:48] * 0.25

    @staticmethod
    def _transform_prog(prog, mean, std):
        return (prog - mean) / std

    def encode_multimodal_context_and_candidates(
        self,
        mm_tokens_10x32,
        mm_mask_10,
        prog_412,
        sos_prior,
        active_modality_override_mask=None,
    ):
        assert active_modality_override_mask is None
        sem24 = self._sem24(prog_412)
        sem24_centered = sem24 - torch.mean(sem24, dim=1, keepdim=True)
        prog_std_412 = self._transform_prog(prog_412, self.prog_mean, self.prog_std)
        eff_mask = mm_mask_10.float().clone()
        eff_mask[:, 0] = 1.0
        mask_sum = torch.sum(eff_mask, dim=1, keepdim=True).clamp(min=1.0)
        key_pad_mask = eff_mask < 0.5
        gates = torch.sigmoid(self.modality_gate_logits).view(1, self.n_modalities, 1)
        t = self.modality_norm(
            mm_tokens_10x32.float() + self.modality_type_emb.unsqueeze(0)
        )
        tok_gated = t * gates * eff_mask.unsqueeze(-1)
        ctx_pooled = torch.sum(tok_gated, dim=1) / mask_sum
        prog_bot16 = self.prog_412_bottleneck(prog_std_412)
        cand_q = self.cand_query_proj(
            torch.cat([sem24, sem24_centered, prog_bot16, sos_prior.float()], dim=-1)
        )
        mm_attn_out, _ = self.mm_cross_attn(
            query=cand_q,
            key=tok_gated,
            value=tok_gated,
            key_padding_mask=key_pad_mask,
            need_weights=False,
        )
        bilinear_16 = self.ctx_bilinear_u(ctx_pooled).unsqueeze(
            1
        ) * self.prog_bilinear_v(sem24)
        raw_mm_delta = self.mm_prior_out(
            torch.cat([mm_attn_out, bilinear_16, sos_prior.float()], dim=-1)
        ).squeeze(-1)
        raw_mm_centered = raw_mm_delta - torch.mean(raw_mm_delta, dim=-1, keepdim=True)
        adjustment = torch.tanh(self.mm_prior_scale) * raw_mm_centered
        return ctx_pooled, cand_q, sem24, adjustment

    def compute_adapter_delta_from_static_terms(
        self, static_terms, scale, revealed_indices, revealed_raw_f, revealed_gate
    ):
        q = static_terms["cand_q"] + static_terms["ctx_pooled"].unsqueeze(1)
        base = self.delta_proj(q).squeeze(-1) + static_terms["eff_prior"]
        rev_factor = (
            revealed_raw_f.mean(dim=-1, keepdim=True)
            * revealed_gate.float().mean(dim=-1, keepdim=True)
            * scale.unsqueeze(-1)
            * 0.01
        )
        return (
            base + rev_factor + 0.0 * revealed_indices.float().sum(dim=-1, keepdim=True)
        )


class _SyntheticNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_10tok = _SyntheticAdapter10Tok()


def test_static_forward_output_and_gradient_parity_including_perturbed_weights():
    torch.manual_seed(171202)
    net = _SyntheticNet().eval()
    params = [p for p in net.parameters() if p.requires_grad]
    total_numel = sum(p.numel() for p in params)

    b, k = 4, 12
    prog = torch.randn(b, k, 412)
    sos = torch.randn(b, k, 3)
    tok = torch.randn(b, 10, 32)
    mask = (torch.rand(b, 10) > 0.3).float()
    scale = torch.ones(b)
    rev = torch.zeros(b, 4, dtype=torch.long)
    raw = torch.randn(b, 4)
    gate = torch.ones(b, 4, dtype=torch.bool)
    anchor_total = torch.randn(b, k)
    anchor_prior = torch.randn(b, k)
    target = torch.randn(b, k)

    static_slate = precompute_static_slate(
        net.adapter_10tok,
        prog,
        sos,
        mask,
        sem24_fn=_SyntheticAdapter10Tok._sem24,
        transform_prog_fn=_SyntheticAdapter10Tok._transform_prog,
    )

    gen = torch.Generator(device="cpu").manual_seed(171777)
    for mode in ("initial", "perturbed"):
        if mode == "perturbed":
            with torch.no_grad():
                for p in params:
                    p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))

        for p in params:
            p.grad = None
        out_dyn = adapter_forward(
            net, prog, sos, tok, mask, scale, rev, raw, gate, anchor_total, anchor_prior
        )
        loss_dyn = torch.nn.functional.smooth_l1_loss(out_dyn, target)
        loss_dyn.backward()
        grad_dyn, act_dyn, none_dyn = flatten_grad_with_none_as_zero(
            params, expected_numel=total_numel
        )

        for p in params:
            p.grad = None
        out_stat = adapter_forward_static(
            net,
            tok,
            static_slate,
            scale,
            rev,
            raw,
            gate,
            anchor_total,
            anchor_prior,
        )
        loss_stat = torch.nn.functional.smooth_l1_loss(out_stat, target)
        loss_stat.backward()
        grad_stat, act_stat, none_stat = flatten_grad_with_none_as_zero(
            params, expected_numel=total_numel
        )

        assert torch.equal(out_dyn.detach(), out_stat.detach())
        assert torch.equal(grad_dyn, grad_stat)
        assert act_dyn == act_stat and act_dyn > 0
        assert none_dyn == none_stat == 6

    net.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        adapter_forward_static(
            net, tok, static_slate, scale, rev, raw, gate, anchor_total, anchor_prior
        )
