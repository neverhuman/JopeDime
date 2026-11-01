"""Unit and protocol tests for H198 per-context test-time training (TTT)."""

import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import analysis as stat
from treatment import (
    DERANGEMENTS_4,
    GRID,
    K_GRID,
    L2_BETA,
    LR_GRID,
    revealed_live_loss,
    shuf_permutation,
    ttt_context_trajectory,
    validate_launch,
    verify_leakage_mask_invariance,
)

HERE = Path(__file__).resolve().parent


class _ToyAdapter10Tok(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(198)
        self.proj = nn.Linear(16, 8)
        self.head = nn.Linear(8, 1)

    def encode_multimodal_context_and_candidates(
        self,
        mm_tokens_10x32,
        mm_mask_10,
        prog_412,
        sos_prior,
        active_modality_override_mask=None,
    ):
        del mm_tokens_10x32, mm_mask_10, sos_prior, active_modality_override_mask
        h = torch.tanh(self.proj(prog_412[:, :, :16]))
        adj = self.head(h).squeeze(-1)
        pooled = h.mean(dim=1)
        return pooled, h, h, adj

    def compute_adapter_delta_from_static_terms(
        self,
        static_terms,
        scale,
        revealed_indices,
        revealed_raw_f,
        revealed_gate,
    ):
        del scale
        q = static_terms["cand_q_norm"]
        rev_q = torch.gather(
            q, dim=1, index=revealed_indices.unsqueeze(-1).expand(-1, -1, q.shape[-1])
        )
        sim = torch.matmul(q, rev_q.transpose(1, 2))
        weights = revealed_gate * torch.tanh(revealed_raw_f)
        kern = torch.matmul(sim, weights.unsqueeze(-1)).squeeze(-1) / 4.0
        return static_terms["mm_prior_adj"] + 0.45 * kern


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_10tok = _ToyAdapter10Tok()


def _toy_runtime():
    rt = types.SimpleNamespace()

    def tensor_hash(state):
        h = hashlib.sha256()
        for k in sorted(state):
            h.update(k.encode())
            t = state[k].detach().cpu().contiguous().numpy()
            h.update(t.tobytes())
        return h.hexdigest()

    def loss(scores, fitness, gate):
        target = torch.where(gate, fitness, torch.full_like(fitness, 2.0))
        target_z = (target - target.mean(dim=1, keepdim=True)) / (
            target.std(dim=1, keepdim=True) + 1e-6
        )
        pred_z = (scores - scores.mean(dim=1, keepdim=True)) / (
            scores.std(dim=1, keepdim=True) + 1e-6
        )
        huber = F.smooth_l1_loss(pred_z, target_z, beta=1.0)
        kl = F.kl_div(
            F.log_softmax(-scores, dim=1),
            F.softmax(-target, dim=1),
            reduction="batchmean",
        )
        return huber + 0.1 * kl

    rt.tensor_hash = tensor_hash
    rt.loss = loss
    return rt


def _make_toy_setup():
    net = _ToyNet()
    rt = _toy_runtime()
    pdict = dict(net.named_parameters())
    names = sorted(pdict)
    params = [pdict[n] for n in names]
    initial = {n: pdict[n].detach().cpu().clone() for n in names}
    initial_hash = rt.tensor_hash(initial)

    rng = np.random.default_rng(198001)
    m = 24
    pool = dict(
        program=rng.standard_normal((m, 412)).astype(np.float32),
        fitness=rng.standard_normal(m).astype(np.float64),
        gate=np.ones(m, dtype=bool),
    )
    pool["gate"][2] = False
    anchor_total = torch.as_tensor(rng.standard_normal(m), dtype=torch.float32)
    anchor_prior = torch.as_tensor(rng.standard_normal(m), dtype=torch.float32)
    tokens = torch.zeros((10, 32), dtype=torch.float32)
    masks = torch.ones((10,), dtype=torch.bool)

    def build_inputs(cid, fit_arr, gate_arr):
        del cid
        rev_f = np.asarray(fit_arr[:4], dtype=np.float32)
        rev_g = np.asarray(gate_arr[:4], dtype=bool)
        return dict(
            prog_412=torch.as_tensor(pool["program"][None, :, :], dtype=torch.float32),
            sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
            tokens=tokens.unsqueeze(0),
            masks=masks.unsqueeze(0),
            scale=torch.full((1,), 0.4208, dtype=torch.float32),
            revealed_indices=torch.arange(4, dtype=torch.long).reshape(1, 4),
            revealed_raw_f=torch.as_tensor(rev_f[None, :], dtype=torch.float32),
            revealed_gate=torch.as_tensor(rev_g[None, :], dtype=torch.float32),
            anchor_total=anchor_total.unsqueeze(0),
            anchor_prior=anchor_prior.unsqueeze(0),
        )

    return net, names, params, pdict, initial, initial_hash, rt, build_inputs, pool


def test_leakage_mask_poisoned_unrevealed_labels():
    net, names, params, pdict, initial, initial_hash, rt, build_inputs, pool = (
        _make_toy_setup()
    )
    assert verify_leakage_mask_invariance(
        net,
        names,
        params,
        pdict,
        initial,
        initial_hash,
        rt,
        build_inputs,
        "cid_test_01",
        pool,
    )

    # Also test extreme finite poisoning (+1e9 / -1e9) on unrevealed candidates 4..23
    clean_in = build_inputs("cid_test_01", pool["fitness"], pool["gate"])
    extreme_fit = pool["fitness"].copy()
    extreme_fit[4:] = 1e9
    extreme_in = build_inputs("cid_test_01", extreme_fit, pool["gate"])
    s_clean, _ = ttt_context_trajectory(
        net, names, params, pdict, initial, initial_hash, rt, clean_in, lr=1e-4
    )
    s_ext, _ = ttt_context_trajectory(
        net, names, params, pdict, initial, initial_hash, rt, extreme_in, lr=1e-4
    )
    for k in K_GRID:
        assert np.array_equal(s_clean[k]["costs"], s_ext[k]["costs"])
        assert s_clean[k]["pick"] == s_ext[k]["pick"]
        assert s_clean[k]["drift"] == s_ext[k]["drift"]

    # Passing full-pool fitness or gate into context_inputs must be rejected immediately
    leaked_in = dict(clean_in, fitness=pool["fitness"])
    with pytest.raises(ValueError, match="unrevealed fitness/gate"):
        ttt_context_trajectory(
            net, names, params, pdict, initial, initial_hash, rt, leaked_in, lr=1e-4
        )

    # Passing non-[0,1,2,3] revealed indices into revealed_live_loss must be rejected
    bad_rev = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)
    with pytest.raises(ValueError, match="revealed_indices must be"):
        revealed_live_loss(
            torch.zeros((1, 24)),
            bad_rev,
            clean_in["revealed_raw_f"],
            clean_in["revealed_gate"],
            rt.loss,
        )


def test_per_context_reset_and_trajectory_prefix_equivalence():
    net, names, params, pdict, initial, initial_hash, rt, build_inputs, pool = (
        _make_toy_setup()
    )
    c_in = build_inputs("cid_prefix", pool["fitness"], pool["gate"])

    snaps_all, steps = ttt_context_trajectory(
        net,
        names,
        params,
        pdict,
        initial,
        initial_hash,
        rt,
        c_in,
        lr=1e-4,
        k_steps=(1, 3, 5),
        arm="TTT_K",
    )
    assert steps == 5
    assert rt.tensor_hash({n: pdict[n] for n in names}) == initial_hash
    assert all(p.grad is None for p in params)

    for k in (1, 3, 5):
        snaps_single, st_k = ttt_context_trajectory(
            net,
            names,
            params,
            pdict,
            initial,
            initial_hash,
            rt,
            c_in,
            lr=1e-4,
            k_steps=(k,),
            arm="TTT_K",
        )
        assert st_k == k
        assert np.array_equal(snaps_all[k]["costs"], snaps_single[k]["costs"])
        assert snaps_all[k]["pick"] == snaps_single[k]["pick"]
        assert snaps_all[k]["drift"] == snaps_single[k]["drift"]

    # L2-anchored TTT_L2 (beta=10.0) curbed multi-step drift relative to unanchored TTT_K
    snaps_l2, _ = ttt_context_trajectory(
        net,
        names,
        params,
        pdict,
        initial,
        initial_hash,
        rt,
        c_in,
        lr=1e-4,
        k_steps=(1, 3, 5),
        arm="TTT_L2",
        beta=L2_BETA,
    )
    assert snaps_l2[5]["drift"] < snaps_all[5]["drift"]
    assert rt.tensor_hash({n: pdict[n] for n in names}) == initial_hash


def test_shuf_derangement_properties():
    assert len(DERANGEMENTS_4) == 9
    for p in DERANGEMENTS_4:
        assert sorted(p) == [0, 1, 2, 3]
        assert all(p[i] != i for i in range(4))
    seen = set()
    for cid_i in range(50):
        perm = shuf_permutation(198101, f"ctx_{cid_i}")
        assert perm in DERANGEMENTS_4
        assert all(perm[i] != i for i in range(4))
        seen.add(perm)
    assert len(seen) >= 6


def test_independent_fold_seeds_and_stage1_b_only_lock(tmp_path):
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    fams = [f"fam_{i:03d}" for i in range(205)]
    rows = []
    for fi, fam in enumerate(fams):
        for c in range(2):
            rows.append(dict(context_sha256=f"{fam}_c{c}", family_id=fam))

    folds_by_seed = {
        s: stat.family_folds(fams, seed=stat.FOLD_SEEDS[s]) for s in stat.SEEDS
    }
    assert len({json.dumps(v) for v in folds_by_seed.values()}) == 3
    jacc = stat.fold_jaccards(folds_by_seed)
    assert len(jacc) == 3
    for pair_stat in jacc.values():
        assert 0.0 < pair_stat["diagonal_mean"] < 0.25

    # Verify fold_manifest.json matches stat.family_folds on actual cohort family IDs
    all_manifest_fams = sorted(
        {f for fold in fold_doc["families_by_seed"]["198101"] for f in fold}
    )
    assert len(all_manifest_fams) == 205
    for s in stat.SEEDS:
        assert fold_doc["families_by_seed"][str(s)] == stat.family_folds(
            all_manifest_fams, seed=stat.FOLD_SEEDS[s]
        )

    # Stage-1 lock invariance to A and FIT perturbations
    rng = np.random.default_rng(19899)
    grid_u = {
        r["context_sha256"]: rng.standard_normal(len(GRID)) for r in rows
    }
    lock1 = stat.stage1_lock(rows, grid_u, folds_by_seed, "foldsha", "initsha")

    # Perturb A and FIT contexts for cell (198101, outer=0)
    b_fams_0 = set(folds_by_seed[198101][1])
    b_rows_0 = [r for r in rows if r["family_id"] in b_fams_0]
    grid_u_perturbed = {
        cid: (vec.copy() if any(cid.startswith(f + "_") for f in b_fams_0) else vec + 500.0 * np.arange(len(GRID)))
        for cid, vec in grid_u.items()
    }
    lock_pert = stat.stage1_lock(rows, grid_u_perturbed, folds_by_seed, "foldsha", "initsha")
    assert lock1["cells"][0] == lock_pert["cells"][0]

    # Tie-breaking selects index 0 (smallest k=1, smallest lr=1e-5)
    grid_tie = {r["context_sha256"]: np.ones(len(GRID)) for r in b_rows_0}
    c_tie = stat.select_cell(b_rows_0, grid_tie)
    assert c_tie["grid_index"] == 0
    assert c_tie["k_star"] == K_GRID[0]
    assert c_tie["lr_star"] == LR_GRID[0]

    lock_path = tmp_path / "ttt_selection_lock.json"
    raw_bytes = stat.lock_bytes(lock1)
    lock_path.write_bytes(raw_bytes)
    verified, sha_val = stat.verify_stage1_lock(lock_path, "foldsha", "initsha")
    assert sha_val == hashlib.sha256(raw_bytes).hexdigest()
    assert verified["cells"] == lock1["cells"]

    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        stat.verify_stage1_lock(lock_path, "wrong_fold_sha", "initsha")


def test_verdict_gates_and_axis0_bootstrap():
    rows = [
        dict(context_sha256=f"c_{i}", family_id=f"f_{i}") for i in range(205)
    ]
    base = np.ones((3, 205), dtype=np.float64)
    # Construct synthetic TTT_K with positive gain +0.02 and 0% harm
    ttt_k = base + 0.02
    ttt_shuf = base + 0.001

    st_k = stat.contrast_stats(rows, ttt_k, base)
    st_shuf = stat.contrast_stats(rows, ttt_shuf, base)
    st_diff = stat.contrast_stats(rows, ttt_k, ttt_shuf)
    sel = stat.pooled_select(rows, base, {"TTT_K": ttt_k})

    v_pass = stat.verdict(st_k, st_shuf, st_diff, sel, h178_ok=True)
    assert v_pass["verdict"] == "ADVANCE"

    # If TTT_SHUF matches TTT_K, placebo gate must reject
    st_same_diff = stat.contrast_stats(rows, ttt_k, ttt_k)
    v_shuf_fail = stat.verdict(st_k, st_k, st_same_diff, sel, h178_ok=True)
    assert v_shuf_fail["verdict"] == "NO_ADVANCE"
    assert not v_shuf_fail["gates"]["ttt_k_gt_ttt_shuf"]

    # If harm > 10% (e.g. 30/205 families lose -0.05 while 175/205 gain +0.05), harm gate must reject
    ttt_harm = base.copy()
    ttt_harm[:, :30] -= 0.05
    ttt_harm[:, 30:] += 0.05
    st_harm = stat.contrast_stats(rows, ttt_harm, base)
    st_harm_diff = stat.contrast_stats(rows, ttt_harm, ttt_shuf)
    sel_harm = stat.pooled_select(rows, base, {"TTT_K": ttt_harm})
    v_harm_fail = stat.verdict(st_harm, st_shuf, st_harm_diff, sel_harm, h178_ok=True)
    assert v_harm_fail["verdict"] == "NO_ADVANCE"
    assert not v_harm_fail["gates"]["ttt_k_harm_le_010"]


def test_validate_launch_contract():
    good = dict(
        manifest_sha256="m_sha",
        merged_source_verified=True,
        source_commit="abc1234",
        review_id=12345,
        prereg_pr=1600,
        prereg_sha256="p_sha",
        fold_manifest_sha256="f_sha",
    )
    validate_launch(good, "abc1234", "m_sha", "p_sha", "f_sha", smoke=False)
    with pytest.raises(RuntimeError, match="launch manifest binding"):
        validate_launch(good, "abc1234", "wrong_m", "p_sha", "f_sha", smoke=True)
    with pytest.raises(RuntimeError, match="external merged/reviewed"):
        validate_launch(
            dict(good, prereg_pr=0), "abc1234", "m_sha", "p_sha", "f_sha", smoke=False
        )
