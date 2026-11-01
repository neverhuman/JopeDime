"""Data-free unit tests for H190 ENS10 confirmation with independent fold seeds per outer seed."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

import analysis
import monitor
import runner
import treatment

HERE = Path(__file__).resolve().parent
H178_RECEIPT_PATH = HERE.parent / "h178_pooled_calibration" / "h178_receipt.json"


def _synthetic_rows():
    rows = []
    cid = 0
    for f in range(205):
        fam_id = f"fam-{f:04d}"
        n_ctx = 6 if f < 194 else 5
        for _ in range(n_ctx):
            rows.append(
                {
                    "context_sha256": f"cid-{cid:04d}",
                    "family_id": fam_id,
                }
            )
            cid += 1
    assert len(rows) == 1219
    return rows


class _ToyAdapter(nn.Module):
    def __init__(
        self,
        frozen_anchor=None,
        prog_mean=None,
        prog_std=None,
        d_token=32,
        n_modalities=10,
        modality_dropout_p=0.0,
    ):
        super().__init__()
        self.anchor = frozen_anchor
        self.prog_mean = (
            torch.zeros(4) if prog_mean is None else torch.as_tensor(prog_mean)
        )
        self.prog_std = torch.ones(4) if prog_std is None else torch.as_tensor(prog_std)
        self.d_token = d_token
        self.n_modalities = n_modalities
        self.modality_dropout_p = modality_dropout_p
        # 47 trainable tensors / 29,211 parameters to match TENSORS=47, PARAMETERS=29211
        self.tensors_list = nn.ParameterList(
            [nn.Parameter(torch.randn(621)) for _ in range(46)]
            + [nn.Parameter(torch.randn(645))]
        )


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_10tok = _ToyAdapter()


def _state_sha256(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_independent_fold_partitions_differ_and_cover_205_families():
    rows = _synthetic_rows()
    proof = analysis.verify_independent_fold_partitions(rows)
    assert proof["independent_partitions_verified"] is True
    assert proof["fold_seeds_by_seed"] == {
        "190101": 190141,
        "190102": 190142,
        "190103": 190143,
    }

    partitions_by_seed = {}
    for seed in analysis.SEEDS:
        fseed = analysis.fold_seed_for(seed)
        cov = proof["per_seed_coverage"][str(seed)]
        assert cov["fold_seed"] == fseed
        assert cov["total_families"] == 205
        assert cov["total_contexts"] == 1219
        assert cov["b_exact_once"] is True
        assert cov["a_exact_once"] is True
        assert cov["zero_fit_b_overlap"] is True
        assert cov["zero_b_a_same_fold_overlap"] is True

        partitions_by_seed[seed] = analysis.folds(rows, fseed)
        for outer in range(5):
            fit_rows, b_rows, a_rows = analysis.partition(rows, outer, fold_seed=fseed)
            fit_fams = {r["family_id"] for r in fit_rows}
            b_fams = {r["family_id"] for r in b_rows}
            a_fams = {r["family_id"] for r in a_rows}
            assert len(fit_fams) == 123
            assert len(b_fams) == 41
            assert len(a_fams) == 41
            assert not (fit_fams & b_fams)
            assert not (fit_fams & a_fams)
            assert not (b_fams & a_fams)

            fit_cids = [r["context_sha256"] for r in fit_rows]
            b_cids = {r["context_sha256"] for r in b_rows}
            a_cids = {r["context_sha256"] for r in a_rows}
            orders = {c: np.arange(30) for c in fit_cids}
            plan = analysis.batch_plan(fit_cids, orders, seed=seed + outer, updates=12)
            assert len(plan) == 12
            for stream, cids_step, ixs_step in plan:
                assert isinstance(stream, int)
                assert len(cids_step) == 8
                assert len(set(cids_step)) == 8
                assert not (set(cids_step) & b_cids)
                assert not (set(cids_step) & a_cids)
                for ix in ixs_step:
                    assert len(ix) == 24
                    assert list(ix[:4]) == [0, 1, 2, 3]

    # Verify explicitly that all 3 outer seeds have different fold partitions on every fold 0..4
    for i, s1 in enumerate(analysis.SEEDS):
        for s2 in analysis.SEEDS[i + 1 :]:
            assert partitions_by_seed[s1] != partitions_by_seed[s2]
            for outer in range(5):
                fit1, b1, a1 = analysis.partition(
                    rows, outer, fold_seed=analysis.fold_seed_for(s1)
                )
                fit2, b2, a2 = analysis.partition(
                    rows, outer, fold_seed=analysis.fold_seed_for(s2)
                )
                assert {r["family_id"] for r in fit1} != {r["family_id"] for r in fit2}
                assert {r["family_id"] for r in b1} != {r["family_id"] for r in b2}
                assert {r["family_id"] for r in a1} != {r["family_id"] for r in a2}


def test_stage_order_guard_and_pilot_fallback(tmp_path):
    guard = analysis.StageOrderGuard()
    assert guard.expected_steps == {
        "ENS10": 40,
        "RENS10": 0,
    }
    assert guard.ensemble_k == 10
    guard.mark_b_read()
    assert guard.b_read_started is True
    with pytest.raises(RuntimeError, match="before B is read"):
        guard.set_candidate_step("ENS10", 20)
    with pytest.raises(RuntimeError, match="forbidden before pooled_b_selection_lock"):
        guard.require_b_locked()

    lock_file = tmp_path / "pooled_b_selection_lock.json"
    with pytest.raises(RuntimeError, match="does not exist"):
        guard.mark_b_locked(lock_file)

    rows = _synthetic_rows()
    base = np.zeros((3, 1219), dtype=np.float64)
    decisions = analysis.pooled_b_select_h190(
        rows,
        base,
        {
            "ENS10": base.copy(),
            "RENS10": base.copy(),
        },
    )
    payload = dict(
        wave="H190",
        pooled_b_families=205,
        pooled_b_contexts=1219,
        fold_seeds_by_seed={"190101": 190141, "190102": 190142, "190103": 190143},
        locked_before_a_confirmation=True,
        decisions=decisions,
    )
    runner.save(lock_file, payload)
    guard.mark_b_locked(lock_file)
    locked = guard.require_b_locked()
    assert locked["wave"] == "H190"
    assert locked["locked_before_a_confirmation"] is True

    pilot_ok = analysis.execution_plan(0.08, cap_seconds=9000, reserve_seconds=900)
    assert pilot_ok["active_ensemble_k"] == 10
    assert pilot_ok["total_optimizer_updates"] == 6000
    assert pilot_ok["fallback_triggered"] is False

    pilot_fb = analysis.execution_plan(1.50, cap_seconds=9000, reserve_seconds=900)
    assert pilot_fb["active_ensemble_k"] == 5
    assert pilot_fb["total_optimizer_updates"] == 3000
    assert pilot_fb["fallback_triggered"] is True


def test_rens10_inits_match_ens10_and_member_seeds_disjoint():
    assert analysis.validate_member_seed_disjointness() is True
    assert analysis.member_seeds_for(190101) == tuple(range(190201, 190211))
    assert analysis.member_seeds_for(190102) == tuple(range(190301, 190311))
    assert analysis.member_seeds_for(190103) == tuple(range(190401, 190411))
    assert analysis.member_seeds_for(190101, k=5) == tuple(range(190201, 190206))

    net = _ToyNet()
    names = [f"adapter_10tok.{n}" for n, _ in net.adapter_10tok.named_parameters()]
    r882_hash = _state_sha256(runner.adapter_state(net, names))

    for seed in analysis.SEEDS:
        m_seeds = analysis.member_seeds_for(seed, k=10)
        ens_hashes = []
        for m_seed in m_seeds:
            st = runner.random_adapter_state(net, names, m_seed)
            ens_hashes.append(_state_sha256(st))
        assert len(analysis.validate_member_init_hashes(ens_hashes, k=10)) == 10
        assert (
            analysis.validate_rens_matches_ens_inits(list(ens_hashes), ens_hashes, k=10)
            is True
        )
        cell_hashes = analysis.validate_cell_init_hashes(r882_hash, ens_hashes, k=10)
        assert len(cell_hashes) == 11

        with pytest.raises(ValueError, match="do not match"):
            analysis.validate_rens_matches_ens_inits(
                ens_hashes[:-1] + ["0" * 64],
                ens_hashes,
                k=10,
            )

    member_scores = np.ones((10, 4), dtype=np.float64)
    member_scores[0] = [0.2, 0.9, 0.5, 0.8]
    avg = analysis.score_ensemble(member_scores, k=10)
    assert np.allclose(avg, np.mean(member_scores, axis=0))
    with pytest.raises(ValueError, match="expected finite"):
        analysis.score_ensemble(member_scores[:9], k=10)


def test_imported_h177_decision_rule_holm_m2_and_h178_verdict():
    assert analysis.H177_DECISION_RULE is analysis.h177_analysis.H177_DECISION_RULE
    assert analysis.pooled_b_select is analysis.h177_analysis.pooled_b_select
    assert analysis.resolve_h177_verdict is analysis.h177_analysis.resolve_h177_verdict

    rows = _synthetic_rows()
    rng = np.random.default_rng(190999)
    base = rng.normal(1.2, 0.1, size=(3, 1219))

    null_sel = analysis.pooled_b_select_h190(
        rows,
        base,
        {
            "ENS10": base.copy(),
            "RENS10": base.copy(),
        },
    )
    for arm in analysis.CANDIDATES:
        assert null_sel[arm]["admitted"] is False
        assert null_sel[arm]["mean_gain"] == 0.0
        assert null_sel[arm]["holm_multiplicity"] == 2

    tie_sel = analysis.pooled_b_select_h190(
        rows,
        base,
        {
            "ENS10": base + 0.001,
            "RENS10": base.copy(),
        },
    )
    assert tie_sel["ENS10"]["admitted"] is False

    pos_sel = analysis.pooled_b_select_h190(
        rows,
        base,
        {
            "ENS10": base + 0.03,
            "RENS10": base.copy(),
        },
    )
    assert pos_sel["ENS10"]["admitted"] is True
    assert pos_sel["RENS10"]["admitted"] is False

    h178_receipt = json.loads(H178_RECEIPT_PATH.read_text(encoding="utf-8"))
    confirm_a_pass = {
        "contrasts": {
            "ENS10_minus_R882": {
                "mean_gain": 0.025,
                "median_gain": 0.024,
                "ci95": [0.01, 0.04],
                "median_seed_ci95": [0.008, 0.039],
                "positive_seeds": 3,
            },
            "RENS10_minus_R882": {
                "mean_gain": -0.002,
                "median_gain": -0.002,
                "ci95": [-0.01, 0.005],
                "median_seed_ci95": [-0.01, 0.005],
                "positive_seeds": 1,
            },
        }
    }
    verdict_ok = analysis.resolve_h190_verdict(pos_sel, confirm_a_pass, h178_receipt)
    assert verdict_ok["verdict"] == "ADVANCE"
    assert verdict_ok["lift_qualified"] is True
    assert verdict_ok["qualifying_arms"] == ["ENS10"]

    # Reject if median_seed_ci95 lower bound <= 0 even when 3/3 seeds are positive
    confirm_a_wide_ci = {
        "contrasts": {
            "ENS10_minus_R882": {
                "mean_gain": 0.015,
                "median_gain": 0.014,
                "ci95": [0.001, 0.03],
                "median_seed_ci95": [-0.002, 0.03],
                "positive_seeds": 3,
            },
            "RENS10_minus_R882": {
                "mean_gain": 0.0,
                "median_gain": 0.0,
                "ci95": [0.0, 0.0],
                "median_seed_ci95": [0.0, 0.0],
                "positive_seeds": 0,
            },
        }
    }
    verdict_no = analysis.resolve_h190_verdict(pos_sel, confirm_a_wide_ci, h178_receipt)
    assert verdict_no["verdict"] == "NO_ADVANCE"
    assert verdict_no["qualifying_arms"] == []


def test_axis0_bootstrap_partition_variance_and_disagreement():
    rows = _synthetic_rows()
    fams = sorted({r["family_id"] for r in rows})
    fam_to_idx = {f: i for i, f in enumerate(fams)}
    rng = np.random.default_rng(190777)

    base = np.zeros((3, 1219), dtype=np.float64)
    base[0, 0] = -2.0
    base[1, 5] = -2.0

    cand = np.zeros((3, 1219), dtype=np.float64)
    s1_fam = 0.02 + rng.normal(0.0, 0.001, size=205)
    for j, r in enumerate(rows):
        fi = fam_to_idx[r["family_id"]]
        cand[0, j] = base[0, j] - 0.20 + 0.001 * (fi % 5)
        cand[1, j] = base[1, j] + s1_fam[fi]
        cand[2, j] = base[2, j] + 0.25 + 0.001 * (fi % 7)

    panel_3 = np.stack([base, cand, base.copy()], axis=-1)
    summary = analysis.summarize_panel(rows, panel_3, seed=analysis.BOOT_SEED)
    e_con = summary["contrasts"]["ENS10_minus_R882"]
    assert abs(e_con["median_gain"] - 0.02) < 0.002
    ci_lo, ci_hi = e_con["median_seed_ci95"]
    assert 0.015 < ci_lo <= e_con["median_gain"] <= ci_hi < 0.025
    assert summary["arms"]["R882"]["minus2_contexts_per_seed"] == [1, 1, 0]
    assert summary["arms"]["ENS10"]["minus2_contexts_per_seed"] == [0, 0, 0]
    assert "ENS10_minus_RENS10" in summary["contrasts"]

    pvar = summary["partition_variance"]["ENS10_minus_R882"]
    assert pvar["between_seed_sd_ddof1"] > 0.20
    assert pvar["within_family_between_seed_sd_ddof1"] > 0.20

    choices = np.zeros((3, 10, 10), dtype=np.int64)
    choices[:, :, 5:] = 1
    dis = analysis.pairwise_member_disagreement(choices, k=10)
    # 5 zeros and 5 ones -> 25 disagreeing pairs out of 45
    assert abs(dis["mean_over_seeds"] - (25.0 / 45.0)) < 1e-12


def test_monitor_and_launch_guards():
    good_sample = dict(
        load=12.0,
        cpu=30.0,
        available=32 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.94,
        rss=3_000_000_000,
        threads=4,
        output_bytes=50_000_000,
    )
    assert monitor.violations(good_sample) == []
    bad_sample = dict(
        good_sample, free=75_000_000_000, threads=9, output_bytes=2_100_000_000
    )
    v = monitor.violations(bad_sample)
    assert "free<80GB" in v
    assert "aggregate threads>8" in v
    assert "output_dir>=2GB" in v

    with pytest.raises(RuntimeError, match="launch manifest binding mismatch"):
        treatment.validate_launch(
            {"manifest_sha256": "wrong"},
            "head",
            "man",
            "pre",
            "fold",
            "h178",
            smoke=False,
        )
    assert (
        treatment.validate_launch(
            {"manifest_sha256": "man"},
            "head",
            "man",
            "pre",
            "fold",
            "h178",
            smoke=True,
        )
        is None
    )
