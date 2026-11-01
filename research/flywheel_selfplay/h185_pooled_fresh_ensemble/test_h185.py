"""Data-free unit tests for H185 power-aware pooled fresh-init & ensemble confirmatory test."""

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


def test_fit_only_training_and_fold_coverage():
    rows = _synthetic_rows()
    cov = analysis.verify_pooled_b_coverage(rows, fold_seed=analysis.FOLD_SEED)
    assert cov["total_families"] == 205
    assert cov["total_contexts"] == 1219
    assert cov["b_exact_once"] is True
    assert cov["a_exact_once"] is True
    assert cov["zero_fit_b_overlap"] is True
    assert cov["zero_b_a_same_fold_overlap"] is True
    assert cov["fold_seed"] == 185140

    for outer in range(5):
        fit_rows, b_rows, a_rows = analysis.partition(rows, outer)
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
        plan = analysis.batch_plan(fit_cids, orders, seed=185101 + outer, updates=16)
        assert len(plan) == 16
        for stream, cids_step, ixs_step in plan:
            assert isinstance(stream, int)
            assert len(cids_step) == 8
            assert len(set(cids_step)) == 8
            assert not (set(cids_step) & b_cids)
            assert not (set(cids_step) & a_cids)
            for ix in ixs_step:
                assert len(ix) == 24
                assert list(ix[:4]) == [0, 1, 2, 3]


def test_stage_order_guard_requires_b_lock_before_a(tmp_path):
    guard = analysis.StageOrderGuard()
    assert guard.expected_steps == {
        "FRESH160": 160,
        "ENS5": 40,
        "RENS5": 0,
    }
    guard.mark_b_read()
    assert guard.b_read_started is True
    with pytest.raises(RuntimeError, match="before B is read"):
        guard.set_candidate_step("FRESH160", 80)
    with pytest.raises(RuntimeError, match="forbidden before pooled_b_selection_lock"):
        guard.require_b_locked()

    lock_file = tmp_path / "pooled_b_selection_lock.json"
    with pytest.raises(RuntimeError, match="does not exist"):
        guard.mark_b_locked(lock_file)

    rows = _synthetic_rows()
    base = np.zeros((3, 1219), dtype=np.float64)
    decisions = analysis.pooled_b_select_h185(
        rows,
        base,
        {
            "FRESH160": base.copy(),
            "ENS5": base.copy(),
            "RENS5": base.copy(),
        },
    )
    payload = dict(
        wave="H185",
        pooled_b_families=205,
        pooled_b_contexts=1219,
        locked_before_a_confirmation=True,
        decisions=decisions,
    )
    runner.save(lock_file, payload)
    guard.mark_b_locked(lock_file)
    locked = guard.require_b_locked()
    assert locked["wave"] == "H185"
    assert locked["locked_before_a_confirmation"] is True

    pilot_ok = analysis.execution_plan(0.06, reserve_seconds=150.0)
    assert pilot_ok["fresh160_updates"] == 160
    assert pilot_ok["fallback_triggered"] is False
    pilot_fb = analysis.execution_plan(2.20, reserve_seconds=300.0)
    assert pilot_fb["fresh160_updates"] == 80
    assert pilot_fb["fallback_triggered"] is True


def test_rens5_inits_match_ens5_and_member_hashes_are_distinct():
    assert analysis.member_seeds_for(185101) == (
        185201,
        185202,
        185203,
        185204,
        185205,
    )
    assert analysis.member_seeds_for(185102) == (
        185301,
        185302,
        185303,
        185304,
        185305,
    )
    assert analysis.member_seeds_for(185103) == (
        185401,
        185402,
        185403,
        185404,
        185405,
    )

    net = _ToyNet()
    names = [f"adapter_10tok.{n}" for n, _ in net.adapter_10tok.named_parameters()]
    r882_hash = _state_sha256(runner.adapter_state(net, names))

    for seed in analysis.SEEDS:
        m_seeds = analysis.member_seeds_for(seed)
        ens_hashes = []
        for m_seed in m_seeds:
            st = runner.random_adapter_state(net, names, m_seed)
            ens_hashes.append(_state_sha256(st))
        assert len(analysis.validate_member_init_hashes(ens_hashes)) == 5
        assert (
            analysis.validate_rens5_matches_ens5_inits(list(ens_hashes), ens_hashes)
            is True
        )

        with pytest.raises(ValueError, match="do not match"):
            analysis.validate_rens5_matches_ens5_inits(
                ens_hashes[:-1] + ["0" * 64],
                ens_hashes,
            )

        for outer in range(5):
            f_seed = analysis.fresh160_init_seed(seed, outer)
            f_hash = _state_sha256(runner.random_adapter_state(net, names, f_seed))
            cell_hashes = analysis.validate_cell_init_hashes(
                r882_hash, f_hash, ens_hashes
            )
            assert len(cell_hashes) == 7
            with pytest.raises(ValueError, match="7 distinct hashes"):
                analysis.validate_cell_init_hashes(r882_hash, r882_hash, ens_hashes)

    member_scores = np.array(
        [
            [1.0, 0.2, 0.9],
            [0.8, 0.4, 0.3],
            [0.6, 0.3, 0.6],
            [0.6, 0.3, 0.6],
            [0.6, 0.3, 0.6],
        ],
        dtype=np.float64,
    )
    avg = analysis.score_ensemble(member_scores, k=5)
    assert np.allclose(avg, np.mean(member_scores, axis=0))
    with pytest.raises(ValueError, match="expected finite"):
        analysis.score_ensemble(member_scores[:4], k=5)


def test_imported_h177_decision_rule_and_h178_verdict():
    assert analysis.H177_DECISION_RULE is analysis.h177_analysis.H177_DECISION_RULE
    assert analysis.pooled_b_select is analysis.h177_analysis.pooled_b_select
    assert analysis.resolve_h177_verdict is analysis.h177_analysis.resolve_h177_verdict

    rows = _synthetic_rows()
    rng = np.random.default_rng(185999)
    base = rng.normal(1.2, 0.1, size=(3, 1219))

    null_sel = analysis.pooled_b_select_h185(
        rows,
        base,
        {
            "FRESH160": base.copy(),
            "ENS5": base.copy(),
            "RENS5": base.copy(),
        },
    )
    for arm in analysis.CANDIDATES:
        assert null_sel[arm]["admitted"] is False
        assert null_sel[arm]["mean_gain"] == 0.0

    tie_sel = analysis.pooled_b_select_h185(
        rows,
        base,
        {
            "FRESH160": base + 0.001,
            "ENS5": base.copy(),
            "RENS5": base.copy(),
        },
    )
    assert tie_sel["FRESH160"]["admitted"] is False

    pos_sel = analysis.pooled_b_select_h185(
        rows,
        base,
        {
            "FRESH160": base.copy(),
            "ENS5": base + 0.03,
            "RENS5": base + 0.025,
        },
    )
    assert pos_sel["ENS5"]["admitted"] is True
    assert pos_sel["RENS5"]["admitted"] is True
    assert pos_sel["FRESH160"]["admitted"] is False

    h178_receipt = json.loads(H178_RECEIPT_PATH.read_text(encoding="utf-8"))
    confirm_a_pass = {
        "contrasts": {
            "FRESH160_minus_R882": {
                "mean_gain": -0.002,
                "median_gain": -0.002,
                "ci95": [-0.01, 0.005],
                "median_seed_ci95": [-0.01, 0.005],
                "positive_seeds": 1,
            },
            "ENS5_minus_R882": {
                "mean_gain": 0.025,
                "median_gain": 0.024,
                "ci95": [0.01, 0.04],
                "median_seed_ci95": [0.01, 0.04],
                "positive_seeds": 3,
            },
            "RENS5_minus_R882": {
                "mean_gain": 0.021,
                "median_gain": 0.020,
                "ci95": [0.008, 0.035],
                "median_seed_ci95": [0.008, 0.035],
                "positive_seeds": 3,
            },
        }
    }
    verdict_ok = analysis.resolve_h185_verdict(pos_sel, confirm_a_pass, h178_receipt)
    assert verdict_ok["verdict"] == "ADVANCE"
    assert verdict_ok["lift_qualified"] is True
    assert verdict_ok["qualifying_arms"] == ["ENS5", "RENS5"]

    confirm_a_two_seeds = {
        "contrasts": {
            "FRESH160_minus_R882": {
                "mean_gain": 0.0,
                "median_gain": 0.0,
                "ci95": [0.0, 0.0],
                "median_seed_ci95": [0.0, 0.0],
                "positive_seeds": 0,
            },
            "ENS5_minus_R882": {
                "mean_gain": 0.010,
                "median_gain": 0.010,
                "ci95": [0.001, 0.02],
                "median_seed_ci95": [0.001, 0.02],
                "positive_seeds": 2,
            },
            "RENS5_minus_R882": {
                "mean_gain": -0.001,
                "median_gain": -0.001,
                "ci95": [-0.01, 0.005],
                "median_seed_ci95": [-0.01, 0.005],
                "positive_seeds": 1,
            },
        }
    }
    verdict_no = analysis.resolve_h185_verdict(
        pos_sel, confirm_a_two_seeds, h178_receipt
    )
    assert verdict_no["verdict"] == "NO_ADVANCE"
    assert verdict_no["qualifying_arms"] == []


def test_axis0_bootstrap_seed_reduction_and_minus_two_counts():
    rows = _synthetic_rows()
    fams = sorted({r["family_id"] for r in rows})
    fam_to_idx = {f: i for i, f in enumerate(fams)}
    rng = np.random.default_rng(185777)

    base = np.zeros((3, 1219), dtype=np.float64)
    base[0, 0] = -2.0
    base[1, 5] = -2.0

    # Construct heterogeneous seed gains so seed 1 is always the median seed across bootstrap draws:
    # Seed 0: large negative (-0.20), Seed 1: moderate positive (+0.02 ± tiny noise), Seed 2: large positive (+0.25)
    cand = np.zeros((3, 1219), dtype=np.float64)
    s1_fam = 0.02 + rng.normal(0.0, 0.001, size=205)
    for j, r in enumerate(rows):
        fi = fam_to_idx[r["family_id"]]
        cand[0, j] = base[0, j] - 0.20 + 0.001 * (fi % 5)
        cand[1, j] = base[1, j] + s1_fam[fi]
        cand[2, j] = base[2, j] + 0.25 + 0.001 * (fi % 7)

    panel_4 = np.stack([base, cand, cand, base.copy()], axis=-1)
    summary = analysis.summarize_panel(rows, panel_4, seed=analysis.BOOT_SEED)
    f_con = summary["contrasts"]["FRESH160_minus_R882"]
    assert abs(f_con["median_gain"] - 0.02) < 0.002
    ci_lo, ci_hi = f_con["median_seed_ci95"]
    # Under true axis=0 median reduction across the 3 seeds per draw, every draw picks seed 1 (~0.02),
    # so the 95% CI stays tight around +0.02 (and never spans [-0.20, +0.25] as an axis=1 bug would).
    assert 0.015 < ci_lo <= f_con["median_gain"] <= ci_hi < 0.025
    assert summary["arms"]["R882"]["minus2_contexts_per_seed"] == [1, 1, 0]
    assert summary["arms"]["FRESH160"]["minus2_contexts_per_seed"] == [0, 0, 0]
    assert "ENS5_minus_RENS5" in summary["contrasts"]
    assert "FRESH160_minus_RENS5" in summary["contrasts"]
    assert "ENS5_minus_FRESH160" in summary["contrasts"]


def test_monitor_and_launch_guards():
    good_sample = dict(
        load=12.0,
        cpu=30.0,
        available=64 * 2**30,
        free=80_000_000_000,
        disk_used_fraction=0.88,
        rss=2_000_000_000,
        threads=2,
    )
    assert monitor.violations(good_sample) == []
    bad_sample = dict(good_sample, disk_used_fraction=0.939, threads=5)
    v = monitor.violations(bad_sample)
    assert "disk>=93.8% user-visible used" in v
    assert "aggregate threads>4" in v

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
