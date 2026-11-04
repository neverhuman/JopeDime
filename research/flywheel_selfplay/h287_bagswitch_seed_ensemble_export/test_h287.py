"""Unit tests for H287 ensemble and analysis_h287."""

import numpy as np
import pytest

import analysis_h287 as stat
import ensemble as ens
import runner


def test_family_folds_partition_exact():
    rows = [{"context_sha256": f"c{i:04d}", "family_id": f"f{i % 205:03d}"} for i in range(1219)]
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in rows}
    folds = ens.family_folds(rows, fam_by_cid, seed=279101, n=4)
    assert folds[0] == []
    seen_cids = []
    seen_fams = set()
    for f in (1, 2, 3, 4):
        f_fams = {fam_by_cid[c] for c in folds[f]}
        assert seen_fams.isdisjoint(f_fams)
        seen_fams |= f_fams
        seen_cids.extend(folds[f])
    assert len(seen_fams) == 205
    assert sorted(seen_cids) == sorted(fam_by_cid.keys())


def test_clip_packets_clips_invalid_at_minus_one():
    packets = {
        "c1": {
            "utility_k": np.array([1.2, -2.0, 0.5, -2.0], dtype=np.float64),
            "r882_u": -2.0,
        }
    }
    clipped = ens.clip_packets(packets, clip_floor=-1.0)
    assert np.allclose(clipped["c1"]["utility_k"], [1.2, -1.0, 0.5, -1.0])
    assert clipped["c1"]["r882_u"] == -1.0


def test_vote_consensus_and_ensemble_aggregation():
    # 4 contexts, K=8 candidates
    n_ctx, k = 4, 8
    m_outs = []
    # Context 0: all 3 members pick candidate 3
    # Context 1: 2 members pick candidate 2, 1 picks 0 -> ENS_VOTE2=2, ENS_VOTE3=0
    # Context 2: 2 members switch but disagree (candidate 1 vs candidate 4) -> ENS_VOTE2=0, ENS_VOTE3=0
    # Context 3: all 3 members pick 0
    picks_by_member = [
        [3, 2, 1, 0],
        [3, 2, 4, 0],
        [3, 0, 0, 0],
    ]
    for m in range(3):
        p_inv = np.zeros((n_ctx, k), dtype=np.float64)
        raw_sc = np.zeros((n_ctx, k), dtype=np.float64)
        bag_sc = np.zeros((8, n_ctx, k), dtype=np.float64)
        # Context 0: strong margin on candidate 3 across all members and all bags
        raw_sc[0, 3] = 0.25
        bag_sc[:, 0, 3] = 0.25
        # Context 1: candidate 2 has veto prob >= 0.10 on mean
        p_inv[1, 2] = 0.15
        raw_sc[1, 2] = 0.30
        bag_sc[:, 1, 2] = 0.30
        m_outs.append(
            {
                "p_inv": p_inv,
                "raw_scores": raw_sc,
                "bag_scores": bag_sc,
                "j_ref": picks_by_member[m],
                "votes": np.array([8, 6, 6, 0], dtype=np.int32),
                "picks_bag": picks_by_member[m],
                "margin": 0.10,
            }
        )

    agg = ens.aggregate_ensemble_picks(m_outs, n_ctx, q_fixed=6, tau=0.10)
    assert agg["ENS_VOTE2"] == [3, 2, 0, 0]
    assert agg["ENS_VOTE3"] == [3, 0, 0, 0]
    assert agg["ENS_MEAN"] == [3, 0, 0, 0]


def test_audit_hook_blocks_fresh_and_sealed_paths():
    for bad in (
        "/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H283.json",
        "/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json",
        "/tmp/fresh_top8_packets.npz",
        "/data/sealed/test.npz",
    ):
        with pytest.raises(RuntimeError):
            runner.audit_hook("open", (bad,))


def test_decide_h287_gate():
    fams = [f"fam_{i:03d}" for i in range(50)]
    panel = np.zeros((3, 50, len(stat.ALL_ARMS)), dtype=np.float64)
    col_bag = stat.ALL_ARMS.index("BAG_SWITCH")
    col_ens = stat.ALL_ARMS.index("ENS_MEAN")
    # BAG_SWITCH: 6 harmed families (< -0.01), +0.04 mean gain
    panel[:, :6, col_bag] = -0.02
    panel[:, 6:, col_bag] = 0.05
    # ENS_MEAN: 4 harmed families (<= 6), +0.05 mean gain, all 3 seeds LCB95 > 0
    panel[:, :4, col_ens] = -0.02
    panel[:, 4:, col_ens] = 0.058

    inv = {a: [2, 2, 2] for a in stat.ALL_ARMS}
    chg = {a: [0, 0, 0] if a == "slot0" else [20, 20, 20] for a in stat.ALL_ARMS}
    s = stat.summarize_panel(panel, fams, inv, chg)
    dec = stat.decide_h287(s)
    assert dec["verdict"] == "EXPORTED_ADVANCE"
    assert "ENS_MEAN" in dec["advancing_ens_arms"]
