"""Unit tests for H284 freeze_bagged and analysis_h284."""

import numpy as np

import analysis_h284 as stat
import bagged
import freeze_bagged as fb


def test_family_folds_partition_exact():
    rows = [{"context_sha256": f"c{i:04d}", "family_id": f"f{i % 205:03d}"} for i in range(1219)]
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in rows}
    folds = fb.family_folds(rows, fam_by_cid, seed=279101, n=4)
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


def test_wtl_ties_are_not_wins():
    d = np.array([0.0, 5e-5, -5e-5, 2e-4, -3e-4], dtype=np.float64)
    wtl = stat.wtl_counts(d)
    assert wtl == {"W": 1, "T": 3, "L": 1}


def test_vote_switching_rule():
    scores = np.zeros((8, 2, 8), dtype=np.float64)
    # Context 0: 6 of 8 bags rank slot 3 above slot 0 by > 0.1
    scores[:6, 0, 3] = 0.25
    # Context 1: 5 of 8 bags rank slot 2 above slot 0 by > 0.1
    scores[:5, 1, 2] = 0.25
    votes = bagged.vote_counts(scores, [3, 2], margin=0.1)
    assert votes.tolist() == [6, 5]
    assert bagged.switch_by_votes([3, 2], votes, 6) == [3, 0]
    assert bagged.switch_by_votes([3, 2], votes, 5) == [3, 2]


def test_decide_fresh_gates():
    fams = [f"fam_{i:02d}" for i in range(40)]
    panel = np.zeros((3, 40, len(stat.ALL_ARMS)), dtype=np.float64)
    # RERANK8_VETO (col 1) and H278_RERANK8_VETO (col 2): 8 harmed families (< -0.01), +0.025 mean gain
    for col in (1, 2):
        panel[:, :8, col] = -0.03
        panel[:, 8:, col] = 0.04
    # BAG_SWITCH (col 3) and BAG_SWITCH_NESTED (col 4): 4 harmed families (<= 0.75 * 8 = 6), +0.025 mean gain
    for col in (3, 4):
        panel[:, :4, col] = -0.02
        panel[:, 4:, col] = 0.035

    inv = {a: [0, 0, 0] for a in stat.ALL_ARMS}
    chg = {a: [0, 0, 0] if a == "slot0" else [20, 20, 20] for a in stat.ALL_ARMS}
    s = stat.summarize_pool_panel(panel, fams, inv, chg)
    dec = stat.decide_fresh(s)
    assert dec["verdict"] == "FRESH_ADVANCE"
    assert dec["BAG_SWITCH"]["pass_all_vs_paired_veto"] is True
    assert dec["BAG_SWITCH"]["pass_all_vs_h278_veto"] is True
