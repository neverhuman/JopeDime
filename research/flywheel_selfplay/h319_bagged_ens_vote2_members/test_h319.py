"""Data-free H319 controls: the bag RNG formula and draw order, the cluster-bootstrap relabel, the 9-member vote
(6/9 for a=2, 9/9 for a=3, ties -> slot 0), leave-one-family-out, seed-Jaccard and the decision rule."""

import numpy as np

import evaluate_h319 as ev
import run_h319 as r


def test_bag_rng_formula_and_draws():
    fams = {f"f{i}": None for i in range(20)}
    d1 = r.bag_draws(fams, "P1", 2, 30002)
    d2 = r.bag_draws(fams, "P1", 2, 30002)
    assert d1 == d2 and len(d1) == 9
    rng = np.random.default_rng(319190 + 1000 * 1 + 30002 + 10 * 2)
    first = [str(f) for f in rng.choice(sorted(fams), size=20, replace=True)]
    assert d1[0][0] == first and d1[0][1] == int(rng.integers(1, 2**31 - 1))
    assert r.bag_draws(fams, "P2", 2, 30002) != d1


def test_relabel_copies_are_distinct_families():
    by_fam = {"a": [{"context_sha256": "c1", "family_id": "a"}], "b": [{"context_sha256": "c2", "family_id": "b"}]}
    rows = r.relabel(["a", "a", "b"], by_fam)
    assert [x["context_sha256"] for x in rows] == ["c1#0", "c1#1", "c2#0"]
    assert [x["family_id"] for x in rows] == ["a#0", "a#1", "b#0"] and rows[1]["_src"] == "c1"


def test_vote_thresholds_and_ties():
    six = [[3]] * 6 + [[0]] * 3
    five = [[3]] * 5 + [[0]] * 4
    assert r.vote(six, 2) == [3] and r.vote(five, 2) == [0]
    assert r.vote([[3]] * 8 + [[0]], 3) == [0] and r.vote([[3]] * 9, 3) == [3]
    tie = [[3]] * 4 + [[5]] * 4 + [[0]]
    assert r.vote(tie, 2) == [0]
    mixed = [[3]] * 6 + [[5]] * 3
    assert r.vote(mixed, 2) == [3]


def test_lofo_jaccard_decision():
    assert abs(ev.lofo_min(np.array([0.3, 0.0, 0.0])) - 0.0) < 1e-12
    picks = np.array([[1, 0, 2, 0], [1, 0, 0, 0], [1, 2, 2, 0]])
    j = ev.seed_jaccard(picks, [0, 0, 0, 1])
    assert abs(j[0] - np.mean([1 / 2, 2 / 3, 1 / 3])) < 1e-12 and j[1] == 1.0
    good = dict(median=0.002, lcb95=0.0005)
    assert ev.decide(good, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9, [0.5] * 12, [0.4] * 12)[1] == "BAG_ADVANCE"
    assert (
        ev.decide(good, [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9, [0.4] * 12, [0.4] * 12)[1] == "BAG_NOT_ADVANCED"
    )
    assert (
        ev.decide(dict(median=0.0, lcb95=0.0), [0.05] * 9, [0.05] * 9, [0.08] * 9, [0.09] * 9, [0.5] * 12, [0.4] * 12)[
            1
        ]
        == "BAG_NOT_ADVANCED"
    )
