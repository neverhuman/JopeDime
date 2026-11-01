"""Model-free controls use toy arrays only; no corpus/checkpoint access."""

import ast
from pathlib import Path
import numpy as np
import pytest
import protocol as p


def fixtures():
    rows = [dict(context_sha256=f"c{i}", family_id=f"f{i // 2}") for i in range(6)]
    payload = {
        r["context_sha256"]: (np.array([0.0, 1.0, 2.0]), np.array([-2.0, 2.0, 0.0]))
        for r in rows
    }
    return rows, payload


def test_fold_family_isolation():
    rows = [dict(context_sha256=str(i), family_id=f"f{i:03}") for i in range(205)]
    groups = p.folds(rows)
    assert [len(x) for x in groups] == [41] * 5
    for i in range(5):
        a, b = set(groups[i]), set(groups[(i + 1) % 5])
        assert not a & b and len(set.union(*map(set, groups)) - a - b) == 123


def test_injections_never_read_outside_role_and_A_cannot_change_B():
    rows, all_payload = fixtures()
    b = rows[:2]
    scope = {r["context_sha256"] for r in b}

    class LocalOnly(dict):
        def __getitem__(self, key):
            assert key in scope, "outside B read"
            return super().__getitem__(key)

    view = LocalOnly(all_payload)
    before = p.inject(b, view, scope, 175101, 0.2, "context", "oracle", "full")
    all_payload["c5"] = (np.array([0.0, 1.0, 2.0]), np.array([999.0, -999.0, 50.0]))
    after = p.inject(b, view, scope, 175101, 0.2, "context", "oracle", "full")
    assert np.array_equal(before["delta"], after["delta"])
    with pytest.raises(ValueError):
        p.inject(rows, view, scope, 175101, 0.2, "context", "oracle", "full")


def test_same_policy_reproduces_when_role_order_changes():
    rows, payload = fixtures()
    scope = {r["context_sha256"] for r in rows}
    a = p.inject(
        rows, payload, scope, 175101, 0.2, "context", "random_nonoracle", "full"
    )
    b = p.inject(
        rows[::-1], payload, scope, 175101, 0.2, "context", "random_nonoracle", "full"
    )
    assert np.array_equal(a["delta"], b["delta"][:, ::-1])


def test_family_clustered_flips_share_activation():
    rows, payload = fixtures()
    a = p.inject(rows, payload, set(payload), 175101, 0.2, "family", "oracle", "full")
    assert all(
        np.array_equal(a["activated"][:, i], a["activated"][:, i + 1])
        for i in [0, 2, 4]
    )


def test_fraction_masks_are_nested_and_zero_is_null():
    rows, payload = fixtures()
    args = (rows, payload, set(payload), 175101)
    a = p.inject(*args, 0.1, "context", "oracle", "full")
    b = p.inject(*args, 0.2, "context", "oracle", "full")
    z = p.inject(*args, 0, "context", "oracle", "full")
    assert np.all(a["activated"] <= b["activated"]) and not z["delta"].any()


def test_slate24_preserves_native_reveals_without_outcome_access():
    x = p.slate_indices("abc", 100, 175101, "24")
    assert len(x) == 24 and len(set(x)) == 24 and list(x[:4]) == [0, 1, 2, 3]
    assert np.array_equal(x, p.slate_indices("abc", 100, 175101, "24"))


def test_exact_scalar_family_bootstrap_replay():
    fm = np.random.default_rng(8).normal(0.004, 0.01, (7, 41))
    seed = 77
    indices = np.random.default_rng(seed).integers(0, 41, (1000, 41))
    g, ci, accept = p.select(fm, indices)
    for i, x in enumerate(fm):
        gold = np.quantile(x[indices].mean(axis=1), [0.025, 0.975])
        assert np.array_equal(ci[i], gold)
        assert g[i] == float(x.mean()) and accept[0, i] == (
            g[i] > 0.001 + p.EPS and gold[0] > 0
        )


def test_ties_and_minimum_gain_strict():
    ix = np.random.default_rng(5).integers(0, 41, (1000, 41))
    fm = np.array(
        [
            np.zeros(41),
            np.full(41, 0.001),
            np.full(41, 0.001 + p.EPS / 2),
            np.full(41, 0.002),
        ]
    )
    _, _, a = p.select(fm, ix)
    assert a.tolist() == [[False, False, False, True], [False, False, False, True]]


def test_oracle_and_random_nonoracle_are_not_mislabeled_harm():
    rows, payload = fixtures()
    x = p.inject(rows, payload, set(payload), 175101, 1, "context", "oracle", "full")
    y = p.inject(
        rows, payload, set(payload), 175101, 1, "context", "random_nonoracle", "full"
    )
    assert np.all(x["delta"] == 4) and (y["delta"] > 0).any() and (y["delta"] < 4).all()
    assert all(x["failures"]) and all(x["avoidable"])


def test_family_weighting_is_not_context_weighting():
    _, fm = p.family_mean(np.array([0.0, 0.0, 0.0, 4.0]), ["a", "a", "a", "b"])
    assert fm.mean() == 2 and np.mean([0, 0, 0, 4]) == 1


def test_mde_grid_and_right_censoring():
    curve = [
        dict(fraction=0.01, injected_gain={"median": 0.002}, power={"median": 0.5}),
        dict(fraction=0.2, injected_gain={"median": 0.04}, power={"median": 0.79}),
    ]
    assert p.mde(curve, 0.2)["status"] == "RIGHT_CENSORED"
    curve[-1]["power"]["median"] = 0.8
    d = p.mde(curve, 0.2)
    assert d["fraction"] == 0.2 and d["gain"] == 0.04 and d["headroom_percent"] == 20


def test_null_elo_anchor_and_seed_summary():
    w = p.report_weights(3)
    x = p.summarize(np.zeros((3, 3)), w)
    e = p.elo(np.full((3, 3), 0.5), w)
    assert x["median"] == 0 and x["ci95"] == [0, 0] and e["median"] == 1500


def test_no_optimizer_or_backward_call_in_owned_execution():
    root = Path(__file__).parent
    for name in ["runner.py", "power.py", "protocol.py"]:
        tree = ast.parse((root / name).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                assert n.func.attr not in [
                    "backward",
                    "step",
                    "AdamW",
                    "Adam",
                    "SGD",
                    "LBFGS",
                ]
    text = (root / "runner.py").read_text()
    assert (
        "torch.optim.Optimizer.__init__=forbidden" in text.replace(" ", "")
        and "param.requires_grad_(False)" in text
    )


def test_full_pipeline_role_locks_and_censored_MDE(monkeypatch, tmp_path):
    import power

    monkeypatch.setattr(p, "SIMULATIONS", 4)
    monkeypatch.setattr(p, "SELECTOR_DRAWS", 16)
    monkeypatch.setattr(p, "REPORT_DRAWS", 100)
    monkeypatch.setattr(p, "FRACTIONS", (0.0, 0.2))
    rows = [dict(context_sha256=f"t{i}", family_id=f"f{i:03}") for i in range(205)]
    val = [dict(context_sha256="v0", family_id="v0")]
    payload = {
        r["context_sha256"]: (np.arange(24, dtype=float), np.linspace(-2, 2, 24))
        for r in rows + val
    }
    head, curves = power.run(rows, val, payload, tmp_path, lambda: None)
    assert len(curves) == 16 and len(list(tmp_path.glob("lock-*.npz"))) == 240
    assert all(
        x["rules"]["STRICT"]["admitted_decisions"] == 0
        for x in curves
        if x["fraction"] == 0
    )
    assert head["full"]["TRAIN"]["equal_family_headroom"]["median"] == 4
    assert head["24"]["MDE80"]["context"]["STRICT"]["status"] in [
        "GRID_DETECTED",
        "RIGHT_CENSORED",
    ]


def test_native_baseline_rejects_single_bit_mutation():
    from runner import baseline_gate

    values = np.zeros(91, dtype=np.float64)
    values[-1] = 91 * 1.1377941898164525
    assert float(values.mean()) == 1.1377941898164525
    baseline_gate(values, values.copy())
    bad = values.copy()
    bad.view(np.uint64)[0] = 1
    with pytest.raises(RuntimeError):
        baseline_gate(bad, values)
