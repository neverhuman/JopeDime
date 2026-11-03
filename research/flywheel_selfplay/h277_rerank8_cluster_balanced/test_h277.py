import hashlib
import json
import copy
from pathlib import Path
import numpy as np
import pytest
from dependencies import ROOT, CLOSURE, PINS, topk, clusters, core
import weighted
import gate
import analysis_h277 as stat
import monitor

HERE = Path(__file__).parent


def test_parent_and_cluster_manifest_exact_identity():
    for name, digest in PINS["files"].items():
        assert hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == digest
    assert (ROOT / "inputs/fold_manifest.json").read_bytes() == (
        CLOSURE / "h276-clusters.json"
    ).read_bytes()
    cm = json.loads((ROOT / "inputs/fold_manifest.json").read_text())
    parts = clusters.partitions_from_manifest(cm)
    assert tuple(parts) == stat.PARTITIONS
    for groups in parts.values():
        assert (
            sorted(f for g in groups for f in g) == cm["families"] and len(groups) == 5
        )
    assert min(map(len, parts["KM276141"])) == 1


def test_equal_cluster_mass_and_train_only_counts():
    fam = {"a1": "a", "a2": "a", "b1": "b", "c1": "c", "c2": "c", "d1": "d"}
    cl = {"a": 0, "b": 1, "c": 1, "d": 1}
    cids = list(fam)
    w, fs, fw, _ = weighted.weights(cids, fam, cl)
    assert sum(w[i] for i, c in enumerate(cids) if cl[fam[c]] == 0) == pytest.approx(
        sum(w[i] for i, c in enumerate(cids) if cl[fam[c]] == 1)
    )
    assert sum(fw[i] for i, f in enumerate(fs) if cl[f] == 0) == pytest.approx(0.5)
    subset = ["a1", "b1"]
    a = weighted.weights(subset, fam, cl)[0]
    b = weighted.weights(subset, {**fam, "OUTER": "foreign"}, {**cl, "foreign": 99})[0]
    np.testing.assert_array_equal(a, b)


def test_weight_placebo_permuted_only_present_clusters():
    fam = {str(i): str(i) for i in range(6)}
    cl = {str(i): 0 if i == 0 else 1 for i in range(6)}
    for seed in range(20):
        w, _, _, info = weighted.weights(list(fam), fam, cl, seed)
        assert (
            set(info["permutation"]) == {0, 1}
            and np.isfinite(w).all()
            and w.mean() == pytest.approx(1)
        )
    assert any(
        not np.allclose(
            weighted.weights(list(fam), fam, cl, s)[0],
            weighted.weights(list(fam), fam, cl)[0],
        )
        for s in range(20)
    )


def test_ranker_parameters_match_parent(monkeypatch):
    import lightgbm as lgb

    captured = []

    class Model:
        def __init__(self, **kw):
            self.params = kw

        def fit(self, *a, **kw):
            captured.append((self.params, kw))
            return self

    monkeypatch.setattr(lgb, "LGBMRanker", Model)
    x = np.zeros((16, 568))
    y = np.zeros(16, dtype=int)
    topk.fit_ranker(x, y, 2, 8, 0.03, 123)
    weighted.fit_ranker(x, y, 2, 8, 0.03, 123, np.ones(16))
    assert captured[0][0] == captured[1][0]
    assert set(captured[1][1]) - set(captured[0][1]) == {"sample_weight"}


def test_h273_veto_semantics_and_parameters():
    sc = np.array([[0, 0.9, 0.8, 0, 0, 0, 0, 0]])
    pr = np.array([[1, 0.11, 0, 0, 0, 0, 0, 0]])
    assert weighted.picks(sc, 0, pr, 0.1).tolist() == [
        0
    ]  # no search for an alternative candidate
    assert weighted.picks(sc, 0, pr, 0.11).tolist() == [1]
    assert gate.select_from_topk_scores_with_veto(sc[0], pr[0], 0.1, 0) == 0
    assert (
        gate.PARAMS["class_weight"] == {0: 1.0, 1: 10.0}
        and gate.PARAMS["n_estimators"] == 100
        and gate.PARAMS["max_depth"] == 3
        and gate.PARAMS["n_jobs"] == 1
    )


def test_nested_tau_selected_only_from_inner_predictions():
    scores = {
        (lr, t): np.tile([0, 0.9, 0, 0, 0, 0, 0, 0], (12, 1))
        for lr in core.LR_GRID
        for t in core.N_EST_GRID
    }
    u = np.zeros((12, 8))
    u[:, 1] = 0.1
    prob = np.zeros((12, 8))
    prob[:, 1] = 0.12
    cfg = weighted.choose(
        scores, u, [str(i) for i in range(12)], np.ones(12) / 12, 123, prob
    )
    assert cfg["tau"] == 0.15 and cfg["strict_admitted"] and cfg["margin"] == 0.75
    assert "outer" not in weighted.choose.__code__.co_varnames
    assert weighted.TAUS == (0.05, 0.10, 0.15, 0.20)


def fixture():
    c = dict(
        positive_seeds=3,
        ci975_holm2=[0.01, 0.05],
        harm_families_gt_1e4_by_seed=[1, 2, 3],
    )
    primary = {n: {a: copy.deepcopy(c) for a in stat.ARMS[1:]} for n in stat.PARTITIONS}
    secondary = {a: dict(positive_seeds=3) for a in stat.ARMS[1:]}
    raw = {n: {a: 0 for a in stat.ARMS} for n in stat.PARTITIONS}
    for n in stat.PARTITIONS:
        primary[n]["PLACEBO"]["positive_seeds"] = 0
    return primary, secondary, raw


def test_all_partitions_all_seeds_harms_and_placebo_gate():
    p, s, raw = fixture()
    assert stat.decide(p, s, raw)["verdict"] == "ADVANCE"
    q = copy.deepcopy(p)
    q["WARD"]["A"]["ci975_holm2"][0] = 1e-12
    assert not stat.decide(q, s, raw)["arms"]["A"]["advance"]
    q = copy.deepcopy(p)
    q["KM276141"]["B"]["harm_families_gt_1e4_by_seed"][1] += 1
    assert not stat.decide(q, s, raw)["arms"]["B"]["advance"]
    raw["WARD"]["A"] = 1
    assert not stat.decide(p, s, raw)["arms"]["A"]["advance"]
    raw["WARD"]["A"] = 0
    for n in stat.PARTITIONS:
        p[n]["PLACEBO"]["positive_seeds"] = 3
    assert stat.decide(p, s, raw)["verdict"] == "NO_ADVANCE"


def test_statistics_zero_ties_cannot_advance():
    groups = [[f"f{g}_{i}" for i in range(g + 1)] for g in range(5)]
    parts = {n: groups for n in stat.PARTITIONS}
    rows = [dict(family_id=f) for g in groups for f in g]
    v = np.ones((3, 4, len(rows), 5))
    vp = np.ones((3, 5, len(rows), 5))
    p, s, raw = stat.summarize(rows, v, rows, vp, parts)
    assert stat.decide(p, s, raw)["verdict"] == "NO_ADVANCE"
    assert p["WARD"]["A"]["ci95"] == [0, 0]


def test_resource_guard_rate_and_memory_pause():
    def sample(t, sin=0, ram=40):
        return dict(
            utc=t,
            swap_sin_bytes=sin,
            page_size=4096,
            available=ram * 2**30,
            free=500 * 10**9,
            rss=2**30,
            threads=3,
            load5=40,
            swap_free=0,
            swap_total=8 * 2**30,
        )

    t = 1793739000
    g = monitor.Guard()
    assert g.update(sample(t)) == ([], False)
    assert g.update(sample(t + 1, ram=11)) == ([], True)
    assert not g.update(sample(t + 32, ram=11))[0]
    assert monitor.Guard().update(sample(t, ram=5))[0]
    g = monitor.Guard()
    g.update(sample(t))
    for i in range(1, 32):
        assert g.update(sample(t + i, i * 2100 * 4096))[1] == (i >= 31)
    assert monitor.COMPUTE_STOP == 1793748600


def test_outer_labels_cannot_change_weighted_fit_or_nested_tau(monkeypatch):
    folds = {f: [f"{f}-{i}" for i in range(3)] for f in range(5)}
    fams = {c: c for cs in folds.values() for c in cs}
    cl = {c: int(c[0]) for c in fams}
    packets = {
        c: dict(
            X=np.tile(np.arange(8, dtype=np.float32)[:, None], (1, 3)),
            rel_k=np.arange(8, dtype=np.int32),
            gate_k=np.ones(8, dtype=bool),
            utility_k=np.arange(8, dtype=float) / 8,
        )
        for c in fams
    }
    fitted = []

    class Booster:
        def predict(self, X, **kw):
            return X[:, 0] / 8

        def model_to_string(self):
            return "synthetic-model"

    class Model:
        booster_ = Booster()

    def fit(X, y, n_groups, k, lr, random_state, sample_weight):
        fitted.append((y.copy(), sample_weight.copy(), random_state))
        return Model()

    monkeypatch.setattr(weighted, "fit_ranker", fit)
    monkeypatch.setattr(gate, "fit", lambda *args: Model())
    a, ev, _ = weighted.run(277101, folds, packets, fams, cl, "synthetic")
    before = [(y.copy(), w.copy(), s) for y, w, s in fitted]
    fitted.clear()
    for cid in folds[0]:
        packets[cid]["rel_k"][:] = -100
        packets[cid]["utility_k"][:] = -100
        packets[cid]["gate_k"][:] = False
    b, _, _ = weighted.run(277101, folds, packets, fams, cl, "synthetic")
    assert {arm: x["receipt"] for arm, x in a.items()} == {
        arm: x["receipt"] for arm, x in b.items()
    }
    assert len(before) == len(fitted)
    for (y, w, s), (yy, ww, ss) in zip(before, fitted):
        np.testing.assert_array_equal(y, yy)
        np.testing.assert_array_equal(w, ww)
        assert s == ss
    assert all(not set(call["cids"]) & set(folds[0]) for call in ev["fit_calls"])


def test_real_lightgbm_weighted_and_gate_fit_path():
    rng = np.random.default_rng(277001)
    X = rng.normal(size=(160, 8)).astype(np.float32)
    y = np.tile(np.arange(8, dtype=np.int32), 20)
    w = np.repeat(np.linspace(0.5, 1.5, 20), 8)
    model = weighted.fit_ranker(X, y, 20, 8, 0.03, 277001, w)
    score = model.booster_.predict(X, num_iteration=50, num_threads=1)
    assert (
        score.shape == (160,)
        and np.isfinite(score).all()
        and model.booster_.current_iteration() > 0
    )
    classifier = gate.fit(X, y < 2, 277002)
    prob = classifier.booster_.predict(X, num_threads=1)
    assert prob.shape == (160,) and np.all((prob >= 0) & (prob <= 1))
