import copy
import json
import hashlib
import numpy as np
from dependencies import ROOT, CLOSURE, clusters, core, topk
import analysis_h280 as stat
import cell
import monitor
import veto


def test_parent_science_and_cluster_manifest_identity():
    pins = {
        "topk.py": "fcd2d100c30d97dfe70fab7b981332f0c8dbffbf8bf7b7badaaef09550983d43",
        "h265_core.py": "5c17ddad3993b5cebb702d2c58882fa2e54dc15572649c663169416f6965de76",
        "veto.py": "bf9c06ddd97f793b43f8ddbd31488768be3567db3f485f2d9377248e9b878ec0",
    }
    for f, d in pins.items():
        assert hashlib.sha256((CLOSURE / f).read_bytes()).hexdigest() == d
    assert (
        hashlib.sha256((ROOT / "source/folds.json").read_bytes()).hexdigest()
        == "e89f9aa130cf1529a684c83c2dc8bd1658dbae69a5c1e0af87760ed8d92208f2"
    )


def toy():
    rows = [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(20)]
    parts = {
        p: [[f"f{i}" for i in range(k * 4, (k + 1) * 4)] for k in range(5)]
        for p in stat.PARTITIONS
    }
    panel = np.ones((3, 4, 20, 4))
    panel[:, :, :, 2] += 0.02
    vp = np.ones((3, 5, 20, 4))
    vp[:, :, :, 2] += 0.02
    return rows, parts, panel, vp


def test_gates_pass_strict_tie_every_partition_and_placebo():
    rows, parts, panel, vp = toy()
    p, s, r = stat.summarize(rows, panel, rows, vp, parts)
    assert stat.decide(p, s, r)["verdict"] == "ADVANCE"
    q = copy.deepcopy(p)
    q["WARD"]["A"]["median_family_gain"] = 0.010
    assert stat.decide(q, s, r)["verdict"] == "NO_ADVANCE"
    q = copy.deepcopy(p)
    q["WARD"]["A"]["positive_seeds"] = 2
    assert stat.decide(q, s, r)["verdict"] == "NO_ADVANCE"
    q = copy.deepcopy(p)
    q["WARD"]["A"]["ci95"][0] = 0
    assert stat.decide(q, s, r)["verdict"] == "NO_ADVANCE"
    for n in stat.PARTITIONS:
        p[n]["PLACEBO"] = copy.deepcopy(p[n]["A"])
    s["PLACEBO"] = copy.deepcopy(s["A"])
    assert stat.decide(p, s, r)["placebo_would_pass"]
    assert stat.decide(p, s, r)["verdict"] == "NO_ADVANCE"


def test_harm_and_raw_invalid_boundaries():
    rows, parts, panel, vp = toy()
    p, s, r = stat.summarize(rows, panel, rows, vp, parts)
    p["WARD"]["A"]["harm_families_gt_1e4_by_seed"] = [7, 7, 7]
    assert stat.decide(p, s, r)["verdict"] == "ADVANCE"
    p["WARD"]["A"]["harm_families_gt_1e4_by_seed"][0] = 8
    assert stat.decide(p, s, r)["verdict"] == "NO_ADVANCE"
    p["WARD"]["A"]["harm_families_gt_1e4_by_seed"][0] = 7
    r["WARD"]["A"] = 1
    assert stat.decide(p, s, r)["verdict"] == "NO_ADVANCE"


def test_h276_veto_boundary_and_fallback():
    scores = np.array([[0, 3, 2, 1, 0, 0, 0, 0]], float)
    prob = np.zeros_like(scores)
    prob[0, 0] = 1
    prob[0, 1] = 0.1
    masked = veto.mask(scores, prob)
    assert masked[0, 0] == 0 and np.isneginf(masked[0, 1])
    assert core.select_from_top4_scores(masked[0], 0.1) == 2


def test_cluster_out_no_family_overlap():
    rows = json.loads((ROOT / "inputs/selected.json").read_text())["train"]
    parts = clusters.partitions_from_manifest(
        json.loads((ROOT / "source/folds.json").read_text())
    )
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    for name in stat.PARTITIONS:
        for seed in stat.SEEDS:
            for held in range(5):
                folds = clusters.cluster_out_folds(rows, parts[name], held, seed)
                sets = [{fam[c] for c in folds[k]} for k in range(5)]
                assert set.union(*sets) == set(fam.values())
                assert all(
                    sets[i].isdisjoint(sets[j]) for i in range(5) for j in range(i)
                )
                assert sets[0] == set(parts[name][held])


def synthetic():
    rng = np.random.default_rng(280)
    P = {}
    families = {}
    folds = {k: [] for k in range(5)}
    for i in range(50):
        c = f"c{i}"
        families[c] = f"f{i // 2}"
        folds[(i // 2) % 5].append(c)
        X = rng.normal(size=(8, 10)).astype(np.float32)
        g = rng.random(8) > 0.2
        g[0] = True
        u = np.where(g, X[:, 0], -2.0)
        P[c] = dict(
            X=X,
            gate_k=g,
            utility_k=u,
            rel_k=topk.topk_relevance(-u, g),
            r882_u=float(u[0]),
        )
    return P, families, folds


def test_wrapper_real_lightgbm_exact_h276_and_model_replay():
    P, fam, folds = synthetic()
    ref = veto.run_cell_veto(280101, 0, folds, P, fam, 8, "test")
    got = cell.run(280101, folds, P, fam, "test")
    for key in ["receipt", "model", "gate", "assess_picks"]:
        assert got[key] == ref[key]
    assert len(got["fit_receipts"]) == 18
    X = np.concatenate([P[c]["X"] for c in folds[0]])
    assert (
        cell.picks(
            got["model"], got["gate"], X, got["n_estimators"], got["margin"]
        ).tolist()
        == got["assess_picks"]
    )


def test_placebo_shuffles_fit_only_and_outer_labels_do_not_change_model():
    P, fam, folds = synthetic()
    a = cell.run(280102, folds, P, fam, "test", True)
    Q = copy.deepcopy(P)
    for c in folds[0]:
        Q[c]["rel_k"] = np.zeros(8, int)
        Q[c]["utility_k"] = np.full(8, 1000.0)
        Q[c]["gate_k"] = np.zeros(8, bool)
    b = cell.run(280102, folds, Q, fam, "test", True)
    assert (
        a["model"] == b["model"]
        and a["gate"] == b["gate"]
        and a["receipt"] == b["receipt"]
    )
    assert set(a["shuffled_labels_sha256"]) == set(
        c for k in range(1, 5) for c in folds[k]
    )
    assert not set(folds[0]).intersection(a["shuffled_labels_sha256"])


def sample(t=0):
    return dict(
        utc=t,
        available=40 * 2**30,
        rss=100,
        free=400_000_000_000,
        threads=3,
        load5=1,
        swap_sin_bytes=0,
        page_size=4096,
        swap_free=0,
    )


def test_resource_guard_ignores_occupancy_and_pauses_on_sustained_swap():
    g = monitor.Guard()
    assert g.update(sample())[1] is False
    for t in [1, 15, 30]:
        s = sample(t)
        s["swap_sin_bytes"] = t * 2500 * 4096
        assert g.update(s)[1] is False
    s = sample(31)
    s["swap_sin_bytes"] = 31 * 2500 * 4096
    assert g.update(s)[1] is True
    s = sample(32)
    s["swap_sin_bytes"] = 31 * 2500 * 4096
    assert g.update(s)[1] is False
    s = sample(33)
    s["available"] = 11 * 2**30
    assert g.update(s)[1] is True
