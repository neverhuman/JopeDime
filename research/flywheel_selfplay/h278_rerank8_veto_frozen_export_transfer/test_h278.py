"""Data-free H278 controls: byte-identical H276 science, freeze_veto equals veto.run_cell_veto on the same folds,
exported-model inference reproduces the in-run picks, the inner family folds, the Part B statistics and gates (ties
never count as wins), the two-contender Elo, the A1 guard and the manifest pins. Toy data only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import analysis as stat
import freeze
import monitor as mon
import topk
import veto

HERE = Path(__file__).resolve().parent
GIB = 2**30


def test_h276_science_byte_identical():
    want = {
        "topk.py": "fcd2d100c30d97df",
        "h265_core.py": "5c17ddad3993b5ce",
        "veto.py": "bf9c06ddd97f793b",
    }
    for name, prefix in want.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest().startswith(prefix)


def synthetic(n_fam=60, k=8, seed=5):
    rng = np.random.default_rng(seed)
    packets, fam, rows = {}, {}, []
    for i in range(n_fam):
        for j in range(3):
            cid = f"f{i:03}c{j}"
            fam[cid] = f"f{i:03}"
            rows.append(dict(context_sha256=cid, family_id=f"f{i:03}"))
            X = rng.normal(size=(k, 10)).astype(np.float32)
            gate = rng.random(k) > 0.2
            gate[0] = True
            u = np.where(gate, X[:, 0] + rng.normal(scale=0.5, size=k), -2.0)
            packets[cid] = dict(
                X=X,
                gate_k=gate,
                utility_k=u,
                rel_k=topk.topk_relevance(-np.where(gate, u, -9), gate),
                r882_u=float(u[0]),
            )
    return rows, packets, fam


def test_freeze_equals_run_cell_veto_and_exported_inference():
    rows, packets, fam = synthetic()
    folds = freeze.family_folds(rows, 278101)
    assert folds[0] == [] and all(folds[k] for k in range(1, 5))
    assess = [r["context_sha256"] for r in rows if r["family_id"] == "f000"]
    cbf = {
        0: assess,
        **{k: [c for c in folds[k] if c not in assess] for k in range(1, 5)},
    }
    ref = veto.run_cell_veto(278101, 0, cbf, packets, fam, 8, salt="h278")
    got = freeze.freeze_veto(278101, {**cbf, 0: []}, packets, fam, 8, salt="h278")
    for key in (
        "lr",
        "n_estimators",
        "margin",
        "inner_fit_family_gain",
        "inner_fit_ci95",
        "strict_admitted",
        "strict_margin",
    ):
        assert got["receipt"][key] == ref["receipt"][key]
    assert got["ranker"] == ref["model"] and got["gate"] == ref["gate"]
    X = np.concatenate([packets[c]["X"] for c in assess])
    picks, _, _ = freeze.frozen_picks(
        got["ranker"], got["gate"], X, len(assess), got["n_estimators"], got["margin"]
    )
    assert picks == ref["assess_picks"]


def test_family_folds_deterministic_and_partition():
    rows, _, fam = synthetic(n_fam=40)
    a, b = freeze.family_folds(rows, 278102), freeze.family_folds(rows, 278102)
    assert a == b and a != freeze.family_folds(rows, 278103)
    fams = [{fam[c] for c in a[k]} for k in range(1, 5)]
    assert sum(len(f) for f in fams) == 40 and not set.intersection(*fams)
    fm = json.loads((HERE / "fold_manifest.json").read_text())
    assert len(fm["families"]) == 205 and sorted(fm["inner_fold_by_seed"]) == [
        "278101",
        "278102",
        "278103",
    ]


def toy_rows(n=19):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(n)]


def test_summary_gates_ties_and_invalid():
    r = toy_rows()
    u = np.ones((3, 19, 2))
    u[:, :, 1] += 0.02
    inv = np.zeros((3, 19, 2), dtype=bool)
    s = stat.summarize(r, u, inv)
    assert (
        s["median_family_mean_gain"] == pytest.approx(0.02)
        and s["positive_seeds"] == 3
        and s["pooled_lcb95"] > 0
    )
    assert stat.adopt_gate(s) and stat.p1_gate(s) and s["wtl_by_seed"][0] == [19, 0, 0]
    tie = stat.summarize(r, np.ones((3, 19, 2)), inv)
    assert (
        not stat.adopt_gate(tie)
        and tie["wtl_by_seed"][0] == [0, 19, 0]
        and tie["elo_challenger_by_seed"][0] == pytest.approx(1500)
    )
    inv2 = inv.copy()
    inv2[0, 0, 1] = True
    assert not stat.adopt_gate(stat.summarize(r, u, inv2))
    assert stat.bt_two(2.0, 8.0) > 1500 > stat.bt_two(8.0, 2.0)
    assert (
        "BLOCKED" in stat.P2_BLOCK_REASON
        and "1.1389572403748425" in stat.P2_BLOCK_REASON
    )


def sample(t, available, pswpin=0, load5=50.0):
    return dict(
        utc=t,
        load5=load5,
        available=available,
        free=10**15,
        rss=GIB,
        threads=2,
        affinity=[44, 45, 46, 47],
        pswpin=pswpin,
    )


def test_a1_guard_and_deadline():
    g = mon.Guard()
    assert g.update(sample(1793745000.0, 40 * GIB, 0)) == ([], False)
    assert g.update(sample(1793745001.0, 40 * GIB, 5000))[1] is True
    assert mon.COMPUTE_STOP == 1793752200.0  # 2026-11-04T00:30:00Z
    assert "absolute compute stop 00:30Z" in mon.violations(
        sample(mon.COMPUTE_STOP, 40 * GIB)
    )


def test_manifest_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert (
        m["wave"] == "H278"
        and m["total_updates"] == 0
        and "freeze.py" in m["source_sha256"]
    )
    assert m["inputs"][m["checkpoint"]].startswith("0084479a")
