"""Synthetic CPU controls for the H124 fixed-lambda shrink protocol."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h124_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def _rows():
    return [
        {
            "context_sha256": f"{index:064x}",
            "family_id": f"family-{index:03d}",
            "dl_split": "DL-Train",
            "source_split": "train",
            "task": "regression",
        }
        for index in range(205)
    ]


def _packets(blend_gain=0.002, pure_gain=0.0, harmed=0):
    rows = _rows()
    packets = {}
    for seed in r.SEEDS:
        packets[str(seed)] = {}
        for index, row in enumerate(rows):
            blend = 0.5 + (-0.02 if index < harmed else blend_gain)
            packets[str(seed)][row["context_sha256"]] = {
                "r882": {"u4": 0.5},
                "blend": {"u4": blend},
                "pure_gbm": {"u4": 0.5 + pure_gain},
            }
    return rows, packets


def test_fold_is_fresh_deterministic_family_grouped_and_disjoint():
    rows = _rows()
    groups, mapping = r.folds(rows)
    reversed_groups, reversed_mapping = r.folds(list(reversed(rows)))
    assert groups == reversed_groups
    assert mapping == reversed_mapping
    assert [len(group) for group in groups] == [41] * 5
    assert len(set(mapping.values())) == 5
    old = [
        sorted(group.tolist())
        for group in np.array_split(
            np.random.default_rng(11900).permutation(
                sorted({row["family_id"] for row in rows})
            ),
            5,
        )
    ]
    assert groups != old
    duplicate = rows + [rows[0]]
    with pytest.raises(ValueError):
        r.folds(duplicate)
    rows[0]["dl_split"] = "DL-Val"
    with pytest.raises(ValueError):
        r.folds(rows)


def test_score_blend_is_fixed_and_matches_registered_utility_formula():
    r882_cost = np.asarray([3.0, 0.5, -1.0])
    gbm_cost = np.asarray([1.0, 1.5, -3.0])
    blended = r.blend_cost(r882_cost, gbm_cost)
    expected_utility = -r882_cost + 0.5 * ((-gbm_cost) - (-r882_cost))
    np.testing.assert_array_equal(blended, -expected_utility.astype(np.float32))
    np.testing.assert_array_equal(r.blend_cost(r882_cost, gbm_cost, 1.0), gbm_cost)
    with pytest.raises(ValueError, match="fixed"):
        r.blend_cost(r882_cost, gbm_cost, 0.75)
    with pytest.raises(ValueError, match="finite"):
        r.blend_cost([0.0], [float("nan")])


def test_family_values_refuse_incomplete_or_duplicate_panels():
    rows, packets = _packets()
    families, values = r.family_values(rows, packets, "blend")
    assert len(families) == 205
    assert values.shape == (3, 205)
    packets["64"].pop(rows[0]["context_sha256"])
    with pytest.raises(ValueError, match="every DL-Train context"):
        r.family_values(rows, packets, "blend")
    with pytest.raises(ValueError, match="unique"):
        r.family_values(rows + [rows[0]], packets, "blend")


def test_fixed_lambda_primary_gate_and_secondary_is_descriptive(monkeypatch):
    monkeypatch.setattr(r, "elo_h124", lambda *_args, **_kwargs: {"ratings": {}})
    rows, packets = _packets(blend_gain=0.002, pure_gain=0.0)
    result = r.h124_summary(rows, packets)
    assert result["primary_vs_r882"]["admitted"]
    assert result["primary_vs_r882"]["lcb95"] == pytest.approx(0.002)
    assert result["secondary_pure_gbm_vs_r882"]["admission_assessed"] is False
    assert result["secondary_pure_gbm_vs_r882"]["admitted"] is None

    rows, tied = _packets(blend_gain=0.001)
    assert not r.h124_summary(rows, tied)["primary_vs_r882"]["admitted"]

    rows, harmed = _packets(blend_gain=0.005, harmed=21)
    rejected = r.h124_summary(rows, harmed)["primary_vs_r882"]
    assert rejected["median_seed_gain"] > 0
    assert rejected["harm_count"] == 21
    assert not rejected["admitted"]


def test_unregistered_fit_refuses_before_input_access(tmp_path):
    for task in ((63, (0,)), (67, (0,)), (64, ()), (64, (1, 0))):
        with pytest.raises(ValueError):
            r.fit_one(task, tmp_path, [], {}, {}, {}, {})


def test_cpu_dry_run_uses_only_synthetic_metadata_and_no_fits():
    receipt = r.cpu_dry_run()
    assert receipt["status"] == "CPU_DRY_RUN_PASS"
    assert receipt["synthetic_folds"] == [41] * 5
    assert receipt["fit_count_planned"] == 45
    assert receipt["data_reads"] == receipt["checkpoint_reads"] == 0
    assert receipt["estimator_fits"] == receipt["outcome_cells"] == 0
    assert not receipt["GPU_used"]
    assert not receipt["qualification_lock_used"]
    assert not receipt["FIFO_ticket_created"]


# Amendment 1 controls: train-row-only cache reads, lazy pools, bit-identical design matrix, RSS ceiling, and
# the registered median seed Elo (audit H124-A1..A3). All synthetic.


def _cache(tmp_path, n_ctx=6, dev=(1, 4)):
    rng = np.random.default_rng(0)
    cids = np.asarray([f"{i:064x}" for i in range(n_ctx)])
    arrays = {
        "cids": cids,
        "tokens_10x32": rng.standard_normal((n_ctx, 10, 32)).astype(np.float32),
        "mask_10": rng.random((n_ctx, 10)) > 0.5,
        "dope629": rng.standard_normal((n_ctx, 629)).astype(np.float32),
        "lm48_z": rng.standard_normal((n_ctx, 48)).astype(np.float32),
    }
    np.savez(tmp_path / "context_features.npz", **arrays)
    sizes = rng.integers(5, 9, n_ctx)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    np.savez(
        tmp_path / "pool_priors.npz",
        cids=cids,
        offsets=offsets,
        proj16=rng.standard_normal((int(offsets[-1]), 16)).astype(np.float32),
    )
    rows = [
        {"context_sha256": c, "family_id": f"f{i}"}
        for i, c in enumerate(cids)
        if i not in dev
    ]
    dev_rows = [{"context_sha256": cids[i], "family_id": f"f{i}"} for i in dev]
    return arrays, offsets, rows, dev_rows


def test_stored_rows_match_numpy_and_compressed_members_refuse(tmp_path):
    arrays, _, _, _ = _cache(tmp_path)
    for member in ("tokens_10x32", "mask_10", "dope629", "lm48_z"):
        reader = r.StoredRows(tmp_path / "context_features.npz", member)
        np.testing.assert_array_equal(reader.rows(2, 4), arrays[member][2:4])
        assert reader.receipt([(1, 2)])["overlapping_reads"] == 0
        assert reader.receipt([(3, 4)])["overlapping_reads"] == 1
    np.savez_compressed(tmp_path / "c.npz", a=np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match="compressed"):
        r.StoredRows(tmp_path / "c.npz", "a")


def test_train_context_and_projection_never_read_dev_rows(tmp_path, monkeypatch):
    arrays, offsets, rows, dev_rows = _cache(tmp_path)
    monkeypatch.setattr(r, "CACHE", tmp_path)
    seen = []
    original = r.StoredRows.rows

    def spy(self, lo, hi):
        seen.append((self.member, lo, hi))
        return original(self, lo, hi)

    monkeypatch.setattr(r.StoredRows, "rows", spy)
    context, receipts = r.train_context(rows, dev_rows)
    assert set(context) == {row["context_sha256"] for row in rows}
    np.testing.assert_array_equal(
        context[rows[1]["context_sha256"]]["dope629"], arrays["dope629"][2]
    )
    assert all(v["overlapping_reads"] == 0 for v in receipts.values())
    assert {lo for member, lo, _ in seen} == {
        0,
        2,
        3,
        5,
    }  # dev positions 1 and 4 never read
    reader, spans, _ = r.train_projection(rows, dev_rows)
    for row in rows:
        i = int(row["context_sha256"], 16)
        assert spans[row["context_sha256"]] == (offsets[i], offsets[i + 1])
        np.testing.assert_array_equal(
            reader.rows(*spans[row["context_sha256"]]).shape,
            (offsets[i + 1] - offsets[i], 16),
        )
    dev_proj = [(int(offsets[i]), int(offsets[i + 1])) for i in (1, 4)]
    assert reader.receipt(dev_proj)["overlapping_reads"] == 0
    with pytest.raises(RuntimeError, match="closure"):
        r.train_context(rows, dev_rows[:1])


def _pool(tmp_path, cid, n=200, seed=0):
    rng = np.random.default_rng(seed)
    path = tmp_path / f"{cid}.npz"
    np.savez_compressed(path, program=rng.standard_normal((n, 412)).astype(np.float32))
    fit = rng.random(n)
    gate = rng.random(n) > 0.2
    return {
        "path": path,
        "pool_sha256": r.sha(path),
        "fitness": fit,
        "gate": gate,
        "u": np.where(gate, -fit, -2.0),
    }


def test_lazy_program_checks_identity(tmp_path):
    pools = {"c": _pool(tmp_path, "c")}
    assert r.program(pools, "c").dtype == np.float32
    pools["c"]["pool_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="identity"):
        r.program(pools, "c")


def test_fit_design_matrix_is_bit_identical_to_float32_concatenation(
    tmp_path, monkeypatch
):
    import sklearn.ensemble

    rows = [{"context_sha256": f"c{i}"} for i in range(5)]
    mapping = {f"c{i}": i for i in range(5)}
    pools = {f"c{i}": _pool(tmp_path, f"c{i}", seed=i) for i in range(5)}
    rng = np.random.default_rng(1)
    samples = {cid: 4 + rng.choice(196, 192, replace=False) for cid in pools}
    context = {
        cid: {
            "lm48_z": rng.standard_normal(48).astype(np.float32),
            "dope629": rng.standard_normal(629).astype(np.float32),
        }
        for cid in pools
    }
    captured = {}

    class Recorder:
        def __init__(self, **kw):
            self.kw, self.n_iter_ = kw, 1

        def fit(self, x, y):
            captured["x"], captured["y"] = x.copy(), y.copy()
            return self

        def predict(self, x):
            return np.zeros(len(x))

        def get_params(self):
            return self.kw

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", Recorder)
    run = tmp_path / "run"
    run.mkdir()
    r.fit_one((64, (0,)), run, rows, context, pools, samples, mapping)
    parts, targets = [], []
    for row in rows[1:]:
        cid = row["context_sha256"]
        cf = np.concatenate([context[cid]["lm48_z"], context[cid]["dope629"]]).astype(
            np.float32
        )
        program = r.program(pools, cid)
        parts.append(
            np.concatenate(
                [program[samples[cid]], np.repeat(cf[None], 192, axis=0)], axis=1
            )
        )
        targets.append(
            (pools[cid]["u"][samples[cid]] - pools[cid]["u"].mean()).astype(np.float32)
        )
    reference = np.concatenate(parts).astype(
        np.float64
    )  # what scikit-learn converts the original input to
    assert captured["x"].dtype == np.float64
    np.testing.assert_array_equal(captured["x"], reference)
    np.testing.assert_array_equal(captured["y"], np.concatenate(targets))


def test_rss_ceiling_is_fixed_and_fails_closed(monkeypatch):
    with pytest.raises(ValueError):
        r.set_rss_limit(8)
    monkeypatch.setattr(r, "RSS_LIMIT_KIB", 1)
    with pytest.raises(RuntimeError, match="RSS limit"):
        r.check_rss()


def test_median_seed_elo_is_the_registered_estimator():
    rows = _rows()
    packets = {}
    for seed, delta in zip(r.SEEDS, (0.03, 0.03, -0.09)):
        packets[str(seed)] = {
            row["context_sha256"]: {"r882": {"u4": 0.5}, "blend": {"u4": 0.5 + delta}}
            for row in rows
        }
    samples = np.random.default_rng(r.ELO_BOOTSTRAP_SEED).integers(
        0, 205, size=(r.ELO_BOOTSTRAP_DRAWS, 205)
    )
    out = r.median_seed_elo(rows, packets, "blend", samples)
    assert out["per_seed_point"] == {"64": 3224.71, "65": 3224.71, "66": -224.71}
    assert out["rating"] == 3224.71


def test_median_seed_elo_agrees_with_canonical_helper_on_mixed_families():
    rows = [
        {"context_sha256": f"{i:064x}", "family_id": f"family-{i % 37:03d}"}
        for i in range(120)
    ]
    rng = np.random.default_rng(5)
    packets = {
        str(seed): {
            row["context_sha256"]: {
                "r882": {"u4": 0.5},
                "blend": {"u4": 0.5 + float(rng.choice([-0.02, 0.0, 0.00005, 0.02]))},
            }
            for row in rows
        }
        for seed in r.SEEDS
    }
    result = r.elo_h124(rows, packets, "blend")
    assert result["registered_estimator"] == "median_seed_elo"
    for seed in r.SEEDS:
        assert result["median_seed_elo"]["per_seed_point"][str(seed)] == pytest.approx(
            result["per_seed_ratings"][str(seed)]["H124"], abs=0.05
        )
    lo, hi = result["median_seed_elo"]["ci95"]
    assert lo <= result["median_seed_elo"]["rating"] <= hi
