"""Data-free H308 controls. The negative fixture: a mixed npz whose trailing (non-TRAIN) feature rows are overwritten
in place with NaN after writing. A whole-member read (np.load -> zipfile) must fail its CRC check on those bytes, while
the TRAIN-only loader must return exactly the TRAIN prefix and never read the poisoned bytes."""

import zipfile

import numpy as np
import pytest

import evaluate_h308 as ev
import run_h308 as r
import trainonly_loader as tl

TRAIN = ["t0", "t1", "t2", "t3", "t4"]
VAL = ["v0", "v1"]


def _poisoned_fixture(path):
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(7, 3)).astype(np.float32)
    off = np.array([0, 2, 3, 6, 7, 9, 12, 14])  # ragged rows; TRAIN rows = [0, off[5]) = [0, 9)
    rows = rng.normal(size=(14, 2)).astype(np.float32)
    np.savez(path, cids=np.array(TRAIN + VAL), offsets=off, lm48_z=feats, proj16=rows)
    clean = dict(lm48_z=feats.copy(), proj16=rows.copy())
    for name, n_pre, arr in (("lm48_z.npy", 5, feats), ("proj16.npy", 9, rows)):
        data_off, shape, dtype = tl._member_layout(path, name)
        row_b = arr[0].nbytes
        poison = np.full(arr[n_pre:].shape, np.nan, dtype=np.float32).tobytes()
        with open(path, "r+b") as f:
            f.seek(data_off + n_pre * row_b)
            f.write(poison)
    return clean


def test_negative_fixture_poisoned_tail_never_read(tmp_path):
    p = tmp_path / "pool_priors.npz"
    clean = _poisoned_fixture(p)
    with np.load(p) as z, pytest.raises(zipfile.BadZipFile):
        z["proj16"]  # a whole-member decode touches the poisoned tail and fails CRC
    tl.COUNTERS.update(feature_bytes_read=0, feature_rows_read=0)
    with tl.TrainOnlyNumpy(set(TRAIN)).load(p) as z:
        assert z.row_prefix == 9 and z.n_train == 5
        got = z["proj16"]
        assert got.shape == (9, 2) and np.array_equal(got, clean["proj16"][:9]) and not np.isnan(got).any()
        assert list(z["cids"][:5]) == TRAIN
    assert tl.COUNTERS["feature_rows_read"] == 9 and tl.COUNTERS["feature_bytes_read"] == 9 * 2 * 4


def test_negative_fixture_unragged(tmp_path):
    p = tmp_path / "context_features.npz"
    rng = np.random.default_rng(2)
    a = rng.normal(size=(7, 4)).astype(np.float32)
    np.savez(p, cids=np.array(TRAIN + VAL), lm48_z=a)
    off, _, _ = tl._member_layout(p, "lm48_z.npy")
    with open(p, "r+b") as f:
        f.seek(off + 5 * 16)
        f.write(np.full((2, 4), np.nan, dtype=np.float32).tobytes())
    with tl.TrainOnlyNumpy(set(TRAIN)).load(p) as z:
        assert np.array_equal(z["lm48_z"], a[:5])


def test_prefix_violation_refused(tmp_path):
    p = tmp_path / "context_features.npz"
    np.savez(p, cids=np.array(["t0", "v0", "t1", "t2", "t3", "t4", "v1"]), lm48_z=np.zeros((7, 2), np.float32))
    with pytest.raises(RuntimeError, match="prefix"):
        tl.TrainOnlyNumpy(set(TRAIN)).load(p)


def test_other_files_use_numpy(tmp_path):
    p = tmp_path / "other.npz"
    np.savez(p, x=np.arange(3))
    with tl.TrainOnlyNumpy(set(TRAIN)).load(p) as z:
        assert np.array_equal(z["x"], np.arange(3))


def test_guarded_packets():
    g = tl.GuardedPackets({"a": 1, "b": 2}, {"b"})
    assert g["a"] == 1 and g.get("a") == 1
    with pytest.raises(KeyError):
        g["b"]
    assert g.denied == 1


def test_partitions_differ_and_cover_4_folds():
    fams = [f"fam-{i}" for i in range(300)]
    p1 = [r.outer_fold(r.SALTS["P1"], f) for f in fams]
    p2 = [r.outer_fold(r.SALTS["P2"], f) for f in fams]
    assert set(p1) == set(p2) == {0, 1, 2, 3} and p1 != p2
    assert r.SALTS == {"P1": b"H308-outer-v1\x00", "P2": b"H308-outer-v2\x00"}


def _res(med, pooled, lcb, d5, harm=0.1, exc=-2):
    return dict(
        summary=dict(median_gain_across_seeds=med, pooled_lcb_one_sided_95=pooled, per_seed_lcb_one_sided={"0.05": lcb},
                     harmed_share_by_seed=[harm] * 3),
        concentration=[dict(drop_top5_lcb95_one_sided=x) for x in d5],
        invalid_detail=[dict(excess=exc)] * 3,
    )  # fmt: skip


def test_decision_rule():
    good = _res(0.05, 0.02, [0.01, 0.01, 0.01], [0.01, 0.01, -0.01])
    assert ev.decide({"P1": good, "P2": good}, ["P1", "P2"])[2] == "PARTITION_ROBUST"
    weak = _res(0.05, 0.02, [0.01, -0.001, -0.002], [0.01, -0.01, -0.01])
    crit, counts, verdict, failed = ev.decide({"P1": good, "P2": weak}, ["P1", "P2"])
    assert verdict == "PARTITION_FRAGILE" and set(failed) == {
        "c2_per_seed_lcb95_gt_0_cells",
        "c3_drop_top5_lcb95_gt_0_cells",
    }
    tie = _res(0.001, 0.02, [0.01] * 3, [0.01] * 3)  # median exactly +0.001 is not > +0.001
    assert ev.decide({"P1": good, "P2": tie}, ["P1", "P2"])[2] == "PARTITION_FRAGILE"
    assert ev.decide({"P1": good}, ["P1"])[2] == "PARTITION_ROBUST"
    assert ev.ELO_LABEL == "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured"
