"""Data-free H283 controls: the primary rule (LCB > 0 on 3/3 seeds, median > +0.010, harmed share <= 35%, invalid not
above slot 0, and the random null must NOT pass), INSUFFICIENT_FRESH below 25 families, ties are not wins, the seeded
random-valid-slot null, the train-only table transform, the scorer's H281 lineage and the pins. Toy data only."""

import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

import analysis as stat
import build_fresh as bf

HERE = Path(__file__).resolve().parent
H281 = HERE.parent / "h281_r3175_gpu_rescore_p2"


def panel(gain, n=40, harmed=0, inv_frozen=0):
    u = np.ones((3, n, 2))
    u[:, :, 1] += gain
    u[:, :harmed, 1] = 1.0 - 0.05
    inv = np.zeros((3, n, 2), dtype=bool)
    inv[:, :inv_frozen, 1] = True
    return u, inv, [f"fresh:f{i:02}" for i in range(n)], [[1] * n] * 3


def test_rule_pass_and_each_failure():
    ok = stat.summarize(*panel(0.03))
    null0 = stat.summarize(*panel(0.0))
    assert ok["passes"] and not null0["passes"]
    assert stat.verdict(40, ok, null0) == "FRESH_TRANSFER_PASS"
    assert (
        stat.verdict(40, ok, ok) == "NOT_TRANSFERRED_FRESH"
    )  # null passing the same rule blocks the pass
    assert stat.verdict(24, ok, null0) == "INSUFFICIENT_FRESH"
    assert not stat.summarize(*panel(0.008))["criteria"]["median_above_min"]
    assert not stat.summarize(*panel(0.03, harmed=15))["criteria"][
        "harmed_share_ok"
    ]  # 37.5% > 35%
    assert not stat.summarize(*panel(0.03, inv_frozen=1))["criteria"]["invalid_ok"]
    tie = stat.summarize(*panel(0.0))
    assert (
        tie["wtl_by_seed"][0] == [0, 40, 0] and abs(tie["elo_by_seed"][0] - 1500) < 1e-9
    )
    assert stat.BOOT_SEED == 283150 and stat.DRAWS == 10000


def test_random_valid_slot_null():
    gates = [
        np.array([True, False, True, True, False, False, False, False]),
        np.array([True] + [False] * 7),
    ]
    a, b = (
        stat.random_valid_slot_picks(gates, 283101),
        stat.random_valid_slot_picks(gates, 283101),
    )
    assert a == b and a[0] in (2, 3) and a[1] == 0


def test_train_only_table(tmp_path):
    nt, nq, w = 3, 2, 2
    folds = bytes([0, 1, 0])
    rows = np.arange((nt + nq) * (w + 1), dtype="<f8").tobytes()
    raw = b"JTT1" + struct.pack("<III", nt, nq, w) + bytes([2]) + folds + rows
    p = tmp_path / "t.jtt1.gz"
    p.write_bytes(gzip.compress(raw))
    out = bf.train_only_table(p)
    assert struct.unpack_from("<III", out, 4) == (nt, 0, w) and out[16] == 2
    assert np.array_equal(
        np.frombuffer(out, dtype="<f8", offset=17 + nt),
        np.arange(nt * (w + 1), dtype="<f8"),
    )


def test_scorer_lineage():
    a = (H281 / "gpu_rescore.py").read_text()
    b = (HERE / "gpu_rescore.py").read_text()
    for anchor in (
        "def predict(cids, selections):",
        "def landmark_tokens(",
        "def classify(",
        "NEED_BYTES = int(",
    ):
        assert anchor in a and anchor in b
    assert "fresh-r3175-gpu-costs.npz" in b and 'if dl_exact == "A_EXACT":' in b


def test_pins():
    m = json.loads((HERE / "manifest.json").read_text())
    assert m["wave"] == "H283" and m["total_updates"] == 0 and m["boot_seed"] == 283150
    assert m["null_seeds"] == [283101, 283102, 283103]
    for root, table in (
        (HERE.parent / "h278_rerank8_veto_frozen_export_transfer", "h278_sources"),
        (H281, "h281_sources"),
    ):
        for name, digest in m[table].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
