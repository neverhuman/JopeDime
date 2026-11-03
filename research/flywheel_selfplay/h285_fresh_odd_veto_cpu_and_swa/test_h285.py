import copy
import hashlib
import json
import numpy as np
import torch
from dependencies import ROOT, CLOSURE
import analysis_h285 as a
from swa import average, state_sha
import monitor


def test_pinned_science_and_swa_manifest():
    assert (
        hashlib.sha256((CLOSURE / "topk.py").read_bytes()).hexdigest()
        == "fcd2d100c30d97dfe70fab7b981332f0c8dbffbf8bf7b7badaaef09550983d43"
    )
    s = json.loads((ROOT / "source/swa-members.json").read_text())
    assert (
        s["forward_calls"] == 0
        and s["optimizer_updates"] == 0
        and s["dope_before"] == s["dope_after"]
    )
    assert s["members"] == [2750, 2795, 3100, 3175]
    for c in json.loads((ROOT / "export/export_manifest.json").read_text())["cells"]:
        for role in ["ranker", "veto"]:
            assert (
                hashlib.sha256(
                    (ROOT / "export" / c[role]["file"]).read_bytes()
                ).hexdigest()
                == c[role]["sha256"]
            )


def test_swa_dtype_and_integer_buffers():
    x = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([1])}
    y = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([1])}
    z = average([x, y], ["w", "b"])
    assert (
        torch.equal(z["w"], torch.tensor([2.0, 3.0]))
        and z["w"].dtype == x["w"].dtype
        and torch.equal(z["b"], x["b"])
    )
    assert state_sha(x, ["w"]) != state_sha(y, ["w"])
    assert torch.equal(x["w"], torch.tensor([1.0, 2.0]))


def toy(n=30):
    u = np.ones((3, n, 4))
    u[:, :, 1] += 0.02
    inv = np.zeros_like(u, bool)
    p = np.zeros_like(u, int)
    p[:, :, 1] = 1
    return a.summarize(u, inv, p, [f"f{i}" for i in range(n)])


def test_primary_gates_and_insufficient():
    s = toy()
    assert a.decide(s)["verdict"] == "ADVANCE"
    q = copy.deepcopy(s)
    q["contrasts"]["A"]["per_seed_ci95"][1][0] = 0
    assert a.decide(q)["verdict"] == "NOT_TRANSFERRED_FRESH"
    assert a.decide(toy(24))["verdict"] == "INSUFFICIENT_FRESH"
    q = copy.deepcopy(s)
    q["contrasts"]["A"]["median_gain"] = 0.01
    assert a.decide(q)["verdict"] == "NOT_TRANSFERRED_FRESH"


def test_invalid_harm_and_null_gate():
    s = toy()
    q = copy.deepcopy(s)
    q["contrasts"]["A"]["invalid_by_seed"][0] = 1
    assert a.decide(q)["verdict"] == "NOT_TRANSFERRED_FRESH"
    q = copy.deepcopy(s)
    q["contrasts"]["A"]["harm_families_loss_gt_0_01"][0] = 11
    assert a.decide(q)["verdict"] == "NOT_TRANSFERRED_FRESH"
    q = copy.deepcopy(s)
    q["contrasts"]["RANDOM_VALID"] = copy.deepcopy(q["contrasts"]["A"])
    assert (
        a.decide(q)["random_null_would_pass"]
        and a.decide(q)["verdict"] == "NOT_TRANSFERRED_FRESH"
    )


def test_null_is_deterministic_valid_only_and_order_independent():
    ids = ["x", "y"]
    g = np.array([[False, True, True, False], [False, False, False, False]])
    p = a.random_valid_picks(ids, g, 285101)
    assert p[0] in [1, 2] and p[1] == 0
    assert np.array_equal(p, a.random_valid_picks(ids, g, 285101))
    assert np.array_equal(p[::-1], a.random_valid_picks(ids[::-1], g[::-1], 285101))


def test_ties_not_wins():
    d = np.full((3, 30), 1e-4)
    r = a.contrast(d, np.tile(np.arange(30), (10000, 1)))
    assert r["positive_seeds"] == 0 and all(x["wins"] == 0 for x in r["wtl_by_seed"])


def test_B_never_advances():
    s = toy()
    s["contrasts"]["A"]["median_gain"] = 0
    s["contrasts"]["SWA"].update(gain=0.03, family_bootstrap_ci95=[0.02, 0.04])
    d = a.decide(s)
    assert d["B_verdict"] == "PROMISING" and d["verdict"] == "NOT_TRANSFERRED_FRESH"


def test_resource_rate_not_occupancy():
    def sample(t):
        return dict(
            utc=t,
            available=40 * 2**30,
            rss=100,
            free=400_000_000_000,
            threads=3,
            load5=1,
            swap_sin_bytes=t * 2500 * 4096,
            page_size=4096,
            swap_free=0,
        )

    g = monitor.Guard()
    assert not g.update(sample(0))[1]
    assert not g.update(sample(1))[1]
    assert g.update(sample(31))[1]


def test_train_reader_never_requests_outer_rows(monkeypatch):
    import io
    import struct
    import fresh_inputs as f

    train = struct.pack("<6d", 1, 2, 3, 4, 5, 6)
    raw = (
        b"JTT1"
        + struct.pack("<IIIB", 2, 1, 2, 2)
        + bytes([0, 1])
        + train
        + b"OUTER_MUST_NOT_BE_REQUESTED"
    )

    class Tracked(io.BytesIO):
        def read(self, n=-1):
            assert 0 <= n <= 17 + 2 + len(train) - self.tell()
            return super().read(n)

    monkeypatch.setattr(f.gzip, "open", lambda *a, **kw: Tracked(raw))
    out = f.train_only_table("fixture")
    assert (
        out[17:] == bytes([0, 1]) + train and struct.unpack_from("<I", out, 8)[0] == 0
    )


def test_corpus_pool_and_table_reconstruction_no_forward():
    from pathlib import Path
    import fresh_inputs as f

    names = json.loads(
        Path(
            "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/program-feature-names.json"
        ).read_text()
    )
    adm = {
        r["context_sha256"]: r
        for r in json.loads(
            Path(
                "/home/ubuntu/jope-evidence/dope-lift-20261021/deep-learning-dl03/admitted-input-contexts.json"
            ).read_text()
        )
    }
    for row in json.loads((ROOT / "inputs/selected.json").read_text())["dev"][:3]:
        cid = row["context_sha256"]
        a = adm[cid]
        tar = Path(a["final_path"]).parent / "summary.tar.gz"
        assert (
            f.train_only_table(Path(a["context_json_path"]).parent / "teacher.jtt1.gz")
            == (
                Path("/home/ubuntu/jope-data/h50/input/tables") / (cid + ".jtt1")
            ).read_bytes()
        )
        pool = f.extract_pool(tar, f.features, names)
        with np.load("/home/ubuntu/jope-data/h50/input/pools/" + cid + ".npz") as z:
            for key in ["program", "fitness", "gate", "program_sha256"]:
                assert np.array_equal(pool[key], z[key]), key


def test_cohort_admission_and_half_identity():
    c = json.loads((ROOT / "inputs/fresh_cohort_v1.json").read_text())
    rows = json.loads((ROOT / "inputs/fresh-selected.json").read_text())
    r = json.loads((ROOT / "inputs/fresh-metadata-receipts.json").read_text())
    assert r["status"] == "PASS" and len(rows) == 60 and len(r["rows"]) == 60
    assert [x["family_key"] for x in rows] == c["assignment"]["H285"]["families"]
    assert all(
        x["context"]["p3_assignment"] in ["train_fit", "train_nonpool"]
        and int(x["family_key_sha256"][0], 16) % 2
        for x in rows
    )
    assert not set(c["assignment"]["H283"]["families"]) & set(
        c["assignment"]["H285"]["families"]
    )


def test_elo_tolerance_base_and_per_seed():
    u = np.zeros((3, 30, 2))
    u[:, :, 1] = 0.00005
    assert np.allclose(a.elo_per_seed(u, [str(i) for i in range(30)]), 1500)
    u[:, :, 1] = 0.001
    e = a.elo_per_seed(u, [str(i) for i in range(30)])
    assert e.shape == (3, 2) and (e[:, 0] == 1500).all() and (e[:, 1] > 1500).all()
