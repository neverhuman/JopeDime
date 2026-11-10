"""Data-free H313 controls: the fresh gate refuses fresh-path opens until all 6 fit locks exist and verify; the
decision rule; the sign-flip test; the 6-seed summary; the v7 split rule; the census regression comparator."""

import hashlib
import json
import subprocess
import sys
import textwrap

import numpy as np

import census_h313 as C
import harness_h313 as H


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, cwd=str(H.HERE)
    )


def test_fresh_gate_until_six_locks(tmp_path):
    fresh = tmp_path / "fresh_v7_pool.npz"
    fresh.write_bytes(b"x")
    r = _run(f"""
        import json, hashlib
        from pathlib import Path
        import harness_h313 as H
        base = Path(r"{tmp_path}") / "harness"
        g = H.FreshGate(base); g.install()
        def try_open():
            try:
                open(r"{fresh}").close(); return True
            except RuntimeError:
                return False
        assert not try_open()                      # no locks, no manifest
        locks = {{}}
        for i, s in enumerate(H.SEEDS):
            p = H.lock_path(base, s); p.parent.mkdir(parents=True)
            p.write_text(json.dumps({{"seed": s}}))
            locks[str(s)] = hashlib.sha256(p.read_bytes()).hexdigest()
            if i == 4:
                (base / "lock_manifest.json").write_text(json.dumps({{"locks": locks}}))
                assert not try_open()              # 5 of 6 locks
        (base / "lock_manifest.json").write_text(json.dumps({{"locks": locks}}))
        assert try_open()                          # all 6 verify
        H.lock_path(base, 30006).write_text("tampered")
        assert not try_open()                      # sha mismatch closes the gate again
        assert g.denied == 3
        print("ok")
    """)
    assert r.stdout.strip() == "ok", r.stderr


def test_sign_flip_exact_and_mc():
    p, m = H.sign_flip_p(np.full(10, 0.01))
    assert m == "exact" and abs(p - 1 / 2**10) < 1e-12
    p0, _ = H.sign_flip_p(np.zeros(8))
    assert p0 == 1.0
    p2, m2 = H.sign_flip_p(np.full(40, 0.01), draws=20000)
    assert m2 == "monte_carlo" and p2 < 0.001


def _toy(gain, S=6, F=30, inv_extra=0):
    rng = np.random.default_rng(0)
    u = np.zeros((S, F * 2, 2))
    u[:, :, 1] = gain + rng.normal(0, 0.001, size=(S, F * 2))
    inv = np.zeros((S, F * 2, 2), dtype=bool)
    inv[:, :inv_extra, 1] = True
    fams = [f"f{i // 2}" for i in range(F * 2)]
    picks = np.ones((S, F * 2), dtype=int)
    return H.summarize_seeds(u, inv, fams, picks)


def test_decision_rule():
    s = _toy(0.01)
    assert s["seeds"] == 6 and s["families"] == 30 and len(s["sign_flip_p"]) == 6
    assert H.decide(s)[1] == "FRESH_CONFIRMED"
    assert H.decide(_toy(0.0))[1] == "FRESH_NOT_CONFIRMED"  # ties never win
    assert H.decide(_toy(0.01, inv_extra=1))[1] == "FRESH_NOT_CONFIRMED"  # invalid above reference
    assert H.ELO_LABEL_FRESH == "reference = r3175 slot0 (fresh v7)"


def test_split_and_regression_comparator():
    halves = {C.split_v7(f"fam-{i}") for i in range(50)}
    assert halves == {"v7A", "v7B"} and C.split_v7("x") == C.split_v7("x")
    want = hashlib.sha256(b"H313-v7-split-v1\x00x").digest()[-1] % 2
    assert C.split_v7("x") == ("v7A" if want == 0 else "v7B")
    m = dict(verdict="NO_COHORT", catalog_coverage={}, prior_scanned_partition_counts={}, eligible_base_families=2729,
             available_unseen_families=0, qualifying_families=0, coverage_sha256="a", flywheel_roster_counts={},
             exclusion_proof={"s": dict(count=1, set_sha256="h", intersection_count=0)})  # fmt: skip
    assert C.compare_regression(m, json.loads(json.dumps(m))) == dict(
        fields_equal=True, exclusion_sets_equal=True, n_exclusion_sets=1
    )
    m2 = json.loads(json.dumps(m))
    m2["exclusion_proof"]["s"]["set_sha256"] = "other"
    assert not C.compare_regression(m, m2)["exclusion_sets_equal"]
