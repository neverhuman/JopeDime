"""H106 structural tests on synthetic fixtures (no data payloads, models or native calls)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_h106 as h  # noqa: E402

H81 = h.load_h81()
S = lambda t: hashlib.sha256(t.encode()).hexdigest()


def test_percentile_target_is_centred_rank_with_shared_ties():
    t = h.percentile_target(np.array([3.0, 1.0, 2.0, 2.0]))
    assert t.tolist() == [0.5, -0.5, 0.0, 0.0] and h.percentile_target(np.array([7.0])).tolist() == [0.0]


def mem(rows_per_source):
    out = []
    for src, n in rows_per_source.items():
        for i in range(n):
            c = f"c{i % 10}"
            out.append(dict(source=src, context=c, fold=i % 5, program_sha256=S(f"{src}{i}"), dsl=f"({src} {i})", utility=float(i)))
    return out


def test_matched_sources_are_cut_to_the_smallest_and_never_use_the_held_fold():
    m = mem({"UNIFORM": 40, "PRIOR": 100, "ADAPTIVE": 60, "CONTEXT": 80})
    for k in range(5):
        sizes = {s: len(h.training_rows(m, s, k)) for s in h.MATCHED}
        assert len(set(sizes.values())) == 1 and sizes["UNIFORM"] == sum(1 for r in m if r["source"] == "UNIFORM" and r["fold"] != k)
        for s in h.SCORERS:
            assert all(r["fold"] != k for r in h.training_rows(m, s, k))
    assert h.training_rows(m, "PRIOR", 0) == h.training_rows(m, "PRIOR", 0)  # deterministic hash order


class FakeProp:
    fam = {"c0": "f0"}; fold = {"c0": 0}

    def __init__(self, dsls):
        self.dsls = dsls

    def batch(self, c, seed, stream):
        return [dict(program_sha256=S(d), dsl=d) for d in self.dsls]


def test_memory_build_refuses_unrecovered_programs_and_changed_logs(tmp_path, monkeypatch):
    row = dict(context="c0", seed=701, arm="random4", status="completed", revealed=[S("(a)"), S("(zz)")], revealed_utility=[1.0, 2.0])
    (tmp_path / "rows_w4.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "rows.jsonl").write_text("")
    logs = {"h103": ("rows_w4.jsonl", hashlib.sha256((tmp_path / "rows_w4.jsonl").read_bytes()).hexdigest(), "s"),
            "h94": ("rows.jsonl", hashlib.sha256(b"").hexdigest(), "s")}
    monkeypatch.setattr(h, "LOGS", logs)
    with pytest.raises(SystemExit):  # (zz) is not in the replayed stream
        h.build_memory(FakeProp(["(a)", "(b)"]), {"h94": tmp_path, "h103": tmp_path})
    m = h.build_memory(FakeProp(["(a)", "(zz)"]), {"h94": tmp_path, "h103": tmp_path})
    assert {(r["source"], r["dsl"], r["utility"]) for r in m} == {("UNIFORM", "(a)", 1.0), ("UNIFORM", "(zz)", 2.0)}
    monkeypatch.setattr(h, "LOGS", dict(logs, h103=("rows_w4.jsonl", "0" * 64, "s")))
    with pytest.raises(SystemExit):
        h.build_memory(FakeProp(["(a)", "(zz)"]), {"h94": tmp_path, "h103": tmp_path})


def test_arm_picks_use_scores_only_and_random_is_seeded():
    rng = np.random.default_rng(0)
    ids = [S(str(i)) for i in range(16)]
    scores = {a: rng.normal(size=16) for a in h.ARMS if a != "random4"}
    p1, p2 = h.arm_picks(scores, ids, H81.ranked, 5), h.arm_picks(scores, ids, H81.ranked, 5)
    assert p1 == p2 and all(len(set(v)) == 4 for v in p1.values())
    assert p1["S_ALL"] == H81.ranked(scores["S_ALL"], ids)[:4]


def rows(delta_all=0.0, delta_prior=0.0, n=60, drop=None):
    out = []
    for i in range(n):
        for seed in h.EVAL_SEEDS:
            for a in h.ARMS + ("ORACLE4",):
                v = 1.0 + 0.01 * i + (delta_all if a == "S_ALL" else 0) + (delta_prior if a == "S_PRIOR" else 0) + (1 if a == "ORACLE4" else 0)
                if drop == (i, seed, a):
                    continue
                out.append(dict(tag="H106", context=f"c{i}", family=f"f{i % 30}", seed=seed, arm=a, status="completed", best_of_four=v))
    return out


ROSTER = [f"c{i}" for i in range(60)]


def test_analyze_two_primaries_and_completeness_gate():
    v = h.analyze(H81, rows(0.02, 0.0), ROSTER, [])["verdict"]
    assert v == dict(A_scorer_vs_random="PASS", B_guided_vs_uniform_memory="FAIL")
    v = h.analyze(H81, rows(0.0, 0.02), ROSTER, [])["verdict"]
    assert v == dict(A_scorer_vs_random="FAIL", B_guided_vs_uniform_memory="PASS")
    assert h.analyze(H81, rows(0.02, 0.02, drop=(3, 802, "S_PRIOR")), ROSTER, [])["verdict"].startswith("INCOMPLETE")
    assert h.analyze(H81, rows(0.02), ROSTER, [dict(context="c0")])["verdict"].startswith("INCOMPLETE")
    with pytest.raises(SystemExit):
        h.analyze(H81, [dict(r, tag="DRY_RUN") for r in rows()], ROSTER, [])


def test_h81_loader_refuses_changed_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "run_prospective.py"; fake.write_text("# not the merged runner\n")
    monkeypatch.setattr(h, "H81_PATH", fake)
    with pytest.raises(SystemExit):
        h.load_h81()
