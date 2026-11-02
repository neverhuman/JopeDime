"""Data-free H225 leakage, selection, exact-tie and resource controls."""

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(
        "h225_test_" + name, HERE / (name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


sci, monitor = module("science"), module("monitor")


def test_exact_grid_and_nested_fit_budget():
    assert len(sci.CONFIGS) == 8
    assert len({c["name"] for c in sci.CONFIGS}) == 8
    assert max(c["steps"] for c in sci.CONFIGS) == 32
    assert sci.fit_inventory() == dict(
        outer_proposals=120,
        inner_cv_fits=240,
        outer_fit_refits=120,
        all_train_refits=0,
        total_fits=360,
        optimizer_updates=8280,
    )


def test_fresh_family_partitions_and_inner_exclusion_closure():
    families = [f"f{i:03}" for i in range(205)]
    seen = []
    for seed in sci.SEEDS:
        groups = sci.folds(families, seed)
        assert sorted(sum(groups, [])) == families
        assert all(len(g) == 41 for g in groups)
        seen.append(groups)
        for k in range(5):
            a, b, fit = sci.cell(k)
            assert set((a, b)) & set(fit) == set()
            for j in fit:
                excluded = {a, b, j}
                training = set().union(
                    *(set(groups[t]) for t in range(5) if t not in excluded)
                )
                assert training.isdisjoint(groups[a])
                assert training.isdisjoint(groups[b])
                assert training.isdisjoint(groups[j])
                assert len(training) == 82
    assert seen[0] != seen[1] != seen[2]


def test_continuous_score_changes_when_top_one_does_not():
    labels = sci.gaps([0.0, 1.0], [True, True])
    assert np.argmin([0, 0.1]) == np.argmin([0, 1])
    assert sci.regret([0, 1], labels) < sci.regret([0, 0.1], labels)
    with pytest.raises(ValueError):
        sci.regret([np.nan, 0], labels)


def test_scoring_enforces_fit_disjointness_and_global_readout_lock():
    sci.require_score_role(["j"], ["fit1", "fit2"], "FIT", False, ["A", "B"])
    for role in ("A", "DL-Val"):
        with pytest.raises(ValueError):
            sci.require_score_role(["test"], ["fit"], role, False)
        sci.require_score_role(["test"], ["fit"], role, True)
    with pytest.raises(ValueError):
        sci.require_score_role(["A"], ["fit"], "FIT", False, ["A", "B"])
    with pytest.raises(ValueError):
        sci.require_score_role(["fit"], ["fit"], "FIT", False)


@pytest.mark.parametrize("arm", sci.ARMS)
def test_exact_baseline_ties_fall_back_and_positive_margin_adopts(arm):
    fam = [f"f{i}" for i in range(12)]
    reference = np.ones(12)
    same = {c["name"]: reference.copy() for c in sci.CONFIGS}
    assert sci.select(arm, same, reference, fam, 7)["chosen"] is None
    best = sci.CONFIGS[0]["name"]
    same[best] = reference + (0.1 if arm == "MSTOP1" else -0.1)
    assert sci.select(arm, same, reference, fam, 7)["chosen"] == best
    same.pop(best)
    with pytest.raises(ValueError):
        sci.select(arm, same, reference, fam, 7)


def test_placebo_permutation_never_crosses_family_boundary():
    rows = [dict(context_sha256=str(i), family_id=str(i // 2)) for i in range(4)]
    labels = {str(i): np.arange(i * 10, i * 10 + 5) for i in range(4)}
    actual = sci.shuffled_gaps(rows, labels, 3)
    for ids in (("0", "1"), ("2", "3")):
        assert sorted(np.concatenate([actual[c] for c in ids])) == sorted(
            np.concatenate([labels[c] for c in ids])
        )


def test_family_sampler_is_fit_only_paired_and_replica_changes_stream():
    rows = [
        dict(context_sha256=str(i), family_id=str(i % 3), dl_split="DL-Train")
        for i in range(9)
    ]
    orders = {str(i): np.arange(200) for i in range(9)}
    short = sci.batch_plan(rows, orders, 1, 8)
    long = sci.batch_plan(rows, orders, 1, 32)
    replica = sci.batch_plan(rows, orders, 2, 8)
    assert [r[1] for r in short] == [r[1] for r in long[:8]]
    assert [r[1] for r in short] != [r[1] for r in replica]
    assert all(set(r[1]) <= set(orders) for r in long)
    for a, b in zip(short, long[:8], strict=True):
        for ia, ib in zip(a[2], b[2], strict=True):
            np.testing.assert_array_equal(ia, ib)
    with pytest.raises(ValueError):
        sci.batch_plan([{**rows[0], "dl_split": "DL-Val"}], orders, 1, 8)


def test_missing_seed_and_ties_cannot_be_candidate_or_plateau_win():
    families = [str(i % 205) for i in range(1219)]
    zero = np.zeros((3, 1219))
    assert not sci.contrast(zero, zero, families, 11)["strict_pass"]
    assert sci.contrast(zero + 0.1, zero, families, 11)["strict_pass"]
    with pytest.raises(ValueError):
        sci.contrast(zero[:2], zero[:2], families, 11)
    vf = [str(i % 19) for i in range(91)]
    val = np.zeros((3, 91))
    assert not sci.contrast(val, val, vf, 11, plateau=True)["strict_pass"]
    assert not sci.contrast(val + 0.002, val, vf, 11, plateau=True)["strict_pass"]
    assert not sci.contrast(val + 0.00199, val, vf, 11, plateau=True)["strict_pass"]
    assert sci.contrast(val + 0.003, val, vf, 11, plateau=True)["strict_pass"]
    assert sci.elo(zero, zero, families)["median"] == 1500


def test_harm_and_each_seed_sign_are_real_gates():
    families = [str(i % 205) for i in range(1219)]
    base = np.zeros((3, 1219))
    candidate = base + 0.1
    candidate[0] = -0.01
    assert not sci.contrast(candidate, base, families, 12)["strict_pass"]
    candidate[:] = 1.0
    candidate[:, :400] = -10.0
    assert not sci.contrast(candidate, base, families, 12)["strict_pass"]


def test_load_spike_pauses_and_absolute_deadline_still_stops():
    obs = dict(
        load5=61,
        available_ram=30 * 2**30,
        free_disk=400_000_000_000,
        process_tree_rss=2**30,
    )
    assert monitor.pause_reasons(obs, monitor.DEADLINE - 100) == ["load5>60"]
    assert monitor.pause_reasons({**obs, "load5": 60}, monitor.DEADLINE - 1) == []
    with pytest.raises(TimeoutError):
        monitor.pause_reasons(obs, monitor.DEADLINE)
    with pytest.raises(MemoryError):
        monitor.pause_reasons(
            {**obs, "process_tree_rss": 4 * 2**30}, monitor.DEADLINE - 1
        )


def test_pause_signals_only_verified_owned_session():
    import psutil

    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    identity = psutil.Process(child.pid).create_time()
    try:
        with pytest.raises(RuntimeError):
            monitor.signal_owned(child, identity - 1, signal.SIGSTOP)
        assert child.poll() is None
        monitor.signal_owned(child, identity, signal.SIGSTOP)
        limit = time.monotonic() + 2
        while (
            psutil.Process(child.pid).status() != psutil.STATUS_STOPPED
            and time.monotonic() < limit
        ):
            time.sleep(0.01)
        assert psutil.Process(child.pid).status() == psutil.STATUS_STOPPED
        monitor.signal_owned(child, identity, signal.SIGCONT)
    finally:
        monitor.stop_owned(child, identity)
    assert child.returncode is not None


@pytest.mark.parametrize("positive", [False, True])
def test_entire_nested_schedule_locks_before_candidate_readouts(
    tmp_path, monkeypatch, positive
):
    runner = module("runner")
    train = [
        dict(context_sha256=str(i), family_id=f"f{i % 205:03}", dl_split="DL-Train")
        for i in range(1219)
    ]
    val = [
        dict(context_sha256=f"v{i}", family_id=f"v{i % 19}", dl_split="DL-Val")
        for i in range(91)
    ]
    frozen = {str(s): sci.folds([r["family_id"] for r in train], s) for s in sci.SEEDS}
    monkeypatch.setattr(
        runner, "blob", lambda *_: json.dumps(dict(folds=frozen)).encode()
    )

    class FakePools:
        def get(self, _cid):
            return dict(fitness=np.array([0.0, 1.0]), gate=np.array([True, True]))

    class FakeEngine:
        output = tmp_path
        pools = FakePools()
        progress = dict(updates=0, fits=0)
        candidate_reads = 0

        def fit_proposal(self, rows, config, _seed, label):
            assert not (tmp_path / "selection-lock.json").exists()
            self.progress["updates"] += config["steps"]
            self.progress["fits"] += 1
            return dict(
                label=label, fit_families=sorted({r["family_id"] for r in rows})
            )

        def baseline(self, rows, _shuffled=None):
            return {
                r["context_sha256"]: dict(utility=0.0, regret=1.0, shuffled_regret=1.0)
                for r in rows
            }

        def evaluate_proposal(self, state, rows, _shuffled=None, *, role, forbidden=()):
            locked = hasattr(self, "selection_lock_sha256")
            sci.require_score_role(
                [r["family_id"] for r in rows],
                state["fit_families"],
                role,
                locked,
                forbidden,
            )
            if role != "FIT":
                assert (
                    runner.sha(tmp_path / "selection-lock.json")
                    == self.selection_lock_sha256
                )
                assert self.progress == dict(fits=360, updates=8280)
                self.candidate_reads += 1
            scores = self.baseline(rows)
            if positive:
                for row in scores.values():
                    row.update(utility=0.1, regret=0.9, shuffled_regret=0.9)
            return scores

        def save(self, name, value):
            runner.save(tmp_path / name, value)

    engine = FakeEngine()
    result = runner.run_experiment(
        "fixture", {}, tmp_path, engine, sci, train, val, np.zeros(91)
    )
    assert result["status"] == "COMPLETE"
    assert result["proposal_count"] == 120
    assert result["verdict"] == ("CANDIDATE" if positive else "NO_ADVANCE")
    assert result["plateau_verdict"] == (
        "PLATEAU_WIN" if positive else "NO_PLATEAU_WIN"
    )
    assert engine.candidate_reads == (135 if positive else 120)
    assert result["distinct_DL_Val_values"] == 1
