"""Data-free controls for H152 soups and readout (synthetic tensors and gains only)."""

import importlib.util
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("h152_test_runner", HERE / "runner.py")
r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r
SPEC.loader.exec_module(r)


def states():
    base = {
        "adapter_10tok.w": torch.tensor([0.0, 4.0]),
        "adapter_10tok.anchor.b": torch.tensor([1.0]),
        "dope_model.x": torch.tensor([7.0]),
    }
    out = {}
    for rnd, v in ((782, 2.0), (785, 4.0), (859, 6.0), (882, 8.0)):
        s = {k: t.clone() for k, t in base.items()}
        s["adapter_10tok.w"] = torch.tensor([v, v])
        out[rnd] = s
    return out


def test_arms_are_fixed_and_weights_sum_to_one():
    assert set(r.ARMS) == {"R882", "MID", "SOUP4", "A025", "A075"}
    assert r.ADMISSIBLE == ("MID", "SOUP4") and set(r.EXPLORATORY) == {"A025", "A075"}
    assert all(abs(sum(w.values()) - 1) < 1e-12 for w in r.ARMS.values())
    assert r.ARMS["MID"] == {782: 0.5, 882: 0.5} and r.SEEDS == (152101, 152102, 152103)


def test_blend_averages_only_trainable_tensors_and_keeps_r882_otherwise():
    s = states()
    s[782]["dope_model.x"] = torch.tensor(
        [99.0]
    )  # a differing frozen tensor must not leak into the soup
    mid = r.blend(s, r.ARMS["MID"], {"adapter_10tok.w"})
    assert mid["adapter_10tok.w"].tolist() == [5.0, 5.0]
    assert mid["dope_model.x"].tolist() == [7.0] and mid[
        "adapter_10tok.anchor.b"
    ].tolist() == [1.0]
    soup = r.blend(s, r.ARMS["SOUP4"], {"adapter_10tok.w"})
    assert soup["adapter_10tok.w"].tolist() == [5.0, 5.0]
    assert r.frozen_identity(s, {"adapter_10tok.w"}) == ["dope_model.x"]
    with pytest.raises(ValueError):
        r.blend(s, {782: 0.7, 882: 0.7}, {"adapter_10tok.w"})


def test_gain_readout_is_deterministic_and_ties_score_half():
    g = np.zeros(19)
    out = r.gain_readout(g)
    assert (
        out["gain"] == 0
        and out["elo"] == pytest.approx(1500.0)
        and out["families_win_tie_loss"] == [0, 19, 0]
    )
    assert out == r.gain_readout(g)


def test_decision_requires_positive_gain_ci_and_holm():
    pos = r.gain_readout(np.full(19, 0.01))
    tie = r.gain_readout(np.zeros(19))
    d = r.decide(pos, tie)
    assert d["MID"]["advance"] and not d["SOUP4"]["advance"]
    d = r.decide(tie, tie)
    assert not d["MID"]["advance"] and not d["MID"]["gates"]["gain_gt_0"]
    neg = r.gain_readout(np.full(19, -0.01))
    assert not r.decide(neg, neg)["MID"]["advance"]


def test_concentrated_gain_is_not_admitted_without_ci_support():
    g = np.zeros(19)
    g[0] = 0.5  # one family carries everything
    out = r.gain_readout(g)
    assert out["median_ci95"][0] <= 0 and not r.decide(out, out)["MID"]["advance"]


def test_elo_ci_brackets_point():
    g = np.r_[np.full(10, 0.01), np.full(9, -0.01)]
    lo, hi = r.elo_ci(g)
    assert lo <= r.gain_readout(g)["elo"] <= hi


# ------------------------------------------------------------------ guards, binding and budget (negative controls)

MSPEC = importlib.util.spec_from_file_location("h152_test_monitor", HERE / "monitor.py")
mon = importlib.util.module_from_spec(MSPEC)
MSPEC.loader.exec_module(mon)


class Usage:
    def __init__(self, total, free):
        self.total, self.free = total, free


def healthy(monkeypatch, free=118_000_000_000, total=1_966_000_000_000, load=17.0):
    monkeypatch.setattr(r.os, "getpriority", lambda *a: 19)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: Usage(total, free))
    monkeypatch.setattr(r.os, "getloadavg", lambda: (load, load, load))


def test_guards_pass_at_xbabe1_94_percent_and_stop_on_each_limit(monkeypatch, tmp_path):
    healthy(monkeypatch)  # 94.0% used, 118 GB free: must run (absolute guard, not 94%)
    assert r.guards(tmp_path)["disk_used_fraction"] < 0.95
    healthy(monkeypatch, free=39_000_000_000)
    with pytest.raises(RuntimeError, match="disk stop"):
        r.guards(tmp_path)
    healthy(monkeypatch, free=90_000_000_000, total=1_900_000_000_000)  # 95.3% used
    with pytest.raises(RuntimeError, match="disk stop"):
        r.guards(tmp_path)
    healthy(monkeypatch, load=85.5)
    with pytest.raises(RuntimeError, match="load stop"):
        r.guards(tmp_path)
    healthy(monkeypatch)
    (tmp_path / "big").write_bytes(b"\0")
    monkeypatch.setattr(r, "SCRATCH_LIMIT_BYTES", 1)
    with pytest.raises(RuntimeError, match="scratch stop"):
        r.guards(tmp_path)
    monkeypatch.setattr(r.os, "getpriority", lambda *a: 0)
    with pytest.raises(RuntimeError, match="nice"):
        r.guards(tmp_path)


def test_supervisor_stops_on_every_limit_and_never_pauses():
    ok = dict(
        rss=1e9,
        threads=3,
        free=118e9,
        disk_used=0.94,
        scratch=7e7,
        load=17.0,
        available=40 * 1024**3,
    )
    assert mon.classify(ok) == ("run", None)
    for key, bad, why in [
        ("rss", 6e9, "RSS"),
        ("threads", 5, "TOTAL_THREADS"),
        ("free", 39e9, "DISK"),
        ("disk_used", 0.95, "DISK"),
        ("scratch", 1e9, "SCRATCH"),
        ("load", 85.1, "LOAD"),
        ("available", 11 * 1024**3, "MEMORY"),
    ]:
        assert mon.classify(dict(ok, **{key: bad})) == ("abort", why)


def test_hardcoded_torch_four_threads_is_held_at_one():
    def baseline_like():
        torch.set_num_threads(4)
        return torch.get_num_threads()

    before = torch.get_num_threads()
    try:
        assert r.bounded_threads(baseline_like) == r.TORCH_THREADS == 1
    finally:
        torch.set_num_threads(before)


def git(repo, *args):
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
        },
    ).strip()


def bound_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    files = {"a/PREREG.md": b"prereg\n", "a/runner.py": b"print(1)\n"}
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "reviewed")
    reviewed = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    squash = git(
        repo, "commit-tree", tree, "-m", "squash"
    )  # same blobs, no ancestry to the reviewed commit
    monkeypatch.setattr(r, "BOUND_FILES", tuple(files))
    monkeypatch.setattr(r, "PREREG_SHA", r.sha(repo / "a/PREREG.md"))
    return repo, reviewed, squash


def test_binding_accepts_squash_merge_and_rejects_mismatched_source(
    tmp_path, monkeypatch
):
    repo, reviewed, squash = bound_repo(tmp_path, monkeypatch)
    out = r.bind_source({"reviewed_head": reviewed, "source_commit": squash}, repo=repo)
    assert out["source_commit"] == squash and set(out["bound_sha256"]) == set(
        r.BOUND_FILES
    )
    (repo / "a/runner.py").write_bytes(
        b"print(2)\n"
    )  # executed bytes edited after review
    with pytest.raises(RuntimeError, match="executed bytes differ"):
        r.bind_source({"reviewed_head": reviewed, "source_commit": squash}, repo=repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "changed source")
    changed = git(
        repo, "rev-parse", "HEAD"
    )  # merged source no longer equals the reviewed head
    with pytest.raises(RuntimeError, match="executed bytes differ"):
        r.bind_source({"reviewed_head": reviewed, "source_commit": changed}, repo=repo)
    monkeypatch.setattr(r, "PREREG_SHA", "0" * 64)
    with pytest.raises(RuntimeError, match="PREREG"):
        r.bind_source({"reviewed_head": reviewed, "source_commit": squash}, repo=repo)


def test_regression_requires_exact_contexts_and_mean():
    vals = np.linspace(0.5, 1.7, 91)
    ref = {
        "utility": {"per_context": vals.tolist()},
        "best_utility": float(vals.mean()),
    }
    assert r.regression(vals.tolist(), ref)["contexts_matched"] == 91
    off = vals.copy()
    off[7] = np.nextafter(off[7], 9)  # one ulp is a failure
    with pytest.raises(RuntimeError, match="91/91"):
        r.regression(off.tolist(), ref)
    with pytest.raises(RuntimeError, match="mean"):
        r.regression(vals.tolist(), dict(ref, best_utility=ref["best_utility"] + 1e-15))
    with pytest.raises(RuntimeError, match="91/91"):
        r.regression(vals[:90].tolist(), ref)


def test_train_folds_are_family_disjoint_complete_and_seeded():
    fams = [f"f{i % 205}" for i in range(1219)]
    a, b = r.train_folds(fams, 152101), r.train_folds(fams, 152102)
    flat = [f for g in a for f in g]
    assert (
        len(a) == 5
        and sorted(flat) == sorted(set(fams))
        and len(flat) == len(set(flat)) == 205
    )
    assert a == r.train_folds(fams, 152101) and a != b


def test_run_refuses_mixed_modes_and_foreign_roots(tmp_path):
    with pytest.raises(RuntimeError):
        r.run(tmp_path / "run-x")  # not under ~/jope-data/h152
