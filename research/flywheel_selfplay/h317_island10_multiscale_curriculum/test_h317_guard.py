"""H317 read-guard v2 tests (leakage red team probes, SYNTHETIC paths only; no DL-Val byte is ever opened).

Every probe uses a nonexistent path whose NAME carries a DL-Val identifier, a synthetic file registered as a forbidden
inode, or a denylisted location. A refusal is GuardViolation; "allowed" probes may raise FileNotFoundError.
Run with H317_DATA pointed at a scratch dir so the cross-process refusal log of real runs stays clean.
"""

from __future__ import annotations

import os
import subprocess

import pytest

import common_h317 as C

NX = C.DATA_ROOT / "guardtest" / "nonexistent_synthetic"


def _dev():
    sel = C.load_selected()
    return (
        sel["dev"][0]["context_sha256"],
        sel["dev"][0]["dataset_id"],
        sel["train"][0]["context_sha256"],
    )


def _refused(fn) -> bool:
    try:
        fn()
    except C.GuardViolation:
        return True
    except (FileNotFoundError, OSError):
        return False
    return False


def test_guard_installed_at_import():
    assert C.guard_report()["installed"] is True
    assert C.guard_report()["guard_version"] == 2


@pytest.mark.parametrize(
    "name",
    [
        "{d}_v0.npz",
        "{d}-search.jtt1",
        "enc_{d}.npz",
        "{p12}.search.jtt1",
        "{p16}.npz",
        "{u}.jtt1",
        "{ds}/train.csv",
        "x/dev91_tables.npz",
        "x/DL-Val_rows.json",
        "x/outer_test.npz",
    ],
)
def test_dev_named_paths_refused(name):
    d, ds, _ = _dev()
    nm = name.format(d=d, p12=d[:12], p16=d[:16], u=d.upper(), ds=ds)
    assert _refused(lambda: open(NX / nm, "rb"))


def test_train_paths_allowed():
    _, _, t = _dev()
    assert not _refused(lambda: open(NX / f"{t}.search.w.jtt1", "rb"))
    assert not _refused(lambda: open(C.table_path(t), "rb").close())
    assert not _refused(lambda: open(C.RES01 / "selected.json", "rb").close())


@pytest.mark.parametrize(
    "path",
    [
        "/home/ubuntu/jope-data/res-01/full-packet.tar",
        "/home/ubuntu/JopeDimeMicro/assets/benchmark_data/x.json",
        "/home/ubuntu/jope-evidence/a/b.csv",
        "/home/ubuntu/jope-data/gov-03/custody/x.json",
        "/home/ubuntu/flywheel/data/x/train.csv",
        "/home/ubuntu/scratch/x.json",
        "/home/ubuntu/JopeDime/research/flywheel_selfplay/custody/files/home/ubuntu/x.jtt1",
        "/home/ubuntu/jope-data/h257/x.pt",
        "/home/ubuntu/jope-data/h256/x.pt",
        "/home/ubuntu/x/sealed/a.json",
    ],
)
def test_denylisted_trees_refused(path):
    assert _refused(lambda: C.assert_allowed(path))


def test_mixed_h58_direct_refused_and_prefix_requires_loader():
    p = C.H58_CACHE / "pool_priors.npz"
    if C._stat_key(p) is None:
        pytest.skip("h58 cache not present on this host/sandbox")
    assert _refused(lambda: open(p, "rb").close())
    with C.mixed_prefix_read():  # not called from trainonly_loader.py -> still refused
        assert _refused(lambda: open(p, "rb").close())


def test_symlink_and_hardlink_by_inode_refused(tmp_path):
    target = tmp_path / "syn_target.bin"
    target.write_bytes(b"SYNTHETIC")
    hl = tmp_path / "syn_hardlink.bin"
    os.link(target, hl)
    sl = tmp_path / "syn_symlink.bin"
    os.symlink(
        target, sl
    )  # links made BEFORE the inode is registered (afterwards os.link/os.symlink are refused too)
    key = C._stat_key(target)
    old = C._STATE["dev_inodes"]
    C._STATE["dev_inodes"] = frozenset(old | {key})
    try:
        assert _refused(lambda: open(hl, "rb"))
        assert _refused(lambda: open(sl, "rb"))
        assert _refused(lambda: os.link(target, tmp_path / "late_link.bin"))
    finally:
        C._STATE["dev_inodes"] = old
    d, _, _ = _dev()
    alias = tmp_path / "alias.jtt1"
    # Creating an alias to a dev-named target is itself refused (os.symlink audit checks both names).
    assert _refused(lambda: os.symlink(NX / f"{d}.jtt1", alias))


def test_exec_argv_and_link_rename_refused(tmp_path):
    d, _, _ = _dev()
    assert _refused(
        lambda: subprocess.run(["cat", str(NX / f"{d}.jtt1")], capture_output=True)
    )
    assert _refused(lambda: os.system(f"test -r {NX}/{d}.jtt1"))
    assert _refused(lambda: os.link(NX / f"{d}.jtt1", tmp_path / "hl2"))
    assert _refused(lambda: os.rename(NX / f"{d}.jtt1", tmp_path / "rn"))
    assert not _refused(
        lambda: subprocess.run([str(C.WORKER_BIN), "--help"], capture_output=True)
    )


def test_stage_v_unlock_rejects_caller_sha_and_is_pinned(tmp_path):
    with pytest.raises((C.GuardViolation, FileNotFoundError, OSError)):
        C.unlock_stage_v(
            tmp_path / "LOCK.json", "0" * 64
        )  # caller args are ignored; receipt/LOCK absent -> raises
    assert C._STATE["stage_v"] is False


def test_refusals_logged_cross_process():
    before = C.guard_report()["cross_process_refusal_log_lines"]
    d, _, _ = _dev()
    # The probe path travels in the environment: putting it in argv would (correctly) be refused by the parent's
    # exec-argv scan before the child even starts.
    code = "import os, common_h317 as C\ntry:\n open(os.environ['H317_PROBE'],'rb')\nexcept C.GuardViolation:\n pass\n"
    env = {**os.environ, "H317_PROBE": f"{NX}/{d}.jtt1"}
    subprocess.run(["python3", "-c", code], cwd=str(C.HERE), env=env, check=True)
    assert C.guard_report()["cross_process_refusal_log_lines"] >= before + 1


def test_no_train_identifier_triggers_dev_matcher():
    for r in C.load_selected()["train"]:
        assert not C._dev_named(r["context_sha256"])
