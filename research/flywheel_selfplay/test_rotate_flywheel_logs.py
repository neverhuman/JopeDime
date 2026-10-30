"""Custody rotation failure fences and an actual open-descriptor append proof."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import rotate_flywheel_logs as rotation


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    monkeypatch.setattr(rotation, "MIN_FREE_BYTES", 0)
    remote = tmp_path / "remote"
    remote.mkdir()
    calls = []

    def publish(path, prefix):
        target = remote / path.name
        shutil.copy2(path, target)
        assert target.read_bytes() == path.read_bytes()
        calls.append(path)
        return {
            "uri": "fixture://" + target.name,
            "version_id": "fixture-version",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }

    monkeypatch.setattr(rotation, "publish_checkpoint_archive", publish)
    return calls


def text_file(tmp_path):
    source = tmp_path / "summary.jsonl"
    source.write_bytes(b'{"round":1}\n{"round":2}\n{"round":3}\npartial-\xff')
    return source


def test_rotates_only_after_full_archive_restore_and_keeps_exact_tail(
    tmp_path, publisher
):
    source = text_file(tmp_path)
    before = source.read_bytes()
    source.chmod(0o640)
    inode = source.stat().st_ino
    receipt = rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == b'{"round":3}\npartial-\xff'
    assert source.stat().st_ino == inode and source.stat().st_mode & 0o777 == 0o640
    assert receipt["removed_rows"] == 2 and receipt["status"] == "rotated"
    assert len(publisher) == 2
    assert subprocess.check_output(["zstd", "-dc", receipt["archive"]]) == before
    disk_receipt = json.loads(
        next((tmp_path / "archives").glob("*.receipt.json")).read_text()
    )
    assert disk_receipt["status"] == "rotated"
    assert disk_receipt["s3_archive"]["version_id"]
    assert rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://") is None


@pytest.mark.parametrize("failed_call", [1, 2])
def test_s3_failure_never_changes_source(tmp_path, publisher, monkeypatch, failed_call):
    source = text_file(tmp_path)
    before = source.read_bytes()
    publish = rotation.publish_checkpoint_archive
    calls = 0

    def fail(path, prefix):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise ValueError("injected S3 readback failure")
        return publish(path, prefix)

    monkeypatch.setattr(rotation, "publish_checkpoint_archive", fail)
    with pytest.raises(ValueError, match="S3"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == before


def test_source_append_during_upload_prevents_rotation(
    tmp_path, publisher, monkeypatch
):
    source = text_file(tmp_path)
    before = source.read_bytes()
    publish = rotation.publish_checkpoint_archive

    def append(path, prefix):
        result = publish(path, prefix)
        if path.suffix == ".zst":
            with source.open("ab") as stream:
                stream.write(b"-arrived-during-network\n")
        return result

    monkeypatch.setattr(rotation, "publish_checkpoint_archive", append)
    with pytest.raises(ValueError, match="changed"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == before + b"-arrived-during-network\n"


def test_corrupt_compressed_copy_never_authorizes_rotation(
    tmp_path, publisher, monkeypatch
):
    source = text_file(tmp_path)
    before = source.read_bytes()
    publish = rotation.publish_checkpoint_archive

    def corrupt(path, prefix):
        result = publish(path, prefix)
        if path.suffix == ".zst":
            path.write_bytes(b"corrupt-after-upload")
        return result

    monkeypatch.setattr(rotation, "publish_checkpoint_archive", corrupt)
    with pytest.raises(ValueError, match="changed"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "append, authorized", [(True, True), (False, True), (True, False)]
)
def test_open_primary_descriptor_appends_without_holes_or_requires_safe_restart(
    tmp_path, publisher, append, authorized
):
    source = text_file(tmp_path)
    before = source.read_bytes()
    program = "import sys; f=open(sys.argv[1],sys.argv[2]); f.seek(0,2); print('ready',flush=True); sys.stdin.readline(); f.write(b'later\\n'); f.flush()"
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(source), "ab" if append else "r+b"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline() == b"ready\n"
        if append and authorized:
            rotation.rotate_file(
                source, tmp_path / "archives", 2, "fixture://", child.pid
            )
            expected = b'{"round":3}\npartial-\xfflater\n'
        else:
            with pytest.raises(ValueError, match="O_APPEND|uncoordinated"):
                rotation.rotate_file(
                    source,
                    tmp_path / "archives",
                    2,
                    "fixture://",
                    child.pid if authorized else None,
                )
            expected = before + b"later\n"
        child.communicate(b"go\n", timeout=10)
        assert child.returncode == 0 and source.read_bytes() == expected
        assert b"\0" not in source.read_bytes()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_active_reader_and_symlink_are_not_rotated(tmp_path, publisher):
    source = text_file(tmp_path)
    before = source.read_bytes()
    with source.open("rb"), pytest.raises(ValueError, match="uncoordinated"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    link = tmp_path / "link.log"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="nonsymlink"):
        rotation.rotate_file(link, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == before


def test_allowlist_preserves_models_scores_journals_and_continues_after_failure(
    tmp_path, publisher
):
    run, custody = tmp_path / "run", tmp_path / "custody"
    run.mkdir()
    custody.mkdir()
    for path in [
        run / "summary.jsonl",
        custody / "worker.log",
        run / "round-882.pt",
        run / "promotions.jsonl",
        run / "round-1-scores.jsonl",
        custody / "AGENT_CHAT.dirty.t533.md",
    ]:
        path.write_bytes(b"one\ntwo\nthree\n")
    unsafe = custody / "unsafe.log"
    unsafe.symlink_to(run / "round-882.pt")
    preserved = {
        p: p.read_bytes()
        for p in [
            run / "round-882.pt",
            run / "promotions.jsonl",
            run / "round-1-scores.jsonl",
            custody / "AGENT_CHAT.dirty.t533.md",
        ]
    }
    result = rotation.rotate_logs(
        run, custody, tmp_path / "archives", 2, s3_prefix="fixture://"
    )
    assert [r["status"] for r in result].count("rotated") == 2
    assert any(r["status"] == "deferred" for r in result)
    assert all(p.read_bytes() == body for p, body in preserved.items())
    assert (
        (run / "summary.jsonl").read_bytes()
        == (custody / "worker.log").read_bytes()
        == b"two\nthree\n"
    )


@pytest.mark.parametrize("where", ["run", "custody", "checkout"])
def test_archives_inside_source_or_checkout_are_refused(tmp_path, publisher, where):
    run, custody = tmp_path / "run", tmp_path / "custody"
    run.mkdir()
    custody.mkdir()
    archives = {"run": run, "custody": custody, "checkout": Path(__file__).parent}[
        where
    ]
    with pytest.raises(ValueError, match="outside"):
        rotation.rotate_logs(run, custody, archives)


def test_census_errors_refuse_all_mutation(tmp_path, publisher, monkeypatch):
    source = text_file(tmp_path)
    before = source.read_bytes()
    original = subprocess.run

    def unavailable(argv, **kwargs):
        if argv[0] == "lsof":
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="unavailable")
        return original(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="unavailable"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == before


def test_free_space_reserve_refuses_rotation(tmp_path, publisher, monkeypatch):
    source = text_file(tmp_path)
    original = source.read_bytes()
    monkeypatch.setattr(rotation, "MIN_FREE_BYTES", 10**30)
    with pytest.raises(ValueError, match="headroom"):
        rotation.rotate_file(source, tmp_path / "archives", 2, "fixture://")
    assert source.read_bytes() == original and not publisher


def test_cli_holds_actual_round_lock_until_all_archive_verification_and_rewrite(
    tmp_path, publisher, monkeypatch
):
    import fcntl

    run, custody = tmp_path / "run", tmp_path / "custody"
    run.mkdir()
    custody.mkdir()
    (run / "summary.jsonl").write_bytes(b"one\ntwo\nthree\n")
    round_lock = tmp_path / "qualification.lock"
    publish = rotation.publish_checkpoint_archive
    checks = []

    def verify_lock(path, prefix):
        with round_lock.open("a") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks.append(True)
        return publish(path, prefix)

    monkeypatch.setattr(rotation, "publish_checkpoint_archive", verify_lock)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rotate",
            "--run",
            str(run),
            "--custody",
            str(custody),
            "--archive-root",
            str(tmp_path / "archives"),
            "--round-lock",
            str(round_lock),
            "--writer-pid",
            str(os.getpid()),
            "--keep-rows",
            "2",
            "--once",
        ],
    )
    rotation.main()
    assert len(checks) == 2 and (run / "summary.jsonl").read_bytes() == b"two\nthree\n"
    with round_lock.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_per_boundary_batch_limit_still_drains_closed_log_backlog(tmp_path, publisher):
    run, custody = tmp_path / "run", tmp_path / "custody"
    run.mkdir()
    custody.mkdir()
    for name in ["a.log", "b.log", "c.log", "d.log"]:
        (custody / name).write_bytes(b"1\n2\n3\n")
    first = rotation.rotate_logs(
        run, custody, tmp_path / "archives", 2, s3_prefix="fixture://", max_files=2
    )
    assert (
        len(first) == 2
        and sum(p.read_bytes() == b"2\n3\n" for p in custody.glob("*.log")) == 2
    )
    second = rotation.rotate_logs(
        run, custody, tmp_path / "archives", 2, s3_prefix="fixture://", max_files=2
    )
    assert len(second) == 2 and all(
        p.read_bytes() == b"2\n3\n" for p in custody.glob("*.log")
    )
