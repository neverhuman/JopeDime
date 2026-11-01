#!/usr/bin/env python3
"""Verify every vendored file listed by the RES-01 custody manifest."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import posixpath
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

CUSTODY_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CUSTODY_DIR / "MANIFEST.json"
FILES_DIR = CUSTODY_DIR / "files"
EXTERNAL_PATH = CUSTODY_DIR / "EXTERNAL.json"
REPO_ROOT = CUSTODY_DIR.parents[2]
CHUNK_SIZE = 1024 * 1024
FILE_COLLECTIONS = ("import_files", "assets", "evaluator.compiled_source_inputs")


def _fail(message: str) -> None:
    print(f"custody verification failed: {message}", file=sys.stderr)


def _compiled_sources(manifest: dict) -> list[dict]:
    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("evaluator must be an object")
    root = evaluator.get("source_root")
    if (
        not isinstance(root, str)
        or not root.startswith("/")
        or any(part in ("", ".", "..") for part in root.split("/")[1:])
    ):
        raise ValueError("evaluator has an unsafe source_root")
    rows = evaluator.get("compiled_source_inputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluator.compiled_source_inputs must be a non-empty list")
    sources = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("compiled source entry must be an object")
        relative = row.get("repo_path")
        if not isinstance(relative, str) or any(
            part in ("", ".", "..") for part in relative.split("/")
        ):
            raise ValueError("compiled source has an unsafe repo_path")
        canonical = root + "/" + relative
        original = row.get("path")
        if not isinstance(original, str) or posixpath.normpath(original) != canonical:
            raise ValueError("compiled source path disagrees with its repository path")
        digest = row.get("actual_sha256")
        if (
            digest != row.get("revision_sha256")
            or row.get("matches_revision") is not True
        ):
            raise ValueError("compiled source hash disagrees with its pinned revision")
        sources.append({"path": canonical, "sha256": digest, "bytes": None})
    return sources


def _manifest_entries(
    manifest: object,
) -> tuple[dict[Path, tuple[str, int | None]], int]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")

    entries: dict[str, tuple[Path, str, int | None]] = {}
    references = 0
    for collection in FILE_COLLECTIONS:
        rows = (
            _compiled_sources(manifest)
            if collection == FILE_COLLECTIONS[2]
            else manifest.get(collection)
        )
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{collection} must be a non-empty list")

        for index, entry in enumerate(rows):
            references += 1
            if not isinstance(entry, dict):
                raise ValueError(f"{collection}[{index}] is not an object")

            source_path = entry.get("path")
            expected_hash = entry.get("sha256")
            expected_size = entry.get("bytes")
            if not isinstance(source_path, str) or not source_path.startswith("/"):
                raise ValueError(f"{collection}[{index}] has no absolute source path")
            path_parts = source_path.split("/")[1:]
            if not path_parts or any(part in ("", ".", "..") for part in path_parts):
                raise ValueError(f"unsafe source path: {source_path}")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)
            ):
                raise ValueError(f"{collection}[{index}] has an invalid SHA-256")
            if collection != FILE_COLLECTIONS[2] and (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
            ):
                raise ValueError(f"{collection}[{index}] has an invalid byte count")

            relative = Path(*path_parts)
            value = (relative, expected_hash, expected_size)
            previous = entries.get(source_path)
            if previous is not None:
                if previous[1] != expected_hash or (
                    previous[2] is not None
                    and expected_size is not None
                    and previous[2] != expected_size
                ):
                    raise ValueError(f"manifest collections disagree on {source_path}")
                if expected_size is None:
                    value = previous
            entries[source_path] = value

    relative_entries: dict[Path, tuple[str, int | None]] = {}
    for source_path, (relative, expected_hash, expected_size) in entries.items():
        previous = relative_entries.get(relative)
        value = (expected_hash, expected_size)
        if previous is not None and previous != value:
            raise ValueError(
                f"distinct manifest paths map to conflicting custody path {relative}"
            )
        relative_entries[relative] = value
    return relative_entries, references


def _actual_files(root: Path) -> set[Path]:
    actual: set[Path] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        directory = Path(current)
        for name in list(directories):
            path = directory / name
            if path.is_symlink():
                raise ValueError(f"custody tree contains a symlink: {path}")
        for name in filenames:
            path = directory / name
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"custody path is not a regular file: {path}")
            actual.add(path.relative_to(root))
    return actual


def _external_entries(document: dict, expected: dict) -> dict[Path, dict]:
    if (
        not isinstance(document, dict)
        or document.get("schema") != "jope.external-custody/v1"
    ):
        raise ValueError("unknown external custody schema")
    commit = document.get("recovery_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("external custody needs an immutable full Git commit")
    rows = document.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("external custody files must be a non-empty list")
    external = {}
    for row in rows:
        relative = Path(row["path"].lstrip("/"))
        if row["path"] != "/" + relative.as_posix() or relative not in expected:
            raise ValueError(
                "external file is absent from the original custody manifest"
            )
        if relative in external or (row["sha256"], row["bytes"]) != expected[relative]:
            raise ValueError("external pin differs from the original custody manifest")
        if row["git_path"] != "research/flywheel_selfplay/custody/files/" + str(
            relative
        ):
            raise ValueError(
                "external recovery path differs from the original custody path"
            )
        object_id = row.get("git_object_id", "")
        if len(object_id) != 40 or any(c not in "0123456789abcdef" for c in object_id):
            raise ValueError("invalid external Git object identity")
        external[relative] = dict(row, recovery_commit=commit)
    return external


@contextlib.contextmanager
def _open_external(row: dict, root: Path | None):
    if root is not None:
        target = root / "sha256" / row["sha256"]
        if not root.is_absolute() or any(
            p.is_symlink() for p in (target, *target.parents)
        ):
            raise ValueError("external cache must use absolute, non-symlink paths")
        if not stat.S_ISREG(target.lstat().st_mode):
            raise ValueError("external cache payload is not a regular file")
        with target.open("rb") as stream:
            yield stream
        return
    # Existing immutable history is a recovery route, not permission to put payloads back in HEAD.
    git = ["git", "-C", str(REPO_ROOT)]
    try:
        object_id = subprocess.check_output(
            git
            + ["rev-parse", "--verify", row["recovery_commit"] + ":" + row["git_path"]],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "pinned Git recovery unavailable; supply --external-root CACHE"
        ) from error
    if object_id != row["git_object_id"]:
        raise ValueError("external Git recovery object identity differs")
    with subprocess.Popen(
        git + ["cat-file", "blob", object_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise ValueError(
                    "cannot read pinned Git recovery blob: " + error.strip()
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root", type=Path, help="cache containing sha256/<digest> files"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected, references = _manifest_entries(manifest)
        external = _external_entries(json.loads(EXTERNAL_PATH.read_text()), expected)
        files_root = FILES_DIR.resolve(strict=True)
        files_root.relative_to(CUSTODY_DIR.resolve(strict=True))
        if FILES_DIR.is_symlink() or not files_root.is_dir():
            raise ValueError("custody files path must be a real directory")
        actual = _actual_files(files_root)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        _fail(str(exc))
        return 1

    vendored = set(expected) - set(external)
    missing = sorted(vendored - actual)
    extra = sorted(actual - vendored)
    if missing or extra:
        for path in missing[:10]:
            _fail(f"missing vendored file: {path}")
        for path in extra[:10]:
            _fail(f"unlisted vendored file: {path}")
        _fail(f"file inventory mismatch: {len(missing)} missing, {len(extra)} unlisted")
        return 1

    errors = 0
    for relative, (expected_hash, expected_size) in expected.items():
        target = FILES_DIR / relative
        try:
            digest = hashlib.sha256()
            size = 0
            with contextlib.ExitStack() as stack:
                if relative in external:
                    file = stack.enter_context(
                        _open_external(external[relative], args.external_root)
                    )
                else:
                    resolved = target.resolve(strict=True)
                    resolved.relative_to(files_root)
                    if target.is_symlink() or not resolved.is_file():
                        raise ValueError(
                            "not a regular file inside the custody directory"
                        )
                    file = stack.enter_context(resolved.open("rb"))
                while chunk := file.read(CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
        except (OSError, ValueError) as exc:
            _fail(f"cannot verify {relative}: {exc}")
            errors += 1
            continue

        if digest.hexdigest() != expected_hash:
            _fail(f"SHA-256 mismatch for {relative}: got {digest.hexdigest()}")
            errors += 1
        if expected_size is not None and size != expected_size:
            _fail(
                f"byte count mismatch for {relative}: expected {expected_size}, got {size}"
            )
            errors += 1

    if errors:
        print(
            f"checked {len(expected)} unique custody files; found {errors} error(s)",
            file=sys.stderr,
        )
        return 1

    print(
        f"verified {len(expected)} unique custody files from {references} manifest entries "
        f"across {', '.join(FILE_COLLECTIONS)}; "
        f"{len(vendored)} vendored and {len(external)} external (all bytes checked)"
    )
    return 0


def self_test() -> None:
    # End-to-end controls exercise all three manifest collections without opening model payloads.
    with tempfile.TemporaryDirectory(prefix="custody-controls-") as directory:
        root = Path(directory)
        files = root / "files"
        files.mkdir()
        cache = root / "cache"
        (cache / "sha256").mkdir(parents=True)
        content = b"frozen binary input"
        digest = hashlib.sha256(content).hexdigest()
        pinned = {"path": "/binary", "sha256": digest, "bytes": len(content)}
        source = b"source"
        source_hash = hashlib.sha256(source).hexdigest()
        (files / "asset").write_bytes(source)
        (files / "src").mkdir()
        (files / "src/input.rs").write_bytes(source)
        manifest = {
            "import_files": [pinned],
            "assets": [{"path": "/asset", "sha256": source_hash, "bytes": len(source)}],
            "evaluator": {
                "source_root": "/src",
                "compiled_source_inputs": [
                    {
                        "path": "/src/input.rs",
                        "repo_path": "input.rs",
                        "actual_sha256": source_hash,
                        "revision_sha256": source_hash,
                        "matches_revision": True,
                    }
                ],
            },
        }
        external = {
            "schema": "jope.external-custody/v1",
            "recovery_commit": "a" * 40,
            "files": [
                dict(
                    pinned,
                    git_object_id="b" * 40,
                    git_path="research/flywheel_selfplay/custody/files/binary",
                )
            ],
        }
        manifest_path = root / "MANIFEST.json"
        external_path = root / "EXTERNAL.json"
        manifest_path.write_text(json.dumps(manifest))
        external_path.write_text(json.dumps(external))
        payload = cache / "sha256" / digest
        payload.write_bytes(content)
        names = {
            "CUSTODY_DIR": root,
            "FILES_DIR": files,
            "MANIFEST_PATH": manifest_path,
            "EXTERNAL_PATH": external_path,
        }
        previous = {key: globals()[key] for key in names}
        globals().update(names)
        count = 0

        def check(expected_code):
            nonlocal count
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                assert main(["--external-root", str(cache)]) == expected_code
            count += 1

        try:
            check(0)
            payload.write_bytes(b"x" * len(content))
            check(1)  # same-size corruption
            payload.write_bytes(content[:-1])
            check(1)  # truncation
            payload.unlink()
            check(1)  # unavailable is not PASS
            payload.symlink_to(files / "asset")
            check(1)
            payload.unlink()
            payload.write_bytes(content)
            (files / "extra").write_bytes(b"extra")
            check(1)
            (files / "extra").unlink()
            external["files"][0]["bytes"] += 1
            external_path.write_text(json.dumps(external))
            check(1)
            external["files"][0]["bytes"] -= 1
            external["files"].append(dict(external["files"][0]))
            external_path.write_text(json.dumps(external))
            check(1)
            external["files"].pop()
            original_git_path = external["files"][0]["git_path"]
            external["files"][0]["git_path"] = "wrong/path"
            external_path.write_text(json.dumps(external))
            check(1)
            external["files"][0]["git_path"] = original_git_path
            external_path.write_text(json.dumps(external))
            hashes = cache / "sha256"
            saved = cache / "saved"
            hashes.rename(saved)
            hashes.symlink_to(saved, target_is_directory=True)
            check(1)
            hashes.unlink()
            saved.rename(hashes)
            from unittest.mock import patch

            for result in (subprocess.CalledProcessError(128, "git"), "c" * 40):
                with patch(
                    "subprocess.check_output",
                    side_effect=result if isinstance(result, Exception) else None,
                    return_value=result if isinstance(result, str) else None,
                ):
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        assert main([]) == 1
                    count += 1
        finally:
            globals().update(previous)
    print(f"external custody verifier controls: {count} passed")


if __name__ == "__main__":
    raise SystemExit(main())
