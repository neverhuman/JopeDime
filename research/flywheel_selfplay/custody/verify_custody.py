#!/usr/bin/env python3
"""Verify every vendored file listed by the RES-01 custody manifest."""

import hashlib
import json
import os
import posixpath
import stat
import sys
from pathlib import Path

CUSTODY_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = CUSTODY_DIR / "MANIFEST.json"
FILES_DIR = CUSTODY_DIR / "files"
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


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected, references = _manifest_entries(manifest)
        files_root = FILES_DIR.resolve(strict=True)
        files_root.relative_to(CUSTODY_DIR.resolve(strict=True))
        if FILES_DIR.is_symlink() or not files_root.is_dir():
            raise ValueError("custody files path must be a real directory")
        actual = _actual_files(files_root)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(str(exc))
        return 1

    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
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
            resolved = target.resolve(strict=True)
            resolved.relative_to(files_root)
            if target.is_symlink() or not resolved.is_file():
                raise ValueError("not a regular file inside the custody directory")

            digest = hashlib.sha256()
            size = 0
            with resolved.open("rb") as file:
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
        f"across {', '.join(FILE_COLLECTIONS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
