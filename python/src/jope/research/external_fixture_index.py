#!/usr/bin/env python3
"""Build, verify, and materialize JopeExternalFixtureIndex.v1.

Materialization reads Git objects from the pinned commits, never mutable
worktree bytes. Remote binary splits are optional unless `--profile all` is
selected and are accepted only after size and SHA-256 verification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "manifests" / "epoch6" / "JopeExternalFixtureIndex.v1.json"
LEGACY_REPO = ROOT / "sotarena"
MODERN_REPO = Path("/home/ubuntu/sotarena")
REPOSITORY_PATHS = {
    "legacy-sotarena": LEGACY_REPO,
    "sotarena-rust": MODERN_REPO,
}
SCHEMA = "JopeExternalFixtureIndex.v1"
RECEIPT_SCHEMA = "JopeExternalFixtureMaterialization.v1"
RECOVERY_IMPORT_SCHEMA = "JopeExternalFixtureRecoveryImport.v1"
DEFAULT_RECOVERY_ROOT = Path("/home/ubuntu/sotarena_old")
DEFAULT_FIXTURE_STORE = Path(
    "/home/ubuntu/jope-dime-snapshots/external-fixtures-v1"
)
GOLDEN_REPORTS = (
    "tabpfn_v3_regressor.json",
    "tabfm_regressor.json",
    "tabfm_ensemble_regressor.json",
)


class FixtureError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def run(*arguments: str, cwd: Path | None = None) -> bytes:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FixtureError(f"command failed: {' '.join(arguments)}: {exc}") from exc


def git_text(repo: Path, *arguments: str) -> str:
    return run("git", *arguments, cwd=repo).decode("utf-8").strip()


def repository(repo_id: str, path: Path) -> dict:
    require((path / ".git").exists(), f"fixture repository is absent: {path}")
    return {
        "id": repo_id,
        "commit": git_text(path, "rev-parse", "HEAD"),
        "tree": git_text(path, "show", "-s", "--format=%T", "HEAD"),
    }


def repository_path(repo: dict) -> Path:
    try:
        return REPOSITORY_PATHS[repo["id"]]
    except (KeyError, TypeError) as exc:
        raise FixtureError("fixture repository has no local observation root") from exc


def git_entry(repo: dict, source_path: str, target_path: str, group: str) -> dict:
    pure = PurePosixPath(source_path)
    require(
        not pure.is_absolute() and ".." not in pure.parts,
        f"unsafe source path {source_path}",
    )
    line = git_text(
        repository_path(repo),
        "ls-tree",
        "--long",
        repo["commit"],
        "--",
        source_path,
    )
    parts = line.split(None, 4)
    require(
        len(parts) == 5 and parts[1] == "blob" and parts[4] == source_path,
        f"missing regular Git fixture {source_path}",
    )
    mode, _, blob_sha1, size, _ = parts
    require(mode == "100644", f"fixture {source_path} has unsupported Git mode {mode}")
    data = run(
        "git",
        "cat-file",
        "blob",
        f"{repo['commit']}:{source_path}",
        cwd=repository_path(repo),
    )
    require(len(data) == int(size), f"Git fixture size drift for {source_path}")
    return {
        "group": group,
        "target_path": target_path,
        "source": {
            "kind": "git_blob",
            "repository": repo["id"],
            "path": source_path,
            "blob_sha1": blob_sha1,
        },
        "file_type": "regular",
        "size_bytes": len(data),
        "mode": 0o444,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def remote_entry(
    modern: dict,
    metadata_path: str,
    dataset: str,
    split: str,
    metadata: dict,
) -> dict:
    source = metadata[split]
    return {
        "group": "binary42",
        "target_path": f"data/tabular-binary-v1/{dataset}/{split}.csv",
        "source": {
            "kind": "verified_url",
            "repository": modern["id"],
            "metadata_path": metadata_path,
            "url": source["url"],
        },
        "file_type": "regular",
        "size_bytes": source["bytes"],
        "mode": 0o444,
        "sha256": source["sha256"],
    }


def unresolved_split_entry(
    modern: dict,
    metadata_path: str,
    dataset: str,
    split: str,
    expected_sha256: str,
) -> dict:
    return {
        "group": "binary42",
        "target_path": f"data/tabular-binary-v1/{dataset}/{split}.csv",
        "source": {
            "kind": "unresolved_external",
            "repository": modern["id"],
            "metadata_path": metadata_path,
            "reason": "pinned metadata has zero bytes and no retrieval URL; source bytes are absent",
        },
        "file_type": "regular",
        "size_bytes": None,
        "mode": 0o444,
        "sha256": expected_sha256,
    }


def committed_json(repo: dict, path: str) -> dict:
    data = run(
        "git",
        "cat-file",
        "blob",
        f"{repo['commit']}:{path}",
        cwd=repository_path(repo),
    )
    value = json.loads(data)
    require(isinstance(value, dict), f"fixture metadata is not an object: {path}")
    return value


def find_committed_sha256(repo: dict, directory: str, expected: str) -> str:
    paths = git_text(
        repository_path(repo),
        "ls-tree",
        "-r",
        "--name-only",
        repo["commit"],
        "--",
        directory,
    ).splitlines()
    for path in paths:
        data = run(
            "git",
            "cat-file",
            "blob",
            f"{repo['commit']}:{path}",
            cwd=repository_path(repo),
        )
        if hashlib.sha256(data).hexdigest() == expected:
            return path
    raise FixtureError(f"no committed object under {directory} has SHA-256 {expected}")


def build_index() -> dict:
    legacy = repository("legacy-sotarena", LEGACY_REPO)
    modern = repository("sotarena-rust", MODERN_REPO)
    entries = [
        git_entry(legacy, "sotarena/elo.py", "sotarena/elo.py", "modules"),
        git_entry(
            legacy,
            "private/jope-dime/src/aspfm_generation/island_extension.rs",
            "private/jope-dime/src/aspfm_generation/island_extension.rs",
            "modules",
        ),
    ]

    golden = [
        line.strip()
        for line in (ROOT / "manifests/panel.txt").read_text().splitlines()
        if line.strip()
    ]
    require(
        len(golden) == 42 and golden == sorted(golden),
        "Golden42 panel is not exact and sorted",
    )
    for dataset in golden:
        metadata_path = f"splits/tabular-regression-v1/{dataset}/metadata.json"
        entries.append(git_entry(legacy, metadata_path, metadata_path, "train_only"))
        for split in ("train", "test"):
            path = f"data/tabular-regression-v1/{dataset}/{split}.csv"
            entries.append(git_entry(legacy, path, path, "train_only"))
        for report in GOLDEN_REPORTS:
            path = f"results/external-references/golden42/{dataset}/{report}"
            entries.append(git_entry(legacy, path, path, "train_only"))

    binary_manifest = json.loads(
        (ROOT / "manifests/binary42-test-adaptive.v1.json").read_text()
    )
    binary = binary_manifest.get("datasets")
    require(
        isinstance(binary, list) and len(binary) == 42, "Binary42 manifest is not exact"
    )
    for row in binary:
        dataset = row["id"]
        metadata_path = f"binary/{dataset}/dataset.json"
        metadata = committed_json(modern, metadata_path)
        require(
            metadata.get("dataset_hash") == dataset,
            f"binary metadata identity drift for {dataset}",
        )
        entries.append(
            git_entry(
                modern,
                metadata_path,
                f"registry/{metadata_path}",
                "binary42",
            )
        )
        for split in ("train", "test"):
            require(
                metadata[split]["sha256"] == row[f"{split}_sha256"],
                f"binary {dataset} {split} digest drift",
            )
            source = metadata[split]
            if (
                isinstance(source.get("bytes"), int)
                and source["bytes"] > 0
                and isinstance(source.get("url"), str)
            ):
                entries.append(
                    remote_entry(modern, metadata_path, dataset, split, metadata)
                )
            else:
                entries.append(
                    unresolved_split_entry(
                        modern,
                        metadata_path,
                        dataset,
                        split,
                        row[f"{split}_sha256"],
                    )
                )
        leader_source = find_committed_sha256(
            modern,
            f"binary/{dataset}/results",
            row["leader_report_sha256"],
        )
        target = f"results/external-references/binary42/{dataset}/{Path(row['leader_report_path']).name}"
        entries.append(git_entry(modern, leader_source, target, "binary42"))

    entries.sort(key=lambda row: row["target_path"])
    require(
        len(entries) == len({row["target_path"] for row in entries}),
        "fixture target paths overlap",
    )
    return {
        "schema": SCHEMA,
        "repositories": [legacy, modern],
        "profiles": {
            "workspace_tests": ["modules", "train_only"],
            "all": ["binary42", "modules", "train_only"],
        },
        "group_counts": {
            group: sum(row["group"] == group for row in entries)
            for group in ("binary42", "modules", "train_only")
        },
        "unresolved_count": sum(
            row["source"]["kind"] == "unresolved_external" for row in entries
        ),
        "entries": entries,
    }


def canonical_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_file(path: Path, data: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def stable_stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_double_read(path: Path) -> tuple[bytes, os.stat_result, str]:
    first_stat = path.stat(follow_symlinks=False)
    require(
        stat.S_ISREG(first_stat.st_mode) and not path.is_symlink(),
        f"recovery source is not a regular non-symlink: {path}",
    )

    def read_once() -> bytes:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(descriptor)

    first_bytes = read_once()
    middle_stat = path.stat(follow_symlinks=False)
    second_bytes = read_once()
    final_stat = path.stat(follow_symlinks=False)
    signatures = {
        stable_stat_signature(first_stat),
        stable_stat_signature(middle_stat),
        stable_stat_signature(final_stat),
    }
    first_digest = hashlib.sha256(first_bytes).hexdigest()
    second_digest = hashlib.sha256(second_bytes).hexdigest()
    require(
        len(signatures) == 1
        and first_bytes == second_bytes
        and first_digest == second_digest,
        f"recovery source changed during stable double read: {path}",
    )
    return first_bytes, final_stat, first_digest


def frozen_binary_digests() -> dict[str, dict[str, str]]:
    value = json.loads((ROOT / "manifests/binary42-test-adaptive.v1.json").read_text())
    return {
        row["id"]: {
            "train": row["train_sha256"],
            "test": row["test_sha256"],
        }
        for row in value["datasets"]
    }


def binary_target_identity(target_path: str) -> tuple[str, str]:
    parts = PurePosixPath(target_path).parts
    require(
        len(parts) == 4
        and parts[:2] == ("data", "tabular-binary-v1")
        and parts[3] in {"train.csv", "test.csv"},
        f"unsupported recovery target {target_path}",
    )
    return parts[2], PurePosixPath(parts[3]).stem


def import_local_recovery(
    value: dict,
    source_root: Path,
    fixture_store: Path,
    import_manifest_path: Path,
    recovered_index_path: Path,
) -> tuple[dict, dict]:
    """Qualify dirty local observations and publish immutable CAS objects.

    The caller chooses every output path. Production publication is therefore
    separable from qualification in a temporary root.
    """

    validate_index(value)
    require(source_root.is_absolute(), "recovery source root must be absolute")
    require(fixture_store.is_absolute(), "fixture store must be absolute")
    require(
        import_manifest_path.is_absolute() and recovered_index_path.is_absolute(),
        "recovery output paths must be absolute",
    )
    unresolved = [
        row for row in value["entries"] if row["source"]["kind"] == "unresolved_external"
    ]
    require(len(unresolved) == 46, "qualified recovery requires the frozen 46-file set")
    modern = next(row for row in value["repositories"] if row["id"] == "sotarena-rust")
    observed_head = git_text(source_root, "rev-parse", "HEAD")
    observed_tree = git_text(source_root, "show", "-s", "--format=%T", "HEAD")
    require(
        re.fullmatch(r"[0-9a-f]{40}", observed_head) is not None
        and re.fullmatch(r"[0-9a-f]{40}", observed_tree) is not None,
        "recovery observation Git identity is invalid",
    )
    manifest_digests = frozen_binary_digests()
    objects = []
    for row in unresolved:
        source_path = source_root / row["target_path"]
        data, observed, digest = stable_double_read(source_path)
        dataset, split = binary_target_identity(row["target_path"])
        metadata = committed_json(modern, row["source"]["metadata_path"])
        require(
            digest == row["sha256"]
            and digest == metadata[split]["sha256"]
            and digest == manifest_digests[dataset][split],
            f"recovery digest has no two-authority match for {row['target_path']}",
        )
        require(
            observed.st_size == len(data) and len(data) > 0,
            f"recovery size is invalid for {row['target_path']}",
        )
        relative_object = PurePosixPath("objects", "sha256", digest[:2], digest)
        object_path = fixture_store / relative_object
        exclusive_file(object_path, data)
        published = object_path.stat(follow_symlinks=False)
        require(
            stat.S_ISREG(published.st_mode)
            and not object_path.is_symlink()
            and stat.S_IMODE(published.st_mode) == 0o444
            and published.st_size == len(data)
            and hashlib.sha256(object_path.read_bytes()).hexdigest() == digest,
            f"published recovery object failed re-verification: {object_path}",
        )
        objects.append(
            {
                "target_path": row["target_path"],
                "cas_object": str(relative_object),
                "file_type": "regular",
                "size_bytes": len(data),
                "target_mode": 0o444,
                "sha256": digest,
                "source_stat": {
                    "device": observed.st_dev,
                    "inode": observed.st_ino,
                    "size_bytes": observed.st_size,
                    "mode": stat.S_IMODE(observed.st_mode),
                    "mtime_ns": observed.st_mtime_ns,
                    "ctime_ns": observed.st_ctime_ns,
                },
            }
        )
    objects.sort(key=lambda row: row["target_path"])
    import_manifest = {
        "schema": RECOVERY_IMPORT_SCHEMA,
        "metadata_authority": {
            "repository": modern["id"],
            "commit": modern["commit"],
            "tree": modern["tree"],
        },
        "recovery_observation": {
            "head": observed_head,
            "tree": observed_tree,
            "dirty_observation_not_authority": True,
        },
        "fixture_store": "external-fixtures-v1",
        "object_count": len(objects),
        "objects_sha256": hashlib.sha256(canonical_bytes(objects)).hexdigest(),
        "objects": objects,
    }
    import_bytes = canonical_bytes(import_manifest)
    import_digest = hashlib.sha256(import_bytes).hexdigest()
    exclusive_file(import_manifest_path, import_bytes)

    recovered = copy.deepcopy(value)
    by_target = {row["target_path"]: row for row in objects}
    for row in recovered["entries"]:
        if row["target_path"] not in by_target:
            continue
        qualified = by_target[row["target_path"]]
        row["size_bytes"] = qualified["size_bytes"]
        row["source"] = {
            "kind": "verified_local_recovery",
            "repository": modern["id"],
            "metadata_path": next(
                original["source"]["metadata_path"]
                for original in unresolved
                if original["target_path"] == row["target_path"]
            ),
            "fixture_store": "external-fixtures-v1",
            "cas_object": qualified["cas_object"],
            "import_manifest_sha256": import_digest,
            "dirty_observation_not_authority": True,
        }
    recovered["unresolved_count"] = 0
    validate_index(recovered, require_complete=True)
    exclusive_file(recovered_index_path, canonical_bytes(recovered))
    return import_manifest, recovered


def validate_index(value: dict, require_complete: bool = False) -> None:
    require(value.get("schema") == SCHEMA, "invalid external-fixture index schema")
    repositories = value.get("repositories")
    require(
        isinstance(repositories, list) and len(repositories) == 2,
        "fixture repositories are incomplete",
    )
    by_id = {row.get("id"): row for row in repositories if isinstance(row, dict)}
    require(
        set(by_id) == {"legacy-sotarena", "sotarena-rust"},
        "fixture repository IDs drift",
    )
    for repo in repositories:
        require(
            re.fullmatch(r"[0-9a-f]{40}", str(repo.get("commit"))) is not None,
            "fixture repository commit is invalid",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", str(repo.get("tree"))) is not None,
            "fixture repository tree is invalid",
        )
        observed = repository_path(repo)
        require(
            git_text(observed, "rev-parse", repo["commit"]) == repo["commit"],
            f"fixture commit unavailable for {repo['id']}",
        )
        require(
            git_text(observed, "show", "-s", "--format=%T", repo["commit"])
            == repo["tree"],
            f"fixture tree drift for {repo['id']}",
        )

    entries = value.get("entries")
    require(isinstance(entries, list) and entries, "fixture entries are absent")
    require(
        [row["target_path"] for row in entries]
        == sorted(row["target_path"] for row in entries),
        "fixture entries are noncanonical",
    )
    require(
        len(entries) == len({row["target_path"] for row in entries}),
        "fixture target paths overlap",
    )
    counts = value.get("group_counts")
    require(
        counts
        == {
            group: sum(row["group"] == group for row in entries)
            for group in ("binary42", "modules", "train_only")
        },
        "fixture group counts drift",
    )
    unresolved_count = sum(
        row["source"].get("kind") == "unresolved_external" for row in entries
    )
    require(
        value.get("unresolved_count") == unresolved_count,
        "fixture unresolved count drift",
    )
    for row in entries:
        target = PurePosixPath(row["target_path"])
        require(
            not target.is_absolute() and ".." not in target.parts,
            f"unsafe fixture target {target}",
        )
        require(
            row.get("file_type") == "regular" and row.get("mode") == 0o444,
            f"fixture type/mode drift for {target}",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is not None,
            f"fixture digest invalid for {target}",
        )
        source = row.get("source")
        require(
            isinstance(source, dict) and source.get("repository") in by_id,
            f"fixture source invalid for {target}",
        )
        if source.get("kind") == "git_blob":
            require(
                isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0,
                f"fixture size invalid for {target}",
            )
            repo = by_id[source["repository"]]
            actual = git_entry(repo, source["path"], row["target_path"], row["group"])
            require(actual == row, f"committed fixture drift for {target}")
        elif source.get("kind") == "verified_url":
            require(
                isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0,
                f"fixture size invalid for {target}",
            )
            require(
                source.get("url", "").startswith("https://"),
                f"fixture URL is not HTTPS for {target}",
            )
            metadata = committed_json(
                by_id[source["repository"]], source["metadata_path"]
            )
            split = target.stem
            require(
                metadata[split]["sha256"] == row["sha256"]
                and metadata[split]["bytes"] == row["size_bytes"],
                f"remote fixture metadata drift for {target}",
            )
        elif source.get("kind") == "unresolved_external":
            require(
                row.get("size_bytes") is None,
                f"unresolved fixture unexpectedly declares a size for {target}",
            )
            metadata = committed_json(
                by_id[source["repository"]], source["metadata_path"]
            )
            split = target.stem
            require(
                metadata[split]["sha256"] == row["sha256"],
                f"unresolved fixture digest drift for {target}",
            )
            require(
                metadata[split].get("bytes") == 0
                and metadata[split].get("url") is None,
                f"unresolved fixture is now retrievable: {target}",
            )
        elif source.get("kind") == "verified_local_recovery":
            require(
                isinstance(row.get("size_bytes"), int) and row["size_bytes"] > 0,
                f"recovered fixture size invalid for {target}",
            )
            require(
                source.get("fixture_store") == "external-fixtures-v1"
                and source.get("dirty_observation_not_authority") is True
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(source.get("import_manifest_sha256"))
                )
                is not None,
                f"recovered fixture authority invalid for {target}",
            )
            expected_object = f"objects/sha256/{row['sha256'][:2]}/{row['sha256']}"
            require(
                source.get("cas_object") == expected_object,
                f"recovered fixture CAS identity drift for {target}",
            )
            metadata = committed_json(
                by_id[source["repository"]], source["metadata_path"]
            )
            dataset, split = binary_target_identity(row["target_path"])
            require(
                metadata[split]["sha256"] == row["sha256"]
                and frozen_binary_digests()[dataset][split] == row["sha256"],
                f"recovered fixture digest authority drift for {target}",
            )
        else:
            raise FixtureError(f"unsupported fixture source for {target}")
    require(
        not require_complete or unresolved_count == 0,
        f"{unresolved_count} external fixture files remain unresolved",
    )


def source_bytes(
    row: dict,
    repositories: dict[str, dict],
    fixture_store: Path,
) -> bytes:
    source = row["source"]
    if source["kind"] == "git_blob":
        repo = repositories[source["repository"]]
        return run(
            "git",
            "cat-file",
            "blob",
            f"{repo['commit']}:{source['path']}",
            cwd=repository_path(repo),
        )
    if source["kind"] == "verified_url":
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            return response.read(row["size_bytes"] + 1)
    if source["kind"] == "verified_local_recovery":
        object_path = fixture_store / source["cas_object"]
        observed = object_path.stat(follow_symlinks=False)
        require(
            stat.S_ISREG(observed.st_mode)
            and not object_path.is_symlink()
            and stat.S_IMODE(observed.st_mode) == 0o444,
            f"recovery CAS object type/mode drift for {row['target_path']}",
        )
        return object_path.read_bytes()
    raise FixtureError(f"cannot materialize unresolved fixture {row['target_path']}")


def materialize(
    value: dict,
    destination: Path,
    profile: str,
    fixture_store: Path = DEFAULT_FIXTURE_STORE,
) -> dict:
    validate_index(value, require_complete=profile == "all")
    require(destination.is_absolute(), "materialization root must be absolute")
    require(not destination.exists(), "materialization root already exists")
    groups = value["profiles"].get(profile)
    require(isinstance(groups, list), f"unknown fixture profile {profile}")
    repositories = {row["id"]: row for row in value["repositories"]}
    destination.mkdir(mode=0o700)
    materialized = []
    try:
        for row in value["entries"]:
            if row["group"] not in groups:
                continue
            target = destination / row["target_path"]
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = source_bytes(row, repositories, fixture_store)
            require(
                len(data) == row["size_bytes"],
                f"materialized size drift for {row['target_path']}",
            )
            require(
                hashlib.sha256(data).hexdigest() == row["sha256"],
                f"materialized digest drift for {row['target_path']}",
            )
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(target, 0o444)
            fsync_directory(target.parent)
            materialized.append(
                {
                    "target_path": row["target_path"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
            )
        for directory, subdirs, _ in os.walk(destination, topdown=False):
            for name in subdirs:
                os.chmod(Path(directory) / name, 0o555)
        os.chmod(destination, 0o555)
        fsync_directory(destination)
    except Exception:
        # Preserve partial bytes for diagnosis; the caller receives no receipt.
        raise
    return {
        "schema": RECEIPT_SCHEMA,
        "index_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
        "profile": profile,
        "file_count": len(materialized),
        "object_sha256": sorted(row["sha256"] for row in materialized),
        "materialized_root_sha256": hashlib.sha256(
            canonical_bytes(sorted(materialized, key=lambda row: row["target_path"]))
        ).hexdigest(),
        "read_only": True,
    }


def read_index() -> dict:
    try:
        value = json.loads(INDEX.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot read external fixture index: {exc}") from exc
    require(isinstance(value, dict), "external fixture index is not an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=INDEX)
    subparsers.add_parser("verify")
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--destination", type=Path, required=True)
    materialize_parser.add_argument(
        "--profile", choices=("workspace_tests", "all"), default="workspace_tests"
    )
    materialize_parser.add_argument(
        "--fixture-store", type=Path, default=DEFAULT_FIXTURE_STORE
    )
    recover = subparsers.add_parser("recover-local")
    recover.add_argument("--source-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    recover.add_argument("--fixture-store", type=Path, required=True)
    recover.add_argument("--import-manifest", type=Path, required=True)
    recover.add_argument("--index-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build":
            value = build_index()
            validate_index(value)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(value))
                handle.flush()
                os.fsync(handle.fileno())
            print(f"published {args.output} with {len(value['entries'])} entries")
        elif args.command == "verify":
            value = read_index()
            validate_index(value, require_complete=True)
            print(f"external fixture index verified: {len(value['entries'])} entries")
        elif args.command == "materialize":
            receipt = materialize(
                read_index(), args.destination, args.profile, args.fixture_store
            )
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        else:
            import_manifest, recovered = import_local_recovery(
                read_index(),
                args.source_root,
                args.fixture_store,
                args.import_manifest,
                args.index_output,
            )
            receipt = {
                "schema": RECOVERY_IMPORT_SCHEMA,
                "import_manifest_sha256": hashlib.sha256(
                    canonical_bytes(import_manifest)
                ).hexdigest(),
                "index_sha256": hashlib.sha256(canonical_bytes(recovered)).hexdigest(),
                "object_count": import_manifest["object_count"],
                "unresolved_count": recovered["unresolved_count"],
            }
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    except FixtureError as exc:
        print(f"external fixture index failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
