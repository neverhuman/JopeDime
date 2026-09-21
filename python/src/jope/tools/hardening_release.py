"""Package an explicit immutable experiment source release; never replace active files."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiment_runtime import checked_path, require, write

TOOLS = (
    "experiment_runtime",
    "experiment_scheduler",
    "experiment_preflight",
    "experiment_canary",
    "hardening_release",
    "sync_hardening_release",
    "seed_experiment",
    "verify_experiment",
    "qualify_matched_v2",
    "test_qualify_matched_v2",
    "build_hardening_release",
    "tui_smoke",
    "propose_ranked",
    "proposal_pool_v2",
    "binary_prepare",
    "binary_encode",
    "binary_verify",
    "binary_serial",
    "binary_reproduce",
    "binary_replay_v2",
    "experiment_stages",
    "extend_component_sampling",
    "test_experiment_hardening",
    "test_seed_experiment",
    "test_binary_workflow",
    "test_binary_inventory",
    "test_extend_component_sampling",
    "qualify_pure_proposals",
    "test_qualify_pure_proposals",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(source):
    paths = [source / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    paths += [source / "tools" / (name + ".py") for name in TOOLS]
    for name in ("src", "tests", "examples", "native", "assets/source/jope-dime/vendor/feat-math"):
        paths += [
            p
            for p in (source / name).rglob("*")
            if p.is_file()
            and not any(part in ("target", "__pycache__", ".git") for part in p.relative_to(source).parts)
        ]
    return {str(checked_path(p).relative_to(source)): digest(p) for p in sorted(set(paths))}


def package(source, destination, evidence):
    files = inventory(source)
    source_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    name = "hardening-v2-" + source_hash
    release = destination / name
    require(not release.exists(), f"release already exists: {release}")
    staging = destination / ("." + name + ".staging")
    require(not staging.exists(), f"preserved incomplete staging requires inspection: {staging}")
    staging.mkdir(parents=True)
    for relative, expected in files.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        require(
            digest(target) == expected == digest(source / relative),
            f"source changed during packaging: {relative}",
        )
        target.chmod(0o444)
    receipt = {
        "schema": "parallel.hardening_release.v2",
        "source_sha256": source_hash,
        "baseline": "builds/v62-experiment",
        "files": files,
        "vendor_binding": {p: h for p, h in files.items() if p.startswith("assets/source/")},
        "verification": json.loads(evidence.read_text()),
        "production_services_changed": False,
        "larger_pool_selector_deployed": False,
    }
    write(staging / "release-manifest.json", receipt)
    (staging / "release-manifest.json").chmod(0o444)
    staging.rename(release)
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(package(args.source.resolve(), args.destination.resolve(), args.evidence.resolve()))
