"""Output inventories validated before completion publication and resume."""
import json
from pathlib import Path

from binary_verify import COMPONENTS, read, sha, vector
from experiment_runtime import inside, require


def python_environment(experiment):
    path = experiment / "python-deps.json"
    if not path.exists():
        return {}
    receipt = read(path)
    for name, expected in receipt["files"].items():
        require(sha(inside(experiment, name)) == expected, f"frozen Python dependency changed: {name}")
    return {"PYTHONPATH": str(experiment / "python-deps")}


def inventory(root, commands):
    paths = set()
    for command in commands:
        def arg(name, default=None):
            return command[command.index(name) + 1] if name in command else default
        output = inside(root, arg("--output"))
        if "--prepare-only" in command:
            paths.update(output / name for name in ("request.json", "target/sampling.json", "foundation/sampling.json"))
        elif "--qualify-cuda" in command:
            path = output / f"qualification-rank-{arg('--rank', '0')}.json"
            require(read(path)["passed"] is True, "sampling qualification failed")
            paths.add(path)
        elif "--merge-qualification" in command:
            paths.update(output / name for name in ("target-spec.json", "foundation-spec.json", "target/parity.json", "foundation/parity.json"))
        elif "apply-binary-contexts" in command:
            index_path = output / "index.json"
            index = read(index_path)
            jobs = read(inside(root, arg("--jobs")))["jobs"]
            require(len(index["encodings"]) == len(jobs) and len({(r["dataset"], r["cut"]) for r in index["encodings"]}) == len(jobs), "stage context inventory incomplete")
            expected = {(read(inside(root, j["prepared"]) / "prepared.json")["dataset"], j["cut"]): j for j in jobs}
            require(set(expected) == {(r["dataset"], r["cut"]) for r in index["encodings"]}, "stage contexts differ from admitted jobs")
            paths.update(output / name for name in ("index.json", "application.json", "reference-parity.json"))
            for row in index["encodings"]:
                path = inside(root, row["path"])
                vector(path, 2048)
                require(sha(path) == row["sha256"] and sha(path.with_suffix(".json")) == row["receipt_sha256"], "stage context changed")
                metadata = read(path.with_suffix(".json"))
                job = expected[row["dataset"], row["cut"]]
                require(metadata["prepared_sha256"] == job["prepared_sha256"] and metadata["cut"] == job["cut"] and metadata["dataset"] == row["dataset"], "stage context identity changed")
                for component, binding in metadata["components"].items():
                    require(component in COMPONENTS and sha(inside(root, binding["receipt"])) == binding["receipt_sha256"]
                            and sha(inside(root, binding["signature"])) == binding["signature_sha256"], "stage context component binding changed")
                require(set(metadata["components"]) == set(COMPONENTS), "stage context component missing")
                paths.update((path, path.with_suffix(".json")))
        else:
            component = next((name for name in COMPONENTS if f"{name}-rank" in command), "hyperion")
            filename, dimensions = COMPONENTS[component]
            jobs = read(inside(root, arg("--jobs")))["jobs"]
            rank, world = int(arg("--rank", "0")), int(arg("--world", "1"))
            for index, job in enumerate(jobs):
                if index % world != rank:
                    continue
                prepared = read(inside(root, job["prepared"]) / "prepared.json")
                directory = output / prepared["dataset"] / f"cut-{job['cut']}"
                path, receipt = directory / filename, directory / f"{component}.json"
                metadata = read(receipt)
                cut = next(c for c in prepared["cuts"] if c["index"] == job["cut"])
                require(metadata["prepared_sha256"] == job["prepared_sha256"] and metadata["cut"] == job["cut"]
                        and metadata["dataset"] == prepared["dataset"] and metadata["family"] == prepared["family"]
                        and metadata["training_sha256"] == cut["training_sha256"] and metadata["preprocessing_sha256"] == cut["preprocessing_sha256"]
                        and metadata["final_test_targets_used"] is False, "stage component identity changed")
                require(metadata["spec_sha256"] == sha(inside(root, arg("--spec"))), "stage encoder specification changed")
                vector(path, dimensions)
                require(metadata["shape"] == [dimensions] and metadata["signature_sha256"] == sha(path), "stage component changed")
                paths.update((path, receipt))
    require(bool(paths), "stage has no output artifacts")
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def verify(root, commands, receipt):
    require(receipt.get("complete") is True and isinstance(receipt.get("outputs"), dict), "stage lacks a verified output inventory; legacy stage cannot be skipped")
    require(inventory(root, commands) == receipt["outputs"], "completed stage output inventory changed")
