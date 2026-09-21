"""Build an immutable release in private storage during approved spare CPU windows."""

import argparse
import json
import shutil
import time
from pathlib import Path

import experiment_runtime as runtime
from experiment_scheduler import spare_lease
from seed_experiment import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=7200)
    args = parser.parse_args()
    output = runtime.output_root(args.root, args.output)
    source = runtime.inside(args.root, args.source)
    manifest = json.loads((source / "release-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        runtime.require(digest(source / name) == expected, f"release source changed: {name}")
    runtime.require(source.name == "hardening-v2-" + manifest["source_sha256"], "unversioned build source")
    complete = output / "build.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        runtime.require(
            receipt["source_sha256"] == manifest["source_sha256"]
            and digest(output / "bin/jope-parallel") == receipt["executable_sha256"],
            "completed build changed",
        )
        return
    cargo_home = output / "cargo-home"
    # Cargo can write only in this private cache; production's cache is read-only.
    registry = Path.home() / ".cargo/registry"
    cargo = Path.home() / ".cargo/bin/cargo"
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            with spare_lease(args.root, "hardening-build", cpus=2):
                if not (cargo_home / "registry").exists():
                    staging = cargo_home / "registry.staging"
                    shutil.copytree(
                        registry,
                        staging,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".cache-lock", "*.lock"),
                    )
                    staging.rename(cargo_home / "registry")
                env = runtime.child_env(
                    CARGO_HOME=str(cargo_home),
                    CARGO_TARGET_DIR=str(output / "target"),
                    CARGO_BUILD_JOBS="2",
                    RUSTUP_HOME=str(Path.home() / ".rustup"),
                    HOME=str(Path.home()),
                    PATH=str(Path.home() / ".cargo/bin") + ":/usr/bin:/bin",
                )
                logpath = output / f"build-{time.time_ns()}.log"
                command = [
                    str(cargo),
                    "build",
                    "--offline",
                    "--locked",
                    "--release",
                    "--bin",
                    "jope-parallel",
                    "-j",
                    "2",
                    "--manifest-path",
                    str(source / "Cargo.toml"),
                ]
                begun = time.monotonic()
                with logpath.open("w") as log:
                    result = runtime.run(
                        command,
                        root=args.root,
                        timeout=max(0.1, deadline - begun),
                        env=env,
                        stdout=log,
                        stderr=log,
                    )
                result.check_returncode()
                executable = output / "bin/jope-parallel"
                executable.parent.mkdir(exist_ok=True)
                with (
                    executable.open("xb") as target,
                    (output / "target/release/jope-parallel").open("rb") as built,
                ):
                    shutil.copyfileobj(built, target)
                executable.chmod(0o555)
                runtime.write(
                    complete,
                    {
                        "source_sha256": manifest["source_sha256"],
                        "executable_sha256": digest(executable),
                        "command": command,
                        "seconds": time.monotonic() - begun,
                        "target_directory": str(output / "target"),
                        "production_executables_replaced": False,
                    },
                )
                return
        except BlockingIOError as error:
            runtime.write(
                output / "progress.json",
                {"phase": "waiting", "reason": str(error), "timestamp": int(time.time())},
            )
            time.sleep(2)
    runtime.write(
        output / "progress.json",
        {"phase": "paused", "reason": "bounded build window expired", "timestamp": int(time.time())},
    )


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "build")
