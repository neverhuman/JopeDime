#!/usr/bin/env python3
"""Resolve Cargo executables/reports without assuming the target-directory layout.

Build writes diagnostics to stderr and exactly one executable path to stdout.
Select consumes successful container build messages, translating a mounted root.
Cargo's compiler-artifact messages include cached (fresh=true) executables too:
https://doc.rust-lang.org/cargo/reference/external-tools.html#artifact-messages
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


def cargo_environment(target_dir=None):
    env = os.environ.copy()
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "line-tables-only")
    if target_dir is not None:
        env["CARGO_TARGET_DIR"] = target_dir
    return env


def metadata(manifest, env):
    return json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1",
         "--manifest-path", manifest], env=env,
    ))


def nextest_config(source, output, target):
    """Nextest's default store is workspace-relative, independently of Cargo."""
    text = Path(source).read_text()
    if "store" in tomllib.loads(text):
        raise ValueError("nextest already configures a store; reconcile it before relocating reports")
    # Keep every profile/filter byte intact. TOML basic strings accept JSON's
    # escaped string form, including spaces in an absolute target directory.
    with Path(output).open("x") as stream:
        stream.write("store = { dir = " + json.dumps(str(Path(target) / "nextest")) + " }\n" + text)


def executable(messages, meta, package, kind, name, path_map=None):
    packages = [p["id"] for p in meta["packages"]
                if p["name"] == package and p["id"] in meta["workspace_members"]]
    if len(packages) != 1:
        raise ValueError(f"expected one workspace package {package!r}, found {len(packages)}")
    matches = []
    finished = False
    for line in messages.splitlines():
        if not line.startswith("{"):
            continue  # Cargo documents possible non-JSON output from other tools.
        message = json.loads(line)
        if message.get("reason") == "build-finished":
            finished = message.get("success") is True
        if (message.get("reason") == "compiler-artifact"
                and message.get("package_id") == packages[0]
                and message.get("target", {}).get("name") == name
                and kind in message.get("target", {}).get("kind", [])
                and not message.get("profile", {}).get("test", False)
                and message.get("executable")):
            matches.append(Path(message["executable"]))
    if not finished:
        raise ValueError("Cargo did not report a successful completed build")
    if len(matches) != 1:
        raise ValueError(f"expected one executable for {package}/{kind}/{name}, found {len(matches)}")
    path = matches[0]
    if path_map:
        path = Path(path_map[1]) / path.relative_to(path_map[0])
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Cargo executable is missing or not executable: {path}")
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    select = commands.add_parser("select", help="read a successful container build")
    directory = commands.add_parser("target-dir")
    config = commands.add_parser("nextest-config", help="bind nextest's store to Cargo's effective target")
    config.add_argument("--source", default=".config/nextest.toml")
    config.add_argument("--output", required=True)
    for p in (build, directory, config):
        p.add_argument("--manifest-path", default="Cargo.toml")
        p.add_argument("--target-dir")
    for p in (build, select):
        p.add_argument("--package", required=True)
        target = p.add_mutually_exclusive_group(required=True)
        target.add_argument("--bin")
        target.add_argument("--example")
    build.add_argument("--profile", default="dev")
    build.add_argument("--features", action="append", default=[])
    build.add_argument("--no-default-features", action="store_true")
    select.add_argument("--metadata", required=True)
    select.add_argument("--messages", required=True)
    select.add_argument("--path-map", nargs=2, metavar=("CONTAINER_ROOT", "HOST_ROOT"))
    args = parser.parse_args()
    try:
        if args.command == "select":
            meta = json.loads(Path(args.metadata).read_text())
            messages = Path(args.messages).read_text()
        else:
            env = cargo_environment(args.target_dir)
            meta = metadata(args.manifest_path, env)
            if args.command == "target-dir":
                print(meta["target_directory"])
                return 0
            if args.command == "nextest-config":
                nextest_config(args.source, args.output, meta["target_directory"])
                return 0
            command = ["cargo", "build", "--locked", "--message-format=json-render-diagnostics",
                       "--manifest-path", args.manifest_path, "--package", args.package,
                       "--profile", args.profile,
                       "--bin" if args.bin else "--example", args.bin or args.example]
            for features in args.features:
                command.extend(["--features", features])
            if args.no_default_features:
                command.append("--no-default-features")
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE, text=True)
            # Never return an artifact from a build that later failed.
            if result.returncode:
                return result.returncode if result.returncode > 0 else 128 - result.returncode
            messages = result.stdout
        print(executable(messages, meta, args.package,
                         "bin" if args.bin else "example", args.bin or args.example,
                         getattr(args, "path_map", None)))
        return 0
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (OSError, ValueError, KeyError) as exc:
        print(f"cargo-artifact: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
