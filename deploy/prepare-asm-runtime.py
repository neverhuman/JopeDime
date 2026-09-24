#!/usr/bin/env python3
"""Build pinned workers and prepare metadata; numerical execution is always separate."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HOST_LOCK = Path("/run/lock/jope-exclusive-cpu.lock")
TARGETS = ("jope", "jope-context-worker", "jope-policy-producer", "jope-context-proof")
FEATURES = "jope-cli/asm,jope-cli/qualified-training-context"
PROTOCOL = {
    "seconds": 120, "max_evaluations": 96, "sampling_rounds": 4,
    "sampling_batch": 64, "sampling_temperature": 1.0, "dataset_wave_width": 1,
    "scheduling_policy": "asm.shared-pool/dataset-wave-rendezvous-round-robin/quantum20/v1",
    "inner_fold_policy": "asm.inner-groups/largest-first-lexicographic-balance-five/v1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file(), "artifact must be physical")
    require(not path.is_relative_to("/tmp"), "artifact must be durable, outside /tmp")
    return {"path": str(path), "sha256": digest(path)}


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "artifact reference required")
    require(artifact(ref["path"]) == ref, "artifact bytes changed")
    return Path(ref["path"])


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fresh_output(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), "output already exists")
    require(path.parent.resolve(strict=True) == path.parent, "output parent must be physical")
    require(not path.is_relative_to("/tmp"), "output must be durable, outside /tmp")
    require(not any((p / ".git").exists() for p in (path, *path.parents)), "output must be outside a checkout")
    path.mkdir(mode=0o700)
    return path


@contextlib.contextmanager
def allocation():
    # Provisioning owns this persistent inode. Never create or replace it.
    with HOST_LOCK.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("exclusive CPU allocation is busy") from error
        yield


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_guard(root, expected):
    require(re.fullmatch(r"[0-9a-f]{40}", expected), "full source SHA required")
    require(git(root, "rev-parse", "HEAD") == expected, "source HEAD changed")
    dirty = git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    require(all(line.strip() == "M AGENT_CHAT.md" for line in dirty), "source must be committed and clean")


def build_workers(args):
    with allocation():
        root = args.source_root.resolve(strict=True)
        source_guard(root, args.expect_source)
        require(1 <= args.jobs <= 4, "build jobs must be between 1 and 4")
        require(not os.environ.get("CARGO_TARGET_DIR"), "use the checkout's existing target")
        out = fresh_output(args.output)
        files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        source = out / "source-manifest.json"
        # The append-only coordination ledger is not a compiled source input.
        write(source, {"schema": "jope.private-source-manifest/v1", "source_sha40": args.expect_source,
                       "tree_sha40": git(root, "rev-parse", "HEAD^{tree}"),
                       "files": {f: digest(root / f) for f in files if f and f != "AGENT_CHAT.md"}})
        env = os.environ.copy()
        temporary = root / "target" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        env.update(CARGO_INCREMENTAL="0", CARGO_PROFILE_DEV_DEBUG="line-tables-only",
                   CARGO_BUILD_JOBS=str(args.jobs), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", TMPDIR=str(temporary))
        command = ["cargo", "build", "--locked", "--release", "-p", "jope-cli", "-p", "jope-policy",
                   "-p", "jope-pipeline", "--features", FEATURES]
        for target in TARGETS:
            command += ["--bin", target]
        command += ["--message-format=json-render-diagnostics"]
        stdout, stderr = out / "cargo.stdout.jsonl", out / "cargo.stderr.log"
        started = time.time_ns() // 1_000_000
        with stdout.open("xb") as a, stderr.open("xb") as b:
            result = subprocess.run(command, cwd=root, env=env, stdout=a, stderr=b)
        receipt = {"schema": "jope.asm-cargo-build/v1", "source_sha40": args.expect_source,
                   "source_manifest_sha256": digest(source), "command": command,
                   "working_directory": str(root), "started_unix_ms": started,
                   "finished_unix_ms": time.time_ns() // 1_000_000, "exit_code": result.returncode,
                   "cargo_messages": artifact(stdout), "executables": []}
        try:
            require(result.returncode == 0, "Cargo build failed; retained cargo.stderr.log")
            source_guard(root, args.expect_source)
            require(all(digest(root / f) == h for f, h in read(source)["files"].items()), "source bytes changed during build")
            messages = [json.loads(line) for line in stdout.read_text().splitlines()]
            require(any(m.get("reason") == "build-finished" and m.get("success") is True for m in messages), "Cargo completion absent")
            (out / "bin").mkdir(mode=0o700)
            copies, features = {}, {}
            for name in TARGETS:
                matches = [m for m in messages if m.get("reason") == "compiler-artifact"
                           and m.get("target", {}).get("name") == name and m.get("executable")]
                require(len(matches) == 1, "missing or duplicate Cargo executable: " + name)
                message = matches[0]
                original = artifact(message["executable"])
                destination = out / "bin" / name
                shutil.copyfile(original["path"], destination)
                destination.chmod(0o700)
                copies[name] = artifact(destination)
                require(copies[name]["sha256"] == original["sha256"], "executable copy changed")
                features[name] = message["features"]
                receipt["executables"].append({"target_name": name, **original})
            for name in ("jope-context-worker", "jope-policy-producer"):
                require("context-producer" in features[name], "qualified context feature absent")
        except Exception as error:
            write(out / "failed-build.json", {"build": receipt, "error": str(error)})
            raise
        write(out / "build.json", receipt)
        inventory = {"schema": "jope.asm-worker-build-inventory/v1", "source_sha40": args.expect_source,
                     "source_manifest": artifact(source), "workers": [
                         {"role": role, "executable": copies[name], "features": features[name],
                          "build_receipt": artifact(out / "build.json")}
                         for role, name in (("policy", "jope-policy-producer"), ("context", "jope-context-worker"))]}
        write(out / "worker-build-inventory.json", inventory)
        # Explicit conversion for prepare-panel-runtime.py's distinct CLI receipt interface.
        write(out / "cli-build.json", {"schema": "jope.asm-cli-build-conversion/v1", "ok": True,
                                       "source_sha": args.expect_source, "binary_sha256": copies["jope"]["sha256"],
                                       "compiler_evidence": artifact(out / "build.json")})
        summary = {"source_sha40": args.expect_source, "source_manifest": artifact(source),
                   "worker_inventory": artifact(out / "worker-build-inventory.json"),
                   "build_receipt": artifact(out / "build.json"), "executables": copies,
                   "cli_build_receipt": artifact(out / "cli-build.json"),
                   "canonicalizer": {"program": copies["jope"], "source_sha40": args.expect_source},
                   "fits": 0, "execution_budget_started": False}
        write(out / "summary.json", summary)
        return summary


def worker_build(path):
    summary = read(path)
    require(re.fullmatch(r"[0-9a-f]{40}", summary["source_sha40"]), "full worker source required")
    source = read(checked(summary["source_manifest"]))
    inventory = read(checked(summary["worker_inventory"]))
    receipt = read(checked(summary["build_receipt"]))
    require(source["source_sha40"] == inventory["source_sha40"] == receipt["source_sha40"] == summary["source_sha40"], "worker source differs")
    require(inventory["schema"] == "jope.asm-worker-build-inventory/v1" and receipt["schema"] == "jope.asm-cargo-build/v1", "unsupported worker build schema")
    require(inventory["source_manifest"] == summary["source_manifest"] and receipt["source_manifest_sha256"] == summary["source_manifest"]["sha256"], "source manifest differs")
    require(receipt["exit_code"] == 0 and receipt["finished_unix_ms"] >= receipt["started_unix_ms"] > 0, "unsuccessful worker build")
    messages = [json.loads(line) for line in checked(receipt["cargo_messages"]).read_text().splitlines()]
    require(any(m.get("reason") == "build-finished" and m.get("success") is True for m in messages), "Cargo completion absent")
    require(len(inventory["workers"]) == 2 and {w["role"] for w in inventory["workers"]} == {"context", "policy"}, "worker roles differ")
    require(set(summary["executables"]) == set(TARGETS), "executable inventory differs")
    for name, ref in summary["executables"].items():
        checked(ref)
        built = [x for x in receipt["executables"] if x["target_name"] == name]
        matches = [m for m in messages if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == name and m.get("executable")]
        require(len(built) == len(matches) == 1 and built[0]["sha256"] == ref["sha256"] and built[0]["path"] == matches[0]["executable"], "executable differs from Cargo evidence")
        for worker in inventory["workers"]:
            if name == {"policy": "jope-policy-producer", "context": "jope-context-worker"}[worker["role"]]:
                require(worker["executable"] == ref and worker["build_receipt"] == summary["build_receipt"] and worker["features"] == matches[0]["features"] and "context-producer" in worker["features"], "worker feature/build binding differs")
    return summary, inventory


def prepare_context(args):
    with allocation():
        build, _ = worker_build(args.worker_build)
        out = fresh_output(args.output)
        command = [build["executables"]["jope-context-proof"]["path"], "prepare"]
        for flag, path in (("sources", args.reference_sources), ("encoder-files", args.encoder_files), ("resources", args.resources)):
            ref = artifact(path)
            command += ["--" + flag, ref["path"], "--" + flag + "-sha256", ref["sha256"]]
        worker = build["executables"]["jope-context-worker"]
        command += ["--worker", worker["path"], "--worker-sha256", worker["sha256"], "--source-sha40", build["source_sha40"], "--output", str(out)]
        started = time.time()
        result = subprocess.run(command, capture_output=True, text=True)
        write(out / "preparation.json", {"command": command, "started_unix": started, "elapsed_seconds": time.time() - started,
                                         "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                                         "fits": 0, "execution_budget_started": False})
        require(result.returncode == 0, "context preparation failed; see preparation.json")
        parts = result.stdout.split()
        require(len(parts) == 2, "unexpected native preparation response")
        plan = {"path": parts[0], "sha256": parts[1]}
        checked(plan)
        next_step = {"command": [command[0], "run", "--plan", parts[0], "--plan-sha256", parts[1]], "admitted": False}
        write(out / "next-command.json", next_step)
        return next_step


def configuration(args):
    with allocation():
        build, inventory = worker_build(args.worker_build)
        policy, assets = read(args.policy_preparation), read(args.asset_preparation)
        encoder_files = read(checked(assets["configuration"]))
        closure = {}
        for ref in assets["model_closure"] + [encoder_files[key] for key in
                                               ("hyperion_weights", "hyperion_manifest", "frozen_transforms")]:
            checked(ref)
            require(closure.setdefault(ref["path"], ref) == ref, "conflicting encoder asset digest")
        qualification = read(args.completed_qualification) if args.completed_qualification else None
        if qualification is not None and qualification.get("schema") == "jope.cpu-context-proof-result/v1":
            require(qualification.get("validation_error") is None and qualification.get("completed") is not None,
                    "CPU qualification did not complete")
            qualification = qualification["completed"]
        if qualification is not None:
            require(set(qualification) == {"qualification", "reference_manifest", "case_outputs", "case_executions"},
                    "actual CompletedQualification required")
        roles = {worker["role"]: worker for worker in inventory["workers"]}

        def runtime(role):
            return {"program": roles[role]["executable"], "argument_prefix": [], "inputs": [],
                    "package_files": [], "environment": {}, "version": build["source_sha40"], "unavailable": None}

        for key in ("checkpoint_receipt", "policy", "vocabulary"):
            checked(policy[key])
        config = {"schema": "jope.asm-configuration/v1", "source_sha40": build["source_sha40"],
                  "release_manifest": build["worker_inventory"], "source_manifest": build["source_manifest"],
                  "checkpoint_receipt": policy["checkpoint_receipt"], "policy_weights": policy["policy"],
                  "vocabulary": policy["vocabulary"], "frozen_transforms": encoder_files["frozen_transforms"],
                  "policy": runtime("policy"), "encoder": runtime("context"),
                  "encoder_assets": {"configuration": assets["configuration"], "files": [closure[k] for k in sorted(closure)]},
                  "qualification": qualification, **PROTOCOL}
        out = fresh_output(args.output)
        write(out / "configuration.json", config)
        ref = artifact(out / "configuration.json")
        command = [build["executables"]["jope"]["path"], "optimize", "validate-configuration",
                   "--configuration", ref["path"], "--configuration-sha256", ref["sha256"]]
        result = subprocess.run(command, capture_output=True, text=True)
        write(out / "native-validation.json", {"command": command, "exit_code": result.returncode,
                                               "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, "native configuration validation failed; see native-validation.json")
        validation = json.loads(result.stdout)
        require(validation.get("schema") == "jope.asm-configuration-validation/v1"
                and validation.get("ok") is True and validation.get("source_sha40") == build["source_sha40"]
                and validation.get("configuration") == ref
                and validation.get("context_qualified") is (qualification is not None),
                "native validation response differs")
        receipt = {"configuration": ref, "native_validation": artifact(out / "native-validation.json"),
                   "context_qualified": qualification is not None, "fits": 0, "execution_budget_started": False,
                   "protocol": "asm-preparation-defaults/v1", "protocol_defaults": PROTOCOL,
                   "preparation_inputs": [artifact(args.worker_build), artifact(args.policy_preparation), artifact(args.asset_preparation)]}
        if args.completed_qualification:
            receipt["preparation_inputs"].append(artifact(args.completed_qualification))
        write(out / "receipt.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-workers")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--expect-source", required=True)
    build.add_argument("--jobs", type=int, default=4)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(run=build_workers)
    context = sub.add_parser("prepare-context-proof")
    for flag in ("worker-build", "reference-sources", "encoder-files", "resources", "output"):
        context.add_argument("--" + flag, type=Path, required=True)
    context.set_defaults(run=prepare_context)
    config = sub.add_parser("configuration")
    for flag in ("worker-build", "policy-preparation", "asset-preparation", "output"):
        config.add_argument("--" + flag, type=Path, required=True)
    config.add_argument("--completed-qualification", type=Path)
    config.set_defaults(run=configuration)
    args = parser.parse_args()
    print(json.dumps(args.run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
