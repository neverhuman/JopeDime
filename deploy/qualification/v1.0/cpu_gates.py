#!/usr/bin/env python3
"""Run same-source CPU gates; retain failures without claiming release qualification."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "jope.qualification-cpu-prestage/v1"
GATE_IDS = "M1 M2 M3 M4 M5 S1 S2 S3 S4 R1 R2 R3 R4 A1 A2 A3 A4 A5 E1 E2 E3 U1 U2 T1 T2 T3 D1 D2 Q1".split()
FIELDS = "collected executed passed failed skipped excluded".split()
CONTROLS = {
    "source-hygiene": ["just", "hygiene"],
    "source-format": ["cargo", "fmt", "--all", "--", "--check"],
    "source-check": ["cargo", "check", "--workspace", "--all-targets", "--locked"],
    "source-clippy": [
        "cargo",
        "clippy",
        "--workspace",
        "--all-targets",
        "--locked",
        "--",
        "-D",
        "warnings",
    ],
    "source-python": ["just", "python-check"],
    "source-audit": ["just", "audit"],
    "source-paths": ["just", "lint-paths"],
    "source-dependencies": ["cargo", "deny", "--all-features", "check"],
}
INPUTS = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "python/uv.lock",
    "justfile",
    "deploy/ci/coverage.json",
    "deploy/ci/full-main.py",
    "deploy/ci/full-lane.sh",
    "deploy/ci/fast-lane.sh",
    "deploy/ci/results.py",
]
MAX_LOG_BYTES = 64 * 1024 * 1024


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return dt.datetime.now(dt.UTC).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def identity(root, expected=None, require_clean=False):
    main = git(root, "rev-parse", "refs/remotes/origin/main")
    head = git(root, "rev-parse", "HEAD")
    if len(main) != 40 or any(c not in "0123456789abcdef" for c in main):
        raise ValueError("origin/main must resolve to a full commit")
    if expected is not None and main != expected:
        raise ValueError("origin/main differs from the explicitly expected source")
    dirty = git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    if require_clean and (head != main or dirty):
        raise ValueError("execution requires a clean checkout at frozen origin/main")
    return dict(
        main_sha=main,
        head_sha=head,
        tree_sha=git(root, "rev-parse", main + "^{tree}"),
        clean_at_main=head == main and not dirty,
        dirty_paths=dirty,
    )


def inventory(root, revision=None):
    def read(name):
        return (
            git(root, "show", revision + ":" + name)
            if revision
            else (root / name).read_text()
        )

    mapping = json.loads(read("deploy/ci/coverage.json"))
    syntax = ast.parse(read("deploy/ci/full-main.py"))
    definitions = [
        n
        for n in syntax.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "GROUPS" for t in n.targets)
    ]
    if len(definitions) != 1:
        raise ValueError("expected one literal full CPU GROUPS declaration")
    groups = ast.literal_eval(definitions[0].value)
    names = [name for entries in groups.values() for name in entries]
    families = {
        name: lane
        for name, lane in mapping["lanes"].items()
        if name.startswith("full-")
    }
    if not names or len(names) != len(set(names)) or set(names) != set(families):
        raise ValueError("full CPU groups and coverage families must partition exactly")
    result = [
        dict(
            id=name,
            command=command,
            result_backend="none",
            requires_nonzero_tests=False,
        )
        for name, command in CONTROLS.items()
    ]
    for name in names:
        if not re.fullmatch(r"full-[a-z0-9-]+", name):
            raise ValueError("invalid CPU family identifier")
        lane = families[name]
        command = lane["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(x, str) or not x for x in command)
        ):
            raise ValueError("invalid CPU command: " + name)
        backend = lane["result_backend"]
        if backend not in {"none", "nextest", "rust", "pytest", "unittest"}:
            raise ValueError("unsupported result backend: " + name)
        if lane["requires_nonzero_tests"] and backend == "none":
            raise ValueError("nonzero test family requires a result backend: " + name)
        result.append(
            dict(
                id=name,
                command=command,
                result_backend=backend,
                requires_nonzero_tests=lane["requires_nonzero_tests"],
            )
        )
    return result


def result_parser(root):
    spec = importlib.util.spec_from_file_location(
        "qualification_cpu_results", root / "deploy/ci/results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_inventory(plan_root):
    """Inventory dated plan/status metadata; never infer signed-byte qualification."""
    plan_root = plan_root.resolve()
    sources = {}
    texts = {}

    def read(relative, required=True):
        path = plan_root / relative
        if not path.resolve().is_relative_to(plan_root):
            raise ValueError("plan metadata escapes its declared root")
        if relative not in sources:
            exists = path.is_file()
            if required and not exists:
                raise ValueError("required plan metadata missing: " + relative)
            data = path.read_bytes() if exists else None
            sources[relative] = dict(
                path=relative,
                exists=exists,
                sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
            )
            texts[relative] = data.decode("utf-8") if data is not None else None
        return texts[relative], sources[relative]

    master, _ = read("master_plan.md")
    specification, _ = read("source/spec-v1.md")
    section = specification.split("## 14. Release gates", 1)[-1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| \*\*([A-Z][0-9]+)\*\* \| ([^|]+)\| (.+) \|$", section, re.M)
    if [row[0] for row in rows] != GATE_IDS[:-1]:
        raise ValueError("specification must enumerate every release gate exactly once")
    q1 = re.search(r"\*\*Capability gate Q1 — learned guidance\.\*\* (.+)", section)
    if q1 is None or "28" not in master or "Q1" not in master:
        raise ValueError("master plan/specification capability-gate mapping missing")
    rows.append(("Q1", "Learned guidance default", q1[1]))
    gates = {
        gate: dict(
            status="not-qualified",
            name=name.strip(),
            criterion=criterion.strip(),
            reason="Signed candidate and qualifying rerun receipts are unavailable; task notes are not qualification",
            release_blocking=gate != "Q1",
            rerun_status="not-run",
            rerun_command=None,
            qualification_evidence=None,
            evidence=[],
        )
        for gate, name, criterion in rows
    }
    task_rows = re.findall(
        r"^\| P[0-9]+ \| \[([A-Z]+-[0-9]+)\]\((tasks/[^/)]+\.md)\).*?\| ([^|]+) \|$",
        master,
        re.M,
    )
    if not task_rows:
        raise ValueError("master plan task index missing")
    for task_id, card_path, plan_status in task_rows:
        card, card_source = read(card_path)
        field = re.search(r"^\| Gates closed \| ([^|]+)\|", card, re.M)
        if field is None:
            raise ValueError("task gate mapping missing: " + task_id)
        mapped = set(re.findall(r"\b[A-Z][0-9]+\b", field[1])) & set(GATE_IDS)
        if not mapped:  # REL-07's aggregate 'all' is not per-gate proof.
            continue
        note, note_source = read("status/" + task_id + ".md", required=False)
        evidence = dict(
            task=task_id,
            task_card=card_source,
            reported_plan_status=plan_status.strip(),
            status_note=dict(
                **note_source,
                reported_status=next(
                    (x.strip() for x in (note or "").splitlines() if x.strip()), None
                ),
            ),
            qualifies_signed_bytes=False,
        )
        for gate in mapped:
            gates[gate]["evidence"].append(evidence)
    # Catch a concurrently rewritten status/plan during this inventory, including
    # an absent note appearing. Later changes require a fresh dated inventory.
    for relative, source in sources.items():
        path = plan_root / relative
        if path.is_file() != source["exists"] or (
            source["exists"] and digest(path) != source["sha256"]
        ):
            raise ValueError(
                "plan/status metadata changed during inventory: " + relative
            )
    return dict(
        plan_root=str(plan_root),
        plan_sources=list(sources.values()),
        release_gates=gates,
    )


def admission(root):
    capacity = os.statvfs(root)
    used = capacity.f_blocks - capacity.f_bfree
    percent = 100 * used / (used + capacity.f_bavail)
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    if load is None or load > 0.85 * cores or percent >= 95:
        raise ValueError(
            "CPU/disk admission unavailable: load <=85%, disk <95% required"
        )
    return dict(cpu_count=cores, load1=load, disk_used_percent=percent)


def child_environment(folder, target, jobs, source):
    forwarded = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CC",
        "CXX",
        "AR",
        "LD_LIBRARY_PATH",
        "LIBTORCH",
        "LIBTORCH_USE_PYTORCH",
        "LIBTORCH_BYPASS_VERSION_CHECK",
        "CMAKE_PREFIX_PATH",
        "PKG_CONFIG_PATH",
        "UV_CACHE_DIR",
    )
    environment = {key: os.environ[key] for key in forwarded if key in os.environ}
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        AWS_EC2_METADATA_DISABLED="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        CARGO_BUILD_JOBS=str(jobs),
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CARGO_TARGET_DIR=str(target),
        JOPE_LANE_EVIDENCE=str(folder),
        JOPE_SOURCE_SHA=source,
        JOPE_PARITY_STRICT="1",
        JOPE_CPU_TEST_THREADS=str(jobs),
        JOPE_BUILD_JOBS=str(jobs),
        JOPE_CPU_BUILD_JOBS=str(jobs),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(folder / "scratch"),
    )
    return environment


def validate_timeout(seconds):
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 3600:
        raise ValueError(
            "gate timeout must be finite, positive and at most 3600 seconds"
        )


def _wait_unreaped(process, seconds):
    # Keep the leader's PID reserved until cleanup has signalled its whole group.
    deadline = time.monotonic() + seconds
    while (
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, seconds)
        time.sleep(min(0.05, remaining))


def _stop_owned_group(process):
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(5)
    finally:
        # The leader may have exited, but a descendant can still ignore SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def invoke(command, root, log, environment, seconds):
    validate_timeout(seconds)
    with log.open("xb") as stream:
        process = subprocess.Popen(
            ["nice", "-n", "10", *command],
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_unreaped(process, seconds)
        except subprocess.TimeoutExpired:
            _stop_owned_group(process)
            return 124, True
        except BaseException:
            _stop_owned_group(process)
            raise
        # The exited leader remains unreaped, reserving its PID/PGID while
        # owned descendants are stopped before receipt/log hashing. Preserve
        # the leader's original exit status rather than treating cleanup as success.
        _stop_owned_group(process)
        return process.returncode, False


def execute_gate(
    gate, root, folder, target, jobs, source, parser, seconds, check=admission
):
    folder.mkdir()
    (folder / "scratch").mkdir()
    record = dict(
        **gate,
        started_utc=utc(),
        finished_utc=None,
        exit_status=None,
        status="unavailable",
        tests=dict.fromkeys(FIELDS),
        diagnostic=None,
        admission=None,
        outputs={},
        omitted_symlinks=[],
    )
    start = time.monotonic()
    log = folder / "command.log"
    try:
        record["admission"] = check(root)
        environment = child_environment(folder, target, jobs, source)
        code, timed_out = invoke(gate["command"], root, log, environment, seconds)
        record["exit_status"] = code
        record["status"] = (
            "timed-out"
            if timed_out or code == 124
            else (
                "unavailable"
                if code in (78, 126, 127)
                else "failed"
                if code != 0
                else "passed"
            )
        )
        if log.is_symlink() or not log.is_file():
            raise ValueError("command log must be a regular file")
        if gate["result_backend"] != "none":
            if log.stat().st_size > MAX_LOG_BYTES:
                raise ValueError(
                    "test log exceeds count-parser bound; complete log retained"
                )
            junit = folder / "junit.xml"
            if log.is_symlink() or junit.is_symlink():
                raise ValueError("test evidence must be regular files, never symlinks")
            if gate["result_backend"] in {"nextest", "pytest"} and not junit.is_file():
                raise ValueError("fresh JUnit evidence is required")
            counts = parser.summarize(
                log.read_text(errors="replace"),
                gate["result_backend"],
                junit if junit.exists() else None,
            )
            record["tests"] = counts
            if any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("unknown or invalid test counts")
            if counts["executed"] != counts["passed"] + counts["failed"] or (
                counts["collected"]
                != counts["executed"] + counts["skipped"] + counts["excluded"]
            ):
                raise ValueError("test counts do not reconcile")
            if counts["executed"] == 0:
                raise ValueError("zero executed tests are not passing evidence")
            if gate["result_backend"] == "nextest":
                collection_path = folder / "collection.json"
                if collection_path.is_symlink():
                    raise ValueError("collection must not be a symlink")
                collection = json.loads(collection_path.read_text())
                listed = sum(
                    len(suite["testcases"])
                    for suite in collection["rust-suites"].values()
                )
                if collection["test-count"] != listed or listed != counts["collected"]:
                    raise ValueError("fresh Nextest collection and execution differ")
            if code == 0 and counts["failed"]:
                record["status"] = "failed"
            elif record["status"] == "passed" and (
                counts["skipped"] or counts["excluded"]
            ):
                record["status"] = "incomplete"
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["diagnostic"] = str(error)
        if record["exit_status"] == 0:
            record["status"] = "failed"
    record["finished_utc"] = utc()
    record["wall_seconds"] = time.monotonic() - start
    for path in sorted(folder.rglob("*")):
        relative = str(path.relative_to(folder))
        try:
            if path.is_symlink():
                record["omitted_symlinks"].append(relative)
            elif path.is_file():
                record["outputs"][relative] = dict(
                    sha256=digest(path), bytes=path.stat().st_size
                )
        except OSError as error:
            record["diagnostic"] = record["diagnostic"] or (
                f"cannot capture output {relative}: {error}"
            )
            if record["status"] in {"passed", "incomplete"}:
                record["status"] = "failed"
    write(folder / "receipt.json", record)
    return record


def run(root, output, expected=None, jobs=4, seconds=1800, execute=False, target=None):
    validate_timeout(seconds)
    root = root.resolve()
    if output.is_symlink():
        raise ValueError("evidence directory must not be a symlink")
    output = output.resolve()
    target = (target or root / "target").resolve()
    if (
        output == root
        or output.is_relative_to(Path("/tmp"))
        or target.is_relative_to(Path("/tmp"))
    ):
        raise ValueError("evidence and builds require an explicit non-/tmp location")
    source = identity(root, expected, require_clean=execute)
    gates = inventory(root, source["main_sha"])
    output.mkdir(parents=True, exist_ok=False)
    actual_jobs = min(max(jobs, 1), 4)
    document = dict(
        schema=SCHEMA,
        phase="cpu-prestage",
        started_utc=utc(),
        finished_utc=None,
        status="planned",
        source=source,
        driver_sha256=digest(Path(__file__)),
        machine=dict(
            host=socket.gethostname(),
            platform=platform.platform(),
            python=platform.python_version(),
        ),
        environment=dict(
            CUDA_VISIBLE_DEVICES="",
            CARGO_BUILD_JOBS=str(actual_jobs),
            CARGO_INCREMENTAL="0",
            CARGO_PROFILE_DEV_DEBUG="line-tables-only",
            CARGO_TARGET_DIR=str(target),
            math_threads=1,
        ),
        inputs={
            path: hashlib.sha256(
                subprocess.check_output(
                    ["git", "-C", str(root), "show", source["main_sha"] + ":" + path]
                )
            ).hexdigest()
            for path in INPUTS
        },
        inventory=gates,
        checks=[],
        release_gates={
            gate: dict(
                status="not-qualified",
                reason="CPU source diagnostics do not qualify signed release bytes",
            )
            for gate in GATE_IDS
        },
        signed_candidate=None,
        owner_approval=None,
        release_qualified=False,
    )
    write(output / "qualification_receipt.json", document)
    if execute:
        document["status"] = "running"
        parser = result_parser(root)
        try:
            for gate in gates:
                document["checks"].append(
                    execute_gate(
                        gate,
                        root,
                        output / gate["id"],
                        target,
                        actual_jobs,
                        source["main_sha"],
                        parser,
                        seconds,
                    )
                )
                write(output / "qualification_receipt.json", document)
        except KeyboardInterrupt:
            document.update(status="failed", interrupted=True, finished_utc=utc())
            write(output / "qualification_receipt.json", document)
            raise
        statuses = {x["status"] for x in document["checks"]}
        document["status"] = (
            "failed"
            if statuses & {"failed", "timed-out"}
            else "passed"
            if statuses == {"passed"}
            else "incomplete"
        )
        try:
            after = identity(root, expected=source["main_sha"], require_clean=True)
            if after != source:
                raise ValueError("source identity changed during CPU checks")
        except (ValueError, subprocess.CalledProcessError) as error:
            document["status"] = "failed"
            document["source_changed"] = str(error)
    document["finished_utc"] = utc()
    write(output / "qualification_receipt.json", document)
    return document


def release_plan(root, output, plan_root, expected=None, jobs=4):
    catalog = release_inventory(plan_root)
    document = run(root, output, expected=expected, jobs=jobs)
    document.update(catalog)
    write(output / "qualification_receipt.json", document)
    # Compact reviewable current-state table; the JSON retains every byte pin.
    lines = [
        "# Release gate inventory — preparation only",
        "",
        "Observed "
        + document["finished_utc"]
        + "; source "
        + document["source"]["main_sha"],
        "",
        "All statuses concern exact signed-byte qualification. No gates were rerun.",
        "",
        "| Gate | Name | Status | Current task/status evidence paths |",
        "| --- | --- | --- | --- |",
    ]
    for gate, record in catalog["release_gates"].items():
        refs = (
            "; ".join(
                ref["status_note"]["path"]
                + (" (missing)" if not ref["status_note"]["exists"] else "")
                for ref in record["evidence"]
            )
            or "No task-specific status note mapped"
        )
        lines.append(
            f"| {gate} | {record['name']} | not-qualified / not-run | {refs} |"
        )
    (output / "gate-status.md").write_text("\n".join(lines) + "\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "release-plan"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--expected-main-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--gate-timeout", type=float, default=1800)
    args = parser.parse_args()
    try:
        validate_timeout(args.gate_timeout)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "release-plan" and args.plan_root is None:
        parser.error(
            "release-plan requires --plan-root with master plan/task/spec metadata"
        )
    if args.command == "run":

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        receipt = (
            release_plan(
                args.repo,
                args.output,
                args.plan_root,
                args.expected_main_sha,
                args.jobs,
            )
            if args.command == "release-plan"
            else run(
                args.repo,
                args.output,
                args.expected_main_sha,
                args.jobs,
                args.gate_timeout,
                args.command == "run",
                args.target,
            )
        )
        print(
            json.dumps(
                dict(
                    status=receipt["status"],
                    source=receipt["source"]["main_sha"],
                    checks=len(receipt["checks"]),
                    release_qualified=False,
                )
            )
        )
        return 0 if receipt["status"] in {"planned", "passed"} else 1
    except KeyboardInterrupt:
        print(
            "CPU qualification interrupted; incomplete checkpoint retained",
            file=sys.stderr,
        )
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print("CPU qualification refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
