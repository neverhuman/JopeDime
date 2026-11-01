#!/usr/bin/env python3
"""Offline Q1 orchestration and paired statistics. Qualification reads metadata, never rows."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import stat
import statistics
import subprocess
import tempfile
import time

import q1_bindings

HERE = Path(__file__).resolve().parent
PREREGISTRATION = HERE.parent / "PREREGISTRATION.md"
PLAN_KEYS = {
    "schema",
    "mode",
    "source_commit",
    "preregistration_sha256",
    "backend_sha256",
    "budget",
    "seeds",
    "datasets",
    "pins",
    "training_families",
    "timeout_seconds",
    "ledger_root",
}
DATA_KEYS = {
    "id",
    "task",
    "dataset_sha256",
    "family",
    "family_authoritative",
    "rows",
    "features",
    "cut_sha256",
    "holdout_group",
}
GROUPS = ("small", "large", "narrow", "wide")
R19_ROSTER_SHA256 = "ab434d145b8cec60399b35c35a1ce9551a881843e8136b1c757e9c93ae62bd9a"
R19_ROSTER = HERE.parent / "rosters" / (R19_ROSTER_SHA256 + ".json")
R19_BUNDLE = "94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
R19_DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R19_WIDE_LABEL = (
    "4/5 families, below preregistered minimum, descriptive only, no subgroup verdict"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(text):
    return json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def load(path):
    return decode(Path(path).read_text())


def integer(value, lower=0):
    return type(value) is int and value >= lower


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def memberships(dataset):
    return (
        "small" if dataset["rows"] <= 1000 else "large",
        "narrow" if dataset["features"] <= 20 else "wide",
    )


def validate_r19(plan):
    """The signed exception applies only to the unchanged R19 roster and candidate."""
    require(
        plan["mode"] == "qualification"
        and type(plan["minimum_wide_families"]) is int
        and plan["minimum_wide_families"] == 4,
        "R19 permits only minimum_wide_families=4 in qualification",
    )
    require(
        plan["pins"]["bundle_sha256"] == R19_BUNDLE
        and plan["pins"]["dope_sha256"] == R19_DOPE,
        "R19 candidate/DOPE mismatch",
    )
    require(sha_file(R19_ROSTER) == R19_ROSTER_SHA256, "R19 roster hash mismatch")
    roster = load(R19_ROSTER)["official_roster"]
    expected = {r["dataset_id"]: r for r in roster}
    require(
        len(plan["datasets"]) == len(expected) == 153
        and {d["id"] for d in plan["datasets"]} == set(expected),
        "R19 requires the unchanged 153-cut roster",
    )
    for dataset in plan["datasets"]:
        record = expected[dataset["id"]]
        require(
            all(
                dataset[k] == record[k]
                for k in ("dataset_sha256", "cut_sha256", "rows", "features")
            )
            and (
                record["family_key"] is None
                or dataset["family"] == record["family_key"]
            ),
            "R19 roster identity/cut/shape/family changed",
        )


def validate_plan(plan, preregistration=PREREGISTRATION):
    require(
        set(plan) in (PLAN_KEYS, PLAN_KEYS | {"minimum_wide_families"}),
        "unsupported/missing plan field",
    )
    require(plan["schema"] == "jope.q1-plan/v1", "unsupported plan schema")
    require(
        plan["preregistration_sha256"] == sha_file(preregistration),
        "preregistration hash mismatch",
    )
    require(sha(plan["backend_sha256"]), "backend hash required")
    require(
        re.fullmatch(r"[0-9a-f]{40}", plan["source_commit"]) is not None,
        "source commit required",
    )
    require(plan["mode"] in ("heuristic_smoke", "qualification"), "invalid mode")
    require(
        isinstance(plan["seeds"], list)
        and plan["seeds"]
        and all(integer(v) for v in plan["seeds"])
        and len(set(plan["seeds"])) == len(plan["seeds"]),
        "invalid/duplicate seeds",
    )
    require(
        isinstance(plan["datasets"], list) and plan["datasets"], "datasets required"
    )
    seen = set()
    families = set()
    for dataset in plan["datasets"]:
        extra = (
            {"train_sha256", "has_meta_test"}
            if plan["mode"] == "heuristic_smoke"
            else set()
        )
        require(
            set(dataset) == DATA_KEYS | extra, "unsupported/missing dataset metadata"
        )
        require(dataset["task"] == "regression", "Q1 is regression-only")
        require(
            sha(dataset["dataset_sha256"]) and sha(dataset["cut_sha256"]),
            "dataset/cut hash required",
        )
        require(
            re.fullmatch(r"[0-9a-f]{16}", dataset["id"]) is not None
            and dataset["dataset_sha256"].startswith(dataset["id"]),
            "invalid dataset identity",
        )
        require(
            dataset["id"] not in seen and dataset["family"] not in families,
            "duplicate dataset/family",
        )
        require(
            dataset["family_authoritative"] is True
            and isinstance(dataset["family"], str)
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", dataset["family"])
            is not None,
            "authoritative family identity required",
        )
        require(
            integer(dataset["rows"], 1) and integer(dataset["features"], 1),
            "invalid shape",
        )
        seen.add(dataset["id"])
        families.add(dataset["family"])
    if plan["mode"] == "qualification":
        require(
            isinstance(plan["ledger_root"], str)
            and Path(plan["ledger_root"]).is_absolute(),
            "signed persistent ledger root required",
        )
        require(len(families) >= 40, "Q1 needs at least 40 sealed meta-test families")
        require(
            plan["seeds"] == [0, 1, 2]
            and plan["budget"] == 24
            and plan["timeout_seconds"] == 600,
            "qualification budget/seeds/limit differ from preregistration",
        )
        require(
            set(plan["pins"])
            == {
                "bundle_sha256",
                "dope_sha256",
                "evaluator_sha256",
                "vocabulary_sha256",
                "protocol_sha256",
                "training_family_manifest_sha256",
            }
            and all(sha(v) for v in plan["pins"].values()),
            "all qualification identities must be pinned",
        )
        q1_bindings.validate_pins(plan["pins"])
        require(
            isinstance(plan["training_families"], list)
            and plan["training_families"]
            and all(isinstance(v, str) for v in plan["training_families"]),
            "training closure missing",
        )
        require(
            not families.intersection(plan["training_families"]),
            "training/cohort family overlap",
        )
        require(
            all(d["holdout_group"] == "sealed_meta_test" for d in plan["datasets"]),
            "sealed meta-test is the ONLY qualification holdout",
        )
        if "minimum_wide_families" in plan:
            validate_r19(plan)
        for group in GROUPS:
            minimum = plan.get("minimum_wide_families", 5) if group == "wide" else 5
            require(
                sum(group in memberships(d) for d in plan["datasets"]) >= minimum,
                "too few datasets in preregistered subgroup " + group,
            )
    else:
        require("minimum_wide_families" not in plan, "R19 is qualification only")
        require(
            plan["ledger_root"] is None, "smoke cannot consume a qualification ledger"
        )
        require(
            integer(plan["budget"], 1)
            and plan["budget"] <= 4
            and plan["timeout_seconds"] == 60,
            "invalid bounded smoke budget/limit",
        )
        require(
            plan["pins"] == {} and plan["training_families"] == [],
            "smoke cannot claim qualification pins",
        )
        roster = {r["id"]: r for r in load(HERE / "smoke-catalog.json")["datasets"]}
        require(
            seen == set(roster),
            "smoke must use the three approved non-sealed catalog choices",
        )
        for dataset in plan["datasets"]:
            record = roster[dataset["id"]]
            require(
                dataset["holdout_group"] == "development_fit"
                and dataset["has_meta_test"] is False,
                "smoke cannot read any holdout",
            )
            require(
                (dataset["dataset_sha256"], dataset["family"], dataset["train_sha256"])
                == (
                    record["canonical_sha256"],
                    record["family_id"],
                    record["train_sha256"],
                ),
                "smoke provenance mismatch",
            )
    return plan


def approval_payload(plan):
    return (
        "jope.q1-approval/v1\n"
        + plan["preregistration_sha256"]
        + "\n"
        + digest(plan)
        + "\n"
        + plan["backend_sha256"]
        + "\n"
    ).encode()


def verify_owner(plan, signature, allowed_signers, preregistration=PREREGISTRATION):
    text = Path(preregistration).read_text()
    match = re.search(
        r"^Owner allowed-signers SHA-256: ([0-9a-f]{64})$", text, re.MULTILINE
    )
    require(match is not None, "OWNER_SIGNATURE_REQUIRED: preregistration is unsigned")
    require(
        signature is not None and allowed_signers is not None,
        "OWNER_SIGNATURE_REQUIRED",
    )
    require(
        sha_file(allowed_signers) == match[1], "owner verification key hash mismatch"
    )
    result = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(allowed_signers),
            "-I",
            "jepson",
            "-n",
            "jopedime-q1",
            "-s",
            str(signature),
        ],
        input=approval_payload(plan),
        capture_output=True,
    )
    require(
        result.returncode == 0,
        "OWNER_SIGNATURE_REQUIRED: invalid/unbound owner signature",
    )


def smoke_payload(dataset, data_root, seed):
    require(data_root is not None, "smoke data root required")
    root = Path(data_root).resolve(strict=True)
    require(
        not any(
            t in str(root).lower()
            for t in ("sealed", "holdout", "meta_test", "meta-test")
        ),
        "forbidden payload root",
    )
    path = root / (dataset["train_sha256"] + ".csv")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= 4 * 1024 * 1024,
            "invalid smoke blob",
        )
        data = stream.read(4 * 1024 * 1024 + 1)
    require(
        hashlib.sha256(data).hexdigest() == dataset["train_sha256"],
        "smoke blob hash mismatch",
    )
    rows = []
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        values = [float(v) for v in row]
        require(
            len(values) == dataset["features"] + 1
            and all(math.isfinite(v) for v in values),
            "invalid normalized numeric FIT row",
        )
        rows.append(values)
        if len(rows) == 128:
            break
    require(len(rows) >= 16, "insufficient FIT rows for smoke CV")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    folds = [
        {"train": order[1::2], "validation": order[::2]},
        {"train": order[::2], "validation": order[1::2]},
    ]
    return {
        "x": [row[:8] if len(row) > 9 else row[:-1] for row in rows],
        "y": [row[-1] for row in rows],
        "folds": folds,
    }


def response_valid(response, request):
    keys = {
        "schema",
        "request_sha256",
        "status",
        "guidance",
        "metric",
        "charged",
        "completed",
        "loss",
        "fold_scores_completed",
        "candidate_failures",
        "champion_program",
        "adapter",
    }
    require(
        set(response) == keys and response["schema"] == "jope.q1-response/v1",
        "invalid response fields/schema",
    )
    require(
        response["request_sha256"] == request["request_sha256"]
        and response["guidance"] == request["guidance"]
        and response["metric"] == "rmse",
        "response identity mismatch",
    )
    require(
        integer(response["charged"]) and response["charged"] == request["budget"],
        "unequal actual charged budget",
    )
    require(
        integer(response["completed"])
        and 0 <= response["completed"] <= response["charged"],
        "invalid completion count",
    )
    require(
        integer(response["fold_scores_completed"])
        and isinstance(response["candidate_failures"], list)
        and len(response["candidate_failures"])
        == response["charged"] - response["completed"],
        "missing attempt evidence",
    )
    failure_attempts = set()
    for failure in response["candidate_failures"]:
        require(
            isinstance(failure, dict)
            and integer(failure.get("attempt"))
            and failure["attempt"] < response["charged"]
            and isinstance(failure.get("reason"), str)
            and bool(failure["reason"].strip()),
            "invalid per-attempt failure evidence",
        )
        require(failure["attempt"] not in failure_attempts, "duplicate failure attempt")
        failure_attempts.add(failure["attempt"])
    require(response["status"] in ("complete", "failed"), "invalid response outcome")
    if response["status"] == "complete":
        require(
            all(
                isinstance(response[name], str) and response[name].strip()
                for name in ("champion_program", "adapter")
            ),
            "complete result requires program and adapter identities",
        )
        require(
            response["completed"] > 0
            and response["fold_scores_completed"] >= 2 * response["completed"]
            and type(response["loss"]) in (float, int)
            and math.isfinite(response["loss"])
            and response["loss"] >= 0,
            "missing/non-finite RMSE",
        )
    else:
        require(
            response["loss"] is None, "failed results must not have placeholder scores"
        )
    return response


def append_receipt(stream, value):
    stream.write(canonical(value).decode() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


class BackendSpawnError(RuntimeError):
    """The backend could not be started; stop the panel without retrying."""


class BackendCleanupError(RuntimeError):
    """The owned process group could not be proven inactive."""


def _proc_identity(pid):
    """Return Linux process identity fields, or None if the process vanished."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
            raw = stream.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = raw.rfind(")")
    if close < 0:
        raise BackendCleanupError("malformed /proc process identity")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        raise BackendCleanupError("short /proc process identity")
    try:
        return {
            "state": fields[0],
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "start": int(fields[19]),
        }
    except ValueError as error:
        raise BackendCleanupError("invalid /proc process identity") from error


def _processes_in_scope(session_id, process_group=None):
    try:
        names = os.listdir("/proc")
    except OSError as error:
        raise BackendCleanupError("cannot enumerate owned process session") from error
    result = []
    for name in names:
        if not name.isdecimal():
            continue
        pid = int(name)
        identity = _proc_identity(pid)
        if identity is None or identity["state"] in ("Z", "X", "x"):
            continue
        if identity["session"] != session_id:
            continue
        if process_group is not None and identity["pgrp"] != process_group:
            continue
        result.append((pid, identity))
    return result


def _signal_scope(session_id, process_group, signum):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise BackendCleanupError("pidfd process identity guards are unavailable")
    for pid, before in _processes_in_scope(session_id, process_group):
        try:
            descriptor = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            continue
        except OSError as error:
            raise BackendCleanupError("cannot pin owned process identity") from error
        try:
            current = _proc_identity(pid)
            if (
                current is None
                or current["state"] in ("Z", "X", "x")
                or current["start"] != before["start"]
                or current["session"] != session_id
                or (process_group is not None and current["pgrp"] != process_group)
            ):
                continue
            try:
                signal.pidfd_send_signal(descriptor, signum, None, 0)
            except ProcessLookupError:
                pass
            except OSError as error:
                raise BackendCleanupError("cannot signal owned process") from error
        finally:
            os.close(descriptor)


def _terminate_scope(session_id, process_group, grace, wait_forever=False):
    """Terminate an owned group/session, never signalling a reused process ID."""
    if not math.isfinite(grace) or grace < 0:
        raise BackendCleanupError("invalid process cleanup grace")

    def attempt(signum):
        while True:
            try:
                _signal_scope(session_id, process_group, signum)
                return
            except BackendCleanupError:
                if not wait_forever:
                    raise
                time.sleep(0.1)

    while True:
        attempt(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while True:
            try:
                active = _processes_in_scope(session_id, process_group)
            except BackendCleanupError:
                if not wait_forever:
                    raise
                time.sleep(0.1)
                continue
            if not active:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            attempt(signal.SIGTERM)
        kill_deadline = time.monotonic() + 2
        while True:
            try:
                active = _processes_in_scope(session_id, process_group)
            except BackendCleanupError:
                if not wait_forever:
                    raise
                time.sleep(0.1)
                continue
            if not active:
                return
            attempt(signal.SIGKILL)
            if not wait_forever and time.monotonic() >= kill_deadline:
                remaining = [
                    pid for pid, _ in _processes_in_scope(session_id, process_group)
                ]
                raise BackendCleanupError(
                    "owned process cleanup could not be established; active PIDs: "
                    + ",".join(map(str, remaining))
                )
            time.sleep(0.02)


def terminate_session(session_id, grace=5):
    """Clean every live process in a private FIFO-launched session before lease release."""
    _terminate_scope(session_id, None, grace, wait_forever=True)


def _terminate_process_group(session_id, process_group, leader_start, grace):
    """Kill one group while its unreaped leader pins the group ID against reuse."""
    leader = _proc_identity(process_group)
    if (
        leader is None
        or leader["pgrp"] != process_group
        or leader["session"] != session_id
        or leader["start"] != leader_start
    ):
        raise BackendCleanupError("backend process group leader identity changed")
    active = _processes_in_scope(session_id, process_group)
    if not active:
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not _processes_in_scope(session_id, process_group):
            return
        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
    active = _processes_in_scope(session_id, process_group)
    if active:
        # The unreaped leader still pins process_group, so this cannot target a
        # recycled PGID. SIGKILL also reaches children forked during the grace.
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
    deadline = time.monotonic() + 2
    while True:
        active = _processes_in_scope(session_id, process_group)
        if not active:
            return
        if time.monotonic() >= deadline:
            raise BackendCleanupError(
                "backend group cleanup could not be established; active PIDs: "
                + ",".join(str(pid) for pid, _ in active)
            )
        time.sleep(0.05)


def _invoke_backend(command, request, env, timeout, cleanup_grace):
    """Capture one backend call while retaining its group leader through cleanup."""
    require(math.isfinite(timeout) and timeout > 0, "backend timeout must be positive")
    require(
        math.isfinite(cleanup_grace) and cleanup_grace >= 0,
        "backend cleanup grace must be non-negative",
    )

    def interrupt(signum, _frame):
        raise InterruptedError(
            "backend invocation interrupted by " + signal.Signals(signum).name
        )

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    child = None
    child_exited = False
    primary = None
    timed_out = False
    cleanup_error = None
    try:
        with (
            tempfile.TemporaryFile() as stdin,
            tempfile.TemporaryFile() as stdout,
            tempfile.TemporaryFile() as stderr,
        ):
            stdin.write(request)
            stdin.seek(0)
            try:
                child = subprocess.Popen(
                    command,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    process_group=0,
                )
            except OSError as error:
                raise BackendSpawnError(
                    "backend process could not be started"
                ) from error
            session_id = os.getsid(0)
            try:
                leader = _proc_identity(child.pid)
            except BaseException as error:
                leader = None
                primary = error
            if leader is None:
                cleanup_error = BackendCleanupError(
                    "backend process group identity is unavailable"
                )
            elif leader["pgrp"] != child.pid or leader["session"] != session_id:
                primary = BackendCleanupError("backend process group identity mismatch")
            else:
                deadline = time.monotonic() + timeout
                try:
                    while True:
                        if (
                            os.waitid(
                                os.P_PID,
                                child.pid,
                                os.WEXITED | os.WNOHANG | os.WNOWAIT,
                            )
                            is not None
                        ):
                            child_exited = True
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            break
                        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
                except BaseException as error:
                    primary = error
            ignored = {
                sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                if child is not None:
                    try:
                        if leader is not None:
                            _terminate_process_group(
                                session_id,
                                child.pid,
                                leader["start"],
                                cleanup_grace,
                            )
                    except BaseException as error:
                        cleanup_error = error
                    if cleanup_error is None or child_exited:
                        try:
                            child.wait()
                        except BaseException as error:
                            cleanup_error = error
            finally:
                for sig, handler in ignored.items():
                    signal.signal(sig, handler)
            if cleanup_error is not None:
                raise BackendCleanupError(
                    "backend cleanup failed; no later request may be dispatched"
                ) from cleanup_error
            if primary is not None:
                raise primary
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read()
            error_output = stderr.read()
            if timed_out:
                raise subprocess.TimeoutExpired(
                    command, timeout, output=output, stderr=error_output
                )
            return subprocess.CompletedProcess(
                command, child.returncode, output, error_output
            )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def execute(
    plan,
    backend,
    output,
    data_root=None,
    signature=None,
    allowed_signers=None,
    ledger=None,
    backend_timeout=None,
    backend_cleanup_grace=5,
):
    validate_plan(plan)
    require(sha_file(backend) == plan["backend_sha256"], "backend binary hash mismatch")
    if plan["mode"] == "qualification":
        verify_owner(plan, signature, allowed_signers)
        require(
            ledger is not None
            and Path(ledger).resolve() == Path(plan["ledger_root"]).resolve(),
            "persistent consumption ledger must match the owner-signed plan",
        )
    if hasattr(os, "getloadavg"):
        cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        require(
            os.getloadavg()[0] <= 0.85 * cpus, "host load exceeds approved CPU budget"
        )
    timeout = plan["timeout_seconds"] if backend_timeout is None else backend_timeout
    require(
        type(timeout) in (int, float)
        and math.isfinite(timeout)
        and 0 < timeout <= plan["timeout_seconds"],
        "backend timeout must be positive and no greater than the signed plan",
    )
    require(
        type(backend_cleanup_grace) in (int, float)
        and math.isfinite(backend_cleanup_grace)
        and backend_cleanup_grace >= 0,
        "backend cleanup grace must be finite and non-negative",
    )
    env = dict(
        os.environ,
        RAYON_NUM_THREADS="8",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        AWS_EC2_METADATA_DISABLED="true",
    )
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "JOPE_AWS_PROFILE",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        env.pop(key, None)
    output = Path(output)
    require(not output.exists(), "refusing to overwrite prior execution receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    if plan["mode"] == "qualification":
        ledger = Path(ledger)
        ledger.mkdir(parents=True, exist_ok=True)
        marker = ledger / (plan["pins"]["bundle_sha256"] + ".started.json")
        with marker.open("x") as stream:
            append_receipt(
                stream,
                {"plan_sha256": digest(plan), "status": "started_outcome_unknown"},
            )
    failures = 0
    with output.open("x") as stream:
        for dataset in plan["datasets"]:
            for seed in plan["seeds"]:
                policies = (
                    ["heuristic"]
                    if plan["mode"] == "heuristic_smoke"
                    else ["heuristic", "learned"]
                )
                if seed % 2:
                    policies.reverse()
                for guidance in policies:
                    request = {
                        "schema": "jope.q1-request/v1",
                        "mode": plan["mode"],
                        "plan_sha256": digest(plan),
                        "source_commit": plan["source_commit"],
                        "pins": plan["pins"],
                        "metric": "rmse",
                        "dataset": dataset,
                        "guidance": guidance,
                        "seed": seed,
                        "budget": plan["budget"],
                    }
                    if plan["mode"] == "heuristic_smoke":
                        request["payload"] = smoke_payload(dataset, data_root, seed)
                        request["actual_cut_sha256"] = digest(
                            request["payload"]["folds"]
                        )
                    request["request_sha256"] = digest(request)
                    identity = {
                        k: request[k]
                        for k in (
                            "mode",
                            "plan_sha256",
                            "source_commit",
                            "pins",
                            "dataset",
                            "guidance",
                            "seed",
                            "budget",
                            "request_sha256",
                        )
                    }
                    if "actual_cut_sha256" in request:
                        identity["actual_cut_sha256"] = request["actual_cut_sha256"]
                    append_receipt(
                        stream,
                        {**identity, "event": "started", "status": "outcome_unknown"},
                    )
                    start = time.monotonic()
                    try:
                        result = _invoke_backend(
                            [str(Path(backend).resolve())],
                            canonical(request),
                            env,
                            timeout,
                            backend_cleanup_grace,
                        )
                        require(result.returncode == 0, "backend process failed")
                        response = response_valid(decode(result.stdout), request)
                        row = {**identity, "event": "finished", "response": response}
                    except (ValueError, subprocess.TimeoutExpired, OSError) as error:
                        row = {
                            **identity,
                            "event": "finished",
                            "response": {
                                "status": "failed",
                                "loss": None,
                                "charged": None,
                                "completed": None,
                                "reason": type(error).__name__ + ": " + str(error),
                            },
                        }
                    row["wall_seconds"] = time.monotonic() - start
                    append_receipt(stream, row)
                    failures += row["response"]["status"] != "complete"
    if plan["mode"] == "qualification":
        with (ledger / (plan["pins"]["bundle_sha256"] + ".finished.json")).open(
            "x"
        ) as finished:
            append_receipt(
                finished,
                {
                    "plan_sha256": digest(plan),
                    "receipts_sha256": sha_file(output),
                    "status": "complete" if failures == 0 else "incomplete",
                    "failed_runs": failures,
                },
            )
    require(
        failures == 0, "incomplete execution panel; retained failed/unknown receipts"
    )


def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(values, seed=20261030, lower_probability=0.025):
    require(
        values and all(math.isfinite(v) for v in values), "finite observations required"
    )
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(10000)
    ]
    return {
        "median": statistics.median(values),
        "lower": quantile(medians, lower_probability),
        "upper": quantile(medians, 0.975),
        "resamples": 10000,
        "seed": seed,
    }


def analyze(plan, receipts):
    validate_plan(plan)
    require(
        plan["mode"] == "qualification", "smoke evidence cannot produce a Q1 decision"
    )
    expected = {
        (d["id"], seed, policy)
        for d in plan["datasets"]
        for seed in plan["seeds"]
        for policy in ("heuristic", "learned")
    }
    panel = {}
    started = set()
    start_hashes = {}
    for row in receipts:
        require(
            integer(row["seed"]) and row["seed"] in plan["seeds"],
            "invalid receipt seed",
        )
        key = (row["dataset"]["id"], row["seed"], row["guidance"])
        require(row["mode"] == "qualification", "non-qualification receipt")
        require(
            row["plan_sha256"] == digest(plan)
            and row["pins"] == plan["pins"]
            and row["source_commit"] == plan["source_commit"],
            "comparison identity mismatch",
        )
        require(key in expected, "unexpected dataset/seed/policy")
        dataset = next(d for d in plan["datasets"] if d["id"] == key[0])
        require(
            row["dataset"] == dataset and row["budget"] == plan["budget"],
            "dataset/cut/budget mismatch",
        )
        if row.get("event") == "started":
            require(
                key in expected and key not in started, "unexpected/duplicate start"
            )
            started.add(key)
            start_hashes[key] = row["request_sha256"]
            continue
        require(
            row.get("event") == "finished" and key in started, "missing start receipt"
        )
        require(
            type(row.get("wall_seconds")) in (int, float)
            and math.isfinite(row["wall_seconds"])
            and row["wall_seconds"] >= 0,
            "finished receipt requires finite nonnegative wall time",
        )
        require(
            row["request_sha256"] == start_hashes[key], "start/finish digest mismatch"
        )
        require(
            key in expected and key not in panel, "unexpected/duplicate paired result"
        )
        request = {
            "schema": "jope.q1-request/v1",
            "mode": "qualification",
            "plan_sha256": digest(plan),
            "source_commit": plan["source_commit"],
            "pins": plan["pins"],
            "metric": "rmse",
            "dataset": dataset,
            "guidance": key[2],
            "seed": key[1],
            "budget": plan["budget"],
        }
        request["request_sha256"] = digest(request)
        require(
            row["request_sha256"] == request["request_sha256"],
            "request digest mismatch",
        )
        response = response_valid(row["response"], request)
        require(
            response["status"] == "complete",
            "incomplete/failed pair; no substitute score",
        )
        require(
            response["charged"] == plan["budget"]
            and response["guidance"] == key[2]
            and response["request_sha256"] == row["request_sha256"]
            and response["metric"] == "rmse",
            "response budget/metric/guidance mismatch",
        )
        require(
            type(response["loss"]) in (float, int)
            and math.isfinite(response["loss"])
            and response["loss"] >= 0,
            "undefined paired result",
        )
        panel[key] = response["loss"]
    require(set(panel) == expected and started == expected, "incomplete paired panel")
    effects = {}
    for dataset in plan["datasets"]:
        heuristic_loss = statistics.mean(
            panel[(dataset["id"], seed, "heuristic")] for seed in plan["seeds"]
        )
        learned_loss = statistics.mean(
            panel[(dataset["id"], seed, "learned")] for seed in plan["seeds"]
        )
        effects[dataset["id"]] = (heuristic_loss - learned_loss) / max(
            heuristic_loss, 1e-12
        )
    primary = bootstrap(list(effects.values()))
    subgroups = {}
    for group in GROUPS:
        values = [effects[d["id"]] for d in plan["datasets"] if group in memberships(d)]
        minimum = plan.get("minimum_wide_families", 5) if group == "wide" else 5
        require(len(values) >= minimum, "insufficient subgroup")
        seed = int.from_bytes(
            hashlib.sha256((group + ":20261030").encode()).digest()[:8], "big"
        )
        subgroups[group] = {
            **bootstrap(values, seed, 0.05 / len(GROUPS)),
            "datasets": len(values),
        }
    if plan.get("minimum_wide_families") == 4:
        subgroups["wide"].update(
            report_label=R19_WIDE_LABEL, descriptive_only=True, verdict=None
        )
    passed = primary["lower"] > 0 and all(
        v["median"] >= -0.02 and v["lower"] >= -0.02
        for group, v in subgroups.items()
        if not (group == "wide" and plan.get("minimum_wide_families") == 4)
    )
    return {
        "schema": "jope.q1-analysis/v1",
        "plan_sha256": digest(plan),
        "task": "regression",
        "datasets": len(effects),
        "primary": primary,
        "subgroups": subgroups,
        "decision": "PASS" if passed else "FAIL",
        **({"amendment": "R19"} if plan.get("minimum_wide_families") == 4 else {}),
        "relative_regression_margin": 0.02,
    }


def make_smoke_plan(backend):
    catalog = load(HERE / "smoke-catalog.json")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    datasets = [
        {
            "id": r["id"],
            "task": "regression",
            "dataset_sha256": r["canonical_sha256"],
            "family": r["family_id"],
            "family_authoritative": True,
            "rows": r["rows"],
            "features": r["features"],
            "train_sha256": r["train_sha256"],
            "has_meta_test": False,
            "holdout_group": "development_fit",
            "cut_sha256": digest(
                {"train": r["train_sha256"], "rows": 128, "columns": 8, "cv": 2}
            ),
        }
        for r in catalog["datasets"]
    ]
    return {
        "schema": "jope.q1-plan/v1",
        "mode": "heuristic_smoke",
        "source_commit": source,
        "preregistration_sha256": sha_file(PREREGISTRATION),
        "backend_sha256": sha_file(backend),
        "budget": 4,
        "seeds": [0, 1],
        "datasets": datasets,
        "pins": {},
        "training_families": [],
        "timeout_seconds": 60,
        "ledger_root": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("smoke", "run", "approval-payload", "analyze")
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--allowed-signers", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    plan = make_smoke_plan(args.backend) if args.command == "smoke" else load(args.plan)
    validate_plan(plan)
    if args.command in ("smoke", "run"):
        require(
            args.backend is not None and args.out is not None, "backend/output required"
        )
        if args.command == "smoke":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            plan_path = args.out.with_suffix(".plan.json")
            with plan_path.open("x") as stream:
                stream.write(json.dumps(plan, indent=2) + "\n")
        execute(
            plan,
            args.backend,
            args.out,
            args.data_root,
            args.signature,
            args.allowed_signers,
            args.ledger,
        )
    elif args.command == "approval-payload":
        print(approval_payload(plan).decode(), end="")
    else:
        verify_owner(plan, args.signature, args.allowed_signers)
        receipts = [
            decode(line) for line in args.receipts.read_text().splitlines() if line
        ]
        result = analyze(plan, receipts)
        with args.out.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
