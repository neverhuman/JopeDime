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
import stat
import statistics
import subprocess
import time

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


def validate_plan(plan, preregistration=PREREGISTRATION):
    require(set(plan) == PLAN_KEYS, "unsupported/missing plan field")
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
        for group in GROUPS:
            require(
                sum(group in memberships(d) for d in plan["datasets"]) >= 5,
                "too few datasets in preregistered subgroup " + group,
            )
    else:
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


def execute(
    plan,
    backend,
    output,
    data_root=None,
    signature=None,
    allowed_signers=None,
    ledger=None,
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
                        result = subprocess.run(
                            [str(Path(backend).resolve())],
                            input=canonical(request),
                            capture_output=True,
                            timeout=plan["timeout_seconds"],
                            env=env,
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
        require(len(values) >= 5, "insufficient subgroup")
        seed = int.from_bytes(
            hashlib.sha256((group + ":20261030").encode()).digest()[:8], "big"
        )
        subgroups[group] = {
            **bootstrap(values, seed, 0.05 / len(GROUPS)),
            "datasets": len(values),
        }
    passed = primary["lower"] > 0 and all(
        v["median"] >= -0.02 and v["lower"] >= -0.02 for v in subgroups.values()
    )
    return {
        "schema": "jope.q1-analysis/v1",
        "plan_sha256": digest(plan),
        "task": "regression",
        "datasets": len(effects),
        "primary": primary,
        "subgroups": subgroups,
        "decision": "PASS" if passed else "FAIL",
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
