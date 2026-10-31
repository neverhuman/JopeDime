"""Metadata-only planning and strict result admission for JRB-01."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("regression", "binary")
ARMS = ("joint-balanced", "joint-natural", "reg-only", "bin-only")
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SCHEMA = "jope.joint-regbin/v1"
EVALUATION_PROTOCOL = "cached-pool-jrb01/v1"
DEFAULTS = dict(
    rounds=50,
    steps=2,
    contexts=32,
    tokens=10,
    eval_every=5,
    seeds=[20261029, 20261030, 20261031],
    bootstrap=1000,
    bootstrap_seed=136,
    wall_seconds=21600,
    output_bytes=10 * 1024**3,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_sha():
    root = Path(__file__).parent
    return digest(
        {
            name: file_sha(root / name)
            for name in (
                "protocol.py",
                "driver.py",
                "report.py",
                "preregistration.json",
            )
        }
    )


PROTECTED_TOKENS = ("holdout", "sealed", "meta_test", "outer_test")


def _is_protected(path):
    text = str(path).lower().replace("-", "_")
    return any(token in text for token in PROTECTED_TOKENS)


def relative_payload(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("payload path must remain inside the pinned packet")
    if _is_protected(path):
        raise ValueError("protected payload refused before open")
    return path


def safe_path(root, relative):
    path = relative_payload(relative)
    packet = Path(root).resolve()
    if _is_protected(root) or _is_protected(packet):
        raise ValueError("protected packet root refused before open")
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(packet):
        raise ValueError("payload symlink escapes packet")
    # A symlink inside the packet may still point at a protected file; check
    # the resolved in-packet components too, before anything opens or hashes it.
    if _is_protected(result.relative_to(packet)):
        raise ValueError("protected payload refused before open (resolved symlink)")
    return result


def check_sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected full lowercase SHA256")


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError("preregistration keys differ")
    for key, value in config.items():
        if key != "seeds" and (type(value) is not int or value < 1):
            raise ValueError("positive integer budget required")
    if (
        config["rounds"] < 50
        or not 1 <= config["eval_every"] <= 10
        or config["rounds"] % config["eval_every"]
        or config["tokens"] != 10
        or config["contexts"] % 2
    ):
        raise ValueError(
            "require >=50 rounds, <=10-round intervals and even fixed-token batches"
        )
    seeds = config["seeds"]
    if (
        len(seeds) != 3
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != 3
    ):
        raise ValueError("exactly three distinct paired integer seeds required")
    if config != DEFAULTS:
        raise ValueError(
            "preregistered campaign budget differs; do not silently reset or enlarge it"
        )


def validate_manifest(manifest):
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("eligibility_verified") is not True
    ):
        raise ValueError("explicit eligible real-S3 metadata required")
    check_sha(manifest["catalog_sha256"])
    check_sha(manifest["learner_sha256"])
    check_sha(manifest["initial_checkpoint_sha256"])
    kept, excluded = [], 0
    ids, datasets, families = set(), {}, defaultdict(set)
    for row in manifest["contexts"]:
        required = {
            "id",
            "dataset",
            "family",
            "task",
            "role",
            "source_split",
            "origin",
            "has_meta_test",
            "payload",
            "payload_sha256",
        }
        if set(row) != required:
            raise ValueError(
                "context metadata fields differ; raw rows are not accepted"
            )
        if row["role"] not in ("train", "val", "panel") or row["source_split"] not in (
            "train",
            "inner",
        ):
            raise ValueError("outside training-row closure")
        if row["has_meta_test"] is not False or row["origin"] != "real":
            raise ValueError("only unsealed real-S3 contexts accepted")
        relative_payload(
            row["payload"]
        )  # Never resolve metadata against the caller's cwd.
        check_sha(row["payload_sha256"])
        for key in ("id", "dataset", "family"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("unresolved dataset/context/family")
        if row["id"] in ids:
            raise ValueError("duplicate context identity")
        ids.add(row["id"])
        previous = datasets.setdefault(row["dataset"], (row["family"], row["task"]))
        if previous != (row["family"], row["task"]):
            raise ValueError("dataset task/family identity changed")
        families[row["family"]].add(row["role"])
        if row["task"] == "multiclass":
            excluded += 1
            continue
        if row["task"] not in TASKS:
            raise ValueError("unknown task")
        kept.append(dict(row))
    if any("train" in roles and "val" in roles for roles in families.values()):
        raise ValueError("family overlap across train/validation")
    for role in ("train", "val", "panel"):
        for task in TASKS:
            if not any(row["role"] == role and row["task"] == task for row in kept):
                raise ValueError(
                    f"missing {role} {task} inputs; regression-only is not joint"
                )
    return kept, excluded


def family_folds(rows, seed):
    families = defaultdict(Counter)
    for row in rows:
        families[row["family"]][row["task"]] += 1
    totals = Counter(row["task"] for row in rows)
    if any(sum(counts[task] > 0 for counts in families.values()) < 5 for task in TASKS):
        raise ValueError("five independent training families per task required")
    ordered = sorted(families, key=lambda f: digest([seed, f]))
    ordered.sort(key=lambda f: -max(families[f][task] / totals[task] for task in TASKS))
    loads = [Counter() for _ in range(5)]
    assignment = {}
    for family in ordered:
        fold = min(
            range(5),
            key=lambda i: (
                sum(
                    (loads[i][task] + families[family][task]) ** 2 / totals[task] ** 2
                    for task in TASKS
                ),
                sum(loads[i].values()),
                i,
            ),
        )
        assignment[family] = fold
        loads[fold].update(families[family])
    if any(not load[task] for load in loads for task in TASKS):
        raise ValueError(
            "fixed family stratification lacks a task; do not tune folds on outcomes"
        )
    return assignment


def make_plan(manifest, config=None):
    config = json.loads(json.dumps(DEFAULTS if config is None else config))
    validate_config(config)
    rows, excluded = validate_manifest(manifest)
    train = [row for row in rows if row["role"] == "train"]
    folds = family_folds(train, config["bootstrap_seed"])
    jobs = []
    for seed in config["seeds"]:
        for fold in [0, 1, 2, 3, 4, "full"]:
            fit = [
                row for row in train if fold == "full" or folds[row["family"]] != fold
            ]
            evaluation = [
                row
                for row in rows
                if (row["role"] == "val" and fold == "full")
                or (
                    row["role"] == "train"
                    and fold != "full"
                    and folds[row["family"]] == fold
                )
                or (row["role"] == "panel" and folds.get(row["family"], "full") == fold)
            ]
            for arm in ARMS:
                selected = [
                    row
                    for row in fit
                    if arm.startswith("joint")
                    or row["task"] == ("regression" if arm == "reg-only" else "binary")
                ]
                if not selected:
                    raise ValueError("empty arm training closure")
                job = dict(
                    arm=arm, seed=seed, fold=fold, train=selected, evaluation=evaluation
                )
                job["id"] = digest(job)[:20]
                jobs.append(job)
    plan = dict(
        schema=SCHEMA,
        config=config,
        manifest_sha256=digest(manifest),
        implementation_sha256=implementation_sha(),
        learner_sha256=manifest["learner_sha256"],
        initial_checkpoint_sha256=manifest["initial_checkpoint_sha256"],
        catalog_sha256=manifest["catalog_sha256"],
        excluded_multiclass=excluded,
        contexts=rows,
        folds=folds,
        jobs=jobs,
        protocol="cached-pool-selection; no raw outer rows; no live promotion",
        evaluation_protocol=EVALUATION_PROTOCOL,
        dope_sha256=DOPE,
    )
    plan["sha256"] = digest(plan)
    return plan


def verify_plan(plan):
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("plan changed after preregistration")
    validate_config(plan["config"])
    if plan["evaluation_protocol"] != EVALUATION_PROTOCOL:
        raise ValueError("evaluation protocol differs from preregistration")
    if plan["implementation_sha256"] != implementation_sha():
        raise ValueError("study implementation changed after preregistration")
    # Recheck closure semantics even if someone edits and rehashes a plan file.
    reconstructed = make_plan(
        dict(
            schema=SCHEMA,
            eligibility_verified=True,
            contexts=plan["contexts"],
            catalog_sha256=plan["catalog_sha256"],
            learner_sha256=plan["learner_sha256"],
            initial_checkpoint_sha256=plan["initial_checkpoint_sha256"],
        ),
        plan["config"],
    )
    if (plan["jobs"], plan["folds"]) != (reconstructed["jobs"], reconstructed["folds"]):
        raise ValueError("job grid differs from fixed family/input closure")


def verify_catalog(manifest, catalog):
    """The supplied file is the curated real-S3 catalog, not a payload or a new roster."""
    if file_sha(catalog) != manifest["catalog_sha256"]:
        raise ValueError("catalog changed")
    records = {
        row["id"]: row for row in json.loads(Path(catalog).read_text())["datasets"]
    }
    for row in manifest["contexts"]:
        actual = records.get(row["dataset"])
        if (
            actual is None
            or not actual.get("family_key")
            or (actual["family_key"], actual["task"]) != (row["family"], row["task"])
        ):
            raise ValueError(
                "unresolved/unreviewed or mismatched authoritative task/family"
            )


def batches(job, config):
    """Seeded cyclic family blocks; task quotas never depend on validation outcomes."""
    rng = random.Random(job["seed"])
    queues, offsets = {}, Counter()
    for task in TASKS:
        blocks = defaultdict(list)
        for row in job["train"]:
            if row["task"] == task:
                blocks[row["family"]].append(row["id"])
        names = sorted(blocks)
        rng.shuffle(names)
        queues[task] = []
        for name in names:
            values = sorted(blocks[name])
            rng.shuffle(values)
            queues[task].extend(values)
    total = sum(map(len, queues.values()))
    reg_count = len(queues["regression"])
    size = config["contexts"]
    for step in range(config["rounds"] * config["steps"]):
        if job["arm"] == "joint-balanced":
            nreg = size // 2
        else:
            nreg = (
                step + 1
            ) * size * reg_count // total - step * size * reg_count // total
        order = list(TASKS)
        rng.shuffle(order)
        batch = []
        for task in order:
            count = nreg if task == "regression" else size - nreg
            for _ in range(count):
                batch.append(queues[task][offsets[task] % len(queues[task])])
                offsets[task] += 1
        yield batch


def expected_counts(config):
    steps = config["rounds"] * config["steps"]
    return dict(
        optimizer_steps=steps,
        context_presentations=steps * config["contexts"],
        conditioning_token_slots=steps * config["contexts"] * config["tokens"],
    )


def admit_result(plan, job, receipt):
    config = plan["config"]
    if (
        receipt["plan_sha256"] != plan["sha256"]
        or receipt["job_id"] != job["id"]
        or receipt["learner_sha256"] != plan["learner_sha256"]
        or receipt["initial_checkpoint_sha256"] != plan["initial_checkpoint_sha256"]
    ):
        raise ValueError("result source identity mismatch")
    if receipt["actual_budget"] != expected_counts(config):
        raise ValueError("unequal actual optimizer/context/token budget")
    if receipt["dope_before"] != DOPE or receipt["dope_after"] != DOPE:
        raise ValueError("DOPE changed")
    for key in ("parameters_before", "parameters_after"):
        check_sha(receipt[key])
    if receipt["parameters_before"] == receipt["parameters_after"]:
        raise ValueError("no changed parameters; not learning evidence")
    traces = receipt["round_receipts"]
    if [trace["round"] for trace in traces] != list(range(1, config["rounds"] + 1)):
        raise ValueError("missing actual training rounds")
    previous = receipt["parameters_before"]
    orders = list(batches(job, config))
    for trace in traces:
        end = trace["round"] * config["steps"]
        expected_order = digest(orders[end - config["steps"] : end])
        if (
            trace["batch_sha256"] != expected_order
            or trace["optimizer_steps"] != config["steps"]
            or trace["dope_sha256"] != DOPE
            or trace["parameters_before"] != previous
        ):
            raise ValueError("round budget/frozen state/batch continuity mismatch")
        check_sha(trace["parameters_after"])
        check_sha(trace["checkpoint_sha256"])
        if trace["parameters_after"] == previous:
            raise ValueError("round has no changed parameters")
        previous = trace["parameters_after"]
    if previous != receipt["parameters_after"]:
        raise ValueError("final parameters disagree with actual rounds")
    rounds = range(config["eval_every"], config["rounds"] + 1, config["eval_every"])
    expected = {(r, row["id"]) for r in rounds for row in job["evaluation"]}
    by_id = {row["id"]: row for row in job["evaluation"]}
    seen = set()
    for row in receipt["scores"]:
        key = row["round"], row["id"]
        if key not in expected or key in seen:
            raise ValueError("unexpected/duplicate evaluation result")
        seen.add(key)
        context = by_id[row["id"]]
        if (
            row.get("evaluation_sha256") != context["payload_sha256"]
            or row.get("evaluation_protocol") != EVALUATION_PROTOCOL
        ):
            raise ValueError("evaluation input/protocol differs from frozen context")
        metric = "rmse" if context["task"] == "regression" else "auc"
        if row["metric"] != metric or row["status"] != "ok":
            raise ValueError("failed or task-incomparable evaluation")
        for name in ("value", "baseline", "curve"):
            if type(row[name]) not in (int, float) or not math.isfinite(row[name]):
                raise ValueError("nonfinite/unscored result")
        if context["task"] == "regression" and (
            row["value"] <= 0 or row["baseline"] <= 0
        ):
            raise ValueError("positive RMSE and ridge RMSE required")
        if context["task"] == "regression" and not math.isclose(
            row["curve"], math.log(row["value"] / row["baseline"]), abs_tol=1e-10
        ):
            raise ValueError("regression curve must be log RMSE/ridge ratio")
        if context["task"] == "binary" and (
            not 0 <= row["value"] <= 1
            or not 0 <= row["baseline"] <= 1
            or row["curve"] < 0
        ):
            raise ValueError("binary AUC/log-loss outside domain")
        check_sha(row["model_sha256"])
        if row["model_sha256"] != traces[row["round"] - 1]["checkpoint_sha256"]:
            raise ValueError("score checkpoint differs from its actual training round")
    if seen != expected:
        raise ValueError("missing evaluation results; no passing subset")
    return receipt
