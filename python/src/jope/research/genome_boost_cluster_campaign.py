#!/usr/bin/env python3
"""Genome-boost cluster planning, shard execution, and controller finalize.

Subcommands:

* ``plan`` — shard the dataset list (50 datasets/shard), stage the release
  binary and worker scripts into the campaign root, write per-shard combined
  feature manifests, and emit SCQ TOML job specs into
  ``<campaign-root>/scq-specs/``:
  - feature shards: 0 GPUs, 8 CPUs, 2 h, non-preemptible, and admitted only
    where the staged campaign binary is available;
  - DL shards (per shard x backend x arm): 4 CPUs, 8 h;
    ``tabpfn*`` backends get ``gpus_per_task = 1``.
  ``--submit`` exists but DEFAULTS TO DRY-RUN: without it the script only
  prints the ``scq check`` / ``scq submit`` commands.
* ``run-shard`` — executed on a node by the DL shard jobs: loops the shard's
  datasets through ``genome_boost_execution.py`` (fit(train)->predict(test),
  no CV).
* ``finalize`` — controller-only scoring: verifies every worker receipt hash,
  reopens the canonical registry ``test.csv`` target column, computes
  ``regression_metrics`` (imported from ``golden46_execution``), and writes
  per-dataset/backend ``final-report.v1.json`` with baseline vs boosted RMSE
  and the test-adaptive ``leader_rmse`` comparison when available. Every
  report is permanently ``test_adaptive: true`` / ``benchmark_valid: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_ROOT))

CAMPAIGN_ID = "genome-boost-20260725-v1"
DEFAULT_CAMPAIGN_ROOT = Path("/home/ubuntu/remote_super/quant/campaigns") / CAMPAIGN_ID
REGISTRY_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
JOPE_ROOT = Path("/home/ubuntu/jope-dime")
PANEL_FILE = JOPE_ROOT / "manifests" / "panel.txt"
TEST_ADAPTIVE_RUN = JOPE_ROOT / "runs" / "test-adaptive-golden42-v1"
RELEASE_BINARY = JOPE_ROOT / "target" / "release" / "jope-dime-regressor"
VENV_PYTHON = Path("/home/ubuntu/remote_super/quant/.venv-model-queue/bin/python")
SCQ_BINARY = "/home/ubuntu/smart_cluster/target/release/scq"
SCQ_PROJECT = "jope"
DATASETS_PER_SHARD = 50
DEFAULT_BACKENDS = ("mlp", "tabm", "tabpfn_v3_regressor_default")
ARMS = ("baseline", "boosted")
FINAL_REPORT_SCHEMA = "jope-dime.genome-boost-final-report.v1"
SHARD_SUMMARY_SCHEMA = "jope-dime.genome-boost-shard-summary.v1"
MANIFEST_SCHEMA = "jope-dime.genome-boost-manifest.v1"
THREAD_ENV = (
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def load_datasets(argument: str) -> list[str]:
    source = PANEL_FILE if argument == "golden42" else Path(argument)
    if source.suffix == ".json":
        payload = read_json(source)
        hashes = payload.get("datasets", [])
    else:
        hashes = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not hashes:
        raise SystemExit(f"no dataset hashes in {source}")
    return hashes


def shard_datasets(datasets: list[str]) -> list[list[str]]:
    return [
        datasets[index : index + DATASETS_PER_SHARD]
        for index in range(0, len(datasets), DATASETS_PER_SHARD)
    ]


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def render_spec(
    job_id: str,
    args: list[str],
    workdir: Path,
    *,
    gpus_per_task: int,
    cpus_reserved: int,
    ram_gb: int,
    max_runtime_hours: int,
    max_restarts: int,
    preemptible: bool,
    capability_path: Path,
) -> str:
    return (
        "[job]\n"
        f"id = {json.dumps(job_id)}\n"
        f"project = {json.dumps(SCQ_PROJECT)}\n"
        'command = "/usr/bin/env"\n'
        f"args = {toml_string_array(args)}\n"
        f"workdir = {json.dumps(str(workdir))}\n"
        'tier = "normal"\n'
        "\n"
        "[resource]\n"
        f"gpus_per_task = {gpus_per_task}\n"
        f"cpus_reserved = {cpus_reserved}\n"
        "omp_threads = 1\n"
        f"ram_gb = {ram_gb}\n"
        "gang_size = 1\n"
        'placement = "single_node"\n'
        "\n"
        "[[dependencies.smoke]]\n"
        'name = "campaign runtime capability available"\n'
        'command = "/usr/bin/test"\n'
        f"args = [\"-e\", {json.dumps(str(capability_path))}]\n"
        "timeout_secs = 5\n"
        "\n"
        "[preempt]\n"
        f"preemptible = {str(preemptible).lower()}\n"
        'signal = "SIGTERM"\n'
        "grace_seconds = 60\n"
        "\n"
        "[limits]\n"
        f"max_runtime_hours = {max_runtime_hours}\n"
        f"max_restarts = {max_restarts}\n"
    )


def combine_manifests(campaign_root: Path, shard: list[str], shard_index: int) -> Path:
    """One Rust-consumable manifest covering every dataset in a feature shard."""
    combined: dict = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": None,
        "test_adaptive": True,
        "benchmark_valid": False,
        "created_at": utc_now(),
        "datasets": [],
    }
    for dataset in shard:
        manifest = read_json(campaign_root / "manifests" / f"{dataset}.manifest.v1.json")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise SystemExit(f"{dataset}: unexpected manifest schema")
        if combined["mode"] is None:
            combined["mode"] = manifest.get("mode")
        elif manifest.get("mode") != combined["mode"]:
            raise SystemExit(f"{dataset}: mixed selection modes inside one shard")
        combined["datasets"].extend(manifest["datasets"])
    path = campaign_root / "scq-manifests" / f"feature-shard-{shard_index:03d}.manifest.v1.json"
    atomic_write_json(path, combined)
    return path


def stage_code(campaign_root: Path) -> dict[str, str]:
    """Copy the exact worker code the shard jobs will run into the campaign
    root, so the node-sync plan distributes pinned bytes."""
    code_dir = campaign_root / "code"
    staged = {}
    for name in (
        "genome_boost_execution.py",
        "genome_boost_cluster_campaign.py",
        "golden46_execution.py",
    ):
        source = SCRIPTS_ROOT / name
        atomic_write_bytes(code_dir / name, source.read_bytes())
        staged[name] = sha256_file(code_dir / name)
    return staged


def stage_binary(campaign_root: Path, binary: Path) -> tuple[Path, str]:
    binary_sha256 = sha256_file(binary)
    bundle_dir = campaign_root / "bundles" / binary_sha256
    target = bundle_dir / "jope-dime-regressor"
    if not target.is_file() or sha256_file(target) != binary_sha256:
        atomic_write_bytes(target, binary.read_bytes())
        target.chmod(0o755)
    return target, binary_sha256


def plan(arguments: argparse.Namespace) -> None:
    campaign_root: Path = arguments.campaign_root
    datasets = load_datasets(arguments.datasets)
    backends = [backend.strip() for backend in arguments.backends.split(",") if backend.strip()]
    shards = shard_datasets(datasets)
    spec_dir = arguments.spec_dir or campaign_root / "scq-specs"
    shard_dir = arguments.shard_dir or campaign_root / "scq-manifests"
    worker_root = arguments.worker_root or campaign_root / "worker-views"
    staged_code = stage_code(campaign_root)
    binary_sha256 = None
    specs: list[Path] = []

    if not arguments.dl_only:
        bundled_binary, binary_sha256 = stage_binary(campaign_root, arguments.binary)
        for shard_index, shard in enumerate(shards):
            shard_manifest = combine_manifests(campaign_root, shard, shard_index)
            digest = hashlib.sha256(
                f"{CAMPAIGN_ID}\0feature\0{shard_index}\0{binary_sha256}".encode()
            ).hexdigest()[:12]
            job_id = f"jope-gb-feat-{shard_index:03d}-{digest}"
            args = [
                *THREAD_ENV,
                "RAYON_NUM_THREADS=1",
                "RUST_NUM_THREADS=1",
                str(bundled_binary),
                "genome-boost-features",
                "--manifest",
                str(shard_manifest),
                "--output-dir",
                str(campaign_root / "features"),
            ]
            spec = render_spec(
                job_id,
                args,
                bundled_binary.parent,
                gpus_per_task=0,
                cpus_reserved=8,
                ram_gb=0,
                max_runtime_hours=2,
                max_restarts=0,
                preemptible=False,
                capability_path=bundled_binary,
            )
            spec_path = spec_dir / f"{job_id}.toml"
            atomic_write_bytes(spec_path, spec.encode())
            specs.append(spec_path)

    for shard_index, shard in enumerate(shards):
        shard_file = shard_dir / f"dl-shard-{shard_index:03d}.datasets.json"
        atomic_write_json(
            shard_file,
            {"campaign_id": CAMPAIGN_ID, "shard": shard_index, "datasets": shard},
        )
        for backend in backends:
            for arm in ARMS:
                digest = hashlib.sha256(
                    f"{CAMPAIGN_ID}\0dl\0{shard_index}\0{backend}\0{arm}".encode()
                ).hexdigest()[:12]
                job_id = f"jope-gb-dl-{backend[:24]}-{arm}-{shard_index:03d}-{digest}"
                args = [
                    *THREAD_ENV,
                    str(VENV_PYTHON),
                    str(campaign_root / "code" / "genome_boost_cluster_campaign.py"),
                    "run-shard",
                    "--campaign-root",
                    str(campaign_root),
                    "--worker-root",
                    str(worker_root),
                    "--shard-file",
                    str(shard_file),
                    "--backend",
                    backend,
                    "--arm",
                    arm,
                ]
                spec = render_spec(
                    job_id,
                    args,
                    campaign_root / "code",
                    gpus_per_task=1 if backend.startswith("tabpfn") else 0,
                    cpus_reserved=4,
                    ram_gb=32 if backend.startswith("tabpfn") else 0,
                    max_runtime_hours=8,
                    max_restarts=1 if backend.startswith("tabpfn") else 0,
                    preemptible=backend.startswith("tabpfn"),
                    capability_path=VENV_PYTHON,
                )
                spec_path = spec_dir / f"{job_id}.toml"
                atomic_write_bytes(spec_path, spec.encode())
                specs.append(spec_path)

    plan_receipt = {
        "schema": "jope-dime.genome-boost-cluster-plan.v1",
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "created_at": utc_now(),
        "datasets": datasets,
        "shards": [len(shard) for shard in shards],
        "backends": backends,
        "arms": list(ARMS),
        "release_binary_sha256": binary_sha256,
        "staged_code_sha256": staged_code,
        "feature_capability": str(bundled_binary),
        "dl_capability": str(VENV_PYTHON),
        "worker_root": str(worker_root),
        "specs": [str(path) for path in specs],
        "submitted": [],
    }
    plan_path = spec_dir / arguments.plan_name
    atomic_write_json(plan_path, plan_receipt)

    if arguments.submit:
        for spec_path in specs:
            with spec_path.open("rb") as handle:
                spec_payload = tomllib.load(handle)
            if int(spec_payload["resource"].get("gpus_per_task", 0)):
                continue
            print(f"submitting {spec_path}")
            subprocess.run([SCQ_BINARY, "submit", str(spec_path)], check=True)
            plan_receipt["submitted"].append(str(spec_path))
        atomic_write_json(plan_path, plan_receipt)
    else:
        print(f"DRY RUN: {len(specs)} SCQ specs written under {spec_dir}")
        print("# validate without enqueueing:")
        for spec_path in specs:
            print(f"scq check {spec_path}")
        print("# submit (NOT executed by this dry run):")
        for spec_path in specs:
            print(f"scq submit {spec_path}")


def scq_jobs() -> dict[str, dict]:
    completed = subprocess.run(
        [SCQ_BINARY, "--json", "ls"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    jobs = payload.get("Jobs", {}).get("jobs", payload.get("jobs", []))
    return {job["id"]: job for job in jobs}


def successful_shard_summary(spec: dict) -> bool:
    """Validate the durable completion artifact addressed by a GPU spec."""
    args = list(spec.get("job", {}).get("args", []))
    if "run-shard" not in args:
        return False

    def value_after(flag: str) -> str | None:
        try:
            return str(args[args.index(flag) + 1])
        except (ValueError, IndexError):
            return None

    campaign_root = value_after("--campaign-root")
    shard_file = value_after("--shard-file")
    backend = value_after("--backend")
    arm = value_after("--arm")
    if not all((campaign_root, shard_file, backend, arm)):
        return False
    try:
        shard = read_json(Path(shard_file))
        shard_index = int(shard["shard"])
        datasets = list(shard["datasets"])
        summary_path = (
            Path(campaign_root)
            / "shard-summaries"
            / f"shard-{shard_index:03d}-{backend}-{arm}.v1.json"
        )
        summary = read_json(summary_path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    results = summary.get("results")
    return (
        summary.get("schema") == SHARD_SUMMARY_SCHEMA
        and summary.get("campaign_id") == CAMPAIGN_ID
        and summary.get("backend") == backend
        and summary.get("arm") == arm
        and int(summary.get("datasets", -1)) == len(datasets)
        and int(summary.get("failures", -1)) == 0
        and isinstance(results, list)
        and len(results) == len(datasets)
        and all(int(result.get("return_code", -1)) == 0 for result in results)
    )


def submit_next(arguments: argparse.Namespace) -> None:
    plan_receipt = read_json(arguments.plan)
    specs = [Path(path) for path in plan_receipt["specs"]]
    jobs = scq_jobs()
    state_path = arguments.state or arguments.plan.parent / "lane-state.v1.json"
    try:
        prior_status = read_json(state_path)
    except (OSError, ValueError):
        prior_status = {}
    completed_ids = set(prior_status.get("completed_ids", []))
    completed = 0
    failed: list[dict] = []
    active: list[dict] = []
    redundant_cancelled: list[str] = []
    next_spec = None
    for spec_path in specs:
        with spec_path.open("rb") as handle:
            spec = tomllib.load(handle)
        if int(spec["resource"].get("gpus_per_task", 0)) != 1:
            continue
        job_id = spec["job"]["id"]
        job = jobs.get(job_id)
        if successful_shard_summary(spec):
            completed_ids.add(job_id)
        if job_id in completed_ids:
            if job and job.get("state") in {"queued", "starting"}:
                subprocess.run([SCQ_BINARY, "cancel", job_id], check=True)
                redundant_cancelled.append(job_id)
                job = None
            elif job and job.get("state") == "running":
                active.append(job)
            completed += 1
            continue
        if job is None:
            if next_spec is None:
                next_spec = spec_path
            continue
        state = job.get("state")
        if state == "completed":
            failed.append(
                {
                    **job,
                    "error_reason": "completed without a valid shard summary",
                }
            )
        elif state in {"failed", "cancelled"}:
            failed.append(job)
        else:
            active.append(job)

    status = {
        "schema": "jope-dime.tabpfn-serial-lane.v1",
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "updated_at": utc_now(),
        "plan": str(arguments.plan),
        "completed": completed,
        "completed_ids": sorted(completed_ids),
        "active": [
            {"id": job["id"], "state": job.get("state"), "attempt": job.get("attempt")}
            for job in active
        ],
        "failed": [
            {
                "id": job["id"],
                "state": job.get("state"),
                "attempt": job.get("attempt"),
                "error_reason": job.get("error_reason"),
            }
            for job in failed
        ],
        "next_spec": str(next_spec) if next_spec else None,
        "submitted": None,
        "redundant_cancelled": redundant_cancelled,
    }
    if failed:
        atomic_write_json(state_path, status)
        raise SystemExit(
            f"TabPFN lane halted after terminal failure: {failed[0]['id']}"
        )
    if len(active) > 1:
        atomic_write_json(state_path, status)
        raise SystemExit(f"TabPFN serialization violated: {len(active)} active jobs")
    if active:
        atomic_write_json(state_path, status)
        print(f"TabPFN lane occupied by {active[0]['id']} ({active[0].get('state')})")
        return
    if next_spec is None:
        atomic_write_json(state_path, status)
        print(f"TabPFN lane complete: {completed} jobs")
        return
    subprocess.run([SCQ_BINARY, "check", str(next_spec)], check=True)
    subprocess.run([SCQ_BINARY, "submit", str(next_spec)], check=True)
    with next_spec.open("rb") as handle:
        submitted_id = tomllib.load(handle)["job"]["id"]
    status["submitted"] = submitted_id
    status["next_spec"] = None
    atomic_write_json(state_path, status)
    print(f"submitted {submitted_id}")


def run_shard(arguments: argparse.Namespace) -> None:
    campaign_root: Path = arguments.campaign_root
    shard = read_json(arguments.shard_file)
    if shard.get("campaign_id") != CAMPAIGN_ID:
        raise SystemExit("shard file campaign identity mismatch")
    datasets = shard["datasets"]
    worker_root = arguments.worker_root or campaign_root / "worker-views"
    results = []
    failures = 0
    for dataset in datasets:
        output = campaign_root / "predictions" / dataset / arguments.backend / arguments.arm
        receipt_path = output / "worker-receipt.json"
        prediction_path = output / "predictions.npy"
        if receipt_path.is_file() and prediction_path.is_file():
            receipt = read_json(receipt_path)
            if (
                receipt.get("dataset_hash") == dataset
                and receipt.get("backend") == arguments.backend
                and receipt.get("arm") == arguments.arm
                and receipt.get("predictions", {}).get("sha256")
                == sha256_file(prediction_path)
            ):
                results.append(
                    {"dataset": dataset, "return_code": 0, "skipped_complete": True}
                )
                print(
                    f"{dataset} {arguments.backend} {arguments.arm}: "
                    "skipped complete receipt"
                )
                continue
        command = [
            sys.executable,
            str(campaign_root / "code" / "genome_boost_execution.py"),
            "--worker-root",
            str(worker_root),
            "--dataset",
            dataset,
            "--backend",
            arguments.backend,
            "--arm",
            arguments.arm,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        record = {
            "dataset": dataset,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode != 0:
            failures += 1
        results.append(record)
        print(f"{dataset} {arguments.backend} {arguments.arm}: rc={completed.returncode}")
    summary = {
        "schema": SHARD_SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "shard": shard.get("shard"),
        "backend": arguments.backend,
        "arm": arguments.arm,
        "datasets": len(datasets),
        "failures": failures,
        "results": results,
        "finished_at": utc_now(),
    }
    atomic_write_json(
        campaign_root
        / "shard-summaries"
        / f"shard-{shard.get('shard'):03d}-{arguments.backend}-{arguments.arm}.v1.json",
        summary,
    )
    if failures:
        raise SystemExit(f"{failures}/{len(datasets)} datasets failed in this shard")


def aggregate_reports(arguments: argparse.Namespace) -> None:
    campaign_root: Path = arguments.campaign_root
    datasets = load_datasets(arguments.datasets)
    backends = [backend.strip() for backend in arguments.backends.split(",") if backend.strip()]
    by_backend = {}
    report_hashes = {}
    missing = []
    for backend in backends:
        reports = []
        for dataset in datasets:
            path = campaign_root / "reports" / dataset / backend / "final-report.v1.json"
            if not path.is_file():
                missing.append(f"{dataset}/{backend}")
                continue
            report = read_json(path)
            if (
                report.get("dataset_hash") != dataset
                or report.get("backend") != backend
                or report.get("test_adaptive") is not True
                or report.get("benchmark_valid") is not False
            ):
                raise SystemExit(f"{path}: report identity or labels are invalid")
            reports.append(report)
            report_hashes[f"{dataset}/{backend}"] = sha256_file(path)
        deltas = [float(report["delta"]) for report in reports]
        delta_pcts = [float(report["delta_pct"]) for report in reports if report["delta_pct"] is not None]
        wins = sum(bool(report["boosted_wins"]) for report in reports)
        by_backend[backend] = {
            "datasets": len(reports),
            "boosted_wins": wins,
            "boosted_win_rate": wins / len(reports) if reports else None,
            "median_rmse_delta": statistics.median(deltas) if deltas else None,
            "median_delta_pct": statistics.median(delta_pcts) if delta_pcts else None,
        }
    if missing:
        raise SystemExit(f"aggregate incomplete: {len(missing)} reports missing")
    payload = {
        "schema": "jope-dime.genome-boost-aggregate-report.v1",
        "campaign_id": CAMPAIGN_ID,
        "test_adaptive": True,
        "benchmark_valid": False,
        "datasets": len(datasets),
        "backends": by_backend,
        "report_sha256": report_hashes,
        "created_at": utc_now(),
    }
    output = arguments.output or campaign_root / "aggregate" / "golden42-final-report.v1.json"
    atomic_write_json(output, payload)
    print(f"wrote complete {len(datasets)}-dataset aggregate to {output}")


def load_predictions(directory: Path) -> tuple["object", dict]:
    import numpy as np

    receipt = read_json(directory / "worker-receipt.json")
    prediction_path = directory / "predictions.npy"
    if sha256_file(prediction_path) != receipt["predictions"]["sha256"]:
        raise SystemExit(f"{prediction_path}: byte hash differs from worker receipt")
    values = np.load(prediction_path, allow_pickle=False).astype(np.float64, copy=False)
    from golden46_execution import sha256_array

    if sha256_array(values) != receipt["predictions"]["array_sha256"]:
        raise SystemExit(f"{prediction_path}: array hash differs from worker receipt")
    return values, receipt


def verify_receipt_inputs(receipt: dict, campaign_root: Path, dataset: str) -> None:
    view = campaign_root / "worker-views" / dataset
    for name, expected in receipt["inputs_sha256"].items():
        if name in ("model_zoo.py", "execution_source"):
            continue
        if sha256_file(view / name) != expected:
            raise SystemExit(f"{dataset}: worker input {name} no longer matches its receipt")


def finalize(arguments: argparse.Namespace) -> None:
    import numpy as np

    from golden46_execution import regression_metrics

    campaign_root: Path = arguments.campaign_root
    datasets = load_datasets(arguments.datasets)
    backends = [backend.strip() for backend in arguments.backends.split(",") if backend.strip()]
    written = 0
    skipped = []
    for dataset in datasets:
        registry_test = REGISTRY_ROOT / dataset / "test.csv"
        target = None
        feature_receipt_path = (
            campaign_root / "features" / dataset / "genome-features-receipt.v1.json"
        )
        for backend in backends:
            base_dir = campaign_root / "predictions" / dataset / backend
            baseline_dir = base_dir / "baseline"
            boosted_dir = base_dir / "boosted"
            if not (baseline_dir / "predictions.npy").is_file() or not (
                boosted_dir / "predictions.npy"
            ).is_file():
                skipped.append(f"{dataset}/{backend}")
                continue
            if target is None:
                rows = [
                    line.split(",")
                    for line in registry_test.read_text().splitlines()
                    if line.strip()
                ]
                target = np.asarray([float(row[-1]) for row in rows], dtype=np.float64)
            baseline_values, baseline_receipt = load_predictions(baseline_dir)
            boosted_values, boosted_receipt = load_predictions(boosted_dir)
            for receipt in (baseline_receipt, boosted_receipt):
                if receipt.get("dataset_hash") != dataset or receipt.get("backend") != backend:
                    raise SystemExit(f"{dataset}/{backend}: worker receipt identity mismatch")
                if receipt.get("benchmark_valid") is not False:
                    raise SystemExit(f"{dataset}/{backend}: worker receipt mislabeled")
                verify_receipt_inputs(receipt, campaign_root, dataset)
            if len(baseline_values) != len(target) or len(boosted_values) != len(target):
                raise SystemExit(f"{dataset}/{backend}: prediction row counts differ from test")
            feature_receipt = read_json(feature_receipt_path)
            baseline_metrics = regression_metrics(target, baseline_values)
            boosted_metrics = regression_metrics(target, boosted_values)
            baseline_rmse = baseline_metrics["rmse"]
            boosted_rmse = boosted_metrics["rmse"]
            delta = baseline_rmse - boosted_rmse
            report = {
                "schema": FINAL_REPORT_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "test_adaptive": True,
                "benchmark_valid": False,
                "in_sample_train_features": True,
                "dataset_hash": dataset,
                "backend": backend,
                "k_genomes": int(feature_receipt["emitted_columns"]),
                "feature_lineage": feature_receipt["feature_lineage"],
                "baseline_rmse": baseline_rmse,
                "boosted_rmse": boosted_rmse,
                "delta": delta,
                "delta_pct": (delta / baseline_rmse * 100.0) if baseline_rmse > 0 else None,
                "boosted_wins": boosted_rmse < baseline_rmse,
                "baseline_metrics": baseline_metrics,
                "boosted_metrics": boosted_metrics,
                "baseline_receipt_sha256": sha256_file(baseline_dir / "worker-receipt.json"),
                "boosted_receipt_sha256": sha256_file(boosted_dir / "worker-receipt.json"),
                "canonical_test_sha256": sha256_file(registry_test),
                "created_at": utc_now(),
            }
            final_result = TEST_ADAPTIVE_RUN / "datasets" / dataset / "final-result.v1.json"
            if final_result.is_file():
                payload = read_json(final_result)
                leader = payload.get("leader_rmse")
                if isinstance(leader, (int, float)) and math.isfinite(leader):
                    report["leader_rmse"] = float(leader)
                    report["leader_models"] = payload.get("leader_models")
                    report["boosted_beats_leader"] = boosted_rmse < float(leader)
                    report["baseline_beats_leader"] = baseline_rmse < float(leader)
            atomic_write_json(
                campaign_root / "reports" / dataset / backend / "final-report.v1.json",
                report,
            )
            written += 1
            print(
                f"{dataset} {backend}: baseline {baseline_rmse:.6g} boosted {boosted_rmse:.6g} "
                f"delta {delta:+.6g}"
            )
    if skipped:
        print(f"skipped {len(skipped)} cells without both arms: {', '.join(skipped)}")
    print(f"wrote {written} final reports under {campaign_root / 'reports'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    plan_parser = subparsers.add_parser("plan", help="emit SCQ specs (dry-run by default)")
    plan_parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    plan_parser.add_argument("--datasets", default="golden42")
    plan_parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    plan_parser.add_argument("--binary", type=Path, default=RELEASE_BINARY)
    plan_parser.add_argument("--dl-only", action="store_true")
    plan_parser.add_argument("--worker-root", type=Path)
    plan_parser.add_argument("--spec-dir", type=Path)
    plan_parser.add_argument("--shard-dir", type=Path)
    plan_parser.add_argument("--plan-name", default="plan.v1.json")
    plan_parser.add_argument(
        "--submit",
        action="store_true",
        help="actually submit via scq (default: dry-run printing the commands)",
    )
    plan_parser.set_defaults(handler=plan)

    submit_parser = subparsers.add_parser(
        "submit-next", help="submit at most one GPU shard from a plan"
    )
    submit_parser.add_argument("--plan", required=True, type=Path)
    submit_parser.add_argument("--state", type=Path)
    submit_parser.set_defaults(handler=submit_next)

    shard_parser = subparsers.add_parser("run-shard", help="node-side DL shard loop")
    shard_parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    shard_parser.add_argument("--worker-root", type=Path)
    shard_parser.add_argument("--shard-file", type=Path, required=True)
    shard_parser.add_argument("--backend", required=True)
    shard_parser.add_argument("--arm", required=True, choices=ARMS)
    shard_parser.set_defaults(handler=run_shard)

    finalize_parser = subparsers.add_parser("finalize", help="controller-side scoring")
    finalize_parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    finalize_parser.add_argument("--datasets", default="golden42")
    finalize_parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    finalize_parser.set_defaults(handler=finalize)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="write a complete controller-side aggregate report"
    )
    aggregate_parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    aggregate_parser.add_argument("--datasets", default="golden42")
    aggregate_parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    aggregate_parser.add_argument("--output", type=Path)
    aggregate_parser.set_defaults(handler=aggregate_reports)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
