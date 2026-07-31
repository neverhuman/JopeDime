#!/usr/bin/env python3
"""Freeze the one-off Golden46 AutoInt runtime-overlay remediation."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


CAMPAIGN_ROOT = Path(
    "/home/ubuntu/remote_super/quant/campaigns/golden46-full42-20260723-v1"
)
DATASET_ID = "6054656956182296"
METHOD_ID = "pytorch_tabular_autoint_regressor"
NODE = "xbabe1"
OVERLAY = (
    CAMPAIGN_ROOT
    / "runtime-overlays"
    / "pytorch-tabular-1.2.0"
    / "site-packages"
)
ARTIFACT = CAMPAIGN_ROOT / "artifacts" / DATASET_ID / METHOD_ID
CLUSTER_INDEX = CAMPAIGN_ROOT / "indexes" / f"{DATASET_ID}.gpu.cluster-run.json"
OOM_LOG = (
    CAMPAIGN_ROOT
    / "nodes"
    / "xbabe3"
    / DATASET_ID
    / "gpu-gpu_heavy"
    / "logs"
    / f"{METHOD_ID}.log"
)
DEPENDENCY_LOG = (
    CAMPAIGN_ROOT
    / "nodes"
    / "xbabe1"
    / DATASET_ID
    / "gpu-gpu_heavy"
    / "logs"
    / f"{METHOD_ID}.log"
)
MANIFEST_PATH = (
    CAMPAIGN_ROOT
    / "runtime-overlays"
    / "pytorch-tabular-1.2.0"
    / "manifest.json"
)
RECEIPT_PATH = (
    CAMPAIGN_ROOT / "indexes" / f"{DATASET_ID}.{METHOD_ID}.remediation.json"
)
COMMAND = [
    "/home/ubuntu/remote_super/quant/.venv-model-queue/bin/python",
    "/home/ubuntu/jope-dime/scripts/golden46_execution.py",
    "--worker-root",
    str(CAMPAIGN_ROOT / "worker-views" / "regression"),
    "--artifact-root",
    str(CAMPAIGN_ROOT / "artifacts"),
    "run-registered",
    "--dataset",
    DATASET_ID,
    "--method",
    METHOD_ID,
]
ENVIRONMENT = {
    "PYTHONPATH": str(OVERLAY),
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def remote_overlay_files() -> list[dict[str, Any]]:
    source = """
import hashlib
import json
from pathlib import Path

root = Path(%r)
if not root.is_dir():
    raise SystemExit("overlay directory is missing")
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    })
print(json.dumps(files, separators=(",", ":")))
""" % str(OVERLAY)
    remote_command = f"/usr/bin/python3 -c {shlex.quote(source)}"
    completed = subprocess.run(
        ["ssh", NODE, remote_command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list) or not value:
        raise ValueError("runtime overlay file manifest is empty")
    return value


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> int:
    worker = read_json(ARTIFACT / "worker-receipt.json")
    prediction = ARTIFACT / "predictions.npy"
    cluster = read_json(CLUSTER_INDEX)
    if (
        worker.get("dataset_hash") != DATASET_ID
        or worker.get("algorithm") != METHOD_ID
        or worker.get("status") != "predicted-unscored"
        or worker.get("environment", {}).get("host") != NODE
        or worker.get("environment", {})
        .get("distributions", {})
        .get("pytorch_tabular")
        != "1.2.0"
    ):
        raise ValueError("final worker receipt does not match the repaired execution")
    if sha256(prediction) != worker["predictions"]["sha256"]:
        raise ValueError("final prediction file disagrees with the worker receipt")

    retries = cluster.get("retries", [])
    matching_retry = next(
        (item for item in retries if item.get("method_id") == METHOD_ID),
        None,
    )
    if (
        matching_retry is None
        or matching_retry.get("original_node") != "xbabe3"
        or matching_retry.get("retry_node") != NODE
        or matching_retry.get("return_code") == 0
    ):
        raise ValueError("cluster index does not contain the expected failed retry")
    if "CUDA out of memory" not in OOM_LOG.read_text():
        raise ValueError("initial failure log is not the expected GPU OOM")
    if "missing dependencies: ['pytorch_tabular']" not in DEPENDENCY_LOG.read_text():
        raise ValueError("retry failure log is not the expected dependency failure")

    files = remote_overlay_files()
    overlay_manifest = {
        "schema": "golden46.runtime-overlay-manifest.v1",
        "node": NODE,
        "root": str(OVERLAY),
        "excluded_generated_files": ["**/__pycache__/**", "**/*.pyc"],
        "file_count": len(files),
        "files": files,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(overlay_manifest, indent=2, sort_keys=True) + "\n")

    started = parse_utc(worker["started_at"])
    ended = parse_utc(worker["ended_at"])
    receipt = {
        "schema": "golden46.runtime-remediation.v1",
        "status": "validated",
        "campaign_id": "golden46-full42-20260723-v1",
        "dataset_id": DATASET_ID,
        "method_id": METHOD_ID,
        "reason": (
            "The frozen AutoInt configuration exhausted the RTX 3090 during its "
            "first attempt; the designated retry node lacked pytorch_tabular. "
            "The same method, model-zoo identity, parameters, rows, and features "
            "were executed on CPU with a checksum-frozen dependency overlay."
        ),
        "method_identity": worker["method_identity"],
        "command": COMMAND,
        "environment_overrides": ENVIRONMENT,
        "hardware_execution": {
            "node": NODE,
            "cuda_visible_devices": "",
            "mode": "cpu",
        },
        "runtime_seconds": (ended - started).total_seconds(),
        "started_at": worker["started_at"],
        "ended_at": worker["ended_at"],
        "failure_chain": [
            {
                "stage": "initial",
                "node": "xbabe3",
                "reason": "CUDA out of memory",
                "log": str(OOM_LOG),
                "log_sha256": sha256(OOM_LOG),
            },
            {
                "stage": "designated-cross-node-retry",
                "node": NODE,
                "reason": "missing pytorch_tabular dependency",
                "log": str(DEPENDENCY_LOG),
                "log_sha256": sha256(DEPENDENCY_LOG),
            },
        ],
        "cluster_index": {
            "path": str(CLUSTER_INDEX),
            "sha256": sha256(CLUSTER_INDEX),
        },
        "runtime_overlay": {
            "path": str(OVERLAY),
            "manifest_path": str(MANIFEST_PATH),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "file_count": len(files),
        },
        "result": {
            "worker_receipt_path": str(ARTIFACT / "worker-receipt.json"),
            "worker_receipt_sha256": sha256(ARTIFACT / "worker-receipt.json"),
            "prediction_path": str(prediction),
            "prediction_sha256": sha256(prediction),
            "prediction_array_sha256": worker["predictions"]["array_sha256"],
        },
        "model_or_data_changes": False,
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
