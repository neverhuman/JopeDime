from __future__ import annotations

from pathlib import Path

SCQ = Path("/home/ubuntu/super_cluster/scq")
PROJECT = "sotarena-b42"
RUN_ID = "golden42-smart-v3-cpu-20260730-v1"
PRIOR_RUN_ID = "golden42-train-only-redteam-20260730-v1"
PARENT_RUN_ID = "golden42-transfer-evolution-20260730-v1"
CHECKPOINT_SHA256 = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
NODES = ("xbabe1", "xbabe2", "xbabe3", "xbabe4")
SHARDS = {
    "xbabe1": ("55f06d147bcc2a32",),
    "xbabe2": (
        "bd49e740a79bcb6c",
        "55b144de19659467",
        "aebf5d861a8365b4",
        "019de5d76a718163",
    ),
    "xbabe3": (
        "b5beff4a53fa12c0",
        "5f91a00bd3b66742",
        "ee56a1015e344c11",
    ),
    "xbabe4": ("7af85a04de0c7018", "f8536dfeef471c2f"),
}
ACTIVE_STATES = {"queued", "held", "dispatching", "running", "preempting", "requeued"}
INFRASTRUCTURE_FAILURES = {"rpc-error", "storage", "vanished"}
DEFAULT_POLL_SECONDS = 60
DISPATCH_VERSION = "v4"
CAMPAIGN_DIR = "eleven-job-v4"
PROGRESS_SCHEMA = "SmartV3CpuSCQProgress.v2"
JOB_COUNT = 11
EVOLUTION_JOB_COUNT = 10


def dataset_owners() -> tuple[tuple[str, str], ...]:
    return tuple(
        (node, dataset)
        for node in NODES
        for dataset in SHARDS[node]
    )


DATASETS = tuple(dataset for _, dataset in dataset_owners())
DATASET_OWNERSHIP = dict(dataset_owners())
