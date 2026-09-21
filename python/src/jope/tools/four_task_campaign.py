"""Isolated four-task campaign identities. Never merges into regression recovery."""

from __future__ import annotations

import hashlib
import json

SCHEMA = "parallel.four_task_campaign.v1"
TASKS = ("regression", "binary", "multiclass", "timeseries")
RECOVERY_CAMPAIGN = "regression-campaign64-qualification-v1"
RECOVERY_SHA = "c2f229c9b8c582e11e0e62ec011a9ccb066820fbcee5e2da9d0ad5d63780160e"
QUANT_DIR = {
    "regression": "regression",
    "binary": "binary",
    "multiclass": "multi_label",
    "timeseries": "timeseries",
}
PRIMARY = {
    "regression": ("rmse", "original_target"),
    "binary": ("logloss", "class_index"),
    "multiclass": ("logloss", "class_index"),
    "timeseries": ("mase", "original_target"),
}


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def campaign_id(task, inventory_sha):
    if task not in TASKS:
        raise ValueError(f"unsupported task {task}")
    return digest({"schema": SCHEMA, "task": task, "inventory_sha256": inventory_sha})


def isolate(task, inventory_sha, parent_campaign=None, parent_checkpoint_sha=None):
    if parent_campaign == RECOVERY_CAMPAIGN or parent_checkpoint_sha == RECOVERY_SHA:
        if task != "regression":
            raise ValueError("non-regression campaign cannot inherit regression recovery identity")
    identity = campaign_id(task, inventory_sha)
    metric, units = PRIMARY[task]
    return dict(
        schema=SCHEMA,
        task=task,
        quant_dir=QUANT_DIR[task],
        campaign_id=identity,
        corpus_id=digest({"campaign": identity, "kind": "corpus"}),
        frozen_panel_id=digest({"campaign": identity, "kind": "frozen_panel"}),
        specialist_checkpoint_bind=digest({"campaign": identity, "kind": "specialist"}),
        primary_metric=metric,
        units=units,
        may_merge_into_recovery=False,
        expand_cuts=task != "timeseries",
        rolling_origin=task == "timeseries",
        multi_label_means_multiclass=task == "multiclass",
        dispatcher="prepared.task",
        routing_ce=False,
    )


def inventory(dataset_dirs, task):
    rows = [{"dataset": name, "task": task} for name in dataset_dirs]
    return dict(
        schema="parallel.task_inventory.v1",
        task=task,
        n=len(rows),
        sha256=digest({"task": task, "datasets": sorted(dataset_dirs)}),
        rows=rows,
    )


def frozen_panel(campaign, families, cuts_per_family=2):
    if campaign["may_merge_into_recovery"]:
        raise ValueError("panel refused: campaign marked mergeable")
    return dict(
        schema="parallel.task_frozen_panel.v1",
        campaign_id=campaign["campaign_id"],
        task=campaign["task"],
        metric=campaign["primary_metric"],
        units=campaign["units"],
        families=list(families),
        cuts_per_family=cuts_per_family,
        independent_evaluation=False,
        recovery_checkpoint_sha256=None if campaign["task"] != "regression" else RECOVERY_SHA,
    )


def specialist_bind(campaign, checkpoint_sha):
    if campaign["task"] != "regression" and checkpoint_sha == RECOVERY_SHA:
        raise ValueError("specialist cannot bind the regression recovery checkpoint")
    return dict(
        schema="parallel.specialist_bind.v1",
        task=campaign["task"],
        campaign_id=campaign["campaign_id"],
        checkpoint_sha256=checkpoint_sha,
        vocabulary_shared=True,
        routing_ce=False,
    )
