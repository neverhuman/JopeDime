"""Protocol/statistical fixtures only; no learned model or sealed payload is executed."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import q1
import q1_bindings

def qualification_plan():
    datasets = []
    for i in range(40):
        dataset_id = f"{i + 1:016x}"
        datasets.append(
            {
                "id": dataset_id,
                "task": "regression",
                "dataset_sha256": dataset_id + "a" * 48,
                "family": f"{i + 100:040x}",
                "family_authoritative": True,
                "rows": 500 if i % 2 == 0 else 2000,
                "features": 10 if i % 4 < 2 else 30,
                "cut_sha256": "c" * 64,
                "holdout_group": "sealed_meta_test",
            }
        )
    return {
        "schema": "jope.q1-plan/v1",
        "mode": "qualification",
        "source_commit": json.loads(q1_bindings.EVALUATOR_PIN.read_text())[
            "source_commit"
        ],
        "preregistration_sha256": q1.sha_file(q1.PREREGISTRATION),
        "backend_sha256": "b" * 64,
        "budget": 24,
        "seeds": [0, 1, 2],
        "datasets": datasets,
        "training_families": ["f" * 40],
        "pins": {
            **{
                k: "d" * 64
                for k in (
                    "bundle_sha256",
                    "dope_sha256",
                    "evaluator_sha256",
                    "training_family_manifest_sha256",
                )
            },
            **q1_bindings.expected_pins(),
            "evaluator_sha256": json.loads(q1_bindings.EVALUATOR_PIN.read_text())[
                "sha256"
            ],
        },
        "timeout_seconds": 600,
        "ledger_root": "/qualification-test-ledger",
    }


def fixture_receipts(plan, learned=0.9):
    rows = []
    for dataset in plan["datasets"]:
        for seed in plan["seeds"]:
            for policy in ("heuristic", "learned"):
                request = {
                    "schema": "jope.q1-request/v1",
                    "mode": "qualification",
                    "plan_sha256": q1.digest(plan),
                    "source_commit": plan["source_commit"],
                    "pins": plan["pins"],
                    "metric": "rmse",
                    "dataset": dataset,
                    "guidance": policy,
                    "seed": seed,
                    "budget": plan["budget"],
                }
                request["request_sha256"] = q1.digest(request)
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
                rows.append(
                    {**identity, "event": "started", "status": "outcome_unknown"}
                )
                response = {
                    "schema": "jope.q1-response/v1",
                    "request_sha256": request["request_sha256"],
                    "status": "complete",
                    "guidance": policy,
                    "metric": "rmse",
                    "charged": 24,
                    "completed": 24,
                    "loss": 1.0 if policy == "heuristic" else learned,
                    "fold_scores_completed": 48,
                    "candidate_failures": [],
                    "champion_program": "unit-fixture",
                    "adapter": "unit-fixture-no-execution",
                }
                rows.append(
                    {
                        **identity,
                        "event": "finished",
                        "response": response,
                        "wall_seconds": 0.1,
                    }
                )
    return rows


