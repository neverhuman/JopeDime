#!/usr/bin/env python3
"""Freeze the Golden46 dataset panel, historical cohort, and campaign policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "golden46.campaign-manifests.v1"
CAMPAIGN_ID = "golden46-full42-20260723-v1"
GOLDEN42_PANEL = Path("/home/ubuntu/golden42/golden42_foundry/data/panel.json")
GOLDEN46_PANEL = Path("/home/ubuntu/golden42/golden42_foundry/data/panel46.json")
DATA_ROOT = Path("/home/ubuntu/remote_super/quant/regression")
PRIMARY_REPORT_ROOT = DATA_ROOT
SUPPLEMENTARY_REPORT_ROOT = Path("/home/ubuntu/super_cluster/quant/regression")
MODEL_ZOO_PATH = Path("/home/ubuntu/remote_super/quant/models/tools/model_zoo.py")
DEFAULT_OUTPUT = Path("/home/ubuntu/remote_super/quant/campaigns") / CAMPAIGN_ID / "manifests"

NEW_DATASET_IDS = (
    "ZTS_small",
    "a3bd0a400ec4d7f9",
    "6054656956182296",
    "ZTS",
)
SPECIAL_RUNNERS = {
    "autogluon_300s": "autogluon_300s",
    "chimera_regression": "chimera_regression",
    "hyp3_chimera_regression": "hyp3_chimera_regression",
    "grok4p5_regressor": "grok4p5_regressor",
    "tabfm_ensemble_regressor": "tabfm_ensemble_regressor",
    "tabpfn_v3_regressor": "tabpfn_v3_regressor",
    "xgboost_hist_trees900_depth4_lr0p02": "xgboost_hist_trees900_depth4_lr0p02",
}
LEGACY_ASPFM_REPORT_ID = "jope-dime-regressor"
ASPFM_METHOD_ID = "jope-dime"
COMPLETED_STATUSES = {None, "ok", "completed", "success"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite changed immutable manifest: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def cohort_hash(dataset_ids: list[str]) -> str:
    return sha256_bytes("".join(f"{item}\n" for item in sorted(dataset_ids)).encode())


def csv_dimensions(path: Path, *, has_header: bool) -> tuple[int, int]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        first = next(reader)
        columns = len(first)
        rows = sum(1 for _ in reader) + 1 - int(has_header)
    return rows, columns


def dataset_record(dataset_id: str, position: int) -> dict[str, Any]:
    path = DATA_ROOT / dataset_id
    meta_path = path / "meta.json"
    train_path = path / "train.csv"
    test_path = path / "test.csv"
    for required in (meta_path, train_path, test_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    metadata = read_json(meta_path)
    headers_removed = metadata.get("headers_removed_from_csv")
    if not isinstance(headers_removed, bool):
        raise ValueError(f"{dataset_id}: meta.json lacks boolean headers_removed_from_csv")
    train_rows, train_columns = csv_dimensions(train_path, has_header=not headers_removed)
    test_rows, test_columns = csv_dimensions(test_path, has_header=not headers_removed)
    expected_train = metadata.get("train_rows", metadata.get("counts", {}).get("train_rows"))
    expected_test = metadata.get("test_rows", metadata.get("counts", {}).get("test_rows"))
    expected_features = metadata.get("n_features", metadata.get("counts", {}).get("n_features"))
    if (train_rows, test_rows) != (expected_train, expected_test):
        raise ValueError(
            f"{dataset_id}: CSV row counts {(train_rows, test_rows)} != metadata "
            f"{(expected_train, expected_test)}"
        )
    if train_columns != test_columns or train_columns != expected_features + 1:
        raise ValueError(
            f"{dataset_id}: expected {expected_features + 1} target-last columns, "
            f"found train={train_columns}, test={test_columns}"
        )
    provenance_group = "ZTS" if dataset_id in {"ZTS", "ZTS_small"} else metadata.get(
        "provenance_group", dataset_id
    )
    if dataset_id in {"ZTS", "ZTS_small"} and metadata.get("provenance_group") != "ZTS":
        raise ValueError(f"{dataset_id}: ZTS-family dataset must declare provenance_group=ZTS")
    if metadata.get("target_is_final_column", True) is not True:
        raise ValueError(f"{dataset_id}: target-last convention is required")
    return {
        "dataset_id": dataset_id,
        "panel_position": position,
        "source_dataset_name": metadata.get("source_dataset_name", dataset_id),
        "provenance_group": provenance_group,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "n_features": expected_features,
        "target_is_final_column": metadata.get("target_is_final_column", True),
        "headers_removed_from_csv": headers_removed,
        "raw_redistribution": (
            "withheld-pending-rights"
            if dataset_id in NEW_DATASET_IDS
            else "unchanged-golden42-policy"
        ),
        "artifacts": {
            "meta.json": sha256_file(meta_path),
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        },
    }


def report_score(path: Path, dataset_id: str) -> tuple[float | None, str]:
    try:
        report = read_json(path)
    except Exception as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if report.get("dataset_hash") != dataset_id:
        return None, "dataset_hash_mismatch"
    if report.get("status") not in COMPLETED_STATUSES:
        return None, f"status:{report.get('status')}"
    metrics = report.get("metrics")
    value = metrics.get("rmse") if isinstance(metrics, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None, "missing_finite_rmse"
    return float(value), ""


def selected_report_paths(dataset_id: str) -> dict[str, tuple[Path, str]]:
    supplementary = SUPPLEMENTARY_REPORT_ROOT / dataset_id / "reports"
    primary = PRIMARY_REPORT_ROOT / dataset_id / "reports"
    selected = {
        path.name: (path, "supplementary")
        for path in supplementary.glob("*.json")
    }
    selected.update({path.name: (path, "primary") for path in primary.glob("*.json")})
    return selected


def load_model_zoo() -> tuple[dict[str, Any], Any]:
    package_root = MODEL_ZOO_PATH.parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    module_spec = importlib.util.spec_from_file_location(
        "models.tools.model_zoo",
        MODEL_ZOO_PATH,
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_ZOO_PATH}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module.specs_by_name("regression"), module.DatasetInfo


def frozen_cohort(golden42_ids: list[str]) -> dict[str, Any]:
    coverage: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id in golden42_ids:
        for filename, (path, root_name) in selected_report_paths(dataset_id).items():
            value, _reason = report_score(path, dataset_id)
            if value is None:
                continue
            method_id = Path(filename).stem
            coverage.setdefault(method_id, {})[dataset_id] = {
                "root": root_name,
                "relative_path": str(path.relative_to(
                    PRIMARY_REPORT_ROOT if root_name == "primary" else SUPPLEMENTARY_REPORT_ROOT
                )),
                "sha256": sha256_file(path),
                "rmse": value,
            }
    report_backed = sorted(
        method_id
        for method_id, reports in coverage.items()
        if set(reports) == set(golden42_ids)
    )
    if len(report_backed) != 124:
        raise RuntimeError(f"expected 124 full-coverage report-backed methods, found {len(report_backed)}")
    if LEGACY_ASPFM_REPORT_ID not in report_backed:
        raise RuntimeError(f"missing historical ASPFM report identity {LEGACY_ASPFM_REPORT_ID}")
    if not set(SPECIAL_RUNNERS).issubset(report_backed):
        raise RuntimeError("one or more frozen special runners lacks complete Golden42 coverage")

    registry, DatasetInfo = load_model_zoo()
    registered = sorted(set(report_backed) - set(SPECIAL_RUNNERS) - {LEGACY_ASPFM_REPORT_ID})
    missing_registered = sorted(set(registered) - set(registry))
    if missing_registered:
        raise RuntimeError(f"report-backed methods missing registered specs: {missing_registered}")
    if len(registered) != 116:
        raise RuntimeError(f"expected 116 registered methods, found {len(registered)}")

    small_info = DatasetInfo("ZTS_small", 1000, 1000, 147)
    small_ineligible = [
        {"method_id": name, "reason": registry[name].check_eligibility(small_info)[1]}
        for name in registered
        if not registry[name].check_eligibility(small_info)[0]
    ]
    if small_ineligible:
        raise RuntimeError(f"registered methods ineligible on ZTS_small: {small_ineligible}")

    zts_info = DatasetInfo("ZTS", 50_000, 10_000, 147)
    zts_ineligible = [
        {"method_id": name, "reason": registry[name].check_eligibility(zts_info)[1]}
        for name in registered
        if not registry[name].check_eligibility(zts_info)[0]
    ]
    if len(zts_ineligible) != 20:
        raise RuntimeError(f"expected 20 explicit full-ZTS ineligibilities, found {len(zts_ineligible)}")

    method_records = []
    for method_id in report_backed:
        if method_id == LEGACY_ASPFM_REPORT_ID:
            execution = "historical-aspfm-reference-not-rerun-as-baseline"
        elif method_id in SPECIAL_RUNNERS:
            execution = "frozen-special-runner"
        else:
            execution = "registered-model-queue"
        method_records.append(
            {
                "method_id": method_id,
                "execution": execution,
                "golden42_reports": coverage[method_id],
            }
        )
    method_records.append(
        {
            "method_id": ASPFM_METHOD_ID,
            "display_name": "ASPFM",
            "execution": "adaptive-discovery",
            "benchmark_valid": False,
            "golden42_source": "immutable existing Golden42 ASPFM ledgers",
        }
    )
    payload = {
        "schema": "golden46.frozen-cohort.v1",
        "campaign_id": CAMPAIGN_ID,
        "derivation": {
            "golden42_panel": str(GOLDEN42_PANEL),
            "golden42_panel_sha256": sha256_file(GOLDEN42_PANEL),
            "primary_report_root": str(PRIMARY_REPORT_ROOT),
            "supplementary_report_root": str(SUPPLEMENTARY_REPORT_ROOT),
            "primary_precedence_on_filename_collision": True,
            "criterion": "finite nonnegative RMSE in a valid report on every ordered Golden42 dataset",
        },
        "counts": {
            "leaderboard_entries": 125,
            "report_backed_entries": 124,
            "pre_aspfm_baselines_to_execute": 123,
            "registered_model_queue": 116,
            "frozen_special_runners": 7,
            "historical_aspfm_reference": 1,
            "adaptive_discovery": 1,
        },
        "registered_method_ids": registered,
        "special_runner_ids": sorted(SPECIAL_RUNNERS),
        "historical_aspfm_report_id": LEGACY_ASPFM_REPORT_ID,
        "aspfm_method_id": ASPFM_METHOD_ID,
        "zts_small_registered_ineligibilities": small_ineligible,
        "full_zts_terminal_ineligibilities": zts_ineligible,
        "methods": method_records,
    }
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def campaign_policy(panel_hash: str, cohort_hash_value: str) -> dict[str, Any]:
    payload = {
        "schema": "golden46.campaign-policy.v1",
        "campaign_id": CAMPAIGN_ID,
        "panel_sha256": panel_hash,
        "cohort_sha256": cohort_hash_value,
        "authoritative_data_root": str(DATA_ROOT),
        "scratch_data_root": "/home/ubuntu/quant/regression",
        "receipts_root": str(DEFAULT_OUTPUT.parent),
        "dataset_priority": [
            "ZTS_small",
            "a3bd0a400ec4d7f9",
            "6054656956182296",
            "ZTS",
        ],
        "launch_gates": {
            "nomad_after": "ZTS_small compatibility wave",
            "zts_after": "confirmed ZTS_small baseline matrix and ASPFM result",
            "publication_after": "all 500 new leaderboard cells are validated scores or terminal receipts",
        },
        "controller_isolation": {
            "worker_inputs": ["training features", "training targets", "test features"],
            "controller_only": ["canonical test targets", "RMSE computation", "report finalization"],
            "legacy_test_target": "dummy final column",
            "reject": [
                "altered checkpoints",
                "row mismatches",
                "non-finite predictions",
                "changed method identities",
                "score/report disagreement",
            ],
        },
        "cluster": {
            "nodes": {
                "xbabe1": {"ram_reserve_gib": 16, "gpu_after_external_workload_clears": True},
                "xbabe2": {"ram_reserve_gib": 24, "authoritative_data": True, "gpu": True},
                "xbabe3": {"ram_reserve_gib": 24, "gpu": True},
            },
            "cpu_workers": {
                "initial_per_node": 96,
                "adjustment": 8,
                "adjustment_interval_seconds": 120,
                "target_cpu_percent": [70, 85],
                "maximum_per_node": 112,
            },
            "pressure_policy": {
                "pause_new_launches_on": ["memory reserve", "swap activity", "sustained I/O pressure"],
                "swapoff": False,
                "kill_unrelated_processes": False,
            },
            "timeouts_seconds": {"standard": 7200, "gpu": 7200, "special": 14400, "aspfm": None},
            "transient_retry": {"count": 1, "different_node": True},
            "one_heavy_model_per_gpu": True,
            "reuse_loaded_tabular_foundation_models": True,
        },
        "aspfm": {
            "islands": 4,
            "members_per_island": 9,
            "elites_per_island": 2,
            "generation_zero_can_terminate": False,
            "winner_minimum_generation": 1,
            "winner_must_be_fresh_mutation_or_crossover": True,
            "generation_zero_seed_mix": {
                "strongest_transferable": 18,
                "behaviorally_diverse": 18,
            },
            "exclude_target_provenance_group": True,
            "full_zts_arms": {
                "primary": {
                    "exclude_provenance_groups": ["ZTS"],
                    "leaderboard_default": True,
                    "label": "leak-safe new-data discovery",
                },
                "transfer": {
                    "include_confirmed_zts_small_winner": True,
                    "leaderboard_default": False,
                    "label": "target-family-informed",
                },
            },
        },
        "publication": {
            "golden42_track": "tabular-regression-v1",
            "golden46_track": "tabular-regression-v2",
            "raw_new_dataset_bytes": "withheld-pending-redistribution-rights",
            "golden46_atomic_commit": True,
        },
    }
    payload["content_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel42 = read_json(GOLDEN42_PANEL)
    panel46 = read_json(GOLDEN46_PANEL)
    golden42_ids = panel42["dataset_ids"]
    expected_ids = [*golden42_ids, *NEW_DATASET_IDS]
    if panel46.get("dataset_ids") != expected_ids:
        raise RuntimeError("panel46 ordering is not Golden42, ZTS_small, NOMAD pair, full ZTS")
    expected_panel_hash = cohort_hash(expected_ids)
    if panel46.get("panel_dataset_sha256") != expected_panel_hash:
        raise RuntimeError("panel46 cohort hash is inconsistent with its dataset ids")

    datasets = [dataset_record(dataset_id, position) for position, dataset_id in enumerate(expected_ids, 1)]
    panel_manifest = {
        "schema": "golden46.dataset-panel-manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_root": str(DATA_ROOT),
        "panel_dataset_sha256": expected_panel_hash,
        "golden42_panel_sha256": sha256_file(GOLDEN42_PANEL),
        "dataset_ids": expected_ids,
        "datasets": datasets,
    }
    panel_manifest["content_sha256"] = sha256_bytes(canonical_json(panel_manifest))
    cohort_manifest = frozen_cohort(golden42_ids)
    policy = campaign_policy(
        panel_manifest["content_sha256"],
        cohort_manifest["content_sha256"],
    )
    outputs = {
        "panel46.json": canonical_json(panel_manifest),
        "cohort.json": canonical_json(cohort_manifest),
        "campaign-policy.json": canonical_json(policy),
    }
    for name, content in outputs.items():
        path = args.output / name
        if args.verify:
            if not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(f"deterministic verification mismatch: {path}")
        else:
            immutable_write(path, content)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "panel_datasets": len(datasets),
                "leaderboard_entries": cohort_manifest["counts"]["leaderboard_entries"],
                "registered": cohort_manifest["counts"]["registered_model_queue"],
                "special": cohort_manifest["counts"]["frozen_special_runners"],
                "full_zts_ineligible": len(cohort_manifest["full_zts_terminal_ineligibilities"]),
                "hashes": {
                    name: sha256_bytes(content)
                    for name, content in outputs.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
