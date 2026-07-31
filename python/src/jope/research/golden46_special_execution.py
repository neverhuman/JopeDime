#!/usr/bin/env python3
"""Leakage-resistant execution for the seven frozen Golden46 special runners.

The public ``run`` command validates a masked worker view and launches a
method-specific backend interpreter.  Backends can read only that view and
write unscored predictions.  The ``finalize`` command is the only path that
opens canonical test labels.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from golden46_execution import (
    CAMPAIGN_ID,
    CANONICAL_ROOT,
    COHORT_PATH,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_CONTROLLER_ROOT,
    DEFAULT_WORKER_ROOT,
    HYP3_OUTPUTS,
    atomic_save_npy,
    atomic_write_json,
    canonical_dataset,
    canonical_json,
    read_canonical_frame,
    read_json,
    regression_metrics,
    sha256_array,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_worker_view,
)


SPECIAL_SOURCE_PATH = Path(__file__).resolve()
HISTORICAL_QUANT_RUNNER = Path("/home/ubuntu/remote_super/guant/benchmark_quant_models.py")
HISTORICAL_TABFM_RUNNER = Path("/home/ubuntu/remote_super/guant/benchmark_tabfm_models.py")
HISTORICAL_CHIMERA_RUNNER = Path(
    "/home/ubuntu/remote_super/fable_god/chimera_regression/python/router/"
    "run_chimera_regression_reports.py"
)
CHIMERA_MODULE = Path(
    "/home/ubuntu/remote_super/fable_god/chimeria_regression_shuffle/chimera.py"
)
CHIMERA_CHECKPOINT = Path(
    "/home/ubuntu/fable_god/chimeria_regression_shuffle/run_FINAL_e10_trunk/"
    "FINAL_stackfusionreg.pt"
)
CHIMERA_SAFETENSOR = Path(
    "/home/ubuntu/jain-split/jain-starforge/artifacts/starforge/"
    "chimera_stackfusionreg.safetensors"
)
CHIMERA_CONVERSION_RECEIPT = Path(
    "/home/ubuntu/TO_DELETE/jain_small/deployment/ops/"
    "starforge-artifacts.receipt.json"
)
CHIMERA_INVENTORY = Path(
    "/home/ubuntu/remote_super/fable_god/chimeria_regression_shuffle/"
    "run_FINAL_e10_trunk/FINAL_file_inventory.json"
)
CHIMERA_EXPERT_A = CHIMERA_MODULE.parent / "weights/expert_a.safetensors"
CHIMERA_EXPERT_B = CHIMERA_MODULE.parent / "weights/expert_b.safetensors"
CHIMERA_EXPERT_CONFIG = CHIMERA_MODULE.parent / "weights/starlight_config.json"
FABLE_PYTHON = Path("/home/ubuntu/remote_super/fable_god/python")
FABLE_MODEL_IO = FABLE_PYTHON / "elo_finetune/model_io.py"
FABLE_PATHS = FABLE_PYTHON / "elo_finetune/paths.py"
FABLE_REGRESSOR_SOURCE = Path(
    "/home/ubuntu/remote_super/fable_god/provenance/regressor_source"
)
FABLE_REGRESSOR_TRAIN = FABLE_REGRESSOR_SOURCE / "starlight_core_regressor_train.py"
GROK_SOURCE = Path("/home/ubuntu/super_cluster/grok4p5_run/cursor_grok4p5hf.py")
GROK_HARNESS = Path("/home/ubuntu/super_cluster/grok4p5_run/benchmark.py")
MODEL_ZOO = Path("/home/ubuntu/remote_super/quant/models/tools/model_zoo.py")
TABPFN_CHECKPOINT = Path(
    "/home/ubuntu/guant/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"
)
TABFM_SNAPSHOT = Path(
    "/home/ubuntu/.cache/huggingface/hub/models--google--tabfm-1.0.0-pytorch/"
    "snapshots/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
)
TABFM_CHECKPOINT = TABFM_SNAPSHOT / "regression/model.safetensors"
TABFM_CONFIG = TABFM_SNAPSHOT / "regression/config.json"
TABFM_ROOT_CONFIG = TABFM_SNAPSHOT / "config.json"

THREAD_LIMITS = {
    "BLIS_NUM_THREADS": "1",
    "LIGHTGBM_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "XGBOOST_NTHREADS": "1",
    "XGBOOST_NUM_THREADS": "1",
}

EXPECTED_FILES = {
    HISTORICAL_QUANT_RUNNER: "77e967d0e7024d2b46c51e1997147957720a514771b2e33cc35200454dff29fa",
    HISTORICAL_TABFM_RUNNER: "e5b7dcb663e5f00ba899e430abb0758928359e203cf01b54d49366e28cbafc82",
    HISTORICAL_CHIMERA_RUNNER: "44ae4d81c2081a84ebf4ebe4e268910e2733bada4e7ec40d6b68042f20922938",
    CHIMERA_MODULE: "babb5f922ae515ccd8a3dba1c779f3c3e5d5e3c023604fabbd785805ed579b67",
    CHIMERA_INVENTORY: "577bab17e9298ea3afa1450973a0f1c4be0d93da746c647bfc261fc88a97d406",
    CHIMERA_EXPERT_A: "041ffd016c0a3a32aa4c2592b24a9fca8db086f7ab6844e5c7260d63aa2f3862",
    CHIMERA_EXPERT_B: "544a4573398e2548dc9ac379d53bdb6a5ae58fc5646619ed029227db60528848",
    CHIMERA_EXPERT_CONFIG: "4acacec8c148702e07147452e3e13dad586bd40f90878a3bbac8355753978acb",
    FABLE_MODEL_IO: "5e142384c871f4df8e59271e5998524ea1806e8a8ea2f240ea9a8190d8e28805",
    FABLE_PATHS: "3db03efb5ca67fcc5a898fc9bbc83fb1e4053fd6e98338b8836b39d3ca3c8cc7",
    FABLE_REGRESSOR_TRAIN: "71dbbce7c0c498917f4bfb57e674177c8240ce7b428decceea20dad0e3c259e9",
    GROK_SOURCE: "14ca62e456388c171b4da837ba88e6aa77df3ae6a36c9833126d0425a827dcad",
    GROK_HARNESS: "13bdf302baaf03b7ab59fb77c040cc5711cca0981772c9a0f2488f139c1d8d79",
    MODEL_ZOO: "3be356f5931021a94639bd53b5730dae9f28f94f20379deeb9b0921e67c1687e",
    TABPFN_CHECKPOINT: "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
    TABFM_CHECKPOINT: "bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9",
    TABFM_CONFIG: "169c7a935965b45c4ed9d981201392743e498115ceddb29931eccd37e0c8f6ad",
    TABFM_ROOT_CONFIG: "21c1e7704490da38026a37920d52fd8f5ba552c9f1eba9424fadb11095d7dd45",
    CHIMERA_SAFETENSOR: "57cdb53ae732a705e6aee7381a6a11e7e82cf357bcd67c46ccfa47b789c7d5d2",
    CHIMERA_CONVERSION_RECEIPT: "a86a63bca8148ca3238472a3daf838559a0666fa6e45b96afd34438bb5e3fb46",
}

CHIMERA_EXPECTED = {
    "path": str(CHIMERA_CHECKPOINT),
    "bytes": 4_612_145,
    "sha256": "087311c6b64ecc15df3022b011b6618d8d2f1040daa42d1c7f9bb382ab2ed20b",
}
CHIMERA_CONVERTED_EXPECTED = {
    "path": str(CHIMERA_SAFETENSOR),
    "bytes": 4_604_636,
    "sha256": EXPECTED_FILES[CHIMERA_SAFETENSOR],
    "format": "pt_to_safetensors",
    "architecture": "StackFusionReg",
    "source_sha256": CHIMERA_EXPECTED["sha256"],
    "tensor_count": 32,
    "conversion_receipt": {
        "path": str(CHIMERA_CONVERSION_RECEIPT),
        "sha256": EXPECTED_FILES[CHIMERA_CONVERSION_RECEIPT],
    },
}
CHIMERA_METHODS = {"chimera_regression", "hyp3_chimera_regression"}

SPECIAL_SPECS: dict[str, dict[str, Any]] = {
    "autogluon_300s": {
        "resource": "cpu",
        "interpreter": "/usr/bin/python3",
        "model_class": "__main__.AutoGluonAdapter",
        "parameters": {
            "problem_type": "regression",
            "eval_metric": "rmse",
            "presets": "medium_quality",
            "time_limit": 270.0,
            "ag_args_fit": {"num_cpus": 1, "num_gpus": 0},
            "seed": 20260621,
        },
        "sources": [HISTORICAL_QUANT_RUNNER],
    },
    "chimera_regression": {
        "resource": "gpu",
        "interpreter": "/home/ubuntu/jmcp-split/jmcp-talk/research/.venv_tabpfn3/bin/python",
        "model_class": "chimera.Chimera",
        "parameters": {
            "device": "cuda",
            "source_checkpoint": CHIMERA_EXPECTED,
            "converted_checkpoint": CHIMERA_CONVERTED_EXPECTED,
        },
        "sources": [
            HISTORICAL_CHIMERA_RUNNER,
            CHIMERA_MODULE,
            CHIMERA_INVENTORY,
            CHIMERA_EXPERT_A,
            CHIMERA_EXPERT_B,
            CHIMERA_EXPERT_CONFIG,
            FABLE_MODEL_IO,
            FABLE_PATHS,
            FABLE_REGRESSOR_TRAIN,
            CHIMERA_SAFETENSOR,
            CHIMERA_CONVERSION_RECEIPT,
        ],
        "checkpoint": CHIMERA_CONVERTED_EXPECTED,
    },
    "grok4p5_regressor": {
        "resource": "cpu_light",
        "interpreter": "/usr/bin/python3",
        "model_class": "cursor_grok4p5hf.run",
        "parameters": {"candidate_seed": 20260719},
        "sources": [GROK_SOURCE, GROK_HARNESS],
    },
    "hyp3_chimera_regression": {
        "resource": "gpu",
        "interpreter": "/home/ubuntu/jmcp-split/jmcp-talk/research/.venv_tabpfn3/bin/python",
        "model_class": "chimera.Chimera",
        "parameters": {
            "device": "cuda",
            "source_checkpoint": CHIMERA_EXPECTED,
            "converted_checkpoint": CHIMERA_CONVERTED_EXPECTED,
            "feature_order": ["train3", "hyperion3", "original_csv_features"],
        },
        "sources": [
            HISTORICAL_CHIMERA_RUNNER,
            CHIMERA_MODULE,
            CHIMERA_INVENTORY,
            CHIMERA_EXPERT_A,
            CHIMERA_EXPERT_B,
            CHIMERA_EXPERT_CONFIG,
            FABLE_MODEL_IO,
            FABLE_PATHS,
            FABLE_REGRESSOR_TRAIN,
            CHIMERA_SAFETENSOR,
            CHIMERA_CONVERSION_RECEIPT,
        ],
        "checkpoint": CHIMERA_CONVERTED_EXPECTED,
        "hyp3": True,
    },
    "tabfm_ensemble_regressor": {
        "resource": "gpu",
        "interpreter": "/home/ubuntu/remote_super/tabfm_venv/bin/python",
        "model_class": "tabfm.src.classifier_and_regressor.TabFMRegressor",
        "parameters": {
            "model_type": "regression",
            "device": "cuda",
            "ensemble": True,
        },
        "sources": [
            HISTORICAL_TABFM_RUNNER,
            TABFM_CHECKPOINT,
            TABFM_CONFIG,
            TABFM_ROOT_CONFIG,
        ],
    },
    "tabpfn_v3_regressor": {
        "resource": "gpu",
        "interpreter": "/home/ubuntu/jmcp-split/jmcp-talk/research/.venv_tabpfn3/bin/python",
        "model_class": "tabpfn.regressor.TabPFNRegressor",
        "parameters": {
            "model_path": str(TABPFN_CHECKPOINT),
            "device": "cuda",
            "random_state": 20260621,
            "ignore_pretraining_limits": True,
        },
        "sources": [HISTORICAL_QUANT_RUNNER, TABPFN_CHECKPOINT],
    },
    "xgboost_hist_trees900_depth4_lr0p02": {
        "resource": "cpu",
        "interpreter": "/usr/bin/python3",
        "model_class": "sklearn.pipeline.Pipeline",
        "parameters": {
            "n_estimators": 900,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_jobs": 1,
            "nthread": 1,
            "random_state": 20260624,
            "verbosity": 0,
            "preprocessing": "SimpleImputer(strategy='median')",
        },
        "sources": [MODEL_ZOO],
    },
}


def frozen_special_ids() -> set[str]:
    return set(read_json(COHORT_PATH)["special_runner_ids"])


def validate_source_files(spec: dict[str, Any]) -> None:
    for path in spec["sources"]:
        expected = EXPECTED_FILES[path]
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source/checkpoint: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen file changed: {path}: {actual} != {expected}")


def validate_chimera_artifacts() -> dict[str, Any]:
    """Prove the converted checkpoint retains the frozen source identity."""
    validate_source_files(SPECIAL_SPECS["chimera_regression"])
    if CHIMERA_SAFETENSOR.stat().st_size != CHIMERA_CONVERTED_EXPECTED["bytes"]:
        raise ValueError("converted Chimera checkpoint size mismatch")
    from safetensors import safe_open

    with safe_open(CHIMERA_SAFETENSOR, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    expected_metadata = {
        "architecture": CHIMERA_CONVERTED_EXPECTED["architecture"],
        "format": CHIMERA_CONVERTED_EXPECTED["format"],
        "source_sha256": CHIMERA_EXPECTED["sha256"],
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"converted Chimera metadata mismatch: {metadata} != {expected_metadata}"
        )
    if len(keys) != CHIMERA_CONVERTED_EXPECTED["tensor_count"] or len(set(keys)) != len(keys):
        raise ValueError("converted Chimera tensor inventory is not exactly 32 unique tensors")

    conversion = read_json(CHIMERA_CONVERSION_RECEIPT)
    source_records = conversion.get("source_artifacts")
    artifact_records = conversion.get("artifacts")
    if not isinstance(source_records, list) or not isinstance(artifact_records, list):
        raise ValueError("Chimera conversion receipt lacks source/artifact records")
    source = next(
        (
            record
            for record in source_records
            if record.get("sha256") == CHIMERA_EXPECTED["sha256"]
            and Path(str(record.get("converted_to", ""))).name == CHIMERA_SAFETENSOR.name
        ),
        None,
    )
    converted = next(
        (
            record
            for record in artifact_records
            if record.get("sha256") == CHIMERA_CONVERTED_EXPECTED["sha256"]
            and Path(str(record.get("path", ""))).name == CHIMERA_SAFETENSOR.name
        ),
        None,
    )
    if source is None or converted is None:
        raise ValueError("conversion receipt does not bind the source and converted Chimera")
    if converted.get("size") != CHIMERA_CONVERTED_EXPECTED["bytes"]:
        raise ValueError("conversion receipt Chimera size mismatch")
    return {
        "source_checkpoint": CHIMERA_EXPECTED,
        "converted_checkpoint": CHIMERA_CONVERTED_EXPECTED,
        "embedded_metadata": metadata,
        "tensor_count": len(keys),
        "tensor_keys_sha256": sha256_bytes(canonical_json(keys)),
        "conversion_receipt_sha256": sha256_file(CHIMERA_CONVERSION_RECEIPT),
        "receipt_source_record": source,
        "receipt_converted_record": converted,
    }


def load_chimera_compatible(device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    """Strict-load the lossless safetensor through a temporary PyTorch state dict."""
    validation = validate_chimera_artifacts()
    import torch
    from safetensors.torch import load_file

    state = load_file(str(CHIMERA_SAFETENSOR), device="cpu")
    if len(state) != CHIMERA_CONVERTED_EXPECTED["tensor_count"]:
        raise ValueError("converted Chimera state dictionary tensor count mismatch")

    os.environ["FABLE_REGRESSOR_SOURCE"] = str(FABLE_REGRESSOR_SOURCE)
    sys.path.insert(0, str(CHIMERA_MODULE.parent))
    from chimera import Chimera

    import python as frozen_python

    fable_python = str(FABLE_PYTHON)
    if fable_python not in frozen_python.__path__:
        frozen_python.__path__.append(fable_python)

    with tempfile.TemporaryDirectory(prefix="golden46-chimera-state-") as temp:
        compatibility_path = Path(temp) / "compatibility-state-dict.pt"
        torch.save(state, compatibility_path)
        model = Chimera(ckpt=str(compatibility_path), device=device)

    loaded = model.trunk.state_dict()
    if set(loaded) != set(state):
        raise ValueError("strict Chimera state-dictionary keys mismatch")
    unequal = [
        name
        for name, tensor in state.items()
        if tensor.shape != loaded[name].shape
        or tensor.dtype != loaded[name].dtype
        or not torch.equal(tensor, loaded[name].detach().cpu())
    ]
    if unequal:
        raise ValueError(f"strict Chimera state-dictionary tensor mismatch: {unequal}")
    validation.update(
        {
            "compatibility_transport": "temporary-pytorch-state-dictionary",
            "strict_state_dict_load": True,
            "loaded_tensor_count": len(loaded),
            "loaded_tensor_keys_sha256": sha256_bytes(canonical_json(sorted(loaded))),
        }
    )
    return model, validation


def special_identity(method_id: str) -> dict[str, Any]:
    if method_id not in frozen_special_ids() or method_id not in SPECIAL_SPECS:
        raise ValueError(f"{method_id} is not in the frozen seven-method special cohort")
    spec = SPECIAL_SPECS[method_id]
    sources = []
    for path in spec["sources"]:
        sources.append({"path": str(path), "sha256": EXPECTED_FILES[path]})
    payload = {
        "method_id": method_id,
        "execution": "frozen-special-runner",
        "resource": spec["resource"],
        "model_class": spec["model_class"],
        "parameters": spec["parameters"],
        "sources": sources,
        "cohort_content_sha256": read_json(COHORT_PATH)["content_sha256"],
    }
    payload["identity_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def compatibility_blocker(
    dataset_id: str,
    method_id: str,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    output = artifact_root / dataset_id / method_id
    receipt = {
        "schema": "golden46.special-compatibility-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": method_id,
        "status": "blocked-missing-frozen-artifact",
        "benchmark_cell_resolved": False,
        "retryable_after_exact_artifact_restore": True,
        "missing_artifact": CHIMERA_EXPECTED,
        "substitution_permitted": False,
        "method_identity_changed": False,
        "test_labels_available_to_worker": False,
        "created_at": utc_now(),
    }
    atomic_write_json(output / "compatibility-receipt.json", receipt)
    return receipt


def load_worker_arrays(worker_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(worker_path / "train.csv", header=None).to_numpy(dtype=np.float32)
    test = pd.read_csv(worker_path / "test.csv", header=None).to_numpy(dtype=np.float32)
    return train[:, :-1], train[:, -1].astype(np.float64), test[:, :-1]


def load_module(name: str, path: Path) -> Any:
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def hyp3_arrays(worker_path: Path) -> tuple[np.ndarray, np.ndarray]:
    receipt_path = worker_path / "hyp3-receipt.json"
    receipt = read_json(receipt_path)
    blocks = []
    for name in HYP3_OUTPUTS:
        expected = receipt["artifacts"][name]["sha256"]
        if sha256_file(worker_path / name) != expected:
            raise ValueError(f"altered HYP3 artifact: {name}")
        blocks.append(pd.read_csv(worker_path / name, header=None).to_numpy(dtype=np.float32))
    train3, test3, hyperion_train, hyperion_test = blocks
    return (
        np.concatenate([train3, hyperion_train], axis=1),
        np.concatenate([test3, hyperion_test], axis=1),
    )


def backend_predict(method_id: str, worker_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    x_train, y_train, x_test = load_worker_arrays(worker_path)
    fit_started = time.monotonic()
    if method_id == "autogluon_300s":
        from autogluon.tabular import TabularPredictor

        columns = [f"f{index}" for index in range(x_train.shape[1])]
        train_frame = pd.DataFrame(x_train, columns=columns)
        train_frame["__ag_label__"] = y_train
        predictor = TabularPredictor(
            label="__ag_label__",
            problem_type="regression",
            eval_metric="rmse",
            verbosity=0,
        ).fit(
            train_frame,
            time_limit=270.0,
            presets="medium_quality",
            ag_args_fit={"num_cpus": 1, "num_gpus": 0},
        )
        training_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predictions = predictor.predict(pd.DataFrame(x_test, columns=columns))
    elif method_id in CHIMERA_METHODS:
        if method_id == "hyp3_chimera_regression":
            train_blocks, test_blocks = hyp3_arrays(worker_path)
            x_train = np.concatenate([train_blocks, x_train], axis=1)
            x_test = np.concatenate([test_blocks, x_test], axis=1)
        model, chimera_identity = load_chimera_compatible("cuda")
        training_seconds = 0.0
        predict_started = time.monotonic()
        predictions = model.predict(x_train, y_train.astype(np.float32), x_test)
    elif method_id == "grok4p5_regressor":
        candidate = load_module("frozen_grok4p5_candidate", GROK_SOURCE)
        training_seconds = math.nan
        predict_started = fit_started
        predictions = candidate.run(
            str(worker_path / "train.csv"),
            str(worker_path / "test.csv"),
        )
    elif method_id == "tabfm_ensemble_regressor":
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

        model = tabfm_v1_0_0_pytorch.load(model_type="regression", device="cuda")
        estimator = TabFMRegressor.ensemble(model=model)
        estimator.fit(x_train, y_train)
        training_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predictions = estimator.predict(x_test)
    elif method_id == "tabpfn_v3_regressor":
        from tabpfn import TabPFNRegressor

        estimator = TabPFNRegressor(
            model_path=str(TABPFN_CHECKPOINT),
            device="cuda",
            random_state=20260621,
            ignore_pretraining_limits=True,
        )
        estimator.fit(x_train, y_train)
        training_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predictions = estimator.predict(x_test)
    elif method_id == "xgboost_hist_trees900_depth4_lr0p02":
        sys.path.insert(0, "/home/ubuntu/remote_super/quant")
        from models.tools.model_zoo import _xgb_regression

        estimator = _xgb_regression(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.02,
        )
        estimator.fit(x_train, y_train)
        training_seconds = time.monotonic() - fit_started
        predict_started = time.monotonic()
        predictions = estimator.predict(x_test)
    else:
        raise ValueError(method_id)
    ended = time.monotonic()
    if math.isnan(training_seconds):
        training_seconds = ended - fit_started
        inference_seconds = 0.0
    else:
        inference_seconds = ended - predict_started
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    metadata = {
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "python": platform.python_version(),
        "executable": sys.executable,
        "host": platform.node(),
        "distributions": distribution_versions(method_id),
    }
    if method_id in CHIMERA_METHODS:
        metadata["chimera_checkpoint"] = chimera_identity
    return values, metadata


def distribution_versions(method_id: str) -> dict[str, str | None]:
    names = {"numpy", "pandas", "scikit-learn"}
    names.update(
        {
            "autogluon_300s": {"autogluon.tabular"},
            "chimera_regression": {"torch", "safetensors"},
            "hyp3_chimera_regression": {"torch", "safetensors"},
            "grok4p5_regressor": set(),
            "tabfm_ensemble_regressor": {"torch"},
            "tabpfn_v3_regressor": {"tabpfn", "torch"},
            "xgboost_hist_trees900_depth4_lr0p02": {"xgboost"},
        }[method_id]
    )
    versions = {}
    for name in sorted(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def backend_worker(method_id: str, worker_path: Path, output_path: Path) -> dict[str, Any]:
    predictions, metadata = backend_predict(method_id, worker_path)
    manifest = read_json(worker_path / "worker-manifest.json")
    if predictions.shape != (manifest["test_rows"],):
        raise ValueError(
            f"{method_id}: prediction row mismatch: {predictions.shape} "
            f"!= {(manifest['test_rows'],)}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{method_id}: non-finite predictions")
    atomic_save_npy(output_path / "predictions.npy", predictions)
    atomic_write_json(output_path / "backend.json", metadata)
    return {"rows": int(predictions.size), **metadata}


def backend_environment(method_id: str, device: str) -> dict[str, str]:
    environment = os.environ.copy()
    limits = dict(THREAD_LIMITS)
    if method_id == "tabfm_ensemble_regressor":
        for name in (
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            limits[name] = "3"
    environment.update(limits)
    if SPECIAL_SPECS[method_id]["resource"] in {"cpu", "cpu_light"}:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def run_special(
    dataset_id: str,
    method_id: str,
    *,
    worker_root: Path,
    artifact_root: Path,
    device: str,
) -> dict[str, Any]:
    worker_path = worker_root / dataset_id
    worker_manifest = validate_worker_view(worker_path, dataset_id)
    identity = special_identity(method_id)
    spec = SPECIAL_SPECS[method_id]
    if spec.get("checkpoint") and (
        not CHIMERA_SAFETENSOR.is_file() or not CHIMERA_CONVERSION_RECEIPT.is_file()
    ):
        return compatibility_blocker(
            dataset_id,
            method_id,
            artifact_root=artifact_root,
        )
    validate_source_files(spec)
    output = artifact_root / dataset_id / method_id
    output.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"golden46-{method_id}-") as temp:
        backend_output = Path(temp)
        command = [
            spec["interpreter"],
            str(SPECIAL_SOURCE_PATH),
            "backend-worker",
            "--method",
            method_id,
            "--worker-path",
            str(worker_path),
            "--output-path",
            str(backend_output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=4 * 60 * 60,
            env=backend_environment(method_id, device),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{method_id} backend failed ({completed.returncode}): "
                f"{completed.stdout[-8000:]}"
            )
        backend = read_json(backend_output / "backend.json")
        predictions = np.load(backend_output / "predictions.npy", allow_pickle=False)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if predictions.shape != (worker_manifest["test_rows"],):
        raise ValueError(f"{method_id}: prediction row mismatch")
    if not np.all(np.isfinite(predictions)):
        raise ValueError(f"{method_id}: non-finite predictions")
    prediction_path = output / "predictions.npy"
    atomic_save_npy(prediction_path, predictions)
    receipt = {
        "schema": "golden46.special-prediction-worker-receipt.v1",
        "campaign_id": CAMPAIGN_ID,
        "dataset_hash": dataset_id,
        "algorithm": method_id,
        "special_identity": identity,
        "worker_implementation": {
            "special_execution_source_sha256": sha256_file(SPECIAL_SOURCE_PATH),
        },
        "status": "predicted-unscored",
        "started_at": started_at,
        "ended_at": utc_now(),
        "training_time_seconds": backend["training_time_seconds"],
        "inference_time_seconds": backend["inference_time_seconds"],
        "wall_time_seconds": time.monotonic() - started,
        "train_rows": worker_manifest["train_rows"],
        "test_rows": worker_manifest["test_rows"],
        "n_features": worker_manifest["n_features"],
        "worker_manifest_sha256": sha256_file(worker_path / "worker-manifest.json"),
        "predictions": {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
            "array_sha256": sha256_array(predictions),
            "rows": int(predictions.size),
        },
        "environment": backend,
        "test_labels_available_to_worker": False,
        "metrics_computed_by_worker": False,
    }
    atomic_write_json(output / "worker-receipt.json", receipt)
    return receipt


def finalize_special(
    dataset_id: str,
    method_id: str,
    *,
    canonical_root: Path,
    worker_root: Path,
    artifact_root: Path,
    controller_root: Path,
) -> dict[str, Any]:
    output = artifact_root / dataset_id / method_id
    receipt_path = output / "worker-receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("dataset_hash") != dataset_id or receipt.get("algorithm") != method_id:
        raise ValueError("worker receipt changed method or dataset identity")
    if receipt.get("special_identity") != special_identity(method_id):
        raise ValueError("worker method/checkpoint identity differs from the frozen identity")
    expected_source = sha256_file(SPECIAL_SOURCE_PATH)
    if receipt.get("worker_implementation", {}).get(
        "special_execution_source_sha256"
    ) != expected_source:
        raise ValueError("worker implementation differs from the frozen implementation")
    worker_manifest_sha = sha256_file(worker_root / dataset_id / "worker-manifest.json")
    if receipt.get("worker_manifest_sha256") != worker_manifest_sha:
        raise ValueError("worker receipt does not match the masked worker view")
    prediction_path = output / Path(receipt["predictions"]["path"]).name
    if sha256_file(prediction_path) != receipt["predictions"]["sha256"]:
        raise ValueError("prediction artifact hash mismatch")
    predictions = np.load(prediction_path, allow_pickle=False).astype(np.float64).reshape(-1)
    if sha256_array(predictions) != receipt["predictions"]["array_sha256"]:
        raise ValueError("prediction array content mismatch")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("non-finite predictions")
    source, metadata = canonical_dataset(dataset_id, canonical_root)
    target = read_canonical_frame(source, metadata, "test").iloc[:, -1].to_numpy(dtype=np.float64)
    if predictions.shape != target.shape:
        raise ValueError(f"prediction row mismatch: {predictions.shape} != {target.shape}")
    labels = read_json(controller_root / dataset_id / "labels.json")
    if labels["test_target_sha256"] != sha256_array(target):
        raise ValueError("canonical labels changed after controller preparation")
    metrics = regression_metrics(target, predictions)
    report = {
        "schema": "golden46.validated-report.v1",
        "campaign_id": CAMPAIGN_ID,
        "algorithm": method_id,
        "model_name": method_id,
        "dataset_hash": dataset_id,
        "task_type": "regression",
        "status": "completed",
        "benchmark_valid": True,
        "run_date": utc_now(),
        "train_rows": receipt["train_rows"],
        "test_rows": receipt["test_rows"],
        "n_features": receipt["n_features"],
        "training_time_seconds": receipt["training_time_seconds"],
        "inference_time_seconds": receipt["inference_time_seconds"],
        "metrics": metrics,
        "provenance": {
            "worker_receipt": str(receipt_path),
            "worker_receipt_sha256": sha256_file(receipt_path),
            "predictions_sha256": receipt["predictions"]["sha256"],
            "canonical_test_sha256": sha256_file(source / "test.csv"),
            "canonical_test_target_sha256": sha256_array(target),
            "controller_computed_metrics": True,
            "worker_had_test_labels": False,
        },
        "environment": {"controller_host": platform.node()},
    }
    atomic_write_json(output / "report.json", report)
    verify_special(
        dataset_id,
        method_id,
        canonical_root=canonical_root,
        artifact_root=artifact_root,
    )
    return report


def verify_special(
    dataset_id: str,
    method_id: str,
    *,
    canonical_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    output = artifact_root / dataset_id / method_id
    report = read_json(output / "report.json")
    receipt = read_json(output / "worker-receipt.json")
    predictions = np.load(output / "predictions.npy", allow_pickle=False).astype(np.float64)
    source, metadata = canonical_dataset(dataset_id, canonical_root)
    target = read_canonical_frame(source, metadata, "test").iloc[:, -1].to_numpy(dtype=np.float64)
    recomputed = regression_metrics(target, predictions.reshape(-1))
    if report.get("dataset_hash") != dataset_id or report.get("algorithm") != method_id:
        raise ValueError("report identity mismatch")
    if report.get("provenance", {}).get("worker_receipt_sha256") != sha256_file(
        output / "worker-receipt.json"
    ):
        raise ValueError("report worker receipt hash mismatch")
    if report.get("metrics") != recomputed:
        raise ValueError("score/report disagreement")
    if receipt["predictions"]["sha256"] != sha256_file(output / "predictions.npy"):
        raise ValueError("prediction artifact changed")
    return {
        "dataset_id": dataset_id,
        "method_id": method_id,
        "rmse": recomputed["rmse"],
        "verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--controller-root", type=Path, default=DEFAULT_CONTROLLER_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--device", default="0")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--dataset", required=True)
    finalize.add_argument("--method", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--method", required=True)
    backend = commands.add_parser("backend-worker")
    backend.add_argument("--method", required=True)
    backend.add_argument("--worker-path", type=Path, required=True)
    backend.add_argument("--output-path", type=Path, required=True)
    identity = commands.add_parser("identity")
    identity.add_argument("--method", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        result = run_special(
            args.dataset,
            args.method,
            worker_root=args.worker_root,
            artifact_root=args.artifact_root,
            device=args.device,
        )
    elif args.command == "finalize":
        result = finalize_special(
            args.dataset,
            args.method,
            canonical_root=args.canonical_root,
            worker_root=args.worker_root,
            artifact_root=args.artifact_root,
            controller_root=args.controller_root,
        )
    elif args.command == "verify":
        result = verify_special(
            args.dataset,
            args.method,
            canonical_root=args.canonical_root,
            artifact_root=args.artifact_root,
        )
    elif args.command == "backend-worker":
        result = backend_worker(args.method, args.worker_path, args.output_path)
    else:
        result = special_identity(args.method)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
