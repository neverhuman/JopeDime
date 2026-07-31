#!/usr/bin/env python3
"""Prove HYP3 features are invariant to masked test-target sentinel values."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from golden46_execution import (
    CAMPAIGN_ROOT,
    DEFAULT_WORKER_ROOT,
    HYP3_BINARY,
    HYP3_MANIFEST,
    HYP3_OUTPUTS,
    HYP3_WEIGHTS,
    atomic_write_json,
    sha256_file,
    utc_now,
)


def invoke(train: Path, test: Path, output: Path) -> None:
    command = [
        str(HYP3_BINARY),
        "--train",
        str(train),
        "--test",
        str(test),
        "--out-cell-train",
        str(output / "train3.vs"),
        "--out-cell-test",
        str(output / "test3.vs"),
        "--out-row-train",
        str(output / "hyperion3train.vs"),
        "--out-row-test",
        str(output / "hyperion3test.vs"),
        "--weights",
        str(HYP3_WEIGHTS),
        "--manifest",
        str(HYP3_MANIFEST),
        "--sweeps",
        "3",
        "--scales",
        "4,7,11,15",
    ]
    output.mkdir(parents=True)
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def validate(dataset_id: str) -> dict[str, Any]:
    worker = DEFAULT_WORKER_ROOT / dataset_id
    train = worker / "train.csv"
    test = pd.read_csv(worker / "test.csv", header=None)
    with tempfile.TemporaryDirectory(prefix="golden46-hyp3-invariance-") as temporary:
        root = Path(temporary)
        test_zero = root / "test-zero.csv"
        test_changed = root / "test-changed.csv"
        zero = test.copy()
        zero.iloc[:, -1] = 0.0
        changed = test.copy()
        changed.iloc[:, -1] = [
            17.0 + index / max(len(changed), 1) for index in range(len(changed))
        ]
        zero.to_csv(test_zero, header=False, index=False, float_format="%.17g")
        changed.to_csv(test_changed, header=False, index=False, float_format="%.17g")
        output_zero = root / "zero"
        output_changed = root / "changed"
        invoke(train, test_zero, output_zero)
        invoke(train, test_changed, output_changed)
        artifacts = {}
        for name in HYP3_OUTPUTS:
            zero_hash = sha256_file(output_zero / name)
            changed_hash = sha256_file(output_changed / name)
            if zero_hash != changed_hash:
                raise ValueError(f"HYP3 output changed with masked test target: {name}")
            artifacts[name] = zero_hash
    return {
        "schema": "golden46.hyp3-target-invariance.v1",
        "created_at": utc_now(),
        "dataset_id": dataset_id,
        "status": "validated",
        "test_target_variants": ["constant-zero", "nonconstant-17-plus-row-fraction"],
        "artifacts": artifacts,
        "implementation": {
            "binary_sha256": sha256_file(HYP3_BINARY),
            "weights_sha256": sha256_file(HYP3_WEIGHTS),
            "manifest_sha256": sha256_file(HYP3_MANIFEST),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ZTS_small")
    args = parser.parse_args()
    result = validate(args.dataset)
    output = CAMPAIGN_ROOT / "manifests" / f"{args.dataset}.hyp3-invariance.json"
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
