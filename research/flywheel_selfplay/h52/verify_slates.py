#!/usr/bin/env python3
"""Require exact training-index slate identity for one matched H52 pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    candidate = json.loads(args.candidate.read_text())
    if control["seed"] != candidate["seed"]:
        raise RuntimeError("matched slate check requires the same seed")
    left = [fold["training_slate_sha256_by_step"] for fold in control["folds"]]
    right = [fold["training_slate_sha256_by_step"] for fold in candidate["folds"]]
    if left != right:
        raise RuntimeError("matched CONTROL/objective training slates differ")
    print(
        json.dumps(
            {
                "seed": control["seed"],
                "folds": len(left),
                "steps_per_fold": [len(row) for row in left],
                "identical": True,
            }
        )
    )


if __name__ == "__main__":
    main()
