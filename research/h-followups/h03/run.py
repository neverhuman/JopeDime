#!/usr/bin/env python3
"""Validate and print the committed historical H03 aggregate; never replay it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
RESULTS = (
    ROOT
    / "docs/research/studies/dope-neural-ranking-20261021/early-history/"
    / "hypotheses-results.json"
)
EXPECTED_SOURCE_SHA256 = (
    "4da33283fc32b609166f2a59bfbf93407f26428921f3f881915d32405a87df20"
)


def historical_h03() -> dict[str, object]:
    """Return H03's saved aggregate after checking its pinned identity."""
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    studies = results.get("studies")
    if not isinstance(studies, list):
        raise ValueError("historical results have no studies list")
    matches = [study for study in studies if study.get("id") == "H03"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one H03 study, found {len(matches)}")
    study = matches[0]
    if study.get("source_artifact_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("H03 source artifact identity does not match the pin")
    return study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the saved aggregate without running a model",
    )
    args = parser.parse_args()
    if not args.dry_run:
        parser.error(
            "fresh replication is unavailable: the historical receipt contains "
            "no arrays/model or recoverable treatment and comparator recipe; "
            "only --dry-run is supported"
        )

    study = historical_h03()
    current = study["contrasts"]["function_vs_current"]
    a0 = study["contrasts"]["features_trained_function_vs_lambda_a0"]
    print(
        json.dumps(
            {
                "study": "H03",
                "fresh_replay": False,
                "source_artifact_sha256": study["source_artifact_sha256"],
                "contexts": study["contexts"],
                "families": study["families"],
                "historical_contrasts": {
                    "readout_aware_vs_current": current,
                    "readout_aware_features_vs_frozen_A0": a0,
                },
                "verdict": "NOT_REPLICATED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def test_pinned_historical_h03_contrasts_are_reported_without_replay() -> None:
    study = historical_h03()

    assert study["source_artifact_sha256"] == EXPECTED_SOURCE_SHA256
    assert study["contexts"] == 1594
    assert study["families"] == 379
    assert study["fits"] == 30

    current = study["contrasts"]["function_vs_current"]
    assert current["confidence"] == 0.975
    assert current["reduction_percent"] == 4.555989014108519
    assert current["ci_percent"] == [0.01750381930485131, 9.50556686823229]

    a0 = study["contrasts"]["features_trained_function_vs_lambda_a0"]
    assert a0["confidence"] == 0.975
    assert a0["reduction_percent"] == -0.7895247464571709
    assert a0["ci_percent"] == [-4.727096189114657, 3.1652944759961112]


def test_cli_requires_explicit_read_only_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "only --dry-run is supported" in result.stderr


if __name__ == "__main__":
    sys.exit(main())
