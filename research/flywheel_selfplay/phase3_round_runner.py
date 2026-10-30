"""Finalize and count only rounds with all Phase 3 gates proven."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Mapping

from phase3_gatekeeper import GateError, qualify_round

EXPECTED_COVERAGE = {"regression": "2720/3603", "binary": "7/843", "datasets": 2727}


def qualify_phase3_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a GPU round has complete keyed-subset evidence."""
    if bundle.get("scope") != "keyed_subset":
        raise GateError("round bundle scope must be keyed_subset")
    fold_manifest = bundle.get("s3_fold_manifest")
    if not isinstance(fold_manifest, Mapping) or fold_manifest.get("scope") != "keyed_subset":
        raise GateError("missing owner-approved keyed-subset fold manifest")
    coverage = fold_manifest.get("coverage")
    if not isinstance(coverage, Mapping) or any(
        coverage.get(key) != value for key, value in EXPECTED_COVERAGE.items()
    ):
        raise GateError("keyed-subset coverage differs from owner decision")
    if fold_manifest.get("sealed_excluded_applied") is not True:
        raise GateError("sealed-family exclusion was not applied")
    if fold_manifest.get("sealed_partition_included") is not False:
        raise GateError("sealed partition must be excluded")
    result = qualify_round(
        winners=bundle["winners"],
        verification_results=bundle["verification_results"],
        s3_fold_manifest=fold_manifest,
        training_receipt=bundle["training_receipt"],
        dope_freeze_receipt=bundle["dope_freeze_receipt"],
        current_metrics=bundle["current_metrics"],
        previous_metrics=bundle["previous_metrics"],
    )
    elo = bundle.get("elo_report")
    if elo is None:
        elo_result = {
            "status": "unavailable",
            "reason": "no round-specific shared-panel method scores supplied",
        }
    else:
        if not isinstance(elo, Mapping) or elo.get("scope") != "keyed_subset":
            raise GateError("round Elo must be computed on the keyed_subset shared panel")
        if not isinstance(elo.get("regression_elo"), (float, int)) or not isinstance(
            elo.get("binary_elo"), (float, int)
        ):
            raise GateError("round Elo must include regression and binary ratings")
        elo_result = dict(elo)
    return {
        "schema": "jope.phase3.round_qualification.v1",
        "scope": "keyed_subset",
        "round": bundle.get("round"),
        "host": socket.gethostname(),
        "qualified": True,
        "qualified_rounds_increment": 1,
        "gates": {
            "verification": "PASS",
            "miss_to_pair_and_retraining": "PASS",
            "grouped_s3": "PASS",
            "comparison": "PASS",
            "dope_frozen": "PASS",
        },
        **result,
        "elo": elo_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "xbabe2":
        raise RuntimeError("qualified Phase 3 finalization must run on xbabe2")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("qualified Phase 3 finalization requires the xbabe2 GPU")
    bundle = json.loads(args.bundle.read_text())
    receipt = qualify_phase3_bundle(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"round": receipt["round"], "qualified": True, "scope": receipt["scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
