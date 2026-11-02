#!/usr/bin/env python3
"""Emit the minimal 28-gate receipt format consumed by signer_watch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gates_receipt_assembly as _assembly
from gates_receipt_contract import (
    HEX_COMMIT,
    HEX_SHA256,
    M3_WAIVER_PROPOSER_SHA256,
    M3_WAIVER_RULING_SHA256,
    RELEASE_GATE_IDS,
    STATUSES,
)
from gates_receipt_inputs import _json, _rows, from_cpu_prestage, from_gate_results

_assembly_json = _assembly._assembly_json
_assembly_artifact = _assembly._assembly_artifact
_source_proposer_sha256 = _assembly._source_proposer_sha256
_within_operator_mix_limit = _assembly._within_operator_mix_limit
_assembly_controller = _assembly._assembly_controller
_assembly_candidate = _assembly._assembly_candidate
_controller_provenance = _assembly._controller_provenance


def _sync_assembly_compatibility() -> None:
    # Preserve the runner module's existing monkeypatch seams for receipt assembly.
    _assembly._assembly_json = _assembly_json
    _assembly._assembly_artifact = _assembly_artifact
    _assembly._source_proposer_sha256 = _source_proposer_sha256
    _assembly._within_operator_mix_limit = _within_operator_mix_limit
    _assembly._assembly_controller = _assembly_controller
    _assembly._assembly_candidate = _assembly_candidate
    _assembly._controller_provenance = _controller_provenance
    _assembly.M3_WAIVER_RULING_SHA256 = M3_WAIVER_RULING_SHA256
    _assembly.M3_WAIVER_PROPOSER_SHA256 = M3_WAIVER_PROPOSER_SHA256


def _m3_waiver(result: dict, ruling: Path, release_version: str) -> dict:
    _sync_assembly_compatibility()
    return _assembly._m3_waiver(result, ruling, release_version)


def from_scaffold(
    scaffold_path: Path,
    receipt_paths: list[Path],
    artifact: Path,
    source_sha: str,
    *,
    m3_waiver_ruling: Path | None = None,
    release_version: str | None = None,
    candidate_manifest: Path | None = None,
    controller_run: Path | None = None,
    controller_inputs: Path | None = None,
) -> dict:
    """Assemble existing PASS receipts; never execute the scaffold or sign bytes."""
    _sync_assembly_compatibility()
    return _assembly.from_scaffold(
        scaffold_path,
        receipt_paths,
        artifact,
        source_sha,
        m3_waiver_ruling=m3_waiver_ruling,
        release_version=release_version,
        candidate_manifest=candidate_manifest,
        controller_run=controller_run,
        controller_inputs=controller_inputs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--gate-results", type=Path)
    inputs.add_argument("--cpu-prestage", type=Path)
    inputs.add_argument(
        "--scaffold",
        type=Path,
        help="assemble existing receipts, without executing commands",
    )
    parser.add_argument("--gate-receipt", type=Path, action="append", default=[])
    parser.add_argument(
        "--source-sha", help="exact source shared by all 28 gate receipts"
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--controller-run", type=Path)
    parser.add_argument("--controller-inputs", type=Path)
    parser.add_argument("--m3-waiver-ruling", type=Path)
    parser.add_argument("--release-version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.scaffold:
            if args.artifact is None or args.source_sha is None:
                raise ValueError("--scaffold requires --artifact and --source-sha")
            receipt = from_scaffold(
                args.scaffold,
                args.gate_receipt,
                args.artifact,
                args.source_sha,
                m3_waiver_ruling=args.m3_waiver_ruling,
                release_version=args.release_version,
                candidate_manifest=args.candidate_manifest,
                controller_run=args.controller_run,
                controller_inputs=args.controller_inputs,
            )
        else:
            if (
                args.gate_receipt
                or args.source_sha is not None
                or args.m3_waiver_ruling
                or args.candidate_manifest
                or args.release_version
                or args.controller_run
                or args.controller_inputs
            ):
                raise ValueError("--gate-receipt and --source-sha require --scaffold")
            receipt = (
                from_cpu_prestage(args.cpu_prestage)
                if args.cpu_prestage
                else from_gate_results(args.gate_results, args.artifact)
            )
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output must be a new path")
        if any(part.is_symlink() for part in args.output.parents):
            raise ValueError("output parents must not contain symlinks")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if any(part.is_symlink() for part in args.output.parents):
            raise ValueError("output parents must not contain symlinks")
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "gates": len(receipt["gates"]),
                "all_pass": all(g["status"] == "PASS" for g in receipt["gates"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
