"""Command-line interface for the three-client capture runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from capture_three_clients_contract import CLIENTS, CaptureError, self_test

# Bound by capture_three_clients.py after loading this CLI module.
run_capture: Callable[[argparse.Namespace], dict] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument(
        "--capture-source-sha",
        help="clean reviewed tooling checkout; candidate source remains --source-sha",
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--input-handoff", type=Path)
    parser.add_argument("--input-handoff-sha256")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--client-plan", type=Path)
    parser.add_argument("--engine-home", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--clients",
        nargs="+",
        choices=CLIENTS,
        help="named clients to run (default: all)",
    )
    parser.add_argument(
        "--exclude-reason", help="required explanation for each omitted client"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-named-clients", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("three-client capture plan controls: PASS (synthetic plan only)")
        return 0
    required = (
        args.repo,
        args.source_sha,
        args.candidate,
        args.candidate_sha256,
        args.input_handoff,
        args.input_handoff_sha256,
        args.train_path,
        args.client_plan,
        args.engine_home,
        args.out_dir,
    )
    if any(value is None for value in required):
        parser.error(
            "all candidate, TRAIN handoff, client plan and output bindings are required"
        )
    if not args.preflight_only and not args.execute_named_clients:
        parser.error(
            "pass --preflight-only or explicitly authorize --execute-named-clients"
        )
    try:
        if run_capture is None:
            raise RuntimeError("capture runner binding is unavailable")
        result = run_capture(args)
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if result["status"] == "PREFLIGHT_READY_NO_CLIENTS_LAUNCHED"
            or result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
            else 2
        )
    except (CaptureError, OSError, subprocess.SubprocessError) as error:
        print(f"three-client capture refused: {error}", file=sys.stderr)
        return 2

