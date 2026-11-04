#!/usr/bin/env python3
"""Fetch, validate, ingest, and atomically index strict test-only receipts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from test_exact_common import (
    atomic_json,
    is_strict_receipt,
    manifest_authority_kind,
    receipt_authority_kind,
    sha256_file,
    sha256_text,
)
from test_exact_pair_registry import PairRegistry, protocol_for_manifest

from test_exact_ingest_constants import (
    SCQ, ROOT, BACKLOG, CORPUS, RECEIPTS, SCQ_RUNS, HOF_POINTER,
    GOLDEN42, CATALOG_POINTER,
)
from test_exact_ingest_receipts import (
    command, scheduler_jobs, corpus_db as _corpus_db, receipt_dir, local_receipt_dir,
    validate_and_ingest,
)
from test_exact_ingest_reconcile import reconcile, rebuild_indexes, import_hof


def corpus_db() -> sqlite3.Connection:
    return _corpus_db(CORPUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    reconcile_command = commands.add_parser("reconcile")
    reconcile_command.add_argument("--limit", type=int, default=100)
    commands.add_parser("rebuild-indexes")
    import_command = commands.add_parser("import-hof")
    import_command.add_argument(
        "--task",
        action="append",
        choices=("regression", "binary"),
        dest="tasks",
        required=True,
    )
    args = parser.parse_args()
    if args.command == "reconcile":
        print(json.dumps(reconcile(args.limit), sort_keys=True))
    elif args.command == "import-hof":
        print(json.dumps(import_hof(sorted(set(args.tasks))), sort_keys=True))
    else:
        print(json.dumps(rebuild_indexes(), sort_keys=True))


if __name__ == "__main__":
    main()
