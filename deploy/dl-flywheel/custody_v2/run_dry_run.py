#!/usr/bin/env python3
"""Read exact S3 versions into a conservative ledger and a local verified sample pack."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from custody_v2.read_only import ReadOnlyS3
from custody_v2.retention import (
    budget_status,
    canonical,
    classify,
    cost_report,
    digest,
    need,
    pack_local,
    verify_local,
)

PROFILE = "veox-jepson-prod"
ACCOUNT = "714044927654"
BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/objects/"


def verify_saved(manifest):
    manifest = Path(manifest)
    index = json.loads(manifest.read_bytes())
    report = verify_local(manifest.parent / (index["pack_sha256"] + ".tar.gz"), index)
    report["manifest_sha256"] = digest(manifest.read_bytes())
    report["remote_calls"] = 0
    print(json.dumps(report, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rates", type=Path)
    parser.add_argument("--max-members", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--verify-local", type=Path)
    args = parser.parse_args(argv)
    if args.verify_local:
        need(
            args.output is None and args.rates is None,
            "verification has no remote/output arguments",
        )
        verify_saved(args.verify_local)
        return 0
    need(
        args.output is not None
        and args.output.is_absolute()
        and args.rates is not None,
        "absolute outside-Git output and rate card required",
    )
    need(1 <= args.max_members <= 512, "bounded member sample")
    need(
        math.isfinite(args.seconds) and 0 < args.seconds <= 900,
        "claimed finite read window, at most 900 seconds",
    )
    checkout = Path(__file__).resolve().parents[3]
    worktrees = subprocess.check_output(
        ["/usr/bin/git", "worktree", "list", "--porcelain", "-z"], cwd=checkout
    )
    for record in worktrees.split(b"\0"):
        if record.startswith(b"worktree "):
            tree = Path(os.fsdecode(record[len(b"worktree ") :])).resolve()
            need(
                not args.output.resolve().is_relative_to(tree),
                "bulk custody output must be outside all registered checkouts",
            )
    rates = json.loads(args.rates.read_bytes())
    args.output.mkdir(mode=0o700)
    started = datetime.now(timezone.utc)
    report = {
        "schema": "jope.custody-v2-readonly-dry-run/v1",
        "started_at": started.isoformat(),
        "bucket": BUCKET,
        "prefix": PREFIX,
        "profile": PROFILE,
        "qualified": False,
        "remote_mutations": 0,
        "complete": False,
        "actual_billed_usd": None,
        "rate_card_sha256": digest(args.rates.read_bytes()),
        "rate_catalog_sha256": rates.get("catalog_sha256"),
        "rate_catalog_url": rates.get("catalog_url"),
    }
    report["source_sha"] = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    report["source_dirty_paths"] = subprocess.check_output(
        [
            "/usr/bin/git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            str(Path(__file__).parent),
        ],
        cwd=checkout,
        text=True,
    ).splitlines()
    need(
        not report["source_dirty_paths"],
        "freeze the executed custody source before remote reads",
    )
    reader = None
    try:
        # Local pack verification above does not import a credential provider or an S3 SDK.
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=PROFILE, region_name="us-east-2")
        config = Config(
            connect_timeout=10, read_timeout=15, retries={"total_max_attempts": 1}
        )
        who = session.client("sts", config=config).get_caller_identity()
        need(who["Account"] == ACCOUNT, "verified product AWS account")
        report["principal"] = {"account": who["Account"], "arn": who["Arn"]}
        report["sdk_version"] = boto3.__version__
        reader = ReadOnlyS3(
            session.client("s3", config=config), BUCKET, PREFIX, seconds=args.seconds
        )
        census = reader.census()
        rows = census["rows"]
        ledger = [classify(row) for row in rows]
        ledger_raw = canonical(ledger)
        (args.output / "object-ledger.json").write_bytes(ledger_raw)
        (args.output / "delete-marker-observations.json").write_bytes(
            canonical(census["delete_markers"])
        )
        current = [r for r in rows if r["is_latest"]]
        cutoff = started - timedelta(hours=24)
        recent = sum(
            r["bytes"]
            for r in rows
            if datetime.fromisoformat(r["last_modified"]) >= cutoff
        )
        report.update(
            {
                "listing": {
                    k: v
                    for k, v in census.items()
                    if k not in ("rows", "delete_markers")
                },
                "listed_versions": len(rows),
                "current_objects": len(current),
                "all_version_bytes": sum(r["bytes"] for r in rows),
                "current_object_bytes": sum(r["bytes"] for r in current),
                "noncurrent_versions": len(rows) - len(current),
                "delete_markers": len(census["delete_markers"]),
                "ledger_sha256": digest(ledger_raw),
                "retention_classes": dict(
                    Counter(r["retention_class"] for r in ledger)
                ),
                "keeper_reasons": dict(Counter(r["reason"] for r in ledger)),
                "guard": budget_status(recent, None, None),
                "new_version_bytes_24h_observed": recent,
                "net_new_bytes_24h": None,
                "net_new_note": "listing interval is not atomic; deleted versions and write/PUT counters are unmeasured",
            }
        )
        # Fixed lexical selection before body reads; no executable downloads or score-dependent selection.
        eligible = [
            r
            for r in current
            if r["storage_class"] == "STANDARD"
            and r["version_id"] != "null"
            and r["key"].rsplit("/", 1)[-1]
            in ("score_records", "evaluation", "rust_parity", "record.json")
            and r["bytes"] <= 64 * 1024
        ]
        selected = eligible[: args.max_members]
        need(selected, "at least one exact small-artifact version for the dry-run pack")
        members = [(row, reader.capture(row)) for row in selected]
        report["pack_sample"] = pack_local(args.output / "pack", members)
        env = os.environ.copy()
        for key in tuple(env):
            if key.startswith("AWS_") or key in (
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "JOPE_LEARNING_URL",
                "JOPE_DATASETS_URL",
            ):
                env.pop(key)
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--verify-local",
                report["pack_sample"]["manifest"],
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        readback = json.loads(child.stdout)
        need(
            readback["manifest_sha256"] == report["pack_sample"]["manifest_sha256"],
            "independent reader authenticates manifest",
        )
        report["independent_readback"] = readback
        report["pack_scope"] = {
            "selected": len(selected),
            "eligible": len(eligible),
            "unpacked_versions": len(rows) - len(selected),
            "entire_prefix_packed": False,
            "selection": "first lexical current standard small-artifact versions, at most 512",
        }
        report["cost"] = cost_report(rows, reader.requests, rates, started)
        reader.check()
        report["complete"] = True
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if reader is not None:
            report["s3_requests"] = reader.receipt()
        (args.output / "receipt.json").write_bytes(canonical(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
