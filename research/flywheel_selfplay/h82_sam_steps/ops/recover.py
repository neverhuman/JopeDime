#!/usr/bin/env python3
"""One-shot H82 operator continuation; numerical worker stays at its frozen commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

WORKER_COMMIT = "037f914a59aabd34279000c40288315dc00500d8"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
CPU_PR = 897
SOURCE_PATH = "research/flywheel_selfplay/h82_sam_steps/ops/recover.py"
PREFIX_SHA = "a69d385a3887b3b4c4237bd1b129070275190a7c3515560ce1c43da069a0bb1c"
LOG_SHA = "d711c58391ae879195e81f56872ed6c0772e30df75d964e09d963faeee9dfae6"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prefix(root, *, expected_sha=PREFIX_SHA, expected_log_sha=LOG_SHA):
    if sha(root / "budget.json") != expected_sha:
        raise RuntimeError("not the reviewed stopped H82 prefix; no reset/resume")
    ledger = json.loads((root / "budget.json").read_text())
    rows = ledger["attempts"]
    if ledger["in_progress"] or ledger["cap_seconds"] != 2700 or len(rows) != 5:
        raise RuntimeError("unknown launch or unexpected cap/attempt inventory")
    a, b = rows[-2:]
    if any(
        row["arm"] != "SAM"
        or row["seed"] != 24
        or row.get("outcome_started") is not False
        for row in (a, b)
    ):
        raise RuntimeError("recovery cannot follow any computed/unknown outcome")
    if (
        a["attempt"] != 1
        or a["error_kind"] != "host_load"
        or b["attempt"] != 2
        or b["error_kind"] != "fatal"
        or b["stage"] != "loading"
    ):
        raise RuntimeError("not the qualifying load void and pre-model setup fault")
    if sha(root / "SAM-seed24-attempt2.log") != expected_log_sha:
        raise RuntimeError("staging traceback changed")
    if any(
        p.name.startswith("SAM-seed24")
        and p.name.endswith(".json")
        and not p.name.endswith(".phase.json")
        for p in root.iterdir()
    ):
        raise RuntimeError("SAM outcome already exists; never retry it")
    allowed_results = {
        "timing-K8-seed24-attempt1.json",
        "CONTROL-seed24-attempt1.json",
        "K8-seed24-attempt1.json",
    }
    allowed_phases = {
        f"{'timing-' if row.get('timing') else ''}{row['arm']}-seed{row['seed']}-attempt{row['attempt']}.phase.json"
        for row in rows
    }
    if any(
        p.name not in allowed_results | allowed_phases
        for p in root.glob("*-seed*.json")
    ):
        raise RuntimeError("unaccounted outcome/phase files; refusing continuation")
    if any(
        p.name
        not in {"CONTROL-seed24-attempt1.adapter.pt", "K8-seed24-attempt1.adapter.pt"}
        for p in root.glob("*.adapter.pt")
    ):
        raise RuntimeError("unaccounted adapter artifact; refusing continuation")
    if (root / "metrics.json").exists() or ledger.get("operator_recovery"):
        raise RuntimeError("already recovered/completed")
    return ledger


def clear_staging(root, custody, *, expected_sha=R882_SHA):
    scratch = root / "scratch"
    try:
        scratch_stat = scratch.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(scratch_stat.st_mode) or scratch.is_symlink():
        raise RuntimeError("staging directory is not a real directory")
    staged = scratch / "r882_adapter_copy.pt"
    if not staged.exists() and not staged.is_symlink():
        return False
    if (
        not staged.resolve().is_relative_to(root.resolve())
        or staged.is_symlink()
        or not stat.S_ISREG(staged.lstat().st_mode)
        or sha(staged) != expected_sha
    ):
        raise RuntimeError("unknown staged input; refusing to remove")
    custody.mkdir(parents=True, exist_ok=True)
    archive = custody / "preserved-stale-r882-copy.pt"
    if not archive.exists():
        shutil.copy2(staged, archive)
    if sha(archive) != expected_sha:
        raise RuntimeError("staged input recovery copy is not verified")
    staged.unlink()  # Exact duplicate of the pinned immutable input, never learned weights.
    return True


def load_worker(path):
    source = path / "research/flywheel_selfplay/h82_sam_steps/run_wave.py"
    spec = importlib.util.spec_from_file_location("h82_frozen_operator_worker", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_launcher(worker, review_pr):
    info = json.loads(
        subprocess.check_output(
            [
                "/home/ubuntu/.local/bin/gh",
                "pr",
                "view",
                str(review_pr),
                "--repo",
                "neverhuman/JopeDime",
                "--json",
                "mergedAt,mergeCommit",
            ],
            text=True,
        )
    )
    if not info["mergedAt"]:
        raise RuntimeError("operator recovery PR must merge before continuation")
    revision = info["mergeCommit"]["oid"]
    approved = subprocess.check_output(
        ["git", "show", f"{revision}:{SOURCE_PATH}"], cwd=worker
    )
    if hashlib.sha256(approved).hexdigest() != sha(Path(__file__)):
        raise RuntimeError("launcher differs from reviewed merged bytes")
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=worker, text=True
        ).strip()
        != WORKER_COMMIT
    ):
        raise RuntimeError(
            "numerical worker checkout is not frozen at the original source"
        )
    return revision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-pr", type=int, required=True)
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path("/home/ubuntu/JopeDime/.agent/worktrees/h82"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path.home() / "jope-data/h82/h82-wave"
    )
    parser.add_argument(
        "--custody", type=Path, default=Path.home() / "jope_plan/evidence/H82/recovery"
    )
    args = parser.parse_args()
    revision = check_launcher(args.worker, args.review_pr)
    w = load_worker(args.worker)
    w.validate_run_root(args.root)
    w.check_dependencies(CPU_PR)
    with (args.root / "coordinator.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = validate_prefix(args.root)
        before = json.dumps(ledger["attempts"], sort_keys=True)
        cells = []
        for row in ledger["attempts"]:
            if row["timing"] or row.get("error"):
                continue
            out = Path(row["command"][row["command"].index("--out") + 1])
            if sha(out) != row["receipt_sha256"]:
                raise RuntimeError("earlier complete receipt changed")
            cell = json.loads(out.read_text())
            if cell["source_commit"] != WORKER_COMMIT or cell["wave"] != "H82":
                raise RuntimeError("earlier cell source mismatch")
            cells.append(cell)
        if {(c["arm"], c["seed"]) for c in cells} != {("CONTROL", 24), ("K8", 24)}:
            raise RuntimeError("unexpected complete prefix; no outcome may be rerun")
        last = ledger["attempts"][-1]
        ended = (
            datetime.fromisoformat(last["started_utc"]).timestamp()
            + last["wall_seconds"]
        )
        downtime = max(0.0, datetime.now(timezone.utc).timestamp() - ended)
        if downtime + sum(a["wait_seconds"] for a in ledger["attempts"]) >= 10800:
            raise RuntimeError("original cumulative wait allowance exhausted")
        # Charge operator downtime conservatively by lowering remaining wait allowance.
        w.fair.MAX_LOCK_WAIT_SECONDS = 10800 - downtime
        ledger["operator_recovery"] = {
            "source_commit": revision,
            "launcher_sha256": sha(Path(__file__)),
            "worker_commit": WORKER_COMMIT,
            "original_prefix_sha256": PREFIX_SHA,
            "setup_log_sha256": LOG_SHA,
            "operator_downtime_seconds_charged_to_wait": downtime,
            "sam24_final_attempt": 3,
            "reason": "verified read-only staged input prevented retry2 before model construction",
        }
        if json.dumps(ledger["attempts"], sort_keys=True) != before:
            raise RuntimeError("prior accounting changed")
        w.r.atomic_json(args.root / "budget.json", ledger)

        def attempt(root, current, arm, seed, timing, index):
            clear_staging(root, args.custody)
            return w.run_attempt(root, current, arm, seed, timing, index)

        try:
            cell, _ = attempt(args.root, ledger, "SAM", 24, False, 3)
            if cell is None:
                raise RuntimeError("third/final SAM attempt cannot retry")
            cells.append(cell)
            for seed in w.r.SEEDS:
                for arm in w.r.ARMS:
                    if (arm, seed) in {("CONTROL", 24), ("K8", 24), ("SAM", 24)}:
                        continue
                    cell, _ = w.execute(
                        args.root, ledger, arm, seed, attempt_fn=attempt
                    )
                    cells.append(cell)
            result = w.r.summarize(cells)
            result.update(
                ledger=ledger,
                gpu_wall_seconds=sum(a["wall_seconds"] for a in ledger["attempts"]),
            )
            w.r.atomic_json(args.root / "metrics.json", result)
            w.fair.note(
                args.root / "H82_STATUS.log",
                "COMPLETE all12 paired cells; original numerical worker; bounded operator recovery",
            )
        except BaseException as exc:
            w.r.atomic_json(
                args.root / "incomplete-operator-recovery.json",
                {"complete": False, "reason": repr(exc), "ledger": ledger},
            )
            w.fair.note(
                args.root / "H82_STATUS.log",
                f"STOP operator recovery {exc}; no additional retries",
            )
            raise


if __name__ == "__main__":
    main()
