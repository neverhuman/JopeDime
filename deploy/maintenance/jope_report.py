#!/usr/bin/env python3
"""Report Cargo candidates and allocated disk usage without changing scanned files."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

APPROVED_ROOTS = ["/home/ubuntu/JopeDime", "/home/ubuntu/jain-split/jope-dime"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fail_walk(error):
    raise error


def walk(root):
    # Even an explicitly supplied root must not introduce a symlink scan root.
    root = Path(os.path.abspath(root))
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError(f"root must be an existing directory without symlink components: {root}")
    return os.walk(root, followlinks=False, onerror=fail_walk)


def allocated(root, privileged=False):
    command = ["/usr/bin/sudo", "-n", "/usr/bin/du"] if privileged else ["/usr/bin/du"]
    result = subprocess.run(
        [*command, "-s", "-B1", "--", str(root)], capture_output=True, text=True,
        check=True, timeout=600,
    )
    return int(result.stdout.split("\t", 1)[0])


def scan(root, hours):
    root = Path(os.path.abspath(root))
    candidates = set()
    for directory, dirs, files in walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        path = Path(directory)
        marker = path / ".rustc_info.json"
        if ".rustc_info.json" in files and stat.S_ISREG(marker.lstat().st_mode):
            candidates.add(path)
        if path.name in ("debug", "release") and ".fingerprint" in dirs:
            if (path.parent == root or root in path.parent.parents) and \
                    stat.S_ISDIR((path / ".fingerprint").lstat().st_mode):
                candidates.add(path.parent)
    rows = []
    cutoff = time.time() - hours * 3600 if hours is not None else None
    for candidate in sorted(candidates):
        if any(parent in candidates for parent in candidate.parents):
            continue
        newest = candidate.lstat().st_mtime
        # Measure all contents, including nested .git evidence, after identifying candidates.
        # Visit fresh trees too so traversal failures stay visible.
        for directory, dirs, files in walk(candidate):
            for name in dirs + files:
                newest = max(newest, (Path(directory) / name).lstat().st_mtime)
        if cutoff is not None and newest > cutoff:
            continue
        rows.append({"path": str(candidate), "allocated_bytes": allocated(candidate),
                     "newest_mtime": newest})
    return {"completed_at": utc(), "root": os.path.abspath(root), "stale_hours": hours,
            "candidates": rows,
            "candidate_allocated_bytes_sum": sum(r["allocated_bytes"] for r in rows),
            "sum_may_double_count_hardlinks": True,
            # This object is returned only after every traversal and measurement succeeds.
            "report_only": True, "complete": True}


def whole_hours(value):
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("--stale requires nonnegative whole hours")
    # Bound arithmetic, not the reporting/deletion policy (there is no deletion policy).
    if len(value) > 8:
        raise argparse.ArgumentTypeError("--stale age is too large")
    return int(value)


def interrupted(signum, _frame):
    raise RuntimeError(f"report interrupted by signal {signum}")


def report(args):
    # Custody is a service argument so bad configuration can still receive a failure receipt.
    output = args.custody.resolve()
    if Path("/tmp") in [output, *output.parents] or any(
            (parent / ".git").exists() for parent in [output, *output.parents]):
        raise ValueError("report custody must be outside /tmp and Git working trees")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = output / stamp
    run.mkdir(mode=0o700)
    receipt = {"started_at": utc(), "status": "running", "exit_status": None,
               "checksums": {}, "measurements": [], "scan_outputs": [],
               "invocation_id": os.environ.get("INVOCATION_ID")}
    source = Path(__file__).resolve().parent
    receipt_path = run / "receipt.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        paths = [Path(__file__).resolve(), source / "cargo-build-dirs.sh",
                 source / "jope-build-report.sh"]
        units = Path.home() / ".config/systemd/user"
        paths.extend(p for p in [units / "jope-build-sweep.service",
                                units / "jope-build-sweep.timer"] if p.is_file())
        for path in paths:
            receipt["checksums"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        config_path = args.config.resolve(strict=True)
        config_bytes = config_path.read_bytes()
        receipt["checksums"][str(config_path)] = hashlib.sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes)
        if not isinstance(config, dict) or not {"cargo_roots", "stale_hours",
                                               "measurement_roots"} <= config.keys():
            raise ValueError("config requires cargo_roots, stale_hours and measurement_roots")
        receipt["config"] = config
        roots = config["cargo_roots"]
        if roots != APPROVED_ROOTS or config["stale_hours"] != 48:
            raise ValueError("service requires the two approved Cargo roots and a 48-hour filter")
        if not isinstance(config["measurement_roots"], list):
            raise ValueError("measurement_roots must be a list")
        seen = []
        for item in config["measurement_roots"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or \
                    not isinstance(item.get("group"), str) or \
                    not isinstance(item.get("privileged", False), bool):
                raise ValueError("each measurement requires string path/group and optional boolean privileged")
            path = Path(item["path"])
            resolved = path.resolve(strict=True)
            if resolved != path or not resolved.is_dir():
                raise ValueError(f"measurement roots must be explicit resolved directories: {path}")
            if any(resolved == prior or prior in resolved.parents or resolved in prior.parents
                   for prior in seen):
                raise ValueError(f"overlapping measurement root: {path}")
            seen.append(resolved)
        for index, root in enumerate(roots):
            result_path = run / f"candidates-{index}.json"
            diagnostic = run / f"scan-{index}.stderr"
            receipt["scan_outputs"].append({"root": root, "output": result_path.name,
                                            "stderr": diagnostic.name})
            diagnostic.write_text("")
            try:
                result_path.write_text(json.dumps(scan(root, 48), indent=2) + "\n")
            except Exception as error:
                diagnostic.write_text(str(error) + "\n" + (getattr(error, "stderr", "") or ""))
                raise
        for item in config["measurement_roots"]:
            resolved = Path(item["path"])
            receipt["measurements"].append({**item, "allocated_bytes": allocated(
                resolved, item.get("privileged", False)),
                                            "observed_at": utc()})
            save()
        result = subprocess.run(["df", "-B1", str(output)], capture_output=True,
                                text=True, check=True, timeout=30)
        (run / "filesystem.txt").write_text(result.stdout)
        receipt.update(status="success", exit_status=0)
    except Exception as error:
        receipt.update(status="failed", exit_status=1, error=str(error))
        (run / "failure.txt").write_text(str(error) + "\n")
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            with (run / "failure.txt").open("a") as out:
                out.write(error.stderr)
    finally:
        receipt["completed_at"] = utc()
        save()
    print(receipt_path)
    return receipt["exit_status"]


def main():
    if "--delete" in sys.argv:
        print("Deletion is disabled. Use a logged preservation and independent verification batch.",
              file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    scanner = sub.add_parser("scan")
    scanner.add_argument("root")
    scanner.add_argument("--stale", type=whole_hours)
    wrapper = sub.add_parser("report")
    wrapper.add_argument("--config", type=Path, required=True)
    wrapper.add_argument("--custody", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "report":
            return report(args)
        print(json.dumps(scan(args.root, args.stale), indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"report failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
