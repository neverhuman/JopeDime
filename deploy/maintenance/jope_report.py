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
    command = ["sudo", "-n", "/usr/bin/du"] if privileged else ["du"]
    result = subprocess.run(
        [*command, "-s", "-B1", "--", str(root)], capture_output=True, text=True,
        check=True, timeout=600,
    )
    return int(result.stdout.split("\t", 1)[0])


def scan(root, hours):
    candidates = set()
    for directory, dirs, files in walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        path = Path(directory)
        marker = path / ".rustc_info.json"
        if ".rustc_info.json" in files and stat.S_ISREG(marker.lstat().st_mode):
            candidates.add(path)
        if path.name in ("debug", "release") and ".fingerprint" in dirs:
            if stat.S_ISDIR((path / ".fingerprint").lstat().st_mode):
                candidates.add(path.parent)
    rows = []
    cutoff = time.time() - hours * 3600 if hours is not None else None
    for candidate in sorted(candidates):
        if any(parent in candidates for parent in candidate.parents):
            continue
        newest = candidate.lstat().st_mtime
        # Visit every descendant, including fresh trees, so traversal failures are visible.
        for directory, dirs, files in walk(candidate):
            for name in dirs + files:
                newest = max(newest, (Path(directory) / name).lstat().st_mtime)
        if cutoff is not None and newest > cutoff:
            continue
        rows.append({"path": str(candidate), "allocated_bytes": allocated(candidate),
                     "newest_mtime": newest})
    return {"completed_at": utc(), "root": os.path.abspath(root), "stale_hours": hours,
            "candidates": rows, "allocated_bytes": sum(r["allocated_bytes"] for r in rows),
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
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = Path(config["custody"]).resolve(strict=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = output / stamp
    run.mkdir(mode=0o700)
    receipt = {"started_at": utc(), "status": "running", "exit_status": None,
               "config": config, "checksums": {}, "measurements": []}
    source = Path(__file__).resolve().parent
    for path in [config_path, Path(__file__).resolve(), source / "cargo-build-dirs.sh",
                 source / "jope-build-report.sh"]:
        receipt["checksums"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = run / "receipt.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        roots = config["cargo_roots"]
        approved = ["/home/ubuntu/JopeDime", "/home/ubuntu/jain-split/jope-dime"]
        if roots != approved or config["stale_hours"] != 48:
            raise ValueError("service requires the two approved Cargo roots and a 48-hour filter")
        for index, root in enumerate(roots):
            with (run / f"candidates-{index}.json").open("w") as out, \
                    (run / f"scan-{index}.stderr").open("w") as err:
                subprocess.run([str(source / "cargo-build-dirs.sh"), root, "--stale", "48"],
                               stdout=out, stderr=err, check=True, timeout=600)
        seen = []
        for item in config["measurement_roots"]:
            path = Path(item["path"])
            resolved = path.resolve(strict=True)
            if resolved != path or not resolved.is_dir():
                raise ValueError(f"measurement roots must be explicit resolved directories: {path}")
            if any(resolved == prior or prior in resolved.parents or resolved in prior.parents
                   for prior in seen):
                raise ValueError(f"overlapping measurement root: {path}")
            seen.append(resolved)
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
