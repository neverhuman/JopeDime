"""Snapshot scratch receipts into daily Parquet; optionally refresh every 30 minutes."""

import argparse
import csv
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from .extract import Copies, digest
from .report import read_daily, render, write_daily
from .scratch import (
    EXPORT_SCHEMA,
    R22_SCHEMAS,
    number,
    receipt_rows,
    verify_r22_metrics,
)
from .context_costs import (
    context_rows,
    native_context_rows,
    indexed_context_rows,
    scope_rows,
)


REMOTE_ROOTS = {
    "CL": "/home/ubuntu/claude-rlfw-lane/rlfw-takeover/lpi-scratch/run",
    "CX-OP": "/home/ubuntu/jope-evidence/rlfw-cx-op/scratch-train/results",
}


def load_events(path):
    events = []
    for line, text in enumerate(Path(path).read_text().splitlines(), 1):
        event = json.loads(text)
        event["source_line"] = line
        events.append(event)
    return events


def remote_read(host, *command):
    if host != "xbabe3":
        raise ValueError("only owner-assigned xbabe3 scratch metadata reads allowed")
    return subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            " ".join(shlex.quote(str(arg)) for arg in command),
        ],
        timeout=30,
    )


def remote_copy(host, path, copies):
    """Read-only SSH stat/hash/cat/hash; no remote writable handle or model import."""
    size = int(remote_read(host, "stat", "-c", "%s", "--", path))
    if size > 32 * 1024**2:
        raise ValueError("remote scratch metadata exceeds32MiB")
    before = remote_read(host, "sha256sum", "--", path).decode().split()[0]
    raw = remote_read(host, "cat", "--", path)
    sha = hashlib.sha256(raw).hexdigest()
    after = remote_read(host, "sha256sum", "--", path).decode().split()[0]
    if len(raw) != size or before != sha or after != sha:
        raise ValueError(
            "remote scratch metadata changed during snapshot; defer this source"
        )
    target = copies.root / sha
    if not target.exists():
        with target.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        target.chmod(0o444)
    if digest(target) != sha:
        raise ValueError("remote copy hash mismatch")
    key = host + ":" + path
    copies.records[key] = {
        "source": key,
        "copy": str(target),
        "sha256": sha,
        "bytes": size,
    }
    return target


def partial_receipt(events):
    evaluations = [e for e in events if e.get("kind") == "eval"]
    updates = {}
    for event in events:
        if event.get("kind") == "step":
            updates[event["arm"]] = updates.get(event["arm"], 0) + 1
    return {
        "schema": "jope.r22-lpi-scratch-train/v2",
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "initial_eval": evaluations,
        "updates_per_arm": updates,
        "stop_reason": "PARTIAL JSONL snapshot; terminal state, final costs and input closure unmeasured",
    }


def live_status_receipt(document):
    """Copy producer LIVE_STATUS as a partial observation, never a terminal receipt."""
    if "evals" not in document:
        raise ValueError(
            "Status-only inventory has no evaluation series; use its final run receipt"
        )
    number(document["unix"])
    events = document["evals"]
    if not isinstance(events, list) or any(e.get("kind") != "eval" for e in events):
        raise ValueError("LIVE_STATUS reported evaluation inventory required")
    result = partial_receipt(events)
    result["updates_per_arm"] = document["updates"]
    result["stop_reason"] = (
        "PARTIAL copied LIVE_STATUS at "
        + document["utc"]
        + "; producer state="
        + document["state"]
        + "; terminal costs/input closure unmeasured"
    )
    return result


def gpu_export(host, copies):
    """Physical read-only device sample; it cannot establish whole-day utilization."""
    if host not in ("xbabe2", "xbabe3"):
        raise ValueError("owner-assigned GPU observation host required")
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,index,timestamp,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    raw = (
        remote_read(host, *command)
        if host == "xbabe3"
        else subprocess.check_output(command, timeout=30)
    )
    clock = (
        remote_read(host, "date", "-u", "+%s.%N")
        if host == "xbabe3"
        else subprocess.check_output(["date", "-u", "+%s.%N"], timeout=30)
    )
    samples, points = [], []
    for uuid, index, device_time, utilization, memory_mb in csv.reader(
        raw.decode().splitlines(), skipinitialspace=True
    ):
        utilization = number(float(utilization), nonnegative=True)
        if utilization > 100:
            raise ValueError("GPU percentage outside 0..100")
        sample = {
            "uuid": uuid,
            "index": int(index),
            "device_timestamp": device_time,
            "utilization_percent": utilization,
            "memory_used_mb": float(memory_mb),
        }
        samples.append(sample)
        for metric, value, unit in (
            ("gpu_utilization_sample_percent", utilization, "%"),
            ("gpu_memory_used_sample_mb", float(memory_mb), "MiB"),
            ("gpu_full_day_time_mean_percent", None, "%"),
        ):
            points.append(
                {
                    "arm": uuid,
                    "epoch": None,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "n": 1 if value is not None else None,
                    "note": "Physical host GPU sample at "
                    + device_time
                    + "; includes all workloads. No full-day trace, lane attribution or extrapolation from this sample.",
                }
            )
    if not samples:
        raise ValueError("no physical GPU observations")
    document = {
        "schema": EXPORT_SCHEMA,
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "started_unix": float(clock),
        "host": host,
        "command": command,
        "raw_csv": raw.decode(),
        "samples": samples,
        "points": points,
        "costs": {},
    }
    encoded = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    sha = hashlib.sha256(encoded).hexdigest()
    target = copies.root / sha
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o444)
    source = host + ":nvidia-smi@" + str(document["started_unix"])
    copies.records[source] = {
        "source": source,
        "copy": str(target),
        "sha256": sha,
        "bytes": len(encoded),
    }
    return document, sha, source


def remote_prefix(host, path, copies):
    """Verify the same complete-line prefix twice while an append-only JSONL grows."""
    size = int(remote_read(host, "stat", "-c", "%s", "--", path))
    if size > 128 * 1024**2:
        raise ValueError("JSONL observation prefix exceeds128MiB")
    raw = remote_read(host, "head", "-c", size, "--", path)
    raw = raw[: raw.rfind(b"\n") + 1]
    repeated = remote_read(host, "head", "-c", len(raw), "--", path)
    if raw != repeated:
        raise ValueError("JSONL prefix rewritten during snapshot; defer")
    sha = hashlib.sha256(raw).hexdigest()
    target = copies.root / sha
    if not target.exists():
        with target.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        target.chmod(0o444)
    if digest(target) != sha:
        raise ValueError("JSONL prefix copy hash mismatch")
    source = host + ":" + path
    copies.records[source] = {
        "source": source,
        "copy": str(target),
        "sha256": sha,
        "bytes": len(raw),
        "source_size_at_start": size,
        "snapshot_kind": "verified complete-line prefix; terminal state unmeasured",
    }
    return target


def scope_export(ref, copies):
    lane, host, unit = ref["lane"], ref.get("host"), ref["unit"]
    prefixes = {
        "CL": "claude-rlfw-lpi-scratch-",
        "CX-OP": (
            "cxop-rloo-pkpo-",
            "cxop-r5-",
            "cxop-r6-",
            "cxop-r7-",
            "cxop-r8-",
            "cxop-r9-scratch-",
        ),
        "G-ABL": ("g-abl-real-pairs-", "g-abl-r23-"),
    }
    if (
        lane not in prefixes
        or not unit.startswith(prefixes[lane])
        or not unit.endswith(".scope")
    ):
        raise ValueError("owner-assigned scratch scope required")
    if host not in (None, "xbabe3"):
        raise ValueError("only local xbabe2 or assigned xbabe3 metadata allowed")
    command = [
        "env",
        "TZ=UTC",
        "LC_ALL=C",
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=Id,LoadState,ActiveState,SubState,ActiveEnterTimestamp,InactiveEnterTimestamp,ActiveEnterTimestampMonotonic,InactiveEnterTimestampMonotonic,CPUUsageNSec,RuntimeMaxUSec,CPUQuotaPerSecUSec,MemoryMax,MemorySwapMax",
    ]
    raw = (
        remote_read(host, *command)
        if host
        else subprocess.check_output(command, timeout=30)
    )
    clock = (
        remote_read(host, "date", "-u", "+%s.%N")
        if host
        else subprocess.check_output(["date", "-u", "+%s.%N"], timeout=30)
    )
    properties = dict(
        line.split("=", 1) for line in raw.decode().splitlines() if "=" in line
    )

    def stamp(value):
        try:
            return (
                dt.datetime.strptime(value, "%a %Y-%m-%d %H:%M:%S UTC")
                .replace(tzinfo=dt.timezone.utc)
                .timestamp()
            )
        except ValueError:
            return None

    document = {
        "lane": lane,
        "run": ref["run"],
        "host": host or "xbabe2",
        "unit": unit,
        "properties": properties,
        "observed_remote_unix": float(clock),
        "active_enter_unix": stamp(properties.get("ActiveEnterTimestamp", "")),
        "inactive_enter_unix": stamp(properties.get("InactiveEnterTimestamp", "")),
        "gpu_count": ref.get("gpu_count"),
        "assignment_basis": "owner COORD GPU scratch claim",
    }
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    sha = hashlib.sha256(raw).hexdigest()
    target = copies.root / sha
    if not target.exists():
        with target.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        target.chmod(0o444)
    document["sha256"] = sha
    source = (
        (host or "xbabe2")
        + ":systemctl-show/"
        + unit
        + "@"
        + str(document["observed_remote_unix"])
    )
    copies.records[source] = {
        "source": source,
        "copy": str(target),
        "sha256": sha,
        "bytes": len(raw),
    }
    return document, source + ":1 sha256=" + sha


def scope():
    relative = next(
        s.split(":", 2)[2]
        for s in Path("/proc/self/cgroup").read_text().splitlines()
        if s.startswith("0::")
    )
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    result = {
        "path": relative,
        **{
            k: (root / k).read_text().strip()
            for k in ("cpu.max", "memory.max", "cpu.stat", "memory.peak")
        },
    }
    quota, period = result["cpu.max"].split()
    if (
        quota == "max"
        or int(quota) / int(period) > 2
        or result["memory.max"] == "max"
        or int(result["memory.max"]) > 4 * 1024**3
    ):
        raise ValueError("scratch analytics requires CPU<=200% and MemoryMax<=4GiB")
    unit = root.name
    properties = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "--property=RuntimeMaxUSec"], text=True
    ).strip()
    result["RuntimeMaxUSec"] = properties.split("=", 1)[1]
    duration = result["RuntimeMaxUSec"]
    tokens = re.findall(r"([0-9.]+)(h|min|s|ms|us)", duration)
    seconds = sum(
        float(n) * {"h": 3600, "min": 60, "s": 1, "ms": 0.001, "us": 0.000001}[u]
        for n, u in tokens
    )
    if not tokens or not 0 < seconds <= 7200:
        raise ValueError("finite external RuntimeMaxSec<=2h required")
    result["runtime_max_seconds"] = seconds
    return result


def build(manifest_path, output, consumer_commit):
    """All foreign reads go through stable content-addressed copies first."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    copies = Copies(output / "inputs")
    manifest = copies.json(manifest_path)
    if manifest.get("schema") != "jope.cx-met-scratch-input/v1":
        raise ValueError("scratch input manifest schema")
    date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    closures = {}
    for pattern in manifest.get("closure_patterns", []):
        for path in sorted(glob.glob(pattern)):
            closure = copies.json(path)
            sha = closure.get("run_receipt", {}).get("sha256")
            if sha:
                closures[sha] = closure
    pairings = {}
    for ref in manifest.get("paired_evidence", []):
        pairing_copy = copies.copy(
            ref["path"], expected_sha=ref["sha256"], expected_bytes=ref["bytes"]
        )
        pairings[ref["sha256"]] = json.loads(pairing_copy.read_bytes())
    rows, inventory, seen, context_sources, native_indexes = [], [], set(), [], []
    for provider in manifest["providers"]:
        native_indexes.extend(
            (None, p)
            for pattern in provider.get("native_index_patterns", [])
            for p in glob.glob(pattern)
        )
        context_sources.extend(
            (provider["lane"], None, p)
            for pattern in provider.get("context_patterns", [])
            for p in glob.glob(pattern)
        )
        for table_path in provider.get("tables", []):
            table_copy = copies.copy(table_path)
            inventory.append(
                {
                    "lane": provider["lane"],
                    "source": table_path,
                    "sha256": digest(table_copy),
                    "status": "copied provider table; numeric values from receipts",
                }
            )
        paths = sorted(
            {
                p
                for pattern in provider.get("patterns", [])
                for p in glob.glob(pattern, recursive=True)
            }
        )
        if len(paths) > 256:
            raise ValueError("provider scratch metadata inventory exceeds256")
        remote_paths = []
        for remote in provider.get("remote_roots", []):
            host, root = remote["host"], remote["path"]
            assigned_op = provider["lane"] == "CX-OP" and re.fullmatch(
                r"/home/ubuntu/jope-evidence/rlfw-cx-op/scratch-train/r[5-9](?:-scratch)?/results",
                root,
            )
            if host != "xbabe3" or not (
                root == REMOTE_ROOTS.get(provider["lane"]) or assigned_op
            ):
                raise ValueError("owner-assigned lane scratch metadata root required")
            try:
                found = (
                    remote_read(
                        host,
                        "find",
                        root,
                        "-maxdepth",
                        "2",
                        "-type",
                        "f",
                        "(",
                        "-name",
                        "receipt.json",
                        "-o",
                        "-name",
                        "metrics.jsonl",
                        "-o",
                        "-name",
                        "context-costs.json",
                        "-o",
                        "-name",
                        "index.json",
                        ")",
                    )
                    .decode()
                    .splitlines()
                )
            except (OSError, subprocess.SubprocessError) as error:
                inventory.append(
                    {
                        "lane": provider["lane"],
                        "source": host + ":" + root,
                        "status": "unmeasured",
                        "reason": "remote inventory unavailable: " + str(error),
                    }
                )
                continue
            if len(found) > 256 or any(
                not p.startswith(root.rstrip("/") + "/") for p in found
            ):
                raise ValueError("remote scratch metadata inventory boundary")
            receipts = {
                str(Path(p).parent) for p in found if p.endswith("/receipt.json")
            }
            remote_paths.extend(
                (host, p)
                for p in found
                if p.endswith("/receipt.json")
                or (
                    p.endswith("/metrics.jsonl") and str(Path(p).parent) not in receipts
                )
            )
            context_sources.extend(
                (provider["lane"], host, p)
                for p in found
                if p.endswith("/context-costs.json")
            )
            native_indexes.extend(
                (host, p)
                for p in found
                if provider["lane"] == "CL"
                and p.endswith("/index.json")
                and "paired" in Path(p).parent.name
            )
        imported = 0
        for host, path in [(None, p) for p in paths] + sorted(remote_paths):
            source = host + ":" + path if host else path
            record = {"lane": provider["lane"], "source": source}
            try:
                if host:
                    copy = (
                        remote_prefix(host, path, copies)
                        if path.endswith("/metrics.jsonl")
                        else remote_copy(host, path, copies)
                    )
                else:
                    if Path(path).stat().st_size > 32 * 1024**2:
                        raise ValueError("scratch metadata exceeds32MiB")
                    copy = copies.copy(path)
                sha = digest(copy)
                record["sha256"] = sha
                if sha in seen:
                    continue
                seen.add(sha)
                live = path.endswith("/LIVE_STATUS.json")
                partial = path.endswith("/metrics.jsonl") or live
                receipt = (
                    live_status_receipt(json.loads(copy.read_bytes()))
                    if live
                    else partial_receipt(load_events(copy))
                    if partial
                    else json.loads(copy.read_bytes())
                )
                events_path = Path(path).parent / "metrics.jsonl"
                if (
                    not partial
                    and receipt.get("schema") in R22_SCHEMAS
                    and (host or events_path.exists())
                ):
                    expected = closures.get(sha, {}).get("metrics_jsonl_sha256")
                    events_copy = (
                        remote_copy(host, str(events_path), copies)
                        if host
                        else copies.copy(events_path, expected_sha=expected)
                    )
                    if expected and digest(events_copy) != expected:
                        raise ValueError("remote JSONL differs from run-closure pin")
                    note = verify_r22_metrics(receipt, load_events(events_copy))
                    if note:
                        record["reconciliation_note"] = note
                    record["metrics_sha256"] = digest(events_copy)
                result = receipt_rows(
                    receipt,
                    run=(
                        Path(path).parent.parent.name
                        if Path(path).parent.name in ("work", "result")
                        else Path(path).parent.parent.parent.name + "/kv-eval"
                        if Path(path).parent.name == "kv-eval"
                        else Path(path).parent.name
                    )
                    + ("/LIVE_STATUS" if live else ""),
                    lane=provider["lane"],
                    source=source + ":1 sha256=" + sha,
                    receipt_sha=sha,
                    consumer_commit=consumer_commit,
                    as_of_date=date,
                    closure=closures.get(sha),
                    verified_pairings=pairings,
                )
                rows.extend(result)
                record.update(
                    status="PARTIAL / NON-QUALIFYING" if partial else "NON-QUALIFYING",
                    rows=len(result),
                )
                imported += 1
            except (
                ValueError,
                KeyError,
                TypeError,
                OSError,
                subprocess.SubprocessError,
            ) as error:
                record.update(status="unmeasured", reason=str(error))
            inventory.append(record)
        if imported == 0:
            inventory.append(
                {
                    "lane": provider["lane"],
                    "status": "unmeasured",
                    "reason": "no supported immutable scratch receipt yet",
                }
            )
    context_count = 0
    for lane, host, path in sorted(
        set(context_sources), key=lambda x: (x[0], x[1] or "", x[2])
    ):
        source = host + ":" + path if host else path
        record = {"lane": lane, "source": source}
        try:
            copy = remote_copy(host, path, copies) if host else copies.copy(path)
            export = json.loads(copy.read_bytes())
            sha = digest(copy)
            parent = None
            if export.get("run_receipt_sha256"):
                parent_copy = copies.root / export["run_receipt_sha256"]
                if not parent_copy.is_file():
                    raise ValueError("physical parent run receipt not in this snapshot")
                parent = json.loads(parent_copy.read_bytes())
            if (
                parent is not None and parent.get("schema") not in R22_SCHEMAS
            ) or export["lane"] != lane:
                raise ValueError("context export parent schema/lane mismatch")
            result = context_rows(
                export,
                parent,
                source=source + ":1 sha256=" + sha,
                receipt_sha=sha,
                consumer_commit=consumer_commit,
                as_of_date=date,
            )
            rows.extend(result)
            context_count += 1
            record.update(status="NON-QUALIFYING", rows=len(result), sha256=sha)
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            record.update(status="unmeasured", reason=str(error))
        inventory.append(record)
    if not context_count:
        inventory.append(
            {
                "lane": "CL",
                "status": "unmeasured",
                "reason": "Per-context full-validation L6 series pending; between-context variance and measured n unmeasured.",
            }
        )
    for host, path in sorted(set(native_indexes), key=lambda x: (x[0] or "", x[1])):
        source = host + ":" + path if host else path
        record = {"lane": "CL", "source": source}
        try:
            copy = remote_copy(host, path, copies) if host else copies.copy(path)
            index = json.loads(copy.read_bytes())
            in_process = "checkpoint_index" in index
            rows_path = str(
                Path(path).with_name(
                    "paired-rows.jsonl" if in_process else "rows.jsonl"
                )
            )
            rows_copy = (
                remote_prefix(host, rows_path, copies)
                if host
                else copies.copy(rows_path)
            )
            rosters = {}
            for name, ref in index["sets"].items():
                filename = ref.get("file", "contexts-" + name + ".json")
                if Path(filename).name != filename or not filename.startswith(
                    "contexts-"
                ):
                    raise ValueError("native roster filename boundary")
                roster_path = str(Path(path).with_name(filename))
                roster_copy = (
                    remote_copy(host, roster_path, copies)
                    if host
                    else copies.copy(roster_path)
                )
                rosters[name] = json.loads(roster_copy.read_bytes())
            if in_process:
                if (
                    host is not None
                    or Path(index["checkpoint_index"]).name != index["checkpoint_index"]
                ):
                    raise ValueError("local indexed metadata basename required")
                checkpoint_copy = copies.copy(
                    Path(path).with_name(index["checkpoint_index"])
                )
                parent_copy = copies.copy(Path(path).with_name("receipt.json"))
                result = indexed_context_rows(
                    index,
                    json.loads(checkpoint_copy.read_bytes()),
                    json.loads(parent_copy.read_bytes()),
                    load_events(rows_copy),
                    rosters,
                    source=source
                    + ":1 index_sha256="
                    + digest(copy)
                    + " rows_sha256="
                    + digest(rows_copy)
                    + " checkpoint_index_sha256="
                    + digest(checkpoint_copy)
                    + " parent_sha256="
                    + digest(parent_copy),
                    index_sha=digest(copy),
                    parent_sha=digest(parent_copy),
                    consumer_commit=consumer_commit,
                    as_of_date=date,
                )
            else:
                result = native_context_rows(
                    index,
                    load_events(rows_copy),
                    rosters,
                    source=source
                    + ":1 index_sha256="
                    + digest(copy)
                    + " rows_sha256="
                    + digest(rows_copy),
                    index_sha=digest(copy),
                    consumer_commit=consumer_commit,
                    as_of_date=date,
                )
            rows.extend(result)
            record.update(
                status="NON-QUALIFYING",
                rows=len(result),
                sha256=digest(copy),
                rows_sha256=digest(rows_copy),
            )
            if not result:
                record.update(
                    status="unmeasured",
                    reason="Native index has no complete matching control/arm epoch cohorts yet.",
                )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            record.update(status="unmeasured", reason=str(error))
        inventory.append(record)
    for ref in manifest.get("scope_exports", []):
        record = {"lane": ref["lane"], "source": ref["unit"]}
        try:
            document, source = scope_export(ref, copies)
            result = scope_rows(
                document,
                consumer_commit=consumer_commit,
                as_of_date=date,
                source=source,
            )
            rows.extend(result)
            record.update(
                status="NON-QUALIFYING scope metadata",
                rows=len(result),
                sha256=document["sha256"],
            )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            record.update(status="unmeasured", reason=str(error))
        inventory.append(record)
    for host in manifest.get("gpu_hosts", []):
        record = {"lane": "CX-MET", "source": host + ":nvidia-smi"}
        try:
            document, sha, source = gpu_export(host, copies)
            result = receipt_rows(
                document,
                run=host + "-gpu-observation",
                lane="CX-MET",
                source=source + ":1 sha256=" + sha,
                receipt_sha=sha,
                consumer_commit=consumer_commit,
                as_of_date=date,
            )
            rows.extend(result)
            record.update(
                status="NON-QUALIFYING host observation", rows=len(result), sha256=sha
            )
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            record.update(status="unmeasured", reason=str(error))
        inventory.append(record)
    for item in inventory:
        if item["status"] == "unmeasured":
            rows.append(
                {
                    "date": date,
                    "kind": "scratch_provider",
                    "metric": "receipt_availability",
                    "scope": item["lane"],
                    "status": "unmeasured / NON-QUALIFYING",
                    "value": None,
                    "n": None,
                    "unit": "receipt",
                    "note": item["reason"],
                    "sources": item.get("source", "pending owner export"),
                    "source_commit": consumer_commit,
                    "context_set_sha256": "unmeasured",
                    "details": json.dumps(
                        item | {"run": "pending", "lane": item["lane"]}
                    ),
                }
            )
    reference = manifest.get("fold_noise_reference")
    if reference:
        copy = copies.copy(
            reference["path"],
            expected_sha=reference["sha256"],
            expected_bytes=reference["bytes"],
        )
        candidates = [
            r
            for r in read_daily([copy])
            if r["kind"] == "summary" and r["metric"] == "SE_pair:p50"
        ]
        if len(candidates) != 1:
            raise ValueError("one pinned retained median SE reference required")
        ref = candidates[0]
        rows.append(
            ref
            | {
                "date": date,
                "kind": "scratch_reference",
                "metric": "retained_SE_pair_median_reference",
                "source_commit": consumer_commit,
                "status": "retained reference / NON-QUALIFYING",
                "note": "Retained evaluator-q noise, not loss/KL SE. Per-comparison identity+SE needed to resolve new q deltas.",
                "details": json.dumps(
                    {"lane": "CX-MET", "run": "retained-fold-reference"}
                ),
            }
        )
    files = write_daily(rows, output / "daily")
    actual = read_daily(files)
    render(actual, output / "dashboard.html")
    receipt = {
        "schema": "jope.cx-met-scratch-ingest/v1",
        "source_commit": consumer_commit,
        "non_qualifying": True,
        "benchmark_valid": False,
        "promotion_input": False,
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": len(actual),
        "inputs": list(copies.records.values()),
        "inventory": inventory,
        "outputs": [
            {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in [*files, output / "dashboard.html"]
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def log_board(board, message):
    board = Path(board)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    line = f"- {now} CX-MET: {message}"
    if len(line) > 300:
        raise ValueError("coordination log exceeds300")
    with Path(str(board) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = board.read_text().splitlines()
        for i, text in enumerate(rows):
            if text.startswith("last_updated:"):
                rows[i] = "last_updated: " + now
            elif text.startswith("owner_now:"):
                rows[i] = "owner_now: CX-MET"
        board.write_text("\n".join(rows) + "\n" + line + "\n")


def release_watch(board, watch, output_root):
    board = Path(board)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    unit = watch["scope"]["path"].rsplit("/", 1)[-1]
    with Path(str(board) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = board.read_text().splitlines()
        for i, text in enumerate(lines):
            if text.startswith("last_updated:"):
                lines[i] = "last_updated: " + now
            elif text.startswith("owner_now:"):
                lines[i] = "owner_now: CX-MET"
            elif text.startswith("| CX-MET |") and unit in text:
                fields = text.split("|")
                fields[-2] = (
                    f" stopped: bounded watch complete at {now}; source {watch['source_commit'][:8]}, PR288 open, no merge "
                )
                lines[i] = "|".join(fields)
        claim = f"- RELEASED CX-MET {now} scratch-run ingest/per-context L6 watch {unit}; {output_root}/terminal.json"
        index = next(
            i for i, text in enumerate(lines) if text.startswith("## review_queue")
        )
        lines.insert(index, claim)
        lines.append(
            f"- {now} CX-MET: bounded scratch watch complete, {watch['iterations']} passes; {output_root}/terminal.json; NON-QUALIFYING, no GPU."
        )
        board.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--board", type=Path)
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--deadline-unix", type=float)
    args = parser.parse_args()
    if args.deadline_unix is not None:
        args.watch_seconds = max(0, int(args.deadline_unix - time.time()))
    if not 0 <= args.watch_seconds <= 7140 or args.interval_seconds < 60:
        raise ValueError("bounded <=7140s watch and >=60s interval required")
    started, deadline = time.time(), time.monotonic() + args.watch_seconds
    caps = scope()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_files = [
        Path(__file__),
        Path(__file__).with_name("scratch.py"),
        Path(__file__).with_name("report.py"),
        Path(__file__).with_name("extract.py"),
        Path(__file__).with_name("context_costs.py"),
    ]
    pins = {str(p): digest(p) for p in source_files}
    args.output_root.mkdir(parents=True, exist_ok=True)
    watch = {
        "schema": "jope.cx-met-scratch-watch/v1",
        "source_commit": commit,
        "code_sha256": pins,
        "pid": os.getpid(),
        "scope": caps,
        "started_unix": started,
        "deadline_unix": started + args.watch_seconds,
        "interval_seconds": args.interval_seconds,
        "non_qualifying": True,
    }
    (args.output_root / "watch.json").write_text(json.dumps(watch, indent=2) + "\n")
    print(json.dumps(watch), flush=True)
    iteration = 0
    while True:
        if args.board:
            args.board.read_text()  # The full shared board is read before every pass.
        if any(digest(p) != sha for p, sha in pins.items()):
            raise ValueError(
                "ingest source changed during watch; new bounded source launch required"
            )
        target = args.output_root / (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-{iteration:03d}"
        )
        receipt = build(args.manifest, target, commit)
        rows = read_daily(
            [
                Path(r["path"])
                for r in receipt["outputs"]
                if r["path"].endswith(".parquet")
            ]
        )
        final = [r for r in rows if r["kind"] == "scratch_epoch"]
        table = [
            {
                "scope": r["scope"],
                "metric": r["metric"],
                "epoch": json.loads(r["details"])["epoch"],
                "value": r["value"],
                "status": r["status"],
            }
            for r in final
        ]
        print(
            json.dumps(
                {"output": str(target), "rows": receipt["rows"], "table": table}
            ),
            flush=True,
        )
        if args.board:
            supported = sum(
                "NON-QUALIFYING" in r.get("status", "") for r in receipt["inventory"]
            )
            deltas = [
                r
                for r in rows
                if r["kind"] == "scratch_delta"
                and r["metric"] == "l6_full_validation_delta_vs_control"
                and r["value"] is not None
                and "smoke" not in r["scope"]
            ]
            paired = [
                r
                for r in rows
                if r["metric"] == "l6_paired_mean_delta"
                and r["n"] == 263
                and r["value"] is not None
            ]
            if paired:
                deltas = paired
            deltas.sort(
                key=lambda r: (json.loads(r["details"])["epoch"], r["scope"]),
                reverse=True,
            )
            headline = (
                ",".join(
                    json.loads(r["details"])["arm"] + f" {r['value']:+.4g}"
                    for r in deltas[:2]
                )
                or "unmeasured"
            )
            measured = [
                r
                for r in rows
                if r["metric"] == "l6_contexts_needed_delta_0_005_power_0_80"
                and r["value"] is not None
                and r["n"] == 263
            ]
            latest = {}
            for r in measured:
                d = json.loads(r["details"])
                key = r["scope"]
                if (
                    key not in latest
                    or d["epoch"] > json.loads(latest[key]["details"])["epoch"]
                ):
                    latest[key] = r
            sizing = (
                "/".join(str(int(r["value"])) for _, r in sorted(latest.items())[-3:])
                or "unmeasured"
            )
            log_board(
                args.board,
                f"ingest#{iteration}: {supported}src/{receipt['rows']}rows; ΔL6 {headline}; measured n {sizing}; {target.relative_to(args.output_root.parent.parent)}/dashboard.html; NON-QUAL.",
            )
        if not args.watch_seconds or time.monotonic() >= deadline:
            break
        next_pass = min(time.monotonic() + args.interval_seconds, deadline)
        while time.monotonic() < next_pass:
            time.sleep(min(45, next_pass - time.monotonic()))
        iteration += 1
    watch.update(
        finished_unix=time.time(), scope_finish=scope(), iterations=iteration + 1
    )
    (args.output_root / "terminal.json").write_text(json.dumps(watch, indent=2) + "\n")
    if args.board and args.watch_seconds:
        release_watch(args.board, watch, args.output_root)


if __name__ == "__main__":
    main()
