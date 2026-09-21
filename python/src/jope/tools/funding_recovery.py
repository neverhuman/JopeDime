"""Inspect retained funding evidence without creating or replenishing a grant.

Legacy snapshots have no stable domain/attempt identities. Equal row names or
equal bytes therefore cannot prove that two charges describe the same work.
This report preserves every origin and leaves aggregate consumption unavailable.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import stat
import time
from pathlib import Path


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field: " + key)
        result[key] = value
    return result


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number: " + value)
    return number


def _stable_bytes(path, deadline, max_bytes):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink requires original-location reconciliation")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("not a bounded regular evidence file")
        chunks, count = [], 0
        while True:
            if time.time() >= deadline:
                raise TimeoutError("absolute evidence-inspection deadline expired")
            block = stream.read(min(1024 * 1024, max_bytes - count + 1))
            if not block:
                break
            count += len(block)
            if count > max_bytes:
                raise ValueError("evidence exceeds byte limit")
            chunks.append(block)
        after = os.fstat(stream.fileno())
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(path.stat()):
            raise ValueError("evidence changed during inspection")
    return b"".join(chunks)


def inspect(paths, *, deadline, max_bytes=16 * 1024 * 1024):
    if not _number(deadline) or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("finite absolute deadline and positive byte limit required")
    records = []
    for value in sorted(set(str(Path(os.path.abspath(p))) for p in paths)):
        row = dict(path=value, host=socket.gethostname(), status="unvisited")
        records.append(row)
        if time.time() >= deadline:
            row["reason"] = "absolute evidence-inspection deadline expired"
            continue
        try:
            raw = _stable_bytes(value, deadline, max_bytes)
            row.update(sha256=_sha(raw), bytes=len(raw))
            doc = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
            if not isinstance(doc, dict):
                raise ValueError("funding evidence must be an object")
            row["schema"] = doc.get("schema")
            if doc.get("schema") != "parallel.qualification_budget.v1":
                row.update(status="unresolved", reason="original authorization/domain verifier required")
                continue
            reservations = doc.get("reservations")
            if not isinstance(reservations, dict):
                raise ValueError("reservation map required")
            for identity, charge in reservations.items():
                if (
                    not identity
                    or not isinstance(charge, dict)
                    or not all(_number(charge.get(k)) for k in ("core_seconds", "node_seconds"))
                    or type(charge.get("searches")) is not int
                    or not 0 <= charge["searches"] <= 16
                    or type(charge.get("complete")) is not bool
                ):
                    raise ValueError("invalid historical charge: " + identity)
                if charge["complete"]:
                    ref = charge.get("receipt")
                    if (
                        not isinstance(ref, dict)
                        or set(ref) != {"path", "sha256"}
                        or not isinstance(ref["path"], str)
                        or not ref["path"]
                        or not isinstance(ref["sha256"], str)
                        or len(ref["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in ref["sha256"])
                    ):
                        raise ValueError("invalid original completion reference: " + identity)
            ceilings = {k: sum(r[k] for r in reservations.values()) for k in ("core_seconds", "node_seconds")}
            if not all(_number(value) for value in ceilings.values()):
                raise ValueError("historical snapshot ceiling overflow")
            row.update(
                status="legacy_snapshot",
                reservations=reservations,
                charged_ceilings=ceilings,
                unfinished=[k for k, r in reservations.items() if not r["complete"]],
                completion_bytes_verified=False,
            )
        except FileNotFoundError:
            row.update(status="missing")
        except PermissionError as error:
            row.update(status="unreadable", reason=str(error))
        except TimeoutError as error:
            row.update(status="unvisited", reason=str(error))
        except (OSError, ValueError, OverflowError) as error:
            row.update(status="unresolved", reason=str(error))
    return dict(
        schema="parallel.funding_recovery_observation.v1",
        observed_at=time.time(),
        records=records,
        scope="explicit paths only; no complete ledger census asserted",
        production_authorized=False,
        remaining_balance=None,
    )


def reconcile(observations):
    """Compare observations; never collapse distinct historical charge origins."""
    origins, variants, same_bytes = {}, {}, {}
    for observation in observations:
        if observation.get("schema") != "parallel.funding_recovery_observation.v1":
            raise ValueError("unsupported funding observation")
        for row in observation["records"]:
            # Repeated observations of one path retain differing versions too.
            key = (row["host"], row["path"], row.get("sha256"), row["status"])
            if key in origins:
                if origins[key] != row:
                    raise ValueError("conflicting observations of identical source bytes")
                continue
            origins[key] = row
            if row["status"] != "legacy_snapshot":
                continue
            origin = dict(host=row["host"], path=row["path"], sha256=row["sha256"])
            same_bytes.setdefault(row["sha256"], []).append(origin)
            for identity, charge in row["reservations"].items():
                encoded = json.dumps(charge, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
                version = variants.setdefault(identity, {}).setdefault(
                    _sha(encoded), dict(charge=charge, origins=[])
                )
                version["origins"].append(origin)
    rows = [
        dict(
            identity=k,
            variants=list(v.values()),
            disposition="unresolved_snapshot_versions" if len(v) > 1 else "origin_join_required",
        )
        for k, v in sorted(variants.items())
    ]
    return dict(
        schema="parallel.funding_recovery_comparison.v1",
        observations=list(origins.values()),
        reservation_names=rows,
        identical_snapshot_bytes=[v for v in same_bytes.values() if len(v) > 1],
        aggregate_consumption=None,
        remaining_balance=None,
        production_authorized=False,
        blockers=[
            "Original funding authorization is not verified",
            "Complete two-host ledger coverage is not verified",
            "Original attempt/domain joins and completion bytes require independent verification",
        ],
        accounting="Each legacy snapshot retains full ceilings, including unfinished charges; "
        "snapshot sums must not be added or deduplicated by name to infer a balance",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="JSON array of original evidence paths")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(), object_pairs_hook=_pairs)
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        parser.error("--paths must contain a JSON array of paths")
    print(json.dumps(inspect(paths, deadline=args.deadline), sort_keys=True, indent=2, allow_nan=False))
