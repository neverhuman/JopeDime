"""Conservative retention classification and a deterministic local custody pack.

Every output is a non-qualifying observation. This module has no remote mutation
or deletion operation. A listing's ETag is never treated as a payload SHA.
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

BYTE_CAP = 25 * 10**9
HOT_CAP = 10**12
PUT_CAP = 1_500_000
PERMANENT_ROLES = frozenset(
    {
        "score_records",
        "evaluation",
        "rust_parity",
        "input",
        "program",
        "rust_source",
        "record.json",
    }
)
KEEPER_FACTS = ("new_context_best", "parity_failure", "cited", "sealed_salt_audit")
SHA = re.compile(r"[0-9a-f]{64}")
MAX_CAPTURE_BYTES = 64 * 1024**2
MAX_PACK_BYTES = 128 * 1024**2
MAX_PHYSICAL_REFS = 512
MAX_MANIFEST_BYTES = 4 * 1024**2


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, name):
    need(type(value) is int and value >= 0, name + " must be a nonnegative integer")
    return value


def identity(row):
    values = tuple(row[k] for k in ("bucket", "key", "version_id"))
    need(all(type(v) is str and v for v in values), "complete object version identity")
    return values


def classify(row, facts=None):
    """Unknown keeper/hold facts retain bytes; other is only an inert recommendation."""
    identity(row)
    integer(row["bytes"], "object bytes")
    need(type(row["is_latest"]) is bool, "latest-version flag")
    role = row["key"].rsplit("/", 1)[-1]
    reason = "unmeasured_keeper_or_hold_facts"
    kind = "keeper"
    if row["version_id"] == "null":
        reason = "immutable_version_unavailable"
    elif not row["is_latest"]:
        reason = "noncurrent_version_review"
    elif any(
        part in row["key"].split("/")
        for part in ("sealed", "judge", "checkpoints", "gp-hwm", "provenance")
    ):
        reason = "protected_evidence"
    elif role in PERMANENT_ROLES:
        reason = "permanent_ledger_or_small_artifact"
    elif role != "rust_executable":
        reason = "unknown_role"
    elif facts is not None:
        need(type(facts) is dict, "keeper facts object")
        need(
            facts.get("identity") == list(identity(row)),
            "keeper facts bind the exact physical version",
        )
        checked = (
            facts.get("legal_hold_checked") is True
            and facts.get("retention_checked") is True
        )
        complete = all(type(facts.get(k)) is bool for k in KEEPER_FACTS)
        salt = facts.get("audit_salt_commitment")
        if checked and complete and type(salt) is str and SHA.fullmatch(salt):
            if (
                facts.get("legal_hold") != "OFF"
                or facts.get("retention_until") is not None
            ):
                reason = "protected_or_unknown_hold"
            elif any(facts[k] for k in KEEPER_FACTS):
                reason = "verified_keeper_fact"
            else:
                kind, reason = "other", "complete_negative_keeper_facts"
    return dict(
        row,
        role=role,
        retention_class=kind,
        reason=reason,
        proposed_retention_days=14 if kind == "other" else None,
        applied=False,
        qualified=False,
    )


def budget_status(new_version_bytes, raw_hot_bytes, puts):
    """Read-only guard: 80% alarm; 100% refuses new scoring. No service is controlled."""
    values = {
        "new_version_bytes": new_version_bytes,
        "raw_hot_bytes": raw_hot_bytes,
        "puts": puts,
    }
    caps = {"new_version_bytes": BYTE_CAP, "raw_hot_bytes": HOT_CAP, "puts": PUT_CAP}
    unknown = [k for k, v in values.items() if v is None]
    for k, v in values.items():
        if v is not None:
            integer(v, k)
    alarm = [
        k
        for k, cap in caps.items()
        if values[k] is not None and values[k] >= cap * 4 // 5
    ]
    exhausted = [
        k for k, cap in caps.items() if values[k] is not None and values[k] >= cap
    ]
    return {
        "caps": caps,
        "values": values,
        "alarm": alarm,
        "exhausted": exhausted,
        "unknown": unknown,
        "admit_new_scoring": not unknown and not exhausted,
        "control_applied": False,
    }


def cost_report(rows, requests, rates, as_of):
    """Price measured bytes/HTTP requests; invoice, discounts and transfer remain unknown."""
    need(
        rates["region"] == "us-east-2" and rates["storage_bytes_per_gb"] == 2**30,
        "Ohio binary-GB rate card",
    )
    price = {}
    for key in ("standard_gb_month", "list_request", "get_request"):
        need(type(rates[key]) is str, "decimal price string")
        value = Decimal(rates[key])
        need(value.is_finite() and value > 0, "positive finite price")
        price[key] = value
    days = calendar.monthrange(as_of.year, as_of.month)[1]
    standard = sum(
        integer(r["bytes"], "object bytes")
        for r in rows
        if r["storage_class"] == "STANDARD"
    )
    unsupported = Counter(
        r["storage_class"] for r in rows if r["storage_class"] != "STANDARD"
    )
    counts = Counter(r["operation"] for r in requests)
    need(
        set(counts)
        <= {"ListObjectVersions", "ListObjectsV2", "GetObject", "GetBucketVersioning"},
        "read-only request roster",
    )
    listings = counts["ListObjectVersions"] + counts["ListObjectsV2"]
    reads = counts["GetObject"] + counts["GetBucketVersioning"]
    storage = Decimal(standard) / Decimal(2**30) * price["standard_gb_month"] / days
    request_cost = listings * price["list_request"] + reads * price["get_request"]
    return {
        "standard_version_bytes": standard,
        "standard_storage_usd_day_estimate": str(storage),
        "request_usd_estimate": str(request_cost),
        "http_requests": dict(sorted(counts.items())),
        "days_in_utc_month": days,
        "storage_rate_assumption": "first 50 TiB tier; calendar-month daily normalization",
        "unsupported_storage_classes": dict(unsupported),
        "all_storage_priced": not unsupported,
        "data_transfer_usd": None,
        "actual_billed_usd": None,
        "discounts_free_tier_account_aggregate_tier": "unmeasured",
        "qualified": False,
    }


def _durable_file(path, raw):
    """Exclusive local name, complete writes and both file and directory fsync."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        pending = memoryview(raw)
        while pending:
            n = os.write(fd, pending)
            need(n > 0, "short local write")
            pending = pending[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pack_local(destination, members):
    """Pack exact captured versions; manifest becomes visible only after full readback."""
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    refs, blobs, seen = [], {}, set()
    for row, body in members:
        need(len(refs) < MAX_PHYSICAL_REFS, "bounded physical reference roster")
        ref = identity(row)
        need(ref not in seen, "duplicate physical object version")
        seen.add(ref)
        need(ref[2] != "null", "immutable version required for capture")
        need(
            type(body) is bytes and len(body) == integer(row["bytes"], "member bytes"),
            "captured size mismatch",
        )
        h = digest(body)
        refs.append(
            {
                "bucket": ref[0],
                "key": ref[1],
                "version_id": ref[2],
                "bytes": len(body),
                "sha256": h,
            }
        )
        need(h not in blobs or blobs[h] == body, "content digest collision")
        blobs[h] = body
    need(refs, "nonempty pack")
    need(
        sum(len(b) for b in blobs.values()) <= MAX_CAPTURE_BYTES,
        "local pack raw-byte cap",
    )
    refs.sort(key=lambda r: identity(r))
    index = {
        "schema": "jope.custody-local-pack/v2",
        "refs": refs,
        "blobs": [
            {"name": "members/" + h, "sha256": h, "bytes": len(b)}
            for h, b in sorted(blobs.items())
        ],
        "qualified": False,
        "remote_publication": False,
    }
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for h, body in sorted(blobs.items()):
                item = tarfile.TarInfo("members/" + h)
                item.size, item.mode, item.mtime = len(body), 0o600, 0
                archive.addfile(item, io.BytesIO(body))
    raw = stream.getvalue()
    pack = destination / (digest(raw) + ".tar.gz")
    _durable_file(pack, raw)
    index["pack_sha256"] = digest(raw)
    # Verify the actual newly reopened file, not the in-memory writer's bytes.
    verify_local(pack, index)
    _durable_file(destination / "manifest.json", canonical(index))
    return {
        "pack": str(pack),
        "manifest": str(destination / "manifest.json"),
        "manifest_sha256": digest(canonical(index)),
        "physical_refs": len(refs),
        "unique_blobs": len(blobs),
        "pack_bytes": len(raw),
        "captured_ref_bytes": sum(r["bytes"] for r in refs),
        "remote_writes": 0,
        "qualified": False,
    }


def verify_local(pack, index):
    """No extraction, links, extra members, aliases or unchecked SHA values."""
    need(type(index) is dict, "pack manifest object")
    need(
        type(index.get("refs")) is list and 0 < len(index["refs"]) <= MAX_PHYSICAL_REFS,
        "bounded physical reference roster",
    )
    need(
        type(index.get("blobs")) is list
        and 0 < len(index["blobs"]) <= MAX_PHYSICAL_REFS,
        "bounded blob roster",
    )
    with Path(pack).open("rb") as file:
        raw = file.read(MAX_PACK_BYTES + 1)
    need(len(raw) <= MAX_PACK_BYTES, "bounded compressed pack")
    need(
        index.get("schema") == "jope.custody-local-pack/v2"
        and index.get("qualified") is False
        and index.get("remote_publication") is False,
        "closed non-qualifying pack",
    )
    need(digest(raw) == index["pack_sha256"], "pack SHA mismatch")
    expected = {}
    for blob in index["blobs"]:
        h = blob["sha256"]
        need(
            type(h) is str and SHA.fullmatch(h) and blob["name"] == "members/" + h,
            "safe content member",
        )
        need(blob["name"] not in expected, "duplicate blob declaration")
        integer(blob["bytes"], "blob bytes")
        expected[blob["name"]] = blob
    need(expected, "nonempty blob roster")
    payload_bytes = sum(b["bytes"] for b in expected.values())
    need(payload_bytes <= MAX_CAPTURE_BYTES, "bounded unpacked bytes")
    # Bound the entire inflated stream before tarfile can allocate PAX/GNU
    # metadata. Our short CAS names need one header and at most 511 padding
    # bytes each, plus the final tar record; payload-only limits miss metadata.
    tar_limit = payload_bytes + 1024 * len(expected) + tarfile.RECORDSIZE
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
        decoded = gz.read(tar_limit + 1)
    need(len(decoded) <= tar_limit, "bounded expanded tar bytes")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:") as archive:
        for item in archive:
            need(
                item.isfile() and item.name in expected and item.name not in seen,
                "unexpected or duplicate archive member",
            )
            blob = expected[item.name]
            need(item.size == blob["bytes"], "archive size mismatch")
            body = archive.extractfile(item).read()
            need(digest(body) == blob["sha256"], "member SHA mismatch")
            seen.add(item.name)
    need(seen == set(expected), "missing archive member")
    physical = set()
    used = set()
    for row in index["refs"]:
        ref = identity(row)
        need(
            ref not in physical and ref[2] != "null",
            "unique immutable captured reference",
        )
        physical.add(ref)
        name = "members/" + row["sha256"]
        need(
            name in expected and row["bytes"] == expected[name]["bytes"],
            "every reference joins a verified blob",
        )
        used.add(name)
    need(physical and used == seen, "every blob has a source reference")
    return {
        "physical_refs": len(physical),
        "verified_blobs": len(seen),
        "all_member_shas_verified": True,
        "qualified": False,
    }
