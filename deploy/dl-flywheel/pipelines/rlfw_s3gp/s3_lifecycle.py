"""Score staleness, storage-class proposals, a byte/PUT guard, and a ledger export.

Read a listing. Write a proposal. Do not call S3.

Prices below are the 2026-09-01 us-east-2 rows in
`.agent/proposals/claude-rlfw-20261009/w6/pricing/s3-us-east-2-price-receipt-v1.json`
(usage types USE2-TimedStorage-ByteHrs, USE2-TimedStorage-GIR-ByteHrs,
USE2-Requests-Tier1, USE2-Requests-Tier2, USE2-EarlyDelete-GIR). A rate that
is not in that receipt is not invented here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "jope.s3-lifecycle-proposal/v1"
LEDGER_SCHEMA = "jope.s3-byte-put-ledger/v1"
# Decimal gigabytes, matching the bucket inventory note in AGENTS.md.
BYTE_CAP = 200 * 10**9
PUT_CAP = 2_000_000
ALARM_RATIO = 0.80
GIB128 = 128 * 1024
STANDARD_USD_PER_GB_MONTH = 0.023
GIR_USD_PER_GB_MONTH = 0.004
PUT_USD = 0.005 / 1000
GET_USD = 0.004 / 10000
GIR_MIN_DAYS = 90
EXECUTABLE_EXPIRE_DAYS = 14

# Prefixes this lane will not transition or expire. gp-hwm and checkpoints are
# the product high-water and checkpoint trees. paper-evidence and provenance
# are records. sealed/judge/legal-hold are the owner's hold.
PROTECTED_MARKERS = (
    "/gp-hwm/",
    "/checkpoints/",
    "/paper-evidence/",
    "/provenance/",
    "/sealed/",
    "/judge/",
    "legal-hold",
)
EXECUTABLE_ROLES = frozenset({"rust_executable"})
SCORE_ROLES = frozenset({"score_records", "evaluation", "rust_parity"})


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def protected(key: str, obj: dict | None = None) -> str | None:
    """Return the reason a key must be left untouched, or None."""
    for marker in PROTECTED_MARKERS:
        if marker in key:
            return marker
    if obj:
        hold = obj.get("legal_hold") or obj.get("ObjectLockLegalHoldStatus")
        if hold not in (None, "", "OFF"):
            return "legal_hold"
        if obj.get("object_lock_mode") or obj.get("ObjectLockMode"):
            return "object_lock"
    return None


def role_of(key: str) -> str | None:
    name = key.rstrip("/").rsplit("/", 1)[-1]
    if name in {
        "rust_executable",
        "rust_source",
        "rust_parity",
        "evaluation",
        "score_records",
        "input",
        "program",
        "record.json",
    }:
        return "record" if name == "record.json" else name
    return None


def artifact_bytes(record: dict) -> dict[str, int]:
    artifacts = record.get("artifacts") or {}
    out = {}
    for role, ref in artifacts.items():
        if isinstance(ref, dict) and isinstance(ref.get("bytes"), int):
            out[role] = ref["bytes"]
    return out


def classify_object(
    obj: dict,
    *,
    current_evaluator: str | None,
    superseded_source_shas: set[str] | None = None,
    now: datetime | None = None,
) -> dict:
    """Classify one listing row. `obj` needs key, size, and optional fields.

    legal_hold left absent means the hold was not read. Those rows are retained.
    A stale score is demoted only when both evaluator ids are present and differ.
    An executable is an expiry candidate only when its rust_source sha is in the
    reproducible-cache set. No class here is applied to S3.
    """
    key = obj["key"]
    size = int(obj["size"])
    reason = protected(key, obj)
    row = {
        "key": key,
        "size": size,
        "storage_class": obj.get("storage_class") or "STANDARD",
        "role": role_of(key),
        "action": "retain",
        "reason": reason or "keep",
    }
    if reason:
        row["action"] = "untouched"
        return row
    if obj.get("legal_hold_checked") is not True:
        row["action"] = "retain"
        row["reason"] = "legal_hold_unmeasured"
        return row
    role = row["role"]
    evaluator = obj.get("evaluator_sha256")
    if role in SCORE_ROLES and current_evaluator and evaluator and evaluator != current_evaluator:
        if size < GIB128:
            row["action"] = "retain"
            row["reason"] = "stale_score_below_128kib_no_individual_transition"
        else:
            row["action"] = "propose_glacier_ir"
            row["reason"] = "stale_evaluator"
        row["evaluator_sha256"] = evaluator
        return row
    if role in SCORE_ROLES and evaluator is None:
        row["reason"] = "evaluator_unmeasured"
        return row
    if role in EXECUTABLE_ROLES:
        source = obj.get("rust_source_sha256")
        known = superseded_source_shas or set()
        if source and source in known:
            row["action"] = "propose_expire"
            row["reason"] = "superseded_by_reproducible_executable"
            row["expire_days"] = EXECUTABLE_EXPIRE_DAYS
        else:
            row["action"] = "retain"
            row["reason"] = "executable_not_yet_superseded"
        return row
    row["reason"] = "no_rule"
    if now is not None:
        row["classified_at"] = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return row


def monthly_usd(size_bytes: int, usd_per_gb_month: float) -> float:
    return (size_bytes / 10**9) * usd_per_gb_month


def proposal_from_rows(
    rows: list[dict],
    *,
    current_evaluator: str | None,
    superseded_source_shas: set[str] | None = None,
    existing_rules: list | None = None,
    sample_note: str,
) -> dict:
    """Build a proposal. `applied` is always false. Passing apply=True raises."""
    classified = [
        classify_object(
            row,
            current_evaluator=current_evaluator,
            superseded_source_shas=superseded_source_shas,
        )
        for row in rows
    ]
    totals: dict[str, dict[str, int]] = {}
    for row in classified:
        bucket = totals.setdefault(row["action"], {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += row["size"]
    expire_bytes = totals.get("propose_expire", {}).get("bytes", 0)
    gir_bytes = totals.get("propose_glacier_ir", {}).get("bytes", 0)
    # A prefix rule is safe only when every scanned object under that prefix
    # has the same proposed action. Mixed prefixes produce no rule.
    by_prefix: dict[str, set[str]] = {}
    for row in classified:
        parent = row["key"].rsplit("/", 1)[0] + "/"
        by_prefix.setdefault(parent, set()).add(row["action"])
    rules = []
    for prefix, actions in sorted(by_prefix.items()):
        if actions == {"propose_expire"}:
            rules.append(
                {
                    "ID": "proposed-expire-superseded-executables",
                    "Status": "proposed-not-applied",
                    "Filter": {"Prefix": prefix},
                    "Expiration": {"Days": EXECUTABLE_EXPIRE_DAYS},
                    "note": "14 days is under the 90-day GIR minimum, so this is expiry, not a transition",
                }
            )
        elif actions == {"propose_glacier_ir"}:
            rules.append(
                {
                    "ID": "proposed-demote-stale-scores",
                    "Status": "proposed-not-applied",
                    "Filter": {"Prefix": prefix},
                    "Transitions": [{"Days": GIR_MIN_DAYS, "StorageClass": "GLACIER_IR"}],
                    "note": "only objects already >=128 KiB; bucket minimum is all_storage_classes_128K",
                }
            )
    body = {
        "schema": SCHEMA,
        "applied": False,
        "sample_note": sample_note,
        "current_evaluator_sha256": current_evaluator,
        "existing_rules_unchanged": existing_rules or [],
        "proposed_rules": rules,
        "totals": totals,
        "classified": classified,
        "cost_if_expire_executables_usd_per_month": round(
            monthly_usd(expire_bytes, STANDARD_USD_PER_GB_MONTH), 6
        ),
        "cost_if_demote_to_gir_usd_per_month": {
            "standard": round(monthly_usd(gir_bytes, STANDARD_USD_PER_GB_MONTH), 6),
            "glacier_ir": round(monthly_usd(gir_bytes, GIR_USD_PER_GB_MONTH), 6),
        },
        "rates_usd": {
            "standard_gb_month": STANDARD_USD_PER_GB_MONTH,
            "glacier_ir_gb_month": GIR_USD_PER_GB_MONTH,
            "put": PUT_USD,
            "get": GET_USD,
        },
    }
    body["proposal_sha256"] = sha256_hex(canonical({k: v for k, v in body.items() if k != "proposal_sha256"}))
    return body


def refuse_apply(apply: bool) -> None:
    if apply:
        raise RuntimeError("refusing to apply an S3 lifecycle or storage-class change")


class BudgetLedger:
    """Hash-chained daily byte and PUT guard. It does not talk to S3."""

    def __init__(self, path: Path, byte_cap: int = BYTE_CAP, put_cap: int = PUT_CAP):
        self.path = Path(path)
        self.byte_cap = byte_cap
        self.put_cap = put_cap
        self.events = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        prev = "0" * 64
        for line in self.path.read_text().splitlines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("prev") != prev:
                raise ValueError("ledger prev link does not match the prior event")
            body = {k: v for k, v in event.items() if k != "sha256"}
            if sha256_hex(canonical(body)) != event.get("sha256"):
                raise ValueError("ledger event hash does not match its body")
            prev = event["sha256"]
            events.append(event)
        return events

    def _day_usage(self, day: str) -> tuple[int, int]:
        nbytes = puts = 0
        for event in self.events:
            if event.get("day") == day and event.get("decision") == "admit":
                nbytes += int(event["bytes"])
                puts += int(event["puts"])
        return nbytes, puts

    def admit(self, nbytes: int, puts: int, *, kind: str, now: datetime | None = None) -> dict:
        if nbytes < 0 or puts < 0:
            raise ValueError("negative usage")
        moment = now or datetime.now(timezone.utc)
        day = moment.astimezone(timezone.utc).strftime("%Y-%m-%d")
        used_b, used_p = self._day_usage(day)
        projected_b = used_b + nbytes
        projected_p = used_p + puts
        alarm = (
            projected_b >= int(self.byte_cap * ALARM_RATIO)
            or projected_p >= int(self.put_cap * ALARM_RATIO)
        )
        over = projected_b > self.byte_cap or projected_p > self.put_cap
        decision = "refuse" if over else "admit"
        prev = self.events[-1]["sha256"] if self.events else "0" * 64
        event = {
            "schema": LEDGER_SCHEMA,
            "day": day,
            "kind": kind,
            "bytes": nbytes,
            "puts": puts,
            "decision": decision,
            "alarm": alarm and decision == "admit",
            "projected_bytes": projected_b,
            "projected_puts": projected_p,
            "byte_cap": self.byte_cap,
            "put_cap": self.put_cap,
            "prev": prev,
        }
        event["sha256"] = sha256_hex(canonical(event))
        if decision == "admit":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            self.events.append(event)
        return event


def export_ledger(rows: list[dict], parquet_path: Path, duckdb_path: Path) -> dict:
    """Write Parquet and a DuckDB file that reproduces the row count.

    Requires the duckdb module. The caller records ImportError as unmeasured
    rather than substituting a different format silently.
    """
    import duckdb

    parquet_path = Path(parquet_path)
    duckdb_path = Path(duckdb_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = parquet_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows))
    dest = parquet_path.as_posix().replace("'", "")
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            "create or replace table objects as select * from read_json_auto(?)",
            [str(json_path)],
        )
        con.execute(f"copy objects to '{dest}' (format parquet)")
        count = con.execute("select count(*) from objects").fetchone()[0]
        check = duckdb.connect()
        try:
            via = check.execute("select count(*) from read_parquet(?)", [str(parquet_path)]).fetchone()[0]
        finally:
            check.close()
    finally:
        con.close()
    if count != via or count != len(rows):
        raise RuntimeError(f"ledger count mismatch rows={len(rows)} duckdb={count} parquet={via}")
    return {
        "rows": count,
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": sha256_hex(parquet_path.read_bytes()),
        "duckdb_bytes": duckdb_path.stat().st_size,
    }
