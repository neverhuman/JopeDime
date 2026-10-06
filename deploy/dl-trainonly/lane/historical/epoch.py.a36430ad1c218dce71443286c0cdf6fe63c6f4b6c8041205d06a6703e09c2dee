"""PLAN §5 P1: corpus epoch 0 (read-only inputs; writes only its manifests and receipt).

The epoch manifest carries only whitelisted references (ref()). The test-exposed fields stay out of it:
- the champion program and its RMSE;
- ratios, the TabPFN 3.5 RMSE and tabpfn35_ref;
- native.best and the winning programs.
They go to a separate exposed side manifest that only reporting code reads.

Commands:
  pin    --out DIR [--publish --g18-receipt PATH]   build epoch-000.json + epoch-000.exposed.json (+ S3 objects and binding)
  verify --epoch PATH --sample N                     exact-version HEADs of sampled refs (<= 600 requests)
"""
import argparse
import gzip
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from dltrain import common

EXPERIMENT_ID = "gp-policy-dl-v1-e0"
INVENTORY = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/win-status-v2/folds.json.gz")
INVENTORY_SHA256 = "125cbc0b4ecfbbfddcf37b9255a95884aa5c196a1397f4caf28798231b798a8d"
FROZEN_ROOT = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/contexts")
INCOMING_ROOT = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/items")
REGISTRY = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/intake.sqlite3")
LEAKAGE = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-20261001-leakage1/gp/leakage-remediation-preparation.json")
LEAKAGE_SHA256 = "84d1098ba14c626753044b260d99d85d02da0002df9b867475e65d3e8ac5c539"
LATEST_POINTER = ("veox-jopedime-datasets-use2", "research/regression-2k/incoming/v1/latest.json")
MASTER = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/input-manifest-master.json")
COHORTS = {"frozen": "frozen", "incoming_qualified_complete": "incoming"}
WHITELIST = ("cohort", "view_id", "item_id", "dataset_hash", "canonical_sha256", "sample_identity_sha256", "fold",
             "context_sha256", "context_schema", "history_ref", "telemetry_ref", "native_source", "prepared_ref", "input_ref", "membership_ref")
OPTIONAL = {"item_id", "canonical_sha256"}
EXPOSED_FIELDS = ("program_sha256", "champion", "gp_rmse", "tabpfn35_rmse", "ratio", "tabpfn35_ref", "generation_zero_rmse")
EXPECT = {"frozen": {"folds": 2834, "datasets": 570, "completed_per_fold": 1280, "history_bytes": 11142261689, "telemetry_bytes": 10603091253},
          "incoming": {"folds": 2183, "history_bytes": 8620892008, "telemetry_bytes": 8115319838}}
WITHDRAWAL_REASONS = ("withdrawn_from_hunt_catalog",)


class EpochStop(RuntimeError):
    """A P1 Failure condition (PLAN §5 P1): sha/bytes mismatch, unresolvable ref, or unlisted exclusion reason."""


def ref(r: dict) -> dict:
    """PLAN P1 whitelist: a missing required key raises (STOP); champion/rmse/ratio/tabpfn35_ref never enter."""
    missing = [k for k in WHITELIST if k not in r and k not in OPTIONAL]
    if missing:
        raise EpochStop(f"context {r.get('view_id')} lacks required keys {missing}")
    return {k: r[k] for k in WHITELIST if k in r}


def load_inventory(path=INVENTORY, expected=INVENTORY_SHA256) -> tuple[list[dict], dict]:
    raw = Path(path).read_bytes()
    got = common.sha256_bytes(raw)
    if got != expected:
        raise EpochStop(f"inventory sha256 {got} != {expected}")
    rows = json.loads(gzip.decompress(raw))
    return rows, {"path": str(path), "sha256": got, "bytes": len(raw), "rows": len(rows)}


def prepared_path(row: dict) -> Path:
    if row["cohort"] == "frozen":
        return FROZEN_ROOT / row["view_id"] / "prepared.json"
    return INCOMING_ROOT / row["item_id"] / "run" / "gp" / "contexts" / row["view_id"] / "prepared.json"


def merged_record(row: dict) -> tuple[dict, dict]:
    """Sealed row + prepared.json context refs + prepared_ref + native_source; returns (record, exposed)."""
    p = prepared_path(row)
    if not p.is_file():
        raise EpochStop(f"prepared.json missing for {row['view_id']}: {p}")
    raw = p.read_bytes()
    prep = json.loads(raw)
    ctx = prep.get("context") or {}
    checks = {"context_sha256": (ctx.get("context_sha256"), row["context_sha256"]), "dataset_hash": (ctx.get("dataset_hash"), row["dataset_hash"]),
              "fold": (ctx.get("outer_fold"), row["fold"]), "view_id": (prep.get("view_id"), row["view_id"])}
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise EpochStop(f"prepared.json disagrees with the sealed row for {row['view_id']}: {bad}")
    rec = {"cohort": COHORTS[row["cohort"]], "view_id": row["view_id"], "dataset_hash": row["dataset_hash"], "fold": row["fold"],
           "context_sha256": row["context_sha256"], "context_schema": ctx.get("schema"), "sample_identity_sha256": ctx.get("sample_identity_sha256"),
           "history_ref": row["history_ref"], "telemetry_ref": row["telemetry_ref"], "native_source": row["native"]["source"],
           "prepared_ref": {"path": str(p), "sha256": common.sha256_bytes(raw), "bytes": len(raw)},
           "input_ref": ctx.get("input_ref"), "membership_ref": ctx.get("membership_ref")}
    if row.get("item_id"):
        rec["item_id"] = row["item_id"]
    if ctx.get("canonical_sha256"):
        rec["canonical_sha256"] = ctx["canonical_sha256"]
    for k in ("sample_identity_sha256", "context_schema", "input_ref", "membership_ref"):
        if not rec.get(k):
            raise EpochStop(f"{row['view_id']}: prepared.json lacks {k}")
    exposed = {k: row.get(k) for k in EXPOSED_FIELDS} | {"native_best": row["native"].get("best"),
                                                         "native_winning_programs": row["native"].get("winning_programs")}
    return rec, exposed


def registry_snapshot(path=REGISTRY) -> dict:
    """One read snapshot (BEGIN) for poll_seq and withdrawals; mode=ro + query_only; reasons cross-checked against events."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only=1")
        con.execute("BEGIN")
        poll_seq = con.execute("select max(seq) from polls").fetchone()[0]
        retired = con.execute("select id, dataset, reason from items where state='retired'").fetchall()
        missing = sorted({d for _, d, r in retired if "missing_from_latest_corpus" in (r or "")})
        reasons = Counter((r or "").split(":")[0] for _, _, r in retired)
        withdrawn = sorted({d for _, d, r in retired if any(w in (r or "") for w in WITHDRAWAL_REASONS)})
        ev = Counter()
        for (payload,) in con.execute("select payload from events where event like '%retire%' or event like '%withdraw%'"):
            try:
                body = json.loads(payload) if payload else {}
            except json.JSONDecodeError:
                body = {}
            ev[str(body.get("reason") or body.get("retired_reason") or "unknown").split(":")[0]] += 1
        con.execute("COMMIT")
    finally:
        con.close()
    return {"path": str(path), "mode": "ro+query_only", "poll_seq": poll_seq, "utc_us": common.utc_us(),
            "retired_reason_counts": dict(reasons), "retire_event_reason_counts": dict(ev),
            "withdrawn_datasets": withdrawn, "withdrawn_sha256": common.sha256_bytes(common.canonical(withdrawn)),
            "withdrawal_rule": f"items.state='retired' and reason contains one of {list(WITHDRAWAL_REASONS)}",
            "retired_missing_from_latest_corpus": missing,
            "events_match_items": dict(reasons) == dict(ev)}


def leakage_quarantine(path=LEAKAGE, expected=LEAKAGE_SHA256) -> dict:
    raw = Path(path).read_bytes()
    if common.sha256_bytes(raw) != expected:
        raise EpochStop("leakage quarantine file changed")
    d = json.loads(raw)
    maps = d["mappings"]
    return {"path": str(path), "sha256": expected, "mappings": len(maps),
            "source_context_sha256": sorted(m["source_context_sha256"] for m in maps),
            "clean_context_sha256": sorted(m["clean_context_sha256"] for m in maps),
            "source_view_ids": sorted(m["source_view_id"] for m in maps), "clean_view_ids": sorted(m["clean_view_id"] for m in maps)}


def eligibility_source(rec: dict) -> str:
    if rec["cohort"] == "frozen" and rec["fold"] == 0:
        return "frozen_fold0_by_construction(clean views substituted; build_campaign.py:236-247)"
    if rec["cohort"] == "frozen":
        return "frozen_fold1_4: P5 target-leak recompute pending"
    return "incoming: intake-qualified at sealing (intake_poll.py:96-108); P5 target-leak recompute pending"


def build(rows: list[dict], snapshot: dict, quarantine: dict, latest: dict | None) -> tuple[dict, dict, dict]:
    withdrawn = set(snapshot["withdrawn_datasets"])
    q_sources = set(quarantine["source_context_sha256"])
    contexts, exposed, excl = [], {}, Counter()
    for row in rows:
        rec, exp = merged_record(row)
        r = ref(rec)
        reason = None
        if rec["context_sha256"] in q_sources:
            reason = "quarantined"
        elif rec["cohort"] == "incoming" and rec["dataset_hash"] in withdrawn:
            reason = "withdrawn"
        r["eligibility_source"] = eligibility_source(rec)
        r["status"] = "excluded" if reason else "included"
        r["exclusion_reason"] = reason
        if reason:
            excl[reason] += 1
        contexts.append(r)
        exposed[rec["context_sha256"]] = exp
    counts = {}
    for coh in ("frozen", "incoming"):
        rs = [r for r in rows if COHORTS[r["cohort"]] == coh]
        counts[coh] = {"folds": len(rs), "datasets": len({r["dataset_hash"] for r in rs}),
                       "completed_total": sum(r["native"]["completed"] for r in rs),
                       "history_bytes": sum(r["history_ref"]["bytes"] for r in rs),
                       "telemetry_bytes": sum((r.get("telemetry_ref") or {}).get("bytes", 0) for r in rs),
                       "included": sum(1 for c in contexts if c["cohort"] == coh and c["status"] == "included")}
    master_raw = MASTER.read_bytes()
    sanitized = {v["view_id"] for v in json.loads(master_raw)["views"] if v.get("feature_sanitization")}
    q_datasets = {v.split("__")[0] for v in quarantine["source_view_ids"]}
    q_rows = [r for r in rows if r["dataset_hash"] in q_datasets]
    quarantine_check = {"master_manifest": {"path": str(MASTER), "sha256": common.sha256_bytes(master_raw), "bytes": len(master_raw)},
                        "quarantined_datasets": len(q_datasets), "inventory_rows": len(q_rows),
                        "rows_via_feature_sanitization_view": sum(r["view_id"] in sanitized for r in q_rows),
                        "rows_with_quarantined_source_context": sum(r["context_sha256"] in q_sources for r in q_rows)}
    missing = set(snapshot.get("retired_missing_from_latest_corpus", []))
    flags = {"incoming_folds_retired_missing_from_latest_corpus": sum(1 for r in rows if COHORTS[r["cohort"]] == "incoming" and r["dataset_hash"] in missing),
             "rule": "kept (PLAN P1 withdrawal SQL covers withdrawn_from_hunt_catalog only); owner may exclude them in a later declared epoch"}
    frozen_withdrawn = sorted({r["dataset_hash"] for r in rows if COHORTS[r["cohort"]] == "frozen" and r["dataset_hash"] in withdrawn})
    checks = {
        "frozen_folds": counts["frozen"]["folds"] == EXPECT["frozen"]["folds"],
        "frozen_datasets": counts["frozen"]["datasets"] == EXPECT["frozen"]["datasets"],
        "frozen_completed": counts["frozen"]["completed_total"] == EXPECT["frozen"]["folds"] * EXPECT["frozen"]["completed_per_fold"],
        "frozen_history_bytes": counts["frozen"]["history_bytes"] == EXPECT["frozen"]["history_bytes"],
        "frozen_telemetry_bytes": counts["frozen"]["telemetry_bytes"] == EXPECT["frozen"]["telemetry_bytes"],
        "incoming_folds_sealed": counts["incoming"]["folds"] == EXPECT["incoming"]["folds"],
        "incoming_history_bytes": counts["incoming"]["history_bytes"] == EXPECT["incoming"]["history_bytes"],
        "incoming_telemetry_bytes": counts["incoming"]["telemetry_bytes"] == EXPECT["incoming"]["telemetry_bytes"],
        "unique_context_sha256": len({c["context_sha256"] for c in contexts}) == len(contexts),
        "only_listed_exclusion_reasons": set(excl) <= {"withdrawn", "quarantined", "intake_ineligible", "target_leak", "sealed_set"},
        "quarantine_only_via_sanitized_views": quarantine_check["inventory_rows"] == quarantine_check["rows_via_feature_sanitization_view"] == 45
                                               and quarantine_check["rows_with_quarantined_source_context"] == 0,
        "registry_events_match_items": bool(snapshot.get("events_match_items")),
    }
    epoch = {
        "schema": "jope.dl-corpus-epoch/v1", "experiment_id": EXPERIMENT_ID, "epoch": 0, "created_utc_us": common.utc_us(),
        "inventory": None, "registry": {k: v for k, v in snapshot.items()}, "leakage_quarantine": quarantine,
        "latest_pointer_observed": latest, "whitelist": list(WHITELIST), "optional": sorted(OPTIONAL),
        "exclusions": dict(excl) | {"target_leak": "pending P5 recompute", "sealed_set": "pending P2/P3", "intake_ineligible": 0},
        "frozen_datasets_also_withdrawn_in_registry": frozen_withdrawn, "quarantine_check": quarantine_check, "flags": flags,
        "counts": counts, "checks": checks,
        "contexts": contexts,
    }
    side = {"schema": "jope.dl-corpus-epoch-exposed/v1", "experiment_id": EXPERIMENT_ID, "epoch": 0,
            "reader_rule": "reporting code only; never an input to training, sampling, labels, encodings or selection",
            "fields": list(EXPOSED_FIELDS) + ["native_best", "native_winning_programs"], "by_context_sha256": exposed}
    return epoch, side, checks


def head_latest_pointer() -> dict:
    from dltrain import s3io
    c = s3io.make_client()
    h = c.head_object(Bucket=LATEST_POINTER[0], Key=LATEST_POINTER[1])
    return {"bucket": LATEST_POINTER[0], "key": LATEST_POINTER[1], "version_id": h.get("VersionId"), "etag": h.get("ETag"),
            "content_length": h.get("ContentLength"), "last_modified": str(h.get("LastModified")), "recorded_only": True,
            "observed_utc_us": common.utc_us()}


def verify(epoch_path: str, sample: int, seed: int = 20261005) -> dict:
    """Exact-version HEADs of sampled refs; each must exist with the recorded byte count (<= 600 requests)."""
    from dltrain import s3io
    e = json.loads(Path(epoch_path).read_text())
    rng = random.Random(seed)
    picks = rng.sample(e["contexts"], min(sample, len(e["contexts"])))
    c = s3io.make_client()
    out, requests = [], 0
    for ctx in picks:
        for kind in ("history_ref", "telemetry_ref", "input_ref"):
            r = ctx.get(kind)
            if not r or requests >= 600:
                continue
            requests += 1
            try:
                h = c.head_object(Bucket=r["bucket"], Key=r["key"], VersionId=r["version_id"])
                ok = h.get("ContentLength") == r["bytes"]
                out.append({"context": ctx["context_sha256"][:16], "kind": kind, "ok": ok, "bytes": h.get("ContentLength")})
            except Exception as err:  # noqa: BLE001
                out.append({"context": ctx["context_sha256"][:16], "kind": kind, "ok": False, "error": str(err)[:300]})
    bad = [o for o in out if not o["ok"]]
    return {"schema": "jope.dl-epoch-verify/v1", "epoch": epoch_path, "sampled_contexts": len(picks), "requests": requests,
            "failures": len(bad), "failed": bad[:50], "verdict": "PASS" if not bad else "FAIL", "utc_us": common.utc_us()}


CLAIM = "2026-10-05T04:45:30.385967+00:00 CLAIM dltrain-impl-v1"
DECISIONS_SHA256 = "77a115c78e927000e77f052c86c2fde5cc66c149d4a16814172b91426aa3a602"


def publish(out_dir) -> dict:
    """Create-only uploads of both manifests, then the deterministic create-only binding (first freeze wins)."""
    from dltrain import s3io
    out = Path(out_dir)
    e_body, x_body = (out / "epoch-000.json").read_bytes(), (out / "epoch-000.exposed.json").read_bytes()
    created = json.loads(e_body)["created_utc_us"]
    w, r = s3io.make_client(writes=True), s3io.make_client()
    refs = {}
    for name, body, kind in (("epoch", e_body, "epoch"), ("exposed", x_body, "epoch-exposed")):
        key = f"{s3io.PREFIX_KEY}objects/{kind}/sha256/{common.sha256_bytes(body)}.json"
        res = s3io.put_create_only(w, key, body, r)
        refs[name] = {"bucket": s3io.BUCKET, "key": key, "version_id": res["version_id"], "sha256": res["sha256"], "bytes": res["bytes"],
                      "outcome": res["outcome"]}
    binding = {"schema": "jope.dl-binding/v1", "experiment_id": EXPERIMENT_ID, "kind": "epoch", "epoch": 0,
               "epoch_ref": {k: refs["epoch"][k] for k in ("bucket", "key", "version_id", "sha256", "bytes")},
               "exposed_ref": {k: refs["exposed"][k] for k in ("bucket", "key", "version_id", "sha256", "bytes")},
               "exposed_reader_rule": "reporting code only", "created_utc_us": created, "claim": CLAIM, "owner_decisions_sha256": DECISIONS_SHA256}
    b_key = f"{s3io.PREFIX_KEY}bindings/{EXPERIMENT_ID}/epoch-000.json"
    b_res = s3io.put_create_only(w, b_key, common.canonical(binding), r)
    refs["binding"] = {"bucket": s3io.BUCKET, "key": b_key, "version_id": b_res["version_id"], "sha256": b_res["sha256"],
                       "bytes": b_res["bytes"], "outcome": b_res["outcome"]}
    return refs


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.epoch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pin = sub.add_parser("pin")
    pin.add_argument("--out", required=True)
    pin.add_argument("--no-pointer", action="store_true", help="skip the recorded-only latest.json HEAD (offline runs)")
    ver = sub.add_parser("verify")
    ver.add_argument("--epoch", required=True)
    ver.add_argument("--sample", type=int, default=256)
    ver.add_argument("--receipt", required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--out", required=True)
    pub.add_argument("--receipt", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "publish":
        rec = Path(args.receipt)
        if rec.exists():
            raise SystemExit("publish receipt exists; refusing")
        refs = publish(args.out)
        res = common.write_receipt(rec, {"schema": "jope.dl-epoch-publish/v1", "utc_us": common.utc_us(), "refs": refs})
        print(json.dumps({"receipt": {k: res[k] for k in ("path", "sha256", "bytes")}, "refs": refs}))
        return 0
    if args.cmd == "pin":
        rows, inv = load_inventory()
        snapshot = registry_snapshot()
        quarantine = leakage_quarantine()
        latest = None if args.no_pointer else head_latest_pointer()
        epoch, side, checks = build(rows, snapshot, quarantine, latest)
        epoch["inventory"] = inv
        out = Path(args.out)
        r1 = common.write_receipt(out / "epoch-000.json", epoch)
        r2 = common.write_receipt(out / "epoch-000.exposed.json", side)
        failed = [k for k, v in checks.items() if not v]
        print(json.dumps({"epoch": r1, "exposed": {k: r2[k] for k in ("path", "sha256", "bytes")}, "counts": epoch["counts"],
                          "exclusions": epoch["exclusions"], "failed_checks": failed}, default=str))
        return 0 if not failed else 3
    res = verify(args.epoch, args.sample)
    common.write_receipt(args.receipt, res)
    print(json.dumps({k: res[k] for k in ("sampled_contexts", "requests", "failures", "verdict")}))
    return 0 if res["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
