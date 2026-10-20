"""Durable research target compilation from exact-version measured GP results.

This journal is not a native teaching verifier, HWM writer, experiment permit or
trainer. Its snapshots remain unadmitted until the existing native qualification
and resource owners bind them. Failed and unresolved attempts still cost slots.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


HEX = re.compile(r"[0-9a-f]{64}\Z")
DIRECTIONS = {"r2": "higher", "oof_r2": "higher", "auc": "higher", "macro_f1": "higher",
              "negative_mase": "higher", "rmse": "lower", "logloss": "lower", "mase": "lower"}
IDENTITY_KEYS = {"task", "dataset_hash", "cut_id", "evaluator_sha256", "vocabulary_sha256", "protocol", "comparison"}
COMPARISON_KEYS = {"numerical_policy", "stage", "aggregation", "label_role", "sealed", "benchmark_valid",
                   "recipe_encoding", "class_order", "checkpoint_sha256"}
MAX_RESULT_BYTES = 16 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)
    return value


def closed(value, keys, name):
    require(type(value) is dict and set(value) == set(keys), "closed " + name + " schema")


def sha(value):
    require(type(value) is str and HEX.fullmatch(value) is not None, "bound SHA256 required")
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite numerical value required")
    return float(value)


def identity(value):
    closed(value, IDENTITY_KEYS, "native comparison identity")
    require(all(type(value[k]) is str and 0 < len(value[k]) <= 512 for k in ("task", "dataset_hash", "protocol")), "named native identity")
    for key in ("cut_id", "evaluator_sha256", "vocabulary_sha256"):
        sha(value[key])
    comp = value["comparison"]
    closed(comp, COMPARISON_KEYS, "native comparison context")
    require(comp["sealed"] is False and comp["benchmark_valid"] is True, "unsealed valid comparison required")
    require(comp["label_role"] == "train", "TRAIN comparison role required")
    require(all(type(comp[k]) is str and 0 < len(comp[k]) <= 512 for k in ("numerical_policy", "stage", "aggregation", "recipe_encoding")), "explicit comparison semantics")
    require(type(comp["class_order"]) is list and all(type(x) is str for x in comp["class_order"])
            and len(set(comp["class_order"])) == len(comp["class_order"]), "explicit unique class order")
    if comp["checkpoint_sha256"] is not None:
        sha(comp["checkpoint_sha256"])
    return value


def manifest(raw):
    doc = decode(raw)
    closed(doc, {"schema", "experiment_sha256", "source_sha256", "runtime_sha256", "target_policy_sha256",
                 "data_manifest_sha256", "max_logical_attempts", "contexts"}, "feedback manifest")
    require(doc["schema"] == "jope.measured-feedback-journal/v1", "feedback manifest version")
    for key in ("experiment_sha256", "source_sha256", "runtime_sha256", "target_policy_sha256", "data_manifest_sha256"):
        sha(doc[key])
    require(type(doc["max_logical_attempts"]) is int and 1 <= doc["max_logical_attempts"] <= 100000, "finite logical budget")
    require(type(doc["contexts"]) is list and 0 < len(doc["contexts"]) <= 100000, "bounded context roster")
    contexts, parents = {}, {}
    for row in doc["contexts"]:
        closed(row, {"context_id", "parent_lineage_sha256", "partition", "identity", "metric", "baseline"}, "context")
        cid = sha(row["context_id"])
        require(cid not in contexts, "duplicate context")
        parent = sha(row["parent_lineage_sha256"])
        require(row["partition"] == "train", "only TRAIN contexts allowed")
        require(parents.setdefault(parent, row["partition"]) == row["partition"], "parent role conflict")
        identity(row["identity"])
        require(row["metric"] in DIRECTIONS, "native metric required")
        closed(row["baseline"], {"program_sha256", "value"}, "measured baseline")
        sha(row["baseline"]["program_sha256"])
        baseline = number(row["baseline"]["value"])
        if row["metric"] in ("rmse", "logloss", "mase"):
            require(baseline >= 0, "nonnegative measured loss baseline")
        contexts[cid] = row
    return doc, contexts


def reference(ref, raw):
    closed(ref, {"s3_uri", "version_id", "sha256", "bytes"}, "immutable result reference")
    require(type(ref["s3_uri"]) is str and re.fullmatch(r"s3://[^/?#]+/[^?#]+", ref["s3_uri"]) is not None, "exact S3 object URI required")
    require(type(ref["version_id"]) is str and ref["version_id"] not in ("", "null", "latest"), "immutable object VersionId required")
    sha(ref["sha256"])
    require(type(raw) is bytes and type(ref["bytes"]) is int and 0 < ref["bytes"] <= MAX_RESULT_BYTES
            and len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "exact result bytes/hash required")


def compile_target(context, attempt_id, program, raw):
    row = decode(raw)
    closed(row, {"schema", "attempt_id", "context_id", "parent_lineage_sha256", "partition", "identity", "program_sha256",
                 "dsl", "status", "score", "runtime_seconds", "reported_fresh_fit"}, "measured outcome")
    require(row["schema"] == "jope.measured-feedback-outcome/v1", "measured outcome version")
    require(row["attempt_id"] == attempt_id and row["context_id"] == context["context_id"]
            and row["parent_lineage_sha256"] == context["parent_lineage_sha256"] and row["partition"] == "train", "attempt/context/role identity mismatch")
    identity(row["identity"])
    require(row["identity"] == context["identity"], "noncomparable native HWM identity")
    require(row["program_sha256"] == program and type(row["dsl"]) is str
            and 0 < len(row["dsl"].encode()) <= 32768 and hashlib.sha256(row["dsl"].encode()).hexdigest() == program, "measured program byte identity")
    require(type(row["reported_fresh_fit"]) is bool, "explicit reported fit/cache provenance required")
    if row["runtime_seconds"] is not None:
        require(number(row["runtime_seconds"]) >= 0, "nonnegative measured runtime")
    require(row["status"] in ("completed", "failed", "timeout", "right_censored"), "known terminal outcome status")
    if row["status"] != "completed":
        require(row["score"] is None, "failed/censored attempt cannot supply numerical reward")
        return None, row
    score = row["score"]
    closed(score, {"metric", "direction", "units", "value"}, "score")
    require(score["metric"] == context["metric"] and score["direction"] == DIRECTIONS[score["metric"]], "metric/direction mismatch")
    require(score["units"] == ("original_target_units" if score["metric"] == "rmse" else "dimensionless"), "native metric units mismatch")
    value = number(score["value"])
    baseline = number(context["baseline"]["value"])
    if score["metric"] in ("rmse", "logloss", "mase"):
        require(value >= 0 and baseline >= 0, "nonnegative loss metric")
    improved = value > baseline if score["direction"] == "higher" else value < baseline
    transition = "improved" if improved else "tie" if value == baseline else "not-improved"
    target = {"context_id": row["context_id"], "parent_lineage_sha256": row["parent_lineage_sha256"],
              "identity": row["identity"], "identity_sha256": digest(row["identity"]),
              "program_sha256": program, "measured_score": score, "baseline": context["baseline"],
              "prospective_hwm_transition": transition, "ce_candidate": improved, "ranking_candidate": True,
              "native_teaching_qualification": "pending", "training_admitted": False, "hwm_published": False}
    return target, row


class Journal:
    """Transactional reservation/result boundary. A failed key is still reserved.

    One immutable result per context/program in this manifest; another numerical
    seed or retry needs a separately declared experiment, never automatic refill.
    """

    def __init__(self, path, manifest_bytes):
        self.manifest, self.contexts = manifest(manifest_bytes)
        self.manifest_bytes = manifest_bytes
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.path = Path(path)
        require(self.path.is_absolute() and not any(p.is_symlink() for p in (self.path, *self.path.parents)), "absolute nonsymlink journal path")
        for suffix in ("-wal", "-shm"):
            require(not Path(str(path)+suffix).is_symlink(), "journal sidecar symlink")
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,context TEXT NOT NULL,program TEXT NOT NULL,
            state TEXT NOT NULL,result_sha TEXT,reference BLOB,raw BLOB,target BLOB,reason TEXT,
            UNIQUE(context,program));""")
        try:
            with self.transaction():
                old = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
                if old is None:
                    require(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0,
                            "existing liabilities have no bound manifest")
                    self.db.execute("INSERT INTO metadata VALUES('manifest',?)", (manifest_bytes,))
                else:
                    require(bytes(old["value"]) == manifest_bytes, "journal manifest drift")
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def reserve(self, cid, program):
        require(cid in self.contexts, "unknown/non-TRAIN context")
        sha(program)
        aid = digest({"manifest_sha256": self.manifest_sha256, "context_id": cid, "program_sha256": program})
        with self.transaction():
            old = self.db.execute("SELECT id FROM attempts WHERE context=? AND program=?", (cid, program)).fetchone()
            if old is not None:
                return old["id"]
            count = self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            require(count < self.manifest["max_logical_attempts"], "logical attempt budget exhausted")
            self.db.execute("INSERT INTO attempts(id,context,program,state) VALUES(?,?,?,'reserved')", (aid, cid, program))
        return aid

    def finish(self, aid, ref, raw, *, before_commit=None):
        reference(ref, raw)
        with self.transaction():
            attempt = self.db.execute("SELECT * FROM attempts WHERE id=?", (aid,)).fetchone()
            require(attempt is not None, "unreserved result")
            if attempt["state"] != "reserved":
                require(attempt["result_sha"] == ref["sha256"], "immutable attempt result conflict")
                return attempt["state"]
            try:
                target, row = compile_target(self.contexts[attempt["context"]], aid, attempt["program"], raw)
                state, reason = ("measured", None) if target is not None else (row["status"], "no numerical target")
            except (ValueError, TypeError, KeyError) as error:
                target, state, reason = None, "rejected-metadata", str(error)
            if target is not None:
                target.update(result_sha256=ref["sha256"], result_ref=ref, attempt_id=aid,
                              target_policy_sha256=self.manifest["target_policy_sha256"])
            self.db.execute("UPDATE attempts SET state=?,result_sha=?,reference=?,raw=?,target=?,reason=? WHERE id=?",
                            (state, ref["sha256"], canonical(ref), raw, canonical(target) if target else None, reason, aid))
            if before_commit is not None:
                before_commit()
        return state

    def snapshot(self):
        stored = self.db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        require(stored is not None and bytes(stored["value"]) == self.manifest_bytes, "stored manifest drift")
        rows = self.db.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        for row in rows:
            require(row["context"] in self.contexts and row["id"] == digest({"manifest_sha256": self.manifest_sha256,
                    "context_id": row["context"], "program_sha256": row["program"]}), "journal reservation identity drift")
            if row["state"] == "reserved":
                require(all(row[key] is None for key in ("result_sha", "reference", "raw", "target", "reason")),
                        "reserved attempt has uncommitted result data")
                continue
            require(row["raw"] is not None and row["reference"] is not None, "terminal result bytes missing")
            ref = decode(row["reference"])
            reference(ref, bytes(row["raw"]))
            require(ref["sha256"] == row["result_sha"], "stored result hash drift")
            try:
                target, outcome = compile_target(self.contexts[row["context"]], row["id"], row["program"], bytes(row["raw"]))
                state, reason = ("measured", None) if target is not None else (outcome["status"], "no numerical target")
            except (ValueError, TypeError, KeyError) as error:
                target, state, reason = None, "rejected-metadata", str(error)
            if target is not None:
                target.update(result_sha256=ref["sha256"], result_ref=ref, attempt_id=row["id"],
                              target_policy_sha256=self.manifest["target_policy_sha256"])
            require(row["state"] == state and row["reason"] == reason
                    and row["target"] == (canonical(target) if target is not None else None), "stored target compilation drift")
        targets = [decode(row["target"]) for row in rows if row["target"] is not None]
        pending = [row["id"] for row in rows if row["state"] == "reserved"]
        body = {"schema": "jope.measured-feedback-target-snapshot/v1", "manifest_sha256": self.manifest_sha256,
                "targets": targets, "charged_logical_attempts": len(rows), "unresolved_attempt_ids": pending,
                "terminal_without_target": [{"attempt_id": row["id"], "state": row["state"], "reason": row["reason"],
                                              "result_sha256": row["result_sha"]} for row in rows if row["target"] is None and row["state"] != "reserved"],
                "native_teaching_qualification": "pending", "training_admitted": False, "hwm_published": False}
        body["snapshot_sha256"] = digest(body)
        return body
