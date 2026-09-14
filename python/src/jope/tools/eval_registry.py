"""Fail-closed evaluation registry. Source-only; not wired to live GP 774452.

Unknown / partial history is a tombstone, never a miss that authorizes a new fit.
A start token is consumed exactly once. rmse==0 is a valid success.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = "parallel.eval_registry.v1"
ADMISSION_FIELDS = (
    "program_sha256", "family", "partition", "prepared_sha256", "cut",
    "training_sha256", "validation_sha256", "evaluator_sha256", "evaluator_protocol",
    "preprocessing_sha256", "target_scale", "metric_units", "vocabulary_sha256",
    "feature_view_sha256", "evaluation_seed", "resource_protocol",
)
EXECUTION_FIELDS = (
    "evaluator_binary_sha256", "entry_point", "loaded_libs_sha256",
    "omp_threads", "rayon_threads", "fit_rng", "sealed_input_sha256",
    "resource_window", "admission_policy_sha256", "clean_completion_sha256",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def evaluation_key(admission, execution):
    missing = [k for k in ADMISSION_FIELDS if k not in admission]
    missing += [k for k in EXECUTION_FIELDS if k not in execution]
    if missing:
        return None, missing
    return digest({"admission": {k: admission[k] for k in ADMISSION_FIELDS},
                   "execution": {k: execution[k] for k in EXECUTION_FIELDS}}), ()


class EvalRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS evaluations (
                key TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                start_token TEXT UNIQUE,
                attempt_token TEXT,
                result_json TEXT,
                primary_score REAL,
                source TEXT,
                reason TEXT
            ) WITHOUT ROWID"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS tombstones (
                partial_key TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                evidence TEXT
            ) WITHOUT ROWID"""
        )
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            ) WITHOUT ROWID"""
        )
        self.db.commit()

    def _bump(self, name, n=1):
        cur = self.db.execute("SELECT value FROM counters WHERE name=?", (name,))
        row = cur.fetchone()
        value = (row[0] if row else 0) + n
        self.db.execute("INSERT OR REPLACE INTO counters(name,value) VALUES(?,?)", (name, value))
        return value

    def counters(self):
        return {name: value for name, value in self.db.execute("SELECT name,value FROM counters")}

    def partial_key(self, admission):
        present = {k: admission[k] for k in ADMISSION_FIELDS if k in admission}
        return digest({"partial": present})

    def tombstone(self, admission, reason, evidence=None):
        key = self.partial_key(admission)
        self.db.execute(
            "INSERT OR IGNORE INTO tombstones(partial_key,reason,evidence) VALUES(?,?,?)",
            (key, reason, json.dumps(evidence) if evidence is not None else None),
        )
        self._bump("tombstone")
        self.db.commit()
        return key

    def is_tombstoned(self, admission):
        return self.db.execute(
            "SELECT 1 FROM tombstones WHERE partial_key=?", (self.partial_key(admission),)
        ).fetchone() is not None

    def lookup(self, admission, execution):
        self._bump("registry_lookup")
        if self.is_tombstoned(admission):
            self._bump("registry_tombstone")
            return dict(action="tombstone", authorize_fit=False)
        key, missing = evaluation_key(admission, execution)
        if key is None:
            self.tombstone(admission, "partial_identity", {"missing": missing})
            self._bump("registry_unknown")
            return dict(action="tombstone", authorize_fit=False, missing=missing)
        row = self.db.execute(
            "SELECT state,start_token,attempt_token,result_json,primary_score,source FROM evaluations WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            self._bump("registry_miss")
            return dict(action="absent", authorize_fit=False, key=key)
        state, start, attempt, result, score, source = row
        if state == "completed":
            self._bump("registry_hit")
            return dict(action="reuse", authorize_fit=False, key=key,
                        result=json.loads(result), primary_score=score, source=source)
        if state in ("failed", "timeout"):
            self._bump("registry_hit")
            return dict(action="reuse_terminal", authorize_fit=False, key=key, state=state)
        self._bump("registry_unknown")
        return dict(action=state, authorize_fit=False, key=key, start_token=start)

    def claim(self, admission, execution, owner):
        looked = self.lookup(admission, execution)
        if looked["action"] != "absent":
            return looked
        key = looked["key"]
        token = digest({"key": key, "owner": owner, "once": True})
        try:
            self.db.execute(
                "INSERT INTO evaluations(key,state,start_token,attempt_token,source) VALUES(?,?,?,?,?)",
                (key, "claimed", token, token, owner),
            )
        except sqlite3.IntegrityError:
            self.db.commit()
            return self.lookup(admission, execution)
        self._bump("registry_claim")
        self.db.commit()
        return dict(action="claimed", authorize_fit=False, key=key, start_token=token)

    def authorize_dispatch(self, key, start_token):
        cur = self.db.execute(
            "UPDATE evaluations SET state='dispatch_authorized' WHERE key=? AND start_token=? AND state='claimed'",
            (key, start_token),
        )
        self.db.commit()
        if cur.rowcount != 1:
            return dict(action="rejected", authorize_fit=False)
        self._bump("registry_authorized")
        return dict(action="dispatch_authorized", authorize_fit=True, key=key, start_token=start_token)

    def consume_start(self, key, start_token):
        cur = self.db.execute(
            "UPDATE evaluations SET state='running', start_token=NULL WHERE key=? AND start_token=? AND state='dispatch_authorized'",
            (key, start_token),
        )
        self.db.commit()
        if cur.rowcount != 1:
            self._bump("registry_replay_blocked")
            return dict(action="rejected", authorize_fit=False)
        self._bump("executed_fits")
        return dict(action="running", authorize_fit=True, key=key, attempt_token=start_token)

    def complete(self, key, attempt_token, result, primary_score, source="test"):
        if primary_score is None or not isinstance(primary_score, (int, float)):
            raise ValueError("primary_score required")
        # rmse/logloss/mase == 0 is a legal success.
        if primary_score < 0:
            raise ValueError("primary_score must be >= 0")
        cur = self.db.execute(
            "UPDATE evaluations SET state='completed', result_json=?, primary_score=?, source=? "
            "WHERE key=? AND attempt_token=? AND state='running'",
            (canonical(result).decode(), float(primary_score), source, key, attempt_token),
        )
        self.db.commit()
        if cur.rowcount != 1:
            return dict(action="rejected")
        self._bump("registry_completed")
        return dict(action="completed", authorize_fit=False)

    def fail(self, key, attempt_token, reason):
        cur = self.db.execute(
            "UPDATE evaluations SET state='failed', reason=? WHERE key=? AND attempt_token=? AND state='running'",
            (reason, key, attempt_token),
        )
        self.db.commit()
        return dict(action="failed" if cur.rowcount == 1 else "rejected", authorize_fit=False)

    def merge_counts(self, shards):
        """Attribute every repeat. Do not drop cross-shard hits from origin totals."""
        totals = {"repeats": 0, "hits_on_exact": 0, "hits_on_provisional": 0,
                  "cross_shard_hits": 0, "per_origin": {}}
        for shard in shards:
            repeats = int(shard.get("repeats") or 0)
            exact = int(shard.get("hits_on_exact") or 0)
            provisional = int(shard.get("hits_on_provisional") or 0)
            cross = int(shard.get("cross_shard_hits") or 0)
            totals["repeats"] += repeats
            totals["hits_on_exact"] += exact
            totals["hits_on_provisional"] += provisional
            totals["cross_shard_hits"] += cross
            origin = shard.get("origin") or "unknown"
            bucket = totals["per_origin"].setdefault(origin, 0)
            totals["per_origin"][origin] = bucket + exact + provisional + cross
        attributed = totals["hits_on_exact"] + totals["hits_on_provisional"] + totals["cross_shard_hits"]
        totals["unattributed"] = totals["repeats"] - attributed
        totals["wall_reclaimed_modeled"] = sum(float(s.get("modeled_seconds") or 0) for s in shards)
        totals["measured_core_seconds"] = sum(float(s.get("measured_core_seconds") or 0) for s in shards)
        return totals

    def shadow_replay(self, observations, owner="shadow"):
        """Claim/complete from existing clean observations. No new fits."""
        replayed = hits = tombstones = skipped = 0
        for row in observations or []:
            admission = row.get("admission") or {}
            execution = row.get("execution") or {}
            result = row.get("result") or {}
            score = row.get("primary_score")
            if row.get("failure") is not None or score is None:
                skipped += 1
                continue
            claimed = self.claim(admission, execution, owner)
            if claimed["action"] == "tombstone":
                tombstones += 1
                continue
            if claimed["action"] == "reuse":
                hits += 1
                continue
            if claimed["action"] != "claimed":
                skipped += 1
                continue
            self.authorize_dispatch(claimed["key"], claimed["start_token"])
            running = self.consume_start(claimed["key"], claimed["start_token"])
            if running["action"] != "running":
                skipped += 1
                continue
            self.complete(claimed["key"], running["attempt_token"], result, float(score), owner)
            replayed += 1
        counters = self.counters()
        return dict(
            schema="parallel.eval_registry_shadow.v1",
            replayed=replayed,
            registry_hit=counters.get("registry_hit", 0),
            tombstones=tombstones,
            skipped=skipped,
            executed_fits=counters.get("executed_fits", 0),
            live_gp_pid=774452,
            wired_to_live_gp=False,
            claimed_savings=False,
        )
