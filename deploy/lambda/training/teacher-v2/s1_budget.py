"""S1 fixed-equal attempt coordination. This module does not launch computation."""

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

GENERATIONS = 18
INITIALIZATION = 96
PER_GENERATION = 64
CONTEXTS = 4873
PER_CONTEXT = INITIALIZATION + GENERATIONS * PER_GENERATION
TOTAL = CONTEXTS * PER_CONTEXT
EVALUATION_TIMEOUT_SECONDS = 90
DEADLINE = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
PROTOCOL = "jope.s1-fixed-equal/v1"


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("full lowercase SHA256 required")
    return value


def contexts_for_stage(stage, rows_by_view):
    """Channel 4: never inspect HWM/test fields when deciding continuation."""
    if type(stage) is not int or not 1 <= stage <= GENERATIONS:
        raise ValueError("stage outside the fixed 18-generation budget")
    return [view for view, _ in rows_by_view]


class Ledger:
    """Reservations and unknown dispatches remain charged across restart.

    Ordinals 0..95 belong to G0; the next eighteen blocks of 64 belong to
    evolution. Retrying requires another available ordinal, not another free
    execution of the old attempt. The receiving supervisor must consume the
    single-use dispatch claim before launching any evaluation.
    """

    def __init__(self, path, plan):
        if set(plan) != {
            "contexts",
            "engine_sha256",
            "worker_sha256",
            "input_manifest_sha256",
        }:
            raise ValueError("incomplete frozen S1 plan")
        contexts = plan["contexts"]
        if not isinstance(contexts, list) or len(contexts) != CONTEXTS:
            raise ValueError("S1 plan must retain all 4,873 context dispositions")
        self.contexts = frozenset(sha(c) for c in contexts)
        if len(self.contexts) != CONTEXTS:
            raise ValueError("duplicate contexts")
        for field in ("engine_sha256", "worker_sha256", "input_manifest_sha256"):
            sha(plan[field])
        self.path = Path(path)
        self.policy = json.dumps(
            {
                "protocol": PROTOCOL,
                "plan": plan,
                "generations": GENERATIONS,
                "initialization": INITIALIZATION,
                "per_generation": PER_GENERATION,
                "total": TOTAL,
                "evaluation_timeout_seconds": EVALUATION_TIMEOUT_SECONDS,
                "deadline": DEADLINE.isoformat(),
            },
            sort_keys=True,
        )
        self.plan_sha256 = hashlib.sha256(self.policy.encode()).hexdigest()
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT PRIMARY KEY, context TEXT NOT NULL, ordinal INTEGER NOT NULL, program TEXT NOT NULL, dispatched INTEGER NOT NULL DEFAULT 0, outcome TEXT, UNIQUE(context, ordinal))"
            )
            db.execute("INSERT OR IGNORE INTO policy VALUES (1,?)", (self.policy,))
            if (
                db.execute("SELECT value FROM policy WHERE id=1").fetchone()[0]
                != self.policy
            ):
                raise ValueError("plan changed; no silent resume or renewed budget")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def check_clock(now):
        if (
            now.tzinfo is None
            or now.utcoffset() != dt.timedelta(0)
            or now + dt.timedelta(seconds=EVALUATION_TIMEOUT_SECONDS) > DEADLINE
        ):
            raise ValueError("original S1 deadline expired or non-UTC clock")

    def reserve(self, attempt_id, context, ordinal, program, *, now=None):
        self.check_clock(now if now is not None else dt.datetime.now(dt.timezone.utc))
        sha(attempt_id), sha(context), sha(program)
        if (
            context not in self.contexts
            or type(ordinal) is not int
            or not 0 <= ordinal < PER_CONTEXT
        ):
            raise ValueError("context/ordinal outside frozen equal allowance")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT context,ordinal,program,outcome FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if old is not None:
                if old != (context, ordinal, program, None):
                    raise ValueError(
                        "changed or terminal attempt cannot be reserved again"
                    )
                new = False
            else:
                if db.execute(
                    "SELECT 1 FROM attempts WHERE context=? AND ordinal=?",
                    (context, ordinal),
                ).fetchone():
                    raise ValueError(
                        "slot already charged; retry needs another available slot"
                    )
                db.execute(
                    "INSERT INTO attempts(attempt_id,context,ordinal,program) VALUES (?,?,?,?)",
                    (attempt_id, context, ordinal, program),
                )
                new = True
        return {
            "protocol": PROTOCOL,
            "plan_sha256": self.plan_sha256,
            "attempt_id": attempt_id,
            "new_reservation": new,
            "phase": "initialization" if ordinal < INITIALIZATION else "evolution",
            "generation": 0
            if ordinal < INITIALIZATION
            else 1 + (ordinal - INITIALIZATION) // PER_GENERATION,
            "execution_admitted": False,
        }

    def claim_dispatch(self, attempt_id, *, now=None):
        self.check_clock(now if now is not None else dt.datetime.now(dt.timezone.utc))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT dispatched,outcome FROM attempts WHERE attempt_id=?",
                (sha(attempt_id),),
            ).fetchone()
            if row != (0, None):
                raise ValueError("unreserved, terminal or already-dispatched attempt")
            db.execute(
                "UPDATE attempts SET dispatched=1 WHERE attempt_id=?", (attempt_id,)
            )
        return {
            "attempt_id": attempt_id,
            "dispatch_claimed": True,
            "execution_admitted": False,
        }

    def finish(self, attempt_id, outcome):
        if outcome not in {"complete", "failed", "cancelled", "unknown"}:
            raise ValueError("invalid terminal disposition")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT outcome FROM attempts WHERE attempt_id=?", (sha(attempt_id),)
            ).fetchone()
            if old is None or old[0] not in (None, outcome):
                raise ValueError("unreserved attempt or changed terminal disposition")
            db.execute(
                "UPDATE attempts SET outcome=? WHERE attempt_id=?",
                (outcome, attempt_id),
            )

    def status(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT ordinal,dispatched,outcome FROM attempts"
            ).fetchall()
        counts = {
            phase: {
                state: 0
                for state in (
                    "reserved",
                    "dispatched",
                    "complete",
                    "failed",
                    "cancelled",
                    "unknown",
                    "pending",
                )
            }
            for phase in ("initialization", "evolution")
        }
        for ordinal, dispatched, outcome in rows:
            phase = counts[
                "initialization" if ordinal < INITIALIZATION else "evolution"
            ]
            phase["reserved"] += 1
            phase["dispatched"] += dispatched
            phase[outcome or "pending"] += 1
        return {
            "protocol": PROTOCOL,
            "plan_sha256": self.plan_sha256,
            "attempt_ceiling": TOTAL,
            "charged_attempts": len(rows),
            "counts": counts,
            "execution_admitted": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--request",
        type=Path,
        help="Frozen reservation with attempt_id, context, ordinal, program",
    )
    args = parser.parse_args()
    try:
        raw = args.plan.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha(args.plan_sha256):
            raise ValueError("plan bytes differ from trusted pin")
        ledger = Ledger(args.ledger, json.loads(raw))
        result = (
            ledger.reserve(**json.loads(args.request.read_bytes()))
            if args.request
            else ledger.status()
        )
        print(json.dumps(result, sort_keys=True))
    except (ValueError, TypeError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    main()
