"""Offline budget reservations. A reservation is never permission to launch a job."""

import argparse
import datetime as dt
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

HOSTS = frozenset({"xbabe1", "xbabe2", "xbabe3"})
PLANNED_SECONDS = 16 * 3600
HARD_SECONDS = 24 * 3600
DEADLINE = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
PROTOCOL = "jope.dl-gpu-budget/v2"
MAX_OUTPUT_ATTEMPTS = 2


class Refusal(ValueError):
    pass


def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise Refusal("GPU seconds must be a positive integer")
    return value


def validate_job(job, now):
    if now.tzinfo is None or now.utcoffset() != dt.timedelta(0):
        raise Refusal("clock must be timezone-aware UTC")
    if now >= DEADLINE:
        raise Refusal("original deadline expired")
    if not isinstance(job, dict) or set(job) != {
        "job_id",
        "host",
        "gpu_seconds",
        "output_ids",
    }:
        raise Refusal("job requires exactly job_id, host, gpu_seconds, output_ids")
    outputs = job["output_ids"]
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
            for v in outputs
        )
        or len(set(outputs)) != len(outputs)
    ):
        raise Refusal("output_ids must be unique full SHA256 expectation identities")
    if not isinstance(job["job_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,128}", job["job_id"]
    ):
        raise Refusal("invalid job identity")
    if not isinstance(job["host"], str) or job["host"] not in HOSTS:
        raise Refusal("host must be xbabe1, xbabe2 or xbabe3")
    seconds = positive_integer(job["gpu_seconds"])
    # Each named host has one admitted GPU; GPU seconds are also worst-case wall seconds.
    if seconds > HARD_SECONDS or now + dt.timedelta(seconds=seconds) > DEADLINE:
        raise Refusal("job exceeds hard cap or original deadline")
    return seconds


class Budget:
    """One persistent ledger shared by all hosts/retries; reservations are not refunded.

    BEGIN IMMEDIATE prevents concurrent planners from oversubscribing. Failures and
    cancellations retain their reserved charge; a restarted planner keeps this ledger.
    An actual overrun is recorded and closes the ledger to new reservations.
    """

    def __init__(self, path):
        self.path = Path(path)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, host TEXT NOT NULL, reserved INTEGER NOT NULL, actual INTEGER, outcome TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (job_id TEXT NOT NULL REFERENCES jobs(job_id), output_id TEXT NOT NULL, PRIMARY KEY(job_id, output_id))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS dispatches (job_id TEXT PRIMARY KEY REFERENCES jobs(job_id))"
            )
            policy = json.dumps(
                [
                    PROTOCOL,
                    sorted(HOSTS),
                    PLANNED_SECONDS,
                    HARD_SECONDS,
                    MAX_OUTPUT_ATTEMPTS,
                    DEADLINE.isoformat(),
                ]
            )
            db.execute("INSERT OR IGNORE INTO policy VALUES (1, ?)", (policy,))
            if (
                db.execute("SELECT value FROM policy WHERE id=1").fetchone()[0]
                != policy
            ):
                raise Refusal("persisted policy changed; cannot renew caps or deadline")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, job, *, contingency=False, now=None):
        now = now if now is not None else dt.datetime.now(dt.timezone.utc)
        seconds = validate_job(job, now)
        if type(contingency) is not bool:
            raise Refusal("contingency must be boolean")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT count(*) FROM jobs WHERE actual > reserved"
            ).fetchone()[0]:
                raise Refusal("measured overrun: stop and reconcile externally")
            old = db.execute(
                "SELECT host, reserved, outcome FROM jobs WHERE job_id=?",
                (job["job_id"],),
            ).fetchone()
            outputs = sorted(job["output_ids"])
            if old and old[2] is not None:
                raise Refusal(
                    "terminal job cannot be reserved again; retry needs a new charged job_id"
                )
            if old and (
                old[:2] != (job["host"], seconds)
                or outputs
                != [
                    r[0]
                    for r in db.execute(
                        "SELECT output_id FROM attempts WHERE job_id=? ORDER BY output_id",
                        (job["job_id"],),
                    )
                ]
            ):
                raise Refusal("retry changed the reservation identity")
            if not old:
                for output in outputs:
                    if (
                        db.execute(
                            "SELECT count(*) FROM attempts WHERE output_id=?", (output,)
                        ).fetchone()[0]
                        >= MAX_OUTPUT_ATTEMPTS
                    ):
                        raise Refusal("two-attempt output cap exhausted")
            charged = db.execute(
                "SELECT coalesce(sum(max(reserved, coalesce(actual, 0))),0) FROM jobs"
            ).fetchone()[0]
            proposed = charged + (0 if old else seconds)
            limit = HARD_SECONDS if contingency else PLANNED_SECONDS
            if proposed > limit:
                raise Refusal(
                    "hard GPU cap exceeded"
                    if contingency
                    else "16-hour plan exceeded; contingency requires separate admission"
                )
            if not old:
                db.execute(
                    "INSERT INTO jobs(job_id,host,reserved) VALUES (?,?,?)",
                    (job["job_id"], job["host"], seconds),
                )
                db.executemany(
                    "INSERT INTO attempts VALUES (?,?)",
                    ((job["job_id"], output) for output in outputs),
                )
        return {
            "protocol": PROTOCOL,
            "budget_reserved": True,
            "execution_admitted": False,
            "job_id": job["job_id"],
            "new_reservation": not bool(old),
            "output_ids": outputs,
            "charged_gpu_seconds": proposed,
            "contingency_seconds": max(0, proposed - PLANNED_SECONDS),
            "deadline": DEADLINE.isoformat(),
        }

    def claim_dispatch(self, job, *, now=None):
        """Single-use dispatch claim; still requires separate runtime admission.

        Consume before starting work. A lost acknowledgement remains spent, so a
        second delivery cannot launch again. Recovery uses a newly charged job.
        """
        validate_job(job, now if now is not None else dt.datetime.now(dt.timezone.utc))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT host,reserved,outcome FROM jobs WHERE job_id=?",
                (job["job_id"],),
            ).fetchone()
            outputs = [
                r[0]
                for r in db.execute(
                    "SELECT output_id FROM attempts WHERE job_id=? ORDER BY output_id",
                    (job["job_id"],),
                )
            ]
            if old != (job["host"], job["gpu_seconds"], None) or outputs != sorted(
                job["output_ids"]
            ):
                raise Refusal("dispatch requires the exact nonterminal reservation")
            if db.execute(
                "SELECT count(*) FROM jobs WHERE actual > reserved"
            ).fetchone()[0]:
                raise Refusal("measured overrun: stop and reconcile externally")
            if db.execute(
                "SELECT 1 FROM dispatches WHERE job_id=?", (job["job_id"],)
            ).fetchone():
                raise Refusal(
                    "dispatch already claimed; ambiguous work remains charged"
                )
            db.execute("INSERT INTO dispatches VALUES (?)", (job["job_id"],))
        return {
            "dispatch_claimed": True,
            "execution_admitted": False,
            "job_id": job["job_id"],
        }

    def finish(self, job_id, actual_seconds, outcome):
        if type(actual_seconds) is not int or actual_seconds < 0:
            raise Refusal("actual seconds must be a nonnegative integer")
        if outcome not in {"complete", "failed", "cancelled", "unknown"}:
            raise Refusal("invalid terminal outcome")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT actual,outcome FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if old is None:
                raise Refusal("unreserved job")
            if old != (None, None) and old != (actual_seconds, outcome):
                raise Refusal("terminal observation changed")
            db.execute(
                "UPDATE jobs SET actual=?, outcome=? WHERE job_id=?",
                (actual_seconds, outcome, job_id),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--job",
        type=Path,
        required=True,
        help="JSON with job_id, host, gpu_seconds, output_ids",
    )
    parser.add_argument(
        "--contingency",
        action="store_true",
        help="budget calculation only; not authorization",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                Budget(args.ledger).reserve(
                    json.loads(args.job.read_bytes()), contingency=args.contingency
                ),
                sort_keys=True,
            )
        )
    except (Refusal, OSError, json.JSONDecodeError, sqlite3.Error) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    main()
