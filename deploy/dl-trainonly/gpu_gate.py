"""Offline budget reservations. A reservation is never permission to launch a job."""

import argparse
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

HOSTS = frozenset({"xbabe1", "xbabe2", "xbabe3"})
PLANNED_SECONDS = 16 * 3600
HARD_SECONDS = 24 * 3600
DEADLINE = dt.datetime(2026, 10, 10, tzinfo=dt.timezone.utc)
PROTOCOL = "jope.dl-gpu-budget/v1"


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
    if not isinstance(job, dict) or set(job) != {"job_id", "host", "gpu_seconds"}:
        raise Refusal("job requires exactly job_id, host, gpu_seconds")
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
            policy = json.dumps(
                [
                    PROTOCOL,
                    sorted(HOSTS),
                    PLANNED_SECONDS,
                    HARD_SECONDS,
                    DEADLINE.isoformat(),
                ]
            )
            db.execute("INSERT OR IGNORE INTO policy VALUES (1, ?)", (policy,))
            if (
                db.execute("SELECT value FROM policy WHERE id=1").fetchone()[0]
                != policy
            ):
                raise Refusal("persisted policy changed; cannot renew caps or deadline")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

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
                "SELECT host, reserved FROM jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()
            if old and old != (job["host"], seconds):
                raise Refusal("retry changed the reservation identity")
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
        return {
            "protocol": PROTOCOL,
            "budget_reserved": True,
            "execution_admitted": False,
            "job_id": job["job_id"],
            "charged_gpu_seconds": proposed,
            "contingency_seconds": max(0, proposed - PLANNED_SECONDS),
            "deadline": DEADLINE.isoformat(),
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
        "--job", type=Path, required=True, help="JSON with job_id, host, gpu_seconds"
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
