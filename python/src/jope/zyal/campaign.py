"""Campaign state: the run directory, the SQLite ledger, and the controller hash lock.

Nothing here is clever. Everything here is the part that, when it is missing, turns a completed
campaign into an unpublishable one:

* **Every artifact write goes through `atomic_json`.** A half-written matrix or receipt that is
  later read back as truth is indistinguishable from a result. tmp-then-`os.replace` is the only
  write path.
* **Budget is debited under `BEGIN IMMEDIATE`, before the work starts, not after.** Roles run
  concurrently; a read-modify-write on the budget row without an write lock held for the whole
  transaction will overspend under exactly the conditions where overspending is expensive. A lease
  is taken at the *estimate*, then reconciled to actual on finish, so a crashed role still costs
  something and cannot be retried for free forever.
* **The controller is sha256-frozen at preflight and re-verified at every epoch boundary.** The
  nomad reference run failed its own audit because the engine was edited between epochs, which
  silently means epoch 1 and epoch 3 measured different things. `verify_controller` returning a
  non-empty list is a stop-the-run condition, not a warning.

The path-traversal guard on `run_root` exists because the controller mints run ids from config, and
a run id of `../../../etc` would otherwise let a config file choose where artifacts land.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import MAX_ATTEMPTS_PER_ROLE, SCHEMA_VERSION

# Relative to the repo root. The run_root guard below treats this as an inescapable jail.
RUNS_SUBDIR = Path("prime") / "runs" / SCHEMA_VERSION

BUDGET_CEILING_USD = 1200.0
MIN_DEBIT_USD = 2.50

# Mirrors nomad's ARTIFACT_DIRS. Created eagerly so no writer ever has to decide whether to mkdir,
# which is where partially-populated run trees come from.
ARTIFACT_DIRS = (
    "inputs/baseline",
    "inputs/controller",
    "memory/negative",
    "novelty",
    "prompts",
    "raw_responses",
    "families",
    "specs",
    "queue/jobs",
    "matrices/temporary",
    "matrices/validated",
    "inbox",
    "inbox/findings",
    "judgments",
    "receipts/epochs",
    "receipts/features",
    "receipts/sandbox",
    "receipts/confirmation",
    "promotion",
    "final_manifests",
)

CONTROLLER_MANIFEST = "inputs/controller/manifest.json"
EVENTS_FILE = "events.jsonl"


class CampaignError(RuntimeError):
    """Run-structural failure: bad run root, budget exhausted, ledger inconsistency."""


# --- settings ----------------------------------------------------------------------------------


def _default_controller_src() -> Path:
    return Path(__file__).resolve().parent


@dataclass
class CampaignSettings:
    """Where a run lives and how big it is allowed to get.

    `run_root` is derived, never supplied: the controller owns every path in this campaign, and a
    settable run_root is a settable escape hatch.
    """

    run_id: str
    epochs: int = 3
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    budget_ceiling_usd: float = BUDGET_CEILING_USD
    controller_src: Path = field(default_factory=_default_controller_src)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.controller_src = Path(self.controller_src).resolve()
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise CampaignError("run_id must be a non-empty string")
        if self.epochs < 1:
            raise CampaignError(f"epochs must be >= 1, got {self.epochs}")
        # Cheap syntactic guard first, so the error names the actual problem.
        if any(sep in self.run_id for sep in ("/", "\\", os.sep)) or self.run_id in {".", ".."}:
            raise CampaignError(f"run_id may not contain a path separator: {self.run_id!r}")
        if self.run_id.startswith("."):
            raise CampaignError(f"run_id may not start with a dot: {self.run_id!r}")
        # Then the real guard: resolve and prove containment, symlinks and all.
        jail = self.runs_base
        candidate = (jail / self.run_id).resolve()
        if candidate == jail or jail not in candidate.parents:
            raise CampaignError(f"run_root {candidate} escapes {jail}; the controller owns every path")
        self._run_root = candidate

    @property
    def runs_base(self) -> Path:
        return (self.repo_root / RUNS_SUBDIR).resolve()

    @property
    def run_root(self) -> Path:
        return self._run_root

    @property
    def db_path(self) -> Path:
        return self.run_root / "campaign.sqlite3"

    @property
    def events_path(self) -> Path:
        return self.run_root / EVENTS_FILE

    @property
    def controller_dir(self) -> Path:
        return self.run_root / "inputs" / "controller"

    def path(self, *parts: str) -> Path:
        """Resolve a run-relative path, refusing anything that leaves the run root."""
        target = (self.run_root.joinpath(*parts)).resolve()
        if target != self.run_root and self.run_root not in target.parents:
            raise CampaignError(f"{target} escapes run root {self.run_root}")
        return target


def ensure_dirs(settings: CampaignSettings) -> None:
    """Create the full artifact tree. Idempotent."""
    settings.run_root.mkdir(parents=True, exist_ok=True)
    for rel in ARTIFACT_DIRS:
        settings.run_root.joinpath(*rel.split("/")).mkdir(parents=True, exist_ok=True)


# --- atomic io ---------------------------------------------------------------------------------


def atomic_json(path, obj) -> str:
    """Write `obj` as JSON via tmp + os.replace. Returns the sha256 of the bytes written.

    The digest is of the exact bytes on disk, so a receipt can be checked without re-serialising
    and hoping the encoder settings match.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    # The temp name MUST be unique per writer, not just per process. Keyed on PID alone, two
    # threads in one process share a single temp path: the second open(..., "wb") truncates the
    # first writer's still-open inode, os.replace publishes a half-written hybrid, and the first
    # writer then flushes into the *already-published* artifact -- corrupting it while reporting
    # FileNotFoundError to its caller, so the caller believes nothing was written. That is strictly
    # worse than a plain crash. uuid4 makes each writer's temp private.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # fsync the directory too: os.replace orders the rename against the data, but without this
        # the rename itself can be lost on power failure while the ledger (synchronous=FULL)
        # already records the write as durable.
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_event(settings: CampaignSettings, kind: str, payload: dict) -> None:
    """Append one JSON line to events.jsonl.

    A single small `O_APPEND` write is atomic on POSIX, so concurrent roles interleave lines but
    never corrupt one.
    """
    settings.run_root.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(
            {"ts_ms": int(time.time() * 1000), "kind": kind, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    with open(settings.events_path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


# --- controller freeze -------------------------------------------------------------------------


def _controller_sources(settings: CampaignSettings) -> dict[str, Path]:
    src = settings.controller_src
    return {
        p.relative_to(src).as_posix(): p for p in sorted(src.rglob("*.py")) if "__pycache__" not in p.parts
    }


def freeze_controller(settings: CampaignSettings) -> dict:
    """Snapshot every .py under the zyal package and sha256-lock it. Returns the manifest."""
    dest_root = settings.controller_dir
    dest_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for rel, path in _controller_sources(settings).items():
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files[rel] = sha256_file(path)
    manifest = {
        "frozen": True,
        "schema_version": SCHEMA_VERSION,
        "run_id": settings.run_id,
        "source_root": str(settings.controller_src),
        "files": files,
        "created_ms": int(time.time() * 1000),
    }
    manifest["manifest_sha256"] = atomic_json(settings.run_root / CONTROLLER_MANIFEST, manifest)
    return manifest


def verify_controller(settings: CampaignSettings) -> list[str]:
    """Re-hash the live controller against the frozen manifest. Returns changed relpaths.

    A non-empty return means epochs before and after this point did not run the same engine. Call
    at EVERY epoch boundary; nomad's reference run is unpublishable for want of this check.
    """
    manifest_path = settings.run_root / CONTROLLER_MANIFEST
    if not manifest_path.exists():
        raise CampaignError(f"controller is not frozen: {manifest_path} missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen: dict[str, str] = manifest["files"]
    live = _controller_sources(settings)

    changed: list[str] = []
    for rel, expected in frozen.items():
        path = live.get(rel)
        if path is None or not path.exists():
            changed.append(rel)
        elif sha256_file(path) != expected:
            changed.append(rel)
    changed.extend(rel for rel in live if rel not in frozen)
    return sorted(set(changed))


# --- ledger ------------------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    role_id     TEXT    NOT NULL,
    epoch       INTEGER NOT NULL,
    attempt     INTEGER NOT NULL,
    status      TEXT    NOT NULL,
    started_ms  INTEGER NOT NULL,
    finished_ms INTEGER,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (role_id, epoch, attempt)
);
CREATE TABLE IF NOT EXISTS budget (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    total_usd  REAL NOT NULL,
    spent_usd  REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS families (
    family_id TEXT    NOT NULL PRIMARY KEY,
    epoch     INTEGER NOT NULL,
    slug      TEXT    NOT NULL,
    json      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
    fingerprint TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    epoch       INTEGER NOT NULL,
    PRIMARY KEY (fingerprint, epoch)
);
CREATE INDEX IF NOT EXISTS attempts_role ON attempts (role_id);
CREATE INDEX IF NOT EXISTS families_epoch ON families (epoch);
"""


class CampaignStore:
    """SQLite ledger for attempts, budget, families and rejections.

    `isolation_level=None` because this class drives transactions explicitly: the budget
    read-modify-write must hold a write lock across both halves, which only `BEGIN IMMEDIATE`
    gives you. Python's implicit transaction management would start a deferred transaction and
    upgrade it mid-way, which is precisely the interleaving that overspends.
    """

    def __init__(self, settings: CampaignSettings, *, timeout: float = 30.0):
        self.settings = settings
        settings.run_root.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(settings.db_path), timeout=timeout, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA busy_timeout=%d" % int(timeout * 1000))
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO budget (id, total_usd, spent_usd) VALUES (1, ?, 0.0)",
            (float(settings.budget_ceiling_usd),),
        )

    # -- lifecycle
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> CampaignStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- budget
    def budget(self) -> dict:
        row = self.conn.execute("SELECT total_usd, spent_usd FROM budget WHERE id = 1").fetchone()
        return {
            "total_usd": row["total_usd"],
            "spent_usd": row["spent_usd"],
            "remaining_usd": row["total_usd"] - row["spent_usd"],
        }

    def submit_lease(self, role_id: str, epoch: int, est_cost: float) -> bool:
        """Reserve budget for one attempt. Returns False if refused; never raises on refusal.

        Debits the *estimate* up front (floored at MIN_DEBIT_USD) so concurrent leases cannot each
        see the same pre-spend balance. `finish_attempt` reconciles to the actual cost.
        """
        est = float(est_cost)
        if est < 0:
            raise CampaignError(f"est_cost must be >= 0, got {est}")
        debit = max(est, MIN_DEBIT_USD)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            used = self.conn.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE role_id = ?", (role_id,)
            ).fetchone()["n"]
            if used >= MAX_ATTEMPTS_PER_ROLE:
                self.conn.execute("ROLLBACK")
                return False

            row = self.conn.execute("SELECT total_usd, spent_usd FROM budget WHERE id = 1").fetchone()
            if row["spent_usd"] + debit > row["total_usd"]:
                self.conn.execute("ROLLBACK")
                return False

            self.conn.execute("UPDATE budget SET spent_usd = spent_usd + ? WHERE id = 1", (debit,))
            self.conn.execute(
                "INSERT INTO attempts (role_id, epoch, attempt, status, started_ms, cost_usd) "
                "VALUES (?, ?, ?, 'leased', ?, ?)",
                (role_id, int(epoch), used + 1, int(time.time() * 1000), debit),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return True

    def finish_attempt(self, role_id: str, epoch: int, status: str, actual_cost: float) -> None:
        """Close the open attempt and reconcile the lease to what was actually spent."""
        actual = max(float(actual_cost), MIN_DEBIT_USD)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT attempt, cost_usd FROM attempts "
                "WHERE role_id = ? AND epoch = ? AND status = 'leased' "
                "ORDER BY attempt DESC LIMIT 1",
                (role_id, int(epoch)),
            ).fetchone()
            if row is not None:
                self.conn.execute(
                    "UPDATE attempts SET status = ?, finished_ms = ?, cost_usd = ? "
                    "WHERE role_id = ? AND epoch = ? AND attempt = ?",
                    (status, int(time.time() * 1000), actual, role_id, int(epoch), row["attempt"]),
                )
                # Reconcile: the lease already moved `row["cost_usd"]`; move only the difference.
                # Clamped at zero so a refund can never drive spent_usd negative and hand back
                # budget that was never there.
                self.conn.execute(
                    "UPDATE budget SET spent_usd = MAX(0.0, spent_usd + ?) WHERE id = 1",
                    (actual - row["cost_usd"],),
                )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        # Reported after the transaction closes: raising inside the try would re-enter the except
        # branch and ROLLBACK an already-finished transaction, masking this error with
        # sqlite3.OperationalError.
        if row is None:
            raise CampaignError(f"no open attempt for {role_id!r} in epoch {epoch}")

    # -- attempts
    def attempts(self, role_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM attempts WHERE role_id = ? ORDER BY attempt", (role_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def attempt_count(self, role_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM attempts WHERE role_id = ?", (role_id,)
        ).fetchone()["n"]

    # -- families
    def record_family(self, family_id: str, epoch: int, slug: str, obj: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO families (family_id, epoch, slug, json) VALUES (?, ?, ?, ?)",
            (family_id, int(epoch), slug, json.dumps(obj, sort_keys=True, ensure_ascii=False)),
        )

    def families(self, epoch: int | None = None) -> list[dict]:
        if epoch is None:
            rows = self.conn.execute("SELECT * FROM families ORDER BY family_id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM families WHERE epoch = ? ORDER BY family_id", (int(epoch),)
            ).fetchall()
        return [
            {
                "family_id": r["family_id"],
                "epoch": r["epoch"],
                "slug": r["slug"],
                "json": json.loads(r["json"]),
            }
            for r in rows
        ]

    def recent_slugs(self, limit: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT slug FROM families ORDER BY rowid DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [r["slug"] for r in rows]

    # -- rejections
    def record_rejection(self, fingerprint: str, reason: str, epoch: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO rejections (fingerprint, reason, epoch) VALUES (?, ?, ?)",
            (fingerprint, reason, int(epoch)),
        )

    def is_rejected(self, fingerprint: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM rejections WHERE fingerprint = ? LIMIT 1", (fingerprint,)
        ).fetchone()
        return row is not None

    def rejections(self, epoch: int | None = None) -> list[dict]:
        if epoch is None:
            rows = self.conn.execute("SELECT * FROM rejections ORDER BY rowid").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM rejections WHERE epoch = ? ORDER BY rowid", (int(epoch),)
            ).fetchall()
        return [dict(r) for r in rows]
