"""Append-only comparable research ledger; predicted scores have no input field."""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

ARTIFACT_ROLES = frozenset(
    (
        "input",
        "program",
        "evaluation",
        "rust_source",
        "rust_executable",
        "rust_parity",
        "score_records",
    )
)
REFUSALS = frozenset(
    (
        "fitness_only",
        "parity_custody",
        "infra_unknown",
        "incompatible_key",
        "role_changed",
        "missing_closure",
    )
)


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(x in "0123456789abcdef" for x in value)
    )


@dataclass(frozen=True)
class HWMKey:
    context_identity: str
    dataset_version: str
    cut: str
    inner_folds: tuple[str, str]
    prepared_input_digest: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    metric: str
    direction: str
    baseline_epsilon_rule: str
    protocol: str
    vocabulary_sha256: str
    trie_sha256: str

    def __post_init__(self):
        if (
            type(self.inner_folds) is not tuple
            or len(self.inner_folds) != 2
            or not all(isinstance(f, str) and f for f in self.inner_folds)
            or self.inner_folds[0] == self.inner_folds[1]
        ):
            raise ValueError("two distinct pinned inner folds required")
        if any(
            not isinstance(value, str)
            for name, value in asdict(self).items()
            if name != "inner_folds"
        ):
            raise TypeError("comparison identity fields must be strings")
        if not all(asdict(self).values()):
            raise ValueError("all comparison identity fields required")
        for name in (
            "prepared_input_digest",
            "evaluator_sha256",
            "canonicalizer_sha256",
            "vocabulary_sha256",
            "trie_sha256",
        ):
            if not is_sha(getattr(self, name)):
                raise ValueError(f"{name}: full sha256 required")

    @property
    def digest(self):
        return sha256(asdict(self))


@dataclass(frozen=True)
class Publication:
    hwm_key: str
    program_sha256: str
    execution_key: str
    outcome: str
    q_grid: int | None = None
    artifacts: dict | None = None
    source: str = "measured"


class ResearchHWM:
    def __init__(self, path, *, state_root):
        self.path = Path(path).resolve()
        state_root = Path(state_root).resolve()
        if state_root.name != "state" or not self.path.is_relative_to(state_root):
            raise ValueError(
                "ledger must be beneath the explicit benchmark state directory"
            )
        if "flywheel_neural_continuous_20261009_v1" in self.path.parts:
            raise ValueError("service writes are forbidden")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS heads(
                    hwm_key TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    q_grid INTEGER, program_sha256 TEXT);
                CREATE TABLE IF NOT EXISTS events(
                    hwm_key TEXT NOT NULL, version INTEGER NOT NULL,
                    event TEXT NOT NULL, reason TEXT, publication TEXT NOT NULL,
                    best_q_grid INTEGER, best_program_sha256 TEXT,
                    PRIMARY KEY(hwm_key, version));
                CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT,'append-only events'); END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT,'append-only events'); END;
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.execute("PRAGMA busy_timeout=60000")
        return db

    def publish(self, key, publication, *, role_check):
        """role_check(key) runs freshly inside the transaction, never from a cached flag.

        Unknown/failing results carry no numeric score. Every event advances the
        CAS version, but only qualified strict improvements change the incumbent.
        """
        if not isinstance(key, HWMKey) or not isinstance(publication, Publication):
            raise TypeError("HWMKey and measured Publication required")
        reason = None
        if publication.hwm_key != key.digest:
            reason = "incompatible_key"
        elif publication.source == "fitness_only":
            reason = "fitness_only"
        elif publication.source != "measured":
            raise ValueError("predicted or other unmeasured scores are forbidden")
        elif publication.outcome in ("parity_custody", "infra_unknown"):
            reason = publication.outcome
        elif publication.outcome != "qualified_complete":
            if publication.q_grid is not None:
                raise ValueError("failure/unknown outcomes cannot carry q")
            reason = "missing_closure"
        elif type(publication.q_grid) is not int:
            raise TypeError("q_grid must be the measured integer micro-unit grid")
        elif not is_sha(publication.program_sha256) or not publication.execution_key:
            reason = "missing_closure"
        else:
            artifacts = publication.artifacts or {}
            if set(artifacts) != ARTIFACT_ROLES or any(
                not isinstance(ref, dict)
                or not is_sha(ref.get("sha256"))
                or not ref.get("version_id")
                or not ref.get("uri")
                for ref in artifacts.values()
            ):
                reason = "missing_closure"
        if (
            publication.outcome != "qualified_complete"
            and publication.q_grid is not None
        ):
            raise ValueError("failure/unknown outcomes cannot carry q")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not role_check(key):
                reason = "role_changed"
            db.execute(
                "INSERT OR IGNORE INTO heads VALUES (?,0,NULL,NULL)", (key.digest,)
            )
            version, best_q, best_program = db.execute(
                "SELECT version,q_grid,program_sha256 FROM heads WHERE hwm_key=?",
                (key.digest,),
            ).fetchone()
            if reason:
                event = "refused"
            elif best_q is None:
                event = "first_label"
                best_q, best_program = publication.q_grid, publication.program_sha256
            elif publication.q_grid > best_q:
                event = "improvement"
                best_q, best_program = publication.q_grid, publication.program_sha256
            elif publication.q_grid == best_q:
                event = "tie"
            else:
                event = "no_improvement"
            changed = db.execute(
                "UPDATE heads SET version=?,q_grid=?,program_sha256=? WHERE hwm_key=? AND version=?",
                (version + 1, best_q, best_program, key.digest, version),
            ).rowcount
            if changed != 1:
                raise RuntimeError("compare-and-swap conflict; retry publication")
            # Refusal rows keep custody but cannot contain a numeric score.
            evidence = asdict(publication)
            if reason:
                evidence["q_grid"] = None
            db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (
                    key.digest,
                    version + 1,
                    event,
                    reason,
                    canonical_bytes(evidence).decode(),
                    best_q,
                    best_program,
                ),
            )
            return {
                "hwm_key": key.digest,
                "version": version + 1,
                "event": event,
                "reason": reason,
                "q_grid": best_q,
                "program_sha256": best_program,
            }

    def events(self, key):
        with self._connect() as db:
            rows = db.execute(
                "SELECT version,event,reason,publication,best_q_grid FROM events "
                "WHERE hwm_key=? ORDER BY version",
                (key.digest,),
            ).fetchall()
        return [
            {
                "version": v,
                "event": e,
                "reason": r,
                "publication": json.loads(p),
                "best_q_grid": q,
            }
            for v, e, r, p, q in rows
        ]
