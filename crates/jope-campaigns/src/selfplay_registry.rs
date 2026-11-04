//! Durable exactly-once pair and attempt registry for self-play.

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PairIdentity {
    pub task: String,
    pub dataset_hash: String,
    pub semantic_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairReservation {
    pub identity: PairIdentity,
    pub canonical_genome: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub attempt_number: i64,
    pub state: String,
    pub owner: String,
    pub manifest_sha256: Option<String>,
    pub job_id: Option<String>,
    pub receipt_sha256: Option<String>,
}

pub struct PairRegistry {
    connection: Connection,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis() as i64
}

fn validate_identity(identity: &PairIdentity) -> Result<()> {
    if !matches!(
        identity.task.as_str(),
        "regression" | "binary" | "timeseries" | "multiclass"
    ) {
        bail!("unsupported pair task {}", identity.task);
    }
    for (name, value) in [
        ("semantic_sha256", identity.semantic_sha256.as_str()),
        ("protocol_sha256", identity.protocol_sha256.as_str()),
    ] {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("{name} is not a SHA-256 value");
        }
    }
    if identity.dataset_hash.is_empty() {
        bail!("pair dataset identity is empty");
    }
    Ok(())
}

impl PairRegistry {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)
            .with_context(|| format!("open pair registry {}", path.display()))?;
        connection.busy_timeout(std::time::Duration::from_secs(60))?;
        connection.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS pairs(
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              semantic_sha256 TEXT NOT NULL,
              protocol_sha256 TEXT NOT NULL,
              canonical_genome TEXT NOT NULL,
              state TEXT NOT NULL,
              owner TEXT NOT NULL,
              manifest_path TEXT,
              manifest_sha256 TEXT,
              job_id TEXT,
              receipt_sha256 TEXT,
              failure_class TEXT,
              updated_at_ms INTEGER NOT NULL,
              PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS attempts(
              id INTEGER PRIMARY KEY,
              task TEXT NOT NULL,
              dataset_hash TEXT NOT NULL,
              semantic_sha256 TEXT NOT NULL,
              protocol_sha256 TEXT NOT NULL,
              attempt_number INTEGER NOT NULL,
              state TEXT NOT NULL,
              owner TEXT NOT NULL,
              manifest_path TEXT,
              manifest_sha256 TEXT,
              job_id TEXT,
              receipt_sha256 TEXT,
              failure_class TEXT,
              created_at_ms INTEGER NOT NULL,
              finalized_at_ms INTEGER,
              UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS attempts_one_active_pair
            ON attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
            WHERE state IN ('reserved','submitted');
            CREATE TABLE IF NOT EXISTS controls(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            ) WITHOUT ROWID;
            INSERT OR IGNORE INTO controls VALUES('admission_paused','0');
            INSERT OR IGNORE INTO controls VALUES('admission_halted','0');
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY,
              kind TEXT NOT NULL,
              task TEXT,
              dataset_hash TEXT,
              semantic_sha256 TEXT,
              protocol_sha256 TEXT,
              detail_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS pairs_dataset_protocol_v2
              ON pairs(task,dataset_hash,protocol_sha256,semantic_sha256,state);
            CREATE INDEX IF NOT EXISTS pairs_semantic_protocol_v2
              ON pairs(task,semantic_sha256,protocol_sha256,dataset_hash,state);
            "#,
        )?;
        Ok(Self { connection })
    }

    pub fn admission_paused(&self) -> Result<bool> {
        let value: String = self.connection.query_row(
            "SELECT value FROM controls WHERE key='admission_paused'",
            [],
            |row| row.get(0),
        )?;
        Ok(value == "1")
    }

    pub fn set_admission_paused(&self, paused: bool, reason: &str) -> Result<bool> {
        if reason.trim().is_empty() {
            bail!("admission state change requires a reason");
        }
        let prior = self.admission_paused()?;
        self.connection.execute(
            "UPDATE controls SET value=? WHERE key='admission_paused'",
            [if paused { "1" } else { "0" }],
        )?;
        self.connection.execute(
            "INSERT INTO events(kind,detail_json,created_at_ms) VALUES(?,?,?)",
            params![
                if paused {
                    "admission_paused"
                } else {
                    "admission_resumed"
                },
                serde_json::json!({"reason": reason}).to_string(),
                now_ms()
            ],
        )?;
        Ok(prior != paused)
    }

    pub fn reserve(&mut self, reservation: &PairReservation, owner: &str) -> Result<bool> {
        validate_identity(&reservation.identity)?;
        if owner.trim().is_empty() {
            bail!("reservation owner is empty");
        }
        if self.admission_paused()? {
            bail!("pair admission is paused");
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let identity = &reservation.identity;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT state,canonical_genome FROM pairs
                 WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?",
                params![
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((state, genome)) = existing {
            if genome != reservation.canonical_genome {
                bail!("semantic identity maps to a different canonical genome");
            }
            if state != "available" {
                tx.commit()?;
                return Ok(false);
            }
        }
        let attempt_number: i64 = tx.query_row(
            include_str!("../sql/selfplay_registry_next_attempt.sql"),
            params![
                identity.task,
                identity.dataset_hash,
                identity.semantic_sha256,
                identity.protocol_sha256
            ],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT INTO pairs VALUES(?,?,?,?,?,'reserved',?,NULL,NULL,NULL,NULL,NULL,?)
             ON CONFLICT(task,dataset_hash,semantic_sha256,protocol_sha256)
             DO UPDATE SET state='reserved',owner=excluded.owner,
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               receipt_sha256=NULL,failure_class=NULL,updated_at_ms=excluded.updated_at_ms",
            params![
                identity.task,
                identity.dataset_hash,
                identity.semantic_sha256,
                identity.protocol_sha256,
                reservation.canonical_genome,
                owner,
                now_ms()
            ],
        )?;
        tx.execute(
            "INSERT INTO attempts(
               task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number,
               state,owner,created_at_ms
             ) VALUES(?,?,?,?,?,'reserved',?,?)",
            params![
                identity.task,
                identity.dataset_hash,
                identity.semantic_sha256,
                identity.protocol_sha256,
                attempt_number,
                owner,
                now_ms()
            ],
        )?;
        tx.commit()?;
        Ok(true)
    }
}

mod attempts;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn reservation() -> PairReservation {
        PairReservation {
            identity: PairIdentity {
                task: "regression".into(),
                dataset_hash: "dataset".into(),
                semantic_sha256: "1".repeat(64),
                protocol_sha256: "2".repeat(64),
            },
            canonical_genome: "(pipe X (ridge))".into(),
        }
    }

    #[test]
    fn retirement_preserves_attempt_and_allows_new_reservation() {
        let root = tempdir().unwrap();
        let mut registry = PairRegistry::open(&root.path().join("pairs.sqlite")).unwrap();
        let value = reservation();
        assert!(registry.reserve(&value, "old").unwrap());
        registry
            .bind_manifest(
                std::slice::from_ref(&value.identity),
                "old",
                "/manifest.json",
                &"3".repeat(64),
            )
            .unwrap();
        registry
            .record_submission(&"3".repeat(64), "old", "job")
            .unwrap();
        registry
            .supersede_unstarted(&"3".repeat(64), "job", &"4".repeat(64))
            .unwrap();
        assert!(registry.reserve(&value, "new").unwrap());
        let attempts = registry.attempts(&value.identity).unwrap();
        assert_eq!(attempts.len(), 2);
        assert_eq!(
            attempts
                .iter()
                .map(|attempt| attempt.attempt_number)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(attempts[0].state, "superseded_unstarted");
        assert_eq!(attempts[1].state, "reserved");
    }

    #[test]
    fn pause_blocks_new_admission() {
        let root = tempdir().unwrap();
        let mut registry = PairRegistry::open(&root.path().join("pairs.sqlite")).unwrap();
        assert!(registry.set_admission_paused(true, "test").unwrap());
        assert!(registry.reserve(&reservation(), "owner").is_err());
    }

    #[test]
    fn pair_evidence_queries_use_composite_indexes() {
        let root = tempdir().unwrap();
        let registry = PairRegistry::open(&root.path().join("pairs.sqlite")).unwrap();
        let dataset_plan: String = registry
            .connection
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT semantic_sha256,state FROM pairs
                 WHERE task='regression' AND dataset_hash='dataset'
                   AND protocol_sha256='protocol'",
                [],
                |row| row.get(3),
            )
            .unwrap();
        assert!(dataset_plan.contains("pairs_dataset_protocol_v2"));
        let semantic_plan: String = registry
            .connection
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT dataset_hash,state FROM pairs
                 WHERE task='regression' AND semantic_sha256='semantic'
                   AND protocol_sha256='protocol'",
                [],
                |row| row.get(3),
            )
            .unwrap();
        assert!(semantic_plan.contains("pairs_semantic_protocol_v2"));
    }
}
