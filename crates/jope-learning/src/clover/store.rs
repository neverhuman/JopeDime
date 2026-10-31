use super::ExperimentIdentity;
use crate::canon::{canonical_json, is_hex64, sha256_hex};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OpenFlags, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

/// Local create-only evidence store. No remote or bucket operations.
pub struct EvidenceStore {
    root: PathBuf,
}

impl EvidenceStore {
    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("objects"))?;
        Ok(Self {
            root: root.to_owned(),
        })
    }

    /// Performs the read operation and returns an error if it cannot be completed.
    pub fn read(&self, digest: &str) -> Result<Vec<u8>> {
        ensure!(is_hex64(digest), "invalid evidence digest");
        let bytes = fs::read(self.root.join("objects").join(digest))?;
        ensure!(sha256_hex(&bytes) == digest, "evidence read-back mismatch");
        Ok(bytes)
    }

    /// Performs the put operation and returns an error if it cannot be completed.
    pub fn put(&self, bytes: &[u8]) -> Result<String> {
        let digest = sha256_hex(bytes);
        let objects = self.root.join("objects");
        let destination = objects.join(&digest);
        // Stage in the same filesystem and publish with a no-replace hard link.
        // A crash before linking cannot expose a partially written object.
        let staging = objects.join(format!(".{}", ulid::Ulid::new()));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        match fs::hard_link(&staging, &destination) {
            Ok(()) => fs::File::open(&objects)?.sync_all()?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        // A staging file is generated scratch, never the published evidence object.
        fs::remove_file(staging)?;
        ensure!(self.read(&digest)? == bytes, "immutable evidence conflict");
        Ok(digest)
    }

    /// Publish only after every payload is read back. A failing observer models
    /// loss/crash at each durable boundary, including a lost manifest response.
    pub fn publish(
        &self,
        experiment: &ExperimentIdentity,
        payloads: &[Vec<u8>],
        mut after_write: impl FnMut(usize, &str) -> Result<()>,
    ) -> Result<String> {
        let identity = experiment.sha256()?;
        ensure!(!payloads.is_empty(), "empty evidence closure");
        let identity_bytes = canonical_json(experiment)?;
        let objects = std::iter::once(&identity_bytes)
            .chain(payloads)
            .map(|bytes| sha256_hex(bytes))
            .collect::<Vec<_>>();
        let manifest = canonical_json(&serde_json::json!({
            "schema": "clover.evidence/v1", "experiment_sha256": identity, "objects": objects,
        }))?;
        let bytes = payloads.iter().try_fold(
            (identity_bytes.len() + manifest.len()) as u64,
            |sum, p| {
                sum.checked_add(p.len() as u64)
                    .context("evidence size overflow")
            },
        )?;
        ensure!(
            bytes <= experiment.budget.daily_bytes
                && payloads.len() as u64 + 2 <= experiment.budget.daily_puts,
            "evidence batch exceeds daily planning allowance"
        );
        for (index, bytes) in std::iter::once(&identity_bytes).chain(payloads).enumerate() {
            let digest = self.put(bytes)?;
            after_write(index, &digest)?;
            self.read(&digest)?;
        }
        // Recheck the whole closure immediately before publishing its manifest.
        for object in &objects {
            self.read(object)?;
        }
        let digest = self.put(&manifest)?;
        after_write(objects.len(), &digest)?;
        self.verify_manifest(&digest, experiment)?;
        Ok(digest)
    }

    /// Performs the verify manifest operation and returns an error if it cannot be completed.
    pub fn verify_manifest(&self, digest: &str, experiment: &ExperimentIdentity) -> Result<()> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Manifest {
            schema: String,
            experiment_sha256: String,
            objects: Vec<String>,
        }
        let m: Manifest = serde_json::from_slice(&self.read(digest)?)?;
        ensure!(
            m.schema == "clover.evidence/v1"
                && m.experiment_sha256 == experiment.sha256()?
                && m.objects.len() >= 2
                && m.objects[0] == experiment.sha256()?,
            "incomplete or foreign manifest"
        );
        for object in m.objects {
            self.read(&object)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// `AttemptOutcome` is a public attempt outcome enum.
pub enum AttemptOutcome {
    /// The `Qualified` case of `AttemptOutcome`.
    Qualified,
    /// The `Failed` case of `AttemptOutcome`.
    Failed,
    /// The `Timeout` case of `AttemptOutcome`.
    Timeout,
    /// The `ParityUnqualified` case of `AttemptOutcome`.
    ParityUnqualified,
    /// The `Unknown` case of `AttemptOutcome`.
    Unknown,
}

/// Durable lane bindings, charged attempts and a persistent HOLD latch. SQLite
/// transactions serialize writers; recovery never clears HOLD automatically.
pub struct RunStore {
    db: Connection,
    identity: ExperimentIdentity,
    digest: String,
}

// Any interrupted or unrecognized state must be reconciled, never assumed complete.
const UNRESOLVED: &str = "outcome NOT IN ('\"qualified\"','\"failed\"','\"timeout\"','\"parity_unqualified\"','\"unknown\"')";

impl RunStore {
    /// Explicit cold initialization; a lost registry is never implicitly reset by open.
    pub fn create(path: &Path, lane: &str, identity: ExperimentIdentity) -> Result<Self> {
        identity.validate()?;
        ensure!(!lane.trim().is_empty(), "empty lane label");
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .sync_all()?;
        Self::bind(path, lane, identity, true)
    }

    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(path: &Path, lane: &str, identity: ExperimentIdentity) -> Result<Self> {
        Self::bind(path, lane, identity, false)
    }

    fn bind(
        path: &Path,
        lane: &str,
        identity: ExperimentIdentity,
        initialize: bool,
    ) -> Result<Self> {
        let digest = identity.sha256()?;
        ensure!(!lane.trim().is_empty(), "empty lane label");
        let mut db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch("PRAGMA synchronous=FULL;")?;
        if initialize {
            db.execute_batch("
            CREATE TABLE IF NOT EXISTS lanes(label TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS holds(digest TEXT PRIMARY KEY, generation INTEGER NOT NULL, reason TEXT);
            CREATE TABLE IF NOT EXISTS attempts(digest TEXT, id TEXT, outcome TEXT NOT NULL, reward REAL, PRIMARY KEY(digest,id));
            CREATE TABLE IF NOT EXISTS recoveries(digest TEXT, generation INTEGER, manifest TEXT, PRIMARY KEY(digest,generation));")?;
        }
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if initialize {
            tx.execute("INSERT INTO holds VALUES(?1,0,NULL)", [&digest])?;
        } else {
            let _: u64 = tx
                .query_row(
                    "SELECT generation FROM holds WHERE digest=?1",
                    [&digest],
                    |r| r.get(0),
                )
                .context("missing experiment state; explicit recovery required")?;
        }
        tx.execute(
            "INSERT OR IGNORE INTO lanes VALUES(?1,?2)",
            params![lane, digest],
        )?;
        let bound: String =
            tx.query_row("SELECT digest FROM lanes WHERE label=?1", [lane], |r| {
                r.get(0)
            })?;
        ensure!(bound == digest, "conflicting lane label");
        tx.execute(&format!("UPDATE holds SET generation=generation+1,reason='unreconciled attempts' WHERE digest=?1 AND reason IS NULL AND EXISTS(SELECT 1 FROM attempts WHERE digest=?1 AND {UNRESOLVED})"), [&digest])?;
        tx.commit()?;
        Ok(Self {
            db,
            identity,
            digest,
        })
    }

    /// Performs the hold operation and returns an error if it cannot be completed.
    pub fn hold(&self, reason: &str) -> Result<()> {
        ensure!(!reason.trim().is_empty(), "empty HOLD reason");
        self.db.execute(
            "UPDATE holds SET generation=generation+1, reason=?2 WHERE digest=?1",
            params![self.digest, reason],
        )?;
        Ok(())
    }

    /// A failed publication (including a lost response) durably fences the run.
    /// Recovery evidence itself can still be retained through EvidenceStore.
    pub fn publish_evidence(
        &self,
        store: &EvidenceStore,
        payloads: &[Vec<u8>],
        after_write: impl FnMut(usize, &str) -> Result<()>,
    ) -> Result<String> {
        self.require_ready()?;
        let result = store.publish(&self.identity, payloads, after_write);
        if result.is_err() {
            self.hold("evidence publication unresolved")?;
        }
        result
    }

    /// Adoption and promotion callers must use this same gate as attempt admission.
    pub fn require_ready(&self) -> Result<()> {
        let (reason, running): (Option<String>, u64) = self.db.query_row(
            &format!("SELECT reason,(SELECT count(*) FROM attempts WHERE digest=?1 AND {UNRESOLVED}) FROM holds WHERE digest=?1"),
            [&self.digest],
            |r| Ok((r.get(0)?,r.get(1)?)),
        )?;
        ensure!(
            reason.is_none(),
            "persistent HOLD: {}",
            reason.unwrap_or_default()
        );
        ensure!(
            running == 0,
            "unterminated attempts block adoption/promotion"
        );
        Ok(())
    }

    /// Performs the start attempt operation and returns an error if it cannot be completed.
    pub fn start_attempt(&mut self, id: &str) -> Result<()> {
        ensure!(!id.trim().is_empty(), "empty attempt id");
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (held, count): (Option<String>, u64) = tx.query_row(
            "SELECT reason,(SELECT count(*) FROM attempts WHERE digest=?1) FROM holds WHERE digest=?1", [&self.digest], |r| Ok((r.get(0)?,r.get(1)?)))?;
        ensure!(held.is_none(), "persistent HOLD");
        ensure!(
            count < self.identity.budget.attempts,
            "attempt budget exhausted"
        );
        tx.execute(
            "INSERT INTO attempts VALUES(?1,?2,'running',NULL)",
            params![self.digest, id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Performs the finish attempt operation and returns an error if it cannot be completed.
    pub fn finish_attempt(
        &self,
        id: &str,
        outcome: AttemptOutcome,
        reward: Option<f64>,
    ) -> Result<()> {
        ensure!(
            match outcome {
                AttemptOutcome::Qualified => reward.is_some_and(f64::is_finite),
                _ => reward.is_none(),
            },
            "unknown/unqualified outcomes cannot become rewards"
        );
        let outcome = serde_json::to_string(&outcome)?;
        let changed = self.db.execute("UPDATE attempts SET outcome=?3,reward=?4 WHERE digest=?1 AND id=?2 AND outcome='running'", params![self.digest,id,outcome,reward])?;
        ensure!(changed == 1, "absent or already terminal attempt");
        Ok(())
    }

    /// Interrupted work becomes terminal UNKNOWN, retains its charged slot and no reward.
    pub fn recover(&mut self) -> Result<usize> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let count = tx.execute(&format!("UPDATE attempts SET outcome='\"unknown\"',reward=NULL WHERE digest=?1 AND {UNRESOLVED}"), [&self.digest])?;
        if count > 0 {
            tx.execute("UPDATE holds SET generation=generation+1,reason='interrupted attempts: unknown' WHERE digest=?1", [&self.digest])?;
        }
        tx.commit()?;
        Ok(count)
    }

    /// Performs the hold generation operation and returns an error if it cannot be completed.
    pub fn hold_generation(&self) -> Result<u64> {
        Ok(self.db.query_row(
            "SELECT generation FROM holds WHERE digest=?1",
            [&self.digest],
            |r| r.get(0),
        )?)
    }

    /// Explicit release binds the current HOLD generation to verified immutable
    /// recovery evidence and the pinned guard. A new violation makes it stale.
    pub fn release_hold(
        &mut self,
        generation: u64,
        manifest: &str,
        evidence: &EvidenceStore,
    ) -> Result<()> {
        evidence.verify_manifest(manifest, &self.identity)?;
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Recovery {
            schema: String,
            experiment_sha256: String,
            hold_generation: u64,
            guard_sha256: String,
        }
        let m: serde_json::Value = serde_json::from_slice(&evidence.read(manifest)?)?;
        let recovery_digest = m["objects"]
            .as_array()
            .context("missing closure")?
            .get(1)
            .and_then(|v| v.as_str())
            .context("missing recovery evidence")?;
        let recovery: Recovery = serde_json::from_slice(&evidence.read(recovery_digest)?)?;
        ensure!(
            recovery.schema == "clover.recovery/v1"
                && recovery.experiment_sha256 == self.digest
                && recovery.hold_generation == generation
                && recovery.guard_sha256 == self.identity.guards["hold_recovery"],
            "foreign recovery evidence"
        );
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let running: u64 = tx.query_row(
            &format!("SELECT count(*) FROM attempts WHERE digest=?1 AND {UNRESOLVED}"),
            [&self.digest],
            |r| r.get(0),
        )?;
        ensure!(running == 0, "unterminated attempts");
        let changed = tx.execute(
            "UPDATE holds SET reason=NULL WHERE digest=?1 AND generation=?2 AND reason IS NOT NULL",
            params![self.digest, generation],
        )?;
        ensure!(changed == 1, "stale or absent HOLD generation");
        tx.execute(
            "INSERT INTO recoveries VALUES(?1,?2,?3)",
            params![self.digest, generation, manifest],
        )?;
        tx.commit()?;
        Ok(())
    }
}
