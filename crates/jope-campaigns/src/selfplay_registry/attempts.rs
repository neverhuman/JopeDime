use super::*;

impl PairRegistry {
    pub fn bind_manifest(
        &mut self,
        identities: &[PairIdentity],
        owner: &str,
        manifest_path: &str,
        manifest_sha256: &str,
    ) -> Result<()> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        for identity in identities {
            validate_identity(identity)?;
            let changed = tx.execute(
                "UPDATE pairs SET manifest_path=?,manifest_sha256=?,updated_at_ms=?
                 WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?
                   AND state='reserved' AND owner=?",
                params![
                    manifest_path,
                    manifest_sha256,
                    now_ms(),
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256,
                    owner
                ],
            )?;
            if changed != 1 {
                bail!("manifest binding lost pair ownership");
            }
            tx.execute(
                "UPDATE attempts SET manifest_path=?,manifest_sha256=?
                 WHERE task=? AND dataset_hash=? AND semantic_sha256=? AND protocol_sha256=?
                   AND state='reserved'",
                params![
                    manifest_path,
                    manifest_sha256,
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_submission(
        &mut self,
        manifest_sha256: &str,
        owner: &str,
        job_id: &str,
    ) -> Result<usize> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed = tx.execute(
            "UPDATE pairs SET state='submitted',job_id=?,updated_at_ms=?
             WHERE manifest_sha256=? AND owner=? AND state='reserved'",
            params![job_id, now_ms(), manifest_sha256, owner],
        )?;
        if changed == 0 {
            let adopted: i64 = tx.query_row(
                "SELECT COUNT(*) FROM pairs WHERE manifest_sha256=? AND owner=?
                   AND state='submitted' AND job_id=?",
                params![manifest_sha256, owner, job_id],
                |row| row.get(0),
            )?;
            if adopted == 0 {
                bail!("submission has no owned reserved protocol pairs");
            }
            tx.commit()?;
            return Ok(adopted as usize);
        }
        tx.execute(
            "UPDATE attempts SET state='submitted',job_id=?
             WHERE manifest_sha256=? AND owner=? AND state='reserved'",
            params![job_id, manifest_sha256, owner],
        )?;
        tx.commit()?;
        Ok(changed)
    }

    pub fn supersede_unstarted(
        &mut self,
        manifest_sha256: &str,
        job_id: &str,
        retirement_receipt_sha256: &str,
    ) -> Result<usize> {
        if retirement_receipt_sha256.len() != 64 {
            bail!("retirement receipt is not a SHA-256 value");
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed_attempts = tx.execute(
            "UPDATE attempts SET state='superseded_unstarted',
               job_id=COALESCE(job_id,?),receipt_sha256=?,finalized_at_ms=?
             WHERE manifest_sha256=?
               AND ((state='submitted' AND job_id=?) OR state='reserved')",
            params![
                job_id,
                retirement_receipt_sha256,
                now_ms(),
                manifest_sha256,
                job_id
            ],
        )?;
        if changed_attempts == 0 {
            let prior: i64 = tx.query_row(
                "SELECT COUNT(*) FROM attempts WHERE manifest_sha256=? AND job_id=?
                   AND state='superseded_unstarted' AND receipt_sha256=?",
                params![manifest_sha256, job_id, retirement_receipt_sha256],
                |row| row.get(0),
            )?;
            if prior == 0 {
                bail!("retirement receipt matches no unstarted attempts");
            }
            tx.commit()?;
            return Ok(prior as usize);
        }
        let changed_pairs = tx.execute(
            "UPDATE pairs SET state='available',owner='available',
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               receipt_sha256=NULL,failure_class=NULL,updated_at_ms=?
             WHERE manifest_sha256=?
               AND ((state='submitted' AND job_id=?) OR state='reserved')",
            params![now_ms(), manifest_sha256, job_id],
        )?;
        if changed_pairs != changed_attempts {
            bail!("retirement attempt/pair count mismatch");
        }
        tx.commit()?;
        Ok(changed_pairs)
    }

    pub fn attempts(&self, identity: &PairIdentity) -> Result<Vec<Attempt>> {
        validate_identity(identity)?;
        let mut statement = self.connection.prepare(
            "SELECT attempt_number,state,owner,manifest_sha256,job_id,receipt_sha256
             FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=?
               AND protocol_sha256=? ORDER BY attempt_number",
        )?;
        let rows = statement.query_map(
            params![
                identity.task,
                identity.dataset_hash,
                identity.semantic_sha256,
                identity.protocol_sha256
            ],
            |row| {
                Ok(Attempt {
                    attempt_number: row.get(0)?,
                    state: row.get(1)?,
                    owner: row.get(2)?,
                    manifest_sha256: row.get(3)?,
                    job_id: row.get(4)?,
                    receipt_sha256: row.get(5)?,
                })
            },
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }
}
