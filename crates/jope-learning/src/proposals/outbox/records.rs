use super::*;

impl ProposalOutbox {
    /// Performs the recover operation and returns an error if it cannot be completed.
    pub fn recover(&self) -> Result<Recovery> {
        let pending = self
            .statuses
            .values()
            .filter(|s| {
                s.attempts.last().is_none_or(|a| {
                    a.terminal.is_none()
                        || matches!(
                            a.terminal.as_ref(),
                            Some(TerminalKind::Failed {
                                retryable: true,
                                ..
                            })
                        )
                })
            })
            .count() as u64;
        Ok(Recovery {
            requests: self.records.len() as u64,
            pending,
            durable_bytes: self.durable_bytes,
        })
    }

    /// Performs the insert operation and returns an error if it cannot be completed.
    pub fn insert(
        &mut self,
        context: RequestContext,
        bundle: ProposalBundle,
        accepted: &impl AcceptedProposalInputs,
    ) -> Result<Inserted> {
        context.validate()?;
        bundle.validate(self.limits)?;
        accepted.verify_request_context(&context)?;
        accepted.verify_draw_bundle(&bundle)?;
        let key = context.physical_request_key()?;
        if let Some(existing) = self.records.get(&key) {
            ensure!(
                existing.context == context && existing.bundle == bundle,
                "physical request key reused with different proposal content"
            );
            return Ok(Inserted::Existing { request_key: key });
        }
        ensure!(
            (self.records.len() as u64) < self.limits.max_requests,
            "proposal history request cap reached"
        );
        let request_dirs = fs::read_dir(self.root.join("requests"))?.count() as u64;
        ensure!(
            request_dirs < self.limits.max_requests,
            "proposal history includes uncommitted rows at its cap"
        );
        ensure!(
            self.recover()?.pending < self.limits.max_pending,
            "proposal pending cap reached; backpressure preserves existing receipts"
        );
        let mut record = ProposalRecord {
            schema: RECORD_SCHEMA.to_string(),
            request_key: key.clone(),
            context,
            bundle,
            record_sha256: String::new(),
        };
        record.record_sha256 = sha256_hex(&record_body(&record.context, &record.bundle)?);
        let record_bytes = canonical_json(&record)?;
        ensure!(
            record_bytes.len() as u64 <= self.limits.max_record_bytes,
            "proposal record exceeds per-record byte bound"
        );
        let status = DeliveryStatus {
            schema: STATUS_SCHEMA.to_string(),
            request_key: key.clone(),
            generation: 0,
            lease: None,
            attempts: Vec::new(),
        };
        let status_bytes = canonical_json(&status)?;
        let existing_bytes = self.durable_bytes;
        let mut existing_headroom = 0u64;
        for status in self.statuses.values() {
            let size = canonical_json(status)?.len() as u64;
            existing_headroom = existing_headroom
                .checked_add(
                    self.limits
                        .max_status_bytes
                        .checked_sub(size)
                        .context("proposal status exceeds reserved headroom")?,
                )
                .context("proposal terminal headroom accounting overflow")?;
        }
        let dir = self.request_dir(&key);
        let new_directory = !dir.exists();
        let status_path = dir.join("status.json");
        let new_status_bytes = if status_path.exists() {
            0
        } else {
            status_bytes.len() as u64
        };
        let row_overhead = if new_directory { 512 } else { 0 };
        ensure!(
            existing_bytes
                .checked_add(record_bytes.len() as u64)
                .and_then(|n| n.checked_add(new_status_bytes))
                .and_then(|n| n.checked_add(row_overhead))
                .and_then(|n| n.checked_add(existing_headroom))
                .and_then(|n| n.checked_add(self.limits.max_status_bytes))
                .is_some_and(|n| n <= self.limits.max_total_bytes),
            "proposal state lacks reserved terminal/index headroom"
        );
        if new_directory {
            fs::create_dir(&dir).context("create proposal request row")?;
            File::open(dir.parent().context("request index parent missing")?)?.sync_all()?;
        }
        let mut added_bytes = row_overhead + new_status_bytes;
        if status_path.exists() {
            let existing: DeliveryStatus = serde_json::from_slice(&fs::read(&status_path)?)?;
            ensure!(
                existing.request_key == key
                    && existing.attempts.is_empty()
                    && existing.generation == 0,
                "uncommitted proposal row has unexpected status; refusing overwrite"
            );
        } else {
            atomic_write(&status_path, &status_bytes)?;
        }
        // The immutable proposal record is the commit point. A crash before this atomic rename
        // leaves only a bounded, ignored row that a later identical insert can finish.
        atomic_write(&dir.join("record.json"), &record_bytes)?;
        added_bytes = added_bytes
            .checked_add(record_bytes.len() as u64)
            .context("proposal byte accounting overflow")?;
        self.durable_bytes = self
            .durable_bytes
            .checked_add(added_bytes)
            .context("proposal byte accounting overflow")?;
        self.records.insert(key.clone(), record);
        self.statuses.insert(key.clone(), status);
        Ok(Inserted::Added { request_key: key })
    }

    /// Performs the record operation and returns an error if it cannot be completed.
    pub fn record(&self, key: &str) -> Result<&ProposalRecord> {
        self.records.get(key).context("proposal request not found")
    }
    /// Performs the status operation and returns an error if it cannot be completed.
    pub fn status(&self, key: &str) -> Result<&DeliveryStatus> {
        self.statuses.get(key).context("proposal status not found")
    }

    /// Performs the total physical cost operation and returns an error if it cannot be completed.
    pub fn total_physical_cost(&self, key: &str) -> Result<u64> {
        self.status(key)?
            .attempts
            .iter()
            .try_fold(0u64, |sum, attempt| {
                sum.checked_add(attempt.physical_cost_units)
            })
            .context("physical attempt cost overflow")
    }
}
