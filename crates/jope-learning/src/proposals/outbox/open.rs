use super::*;

impl ProposalOutbox {
    /// Performs the dir operation.
    pub fn dir(state_root: &Path) -> PathBuf {
        state_root.join("proposals").join("delivery-v1")
    }

    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(
        state_root: &Path,
        limits: PendingLimits,
        accepted: &impl AcceptedProposalInputs,
    ) -> Result<Self> {
        limits.validate()?;
        let root = Self::dir(state_root);
        fs::create_dir_all(root.join("requests"))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("LOCK"))?;
        lock.try_lock()
            .map_err(|error| anyhow::anyhow!("proposal outbox is already owned: {error}"))?;
        let mut outbox = Self {
            root,
            limits,
            records: BTreeMap::new(),
            statuses: BTreeMap::new(),
            durable_bytes: 0,
            _lock: lock,
        };
        outbox.load(accepted)?;
        Ok(outbox)
    }

    fn load(&mut self, accepted: &impl AcceptedProposalInputs) -> Result<()> {
        let dir = self.root.join("requests");
        let mut entries = fs::read_dir(&dir)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        ensure!(
            entries.len() as u64 <= self.limits.max_startup_requests,
            "proposal startup index exceeds its explicit bound"
        );
        ensure!(
            entries.len() as u64 <= self.limits.max_requests,
            "proposal total-history request cap exceeded"
        );
        let mut bytes = NAMESPACE_INDEX_OVERHEAD_BYTES;
        for entry in entries {
            ensure!(
                entry.file_type()?.is_dir(),
                "unexpected proposal namespace entry"
            );
            let key = entry.file_name().to_string_lossy().into_owned();
            ensure!(is_hex64(&key), "invalid proposal request directory");
            let entry_size = request_directory_size(&entry.path(), self.limits)?;
            let record_path = entry.path().join("record.json");
            let record_bytes = match fs::read(&record_path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    bytes = bytes
                        .checked_add(entry_size)
                        .context("proposal byte accounting overflow")?;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            let record: ProposalRecord = serde_json::from_slice(&record_bytes)?;
            ensure!(
                canonical_json(&record)? == record_bytes,
                "proposal record is not canonical"
            );
            ensure!(
                record.schema == RECORD_SCHEMA,
                "unsupported proposal record schema"
            );
            ensure!(
                record.request_key == key && record.context.physical_request_key()? == key,
                "proposal request key mismatch"
            );
            record.bundle.validate(self.limits)?;
            accepted.verify_request_context(&record.context)?;
            accepted.verify_draw_bundle(&record.bundle)?;
            let body = record_body(&record.context, &record.bundle)?;
            ensure!(
                sha256_hex(&body) == record.record_sha256,
                "proposal record content digest mismatch"
            );
            ensure!(
                record_bytes.len() as u64 <= self.limits.max_record_bytes,
                "proposal record exceeds byte bound"
            );
            let status_path = entry.path().join("status.json");
            let status_bytes = match fs::read(&status_path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let status = DeliveryStatus {
                        schema: STATUS_SCHEMA.to_string(),
                        request_key: key.clone(),
                        generation: 0,
                        lease: None,
                        attempts: Vec::new(),
                    };
                    let bytes = canonical_json(&status)?;
                    ensure!(
                        bytes.len() as u64 <= self.limits.max_status_bytes,
                        "initial proposal status exceeds byte bound"
                    );
                    atomic_write(&status_path, &bytes)?;
                    bytes
                }
                Err(error) => return Err(error.into()),
            };
            ensure!(
                status_bytes.len() as u64 <= self.limits.max_status_bytes,
                "proposal status exceeds byte bound"
            );
            let mut status: DeliveryStatus = serde_json::from_slice(&status_bytes)?;
            ensure!(
                canonical_json(&status)? == status_bytes,
                "proposal status is not canonical"
            );
            ensure!(
                status.schema == STATUS_SCHEMA,
                "unsupported proposal status schema"
            );
            validate_status(&record, &status, self.limits)?;
            // The exclusive process lock proves no previous writer is still executing. A
            // durable start without a terminal receipt is therefore recovered as UNKNOWN.
            if let Some(attempt) = status.attempts.last_mut() {
                if attempt.terminal.is_none() {
                    attempt.unknown = true;
                    status.lease = None;
                    let recovered = canonical_json(&status)?;
                    ensure!(
                        recovered.len() as u64 <= self.limits.max_status_bytes,
                        "recovered unknown status exceeds byte bound"
                    );
                    atomic_write(&status_path, &recovered)?;
                    bytes = bytes
                        .checked_add(request_directory_size(&entry.path(), self.limits)?)
                        .context("proposal byte accounting overflow")?;
                    validate_status(&record, &status, self.limits)?;
                    self.records.insert(key.clone(), record);
                    self.statuses.insert(key, status);
                    continue;
                }
            }
            bytes = bytes
                .checked_add(request_directory_size(&entry.path(), self.limits)?)
                .context("proposal byte accounting overflow")?;
            self.records.insert(key.clone(), record);
            self.statuses.insert(key, status);
        }
        ensure!(
            bytes <= self.limits.max_total_bytes,
            "proposal state exceeds total byte bound"
        );
        self.durable_bytes = bytes;
        Ok(())
    }
}
