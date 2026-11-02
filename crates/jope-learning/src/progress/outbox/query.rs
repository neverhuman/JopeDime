use super::*;

impl Outbox {
    /// Performs the root operation.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The state tree this outbox belongs to: its blob store, and the host's own HWM ledger that
    /// publication checks decisions against.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Performs the limits operation.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// What [`Outbox::open`] found.
    pub fn recovery(&self) -> &Recovery {
        &self.recovery
    }

    /// Performs the records operation.
    pub fn records(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Performs the pending entries operation.
    pub fn pending_entries(&self) -> u64 {
        self.pending_entries
    }

    /// Performs the pending bytes operation.
    pub fn pending_bytes(&self) -> u64 {
        self.pending_bytes
    }

    /// Pending entries in delivery (sequence) order.
    pub fn pending(&self) -> Vec<Pending> {
        self.entries
            .iter()
            .filter(|e| e.ack.is_none())
            .map(|e| Pending {
                seq: e.record.seq,
                key: e.key.clone(),
                bytes: e.record.bytes,
            })
            .collect()
    }

    /// The sequence number `item` was recorded at.
    pub fn seq_of(&self, item: &Dependency) -> Option<u64> {
        self.by_key.get(&item.key()).copied()
    }

    /// Performs the record operation.
    pub fn record(&self, seq: u64) -> Option<&Record> {
        self.entry(seq).map(|e| &e.record)
    }

    /// Performs the ack operation.
    pub fn ack(&self, seq: u64) -> Option<&Ack> {
        self.entry(seq).and_then(|e| e.ack.as_ref())
    }

    /// The object recorded at `seq`, read back from its payload and verified against its record
    /// and its digest. `None` for a blob record or an unknown sequence number.
    pub fn object(&self, seq: u64) -> Result<Option<Encoded>> {
        let Some(entry) = self.entry(seq) else {
            return Ok(None);
        };
        let Dependency::Object(o) = &entry.record.item else {
            return Ok(None);
        };
        let bytes = self.read_payload(&entry.record.item)?;
        ensure!(
            sha256_hex(&bytes) == entry.record.bytes_sha256,
            "outbox payload of record {seq} ({}) changed after it was committed",
            entry.key
        );
        Encoded::from_bytes(o.kind, &o.digest, &bytes).map(Some)
    }

    pub(super) fn entry(&self, seq: u64) -> Option<&Entry> {
        usize::try_from(seq)
            .ok()
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| self.entries.get(i))
    }
}
