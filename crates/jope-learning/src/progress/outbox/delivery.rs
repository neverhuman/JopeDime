use super::*;

impl Outbox {
    /// Upload an already acknowledged item again, create-only, from its verified local payload:
    /// the repair when the remote lost an object after acknowledging it. `Ok(false)` when this
    /// outbox cannot back the key (never recorded, not yet acknowledged, or its local payload is
    /// missing or no longer verifies); nothing is written then. The existing ack is kept.
    pub fn redeliver(&self, remote: &dyn Remote, key: &str) -> Result<bool> {
        let Some(entry) = self.by_key.get(key).and_then(|&seq| self.entry(seq)) else {
            return Ok(false);
        };
        if entry.ack.is_none() {
            return Ok(false);
        }
        let Ok(bytes) = self.read_payload(&entry.record.item) else {
            return Ok(false);
        };
        if sha256_hex(&bytes) != entry.record.bytes_sha256 {
            return Ok(false);
        }
        remote.put_immutable(key, &bytes)?;
        Ok(true)
    }

    /// Deliver up to `limit` pending entries in sequence order: re-verify the payload, write it
    /// create-only, then acknowledge it. Stops at the first entry the remote does not accept,
    /// which stays pending, as does everything after it. A local integrity failure is an error.
    pub fn deliver(&mut self, remote: &dyn Remote, limit: Option<u64>) -> Result<Delivery> {
        let mut report = Delivery::default();
        let todo: Vec<usize> = (0..self.entries.len())
            .filter(|&i| self.entries[i].ack.is_none())
            .take(limit.map_or(usize::MAX, |l| usize::try_from(l).unwrap_or(usize::MAX)))
            .collect();
        for i in todo {
            let entry = self.entries[i].clone();
            let seq = entry.record.seq;
            let unacknowledged: Vec<String> = entry
                .record
                .dependencies
                .iter()
                .filter(|d| self.seq_of(d).and_then(|s| self.ack(s)).is_none())
                .map(Dependency::key)
                .collect();
            if !unacknowledged.is_empty() {
                report.failure = Some(DeliveryFailure {
                    seq,
                    key: entry.key.clone(),
                    error: format!(
                        "dependencies not yet acknowledged: {}",
                        unacknowledged.join(", ")
                    ),
                });
                break;
            }
            let bytes = self.read_payload(&entry.record.item)?;
            ensure!(
                sha256_hex(&bytes) == entry.record.bytes_sha256,
                "outbox payload of record {seq} ({}) changed after it was committed; nothing \
                 further is delivered",
                entry.key
            );
            report.attempted += 1;
            let outcome = match remote.put_immutable(&entry.key, &bytes) {
                Ok(outcome) => outcome,
                Err(error) => {
                    report.failure = Some(DeliveryFailure {
                        seq,
                        key: entry.key.clone(),
                        error: format!("{error:#}"),
                    });
                    break;
                }
            };
            self.crash_point(Boundary::RemotePut)?;
            let ack = Ack {
                schema: ACK_SCHEMA.into(),
                seq,
                key: entry.key.clone(),
                record_sha256: entry.record_sha256.clone(),
                bytes_sha256: entry.record.bytes_sha256.clone(),
                remote: remote.url().into(),
                outcome: match outcome {
                    PutOutcome::Created => Outcome::Created,
                    PutOutcome::Exists => Outcome::Existed,
                },
            };
            write_create_only(&self.ack_path(seq), &canonical_json(&ack)?)?;
            match ack.outcome {
                Outcome::Created => report.created += 1,
                Outcome::Existed => report.existed += 1,
            }
            self.entries[i].ack = Some(ack);
            self.pending_entries -= 1;
            self.pending_bytes -= entry.record.bytes;
            self.crash_point(Boundary::Ack)?;
        }
        report.pending = self.pending_entries;
        report.pending_bytes = self.pending_bytes;
        Ok(report)
    }
}
