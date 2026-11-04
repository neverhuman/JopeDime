use super::*;

impl Outbox {
    /// Whether new work may start, from the durable pending state.
    pub fn admission(&self) -> Admission {
        if self.pending_entries >= self.limits.pause_entries
            || self.pending_bytes >= self.limits.pause_bytes
        {
            Admission::Paused {
                pending_entries: self.pending_entries,
                pending_bytes: self.pending_bytes,
            }
        } else {
            Admission::Open
        }
    }

    fn admit(&self, opens_work: bool, item_bytes: u64) -> Result<()> {
        if let (
            true,
            Admission::Paused {
                pending_entries,
                pending_bytes,
            },
        ) = (opens_work, self.admission())
        {
            return Err(Refused::Paused {
                pending_entries,
                pending_bytes,
            }
            .into());
        }
        if self.pending_entries + 1 > self.limits.max_entries
            || self.pending_bytes + item_bytes > self.limits.max_bytes
        {
            return Err(Refused::Full {
                pending_entries: self.pending_entries,
                pending_bytes: self.pending_bytes,
                item_bytes,
            }
            .into());
        }
        Ok(())
    }

    fn require_dependencies(&self, item: &Dependency, deps: &[Dependency]) -> Result<()> {
        let missing: Vec<String> = deps
            .iter()
            .map(Dependency::key)
            .filter(|k| !self.by_key.contains_key(k))
            .collect();
        ensure!(
            missing.is_empty(),
            "{} names {} item(s) the outbox has not recorded: {}. Append dependencies first; \
             nothing was written",
            item.key(),
            missing.len(),
            missing.join(", ")
        );
        Ok(())
    }

    /// Append one verified inventory object. Its bytes must be the canonical encoding of its
    /// digest and everything it names must already be recorded.
    pub fn append_object(&mut self, object: &Encoded) -> Result<Appended> {
        let item = Dependency::Object(ObjectRef {
            kind: object.kind,
            digest: object.digest.clone(),
        });
        let deps = item.verify(&object.bytes).with_context(|| {
            format!(
                "the outbox holds only verified canonical inventory objects; {} is not one. \
                 Nothing was written",
                item.key()
            )
        })?;
        ensure!(
            deps == object.dependencies,
            "{}: the dependencies offered differ from what its bytes name; nothing was written",
            item.key()
        );
        if let Some(seq) = self.seq_of(&item) {
            return Ok(Appended::Existing { seq });
        }
        self.require_dependencies(&item, &deps)?;
        let opens_work = object.kind == Kind::Event
            && matches!(
                decode::<Event>(&object.bytes)?.value().body,
                EventBody::Scheduled | EventBody::Retry { .. }
            );
        self.admit(opens_work, object.bytes.len() as u64)?;
        write_create_only(
            &self.object_path(object.kind, &object.digest),
            &object.bytes,
        )?;
        self.crash_point(Boundary::Payload)?;
        self.commit(item, &object.bytes, deps)
    }

    /// Append a worker result: an immutable artifact that must be exactly the bytes `expected`
    /// names (the reference the `completed` event will carry).
    pub fn append_result(&mut self, bytes: &[u8], expected: &BlobRef) -> Result<Appended> {
        let got = BlobRef::of(bytes);
        ensure!(
            got == *expected,
            "result is {} ({} bytes), not the referenced {} ({} bytes); nothing was written",
            got.sha256,
            got.bytes,
            expected.sha256,
            expected.bytes
        );
        self.append_blob(bytes)
    }

    /// Append a content-addressed blob.
    pub fn append_blob(&mut self, bytes: &[u8]) -> Result<Appended> {
        let item = Dependency::Blob(BlobRef::of(bytes));
        if let Some(seq) = self.seq_of(&item) {
            return Ok(Appended::Existing { seq });
        }
        self.admit(false, bytes.len() as u64)?;
        self.state.put_blob(bytes)?;
        self.crash_point(Boundary::Payload)?;
        self.commit(item, bytes, Vec::new())
    }

    fn commit(
        &mut self,
        item: Dependency,
        bytes: &[u8],
        deps: Vec<Dependency>,
    ) -> Result<Appended> {
        let seq = self.entries.len() as u64 + 1;
        let record = Record {
            schema: RECORD_SCHEMA.into(),
            seq,
            prev_record_sha256: self.entries.last().map(|e| e.record_sha256.clone()),
            item,
            bytes: bytes.len() as u64,
            bytes_sha256: sha256_hex(bytes),
            dependencies: deps,
        };
        let encoded = canonical_json(&record)?;
        let path = self.record_path(seq);
        ensure!(
            !path.exists(),
            "outbox record {} already exists although this writer holds the lock",
            path.display()
        );
        atomic_write(&path, &encoded)?;
        self.index(record, sha256_hex(&encoded));
        self.crash_point(Boundary::Record)?;
        Ok(Appended::Added { seq })
    }
}
