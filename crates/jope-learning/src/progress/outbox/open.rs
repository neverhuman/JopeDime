use super::*;

impl Outbox {
    /// `<state>/progress/outbox`.
    pub fn dir(state: &State) -> PathBuf {
        state.root().join("progress").join("outbox")
    }

    /// Open (creating if needed) and verify every committed record and acknowledgement. Fails
    /// closed on anything committed that does not verify: a sequence gap, a broken chain, a
    /// missing or altered payload, an acknowledgement for another record or other bytes.
    /// Uncommitted leftovers are reported and ignored.
    pub fn open(state: &State, limits: Limits) -> Result<Outbox> {
        limits.check()?;
        let root = Outbox::dir(state);
        for sub in ["log", "objects", "acks"] {
            fs::create_dir_all(root.join(sub))
                .with_context(|| format!("create {}", root.join(sub).display()))?;
        }
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("LOCK"))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => bail!(
                "outbox {} is held by another writer; an outbox has exactly one writer",
                root.display()
            ),
            Err(fs::TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("lock {}", root.display()))
            }
        }
        let mut outbox = Outbox {
            state: State::at(state.root()),
            root,
            limits,
            entries: Vec::new(),
            by_key: BTreeMap::new(),
            pending_entries: 0,
            pending_bytes: 0,
            recovery: Recovery::default(),
            _lock: lock,
            #[cfg(test)]
            crash_at: None,
        };
        outbox.load()?;
        Ok(outbox)
    }

    fn load(&mut self) -> Result<()> {
        let mut uncommitted = Vec::new();
        let seqs = scan_seqs(&self.root, "log", &mut uncommitted)?;
        ensure!(
            seqs.iter().copied().eq(1..=seqs.len() as u64),
            "outbox log is not the contiguous sequence 1..={}: found {:?}. A committed record is \
             missing; nothing is delivered from an outbox whose history has a hole",
            seqs.len(),
            seqs
        );
        for seq in seqs {
            let path = self.record_path(seq);
            let (record, bytes): (Record, Vec<u8>) = read_canonical(&path)?;
            self.check_record(&record, seq)
                .with_context(|| format!("outbox record {}", path.display()))?;
            self.index(record, sha256_hex(&bytes));
        }

        for seq in scan_seqs(&self.root, "acks", &mut uncommitted)? {
            let path = self.ack_path(seq);
            let (ack, _): (Ack, Vec<u8>) = read_canonical(&path)?;
            let entry = usize::try_from(seq)
                .ok()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| self.entries.get_mut(i))
                .with_context(|| format!("{}: acknowledges no record", path.display()))?;
            ensure!(
                ack.schema == ACK_SCHEMA
                    && ack.seq == seq
                    && ack.key == entry.key
                    && ack.record_sha256 == entry.record_sha256
                    && ack.bytes_sha256 == entry.record.bytes_sha256,
                "{}: acknowledges other bytes or another record than record {seq}",
                path.display()
            );
            entry.ack = Some(ack);
            self.pending_entries -= 1;
            self.pending_bytes -= entry.record.bytes;
        }

        let mut orphans = Vec::new();
        for kind in Kind::ALL {
            let dir = self.root.join("objects").join(kind.dir());
            if !dir.exists() {
                continue;
            }
            for entry in fs::read_dir(&dir)? {
                let name = entry?.file_name().to_string_lossy().into_owned();
                let rel = format!("objects/{}/{name}", kind.dir());
                if is_temporary(&name) {
                    uncommitted.push(rel);
                    continue;
                }
                let digest = name.strip_suffix(".json").unwrap_or(&name);
                let item = Dependency::Object(ObjectRef {
                    kind,
                    digest: digest.into(),
                });
                if !self.by_key.contains_key(&item.key()) {
                    orphans.push(rel);
                }
            }
        }
        uncommitted.sort();
        orphans.sort();
        self.recovery = Recovery {
            records: self.entries.len() as u64,
            acknowledged: self.entries.iter().filter(|e| e.ack.is_some()).count() as u64,
            pending: self.pending_entries,
            pending_bytes: self.pending_bytes,
            uncommitted,
            orphan_payloads: orphans,
        };
        Ok(())
    }

    fn check_record(&self, record: &Record, seq: u64) -> Result<()> {
        ensure!(record.schema == RECORD_SCHEMA, "schema {:?}", record.schema);
        ensure!(record.seq == seq, "names sequence {}", record.seq);
        ensure!(
            record.prev_record_sha256 == self.entries.last().map(|e| e.record_sha256.clone()),
            "does not chain onto record {}",
            seq - 1
        );
        record.item.check()?;
        ensure!(
            is_hex64(&record.bytes_sha256),
            "bytes_sha256 is not a sha256"
        );
        ensure!(
            !self.by_key.contains_key(&record.item.key()),
            "records {} a second time",
            record.item.key()
        );
        ensure!(
            record.dependencies.windows(2).all(|w| w[0] < w[1]),
            "dependencies are not sorted and unique"
        );
        for dep in &record.dependencies {
            ensure!(
                self.by_key.contains_key(&dep.key()),
                "names {} before it was recorded",
                dep.key()
            );
        }
        let bytes = self.read_payload(&record.item)?;
        ensure!(
            bytes.len() as u64 == record.bytes && sha256_hex(&bytes) == record.bytes_sha256,
            "payload of {} is not the recorded {} bytes {}",
            record.item.key(),
            record.bytes,
            record.bytes_sha256
        );
        ensure!(
            record.item.verify(&bytes)? == record.dependencies,
            "payload of {} names other dependencies than the record",
            record.item.key()
        );
        Ok(())
    }

    pub(super) fn index(&mut self, record: Record, record_sha256: String) {
        let key = record.item.key();
        self.by_key.insert(key.clone(), record.seq);
        self.pending_entries += 1;
        self.pending_bytes += record.bytes;
        self.entries.push(Entry {
            record,
            record_sha256,
            key,
            ack: None,
        });
    }

    pub(super) fn record_path(&self, seq: u64) -> PathBuf {
        self.root.join("log").join(seq_name(seq))
    }

    pub(super) fn ack_path(&self, seq: u64) -> PathBuf {
        self.root.join("acks").join(seq_name(seq))
    }

    pub(super) fn object_path(&self, kind: Kind, digest: &str) -> PathBuf {
        self.root
            .join("objects")
            .join(kind.dir())
            .join(format!("{digest}.json"))
    }

    pub(super) fn read_payload(&self, item: &Dependency) -> Result<Vec<u8>> {
        match item {
            Dependency::Object(o) => {
                let path = self.object_path(o.kind, &o.digest);
                fs::read(&path).with_context(|| format!("payload {} is missing", path.display()))
            }
            Dependency::Blob(b) => self
                .state
                .read_blob(&b.sha256)?
                .with_context(|| format!("blob {} is missing from the state store", b.sha256)),
        }
    }

    pub(super) fn crash_point(&self, at: Boundary) -> Result<()> {
        #[cfg(test)]
        if self.crash_at == Some(at) {
            bail!("simulated crash at {at:?}");
        }
        let _ = at;
        Ok(())
    }
}
