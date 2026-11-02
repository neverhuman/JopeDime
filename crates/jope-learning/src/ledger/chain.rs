use super::*;

impl Ledger {
    /// `<state>/<name>/ledger`.
    pub fn open(state: &Path, name: &str) -> Ledger {
        Ledger {
            root: state.join(name).join("ledger"),
            name: name.to_string(),
        }
    }

    /// Performs the root operation.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Performs the name operation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Performs the head operation and returns an error if it cannot be completed.
    pub fn head(&self) -> Result<Option<Head>> {
        let p = self.root.join("HEAD.json");
        if !p.exists() {
            return Ok(None);
        }
        let head: Head = serde_json::from_slice(&fs::read(&p)?)
            .with_context(|| format!("parse {}", p.display()))?;
        ensure!(head.schema == HEAD_SCHEMA, "HEAD.json schema mismatch");
        Ok(Some(head))
    }

    fn lock(&self) -> Result<fs::File> {
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join(".lock"))?;
        lock.lock()?;
        Ok(lock)
    }

    /// The `HEAD.json` this ledger should have, when the one on disk lags a sealed continuation
    /// of the chain it names. That is the only state an interrupted [`Ledger::append`] can leave:
    /// the batch's atomic rename completed and `HEAD.json`'s did not. Otherwise `None`. Read-only.
    ///
    /// A head is rolled forward only when all of these hold:
    /// - `HEAD.json` names a batch of this chain at the path, position, file hash and batch count
    ///   it claims, or it is absent;
    /// - at least one batch follows it;
    /// - every following batch is readable, sealed for this ledger under its own name, carries
    ///   the event count and hash its seal claims, and links to its predecessor's id and file
    ///   hash.
    ///
    /// Anything else stays a break for [`Ledger::verify`] to report. A batch file exists only
    /// once its rename has completed, and every reader ([`Ledger::events`]) already counts it as
    /// part of the ledger. Rolling the head forward therefore makes the anchor agree with what is
    /// already read; it admits nothing new. It cannot see a removed tail any better than before
    /// (see the module docs).
    pub fn lagging_head(&self) -> Result<Option<Head>> {
        let batches = self.batches()?;
        let Ok(head) = self.head() else {
            return Ok(None);
        };
        let (start, mut prev, mut batch_count, mut event_count) = match &head {
            None => (0, None, 0, 0),
            Some(h) => {
                let Some(i) = batches.iter().position(|(rel, _)| *rel == h.path) else {
                    return Ok(None);
                };
                let sha = sha256_hex(&fs::read(&batches[i].1)?);
                if sha != h.sha256
                    || h.batches != (i + 1) as u64
                    || h.path.rsplit('/').next() != Some(format!("{}.jsonl", h.batch).as_str())
                {
                    return Ok(None);
                }
                (
                    i + 1,
                    Some((h.batch.clone(), sha)),
                    h.batches,
                    h.events_total,
                )
            }
        };
        let mut tip = None;
        for (rel, path) in batches.iter().skip(start) {
            let Ok((events, seal, file_sha, events_sha)) = Ledger::read_batch(path) else {
                return Ok(None);
            };
            let stem = rel
                .rsplit('/')
                .next()
                .unwrap_or(rel)
                .trim_end_matches(".jsonl");
            let linked = match &prev {
                None => seal.prev_batch.is_none() && seal.prev_batch_sha256.is_none(),
                Some((b, s)) => {
                    seal.prev_batch.as_ref() == Some(b)
                        && seal.prev_batch_sha256.as_ref() == Some(s)
                }
            };
            if !linked
                || seal.batch != stem
                || seal.ledger != self.name
                || seal.events != events.len() as u64
                || seal.events_sha256 != events_sha
            {
                return Ok(None);
            }
            batch_count += 1;
            event_count += events.len() as u64;
            tip = Some(Head {
                schema: HEAD_SCHEMA.into(),
                ledger: self.name.clone(),
                batch: seal.batch.clone(),
                path: rel.clone(),
                sha256: file_sha.clone(),
                batches: batch_count,
                events_total: event_count,
            });
            prev = Some((seal.batch, file_sha));
        }
        Ok(tip)
    }

    /// Roll a lagging `HEAD.json` forward ([`Ledger::lagging_head`]). The caller holds the
    /// ledger lock. Returns the head it wrote, if any.
    fn roll_forward_locked(&self) -> Result<Option<Head>> {
        let Some(head) = self.lagging_head()? else {
            return Ok(None);
        };
        atomic_write(
            &self.root.join("HEAD.json"),
            &serde_json::to_vec_pretty(&head)?,
        )?;
        Ok(Some(head))
    }

    /// Recover from an interrupted append: roll a lagging `HEAD.json` forward under the ledger
    /// lock. Writes nothing (and takes no lock) when the head does not lag.
    pub fn recover(&self) -> Result<Option<Head>> {
        if self.lagging_head()?.is_none() {
            return Ok(None);
        }
        let _lock = self.lock()?;
        self.roll_forward_locked()
    }

    /// Append one sealed batch. Serialises against a ledger-wide lock.
    ///
    /// Two durable steps: the batch's rename, then `HEAD.json`'s. A crash between them leaves a
    /// sealed batch the head does not name; the next append (or [`Ledger::require_intact`])
    /// rolls the head forward before chaining onto it, instead of forking the chain.
    pub fn append(&self, events: &[Value], source: Value) -> Result<BatchReceipt> {
        ensure!(!events.is_empty(), "refusing to seal an empty batch");
        let lock = self.lock()?;
        self.roll_forward_locked()?;
        let head_path = self.root.join("HEAD.json");
        let head = self.head()?;
        // ULIDs are lexicographically ordered only ACROSS milliseconds. Within one,
        // `Ulid::new()` fills the low 80 bits with random entropy, so two batches sealed in the
        // same millisecond sort randomly against each other -- measured on this machine, 9923 of
        // 19995 consecutive same-millisecond pairs came out lexicographically reversed. Since
        // `batches()` orders by exactly this name and `verify()` walks the prev-links in that
        // order, an intact ledger was reported `prev-link` BROKEN whenever two appends landed in
        // one millisecond: a false alarm from the one structure whose job is telling the truth
        // about tampering, and one that only appears on hardware fast enough to do two appends
        // inside a millisecond.
        //
        // Appends are serialised by the lock taken above, and `head` is the batch being chained
        // onto, so clamping against it makes each id strictly greater than its predecessor by
        // construction -- across processes, not merely within one -- which is what `batches()`
        // needs in order to be sorting by anything meaningful.
        let ulid = next_batch_id(head.as_ref().map(|h| h.batch.as_str()))?.to_string();
        let ts = now_unix();
        let (yyyy, mm) = year_month(ts);
        let rel = format!("{yyyy}/{mm}/{ulid}.jsonl");
        let mut body = Vec::with_capacity(events.len() * 1024);
        for e in events {
            serde_json::to_writer(&mut body, e)?;
            body.push(b'\n');
        }
        let seal = Seal {
            schema: SEAL_SCHEMA.into(),
            ledger: self.name.clone(),
            batch: ulid.clone(),
            events: events.len() as u64,
            events_sha256: sha256_hex(&body),
            prev_batch: head.as_ref().map(|h| h.batch.clone()),
            prev_batch_sha256: head.as_ref().map(|h| h.sha256.clone()),
            source,
            tool: crate::TOOL_VERSION.into(),
            ts_unix: ts,
        };
        serde_json::to_writer(&mut body, &seal)?;
        body.push(b'\n');
        let path = self.root.join(&rel);
        ensure!(!path.exists(), "batch {rel} already exists");
        atomic_write(&path, &body)?;
        let sha = sha256_hex(&body);
        let new_head = Head {
            schema: HEAD_SCHEMA.into(),
            ledger: self.name.clone(),
            batch: ulid.clone(),
            path: rel.clone(),
            sha256: sha.clone(),
            batches: head.as_ref().map_or(0, |h| h.batches) + 1,
            events_total: head.as_ref().map_or(0, |h| h.events_total) + events.len() as u64,
        };
        atomic_write(&head_path, &serde_json::to_vec_pretty(&new_head)?)?;
        lock.unlock()?;
        Ok(BatchReceipt {
            batch: ulid,
            path: rel,
            sha256: sha,
            events: events.len() as u64,
            prev_batch_sha256: seal.prev_batch_sha256,
        })
    }

    /// Every batch file, relative path and absolute path, in ULID (= time) order.
    pub fn batches(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut out = Vec::new();
        if !self.root.exists() {
            return Ok(out);
        }
        for entry in walkdir::WalkDir::new(&self.root).min_depth(3).max_depth(3) {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let rel = p
                .strip_prefix(&self.root)?
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, p.to_path_buf()));
        }
        // Batch ids are strictly increasing by construction (see `next_batch_id`), which is
        // what makes this sort the append order; the yyyy/mm prefix agrees with them.
        out.sort_by(|a, b| {
            let stem = |r: &str| r.rsplit('/').next().unwrap_or(r).to_string();
            stem(&a.0).cmp(&stem(&b.0))
        });
        Ok(out)
    }

    /// Parse one batch: (events, seal, file sha256, recomputed events sha256).
    pub fn read_batch(path: &Path) -> Result<(Vec<Value>, Seal, String, String)> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let file_sha = sha256_hex(&bytes);
        let text = std::str::from_utf8(&bytes).context("batch is not UTF-8")?;
        ensure!(text.ends_with('\n'), "batch does not end with a newline");
        let last_start = text[..text.len() - 1].rfind('\n').map_or(0, |i| i + 1);
        let seal: Seal = serde_json::from_str(text[last_start..].trim_end())
            .with_context(|| format!("seal of {}", path.display()))?;
        ensure!(seal.schema == SEAL_SCHEMA, "seal schema mismatch");
        let body = &bytes[..last_start];
        let events_sha = sha256_hex(body);
        // `seal.events` is a `u64` read out of the file being checked for tampering, so it is not
        // an allocation size until it has been reconciled with what the file can actually hold.
        // Every event is at least one byte and its newline, so the body bounds the count outright;
        // a seal claiming `u64::MAX` used to reach `Vec::with_capacity` and abort the process with
        // `capacity overflow`, which walks straight past `verify`'s careful conversion of a
        // `read_batch` failure into a `Break` — the one command whose job is to report corruption
        // died on it, and so did `require_intact` on the write path. Refusing here makes it an
        // ordinary `unreadable` break, like every other malformed batch.
        let max_events = (body.len() / 2) as u64;
        ensure!(
            seal.events <= max_events,
            "seal of {} declares {} events, which {} bytes of batch body cannot hold (at most {})",
            path.display(),
            seal.events,
            body.len(),
            max_events
        );
        let mut events = Vec::with_capacity(seal.events as usize);
        for (i, line) in text[..last_start].lines().enumerate() {
            events.push(
                serde_json::from_str(line)
                    .with_context(|| format!("{} line {}", path.display(), i + 1))?,
            );
        }
        Ok((events, seal, file_sha, events_sha))
    }

    /// Every event of every batch that *can* be read, skipping the batches that cannot.
    ///
    /// Only for the verifiers. [`Ledger::verify`] already records an `unreadable` break for each
    /// batch it could not parse, and the tool whose job is to diagnose a broken ledger is the one
    /// tool that must not stop at the first break — a forged seal in batch 1 should not hide what
    /// is wrong with batches 2 to 40. Every other caller uses [`Ledger::events`] and fails closed,
    /// because a partial read is not a ledger and must never be mistaken for one.
    pub fn events_readable(&self) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        for (_, p) in self.batches()? {
            if let Ok((events, _, _, _)) = Ledger::read_batch(&p) {
                all.extend(events);
            }
        }
        Ok(all)
    }

    /// All events across all batches in chain order (does not verify).
    pub fn events(&self) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        for (_, p) in self.batches()? {
            let (events, _, _, _) = Ledger::read_batch(&p)?;
            all.extend(events);
        }
        Ok(all)
    }
}
