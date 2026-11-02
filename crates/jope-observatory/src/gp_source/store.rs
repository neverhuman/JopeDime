use super::prelude::*;
use super::types::{Artifact, Event, LanguageMetrics, Payload, Status};
use super::{complete_line, emit, hash, EVENT_SCHEMA, LANGUAGES};

#[derive(Clone, Debug)]
struct Pending {
    occurrence_id: String,
    artifact: Artifact,
    inflight: bool,
}

/// Exclusive append-only source event store and in-memory accounting index.
pub struct Store {
    /// Root containing the event log, source files, and check diagnostics.
    pub root: PathBuf,
    log: File,
    _lock: File,
    /// Last durable event sequence loaded or appended.
    pub sequence: u64,
    seen: BTreeMap<String, String>,
    pending: BTreeMap<(String, String), Pending>,
    ready: VecDeque<(String, String)>,
    offsets: Vec<u64>,
    programs: BTreeSet<String>,
    /// Cumulative counters keyed by language identifier.
    pub metrics: BTreeMap<String, LanguageMetrics>,
    /// Cumulative source emission time in nanoseconds.
    pub emit_ns: u64,
    first_unix_ns: Option<u64>,
    partial_journal_tails: Vec<String>,
    backlogged_journals: Vec<String>,
    ingress_scanned: bool,
    failed: bool,
}
impl Store {
    /// Open or create a source store and acquire its exclusive writer lock.
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("sources"))?;
        fs::create_dir_all(root.join("checks"))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("writer.lock"))?;
        lock.try_lock()
            .context("another GP source service owns this store")?;
        let log_path = root.join("events.jsonl");
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&log_path)?;
        let mut store = Self {
            root: root.into(),
            log,
            _lock: lock,
            sequence: 0,
            seen: BTreeMap::new(),
            pending: BTreeMap::new(),
            ready: VecDeque::new(),
            offsets: vec![0],
            programs: BTreeSet::new(),
            metrics: LANGUAGES
                .into_iter()
                .map(|l| (l.into(), LanguageMetrics::default()))
                .collect(),
            emit_ns: 0,
            first_unix_ns: None,
            partial_journal_tails: Vec::new(),
            backlogged_journals: Vec::new(),
            ingress_scanned: false,
            failed: false,
        };
        let mut reader = BufReader::new(File::open(log_path)?);
        while let Some(bytes) = complete_line(&mut reader, false)? {
            let event: Event =
                serde_json::from_slice(&bytes).context("invalid GP event log; preserve it")?;
            store.apply(&event)?;
            store.offsets.push(reader.stream_position()?);
        }
        let pending = &store.pending;
        store.ready.retain(|key| pending.contains_key(key));
        Ok(store)
    }
    fn apply(&mut self, event: &Event) -> Result<()> {
        ensure!(
            event.schema == EVENT_SCHEMA && event.sequence == self.sequence + 1,
            "GP event schema/sequence mismatch"
        );
        ensure!(event.at_unix_ns > 0, "missing event time");
        match &event.payload {
            Payload::Source {
                occurrence,
                program_sha256,
                emit_ns,
                artifacts,
            } => {
                ensure!(
                    occurrence.schema == SCHEMA && !occurrence.id.is_empty(),
                    "invalid occurrence"
                );
                ensure!(
                    *program_sha256 == hash(occurrence.program.as_bytes()),
                    "program hash mismatch"
                );
                ensure!(
                    !self.seen.contains_key(&occurrence.id),
                    "duplicate occurrence in event log"
                );
                ensure!(
                    artifacts.len() == LANGUAGES.len(),
                    "incomplete source target set"
                );
                let languages: BTreeSet<_> =
                    artifacts.iter().map(|a| a.language.as_str()).collect();
                ensure!(
                    languages == LANGUAGES.into_iter().collect(),
                    "invalid source target set"
                );
                for artifact in artifacts {
                    let m = self
                        .metrics
                        .get_mut(&artifact.language)
                        .context("unknown language")?;
                    m.artifacts += 1;
                    match (&artifact.source, &artifact.sha256, &artifact.status) {
                        (Some(source), Some(digest), Status::Pending) => {
                            ensure!(
                                *digest == hash(source.as_bytes())
                                    && artifact.bytes == source.len() as u64
                                    && artifact.error.is_none(),
                                "source bytes/hash/status mismatch"
                            );
                            let count = count_tokens(source, language_kind(&artifact.language)?);
                            ensure!(
                                artifact.code_tokens == count.code_tokens as u64
                                    && artifact.doc_tokens == count.doc_tokens as u64,
                                "source token count mismatch"
                            );
                            m.emitted_tokens += artifact.tokens();
                            m.code_tokens += artifact.code_tokens;
                            m.doc_tokens += artifact.doc_tokens;
                            m.pending_tokens += artifact.tokens();
                            m.pending_artifacts += 1;
                            let mut descriptor = artifact.clone();
                            descriptor.source = None;
                            self.ready
                                .push_back((occurrence.id.clone(), artifact.language.clone()));
                            self.pending.insert(
                                (occurrence.id.clone(), artifact.language.clone()),
                                Pending {
                                    occurrence_id: occurrence.id.clone(),
                                    artifact: descriptor,
                                    inflight: false,
                                },
                            );
                        }
                        (None, None, Status::Failed) => {
                            ensure!(
                                artifact.tokens() == 0
                                    && artifact.bytes == 0
                                    && artifact.error.as_ref().is_some_and(|e| !e.is_empty()),
                                "invalid emission error"
                            );
                            m.emission_errors += 1;
                        }
                        _ => bail!("invalid initial artifact state"),
                    }
                }
                self.programs.insert(program_sha256.clone());
                self.emit_ns += emit_ns;
                self.seen.insert(
                    occurrence.id.clone(),
                    hash(&serde_json::to_vec(occurrence)?),
                );
            }
            Payload::Checked {
                occurrence_id,
                language,
                source_sha256,
                status,
                tool,
                ..
            } => {
                ensure!(
                    *status != Status::Pending && !tool.is_empty(),
                    "invalid compiler result"
                );
                let key = (occurrence_id.clone(), language.clone());
                let pending = self
                    .pending
                    .get(&key)
                    .context("compiler result without pending artifact")?;
                ensure!(
                    pending.artifact.sha256.as_ref() == Some(source_sha256),
                    "compiler checked wrong source"
                );
                let m = self.metrics.get_mut(language).context("unknown language")?;
                m.pending_tokens -= pending.artifact.tokens();
                m.pending_artifacts -= 1;
                if *status == Status::Passed {
                    m.passed_tokens += pending.artifact.tokens();
                    m.passed_artifacts += 1;
                } else {
                    m.lost_tokens += pending.artifact.tokens();
                    m.compile_errors += 1;
                }
                self.pending.remove(&key);
                if self.ready.front() == Some(&key) {
                    self.ready.pop_front();
                }
            }
        }
        self.first_unix_ns.get_or_insert(event.at_unix_ns);
        self.sequence = event.sequence;
        Ok(())
    }
    fn append(&mut self, payload: Payload) -> Result<()> {
        ensure!(
            !self.failed,
            "source store stopped after earlier publication failure"
        );
        let at_unix_ns = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
        let event = Event {
            schema: EVENT_SCHEMA.into(),
            sequence: self
                .sequence
                .checked_add(1)
                .context("event sequence overflow")?,
            at_unix_ns,
            payload,
        };
        let mut bytes = serde_json::to_vec(&event)?;
        bytes.push(b'\n');
        ensure!(
            bytes.len() <= MAX_LINE,
            "source event exceeds line limit; retain producer journal"
        );
        // Validate before publishing. Apply errors are fatal; never keep serving partial state.
        if let Err(error) = self.apply(&event) {
            self.failed = true;
            return Err(error);
        }
        if let Err(error) = self
            .log
            .write_all(&bytes)
            .and_then(|_| self.log.flush())
            .and_then(|_| self.log.sync_data())
        {
            self.failed = true;
            return Err(error.into());
        }
        self.offsets.push(self.log.metadata()?.len());
        Ok(())
    }
    /// Return the byte offset immediately after `sequence` in the event log.
    pub fn offset_after(&self, sequence: u64) -> Result<u64> {
        self.offsets
            .get(usize::try_from(sequence)?)
            .copied()
            .context("future sequence cursor")
    }
    /// Return the number of distinct source occurrences in the store.
    pub fn occurrence_count(&self) -> usize {
        self.seen.len()
    }
    /// Check whether this occurrence was published, rejecting reused IDs with new content.
    pub fn contains(&self, occurrence: &Occurrence) -> Result<bool> {
        match self.seen.get(&occurrence.id) {
            None => Ok(false),
            Some(digest) => {
                ensure!(
                    *digest == hash(&serde_json::to_vec(occurrence)?),
                    "occurrence identity reused with different content"
                );
                Ok(true)
            }
        }
    }
    /// Persist an occurrence and its emitted artifacts before they are checked.
    pub fn publish(
        &mut self,
        occurrence: Occurrence,
        artifacts: Vec<Artifact>,
        emit_ns: u64,
    ) -> Result<()> {
        ensure!(!self.contains(&occurrence)?, "occurrence already published");
        for artifact in &artifacts {
            if let (Some(source), Some(digest)) = (&artifact.source, &artifact.sha256) {
                ensure!(
                    valid_hash(digest) && *digest == hash(source.as_bytes()),
                    "invalid artifact digest"
                );
                let path = self
                    .root
                    .join("sources")
                    .join(format!("{digest}.{}", extension(&artifact.language)?));
                if path.exists() {
                    ensure!(
                        fs::read(&path)? == source.as_bytes(),
                        "cached source mismatch"
                    );
                } else {
                    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                    let staging = self
                        .root
                        .join("sources")
                        .join(format!(".{digest}-{stamp}.partial"));
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&staging)?;
                    file.write_all(source.as_bytes())?;
                    file.sync_all()?;
                    fs::rename(staging, &path)?;
                }
            }
        }
        let digest = hash(occurrence.program.as_bytes());
        self.append(Payload::Source {
            occurrence,
            program_sha256: digest,
            emit_ns,
            artifacts,
        })
    }
    /// Claim the next queued artifact for checking, returning its ID, metadata, and store root.
    pub fn next_check(&mut self) -> Option<(String, Artifact, PathBuf)> {
        while let Some(key) = self.ready.pop_front() {
            if let Some(next) = self.pending.get_mut(&key) {
                if !next.inflight {
                    next.inflight = true;
                    return Some((
                        next.occurrence_id.clone(),
                        next.artifact.clone(),
                        self.root.clone(),
                    ));
                }
            }
        }
        None
    }
    /// Persist a check result for an artifact previously returned by [`Store::next_check`].
    pub fn checked(&mut self, id: String, artifact: &Artifact, check: Check) -> Result<()> {
        self.append(Payload::Checked {
            occurrence_id: id,
            language: artifact.language.clone(),
            source_sha256: artifact.sha256.clone().context("missing source hash")?,
            status: check.status,
            tool: check.tool,
            diagnostic: check.diagnostic,
            elapsed_ns: check.elapsed_ns,
        })
    }
    /// Record the partial or backlogged producer journals observed at ingress.
    pub fn observed_ingress(
        &mut self,
        partial_journal_tails: Vec<String>,
        backlogged_journals: Vec<String>,
    ) {
        self.partial_journal_tails = partial_journal_tails;
        self.backlogged_journals = backlogged_journals;
        self.ingress_scanned = true;
    }
    /// Return a JSON snapshot of cumulative counts, rates, and pending-ingress state.
    pub fn snapshot(&self) -> serde_json::Value {
        let mut total = LanguageMetrics::default();
        for metrics in self.metrics.values() {
            total.add(metrics);
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let wall = self
            .first_unix_ns
            .map(|first| now.saturating_sub(first as u128) as f64 / 1e9)
            .unwrap_or(0.0)
            .max(1e-9);
        let stats = |m: &LanguageMetrics| {
            serde_json::json!({
                "counts":m, "token_yield":if m.emitted_tokens == 0 {None} else {Some(m.passed_tokens as f64 / m.emitted_tokens as f64)},
                "artifact_yield":if m.artifacts == 0 {None} else {Some(m.passed_artifacts as f64 / m.artifacts as f64)},
                "emitted_tokens_per_second":m.emitted_tokens as f64 / wall,
            })
        };
        serde_json::json!({"schema":"jope.gp-source-metrics/v1", "tokenizer":"feat-math lexical code tokens plus documentation words; not model BPE", "sequence":self.sequence, "occurrences":self.seen.len(), "unique_programs":self.programs.len(), "elapsed_seconds_including_downtime":wall, "store_failed":self.failed, "ingress":{"scanned":self.ingress_scanned,"partial_journal_tails":self.partial_journal_tails, "backlogged_journals":self.backlogged_journals}, "observed_backlog_clear":self.ingress_scanned && self.partial_journal_tails.is_empty() && self.backlogged_journals.is_empty() && self.pending.is_empty() && total.emission_errors == 0 && total.compile_errors == 0, "cumulative_emit_ns":self.emit_ns, "assembly_tokens_per_second":if self.emit_ns == 0 {None} else {Some(total.emitted_tokens as f64 * 1e9 / self.emit_ns as f64)}, "total":stats(&total), "languages":self.metrics.iter().map(|(l,m)| (l.clone(),stats(m))).collect::<BTreeMap<_,_>>()})
    }
}
