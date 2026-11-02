use super::*;

/// Byte-exact binding of a journal prefix, recorded in checkpoint receipts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub schema: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Append-only writer; every event is validated before it is written.
pub struct Journal {
    file: fs::File,
    path: PathBuf,
    digest: Sha256,
    bytes: u64,
    tracker: Tracker,
    poisoned: bool,
}

impl Journal {
    /// A partition-v1 journal.
    pub fn create(path: &Path, header: Header) -> Result<Self> {
        Self::create_with_shard(path, header, None)
    }

    /// A journal for either partition; `shard` is the rank's v2 shard.
    pub fn create_with_shard(path: &Path, header: Header, shard: Option<Shard>) -> Result<Self> {
        header.validate(shard.as_ref())?;
        let file = fs::OpenOptions::new()
            .append(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("consumption journal exists: {}", path.display()))?;
        let mut journal = Self {
            file,
            path: path.to_path_buf(),
            digest: Sha256::new(),
            bytes: 0,
            tracker: Tracker::default(),
            poisoned: false,
        };
        journal.append(&TrainingConsumptionEvent::Attempt { header, shard })?;
        journal.sync()?;
        if let Some(parent) = path.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(journal)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn append(&mut self, event: &TrainingConsumptionEvent) -> Result<()> {
        ensure!(
            !self.poisoned,
            "consumption journal failed an earlier write"
        );
        let mut next = self.tracker.clone();
        next.accept(event)?;
        let mut line = serde_json::to_vec(event)?;
        line.push(b'\n');
        if let Err(error) = self.file.write_all(&line) {
            // A torn line may exist; never append after it.
            self.poisoned = true;
            return Err(error.into());
        }
        self.digest.update(&line);
        self.bytes += line.len() as u64;
        self.tracker = next;
        Ok(())
    }

    pub fn microbatch(
        &mut self,
        step: u64,
        microbatch: usize,
        serial: u64,
        dropout_seed: i64,
        records: &[usize],
        supervision: Supervision,
    ) -> Result<()> {
        self.append(&TrainingConsumptionEvent::Microbatch {
            step,
            microbatch,
            serial,
            dropout_seed,
            records: records.iter().map(|&r| r as u64).collect(),
            supervision,
        })
    }

    pub fn applied(&mut self, step: u64, gradient_norm: f64) -> Result<()> {
        self.append(&TrainingConsumptionEvent::Update {
            step,
            outcome: TrainingUpdateOutcome::Applied,
            microbatches: self.tracker.step.units.microbatches as usize,
            gradient_norm: Some(gradient_norm),
            clipped: Some(gradient_norm > 1.0),
            reason: None,
        })
    }

    /// Record that the attempted update of `step` was not applied. The
    /// attempt ends; the journal refuses every later event.
    pub fn rejected(&mut self, step: u64, reason: &str) -> Result<()> {
        self.append(&TrainingConsumptionEvent::Update {
            step,
            outcome: TrainingUpdateOutcome::Rejected,
            microbatches: self.tracker.step.units.microbatches as usize,
            gradient_norm: None,
            clipped: None,
            reason: Some(if reason.is_empty() {
                "unspecified".into()
            } else {
                reason.into()
            }),
        })
    }

    /// Make every written event durable; return the binding of the prefix.
    pub fn sync(&mut self) -> Result<Binding> {
        ensure!(
            !self.poisoned,
            "consumption journal failed an earlier write"
        );
        self.file.sync_all()?;
        Ok(Binding {
            schema: SCHEMA.into(),
            bytes: self.bytes,
            sha256: hex::encode(self.digest.clone().finalize()),
        })
    }

    /// Record, durably, that a checkpoint covers every applied update through `step`.
    pub fn commit(&mut self, step: u64, state_sha256: &str) -> Result<Binding> {
        self.append(&TrainingConsumptionEvent::Commit {
            step,
            state_sha256: state_sha256.into(),
        })?;
        self.sync()
    }

    pub fn applied_through(&self) -> u64 {
        self.tracker.applied_through
    }
}
