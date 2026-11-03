use super::*;
use crate::progress::inventory::Head;

impl<'r> Driver<'r> {
    pub fn init(
        state: &State,
        remote: &'r dyn Remote,
        config: Config,
        manifest: RunManifest,
        publisher: &str,
        now_ms: u64,
    ) -> Result<Driver<'r>> {
        config.check()?;
        ensure!(
            manifest.publisher == publisher,
            "{publisher} is not the publisher run {} names ({}); nothing was written",
            manifest.run_id,
            manifest.publisher
        );
        let manifest = Decoded::new(manifest)?;
        let run_id = manifest.value().run_id.clone();
        let owed_before = {
            let mut outbox = Outbox::open(state, config.limits)?;
            let others: Vec<String> = run_objects(&outbox, &run_id)?
                .into_iter()
                .filter(|o| o.kind == Kind::RunManifest && o.digest != manifest.digest())
                .map(|o| o.digest)
                .collect();
            ensure!(
                others.is_empty(),
                "run {run_id} already has manifest {} on this host; a run's planned census is \
                 frozen once. Nothing was written",
                others.join(", ")
            );
            let owed = outbox.pending_entries() > 0;
            outbox.append_object(&manifest.encoded()?)?;
            owed
        };
        // The manifest just appended is new work, not a debt from before a restart.
        Driver::open_with(
            state,
            remote,
            config,
            &run_id,
            publisher,
            now_ms,
            owed_before,
        )
    }

    /// Open the publisher of a run whose manifest this host recorded, re-deriving its state from
    /// the outbox and the remote's committed closure. Whatever it finds owed is a restart and is
    /// flushed at once.
    pub fn open(
        state: &State,
        remote: &'r dyn Remote,
        config: Config,
        run_id: &str,
        publisher: &str,
        now_ms: u64,
    ) -> Result<Driver<'r>> {
        Driver::open_with(state, remote, config, run_id, publisher, now_ms, true)
    }

    fn open_with(
        state: &State,
        remote: &'r dyn Remote,
        config: Config,
        run_id: &str,
        publisher: &str,
        now_ms: u64,
        restart: bool,
    ) -> Result<Driver<'r>> {
        config.check()?;
        let outbox = Outbox::open(state, config.limits)?;
        let local = run_objects(&outbox, run_id)?;
        let manifests: Vec<&Encoded> = local
            .iter()
            .filter(|o| o.kind == Kind::RunManifest)
            .collect();
        let manifest = match manifests.as_slice() {
            [one] => decode_as::<RunManifest>(&one.bytes, &one.digest)?,
            [] => bail!("run {run_id} has no manifest on this host; `progress init` records it"),
            many => bail!(
                "run {run_id} has {} manifests on this host; refusing to guess which is the run",
                many.len()
            ),
        };
        ensure!(
            manifest.value().publisher == publisher,
            "{publisher} is not the publisher run {run_id} names ({}); nothing was written",
            manifest.value().publisher
        );
        let mut known: BTreeMap<ObjectRef, Encoded> =
            local.into_iter().map(|o| (o.reference(), o)).collect();
        let (head, committed) = match reusable(remote, run_id)? {
            Some(state) => {
                for (r, bytes) in &state.graph.objects {
                    if !known.contains_key(r) {
                        known.insert(r.clone(), Encoded::from_bytes(r.kind, &r.digest, bytes)?);
                    }
                }
                (Some(state.head), state.graph.objects.into_keys().collect())
            }
            None => (None, BTreeSet::new()),
        };
        let staged: BTreeSet<ObjectRef> = known
            .keys()
            .filter(|r| !matches!(r.kind, Kind::RunManifest | Kind::Commit | Kind::Summary))
            .filter(|r| !committed.contains(*r))
            .cloned()
            .collect();
        let mut driver = Driver {
            cadence: if restart {
                Cadence::start(config.cadence, now_ms, &outbox)
            } else {
                Cadence::new(config.cadence, now_ms)
            },
            checkpoints: Checkpoints::new(config.cadence, now_ms),
            outbox,
            remote,
            config,
            manifest,
            publisher: publisher.into(),
            head,
            committed,
            staged,
            previous: None,
            observed_step: None,
            checkpoint_pending: false,
        };
        driver.previous = driver.last_checkpoint(known.values())?;
        driver.observed_step = driver.previous.as_ref().map(|p| p.step);
        driver.checkpoint_pending = driver.previous.as_ref().is_some_and(|p| {
            driver.staged.contains(&ObjectRef {
                kind: Kind::Checkpoint,
                digest: p.checkpoint.digest().to_string(),
            })
        });
        // Appended but uncommitted progress may be owed since before a crash, even when every
        // entry was delivered: the restart flushes (commits) it at once.
        if restart && !driver.staged.is_empty() {
            driver.cadence.request(Urgent::Restart);
        }
        Ok(driver)
    }

    /// The run's last W8 checkpoint (committed or appended) and the engine step it was taken at,
    /// read back from its own `state.json` blob (locally, or from the remote).
    fn last_checkpoint<'a>(
        &self,
        objects: impl Iterator<Item = &'a Encoded>,
    ) -> Result<Option<Previous>> {
        let mut best: Option<Decoded<Checkpoint>> = None;
        for o in objects.filter(|o| o.kind == Kind::Checkpoint) {
            let k = decode_as::<Checkpoint>(&o.bytes, &o.digest)?;
            if best
                .as_ref()
                .is_none_or(|b| k.value().update_index > b.value().update_index)
            {
                best = Some(k);
            }
        }
        let Some(checkpoint) = best else {
            return Ok(None);
        };
        for blob in &checkpoint.value().state {
            let bytes = match self.outbox.state().read_blob(&blob.sha256)? {
                Some(bytes) => bytes,
                None => self
                    .remote
                    .get(&blob_key(&blob.sha256))?
                    .filter(|b| BlobRef::of(b) == *blob)
                    .with_context(|| {
                        format!("checkpoint state blob {} is not available", blob.sha256)
                    })?,
            };
            if let Ok(step) = engine::checkpoint_step(&bytes) {
                return Ok(Some(Previous { checkpoint, step }));
            }
        }
        bail!(
            "W8 checkpoint {} holds no engine state.json; its step cannot be recovered",
            checkpoint.digest()
        )
    }

    /// Performs the manifest operation.
    pub fn manifest(&self) -> &Decoded<RunManifest> {
        &self.manifest
    }

    /// Performs the outbox operation.
    pub fn outbox(&self) -> &Outbox {
        &self.outbox
    }

    /// Performs the config operation.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Performs the head operation.
    pub fn head(&self) -> Option<&Head> {
        self.head.as_ref()
    }

    /// Performs the staged operation.
    pub fn staged(&self) -> u64 {
        self.staged.len() as u64
    }

    /// Performs the previous operation.
    pub fn previous(&self) -> Option<&Previous> {
        self.previous.as_ref()
    }

    /// Performs the uncheckpointed operation.
    pub fn uncheckpointed(&self) -> u64 {
        self.checkpoints.uncheckpointed()
    }

    pub(super) fn finish(&self, mut step: Step) -> Step {
        step.staged = self.staged.len() as u64;
        step.head = self.head.clone();
        step
    }
}
