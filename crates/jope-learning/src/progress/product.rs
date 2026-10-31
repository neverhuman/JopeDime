//! W8 product integration: the publisher that turns the engine's durable events into committed
//! W8 progress.
//!
//! [`Driver`] is the product path that appends to the durable outbox, commits through the
//! conditional publisher and drives the cadence. For one run it:
//!
//! 1. maps each engine event ([`crate::progress::engine`]) to verified W8 objects and appends
//!    them, blobs first, to the outbox. Nothing reaches the outbox that did not verify;
//! 2. commits what it appended on top of the run's committed tip through
//!    [`crate::progress::publish::publish`], which verifies the remote closure, the graph rules
//!    (one consecutive chain of update indexes included) and every HWM decision against this
//!    host's HWM ledger before the pointer moves;
//! 3. drives the cadence ([`crate::progress::cadence`]): a flush (commit and delivery) at once
//!    after a corpus freeze, a checkpoint commit, an HWM change or a shutdown, and otherwise at
//!    the size or interval trigger; and a W8 checkpoint only when the checkpoint cadence says one
//!    is due at the engine's committed optimizer steps, or at controlled shutdown.
//!
//! **Configuration.** The outbox limits and the cadence are explicit [`Config`] values. Their
//! defaults ([`Limits::ADOPTED`], [`Policy::ADOPTED`]) were adopted by owner delegation on
//! 2026-09-26, and every report says so ([`APPROVAL`]). They stay configuration: a
//! `jope.progress.config/v1` file can set other values, and a report then says the values in
//! effect are not the adopted ones ([`OVERRIDDEN`]).
//!
//! **Store.** The remote is whatever [`Remote`] the caller opened: the CLI opens only the store
//! named explicitly by `--store`/`JOPE_LEARNING_URL`, and the tests use `file://` and
//! `memory:///` stores. Nothing here reaches for a default bucket. An `s3://bucket/prefix` store
//! needs no code change; see `jope-learning progress --help` for the environment it reads.
//!
//! **Restart.** Everything the driver needs is re-derived on open, from the outbox and the
//! remote's committed closure: the run manifest, what is committed, what is appended but not yet
//! committed (it is committed at the next flush), and the last W8 checkpoint with the engine step
//! it was taken at. A restarted driver therefore neither loses appended progress nor maps an
//! engine checkpoint twice; re-mapping the same bytes yields the same objects, which the outbox
//! reports as existing.

use crate::canon::sha256_hex;
use crate::hwm::{parse_event, Transition};
use crate::ledger::Ledger;
use crate::progress::cadence::{
    Cadence, CheckpointDue, Checkpoints, Clock, FlushDue, Policy, Tick, Urgent,
};
use crate::progress::engine::{self, EngineCorpus, Mapped, Previous};
use crate::progress::inventory::{
    blob_key, decode_as, encode, BlobRef, Checkpoint, Commit, Decoded, Dependency, Encoded, Event,
    EventBody, Head, HwmDecision, Kind, ObjectRef, RunManifest, HEAD_SCHEMA, MAX_REFS,
};
use crate::progress::outbox::{Appended, Limits, Outbox, Refused};
use crate::progress::publish::{publish, reusable, Publication};
use crate::progress::remote::Remote;
use crate::state::State;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Schema identifier for config values.
pub const CONFIG_SCHEMA: &str = "jope.progress.config/v1";

/// The approval state of the default [`Config`] values: the outbox limits (W8-T02) and the
/// cadence (W8-T05) were adopted by owner delegation on 2026-09-26 (owner decision D6). This is
/// reported beside them; it is not a setting, and no configuration file can change it.
pub const APPROVAL: &str = "adopted-by-owner-delegation-2026-09-26";

/// The approval state reported when a configuration file sets values other than the adopted
/// defaults: they are the operator's, not the owner's.
pub const OVERRIDDEN: &str = "operator-configured-not-the-adopted-defaults";

/// The publisher's explicit configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Outbox capacity: pause new work, then refuse everything (W8-T02).
    pub limits: Limits,
    /// Flush and checkpoint cadence (W8-T05).
    pub cadence: Policy,
}

impl Config {
    /// The documented defaults, adopted by owner delegation on 2026-09-26 ([`APPROVAL`]):
    /// - outbox: pause new work at 7,500 entries or 768 MiB pending; refuse every append at
    ///   10,000 entries or 1 GiB;
    /// - cadence: flush every 60 s or at 1 MiB pending; checkpoint every 100 committed updates
    ///   or 5 minutes, and at controlled shutdown.
    pub fn adopted() -> Config {
        Config {
            schema: CONFIG_SCHEMA.into(),
            limits: Limits::ADOPTED,
            cadence: Policy::ADOPTED,
        }
    }

    /// Performs the check operation and returns an error if it cannot be completed.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.schema == CONFIG_SCHEMA,
            "config schema {:?} is not {CONFIG_SCHEMA}",
            self.schema
        );
        self.limits.check()?;
        let p = &self.cadence;
        ensure!(
            p.flush_interval_ms > 0
                && p.flush_bytes > 0
                && p.checkpoint_updates > 0
                && p.checkpoint_interval_ms > 0,
            "every cadence value must be positive"
        );
        Ok(())
    }

    /// Read an explicit configuration file. Unknown and missing fields are refused: there is no
    /// partial override that silently keeps some defaults.
    pub fn load(path: &Path) -> Result<Config> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let config: Config = serde_json::from_slice(&bytes)
            .with_context(|| format!("{} is not a {CONFIG_SCHEMA} file", path.display()))?;
        config.check()?;
        Ok(config)
    }

    /// The configuration as reported: its values, whether they are the adopted defaults, and
    /// their approval state.
    pub fn report(&self) -> ConfigReport {
        let adopted_defaults = *self == Config::adopted();
        ConfigReport {
            values: self.clone(),
            adopted_defaults,
            approval: if adopted_defaults {
                APPROVAL
            } else {
                OVERRIDDEN
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
/// `ConfigReport` is a record for the config report contract in this module: W8 product integration: the publisher that turns the engine's durable events into committed W8 progress.
pub struct ConfigReport {
    /// Values recorded for this config report.
    pub values: Config,
    /// Adopted defaults recorded for this config report.
    pub adopted_defaults: bool,
    /// Approval recorded for this config report.
    pub approval: &'static str,
}

/// The wall clock, in milliseconds since the Unix epoch.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

/// What the checkpoint cadence decided at an observed engine checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "decision")]
pub enum CheckpointStep {
    /// No checkpoint is published yet, or none newer than the last one counted.
    Nothing,
    /// Not due: the engine's newer optimizer steps are counted and wait for the next W8
    /// checkpoint.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Deferred {
        /// Step recorded for this outcome.
        step: u64,
        /// Uncheckpointed recorded for this outcome.
        uncheckpointed: u64,
    },
    /// Mapped as the next W8 update.
    Mapped {
        /// Step recorded for this checkpoint step.
        step: u64,
        /// Update index recorded for this checkpoint step.
        update_index: u64,
        /// Number of consumed rows represented here.
        consumed_rows: u64,
        /// Due recorded for this checkpoint step.
        due: CheckpointDue,
    },
}

/// What one driver operation did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Step {
    /// Outbox keys this operation appended.
    pub appended: Vec<String>,
    /// Keys it offered that the outbox already held (a replay after a crash, or the same bytes).
    pub existing: Vec<String>,
    /// Optional checkpoint value; `None` means it is not available.
    pub checkpoint: Option<CheckpointStep>,
    /// Why a flush was due, if one was.
    pub due: Option<FlushDue>,
    /// Optional publication value; `None` means it is not available.
    pub publication: Option<Publication>,
    /// Optional tick value; `None` means it is not available.
    pub tick: Option<Tick>,
    /// A flush was due and left nothing staged and nothing pending: everything appended so far
    /// is committed and acknowledged.
    pub flushed: bool,
    /// Objects appended but not yet committed after this operation.
    pub staged: u64,
    /// Optional head value; `None` means it is not available.
    pub head: Option<Head>,
}

/// The W8 publisher for one run.
pub struct Driver<'r> {
    outbox: Outbox,
    remote: &'r dyn Remote,
    config: Config,
    manifest: Decoded<RunManifest>,
    publisher: String,
    cadence: Cadence,
    checkpoints: Checkpoints,
    head: Option<Head>,
    committed: BTreeSet<ObjectRef>,
    staged: BTreeSet<ObjectRef>,
    previous: Option<Previous>,
    /// The newest engine step whose optimizer updates the checkpoint cadence has counted.
    observed_step: Option<u64>,
    /// A mapped W8 checkpoint waits for its commit before the checkpoint cadence restarts.
    checkpoint_pending: bool,
}

/// Every object this outbox holds for `run_id`, in sequence order.
fn run_objects(outbox: &Outbox, run_id: &str) -> Result<Vec<Encoded>> {
    let mut out = Vec::new();
    for seq in 1..=outbox.records() {
        if let Some(object) = outbox.object(seq)? {
            if object.run_id == run_id {
                out.push(object);
            }
        }
    }
    Ok(out)
}

impl<'r> Driver<'r> {
    /// Record a new run's manifest (its complete planned census, before any work) and open its
    /// publisher. Refuses a publisher the manifest does not name, and a run this host already
    /// holds another manifest for.
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

    /// Performs the manifest operation and returns an error if it cannot be completed.
    pub fn manifest(&self) -> &Decoded<RunManifest> {
        &self.manifest
    }

    /// Performs the outbox operation and returns an error if it cannot be completed.
    pub fn outbox(&self) -> &Outbox {
        &self.outbox
    }

    /// Performs the config operation and returns an error if it cannot be completed.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Performs the head operation and returns an error if it cannot be completed.
    pub fn head(&self) -> Option<&Head> {
        self.head.as_ref()
    }

    /// Performs the staged operation and returns an error if it cannot be completed.
    pub fn staged(&self) -> u64 {
        self.staged.len() as u64
    }

    /// Performs the previous operation and returns an error if it cannot be completed.
    pub fn previous(&self) -> Option<&Previous> {
        self.previous.as_ref()
    }

    /// Performs the uncheckpointed operation and returns an error if it cannot be completed.
    pub fn uncheckpointed(&self) -> u64 {
        self.checkpoints.uncheckpointed()
    }

    fn finish(&self, mut step: Step) -> Step {
        step.staged = self.staged.len() as u64;
        step.head = self.head.clone();
        step
    }

    /// Append a mapping: every blob, then every object, staging the objects for the next commit.
    /// A mapping is admitted whole or not at all: if its new items would pass the outbox maximum,
    /// it is refused ([`Refused::Full`]) before any of them is written.
    fn append(&mut self, mapped: &Mapped, step: &mut Step) -> Result<()> {
        // W12-T05: revoked evidence is never appended, let alone published.
        let revocations = crate::revocation::Revocations::load(self.outbox.state())?;
        if !revocations.is_empty() {
            let mut digests: Vec<String> = mapped.blobs.iter().map(|b| sha256_hex(b)).collect();
            for o in &mapped.objects {
                digests.push(o.digest.clone());
                digests.push(sha256_hex(&o.bytes));
            }
            revocations.refuse("an engine publication", digests.iter().map(String::as_str))?;
        }
        let (mut entries, mut bytes) = (0u64, 0u64);
        for blob in &mapped.blobs {
            if self
                .outbox
                .seq_of(&Dependency::Blob(BlobRef::of(blob)))
                .is_none()
            {
                entries += 1;
                bytes += blob.len() as u64;
            }
        }
        for object in &mapped.objects {
            if self
                .outbox
                .seq_of(&Dependency::Object(object.reference()))
                .is_none()
            {
                entries += 1;
                bytes += object.bytes.len() as u64;
            }
        }
        let limits = self.outbox.limits();
        let (pending_entries, pending_bytes) =
            (self.outbox.pending_entries(), self.outbox.pending_bytes());
        if pending_entries + entries > limits.max_entries
            || pending_bytes + bytes > limits.max_bytes
        {
            return Err(Refused::Full {
                pending_entries,
                pending_bytes,
                item_bytes: bytes,
            }
            .into());
        }
        for blob in &mapped.blobs {
            let key = Dependency::Blob(BlobRef::of(blob)).key();
            match self.outbox.append_blob(blob)? {
                Appended::Added { .. } => step.appended.push(key),
                Appended::Existing { .. } => step.existing.push(key),
            }
        }
        for object in &mapped.objects {
            match self.outbox.append_object(object)? {
                Appended::Added { .. } => step.appended.push(object.key()),
                Appended::Existing { .. } => step.existing.push(object.key()),
            }
            let r = object.reference();
            if !self.committed.contains(&r) && object.kind != Kind::RunManifest {
                self.staged.insert(r);
            }
        }
        Ok(())
    }

    /// Commit every staged object on top of the committed tip. `None` when nothing is staged.
    /// Only an advanced (or already published) pointer moves the staged objects to committed;
    /// a conflict or a block leaves them staged, and the publication says why.
    pub fn commit(&mut self) -> Result<Option<Publication>> {
        if self.staged.is_empty() {
            return Ok(None);
        }
        let objects: Vec<ObjectRef> = self.staged.iter().cloned().collect();
        ensure!(
            objects.len() <= MAX_REFS,
            "{} objects are staged, more than one commit may add ({MAX_REFS}); flush more often",
            objects.len()
        );
        let (parent, height) = match &self.head {
            Some(h) => (
                Some(h.commit.clone()),
                h.height.checked_add(1).context("height overflow")?,
            ),
            None => (None, 0),
        };
        if let Some(parent) = &parent {
            let recorded = self.outbox.seq_of(&Dependency::Object(ObjectRef {
                kind: Kind::Commit,
                digest: parent.clone(),
            }));
            ensure!(
                recorded.is_some(),
                "run {} is committed at {parent}, which this host never recorded: another \
                 publisher advanced the run. `jope-learning progress reconcile` adopts it and \
                 commits this host's work on top; nothing was written",
                self.manifest.value().run_id
            );
        }
        let commit = encode(&Commit {
            schema: Kind::Commit.schema().into(),
            run_id: self.manifest.value().run_id.clone(),
            run_manifest: self.manifest.digest().into(),
            parent,
            height,
            objects,
        })?;
        let publication = publish(
            &mut self.outbox,
            self.remote,
            &self.publisher,
            &self.manifest,
            &commit,
        )?;
        if matches!(
            publication,
            Publication::Advanced { .. } | Publication::AlreadyPublished { .. }
        ) {
            self.head = Some(Head {
                schema: HEAD_SCHEMA.into(),
                run_id: self.manifest.value().run_id.clone(),
                commit: commit.digest.clone(),
                height,
            });
            self.committed.extend(std::mem::take(&mut self.staged));
            self.committed.insert(commit.reference());
        }
        Ok(Some(publication))
    }

    /// Flush if the cadence says one is due. Staged objects count as pending: a flush commits
    /// them, which delivers everything they depend on first, and then delivers anything else
    /// still pending. A failed or refused commit leaves everything owed and the urgency set.
    pub fn tick(&mut self, clock: &dyn Clock) -> Result<Step> {
        let now = clock.now_ms();
        let mut step = Step::default();
        let pending = self.outbox.pending_entries() + self.staged.len() as u64;
        step.due = self
            .cadence
            .due_for(now, pending, self.outbox.pending_bytes());
        if step.due.is_some() && !self.staged.is_empty() {
            step.publication = self.commit()?;
            if self.staged.is_empty() && self.checkpoint_pending {
                self.checkpoints.checkpointed(now);
                self.checkpoint_pending = false;
            }
        }
        // Delivery (and the cadence's own bookkeeping) only once nothing is left uncommitted: a
        // staged object whose commit failed or was refused keeps the flush owed, and its
        // urgency, until a commit takes it. Everything pending belongs to a staged object or to
        // the manifest, which the next commit delivers first.
        if self.staged.is_empty() {
            step.tick = Some(self.cadence.tick(clock, &mut self.outbox, self.remote)?);
        }
        step.flushed =
            step.due.is_some() && self.staged.is_empty() && self.outbox.pending_entries() == 0;
        Ok(self.finish(step))
    }

    /// Request an urgent flush for `why` and flush now.
    fn urgent(&mut self, why: Urgent, clock: &dyn Clock, step: Step) -> Result<Step> {
        self.cadence.request(why);
        let flushed = self.tick(clock)?;
        Ok(self.finish(Step {
            due: flushed.due,
            publication: flushed.publication,
            tick: flushed.tick,
            flushed: flushed.flushed,
            ..step
        }))
    }

    /// A published engine corpus: its payload and manifest are appended, and the corpus freeze
    /// flushes immediately.
    pub fn corpus(&mut self, dir: &Path, clock: &dyn Clock) -> Result<Step> {
        let mapped = self.map_corpus(dir)?;
        let mut step = Step::default();
        self.append(&mapped.mapped, &mut step)?;
        self.urgent(Urgent::CorpusFreeze, clock, step)
    }

    fn map_corpus(&self, dir: &Path) -> Result<EngineCorpus> {
        engine::corpus(
            &self.manifest.value().run_id,
            dir,
            None,
            self.config.limits.max_bytes,
        )
    }

    /// The engine run `run` (relative to `engine_root`), training on the corpus at `corpus_dir`,
    /// has published a checkpoint. Its new optimizer steps are counted by the checkpoint cadence;
    /// when a checkpoint is due, the published checkpoint becomes the next consecutive W8 update
    /// and flushes immediately. Otherwise the steps wait for the next due checkpoint.
    /// `start_step` is the optimizer step the run started from (0 for a fresh run).
    pub fn checkpoint(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.observe(engine_root, run, corpus_dir, start_step, clock, false)
    }

    /// Controlled shutdown: if any counted optimizer step is not yet in a W8 checkpoint, the
    /// published engine checkpoint becomes one; then everything flushes.
    pub fn shutdown(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.observe(engine_root, run, corpus_dir, start_step, clock, true)
    }

    fn observe(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
        shutdown: bool,
    ) -> Result<Step> {
        let now = clock.now_ms();
        let mut step = Step::default();
        let published = engine::published(engine_root, run)?;
        let counted = self.observed_step.unwrap_or(start_step);
        // Counted on a copy: the new steps are counted only once this observation succeeds, so
        // a checkpoint that fails to map is counted, and due, again when it is fixed.
        let mut counter = self.checkpoints.clone();
        let mut due = None;
        let mut observed = self.observed_step;
        if let Some(p) = &published {
            if p.step > counted {
                due = counter.committed_updates(p.step - counted, now);
                observed = Some(p.step);
            }
        }
        if shutdown {
            due = due.or_else(|| counter.shutdown());
        }
        let mapped_through = self.previous.as_ref().map_or(start_step, |p| p.step);
        let decided = match (published, due) {
            (Some(p), Some(due)) if p.step > mapped_through => {
                let corpus = self.map_corpus(corpus_dir)?;
                let mapped = engine::checkpoint(
                    engine_root,
                    run,
                    &corpus,
                    self.previous.as_ref(),
                    start_step,
                    self.config.limits.max_bytes,
                )?;
                // The corpus is a dependency of the exposure: appended (or found) first.
                self.append(&corpus.mapped, &mut step)?;
                self.append(&mapped.mapped, &mut step)?;
                let decided = CheckpointStep::Mapped {
                    step: mapped.step,
                    update_index: mapped.checkpoint.value().update_index,
                    consumed_rows: mapped.exposure.value().consumed_rows,
                    due,
                };
                self.previous = Some(Previous {
                    checkpoint: mapped.checkpoint,
                    step: mapped.step,
                });
                self.checkpoint_pending = true;
                self.cadence.request(Urgent::CheckpointCommit);
                decided
            }
            (Some(p), None) if p.step > mapped_through => CheckpointStep::Deferred {
                step: p.step,
                uncheckpointed: counter.uncheckpointed(),
            },
            _ => CheckpointStep::Nothing,
        };
        self.checkpoints = counter;
        self.observed_step = observed;
        let mapped = matches!(decided, CheckpointStep::Mapped { .. });
        step.checkpoint = Some(decided);
        if shutdown {
            self.cadence.request(Urgent::Shutdown);
        }
        if mapped || shutdown {
            let flushed = self.tick(clock)?;
            step.due = flushed.due;
            step.publication = flushed.publication;
            step.tick = flushed.tick;
            step.flushed = flushed.flushed;
        }
        Ok(self.finish(step))
    }

    fn planned(&self, work_id: &str) -> Result<()> {
        ensure!(
            self.manifest
                .value()
                .units
                .iter()
                .any(|u| u.work_id == work_id),
            "work {work_id} is not planned in run {}; nothing was written",
            self.manifest.value().run_id
        );
        Ok(())
    }

    fn event(&mut self, event: Decoded<Event>, blobs: Vec<Vec<u8>>) -> Result<Step> {
        let mut step = Step::default();
        self.append(
            &Mapped {
                blobs,
                objects: vec![event.encoded()?],
            },
            &mut step,
        )?;
        Ok(self.finish(step))
    }

    /// Attempt 1 of a planned unit was scheduled. It opens new work, so a paused outbox refuses
    /// it (and writes nothing).
    pub fn schedule(&mut self, work_id: &str) -> Result<Step> {
        self.planned(work_id)?;
        let e = Decoded::new(Event::new(&self.manifest, work_id, 1, EventBody::Scheduled))?;
        self.event(e, Vec::new())
    }

    /// Attempt `attempt` of a planned unit was scheduled as a retry of the recorded failure of
    /// the attempt before it.
    pub fn retry(&mut self, work_id: &str, attempt: u32) -> Result<Step> {
        self.planned(work_id)?;
        ensure!(attempt >= 2, "a retry opens attempt 2 or later");
        let failure = self
            .find_event(work_id, attempt - 1, |b| {
                matches!(b, EventBody::Failed { .. })
            })?
            .with_context(|| {
                format!(
                    "attempt {} of {work_id} has no recorded failure to retry",
                    attempt - 1
                )
            })?;
        let e = Decoded::new(Event::new(
            &self.manifest,
            work_id,
            attempt,
            EventBody::Retry {
                failed_event: failure,
            },
        ))?;
        self.event(e, Vec::new())
    }

    /// The digest of the recorded event of `(work_id, attempt)` whose body `matches`.
    fn find_event(
        &self,
        work_id: &str,
        attempt: u32,
        matches: impl Fn(&EventBody) -> bool,
    ) -> Result<Option<String>> {
        for o in run_objects(&self.outbox, &self.manifest.value().run_id)? {
            if o.kind != Kind::Event {
                continue;
            }
            let e = decode_as::<Event>(&o.bytes, &o.digest)?;
            let v = e.value();
            if v.work_id == work_id && v.attempt == attempt && matches(&v.body) {
                return Ok(Some(o.digest));
            }
        }
        Ok(None)
    }

    /// A checkpoint recovery reaped the ranks of `(work_id, attempt)`: its retryable failure is
    /// recorded. The receipt must be complete; its sha256 is in the failure detail.
    pub fn recovery(&mut self, work_id: &str, attempt: u32, receipt: &Path) -> Result<Step> {
        let (_, event) = engine::recovery(&self.manifest, work_id, attempt, receipt)?;
        self.event(event, Vec::new())
    }

    /// A no-replace engine publication is the verified result of `(work_id, attempt)`.
    pub fn result(
        &mut self,
        work_id: &str,
        attempt: u32,
        engine_root: &Path,
        artifact: &Path,
        sha256: &str,
    ) -> Result<Step> {
        let (bytes, event) = engine::result(
            &self.manifest,
            work_id,
            attempt,
            engine_root,
            artifact,
            sha256,
            self.config.limits.max_bytes,
        )?;
        self.event(event, vec![bytes])
    }

    /// Link the completed `(work_id, attempt)` to the decision this host's HWM ledger recorded
    /// at `ledger_path` (`yyyy/mm/<batch>.jsonl`, relative to the ledger) event `seq`. The link
    /// is built from the ledger itself and checked against it again before anything is
    /// appended. An initialization or improvement flushes immediately.
    pub fn decision(
        &mut self,
        work_id: &str,
        attempt: u32,
        ledger_path: &str,
        seq: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.planned(work_id)?;
        let completion = self
            .find_event(work_id, attempt, |b| {
                matches!(b, EventBody::Completed { .. })
            })?
            .with_context(|| {
                format!("attempt {attempt} of {work_id} has no recorded completion")
            })?;
        let state = State::at(self.outbox.state().root());
        let ledger = Ledger::open(state.root(), "hwm");
        let path = ledger
            .batches()?
            .into_iter()
            .find(|(rel, _)| rel == ledger_path)
            .map(|(_, path)| path)
            .with_context(|| format!("{ledger_path} is not a batch of this host's HWM ledger"))?;
        let (events, seal, file_sha, _) = Ledger::read_batch(&path)?;
        let index = usize::try_from(seq)?;
        let event = parse_event(
            index,
            events
                .get(index)
                .with_context(|| format!("{ledger_path} has no event {seq}"))?,
        )?;
        let transition = event
            .transition
            .with_context(|| format!("{ledger_path}#{seq} records no transition"))?;
        let record_sha = &event.record.provenance.record_sha256;
        let blob = state
            .read_blob(record_sha)?
            .with_context(|| format!("record blob {record_sha} is missing from this host"))?;
        let decision = HwmDecision {
            schema: Kind::HwmDecision.schema().into(),
            run_id: self.manifest.value().run_id.clone(),
            work_id: work_id.into(),
            attempt,
            completion,
            record: BlobRef::of(&blob),
            identity_sha256: event.record.identity_sha256.clone(),
            ledger_batch: seal.batch,
            ledger_path: ledger_path.into(),
            ledger_batch_sha256: file_sha,
            seq,
            transition,
            qualification_sha256: event.qualification_sha256.clone(),
        };
        decision.verify_against(&ledger)?;
        let mut step = Step::default();
        self.append(
            &Mapped {
                blobs: vec![blob],
                objects: vec![encode(&decision)?],
            },
            &mut step,
        )?;
        if matches!(transition, Transition::Initialized | Transition::Improved) {
            return self.urgent(Urgent::HwmChange, clock, step);
        }
        Ok(self.finish(step))
    }
}
