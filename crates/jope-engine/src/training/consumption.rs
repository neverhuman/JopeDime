//! Per-rank training consumption journal (W6-T04).
//!
//! Each rank writes one append-only JSON-lines file per attempt, recording what
//! it actually consumed and committed:
//!
//! - `attempt`: the run constants that fix the sampler cursor and the dropout
//!   RNG identity of every microbatch.
//! - `microbatch`: one per completed forward/backward. It holds the optimizer
//!   step, microbatch index, sampler cursor, dropout seed, the sampled record
//!   ids in row order, and the supervision they delivered after masking.
//! - `update`: the outcome of each attempted optimizer update.
//!   - `applied`: the in-memory optimizer step advanced by exactly one.
//!   - `rejected`: nothing changed, and the attempt ends. Lineage B has no
//!     skip-and-continue.
//! - `commit`: a durable checkpoint now covers every applied update through `step`.
//!
//! Commit semantics keep three units apart:
//! - *computed* microbatches are physical work;
//! - *applied* updates are in-memory optimizer steps;
//! - *committed* updates are those a durable checkpoint covers.
//!
//! Applied-but-uncommitted updates are lost when an attempt ends and are
//! re-executed from the committed cursor. They count as computation, never as
//! committed exposure. The same state machine validates the writer and the
//! reader, so the trainer cannot emit a journal that reconciliation would
//! reject.
use super::objective::Supervision;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "jope.training_consumption.v1";
/// Identity of the per-microbatch dropout seed derivation, see [`dropout_seed`].
pub const RNG_ID: &str = "mix_seed_local_step_rank_microbatch_v1";

/// The seed passed to `tch::manual_seed` before each microbatch.
pub fn dropout_seed(seed: u64, local_step: u64, rank: usize, microbatch: usize) -> i64 {
    super::mix(
        seed ^ local_step.wrapping_mul(65537) ^ (rank as u64).wrapping_mul(257) ^ microbatch as u64,
    ) as i64
}

/// Sampler cursor of one rank-local microbatch: the global example index its
/// first comparison is drawn at.
pub fn sampler_serial(
    local_step: u64,
    global_batch: usize,
    world: usize,
    rank: usize,
    microbatch: usize,
    micro: usize,
) -> Result<u64> {
    ensure!(
        world > 0 && rank < world && global_batch % world == 0,
        "invalid sampler partition"
    );
    local_step
        .checked_mul(global_batch as u64)
        .and_then(|x| x.checked_add((rank * (global_batch / world)) as u64))
        .and_then(|x| x.checked_add((microbatch * micro) as u64))
        .context("sampler cursor overflow")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub schema: String,
    pub rank: usize,
    pub world: usize,
    pub microbatch: usize,
    pub global_batch: usize,
    pub accumulation: usize,
    pub seed: u64,
    pub sampler_id: String,
    pub rng: String,
    /// Optimizer step at which this lineage's sampler counter is zero.
    pub sampling_offset: u64,
    pub initial_optimizer_step: u64,
    pub target_optimizer_step: u64,
    pub corpus_sha256: String,
    pub replay_policy_sha256: Option<String>,
    /// Digest of the checkpoint state this attempt resumed from, if any.
    pub resume_state_sha256: Option<String>,
}

impl Header {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.rng == RNG_ID,
            "unknown consumption journal schema or RNG identity"
        );
        ensure!(
            self.world > 0
                && self.rank < self.world
                && self.microbatch > 0
                && self.microbatch % super::objective::COMPARISON_ROWS == 0
                && self.accumulation > 0
                && self
                    .accumulation
                    .checked_mul(self.world)
                    .and_then(|x| x.checked_mul(self.microbatch))
                    == Some(self.global_batch),
            "consumption journal partition is inconsistent"
        );
        ensure!(
            self.sampling_offset <= self.initial_optimizer_step
                && self.initial_optimizer_step <= self.target_optimizer_step,
            "consumption journal step window is inconsistent"
        );
        ensure!(
            is_sha256(&self.corpus_sha256)
                && self.replay_policy_sha256.as_deref().is_none_or(is_sha256)
                && self.resume_state_sha256.as_deref().is_none_or(is_sha256),
            "consumption journal identity digest is malformed"
        );
        Ok(())
    }

    /// Sampler cursor of `microbatch` within optimizer `step`.
    pub fn serial(&self, step: u64, microbatch: usize) -> Result<u64> {
        let local = step
            .checked_sub(self.sampling_offset)
            .context("step precedes the sampling origin")?;
        sampler_serial(
            local,
            self.global_batch,
            self.world,
            self.rank,
            microbatch,
            self.microbatch,
        )
    }

    pub fn dropout_seed(&self, step: u64, microbatch: usize) -> Result<i64> {
        let local = step
            .checked_sub(self.sampling_offset)
            .context("step precedes the sampling origin")?;
        Ok(dropout_seed(self.seed, local, self.rank, microbatch))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Applied,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Attempt {
        header: Header,
    },
    Microbatch {
        step: u64,
        microbatch: usize,
        serial: u64,
        dropout_seed: i64,
        records: Vec<u64>,
        supervision: Supervision,
    },
    Update {
        step: u64,
        outcome: Outcome,
        microbatches: usize,
        gradient_norm: Option<f64>,
        clipped: Option<bool>,
        reason: Option<String>,
    },
    Commit {
        step: u64,
        state_sha256: String,
    },
}

impl Event {
    fn key(&self) -> String {
        match self {
            Self::Attempt { .. } => "attempt".into(),
            Self::Microbatch {
                step, microbatch, ..
            } => format!("microbatch:{step}:{microbatch}"),
            Self::Update { step, .. } => format!("update:{step}"),
            Self::Commit { step, .. } => format!("commit:{step}"),
        }
    }
}

/// Consumption measured in three units. Presentations count sampled rows,
/// including a record presented again; supervised positions count targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Units {
    pub microbatches: u64,
    pub presentations: u64,
    pub supervised_positions: u64,
}

impl Units {
    fn add(&mut self, other: &Self) -> Result<()> {
        let sum = |a: u64, b: u64| a.checked_add(b).context("consumption counter overflow");
        *self = Self {
            microbatches: sum(self.microbatches, other.microbatches)?,
            presentations: sum(self.presentations, other.presentations)?,
            supervised_positions: sum(self.supervised_positions, other.supervised_positions)?,
        };
        Ok(())
    }
}

/// An applied update not yet covered by a commit.
#[derive(Clone, Debug, Default)]
struct Pending {
    units: Units,
    records: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
struct Tracker {
    header: Option<Header>,
    track_records: bool,
    next_step: u64,
    step: Pending,
    ended: Option<String>,
    applied_through: u64,
    committed_through: u64,
    computed: Units,
    committed: Units,
    applied: u64,
    rejected: u64,
    commits: u64,
    pending: Vec<Pending>,
    committed_records: BTreeSet<u64>,
}

impl Tracker {
    fn accept(&mut self, event: &Event) -> Result<()> {
        if let Some(reason) = &self.ended {
            bail!("consumption journal continues after a rejected update ({reason})");
        }
        if let Event::Attempt { header } = event {
            ensure!(
                self.header.is_none(),
                "consumption journal has two attempt headers"
            );
            header.validate()?;
            self.next_step = header.initial_optimizer_step;
            self.applied_through = header.initial_optimizer_step;
            self.committed_through = header.initial_optimizer_step;
            self.header = Some(header.clone());
            return Ok(());
        }
        let h = self
            .header
            .clone()
            .context("consumption journal must begin with its attempt header")?;
        match event {
            Event::Attempt { .. } => unreachable!(),
            Event::Microbatch {
                step,
                microbatch,
                serial,
                dropout_seed,
                records,
                supervision,
            } => {
                ensure!(
                    *step == self.next_step && *step < h.target_optimizer_step,
                    "microbatch for step {step} out of order or beyond the target"
                );
                ensure!(
                    *microbatch == self.step.units.microbatches as usize
                        && *microbatch < h.accumulation,
                    "microbatch {microbatch} of step {step} out of order"
                );
                ensure!(
                    *serial == h.serial(*step, *microbatch)?,
                    "sampler cursor of step {step} microbatch {microbatch} differs"
                );
                ensure!(
                    *dropout_seed == h.dropout_seed(*step, *microbatch)?,
                    "dropout RNG identity of step {step} microbatch {microbatch} differs"
                );
                ensure!(
                    records.len() == h.microbatch && supervision.rows == records.len(),
                    "microbatch record count differs from the partition"
                );
                self.step.units.add(&Units {
                    microbatches: 1,
                    presentations: records.len() as u64,
                    supervised_positions: supervision.supervised_positions as u64,
                })?;
                if self.track_records {
                    self.step.records.extend(records);
                }
            }
            Event::Update {
                step,
                outcome,
                microbatches,
                gradient_norm,
                clipped,
                reason,
            } => {
                ensure!(
                    *step == self.next_step && *step < h.target_optimizer_step,
                    "update for step {step} out of order or beyond the target"
                );
                ensure!(
                    *microbatches as u64 == self.step.units.microbatches,
                    "update for step {step} counts microbatches the journal did not record"
                );
                let step_work = std::mem::take(&mut self.step);
                self.computed.add(&step_work.units)?;
                match outcome {
                    Outcome::Applied => {
                        ensure!(
                            *microbatches == h.accumulation,
                            "update for step {step} applied before every accumulation microbatch"
                        );
                        let norm =
                            gradient_norm.context("applied update lacks its gradient norm")?;
                        ensure!(
                            norm.is_finite()
                                && norm >= 0.0
                                && *clipped == Some(norm > 1.0)
                                && reason.is_none(),
                            "applied update record for step {step} is inconsistent"
                        );
                        self.applied += 1;
                        self.next_step += 1;
                        self.applied_through = self.next_step;
                        self.pending.push(step_work);
                    }
                    Outcome::Rejected => {
                        let reason = reason
                            .as_deref()
                            .filter(|r| !r.is_empty())
                            .context("rejected update lacks its reason")?;
                        ensure!(
                            gradient_norm.is_none_or(|n| !n.is_finite() || n >= 0.0)
                                && clipped.is_none(),
                            "rejected update for step {step} claims an applied clip"
                        );
                        self.rejected += 1;
                        self.ended = Some(reason.to_string());
                    }
                }
            }
            Event::Commit { step, state_sha256 } => {
                ensure!(
                    self.step.units.microbatches == 0,
                    "commit inside an unfinished update"
                );
                ensure!(
                    *step == self.applied_through && *step > self.committed_through,
                    "commit at step {step} must cover exactly the applied updates and advance"
                );
                ensure!(is_sha256(state_sha256), "commit state digest is malformed");
                self.commit_through(*step)?;
                self.commits += 1;
            }
        }
        Ok(())
    }

    fn commit_through(&mut self, step: u64) -> Result<()> {
        let count = (step - self.committed_through) as usize;
        ensure!(
            count <= self.pending.len(),
            "commit covers unapplied updates"
        );
        for pending in self.pending.drain(..count) {
            self.committed.add(&pending.units)?;
            self.committed_records.extend(pending.records);
        }
        self.committed_through = step;
        Ok(())
    }
}

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
    pub fn create(path: &Path, header: Header) -> Result<Self> {
        header.validate()?;
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
        journal.append(&Event::Attempt { header })?;
        journal.sync()?;
        if let Some(parent) = path.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(journal)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn append(&mut self, event: &Event) -> Result<()> {
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
        self.append(&Event::Microbatch {
            step,
            microbatch,
            serial,
            dropout_seed,
            records: records.iter().map(|&r| r as u64).collect(),
            supervision,
        })
    }

    pub fn applied(&mut self, step: u64, gradient_norm: f64) -> Result<()> {
        self.append(&Event::Update {
            step,
            outcome: Outcome::Applied,
            microbatches: self.tracker.step.units.microbatches as usize,
            gradient_norm: Some(gradient_norm),
            clipped: Some(gradient_norm > 1.0),
            reason: None,
        })
    }

    /// Record that the attempted update of `step` was not applied. The
    /// attempt ends; the journal refuses every later event.
    pub fn rejected(&mut self, step: u64, reason: &str) -> Result<()> {
        self.append(&Event::Update {
            step,
            outcome: Outcome::Rejected,
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
        self.append(&Event::Commit {
            step,
            state_sha256: state_sha256.into(),
        })?;
        self.sync()
    }

    pub fn applied_through(&self) -> u64 {
        self.tracker.applied_through
    }
}

/// The next sampler position after a journal's durable commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Cursor {
    pub optimizer_step: u64,
    pub serial: u64,
    pub dropout_seed: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reconciliation {
    pub header: Header,
    /// Every journaled microbatch whose update outcome was recorded,
    /// including rejected and re-executed work.
    pub computed: Units,
    /// Microbatches of a step interrupted before its update outcome.
    pub unfinished: Units,
    pub unfinished_step: Option<u64>,
    pub attempted_updates: u64,
    pub applied_updates: u64,
    pub rejected_updates: u64,
    pub ended_by_rejection: bool,
    pub committed_updates: u64,
    pub committed: Units,
    pub committed_through: u64,
    pub applied_through: u64,
    /// Applied updates no durable checkpoint covers: lost at attempt end.
    pub uncommitted_applied_updates: u64,
    pub commit_events: u64,
    /// The durable checkpoint is ahead of the journal's last commit event.
    pub commit_event_missing: bool,
    pub torn_tail: bool,
    pub duplicate_events_ignored: u64,
    pub unique_committed_records: u64,
    #[serde(skip)]
    pub committed_records: BTreeSet<u64>,
    pub next: Cursor,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Complete lines of a journal and whether a torn final line was dropped.
fn complete_lines(bytes: &[u8]) -> (Vec<&[u8]>, bool) {
    match bytes.iter().rposition(|&b| b == b'\n') {
        Some(end) => (
            bytes[..end].split(|&b| b == b'\n').collect(),
            end + 1 < bytes.len(),
        ),
        None => (Vec::new(), !bytes.is_empty()),
    }
}

/// Validate a journal and derive its consumption. `durable_step` is the
/// optimizer step of the run's durable `latest.json`, when known: it may be
/// ahead of the last commit event (a crash between checkpoint publication and
/// the journal commit) but never behind it and never beyond applied work.
pub fn reconcile(bytes: &[u8], durable_step: Option<u64>) -> Result<Reconciliation> {
    let (lines, torn_tail) = complete_lines(bytes);
    let mut tracker = Tracker {
        track_records: true,
        ..Tracker::default()
    };
    let mut seen = BTreeMap::<String, &[u8]>::new();
    let mut duplicates = 0;
    for (n, line) in lines.into_iter().enumerate() {
        let event: Event = serde_json::from_slice(line)
            .with_context(|| format!("malformed consumption journal line {}", n + 1))?;
        let key = event.key();
        if let Some(previous) = seen.get(&key) {
            ensure!(
                *previous == line,
                "conflicting duplicate consumption event {key}"
            );
            duplicates += 1;
            continue;
        }
        seen.insert(key, line);
        tracker.accept(&event)?;
    }
    let header = tracker
        .header
        .clone()
        .context("consumption journal has no attempt header")?;
    let commit_events = tracker.commits;
    let by_events = tracker.committed_through;
    let mut commit_event_missing = false;
    if let Some(durable) = durable_step {
        ensure!(
            durable >= by_events,
            "consumption journal commits step {by_events} beyond the durable checkpoint {durable}"
        );
        ensure!(
            durable <= tracker.applied_through,
            "durable checkpoint {durable} is beyond the journal's applied updates"
        );
        if durable > by_events {
            tracker.commit_through(durable)?;
            commit_event_missing = true;
        }
    }
    let committed_through = tracker.committed_through;
    let unfinished = tracker.step.units;
    Ok(Reconciliation {
        computed: tracker.computed,
        unfinished,
        unfinished_step: (unfinished.microbatches > 0).then_some(tracker.next_step),
        attempted_updates: tracker.applied + tracker.rejected,
        applied_updates: tracker.applied,
        rejected_updates: tracker.rejected,
        ended_by_rejection: tracker.ended.is_some(),
        committed_updates: committed_through - header.initial_optimizer_step,
        committed: tracker.committed,
        committed_through,
        applied_through: tracker.applied_through,
        uncommitted_applied_updates: tracker.applied_through - committed_through,
        commit_events,
        commit_event_missing,
        torn_tail,
        duplicate_events_ignored: duplicates,
        unique_committed_records: tracker.committed_records.len() as u64,
        next: Cursor {
            optimizer_step: committed_through,
            serial: header.serial(committed_through, 0)?,
            dropout_seed: header.dropout_seed(committed_through, 0)?,
        },
        committed_records: tracker.committed_records,
        header,
    })
}

/// A prefix recorded in a checkpoint receipt must still be the journal's prefix.
pub fn verify_binding(bytes: &[u8], binding: &Binding) -> Result<()> {
    ensure!(
        binding.schema == SCHEMA
            && bytes.len() as u64 >= binding.bytes
            && hex::encode(Sha256::digest(&bytes[..binding.bytes as usize])) == binding.sha256,
        "consumption journal no longer contains its committed prefix"
    );
    Ok(())
}

/// Re-derive every journaled microbatch from its cursor with the frozen
/// sampler; the recorded records must be exactly what the sampler yields.
pub fn verify_records(
    bytes: &[u8],
    mut sampler: impl FnMut(usize, usize) -> Vec<usize>,
) -> Result<u64> {
    let (lines, _) = complete_lines(bytes);
    let mut verified = 0;
    for line in lines {
        if let Event::Microbatch {
            step,
            microbatch,
            serial,
            records,
            ..
        } = serde_json::from_slice::<Event>(line)?
        {
            let expected = sampler(usize::try_from(serial)?, records.len());
            ensure!(
                expected.len() == records.len()
                    && expected.iter().zip(&records).all(|(&e, &r)| e as u64 == r),
                "recorded records of step {step} microbatch {microbatch} differ from the sampler"
            );
            verified += 1;
        }
    }
    Ok(verified)
}

/// Run-level totals across consecutive attempts of one rank.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunTotals {
    pub attempts: usize,
    pub initial_optimizer_step: u64,
    pub committed_through: u64,
    pub committed_updates: u64,
    /// In-memory optimizer steps, including lost and re-executed ones.
    pub applied_updates: u64,
    pub lost_applied_updates: u64,
    pub rejected_updates: u64,
    pub computed: Units,
    pub committed: Units,
    pub unique_committed_records: u64,
}

/// Each attempt must resume at its predecessor's durable commit with the
/// same run constants; unresolved gaps are refused, never bridged.
pub fn reconcile_run(attempts: &[Reconciliation]) -> Result<RunTotals> {
    let first = attempts.first().context("run has no attempts")?;
    let same = |a: &Header, b: &Header| {
        let strip = |h: &Header| Header {
            initial_optimizer_step: 0,
            resume_state_sha256: None,
            ..h.clone()
        };
        strip(a) == strip(b)
    };
    let mut totals = RunTotals {
        attempts: attempts.len(),
        initial_optimizer_step: first.header.initial_optimizer_step,
        committed_through: first.header.initial_optimizer_step,
        committed_updates: 0,
        applied_updates: 0,
        lost_applied_updates: 0,
        rejected_updates: 0,
        computed: Units::default(),
        committed: Units::default(),
        unique_committed_records: 0,
    };
    let mut records = BTreeSet::new();
    for (i, attempt) in attempts.iter().enumerate() {
        if i > 0 {
            let previous = &attempts[i - 1];
            ensure!(
                same(&previous.header, &attempt.header),
                "attempt {i} changed the run constants"
            );
            ensure!(
                attempt.header.initial_optimizer_step == previous.committed_through
                    && attempt.header.resume_state_sha256.is_some(),
                "attempt {i} does not resume at its predecessor's durable commit"
            );
            totals.lost_applied_updates += previous.uncommitted_applied_updates;
        }
        totals.committed_through = attempt.committed_through;
        totals.committed_updates += attempt.committed_updates;
        totals.applied_updates += attempt.applied_updates;
        totals.rejected_updates += attempt.rejected_updates;
        totals.computed.add(&attempt.computed)?;
        totals.computed.add(&attempt.unfinished)?;
        totals.committed.add(&attempt.committed)?;
        records.extend(attempt.committed_records.iter().copied());
    }
    totals.unique_committed_records = records.len() as u64;
    Ok(totals)
}
