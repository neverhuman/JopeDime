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
use super::{mix, objective, partition};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

mod journal;
mod model;
mod reconcile;

pub use journal::*;
pub use model::*;
pub use reconcile::*;

/// An applied update not yet covered by a commit.
#[derive(Clone, Debug, Default)]
struct Pending {
    units: Units,
    records: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
struct Tracker {
    header: Option<Header>,
    shard: Option<Shard>,
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
    fn accept(&mut self, event: &TrainingConsumptionEvent) -> Result<()> {
        if let Some(reason) = &self.ended {
            bail!("consumption journal continues after a rejected update ({reason})");
        }
        if let TrainingConsumptionEvent::Attempt { header, shard } = event {
            ensure!(
                self.header.is_none(),
                "consumption journal has two attempt headers"
            );
            header.validate(shard.as_ref())?;
            self.next_step = header.initial_optimizer_step;
            self.applied_through = header.initial_optimizer_step;
            self.committed_through = header.initial_optimizer_step;
            self.header = Some(header.clone());
            self.shard = shard.clone();
            return Ok(());
        }
        let h = self
            .header
            .clone()
            .context("consumption journal must begin with its attempt header")?;
        let shard = self.shard.clone();
        match event {
            TrainingConsumptionEvent::Attempt { .. } => unreachable!(),
            TrainingConsumptionEvent::Microbatch {
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
                    *serial == h.serial_in(shard.as_ref(), *step, *microbatch)?,
                    "sampler cursor of step {step} microbatch {microbatch} differs"
                );
                ensure!(
                    *dropout_seed == h.dropout_seed(*step, *microbatch)?,
                    "dropout RNG identity of step {step} microbatch {microbatch} differs"
                );
                ensure!(
                    records.len() == h.rows_in(shard.as_ref(), *microbatch)?
                        && supervision.rows == records.len(),
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
            TrainingConsumptionEvent::Update {
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
                    TrainingUpdateOutcome::Applied => {
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
                    TrainingUpdateOutcome::Rejected => {
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
            TrainingConsumptionEvent::Commit { step, state_sha256 } => {
                ensure!(
                    self.step.units.microbatches == 0,
                    "commit inside an unfinished update"
                );
                ensure!(
                    *step <= self.applied_through && *step > self.committed_through,
                    "commit at step {step} must cover applied updates and advance"
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

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
