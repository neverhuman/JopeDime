//! Private, versioned research records (spec §13.2).
//!
//! The producer supplies the actual pre-admission slate and committed outcomes.
//! This module does not reconstruct missing costs, run a model, qualify guidance,
//! change discovery admission, or expose a public/release CLI command.

mod native_inputs;
pub use native_inputs::NativeCaptureInputs;

mod inventory;
pub use inventory::{RecordCounts, RecordInventory};
mod job_resolver;
pub use job_resolver::{AcceptedJobAdmission, TrustedJobResolver};
mod run;
pub use run::{record_run, RecordingBackend};

mod producer;
pub use producer::RoundCapture;
pub(crate) mod capture;
pub(crate) mod outbox;
pub use capture::{CaptureConfig, CaptureContext, ObservedSlate};
pub use outbox::{materialize_committed, materialize_committed_with_inventory};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs::File, io::Write, path::Path};

pub const VERSION: &str = "jope.research.round/v1";
pub const SCHEMA: &str = include_str!("round.schema.json");
pub const DOPE_SHA256: &str = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814";
pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordEvidence {
    Engine,
    SyntheticControl,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordTask {
    Regression,
    Binary,
}

/// These pins are producer attestations, not proof of corpus eligibility or Q1.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordIdentity {
    pub run_id: String,
    pub task: RecordTask,
    pub evidence: RecordEvidence,
    pub partition: String,
    pub source_commit: String,
    pub corpus_manifest_sha256: String,
    pub dataset_sha256: String,
    pub family_sha256: String,
    pub problem_sha256: String,
    pub bundle_sha256: String,
    pub conditioning_sha256: String,
    pub evaluator_sha256: String,
    pub dope_weights_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedProgram {
    pub program_sha256: String,
    pub operator_family: String,
    pub feat412: Vec<f32>,
    pub proj16: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedReveal {
    pub program: RecordedProgram,
    pub surrogate: f64,
    pub gate: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedPolicyKind {
    Heuristic,
    Learned,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorPolicy {
    pub kind: RecordedPolicyKind,
    pub explore: bool,
    /// RecordedReveal entries first, then all proposals in original order. Heuristic: null.
    pub slate_costs: Option<Vec<f64>>,
    /// Original proposal indices, independent of evaluation order or evaluate-all.
    pub admitted: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Rung0Measurement {
    Success {
        surrogate: f64,
        objective: f64,
        gate: bool,
    },
    Failed {
        code: crate::worker::WorkerFailure,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedOutcome {
    pub proposal_index: usize,
    pub wall_ms: u64,
    pub measurement: Rung0Measurement,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchRound {
    pub schema: String,
    pub identity: RecordIdentity,
    /// Zero is the seed round: no revealed context and every seed is admitted.
    pub round: u64,
    pub seed: u64,
    pub tokens: Vec<Vec<f32>>,
    pub mask: Vec<bool>,
    pub revealed: Vec<RecordedReveal>,
    pub proposals: Vec<RecordedProgram>,
    /// Each eligible proposal's index in the original source population, before dedup.
    pub source_indices: Vec<u64>,
    pub policy: BehaviorPolicy,
    pub evaluate_all_proposals: bool,
    pub dispatched: Vec<usize>,
    pub capture_complete: bool,
    pub outcomes: Vec<RecordedOutcome>,
}

fn hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./:-".contains(&b))
}

fn validate_program(program: &RecordedProgram) -> Result<()> {
    ensure!(hex(&program.program_sha256, 64), "invalid program digest");
    ensure!(name(&program.operator_family), "invalid operator family");
    ensure!(
        program.feat412.len() == 412 && program.feat412.iter().all(|v| v.is_finite()),
        "expected finite feat412"
    );
    ensure!(
        program.proj16.len() == 16 && program.proj16.iter().all(|v| v.is_finite()),
        "expected finite proj16"
    );
    Ok(())
}

impl ResearchRound {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == VERSION,
            "unsupported research record version"
        );
        let id = &self.identity;
        ensure!(name(&id.run_id), "invalid run identity");
        ensure!(
            id.partition == "train",
            "research capture requires train-only inputs"
        );
        ensure!(hex(&id.source_commit, 40), "invalid source commit");
        for value in [
            &id.corpus_manifest_sha256,
            &id.dataset_sha256,
            &id.family_sha256,
            &id.problem_sha256,
            &id.bundle_sha256,
            &id.conditioning_sha256,
            &id.evaluator_sha256,
        ] {
            ensure!(hex(value, 64), "invalid research input digest");
        }
        ensure!(
            id.dope_weights_sha256 == DOPE_SHA256,
            "frozen DOPE identity mismatch"
        );
        ensure!(
            self.tokens.len() == 10
                && self
                    .tokens
                    .iter()
                    .all(|row| row.len() == 32 && row.iter().all(|v| v.is_finite()))
                && self.mask.len() == 10,
            "expected ten finite 32-wide conditioning tokens and mask"
        );
        let n = self.proposals.len();
        ensure!((1..=16).contains(&n), "expected one to sixteen proposals");
        ensure!(
            self.source_indices.len() == n
                && self.source_indices.iter().collect::<BTreeSet<_>>().len() == n,
            "invalid original source indices"
        );
        let mut programs = BTreeSet::new();
        for p in self
            .proposals
            .iter()
            .chain(self.revealed.iter().map(|r| &r.program))
        {
            validate_program(p)?;
            ensure!(
                programs.insert(&p.program_sha256),
                "duplicate slate program"
            );
        }
        ensure!(
            self.revealed.iter().all(|r| r.surrogate.is_finite()),
            "nonfinite revealed value"
        );
        let policy = &self.policy;
        let admitted: BTreeSet<_> = policy.admitted.iter().copied().collect();
        ensure!(
            admitted.len() == policy.admitted.len() && admitted.iter().all(|i| *i < n),
            "invalid admitted indices"
        );
        if self.round == 0 {
            ensure!(
                self.revealed.is_empty()
                    && !policy.explore
                    && policy.kind == RecordedPolicyKind::Heuristic,
                "invalid seed round context"
            );
            ensure!(
                admitted == (0..n).collect(),
                "seed round must admit every seed"
            );
        } else {
            ensure!(
                self.revealed.len() == 4,
                "proposal round requires four revealed algorithms"
            );
            ensure!(
                n <= if policy.explore { 16 } else { 8 },
                "proposal count exceeds policy shape"
            );
            ensure!(admitted.len() == n.min(4), "wrong admission count");
        }
        match (&policy.kind, &policy.slate_costs) {
            (RecordedPolicyKind::Heuristic, None) => {}
            (RecordedPolicyKind::Learned, Some(costs)) => {
                ensure!(
                    self.round > 0 && n == if policy.explore { 16 } else { 8 },
                    "learned policy requires full serving slate"
                );
                ensure!(
                    costs.len() == 4 + n && costs.iter().all(|v| v.is_finite()),
                    "missing or nonfinite behavior costs"
                );
            }
            _ => anyhow::bail!("behavior costs do not match policy kind"),
        }
        let expected = if self.evaluate_all_proposals {
            (0..n).collect()
        } else {
            admitted
        };
        let dispatched: BTreeSet<_> = self.dispatched.iter().copied().collect();
        ensure!(
            dispatched.len() == self.dispatched.len() && dispatched.is_subset(&expected),
            "invalid actual dispatch population"
        );
        ensure!(
            self.capture_complete == (dispatched == expected),
            "incorrect capture completeness"
        );
        let mut observed = BTreeSet::new();
        for outcome in &self.outcomes {
            ensure!(
                outcome.proposal_index < n && observed.insert(outcome.proposal_index),
                "duplicate or unknown outcome"
            );
            if let Rung0Measurement::Success {
                surrogate,
                objective,
                ..
            } = &outcome.measurement
            {
                ensure!(
                    surrogate.is_finite() && objective.is_finite(),
                    "nonfinite successful measurement"
                );
            }
        }
        ensure!(
            observed == dispatched,
            "incomplete or unexpected evaluation inventory"
        );
        Ok(())
    }
}

/// Create a fresh private JSONL stream. No append-to-existing or automatic resume.
/// A successful append means the whole validated round was written and synced.
/// A failed write/sync poisons the stream; its partial file is evidence to retain.
pub struct RecordWriter {
    file: File,
    binding: Option<Vec<u8>>,
    next_round: u64,
    poisoned: bool,
}

impl RecordWriter {
    pub fn create(path: &Path) -> Result<Self> {
        #[cfg(not(unix))]
        {
            let _ = path;
            anyhow::bail!("private research file creation is not qualified on this platform");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)?;
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            File::open(parent)?.sync_all()?;
            Ok(Self {
                file,
                binding: None,
                next_round: 0,
                poisoned: false,
            })
        }
    }

    pub fn append(&mut self, record: &ResearchRound) -> Result<()> {
        ensure!(
            !self.poisoned,
            "record stream is poisoned; preserve it and start a new run"
        );
        record.validate()?;
        ensure!(
            record.round == self.next_round,
            "duplicate, missing or out-of-order round"
        );
        let binding = serde_json::to_vec(&(
            &record.identity,
            record.seed,
            &record.tokens,
            &record.mask,
            record.evaluate_all_proposals,
        ))?;
        ensure!(
            self.binding
                .as_ref()
                .is_none_or(|previous| previous == &binding),
            "run/input identity, conditioning, seed or capture mode changed"
        );
        let next = self
            .next_round
            .checked_add(1)
            .context("round counter exhausted")?;
        let mut bytes = serde_json::to_vec(record)?;
        bytes.push(b'\n');
        ensure!(
            bytes.len() <= MAX_LINE_BYTES,
            "research record line exceeds size limit"
        );
        self.poisoned = true;
        self.file.write_all(&bytes)?;
        self.file.sync_all()?;
        self.binding = Some(binding);
        self.next_round = next;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

pub(crate) mod qualified;
