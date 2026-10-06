//! Track E4/E5 library measurement and training-only constant refinement.
//! Shipping offspring, CLI selection and receipt v4 belong to E6. Legacy search
//! continues through its existing evaluator and publishes unchanged observations.

use anyhow::{ensure, Result};
use feat_math::{
    evolve::FaultPolicy,
    kernel::{admission::FaultTotals, profile::SearchProfile},
    GenomeIr, Rows,
};
use serde::Serialize;

use crate::{clock::Clock, prepare::Cut, search::Observation, sha256_hex};

mod fitting;
mod measurement;
mod refinement;
pub use refinement::{RefinedChild, TrainingRefinement};
// laneD:B17-lm-refinement begin
mod engine;
pub use engine::{
    EngineRecord, EngineSelection, RefineMethod, RefineRecord, DAMPED_LS, DAMPED_LS_MIN_TRIALS,
};
pub use refinement::{IterationBits, PipelineCheck, RefinementTrace, StageTrace, SurrogateTrace};
// laneD:B17-lm-refinement end
// laneD:B18-pareto-archive begin
pub mod archive;
pub use engine::{ArchivePolicy, ArchiveRecord, FRONT_LOSS_COST};
// laneD:B18-pareto-archive end

/// A structural or traced Hard refusal, with no candidate score.
pub const INADMISSIBLE: &str = "inadmissible";
/// A child produced by training-only constant refinement.
pub const REFINE: &str = "refine";

/// Kernel2 evidence adds fields without extending legacy observation JSON.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Kernel2Observation {
    #[serde(flatten)]
    pub observation: Observation,
    pub domain_fault_rate: f64,
    pub saturation_fault_rate: f64,
    pub constant_output: bool,
    pub faulted_cells: u64,
    /// Held-out R2 after the configured WP1.5 penalty. Absent on refusal/failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_score: Option<f64>,
    /// Full counts for library ranking; E6 owns the versioned receipt mapping.
    #[serde(skip)]
    pub faults: FaultTotals,
}

impl Kernel2Observation {
    fn refresh_faults(&mut self) {
        self.domain_fault_rate = self.faults.domain_rate();
        self.saturation_fault_rate = self.faults.saturation_rate();
        self.constant_output = self.faults.constant_passes > 0;
        self.faulted_cells = self.faults.faulted_cells();
    }
}

/// One shared per-cut allowance for measurements and all optimizer trials.
/// Rejected attempts spend budget too; an exhausted allowance starts no work.
#[derive(Clone, Debug)]
pub struct EvaluationBudget {
    limit: usize,
    used: usize,
}

impl EvaluationBudget {
    pub fn new(limit: usize) -> Self {
        Self { limit, used: 0 }
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn remaining(&self) -> usize {
        self.limit - self.used
    }
    fn charge(&mut self, count: usize) -> bool {
        if count > self.remaining() {
            return false;
        }
        self.used += count;
        true
    }
}

/// Opt-in measurement for the shipped revision-three wrapper profiles.
#[derive(Clone, Debug)]
pub struct Kernel2Evaluator {
    profile: SearchProfile,
    fault_policy: FaultPolicy,
    /// Opt-in engine methods; the default selects none.
    engine: EngineSelection,
}

impl Kernel2Evaluator {
    pub fn new(profile: SearchProfile) -> Result<Self> {
        ensure!(
            matches!(profile.id, "kernel2-baseline" | "kernel2-experimental")
                && profile.revision == 3,
            "kernel2 revision-three search profile required"
        );
        Ok(Self {
            profile,
            // Settled OQ4: no automatic Penalize default flip before R1.
            fault_policy: FaultPolicy::RejectHard,
            engine: EngineSelection::default(),
        })
    }

    pub fn with_fault_policy(mut self, policy: FaultPolicy) -> Result<Self> {
        policy.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            policy != FaultPolicy::Legacy,
            "kernel2 requires a Hard-refusing policy"
        );
        self.fault_policy = policy;
        Ok(self)
    }
    pub fn profile(&self) -> &SearchProfile {
        &self.profile
    }
    pub fn fault_policy(&self) -> FaultPolicy {
        self.fault_policy
    }

    // laneD:B17-lm-refinement begin
    /// Select opt-in engine methods. They apply to the guarded route that searches use
    /// (`refine_training_guarded` and `finish_generation_guarded`).
    pub fn with_engine(mut self, engine: EngineSelection) -> Result<Self> {
        engine.check_options()?;
        self.engine = engine;
        Ok(self)
    }
    pub fn engine(&self) -> &EngineSelection {
        &self.engine
    }
    /// `None` for the default selection, so a default receipt keeps its bytes.
    pub fn engine_record(&self) -> Option<EngineRecord> {
        self.engine.record()
    }
    // laneD:B17-lm-refinement end
}

/// Refuse malformed rows before fitting a stage or executing a fitted prefix.
fn rectangular_width(rows: &Rows) -> Result<usize> {
    use feat_math::kernel::width::{MismatchKind, WidthMismatch, WidthReq};
    let width = rows.first().map_or(0, Vec::len);
    if let Some((row, values)) = rows.iter().enumerate().find(|(_, r)| r.len() != width) {
        return Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::Exactly(width),
            got: values.len(),
            kind: MismatchKind::RaggedInput { row },
        }
        .into());
    }
    Ok(width)
}

#[cfg(test)]
mod refinement_tests;
#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod refinement_guarded_tests;

#[cfg(test)]
mod refinement_lm_tests;

#[cfg(test)]
mod archive_tests;

pub use refinement::{GuardedGeneration, GuardedTrainingRefinement};

/// Bounded profile-driven generation coordinator; CLI/v4 integration is separate.
pub mod search;

#[cfg(test)]
mod cli_tests;
#[cfg(test)]
mod publication_tests;

pub mod cli;
pub mod publication;
pub mod receipt;
pub mod records;
pub mod run;

mod messages;
