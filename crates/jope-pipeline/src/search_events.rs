//! Observational draw/request/physical-attempt joins. No replay or fitting authority.

use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

// Reuse I02's actual codec without changing the coordinator-owned Cargo graph. The integrator
// can replace this source-module import with a jope-records dependency; no codec is forked.
// Extract I02's jope_records-importing doctests only in their owning crate. Runtime
// codec checks remain compiled here; I05's nonzero integration fixtures exercise this adapter.
#[cfg(not(doctest))]
// The adopted I02 Measured* enum names remain unchanged in this read-only source import.
#[allow(dead_code, clippy::enum_variant_names)]
#[path = "../../jope-records/src/wire_contract.rs"]
mod wire;

pub const LOG_SCHEMA: &str = "jope.search-events/v1";
pub const DECISION_SCHEMA: &str = "jope.search-decision/v1";
pub const EDIT_SCHEMA: &str = "jope.edit-transition/v1";

/// I02 canonical bytes, including exact float tags and length-prefixed domain separation.
pub fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(
        wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value)?)?
            .as_bytes()
            .to_vec(),
    )
}

pub fn identity<T: Serialize>(value: &T) -> Result<String> {
    Ok(
        wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value)?)?
            .domain_separated_id()?
            .sha256_hex()
            .to_owned(),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloatBits {
    pub f64_bits: String,
}

impl FloatBits {
    pub fn new(value: f64) -> Result<Self> {
        ensure!(value.is_finite(), "nonfinite event measurement");
        Ok(Self {
            f64_bits: format!("{:016x}", value.to_bits()),
        })
    }

    pub fn value(&self) -> Result<f64> {
        let value = wire::ExactF64::from_hex("measurement", &self.f64_bits)?.to_f64();
        ensure!(value.is_finite(), "nonfinite event measurement");
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceRef {
    Known { sha256: String },
    Unknown { reason: String },
}

impl EvidenceRef {
    pub fn unknown(reason: &str) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }
    fn check(&self) -> Result<()> {
        match self {
            Self::Known { sha256 } => ensure!(
                crate::experiment::sha256(sha256),
                "invalid evidence reference"
            ),
            Self::Unknown { reason } => {
                ensure!(!reason.is_empty(), "unknown evidence needs a reason")
            }
        }
        Ok(())
    }
}

/// Authenticated role/exposure remains an I04 input. Its presence here never confers eligibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBinding {
    pub run_id: String,
    pub task: String,
    pub experiment: EvidenceRef,
    pub role_exposure: EvidenceRef,
    pub context_sha256: String,
    pub source_sha256: String,
    pub evaluator: EvidenceRef,
    pub profile: EvidenceRef,
    pub legal_set: EvidenceRef,
    pub fidelity: String,
}

impl SearchBinding {
    fn check(&self) -> Result<()> {
        ensure!(
            !self.run_id.is_empty() && !self.task.is_empty() && !self.fidelity.is_empty(),
            "missing search binding"
        );
        ensure!(
            crate::experiment::sha256(&self.context_sha256)
                && crate::experiment::sha256(&self.source_sha256),
            "invalid context/source identity"
        );
        for r in [
            &self.experiment,
            &self.role_exposure,
            &self.evaluator,
            &self.profile,
            &self.legal_set,
        ] {
            r.check()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProposalBehaviorLaw {
    Unknown {
        reason: String,
    },
    NotApplicable {
        reason: String,
    },
    DiscreteProbability {
        subject_sha256: String,
        probability: FloatBits,
    },
    ContinuousDensity {
        subject_sha256: String,
        density: FloatBits,
        measure: String,
    },
}

/// Backwards-compatible name for the proposal probability law recorded in search events.
pub type BehaviorLaw = ProposalBehaviorLaw;

impl ProposalBehaviorLaw {
    fn check(&self, source: &str, program: &Option<String>) -> Result<()> {
        // An edit's unconditional child law is not established by parent selection or seed law.
        if matches!(source, "mutate" | "crossover" | "random_fallback") {
            ensure!(
                matches!(self, Self::Unknown { .. }),
                "GP child law must be unknown; ancestry is not density"
            );
        }
        match self {
            Self::Unknown { reason } | Self::NotApplicable { reason } => {
                ensure!(!reason.is_empty(), "missing behavior-law reason")
            }
            Self::DiscreteProbability {
                subject_sha256,
                probability,
            } => {
                let p = probability.value()?;
                ensure!(
                    program.as_ref() == Some(subject_sha256) && p > 0. && p <= 1.,
                    "probability has wrong subject or range"
                );
            }
            Self::ContinuousDensity {
                subject_sha256,
                density,
                measure,
            } => {
                ensure!(
                    program.as_ref() == Some(subject_sha256)
                        && density.value()? >= 0.
                        && !measure.is_empty(),
                    "density has wrong subject or measure"
                );
            }
        }
        Ok(())
    }
}

/// Effort is per stage, never inferred from an absent physical evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effort {
    pub wall_seconds: Option<FloatBits>,
    pub cpu_seconds: Option<FloatBits>,
    pub gpu_seconds: Option<FloatBits>,
    pub peak_rss_bytes: Option<u64>,
    pub reason: String,
}

impl Effort {
    pub fn unknown(reason: &str) -> Self {
        Self {
            wall_seconds: None,
            cpu_seconds: None,
            gpu_seconds: None,
            peak_rss_bytes: None,
            reason: reason.into(),
        }
    }
    pub fn wall(seconds: f64) -> Result<Self> {
        Ok(Self {
            wall_seconds: Some(FloatBits::new(seconds)?),
            ..Self::unknown("caller-measured wall span; CPU/GPU/memory not instrumented")
        })
    }
    fn check(&self) -> Result<()> {
        ensure!(!self.reason.is_empty(), "missing cost reason");
        for n in [&self.wall_seconds, &self.cpu_seconds, &self.gpu_seconds]
            .into_iter()
            .flatten()
        {
            ensure!(n.value()? >= 0., "negative effort");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchDecision {
    pub schema: String,
    pub coordinate: String,
    pub source: String,
    pub seed_hex: Option<String>,
    pub program_sha256: Option<String>,
    pub canonical_sha256: Option<String>,
    pub request_id: Option<String>,
    #[serde(rename = "draw_disposition")]
    pub outcome: String,
    pub detail: Option<String>,
    pub behavior_law: ProposalBehaviorLaw,
    pub proposal_cost: Effort,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditTransition {
    pub schema: String,
    pub draw_id: String,
    /// Ordered, including both choices even when crossover selected the same parent twice.
    pub parents: Vec<String>,
    pub parent_canonical_sha256: Vec<Option<String>>,
    pub donor: Option<String>,
    pub sibling_group: String,
    pub operator: String,
    pub parameters: BTreeMap<String, String>,
    pub post_canonical_sha256: Option<String>,
    pub legal: bool,
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SearchEvent {
    Draw {
        decision: SearchDecision,
    },
    Edit {
        transition: EditTransition,
    },
    Admission {
        draw_id: String,
        disposition: String,
    },
    Started {
        draw_id: String,
        request_id: String,
        attempt_id: String,
        ordinal: u32,
        retry_of: Option<String>,
    },
    Finished {
        attempt_id: String,
        status: String,
        reason: Option<String>,
        observation_sha256: Option<String>,
        cost: Effort,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventRecord {
    pub event_id: String,
    pub event: SearchEvent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchEventLog {
    pub schema: String,
    pub binding: SearchBinding,
    pub events: Vec<EventRecord>,
    /// Sorted capture inventory: losing even an unattempted/shadow draw is detectable.
    pub captured_event_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reconciliation {
    pub draws: usize,
    pub duplicate_draws: usize,
    pub migration_decisions: usize,
    pub unique_requests: usize,
    pub evaluation_attempts: usize,
    pub failed_attempts: usize,
    pub logical_completed: usize,
    pub pending_attempts: usize,
    pub duplicate_deliveries: usize,
    pub proposal_costs: usize,
    pub terminal_costs: usize,
    pub evaluation_wall_seconds: f64,
    pub unknown_evaluation_wall_costs: usize,
}

mod log;
mod recording;
pub(crate) use recording::Recorder;

pub fn canonical_recipe(genome: &str) -> Option<String> {
    feat_math::genome::GenomeIr::parse(genome)
        .ok()
        .map(|ir| crate::sha256_hex(ir.canonical().as_bytes()))
}

/// Exact ordered input identity without exporting labels or floating-point JSON numbers.
pub fn cut_context_sha256(cut: &crate::prepare::Cut) -> Result<String> {
    let bits = |rows: &[Vec<f64>]| {
        rows.iter()
            .map(|r| {
                r.iter()
                    .map(|v| format!("{:016x}", v.to_bits()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    Ok(crate::sha256_hex(&serde_json::to_vec(&(
        cut.index,
        bits(&cut.train_x),
        bits(&cut.hold_x),
        bits(&[cut.train_y.clone(), cut.hold_y.clone()]),
        &cut.hold_indices,
    ))?))
}

/// Logger/search/operator source identity. Evaluator/build qualification remains separate.
pub fn source_sha256() -> String {
    crate::sha256_hex(
        &[
            include_bytes!("search.rs").as_slice(),
            include_bytes!("search_events.rs").as_slice(),
            include_bytes!("search_events/log.rs").as_slice(),
            include_bytes!("search_events/recording.rs").as_slice(),
            include_bytes!("../../feat-math/src/evolve.rs").as_slice(),
            include_bytes!("../../jope-records/src/wire_contract.rs").as_slice(),
        ]
        .concat(),
    )
}
