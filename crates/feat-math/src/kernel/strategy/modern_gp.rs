//! Incremental controls for named modern GP candidates; no default profile or driver dispatch.

use super::race::{ChampionDecision, ResourceLadder, RungResult};

pub const ISLAND128_ID: &str = "modern-gp-island128.v1";
pub const STEADY_STATE_ID: &str = "modern-gp-steady-state.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateKind {
    Island128,
    SteadyState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateConfig {
    pub kind: CandidateKind,
    pub population: usize,
}

impl CandidateConfig {
    pub fn island128() -> Self {
        Self {
            kind: CandidateKind::Island128,
            population: 128,
        }
    }
    pub fn steady_state(population: usize) -> Option<Self> {
        (16..=64).contains(&population).then_some(Self {
            kind: CandidateKind::SteadyState,
            population,
        })
    }
    pub fn identity(self) -> &'static str {
        match self.kind {
            CandidateKind::Island128 => ISLAND128_ID,
            CandidateKind::SteadyState => STEADY_STATE_ID,
        }
    }
    pub fn validate(self) -> bool {
        match self.kind {
            CandidateKind::Island128 => self.population == 128,
            CandidateKind::SteadyState => (16..=64).contains(&self.population),
        }
    }
}

/// Candidate controls bind a named population strategy to its complete caller-supplied ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidatePlan {
    pub config: CandidateConfig,
    pub ladder: ResourceLadder,
}

impl CandidatePlan {
    pub fn validate(self) -> bool {
        self.config.validate() && self.ladder.validate()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedIncumbent {
    pub genome_id: String,
    pub score: f64,
    pub verification_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SearchOutcome {
    Champion(ChampionDecision),
    Incumbent(VerifiedIncumbent),
    Unavailable {
        reason: &'static str,
        effort_charged: u64,
    },
}

/// On exhaustion, retain only a caller-provided verified incumbent; unverified cheap-rung
/// observations cannot become fallback champions.
pub fn exhausted(incumbent: Option<VerifiedIncumbent>, effort_charged: u64) -> SearchOutcome {
    match incumbent {
        Some(value) if value.score.is_finite() && !value.verification_id.is_empty() => {
            SearchOutcome::Incumbent(value)
        }
        _ => SearchOutcome::Unavailable {
            reason: "budget exhausted without verified incumbent",
            effort_charged,
        },
    }
}

pub fn complete(results: &[RungResult], expected_source_head: &str) -> SearchOutcome {
    SearchOutcome::Champion(super::race::final_champion(results, expected_source_head))
}
