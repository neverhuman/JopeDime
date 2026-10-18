//! Multi-fidelity race: cheap rungs may screen candidates but cannot certify a champion.

use super::selection::{rank, ScoredCandidate};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    Cheap,
    Medium,
    Deep,
    Final,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RungResult {
    pub candidate: ScoredCandidate,
    pub rung: Rung,
    /// Immutable source/input identity for the evidence at this rung.
    pub source_head: String,
    pub eligible: bool,
}

/// Caller-pinned limits for each rung. Values are cumulative ceilings, not execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RungLimit {
    pub evaluations: u64,
    pub effort: u64,
    pub memory_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLadder {
    pub cheap: RungLimit,
    pub medium: RungLimit,
    pub deep: RungLimit,
    pub final_rung: RungLimit,
}

impl ResourceLadder {
    /// Every rung must strictly increase at least one dimension and never reduce another.
    pub fn validate(self) -> bool {
        let levels = [self.cheap, self.medium, self.deep, self.final_rung];
        levels.windows(2).all(|pair| {
            pair[1].evaluations >= pair[0].evaluations
                && pair[1].effort >= pair[0].effort
                && pair[1].memory_bytes >= pair[0].memory_bytes
                && (pair[1].evaluations > pair[0].evaluations
                    || pair[1].effort > pair[0].effort
                    || pair[1].memory_bytes > pair[0].memory_bytes)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ChampionDecision {
    Verified {
        genome_id: String,
        score: f64,
        proposal_sequence: u64,
    },
    Unavailable {
        reason: &'static str,
    },
}

/// Selects only from final-rung eligible evidence. Cheap-rung ordering is deliberately ignored.
pub fn final_champion(results: &[RungResult], expected_source_head: &str) -> ChampionDecision {
    let mut finals: Vec<_> = results
        .iter()
        .filter(|r| r.rung == Rung::Final && r.eligible && r.source_head == expected_source_head)
        .map(|r| r.candidate.clone())
        .collect();
    rank(&mut finals);
    match finals.into_iter().next() {
        Some(best) if best.score.is_finite() => ChampionDecision::Verified {
            genome_id: best.genome_id,
            score: best.score,
            proposal_sequence: best.proposal_sequence,
        },
        _ => ChampionDecision::Unavailable {
            reason: "no eligible final-rung candidate",
        },
    }
}
