use super::*;
use anyhow::ensure;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeagueGateReason {
    NoAcceptedCompetitors,
    NoMeasuredCycleOrSpecialization,
    QualificationPurpose,
    FrozenBaselinePurpose,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LeagueGate {
    Deferred {
        reason: LeagueGateReason,
    },
    CandidateForSeparateReview {
        evidence_sha256: String,
        basis: String,
    },
}

/// Readiness gate only. It does not implement a PFSP sampler or authorize policy/serving
/// promotion. Member counts alone never activate the adaptive league.
pub struct Pfsp;
impl Pfsp {
    pub fn assess(manifest: &PayoffManifest, registry: &LeagueRegistry) -> Result<LeagueGate> {
        manifest.validate(registry)?;
        match manifest.purpose {
            PayoffPurpose::Qualification => {
                return Ok(LeagueGate::Deferred {
                    reason: LeagueGateReason::QualificationPurpose,
                })
            }
            PayoffPurpose::FrozenBaseline => {
                return Ok(LeagueGate::Deferred {
                    reason: LeagueGateReason::FrozenBaselinePurpose,
                })
            }
            PayoffPurpose::TrainingMatchmaking => {}
        }
        let ids = manifest.accepted_competitor_ids();
        if ids.is_empty() {
            return Ok(LeagueGate::Deferred {
                reason: LeagueGateReason::NoAcceptedCompetitors,
            });
        }
        let identity = manifest.identity_sha256(registry)?;
        if manifest.has_nontransitive_cycle(&ids) {
            return Ok(LeagueGate::CandidateForSeparateReview {
                evidence_sha256: identity,
                basis: "measured_nontransitive_cycle".into(),
            });
        }
        if manifest.has_family_specialization(&ids) {
            return Ok(LeagueGate::CandidateForSeparateReview {
                evidence_sha256: identity,
                basis: "measured_family_specialization".into(),
            });
        }
        Ok(LeagueGate::Deferred {
            reason: LeagueGateReason::NoMeasuredCycleOrSpecialization,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingMatchRecord {
    pub manifest_sha256: String,
    pub left_member_id: String,
    pub right_member_id: String,
    pub context_id: String,
    pub payoff_difference: f64,
    pub left_cpu_seconds: f64,
    pub right_cpu_seconds: f64,
}

/// Training-only matchmaking memory. Qualification outcomes are rejected before any row is
/// appended; callers cannot accidentally turn a qualification manifest into training memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchmakingMemory {
    rows: Vec<TrainingMatchRecord>,
}
impl MatchmakingMemory {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn rows(&self) -> &[TrainingMatchRecord] {
        &self.rows
    }

    pub fn absorb(&mut self, manifest: &PayoffManifest, registry: &LeagueRegistry) -> Result<()> {
        manifest.validate(registry)?;
        ensure!(
            manifest.purpose == PayoffPurpose::TrainingMatchmaking,
            "only training matchmaking evidence may enter training memory"
        );
        let manifest_sha256 = manifest.identity_sha256(registry)?;
        let mut additions = Vec::with_capacity(manifest.cells.len());
        for cell in &manifest.cells {
            additions.push(TrainingMatchRecord {
                manifest_sha256: manifest_sha256.clone(),
                left_member_id: cell.left_member_id.clone(),
                right_member_id: cell.right_member_id.clone(),
                context_id: cell.context_id.clone(),
                payoff_difference: cell.left_payoff - cell.right_payoff,
                left_cpu_seconds: cell.left_cpu_seconds,
                right_cpu_seconds: cell.right_cpu_seconds,
            });
        }
        self.rows.extend(additions);
        Ok(())
    }
}
