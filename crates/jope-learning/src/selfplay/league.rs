//! Minimal frozen comparison registry. Adaptive league scheduling is deliberately deferred.
//!
//! This is a worker-local v1 proposal, not a canonical agy schema or a live arena adapter. All
//! measurements, acceptance receipts and identities are caller-supplied evidence references.
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const LEAGUE_SCHEMA: &str = "jope.selfplay.frozen_league.v1";
pub const REGISTRY_SCHEMA: &str = "jope.selfplay.league_registry.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberKind {
    FrozenParent,
    SimpleControl,
    AcceptedCompetitor,
}

/// Immutable historical acquisition identity and measured production cost.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoricalCost {
    pub evidence_sha256: String,
    pub cpu_seconds: f64,
    pub wall_seconds: f64,
}
impl HistoricalCost {
    fn validate(&self) -> Result<()> {
        ensure!(
            is_sha256(&self.evidence_sha256),
            "historical cost evidence digest required"
        );
        ensure!(
            self.cpu_seconds.is_finite() && self.cpu_seconds >= 0.0,
            "finite nonnegative historical CPU cost required"
        );
        ensure!(
            self.wall_seconds.is_finite() && self.wall_seconds >= 0.0,
            "finite nonnegative historical wall cost required"
        );
        Ok(())
    }
}

/// Historical policy/control identity. A repeated `member_id` may only be registered with the
/// exact same digest, including its original acquisition cost and protocol/support identities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeagueMember {
    pub member_id: String,
    pub kind: MemberKind,
    pub policy_sha256: String,
    pub source_manifest_sha256: String,
    pub protocol_sha256: String,
    pub support_sha256: String,
    pub task_distribution_sha256: String,
    pub historical_cost: HistoricalCost,
    /// Present only when an external accepted-policy process supplied a receipt.
    pub acceptance_receipt_sha256: Option<String>,
}
impl LeagueMember {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.member_id.trim().is_empty(), "member ID required");
        for (name, value) in [
            ("policy", &self.policy_sha256),
            ("source manifest", &self.source_manifest_sha256),
            ("protocol", &self.protocol_sha256),
            ("support", &self.support_sha256),
            ("task distribution", &self.task_distribution_sha256),
        ] {
            ensure!(is_sha256(value), "{name} SHA-256 required");
        }
        self.historical_cost.validate()?;
        if let Some(receipt) = &self.acceptance_receipt_sha256 {
            ensure!(is_sha256(receipt), "acceptance receipt SHA-256 malformed");
        }
        ensure!(
            (self.kind == MemberKind::AcceptedCompetitor)
                == self.acceptance_receipt_sha256.is_some(),
            "only an accepted competitor may carry an accepted-policy receipt"
        );
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}

/// Historical member store. Registration is append-only by identity; replacing cost or a source
/// digest under the same display ID is refused.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeagueRegistry {
    schema: String,
    members: BTreeMap<String, LeagueMember>,
}
impl Default for LeagueRegistry {
    fn default() -> Self {
        Self {
            schema: REGISTRY_SCHEMA.into(),
            members: BTreeMap::new(),
        }
    }
}
impl LeagueRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, member: LeagueMember) -> Result<bool> {
        member.validate()?;
        match self.members.get(&member.member_id) {
            Some(existing) if existing == &member => Ok(false),
            Some(_) => bail!("historical member ID cannot be rebound to a new identity or cost"),
            None => {
                self.members.insert(member.member_id.clone(), member);
                Ok(true)
            }
        }
    }

    pub fn member(&self, member_id: &str) -> Option<&LeagueMember> {
        self.members.get(member_id)
    }
    pub fn members(&self) -> impl Iterator<Item = &LeagueMember> {
        self.members.values()
    }

    /// The baseline registry exists before an adaptive league: it must retain the frozen parent
    /// and at least one simple control. Competitors are optional and require accepted receipts.
    pub fn validate_frozen_baseline(&self) -> Result<()> {
        ensure!(
            self.schema == REGISTRY_SCHEMA,
            "unsupported league registry schema"
        );
        ensure!(
            self.members
                .values()
                .any(|m| m.kind == MemberKind::FrozenParent),
            "frozen parent is required"
        );
        ensure!(
            self.members
                .values()
                .any(|m| m.kind == MemberKind::SimpleControl),
            "at least one simple control is required"
        );
        let mut ids = BTreeSet::new();
        for (key, member) in &self.members {
            member.validate()?;
            ensure!(
                key == &member.member_id,
                "registry key/member identity mismatch"
            );
            ensure!(
                ids.insert(member.member_id.as_str()),
                "duplicate historical member identity"
            );
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String> {
        self.validate_frozen_baseline()?;
        digest(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContextIdentity {
    pub context_id: String,
    pub family_id: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayoffPurpose {
    FrozenBaseline,
    TrainingMatchmaking,
    Qualification,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemberSnapshot {
    pub member_id: String,
    pub identity_sha256: String,
    pub kind: MemberKind,
    pub acceptance_receipt_sha256: Option<String>,
    pub historical_cost: HistoricalCost,
}
impl From<&LeagueMember> for MemberSnapshot {
    fn from(m: &LeagueMember) -> Self {
        Self {
            member_id: m.member_id.clone(),
            identity_sha256: m.identity_sha256().expect("validated member serializes"),
            kind: m.kind,
            acceptance_receipt_sha256: m.acceptance_receipt_sha256.clone(),
            historical_cost: m.historical_cost.clone(),
        }
    }
}

/// One paired evaluation under the manifest's single fixed protocol, support, budget, and
/// evaluator. Costs are measured observations and are preserved alongside the payoff.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PayoffCell {
    pub left_member_id: String,
    pub right_member_id: String,
    pub context_id: String,
    pub protocol_sha256: String,
    pub support_sha256: String,
    pub budget_sha256: String,
    pub evaluator_sha256: String,
    pub left_payoff: f64,
    pub right_payoff: f64,
    pub left_cpu_seconds: f64,
    pub right_cpu_seconds: f64,
    pub evidence_sha256: String,
}

/// Fully covered, frozen, paired payoff table. A table binds its complete context roster and
/// exact member snapshots; partial rows, mixed protocols/support/budgets and identity rebinding
/// are refused. Qualification tables remain a separate purpose and cannot enter training memory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PayoffManifest {
    pub schema: String,
    pub purpose: PayoffPurpose,
    pub registry_sha256: String,
    pub protocol_sha256: String,
    pub support_sha256: String,
    pub task_distribution_sha256: String,
    pub budget_sha256: String,
    pub evaluator_sha256: String,
    pub contexts: Vec<ContextIdentity>,
    pub members: Vec<MemberSnapshot>,
    pub cells: Vec<PayoffCell>,
}
impl PayoffManifest {
    pub fn freeze(
        registry: &LeagueRegistry,
        purpose: PayoffPurpose,
        contexts: Vec<ContextIdentity>,
        member_ids: &[String],
        cells: Vec<PayoffCell>,
    ) -> Result<Self> {
        registry.validate_frozen_baseline()?;
        ensure!(
            !member_ids.is_empty(),
            "at least one payoff member required"
        );
        let mut members = Vec::with_capacity(member_ids.len());
        for id in member_ids {
            members.push(MemberSnapshot::from(
                registry
                    .member(id)
                    .ok_or_else(|| anyhow::anyhow!("unknown member {id}"))?,
            ));
        }
        let first = registry
            .member(&members[0].member_id)
            .expect("snapshot came from registry");
        let manifest = Self {
            schema: LEAGUE_SCHEMA.into(),
            purpose,
            registry_sha256: registry.identity_sha256()?,
            protocol_sha256: first.protocol_sha256.clone(),
            support_sha256: first.support_sha256.clone(),
            task_distribution_sha256: first.task_distribution_sha256.clone(),
            budget_sha256: cells
                .first()
                .map(|c| c.budget_sha256.clone())
                .unwrap_or_default(),
            evaluator_sha256: cells
                .first()
                .map(|c| c.evaluator_sha256.clone())
                .unwrap_or_default(),
            contexts,
            members,
            cells,
        };
        manifest.validate(registry)?;
        Ok(manifest)
    }

    pub fn validate(&self, registry: &LeagueRegistry) -> Result<()> {
        ensure!(
            self.schema == LEAGUE_SCHEMA,
            "unsupported league manifest schema"
        );
        registry.validate_frozen_baseline()?;
        ensure!(
            self.registry_sha256 == registry.identity_sha256()?,
            "registry identity changed after freeze"
        );
        for (name, value) in [
            ("protocol", &self.protocol_sha256),
            ("support", &self.support_sha256),
            ("task distribution", &self.task_distribution_sha256),
            ("budget", &self.budget_sha256),
            ("evaluator", &self.evaluator_sha256),
        ] {
            ensure!(is_sha256(value), "{name} SHA-256 required");
        }
        ensure!(
            !self.contexts.is_empty(),
            "complete nonempty context roster required"
        );
        let mut context_ids = BTreeSet::new();
        for context in &self.contexts {
            ensure!(
                !context.context_id.trim().is_empty() && !context.family_id.trim().is_empty(),
                "context and family IDs required"
            );
            ensure!(
                context_ids.insert(context.context_id.as_str()),
                "duplicate expected context"
            );
        }
        ensure!(
            self.members.len() >= 2,
            "a comparison table requires at least two members"
        );
        if self.purpose == PayoffPurpose::FrozenBaseline {
            ensure!(
                self.members
                    .iter()
                    .any(|member| member.kind == MemberKind::FrozenParent),
                "frozen baseline table must retain the frozen parent"
            );
            ensure!(
                self.members
                    .iter()
                    .any(|member| member.kind == MemberKind::SimpleControl),
                "frozen baseline table must retain a simple control"
            );
        }
        let mut members = BTreeMap::new();
        for snapshot in &self.members {
            let member = registry
                .member(&snapshot.member_id)
                .ok_or_else(|| anyhow::anyhow!("unknown member {}", snapshot.member_id))?;
            ensure!(
                snapshot.identity_sha256 == member.identity_sha256()?,
                "historical member identity changed"
            );
            ensure!(
                snapshot.kind == member.kind
                    && snapshot.acceptance_receipt_sha256 == member.acceptance_receipt_sha256
                    && snapshot.historical_cost == member.historical_cost,
                "historical member attributes/cost changed"
            );
            ensure!(
                member.protocol_sha256 == self.protocol_sha256
                    && member.support_sha256 == self.support_sha256
                    && member.task_distribution_sha256 == self.task_distribution_sha256,
                "incompatible member protocol/support/task distribution"
            );
            ensure!(
                members
                    .insert(snapshot.member_id.as_str(), snapshot)
                    .is_none(),
                "duplicate payoff member"
            );
        }
        let expected_cells = self
            .contexts
            .len()
            .checked_mul(
                members
                    .len()
                    .saturating_mul(members.len().saturating_sub(1))
                    / 2,
            )
            .ok_or_else(|| anyhow::anyhow!("comparison roster size overflow"))?;
        ensure!(
            self.cells.len() == expected_cells,
            "partial or over-complete payoff coverage refused"
        );
        let mut cell_keys = BTreeSet::new();
        for cell in &self.cells {
            ensure!(
                cell.left_member_id < cell.right_member_id,
                "payoff member pairs must be unique and canonically ordered"
            );
            ensure!(
                members.contains_key(cell.left_member_id.as_str())
                    && members.contains_key(cell.right_member_id.as_str()),
                "payoff cell references an unregistered member"
            );
            ensure!(
                context_ids.contains(cell.context_id.as_str()),
                "payoff cell context is outside the frozen roster"
            );
            ensure!(
                cell.protocol_sha256 == self.protocol_sha256
                    && cell.support_sha256 == self.support_sha256
                    && cell.budget_sha256 == self.budget_sha256
                    && cell.evaluator_sha256 == self.evaluator_sha256,
                "mixed protocol, support, budget, or evaluator in payoff table"
            );
            ensure!(
                cell.evidence_sha256.len() == 64 && is_sha256(&cell.evidence_sha256),
                "cell evidence digest required"
            );
            ensure!(
                cell.left_payoff.is_finite() && cell.right_payoff.is_finite(),
                "finite payoff values required"
            );
            ensure!(
                cell.left_cpu_seconds.is_finite()
                    && cell.left_cpu_seconds >= 0.0
                    && cell.right_cpu_seconds.is_finite()
                    && cell.right_cpu_seconds >= 0.0,
                "finite nonnegative measured costs required"
            );
            ensure!(
                cell_keys.insert((
                    cell.left_member_id.as_str(),
                    cell.right_member_id.as_str(),
                    cell.context_id.as_str()
                )),
                "duplicate payoff cell"
            );
        }
        Ok(())
    }

    pub fn identity_sha256(&self, registry: &LeagueRegistry) -> Result<String> {
        self.validate(registry)?;
        digest(self)
    }

    fn pair_mean_difference(&self, left: &str, right: &str) -> Option<f64> {
        let mut values = Vec::new();
        for cell in &self.cells {
            if cell.left_member_id == left && cell.right_member_id == right {
                values.push(cell.left_payoff - cell.right_payoff);
            } else if cell.left_member_id == right && cell.right_member_id == left {
                values.push(cell.right_payoff - cell.left_payoff);
            }
        }
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    }

    fn accepted_competitor_ids(&self) -> Vec<&str> {
        self.members
            .iter()
            .filter(|m| {
                m.kind == MemberKind::AcceptedCompetitor && m.acceptance_receipt_sha256.is_some()
            })
            .map(|m| m.member_id.as_str())
            .collect()
    }

    fn has_nontransitive_cycle(&self, ids: &[&str]) -> bool {
        for &a in ids {
            for &b in ids {
                if a == b || self.pair_mean_difference(a, b).unwrap_or(0.0) <= 0.0 {
                    continue;
                }
                for &c in ids {
                    if c == a || c == b {
                        continue;
                    }
                    if self.pair_mean_difference(b, c).unwrap_or(0.0) > 0.0
                        && self.pair_mean_difference(c, a).unwrap_or(0.0) > 0.0
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn has_family_specialization(&self, ids: &[&str]) -> bool {
        if ids.len() < 2 {
            return false;
        }
        let families: BTreeSet<&str> = self.contexts.iter().map(|c| c.family_id.as_str()).collect();
        for &a in ids {
            for &b in ids {
                if a == b {
                    continue;
                }
                let mut signs = BTreeSet::new();
                for family in &families {
                    let mut values = Vec::new();
                    for cell in &self.cells {
                        let context = self
                            .contexts
                            .iter()
                            .find(|c| c.context_id == cell.context_id);
                        if context.is_some_and(|c| c.family_id == *family) {
                            if cell.left_member_id == a && cell.right_member_id == b {
                                values.push(cell.left_payoff - cell.right_payoff);
                            } else if cell.left_member_id == b && cell.right_member_id == a {
                                values.push(cell.right_payoff - cell.left_payoff);
                            }
                        }
                    }
                    if !values.is_empty() {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        if mean > 0.0 {
                            signs.insert(1i8);
                        }
                        if mean < 0.0 {
                            signs.insert(-1i8);
                        }
                    }
                }
                if signs.len() == 2 {
                    return true;
                }
            }
        }
        false
    }
}

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
pub struct PFSP;
impl PFSP {
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

fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
