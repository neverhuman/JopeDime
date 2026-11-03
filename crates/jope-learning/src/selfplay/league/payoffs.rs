use super::*;
use anyhow::ensure;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

    pub(crate) fn accepted_competitor_ids(&self) -> Vec<&str> {
        self.members
            .iter()
            .filter(|m| {
                m.kind == MemberKind::AcceptedCompetitor && m.acceptance_receipt_sha256.is_some()
            })
            .map(|m| m.member_id.as_str())
            .collect()
    }

    pub(crate) fn has_nontransitive_cycle(&self, ids: &[&str]) -> bool {
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

    pub(crate) fn has_family_specialization(&self, ids: &[&str]) -> bool {
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
