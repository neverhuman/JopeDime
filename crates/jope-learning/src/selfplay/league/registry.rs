use super::*;
use anyhow::{bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
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
