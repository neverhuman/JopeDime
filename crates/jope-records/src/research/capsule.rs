use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvidenceLocator {
    pub uri: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceValue {
    pub locator: EvidenceLocator,
    pub unit: String,
    pub numerator: u64,
    pub denominator: u64,
    pub estimate: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub status: String,
    pub estimator: String,
    pub protocol: String,
    pub disclosed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Synthetic,
    Shadow,
    Adaptive,
    Confirmatory,
}

impl Stage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Shadow => "shadow",
            Self::Adaptive => "adaptive",
            Self::Confirmatory => "confirmatory",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoundInput {
    pub round_id: String,
    pub source_commit: Option<String>,
    pub stage: Stage,
    pub outcome: String,
    pub requested: u64,
    pub attempted: u64,
    pub computed: u64,
    pub applied: u64,
    pub committed: u64,
    pub exported: u64,
    pub reloaded: u64,
    pub next_actor: u64,
    pub facts: Vec<String>,
    pub inference: Vec<String>,
    pub limitations: Vec<String>,
    pub evidence: Vec<EvidenceLocator>,
    pub disclosure_granted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResearchCapsuleV2 {
    pub round_id: String,
    pub source_commit: Option<String>,
    pub stage: Stage,
    pub outcome: String,
    pub denominator: BTreeMap<String, u64>,
    pub facts: Vec<String>,
    pub inference: Vec<String>,
    pub limitations: Vec<String>,
    pub evidence: Vec<EvidenceValue>,
    pub assertion_supported: bool,
}
