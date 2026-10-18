use super::{EvidenceLocator, EvidenceValue, ResearchCapsuleV2, RoundInput};
use std::collections::BTreeMap;

pub trait EvidenceResolver {
    type Error: std::fmt::Display;
    fn is_revoked(&self, locator: &EvidenceLocator) -> Result<bool, Self::Error>;
    fn resolve(&self, locator: &EvidenceLocator) -> Result<Option<EvidenceValue>, Self::Error>;
}

#[derive(Debug, PartialEq, Eq)]
pub enum CompileError {
    Resolver(String),
    Invalid(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolver(e) => write!(f, "evidence resolver: {e}"),
            Self::Invalid(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for CompileError {}

pub fn compile<R: EvidenceResolver>(
    input: &RoundInput,
    resolver: &R,
) -> Result<ResearchCapsuleV2, CompileError> {
    if input.round_id.trim().is_empty() {
        return Err(CompileError::Invalid("round_id is required".into()));
    }
    if !input.disclosure_granted {
        return Ok(ResearchCapsuleV2 {
            round_id: "withheld".into(),
            source_commit: None,
            stage: input.stage,
            outcome: "disclosure_withheld".into(),
            denominator: BTreeMap::new(),
            facts: Vec::new(),
            inference: Vec::new(),
            limitations: vec!["disclosure authorization is absent".into()],
            evidence: Vec::new(),
            assertion_supported: false,
        });
    }
    let mut limitations = input.limitations.clone();
    let mut values = Vec::new();
    let mut supported = true;
    let mut locators = input.evidence.clone();
    locators.sort();
    locators.dedup();
    for loc in locators {
        if loc.uri.trim().is_empty()
            || loc.version.trim().is_empty()
            || loc.sha256.len() != 64
            || !loc
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            limitations.push("evidence locator is not a canonical SHA-256 identity".into());
            supported = false;
            continue;
        }
        let revoked_before = resolver
            .is_revoked(&loc)
            .map_err(|e| CompileError::Resolver(e.to_string()))?;
        let value = resolver
            .resolve(&loc)
            .map_err(|e| CompileError::Resolver(e.to_string()))?;
        let revoked_after = resolver
            .is_revoked(&loc)
            .map_err(|e| CompileError::Resolver(e.to_string()))?;
        if revoked_before || revoked_after {
            limitations.push(format!("evidence {} is revoked", loc.uri));
            supported = false;
            continue;
        }
        match value {
            None => {
                limitations.push(format!("evidence {} is missing", loc.uri));
                supported = false;
            }
            Some(v) if v.locator != loc || !v.disclosed => {
                limitations.push(format!("evidence {} is mismatched or undisclosed", loc.uri));
                supported = false;
            }
            Some(v)
                if v.denominator == 0
                    || ![v.estimate, v.ci_low, v.ci_high]
                        .into_iter()
                        .flatten()
                        .all(f64::is_finite) =>
            {
                limitations.push(format!(
                    "evidence {} has invalid denominator or estimate",
                    loc.uri
                ));
                supported = false;
            }
            Some(v) => {
                if matches!((v.ci_low, v.ci_high), (Some(lo), Some(hi)) if lo <= 0.0 && hi >= 0.0) {
                    limitations.push(format!(
                        "evidence {} confidence interval includes zero; outcome is inconclusive",
                        loc.uri
                    ));
                    supported = false;
                }
                values.push(v);
            }
        }
    }
    values.sort_by(|a, b| a.locator.cmp(&b.locator));
    let denominator = BTreeMap::from([
        ("requested".into(), input.requested),
        ("attempted".into(), input.attempted),
        ("computed".into(), input.computed),
        ("applied".into(), input.applied),
        ("committed".into(), input.committed),
        ("exported".into(), input.exported),
        ("reloaded".into(), input.reloaded),
        ("next_actor".into(), input.next_actor),
    ]);
    limitations.sort();
    limitations.dedup();
    let mut facts = input.facts.clone();
    facts.sort();
    facts.dedup();
    // Inference is retained verbatim as an explicitly labeled hypothesis; no stage promotion occurs.
    let mut inference = input.inference.clone();
    inference.sort();
    inference.dedup();
    Ok(ResearchCapsuleV2 {
        round_id: input.round_id.clone(),
        source_commit: input.source_commit.clone(),
        stage: input.stage,
        outcome: input.outcome.clone(),
        denominator,
        facts,
        inference,
        limitations,
        evidence: values,
        assertion_supported: supported,
    })
}
