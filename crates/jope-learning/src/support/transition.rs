//! Versioned extension of frozen CLOVER support.
//!
//! Authentication is delegated to the I04 issuer. A role string or receipt digest by itself
//! is not accepted as TRAIN authority; production wiring must provide an issuer-backed verifier.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{Candidate, FrozenSupport};
use crate::canon::{canonical_sha256, is_hex64};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `TrainEvidence` is a record for the train evidence contract in this module: Versioned extension of frozen CLOVER support.
pub struct TrainEvidence {
    /// Identifier for the associated context.
    pub context_id: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// SHA-256 digest identifying the associated source receipt.
    pub source_receipt_sha256: String,
    /// SHA-256 digest identifying the associated parity receipt.
    pub parity_receipt_sha256: String,
    /// SHA-256 digest identifying the associated execution receipt.
    pub execution_receipt_sha256: String,
}

/// Implemented by the I04 authenticated TRAIN capability issuer at integration time.
pub trait AdmissionVerifier {
    /// Performs the verify train evidence operation and returns an error if it cannot be completed.
    fn verify_train_evidence(&self, evidence: &TrainEvidence) -> Result<bool>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `SupportTransition` is a record for the support transition contract in this module: Versioned extension of frozen CLOVER support.
pub struct SupportTransition {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated from support.
    pub from_support_sha256: String,
    /// SHA-256 digest identifying the associated to support.
    pub to_support_sha256: String,
    /// From version recorded for this support transition.
    pub from_version: u32,
    /// To version recorded for this support transition.
    pub to_version: u32,
    /// SHA-256 digest identifying the associated admission evidence.
    pub admission_evidence_sha256: String,
    /// SHA-256 digest identifying the associated admitted program.
    pub admitted_program_sha256: String,
    /// SHA-256 digest identifying the associated transition.
    pub transition_sha256: String,
}

impl SupportTransition {
    /// Verifies the bound evidence and returns an error when verification fails.
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.support-transition/v1",
            "support transition schema mismatch"
        );
        ensure!(
            is_hex64(&self.from_support_sha256)
                && is_hex64(&self.to_support_sha256)
                && is_hex64(&self.admission_evidence_sha256)
                && is_hex64(&self.admitted_program_sha256)
                && is_hex64(&self.transition_sha256)
                && self.to_version == self.from_version.checked_add(1).unwrap_or(0),
            "support transition binding/version mismatch"
        );
        let mut content = self.clone();
        let expected = content.transition_sha256.clone();
        content.transition_sha256.clear();
        ensure!(
            canonical_sha256(&content)? == expected,
            "support transition digest mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
/// `AdmissionDecision` is an enum for the admission decision contract in this module: Versioned extension of frozen CLOVER support.
pub enum AdmissionDecision {
    /// The `Admitted` case of `AdmissionDecision`.
    Admitted {
        /// Transition recorded for this outcome.
        transition: SupportTransition,
    },
    /// The `Refused` case of `AdmissionDecision`.
    Refused {
        /// Refusal recorded for this outcome.
        refusal: AdmissionRefusal,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `AdmissionRefusal` is a record for the admission refusal contract in this module: Versioned extension of frozen CLOVER support.
pub struct AdmissionRefusal {
    /// SHA-256 digest identifying the associated observation.
    pub observation_sha256: String,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

fn refused(candidate: &Candidate, reason: &str) -> Result<AdmissionDecision> {
    Ok(AdmissionDecision::Refused {
        refusal: AdmissionRefusal {
            observation_sha256: canonical_sha256(candidate)?,
            reason: reason.to_owned(),
        },
    })
}

/// Add exactly one canonical, bounded leaf after the I04 verifier authenticates TRAIN scope.
/// The support's existing role binding must also mark the context TRAIN. Refusals retain a
/// content digest for audit and never mutate the parent support.
pub fn admit_novel_leaf<V: AdmissionVerifier>(
    parent: &FrozenSupport,
    candidate: Candidate,
    evidence: &TrainEvidence,
    verifier: &V,
) -> Result<(AdmissionDecision, Option<FrozenSupport>)> {
    parent.verify()?;
    ensure!(
        is_hex64(&evidence.context_id),
        "bound TRAIN context required"
    );
    ensure!(
        evidence.source_receipt_sha256 == candidate.source_receipt_sha256
            && evidence.context_id == candidate.context_id
            && evidence.program_sha256 == candidate.program.program_sha256,
        "TRAIN evidence does not bind candidate"
    );
    if parent.role_map.get(&candidate.context_id) != Some(&super::Role::Train) {
        return Ok((
            refused(&candidate, "context_is_not_in_frozen_train_role")?,
            None,
        ));
    }
    if ![
        &evidence.source_receipt_sha256,
        &evidence.parity_receipt_sha256,
        &evidence.execution_receipt_sha256,
    ]
    .into_iter()
    .all(|pin| is_hex64(pin))
        || candidate.parity_receipt_sha256.as_deref() != Some(&evidence.parity_receipt_sha256)
    {
        return Ok((
            refused(&candidate, "missing_or_mismatched_execution_parity_proof")?,
            None,
        ));
    }
    if !verifier.verify_train_evidence(evidence)? {
        return Ok((refused(&candidate, "I04_train_capability_refused")?, None));
    }
    if parent.contains_program(&candidate.program) {
        return Ok((refused(&candidate, "program_already_in_support")?, None));
    }
    let representable = parent
        .vocabulary
        .program(&candidate.program.dsl)
        .is_ok_and(|canonical| canonical == candidate.program);
    if !representable {
        return Ok((
            refused(&candidate, "program_is_unexecutable_or_overlength")?,
            None,
        ));
    }
    let next = match parent.extend(vec![candidate.clone()]) {
        Ok(next) => next,
        Err(_) => {
            return Ok((
                refused(&candidate, "candidate_failed_support_validation")?,
                None,
            ))
        }
    };
    let mut transition = SupportTransition {
        schema: "jope.support-transition/v1".into(),
        from_support_sha256: parent.support_sha256.clone(),
        to_support_sha256: next.support_sha256.clone(),
        from_version: parent.version,
        to_version: next.version,
        admission_evidence_sha256: canonical_sha256(evidence)?,
        admitted_program_sha256: candidate.program.program_sha256,
        transition_sha256: String::new(),
    };
    transition.transition_sha256 = canonical_sha256(&transition)?;
    transition.verify()?;
    Ok((AdmissionDecision::Admitted { transition }, Some(next)))
}
