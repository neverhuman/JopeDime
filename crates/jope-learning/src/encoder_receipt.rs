//! Encoder/admission binding. Teaching admission (`eligibility::qualify` and every
//! revalidating consumer) derives the scope and encoder pins from the preoptimization
//! freeze and requires [`check_teaching_binding`] before any teaching decision.
//!
//! Matching these bindings proves neither absence of undeclared reads nor measured
//! teaching eligibility, and a receipt is a producer claim: nothing here executes or
//! authenticates the encoder run itself.

use crate::canon::{canonical_sha256, is_hex64, sha256_hex};
use crate::record::HwmIdentity;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for  values.
pub const SCHEMA: &str = "jope.encoder_receipt/v1";

/// Semantic identity, not just vector width. All spike outputs are LE f32 vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Product {
    /// The `RowEmbedding128` case of `Product`.
    RowEmbedding128,
    /// The `DatasetSignature512` case of `Product`.
    DatasetSignature512,
    /// The `TargetEmbedding512` case of `Product`.
    TargetEmbedding512,
    /// The `TeachingContext2048` case of `Product`.
    TeachingContext2048,
}
impl Product {
    fn width(self) -> usize {
        match self {
            Self::RowEmbedding128 => 128,
            Self::DatasetSignature512 | Self::TargetEmbedding512 => 512,
            Self::TeachingContext2048 => 2048,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Member` is a record for the member contract in this module: Encoder/admission binding. Teaching admission (eligibility::qualify and every revalidating consumer) derives the scope and encoder pins from the preoptimization freeze and requires checkteachingbinding before any.
pub struct Member {
    /// Row recorded for this member.
    pub row: String,
    /// Group recorded for this member.
    pub group: String,
}

/// Ordered fitting membership and explicit forbidden membership from a trusted split.
/// Full comparison identity binds task, dataset, cut, vocabulary and evaluator.
/// `outer_fold: None` scopes the whole development partition (no outer hold).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    /// Comparison recorded for this scope.
    pub comparison: HwmIdentity,
    /// Source family recorded for this scope.
    pub source_family: String,
    /// Optional outer fold value; `None` means it is not available.
    pub outer_fold: Option<u8>,
    /// Optional inner fold value; `None` means it is not available.
    pub inner_fold: Option<u8>,
    /// Collection of fitting values carried by this scope.
    pub fitting: Vec<Member>,
    /// Collection of forbidden values carried by this scope.
    pub forbidden: Vec<Member>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Encoder` is a record for the encoder contract in this module: Encoder/admission binding. Teaching admission (eligibility::qualify and every revalidating consumer) derives the scope and encoder pins from the preoptimization freeze and requires checkteachingbinding before any.
pub struct Encoder {
    /// SHA-256 digest identifying the associated artifact.
    pub artifact_sha256: String,
    /// SHA-256 digest identifying the associated configuration.
    pub configuration_sha256: String,
    /// Seed recorded for this encoder.
    pub seed: u64,
    /// Product recorded for this encoder.
    pub product: Product,
}

/// Digests of *projected* inputs, not the whole dataset containing forbidden rows.
/// Parents are ordered, externally verified lineage pins; empty means a root encoder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    /// SHA-256 digest identifying the associated features.
    pub features_sha256: String,
    /// SHA-256 digest identifying the associated labels.
    pub labels_sha256: Option<String>,
    /// Collection of parent receipts values carried by this inputs.
    pub parent_receipts: Vec<String>,
}

/// In-process capability. Receipts cannot deserialize/mint it. `bind` is an issuer
/// trust boundary, not an authorization verifier; do not feed it an untrusted receipt.
///
/// ```compile_fail
/// let _: jope_learning::encoder_receipt::EncoderCapability =
///     serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Serialize)]
pub struct EncoderCapability {
    scope: Scope,
    encoder: Encoder,
    inputs: Inputs,
}
impl EncoderCapability {
    /// Performs the bind operation and returns an error if it cannot be completed.
    pub fn bind(scope: Scope, encoder: Encoder, inputs: Inputs) -> Result<Self> {
        ensure!(
            !scope.source_family.trim().is_empty()
                && !scope.comparison.task.trim().is_empty()
                && !scope.comparison.protocol.trim().is_empty()
                && scope.outer_fold.is_none_or(|f| f < 5)
                && scope
                    .inner_fold
                    .is_none_or(|f| f < 5 && scope.outer_fold.is_some()),
            "incomplete encoder scope"
        );
        let comparison = scope
            .comparison
            .comparison
            .as_ref()
            .context("missing teaching comparison context")?;
        ensure!(
            matches!(
                scope.comparison.task.as_str(),
                "regression" | "binary" | "multiclass"
            ) && comparison.stage == "train_oof"
                && comparison.label_role == "development"
                && !comparison.sealed
                && !comparison.benchmark_valid
                && !comparison.numerical_policy.trim().is_empty()
                && !comparison.recipe_encoding.trim().is_empty()
                && comparison
                    .checkpoint_sha256
                    .as_ref()
                    .is_none_or(|s| is_hex64(s)),
            "unsupported teaching comparison context"
        );
        for sha in [
            &scope.comparison.dataset_hash,
            &scope.comparison.cut_id,
            &scope.comparison.evaluator_sha256,
            &scope.comparison.vocabulary_sha256,
            &encoder.artifact_sha256,
            &encoder.configuration_sha256,
            &inputs.features_sha256,
        ] {
            ensure!(is_hex64(sha), "unbound encoder provenance");
        }
        ensure!(
            inputs.labels_sha256.as_ref().is_none_or(|s| is_hex64(s))
                && inputs.parent_receipts.iter().all(|s| is_hex64(s)),
            "invalid input provenance"
        );
        ensure!(
            inputs.parent_receipts.iter().collect::<BTreeSet<_>>().len()
                == inputs.parent_receipts.len(),
            "duplicate parent receipt"
        );
        let fitting = members(&scope.fitting)?;
        let forbidden = members(&scope.forbidden)?;
        ensure!(!fitting.is_empty(), "empty fitting scope");
        ensure!(fitting.is_disjoint(&forbidden), "forbidden fitting row");
        let fit_groups: BTreeSet<_> = scope.fitting.iter().map(|m| &m.group).collect();
        ensure!(
            scope
                .forbidden
                .iter()
                .all(|m| !fit_groups.contains(&m.group)),
            "group crosses fitting boundary"
        );
        Ok(Self {
            scope,
            encoder,
            inputs,
        })
    }

    /// Ordered projection for a reference encoder. No forbidden payload is looked up.
    /// Authentication of row IDs, group mapping and payload bytes is the issuer's job.
    pub fn project_fitting<'a, T>(&self, rows: &'a BTreeMap<String, T>) -> Result<Vec<&'a T>> {
        self.scope
            .fitting
            .iter()
            .map(|m| {
                rows.get(&m.row)
                    .with_context(|| format!("missing fitting row {}", m.row))
            })
            .collect()
    }

    /// Construct a receipt from retained state and one typed output vector.
    /// This records claims; it does not certify how an external encoder ran.
    pub fn receipt(&self, fitted_state: &[u8], output: &[u8]) -> Result<EncoderReceipt> {
        ensure!(!fitted_state.is_empty(), "missing fitted transform");
        ensure!(
            output.len() == self.encoder.product.width() * 4,
            "wrong output ABI"
        );
        ensure!(
            output
                .chunks_exact(4)
                .all(|b| { f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite() }),
            "nonfinite encoder output"
        );
        let rows: Vec<_> = self.scope.fitting.iter().map(|m| &m.row).collect();
        let label_rows = if self.inputs.labels_sha256.is_some() {
            rows.clone()
        } else {
            vec![]
        };
        Ok(EncoderReceipt {
            schema: SCHEMA.into(),
            capability_sha256: canonical_sha256(self)?,
            scope_sha256: canonical_sha256(&self.scope)?,
            comparison_sha256: self.scope.comparison.sha256()?,
            ordered_fitting_rows_sha256: canonical_sha256(&rows)?,
            permitted_label_rows_sha256: canonical_sha256(&label_rows)?,
            encoder: self.encoder.clone(),
            inputs: self.inputs.clone(),
            fitted_state_sha256: sha256_hex(fitted_state),
            output_sha256: sha256_hex(output),
        })
    }
}

fn members(rows: &[Member]) -> Result<BTreeSet<&str>> {
    let mut ids = BTreeSet::new();
    for m in rows {
        ensure!(
            !m.row.is_empty() && !m.group.is_empty(),
            "empty row/group identity"
        );
        ensure!(ids.insert(m.row.as_str()), "duplicate scope row");
    }
    Ok(ids)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `EncoderReceipt` is a record for the encoder receipt contract in this module: Encoder/admission binding. Teaching admission (eligibility::qualify and every revalidating consumer) derives the scope and encoder pins from the preoptimization freeze and requires checkteachingbinding before any.
pub struct EncoderReceipt {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated capability.
    pub capability_sha256: String,
    /// SHA-256 digest identifying the associated scope.
    pub scope_sha256: String,
    /// SHA-256 digest identifying the associated comparison.
    pub comparison_sha256: String,
    /// SHA-256 digest identifying the associated ordered fitting rows.
    pub ordered_fitting_rows_sha256: String,
    /// SHA-256 digest identifying the associated permitted label rows.
    pub permitted_label_rows_sha256: String,
    /// Encoder recorded for this encoder receipt.
    pub encoder: Encoder,
    /// Inputs recorded for this encoder receipt.
    pub inputs: Inputs,
    /// SHA-256 digest identifying the associated fitted state.
    pub fitted_state_sha256: String,
    /// SHA-256 digest identifying the associated output.
    pub output_sha256: String,
}

/// Pins expected by the teaching consumer, obtained independently of the receipt.
/// Program binding belongs here: an encoder may be shared by several candidates.
#[derive(Clone, Debug)]
pub struct TeachingBinding {
    /// SHA-256 digest identifying the associated comparison.
    pub comparison_sha256: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// SHA-256 digest identifying the associated encoder receipt.
    pub encoder_receipt_sha256: String,
    /// Product recorded for this teaching binding.
    pub product: Product,
}

/// Structural/provenance equality only. Success MUST NOT be called teaching admission.
/// A production adapter must additionally revalidate measured qualification, issuer
/// authority, complete parent lineage and actual encoder input isolation.
pub fn check_teaching_binding(
    capability: &EncoderCapability,
    receipt: Option<&EncoderReceipt>,
    binding: &TeachingBinding,
    candidate_program_sha256: &str,
    fitted_state: &[u8],
    output: &[u8],
) -> Result<()> {
    let receipt = receipt.context("MissingEncoderProvenance")?;
    ensure!(
        is_hex64(&binding.program_sha256)
            && binding.program_sha256 == candidate_program_sha256
            && binding.comparison_sha256 == capability.scope.comparison.sha256()?
            && binding.product == capability.encoder.product,
        "EncoderScopeMismatch: teaching subject"
    );
    ensure!(
        *receipt == capability.receipt(fitted_state, output)?
            && canonical_sha256(receipt)? == binding.encoder_receipt_sha256,
        "EncoderScopeMismatch: receipt, input closure, state or output"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
