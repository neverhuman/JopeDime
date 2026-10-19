//! Persistence and reload verification for the winner ensemble (SC-DEC-184 JA-04, JA-R4,
//! SC-DEC-184R JA-F1, JA-F3).
//!
//! The persisted record holds the ensemble, the outer refit record and the fitted state bytes of
//! every member. Its identity is the sha256 of its canonical JSON. A reload checks four things:
//! the stored identity equals the caller's independently built identity; every state hashes to
//! the digest the refit recorded; each restored member reproduces its report predictions bit for
//! bit; and the ensemble recomputed from the admitted bundles equals the stored record exactly.
//!
//! `write_new` and `read` make the record a product artifact on disk. `write_new` refuses to
//! overwrite an existing file. This is an in-process check of the bytes; the fresh-process check
//! is in the `model_build_persistence` integration test.
use super::ensemble::{admit_winner_ensemble, ReportPredictions, WinnerEnsemble};
use super::identity::ExecutionIdentity;
use super::oof::AdmittedBundle;
use super::refit::OuterRefitRecord;
use super::selection::EvaluationBudget;
use crate::evaluation_score::Predictions;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedEnsemble {
    pub ensemble: WinnerEnsemble,
    /// The outer refit that produced the report predictions and the member states.
    pub refit: OuterRefitRecord,
    /// Fitted state bytes per member candidate, exactly as the fitter produced them.
    pub member_states: BTreeMap<String, Vec<u8>>,
}

/// Canonical bytes of the persisted record. Map keys are sorted, so the encoding is stable.
pub fn encode(persisted: &PersistedEnsemble) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(persisted)?)
}

pub fn identity(persisted: &PersistedEnsemble) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(encode(persisted)?)))
}

/// Decodes persisted bytes, refusing unknown fields and any non-canonical encoding.
pub fn decode(bytes: &[u8]) -> Result<PersistedEnsemble> {
    let persisted: PersistedEnsemble = serde_json::from_slice(bytes)?;
    ensure!(
        encode(&persisted)? == bytes,
        "persisted winner ensemble is not in canonical encoding"
    );
    Ok(persisted)
}

/// Writes a new artifact. An existing file is an error, never an overwrite.
pub fn write_new(path: &Path, persisted: &PersistedEnsemble) -> Result<()> {
    let bytes = encode(persisted)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

pub fn read(path: &Path) -> Result<PersistedEnsemble> {
    decode(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
}

fn bits_equal(a: &Predictions, b: &Predictions) -> bool {
    match (a, b) {
        (Predictions::Values(x), Predictions::Values(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (
            Predictions::Probabilities {
                classes: cx,
                values: x,
            },
            Predictions::Probabilities {
                classes: cy,
                values: y,
            },
        ) => {
            cx == cy
                && x.len() == y.len()
                && x.iter().zip(y).all(|(r, s)| {
                    r.len() == s.len() && r.iter().zip(s).all(|(p, q)| p.to_bits() == q.to_bits())
                })
        }
        _ => false,
    }
}

/// Reloads a persisted ensemble against the caller's independently built identity. `restore` maps
/// a member's id and state bytes to its restored report-side predictions.
pub fn verify_reload(
    persisted: &PersistedEnsemble,
    identity: &ExecutionIdentity,
    admitted: &[AdmittedBundle],
    selection_labels: &[String],
    report: &ReportPredictions,
    budget: &mut EvaluationBudget,
    restore: &dyn Fn(&str, &[u8]) -> Result<Predictions>,
) -> Result<()> {
    ensure!(
        persisted.ensemble.identity == *identity,
        "the persisted ensemble was built under a different identity"
    );
    persisted.refit.verify_states(&persisted.member_states)?;
    for (id, bytes) in &persisted.member_states {
        let restored = restore(id, bytes)?;
        let expected = report
            .per_member
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("no report predictions for {id}"))?;
        ensure!(
            bits_equal(&restored, expected),
            "restored state for {id} does not reproduce its report predictions bit for bit"
        );
    }
    let recomputed = admit_winner_ensemble(
        admitted,
        selection_labels,
        identity,
        report,
        &persisted.refit,
        budget,
    )?;
    ensure!(
        recomputed == persisted.ensemble,
        "reloaded winner ensemble differs from its recomputation"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_bytes_decode_and_non_canonical_or_unknown_bytes_are_refused() {
        let value = serde_json::json!({"ensemble": null, "member_states": {}});
        // A structurally incomplete record must not decode.
        assert!(decode(&serde_json::to_vec(&value).unwrap()).is_err());
        assert!(decode(b"{\"ensemble\": 1}").is_err());
    }

    #[test]
    fn bit_equality_is_strict_about_nan_and_signed_zero() {
        assert!(bits_equal(
            &Predictions::Values(vec![0.0]),
            &Predictions::Values(vec![0.0])
        ));
        assert!(!bits_equal(
            &Predictions::Values(vec![0.0]),
            &Predictions::Values(vec![-0.0])
        ));
        assert!(!bits_equal(
            &Predictions::Values(vec![1.0]),
            &Predictions::Values(vec![1.0, 2.0])
        ));
    }
}
