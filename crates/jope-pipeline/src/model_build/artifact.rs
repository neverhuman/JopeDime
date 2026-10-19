//! Persistence and reload verification for the winner ensemble (SC-DEC-184 JA-04, JA-R4).
//!
//! The persisted record holds the ensemble and the fitted state bytes of every member. Its
//! identity is the sha256 of its canonical JSON. A reload restores each member's state through a
//! caller-supplied function (in production, the native fitter's restore), checks that the restored
//! model reproduces the report predictions bit for bit, and recomputes the ensemble from the
//! admitted bundles, which must equal the stored record exactly.
//!
//! This is an in-process reload of persisted bytes. It is not a fresh-process reload, and it does
//! not claim a registry publication.
use super::ensemble::{build_winner_ensemble, ReportPredictions, WinnerEnsemble};
use super::oof::AdmittedBundle;
use super::selection::EvaluationBudget;
use crate::evaluation_score::Predictions;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedEnsemble {
    pub ensemble: WinnerEnsemble,
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

/// Reloads a persisted ensemble. `restore` maps a member's id and state bytes to its restored
/// report-side predictions.
pub fn verify_reload(
    persisted: &PersistedEnsemble,
    admitted: &[AdmittedBundle],
    selection_labels: &[String],
    report: &ReportPredictions,
    budget: &mut EvaluationBudget,
    restore: &dyn Fn(&str, &[u8]) -> Result<Predictions>,
) -> Result<()> {
    let member_ids: Vec<&str> = admitted
        .iter()
        .map(|m| m.bundle().candidate_id.as_str())
        .collect();
    ensure!(
        persisted
            .member_states
            .keys()
            .map(String::as_str)
            .eq(member_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()),
        "persisted states do not cover exactly the admitted members"
    );
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
    let recomputed = build_winner_ensemble(
        admitted,
        selection_labels,
        &persisted.ensemble.identity,
        report,
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
