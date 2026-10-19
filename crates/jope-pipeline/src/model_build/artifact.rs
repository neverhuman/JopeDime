//! Persistence and reload verification for the winner ensemble (SC-DEC-184 JA-04).
//!
//! The persisted record is canonical JSON. Its identity is the sha256 of those bytes. Reload
//! verification recomputes the ensemble from the member bundles and requires an identical record.
//! This proves the weights are a deterministic function of the OOF predictions. It does not prove
//! a fitted model can be refitted or reloaded. That needs a real fitter and is not in this file.
use super::ensemble::{build_winner_ensemble, WinnerEnsemble};
use super::oof::AdmittedBundle;
use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};

/// Canonical bytes of the ensemble record.
pub fn encode(ensemble: &WinnerEnsemble) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(ensemble)?)
}

/// Identity of the persisted record.
pub fn identity(ensemble: &WinnerEnsemble) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(encode(ensemble)?)))
}

/// Decodes persisted bytes, refusing unknown fields and any non-canonical encoding.
pub fn decode(bytes: &[u8]) -> Result<WinnerEnsemble> {
    let ensemble: WinnerEnsemble = serde_json::from_slice(bytes)?;
    ensure!(
        encode(&ensemble)? == bytes,
        "persisted winner ensemble is not in canonical encoding"
    );
    Ok(ensemble)
}

/// Recomputes the ensemble from the member bundles and requires an identical record.
pub fn verify_reload(
    persisted: &WinnerEnsemble,
    members: &[AdmittedBundle],
    labels: &[String],
) -> Result<()> {
    let recomputed = build_winner_ensemble(members, labels)?;
    ensure!(
        &recomputed == persisted,
        "reloaded winner ensemble differs from its recomputation"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation_score::Predictions;
    use crate::model_build::oof::{admit_for_test, OofPredictionBundle, OOF_SCHEMA};

    fn admitted(members: &[OofPredictionBundle]) -> Vec<AdmittedBundle> {
        members
            .iter()
            .cloned()
            .map(|m| admit_for_test(m).unwrap())
            .collect()
    }
    use jope_datasets::catalog::Task;

    fn members() -> (Vec<OofPredictionBundle>, Vec<String>) {
        let labels: Vec<String> = (0..12).map(|i| (i as f64 * 0.5).to_string()).collect();
        let truth: Vec<f64> = labels.iter().map(|l| l.parse().unwrap()).collect();
        let make = |id: &str, values: Vec<f64>| OofPredictionBundle {
            schema: OOF_SCHEMA.into(),
            task: Task::Regression,
            candidate_id: id.into(),
            request_sha256: "a".repeat(64),
            fitted_artifact_sha256: vec!["b".repeat(64); 4],
            row_ids: (0..12).map(|i| format!("r{i}")).collect(),
            folds: (0..12).map(|i| (i % 4) as u32).collect(),
            group_ids: vec![],
            classes: vec![],
            predictions: Predictions::Values(values),
            failed_folds: vec![],
        };
        (
            vec![
                make("rust.a", truth.iter().map(|t| t + 0.4).collect()),
                make("rust.b", truth.iter().map(|t| t - 0.4).collect()),
            ],
            labels,
        )
    }

    #[test]
    fn a_persisted_record_round_trips_byte_for_byte_and_keeps_its_identity() {
        let (members, labels) = members();
        let ensemble = build_winner_ensemble(&admitted(&members), &labels).unwrap();
        let bytes = encode(&ensemble).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back, ensemble);
        assert_eq!(identity(&back).unwrap(), identity(&ensemble).unwrap());
    }

    #[test]
    fn non_canonical_or_unknown_persisted_bytes_are_refused() {
        let (members, labels) = members();
        let ensemble = build_winner_ensemble(&admitted(&members), &labels).unwrap();
        let mut pretty = serde_json::to_vec_pretty(&ensemble).unwrap();
        assert!(decode(&pretty).is_err());
        pretty.clear();
        let mut value: serde_json::Value = serde_json::to_value(&ensemble).unwrap();
        value["extra"] = serde_json::json!(1);
        assert!(decode(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn reload_verification_recomputes_the_same_record_and_refuses_a_tampered_one() {
        let (members, labels) = members();
        let ensemble = build_winner_ensemble(&admitted(&members), &labels).unwrap();
        verify_reload(&ensemble, &admitted(&members), &labels).unwrap();
        let mut tampered = ensemble.clone();
        tampered.report_loss += 1.0;
        assert!(verify_reload(&tampered, &admitted(&members), &labels).is_err());
    }
}
