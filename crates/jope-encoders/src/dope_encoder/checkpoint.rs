//! A checkpoint pin is an integrity/lineage binding, not a quality or training-admission proof.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{DOPE512_RECIPE, DOPE_WEIGHTS_SHA256};

pub const CANDIDATE512_RECIPE: &str =
    "dope-mlcv3-unqualified-candidate-high-f32-descriptor256-task128-dist128/v1";

/// Select once when loading the resident encoder; every mask in that session uses this pin.
/// Caller-supplied candidate pins remain unqualified: a receipt digest alone authenticates neither
/// training nor ancestry. A trained product identity requires a separately reviewed registration.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "lineage", rename_all = "kebab-case", deny_unknown_fields)]
pub enum DopeCheckpointIdentity {
    #[default]
    ReleasedV1,
    UnqualifiedCandidateV1 {
        weights_sha256: String,
        training_receipt_sha256: String,
    },
}

impl DopeCheckpointIdentity {
    pub fn unqualified_candidate(
        weights_sha256: String,
        training_receipt_sha256: String,
    ) -> Result<Self> {
        let identity = Self::UnqualifiedCandidateV1 {
            weights_sha256,
            training_receipt_sha256,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<()> {
        if let Self::UnqualifiedCandidateV1 {
            weights_sha256,
            training_receipt_sha256,
        } = self
        {
            for digest in [weights_sha256, training_receipt_sha256] {
                ensure!(
                    digest.len() == 64
                        && digest
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "invalid checkpoint/training receipt digest"
                );
            }
            ensure!(
                weights_sha256 != DOPE_WEIGHTS_SHA256,
                "released baseline cannot be relabeled as a candidate checkpoint"
            );
        }
        Ok(())
    }

    pub fn weights_sha256(&self) -> &str {
        match self {
            Self::ReleasedV1 => DOPE_WEIGHTS_SHA256,
            Self::UnqualifiedCandidateV1 { weights_sha256, .. } => weights_sha256,
        }
    }

    pub fn training_receipt_sha256(&self) -> Option<&str> {
        match self {
            Self::ReleasedV1 => None,
            Self::UnqualifiedCandidateV1 {
                training_receipt_sha256,
                ..
            } => Some(training_receipt_sha256),
        }
    }

    pub fn embedding_recipe(&self) -> &'static str {
        match self {
            Self::ReleasedV1 => DOPE512_RECIPE,
            Self::UnqualifiedCandidateV1 { .. } => CANDIDATE512_RECIPE,
        }
    }

    pub fn fusion_recipe(&self) -> &'static str {
        match self {
            Self::ReleasedV1 => "dope512-foundation1024-target512/v1",
            Self::UnqualifiedCandidateV1 { .. } => {
                "dope512-unqualified-candidate-foundation1024-target512/v1"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_identity_cannot_relabel_baseline_or_accept_malformed_receipts() {
        assert!(DopeCheckpointIdentity::unqualified_candidate(
            DOPE_WEIGHTS_SHA256.into(),
            "b".repeat(64)
        )
        .is_err());
        assert!(
            DopeCheckpointIdentity::unqualified_candidate("a".repeat(64), "missing".into())
                .is_err()
        );
        let pin =
            DopeCheckpointIdentity::unqualified_candidate("a".repeat(64), "b".repeat(64)).unwrap();
        assert_eq!(pin.weights_sha256(), "a".repeat(64));
        assert_eq!(pin.training_receipt_sha256(), Some("b".repeat(64).as_str()));
        assert_ne!(
            pin.embedding_recipe(),
            DopeCheckpointIdentity::ReleasedV1.embedding_recipe()
        );
        assert_eq!(
            serde_json::from_value::<DopeCheckpointIdentity>(serde_json::to_value(&pin).unwrap())
                .unwrap(),
            pin
        );
    }
}
