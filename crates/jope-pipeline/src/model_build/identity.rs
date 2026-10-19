//! One execution identity shared by every member of a winner ensemble (SC-DEC-184M JA-R2).
//!
//! The identity binds the request, the selection-side rows and labels, the report-side rows and
//! labels, the groups, the protocol and the data bytes. Every admitted member must carry the same
//! request and group identity, and the labels supplied to the ensemble must hash to the same
//! cohort. A reloaded ensemble must reproduce the identity it was built under.
use super::oof::AdmittedBundle;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROTOCOL: &str = "outer-selection-refit-report/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub protocol: String,
    pub request_sha256: String,
    /// Digest of the feature and target bytes the fitters were given.
    pub data_sha256: String,
    /// Digest of the selection-side row identifiers.
    pub row_sha256: String,
    /// Digest of the selection-side row identifiers paired with their labels.
    pub cohort_sha256: String,
    /// Digest of the report-side row identifiers paired with their labels.
    pub report_sha256: String,
    /// Digest of the group identifier of every selection-side row, in row order.
    pub group_sha256: String,
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Digest of ordered identifiers, each terminated so concatenations cannot collide.
pub fn digest_ids(ids: &[String]) -> String {
    let mut buffer = Vec::new();
    for id in ids {
        buffer.extend_from_slice(id.as_bytes());
        buffer.push(0);
    }
    hex(buffer)
}

/// Digest of ordered identifiers paired with ordered labels.
pub fn digest_pairs(ids: &[String], labels: &[String]) -> String {
    let mut buffer = Vec::new();
    for (id, label) in ids.iter().zip(labels) {
        buffer.extend_from_slice(id.as_bytes());
        buffer.push(0x1f);
        buffer.extend_from_slice(label.as_bytes());
        buffer.push(0);
    }
    hex(buffer)
}

impl ExecutionIdentity {
    pub fn new(
        request_sha256: &str,
        data_sha256: &str,
        selection_ids: &[String],
        selection_labels: &[String],
        selection_groups: &[String],
        report_ids: &[String],
        report_labels: &[String],
    ) -> Result<Self> {
        ensure!(
            selection_ids.len() == selection_labels.len(),
            "selection ids and labels are not aligned"
        );
        ensure!(
            report_ids.len() == report_labels.len(),
            "report ids and labels are not aligned"
        );
        Ok(Self {
            protocol: PROTOCOL.into(),
            request_sha256: request_sha256.into(),
            data_sha256: data_sha256.into(),
            row_sha256: digest_ids(selection_ids),
            cohort_sha256: digest_pairs(selection_ids, selection_labels),
            report_sha256: digest_pairs(report_ids, report_labels),
            group_sha256: digest_ids(selection_groups),
        })
    }

    /// Refuses any admitted member whose request or group identity differs from this identity,
    /// and any selection-side labels that do not reproduce the cohort digest.
    pub fn verify_members(
        &self,
        admitted: &[AdmittedBundle],
        selection_labels: &[String],
    ) -> Result<()> {
        ensure!(!admitted.is_empty(), "no admitted members");
        ensure!(
            self.protocol == PROTOCOL,
            "protocol differs from {PROTOCOL}"
        );
        for member in admitted {
            let bundle = member.bundle();
            ensure!(
                bundle.request_sha256 == self.request_sha256,
                "member {} was produced under a different request",
                bundle.candidate_id
            );
            ensure!(
                digest_ids(&bundle.group_ids) == self.group_sha256,
                "member {} has a different group identity",
                bundle.candidate_id
            );
            ensure!(
                digest_ids(&bundle.row_ids) == self.row_sha256,
                "member {} has different rows",
                bundle.candidate_id
            );
        }
        ensure!(
            digest_pairs(&admitted[0].bundle().row_ids, selection_labels) == self.cohort_sha256,
            "selection labels do not reproduce the cohort identity"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("row-{i}")).collect()
    }

    fn base() -> ExecutionIdentity {
        let rows = ids(4);
        let labels: Vec<String> = (0..4).map(|i| i.to_string()).collect();
        let groups: Vec<String> = (0..4).map(|i| format!("g{i}")).collect();
        ExecutionIdentity::new(
            &"a".repeat(64),
            &"b".repeat(64),
            &rows,
            &labels,
            &groups,
            &ids(2),
            &labels[..2],
        )
        .unwrap()
    }

    #[test]
    fn identical_inputs_give_identical_identities() {
        assert_eq!(base(), base());
    }

    #[test]
    fn a_changed_label_changes_the_cohort_digest() {
        let rows = ids(4);
        let groups: Vec<String> = (0..4).map(|i| format!("g{i}")).collect();
        let a = base();
        let changed = ExecutionIdentity::new(
            &"a".repeat(64),
            &"b".repeat(64),
            &rows,
            &["0".into(), "1".into(), "2".into(), "9".into()],
            &groups,
            &ids(2),
            &["0".to_string(), "1".to_string()],
        )
        .unwrap();
        assert_ne!(a.cohort_sha256, changed.cohort_sha256);
    }

    #[test]
    fn id_boundaries_cannot_collide_by_concatenation() {
        assert_ne!(
            digest_ids(&["ab".into(), "c".into()]),
            digest_ids(&["a".into(), "bc".into()])
        );
    }
}
