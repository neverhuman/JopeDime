//! One execution identity shared by every member of a winner ensemble (SC-DEC-184M JA-R2, JA-F1,
//! JA-F2).
//!
//! The identity binds the request, the validated bake-off plan, the data bytes, the selection-side
//! rows and labels, the report-side rows and labels, the groups and the protocol. Admission
//! refuses duplicate row identifiers within a side and any identifier shared by the selection and
//! report sides, since a shared row would let the report side leak into the selection side.
//! Every admitted member must carry the same request and group identity, and the selection labels
//! must reproduce the cohort digest.
use super::oof::AdmittedBundle;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PROTOCOL: &str = "outer-selection-refit-report/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub protocol: String,
    pub request_sha256: String,
    /// Digest of the validated bake-off plan: task, class order, rows, labels, folds, groups and
    /// request. A fitter must be bound to this exact plan before any fit starts.
    pub plan_sha256: String,
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
    /// Digest of the group identifier of every report-side row, in row order. Empty when grouping
    /// does not apply.
    pub report_group_sha256: String,
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

fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

impl ExecutionIdentity {
    /// One argument per identity input, so each binding stays named at the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_sha256: &str,
        plan_sha256: &str,
        data_sha256: &str,
        selection_ids: &[String],
        selection_labels: &[String],
        selection_groups: &[String],
        report_ids: &[String],
        report_labels: &[String],
        report_groups: &[String],
    ) -> Result<Self> {
        ensure!(
            is_sha256(request_sha256),
            "request identity is not a sha256"
        );
        ensure!(is_sha256(plan_sha256), "plan identity is not a sha256");
        ensure!(
            !selection_ids.is_empty() && !report_ids.is_empty(),
            "both the selection and the report side need rows"
        );
        ensure!(
            selection_ids.len() == selection_labels.len(),
            "selection ids and labels are not aligned"
        );
        ensure!(
            report_ids.len() == report_labels.len(),
            "report ids and labels are not aligned"
        );
        ensure!(
            selection_groups.is_empty() || selection_groups.len() == selection_ids.len(),
            "selection groups are not row-aligned"
        );
        let selection_set: BTreeSet<&str> = selection_ids.iter().map(String::as_str).collect();
        let report_set: BTreeSet<&str> = report_ids.iter().map(String::as_str).collect();
        ensure!(
            selection_set.len() == selection_ids.len(),
            "selection side repeats a row id"
        );
        ensure!(
            report_set.len() == report_ids.len(),
            "report side repeats a row id"
        );
        ensure!(
            selection_set.is_disjoint(&report_set),
            "a row appears on both the selection and the report side"
        );
        // Grouped CV: a group that spans both sides would let the report side leak into selection,
        // so grouping must cover both sides or neither, and the group sets must be disjoint.
        ensure!(
            report_groups.is_empty() || report_groups.len() == report_ids.len(),
            "report groups are not row-aligned"
        );
        ensure!(
            selection_groups.is_empty() == report_groups.is_empty(),
            "grouping must cover both the selection and the report side, or neither"
        );
        let selection_groups_set: BTreeSet<&str> =
            selection_groups.iter().map(String::as_str).collect();
        let report_groups_set: BTreeSet<&str> = report_groups.iter().map(String::as_str).collect();
        ensure!(
            selection_groups_set.is_disjoint(&report_groups_set),
            "a group appears on both the selection and the report side"
        );
        Ok(Self {
            protocol: PROTOCOL.into(),
            request_sha256: request_sha256.into(),
            plan_sha256: plan_sha256.into(),
            data_sha256: data_sha256.into(),
            row_sha256: digest_ids(selection_ids),
            cohort_sha256: digest_pairs(selection_ids, selection_labels),
            report_sha256: digest_pairs(report_ids, report_labels),
            group_sha256: digest_ids(selection_groups),
            report_group_sha256: digest_ids(report_groups),
        })
    }

    /// Refuses any admitted member whose request, plan or group identity differs from this
    /// identity, and any selection-side labels that do not reproduce the cohort digest.
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

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn ids(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}-{i}")).collect()
    }

    fn build(sel: &[String], rep: &[String]) -> Result<ExecutionIdentity> {
        let sel_labels: Vec<String> = sel.iter().map(|_| "0".into()).collect();
        let rep_labels: Vec<String> = rep.iter().map(|_| "0".into()).collect();
        ExecutionIdentity::new(SHA, SHA, SHA, sel, &sel_labels, &[], rep, &rep_labels, &[])
    }

    #[test]
    fn identical_inputs_give_identical_identities() {
        let sel = ids("s", 4);
        let rep = ids("r", 2);
        assert_eq!(build(&sel, &rep).unwrap(), build(&sel, &rep).unwrap());
    }

    #[test]
    fn a_changed_label_changes_the_cohort_digest() {
        let sel = ids("s", 4);
        let rep = ids("r", 2);
        let a = build(&sel, &rep).unwrap();
        let mut labels: Vec<String> = sel.iter().map(|_| "0".into()).collect();
        labels[3] = "9".into();
        let rep_labels: Vec<String> = rep.iter().map(|_| "0".into()).collect();
        let b = ExecutionIdentity::new(SHA, SHA, SHA, &sel, &labels, &[], &rep, &rep_labels, &[])
            .unwrap();
        assert_ne!(a.cohort_sha256, b.cohort_sha256);
    }

    #[test]
    fn id_boundaries_cannot_collide_by_concatenation() {
        assert_ne!(
            digest_ids(&["ab".into(), "c".into()]),
            digest_ids(&["a".into(), "bc".into()])
        );
    }

    #[test]
    fn a_repeated_selection_row_is_refused() {
        let mut sel = ids("s", 4);
        sel[3] = sel[0].clone();
        assert!(build(&sel, &ids("r", 2)).is_err());
    }

    #[test]
    fn a_repeated_report_row_is_refused() {
        let mut rep = ids("r", 2);
        rep[1] = rep[0].clone();
        assert!(build(&ids("s", 4), &rep).is_err());
    }

    #[test]
    fn a_row_shared_by_both_sides_is_refused() {
        let sel = ids("s", 4);
        let mut rep = ids("r", 2);
        rep[0] = sel[2].clone();
        assert!(build(&sel, &rep).is_err());
    }

    #[test]
    fn a_plan_identity_that_is_not_a_sha256_is_refused() {
        let sel = ids("s", 4);
        let rep = ids("r", 2);
        let labels: Vec<String> = sel.iter().map(|_| "0".into()).collect();
        let rep_labels: Vec<String> = rep.iter().map(|_| "0".into()).collect();
        assert!(ExecutionIdentity::new(
            SHA,
            "plan",
            SHA,
            &sel,
            &labels,
            &[],
            &rep,
            &rep_labels,
            &[]
        )
        .is_err());
    }
}
