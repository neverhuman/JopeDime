//! Contract rules for the reasoner pilot (RP-02). These are the executable forms of the contract
//! fixtures FX-01 to FX-05 in `cards_and_fixtures/TYPED_EDIT_REPLAY_AND_ABLATION_CONTRACT_FIXTURES.json`.
//!
//! This module is a contract model over synthetic fixtures. It runs no model, reads no dataset, and
//! performs no evaluation. It records the rules that a later pilot must satisfy, so that they are
//! tested before any run exists.
use anyhow::{bail, ensure, Result};

pub mod adapters;

/// Runs one arm of the named reasoner study under the no-time-limit policy
/// (`NO_TIME_LIMIT_POLICY_20261019`). The teacher arm and every matched baseline arm must go through
/// this function, so the policy is identical for each by construction.
pub fn run_study_arm(options: &crate::run::RunOptions) -> Result<crate::receipt::Receipt> {
    crate::run::run_no_time_limit(options)
}

/// A measured value that is either recorded or explicitly unknown. An unknown value is never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measured<T> {
    Null,
    Value(T),
}

impl Measured<u64> {
    /// Adds recorded values. One unknown addend makes the total unknown.
    pub fn sum_known(parts: &[Measured<u64>]) -> Measured<u64> {
        let mut total = 0u64;
        for part in parts {
            match part {
                Measured::Null => return Measured::Null,
                Measured::Value(v) => total += v,
            }
        }
        Measured::Value(total)
    }
}

/// The outcome of one typed-edit attempt. Only a constructive edit verified on inner CV earns credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditOutcome {
    ConstructiveVerified,
    Rejected,
    Invalid,
    Timeout,
    JudgedOrReranked,
}

/// Number of credited edits. Every other outcome earns nothing.
pub fn credited_edits(outcomes: &[EditOutcome]) -> usize {
    outcomes
        .iter()
        .filter(|o| **o == EditOutcome::ConstructiveVerified)
        .count()
}

/// Stage 1 admits evidence only when a constructive verified edit is present. Judging or reranking
/// candidates alone is refused. The pass criterion and the equal-budget value are owner decisions
/// that are not set here, so this function admits evidence for comparison and does not declare a pass.
pub fn admit_stage1_evidence(outcomes: &[EditOutcome]) -> Result<()> {
    ensure!(
        !outcomes.is_empty(),
        "stage 1 has no edit attempts to admit"
    );
    if credited_edits(outcomes) == 0 {
        bail!("no constructive verified edit: judging or reranking alone does not qualify for stage 1");
    }
    Ok(())
}

/// The two numerical interfaces. Their widths and vocabularies are distinct, and a receipt for one
/// is never valid for the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interface {
    Kumo2051,
    Hyperion2048,
}

impl Interface {
    pub fn embedding_width(self) -> u32 {
        match self {
            Interface::Kumo2051 => 2051,
            Interface::Hyperion2048 => 2048,
        }
    }
}

/// Refuses a receipt that was produced under a different interface than the target.
pub fn check_receipt_interface(receipt: Interface, target: Interface) -> Result<()> {
    ensure!(
        receipt == target,
        "receipt for {:?} presented to a {:?} interface; the two are never interchanged",
        receipt,
        target
    );
    Ok(())
}

/// Status of a pilot stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageStatus {
    Pass,
    Fail,
    Locked,
}

/// Stage 3 may start only after stage 1 and stage 2 both pass.
pub fn stage3_may_start(stage1: StageStatus, stage2: StageStatus) -> Result<()> {
    ensure!(
        stage1 == StageStatus::Pass && stage2 == StageStatus::Pass,
        "stage 3 is locked until stage 1 and stage 2 both pass (got {stage1:?} and {stage2:?})"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FX-01: a rejected edit is recorded, its cost is counted, and it earns no credit.
    #[test]
    fn fx01_rejected_edit_costs_are_counted_and_earn_no_credit() {
        let attempts = [
            EditOutcome::Rejected,
            EditOutcome::ConstructiveVerified,
            EditOutcome::Timeout,
        ];
        assert_eq!(credited_edits(&attempts), 1);
        // Every attempt is a physical start and counts toward cost, whatever its outcome.
        let starts: Vec<Measured<u64>> = attempts.iter().map(|_| Measured::Value(1)).collect();
        assert_eq!(Measured::sum_known(&starts), Measured::Value(3));
        let only_rejected = [EditOutcome::Rejected, EditOutcome::Invalid];
        assert_eq!(credited_edits(&only_rejected), 0);
    }

    /// FX-02: candidate judging or reranking without a verified constructive edit is refused.
    #[test]
    fn fx02_judging_or_reranking_alone_is_refused_for_stage1() {
        let judged_only = [EditOutcome::JudgedOrReranked, EditOutcome::JudgedOrReranked];
        let err = admit_stage1_evidence(&judged_only).unwrap_err().to_string();
        assert!(err.contains("judging or reranking alone"), "{err}");
        assert!(admit_stage1_evidence(&[EditOutcome::ConstructiveVerified]).is_ok());
        assert!(admit_stage1_evidence(&[]).is_err());
    }

    /// FX-03: a Kumo2051 receipt is refused by a Hyperion2048 interface, and the reverse is refused too.
    #[test]
    fn fx03_kumo2051_and_hyperion2048_receipts_are_never_interchanged() {
        assert_eq!(Interface::Kumo2051.embedding_width(), 2051);
        assert_eq!(Interface::Hyperion2048.embedding_width(), 2048);
        assert!(check_receipt_interface(Interface::Kumo2051, Interface::Hyperion2048).is_err());
        assert!(check_receipt_interface(Interface::Hyperion2048, Interface::Kumo2051).is_err());
        assert!(check_receipt_interface(Interface::Kumo2051, Interface::Kumo2051).is_ok());
        assert!(check_receipt_interface(Interface::Hyperion2048, Interface::Hyperion2048).is_ok());
    }

    /// FX-04: stage 3 cannot start while stage 1 or stage 2 is not a pass.
    #[test]
    fn fx04_stage3_stays_locked_until_stages_1_and_2_pass() {
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Pass).is_ok());
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Fail).is_err());
        assert!(stage3_may_start(StageStatus::Fail, StageStatus::Pass).is_err());
        assert!(stage3_may_start(StageStatus::Pass, StageStatus::Locked).is_err());
        assert!(stage3_may_start(StageStatus::Locked, StageStatus::Locked).is_err());
    }

    /// FX-05: an unknown measured value never turns into zero in a total.
    #[test]
    fn fx05_a_null_measured_field_never_becomes_zero() {
        let parts = [Measured::Value(3), Measured::Null];
        let total = Measured::sum_known(&parts);
        assert_eq!(total, Measured::Null);
        assert_ne!(total, Measured::Value(0));
        assert_eq!(Measured::sum_known(&[]), Measured::Value(0));
    }
}
