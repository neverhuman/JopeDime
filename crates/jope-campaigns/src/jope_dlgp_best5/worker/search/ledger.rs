use super::*;

pub(super) fn budget_ledger(arm: Best5ArmV1) -> BudgetLedgerV1 {
    let arm_specific_operations = if arm == Best5ArmV1::CrossFittedResidualDlgp {
        24
    } else if arm == Best5ArmV1::EvolvabilityUcbLineages {
        140
    } else {
        0
    };
    BudgetLedgerV1 {
        logical_operations: 950 + arm_specific_operations,
        predicted_cpu_seconds: 0,
        mandatory_anchor_evaluations: 80,
        completed_anchor_evaluations: 80,
        mandatory_search_evaluations: 640,
        completed_search_evaluations: 640,
        mandatory_archive_evaluations: 160,
        completed_archive_evaluations: 160,
        mandatory_cv7_evaluations: 70,
        completed_cv7_evaluations: 70,
        arm_specific_operations,
    }
}
