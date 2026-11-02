use super::*;

pub const BUDGET_CAMPAIGN_ID: &str = "jope-dlgp-budget-first-7of22";
pub const DEVELOPMENT_DATASETS: usize = 22;
pub const FROZEN_CONFIRMATION_DATASETS: usize = 20;
pub const BUDGET_RUNGS: [u32; 4] = [1_000, 2_000, 5_000, 10_000];
pub const CONFIRMATION_SEEDS: [u64; 2] = [303, 404];
pub const STAGE1_MAX_SEARCH_EVALUATIONS: u64 = 200_000;
pub const HYPERION_BOOTSTRAP_SEED: u64 = 0x4843_4f4e_4649_524d;
pub const HYPERION_BOOTSTRAP_REPETITIONS: usize = 10_000;
pub const HYPERION_FULL_TRAIN_EXECUTIONS: usize = 42;
pub const HYPERION_CV5_EXECUTIONS: usize = 42 * 5;
pub const HYPERION_CV7_EXECUTIONS: usize = 42 * 7;
pub const HYPERION_TOTAL_ENCODER_EXECUTIONS: usize =
    HYPERION_FULL_TRAIN_EXECUTIONS + HYPERION_CV5_EXECUTIONS + HYPERION_CV7_EXECUTIONS;

/// The Hyperion cache command is campaign-specific: an arbitrary train-only
/// registry must not accidentally become the authority for this confirmation.
pub fn validate_hyperion_campaign_registry_ids(ids: &[String]) -> Result<(), String> {
    let expected = DEVELOPMENT_DATASET_IDS
        .iter()
        .chain(CONFIRMATION_DATASET_IDS.iter())
        .map(|id| (*id).to_owned())
        .collect::<Vec<_>>();
    if ids != expected {
        return Err(
            "Hyperion registry must contain the exact ordered 22-development plus 20-confirmation cohort"
                .into(),
        );
    }
    Ok(())
}

pub const DEVELOPMENT_DATASET_IDS: [&str; DEVELOPMENT_DATASETS] = [
    "1a803b27c2400dc4",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "e5a3a7b3668a1cc2",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
];

pub const CONFIRMATION_DATASET_IDS: [&str; FROZEN_CONFIRMATION_DATASETS] = [
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1cb97b1d7314d43d",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "542e75bc0cfdd1bf",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e",
    "b253fbfae8460dc1",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "ee56a1015e344c11",
    "fe3cc716a05d036f",
];

pub const STAGE1_DATASET_IDS: [&str; 10] = [
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "7af85a04de0c7018",
    "b5beff4a53fa12c0",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
];

pub(super) fn sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub(super) fn finite_positive(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite() && *value > 0.0)
}

mod confirmation;
mod evidence;
mod freeze;
mod policy;
mod stage;

pub use confirmation::{
    HyperionConfirmationDatasetV1, HyperionConfirmationReportV1, HyperionSeedConfirmationV1,
};
pub use evidence::{
    HyperionDevelopmentEvidenceV1, HyperionDevelopmentSeedEvidenceV1, HyperionSeedSelectionV1,
    HyperionSelectionInventoryV1,
};
pub use freeze::{
    BudgetFirstCampaignFreezeV1, CampaignLayerV1, SealedTestBindingV1, SealedTestInventoryV1,
    SearchDatasetBindingV1, TestAccessPolicyV1,
};
pub use policy::{HyperionDecisionRuleV1, HyperionPolicyFreezeV1};
pub use stage::{
    choose_predeclared_raw_fallback, stage_hyperion_sealed_features, verify_full_train_replay,
    HyperionSealedFeatureStageV1,
};
