use feat_math::kernel::operation_profile::{
    operation_profile_for, operation_profiles, validate_successor_params,
};
use feat_math::kernel::registry_v2::{source_sha256, OPERATORS};

#[test]
fn join_covers_every_registry_entry() {
    let profiles = operation_profiles();
    assert_eq!(profiles.len(), OPERATORS.len());
    assert_eq!(profiles.len(), 90);
}

#[test]
fn registry_identity_is_unchanged() {
    assert_eq!(
        source_sha256(),
        "4b674c994a6cbeb9dcf5c4bf80ad63efe8bc1fc2dacb54de19e56e3ce82f4f7d"
    );
}

#[test]
fn elasticnet_profile_carries_l1_ratio_bound() {
    let profile = operation_profile_for("legacy_recipe", "elasticnet").expect("in registry");
    assert_eq!(profile.admission, "enabled");
    assert!(profile.native_policy_mask);
    assert_eq!(profile.param_bounds.len(), 1);
    assert!(operation_profile_for("legacy_recipe", "no_such_op").is_none());
}

#[test]
fn l1_ratio_unit_interval_is_enforced() {
    for ok in [0.0, 0.5, 1.0] {
        validate_successor_params("elasticnet", &[("l1_ratio", ok)]).unwrap();
    }
    for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
        assert!(
            validate_successor_params("elasticnet", &[("l1_ratio", bad)]).is_err(),
            "{bad} accepted"
        );
    }
    validate_successor_params("ridge", &[("l1_ratio", 7.0)]).unwrap();
}
