#[path = "../src/retention/executor_candidate.rs"]
mod candidate_executor;

mod canon {
    pub use jope_learning::canon::*;
}

mod retention {
    pub use jope_learning::retention::*;
}

use candidate_executor::{
    AuthoritySnapshot, ExactVersionExecutor, MockNamespace, MockRecoveryNamespace, OwnerApproval,
    RetirementPlan, RetirementReservation, SimulatedCrash, TombstoneState,
};
use jope_learning::retention::obligations::{StrongPhysicalRef, VersionId};
use std::collections::BTreeSet;

fn object(key: &str, version: &str, digest: &str) -> StrongPhysicalRef {
    StrongPhysicalRef {
        account: "714044927654".into(),
        bucket: "learning-use2".into(),
        key: key.into(),
        version_id: VersionId::Explicit(version.into()),
        sha256: digest.into(),
        bytes: 256,
    }
}

fn authority() -> AuthoritySnapshot {
    AuthoritySnapshot {
        root_generation: "root-30".into(),
        pin_generation: "pin-30".into(),
        lifecycle_generation: "life-8".into(),
        closure_complete: true,
        registered_publishers: BTreeSet::from(["trainer".into()]),
        observed_publishers: BTreeSet::from(["trainer".into()]),
        active_pins: BTreeSet::new(),
        retired: BTreeSet::new(),
    }
}

fn approval(plan: &RetirementPlan) -> OwnerApproval {
    assert_eq!(plan.exact_targets.len(), 1);
    let target = &plan.exact_targets[0];
    let mut replacement = target.clone();
    replacement.key = format!("recovery/{}", target.key);
    replacement.version_id = VersionId::Explicit("recovery-v1".into());
    let mut recovery =
        MockRecoveryNamespace::with_objects("mock-recovery-isolated", [replacement.clone()]);
    let restore = recovery
        .restore_exact(target, &replacement, &plan.target_set_sha256)
        .unwrap();
    assert!(recovery.contains(target));
    OwnerApproval {
        signed_plan_sha256: plan.plan_sha256.clone(),
        signer: "independent-owner".into(),
        signature_verified: true,
        independent_reviewer: true,
        restore,
    }
}

fn setup(target: StrongPhysicalRef) -> (RetirementPlan, RetirementReservation, AuthoritySnapshot) {
    let snapshot = authority();
    let plan = RetirementPlan::propose(&snapshot, "planner", vec![target]).unwrap();
    let reservation = RetirementReservation::reserve(&plan, &snapshot, &approval(&plan)).unwrap();
    (plan, reservation, snapshot)
}

#[test]
fn disabled_candidate_cannot_run_and_default_is_off() {
    let target = object("packs/old", "v1", &"1".repeat(64));
    let (_, reservation, state) = setup(target.clone());
    let mut namespace = MockNamespace::with_objects([target]);
    let mut executor = ExactVersionExecutor::disabled();
    assert!(!executor.is_enabled());
    assert!(executor
        .execute_mock(&reservation, &state, &mut namespace, SimulatedCrash::None)
        .is_err());
    assert_eq!(namespace.exact_request_count, 0);
}

#[test]
fn publisher_pin_during_plan_or_reservation_aborts_without_mock_effect() {
    let target = object("packs/old", "v1", &"2".repeat(64));
    let (plan, _, state) = setup(target.clone());
    let mut publisher_race = state.clone();
    publisher_race.pin_generation = "pin-31".into();
    publisher_race.active_pins.insert(target.clone());
    assert!(RetirementReservation::reserve(&plan, &publisher_race, &approval(&plan)).is_err());

    let (_, reservation, mut state) = setup(target.clone());
    let mut namespace = MockNamespace::with_objects([target.clone()]);
    state.pin_generation = "pin-31".into();
    state.active_pins.insert(target.clone());
    let mut executor = ExactVersionExecutor::mock_candidate();
    assert!(executor
        .execute_mock(&reservation, &state, &mut namespace, SimulatedCrash::None)
        .is_err());
    assert!(namespace.contains(&target));
    assert_eq!(namespace.exact_request_count, 0);
}

#[test]
fn crash_before_request_reconciles_only_same_exact_version() {
    let target = object("packs/old", "v4", &"3".repeat(64));
    let (_, reservation, state) = setup(target.clone());
    let mut namespace = MockNamespace::with_objects([target.clone()]);
    let mut executor = ExactVersionExecutor::mock_candidate();
    let interrupted = executor
        .execute_mock(
            &reservation,
            &state,
            &mut namespace,
            SimulatedCrash::BeforeExactRequest,
        )
        .unwrap();
    assert!(interrupted.crashed);
    assert!(namespace.contains(&target));
    assert_eq!(namespace.exact_request_count, 0);

    let reconciled = executor
        .reconcile_mock(&reservation, &state, &mut namespace)
        .unwrap();
    assert!(!namespace.contains(&target));
    assert_eq!(namespace.exact_request_count, 1);
    assert_eq!(reconciled.live_removal_count, 0);
    assert_eq!(reconciled.simulated_effect_count, 1);
    assert_eq!(reconciled.tombstones[0].state, TombstoneState::Acknowledged);
}

#[test]
fn response_loss_after_exact_request_is_observed_without_retry() {
    let target = object("packs/old", "v5", &"4".repeat(64));
    let (_, reservation, state) = setup(target.clone());
    let mut namespace = MockNamespace::with_objects([target.clone()]);
    let mut executor = ExactVersionExecutor::mock_candidate();
    let interrupted = executor
        .execute_mock(
            &reservation,
            &state,
            &mut namespace,
            SimulatedCrash::AfterExactRequestBeforeAcknowledgement,
        )
        .unwrap();
    assert!(interrupted.crashed);
    assert!(!namespace.contains(&target));
    assert_eq!(
        namespace.request_count_for(&interrupted.tombstones[0].request_id),
        1
    );
    let reconciled = executor
        .reconcile_mock(&reservation, &state, &mut namespace)
        .unwrap();
    assert_eq!(namespace.exact_request_count, 1);
    assert_eq!(reconciled.simulated_effect_count, 1);
    assert_eq!(reconciled.live_removal_count, 0);
    assert_eq!(
        reconciled.tombstones[0].state,
        TombstoneState::ReconciledAbsent
    );
    assert!(reconciled.tombstones[0].detail.contains("no retry"));
}

#[test]
fn approval_must_bind_exact_plan_and_independent_restore_proof() {
    let target = object("checkpoints/old", "v7", &"5".repeat(64));
    let (plan, _, state) = setup(target);
    let mut bad = approval(&plan);
    bad.signed_plan_sha256 = "0".repeat(64);
    assert!(RetirementReservation::reserve(&plan, &state, &bad).is_err());
    let mut bad = approval(&plan);
    bad.signer = plan.planner.clone();
    assert!(RetirementReservation::reserve(&plan, &state, &bad).is_err());
    let mut bad = approval(&plan);
    bad.restore.original_absent_during_restore = false;
    assert!(RetirementReservation::reserve(&plan, &state, &bad).is_err());
}
