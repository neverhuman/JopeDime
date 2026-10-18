#[path = "../src/selfplay/curriculum.rs"]
mod curriculum;
#[path = "../src/selfplay/ranked.rs"]
mod ranked;

use curriculum::{CurriculumError, CurriculumSnapshot, PlantedTask, QualificationDisposition};
use ranked::{rank_reward_view, RankedError, RankedViewStatus, RawUtilityMeasurement};

fn raw() -> Vec<RawUtilityMeasurement> {
    vec![
        RawUtilityMeasurement {
            measurement_id: "m-a".into(),
            family_id: "family-real".into(),
            raw_utility: 0.7,
            synthetic: false,
        },
        RawUtilityMeasurement {
            measurement_id: "m-b".into(),
            family_id: "family-real".into(),
            raw_utility: 0.7,
            synthetic: false,
        },
        RawUtilityMeasurement {
            measurement_id: "m-c".into(),
            family_id: "family-synth".into(),
            raw_utility: 0.2,
            synthetic: true,
        },
    ]
}

fn schedule() -> Vec<PlantedTask> {
    vec![
        PlantedTask {
            task_id: "planted-a".into(),
            family_id: "synthetic-family".into(),
            synthetic: true,
        },
        PlantedTask {
            task_id: "planted-b".into(),
            family_id: "real-family".into(),
            synthetic: false,
        },
    ]
}

#[test]
fn empty_buffer_is_explicitly_unavailable_and_repeatable() {
    let first = rank_reward_view(&[], 1.0, 55, "policy-gen-1").unwrap();
    let second = rank_reward_view(&[], 1.0, 55, "policy-gen-1").unwrap();
    assert_eq!(first.status, RankedViewStatus::EmptyUnavailable);
    assert!(first.entries.is_empty());
    assert_eq!(first.view_id, second.view_id);
}

#[test]
fn ties_are_deterministic_under_the_pinned_seed() {
    let input = raw();
    let first = rank_reward_view(&input, 1.0, 1234, "policy-gen-1").unwrap();
    let again = rank_reward_view(&input, 1.0, 1234, "policy-gen-1").unwrap();
    let different_seed = rank_reward_view(&input, 1.0, 9999, "policy-gen-1").unwrap();
    assert_eq!(first.entries, again.entries);
    assert_eq!(first.view_id, again.view_id);
    assert_ne!(first.view_id, different_seed.view_id);
}

#[test]
fn alpha_mints_a_new_view_without_changing_raw_measurements() {
    let input = raw();
    let baseline = rank_reward_view(&input, 0.0, 1234, "policy-gen-1").unwrap();
    let changed_alpha = rank_reward_view(&input, 2.0, 1234, "policy-gen-1").unwrap();
    assert_ne!(baseline.view_id, changed_alpha.view_id);
    assert_eq!(
        baseline.raw_snapshot_sha256,
        changed_alpha.raw_snapshot_sha256
    );
    assert_eq!(
        baseline.entries.iter().map(|e| &e.0).collect::<Vec<_>>(),
        changed_alpha
            .entries
            .iter()
            .map(|e| &e.0)
            .collect::<Vec<_>>()
    );
    assert_ne!(
        baseline.entries[0].1.reward_weight,
        changed_alpha.entries[0].1.reward_weight
    );
}

#[test]
fn invalid_or_ambiguous_rank_inputs_are_refused() {
    assert_eq!(
        rank_reward_view(&raw(), f64::NAN, 1, "p").unwrap_err(),
        RankedError::InvalidAlpha
    );
    let mut duplicate = raw();
    duplicate[1].measurement_id = duplicate[0].measurement_id.clone();
    assert_eq!(
        rank_reward_view(&duplicate, 1.0, 1, "p").unwrap_err(),
        RankedError::DuplicateMeasurementId
    );
}

#[test]
fn curriculum_schedule_and_actual_exposure_parent_chain_are_enforced() {
    let mut snapshot =
        CurriculumSnapshot::new("schedule-1", "policy-gen-1", schedule(), 3).unwrap();
    let initial_id = snapshot.snapshot_id.clone();
    assert_eq!(
        snapshot.qualification,
        QualificationDisposition::NotEvaluated
    );
    assert_eq!(
        snapshot.record_exposure("e1", "planted-b", None, true),
        Err(CurriculumError::WrongScheduledTask)
    );
    snapshot
        .record_exposure("e1", "planted-a", None, true)
        .unwrap();
    assert_eq!(
        snapshot.record_exposure("e2", "planted-b", None, true),
        Err(CurriculumError::WrongParent)
    );
    snapshot
        .record_exposure("e2", "planted-b", Some("e1"), false)
        .unwrap();
    assert_ne!(snapshot.snapshot_id, initial_id);
    assert_eq!(
        snapshot.exposures[1].parent_exposure_id.as_deref(),
        Some("e1")
    );
    assert_eq!(snapshot.exposures[1].schedule_position, 1);
    assert_eq!(snapshot.can_claim_real_family_improvement(), false);
}

#[test]
fn synthetic_success_is_training_signal_only_and_replay_is_bounded() {
    let mut snapshot =
        CurriculumSnapshot::new("schedule-1", "policy-gen-1", schedule(), 2).unwrap();
    snapshot
        .record_exposure("e1", "planted-a", None, true)
        .unwrap();
    snapshot
        .record_exposure("e2", "planted-b", Some("e1"), true)
        .unwrap();
    assert_eq!(snapshot.synthetic_successes(), 1);
    assert!(!snapshot.can_claim_real_family_improvement());
    assert_eq!(
        snapshot.record_exposure("e3", "planted-a", Some("e2"), true),
        Err(CurriculumError::ExposureLimit)
    );
}
