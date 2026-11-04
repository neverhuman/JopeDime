use super::*;

#[test]
fn disk_low_stops_admission_then_checkpoints_and_pauses_once_without_auto_resume() {
    let mut gov = governor(8);
    let mut effects = Effects::default();
    let mut sensors = Readings::new(500, 99);
    let sample = gov.poll(&mut sensors, &mut effects).unwrap();
    assert!(!sample.admission_allowed);
    assert_eq!(sample.warnings[0].code, "DISK_LOW");
    assert!(sample.warnings[0].host);
    assert_eq!(effects.calls, ["admit:false"]);
    sensors.disk = Ok(100);
    let recovered = gov.poll(&mut sensors, &mut effects).unwrap();
    assert!(
        !recovered.admission_allowed,
        "confirmed pause survives early space recovery"
    );
    assert_eq!(recovered.warnings[0].state, WarningState::Cleared);
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::Paused(ErrorCode::DiskLow)
    );
    assert_eq!(
        effects.calls,
        ["admit:false", "admit:false", "commit", "pause:DISK_LOW"]
    );
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::None
    );
    assert!(
        gov.poll(&mut sensors, &mut effects)
            .unwrap()
            .admission_allowed
    );
    assert!(!effects.calls.iter().any(|call| call.contains("continue")));
    sensors.disk = Ok(99);
    gov.poll(&mut sensors, &mut effects).unwrap();
    gov.round_boundary(1, &mut effects).unwrap();
    gov.poll(&mut sensors, &mut effects).unwrap();
    assert_eq!(
        gov.round_boundary(2, &mut effects).unwrap(),
        RoundEffect::None
    );
    assert_eq!(gov.planned_r().get(), 2);
}

#[test]
fn unavailable_sources_hold_memory_state_and_never_claim_disk_low_or_recovery() {
    let mut gov = governor(8);
    let mut effects = Effects::default();
    let mut sensors = Readings::new(99, 101);
    gov.poll(&mut sensors, &mut effects).unwrap();
    gov.round_boundary(0, &mut effects).unwrap();
    sensors.memory = Err("unavailable".into());
    sensors.disk = Err("permission_denied".into());
    let sample = gov.poll(&mut sensors, &mut effects).unwrap();
    assert!(!sample.admission_allowed);
    assert!(sample.warnings.is_empty());
    assert_eq!(sample.disk_free, Err("permission_denied".into()));
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::None
    );
    assert_eq!(gov.live_workers().get(), 7);
    sensors.memory = Ok(150);
    sensors.disk = Ok(101);
    assert!(gov
        .poll(&mut sensors, &mut effects)
        .unwrap()
        .warnings
        .is_empty());
    assert_eq!(
        gov.round_boundary(2, &mut effects).unwrap(),
        RoundEffect::None
    );
    sensors.memory = Ok(201);
    assert_eq!(
        gov.poll(&mut sensors, &mut effects).unwrap().warnings[0].state,
        WarningState::Cleared
    );
}

#[test]
fn failed_actions_retry_without_advancing_worker_count_or_boundary() {
    let mut gov = governor(2);
    let mut effects = Effects {
        fail_resize: true,
        ..Default::default()
    };
    let mut sensors = Readings::new(99, 100);
    gov.poll(&mut sensors, &mut effects).unwrap();
    assert!(gov.round_boundary(0, &mut effects).is_err());
    assert_eq!(gov.live_workers().get(), 2);
    effects.fail_resize = false;
    gov.round_boundary(0, &mut effects).unwrap();
    assert_eq!(gov.live_workers().get(), 1);
    sensors.disk = Ok(99);
    gov.poll(&mut sensors, &mut effects).unwrap();
    effects.fail_checkpoint = true;
    assert!(gov.round_boundary(1, &mut effects).is_err());
    assert!(!effects.admission);
    effects.fail_checkpoint = false;
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::Paused(ErrorCode::DiskLow)
    );
    assert_eq!(
        effects
            .calls
            .iter()
            .filter(|call| *call == "commit")
            .count(),
        1
    );
}

#[test]
fn unacknowledged_admission_cannot_checkpoint_or_resize() {
    let mut gov = governor(8);
    let mut effects = Effects {
        fail_admission: true,
        ..Default::default()
    };
    let mut sensors = Readings::new(99, 99);
    assert!(gov.round_boundary(0, &mut effects).is_err());
    assert!(gov.poll(&mut sensors, &mut effects).is_err());
    assert!(gov.round_boundary(0, &mut effects).is_err());
    assert!(effects.calls.is_empty());
    effects.fail_admission = false;
    gov.poll(&mut sensors, &mut effects).unwrap();
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::Paused(ErrorCode::DiskLow)
    );
}

#[test]
fn pressure_seen_mid_round_is_applied_once_even_if_next_sample_is_in_the_band() {
    let mut gov = governor(8);
    let mut effects = Effects::default();
    let mut sensors = Readings::new(99, 100);
    gov.poll(&mut sensors, &mut effects).unwrap();
    sensors.memory = Ok(150);
    gov.poll(&mut sensors, &mut effects).unwrap();
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::Workers {
            previous: 8,
            current: 7
        }
    );
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::None
    );
    assert_eq!(gov.live_workers().get(), 7);
}

#[test]
fn partial_resize_error_keeps_target_and_gate_until_confirmed() {
    let mut gov = governor(8);
    let mut effects = Effects {
        workers: 8,
        fail_resize: true,
        partial_resize: true,
        ..Default::default()
    };
    let mut sensors = Readings::new(99, 100);
    gov.poll(&mut sensors, &mut effects).unwrap();
    assert!(gov.round_boundary(0, &mut effects).is_err());
    assert_eq!(
        effects.workers, 7,
        "the actuator took effect before reporting failure"
    );
    assert_eq!(
        gov.live_workers().get(),
        8,
        "only the last confirmed count is known"
    );
    assert!(
        !effects.admission,
        "in-doubt worker effects close admission"
    );
    sensors.memory = Ok(201);
    assert!(
        !gov.poll(&mut sensors, &mut effects)
            .unwrap()
            .admission_allowed
    );
    effects.fail_resize = false;
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::Workers {
            previous: 8,
            current: 7
        }
    );
    assert_eq!(effects.workers, gov.live_workers().get());
    assert!(
        gov.poll(&mut sensors, &mut effects)
            .unwrap()
            .admission_allowed
    );
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::Workers {
            previous: 7,
            current: 8
        }
    );
    assert_eq!(gov.planned_r().get(), 2);
}
