use super::*;

#[test]
fn preflight_populates_canonical_estimates_and_preserves_exact_reserve() {
    let gov = governor(8);
    let needs = Requirements {
        bundle_ram_bytes: 10,
        matrix_ram_bytes: 20,
        per_worker_ram_bytes: 30,
        dataset_copy_bytes: 40,
        matrix_disk_bytes: 50,
        artifact_bytes: 60,
    };
    let mut sensors = Readings::new(500, 250);
    let estimates = gov.preflight(needs, &mut sensors).unwrap();
    assert_eq!(estimates.peak_ram_bytes.0, 270);
    assert_eq!(estimates.disk_bytes.0, 150);
    assert_eq!(gov.planned_workers().get(), 8);
    assert_eq!(gov.planned_r().get(), 2);
    sensors.disk = Ok(249);
    let error = gov.preflight(needs, &mut sensors).unwrap_err();
    assert_eq!((error.code, error.status), (ErrorCode::DiskLow, 507));
    assert!(error.remedy.contains("jopedime gc"));
    sensors.disk = Err("permission_denied".into());
    assert_eq!(
        gov.preflight(needs, &mut sensors).unwrap_err().code,
        ErrorCode::CapabilityUnavailable
    );
    sensors.disk = Ok(u64::MAX);
    for needs in [
        Requirements {
            per_worker_ram_bytes: u64::MAX,
            ..Default::default()
        },
        Requirements {
            bundle_ram_bytes: u64::MAX,
            matrix_ram_bytes: 1,
            ..Default::default()
        },
        Requirements {
            dataset_copy_bytes: u64::MAX,
            artifact_bytes: 1,
            ..Default::default()
        },
        Requirements {
            dataset_copy_bytes: u64::MAX,
            ..Default::default()
        },
    ] {
        assert_eq!(
            gov.preflight(needs, &mut sensors).unwrap_err().code,
            ErrorCode::InvalidInput
        );
    }
}

#[test]
fn strict_thresholds_hold_the_band_and_change_only_one_worker_per_boundary() {
    let mut gov = governor(8);
    let mut effects = Effects::default();
    let mut sensors = Readings::new(100, 100);
    assert!(gov
        .poll(&mut sensors, &mut effects)
        .unwrap()
        .warnings
        .is_empty());
    assert_eq!(
        gov.round_boundary(0, &mut effects).unwrap(),
        RoundEffect::None
    );
    sensors.memory = Ok(99);
    let sample = gov.poll(&mut sensors, &mut effects).unwrap();
    assert_eq!(sample.warnings.len(), 1);
    assert_eq!(sample.warnings[0].code, "MEMORY_PRESSURE");
    assert!(sample.warnings[0].host);
    assert_eq!(sample.warnings[0].state, WarningState::Raised);
    assert_eq!(
        gov.live_workers().get(),
        8,
        "poll does not resize mid-round"
    );
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::Workers {
            previous: 8,
            current: 7
        }
    );
    assert_eq!(
        gov.round_boundary(1, &mut effects).unwrap(),
        RoundEffect::None
    );
    for (boundary, memory) in [(2, 100), (3, 150), (4, 200)] {
        sensors.memory = Ok(memory);
        assert!(gov
            .poll(&mut sensors, &mut effects)
            .unwrap()
            .warnings
            .is_empty());
        assert_eq!(
            gov.round_boundary(boundary, &mut effects).unwrap(),
            RoundEffect::None
        );
        assert_eq!(gov.live_workers().get(), 7);
    }
    sensors.memory = Ok(99);
    gov.poll(&mut sensors, &mut effects).unwrap();
    for boundary in 5..20 {
        gov.round_boundary(boundary, &mut effects).unwrap();
        assert_eq!(gov.planned_r().get(), 2);
    }
    assert_eq!(gov.live_workers().get(), 1);
    sensors.memory = Ok(201);
    let recovered = gov.poll(&mut sensors, &mut effects).unwrap();
    assert_eq!(recovered.warnings[0].state, WarningState::Cleared);
    assert_eq!(
        gov.round_boundary(20, &mut effects).unwrap(),
        RoundEffect::Workers {
            previous: 1,
            current: 2
        }
    );
    for boundary in 21..40 {
        gov.round_boundary(boundary, &mut effects).unwrap();
    }
    assert_eq!(gov.live_workers(), gov.planned_workers());
    assert_eq!(gov.planned_r().get(), 2);
    assert!(gov.round_boundary(38, &mut effects).is_err());
    assert_eq!(governor(1).planned_r().get(), 1);
}

#[test]
fn byte_thresholds_remain_exact_above_f64_integer_precision() {
    let total = u64::MAX - 5; // exactly divisible by ten
    let mut gov = Governor::new(
        NonZeroU64::new(total).unwrap(),
        NonZeroU32::new(2).unwrap(),
        0,
    );
    let mut effects = Effects::default();
    let mut sensors = Readings::new(total / 10, 0);
    assert!(gov
        .poll(&mut sensors, &mut effects)
        .unwrap()
        .warnings
        .is_empty());
    gov.round_boundary(0, &mut effects).unwrap();
    assert_eq!(gov.live_workers().get(), 2);
    sensors.memory = Ok(total / 10 - 1);
    assert_eq!(
        gov.poll(&mut sensors, &mut effects).unwrap().warnings.len(),
        1
    );
    gov.round_boundary(1, &mut effects).unwrap();
    assert_eq!(gov.live_workers().get(), 1);
    sensors.memory = Ok(total / 5);
    assert!(gov
        .poll(&mut sensors, &mut effects)
        .unwrap()
        .warnings
        .is_empty());
    gov.round_boundary(2, &mut effects).unwrap();
    assert_eq!(gov.live_workers().get(), 1);
    sensors.memory = Ok(total / 5 + 1);
    assert_eq!(
        gov.poll(&mut sensors, &mut effects).unwrap().warnings[0].state,
        WarningState::Cleared
    );
    gov.round_boundary(3, &mut effects).unwrap();
    assert_eq!(gov.live_workers().get(), 2);
}
