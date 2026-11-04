fn ensure_numerical_env_or_reexec(test_name: &str) -> bool {
    let raw = fs::read("/proc/self/environ") // default fallback
        .unwrap_or_default();
    let env_map: std::collections::BTreeMap<String, String> = raw
        .split(|b| *b == 0)
        .filter_map(|v| std::str::from_utf8(v).ok()?.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let has_all = [
        "OMP_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    .iter()
    .all(|k| env_map.get(*k).map(String::as_str) == Some("1"));
    if has_all {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env("OMP_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .status()
        .expect("re-exec test with numerical environment");
    assert!(
        status.success(),
        "{test_name} failed in numerical env child"
    );
    true
}

#[test]
fn controller_calls_verify_gpu_task_grant() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::controller_calls_verify_gpu_task_grant",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(1, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_training",
    )
    .unwrap();
    assert_eq!(witness.join().unwrap().len(), 1);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn participant_per_rank_grant_verified() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::participant_per_rank_grant_verified",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let _start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    let rank0 = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        0,
        "regression_training",
    )
    .unwrap();
    // Same process cannot claim a second participant rank slot
    assert!(fixture.request(&fixture.participant(1)).is_err());
    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}

#[test]
fn negative_grant_mismatch_fails_closed() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::negative_grant_mismatch_fails_closed",
    ) {
        return;
    }
    let fixture = Fixture::new_resident_training(true);
    let service = fixture.launch();
    fixture.reserve();
    let witness = fixture.original.server(2, false);
    let start = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.start(),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();
    let rank0 = funding_ipc::funding_admission_installed(
        &fixture.installation,
        &fixture.participant(0),
        request_monotonic_ns().unwrap(),
    )
    .unwrap();

    // Wrong capability
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_proposals"
    )
    .is_err());

    // Controller admission cannot impersonate a participant rank
    assert!(crate::controller::verify_participant_gpu_grant(
        &start,
        &fixture.descriptor,
        0,
        "regression_training"
    )
    .is_err());

    // Participant rank 0 cannot impersonate controller or rank 1
    assert!(crate::controller::verify_controller_gpu_grant(
        &rank0,
        &fixture.descriptor,
        "regression_training"
    )
    .is_err());
    assert!(crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        1,
        "regression_training"
    )
    .is_err());

    // Tampered descriptor world / deadline
    let mut wrong_world = fixture.descriptor.clone();
    wrong_world["task"]["world"] = json!(8);
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &wrong_world,
        "regression_training"
    )
    .is_err());

    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}
