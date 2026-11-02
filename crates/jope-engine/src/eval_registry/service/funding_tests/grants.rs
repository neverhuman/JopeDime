#[test]
fn verify_gpu_task_grant_positive_and_negative() {
    if ensure_numerical_env_or_reexec(
        "eval_registry::service::funding_tests::verify_gpu_task_grant_positive_and_negative",
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

    // Positive: controller and participant rank 0 pass
    crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "regression_training",
    )
    .unwrap();
    crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        0,
        "regression_training",
    )
    .unwrap();

    // Negative: capability mismatch, slot mismatch, second slot from same PID, and descriptor mismatch fail closed
    assert!(fixture.request(&fixture.participant(1)).is_err());
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &fixture.descriptor,
        "unfunded_capability"
    )
    .is_err());
    assert!(crate::controller::verify_participant_gpu_grant(
        &rank0,
        &fixture.descriptor,
        1,
        "regression_training"
    )
    .is_err());
    assert!(crate::controller::verify_controller_gpu_grant(
        &rank0,
        &fixture.descriptor,
        "regression_training"
    )
    .is_err());
    let mut tampered = fixture.descriptor.clone();
    tampered["task"]["world"] = json!(8);
    assert!(crate::controller::verify_controller_gpu_grant(
        &start,
        &tampered,
        "regression_training"
    )
    .is_err());

    assert_eq!(witness.join().unwrap().len(), 2);
    assert_eq!(service.finish().worker_spawns, 0);
}
