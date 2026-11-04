use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn completion_verification_does_not_create_missing_authority_or_witness() {
    let data = Data::new();
    for (authority, witness) in [
        (
            data.root.join("missing-authority"),
            data.authority.witness.clone(),
        ),
        (
            data.authority.authority.clone(),
            data.root.join("missing-witness"),
        ),
    ] {
        let result = data
            .authority
            .command()
            .args(["verify-fit-completion", "--authority"])
            .arg(&authority)
            .arg("--witness")
            .arg(&witness)
            .args([
                "--completion-sha256",
                &"a".repeat(64),
                "--authority-instance",
                &"b".repeat(64),
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    assert!(!data.root.join("missing-authority").exists());
    assert!(!data.root.join("missing-witness").exists());
    data.authority.assert_no_fit_starts();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn self_hashed_but_uncommitted_completion_is_rejected_by_cli() {
    let data = Data::new();
    let registry = Authority::open(&data.authority.authority, &data.authority.witness).unwrap();
    let identity = FitIdentity {
        schema: "parallel.fit_identity.v2".into(),
        canonical_program: PROGRAM.into(),
        program_sha256: hash(PROGRAM.as_bytes()),
        training_sha256: "a".repeat(64),
        preprocessing_sha256: "b".repeat(64),
        feature_view_sha256: "c".repeat(64),
        executable_sha256: file_hash(Path::new(env!("CARGO_BIN_EXE_jope-parallel"))).unwrap(),
        runtime_sha256: "d".repeat(64),
        fit_seed: 0,
        numerical_contract_sha256: numerical_contract_sha256(&contract()).unwrap(),
    };
    let mut receipt = CompletionReceipt {
        schema: "parallel.fit_completion.v2".into(),
        authority_instance: registry.instance().into(),
        fit_key: hash(&serde_json::to_vec(&identity).unwrap()),
        attempt_token: "e".repeat(64),
        process: ProcessIdentity::current().unwrap(),
        task_id: "forged fixture".into(),
        executable_sha256: identity.executable_sha256.clone(),
        program_sha256: identity.program_sha256.clone(),
        training_sha256: identity.training_sha256.clone(),
        fit_identity: identity,
        outcome: ExecutionOutcome::Success,
        observed_exit_code: Some(0),
        observed_signal: None,
        elapsed_seconds: Some(0.1),
        deadline_seconds: 30.,
        output_sha256: hash(b"forged output"),
        model_sha256: Some(hash(b"forged model")),
        predictions_sha256: None,
        metrics_sha256: None,
        clean_completion_sha256: String::new(),
    };
    let mut value = serde_json::to_value(&receipt).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    receipt.clean_completion_sha256 = hash(&serde_json::to_vec(&value).unwrap());
    let bytes = serde_json::to_vec(&receipt).unwrap();
    verify_completion_receipt(&bytes).unwrap();
    let digest = hash(&bytes);
    fs::write(
        data.authority.authority.join("objects").join(&digest),
        bytes,
    )
    .unwrap();
    let result = verification_command(&data.authority, &digest, registry.instance())
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "self-hashed artifact was accepted without a committed original start"
    );
    assert!(result.stdout.is_empty());
    data.authority.assert_no_fit_starts();
}

pub(super) const ENTRIES: [&str; 5] = [
    "fit",
    "fit-view",
    "evaluate-view",
    "evaluate-memory",
    "evaluate-worker",
];
