use super::*;

fn json_reference(root: &Path, name: &str, value: &Value) -> JsonEvidenceRefV2 {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    fs::write(path, &bytes).unwrap();
    JsonEvidenceRefV2 {
        path: name.to_owned(),
        raw_file_sha256: sha256_bytes(&bytes),
        canonical_json_sha256: crate::selfplay_data::canonical_json_sha256(value).unwrap(),
    }
}

fn raw_reference(root: &Path, name: &str, bytes: &[u8]) -> RawEvidenceRefV2 {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
    RawEvidenceRefV2 {
        path: name.to_owned(),
        raw_file_sha256: sha256_bytes(bytes),
    }
}

#[derive(Default)]
struct FakeHost {
    enabled: BTreeSet<String>,
    active: BTreeSet<String>,
    reloads: usize,
    enabled_batches: Vec<Vec<String>>,
    restarted: Vec<Vec<String>>,
}

impl ActivationHost for FakeHost {
    fn is_enabled(&mut self, unit: &str) -> Result<bool> {
        Ok(self.enabled.contains(unit))
    }
    fn is_active(&mut self, unit: &str) -> Result<bool> {
        Ok(self.active.contains(unit))
    }
    fn validate_candidate(&mut self, _root: &Path, _units: &[String]) -> Result<()> {
        Ok(())
    }
    fn daemon_reload(&mut self) -> Result<()> {
        self.reloads += 1;
        Ok(())
    }
    fn enable_timers(&mut self, timers: &[String]) -> Result<()> {
        self.enabled_batches.push(timers.to_vec());
        Ok(())
    }
    fn try_restart_services(&mut self, services: &[String]) -> Result<()> {
        self.restarted.push(services.to_vec());
        Ok(())
    }
    fn restore_states(&mut self, _units: &[ActivationUnitPreimageV1]) -> Result<()> {
        Ok(())
    }
}

fn fixture(root: &Path) -> (PathBuf, PathBuf, String, String) {
    let output = root.join("state/runtime-bundles");
    let units = root.join("user-units");
    let old = "a".repeat(64);
    let candidate = "b".repeat(64);
    for (identity, service) in [
        (&old, b"old-service".as_slice()),
        (&candidate, b"new-service".as_slice()),
    ] {
        let bundle = output.join(identity);
        fs::create_dir_all(bundle.join("runtime/systemd")).unwrap();
        fs::create_dir_all(bundle.join("runtime/bin")).unwrap();
        fs::write(
            bundle.join("bundle.json"),
            br#"{"service_manifest":{"services":[{"unit":"dime.service","active":true}]}}"#,
        )
        .unwrap();
        fs::write(bundle.join("runtime/systemd/dime.service"), service).unwrap();
        fs::write(bundle.join("runtime/systemd/dime.timer"), b"timer").unwrap();
        fs::write(bundle.join("runtime/bin/jope_selfplay_bundle"), b"recovery").unwrap();
    }
    fs::create_dir_all(&units).unwrap();
    fs::write(units.join("dime.service"), b"old-service").unwrap();
    symlink(&old, output.join("current")).unwrap();
    fs::write(output.join("active.json"), b"{\"old\":true}\n").unwrap();
    fs::create_dir_all(root.join("state/training")).unwrap();
    fs::write(root.join("state/training/accepted.json"), b"science").unwrap();
    (output, units, old, candidate)
}

#[test]
fn interruption_at_every_step_resumes_without_touching_science() {
    let stages = [
        ActivationStageV1::Prepared,
        ActivationStageV1::CandidateValidated,
        ActivationStageV1::UnitsStaged,
        ActivationStageV1::UnitsInstalled,
        ActivationStageV1::RuntimeSwitched,
        ActivationStageV1::ActivePublished,
        ActivationStageV1::DaemonReloaded,
        ActivationStageV1::TimersEnabled,
        ActivationStageV1::ServicesRestarted,
    ];
    for stage in stages {
        let temporary = tempfile::tempdir().unwrap();
        let (output, units, _old, candidate) = fixture(temporary.path());
        let transaction = temporary.path().join("transaction");
        let capsule = "c".repeat(64);
        let gate = "d".repeat(64);
        let request = ActivationRequest {
            output_root: &output,
            user_unit_root: &units,
            transaction_root: &transaction,
            candidate_bundle_sha256: &candidate,
            rollback_capsule_sha256: &capsule,
            candidate_review_gate_sha256: Some(&gate),
            external_gate_sha256: Some(&gate),
            active_document: serde_json::json!({"bundle_sha256": candidate}),
            operation: ActivationOperationV1::Activate,
        };
        let mut host = FakeHost::default();
        let stopped = execute_activation_transaction(&request, &mut host, Some(stage)).unwrap();
        assert_eq!(stopped.stage, stage);
        fs::write(
            temporary.path().join("state/training/accepted.json"),
            b"science-advanced",
        )
        .unwrap();
        let complete = execute_activation_transaction(&request, &mut host, None).unwrap();
        assert_eq!(complete.stage, ActivationStageV1::Complete);
        assert_eq!(
            fs::read(temporary.path().join("state/training/accepted.json")).unwrap(),
            b"science-advanced"
        );
        assert_ne!(
            fs::metadata(transaction.join("recovery-tool/jope_selfplay_bundle"))
                .unwrap()
                .permissions()
                .mode()
                & 0o111,
            0
        );
        assert_eq!(
            fs::read_link(output.join("current")).unwrap(),
            PathBuf::from(&candidate)
        );
        assert_eq!(
            fs::read(units.join("dime.service")).unwrap(),
            b"new-service"
        );
    }
}

#[test]
fn abort_restores_runtime_units_and_service_preimages_only() {
    let temporary = tempfile::tempdir().unwrap();
    let (output, units, old, candidate) = fixture(temporary.path());
    let transaction = temporary.path().join("transaction");
    let capsule = "c".repeat(64);
    let request = ActivationRequest {
        output_root: &output,
        user_unit_root: &units,
        transaction_root: &transaction,
        candidate_bundle_sha256: &candidate,
        rollback_capsule_sha256: &capsule,
        candidate_review_gate_sha256: None,
        external_gate_sha256: None,
        active_document: serde_json::json!({"bundle_sha256": candidate}),
        operation: ActivationOperationV1::Rollback,
    };
    let mut host = FakeHost::default();
    execute_activation_transaction(
        &request,
        &mut host,
        Some(ActivationStageV1::ActivePublished),
    )
    .unwrap();
    fs::write(
        temporary.path().join("state/training/accepted.json"),
        b"science-advanced",
    )
    .unwrap();
    let aborted = abort_activation_transaction(&output, &units, &transaction, &mut host).unwrap();
    assert_eq!(aborted.stage, ActivationStageV1::Aborted);
    assert_eq!(
        fs::read_link(output.join("current")).unwrap(),
        PathBuf::from(old)
    );
    assert_eq!(
        fs::read(units.join("dime.service")).unwrap(),
        b"old-service"
    );
    assert_eq!(
        fs::read(temporary.path().join("state/training/accepted.json")).unwrap(),
        b"science-advanced"
    );
}

#[test]
fn external_gate_requires_real_companion_capacity_and_named_cancellation_owner() {
    let temporary = tempfile::tempdir().unwrap();
    let raw = raw_reference(temporary.path(), "raw.txt", b"observed");
    let mut receipts = Vec::new();
    for node in REQUIRED_NODES {
        let eligible = node != "xbabe4";
        receipts.push(json_reference(
            temporary.path(),
            &format!("{node}.json"),
            &serde_json::json!(ScqCapabilityObservationV1 {
                schema: SCQ_CAPABILITY_RECEIPT_SCHEMA.to_owned(),
                node: node.to_owned(),
                observed_hostname: node.to_owned(),
                scheduler_binary_sha256: "a".repeat(64),
                eligible_for_dime_gpu: eligible,
                gpu_companion_core_reserve: Some(if eligible { 4 } else { 0 }),
                observed_at_unix_ms: 1_100,
                raw_output: raw.clone(),
            }),
        ));
    }
    let ownership = json_reference(
        temporary.path(),
        "ownership.json",
        &serde_json::json!(CancellationOwnershipEvidenceV1 {
            schema: CANCELLATION_OWNERSHIP_SCHEMA.to_owned(),
            owner: "DLGP prison-hold campaign".to_owned(),
            cancellation_message: "cancelled by operator".to_owned(),
            source: "events.jsonl".to_owned(),
            observed_at_unix_ms: 1_100,
            raw_events: raw,
        }),
    );
    let mut gate = DimeExternalActivationGateV1 {
        schema: EXTERNAL_ACTIVATION_GATE_SCHEMA.to_owned(),
        candidate_bundle_sha256: "b".repeat(64),
        generated_at_unix_ms: 1_000,
        expires_at_unix_ms: 2_000,
        scq_capability_receipts: receipts,
        cancellation_ownership: ownership,
    };
    gate.validate_at(temporary.path(), 1_200).unwrap();

    let mut broken: ScqCapabilityObservationV1 = gate.scq_capability_receipts[0]
        .load(temporary.path())
        .unwrap();
    broken.gpu_companion_core_reserve = None;
    gate.scq_capability_receipts[0] = json_reference(
        temporary.path(),
        "broken-capability.json",
        &serde_json::json!(broken),
    );
    assert!(gate.validate_at(temporary.path(), 1_200).is_err());
}

#[test]
fn candidate_review_gate_requires_independent_hash_bound_test_reproduction() {
    let temporary = tempfile::tempdir().unwrap();
    let bundle = "a".repeat(64);
    let integration_commit = "b".repeat(40);
    let manifest_file = "c".repeat(64);
    let manifest_canonical = "d".repeat(64);
    let mut acknowledgements = Vec::new();
    let mut test_references = Vec::new();
    for reviewer in ["codex", "claude"] {
        let run = DimeCandidateTestRunV1 {
            schema: CANDIDATE_TEST_RUN_SCHEMA.to_owned(),
            candidate_bundle_sha256: bundle.clone(),
            integration_commit: integration_commit.clone(),
            command: vec!["cargo".to_owned(), "test".to_owned()],
            exit_status: 0,
            passed: 14,
            failed: 0,
            ignored: 0,
            started_at_unix_ms: 1_100,
            finished_at_unix_ms: 1_500,
            stdout: raw_reference(
                temporary.path(),
                &format!("{reviewer}-stdout.txt"),
                b"14 passed",
            ),
            stderr: raw_reference(temporary.path(), &format!("{reviewer}-stderr.txt"), b""),
        };
        let test_reference = json_reference(
            temporary.path(),
            &format!("{reviewer}-test.json"),
            &serde_json::to_value(run).unwrap(),
        );
        test_references.push(test_reference.clone());
        let acknowledgement = DimeCandidateReviewAcknowledgementV1 {
            schema: CANDIDATE_REVIEW_ACK_SCHEMA.to_owned(),
            reviewer: reviewer.to_owned(),
            candidate_bundle_sha256: bundle.clone(),
            integration_commit: integration_commit.clone(),
            bundle_manifest_file_sha256: manifest_file.clone(),
            bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
            test_runs: vec![test_reference],
            reviewed_at_unix_ms: 1_600,
            raw_review: raw_reference(
                temporary.path(),
                &format!("{reviewer}-review.txt"),
                reviewer.as_bytes(),
            ),
        };
        acknowledgements.push(json_reference(
            temporary.path(),
            &format!("{reviewer}-ack.json"),
            &serde_json::to_value(acknowledgement).unwrap(),
        ));
    }
    let mut gate = DimeCandidateReviewGateV1 {
        schema: CANDIDATE_REVIEW_GATE_SCHEMA.to_owned(),
        issued_at_unix_ms: 1_000,
        expires_at_unix_ms: 3_000,
        candidate_bundle_sha256: bundle.clone(),
        integration_commit: integration_commit.clone(),
        bundle_manifest_file_sha256: manifest_file.clone(),
        bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
        acknowledgements,
    };
    gate.validate_at(temporary.path(), 2_000).unwrap();

    let duplicate_ack = DimeCandidateReviewAcknowledgementV1 {
        schema: CANDIDATE_REVIEW_ACK_SCHEMA.to_owned(),
        reviewer: "claude".to_owned(),
        candidate_bundle_sha256: bundle,
        integration_commit,
        bundle_manifest_file_sha256: manifest_file,
        bundle_manifest_canonical_json_sha256: manifest_canonical,
        test_runs: vec![test_references[0].clone()],
        reviewed_at_unix_ms: 1_600,
        raw_review: raw_reference(temporary.path(), "duplicate-review.txt", b"claude"),
    };
    gate.acknowledgements[1] = json_reference(
        temporary.path(),
        "duplicate-ack.json",
        &serde_json::to_value(duplicate_ack).unwrap(),
    );
    assert!(gate.validate_at(temporary.path(), 2_000).is_err());
}
