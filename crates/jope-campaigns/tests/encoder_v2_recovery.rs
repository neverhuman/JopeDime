use anyhow::bail;
use jope_dime_regressor::encoder_v2_authority::{
    eight_slot_sampling_weights, BenchmarkLaneMetrics, EncoderBenchmarkReportV2,
    EncoderLiftReportV2, LiftTaskResult, OptionalTaskReadinessV2, OptionalTaskRow,
    BENCHMARK_REPORT_SCHEMA, LIFT_REPORT_SCHEMA, OPTIONAL_READINESS_SCHEMA,
};
use jope_dime_regressor::encoder_v2_capacity::{
    CapacityDecision, EncoderCapacityObservation, CAPACITY_OBSERVATION_SCHEMA,
};
use jope_dime_regressor::encoder_v2_execution::{
    validate_output_bytes, EncoderAttemptReceiptV2, EncoderExecutionReceiptV2, EncoderStageTiming,
    ObservedEncoderRuntime, ATTEMPT_SCHEMA, EXECUTION_SCHEMA,
};
use jope_dime_regressor::encoder_v2_identity::{
    cache_key, ArtifactIdentity, BuildIdentity, DatasetReadIdentity, FoundationDatasetInputV2,
    FoundationEncoderIdentityV2, NumericalRuntimeIdentity, OutputContractIdentity,
    RuntimePackageIdentity, DATASET_INPUT_SCHEMA, ENCODER_IDENTITY_SCHEMA, OUTPUT_BYTES,
    OUTPUT_BYTE_ORDER, OUTPUT_DTYPE, OUTPUT_FEATURES, OUTPUT_LAYOUT,
};
use jope_dime_regressor::encoder_v2_store::{EncoderV2Store, PublishOutcome, PublishStep};
use jope_dime_regressor::encoder_v2_worker::{
    plan_chunks, ChunkMode, EncoderWorkItem, ResidentWorkerProgress,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::{
    immutable_flywheel_root_str, relocate_v1_quarantine, runtime_current_root_str,
    MutableScriptObservation, QuarantineFileIdentity, QuarantinePolicyFlags,
    QuarantinedEncodingTree, ReferenceExclusionProof, RollbackCapsuleV1, RollbackNodeReceipt,
    RollbackUnitIdentity, V1EncodingQuarantineV1, REQUIRED_NODES, ROLLBACK_CAPSULE_SCHEMA,
    ROLLBACK_RUNTIME_BUNDLE_SHA256, V1_QUARANTINE_SCHEMA,
};
use jope_dime_regressor::selfplay_data::canonical_json_sha256;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn artifact(path: &str, byte: u8) -> ArtifactIdentity {
    ArtifactIdentity {
        path: path.to_owned(),
        sha256: digest(byte),
        size_bytes: u64::from(byte) + 1,
    }
}

fn encoder() -> FoundationEncoderIdentityV2 {
    FoundationEncoderIdentityV2 {
        schema: ENCODER_IDENTITY_SCHEMA.to_owned(),
        bundle_sha256: digest(1),
        prime_source_bundle_sha256: digest(2),
        prime_package_tree_sha256: digest(3),
        protocol_name: "foundation-pooled-vector-v2".to_owned(),
        protocol: artifact("protocol/encoder.v2.json", 4),
        python_implementation_tree_sha256: digest(5),
        python_encoder: artifact("python/foundation.py", 6),
        python_support: vec![artifact("python/support.py", 7)],
        wrapper: artifact("pytorch/encoding/wrapper.py", 8),
        batch_driver: artifact("pytorch/encoding/batch.py", 9),
        native_supervisor: artifact("bin/jope_encoder_supervisor", 10),
        build: BuildIdentity {
            repository_commit: "ab".repeat(20),
            compiler: "rustc-test".to_owned(),
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            profile: "release".to_owned(),
            enabled_features: vec!["cuda".to_owned()],
            lockfiles: vec![artifact("Cargo.lock", 11)],
        },
        model_config: artifact("weights/config.json", 12),
        model_manifests: vec![artifact("weights/manifest.json", 13)],
        model_weights: vec![artifact("weights/model.safetensors", 14)],
        prime_support_tree_sha256: digest(15),
        prime_support: vec![artifact("prime/support.py", 16)],
        preprocessing_sha256: digest(17),
        seed: 23,
        runtime: NumericalRuntimeIdentity {
            requested_backend: "cuda".to_owned(),
            required_actual_backend: "cuda".to_owned(),
            device_class: "sm_89".to_owned(),
            gpu_compute_capability: Some("8.9".to_owned()),
            cuda_version: Some("13.test".to_owned()),
            driver_version: Some("999.test".to_owned()),
            deterministic: true,
            deterministic_flags: BTreeMap::from([
                ("cudnn_benchmark_disabled".to_owned(), true),
                ("deterministic_algorithms".to_owned(), true),
            ]),
            input_dtype: "float32".to_owned(),
            accumulation_dtype: "float32".to_owned(),
            batch_size: 16,
            thread_count: 4,
            packages: BTreeMap::from([
                (
                    "cuda".to_owned(),
                    RuntimePackageIdentity {
                        version: "13.test".to_owned(),
                        package_tree_sha256: digest(18),
                    },
                ),
                (
                    "pytorch".to_owned(),
                    RuntimePackageIdentity {
                        version: "2.test".to_owned(),
                        package_tree_sha256: digest(19),
                    },
                ),
            ]),
            kernel_policy_sha256: digest(20),
            memory_policy_sha256: digest(21),
        },
        output: OutputContractIdentity {
            dtype: OUTPUT_DTYPE.to_owned(),
            vector_count: 1,
            feature_count: OUTPUT_FEATURES,
            byte_length: OUTPUT_BYTES,
            byte_order: OUTPUT_BYTE_ORDER.to_owned(),
            finite_only: true,
            feature_layout: OUTPUT_LAYOUT.to_owned(),
            format_sha256: digest(22),
        },
    }
}

fn input() -> FoundationDatasetInputV2 {
    FoundationDatasetInputV2 {
        schema: DATASET_INPUT_SCHEMA.to_owned(),
        task: "multiclass".to_owned(),
        source_family: "family:test".to_owned(),
        train_sha256: digest(23),
        train_size_bytes: 24,
        feature_target_schema_sha256: digest(24),
        row_count: 128,
        column_count: 12,
        train_only: true,
        files_read: vec![DatasetReadIdentity {
            purpose: "train_data".to_owned(),
            artifact: ArtifactIdentity {
                path: "dataset/train.csv".to_owned(),
                sha256: digest(23),
                size_bytes: 24,
            },
        }],
        time_series_reduction: None,
    }
}

fn output(value: f32) -> Vec<u8> {
    (0..OUTPUT_FEATURES)
        .flat_map(|_| value.to_le_bytes())
        .collect()
}

fn execution_and_attempt(
    encoder: &FoundationEncoderIdentityV2,
    input: &FoundationDatasetInputV2,
    bytes: &[u8],
    job: &str,
    generated: u64,
) -> (EncoderExecutionReceiptV2, EncoderAttemptReceiptV2) {
    let output_sha256 = validate_output_bytes(bytes).unwrap();
    let packages = encoder
        .runtime
        .packages
        .iter()
        .map(|(name, package)| (name.clone(), package.package_tree_sha256.clone()))
        .collect();
    let execution = EncoderExecutionReceiptV2 {
        schema: EXECUTION_SCHEMA.to_owned(),
        generated_at_unix_ms: generated,
        freshness_budget_seconds: 300,
        scq_job_identity_sha256: job.to_owned(),
        encoder_identity_sha256: encoder.digest().unwrap(),
        input_identity_sha256: input.digest().unwrap(),
        cache_key_sha256: cache_key(encoder, input).unwrap(),
        declared_inputs: input
            .files_read
            .iter()
            .map(|read| read.artifact.clone())
            .collect(),
        sandbox_policy_sha256: digest(25),
        child_receipt_sha256: digest(26),
        runtime_artifacts_sha256: digest(27),
        observed: ObservedEncoderRuntime {
            actual_backend: "cuda".to_owned(),
            selected_device: "cuda:0".to_owned(),
            device_class: "sm_89".to_owned(),
            gpu_compute_capability: Some("8.9".to_owned()),
            cuda_version: Some("13.test".to_owned()),
            driver_version: Some("999.test".to_owned()),
            pytorch_version: "2.test".to_owned(),
            package_tree_sha256: packages,
            input_dtype: "float32".to_owned(),
            accumulation_dtype: "float32".to_owned(),
            deterministic_flags: encoder.runtime.deterministic_flags.clone(),
            loaded_weights: encoder.model_weights.clone(),
            peak_rss_bytes: 1_024,
            peak_vram_bytes: Some(2_048),
        },
        output_sha256: output_sha256.clone(),
        output_vector_count: 1,
        output_feature_count: OUTPUT_FEATURES,
        output_size_bytes: OUTPUT_BYTES,
        output_finite: true,
        extra_output_count: 0,
        timing: EncoderStageTiming {
            input_hash_ns: 1,
            preprocess_ns: 1,
            expert_load_ns: 1,
            kernel_ns: 1,
            output_hash_ns: 1,
            verification_ns: 1,
            total_ns: 10,
        },
    };
    let attempt = EncoderAttemptReceiptV2 {
        schema: ATTEMPT_SCHEMA.to_owned(),
        generated_at_unix_ms: generated,
        freshness_budget_seconds: 300,
        node: "xbabe2".to_owned(),
        scq_job_identity_sha256: job.to_owned(),
        cache_key_sha256: execution.cache_key_sha256.clone(),
        status: "succeeded".to_owned(),
        failure_class: None,
        execution_receipt_sha256: Some(execution.digest().unwrap()),
        output_sha256: Some(output_sha256),
        queue_wait_ns: 1,
        total_ns: 20,
    };
    (execution, attempt)
}

#[test]
fn identity_roles_backend_and_train_test_boundary_fail_closed() {
    let encoder = encoder();
    let input = input();
    let stable = cache_key(&encoder, &input).unwrap();

    let mut changed = encoder.clone();
    changed.wrapper.sha256 = digest(30);
    assert_ne!(stable, cache_key(&changed, &input).unwrap());
    let mut changed = encoder.clone();
    changed.runtime.memory_policy_sha256 = digest(31);
    assert_ne!(stable, cache_key(&changed, &input).unwrap());
    let mut changed = encoder.clone();
    changed.output.byte_order = "big-endian".to_owned();
    assert!(changed.validate().is_err());

    let mut leaked = input.clone();
    leaked.files_read[0].purpose = "dataset_metadata".to_owned();
    leaked.files_read[0].artifact.path = "dataset/test.csv".to_owned();
    assert!(leaked.validate().is_err());
    let mut unknown = serde_json::to_value(encoder).unwrap();
    unknown["fallback_allowed"] = Value::Bool(true);
    assert!(serde_json::from_value::<FoundationEncoderIdentityV2>(unknown).is_err());
}

#[test]
fn output_and_execution_receipts_reject_nonfinite_fallback_and_package_drift() {
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let (execution, _) = execution_and_attempt(&encoder, &input, &bytes, &digest(32), 1_000);
    execution.validate_for(&encoder, &input, 1_000).unwrap();

    let mut fallback = execution.clone();
    fallback.observed.actual_backend = "cpu".to_owned();
    assert!(fallback.validate_for(&encoder, &input, 1_000).is_err());
    let mut drift = execution;
    drift
        .observed
        .package_tree_sha256
        .insert("pytorch".to_owned(), digest(33));
    assert!(drift.validate_for(&encoder, &input, 1_000).is_err());

    let mut nonfinite = bytes;
    nonfinite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(validate_output_bytes(&nonfinite).is_err());
}

#[test]
fn object_store_converges_duplicates_and_rejects_nondeterminism() {
    let directory = tempfile::tempdir().unwrap();
    let store = EncoderV2Store::new(directory.path());
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let job1 = digest(40);
    let (execution1, attempt1) = execution_and_attempt(&encoder, &input, &bytes, &job1, 1_000);
    let first = store
        .publish_success(
            "tx-1",
            &encoder,
            &input,
            &execution1,
            &attempt1,
            &bytes,
            &job1,
            1_000,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap();
    assert!(matches!(first, PublishOutcome::Published(_)));

    let job2 = digest(41);
    let (execution2, attempt2) = execution_and_attempt(&encoder, &input, &bytes, &job2, 1_001);
    let duplicate = store
        .publish_success(
            "tx-2",
            &encoder,
            &input,
            &execution2,
            &attempt2,
            &bytes,
            &job2,
            1_001,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap();
    assert!(matches!(duplicate, PublishOutcome::Converged(_)));

    let conflicting = output(1.0);
    let job3 = digest(42);
    let (execution3, attempt3) =
        execution_and_attempt(&encoder, &input, &conflicting, &job3, 1_002);
    let error = store
        .publish_success(
            "tx-3",
            &encoder,
            &input,
            &execution3,
            &attempt3,
            &conflicting,
            &job3,
            1_002,
            None,
            None,
            |_| Ok(()),
        )
        .unwrap_err();
    assert!(error.to_string().contains("nondeterminism"));
    assert_eq!(
        fs::read_dir(directory.path().join("conflicts"))
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn store_recovers_persistent_lock_inode_and_rejects_every_v1_import() {
    let directory = tempfile::tempdir().unwrap();
    let store = EncoderV2Store::new(directory.path());
    store.initialize().unwrap();
    let encoder = encoder();
    let input = input();
    let bytes = output(0.0);
    let job = digest(79);
    let key = cache_key(&encoder, &input).unwrap();
    fs::write(
        directory.path().join("locks").join(format!("{key}.lock")),
        b"stale owner metadata",
    )
    .unwrap();
    let (execution, attempt) = execution_and_attempt(&encoder, &input, &bytes, &job, 1_000);
    let published = store.publish_success(
        "stale-lock-recovery",
        &encoder,
        &input,
        &execution,
        &attempt,
        &bytes,
        &job,
        1_000,
        None,
        None,
        |_| Ok(()),
    );
    assert!(published.is_ok(), "{published:?}");

    let rejected = store.publish_success(
        "v1-import-rejected",
        &encoder,
        &input,
        &execution,
        &attempt,
        &bytes,
        &job,
        1_000,
        Some("legacy-v1-cache"),
        None,
        |_| Ok(()),
    );
    assert!(rejected
        .unwrap_err()
        .to_string()
        .contains("may never be imported"));
}

#[test]
fn publication_crashes_leave_only_valid_bindings_or_no_binding() {
    for (index, crash_step) in [
        PublishStep::Staged,
        PublishStep::ObjectSynced,
        PublishStep::ObjectPublished,
        PublishStep::MetadataPublished,
        PublishStep::BindingPublished,
        PublishStep::AttemptPublished,
        PublishStep::PointerPublished,
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempfile::tempdir().unwrap();
        let store = EncoderV2Store::new(directory.path());
        let encoder = encoder();
        let input = input();
        let bytes = output(0.0);
        let job = digest(50 + index as u8);
        let (execution, attempt) = execution_and_attempt(&encoder, &input, &bytes, &job, 1_000);
        let error = store
            .publish_success(
                &format!("crash-{index}"),
                &encoder,
                &input,
                &execution,
                &attempt,
                &bytes,
                &job,
                1_000,
                None,
                None,
                |step| {
                    if step == crash_step {
                        bail!("injected crash")
                    }
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("injected crash"));

        let recovered = store.publish_success(
            &format!("recover-{index}"),
            &encoder,
            &input,
            &execution,
            &attempt,
            &bytes,
            &job,
            1_000,
            None,
            None,
            |_| Ok(()),
        );
        assert!(recovered.is_ok(), "{crash_step:?}: {recovered:?}");
    }
}

fn quarantine_tree(index: usize, state: &str) -> QuarantinedEncodingTree {
    let file = QuarantineFileIdentity {
        path: "cache-manifest.v1.json".to_owned(),
        file_type: "regular".to_owned(),
        size_bytes: 10,
        mode: 0o444,
        sha256: digest(index as u8 + 1),
    };
    let mut tree = QuarantinedEncodingTree {
        identity: format!("incident-{index:02}"),
        state: state.to_owned(),
        original_path: format!("/store/{state}/incident-{index:02}"),
        quarantine_path: format!("/store/incident-quarantine/{state}/incident-{index:02}"),
        files: vec![file],
        aggregate_directory_sha256: digest(200),
    };
    tree.aggregate_directory_sha256 = tree.computed_directory_sha256().unwrap();
    tree
}

fn quarantine() -> V1EncodingQuarantineV1 {
    let trees = (0..18)
        .map(|index| quarantine_tree(index, if index < 11 { "complete" } else { "partial" }))
        .collect();
    V1EncodingQuarantineV1 {
        schema: V1_QUARANTINE_SCHEMA.to_owned(),
        incident_id: "encoder-v1-20260806".to_owned(),
        generated_at_unix_ms: 300,
        freshness_budget_seconds: 86_400,
        discovered_at_unix_ms: 100,
        contained_at_unix_ms: 200,
        governing_freeze_entry_sha256: digest(210),
        responsible_execution_paths: vec![artifact("scripts/incident.py", 211)],
        trees,
        policy: QuarantinePolicyFlags {
            catalog_allowed: false,
            training_allowed: false,
            benchmark_allowed: false,
            v2_import_allowed: false,
            gc_allowed: false,
        },
        mutable_scripts: vec![
            MutableScriptObservation {
                script_name: "multiclass_drain.py".to_owned(),
                sha256: digest(212),
                observed_paths: vec![
                    format!(
                        "xbabe1:{}",
                        paths::research_script("multiclass_drain.py").display()
                    ),
                    format!(
                        "xbabe3:{}",
                        paths::research_script("multiclass_drain.py").display()
                    ),
                ],
            },
            MutableScriptObservation {
                script_name: "test_exact_expansion.py".to_owned(),
                sha256: digest(213),
                observed_paths: vec![
                    format!(
                        "xbabe1:{}",
                        paths::research_script("test_exact_expansion.py").display()
                    ),
                    format!(
                        "xbabe3:{}",
                        paths::research_script("test_exact_expansion.py").display()
                    ),
                ],
            },
        ],
        non_consumption: [
            "catalog",
            "corpus",
            "packed_corpus",
            "observations",
            "accepted_model",
            "runtime_pointer",
            "model_pointer",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, authority)| ReferenceExclusionProof {
            authority: authority.to_owned(),
            authority_sha256: digest(220 + index as u8),
            quarantined_reference_count: 0,
        })
        .collect(),
    }
}

#[test]
fn quarantine_is_exact_and_denied_for_every_consumption_class() {
    let quarantine = quarantine();
    quarantine.validate().unwrap();
    assert!(quarantine.digest().is_ok());
    for operation in [
        "discovery",
        "catalog",
        "training",
        "benchmark",
        "V2 import",
        "GC",
    ] {
        assert!(quarantine
            .require_operation_allowed("incident-00", operation)
            .is_err());
    }
    let mut permissive = quarantine;
    permissive.policy.gc_allowed = true;
    assert!(permissive.validate().is_err());
}

fn filesystem_quarantine(root: &std::path::Path) -> V1EncodingQuarantineV1 {
    let mut index = quarantine();
    fs::create_dir_all(root.join("incident-quarantine/complete")).unwrap();
    fs::create_dir_all(root.join("incident-quarantine/partial")).unwrap();
    for (number, tree) in index.trees.iter_mut().enumerate() {
        let source = root.join("source").join(&tree.identity);
        fs::create_dir_all(&source).unwrap();
        let contents = [number as u8];
        let file = source.join("cache-manifest.v1.json");
        fs::write(&file, contents).unwrap();
        tree.original_path = source.to_string_lossy().into_owned();
        tree.quarantine_path = root
            .join("incident-quarantine")
            .join(&tree.state)
            .join(&tree.identity)
            .to_string_lossy()
            .into_owned();
        tree.files[0].size_bytes = fs::metadata(&file).unwrap().len();
        tree.files[0].mode = fs::metadata(&file).unwrap().permissions().mode() & 0o7777;
        tree.files[0].sha256 = hex::encode(Sha256::digest(contents));
        tree.aggregate_directory_sha256 = tree.computed_directory_sha256().unwrap();
    }
    index
}

#[test]
fn quarantine_relocation_is_same_filesystem_atomic_and_stops_before_a_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let index = filesystem_quarantine(directory.path());
    let receipts = relocate_v1_quarantine(&index).unwrap();
    assert_eq!(receipts.len(), 18);
    assert!(receipts.iter().all(|receipt| receipt.atomic_rename));
    assert!(index
        .trees
        .iter()
        .all(|tree| !std::path::Path::new(&tree.original_path).exists()
            && std::path::Path::new(&tree.quarantine_path).is_dir()));

    let mismatch_root = tempfile::tempdir().unwrap();
    let mismatch = filesystem_quarantine(mismatch_root.path());
    fs::write(
        std::path::Path::new(&mismatch.trees[17].original_path).join("cache-manifest.v1.json"),
        b"drift",
    )
    .unwrap();
    assert!(relocate_v1_quarantine(&mismatch).is_err());
    assert!(mismatch
        .trees
        .iter()
        .all(|tree| std::path::Path::new(&tree.original_path).is_dir()));
}

fn rollback_unit(name: &str, evaluator: bool, byte: u8) -> RollbackUnitIdentity {
    RollbackUnitIdentity {
        unit: format!("{name}.service"),
        unit_file_sha256: digest(byte),
        resolved_effective_unit_sha256: digest(byte + 1),
        exec_start_argv: vec![format!(
            "{runtime_current_root_str}/runtime/bin/{name}",
            runtime_current_root_str = runtime_current_root_str()
        )],
        working_directory: immutable_flywheel_root_str().to_owned(),
        drop_ins: Vec::new(),
        evaluator,
    }
}

fn rollback() -> RollbackCapsuleV1 {
    let accepted = json!({"checkpoint_sha256": digest(230)});
    let cursors = json!({"ledger": 42, "observations": 99});
    let units = vec![
        rollback_unit("corpus_eval", true, 1),
        rollback_unit("jope_reporting", false, 3),
    ];
    let mut capsule = RollbackCapsuleV1 {
        schema: ROLLBACK_CAPSULE_SCHEMA.to_owned(),
        generated_at_unix_ms: 1_000,
        freshness_budget_seconds: 300,
        runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
        runtime_bundle_manifest_sha256: digest(231),
        unit_overlay_sha256: digest(232),
        units,
        service_manifest_sha256: digest(233),
        evaluator_retention_index_sha256: digest(234),
        accepted_model_pointer_snapshot_sha256: canonical_json_sha256(&accepted).unwrap(),
        accepted_model_pointer_snapshot: accepted,
        model_pointer_sha256: digest(235),
        database_cursor_snapshot_sha256: canonical_json_sha256(&cursors).unwrap(),
        database_cursor_snapshot: cursors,
        node_receipts: Vec::new(),
        absolute_path_audit_receipt_sha256: digest(236),
        simulated_install_receipt_sha256: digest(237),
        simulated_install_passed: true,
    };
    capsule.unit_overlay_sha256 = capsule.computed_unit_overlay_sha256().unwrap();
    capsule.node_receipts = REQUIRED_NODES
        .into_iter()
        .map(|node| RollbackNodeReceipt {
            node: node.to_owned(),
            runtime_bundle_sha256: capsule.runtime_bundle_sha256.clone(),
            unit_overlay_sha256: capsule.unit_overlay_sha256.clone(),
            runtime_file_count: 315,
            unit_count: capsule.units.len() as u64,
            verified_at_unix_ms: 1_000,
        })
        .collect();
    capsule
}

#[test]
fn composite_rollback_rejects_runtime_unit_staleness_and_inventory_drift() {
    let capsule = rollback();
    capsule.validate_at(1_000).unwrap();
    assert!(capsule.digest_at(1_000).is_ok());

    let mut mismatch = capsule.clone();
    mismatch.node_receipts[0].unit_overlay_sha256 = digest(240);
    assert!(mismatch.validate_at(1_000).is_err());
    assert!(capsule.validate_at(302_000).is_err());

    let mut no_evaluator = capsule.clone();
    no_evaluator.units[0].evaluator = false;
    no_evaluator.unit_overlay_sha256 = no_evaluator.computed_unit_overlay_sha256().unwrap();
    assert!(no_evaluator.validate_at(1_000).is_err());

    let mut observed = capsule.units.clone();
    observed[0]
        .drop_ins
        .push(artifact("drop-ins/unknown.conf", 241));
    assert!(capsule.verify_effective_units(&observed).is_err());
}

fn readiness_row(task: &str, ready: bool) -> OptionalTaskRow {
    let multiclass = task == "multiclass";
    OptionalTaskRow {
        task: task.to_owned(),
        state: if ready { "shadow_ready" } else { "collecting" }.to_owned(),
        distinct_dataset_identities: if multiclass { 500 } else { 42 },
        minimum_required: if multiclass { 500 } else { 42 },
        minimum_rolling_origins_per_dataset: (!multiclass).then_some(3),
        valid_v2_cache_bindings: if multiclass { 500 } else { 42 },
        valid_context_transform_bindings: if multiclass { 500 } else { 42 },
        evaluator_retained: true,
        evaluator_sha256: digest(242),
        task_valid_head_count: 1,
        robust_bank_sha256: digest(243),
        fresh_unbiased_shadow_evidence: true,
        causal_reduction_verified: (!multiclass).then_some(true),
        no_future_leakage: (!multiclass).then_some(true),
        primary_metric: if multiclass {
            "macro_f1"
        } else {
            "negative_mase"
        }
        .to_owned(),
        native_error_metric: if multiclass {
            "one_minus_macro_f1"
        } else {
            "mase"
        }
        .to_owned(),
        training_allowed: ready,
        blockers: if ready {
            Vec::new()
        } else {
            vec!["dataset_floor".to_owned()]
        },
    }
}

#[test]
fn optional_task_floors_and_sampler_ratios_are_task_specific() {
    let readiness = OptionalTaskReadinessV2 {
        schema: OPTIONAL_READINESS_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(244)],
        tasks: vec![
            readiness_row("multiclass", true),
            readiness_row("timeseries", true),
        ],
    };
    readiness.validate().unwrap();
    let all = eight_slot_sampling_weights(true, true);
    assert_eq!(all["regression"].numerator, 6);
    assert_eq!(all["multiclass"].numerator, 2);
    let none = eight_slot_sampling_weights(false, false);
    assert_eq!(none["regression"].numerator, 8);
    assert_eq!(none["multiclass"].numerator, 0);

    let mut generic_floor = readiness;
    generic_floor.tasks[1].distinct_dataset_identities = 500;
    generic_floor.tasks[1].valid_v2_cache_bindings = 500;
    generic_floor.tasks[1].valid_context_transform_bindings = 500;
    assert!(generic_floor.validate().is_err());
}

fn lane(throughput: f64, p95: f64) -> BenchmarkLaneMetrics {
    BenchmarkLaneMetrics {
        samples: 384,
        p50_latency_ms: p95 / 2.0,
        p90_latency_ms: p95 * 0.9,
        p95_latency_ms: p95,
        p99_latency_ms: p95 * 1.1,
        throughput_per_gpu_slot: throughput,
        cpu_hours: 1.0,
        load_time_ms: 1.0,
        preprocess_time_ms: 1.0,
        kernel_time_ms: 1.0,
        verification_publication_ms: 1.0,
        cache_lookup_ms: 1.0,
        peak_ram_bytes: 1,
        peak_vram_bytes: 1,
        failures: 0,
    }
}

#[test]
fn speed_gate_has_no_sub_three_x_exception() {
    let mut report = EncoderBenchmarkReportV2 {
        schema: BENCHMARK_REPORT_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(245)],
        benchmark_plan_sha256: digest(246),
        baseline: lane(100.0, 100.0),
        candidate: lane(300.0, 50.0),
        cache_hit: lane(10_000.0, 2.0),
        paired_bootstrap_throughput_ratio_lower_95: 2.5,
        control_plane_overhead_fraction: 0.04,
        output_drift_count: 0,
        fallback_count: 0,
        backend_mismatch_count: 0,
        collision_count: 0,
        oom_as_success_count: 0,
        useful_work_green: true,
        protected_reserves_green: true,
        companion_cpu_credits_green: true,
        release_speed_gate_passed: true,
        blockers: Vec::new(),
    };
    report.validate().unwrap();
    report.candidate.throughput_per_gpu_slot = 299.9;
    report.release_speed_gate_passed = false;
    report.blockers = vec!["throughput_below_3x".to_owned()];
    report.validate().unwrap();
}

#[test]
fn lift_requires_noninferiority_speed_or_quality_and_all_common_gates() {
    let task = |name: &str| LiftTaskResult {
        task: name.to_owned(),
        corrected_primary_delta_lower_95: matches!(name, "binary" | "multiclass").then_some(0.006),
        native_error_ratio_upper_95: matches!(name, "regression" | "timeseries").then_some(0.99),
        strict_win_rate_lower_95: 0.51,
        p95_latency_ratio: 1.0,
        failure_rate_delta: 0.0,
        bounded_metric_worst_delta: Some(0.0),
        native_error_worst_ratio: Some(1.0),
        directional_agreement: true,
        typed_absence: None,
    };
    let mut report = EncoderLiftReportV2 {
        schema: LIFT_REPORT_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(247)],
        experiment_sha256: digest(248),
        benchmark_report_sha256: digest(249),
        speed_gate_passed: true,
        tasks: ["regression", "binary", "multiclass", "timeseries"]
            .into_iter()
            .map(task)
            .collect(),
        leakage_detected: false,
        identity_collision_detected: false,
        reproducible: true,
        advancement_allowed: true,
        blockers: Vec::new(),
    };
    report.validate().unwrap();
    report.tasks[2].bounded_metric_worst_delta = Some(-0.021);
    report.advancement_allowed = false;
    report.blockers = vec!["catastrophic_multiclass_loss".to_owned()];
    report.validate().unwrap();
}

fn capacity() -> EncoderCapacityObservation {
    EncoderCapacityObservation {
        schema: CAPACITY_OBSERVATION_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(252)],
        current_workers: 1,
        eligible_nodes: 4,
        successful_outputs_at_rung: 128,
        green_minutes_at_rung: 30,
        collision_count: 0,
        fallback_count: 0,
        oom_as_success_count: 0,
        unexplained_drift_count: 0,
        ram_headroom_ratio: 1.5,
        vram_headroom_ratio: 1.5,
        cpu_psi: 0.25,
        memory_psi: 0.05,
        io_psi: 0.10,
        free_memory_fraction: 0.15,
        supply_status_v2_fresh_green: true,
        useful_work_at_target: true,
        protected_complete_waves: 2,
        companion_cpu_credits: 4,
        trainer_active: false,
        critical_runway: false,
        blockers: Vec::new(),
    }
}

#[test]
fn capacity_advances_only_after_both_proof_thresholds_and_preempts_for_recovery() {
    assert_eq!(
        capacity().decide_at(1).unwrap(),
        CapacityDecision::Advance { workers: 2 }
    );
    let mut early = capacity();
    early.green_minutes_at_rung = 29;
    assert_eq!(
        early.decide_at(1).unwrap(),
        CapacityDecision::Hold { workers: 1 }
    );
    let mut saturated = capacity();
    saturated.cpu_psi = 0.251;
    assert_eq!(
        saturated.decide_at(1).unwrap(),
        CapacityDecision::GracefulPreempt {
            reason: "resource_pressure"
        }
    );
    let mut runway = capacity();
    runway.useful_work_at_target = false;
    assert_eq!(
        runway.decide_at(1).unwrap(),
        CapacityDecision::GracefulPreempt {
            reason: "recovery_runway"
        }
    );
    assert!(capacity().decide_at(300_002).is_err());
}

#[test]
fn resident_worker_chunk_ladder_is_exact_and_redoes_at_most_one_dataset() {
    let tasks = ["regression", "binary", "multiclass", "timeseries"];
    let items = tasks
        .into_iter()
        .flat_map(|task| (0..8).map(move |index| (task, index)))
        .enumerate()
        .map(|(number, (task, index))| EncoderWorkItem {
            input_identity_sha256: format!("{:064x}", number + 1),
            task: task.to_owned(),
            rows: 128 + index,
            columns: 16,
            input_bytes: 4_096,
            estimated_vram_bytes: 1_024,
        })
        .collect::<Vec<_>>();
    let chunks = plan_chunks(items, ChunkMode::Proof, 2_048).unwrap();
    assert_eq!(
        chunks.iter().map(|chunk| chunk.items.len()).sum::<usize>(),
        32
    );
    assert!(plan_chunks(Vec::new(), ChunkMode::Replay, 2_048).is_err());

    let progress = ResidentWorkerProgress {
        worker_capsule_sha256: digest(250),
        loaded_expert_sha256: digest(251),
        committed_input_identities: vec![digest(1), digest(2)],
        active_input_identity: Some(digest(3)),
    };
    progress.validate().unwrap();
    assert_eq!(progress.maximum_redo_datasets(), 1);
}
