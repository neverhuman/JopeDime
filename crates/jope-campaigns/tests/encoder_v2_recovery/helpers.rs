use super::*;

pub(crate) fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

pub(crate) fn artifact(path: &str, byte: u8) -> ArtifactIdentity {
    ArtifactIdentity {
        path: path.to_owned(),
        sha256: digest(byte),
        size_bytes: u64::from(byte) + 1,
    }
}

pub(crate) fn encoder() -> FoundationEncoderIdentityV2 {
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

pub(crate) fn input() -> FoundationDatasetInputV2 {
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

pub(crate) fn output(value: f32) -> Vec<u8> {
    (0..OUTPUT_FEATURES)
        .flat_map(|_| value.to_le_bytes())
        .collect()
}

pub(crate) fn execution_and_attempt(
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

pub(crate) fn quarantine_tree(index: usize, state: &str) -> QuarantinedEncodingTree {
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

pub(crate) fn quarantine() -> V1EncodingQuarantineV1 {
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

pub(crate) fn filesystem_quarantine(root: &std::path::Path) -> V1EncodingQuarantineV1 {
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

pub(crate) fn rollback_unit(name: &str, evaluator: bool, byte: u8) -> RollbackUnitIdentity {
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

pub(crate) fn rollback() -> RollbackCapsuleV1 {
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

pub(crate) fn readiness_row(task: &str, ready: bool) -> OptionalTaskRow {
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

pub(crate) fn lane(throughput: f64, p95: f64) -> BenchmarkLaneMetrics {
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

pub(crate) fn capacity() -> EncoderCapacityObservation {
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
