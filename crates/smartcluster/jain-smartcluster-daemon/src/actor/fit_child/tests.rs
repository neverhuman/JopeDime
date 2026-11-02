use super::*;
fn candidate_case(
    kind: &str,
    stage: &str,
    evaluations: u64,
    index: u32,
    partition: &str,
    candidate_generation: u64,
    tamper: bool,
) -> Result<(), ServerError> {
    use jain_smartcluster::jope::{TaskKind, TaskManifest};
    use jain_smartcluster_core::{IdempotencyKey, ProjectId, ResourceRequest, TenantId};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let candidate = serde_json::json!({"schema":"jope.native_candidate.v1", "run":"test-run",
        "generation":candidate_generation,"context":{"partition":partition}});
    let candidate_bytes = serde_json::to_vec(&candidate).unwrap();
    fs::write(root.join("candidate.json"), &candidate_bytes).unwrap();
    let genome = serde_json::json!({"schema":"jope.native_genome_task.v1", "stage":stage,
        "evaluations":evaluations,"seconds":30,"direct_parent":if stage=="direct" {serde_json::Value::Null} else {serde_json::json!({"path":"parent.json","sha256":"a".repeat(64)})},
        "candidate":{"path":"candidate.json","sha256":digest(&candidate_bytes)}});
    let genome_bytes = serde_json::to_vec(&genome).unwrap();
    fs::write(root.join("genome.json"), &genome_bytes).unwrap();
    let input = serde_json::json!({"schema":"jope.smartcluster_input.v1","root":root,
        "operation":{"kind":kind,"manifest":"genome.json","sha256":digest(&genome_bytes)}});
    let input_bytes = serde_json::to_vec(&input).unwrap();
    fs::write(root.join("task.input.json"), &input_bytes).unwrap();
    let now = now_timestamp();
    let task = TaskManifest {
        schema: "jope.smartcluster_task.v1".into(),
        run: "test-run".into(),
        generation: 1,
        task_id: "candidate".into(),
        kind: if kind == "DirectEvaluation" {
            TaskKind::GpSearch
        } else {
            TaskKind::Continuation
        },
        input_manifest_sha256: digest(&input_bytes),
        dependencies: std::collections::BTreeMap::from([(
            "candidate.json".into(),
            digest(&candidate_bytes),
        )]),
        parent_checkpoint_sha256: None,
        corpus_sha256: None,
        idempotency_key: IdempotencyKey::new("candidate").unwrap(),
        deadline: Timestamp::from(u64::MAX - 1),
        retry_allowance: 0,
        resources: ResourceRequest::cpu_and_ram(1000, 1073741824),
        expected_receipt_id: "candidate/result".into(),
        checkpoint_allowance_ms: None,
    };
    let task_bytes = serde_json::to_vec(&task).unwrap();
    let path = root.join("task.json");
    fs::write(&path, &task_bytes).unwrap();
    let spec = jain_smartcluster::jope::job_spec(
        &path,
        &digest(&task_bytes),
        TenantId::new("jope").unwrap(),
        ProjectId::new("qualification").unwrap(),
        now,
    )
    .unwrap();
    let id = JobId::new();
    let job = JobRecord::queued(
        id,
        spec,
        Principal::System {
            name: jain_smartcluster_core::SystemPrincipal::new("jope").unwrap(),
        },
        now,
    );
    let attempt = FencedAttempt::new(
        id,
        jain_smartcluster_core::AttemptNumber::new(1).unwrap(),
        jain_smartcluster_core::FenceToken::new(1).unwrap(),
    );
    let request = AttachTrustedFitChildRequest {
        authority_attempt: attempt,
        target_attempt: attempt,
        child_pid: 42,
        child_birth_ticks: 1,
        child_boot_id: "11111111-1111-1111-1111-111111111111".into(),
        task_manifest_sha256: digest(&task_bytes),
        descriptor_sha256: "b".repeat(64),
        candidate_index: index,
        idempotency_key: IdempotencyKey::new("candidate").unwrap(),
        child_executable_sha256: "c".repeat(64),
    };
    if tamper {
        fs::write(root.join("candidate.json"), b"{}").unwrap();
    }
    validate_candidate(&job, &request)
}
#[test]
fn genome_child_admission_binds_budget_stage_partition_and_original_bytes() {
    assert!(candidate_case("DirectEvaluation", "direct", 1, 0, "meta_train", 1, false).is_ok());
    assert!(candidate_case(
        "SeededContinuation",
        "continuation",
        16,
        15,
        "meta_train",
        1,
        false
    )
    .is_ok());
    for (kind, stage, n, index, partition, generation, tamper) in [
        ("DirectEvaluation", "direct", 2, 0, "meta_train", 1, false),
        (
            "DirectEvaluation",
            "continuation",
            1,
            0,
            "meta_train",
            1,
            false,
        ),
        (
            "SeededContinuation",
            "continuation",
            16,
            16,
            "meta_train",
            1,
            false,
        ),
        (
            "SeededContinuation",
            "continuation",
            0,
            0,
            "meta_train",
            1,
            false,
        ),
        (
            "SeededContinuation",
            "continuation",
            1001,
            0,
            "meta_train",
            1,
            false,
        ),
        ("DirectEvaluation", "direct", 1, 0, "sealed_test", 1, false),
        ("DirectEvaluation", "direct", 1, 0, "meta_train", 2, false),
        ("DirectEvaluation", "direct", 1, 0, "meta_train", 1, true),
    ] {
        assert!(candidate_case(kind, stage, n, index, partition, generation, tamper).is_err());
    }
}
#[test]
fn placement_intent_survives_store_reopen_with_exact_candidate_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fit.redb");
    let attempt = |fence| {
        FencedAttempt::new(
            JobId::new(),
            jain_smartcluster_core::AttemptNumber::new(1).unwrap(),
            jain_smartcluster_core::FenceToken::new(fence).unwrap(),
        )
    };
    let intent = FitPlacementIntent {
        request: AttachTrustedFitChildRequest {
            authority_attempt: attempt(1),
            target_attempt: attempt(2),
            child_pid: 42,
            child_birth_ticks: 123,
            child_boot_id: "11111111-1111-1111-1111-111111111111".into(),
            task_manifest_sha256: "a".repeat(64),
            descriptor_sha256: "b".repeat(64),
            candidate_index: 0,
            idempotency_key: jain_smartcluster_core::IdempotencyKey::new("fit-0").unwrap(),
            child_executable_sha256: "c".repeat(64),
        },
        authority_pid: 41,
        authority_birth: Some(122),
        authority_cgroup: "/sys/fs/cgroup/authority".into(),
        target_cgroup: "/sys/fs/cgroup/target".into(),
    };
    {
        let store = Store::open(&path).unwrap();
        store.put_meta("test-fit-intent", &intent).unwrap();
    }
    let store = Store::open(&path).unwrap();
    assert_eq!(
        store
            .get_meta::<FitPlacementIntent>("test-fit-intent")
            .unwrap(),
        Some(intent.clone())
    );
    let mut conflicting = intent.clone();
    conflicting.request.descriptor_sha256 = "d".repeat(64);
    assert_ne!(
        Some(conflicting),
        store
            .get_meta::<FitPlacementIntent>("test-fit-intent")
            .unwrap()
    );
}
#[test]
fn proc_identity_uses_final_parenthesis_and_rejects_traced_or_running_child() {
    let fields = |state: &str| {
        format!("99 (a ) difficult name) {state} 42 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 123 0")
    };
    let stopped = parse_stat(&fields("T")).unwrap();
    assert_eq!(stopped.parent, 42);
    assert_eq!(stopped.birth, 123);
    assert!(stopped.stopped);
    assert!(!parse_stat(&fields("t")).unwrap().stopped);
    assert!(!parse_stat(&fields("R")).unwrap().stopped);
    assert!(parse_stat("broken").is_err());
}
