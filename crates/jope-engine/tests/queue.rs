use jope_parallel::{
    manifest::*,
    queue,
    service::{GpJob, GpJobs},
};

#[test]
fn queue_merge_is_idempotent_and_preserves_other_work() {
    let root = tempfile::tempdir().unwrap();
    let job = |n| GpJob {
        prepared: format!("data/prepared/{n}").into(),
        prepared_sha256: "a".repeat(64),
        cut: 0,
        output: format!("runs/search/{n}").into(),
        seeds: None,
        seeds_sha256: None,
        ..Default::default()
    };
    let mut incoming = GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: vec![job(1)],
        vocabulary: None,
    };
    assert_eq!(queue::merge(root.path(), &incoming).unwrap(), 1);
    assert_eq!(queue::merge(root.path(), &incoming).unwrap(), 0);
    incoming.jobs = vec![job(2)];
    assert_eq!(queue::merge(root.path(), &incoming).unwrap(), 1);
    let queue_path = root.path().join("runs/gp-priority-jobs.json");
    let previous = file_hash(&queue_path).unwrap();
    incoming.jobs[0].prepared_sha256 = "b".repeat(64);
    assert!(queue::merge(root.path(), &incoming).is_err());
    assert_eq!(file_hash(&queue_path).unwrap(), previous);
    let saved: GpJobs = read_json(&queue_path).unwrap();
    assert_eq!(saved.jobs.len(), 2);
    assert_eq!(saved.jobs[0].output, job(2).output);
    assert_eq!(saved.jobs[1].output, job(1).output);
}

#[test]
fn unencoded_jobs_cannot_replace_a_live_queue() {
    let root = tempfile::tempdir().unwrap();
    let jobs = GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: vec![GpJob {
            prepared: "missing-prepared".into(),
            prepared_sha256: "a".repeat(64),
            cut: 0,
            output: "runs/search/1".into(),
            seeds: None,
            seeds_sha256: None,
            ..Default::default()
        }],
        vocabulary: None,
    };
    queue::merge(root.path(), &jobs).unwrap();
    let queue_path = root.path().join("runs/gp-priority-jobs.json");
    let previous = file_hash(&queue_path).unwrap();
    let job_path = root.path().join("jobs.json");
    let context_path = root.path().join("contexts.json");
    write_json(&job_path, &jobs).unwrap();
    write_json(&context_path,&serde_json::json!({"schema":"parallel.context_index.v1","dimensions":2048,"qualification":true,"encodings":[]})).unwrap();
    assert!(queue::enqueue(root.path(), &job_path, &context_path).is_err());
    assert_eq!(file_hash(&queue_path).unwrap(), previous);
}

fn prepared_queue_fixture(
    root: &std::path::Path,
    status: &str,
    task: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let prepared = serde_json::json!({"schema":"parallel.prepared.v1", "dataset":"new-source", "family":"new-family",
        "task":task,"meta_partition":"meta_train","seed":20260907,"family_partition_frozen":false,
        "clean_evaluation":false,"qualification_only":true,"clean_status":status,
        "original_train_sha256":"training","original_test_sha256":"sealed",
        "selected_train_indices":[0,1],"selected_test_indices":[2],"target_last":true,"rows_total":3,
        "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"train-cut",
            "validation_sha256":"validation-cut","preprocessing_sha256":"preprocessing"}]});
    let prepared_path = root.join("prepared/prepared.json");
    write_json(&prepared_path, &prepared).unwrap();
    let prepared_sha = file_hash(&prepared_path).unwrap();
    let payload = vec![0u8; 8192];
    let payload_path = root.join("context.f32");
    std::fs::write(&payload_path, &payload).unwrap();
    let metadata = serde_json::json!({"schema":"parallel.encoding.v1","dataset":"new-source","family":"new-family",
        "partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,"training_sha256":"train-cut",
        "selected_training_rows_sha256":hash(&serde_json::to_vec(&vec![0usize]).unwrap()),
        "preprocessing_sha256":"preprocessing","encoder_spec_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","numerical_backend":"fixture",
        "dataset_dimensions":1536,"target_dimensions":512,"output_sha256":hash(&payload)});
    write_json(&payload_path.with_extension("json"), &metadata).unwrap();
    let jobs = GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs: vec![GpJob {
            prepared: "prepared".into(),
            prepared_sha256: prepared_sha.clone(),
            cut: 0,
            output: "runs/new-source/cut-0".into(),
            seeds: None,
            seeds_sha256: None,
            ..Default::default()
        }],
        vocabulary: None,
    };
    let jobs_path = root.join("jobs.json");
    write_json(&jobs_path, &jobs).unwrap();
    let contexts_path = root.join("contexts.json");
    write_json(&contexts_path,&serde_json::json!({"schema":"parallel.context_index.v1","dimensions":2048,
        "assembly_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","qualification":true,"encodings":[{"dataset":"new-source","cut":0,
        "prepared_sha256":prepared_sha,"prepared":"prepared","family":"new-family","partition":"meta_train","path":"context.f32","sha256":hash(&payload)}]})).unwrap();
    (jobs_path, contexts_path)
}

#[test]
fn native_preparation_status_cannot_change_a_live_gp_queue() {
    // Each row is refused by the gate it exercises: an unqualified preparation status by the
    // GP-worker compatibility check, and a non-regression task by the regression-only task
    // contract (task_contract::require_task), which runs first. Neither may touch the queue.
    for (status, task, reason) in [
        (
            "native_raw_verified_qualification_only",
            "regression",
            "incompatible with GP workers",
        ),
        (
            "raw_training_reconstructed_qualification_only",
            "binary",
            "task binary is not qualified for parallel execution or teaching; regression only",
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let queue_path = root.path().join("runs/gp-priority-jobs.json");
        write_json(
            &queue_path,
            &serde_json::json!({"preserved":"existing production queue"}),
        )
        .unwrap();
        let before = std::fs::read(&queue_path).unwrap();
        let (jobs, contexts) = prepared_queue_fixture(root.path(), status, task);
        let error = queue::enqueue(root.path(), &jobs, &contexts)
            .unwrap_err()
            .to_string();
        assert!(error.contains(reason), "{error}");
        if task == "regression" {
            // The status gate, not the task contract, must be what refused this row.
            assert!(
                !error.contains("not qualified for parallel execution"),
                "{error}"
            );
        }
        assert_eq!(std::fs::read(&queue_path).unwrap(), before);
        assert!(!root.path().join("runs/new-source").exists());
        assert!(!root.path().join("runs/queue-receipts").exists());
    }
}

#[test]
fn supported_preparation_with_bound_context_remains_admissible() {
    let root = tempfile::tempdir().unwrap();
    let (jobs, contexts) = prepared_queue_fixture(
        root.path(),
        "raw_training_reconstructed_qualification_only",
        "regression",
    );
    assert_eq!(queue::enqueue(root.path(), &jobs, &contexts).unwrap(), 1);
    assert_eq!(queue::enqueue(root.path(), &jobs, &contexts).unwrap(), 0);
}

#[test]
fn campaign_bound_admission_refuses_union_only_cuts() {
    let root = tempfile::tempdir().unwrap();
    let (jobs, contexts) = prepared_queue_fixture(
        root.path(),
        "raw_training_reconstructed_qualification_only",
        "regression",
    );
    let empty = root.path().join("campaign.json");
    write_json(&empty,&serde_json::json!({"schema":"parallel.context_index.v1","dimensions":2048,
        "assembly_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","qualification":true,"encodings":[]})).unwrap();
    let before = root.path().join("runs/gp-priority-jobs.json");
    assert!(!before.exists());
    let error = queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&empty))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("outside the bound campaign context index"),
        "{error}"
    );
    assert!(!before.exists());
    assert_eq!(
        queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&contexts)).unwrap(),
        1
    );
    let receipt: serde_json::Value = read_json(
        &std::fs::read_dir(root.path().join("runs/queue-receipts"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path(),
    )
    .unwrap();
    assert_eq!(receipt["campaign_bound"], true);
    assert!(receipt["campaign_contexts_sha256"].as_str().unwrap().len() == 64);
}

#[test]
fn coverage_separates_wasted_gp_from_unused_campaign_cuts() {
    let root = tempfile::tempdir().unwrap();
    let campaign = serde_json::json!({"schema":"parallel.context_index.v1","encodings":[
        {"dataset":"in","cut":0},{"dataset":"unused","cut":1}]});
    let union = serde_json::json!({"schema":"parallel.context_index.v1","encodings":[
        {"dataset":"in","cut":0},{"dataset":"unused","cut":1},{"dataset":"extra","cut":9}]});
    std::fs::create_dir_all(root.path().join("gp/in/cut-0")).unwrap();
    std::fs::create_dir_all(root.path().join("gp/extra/cut-9")).unwrap();
    let gp = root.path().join("gp");
    let report = queue::coverage_report(&campaign, &union, &[gp.as_path()]).unwrap();
    assert_eq!(report["campaign_encodings"], 2);
    assert_eq!(report["union_encodings"], 3);
    assert_eq!(report["in_cohort_gp"], 1);
    assert_eq!(report["wasted_gp"], 1);
    assert_eq!(report["unused_campaign_cuts"], 1);
}

#[test]
fn campaign_identity_mismatches_never_modify_an_existing_queue() {
    let mutations = vec![
        ("/assembly_sha256", serde_json::json!("b".repeat(64))),
        ("/dimensions", serde_json::json!(1024)),
        ("/qualification", serde_json::json!(false)),
        ("/schema", serde_json::json!("unknown")),
        ("/encodings/0/sha256", serde_json::json!("f".repeat(64))),
        ("/encodings/0/path", serde_json::json!("other-context.f32")),
        ("/encodings/0/prepared", serde_json::json!("other-prepared")),
        ("/encodings/0/prepared", serde_json::json!("../escape")),
        (
            "/encodings/0/prepared_sha256",
            serde_json::json!("f".repeat(64)),
        ),
        ("/encodings/0/prepared_sha256", serde_json::Value::Null),
        ("/encodings/0/sha256", serde_json::json!("malformed")),
        ("/encodings/0/family", serde_json::json!("other-family")),
        ("/encodings/0/partition", serde_json::json!("meta_test")),
        ("/encodings/0/partition", serde_json::json!("unknown")),
        ("/encodings/0/dataset", serde_json::json!("")),
        ("/encodings/0/cut", serde_json::json!(true)),
        ("/encodings/0/cut", serde_json::json!(1)),
    ];
    for (pointer, value) in mutations {
        let root = tempfile::tempdir().unwrap();
        let (jobs, contexts) = prepared_queue_fixture(
            root.path(),
            "raw_training_reconstructed_qualification_only",
            "regression",
        );
        let mut existing: GpJobs = read_json(&jobs).unwrap();
        existing.jobs[0].output = "runs/existing-job".into();
        queue::merge(root.path(), &existing).unwrap();
        let queue = root.path().join("runs/gp-priority-jobs.json");
        let before = std::fs::read(&queue).unwrap();
        let mut campaign: serde_json::Value = read_json(&contexts).unwrap();
        *campaign.pointer_mut(pointer).unwrap() = value;
        let bound = root.path().join("campaign.json");
        write_json(&bound, &campaign).unwrap();
        let result = queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&bound));
        assert!(result.is_err(), "{pointer} was admitted");
        assert_eq!(
            std::fs::read(&queue).unwrap(),
            before,
            "{pointer} changed queue"
        );
        assert!(!root.path().join("runs/queue-receipts").exists());
    }
}

#[test]
fn valid_union_vector_with_a_different_campaign_vector_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let (jobs, contexts) = prepared_queue_fixture(
        root.path(),
        "raw_training_reconstructed_qualification_only",
        "regression",
    );
    let campaign = root.path().join("campaign.json");
    std::fs::copy(&contexts, &campaign).unwrap();
    let bytes = 1f32.to_le_bytes().repeat(2048);
    std::fs::write(root.path().join("context.f32"), &bytes).unwrap();
    let mut metadata: serde_json::Value = read_json(&root.path().join("context.json")).unwrap();
    metadata["output_sha256"] = hash(&bytes).into();
    write_json(&root.path().join("context.json"), &metadata).unwrap();
    let mut union: serde_json::Value = read_json(&contexts).unwrap();
    union["encodings"][0]["sha256"] = hash(&bytes).into();
    write_json(&contexts, &union).unwrap();
    let error = queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&campaign))
        .unwrap_err()
        .to_string();
    assert!(error.contains("sha256 binding differs"), "{error}");
    assert!(!root.path().join("runs/gp-priority-jobs.json").exists());
    // Independent research may intentionally use this separately bound vector.
    assert_eq!(queue::enqueue(root.path(), &jobs, &contexts).unwrap(), 1);
}

#[test]
fn duplicate_or_unbound_campaign_receipts_are_rejected() {
    for variant in [
        "duplicate-campaign",
        "duplicate-union",
        "receipt",
        "components",
    ] {
        let root = tempfile::tempdir().unwrap();
        let (jobs, contexts) = prepared_queue_fixture(
            root.path(),
            "raw_training_reconstructed_qualification_only",
            "regression",
        );
        let mut union: serde_json::Value = read_json(&contexts).unwrap();
        let mut campaign = union.clone();
        match variant {
            "duplicate-campaign" => {
                let row = campaign["encodings"][0].clone();
                campaign["encodings"].as_array_mut().unwrap().push(row);
            }
            "duplicate-union" => {
                let row = union["encodings"][0].clone();
                union["encodings"].as_array_mut().unwrap().push(row);
            }
            "receipt" => {
                union["encodings"][0]["receipt_sha256"] = "e".repeat(64).into();
                campaign = union.clone();
            }
            "components" => {
                union["encodings"][0]["components_sha256"] = "e".repeat(64).into();
                campaign = union.clone();
            }
            _ => unreachable!(),
        }
        write_json(&contexts, &union).unwrap();
        let bound = root.path().join("campaign.json");
        write_json(&bound, &campaign).unwrap();
        assert!(
            queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&bound)).is_err(),
            "{variant}"
        );
        assert!(!root.path().join("runs/gp-priority-jobs.json").exists());
    }
}

#[test]
fn a_changed_selected_campaign_revision_is_checked_on_every_admission() {
    let root = tempfile::tempdir().unwrap();
    let (jobs, contexts) = prepared_queue_fixture(
        root.path(),
        "raw_training_reconstructed_qualification_only",
        "regression",
    );
    let campaign = root.path().join("campaign.json");
    std::fs::copy(&contexts, &campaign).unwrap();
    assert_eq!(
        queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&campaign)).unwrap(),
        1
    );
    let queue_path = root.path().join("runs/gp-priority-jobs.json");
    let before = std::fs::read(&queue_path).unwrap();
    let receipt_path = root
        .path()
        .join("runs/queue-receipts")
        .join(format!("{}.json", file_hash(&jobs).unwrap()));
    let receipt_before = std::fs::read(&receipt_path).unwrap();
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_before).unwrap();
    assert_eq!(
        receipt["campaign_contexts_sha256"],
        file_hash(&campaign).unwrap()
    );
    assert_eq!(receipt["contexts_sha256"], file_hash(&contexts).unwrap());
    assert_eq!(receipt["inputs_rechecked_before_merge"], true);
    let mut revised: serde_json::Value = read_json(&campaign).unwrap();
    revised["encodings"][0]["path"] = "changed.f32".into();
    write_json(&campaign, &revised).unwrap();
    assert!(queue::enqueue_bound(root.path(), &jobs, &contexts, Some(&campaign)).is_err());
    assert_eq!(std::fs::read(queue_path).unwrap(), before);
    assert_eq!(std::fs::read(receipt_path).unwrap(), receipt_before);
}

#[test]
fn command_line_admission_uses_the_exact_campaign_binding() {
    let root = tempfile::tempdir().unwrap();
    let (jobs, contexts) = prepared_queue_fixture(
        root.path(),
        "raw_training_reconstructed_qualification_only",
        "regression",
    );
    let campaign = root.path().join("campaign.json");
    let mut wrong: serde_json::Value = read_json(&contexts).unwrap();
    wrong["encodings"][0]["sha256"] = "f".repeat(64).into();
    write_json(&campaign, &wrong).unwrap();
    let invoke = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(root.path())
            .arg("enqueue")
            .arg("--jobs")
            .arg(&jobs)
            .arg("--contexts")
            .arg(&contexts)
            .arg("--campaign-contexts")
            .arg(&campaign)
            .output()
            .unwrap()
    };
    let refused = invoke();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("sha256 binding differs"));
    assert!(!root.path().join("runs/gp-priority-jobs.json").exists());
    std::fs::copy(&contexts, &campaign).unwrap();
    let admitted = invoke();
    assert!(
        admitted.status.success(),
        "{}",
        String::from_utf8_lossy(&admitted.stderr)
    );
    let queue: GpJobs = read_json(&root.path().join("runs/gp-priority-jobs.json")).unwrap();
    assert_eq!(queue.jobs.len(), 1);
    assert!(
        !root.path().join("runs/new-source/cut-0").exists(),
        "admission must not execute fits"
    );
}
