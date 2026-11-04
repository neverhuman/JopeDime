use super::*;
fn continuation_fixture(root: &Path) -> Value {
    let initialization = json!({"state_sha256":"a".repeat(64),"kind":"warm_start"});
    let state = json!({"schema":"parallel.checkpoint.v1","step":3000,"world":8,"global_batch":2048,
            "initialization":initialization,"corpus_sha256":"b".repeat(64),"vocabulary_sha256":"c".repeat(64),
            "seed":20260907,"sampler_id":"family_uniform_champion_plus_seven_quantiles_v2",
            "rng":"counter reset before every microbatch: mix(seed ^ step*65537 ^ rank*257 ^ microbatch); LibTorch CPU/CUDA manual_seed",
            "sampler":"family-uniform best teacher plus seven quantile comparators; next global example = step*2048",
            "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
            "autocast_cache_lifetime":"scoped_v2"});
    write_json(&root.join("state.json"), &state).unwrap();
    let parent = file_hash(&root.join("state.json")).unwrap();
    write_json(&root.join("run.json"),&json!({"schema":"parallel.training_run.v1","world":8,
            "microbatch":256,"global_batch":2048,"initialization_sha256":initialization["state_sha256"],
            "corpus_sha256":state["corpus_sha256"],"sampler_id":state["sampler_id"],
            "executable_sha256":"d".repeat(64)})).unwrap();
    write_json(
        &root.join("complete.json"),
        &json!({"schema":"parallel.training_complete.v1","world":8,
            "microbatch":256,"steps":3000,"latest":{"step":3000,"state_sha256":parent},
            "corpus_sha256":state["corpus_sha256"]}),
    )
    .unwrap();
    write_json(&root.join("initialization.json"), &initialization).unwrap();
    let mut policy = json!({"schema":CONTINUATION_SCHEMA,"mode":"optimizer_carry_new_window",
            "optimizer":"carry","exact_resume":false,"sampling_seed":20260909,
            "parent_checkpoint_sha256":parent,"corpus_sha256":"e".repeat(64),
            "vocabulary_sha256":"c".repeat(64),"execution_partition":crate::task_contract::training_partition(8,256,2048).unwrap(),
            "requested_step_increment":1000});
    for (key, path) in [
        ("original_run", "run.json"),
        ("original_completion", "complete.json"),
        ("original_initialization", "initialization.json"),
        ("parent_state", "state.json"),
    ] {
        policy[key] = json!({"path":path,"sha256":file_hash(&root.join(path)).unwrap()});
    }
    policy
}
fn recover(root: &Path, p: &Value) -> Result<Value> {
    verify_continuation(
        root,
        p,
        &"e".repeat(64),
        &"c".repeat(64),
        crate::checkpoint_recovery::TrainingTopology {
            world: 8,
            micro: 256,
            global: 2048,
        },
        1000,
    )
}
#[test]
fn recovered_carry_preserves_parent_and_adds_new_steps_without_claiming_exact_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = continuation_fixture(tmp.path());
    let before = fs::read(tmp.path().join("state.json")).unwrap();
    let recovered = recover(tmp.path(), &policy).unwrap();
    assert_eq!(recovered["optimizer_parent_step"], 3000);
    assert_eq!(recovered["target_optimizer_step"], 4000);
    assert_eq!(recovered["original_seed"], 20260907);
    assert_eq!(recovered["seed"], 20260909);
    assert_eq!(recovered["execution_partition"]["accumulation"], 1);
    assert_eq!(recovered["original_next_global_example"], 6144000);
    assert_eq!(recovered["new_window_counter_origin"], 0);
    assert_eq!(recovered["exact_same_experiment_resume"], false);
    assert_eq!(recovered["execution_authority"], false);
    assert_eq!(before, fs::read(tmp.path().join("state.json")).unwrap());
}
#[test]
fn continuation_refuses_missing_original_partition_changed_reference_or_mixed_mode() {
    for kind in [
        "missing_micro",
        "boolean_micro",
        "changed_original",
        "mixed_replay",
        "exact_claim",
        "wrong_increment",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mut policy = continuation_fixture(root);
        match kind {
            "missing_micro" | "boolean_micro" => {
                let mut run: Value = read_json(&root.join("run.json")).unwrap();
                if kind == "missing_micro" {
                    run.as_object_mut().unwrap().remove("microbatch");
                } else {
                    run["microbatch"] = true.into();
                }
                write_json(&root.join("run.json"), &run).unwrap();
                policy["original_run"]["sha256"] =
                    file_hash(&root.join("run.json")).unwrap().into();
            }
            "changed_original" => fs::write(root.join("complete.json"), b"{}").unwrap(),
            "mixed_replay" => policy["replay"] = json!("fresh_balanced"),
            "exact_claim" => policy["exact_resume"] = true.into(),
            "wrong_increment" => policy["requested_step_increment"] = 1001.into(),
            _ => unreachable!(),
        }
        assert!(recover(root, &policy).is_err(), "{kind}");
    }
}
#[test]
fn continuation_composes_a_separately_bound_replay_policy() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut policy = continuation_fixture(root);
    let replay = json!({"schema":"parallel.replay_policy.v2","optimizer":"carry","sampling_seed":policy["sampling_seed"],
            "parent_checkpoint_sha256":policy["parent_checkpoint_sha256"],"corpus_sha256":policy["corpus_sha256"],
            "vocabulary_sha256":policy["vocabulary_sha256"],"replay":"fresh_balanced"});
    write_json(&root.join("replay.json"), &replay).unwrap();
    policy["replay_policy_ref"] =
        json!({"path":"replay.json","sha256":file_hash(&root.join("replay.json")).unwrap()});
    assert_eq!(
        recover(root, &policy).unwrap()["new_window_replay"],
        "fresh_balanced"
    );
    assert_eq!(continuation_references(&policy).unwrap().len(), 5);
    fs::write(root.join("replay.json"), b"{}").unwrap();
    assert!(recover(root, &policy).is_err());
}
fn checkpoint(root: &Path, step: u64) -> PathBuf {
    let directory = root.join(format!("run/checkpoints/step-{step}"));
    fs::create_dir_all(&directory).unwrap();
    let mut state = json!({"schema":"parallel.checkpoint.v1","step":step});
    for (file, key) in [
        ("model.ot", "model_sha256"),
        ("optimizer.ot", "optimizer_sha256"),
        ("optimizer.json", "optimizer_metadata_sha256"),
        ("config.json", "config_sha256"),
    ] {
        fs::write(directory.join(file), format!("{step}:{file}")).unwrap();
        state[key] = file_hash(&directory.join(file)).unwrap().into();
    }
    write_json(&directory.join("state.json"), &state).unwrap();
    directory
}
fn latest(root: &Path, step: u64) {
    let directory = checkpoint(root, step);
    write_json(&root.join("run/latest.json"),&json!({"step":step,
            "checkpoint":format!("run/checkpoints/step-{step}"),"state_sha256":file_hash(&directory.join("state.json")).unwrap()})).unwrap();
}
#[test]
fn orphan_and_staging_are_preserved_and_published_history_stays_put() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    checkpoint(root, 5);
    latest(root, 10);
    let orphan = checkpoint(root, 20);
    let bytes = fs::read(orphan.join("model.ot")).unwrap();
    let staging = root.join("run/checkpoints/.step-30-123");
    fs::create_dir(&staging).unwrap();
    fs::write(staging.join("partial"), b"saved").unwrap();
    let published = fs::read(root.join("run/latest.json")).unwrap();
    reconcile(root, &root.join("run")).unwrap();
    assert!(root.join("run/checkpoints/step-5").exists());
    assert!(root.join("run/checkpoints/step-10").exists());
    assert!(!orphan.exists());
    assert!(!staging.exists());
    assert_eq!(published, fs::read(root.join("run/latest.json")).unwrap());
    let quarantine = fs::read_dir(root.join("run/checkpoint-recovery"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert_eq!(
        bytes,
        fs::read(quarantine.join("step-20/model.ot")).unwrap()
    );
    assert_eq!(
        fs::read(quarantine.join(".step-30-123/partial")).unwrap(),
        b"saved"
    );
    let receipt: Value = read_json(&quarantine.join("receipt.json")).unwrap();
    assert_eq!(receipt["completed"], true);
    reconcile(root, &root.join("run")).unwrap();
    assert_eq!(
        fs::read_dir(root.join("run/checkpoint-recovery"))
            .unwrap()
            .count(),
        1
    );
}
#[test]
fn corrupted_latest_aborts_before_moving_anything() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    latest(root, 10);
    checkpoint(root, 20);
    fs::write(root.join("run/checkpoints/step-10/model.ot"), b"corrupt").unwrap();
    assert!(reconcile(root, &root.join("run")).is_err());
    assert!(root.join("run/checkpoints/step-20").exists());
    assert!(!root.join("run/checkpoint-recovery").exists());
}
#[test]
fn fresh_unpublished_work_can_be_quarantined_but_completed_run_without_latest_cannot() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    checkpoint(root, 1);
    reconcile(root, &root.join("run")).unwrap();
    assert!(!root.join("run/checkpoints/step-1").exists());
    checkpoint(root, 2);
    fs::write(root.join("run/complete.json"), b"{}").unwrap();
    assert!(reconcile(root, &root.join("run")).is_err());
    assert!(root.join("run/checkpoints/step-2").exists());
}
#[test]
fn published_path_cannot_point_to_a_different_run() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    latest(root, 10);
    checkpoint(root, 20);
    let mut marker: Value = read_json(&root.join("run/latest.json")).unwrap();
    marker["checkpoint"] = json!("other/checkpoints/step-10");
    write_json(&root.join("run/latest.json"), &marker).unwrap();
    assert!(reconcile(root, &root.join("run")).is_err());
    assert!(root.join("run/checkpoints/step-20").exists());
}
#[test]
fn published_marker_with_missing_checkpoint_directory_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    write_json(
        &root.join("run/latest.json"),
        &json!({"step":10,"checkpoint":"run/checkpoints/step-10","state_sha256":"x"}),
    )
    .unwrap();
    assert!(reconcile(root, &root.join("run"))
        .unwrap_err()
        .to_string()
        .contains("directory is missing"));
}
