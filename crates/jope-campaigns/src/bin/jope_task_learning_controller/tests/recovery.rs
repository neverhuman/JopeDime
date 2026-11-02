fn recovery_intent(root: &Path, job: &str) -> ActiveIntent {
    let spec_path = root.join(format!("{job}.toml"));
    let body = format!(
        "[job]\nid = {}\n\n[[input]]\nname = \"script\"\nsource = \"/frozen/script\"\ntarget = \"script\"\nsha256 = \"{}\"\n",
        serde_json::to_string(job).unwrap(),
        "d".repeat(64),
    );
    fs::write(&spec_path, &body).unwrap();
    let spec = Sha256Digest::parse(sha256_bytes(body.as_bytes())).unwrap();
    ActiveIntent {
        trainer: TrainerIntentV2 {
            schema: "JopeSelfplayTrainerIntent.v2".to_owned(),
            task: Task::Regression,
            mode: TrainerMode::Shadow,
            accepted_parent_checkpoint_sha256: Sha256Digest::parse("a".repeat(64)).unwrap(),
            shadow_parent_checkpoint_sha256: None,
            warm_start_checkpoint_sha256: Sha256Digest::parse("a".repeat(64)).unwrap(),
            corpus_identity_sha256: Sha256Digest::parse("c".repeat(64)).unwrap(),
            corpus_watermark: 1,
            seed: 1,
            immutable_job_identity: job.to_owned(),
            immutable_spec_sha256: spec,
            created_at_ms: 1,
            runtime_bundle_path: None,
            runtime_bundle_sha256: None,
            catalog_binding_path: None,
            catalog_binding_sha256: None,
            recipe: None,
        },
        spec_path,
        output_name: "out".to_owned(),
        warm_start_path: PathBuf::new(),
    }
}

#[test]
fn a_vanished_trainer_retries_the_same_frozen_science_automatically() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let intent = recovery_intent(root, "vanished-job");
    assert_eq!(
        trainer_terminal_state(&intent, &[], 60_001),
        Some("vanished")
    );

    let running = [SchedulerJob {
        id: "vanished-job".to_owned(),
        project: PROJECT.to_owned(),
        state: "running".to_owned(),
    }];
    assert_eq!(trainer_terminal_state(&intent, &running, 60_001), None);

    let args = Args {
        root: root.to_owned(),
        scq: PathBuf::from("/bin/false"),
        runs_root: root.join("runs"),
        trainer_script: root.join("trainer"),
        common_script: root.join("common"),
        packed_script: root.join("packed"),
        prime_source: root.join("prime"),
        submit: false,
        once: true,
        interval_seconds: 1,
        seed_base: 1,
    };
    let original_payload = trainer_scientific_payload(&intent).unwrap();
    let original_corpus = intent.trainer.corpus_identity_sha256.clone();
    let original_seed = intent.trainer.seed;
    let original_recipe = intent.trainer.recipe.clone();
    let mut state = ControllerState {
        intent: Some(intent.clone()),
        ..ControllerState::default()
    };
    let disposition =
        recover_trainer_intent(&args, &mut state, &intent, "vanished", 60_001).unwrap();
    assert_eq!(disposition.action, "trainer_infrastructure_retry_audited");
    let recovered = state.intent.unwrap();
    assert_eq!(recovered.trainer.corpus_identity_sha256, original_corpus);
    assert_eq!(recovered.trainer.seed, original_seed);
    assert_eq!(recovered.trainer.recipe, original_recipe);
    assert_eq!(
        recovered.trainer.immutable_job_identity,
        "vanished-job-infra000002"
    );
    let projection = load_or_rebuild_recovery_projection(
        root,
        DimeRoleV1::AlternatingTrainer,
        &original_payload,
        "vanished-job",
    )
    .unwrap();
    assert_eq!(projection.current_attempt_number, 2);
    assert_eq!(projection.scientific_payload, original_payload);
    assert!(projection.retry_pending_submission);
    let primary = serde_json::to_string(&recovered).unwrap();
    for forbidden in [
        "scientific_payload",
        "infrastructure_attempt",
        "scientific_failures",
        "foreign_cancellations",
        "next_retry_not_before_ms",
    ] {
        assert!(!primary.contains(forbidden));
    }
}
