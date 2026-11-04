use super::*;

#[test]
fn public_start_commits_snapshot_and_receipt_together_and_aborts_competing_creation() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let plan = plan(directory.path(), 24, Effort::Fast, 4);
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let log = Arc::new(EventLog::new(Arc::clone(&store)));
    let input = store
        .publish_blob(fs::read(&plan.matrix_path).unwrap().as_slice())
        .unwrap();
    let identity = crate::idempotency::RequestIdentity::new(
        "local-engine-owner",
        "same-key",
        6,
        "/v1/runs",
        None,
        &json!({"test":"actual-native-plan-creation"}),
    )
    .unwrap();
    let run = Discovery::start_command(
        Arc::clone(&store),
        plan.clone(),
        Arc::clone(&log),
        identity.clone(),
        vec![input.clone()],
        vec![],
    )
    .unwrap();
    let original = store.idempotent_receipt(&identity).unwrap().unwrap();
    assert_eq!(
        original.outcome.body(),
        &serde_json::to_value(run.snapshot()).unwrap()
    );
    assert_eq!(
        log.snapshot(&plan.run_id).unwrap().as_ref(),
        Some(run.snapshot())
    );
    assert_eq!(
        run.admitted(),
        0,
        "no worker admission before the committed creation boundary"
    );
    assert!(store
        .get(&RecordKey::Run(plan.run_id.clone()))
        .unwrap()
        .unwrap()
        .blobs
        .contains(&input));
    let mut competing = plan.clone();
    competing.run_id = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap();
    assert!(Discovery::start_command(
        Arc::clone(&store),
        competing.clone(),
        Arc::clone(&log),
        identity.clone(),
        vec![input],
        vec![]
    )
    .is_err());
    assert!(
        log.snapshot(&competing.run_id).unwrap().is_none(),
        "competing run state escaped its aborted receipt transaction"
    );
    assert_eq!(
        store
            .idempotent_receipt(&identity)
            .unwrap()
            .unwrap()
            .outcome,
        original.outcome
    );
    assert_eq!(store.run_ids_after(None, 257).unwrap(), vec![plan.run_id]);
}
