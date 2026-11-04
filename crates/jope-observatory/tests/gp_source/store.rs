use super::*;

#[test]
fn every_language_compiles_and_every_token_survives_restart_and_duplicates() {
    let directory = test_dir();
    let mut store = Store::open(directory.path()).unwrap();
    assert!(
        Store::open(directory.path()).is_err(),
        "exclusive store ownership"
    );
    assert!(store.snapshot()["total"]["token_yield"].is_null());
    publish(&mut store, "a", "(pipe X (standardize) (ridge))");
    assert_eq!(counts(&store)["artifacts"], 4);
    assert_eq!(counts(&store)["pending_artifacts"], 4);
    assert_eq!(store.snapshot()["total"]["token_yield"], 0.0);
    drain(&mut store);
    assert_eq!(store.snapshot()["total"]["token_yield"], 1.0);
    let first = counts(&store)["emitted_tokens"].as_u64().unwrap();
    let sum: u64 = store.snapshot()["languages"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["counts"]["emitted_tokens"].as_u64().unwrap())
        .sum();
    assert_eq!(first, sum);
    drop(store);
    let mut store = Store::open(directory.path()).unwrap();
    assert_eq!(counts(&store)["emitted_tokens"], first);
    assert!(store
        .contains(&occurrence("a", "(pipe X (standardize) (ridge))"))
        .unwrap());
    assert!(store
        .contains(&occurrence("a", "(pipe X (ridge))"))
        .is_err());
    publish(&mut store, "b", "(pipe X (standardize) (ridge))");
    assert_eq!(counts(&store)["emitted_tokens"], first * 2);
    assert_eq!(store.snapshot()["unique_programs"], 1);
    drain(&mut store);
    assert_eq!(store.snapshot()["total"]["token_yield"], 1.0);
    assert_eq!(store.sequence, 10);
}

#[test]
fn emission_status_accounting_cases() {
    fn assert_failed_artifact_has_no_forged_source(artifact: &gp_source::Artifact) {
        assert_eq!(artifact.status, Status::Failed, "{}", artifact.language);
        assert!(artifact.source.is_none(), "{}", artifact.language);
        assert!(artifact.sha256.is_none(), "{}", artifact.language);
        assert_eq!(
            (artifact.code_tokens, artifact.doc_tokens, artifact.bytes),
            (0, 0, 0),
            "failed emitter produced accounting for {}",
            artifact.language
        );
        assert!(
            artifact
                .error
                .as_deref()
                .is_some_and(|error| !error.is_empty()),
            "failed emitter omitted its diagnostic for {}",
            artifact.language
        );
    }

    let directory = test_dir();
    let mut store = Store::open(directory.path()).unwrap();
    let invalid = occurrence("invalid", "this does not parse");
    let (invalid_artifacts, invalid_emit_ns) = gp_source::emit(&invalid).unwrap();
    assert_eq!(invalid_artifacts.len(), gp_source::LANGUAGES.len());
    for artifact in &invalid_artifacts {
        assert_failed_artifact_has_no_forged_source(artifact);
    }
    store
        .publish(invalid, invalid_artifacts, invalid_emit_ns)
        .unwrap();
    assert_eq!(counts(&store)["emission_errors"], 4);
    assert_eq!(counts(&store)["emitted_tokens"], 0);
    assert_eq!(store.snapshot()["total"]["artifact_yield"], 0.0);
    assert!(store.snapshot()["total"]["token_yield"].is_null());

    let unsupported = occurrence("histgbm", "(pipe X (histgbm depth=2 rounds=2))");
    let (artifacts, emit_ns) = gp_source::emit(&unsupported).unwrap();
    assert_eq!(artifacts.len(), gp_source::LANGUAGES.len());
    let failures = artifacts
        .iter()
        .filter(|artifact| artifact.status == Status::Failed)
        .count();
    assert_eq!(
        failures, 2,
        "unsupported targets must remain explicit failures"
    );
    for artifact in &artifacts {
        match artifact.status {
            Status::Failed => assert_failed_artifact_has_no_forged_source(artifact),
            Status::Pending => {
                let source = artifact.source.as_deref().expect("pending source bytes");
                let digest = gp_source::hash(source.as_bytes());
                assert_eq!(artifact.sha256.as_deref(), Some(digest.as_str()));
                assert_eq!(artifact.bytes, source.len() as u64);
                assert!(artifact.tokens() > 0, "{}", artifact.language);
            }
            Status::Passed => panic!("emission cannot pre-approve {}", artifact.language),
        }
    }
    let emitted_tokens: u64 = artifacts.iter().map(gp_source::Artifact::tokens).sum();
    assert!(
        emitted_tokens > 0,
        "supported emitters must retain source tokens"
    );
    store.publish(unsupported, artifacts, emit_ns).unwrap();

    let observed = counts(&store);
    assert_eq!(observed["emission_errors"], 6);
    assert_eq!(observed["emitted_tokens"].as_u64(), Some(emitted_tokens));
    assert_eq!(observed["pending_tokens"].as_u64(), Some(emitted_tokens));
    assert_eq!(observed["pending_artifacts"], 2);
    assert_eq!(observed["artifacts"], 8);
    assert_eq!(store.snapshot()["total"]["token_yield"], 0.0);
}

#[test]
fn compiler_failure_charges_all_failed_source_tokens_as_loss() {
    let directory = test_dir();
    let mut store = Store::open(directory.path()).unwrap();
    let input = occurrence("broken-rust", "(pipe X (ridge))");
    let (mut artifacts, ns) = gp_source::emit(&input).unwrap();
    let artifact = artifacts.iter_mut().find(|a| a.language == "rust").unwrap();
    let bad = "fn broken( {\n";
    let n = feat_math::tokens::count_tokens(bad, feat_math::tokens::Lang::Rust);
    artifact.source = Some(bad.into());
    artifact.sha256 = Some(gp_source::hash(bad.as_bytes()));
    artifact.code_tokens = n.code_tokens as u64;
    artifact.doc_tokens = n.doc_tokens as u64;
    artifact.bytes = bad.len() as u64;
    let expected_loss = artifact.tokens();
    store.publish(input, artifacts, ns).unwrap();
    while let Some((id, artifact, root)) = store.next_check() {
        let result = gp_source::check(&root, &artifact, 15).unwrap();
        assert_eq!(
            result.status,
            if artifact.language == "rust" {
                Status::Failed
            } else {
                Status::Passed
            }
        );
        store.checked(id, &artifact, result).unwrap();
    }
    let c = counts(&store);
    assert_eq!(c["lost_tokens"], expected_loss);
    assert_eq!(c["compile_errors"], 1);
    assert_eq!(c["pending_tokens"], 0);
    assert_eq!(
        c["passed_tokens"].as_u64().unwrap() + expected_loss,
        c["emitted_tokens"].as_u64().unwrap()
    );
    assert!(store.snapshot()["total"]["token_yield"].as_f64().unwrap() < 1.0);
}

#[test]
fn pending_checks_recover_and_mismatched_results_are_refused() {
    let directory = test_dir();
    let mut store = Store::open(directory.path()).unwrap();
    publish(&mut store, "pending", "(pipe X (ridge))");
    let _inflight = store.next_check().unwrap();
    drop(store);
    let mut store = Store::open(directory.path()).unwrap();
    assert_eq!(counts(&store)["pending_artifacts"], 4);
    let (id, artifact, root) = store.next_check().unwrap();
    let checked = gp_source::check(&root, &artifact, 15).unwrap();
    store.checked(id.clone(), &artifact, checked).unwrap();
    let duplicate = Check {
        status: Status::Passed,
        tool: "fixture".into(),
        diagnostic: String::new(),
        elapsed_ns: 1,
    };
    assert!(store.checked(id, &artifact, duplicate).is_err());
    assert_eq!(store.snapshot()["store_failed"], true);
}

#[test]
fn partial_producer_line_waits_but_corrupt_complete_or_event_lines_fail() {
    let mut reader = BufReader::new(Cursor::new(b"unfinished".to_vec()));
    assert!(gp_source::complete_line(&mut reader, true)
        .unwrap()
        .is_none());
    assert_eq!(reader.stream_position().unwrap(), 0);
    assert!(gp_source::complete_line(&mut reader, false).is_err());
    let directory = test_dir();
    fs::write(directory.path().join("events.jsonl"), "bad\n").unwrap();
    assert!(Store::open(directory.path()).is_err());
    fs::write(directory.path().join("events.jsonl"), "{\"schema\":").unwrap();
    assert!(Store::open(directory.path()).is_err());
}

#[test]
fn journals_keep_identical_occurrences_distinct_and_unicode_counts_do_not_panic() {
    let directory = test_dir();
    let mut journal = Journal::create(directory.path()).unwrap();
    let a = journal
        .append("fixture", "slot/0", "(pipe X (ridge))")
        .unwrap();
    let b = journal
        .append("fixture", "slot/1", "(pipe X (ridge))")
        .unwrap();
    assert_ne!(a.id, b.id);
    assert_eq!(fs::read_to_string(journal.path).unwrap().lines().count(), 2);
    let count =
        feat_math::tokens::count_tokens("café = 1\n# résumé λ\n", feat_math::tokens::Lang::Python);
    assert_eq!(
        feat_math::tokens::count_tokens("\"\"\"café λ", feat_math::tokens::Lang::Python).doc_tokens,
        2
    );
    assert_eq!(count.doc_tokens, 2);
    assert!(count.code_tokens > 0);
}
