use super::*;

#[test]
fn partial_ingress_tail_remains_visible_after_complete_sources_validate() {
    use std::io::Write;
    let journals = test_dir();
    let state = test_dir();
    let mut journal = Journal::create(journals.path()).unwrap();
    journal
        .append("fixture", "slot/0", "(pipe X (ridge))")
        .unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&journal.path)
        .unwrap()
        .write_all(b"{")
        .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.path().to_str().unwrap(),
            "--state-dir",
            state.path().to_str().unwrap(),
            "--once",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metrics: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(metrics["total"]["token_yield"], 1.0);
    assert_eq!(
        metrics["ingress"]["partial_journal_tails"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(metrics["observed_backlog_clear"], false);
}

#[test]
fn oversized_event_is_rejected_before_any_token_or_sequence_credit() {
    let root = test_dir();
    let mut store = Store::open(root.path()).unwrap();
    let artifact = gp_source::Artifact {
        language: "rust".into(),
        source: None,
        sha256: Some("a".repeat(64)),
        code_tokens: 0,
        doc_tokens: 0,
        bytes: 0,
        status: Status::Pending,
        error: None,
    };
    let checked = Check {
        status: Status::Failed,
        tool: "fixture".into(),
        diagnostic: "x".repeat(gp_source::MAX_LINE),
        elapsed_ns: 1,
    };
    assert!(store
        .checked("not-published".into(), &artifact, checked)
        .unwrap_err()
        .to_string()
        .contains("exceeds line limit"));
    assert_eq!(store.sequence, 0);
    assert_eq!(counts(&store)["emitted_tokens"], 0);
    assert_eq!(
        fs::metadata(root.path().join("events.jsonl"))
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn multiple_producers_make_progress_before_an_old_backlog_drains() {
    let journals = test_dir();
    let state = test_dir();
    let mut old = Journal::create(journals.path()).unwrap();
    for slot in 0..17 {
        old.append("old-producer", &format!("slot/{slot}"), "(pipe X (ridge))")
            .unwrap();
    }
    let mut new = Journal::create(journals.path()).unwrap();
    let expected = new
        .append("new-producer", "slot/0", "(pipe X (ridge))")
        .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.path().to_str().unwrap(),
            "--state-dir",
            state.path().to_str().unwrap(),
            "--once",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let events = fs::read_to_string(state.path().join("events.jsonl")).unwrap();
    let sources = events
        .lines()
        .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
        .filter(|e| e["kind"] == "source")
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 18, "once must drain every bounded turn");
    let position = sources
        .iter()
        .position(|e| e["occurrence"]["id"] == expected.id)
        .unwrap();
    assert!(
        position < 17,
        "new producer starved behind the entire old journal"
    );
    let metrics: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(metrics["total"]["token_yield"], 1.0);
    assert_eq!(metrics["observed_backlog_clear"], true);
}

#[test]
fn compiler_queue_follows_source_arrival_instead_of_lexical_ids_across_restart() {
    let root = test_dir();
    let mut store = Store::open(root.path()).unwrap();
    publish(&mut store, "z-arrived-first", "(pipe X (ridge))");
    publish(&mut store, "a-arrived-later", "(pipe X (ridge))");
    assert_eq!(store.next_check().unwrap().0, "z-arrived-first");
    drop(store);
    let mut store = Store::open(root.path()).unwrap();
    assert_eq!(
        store.next_check().unwrap().0,
        "z-arrived-first",
        "in-flight work recovers in source order"
    );
}

#[test]
fn shared_emission_matches_legacy_source_bytes_errors_and_token_rules() {
    use feat_math::{
        tokens::{count_tokens, Lang},
        transpile::{self, EmitOptions},
        GenomeIr,
    };
    let languages = ["rust", "python", "cpp17", "javascript"];
    for (program, expected_success) in [
        ("(pipe X (standardize) (ridge))", [true, true, true, true]),
        (
            "(pipe X (histgbm depth=2 rounds=2))",
            [true, true, false, false],
        ),
        ("this does not parse", [false; 4]),
    ] {
        let parsed = GenomeIr::parse(program);
        let pair = parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
            transpile::emit(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
        });
        let expected = [
            pair.as_ref().map(|p| p.rust.clone()).map_err(Clone::clone),
            pair.as_ref()
                .map(|p| p.python.clone())
                .map_err(Clone::clone),
            parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
                transpile::emit_cpp17(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
            }),
            parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
                transpile::emit_javascript(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
            }),
        ];
        assert_eq!(
            expected.each_ref().map(|row| row.is_ok()),
            expected_success,
            "fixture emitter outcomes changed: {program}"
        );
        let (actual, _emit_ns) = gp_source::emit(&occurrence("cross-surface", program)).unwrap();
        assert_eq!(actual.len(), 4);
        let mut expected_tokens = [0u64; 4];
        for (index, ((artifact, language), result)) in
            actual.iter().zip(languages).zip(&expected).enumerate()
        {
            assert_eq!(artifact.language, language);
            match result {
                Ok(source) => {
                    let kind = match index {
                        0 => Lang::Rust,
                        1 => Lang::Python,
                        2 => Lang::Cpp,
                        _ => Lang::JavaScript,
                    };
                    let counted = count_tokens(source, kind);
                    assert_eq!(artifact.source.as_deref(), Some(source.as_str()));
                    assert_eq!(
                        artifact.sha256.as_deref(),
                        Some(gp_source::hash(source.as_bytes()).as_str())
                    );
                    assert_eq!(artifact.status, Status::Pending);
                    assert_eq!(artifact.error, None);
                    assert_eq!(artifact.code_tokens, counted.code_tokens as u64);
                    assert_eq!(artifact.doc_tokens, counted.doc_tokens as u64);
                    assert_eq!(artifact.bytes, counted.bytes as u64);
                    expected_tokens[index] = (counted.code_tokens + counted.doc_tokens) as u64;
                }
                Err(error) => {
                    assert_eq!(artifact.status, Status::Failed);
                    assert_eq!(artifact.error.as_deref(), Some(error.as_str()));
                    assert_eq!(artifact.source, None);
                    assert_eq!(artifact.sha256, None);
                    assert_eq!(
                        (artifact.code_tokens, artifact.doc_tokens, artifact.bytes),
                        (0, 0, 0)
                    );
                }
            }
        }
        assert_eq!(feat_math::source_stream::emit_languages(program), expected);
        assert_eq!(
            feat_math::source_stream::emitted_tokens(program),
            expected_tokens
        );
    }
}
