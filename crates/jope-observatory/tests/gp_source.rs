use feat_math::source_stream::{Journal, Occurrence, SCHEMA};
use jope_observatory::gp_source::{self, Check, Status, Store};
use std::{
    fs,
    io::{BufReader, Cursor, Seek},
    path::Path,
    process::{Child, Command, Stdio},
    time::Duration,
};
use tempfile::TempDir;

fn test_dir() -> TempDir {
    let root = std::env::var_os("JOPE_GP_SOURCE_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests")
        });
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

fn occurrence(id: &str, program: &str) -> Occurrence {
    Occurrence {
        schema: SCHEMA.into(),
        id: id.into(),
        engine: "fixture".into(),
        coordinate: "generation/0/slot/0".into(),
        program: program.into(),
    }
}
fn publish(store: &mut Store, id: &str, program: &str) {
    let input = occurrence(id, program);
    let (artifacts, ns) = gp_source::emit(&input).unwrap();
    store.publish(input, artifacts, ns).unwrap();
}
fn drain(store: &mut Store) {
    while let Some((id, artifact, root)) = store.next_check() {
        let check = gp_source::check(&root, &artifact, 15).unwrap();
        assert_eq!(
            check.status,
            Status::Passed,
            "{}: {}",
            artifact.language,
            check.diagnostic
        );
        store.checked(id, &artifact, check).unwrap();
    }
}
fn counts(store: &Store) -> serde_json::Value {
    store.snapshot()["total"]["counts"].clone()
}

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
fn unsupported_and_invalid_programs_remain_errors_without_fabricated_tokens() {
    let directory = test_dir();
    let mut store = Store::open(directory.path()).unwrap();
    publish(&mut store, "invalid", "this does not parse");
    assert_eq!(counts(&store)["emission_errors"], 4);
    assert_eq!(counts(&store)["emitted_tokens"], 0);
    assert_eq!(store.snapshot()["total"]["artifact_yield"], 0.0);
    assert!(store.snapshot()["total"]["token_yield"].is_null());
    publish(&mut store, "histgbm", "(pipe X (histgbm depth=2 rounds=2))");
    assert!(counts(&store)["emission_errors"].as_u64().unwrap() >= 6);
    assert!(counts(&store)["emitted_tokens"].as_u64().unwrap() > 0);
    assert_eq!(counts(&store)["artifacts"], 8);
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

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start(journals: &Path, state: &Path) -> (Server, std::net::SocketAddr) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let child = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--listen",
            &address.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    for _ in 0..200 {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "source service exited"
        );
        if std::net::TcpStream::connect(address).is_ok() {
            return (server, address);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("source service did not start");
}

#[tokio::test]
async fn websocket_delivers_full_sources_checks_and_cursor_replay() {
    use futures_util::StreamExt;
    let journals = test_dir();
    let state = test_dir();
    let (_server, address) = start(journals.path(), state.path());
    let url = format!("ws://{address}/api/v1/gp/source");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let mut journal = Journal::create(journals.path()).unwrap();
    journal
        .append("fixture", "slot/0", "(pipe X (standardize) (ridge))")
        .unwrap();
    let mut events = Vec::new();
    for _ in 0..5 {
        let message = tokio::time::timeout(Duration::from_secs(30), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let event: serde_json::Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
        events.push(event);
    }
    assert_eq!(events[0]["kind"], "source");
    assert_eq!(events[0]["artifacts"].as_array().unwrap().len(), 4);
    assert!(events[0]["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .all(|a| a["source"].as_str().unwrap().len() > 100));
    assert!(events[1..]
        .iter()
        .all(|e| e["kind"] == "checked" && e["status"] == "passed"));
    drop(ws);
    let (mut replay, _) = tokio_tungstenite::connect_async(format!("{url}?after=3"))
        .await
        .unwrap();
    for sequence in [4, 5] {
        let message = tokio::time::timeout(Duration::from_secs(5), replay.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let event: serde_json::Value = serde_json::from_str(message.to_text().unwrap()).unwrap();
        assert_eq!(event["sequence"], sequence);
    }
    let mut request =
        tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(&url)
            .unwrap();
    request
        .headers_mut()
        .insert("origin", "http://hostile.invalid".parse().unwrap());
    assert!(tokio_tungstenite::connect_async(request).await.is_err());
}

// Match drain()'s syntax-check budget so a healthy C++ compiler under CI load does
// not become a second failure in the missing-Rust test. The timeout fixture below
// deliberately outlives this budget; production defaults and limits are unchanged.
fn once_with_rustc(journals: &Path, state: &Path, rustc: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--compile-timeout-seconds",
            "15",
            "--once",
        ])
        .env("JOPE_GP_RUSTC", rustc)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
#[test]
fn missing_compiler_is_loss_and_once_restart_does_not_double_count() {
    let journals = test_dir();
    let state = test_dir();
    Journal::create(journals.path())
        .unwrap()
        .append("fixture", "slot/0", "(pipe X (ridge))")
        .unwrap();
    let first = once_with_rustc(
        journals.path(),
        state.path(),
        Path::new("/nonexistent-jope-rustc"),
    );
    let m = &first["total"]["counts"];
    assert_eq!(m["compile_errors"], 1, "{first}");
    assert_eq!(m["passed_artifacts"], 3, "{first}");
    assert_eq!(first["languages"]["rust"]["counts"]["compile_errors"], 1);
    for language in ["python", "cpp17", "javascript"] {
        assert_eq!(
            first["languages"][language]["counts"]["passed_artifacts"], 1,
            "{language}: {first}"
        );
    }
    assert!(m["lost_tokens"].as_u64().unwrap() > 0);
    let second = once_with_rustc(
        journals.path(),
        state.path(),
        Path::new("/nonexistent-jope-rustc"),
    );
    assert_eq!(first["sequence"], second["sequence"]);
    assert_eq!(first["total"]["counts"], second["total"]["counts"]);
}
#[cfg(unix)]
#[test]
fn compiler_timeout_charges_loss_and_retains_the_timeout_exit() {
    use std::os::unix::fs::PermissionsExt;
    let journals = test_dir();
    let state = test_dir();
    let tool = journals.path().join("slow-rustc");
    fs::write(&tool,"#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo fixture-slow-compiler; exit 0; fi\nsleep 60\n").unwrap();
    fs::set_permissions(&tool, fs::Permissions::from_mode(0o755)).unwrap();
    Journal::create(journals.path())
        .unwrap()
        .append("fixture", "slot/0", "(pipe X (ridge))")
        .unwrap();
    let result = once_with_rustc(journals.path(), state.path(), &tool);
    assert_eq!(result["total"]["counts"]["compile_errors"], 1);
    let events = fs::read_to_string(state.path().join("events.jsonl")).unwrap();
    assert!(events
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .any(|e| e["kind"] == "checked"
            && e["language"] == "rust"
            && e["status"] == "failed"
            && e["diagnostic"].as_str().unwrap().contains("signal")
            && e["tool"].as_str().unwrap().contains("timeout=15s")));
}

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
