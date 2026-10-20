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
    let directory = TempDir::new().unwrap();
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
    let directory = TempDir::new().unwrap();
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
    let directory = TempDir::new().unwrap();
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
    let directory = TempDir::new().unwrap();
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
    let directory = TempDir::new().unwrap();
    fs::write(directory.path().join("events.jsonl"), "bad\n").unwrap();
    assert!(Store::open(directory.path()).is_err());
    fs::write(directory.path().join("events.jsonl"), "{\"schema\":").unwrap();
    assert!(Store::open(directory.path()).is_err());
}

#[test]
fn journals_keep_identical_occurrences_distinct_and_unicode_counts_do_not_panic() {
    let directory = TempDir::new().unwrap();
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
    let journals = TempDir::new().unwrap();
    let state = TempDir::new().unwrap();
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

fn once_with_rustc(journals: &Path, state: &Path, rustc: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--compile-timeout-seconds",
            "1",
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
    let journals = TempDir::new().unwrap();
    let state = TempDir::new().unwrap();
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
    assert_eq!(m["compile_errors"], 1);
    assert_eq!(m["passed_artifacts"], 3);
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
    let journals = TempDir::new().unwrap();
    let state = TempDir::new().unwrap();
    let tool = journals.path().join("slow-rustc");
    fs::write(&tool,"#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo fixture-slow-compiler; exit 0; fi\nsleep 5\n").unwrap();
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
            && e["diagnostic"].as_str().unwrap().contains("signal")));
}
