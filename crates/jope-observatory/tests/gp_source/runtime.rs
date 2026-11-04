use super::*;

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
fn rust_compiler_kill_is_recorded_as_failed_lost_work() {
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
    assert!(result["total"]["counts"]["lost_tokens"].as_u64().unwrap() > 0);
    let events = fs::read_to_string(state.path().join("events.jsonl")).unwrap();
    let occurrence_id = events
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|event| event["kind"] == "source")
        .and_then(|event| event["occurrence"]["id"].as_str().map(str::to_owned))
        .unwrap();
    assert!(events
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .any(|e| e["kind"] == "checked"
            && e["occurrence_id"].as_str() == Some(occurrence_id.as_str())
            && e["language"] == "rust"
            && e["status"] == "failed"
            && e["diagnostic"].as_str().unwrap().contains("signal")
            && e["tool"].as_str().unwrap().contains("slow-rustc")
            && e["tool"].as_str().unwrap().contains("timeout=15s")
            && e["elapsed_ns"].as_u64().unwrap() >= 15_000_000_000));
}
