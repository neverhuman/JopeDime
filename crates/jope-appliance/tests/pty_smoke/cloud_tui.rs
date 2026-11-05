//! The terminal's routing dialogue, driven through a PTY against a mock veox Cloud Workspace:
//! `/run --cloud` while signed out (just-in-time login, then the same run continues), the upload
//! consent, cloud progress as thinking cards, a limit refusal, and a finished local run whose
//! results are synced after the one-time question. Prints the transcripts it saw.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "client",
    feature = "engine"
))]

use super::{pty, workspace};

use std::{
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use workspace::{account, Recorded, Workspace, CLOUD_RUN, SESSION};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");

fn host(request: &Recorded) -> String {
    format!(
        "http://{}",
        request.headers.get("host").cloned().unwrap_or_default()
    )
}

fn handler(limited: Arc<AtomicBool>) -> Box<workspace::Handler> {
    Box::new(move |r: &Recorded, previous: &[Recorded]| {
        let count = |path: &str| previous.iter().filter(|p| p.path == path).count();
        let header = |name: &str| r.headers.get(name).cloned().unwrap_or_default();
        let base = format!("/api/cli/runs/{CLOUD_RUN}");
        let event = |n: u32, m: &str| {
            format!(
                r#"{{"sequence":{n},"phase":"p","message":"{m}","payload":{{"type":"phase"}}}}"#
            )
        };
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/api/cli/device") => (200, format!(
                r#"{{"device_code":"secret-device-code","user_code":"WXYZ-9876","verification_uri":"{0}/cli/connect","verification_uri_complete":"{0}/cli/connect?user_code=WXYZ-9876","expires_in":600,"interval":1}}"#,
                host(r)
            )),
            ("POST", "/api/cli/token") if count("/api/cli/token") == 0 => (400, r#"{"error":"authorization_pending"}"#.into()),
            ("POST", "/api/cli/token") => (200, r#"{"access_token":"veox_testtoken","token_type":"Bearer","expires_in":2592000,"scope":"cli","token_id":"t1"}"#.into()),
            ("GET", "/api/cli/account") => (200, account("free")),
            ("POST", "/api/sessions") if limited.load(Ordering::SeqCst) => (429, r#"{"error":"limit","limit":"daily_runs","used":10,"allowed":10,"resets_at":"2026-11-05T14:20:00Z","upgrade_url":"https://veox.ai/pricing","message":"You've started 10 runs today."}"#.into()),
            ("POST", "/api/sessions") => (200, format!(r#"{{"id":"{SESSION}","origin":"cli"}}"#)),
            ("GET", p) if p == format!("/api/sessions/{SESSION}") => {
                let telemetry = r#"{"sequence":3,"phase":"search","message":"","payload":{"type":"telemetry","stage":"search","step":2,"steps":4,"evaluations":40,"bestScore":0.25,"bestScoreMetric":"rmse","etaSeconds":12}}"#;
                let events = format!("{},{},{telemetry}", event(1, "Container started"), event(2, "Profiling the table"));
                (200, if count(p) < 2 {
                    format!(r#"{{"state":"running","progressEvents":[{events}]}}"#)
                } else {
                    format!(r#"{{"state":"completed","progressEvents":[{events}],"summary":{{"narrative":["Selected 'price' as the target.","Best model: CV rmse 0.25 on 149 rows."]}}}}"#)
                })
            }
            ("POST", "/api/cli/runs") => (200, format!(r#"{{"id":"{CLOUD_RUN}","url":"{}/runs/{CLOUD_RUN}"}}"#, host(r))),
            ("POST", p) if p == format!("{base}/events") => (200, "{}".into()),
            ("POST", p) if p == format!("{base}/model") => (200, format!(r#"{{"modelSha256":"{}"}}"#, header("x-model-sha256"))),
            ("POST", p) if p == format!("{base}/report") => (200, format!(r#"{{"reportSha256":"{}"}}"#, header("x-report-sha256"))),
            ("POST", p) if p == format!("{base}/code") => (200, format!(r#"{{"codeSha256":"{}"}}"#, header("x-code-sha256"))),
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    })
}

fn transcript(screen: &str, from: &str) -> String {
    screen[screen.find(from).unwrap_or(0)..].to_owned()
}

#[test]
fn the_terminal_routes_logs_in_just_in_time_streams_and_syncs_results() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let csv = dir.path().join("rent.csv");
    let demo = include_str!("../../assets/demos/regression_apartment_rent.csv");
    std::fs::write(
        &csv,
        demo.lines().take(150).collect::<Vec<_>>().join("\n") + "\n",
    )
    .unwrap();
    let limited = Arc::new(AtomicBool::new(false));
    let server = Workspace::start(handler(Arc::clone(&limited)));
    let home_text = home.to_str().unwrap();
    let env = [
        ("JOPEDIME_HOME", home_text),
        ("JAIN_WORKSPACE", server.origin.as_str()),
        ("JAIN_NO_BROWSER", "1"),
        ("JAIN_NO_UPDATE_CHECK", "1"),
        ("NO_COLOR", "1"),
        ("TERM", "dumb"),
    ];
    let mut terminal = pty::PtySession::spawn_jopedime_with_env(&[], &env).unwrap();
    let wait = |t: &mut pty::PtySession, s: &str, secs: u64| {
        t.wait_for_text(s, Duration::from_secs(secs))
            .unwrap_or_else(|e| panic!("{e}"))
    };
    wait(&mut terminal, "engine background", 15);
    for line in [
        "\\landmarks none",
        "\\lang rust",
        "\\target price --task regression",
        "\\effort fast --evals 4",
    ] {
        terminal.type_text(line).unwrap();
        terminal.press_key(pty::Key::Enter).unwrap();
    }
    terminal
        .type_text(&format!("\\open {}", csv.display()))
        .unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    wait(&mut terminal, "registered rent.csv as", 15);

    // 1. Signed out, `/run --cloud`: log in just in time, then the same run continues.
    terminal.type_text("/run --cloud").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    wait(&mut terminal, "Upload rent.csv (149 rows,", 20);
    terminal.type_text("y").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    let screen = wait(&mut terminal, "Open it in your workspace:", 20);
    let cloud = transcript(&screen, "/run --cloud");
    println!("=== TUI transcript: cloud run with just-in-time login ===\n{cloud}");
    assert!(cloud
        .contains("This run needs your veox Cloud Workspace — opening your browser to log in…"));
    assert!(cloud.contains("Pairing code: WXYZ-9876"));
    assert!(!cloud.contains("secret-device-code"));
    assert_eq!(
        cloud.matches("Thinking Container started").count(),
        1,
        "{cloud}"
    );
    assert_eq!(cloud.matches("Thinking Profiling the table").count(), 1);
    assert!(cloud.contains("Thinking Searching · step 2/4 · 40 evaluations · best rmse 0.2500"));
    assert!(cloud.contains("Selected 'price' as the target."));
    assert!(cloud.contains(&format!("{}/runs/{SESSION}", server.origin)));

    // 2. The same file again: consent remembered, the daily limit in plain words, run locally? no.
    limited.store(true, Ordering::SeqCst);
    terminal.type_text("/run --cloud").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    let screen = wait(&mut terminal, "Free includes 10 cloud runs a day", 20);
    terminal.type_text("n").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    let screen = wait(&mut terminal, "Not started.", 10)
        .len()
        .max(screen.len());
    let _ = screen;
    let limit = transcript(&terminal.read_screen_as_text(), "Free includes 10");
    println!("=== TUI transcript: limit hit ===\n{limit}");
    assert!(limit.starts_with(
        "Free includes 10 cloud runs a day; you've used 10. Resets at 14:20 UTC. Run locally instead? [Y/n] · Upgrade: https://veox.ai/pricing"
    ));

    // 3. A local run while signed in: when it finishes, the one-time question, then the sync.
    terminal.type_text("/run").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    wait(
        &mut terminal,
        "Sync results (model + report, not your data)",
        60,
    );
    terminal.type_text("y").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    let screen = wait(&mut terminal, "Explore it in your workspace:", 60);
    let local = transcript(&screen, "Not started.");
    println!("=== TUI transcript: local run, then results sync ===\n{local}");
    assert!(local.contains("Running locally on this machine · fast search · 149 rows."));
    assert!(local.contains("Results: model synced."));
    assert!(local.contains("Results: report synced."));
    let calls = server.calls();
    let model = calls.iter().position(|c| c.ends_with("/model")).unwrap();
    let report = calls.iter().position(|c| c.ends_with("/report")).unwrap();
    assert!(model < report);
    assert_eq!(
        crate_state(&home)["results_sync"],
        serde_json::json!(true),
        "the answer is remembered"
    );
    drop(terminal);
    let _ = Command::new(BIN)
        .args(["serve", "--stop", "--force"])
        .env("JOPEDIME_HOME", &home)
        .output();
}

fn crate_state(home: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(home.join("state.json")).unwrap()).unwrap()
}
