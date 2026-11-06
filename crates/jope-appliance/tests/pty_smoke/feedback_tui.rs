//! `/feedback` in the terminal, driven through a PTY against a mock veox Cloud Workspace while
//! signed out: the usage line for an empty command, then one issue that reaches
//! `POST /api/feedback` with the documented body and no credential. Prints the transcript it saw.
#![cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "client",
    feature = "engine"
))]

use super::{pty, workspace};

use std::{process::Command, time::Duration};
use workspace::{Recorded, Workspace};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");

fn handler() -> Box<workspace::Handler> {
    Box::new(
        |r: &Recorded, _previous: &[Recorded]| match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/api/feedback") => (201, r#"{"id":"fb-test-1","status":"received"}"#.into()),
            _ => (404, r#"{"error":"not found"}"#.into()),
        },
    )
}

#[test]
fn feedback_from_the_terminal_reaches_the_workspace_signed_out() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let server = Workspace::start(handler());
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

    // An empty command explains itself and sends nothing.
    terminal.type_text("/feedback").unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    wait(&mut terminal, "usage: /feedback", 10);
    assert!(server.requests().iter().all(|r| r.path != "/api/feedback"));

    // One issue, signed out.
    terminal
        .type_text("/feedback issue the upload stalls at 90%")
        .unwrap();
    terminal.press_key(pty::Key::Enter).unwrap();
    let screen = wait(&mut terminal, "feedback sent to the VEOX team", 15);
    println!("=== TUI transcript: /feedback ===\n{screen}");
    assert!(screen.contains("ref fb-test-1"));
    assert!(screen.contains("without an account"));

    let sent: Vec<_> = server
        .requests()
        .into_iter()
        .filter(|r| r.method == "POST" && r.path == "/api/feedback")
        .collect();
    assert_eq!(sent.len(), 1, "exactly one feedback request");
    let request = &sent[0];
    assert!(
        !request.headers.contains_key("authorization"),
        "signed out, no credential is sent"
    );
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["kind"], "issue");
    assert_eq!(body["source"], "user");
    assert_eq!(body["summary"], "the upload stalls at 90%");
    assert_eq!(body["surface_hint"], "cli");
    assert!(body["client"].as_str().unwrap().starts_with("jopedime "));
    assert!(body.get("details").is_none());

    drop(terminal);
    let _ = Command::new(BIN)
        .args(["serve", "--stop", "--force"])
        .env("JOPEDIME_HOME", &home)
        .output();
}

/// A workspace without `POST /api/feedback` (today's production before the app deploy) gives a
/// plain sentence and a failed exit, and nothing else breaks.
#[test]
fn feedback_to_a_workspace_without_the_endpoint_says_so_plainly() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let server = Workspace::start(Box::new(|_: &Recorded, _: &[Recorded]| {
        (404, r#"{"error":"not found"}"#.into())
    }));
    let out = Command::new(BIN)
        .args(["cloud", "feedback", "issue", "the upload stalls"])
        .env("JOPEDIME_HOME", &home)
        .env("JAIN_WORKSPACE", &server.origin)
        .env("JAIN_ACCEPT_TERMS", "1")
        .env("JAIN_NO_UPDATE_CHECK", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("does not accept feedback from the CLI yet"),
        "{stderr}"
    );
}
