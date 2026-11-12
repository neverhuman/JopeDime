//! The terminal's routing dialogue, driven through a PTY against a mock veox Cloud Workspace:
//! `/run --cloud` while signed out (the login step, no just-in-time upload), `/login`, the
//! upload consent, cloud progress as thinking cards, a limit refusal, and a finished local run
//! whose results are synced after the one-time question; then the public rule (owner rule
//! 2026-11-06): own files need a login, `/run` trains them in the cloud after the consent
//! (Enter is no), `--local` is for the demos, and `/demo` still runs here. Prints the
//! transcripts it saw.
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
            ("POST", p) if p == format!("{base}/code") => (200, format!(r#"{{"codeSha256":"{}"}}"#, header("x-code-sha256"))),
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    })
}

fn transcript(screen: &str, from: &str) -> String {
    screen[screen.find(from).unwrap_or(0)..].to_owned()
}

struct Fixture {
    _dir: tempfile::TempDir,
    home: std::path::PathBuf,
    csv: std::path::PathBuf,
}

fn fixture() -> Fixture {
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
    Fixture {
        _dir: dir,
        home,
        csv,
    }
}

fn wait(t: &mut pty::PtySession, s: &str, secs: u64) -> String {
    t.wait_for_text(s, Duration::from_secs(secs))
        .unwrap_or_else(|e| panic!("{e}"))
}

fn line(t: &mut pty::PtySession, text: &str) {
    t.type_text(text).unwrap();
    t.press_key(pty::Key::Enter).unwrap();
}

// Trains the user's own CSV on this machine, which the public edition (cloud-tls) refuses
// by design (owner rule 2026-11-06); the default-feature lanes run it.
#[cfg(not(feature = "public-edition"))]
#[test]
fn the_terminal_routes_logs_in_streams_and_syncs_results() {
    let f = fixture();
    let limited = Arc::new(AtomicBool::new(false));
    let server = Workspace::start(handler(Arc::clone(&limited)));
    let home_text = f.home.to_str().unwrap();
    let env = [
        ("JOPEDIME_HOME", home_text),
        ("JAIN_WORKSPACE", server.origin.as_str()),
        ("JAIN_NO_BROWSER", "1"),
        ("JAIN_NO_UPDATE_CHECK", "1"),
        ("NO_COLOR", "1"),
        ("TERM", "dumb"),
    ];
    let mut terminal = pty::PtySession::spawn_jopedime_with_env(&[], &env).unwrap();
    // Outlast `jope_client::discover::STARTUP_TIMEOUT`: the child prints this only after discovery.
    wait(
        &mut terminal,
        "engine background",
        (jope_client::discover::STARTUP_TIMEOUT + Duration::from_secs(15)).as_secs(),
    );
    for setting in [
        "\\landmarks none",
        "\\lang rust",
        "\\target price --task regression",
        "\\effort fast --evals 4",
    ] {
        line(&mut terminal, setting);
    }
    line(&mut terminal, &format!("\\open {}", f.csv.display()));
    wait(&mut terminal, "registered rent.csv as", 15);

    // 1. Signed out, `/run --cloud`: the login step, nothing uploaded; then /login.
    line(&mut terminal, "/run --cloud");
    wait(&mut terminal, "needs you signed in: type /login", 20);
    assert!(!server.calls().iter().any(|c| c == "POST /api/sessions"));
    line(&mut terminal, "/login");
    wait(
        &mut terminal,
        "Signed in: type /run to train the opened table",
        30,
    );
    line(&mut terminal, "/run --cloud");
    wait(&mut terminal, "Upload rent.csv (149 rows, 12 columns,", 20);
    line(&mut terminal, "y");
    let screen = wait(
        &mut terminal,
        "Next: predict new rows in your workspace:",
        30,
    );
    let cloud = transcript(&screen, "Pairing code");
    println!("=== TUI transcript: login, then a cloud run ===\n{cloud}");
    assert!(cloud.contains("Pairing code: WXYZ-9876"));
    assert!(!cloud.contains("secret-device-code"));
    assert!(
        cloud.contains("to train in the veox Cloud? [y/N]"),
        "{cloud}"
    );
    assert_eq!(
        cloud.matches("Thinking Container started").count(),
        1,
        "{cloud}"
    );
    assert_eq!(cloud.matches("Thinking Profiling the table").count(), 1);
    assert!(cloud.contains("Thinking Searching · step 2/4 · 40 evaluations · best rmse 0.2500"));
    assert!(cloud.contains("Selected 'price' as the target."));
    assert!(cloud.contains(&format!("{}/runs/{SESSION}", server.origin)));
    assert!(cloud.contains("/copy link"), "{cloud}");

    // 2. The same file again: consent remembered, the daily limit in plain words.
    limited.store(true, Ordering::SeqCst);
    line(&mut terminal, "/run --cloud");
    wait(&mut terminal, "Free includes 10 cloud runs a day", 20);
    let limit = transcript(&terminal.read_screen_as_text(), "Free includes 10");
    println!("=== TUI transcript: limit hit ===\n{limit}");
    assert!(limit.starts_with(
        "Free includes 10 cloud runs a day; you've used 10. Resets at 14:20 UTC. No run started. · Upgrade: https://veox.ai/pricing"
    ), "{limit}");

    // 3. This development build still trains own data here with plain /run: when it finishes,
    // the one-time results question, then the sync.
    line(&mut terminal, "/run");
    wait(
        &mut terminal,
        "Sync results (model + code, not your data)",
        60,
    );
    line(&mut terminal, "y");
    let screen = wait(&mut terminal, "Explore it in your workspace:", 60);
    let local = transcript(&screen, "No run started.");
    println!("=== TUI transcript: local run, then results sync ===\n{local}");
    assert!(local.contains(
        "Running locally on this machine · fast search (this build trains own data here) · 149 rows."
    ), "{local}");
    assert!(local.contains("Results: model synced."));
    let calls = server.calls();
    assert!(calls.iter().any(|c| c.ends_with("/model")));
    assert!(!calls.iter().any(|c| c.ends_with("/report")));
    assert_eq!(
        crate_state(&f.home)["results_sync"],
        serde_json::json!(true),
        "the answer is remembered"
    );
    drop(terminal);
    let _ = Command::new(BIN)
        .args(["serve", "--stop", "--force"])
        .env("JOPEDIME_HOME", &f.home)
        .output();
}

#[test]
fn under_the_public_rule_own_files_need_a_login_and_train_in_the_cloud() {
    let f = fixture();
    let server = Workspace::start(handler(Arc::new(AtomicBool::new(false))));
    let home_text = f.home.to_str().unwrap();
    let env = [
        ("JOPEDIME_HOME", home_text),
        ("JAIN_WORKSPACE", server.origin.as_str()),
        ("JAIN_NO_BROWSER", "1"),
        ("JAIN_NO_UPDATE_CHECK", "1"),
        ("JAIN_PUBLIC_EDITION", "1"),
        ("NO_COLOR", "1"),
        ("TERM", "dumb"),
    ];
    let mut terminal = pty::PtySession::spawn_jopedime_with_env(&[], &env).unwrap();
    // Outlast `jope_client::discover::STARTUP_TIMEOUT`: the child prints this only after discovery.
    wait(
        &mut terminal,
        "engine background",
        (jope_client::discover::STARTUP_TIMEOUT + Duration::from_secs(15)).as_secs(),
    );
    let open = format!("/open {}", f.csv.display());

    // Signed out: an own file does not open (JAIN_ALLOW_LOCAL_FILES has no effect here).
    line(&mut terminal, &open);
    let screen = wait(&mut terminal, "needs you signed in: type /login", 20);
    let gated = transcript(&screen, &open);
    println!("=== TUI transcript: signed out ===\n{gated}");
    assert!(gated.contains("https://app.veox.ai/register"), "{gated}");
    assert!(!gated.contains("registered rent.csv"), "{gated}");

    // The bundled demo still runs here.
    line(&mut terminal, "/demo regression_parkinsons");
    let screen = wait(&mut terminal, "Next: predict with", 120);
    let demo = transcript(&screen, "/demo regression_parkinsons");
    println!("=== TUI transcript: demo ===\n{demo}");
    assert!(
        demo.contains("Running locally on this machine · a bundled demo, no account needed"),
        "{demo}"
    );
    assert!(
        demo.contains("held-out"),
        "the result leads with the held-out score: {demo}"
    );

    // Signed in: the file opens, /run asks before uploading; Enter is no.
    line(&mut terminal, "/login");
    wait(&mut terminal, "Signed in to", 30);
    for setting in [
        "\\landmarks none",
        "\\lang rust",
        "\\target price --task regression",
    ] {
        line(&mut terminal, setting);
    }
    line(&mut terminal, &open);
    wait(&mut terminal, "Type /run to train it in the veox Cloud", 20);
    line(&mut terminal, "/run --local");
    wait(&mut terminal, "--local runs only the bundled examples", 20);
    line(&mut terminal, "/run");
    wait(&mut terminal, "to train in the veox Cloud? [y/N]", 20);
    line(&mut terminal, "");
    wait(
        &mut terminal,
        "Nothing was uploaded, and no run started.",
        20,
    );
    assert!(!server.calls().iter().any(|c| c == "POST /api/sessions"));
    line(&mut terminal, "/run");
    wait(&mut terminal, "to train in the veox Cloud? [y/N]", 20);
    line(&mut terminal, "y");
    let screen = wait(
        &mut terminal,
        "Next: predict new rows in your workspace:",
        30,
    );
    let cloud = transcript(&screen, "Training in the veox Cloud");
    println!("=== TUI transcript: own data in the cloud ===\n{cloud}");
    assert!(
        cloud.starts_with(
            "Training in the veox Cloud · your own data trains there (fast search) · 149 rows."
        ),
        "{cloud}"
    );
    assert!(cloud.contains("Uploading rent.csv"), "{cloud}");
    assert!(cloud.contains(&format!(
        "Open it in your workspace: {}/runs/{SESSION}",
        server.origin
    )));
    assert!(!cloud.contains("Running locally"), "{cloud}");
    drop(terminal);
    let _ = Command::new(BIN)
        .args(["serve", "--stop", "--force"])
        .env("JOPEDIME_HOME", &f.home)
        .output();
}

fn crate_state(home: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(home.join("state.json")).unwrap()).unwrap()
}
