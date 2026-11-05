//! End to end against a mock veox Cloud Workspace that speaks the documented contract: a cloud
//! run with a just-in-time login that continues the same run, progress rendered once per event,
//! the remembered upload consent, a limit refusal in plain words, a finished local run's results
//! synced in order with a retry, the X-Jain-Device header, and the global first-run flags.

use super::workspace::{account, Recorded, Workspace, CLOUD_RUN, SESSION};
use super::BIN;
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

struct Home {
    _dir: tempfile::TempDir,
    root: PathBuf,
    csv: PathBuf,
}

fn home() -> Home {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("home");
    std::fs::create_dir(&root).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let csv = dir.path().join("rent.csv");
    let demo = include_str!("../../assets/demos/regression_apartment_rent.csv");
    let rows: Vec<&str> = demo.lines().take(150).collect();
    std::fs::write(&csv, rows.join("\n") + "\n").unwrap();
    Home {
        _dir: dir,
        root,
        csv,
    }
}

fn jopedime(home: &Home, workspace: &str, args: &[&str], extra: &[(&str, &str)]) -> Output {
    let mut command = Command::new(BIN);
    command
        .args(args)
        .env_clear()
        .env("HOME", home.root.parent().unwrap())
        .env("JOPEDIME_HOME", &home.root)
        .env("JAIN_WORKSPACE", workspace)
        .env("JAIN_NO_BROWSER", "1")
        .env("JAIN_NO_UPDATE_CHECK", "1")
        .env("JAIN_CLOUD_POLL_MS", "100")
        .env("JAIN_ACCEPT_TERMS", "1");
    for (key, value) in extra {
        command.env(key, value);
    }
    command.output().unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn host(request: &Recorded) -> String {
    format!(
        "http://{}",
        request.headers.get("host").cloned().unwrap_or_default()
    )
}

#[test]
fn a_cloud_run_logs_in_just_in_time_continues_and_streams_each_event_once() {
    let home = home();
    let limited = Arc::new(AtomicBool::new(false));
    let limit = Arc::clone(&limited);
    let server = Workspace::start(Box::new(move |r: &Recorded, previous: &[Recorded]| {
        let count = |path: &str| previous.iter().filter(|p| p.path == path).count();
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/api/cli/device") => (200, format!(
                r#"{{"device_code":"secret-device-code","user_code":"ABCD-1234","verification_uri":"{0}/cli/connect","verification_uri_complete":"{0}/cli/connect?user_code=ABCD-1234","expires_in":600,"interval":1}}"#,
                host(r)
            )),
            ("POST", "/api/cli/token") if count("/api/cli/token") == 0 => {
                (400, r#"{"error":"authorization_pending"}"#.into())
            }
            ("POST", "/api/cli/token") => (200, r#"{"access_token":"veox_testtoken","token_type":"Bearer","expires_in":2592000,"scope":"cli","token_id":"t1"}"#.into()),
            ("GET", "/api/cli/account") => (200, account("pro")),
            ("POST", "/api/sessions") if limit.load(Ordering::SeqCst) => (429, r#"{"error":"limit","limit":"daily_runs","used":10,"allowed":10,"resets_at":"2026-11-05T14:20:00Z","upgrade_url":"https://veox.ai/pricing","message":"You've started 10 runs today."}"#.into()),
            ("POST", "/api/sessions") => (200, format!(r#"{{"id":"{SESSION}","origin":"cli"}}"#)),
            ("GET", path) if path == format!("/api/sessions/{SESSION}") => {
                let polls = count(path);
                let e = |n: u32, m: &str| format!(r#"{{"sequence":{n},"phase":"p","message":"{m}","payload":{{"type":"phase"}},"recordedAt":"2026-11-05T10:00:0{n}Z"}}"#);
                let telemetry = r#"{"sequence":3,"phase":"search","message":"","payload":{"type":"telemetry","stage":"search","step":2,"steps":4,"evaluations":40,"bestScore":0.25,"bestScoreMetric":"rmse"},"recordedAt":"2026-11-05T10:00:03Z"}"#;
                let body = match polls {
                    0 => format!(r#"{{"state":"running","target":{{"column":"price","open":false}},"progressEvents":[{},{}]}}"#, e(1, "Container started"), e(2, "Profiling the table")),
                    1 => format!(r#"{{"state":"running","progressEvents":[{},{},{telemetry}]}}"#, e(1, "Container started"), e(2, "Profiling the table")),
                    _ => format!(r#"{{"state":"completed","progressEvents":[{},{},{telemetry}],"summary":{{"narrative":["Selected 'price' as the target.","CV rmse 0.25"]}}}}"#, e(1, "Container started"), e(2, "Profiling the table")),
                };
                (200, body)
            }
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    }));
    let csv = home.csv.to_str().unwrap();
    let out = jopedime(
        &home,
        &server.origin,
        &[
            "run",
            csv,
            "--target",
            "price",
            "--effort",
            "fast",
            "--cloud",
            "--consent-upload",
        ],
        &[],
    );
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    // The routing line, the just-in-time login, and the same run continuing.
    assert!(
        stderr.contains("Running in your veox Cloud Workspace · you asked for --cloud · 149 rows."),
        "{stderr}"
    );
    assert!(stderr
        .contains("This run needs your veox Cloud Workspace — opening your browser to log in…"));
    assert!(stderr.contains(&format!(
        "Open {}/cli/connect?user_code=ABCD-1234",
        server.origin
    )));
    assert!(
        !stderr.contains("secret-device-code"),
        "the device code never prints"
    );
    assert!(stderr.contains(&format!(
        "Cloud run started · {}/runs/{SESSION}",
        server.origin
    )));
    // Each progress sequence once, as thinking cards.
    for line in [
        "  Thinking Container started",
        "  Thinking Profiling the table",
        "  Thinking Searching · step 2/4 · 40 evaluations · best rmse 0.2500",
    ] {
        assert_eq!(stderr.matches(line).count(), 1, "{line}\n{stderr}");
    }
    assert!(stderr.contains("Selected 'price' as the target."));
    let stdout: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stdout["kind"], "cloud_run");
    assert_eq!(stdout["state"], "completed");
    assert_eq!(stdout["url"], format!("{}/runs/{SESSION}", server.origin));

    let requests = server.requests();
    let calls = server.calls();
    let login = calls
        .iter()
        .position(|c| c == "GET /api/cli/account")
        .unwrap();
    let upload = calls
        .iter()
        .position(|c| c == "POST /api/sessions")
        .unwrap();
    assert!(
        calls[0] == "POST /api/cli/device" && login < upload,
        "{calls:?}"
    );
    let post = &requests[upload];
    let body = post.text();
    let bytes = std::fs::metadata(&home.csv).unwrap().len();
    assert!(body.contains(&format!(
        "name=\"consent\"\r\n\r\n{{\"uploads_data\":true,\"rows\":149,\"columns\":12,\"bytes\":{bytes}}}"
    )), "{body}");
    for field in [
        "name=\"cloud_fallback\"\r\n\r\nuser_choice",
        "name=\"profile\"\r\n\r\nfast",
        "name=\"mode\"\r\n\r\ndetached",
        "filename=\"rent.csv\"",
    ] {
        assert!(body.contains(field), "{field}");
    }
    assert_eq!(post.headers["authorization"], "Bearer veox_testtoken");
    let device = &post.headers["x-jain-device"];
    assert!(
        device.len() == 32 && device.bytes().all(|b| b.is_ascii_hexdigit()),
        "{device}"
    );

    // Second run of the same file: no login, no consent flag (remembered), a limit refusal in
    // plain words, and with DO_NOT_TRACK no device header.
    limited.store(true, Ordering::SeqCst);
    let before = server.requests().len();
    let out = jopedime(
        &home,
        &server.origin,
        &[
            "run", csv, "--target", "price", "--effort", "fast", "--cloud",
        ],
        &[("DO_NOT_TRACK", "1")],
    );
    let stderr = text(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains(
        "Free includes 10 cloud runs a day; you've used 10. Resets at 14:20 UTC. Run locally instead? [Y/n] · Upgrade: https://veox.ai/pricing"
    ), "{stderr}");
    let later = &server.requests()[before..];
    assert!(
        !later.iter().any(|r| r.path.starts_with("/api/cli/device")),
        "no second login"
    );
    let post = later.iter().find(|r| r.path == "/api/sessions").unwrap();
    assert!(!post.headers.contains_key("x-jain-device"));
}

#[test]
fn a_finished_local_run_syncs_metadata_events_model_and_report_in_order() {
    let home = home();
    let server = Workspace::start(Box::new(|r: &Recorded, previous: &[Recorded]| {
        let header = |name: &str| r.headers.get(name).cloned().unwrap_or_default();
        let base = format!("/api/cli/runs/{CLOUD_RUN}");
        match (r.method.as_str(), r.path.as_str()) {
            ("GET", "/api/cli/account") => (200, account("free")),
            ("POST", "/api/cli/runs") => (
                200,
                format!(
                    r#"{{"id":"{CLOUD_RUN}","url":"{}/runs/{CLOUD_RUN}"}}"#,
                    host(r)
                ),
            ),
            ("POST", p) if p == format!("{base}/events") => (200, "{}".into()),
            // The first model upload fails transiently; the retry succeeds.
            ("POST", p)
                if p == format!("{base}/model") && !previous.iter().any(|q| q.path == p) =>
            {
                (503, "{}".into())
            }
            ("POST", p) if p == format!("{base}/model") => (
                200,
                format!(r#"{{"modelSha256":"{}"}}"#, header("x-model-sha256")),
            ),
            ("POST", p) if p == format!("{base}/report") => (
                200,
                format!(r#"{{"reportSha256":"{}"}}"#, header("x-report-sha256")),
            ),
            ("POST", p) if p == format!("{base}/code") => (
                200,
                format!(r#"{{"codeSha256":"{}"}}"#, header("x-code-sha256")),
            ),
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    }));
    let token = home.root.parent().unwrap().join("token");
    std::fs::write(&token, "veox_testtoken").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let login = jopedime(
        &home,
        &server.origin,
        &[
            "auth",
            "login",
            "--workspace",
            &server.origin,
            "--token-file",
            token.to_str().unwrap(),
        ],
        &[],
    );
    assert!(login.status.success(), "{}", text(&login.stderr));
    let on = jopedime(&home, &server.origin, &["cloud", "results", "on"], &[]);
    assert!(text(&on.stdout).contains("No training rows are uploaded, except that kernel, smoother and Gaussian-process models"), "{}", text(&on.stdout));
    let out = jopedime(
        &home,
        &server.origin,
        &[
            "run",
            home.csv.to_str().unwrap(),
            "--target",
            "price",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"4","wall_seconds":null}"#,
        ],
        &[],
    );
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let calls = server.calls();
    let at = |suffix: &str| {
        calls
            .iter()
            .position(|c| c.ends_with(suffix))
            .unwrap_or_else(|| panic!("{suffix} missing: {calls:?}\n{stderr}"))
    };
    let metadata = at("POST /api/cli/runs");
    let events = at("/events");
    let model = at("/model");
    let report = at("/report");
    assert!(
        metadata < events && events < model && model < report,
        "{calls:?}"
    );
    assert_eq!(
        calls.iter().filter(|c| c.ends_with("/model")).count(),
        2,
        "one retry"
    );
    if let Some(code) = calls.iter().position(|c| c.ends_with("/code")) {
        assert!(report < code);
    }
    assert!(stderr.contains("Results: model synced."), "{stderr}");
    assert!(stderr.contains("Results: report synced."));
    assert!(stderr.contains(&format!(
        "Explore it in your workspace: {}/runs/{CLOUD_RUN}",
        server.origin
    )));
    // Nothing that was uploaded besides the model carries a row of the table.
    let row = std::fs::read_to_string(&home.csv)
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .to_owned();
    for request in server.requests() {
        if !request.path.ends_with("/model") {
            assert!(
                !request.text().contains(&row),
                "{} carried a data row",
                request.path
            );
        }
        if request.path.ends_with("/report") {
            assert!(request.body.starts_with(b"%PDF-1.4"));
        }
    }
}

fn state(home: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(home.join("state.json")).unwrap()).unwrap()
}

#[test]
fn accept_terms_and_telemetry_flags_work_before_or_after_the_subcommand() {
    let home = home();
    let mut command = Command::new(BIN);
    let out = command
        .args(["--accept-terms", "--telemetry", "off", "--version"])
        .env_clear()
        .env("JOPEDIME_HOME", &home.root)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    let saved = state(&home.root);
    assert_eq!(saved["terms_source"], "flag");
    assert_eq!(saved["telemetry"], false);

    let other = self::home();
    let out = Command::new(BIN)
        .args(["serve", "--status", "--telemetry=on", "--accept-terms"])
        .env_clear()
        .env("JOPEDIME_HOME", &other.root)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert_eq!(state(&other.root)["telemetry"], true);

    // DO_NOT_TRACK still forces telemetry off.
    let dnt = self::home();
    Command::new(BIN)
        .args(["--telemetry", "on", "--version"])
        .env_clear()
        .env("JOPEDIME_HOME", &dnt.root)
        .env("DO_NOT_TRACK", "1")
        .output()
        .unwrap();
    assert_eq!(state(&dnt.root)["telemetry"], false);

    let bad = Command::new(BIN)
        .args(["--telemetry", "maybe", "--version"])
        .env_clear()
        .env("JOPEDIME_HOME", &dnt.root)
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));

    let help = Command::new(BIN)
        .args(["--help"])
        .env_clear()
        .output()
        .unwrap();
    let help = text(&help.stdout);
    assert!(help.contains("--accept-terms") && help.contains("--telemetry"));
    assert!(help.contains("https://veox.ai/terms"));
}
