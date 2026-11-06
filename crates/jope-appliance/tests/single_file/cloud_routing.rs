//! End to end against a mock veox Cloud Workspace that speaks the documented contract: a cloud
//! run from a signed-in account, progress rendered once per event, the remembered upload
//! consent, a limit refusal in plain words, the public rule's login, waitlist and consent gates
//! (owner rule 2026-11-06), a finished local run's results synced in order with a retry, the
//! X-Jain-Device header, and the global first-run flags.

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

/// `jain auth login --token-file` against the mock workspace.
fn login(home: &Home, workspace: &str) {
    let token = home.root.parent().unwrap().join("token");
    std::fs::write(&token, "veox_testtoken").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let out = jopedime(
        home,
        workspace,
        &[
            "auth",
            "login",
            "--workspace",
            workspace,
            "--token-file",
            token.to_str().unwrap(),
        ],
        &[],
    );
    assert!(out.status.success(), "{}", text(&out.stderr));
}

fn host(request: &Recorded) -> String {
    format!(
        "http://{}",
        request.headers.get("host").cloned().unwrap_or_default()
    )
}

#[test]
fn a_cloud_run_from_a_signed_in_account_streams_each_event_once() {
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
    // Signed out, --cloud stops before anything is registered or uploaded: no just-in-time login.
    let out = jopedime(
        &home,
        &server.origin,
        &[
            "run",
            csv,
            "--target",
            "price",
            "--cloud",
            "--upload-consent",
        ],
        &[],
    );
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains("needs you signed in: run `jain auth login`")
            && stderr.contains("https://app.veox.ai/register"),
        "{stderr}"
    );
    assert!(server.calls().is_empty(), "{:?}", server.calls());
    login(&home, &server.origin);
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
            // The rc.3 spelling still works.
            "--consent-upload",
        ],
        &[],
    );
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stderr.contains("Training in the veox Cloud · you asked for --cloud · 149 rows."),
        "{stderr}"
    );
    assert!(
        stderr.contains("Upload rent.csv (149 rows, 12 columns, ")
            && stderr.contains("to train in the veox Cloud? [y/N] yes (--upload-consent)"),
        "{stderr}"
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
    assert!(stderr.contains(&format!(
        "Next: predict new rows in your workspace: {}/runs/{SESSION}/predict",
        server.origin
    )));
    let stdout: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stdout["kind"], "cloud_run");
    assert_eq!(stdout["state"], "completed");
    assert_eq!(stdout["url"], format!("{}/runs/{SESSION}", server.origin));

    let requests = server.requests();
    let calls = server.calls();
    let admission = calls
        .iter()
        .rposition(|c| c == "GET /api/cli/account")
        .unwrap();
    let upload = calls
        .iter()
        .position(|c| c == "POST /api/sessions")
        .unwrap();
    assert!(
        !calls.iter().any(|c| c.starts_with("POST /api/cli/device")) && admission < upload,
        "the plan is read before the upload, with no device login: {calls:?}"
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
        "Free includes 10 cloud runs a day; you've used 10. Resets at 14:20 UTC. No run started. · Upgrade: https://veox.ai/pricing"
    ), "{stderr}");
    assert!(!stderr.contains("Run locally"), "{stderr}");
    let later = &server.requests()[before..];
    assert!(
        !later.iter().any(|r| r.path.starts_with("/api/cli/device")),
        "no second login"
    );
    let post = later.iter().find(|r| r.path == "/api/sessions").unwrap();
    assert!(!post.headers.contains_key("x-jain-device"));
}

/// The public edition's rule (owner rule 2026-11-06), applied here with JAIN_PUBLIC_EDITION=1:
/// own data trains only in the veox Cloud, after a login on an admitted account and the user's
/// consent; `--local`, `fit` and `fit-predict` take only the bundled examples, which need no
/// account; `--wait` is checked before anything starts.
#[test]
fn the_public_rule_gates_own_data_and_keeps_the_bundled_examples_local() {
    let home = home();
    let waitlisted = Arc::new(AtomicBool::new(true));
    let waiting = Arc::clone(&waitlisted);
    let server = Workspace::start(Box::new(move |r: &Recorded, previous: &[Recorded]| {
        let count = |path: &str| previous.iter().filter(|p| p.path == path).count();
        match (r.method.as_str(), r.path.as_str()) {
            ("GET", "/api/cli/account") if waiting.load(Ordering::SeqCst) => (
                200,
                account("waitlist").replace(r#""mayRun":true"#, r#""mayRun":false"#),
            ),
            ("GET", "/api/cli/account") => (200, account("free")),
            ("POST", "/api/sessions") => (200, format!(r#"{{"id":"{SESSION}","origin":"cli"}}"#)),
            ("GET", path) if path == format!("/api/sessions/{SESSION}") => (
                200,
                if count(path) == 0 {
                    r#"{"state":"running","progressEvents":[{"sequence":1,"message":"Container started","payload":{"type":"phase"}}]}"#.to_owned()
                } else {
                    r#"{"state":"completed","progressEvents":[],"headline":{"basis":"real_holdout","rows":30,"score":{"label":"RMSE","metric":"rmse","value":101.5,"direction":"minimize","unit":"price"},"baselines":[{"kind":"linear","description":"a linear model","rows":30,"score":{"label":"RMSE","value":180.25},"lift":78.75,"outcome":"veox"}],"notes":[]},"method":{"name":"Bayesian model averaging"},"summary":{"final_model":"bma","narrative":["Selected 'price' as the target.","Selected bma as the served model."]}}"#.to_owned()
                },
            ),
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    }));
    let public = [("JAIN_PUBLIC_EDITION", "1")];
    let csv = home.csv.to_str().unwrap();
    let run = |args: &[&str]| {
        let out = jopedime(&home, &server.origin, args, &public);
        (out.status.code(), text(&out.stderr), text(&out.stdout))
    };

    // --wait is checked before anything starts (journey J2-F05).
    let (code, stderr, _) = run(&["run", csv, "--target", "price", "--wait", "120"]);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("--wait") && stderr.contains("0..=60"),
        "{stderr}"
    );

    // Signed out: own data stops with the login step and the waitlist link; nothing is sent.
    let (code, stderr, _) = run(&["run", csv, "--target", "price"]);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(stderr.contains("Your own data trains in the veox Cloud, so it needs you signed in: run `jain auth login`"), "{stderr}");
    assert!(stderr.contains("Accounts open from the waitlist: https://app.veox.ai/register"));
    for fit in [
        &["fit", "--train", csv, "--target", "price"][..],
        &[
            "fit-predict",
            "--train",
            csv,
            "--target",
            "price",
            "--input",
            csv,
            "--output",
            "p.csv",
        ],
    ] {
        let (code, stderr, _) = run(fit);
        assert_eq!(code, Some(2), "{stderr}");
        assert!(stderr.contains("needs you signed in"), "{stderr}");
    }
    assert!(server.calls().is_empty(), "{:?}", server.calls());

    // The bundled example runs here, signed out.
    let (code, stderr, stdout) = run(&[
        "run",
        "example/regression_parkinsons.csv",
        "--target",
        "target",
        "--effort",
        "fast",
        "--budget",
        r#"{"evaluations":"2","wall_seconds":null}"#,
    ]);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stdout.contains(r#""state":"completed""#), "{stdout}");
    assert!(server.calls().is_empty(), "a demo talks to no workspace");

    // Waitlisted: the waitlist message, no upload.
    login(&home, &server.origin);
    let (code, stderr, _) = run(&["run", csv, "--target", "price"]);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(stderr.contains("Your account (ada@example.test) is on the waitlist; we'll email you when you're admitted"), "{stderr}");

    // Admitted (Free): --local is for the bundled examples; fit points to `jain run`.
    waitlisted.store(false, Ordering::SeqCst);
    let (code, stderr, _) = run(&["run", csv, "--target", "price", "--local"]);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("--local runs only the bundled examples"),
        "{stderr}"
    );
    let (code, stderr, _) = run(&["fit", "--train", csv, "--target", "price"]);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("`jain run FILE.csv --target COLUMN`"),
        "{stderr}"
    );

    // No terminal and no --upload-consent: the error names the flag, and nothing is uploaded.
    let (code, stderr, _) = run(&["run", csv, "--target", "price"]);
    assert_ne!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("needs the user's consent: ask them first, then pass --upload-consent"),
        "{stderr}"
    );
    assert!(!server.calls().iter().any(|c| c == "POST /api/sessions"));

    // With the consent: the cloud run, Free's Fast search for the default medium effort, the
    // held-out result with its baseline, the workspace link and the next steps.
    let (code, stderr, stdout) = run(&["run", csv, "--target", "price", "--upload-consent"]);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains(
            "Training in the veox Cloud · your own data trains there (medium search) · 149 rows."
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("Free runs the Fast search in the veox Cloud"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Result: RMSE 101.5 price (lower is better) on 30 held-back real rows."),
        "{stderr}"
    );
    assert!(
        stderr.contains(
            "Baseline on the same rows (a linear model): RMSE 180.2 price · VEOX ahead by 78.75."
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("Model: Bayesian model averaging.") && !stderr.contains("Selected bma"),
        "{stderr}"
    );
    assert!(stderr.contains(&format!(
        "Open it in your workspace: {}/runs/{SESSION}",
        server.origin
    )));
    assert!(
        stderr.contains("Next: predict new rows in your workspace:"),
        "{stderr}"
    );
    assert!(!stderr.contains("Running locally"), "{stderr}");
    let line: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(line["state"], "completed");
    let post = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/api/sessions")
        .unwrap();
    let body = post.text();
    assert!(body.contains("name=\"profile\"\r\n\r\nfast"), "{body}");
    assert!(
        body.contains("name=\"cloud_fallback\"\r\n\r\nno_gpu"),
        "{body}"
    );
}

/// The bundled demo produces a champion, in every edition and signed out (journey J6: a demo
/// that ended "no champion" must never ship). `--json` gives the final run snapshot.
#[test]
fn the_bundled_demo_runs_here_signed_out_and_produces_a_champion() {
    let home = home();
    let out = jopedime(
        &home,
        "http://127.0.0.1:9",
        &["demo", "run", "regression_parkinsons", "--json"],
        &[("JAIN_PUBLIC_EDITION", "1")],
    );
    let stderr = text(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["run"]["state"], "completed", "{value}");
    assert!(
        value["run"]["best"]["algorithm_id"].is_string(),
        "no champion: {value}"
    );
    let _ = Command::new(BIN)
        .args(["serve", "--stop", "--force"])
        .env("JOPEDIME_HOME", &home.root)
        .output();
}

// Trains the user's own CSV on this machine, which the public edition (cloud-tls) refuses
// by design (owner rule 2026-11-06); the default-feature lanes run it.
#[cfg(not(feature = "public-edition"))]
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
            ("POST", p) if p == format!("{base}/code") => (
                200,
                format!(r#"{{"codeSha256":"{}"}}"#, header("x-code-sha256")),
            ),
            _ => (404, r#"{"error":"not found"}"#.into()),
        }
    }));
    login(&home, &server.origin);
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
    assert!(metadata < events && events < model, "{calls:?}");
    assert!(
        !calls.iter().any(|c| c.ends_with("/report")),
        "the CLI writes no report PDF"
    );
    assert_eq!(
        calls.iter().filter(|c| c.ends_with("/model")).count(),
        2,
        "one retry"
    );
    let last_model = calls.iter().rposition(|c| c.ends_with("/model")).unwrap();
    if let Some(code) = calls.iter().position(|c| c.ends_with("/code")) {
        assert!(last_model < code);
    }
    assert!(stderr.contains("Results: model synced."), "{stderr}");
    // model_published comes in an events batch after the model upload, with its digest.
    let requests = server.requests();
    let model_sha = requests
        .iter()
        .rev()
        .find(|r| r.path.ends_with("/model"))
        .map(|r| r.headers["x-model-sha256"].clone())
        .unwrap();
    let published = requests
        .iter()
        .position(|r| {
            r.path.ends_with("/events")
                && r.text().contains("\"model_published\"")
                && r.text().contains(&model_sha)
        })
        .unwrap_or_else(|| panic!("no model_published: {calls:?}"));
    assert!(published > last_model);
    // The first events batch names the target, task, real metric and direction, and the profile;
    // the metadata label is the file and target, not the run id.
    let first_events = requests
        .iter()
        .find(|r| r.path.ends_with("/events"))
        .unwrap()
        .text();
    assert!(
        first_events.contains(r#""kind":"run_started""#),
        "{first_events}"
    );
    assert!(
        first_events.contains(r#""target":"price""#),
        "{first_events}"
    );
    assert!(
        first_events.contains(r#""task":"regression""#),
        "{first_events}"
    );
    assert!(
        first_events.contains(r#""metric":"rmse""#)
            && first_events.contains(r#""direction":"minimize""#),
        "{first_events}"
    );
    assert!(
        first_events.contains(r#""kind":"dataset_profiled""#),
        "{first_events}"
    );
    assert!(
        !first_events.contains(r#""metric":"score""#),
        "{first_events}"
    );
    let metadata_body = requests
        .iter()
        .find(|r| r.path == "/api/cli/runs")
        .unwrap()
        .text();
    assert!(
        metadata_body.contains(r#""label":"rent.csv · price""#),
        "{metadata_body}"
    );
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
