// ---------------------------------------------------------------- W7-T02 CLI

fn cli(args: &[&str]) -> (bool, serde_json::Value, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let json = serde_json::from_str(stdout.trim()).unwrap_or(serde_json::Value::Null);
    (
        out.status.success(),
        json,
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn cli_plan_status_verify_cancel_round_trip() {
    let (_t, root) = tempdir();
    let k = keys(3);
    let c = contract(now_ms(), 3, &k);
    let file = root.join("contract.json");
    fs::write(&file, bytes(&c)).unwrap();
    let dir = root.join("run");
    let d = dir.to_str().unwrap();
    let (ok, plan, err) = cli(&["plan", "--run-dir", d, "--contract", file.to_str().unwrap()]);
    assert!(ok, "{err}");
    let sha = plan["plan_sha256"].as_str().unwrap().to_string();
    assert_eq!(plan["review"]["world_size"], 3);
    assert_eq!(plan["review"]["dispatch_started"], false);
    let (ok, status, _) = cli(&["status", "--run-dir", d]);
    assert!(ok);
    assert!(status["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["state"] == "scheduled"));
    let (ok, verify, _) = cli(&["verify", "--run-dir", d, "--plan-sha256", &sha]);
    assert!(ok);
    assert_eq!(verify["numerical_work_started"], false);
    let (ok, _, _) = cli(&["verify", "--run-dir", d, "--plan-sha256", &"0".repeat(64)]);
    assert!(!ok);
    let (ok, cancel, _) = cli(&[
        "cancel",
        "--run-dir",
        d,
        "--plan-sha256",
        &sha,
        "--reason",
        "stop",
    ]);
    assert!(ok);
    assert_eq!(cancel["outcome"], "cancelled");
    let (ok, again, _) = cli(&[
        "cancel",
        "--run-dir",
        d,
        "--plan-sha256",
        &sha,
        "--reason",
        "stop",
    ]);
    assert!(ok);
    assert_eq!(again["already_requested"], true);
    assert_eq!(again["journaled"], false);
}

#[test]
fn cli_refuses_mixed_legacy_arguments_and_bad_contracts_before_creating_state() {
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let d = dir.to_str().unwrap();
    let legacy_args = [
        "--root",
        "/x",
        "--state",
        "/x/s",
        "--graph-state",
        "/x/g",
        "--worker-state",
        "/x/w",
        "--config",
        "/x/c",
        "--config-sha256",
        "00",
        "--socket",
        "/x/sock",
        "--tenant",
        "t",
        "--project",
        "p",
    ];
    let mut mixed: Vec<&str> = legacy_args.to_vec();
    mixed.extend(["status", "--run-dir", d]);
    let (ok, _, err) = cli(&mixed);
    assert!(!ok && err.contains("take no fixed-loop arguments"), "{err}");
    let (ok, _, err) = cli(&["tick"]);
    assert!(!ok && err.contains("require --root"), "{err}");
    let k = keys(3);
    let mut bad = contract(now_ms(), 3, &k);
    bad.work_census[0].cpu_core_ms = 3_000_000;
    resign(&mut bad, &k);
    let file = root.join("bad.json");
    fs::write(&file, bytes(&bad)).unwrap();
    let (ok, _, err) = cli(&["plan", "--run-dir", d, "--contract", file.to_str().unwrap()]);
    assert!(!ok && err.contains("underfunded"), "{err}");
    assert!(!dir.exists());
    let help = Command::new(env!("CARGO_BIN_EXE_jope-native-loop"))
        .arg("--help")
        .output()
        .unwrap();
    let help = String::from_utf8_lossy(&help.stdout);
    for verb in [
        "bind",
        "bind-parent-decision",
        "tick",
        "serve",
        "plan",
        "status",
        "verify",
        "cancel",
    ] {
        assert!(
            help.lines().any(|l| l.trim_start().starts_with(verb)),
            "help lacks {verb}: {help}"
        );
    }
}
