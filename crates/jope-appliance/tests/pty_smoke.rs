#[path = "support/pty.rs"]
mod pty;

const SENTINELS: [&str; 4] = [
    "T2_DSL_SENTINEL_8f27",
    "T2_OPERATOR_SENTINEL_19ac",
    "T2_LINEAGE_SENTINEL_43de",
    "T2_ROUND_SENTINEL_71b0",
];

fn assert_no_internal_sentinels(output: &str) {
    for sentinel in SENTINELS {
        assert!(
            !output.contains(sentinel),
            "public CLI/PTY output leaked {sentinel}: {output}"
        );
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn jopedime_help_runs_in_a_pty_and_resize_is_applied() -> std::io::Result<()> {
    use std::time::Duration;

    let mut session = pty::PtySession::spawn_jopedime(&["--help"])?;
    session.resize(100, 40)?;
    assert_eq!(session.size()?, (100, 40));

    let screen = session.wait_for_text("Usage:", Duration::from_secs(10))?;
    assert!(
        screen.contains("jopedime"),
        "unexpected help output: {screen}"
    );
    assert_no_internal_sentinels(&screen);
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn dataset_id_error_does_not_echo_private_sentinel_in_cli_or_pty() -> std::io::Result<()> {
    use std::time::Duration;

    let mut session =
        pty::PtySession::spawn_jopedime(&["dataset", "show", SENTINELS[0], "--json"])?;
    let transcript = session.wait_for_text("invalid dataset id", Duration::from_secs(10))?;
    assert_no_internal_sentinels(&transcript);
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn live_dataset_cli_and_terminal_surfaces_do_not_echo_private_cells() -> std::io::Result<()> {
    use std::{
        fs,
        process::{Child, Command, Output, Stdio},
        thread,
        time::{Duration, Instant},
    };

    struct EngineChild(Child);
    impl Drop for EngineChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn run_jopedime(home: &str, args: &[&str]) -> std::io::Result<Output> {
        Command::new(env!("CARGO_BIN_EXE_jopedime"))
            .args(args)
            .env("JOPEDIME_HOME", home)
            .output()
    }

    fn assert_output_clean(output: &Output) {
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_no_internal_sentinels(&text);
    }

    let root = tempfile::tempdir()?;
    let home = root.path().join("engine-home");
    fs::create_dir_all(&home)?;
    let csv = root.path().join("safe-name.csv");
    fs::write(
        &csv,
        format!(
            "feature,target\nalpha,{}\nbeta,{}\ngamma,{}\ndelta,{}\n",
            SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
        ),
    )?;
    let home_text = home.to_str().expect("temporary path is UTF-8");

    let child = Command::new(env!("CARGO_BIN_EXE_jopedime"))
        .args(["serve", "--idle-exit", "60s"])
        .env("JOPEDIME_HOME", &home)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let mut engine = EngineChild(child);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if engine.0.try_wait()?.is_some() {
            return Err(std::io::Error::other(
                "isolated engine exited during startup",
            ));
        }
        let status = run_jopedime(home_text, &["serve", "--status"])?;
        let status_text = String::from_utf8_lossy(&status.stdout);
        if status.status.success() && status_text.starts_with("running ") {
            assert_output_clean(&status);
            break;
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "isolated engine did not become ready: {}{}",
                    status_text,
                    String::from_utf8_lossy(&status.stderr)
                ),
            ));
        }
        thread::sleep(Duration::from_millis(20));
    }

    let add = run_jopedime(
        home_text,
        &[
            "dataset",
            "add",
            csv.to_str().unwrap(),
            "--name",
            "safe-name",
            "--idempotency-key",
            "pub06-sentinel-cli",
            "--json",
        ],
    )?;
    assert!(
        add.status.success(),
        "dataset register failed: {}",
        String::from_utf8_lossy(&add.stderr)
    );
    assert_output_clean(&add);
    let profile: serde_json::Value = serde_json::from_slice(&add.stdout).unwrap();
    let id = profile["dataset_id"]
        .as_str()
        .expect("dataset id in response");

    for args in [
        vec!["dataset", "list", "--json"],
        vec!["dataset", "show", id, "--json"],
    ] {
        let output = run_jopedime(home_text, &args)?;
        assert!(
            !output.status.success(),
            "unimplemented dataset read unexpectedly succeeded"
        );
        assert_output_clean(&output);
    }

    let mut terminal =
        pty::PtySession::spawn_jopedime_with_env(&[], &[("JOPEDIME_HOME", home_text)])?;
    let screen = terminal.wait_for_text("engine background", Duration::from_secs(10))?;
    assert_no_internal_sentinels(&screen);
    terminal.type_text(&format!("\\open {}", csv.display()))?;
    terminal.press_key(pty::Key::Enter)?;
    terminal.type_text("\\help")?;
    terminal.press_key(pty::Key::Enter)?;
    let screen = terminal.wait_for_text("\\open PATH", Duration::from_secs(10))?;
    assert_no_internal_sentinels(&screen);
    drop(terminal);

    let stopped = run_jopedime(home_text, &["serve", "--stop"])?;
    assert!(stopped.status.success(), "isolated engine stop failed");
    assert_output_clean(&stopped);
    Ok(())
}

// Windows ConPTY coverage is tracked separately from this Linux/macOS smoke test.
