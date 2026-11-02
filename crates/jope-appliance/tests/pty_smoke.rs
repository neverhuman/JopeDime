#[path = "support/pty.rs"]
mod pty;

const SENTINELS: [&str; 4] = ["918273.125", "817263.25", "716254.375", "615245.5"];

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
            "feature,target\n1,{}\n2,{}\n3,{}\n4,{}\n",
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

    let listed = run_jopedime(home_text, &["dataset", "list", "--json"])?;
    assert!(
        listed.status.success(),
        "dataset list failed: {}",
        String::from_utf8_lossy(&listed.stderr)
    );
    assert_output_clean(&listed);
    let listing: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert!(
        listing["datasets"]
            .as_array()
            .is_some_and(|datasets| datasets.iter().any(|profile| profile["dataset_id"] == id)),
        "dataset list omitted the registered profile: {listing}"
    );

    let shown = run_jopedime(home_text, &["dataset", "show", id, "--json"])?;
    assert!(
        shown.status.success(),
        "dataset show failed: {}",
        String::from_utf8_lossy(&shown.stderr)
    );
    assert_output_clean(&shown);
    let shown_profile: serde_json::Value = serde_json::from_slice(&shown.stdout).unwrap();
    assert_eq!(shown_profile["dataset_id"], id);

    let mut terminal = pty::PtySession::spawn_jopedime_with_env(
        &[],
        &[
            ("JOPEDIME_HOME", home_text),
            ("NO_COLOR", "1"),
            ("TERM", "dumb"),
        ],
    )?;
    let screen = terminal.wait_for_text("engine background", Duration::from_secs(10))?;
    assert_no_internal_sentinels(&screen);
    // The local run backend validates a bounded regression task with no landmarks
    // when a dataset is opened. Configure that draft before registration.
    terminal.type_text("\\landmarks none")?;
    terminal.press_key(pty::Key::Enter)?;
    terminal.type_text("\\lang rust")?;
    terminal.press_key(pty::Key::Enter)?;
    terminal.type_text("\\target target --task regression")?;
    terminal.press_key(pty::Key::Enter)?;
    // This unique plan response is a barrier proving both earlier draft commands
    // were processed before the dataset registration starts.
    terminal.type_text("\\plan")?;
    terminal.press_key(pty::Key::Enter)?;
    let screen = terminal.wait_for_text(
        "draft needs a registered dataset and target before it can be validated",
        Duration::from_secs(10),
    )?;
    assert_no_internal_sentinels(&screen);
    terminal.type_text(&format!("\\open {}", csv.display()))?;
    terminal.press_key(pty::Key::Enter)?;
    // Wait for this command's unique completion marker before sending the next line. PTY
    // input and output are asynchronous; back-to-back commands can otherwise race the reader.
    let screen = terminal.wait_for_text("registered safe-name.csv as", Duration::from_secs(10))?;
    assert_no_internal_sentinels(&screen);
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

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    feature = "client",
    feature = "engine"
))]
mod interactive_exit {
    use super::pty;

    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };

    struct Engine(Child);
    impl Drop for Engine {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    fn interactive_exit_returns_to_shell_with_terminal_settings_restored() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let home = root.path().join("home");
        fs::create_dir(&home)?;
        fs::set_permissions(&home, fs::Permissions::from_mode(0o700))?;
        let binary = env!("CARGO_BIN_EXE_jopedime");
        let mut engine = Engine(
            Command::new(binary)
                .args(["serve", "--idle-exit", "60s"])
                .env("JOPEDIME_HOME", &home)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?,
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            assert!(engine.0.try_wait()?.is_none(), "isolated engine exited");
            let status = Command::new(binary)
                .args(["serve", "--status"])
                .env("JOPEDIME_HOME", &home)
                .output()?;
            if status.status.success() && status.stdout.starts_with(b"running ") {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "isolated engine never became ready"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        for keys in [
            "\u{3}",
            "unsent draft\u{3}",
            "\u{12}\u{3}",
            "\u{4}",
            "\\quit\r",
        ] {
            // Compare the shell's actual termios before and after the CLI; a reset
            // escape alone does not prove canonical input and echo were restored.
            let script = r#"saved=$(stty -g) || exit 91
    "$JAIN_EXIT_BINARY"
    result=$?
    if [ "$result" = 0 ] && [ "$(stty -g)" = "$saved" ]; then
        echo JAIN_EXIT_RESTORED
    else
        echo JAIN_EXIT_BROKEN
        exit 92
    fi"#;
            let mut terminal = pty::PtySession::spawn_with_env(
                "/usr/bin/env",
                &["-u", "NO_COLOR", "sh", "-c", script],
                &[
                    ("JOPEDIME_HOME", home.to_str().unwrap()),
                    ("JAIN_EXIT_BINARY", binary),
                ],
            )?;
            terminal.wait_for_output_bytes(b"\x1b[?1049h", Duration::from_secs(15))?;
            terminal.wait_for_text("Powered by JopeDime", Duration::from_secs(10))?;
            terminal.type_text(keys)?;
            terminal.wait_for_text("JAIN_EXIT_RESTORED", Duration::from_secs(5))?;
            terminal.wait_for_output_bytes(b"\x1b[?1049l", Duration::from_secs(5))?;
            terminal.wait_for_output_bytes(b"\x1b[?1000l", Duration::from_secs(5))?;
            assert!(
                engine.0.try_wait()?.is_none(),
                "leaving killed the shared engine"
            );
        }
        Ok(())
    }
}
