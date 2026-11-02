use super::*;

#[test]
fn linux_u2_restores_normal_error_panic_and_three_real_signals() {
    for (scenario, command) in [
        ("normal", "replay"),
        ("error", "error"),
        ("transport-error", "transport-error"),
        ("panic", "panic"),
        ("int", "int"),
        ("term", "term"),
        ("hup", "hup"),
    ] {
        let mut session = spawn_probe(scenario, true);
        session.type_text(&format!("{command}\r")).unwrap();
        let screen = session
            .wait_for_text(
                "PROBE_RESTORED raw=false termios=true",
                Duration::from_secs(10),
            )
            .unwrap();
        if scenario == "normal" {
            assert!(
                screen.contains("First model"),
                "recorded fixture did not reach transcript: {screen}"
            );
            assert!(
                screen.contains("Run finished"),
                "recorded fixture did not finish: {screen}"
            );
        }
        retain_screen(scenario, &screen);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn linux_fallbacks_never_change_raw_mode_and_resize_preserves_draft() {
    for scenario in ["plain", "no-color", "dumb", "narrow"] {
        let mut session = spawn_probe(scenario, false);
        session.type_text("\\quit\r").unwrap();
        let screen = session
            .wait_for_text(
                "PROBE_RESTORED raw=false termios=true",
                Duration::from_secs(10),
            )
            .unwrap();
        retain_screen(scenario, &screen);
    }
    let mut session = spawn_probe("resize", true);
    session.type_text("draft").unwrap();
    session
        .wait_for_text("draft", Duration::from_secs(10))
        .unwrap();
    session.resize(39, 24).unwrap();
    // Send Enter immediately after resize so the newline can be queued behind SIGWINCH.
    session.press_key(pty::Key::Enter).unwrap();
    session
        .wait_for_text("COMMAND draft", Duration::from_secs(10))
        .unwrap();
    session.type_text("\\quit\r").unwrap();
    let screen = session
        .wait_for_text(
            "PROBE_RESTORED raw=false termios=true",
            Duration::from_secs(10),
        )
        .unwrap();
    retain_screen("resize", &screen);
}

#[cfg(target_os = "linux")]
#[test]
fn linux_queued_resize_and_enter_before_next_poll_submits() {
    let mut session = spawn_probe("resize", true);
    session.type_text("arm\r").unwrap();
    let output = session
        .wait_for_text("PROBE_BEFORE_POLL", Duration::from_secs(10))
        .unwrap();
    let pid = output
        .split_whitespace()
        .skip_while(|part| *part != "PROBE_BEFORE_POLL")
        .nth(1)
        .unwrap_or_else(|| panic!("stopped-child marker has no PID: {output:?}"))
        .parse::<u32>()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
        if status
            .lines()
            .any(|line| line.starts_with("State:") && line.contains("T (stopped)"))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "PTY app did not stop before poll"
        );
        std::thread::yield_now();
    }

    // The child queued SIGWINCH before stopping. Queue the matching size change and CR while
    // it cannot poll either source, then resume into one simultaneous readiness batch.
    session.resize(39, 24).unwrap();
    session.press_key(pty::Key::Enter).unwrap();
    assert!(std::process::Command::new("/bin/kill")
        .args(["-CONT", &pid.to_string()])
        .status()
        .unwrap()
        .success());

    session
        .wait_for_text("PROBE_FIRST_EVENT=resize", Duration::from_secs(10))
        .unwrap();
    session
        .wait_for_text("COMMAND_EMPTY", Duration::from_secs(10))
        .unwrap();
    session.type_text("\\quit\r").unwrap();
    session
        .wait_for_text(
            "PROBE_RESTORED raw=false termios=true",
            Duration::from_secs(10),
        )
        .unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn linux_redraw_downgrade_drains_buffered_key_burst_and_enter() {
    let payload = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut session = spawn_probe("resize", true);
    session.type_text(&format!("arm\r{payload}")).unwrap();
    let output = session
        .wait_for_text("PROBE_BEFORE_POLL", Duration::from_secs(10))
        .unwrap();
    let pid = output
        .split_whitespace()
        .skip_while(|part| *part != "PROBE_BEFORE_POLL")
        .nth(1)
        .unwrap_or_else(|| panic!("stopped-child marker has no PID: {output:?}"))
        .parse::<u32>()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
        if status
            .lines()
            .any(|line| line.starts_with("State:") && line.contains("T (stopped)"))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "PTY app did not stop before poll"
        );
        std::thread::yield_now();
    }

    // Resize and Enter while the same child is stopped. Let its redraw deadline expire so
    // reconcile_width downgrades from the redraw path while parsed characters remain queued.
    session.resize(39, 24).unwrap();
    session.press_key(pty::Key::Enter).unwrap();
    std::thread::sleep(Duration::from_millis(250));
    assert!(std::process::Command::new("/bin/kill")
        .args(["-CONT", &pid.to_string()])
        .status()
        .unwrap()
        .success());

    session
        .wait_for_text("PROBE_FIRST_EVENT=key", Duration::from_secs(10))
        .unwrap();
    session
        .wait_for_text(&format!("COMMAND {payload}"), Duration::from_secs(10))
        .unwrap();
    session.type_text("\\quit\r").unwrap();
    session
        .wait_for_text(
            "PROBE_RESTORED raw=false termios=true",
            Duration::from_secs(10),
        )
        .unwrap();
}

#[test]
fn injected_resize_then_enter_queue_drains_after_plain_downgrade() {
    use std::collections::VecDeque;

    // The old Plain transition stopped reading the Crossterm queue here. Model that
    // one-event turn as the red control: Resize downgrades, leaving Enter stranded.
    let mut legacy_queue = VecDeque::from([
        Event::Resize(39, 24),
        Event::Key(event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
    ]);
    let mut legacy_plain = false;
    let legacy_commands: Vec<&str> = Vec::new();
    if let Some(Event::Resize(columns, _)) = legacy_queue.pop_front() {
        legacy_plain = columns < 40;
    }
    assert!(legacy_plain);
    assert!(
        legacy_commands.is_empty(),
        "legacy Plain turn loses queued Enter"
    );
    assert_eq!(legacy_queue.len(), 1, "the queued Enter remains unhandled");

    let mut queue = VecDeque::from([
        Event::Resize(39, 24),
        Event::Key(event::KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
    ]);
    let mut plain = false;
    let mut commands = Vec::new();

    let exit = drain_event_queue(
        || Ok(queue.pop_front()),
        |event| {
            match event {
                Event::Resize(columns, _) if columns < 40 => plain = true,
                Event::Key(key) if plain && key.code == KeyCode::Enter => {
                    commands.push("submit");
                }
                _ => {}
            }
            Ok(None)
        },
    )
    .unwrap();

    assert!(plain, "Resize must trigger the Plain downgrade");
    assert_eq!(commands, ["submit"], "queued Enter must survive downgrade");
    assert!(queue.is_empty(), "the event queue must be fully drained");
    assert!(exit.is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn no_color_replays_all_five_real_recorded_fixtures_without_escape_bytes() {
    for (name, _) in FIXTURES {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "terminal::app::tests::pty_probe",
                "--ignored",
                "--nocapture",
                "--color=never",
                "--test-threads=1",
            ])
            .env("JOPE_TUI_SCENARIO", format!("pipe-{name}"))
            .env("NO_COLOR", "1")
            .env("TERM", "xterm-256color")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output.stdout.contains(&0x1b),
            "NO_COLOR output contains escapes"
        );
        let screen = String::from_utf8(output.stdout).unwrap();
        assert!(screen.contains("JopeDime"));
        if *name == "success" {
            assert!(screen.contains("First model"));
            assert!(screen.contains("Run finished"));
        }
        retain_screen(&format!("no-color-{name}"), &screen);
    }
}
