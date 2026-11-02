use super::*;

#[test]
fn redraw_caps_cover_bursts_idle_and_activity_transitions() {
    let now = Instant::now();
    let mut clock = RedrawClock::new();
    assert!(clock.remaining(now, true).is_zero());
    clock.drawn(now);
    assert_eq!(clock.remaining(now, true), Duration::from_millis(50));
    assert_eq!(clock.remaining(now, false), Duration::from_secs(1));
    assert_eq!(
        clock.remaining(now + Duration::from_millis(49), true),
        Duration::from_millis(1)
    );
    assert!(clock
        .remaining(now + Duration::from_millis(50), true)
        .is_zero());
    let delayed = now + Duration::from_secs(5);
    clock.drawn(delayed);
    assert_eq!(clock.remaining(delayed, true), Duration::from_millis(50));
    assert_eq!(clock.remaining(delayed, false), Duration::from_secs(1));
}

#[test]
fn wrapping_preserves_wide_combining_and_long_transcript_text() {
    for text in ["hello this is a very long transcript", "界界e\u{301}界界"] {
        let lines = wrap_line(text, 4);
        assert_eq!(lines.concat(), text);
        assert!(lines
            .iter()
            .all(|line| Line::from(line.as_str()).width() <= 4));
    }
}

#[cfg(target_os = "linux")]
const FIXTURES: &[(&str, &str)] = &[
    (
        "success",
        include_str!("../../../tests/fixtures/events/success.jsonl"),
    ),
    (
        "warnings",
        include_str!("../../../tests/fixtures/events/warnings.jsonl"),
    ),
    (
        "failed",
        include_str!("../../../tests/fixtures/events/failed-candidate-burst.jsonl"),
    ),
    (
        "paused",
        include_str!("../../../tests/fixtures/events/paused-continued.jsonl"),
    ),
    (
        "cancelled",
        include_str!("../../../tests/fixtures/events/cancelled.jsonl"),
    ),
];

#[cfg(target_os = "linux")]
fn replay(sender: SyncSender<AppEvent>, fixture: &str) -> std::thread::JoinHandle<()> {
    let replay = jope_client::fake::FakeTransport::from_jsonl(fixture, Duration::from_millis(25))
        .expect("recorded TUI-11 fixture");
    std::thread::spawn(move || {
        for event in replay {
            std::thread::sleep(event.delay_before);
            if sender
                .send(AppEvent::Public(Box::new(event.event)))
                .is_err()
            {
                return;
            }
        }
        let _ = sender.send(AppEvent::TransportClosed);
    })
}

#[cfg(target_os = "linux")]
fn termios() -> String {
    let output = std::process::Command::new("stty")
        .arg("-g")
        .stdin(std::process::Stdio::inherit())
        .output()
        .expect("Linux stty");
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

/// One supervised helper, executed in separate real PTYs by the acceptance tests below.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "PTY child helper; executed explicitly by the Linux U2 parent tests"]
fn pty_probe() {
    use std::io::Write;
    let scenario = std::env::var("JOPE_TUI_SCENARIO").expect("supervised PTY scenario");
    let (sender, receiver) = channel();
    if let Some(fixture) = scenario.strip_prefix("pipe-") {
        let jsonl = FIXTURES
            .iter()
            .find(|(name, _)| *name == fixture)
            .unwrap()
            .1;
        let producer = replay(sender, jsonl);
        let result = run(
            Options {
                exit_on_stream_end: true,
                ..Options::default()
            },
            receiver,
            |_| Ok(()),
        )
        .unwrap();
        producer.join().unwrap();
        assert_eq!(result, Exit::StreamEnded);
        return;
    }
    println!("PROBE_START");
    io::stdout().flush().unwrap();
    let mut barrier = String::new();
    io::stdin().read_line(&mut barrier).unwrap();
    assert_eq!(barrier.trim(), "go");
    let before = termios();
    println!("PROBE_ENTER");
    io::stdout().flush().unwrap();
    let mut producers = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(
            Options {
                plain: scenario == "plain",
                exit_on_stream_end: scenario == "normal",
                ..Options::default()
            },
            receiver,
            |command| {
                match command {
                    "arm" => {
                        STOP_BEFORE_EVENT_POLL.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                    "check" => {
                        let raw = ratatui::crossterm::terminal::is_raw_mode_enabled()?;
                        let changed = termios() != before;
                        sender
                            .try_send(AppEvent::Transcript(vec![format!(
                                "PROBE_RAW raw={raw} changed={changed}"
                            )]))
                            .unwrap();
                    }
                    "replay" => producers.push(replay(sender.clone(), FIXTURES[0].1)),
                    "error" => return Err(io::Error::other("scripted command failure")),
                    "transport-error" => sender
                        .try_send(AppEvent::TransportError(
                            "scripted transport failure".into(),
                        ))
                        .unwrap(),
                    "panic" => panic!("scripted terminal panic"),
                    "int" => signal_hook::low_level::raise(signal_hook::consts::SIGINT)?,
                    "term" => signal_hook::low_level::raise(signal_hook::consts::SIGTERM)?,
                    "hup" => signal_hook::low_level::raise(signal_hook::consts::SIGHUP)?,
                    "" => sender
                        .try_send(AppEvent::Transcript(vec!["COMMAND_EMPTY".into()]))
                        .unwrap(),
                    other => sender
                        .try_send(AppEvent::Transcript(vec![format!("COMMAND {other}")]))
                        .unwrap(),
                }
                Ok(())
            },
        )
    }));
    for producer in producers {
        producer.join().unwrap();
    }
    match scenario.as_str() {
        "panic" => assert!(result.is_err()),
        "error" | "transport-error" => assert!(result.unwrap().is_err()),
        "int" => assert_eq!(
            result.unwrap().unwrap(),
            Exit::Signal(ExitSignal::Interrupt)
        ),
        "term" => assert_eq!(
            result.unwrap().unwrap(),
            Exit::Signal(ExitSignal::Terminate)
        ),
        "hup" => assert_eq!(result.unwrap().unwrap(), Exit::Signal(ExitSignal::Hangup)),
        "normal" => assert_eq!(result.unwrap().unwrap(), Exit::StreamEnded),
        _ => assert_eq!(result.unwrap().unwrap(), Exit::Quit),
    }
    let raw = ratatui::crossterm::terminal::is_raw_mode_enabled().unwrap();
    let restored = termios() == before;
    println!("PROBE_RESTORED raw={raw} termios={restored}");
    assert!(!raw);
    assert!(
        restored,
        "exact pre-session Linux terminal flags must be restored"
    );
}

#[cfg(target_os = "linux")]
fn spawn_probe(scenario: &str, inline: bool) -> pty::PtySession {
    let executable = std::env::current_exe().unwrap();
    let scenario_env = format!("JOPE_TUI_SCENARIO={scenario}");
    let color = if scenario == "no-color" {
        "NO_COLOR=1"
    } else {
        "COLORTERM=truecolor"
    };
    let term = if scenario == "dumb" {
        "TERM=dumb"
    } else {
        "TERM=xterm-256color"
    };
    let args = [
        "-u",
        "NO_COLOR",
        &scenario_env,
        color,
        term,
        "RUST_BACKTRACE=0",
        executable.to_str().unwrap(),
        "--exact",
        "terminal::app::tests::pty_probe",
        "--ignored",
        "--nocapture",
        "--color=never",
        "--test-threads=1",
    ];
    let mut session = pty::PtySession::spawn("/usr/bin/env", &args).unwrap();
    session
        .wait_for_text("PROBE_START", Duration::from_secs(10))
        .unwrap();
    if scenario == "narrow" {
        session.resize(39, 24).unwrap();
    }
    session.type_text("go\n").unwrap();
    session
        .wait_for_text("PROBE_ENTER", Duration::from_secs(10))
        .unwrap();
    if inline {
        // TUI-10 supplies a raw PTY, not a terminal emulator: answer the initial CPR query.
        session.type_text("\x1b[1;1R").unwrap();
    }
    session
        .wait_for_text("JopeDime", Duration::from_secs(10))
        .unwrap();
    session.type_text("check\r").unwrap();
    let marker = if inline {
        "PROBE_RAW raw=true changed=true"
    } else {
        "PROBE_RAW raw=false changed=false"
    };
    session
        .wait_for_text(marker, Duration::from_secs(10))
        .unwrap();
    session
}

#[cfg(target_os = "linux")]
fn retain_screen(scenario: &str, screen: &str) {
    if let Some(directory) = std::env::var_os("JOPE_TUI_EVIDENCE_DIR") {
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{scenario}.txt")),
            screen,
        )
        .unwrap();
    }
}

#[cfg(target_os = "linux")]
#[path = "tests/linux.rs"]
mod linux;
