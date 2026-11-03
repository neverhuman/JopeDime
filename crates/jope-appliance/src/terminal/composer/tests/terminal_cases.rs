use super::*;

#[cfg(target_os = "linux")]
#[test]
#[ignore = "supervised real-PTY child; explicitly executed by the U1 parent"]
fn pty_composer_probe() {
    use super::super::super::restore::TerminalGuard;
    use ratatui::crossterm::event;
    use ratatui::widgets::Widget;
    use ratatui::{backend::CrosstermBackend, Terminal, TerminalOptions, Viewport};
    let history = std::env::var_os("JOPE_COMPOSER_HISTORY").unwrap();
    let report = PathBuf::from(std::env::var_os("JOPE_COMPOSER_REPORT").unwrap());
    let mut composer = Composer::with_history(Path::new(&history)).unwrap();
    println!("COMPOSER_START");
    io::stdout().flush().unwrap();
    let mut go = String::new();
    io::stdin().read_line(&mut go).unwrap();
    assert_eq!(go.trim(), "go");
    let guard = TerminalGuard::acquire(true).unwrap();
    println!("COMPOSER_ENTER");
    io::stdout().flush().unwrap();
    let mut terminal = Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(8),
        },
    )
    .unwrap();
    let (send, receive) = std::sync::mpsc::sync_channel(256);
    let started = Instant::now();
    let producer = std::thread::spawn(move || {
        let start = Instant::now();
        for i in 0..150_u64 {
            let deadline = start + Duration::from_millis(i * 20);
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
            if send.send(i).is_err() {
                break;
            }
        }
    });
    let mut events = 0;
    let mut submissions = 0;
    let mut snapshots = Vec::new();
    let mut clock = super::super::super::app::RedrawClock::new();
    loop {
        while let Ok(sequence) = receive.try_recv() {
            events += 1;
            terminal
                .insert_before(1, |buffer| {
                    Paragraph::new(format!("EVENT {sequence}")).render(buffer.area, buffer)
                })
                .unwrap();
        }
        if event::poll(Duration::from_millis(5)).unwrap() {
            let input = event::read().unwrap();
            if matches!(&input,Event::Key(k) if k.code==KeyCode::F(12)) {
                let view = composer.view(80, 7);
                let menu=composer.completion().map(|(s,i)|serde_json::json!({"selected":i,"choices":s.iter().map(|s|&s.insert).collect::<Vec<_>>()}));
                snapshots.push(serde_json::json!({"text":composer.text(),"cursor":composer.cursor_byte(),"view_lines":view.lines,"view_cursor":view.cursor,"menu":menu,"events":events,"submissions":submissions,"elapsed_ms":started.elapsed().as_millis()}));
                fs::write(&report, serde_json::to_vec(&snapshots).unwrap()).unwrap();
            } else {
                match composer.handle(input, &CompletionData::default()).unwrap() {
                    ComposerAction::ExitRequested => break,
                    ComposerAction::Submit(_) => submissions += 1,
                    _ => {}
                }
            }
        }
        if clock.remaining(Instant::now(), true).is_zero() {
            terminal
                .draw(|frame| {
                    let area = frame.area();
                    frame.render_widget(
                        Paragraph::new(format!("COMPOSER_READY events={events}")),
                        Rect::new(area.x, area.y, area.width, 1),
                    );
                    composer.draw(
                        frame,
                        Rect::new(
                            area.x,
                            area.y + 1,
                            area.width,
                            area.height.saturating_sub(1),
                        ),
                    );
                })
                .unwrap();
            clock.drawn(Instant::now());
        }
    }
    drop(receive);
    producer.join().unwrap();
    drop(terminal);
    drop(guard);
    println!("COMPOSER_RESTORED");
    io::stdout().flush().unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn linux_pty_typing_paste_cursor_and_menu_survive_fifty_hz_events() {
    let directory = tempfile::tempdir().unwrap();
    let history = directory.path().join("history");
    let report = directory.path().join("report.json");
    let executable = std::env::current_exe().unwrap();
    let h = format!("JOPE_COMPOSER_HISTORY={}", history.display());
    let r = format!("JOPE_COMPOSER_REPORT={}", report.display());
    let mut session = pty::PtySession::spawn(
        "/usr/bin/env",
        &[
            &h,
            &r,
            "RUST_BACKTRACE=0",
            executable.to_str().unwrap(),
            "--exact",
            "terminal::composer::tests::terminal_cases::pty_composer_probe",
            "--ignored",
            "--nocapture",
            "--color=never",
            "--test-threads=1",
        ],
    )
    .unwrap();
    session
        .wait_for_text("COMPOSER_START", Duration::from_secs(10))
        .unwrap();
    session.type_text("go\n").unwrap();
    session
        .wait_for_text("COMPOSER_ENTER", Duration::from_secs(10))
        .unwrap();
    session.type_text("\x1b[1;1R").unwrap();
    session
        .wait_for_text("COMPOSER_READY", Duration::from_secs(10))
        .unwrap();
    let snapshot = |session: &mut pty::PtySession, count: usize| {
        session.type_text("\x1b[24~").unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(bytes) = fs::read(&report) {
                if let Ok(rows) = serde_json::from_slice::<Vec<serde_json::Value>>(&bytes) {
                    if rows.len() >= count {
                        return rows[count - 1].clone();
                    }
                }
            }
            assert!(
                Instant::now() < deadline,
                "missing PTY snapshot: {}",
                session.read_screen_as_text()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    };
    session.type_text("e\u{301}中").unwrap();
    session.type_text("\x1b[200~pasted\ntext\x1b[201~").unwrap();
    session.press_key(pty::Key::Left).unwrap();
    let typed = snapshot(&mut session, 1);
    assert_eq!(typed["text"], "e\u{301}中pasted\ntext");
    assert_eq!(
        typed["cursor"].as_u64().unwrap() as usize,
        "e\u{301}中pasted\ntex".len()
    );
    std::thread::sleep(Duration::from_millis(440));
    let after = snapshot(&mut session, 2);
    for field in ["text", "cursor", "view_lines", "view_cursor", "menu"] {
        assert_eq!(typed[field], after[field], "{field}");
    }
    assert!(after["events"].as_u64().unwrap() >= typed["events"].as_u64().unwrap() + 18);
    // Clear both draft lines with editing keys; Ctrl-C now exits.
    session.type_text("\x05\x15\x7f\x15\\o\t").unwrap();
    let menu = snapshot(&mut session, 3);
    assert!(!menu["menu"].is_null());
    std::thread::sleep(Duration::from_millis(440));
    let menu_after = snapshot(&mut session, 4);
    for field in ["text", "cursor", "view_lines", "view_cursor", "menu"] {
        assert_eq!(menu[field], menu_after[field], "{field}");
    }
    assert!(menu_after["events"].as_u64().unwrap() >= menu["events"].as_u64().unwrap() + 18);
    session.type_text("\x03").unwrap();
    let _ = session
        .wait_for_text("COMPOSER_RESTORED", Duration::from_secs(10))
        .unwrap();
    let screen = session
        .wait_for_text("1 passed; 0 failed", Duration::from_secs(10))
        .unwrap();
    for point in [&typed, &after, &menu, &menu_after] {
        assert_eq!(point["submissions"], 0);
    }
    if let Some(path) = std::env::var_os("JOPE_COMPOSER_EVIDENCE_DIR") {
        fs::write(
            Path::new(&path).join("u1-snapshots.json"),
            fs::read(&report).unwrap(),
        )
        .unwrap();
        fs::write(Path::new(&path).join("u1-screen.txt"), screen).unwrap();
    }
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        fs::metadata(&history).unwrap().permissions().mode() & 0o777,
        0o600
    );
}
