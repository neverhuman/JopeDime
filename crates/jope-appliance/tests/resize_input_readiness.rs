#![cfg(target_os = "linux")]

#[path = "support/pty.rs"]
mod pty;

use ratatui::crossterm::{
    event::{self, Event, KeyCode},
    terminal,
};
use std::{
    io::Write,
    process::Command,
    time::{Duration, Instant},
};

#[test]
fn queued_resize_does_not_drop_pending_enter() {
    assert_queued_readiness("signal-first");
}

#[test]
fn queued_enter_does_not_drop_pending_resize() {
    assert_queued_readiness("key-first");
}

fn assert_queued_readiness(order: &str) {
    let binary = std::env::current_exe().unwrap();
    let mut session = pty::PtySession::spawn_with_env(
        binary,
        &[
            "--exact",
            "readiness_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        &[("JOPE_READINESS_ORDER", order)],
    )
    .unwrap();
    let output = session
        .wait_for_text("READINESS_READY", Duration::from_secs(10))
        .unwrap();
    let pid = output
        .split_whitespace()
        .skip_while(|s| *s != "READINESS_READY")
        .nth(1)
        .unwrap()
        .parse::<u32>()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let state = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
        if state
            .lines()
            .any(|line| line.starts_with("State:") && line.contains("T (stopped)"))
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "owned readiness child did not stop"
        );
        std::thread::yield_now();
    }
    // Queue both readiness tokens while the owned child cannot consume either.
    if order == "signal-first" {
        session.resize(39, 24).unwrap();
        session.press_key(pty::Key::Enter).unwrap();
    } else {
        session.press_key(pty::Key::Enter).unwrap();
        session.resize(39, 24).unwrap();
    }
    assert!(Command::new("/bin/kill")
        .args(["-CONT", &pid.to_string()])
        .status()
        .unwrap()
        .success());
    session
        .wait_for_text("READINESS_COMPLETE", Duration::from_secs(15))
        .unwrap();
}

#[test]
#[ignore = "supervised real-PTY child, executed by the queued-readiness parent tests"]
fn readiness_child() {
    terminal::enable_raw_mode().unwrap();
    let _ = event::poll(Duration::ZERO).unwrap();
    let pid = std::process::id().to_string();
    if std::env::var("JOPE_READINESS_ORDER").unwrap() == "signal-first" {
        assert!(Command::new("/bin/kill")
            .args(["-WINCH", &pid])
            .status()
            .unwrap()
            .success());
    }
    println!("READINESS_READY {pid}");
    std::io::stdout().flush().unwrap();
    assert!(Command::new("/bin/kill")
        .args(["-STOP", &pid])
        .status()
        .unwrap()
        .success());
    let (mut resize, mut enter) = (false, false);
    for index in 0..2 {
        assert!(
            event::poll(Duration::from_secs(10)).unwrap(),
            "prequeued readiness was lost"
        );
        let event = event::read().unwrap();
        if index == 0 {
            let signal_first = std::env::var("JOPE_READINESS_ORDER").unwrap() == "signal-first";
            assert_eq!(matches!(event, Event::Resize(..)), signal_first);
        }
        match event {
            Event::Resize(39, 24) => resize = true,
            Event::Key(key) if key.code == KeyCode::Enter => enter = true,
            other => panic!("unexpected queued event: {other:?}"),
        }
    }
    terminal::disable_raw_mode().unwrap();
    assert!(
        resize && enter,
        "both already-ready events must survive one poll batch"
    );
    println!("READINESS_COMPLETE");
}
