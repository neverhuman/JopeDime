#[path = "support/pty.rs"]
mod pty;

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
    Ok(())
}

// Windows ConPTY coverage is tracked separately from this Linux/macOS smoke test.
