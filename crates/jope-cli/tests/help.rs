//! `jope --help` snapshot: the verbs the umbrella exposes, and that each verb
//! really reaches the crate it delegates to.

use std::process::Command;

fn jope(args: &[&str]) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_jope"))
        .args(args)
        .output()
        .expect("run jope");
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    (output.status.success(), text)
}

#[test]
fn help_lists_the_verbs() {
    let (ok, text) = jope(&["--help"]);
    assert!(ok, "{text}");
    let expected = include_str!("fixtures/help.txt");
    assert_eq!(
        text.trim_end(),
        expected.trim_end(),
        "jope --help changed; update tests/fixtures/help.txt if that is intended"
    );
    for verb in ["engine", "selfplay", "campaign"] {
        assert!(text.contains(&format!("\n  {verb} ")), "verb {verb} missing from --help");
    }
}

#[test]
fn engine_verb_reaches_the_engine_command_line() {
    let (ok, text) = jope(&["engine", "--help"]);
    assert!(ok, "{text}");
    assert!(text.contains("Usage: jope engine"), "{text}");
    assert!(text.contains("--root"), "{text}");
    for subcommand in ["search", "gp-service", "doctor", "snapshot", "audit"] {
        assert!(text.contains(&format!("\n  {subcommand}")), "{subcommand} missing: {text}");
    }
    // The consumer argv shape (jain-core) parses through the umbrella too.
    let (ok, text) = jope(&["engine", "--root", "/nonexistent", "search", "--help"]);
    assert!(ok, "{text}");
    assert!(text.contains("--qualification") && text.contains("--vocabulary"), "{text}");
}

#[test]
fn program_verbs_reach_the_program_tables() {
    let (ok, text) = jope(&["selfplay", "--help"]);
    assert!(ok, "{text}");
    assert!(text.starts_with("jope-selfplay — "), "{text}");
    for name in jope_selfplay::TABLE.names() {
        assert!(text.contains(&format!("  {name} ")), "{name} missing: {text}");
    }
    let (ok, text) = jope(&["campaign", "--help"]);
    assert!(ok, "{text}");
    assert!(text.starts_with("jope-campaigns — "), "{text}");
    for name in jope_campaigns::TABLE.names() {
        assert!(text.contains(&format!("  {name} ")), "{name} missing: {text}");
    }
    let (ok, text) = jope(&["campaign", "no_such_program"]);
    assert!(!ok);
    assert!(text.contains("unknown program `no_such_program`"), "{text}");
}
