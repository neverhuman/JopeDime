//! The `jope-campaigns` binary: its program list and the two dispatch forms.

use std::path::Path;
use std::process::Command;

fn run(program: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::new(program).args(args).output().expect("run");
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    (output.status.success(), text)
}

fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_jope-campaigns"))
}

#[test]
fn help_lists_the_twenty_five_programs_and_the_former_root_binary() {
    let (ok, text) = run(binary(), &["--help"]);
    assert!(ok, "{text}");
    assert_eq!(jope_campaigns::TABLE.programs.len(), 26, "25 src/bin programs + jope-dime-regressor");
    assert_eq!(jope_campaigns::TABLE.programs[0].name, "jope-dime-regressor");
    assert!(text.contains("\n26 programs:\n"), "{text}");
    let expected = include_str!("fixtures/programs.txt");
    let (ok, list) = run(binary(), &["--list"]);
    assert!(ok, "{list}");
    assert_eq!(list, expected, "the program list changed; update tests/fixtures/programs.txt if intended");
    for name in jope_campaigns::TABLE.names() {
        assert!(text.contains(&format!("  {name} ")), "{name} missing from --help");
    }
}

#[test]
fn programs_run_by_first_argument_and_through_a_link() {
    // The former root binary keeps its clap name and subcommands.
    let (ok, text) = run(binary(), &["jope-dime-regressor", "--help"]);
    assert!(ok, "{text}");
    assert!(text.contains("Usage: jope-dime-regressor"), "{text}");
    assert!(text.contains("run-panel") && text.contains("validate-ratchet-artifact"), "{text}");
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("jope-dime-regressor");
    std::os::unix::fs::symlink(binary(), &link).unwrap();
    let (ok, linked) = run(&link, &["--help"]);
    assert!(ok, "{linked}");
    assert_eq!(linked, text);
    // A hyphenated program name resolves with either spelling.
    let (ok, text) = run(binary(), &["search_v3", "--help"]);
    assert!(ok, "{text}");
    let (ok, text) = run(binary(), &["search-v3", "--help"]);
    assert!(ok, "{text}");
    // A non-clap program keeps its own usage on stderr and its exit status.
    let (ok, text) = run(binary(), &["jope_strategy_screen"]);
    assert!(!ok);
    assert!(text.contains("jope_strategy_screen: subcommands: run | score"), "{text}");
}
