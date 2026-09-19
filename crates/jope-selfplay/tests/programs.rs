//! The `jope-selfplay` binary: its program list, the two dispatch forms, and
//! the committed link table (`deploy/systemd/BINARIES.md`).

use std::path::Path;
use std::process::Command;

fn run(program: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::new(program).args(args).output().expect("run");
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    (output.status.success(), text)
}

fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_jope-selfplay"))
}

#[test]
fn help_lists_the_thirty_programs() {
    let (ok, text) = run(binary(), &["--help"]);
    assert!(ok, "{text}");
    assert_eq!(jope_selfplay::TABLE.programs.len(), 30);
    assert!(text.contains("\n30 programs:\n"), "{text}");
    let expected = include_str!("fixtures/programs.txt");
    let (ok, list) = run(binary(), &["--list"]);
    assert!(ok, "{list}");
    assert_eq!(list, expected, "the program list changed; update tests/fixtures/programs.txt if intended");
    for name in jope_selfplay::TABLE.names() {
        assert!(text.contains(&format!("  {name} ")), "{name} missing from --help");
        assert!(name.starts_with("jope_") || name == "scq_runs_gc");
    }
    let (ok, text) = run(binary(), &[]);
    assert!(ok && text.contains("30 programs:"), "{text}");
}

#[test]
fn a_program_runs_by_first_argument_and_through_a_link() {
    // clap-based program: `--help` exits 0 and names the program.
    let (ok, text) = run(binary(), &["jope_selfplay_status", "--help"]);
    assert!(ok, "{text}");
    assert!(text.contains("jope_selfplay_status"), "{text}");
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("jope_selfplay_status");
    std::os::unix::fs::symlink(binary(), &link).unwrap();
    let (ok, linked) = run(&link, &["--help"]);
    assert!(ok, "{linked}");
    assert_eq!(linked, text, "argv[0] dispatch must run the same program");
    // Bare `--help` through a link still belongs to the program, not the table.
    assert!(!linked.contains("30 programs:"));
    let (ok, text) = run(binary(), &["no_such_program"]);
    assert!(!ok);
    assert!(text.contains("unknown program `no_such_program`"), "{text}");
}

#[test]
fn binaries_md_matches_both_program_tables() {
    let table = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/systemd/BINARIES.md"),
    )
    .expect("deploy/systemd/BINARIES.md");
    let mut rows: Vec<(String, String)> = table
        .lines()
        .filter(|line| line.starts_with("| `"))
        .map(|line| {
            let cells: Vec<_> = line.split('|').map(str::trim).collect();
            (
                cells[1].trim_matches('`').to_owned(),
                cells[2].trim_matches('`').to_owned(),
            )
        })
        .collect();
    let mut expected: Vec<(String, String)> = jope_selfplay::TABLE
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), "jope-selfplay".to_owned()))
        .chain(
            jope_campaigns::TABLE
                .names()
                .into_iter()
                .map(|name| (name.to_owned(), "jope-campaigns".to_owned())),
        )
        .collect();
    rows.sort();
    expected.sort();
    assert_eq!(rows, expected, "deploy/systemd/BINARIES.md is out of step with the program tables");
    assert_eq!(rows.len(), 56, "30 + 25 programs + the former root binary");
}

#[test]
fn every_unit_execstart_names_a_program_or_a_system_interpreter() {
    let units = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/systemd/selfplay");
    let mut checked = 0;
    for entry in std::fs::read_dir(&units).unwrap() {
        let path = entry.unwrap().path();
        let text = std::fs::read_to_string(&path).unwrap();
        for line in text.lines().filter(|l| l.starts_with("ExecStart=")) {
            let Some(rest) = line.strip_prefix("ExecStart=") else { continue };
            for word in rest.split_whitespace() {
                if let Some(name) = word.strip_prefix("@JOPE_BUNDLE@/") {
                    assert!(
                        jope_selfplay::TABLE.find(name).is_some()
                            || jope_campaigns::TABLE.find(name).is_some()
                            || name.starts_with("jope-parallel"),
                        "{}: `{name}` is not a program of jope-selfplay or jope-campaigns",
                        path.display()
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 30, "only {checked} bundle programs seen in the self-play units");
}
