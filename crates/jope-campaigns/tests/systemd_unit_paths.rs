//! Service units must not depend on a mutable checkout.
//!
//! A rollback candidate was held red because seven units declared
//! `WorkingDirectory=/home/ubuntu/jope-dime` -- a git worktree that is edited
//! continuously, absent entirely on xbabe0, and therefore not a thing a
//! deployed service may resolve against. Rollback closure gates activation, so
//! four lines of unit text held up every deployment.
//!
//! This lives in `tests/` rather than inside the bundle validator because that
//! validator is owned by the other agent and is under active edit; the
//! invariant is worth guarding either way, and a test here cannot collide.

use std::fs;
use std::path::{Path, PathBuf};

/// Roots that are edited in place. A unit resolving against one of these has
/// no stable identity: the bytes under it can change between the moment a
/// bundle is built and the moment a service starts.
const MUTABLE_ROOTS: [&str; 3] = [
    "/home/ubuntu/jope-dime",
    "/home/ubuntu/jope-prime",
    "/home/ubuntu/jope\u{2d}slime",
];

fn unit_files() -> Vec<PathBuf> {
    let systemd = Path::new(env!("CARGO_MANIFEST_DIR")).join("systemd");
    let mut units: Vec<PathBuf> = fs::read_dir(&systemd)
        .unwrap_or_else(|error| panic!("read {}: {error}", systemd.display()))
        .filter_map(|entry| entry.ok().map(|value| value.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("service"))
        .collect();
    units.sort();
    assert!(!units.is_empty(), "no service units found in {}", systemd.display());
    units
}

fn directive<'a>(contents: &'a str, key: &str) -> Option<&'a str> {
    contents
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(key))
}

#[test]
fn units_never_resolve_against_a_mutable_checkout() {
    let mut offenders = Vec::new();
    for unit in unit_files() {
        let contents = fs::read_to_string(&unit).expect("read unit");
        let Some(working) = directive(&contents, "WorkingDirectory=") else {
            continue;
        };
        let working = working.trim();
        if MUTABLE_ROOTS
            .iter()
            .any(|root| working == *root || working.starts_with(&format!("{root}/")))
        {
            offenders.push(format!(
                "{}: WorkingDirectory={working}",
                unit.file_name().unwrap().to_string_lossy()
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "units resolve against a mutable checkout, which holds rollback \
         candidates red and cannot be relied on across nodes:\n  {}",
        offenders.join("\n  ")
    );
}

/// The safety premise of pointing every unit at one working directory: none of
/// them resolves anything relative to it. If a unit ever gains a relative
/// ExecStart, changing its WorkingDirectory silently changes what it runs.
#[test]
fn unit_exec_start_paths_are_absolute() {
    let mut relative = Vec::new();
    for unit in unit_files() {
        let contents = fs::read_to_string(&unit).expect("read unit");
        for line in contents.lines().map(str::trim) {
            let Some(exec) = line.strip_prefix("ExecStart=") else {
                continue;
            };
            // systemd allows a leading '-' (ignore failure) and '@' (argv[0]).
            let command = exec.trim_start_matches(['-', '@', '+', '!']);
            if !command.starts_with('/') {
                relative.push(format!(
                    "{}: ExecStart={command}",
                    unit.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    assert!(
        relative.is_empty(),
        "ExecStart must be absolute, or WorkingDirectory becomes load-bearing:\n  {}",
        relative.join("\n  ")
    );
}
