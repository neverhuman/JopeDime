//! Service units must not depend on a mutable checkout, and must carry no
//! host-specific root.
//!
//! A rollback candidate was held red because seven units declared the source
//! checkout as their `WorkingDirectory` -- a git worktree that is edited
//! continuously, absent entirely on xbabe0, and therefore not a thing a
//! deployed service may resolve against. Rollback closure gates activation, so
//! four lines of unit text held up every deployment.
//!
//! The tracked units are templates (`deploy/systemd/**/*.in`, ADR-0003) with
//! four placeholders. This test renders them the way `deploy/systemd/render.sh`
//! does and checks the rendered text: every path a unit resolves lives under
//! one of the rendered roots (or is a system interpreter), no `/home/` literal
//! survives, no `@…@` placeholder is left, and no self-play unit works out of
//! the source checkout.
//!
//! This lives in `tests/` rather than inside the bundle validator because that
//! validator is owned by the other agent and is under active edit; the
//! invariant is worth guarding either way, and a test here cannot collide.

use std::fs;
use std::path::{Path, PathBuf};

use jope_paths::systemd::{render_tree, templates_under, Roots};

/// Interpreters a unit may name outside the rendered roots.
const SYSTEM_PREFIXES: [&str; 3] = ["/usr/bin/", "/bin/", "/usr/local/bin/"];

fn templates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/systemd")
}

fn roots() -> Roots {
    Roots {
        home: PathBuf::from("/opt/jope/checkout"),
        state: PathBuf::from("/var/lib/jope"),
        bundle: PathBuf::from("/opt/jope/bundle/bin"),
        user: "jope".to_owned(),
    }
}

/// Renders every template into a fresh directory and returns the rendered
/// unit files (services, timers and targets), sorted.
fn rendered_units(label: &str) -> (PathBuf, Vec<PathBuf>) {
    let out = std::env::temp_dir().join(format!(
        "jope-systemd-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = fs::remove_dir_all(&out);
    let units = render_tree(&templates(), &out, &roots()).expect("render unit templates");
    assert!(
        !units.is_empty(),
        "no unit templates under {}",
        templates().display()
    );
    (out, units)
}

fn directive_lines(contents: &str, key: &str) -> Vec<String> {
    logical_lines(contents)
        .into_iter()
        .filter_map(|line| line.strip_prefix(key).map(str::to_owned))
        .collect()
}

/// systemd joins lines ending in `\`; the shell fragments in a few units span
/// several physical lines.
fn logical_lines(contents: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if let Some(head) = line.strip_suffix('\\') {
            current.push_str(head);
            current.push(' ');
        } else {
            current.push_str(line);
            lines.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn under_roots(path: &str, roots: &Roots) -> bool {
    let path = Path::new(path);
    path.starts_with(&roots.home)
        || path.starts_with(&roots.state)
        || path.starts_with(&roots.bundle)
}

#[test]
fn every_tracked_unit_is_a_template() {
    let stray: Vec<_> = walk(&templates())
        .into_iter()
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            (name.ends_with(".service") || name.ends_with(".timer") || name.ends_with(".target"))
                && !name.ends_with(".in")
        })
        .collect();
    assert!(
        stray.is_empty(),
        "rendered units committed next to the templates: {stray:?}"
    );
    let count = templates_under(&templates()).unwrap().len();
    assert!(
        count >= 60,
        "expected the 39 + 22 + 1 units as templates, found {count}"
    );
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            found.extend(walk(&path));
        } else {
            found.push(path);
        }
    }
    found
}

#[test]
fn rendered_units_resolve_only_under_the_rendered_roots() {
    let roots = roots();
    let (out, units) = rendered_units("roots");
    let mut offenders = Vec::new();
    for unit in &units {
        let contents = fs::read_to_string(unit).expect("read rendered unit");
        let name = unit.strip_prefix(&out).unwrap().display().to_string();
        if contents.contains("/home/") {
            offenders.push(format!("{name}: a /home/ literal survived rendering"));
        }
        if contents.contains('@') && contents.split('@').count() > 1 {
            for token in contents.split('@').skip(1).step_by(2) {
                if !token.is_empty() && token.chars().all(|c| c.is_ascii_uppercase() || c == '_') {
                    offenders.push(format!("{name}: placeholder @{token}@ left unrendered"));
                }
            }
        }
        for key in [
            "ExecStart=",
            "ExecStartPre=",
            "ExecStartPost=",
            "WorkingDirectory=",
            "EnvironmentFile=",
        ] {
            for value in directive_lines(&contents, key) {
                let value = value.trim_start_matches(['-', '@', '+', '!', ':']);
                for token in
                    value.split(|c: char| c.is_whitespace() || c == '=' || c == '\'' || c == ';')
                {
                    let token =
                        token.trim_matches(|c| c == '"' || c == ',' || c == '[' || c == ']');
                    if !token.starts_with('/') {
                        continue;
                    }
                    if under_roots(token, &roots)
                        || SYSTEM_PREFIXES.iter().any(|p| token.starts_with(p))
                    {
                        continue;
                    }
                    offenders.push(format!(
                        "{name}: {key}{token} is outside the rendered roots"
                    ));
                }
            }
        }
    }
    fs::remove_dir_all(&out).ok();
    assert!(
        offenders.is_empty(),
        "rendered units name paths no deployment can rely on:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn selfplay_units_never_resolve_against_the_source_checkout() {
    let roots = roots();
    let (out, units) = rendered_units("checkout");
    let mut offenders = Vec::new();
    for unit in units
        .iter()
        .filter(|u| u.strip_prefix(&out).unwrap().starts_with("selfplay"))
    {
        let contents = fs::read_to_string(unit).expect("read rendered unit");
        for working in directive_lines(&contents, "WorkingDirectory=") {
            if Path::new(working.trim()).starts_with(&roots.home) {
                offenders.push(format!(
                    "{}: WorkingDirectory={working}",
                    unit.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    fs::remove_dir_all(&out).ok();
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
    let (out, units) = rendered_units("absolute");
    let mut relative = Vec::new();
    for unit in &units {
        let contents = fs::read_to_string(unit).expect("read rendered unit");
        for exec in directive_lines(&contents, "ExecStart=") {
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
    fs::remove_dir_all(&out).ok();
    assert!(
        relative.is_empty(),
        "ExecStart must be absolute, or WorkingDirectory becomes load-bearing:\n  {}",
        relative.join("\n  ")
    );
}
