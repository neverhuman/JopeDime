//! `jope-units-render --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir> [--templates <dir>]`
//!
//! Renders `deploy/systemd/**/*.in` (default: `<JOPE_HOME>/deploy/systemd`)
//! into `<out>/`, keeping the `selfplay/`, `engine/`, `pending/` layout.
//! `deploy/systemd/render.sh` is a thin wrapper around this binary.

use std::path::PathBuf;
use std::process::ExitCode;

use jope_paths::systemd::{render_tree, Roots};

const USAGE: &str = "usage: jope-units-render --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir> [--templates <dir>]";

fn main() -> ExitCode {
    let mut home = None;
    let mut state = None;
    let mut bundle = None;
    let mut user = None;
    let mut out = None;
    let mut templates = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let slot = match flag.to_str() {
            Some("--home") => &mut home,
            Some("--state") => &mut state,
            Some("--bundle") => &mut bundle,
            Some("--user") => &mut user,
            Some("--out") => &mut out,
            Some("--templates") => &mut templates,
            Some("-h" | "--help") => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("unknown argument {flag:?}\n{USAGE}");
                return ExitCode::from(2);
            }
        };
        match args.next() {
            Some(value) if !value.is_empty() => *slot = Some(value),
            _ => {
                eprintln!("{flag:?} needs a non-empty value\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(home), Some(state), Some(bundle), Some(user), Some(out)) =
        (home, state, bundle, user, out)
    else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let roots = Roots {
        home: PathBuf::from(home),
        state: PathBuf::from(state),
        bundle: PathBuf::from(bundle),
        user: user.to_string_lossy().into_owned(),
    };
    let templates = templates.map_or_else(
        || jope_paths::jope_home().join("deploy/systemd"),
        PathBuf::from,
    );
    match render_tree(&templates, &PathBuf::from(out), &roots) {
        Ok(rendered) => {
            for path in rendered {
                println!("{}", path.display());
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("jope-units-render: {error}");
            ExitCode::FAILURE
        }
    }
}
