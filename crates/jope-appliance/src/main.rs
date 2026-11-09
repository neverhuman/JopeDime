//! `jopedime`: one signed executable that fits and predicts tabular models with native Rust
//! methods (deploy/appliance/README.md).
//!
//! Every command that fits, predicts, verifies or prints notices first authenticates the
//! executable's own package (footer, manifest, signature, executable region); an integrity
//! failure is fatal there and never becomes a fallback. `doctor` reports it with exit 3;
//! `--version --json` reports it in its output. Exit statuses follow SPECIFICATION.md §13.3:
//! 0 success, 2 input, 3 integrity, 4 backend, 5 budget, 6 numerical, 7 output.
#![forbid(unsafe_code)]

mod bootstrap;
mod cli;
#[cfg(feature = "client")]
mod cloud;
#[cfg(feature = "client")]
mod cloud_events;
#[path = "cmd/mod.rs"]
mod cmd;
#[path = "cmd/common.rs"]
mod common;
#[cfg(feature = "client")]
#[path = "terminal/draft.rs"]
mod draft;
mod io;
mod landmarks;
#[cfg(feature = "mcp")]
pub mod mcp;
mod model;
#[cfg(feature = "research")]
#[path = "cmd/research.rs"]
mod research;
#[cfg(feature = "client")]
#[path = "terminal/session.rs"]
mod session;
#[path = "terminal/theme.rs"]
pub mod terminal;
#[cfg(feature = "client")]
mod update;
#[cfg(feature = "engine")]
mod worker;

use clap::Parser;
use cli::{ApplianceCommand, Cli, CommonArgs, TrainArgs};
use jope_assets::Package;
use jope_runtime::api::{self, FitOptions};
use jope_runtime::backend::{self, DeviceRequest};
use jope_runtime::budget::Budget;
use jope_runtime::error::kind_of;
use jope_runtime::fitted::{CreatedBy, FittedModel};
use jope_runtime::limits::Limits;
use jope_runtime::methods;
use jope_runtime::receipt::{
    ErrorRecord, FileRecord, MethodRecord, PackageRecord, Receipt, TaskRecord, Timings,
};
use jope_runtime::table;
use jope_runtime::task::{self, TaskChoice};
use jope_runtime::{Failure, FailureKind, RUNTIME_ABI};
use serde_json::{json, Value};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The command users type (`jain`, the jAIn CLI). The executable, its crate, its data directory
/// (`~/.local/share/jopedime`) and machine-readable fields (`version --json` `product`) keep
/// the `jopedime` name; only what people read says jAIn / `jain` (#2144).
pub(crate) const CLI: &str = "jain";

/// `jain 1.0.0`. Invoked under its build name (`jopedime…`, as the release checks run the
/// unrenamed artifact), the line keeps that name, so `jopedime version` still prints
/// `jopedime X.Y.Z` first.
fn version_line() -> String {
    let invoked = std::env::args_os().next().and_then(|argv0| {
        Path::new(&argv0)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    });
    version_line_for(invoked.as_deref())
}

fn version_line_for(invoked: Option<&str>) -> String {
    version_line_with(invoked, release_version(), &bootstrap::source_sha40())
}

/// `jain 1.0.0-rc.4 (2ab7f551c9b1)`: the release identity and the source commit, so a report
/// names the exact build (journey J3-F09). Under the build name the line stays exactly
/// `jopedime X.Y.Z`, which deploy/release/release.sh checks.
fn version_line_with(invoked: Option<&str>, release: &str, sha40: &str) -> String {
    if invoked.is_some_and(|name| name.starts_with("jopedime")) {
        return format!("jopedime {VERSION}\n");
    }
    match sha40.get(..12).filter(|_| sha40 != "unknown") {
        Some(sha) => format!("{CLI} {release} ({sha})\n"),
        None => format!("{CLI} {release}\n"),
    }
}

/// The release this executable was built as: `JAIN_RELEASE_VERSION` at build time when it is
/// this Cargo version or a release candidate of it (`1.0.0-rc.4` is built from 1.0.0), else the
/// Cargo version. `version --json` keeps `version` = the Cargo version for the installers.
pub(crate) fn release_version() -> &'static str {
    release_version_for(option_env!("JAIN_RELEASE_VERSION"))
}

fn release_version_for(var: Option<&'static str>) -> &'static str {
    let candidate = |v: &str| {
        v.strip_prefix(VERSION)
            .and_then(|rest| rest.strip_prefix("-rc."))
            .is_some_and(|n| {
                !n.is_empty() && !n.starts_with('0') && n.bytes().all(|b| b.is_ascii_digit())
            })
    };
    match var {
        Some(v) if v == VERSION || candidate(v) => v,
        _ => VERSION,
    }
}

/// `example/NAME.csv` names a bundled demo (#2142). Builds without the client carry no demos,
/// so their bytes never enlarge an engine-only executable.
fn example_path(path: std::path::PathBuf) -> std::path::PathBuf {
    #[cfg(feature = "client")]
    {
        terminal::demos::resolve(&path)
    }
    #[cfg(not(feature = "client"))]
    {
        path
    }
}

/// `jain fit` / `fit-predict` train on this machine. Where own data trains only in the veox
/// Cloud (the public edition, owner rule 2026-11-06), they take only the bundled examples; the
/// refusal says what to run instead (or to sign in first).
#[cfg(feature = "client")]
fn own_data_fit(train: &Path) -> Option<String> {
    if !cloud::route::cloud_first() || terminal::demos::is_bundled(train) {
        return None;
    }
    Some(match cloud::connection() {
        Err(_) => cloud::route::Gate::SignedOut.message(false),
        Ok(_) => FIT_OWN_DATA.to_owned(),
    })
}

#[cfg(not(feature = "client"))]
fn own_data_fit(_train: &Path) -> Option<String> {
    None
}

#[cfg(feature = "client")]
const FIT_OWN_DATA: &str = "`jain fit` and `jain fit-predict` train on this machine, which runs only the bundled examples (example/NAME.csv). Your own data trains in the veox Cloud: `jain run FILE.csv --target COLUMN` (jAIn asks before uploading), then predict new rows in your workspace.";

/// Printed when `jain` runs without arguments and without a terminal: the full-screen TUI
/// cannot start, so say why and what works without one, then show the help (#2142).
#[cfg(feature = "client")]
const NO_TERMINAL: &str =
    "jain: the interactive TUI needs a terminal (stdin and stdout are not one here).\n\
Without a terminal, use the commands directly, for example:\n  \
jain demo list                                   the bundled demos\n  \
jain demo run regression_parkinsons --json       run one and print the result\n  \
jain auth login, then jain run data.csv --target COLUMN --cloud --upload-consent\n  \
                                                 your own data trains in the veox Cloud; --upload-consent is\n  \
                                                 the user's yes to the upload, so ask them first\n  \
jain run example/regression_parkinsons.csv --target target   a bundled demo file\n";

/// `jain demo NAME`: say the command that runs it.
#[cfg(feature = "mcp")]
fn demo_hint(args: &[String]) -> Option<String> {
    let [command, name, ..] = args else {
        return None;
    };
    if command != "demo"
        || name.starts_with('-')
        || matches!(name.as_str(), "list" | "run" | "help")
    {
        return None;
    }
    Some(match terminal::demos::find(name) {
        Some(demo) => format!(
            "`jain demo {name}` is not a command; did you mean `jain demo run {}`? (`jain demo list` names them all)",
            demo.name
        ),
        None => format!(
            "there is no `jain demo {name}`: use `jain demo list`, then `jain demo run NAME`"
        ),
    })
}

/// The status a run will exit with, so a closed stdout while printing its receipt keeps it.
static PENDING_EXIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Write to stdout. A closed pipe (`jopedime licenses | head`) ends the program quietly instead
/// of panicking; any other write failure is an output failure.
fn out(text: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if let Err(error) = stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        if error.kind() == std::io::ErrorKind::BrokenPipe {
            std::process::exit(i32::from(
                PENDING_EXIT.load(std::sync::atomic::Ordering::SeqCst),
            ));
        }
        eprintln!("jain: stdout: {error}");
        std::process::exit(i32::from(FailureKind::Output.exit_code()));
    }
}

fn print_json(value: &Value) {
    let mut text = serde_json::to_string_pretty(value).expect("JSON values serialize");
    text.push('\n');
    out(&text);
}

mod info;
mod run;

use info::{doctor, licenses, methods_value, verify_self, version_value};
use run::Run;

mod commands;
use commands::{load_fitness, report, run_fit, run_fit_predict, run_predict, serve, summary_line};

fn main() -> ExitCode {
    #[allow(unused_mut)]
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--accept-terms` and `--telemetry on|off` are global: accepted before or after any
    // subcommand, applied once, and removed before the subcommand parses its own arguments.
    #[cfg(feature = "client")]
    if let Err(message) = terminal::onboarding::take_flags(&mut args) {
        eprintln!("jain: {message}");
        return ExitCode::from(2);
    }
    // `--version` alone, and `--version --json` (SPECIFICATION.md §3.3).
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--version"] | ["-V"] => {
            out(&version_line());
            #[cfg(feature = "client")]
            update::notice_blocking(release_version());
            return ExitCode::SUCCESS;
        }
        ["--version", "--json"] | ["--json", "--version"] => {
            print_json(&version_value());
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    #[cfg(feature = "client")]
    if let Some(attach) = session::requested(&args) {
        return report(session::run(attach));
    }
    #[cfg(feature = "client")]
    if args.is_empty() {
        use clap::CommandFactory;
        eprint!("{NO_TERMINAL}");
        let mut command = Cli::command().bin_name(CLI);
        let _ = command.print_help();
        return ExitCode::from(2);
    }
    // `jain demo NAME` is the TUI's `/demo NAME` typed as a command (journey J2-F06).
    #[cfg(feature = "mcp")]
    if let Some(hint) = demo_hint(&args) {
        eprintln!("jain: {hint}");
        return ExitCode::from(2);
    }
    let cli = Cli::parse_from(std::iter::once(CLI.to_owned()).chain(args.iter().cloned()));
    match cli.command {
        #[cfg(feature = "research")]
        ApplianceCommand::Research(args) => report(research::run(args)),
        #[cfg(feature = "engine")]
        ApplianceCommand::Worker { protocol } => report(worker::run(protocol)),
        ApplianceCommand::Version(flag) => {
            if flag.json {
                print_json(&version_value());
            } else {
                out(&version_line());
            }
            ExitCode::SUCCESS
        }
        ApplianceCommand::Doctor(flag) => match doctor() {
            Ok((value, ok)) => {
                #[cfg(feature = "client")]
                update::notice_blocking(release_version());
                let code = if ok {
                    0
                } else {
                    FailureKind::Integrity.exit_code()
                };
                PENDING_EXIT.store(code, std::sync::atomic::Ordering::SeqCst);
                if flag.json {
                    print_json(&value);
                } else {
                    eprintln!(
                        "jain doctor: {} (package: {}; devices: {})",
                        if ok { "ok" } else { "NOT OK" },
                        value["package"]["status"].as_str().unwrap_or("?"),
                        value["devices"]["reason"].as_str().unwrap_or("?")
                    );
                    for item in value["unexecutable_promised_methods"]
                        .as_array()
                        .into_iter()
                        .flatten()
                    {
                        eprintln!(
                            "  promised but not executable: {} ({})",
                            item["id"], item["reason"]
                        );
                    }
                }
                ExitCode::from(code)
            }
            Err(error) => report(Err(error)),
        },
        #[cfg(feature = "client")]
        ApplianceCommand::Auth { command } => report(cloud::auth_command(command)),
        #[cfg(feature = "client")]
        ApplianceCommand::Cloud { command } => report(cloud::cloud_command(command)),
        ApplianceCommand::Methods { task, json } => {
            report(methods_value(task.as_deref()).map(|value| {
                if json.json {
                    print_json(&value);
                } else {
                    for m in value["methods"].as_array().into_iter().flatten() {
                        out(&format!(
                            "{:<36} {:<11} {:<7} {}
",
                            m["id"].as_str().unwrap_or(""),
                            m["task"].as_str().unwrap_or(""),
                            m["kind"].as_str().unwrap_or(""),
                            if m["executable"] == true {
                                "executable"
                            } else {
                                "not executable"
                            }
                        ));
                    }
                    out(&format!(
                        "auto: bounded selection ({})\n",
                        jope_runtime::automl::SELECTION_POLICY
                    ));
                }
            }))
        }
        ApplianceCommand::Fit {
            mut train,
            output,
            model_root,
            common,
        } => {
            train.train = example_path(train.train);
            if let Some(refusal) = own_data_fit(&train.train) {
                // Package integrity still comes first (exit 3).
                if let Err(error) = bootstrap::open() {
                    return report(Err(error));
                }
                eprintln!("jain: {refusal}");
                return ExitCode::from(2);
            }
            run_fit(&train, output.as_deref(), &model_root, &common)
        }
        ApplianceCommand::Model(args) => report(model::run(args)),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Run(args) => match cmd::run::dispatch(args) {
            Err(error) => match cmd::run::dispatch_exit_code(&error) {
                Some(code) => {
                    if code != 130 {
                        eprintln!("jain: {error:#}");
                    }
                    ExitCode::from(code)
                }
                None => report(Err(error)),
            },
            result => report(result),
        },
        #[cfg(feature = "mcp")]
        ApplianceCommand::Demo { command } => report(cmd::demo::run(command)),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Attach { run } => report((|| {
            let client = cmd::client()?;
            cmd::attach::run(run, &client)
        })()),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Status(args) => report((|| {
            let client = cmd::client()?;
            cmd::status::run(args, &client)
        })()),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Events(args) => report((|| {
            let client = cmd::client()?;
            cmd::events::run(args, &client)
        })()),
        #[cfg(feature = "client")]
        ApplianceCommand::Control {
            run,
            action,
            body,
            idempotency_key,
        } => report(cmd::control::run(run, action, body, idempotency_key)),
        #[cfg(feature = "client")]
        ApplianceCommand::Restart {
            run,
            new_seed,
            warm,
            body,
            idempotency_key,
        } => report(cmd::restart::run(
            run,
            new_seed,
            warm,
            body,
            idempotency_key,
        )),
        #[cfg(feature = "client")]
        ApplianceCommand::Runs { command } => report(cmd::runs::run(command)),
        #[cfg(feature = "client")]
        ApplianceCommand::Code { run, alg, lang } => report(cmd::code::run(run, alg, lang)),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Export {
            run,
            alg,
            kind,
            lang,
            out,
            force,
            idempotency_key,
            json,
        } => cmd::dataset::report(cmd::export::run(cmd::export::Options {
            run: &run,
            alg: alg.as_deref(),
            kind,
            language: lang,
            out: &out,
            force,
            idempotency_key: idempotency_key.as_deref(),
            json_out: json,
        })),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Dataset { command } => cmd::dataset::report(cmd::dataset::run(command)),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Objective {
            command: cli::ObjectiveCommand::Validate { spec, json },
        } => cmd::dataset::report(cmd::objective::run(&spec, json)),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Gc {
            dry_run,
            idempotency_key,
            json,
        } => cmd::dataset::report(cmd::gc::run(dry_run, idempotency_key.as_deref(), json)),
        ApplianceCommand::Predict {
            model,
            input,
            output,
            common,
        } => run_predict(&model, &input, &output, &common),
        ApplianceCommand::FitPredict {
            mut train,
            input,
            output,
            common,
        } => {
            train.train = example_path(train.train);
            if let Some(refusal) = own_data_fit(&train.train) {
                // Package integrity still comes first (exit 3).
                if let Err(error) = bootstrap::open() {
                    return report(Err(error));
                }
                eprintln!("jain: {refusal}");
                return ExitCode::from(2);
            }
            let input = example_path(input);
            run_fit_predict(&train, &input, &output, &common)
        }
        ApplianceCommand::Fitness { command } => report(match command {
            cli::FitnessCommand::List => {
                print_json(&jope_runtime::service::catalog());
                Ok(())
            }
            cli::FitnessCommand::Validate { spec } => (|| {
                let fitness = load_fitness(&spec)?;
                print_json(&jope_runtime::service::describe(&fitness)?);
                Ok(())
            })(),
            cli::FitnessCommand::Schema => {
                print_json(&jope_runtime::fitness::schema());
                Ok(())
            }
            cli::FitnessCommand::Evaluate { input } => (|| {
                use std::io::Read;
                // Read at most one byte past the limit, so an oversized file is never loaded.
                let mut bytes = Vec::new();
                std::fs::File::open(input)?
                    .take((jope_runtime::service::MAX_MESSAGE_BYTES + 1) as u64)
                    .read_to_end(&mut bytes)?;
                anyhow::ensure!(
                    bytes.len() <= jope_runtime::service::MAX_MESSAGE_BYTES,
                    "evaluation request too large"
                );
                let request = serde_json::from_slice(&bytes)?;
                print_json(&jope_runtime::service::evaluate(request)?);
                Ok(())
            })(),
        }),
        #[cfg(feature = "mcp")]
        ApplianceCommand::Mcp => report(cmd::mcp_entry::run()),
        // Without the shared client there is no discovery: serve in process, as before #656.
        #[cfg(not(feature = "mcp"))]
        ApplianceCommand::Mcp => report(serve(true)),
        ApplianceCommand::Api => report(serve(false)),
        ApplianceCommand::Serve {
            status,
            stop,
            force,
            idle_exit,
        } => ExitCode::from(cmd::serve::run(status, stop, force, idle_exit.as_deref())),
        ApplianceCommand::VerifySelf(flag) => report(verify_self(flag.json)),
        ApplianceCommand::Licenses(flag) => report(licenses(flag.json)),
    }
}

#[cfg(test)]
mod version_line_tests {
    #[test]
    fn the_version_line_says_jain_unless_invoked_under_the_build_name() {
        let version = env!("CARGO_PKG_VERSION");
        let sha = "2ab7f551c9b1e2b48d2a0f6f5a7c1f0e9d8c7b6a";
        assert_eq!(
            super::version_line_with(Some("jain"), "1.0.0-rc.4", sha),
            "jain 1.0.0-rc.4 (2ab7f551c9b1)\n"
        );
        assert_eq!(
            super::version_line_with(None, version, "unknown"),
            format!("jain {version}\n")
        );
        assert!(super::version_line_for(None)
            .starts_with(&format!("jain {}", super::release_version())));
        // deploy/release/release.sh runs the unrenamed `jopedime-<tag>-<target>` artifact.
        assert_eq!(
            super::version_line_with(
                Some("jopedime-v1.0.0-x86_64-unknown-linux-gnu"),
                "1.0.0-rc.4",
                sha
            ),
            format!("jopedime {version}\n")
        );
    }

    #[test]
    fn the_release_identity_is_this_version_or_a_candidate_of_it() {
        let version = env!("CARGO_PKG_VERSION");
        assert_eq!(super::release_version_for(None), version);
        let rc: &'static str = Box::leak(format!("{version}-rc.4").into_boxed_str());
        assert_eq!(super::release_version_for(Some(rc)), rc);
        for bad in ["9.9.9", "-beta", "-rc.x", "-rc.", "x"] {
            let bad: &'static str = Box::leak(format!("{version}{bad}").into_boxed_str());
            assert_eq!(super::release_version_for(Some(bad)), version, "{bad}");
        }
        let zero: &'static str = Box::leak(format!("{version}-rc.0").into_boxed_str());
        assert_eq!(super::release_version_for(Some(zero)), version);
    }

    #[cfg(feature = "mcp")]
    #[test]
    fn jain_demo_name_suggests_jain_demo_run_name() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let hint = super::demo_hint(&args(&["demo", "regression_parkinsons"])).unwrap();
        assert!(
            hint.contains("did you mean `jain demo run regression_parkinsons`?"),
            "{hint}"
        );
        assert!(super::demo_hint(&args(&["demo", "nonsense"]))
            .unwrap()
            .contains("`jain demo list`"));
        for fine in [
            &["demo", "run", "x"][..],
            &["demo", "list"],
            &["demo", "--help"],
            &["run", "x"],
        ] {
            assert_eq!(super::demo_hint(&args(fine)), None, "{fine:?}");
        }
    }

    #[cfg(feature = "client")]
    #[test]
    fn without_a_terminal_the_message_names_the_noninteractive_commands() {
        assert!(super::NO_TERMINAL.contains("needs a terminal"));
        assert!(super::NO_TERMINAL.contains("jain demo run"));
    }
}
