//! `jain`, the jAIn CLI (also built as `jopedime`): one signed executable that fits and
//! predicts tabular models with native Rust methods (deploy/appliance/README.md).
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
/// The user-facing product name, whatever the executable is called (`jain` or `jopedime`).
pub const PRODUCT: &str = "jain";

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--version` alone, and `--version --json` (SPECIFICATION.md §3.3).
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--version"] | ["-V"] => {
            out(&format!("{PRODUCT} {VERSION}\n"));
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
    let argv0 = std::env::args_os().next().unwrap_or_else(|| PRODUCT.into());
    let cli = Cli::parse_from(std::iter::once(argv0).chain(args.into_iter().map(Into::into)));
    match cli.command {
        #[cfg(feature = "research")]
        ApplianceCommand::Research(args) => report(research::run(args)),
        #[cfg(feature = "engine")]
        ApplianceCommand::Worker { protocol } => report(worker::run(protocol)),
        ApplianceCommand::Version(flag) => {
            if flag.json {
                print_json(&version_value());
            } else {
                out(&format!("{PRODUCT} {VERSION}\n"));
            }
            ExitCode::SUCCESS
        }
        ApplianceCommand::Doctor(flag) => match doctor() {
            Ok((value, ok)) => {
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
            train,
            output,
            model_root,
            common,
        } => run_fit(&train, output.as_deref(), &model_root, &common),
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
            train,
            input,
            output,
            common,
        } => run_fit_predict(&train, &input, &output, &common),
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
            idle_exit,
        } => ExitCode::from(cmd::serve::run(status, stop, idle_exit.as_deref())),
        ApplianceCommand::VerifySelf(flag) => report(verify_self(flag.json)),
        ApplianceCommand::Licenses(flag) => report(licenses(flag.json)),
    }
}
