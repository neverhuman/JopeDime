//! `jope`: the one command line. Fixture, kernel, champion and router verbs live in this
//! crate; `jope models …`, `jope datasets …` and `jope pipeline …` delegate to the library
//! crates' own clap commands (no process is spawned).

use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jope_cli::{
    emit_genome, encode_dataset, fit_dataset, generate_fixtures, load_router, local_champions,
    local_train_step, refuse_s3, resolve_root, route_dataset, search_dataset, verify_inputs,
    write_json, Verification, DEFAULT_GENOME,
};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(
    name = "jope",
    version,
    about = "JOPE-Dime fixtures, encode, fit/emit/search, local champions, centroid router, model bundles, dataset catalog and the product pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Report current ASM capabilities and remaining qualification gaps.
    Doctor,
    /// Verify and plan the private, fixed-panel benchmark.
    Benchmark(jope_bench::panel::cli::BenchmarkCli),
    /// Generate and optimize executable algorithms from the verified development panel.
    Optimize(jope_bench::panel::optimize_cli::OptimizeCli),
    /// Verify the ten synthetic fixtures against the pinned deterministic generator.
    VerifyInputs {
        /// Explicitly select offline qualification; no network access is performed.
        #[arg(long)]
        offline: bool,
        /// Emit one machine-readable result, including on verification failure.
        #[arg(long)]
        json: bool,
        /// Directory containing catalog.json; otherwise resolve at runtime.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Create the versioned synthetic corpus in a new output directory.
    GenerateFixtures {
        #[arg(long)]
        output: PathBuf,
        /// Check an existing directory for exact deterministic regeneration instead.
        #[arg(long)]
        check: bool,
        #[arg(long)]
        json: bool,
    },
    /// Encode one admitted dataset with CPU Hyperion. Needs HYPERION_V7_WEIGHTS; does not download.
    Encode {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Fit a GP genome on one admitted dataset (train R², holdout R²).
    Fit {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        #[arg(long)]
        json: bool,
    },
    /// Emit working Python / Rust / C++17 / JavaScript from a GP genome.
    Emit {
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        /// rust, python, cpp17, javascript, or all
        #[arg(long)]
        lang: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Score a tiny ridge-family GP candidate set on one admitted dataset.
    Search {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Local champion bank over admitted fixtures. --pull/--push fail closed (no live S3).
    Champions {
        #[arg(long)]
        local: bool,
        #[arg(long)]
        pull: bool,
        #[arg(long)]
        push: bool,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// One local train step: fit a 128-d centroid router on champion genomes. Live --env fails closed.
    TrainStep {
        #[arg(long)]
        local: bool,
        #[arg(long)]
        env: Option<String>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Route one dataset through a trained local centroid router, then fit the chosen genome.
    Route {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        router: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Model bundles: fetch, verify, publish and locate content-addressed weights.
    Models(jope_models::cli::Cli),
    /// Dataset catalog: ingest, fetch, verify and link content-addressed datasets.
    Datasets(jope_datasets::cli::Cli),
    /// Product pipeline: dataset, prepared cuts, GP search, emitted code and a receipt.
    Pipeline(jope_pipeline::cli::Cli),
}

fn execute(command: &Command) -> Result<Verification> {
    match command {
        Command::VerifyInputs { offline, root, .. } => {
            ensure!(
                *offline,
                "select --offline for synthetic input qualification"
            );
            verify_inputs(&resolve_root(root.clone())?)
        }
        Command::GenerateFixtures { output, check, .. } => {
            if !check {
                generate_fixtures(output)?;
            }
            verify_inputs(output)
        }
        Command::Doctor
        | Command::Benchmark(_)
        | Command::Optimize(_)
        | Command::Encode { .. }
        | Command::Fit { .. }
        | Command::Emit { .. }
        | Command::Search { .. }
        | Command::Champions { .. }
        | Command::TrainStep { .. }
        | Command::Route { .. }
        | Command::Models(_)
        | Command::Datasets(_)
        | Command::Pipeline(_) => {
            unreachable!("kernel, models, datasets and pipeline commands are dispatched separately")
        }
    }
}

fn print_json<T: serde::Serialize>(
    json: bool,
    report: &T,
    text: impl FnOnce() -> String,
) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::to_string(report).expect("serializable report")
        );
    } else {
        println!("{}", text());
    }
    ExitCode::SUCCESS
}

fn fail(json: bool, schema: &str, error: anyhow::Error) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({"schema": schema, "ok": false, "error": format!("{error:#}")})
        );
    } else {
        eprintln!("jope: {error:#}");
    }
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let Some(command) = Cli::parse().command else {
        println!("jope {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    };
    let command = match command {
        Command::Doctor => {
            println!("{}", jope_pipeline::front_door::doctor());
            return ExitCode::SUCCESS;
        }
        Command::Benchmark(cli) => return jope_bench::panel::cli::execute(cli),
        Command::Optimize(cli) => return jope_bench::panel::optimize_cli::execute(cli),
        Command::Models(cli) => return jope_models::cli::exec(cli),
        Command::Datasets(cli) => return jope_datasets::cli::exec(cli),
        Command::Pipeline(cli) => return jope_pipeline::cli::exec(cli),
        command => command,
    };
    match &command {
        Command::Encode { input, json } => {
            return match encode_dataset(input) {
                Ok(report) => print_json(*json, &report, || {
                    format!(
                        "Encoded {} ({} train rows → {} cols).",
                        report.dataset_id, report.train_rows, report.output_cols
                    )
                }),
                Err(error) => fail(*json, "jope.encode.v1", error),
            };
        }
        Command::Fit {
            input,
            genome,
            json,
        } => {
            return match fit_dataset(input, genome) {
                Ok(report) => print_json(*json, &report, || {
                    format!(
                        "Fitted {} on {} (train R² {:.4}, hold R² {:.4}).",
                        report.genome, report.dataset_id, report.train_r2, report.hold_r2
                    )
                }),
                Err(error) => fail(*json, "jope.fit.v1", error),
            };
        }
        Command::Emit {
            genome,
            lang,
            output,
            json,
        } => {
            return match emit_genome(genome, lang, output.as_deref()) {
                Ok(report) => print_json(*json, &report, || {
                    format!(
                        "Emitted {} artifact(s) from {}.",
                        report.artifacts.len(),
                        report.genome
                    )
                }),
                Err(error) => fail(*json, "jope.emit.v1", error),
            };
        }
        Command::Search { input, json } => {
            return match search_dataset(input) {
                Ok(report) => print_json(*json, &report, || {
                    format!(
                        "Search picked {} on {} (hold R² {:.4}).",
                        report.genome, report.dataset_id, report.hold_r2
                    )
                }),
                Err(error) => fail(*json, "jope.search.v1", error),
            };
        }
        Command::Champions {
            local,
            pull,
            push,
            root,
            output,
            json,
        } => {
            let selected = [*local, *pull, *push].into_iter().filter(|v| *v).count();
            if selected != 1 {
                return fail(
                    *json,
                    "jope.champions.v1",
                    anyhow::anyhow!("select exactly one of --local, --pull, --push"),
                );
            }
            if *pull || *push {
                return match refuse_s3() {
                    Ok(report) => print_json(*json, &report, || "unreachable".into()),
                    Err(error) => fail(*json, "jope.champions.v1", error),
                };
            }
            return match resolve_root(root.clone()).and_then(|root| local_champions(&root)) {
                Ok(report) => {
                    if let Some(path) = output {
                        if let Err(error) = write_json(&path.join("champions.json"), &report) {
                            return fail(*json, "jope.champions.v1", error);
                        }
                    }
                    print_json(*json, &report, || {
                        format!("Local champion bank: {} datasets.", report.dataset_count)
                    })
                }
                Err(error) => fail(*json, "jope.champions.v1", error),
            };
        }
        Command::TrainStep {
            local,
            env,
            root,
            output,
            json,
        } => {
            if env.as_deref() == Some("dev") || env.as_deref() == Some("prod") {
                return fail(
                    *json,
                    "jope.train-step.v1",
                    anyhow::anyhow!(
                        "live --env={env} train-step is not wired on this SHA; use --local",
                        env = env.as_deref().unwrap_or("")
                    ),
                );
            }
            if !*local {
                return fail(
                    *json,
                    "jope.train-step.v1",
                    anyhow::anyhow!("select --local (live S3 train-step is fail-closed)"),
                );
            }
            return match resolve_root(root.clone()).and_then(|root| local_train_step(&root)) {
                Ok(report) => {
                    if let Some(path) = output {
                        if let Err(error) =
                            write_json(&path.join("routing-labels.v1.json"), &report)
                        {
                            return fail(*json, "jope.train-step.v1", error);
                        }
                        if let Err(error) = write_json(&path.join("router.v1.json"), &report.router)
                        {
                            return fail(*json, "jope.train-step.v1", error);
                        }
                    }
                    print_json(*json, &report, || {
                        format!(
                            "Local train-step fitted a centroid router on {} labels ({} classes).",
                            report.labels.len(),
                            report.router.classes.len()
                        )
                    })
                }
                Err(error) => fail(*json, "jope.train-step.v1", error),
            };
        }
        Command::Route {
            input,
            router,
            json,
        } => {
            return match load_router(router).and_then(|router| route_dataset(input, &router)) {
                Ok(report) => print_json(*json, &report, || {
                    format!(
                        "Routed {} to {} (cosine {:.4}, hold R² {:.4}).",
                        report.dataset_id, report.genome, report.cosine, report.hold_r2
                    )
                }),
                Err(error) => fail(*json, "jope.route.v1", error),
            };
        }
        _ => {}
    }
    let json = match &command {
        Command::VerifyInputs { json, .. } | Command::GenerateFixtures { json, .. } => *json,
        _ => unreachable!("kernel commands returned above"),
    };
    match execute(&command) {
        Ok(report) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&report).expect("serializable verification")
                );
            } else {
                println!(
                    "Verified {} synthetic datasets, {} bytes, offline.",
                    report.dataset_count, report.bytes
                );
            }
            ExitCode::SUCCESS
        }
        Err(error) => fail(json, "jope.input-verification.v1", error),
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = env!("CARGO_PKG_VERSION").split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|character| character.is_ascii_digit())));
    }

    #[test]
    fn umbrella_keeps_every_verb_and_lists_models_and_datasets() {
        let cmd = Cli::command();
        let names: Vec<&str> = cmd.get_subcommands().map(|c| c.get_name()).collect();
        for verb in [
            "verify-inputs",
            "generate-fixtures",
            "encode",
            "fit",
            "emit",
            "search",
            "champions",
            "train-step",
            "route",
            "models",
            "datasets",
            "pipeline",
        ] {
            assert!(names.contains(&verb), "{verb} missing from {names:?}");
        }
        let models = cmd.find_subcommand("models").unwrap();
        let verbs: Vec<&str> = models.get_subcommands().map(|c| c.get_name()).collect();
        assert_eq!(verbs, ["fetch", "verify", "publish", "path"]);
        let datasets = cmd.find_subcommand("datasets").unwrap();
        assert!(datasets.find_subcommand("fetch").is_some());
        let pipeline = cmd.find_subcommand("pipeline").unwrap();
        let verbs: Vec<&str> = pipeline.get_subcommands().map(|c| c.get_name()).collect();
        assert_eq!(verbs, ["run", "verify-receipt"]);
        Cli::command().debug_assert();
    }
}
