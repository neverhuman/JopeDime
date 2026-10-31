//! `jope`: the one command line. Fixture, kernel, champion and router verbs live in this
//! crate; `jope models …`, `jope datasets …`, `jope pipeline …` and `jope motifs …` delegate
//! to the library crates' own clap commands (no process is spawned).

use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jope_cli::{
    emit_genome_with_train, fit_dataset, generate_fixtures, load_router, local_champions,
    local_train_step, refuse_s3, resolve_root, route_dataset, search_dataset, verify_inputs,
    write_json, Verification, DEFAULT_GENOME,
};
use std::{path::PathBuf, process::ExitCode};

#[cfg(target_os = "linux")]
mod train;

#[derive(Parser)]
#[command(
    name = "jope",
    version,
    about = "JOPE-Dime command line for diagnostics, fixtures, benchmarking, optimization, training, encoding and fitting, models, datasets, and the product pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Report current ASM capabilities and remaining qualification gaps.
    Doctor,
    /// Fit, save, reload and score one bounded demonstration with measured elapsed time.
    Demo(jope_bench::panel::demo::DemoCli),
    /// Verify and plan the private, fixed-panel benchmark.
    Benchmark(jope_bench::panel::cli::BenchmarkCli),
    /// Plan, run, resume and verify bounded experimental policy training.
    #[cfg(target_os = "linux")]
    Train(train::TrainCli),
    /// Generate and optimize executable algorithms from the verified development panel.
    Optimize(jope_bench::panel::optimize_cli::OptimizeCli),
    /// Verify all forty synthetic fixtures against the pinned deterministic generator.
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
        #[command(flatten)]
        landmarks: jope_runtime::landmark_args::LandmarkArgs,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Fit a GP genome on one validated regression fixture (train R², validation R²).
    Fit {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        #[arg(long)]
        json: bool,
    },
    /// Emit working Python / Rust / C++17 / C++20 / JavaScript / TypeScript / Go / C17 / CUDA C++, or fitted C++17 from numeric training CSV.
    Emit {
        #[arg(long, default_value = DEFAULT_GENOME)]
        genome: String,
        /// rust, python, cpp17, cpp17-recipe-v2, cpp20, javascript (or js), typescript (or ts), go (or golang), c17 (or c), cuda (or cuda-cpp, cu), cpp17-fitted, or all (the four default recipes)
        #[arg(long)]
        lang: String,
        /// Numeric CSV with a final `target` column; required for cpp17-fitted.
        #[arg(long)]
        train: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Score a tiny ridge-family GP candidate set on a regression fixture's validation split.
    Search {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Search supported regression fixtures and report the rest as unsupported. --pull/--push fail closed (no live S3).
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
    /// One local train step: fit a 128-d centroid baseline; accepted policy weights unchanged. Live --env fails closed.
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
    /// Route one regression fixture through the local centroid baseline (accepted policy weights unchanged), then fit the chosen genome.
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
    Pipeline(Box<jope_pipeline::cli::Cli>),
    /// Reusable program fragments kept in a private table, with exact expansion checks.
    /// Hidden from the default help until the capability is published (owner decision,
    /// 2026-09-24); `jope motifs --help` still works.
    #[command(hide = true)]
    Motifs(jope_motifs::cli::Cli),
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
        #[cfg(target_os = "linux")]
        Command::Train(_) => unreachable!("training is dispatched separately"),
        Command::Doctor
        | Command::Demo(_)
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
        | Command::Pipeline(_)
        | Command::Motifs(_) => {
            unreachable!(
                "kernel, models, datasets, pipeline and motifs commands are dispatched separately"
            )
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
            println!("{}", jope_bench::panel::demo::doctor());
            return ExitCode::SUCCESS;
        }
        Command::Demo(cli) => return jope_bench::panel::demo::execute(cli),
        Command::Benchmark(cli) => return jope_bench::panel::cli::execute(cli),
        #[cfg(target_os = "linux")]
        Command::Train(cli) => return train::execute(cli),
        Command::Optimize(cli) => return jope_bench::panel::optimize_cli::execute(cli),
        Command::Models(cli) => return jope_models::cli::exec(cli),
        Command::Datasets(cli) => return jope_datasets::cli::exec(cli),
        Command::Pipeline(cli) => return jope_pipeline::cli::exec(*cli),
        Command::Motifs(cli) => return jope_motifs::cli::exec(cli),
        command => command,
    };
    match &command {
        Command::Encode {
            input,
            json,
            landmarks,
            device,
        } => {
            return match jope_cli::encode_dataset_with_landmarks(input, landmarks, device) {
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
            train,
            output,
            json,
        } => {
            return match emit_genome_with_train(genome, lang, train.as_deref(), output.as_deref()) {
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
                        format!(
                            "Local champion bank: {} supported, {} unsupported ({} fixtures).",
                            report.supported_count, report.unsupported_count, report.dataset_count
                        )
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
                            "Local train-step fitted a centroid baseline on {} supported labels ({} classes); {} fixtures excluded. Accepted policy weights unchanged.",
                            report.supported_count,
                            report.router.classes.len(),
                            report.unsupported_count
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
    use clap::{CommandFactory, Parser};

    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = env!("CARGO_PKG_VERSION").split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|character| character.is_ascii_digit())));
    }

    #[test]
    fn about_mentions_the_current_command_groups() {
        let about = Cli::command().get_about().unwrap().to_string();
        for topic in [
            "diagnostics",
            "benchmarking",
            "optimization",
            "training",
            "models",
            "datasets",
        ] {
            assert!(about.contains(topic), "help summary omits {topic}: {about}");
        }
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
            "motifs",
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
        assert_eq!(
            verbs,
            [
                "run",
                "verify-receipt",
                "evolve",
                "lineage",
                "score",
                "invent"
            ]
        );
        let lineage = pipeline.find_subcommand("lineage").unwrap();
        let verbs: Vec<&str> = lineage.get_subcommands().map(|c| c.get_name()).collect();
        assert_eq!(verbs, ["verify"]);
        let motifs = cmd.find_subcommand("motifs").unwrap();
        let verbs: Vec<&str> = motifs.get_subcommands().map(|c| c.get_name()).collect();
        assert_eq!(verbs, ["mine", "validate"]);
        Cli::command().debug_assert();
    }

    #[test]
    fn motifs_help_passes_shipped_text_lint_and_owner_ban() {
        let mut cmd = Cli::command();
        cmd.build();
        assert!(
            !cmd.render_help().to_string().contains("motifs"),
            "motifs must stay out of the default help"
        );
        let motifs = cmd.find_subcommand_mut("motifs").expect("motifs");
        assert!(motifs.is_hide_set());
        let mut texts = vec![motifs.render_long_help().to_string()];
        for verb in ["mine", "validate"] {
            let command = motifs.find_subcommand_mut(verb).expect("verb");
            texts.push(command.render_long_help().to_string());
            texts.push(command.render_help().to_string());
        }
        // The implemented arguments the acceptance commands use.
        for flag in [
            "--state",
            "--out",
            "--json",
            "--min-datasets",
            "--max-stages",
        ] {
            assert!(texts[1].contains(flag), "mine lacks {flag}");
        }
        for flag in ["--table", "--state", "--report", "--public-out", "--json"] {
            assert!(texts[3].contains(flag), "validate lacks {flag}");
        }
        for text in &texts {
            assert!(
                feat_math::transpile::lint_shipped_text(text).is_empty(),
                "{text}"
            );
            let lower = text.to_lowercase();
            for word in jope_motifs::export::OWNER_BANNED {
                assert!(!lower.contains(word), "{word} in {text}");
            }
            assert!(!text.contains(jope_motifs::NAME_PREFIX), "{text}");
        }
    }

    #[test]
    fn motifs_verbs_refuse_missing_required_arguments() {
        for arguments in [
            ["jope", "motifs"].as_slice(),
            &["jope", "motifs", "mine"],
            &["jope", "motifs", "mine", "--state", "/state"],
            &["jope", "motifs", "validate"],
            &["jope", "motifs", "validate", "--state", "/state"],
        ] {
            assert!(Cli::try_parse_from(arguments).is_err(), "{arguments:?}");
        }
        assert!(Cli::try_parse_from(["jope", "motifs", "mine", "--out", "/mined"]).is_ok());
        assert!(Cli::try_parse_from(["jope", "motifs", "validate", "--table", "/t.json"]).is_ok());
    }
}
