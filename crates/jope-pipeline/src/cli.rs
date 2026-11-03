//! `jope pipeline …`: the same typed `Cli`/`Cmd` dispatch shape the other
//! subcommand crates expose, so the umbrella delegates to it in process.

use anyhow::Result;
use std::process::ExitCode;

use crate::{receipt, run::RunOptions, search::Budget};

#[path = "cli/args.rs"]
mod args;
pub use args::{Cli, Cmd, LineageCmd, ParallelArgs, SearchArgs};

/// `--fitness`: a catalog name, `name:key=value,…`, inline JSON or `@path.json`.
pub fn load_objective(text: &str) -> Result<jope_runtime::fitness::Fitness> {
    use std::io::Read;
    let spec = match text.strip_prefix('@') {
        Some(path) => {
            let mut bytes = Vec::new();
            std::fs::File::open(path)
                .map_err(|e| anyhow::anyhow!("--fitness {path}: {e}"))?
                .take((jope_runtime::fitness::MAX_SPEC_BYTES + 1) as u64)
                .read_to_end(&mut bytes)?;
            String::from_utf8(bytes)?
        }
        None => text.to_owned(),
    };
    let objective = jope_runtime::fitness::Fitness::parse(&spec)
        .map_err(|e| anyhow::anyhow!("--fitness: {e:#}"))?;
    crate::run::check_objective(&objective)?;
    Ok(objective)
}

pub fn dispatch(cli: Cli) -> ExitCode {
    match run(cli) {
        Ok(code) => code,
        Err(error) if crate::parallel::is_cancelled(&error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(crate::parallel::CANCELLED_EXIT)
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

pub fn run(cli: Cli) -> Result<ExitCode> {
    match cli.cmd {
        Cmd::Run {
            search,
            profile,
            output,
            seed_genomes,
            incumbent,
            fitness,
            json,
        } => {
            let objective = fitness.as_deref().map(load_objective).transpose()?;
            if let Some(mut options) = profile.options(&search, output.clone(), true)? {
                anyhow::ensure!(
                    objective.is_none(),
                    "--fitness ranks the legacy search profile; {} keeps its own objective",
                    profile.search_profile
                );
                options.run.seed_genomes = seed_genomes;
                options.run.incumbent = incumbent;
                return crate::kernel2::cli::print_run(&options, json);
            }
            let options = RunOptions {
                seed_genomes,
                incumbent,
                objective,
                ..search.options(output, true)?
            };
            let receipt = crate::run::run(&options)?;
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                let measured: u64 = receipt
                    .search
                    .cuts
                    .iter()
                    .map(|c| c.successful_measurements)
                    .sum();
                let score = match &receipt.search.best {
                    Some(best) => {
                        let objective = match (&receipt.search.objective, best.validation_objective)
                        {
                            (Some(objective), Some(loss)) => {
                                format!("hold {} loss {loss:.6}, ", objective.name)
                            }
                            _ => String::new(),
                        };
                        format!(
                            "{objective}hold RMSE {:.6}, hold R2 {:.6} over {} cut(s)",
                            best.validation_rmse, best.validation_r2, best.cuts_measured
                        )
                    }
                    None => "no measurement".to_owned(),
                };
                println!(
                    "Pipeline ran {} on {} cut(s): {measured} measurement(s); \
                     selected {} by {} ({score}); emitted {}; receipt in {}.",
                    receipt.dataset.id,
                    receipt.prepared.cuts.len(),
                    receipt.search.selected_genome,
                    receipt.search.selection,
                    receipt
                        .artifacts
                        .iter()
                        .map(|a| a.language.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    options.output.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Evolve {
            search,
            output,
            rounds,
            control,
            seeds,
            json,
        } => {
            let options = crate::lineage::EvolveOptions {
                run: search.options(output.clone(), true)?,
                output,
                rounds,
                control,
                seeds,
            };
            let lineage = crate::lineage::evolve(&options)?;
            if json {
                println!("{}", serde_json::to_string(&lineage)?);
            } else {
                for round in lineage.rounds.iter().chain(lineage.control.iter()) {
                    println!(
                        "{} {} ({}): mean held-out RMSE {:.6}, R2 {:.6} over {} cut(s); \
                         {} generation-1+ measurement(s); champion {}: {}",
                        round.kind,
                        round.round,
                        round.directory,
                        round.validation_rmse,
                        round.validation_r2,
                        round.cuts_measured,
                        round.generation_ge1_measured,
                        round.champion_origin(),
                        round.selected_genome
                    );
                }
                if let Some(control) = &lineage.control {
                    println!(
                        "the rounds made {} evaluation(s) in total, the control {} (one round's \
                         budget: not a compute-matched comparison)",
                        lineage
                            .rounds
                            .iter()
                            .map(|r| r.completed_evaluations)
                            .sum::<u64>(),
                        control.completed_evaluations
                    );
                }
                println!(
                    "lineage in {}",
                    options.output.join(crate::lineage::LINEAGE_FILE).display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Lineage {
            cmd:
                LineageCmd::Verify {
                    lineage,
                    require_bred_candidates,
                    json,
                },
        } => match crate::lineage::verify(&lineage, require_bred_candidates) {
            Ok(report) => {
                if json {
                    println!("{}", serde_json::to_string(&report)?);
                } else {
                    println!(
                        "ok {} ({} round(s) on prepared cuts {}; mean held-out RMSE by round \
                         {:?}; control {:?}; champion found in the round itself {:?}; \
                         improved on the round before {:?}; evaluations: rounds {} in total, \
                         control {:?})",
                        lineage.display(),
                        report.rounds,
                        report.prepared_sha256,
                        report.validation_rmse,
                        report.control_validation_rmse,
                        report.champion_found_in_round,
                        report.improved_on_previous_round,
                        report.rounds_completed_evaluations,
                        report.control_completed_evaluations
                    );
                }
                Ok(ExitCode::SUCCESS)
            }
            Err(error) => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({"schema": crate::lineage::LINEAGE_SCHEMA_ID,
                                           "ok": false, "error": format!("{error:#}")})
                    );
                } else {
                    eprintln!("invalid lineage: {error:#}");
                }
                Ok(ExitCode::FAILURE)
            }
        },
        Cmd::Score {
            receipt,
            split,
            dataset,
            registry,
            fitness,
            json,
        } => {
            anyhow::ensure!(split == "test", "--split must be test, got {split:?}");
            let (score, path) = crate::test_score::score(&crate::test_score::ScoreOptions {
                receipt,
                dataset,
                registry: registry.unwrap_or_else(crate::test_score::default_registry),
                clock: crate::clock::Clock::Wall,
                objective: fitness.as_deref().map(load_objective).transpose()?,
            })?;
            if !score.earlier_looks.is_empty() {
                eprintln!(
                    "jope pipeline score: the test split of {} was already scored for {} other \
                     genome(s) in {} ({}). This is look {} at it: keeping the test-best of \
                     several makes the test split a selection input. Report the genome chosen \
                     before any test score was seen, and name every look.",
                    score.dataset.dataset_hash,
                    score.earlier_looks.len(),
                    score.registry,
                    score.earlier_looks.join(", "),
                    score.earlier_looks.len() + 1
                );
            }
            if json {
                println!("{}", serde_json::to_string(&score)?);
            } else {
                println!(
                    "scored {} on {} test row(s) of {}: test RMSE {:.6}, test R2 {:.6} \
                     (development held-out RMSE {:.6}); score receipt {}",
                    score.genome,
                    score.dataset.test_rows,
                    score.dataset.id,
                    score.test_rmse,
                    score.test_r2,
                    score.validation_rmse,
                    path.display()
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::VerifyReceipt {
            receipt: path,
            evidence,
            json,
        } => {
            if crate::kernel2::run::is_v4(&path) {
                return crate::kernel2::cli::verify(&path, evidence, json);
            }
            match if evidence {
                crate::evidence::verify_run(&path).map(|run| run.receipt)
            } else {
                receipt::read(&path)
            } {
                Ok(receipt) => {
                    if json {
                        println!("{}", serde_json::to_string(&receipt)?);
                    } else {
                        println!(
                            "ok {} ({} stage(s), {} artifact(s), {})",
                            path.display(),
                            receipt.stages.len(),
                            receipt.artifacts.len(),
                            receipt.schema
                        );
                    }
                    Ok(ExitCode::SUCCESS)
                }
                Err(error) => {
                    if json {
                        println!(
                            "{}",
                            serde_json::json!({"schema": receipt::RECEIPT_SCHEMA_ID, "ok": false,
                                           "error": format!("{error:#}")})
                        );
                    } else {
                        eprintln!("invalid receipt: {error:#}");
                    }
                    Ok(ExitCode::FAILURE)
                }
            }
        }
        Cmd::Invent {
            train,
            target,
            work_dir,
            budget,
            seed,
            rng_seed,
            cuts,
            top_k,
            parallel,
            source_commit,
            json,
            events,
            fitness,
            stop_file,
        } => {
            let outcome = Budget::parse(&budget).and_then(|budget| {
                let objective = fitness.as_deref().map(load_objective).transpose()?;
                let options = crate::invent::InventOptions {
                    objective,
                    budget,
                    seed,
                    rng_seed,
                    cuts,
                    top_k,
                    threads: parallel.threads,
                    islands: parallel.islands,
                    migration_interval: parallel.migration_interval,
                    cpu_ceiling_millicores: parallel.cpu_ceiling,
                    source_commit,
                    cancel: Some(crate::parallel::termination_flag()?),
                    finish: Some(crate::parallel::finish_flag(stop_file.as_deref())?),
                    events,
                    ..crate::invent::InventOptions::new(train, target, work_dir)
                };
                crate::invent::invent(&options)
            });
            match outcome {
                Ok(candidates) => {
                    if json {
                        println!("{}", serde_json::to_string(&candidates)?);
                    } else {
                        for c in &candidates.candidates {
                            let objective = c
                                .mean_hold_objective
                                .map(|loss| format!("objective loss {loss:.6}, "))
                                .unwrap_or_default();
                            println!(
                                "{:>2}. {objective}hold RMSE {:.6}, hold R2 {:.6}  {}",
                                c.rank, c.mean_hold_rmse, c.mean_hold_r2, c.genome
                            );
                        }
                        println!(
                            "{} candidate(s) from {} ranked program(s); {} left out as not \
                             executable in jain-math {}",
                            candidates.candidates.len(),
                            candidates.ranked,
                            candidates.excluded_not_executable,
                            candidates.vocabulary.jain_math_rev
                        );
                    }
                    Ok(if candidates.ok {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(crate::invent::NO_CANDIDATES_EXIT)
                    })
                }
                // Cancellation keeps its own exit status (see `exec`).
                Err(error) if crate::parallel::is_cancelled(&error) => Err(error),
                Err(error) if json => {
                    println!(
                        "{}",
                        serde_json::json!({"schema": crate::invent::SCHEMA, "ok": false,
                                           "error": format!("{error:#}")})
                    );
                    Ok(ExitCode::from(2))
                }
                Err(error) => Err(error),
            }
        }
    }
}
