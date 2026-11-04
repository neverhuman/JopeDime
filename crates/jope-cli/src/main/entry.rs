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
        Command::Models(cli) => return jope_models::cli::dispatch(cli),
        Command::Datasets(cli) => return dispatch_datasets(cli),
        Command::Pipeline(cli) => return jope_pipeline::cli::dispatch(*cli),
        Command::Motifs(cli) => return jope_motifs::cli::dispatch(cli),
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
    use super::{Cli, Command};
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
    fn models_cli_is_parsed_as_a_typed_command() {
        let cli = Cli::try_parse_from(["jope", "models", "verify", "--manifests-only"])
            .expect("valid typed models command");
        assert!(matches!(
            cli.command,
            Some(Command::Models(jope_models::cli::Cli {
                cmd: jope_models::cli::Cmd::Verify {
                    manifests_only: true,
                    ..
                },
                ..
            }))
        ));

        assert!(Cli::try_parse_from(["jope", "models", "unknown"]).is_err());
    }

    #[test]
    fn datasets_cli_is_parsed_as_a_typed_command() {
        let cli = Cli::try_parse_from(["jope", "datasets", "verify"])
            .expect("valid typed datasets command");
        assert!(matches!(
            cli.command,
            Some(Command::Datasets(jope_datasets::cli::Cli {
                cmd: jope_datasets::cli::Cmd::Verify { .. },
            }))
        ));
        assert!(Cli::try_parse_from(["jope", "datasets", "unknown"]).is_err());
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
