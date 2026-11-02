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

fn dispatch_datasets(cli: jope_datasets::cli::Cli) -> ExitCode {
    match jope_datasets::cli::run(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}
