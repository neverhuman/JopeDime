use super::*;

pub(super) fn handle(root: &Path, command: Cmd) -> Result<Option<Cmd>> {
    match command {
        Cmd::Status => println!(
            "{}",
            serde_json::to_string_pretty(&operations::status(&root)?)?
        ),
        Cmd::Tui {
            experiment_manifest,
        } => jope_parallel::dashboard::run_with_manifest(&root, experiment_manifest.as_deref())?,
        Cmd::LearningHistory => println!(
            "{}",
            serde_json::to_string_pretty(
                &jope_parallel::learning_history::Cache::default()
                    .refresh(&root)
                    .report()
            )?
        ),
        Cmd::Monitor => jope_parallel::telemetry::monitor(&root)?,
        Cmd::PilotGate => operations::pilot_gate(&root)?,
        #[cfg(feature = "neural")]
        Cmd::Campaign { spec } => jope_parallel::campaign::run(&root, &spec)?,
        #[cfg(feature = "neural")]
        Cmd::Cycle {
            corpus,
            output,
            steps,
            cuts,
            cut_start,
            contexts,
            initialize,
            observations,
            qualification,
        } => jope_parallel::pipeline::run(
            &root,
            &corpus,
            &output,
            steps,
            cuts,
            cut_start,
            contexts.as_deref(),
            initialize.as_deref(),
            &observations,
            qualification,
        )?,
        other => return Ok(Some(other)),
    }
    Ok(None)
}
