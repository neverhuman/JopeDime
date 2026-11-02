use super::*;

pub(crate) fn handle(root: &Path, command: EngineCmd) -> Result<Option<EngineCmd>> {
    match command {
        EngineCmd::Status => println!(
            "{}",
            serde_json::to_string_pretty(&operations::status(&root)?)?
        ),
        EngineCmd::Tui {
            experiment_manifest,
        } => jope_parallel::dashboard::run_with_manifest(&root, experiment_manifest.as_deref())?,
        EngineCmd::LearningHistory => println!(
            "{}",
            serde_json::to_string_pretty(
                &jope_parallel::learning_history::Cache::default()
                    .refresh(&root)
                    .report()
            )?
        ),
        EngineCmd::Monitor => jope_parallel::telemetry::monitor(&root)?,
        EngineCmd::PilotGate => operations::pilot_gate(&root)?,
        #[cfg(feature = "neural")]
        EngineCmd::Campaign { spec } => jope_parallel::campaign::run(&root, &spec)?,
        #[cfg(feature = "neural")]
        EngineCmd::Cycle {
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
