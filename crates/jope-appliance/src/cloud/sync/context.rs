//! The local record of what a run was started with (see `cloud_events::RunContext`).
use super::*;

fn context_path(run: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        run.len() == 30
            && run.starts_with("run_")
            && run[4..].bytes().all(|b| b.is_ascii_alphanumeric()),
        "invalid local run identity"
    );
    Ok(cloud_root()?
        .join("run-context")
        .join(format!("{run}.json")))
}

/// Remember a started run's target, task, objective, table name and profile. Never fatal for
/// the run: a failure only means its synced page has less to show.
pub fn remember_run(run: &jope_api::ids::RunId, context: &crate::cloud_events::RunContext) {
    let _ = (|| -> Result<()> {
        write_private(
            &context_path(&run.to_string())?,
            &serde_json::to_vec(context)?,
        )
    })();
}

pub fn run_context(run: &str) -> Option<crate::cloud_events::RunContext> {
    serde_json::from_slice(&read_private(&context_path(run).ok()?).ok()?).ok()
}
