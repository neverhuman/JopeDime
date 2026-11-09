//! `jain attach [RUN]` without a terminal: follow a run of the local engine until it ends, then
//! print its result and next steps, as `jain demo run` does. With a terminal, `jain attach`
//! opens the interactive interface on the run instead (`session::requested`).
use jope_api::{ids::RunId, RunState};
use jope_client::Transport;
use std::str::FromStr;

pub fn run<T: Transport>(run: Option<String>, client: &T) -> anyhow::Result<()> {
    let run = match run {
        Some(run) => RunId::from_str(&run).map_err(|_| {
            jope_runtime::Failure::input(format!("{run} is not a run identifier (run_…)"))
        })?,
        None => crate::terminal::runs::active_run(client).ok_or_else(|| {
            jope_runtime::Failure::input(
                "no run is going on this machine; name one: jain attach RUN (jain runs list shows them)",
            )
        })?,
    };
    let snapshot = client.run_status(&run)?;
    if !matches!(
        snapshot.state,
        RunState::Completed | RunState::Cancelled | RunState::Failed
    ) {
        eprintln!(
            "Following {run} ({}); Ctrl-C stops following, the run continues.",
            crate::terminal::status::state_name(snapshot.state)
        );
    }
    let current = super::demo::wait_terminal(client, snapshot)?;
    for line in super::demo::result_lines(&current) {
        println!("{line}");
    }
    anyhow::ensure!(
        current.state == RunState::Completed,
        "run {} ended {}",
        current.run_id,
        crate::terminal::status::state_name(current.state)
    );
    Ok(())
}
