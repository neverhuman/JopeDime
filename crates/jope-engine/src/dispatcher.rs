use crate::command::{Cli, EngineCmd};
use anyhow::Result;
use jope_parallel::{audit, operations, prepare, snapshot, transport};
use std::path::{Path, PathBuf};

#[path = "dispatcher/mod.rs"]
mod dispatch;

fn write_query_result(path: &std::path::Path, result: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(result)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    std::fs::File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new(".")),
    )?
    .sync_all()?;
    Ok(())
}

pub(crate) fn run(_cli: crate::command::Cli) -> Result<()> {
    let cli = Cli::parse();
    let Cli { command, root, .. } = cli;
    dispatch::run(&root, command)
}
