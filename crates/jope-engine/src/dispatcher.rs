use crate::command::{Cli, EngineCmd};
use anyhow::Result;
use jope_parallel::{audit, operations, prepare, snapshot, transport};
use std::path::{Path, PathBuf};

#[path = "dispatcher/context.rs"]
mod context;
#[path = "dispatcher/fit.rs"]
mod fit;
#[path = "dispatcher/learning.rs"]
mod learning;
#[path = "dispatcher/misc.rs"]
mod misc;
#[path = "dispatcher/neural.rs"]
mod neural;
#[path = "dispatcher/registry.rs"]
mod registry;

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

pub(crate) fn run(cli: Cli) -> Result<()> {
    let Cli { command, root, .. } = cli;
    let mut command = command;
    macro_rules! try_handler {
        ($module:ident) => {
            match $module::handle(&root, command)? {
                Some(next) => command = next,
                None => return Ok(()),
            }
        };
    }
    try_handler!(registry);
    try_handler!(context);
    try_handler!(learning);
    try_handler!(fit);
    try_handler!(neural);
    match misc::handle(&root, command)? {
        Some(_) => anyhow::bail!("no dispatcher handler accepts the selected command"),
        None => Ok(()),
    }
}
