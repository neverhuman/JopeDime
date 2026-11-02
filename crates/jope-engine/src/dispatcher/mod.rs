use super::*;

mod context;
mod fit;
mod learning;
mod misc;
mod neural;
mod registry;

pub(super) fn run(root: &Path, mut command: EngineCmd) -> Result<()> {
    macro_rules! try_handler {
        ($module:ident) => {
            match $module::handle(root, command)? {
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
    try_handler!(misc);
    anyhow::bail!("no dispatcher handler accepts the selected command")
}
