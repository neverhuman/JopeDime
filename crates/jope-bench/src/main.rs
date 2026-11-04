//! `jope-bench`: same-cut benchmark over a dataset tree, then a local Elo track from the reports.

mod cli;
#[cfg(test)]
mod tests;

fn main() -> anyhow::Result<()> {
    cli::run()
}
