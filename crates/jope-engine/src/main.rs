use anyhow::Result;
use clap::Parser;

mod command;
mod dispatcher;

fn main() -> Result<()> {
    dispatcher::run(command::Cli::parse())
}
