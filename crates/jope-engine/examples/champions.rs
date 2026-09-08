//! Import completed, explicitly named teacher roots into durable incumbents.
use anyhow::Result;
use clap::Parser;
use jope_parallel::{champions, manifest::*};
use std::path::PathBuf;
#[derive(Parser)]
struct Cli {
    #[arg(long)] root:PathBuf,
    #[arg(long,required=true)] source:Vec<PathBuf>,
    #[arg(long)] output:PathBuf,
}
fn main()->Result<()> {
    let args=Cli::parse();
    let receipt=champions::import(&args.root,&args.source)?;
    write_json(&resolve(&args.root,&args.output)?,&receipt)?;
    println!("{}",serde_json::to_string(&receipt)?);
    Ok(())
}
