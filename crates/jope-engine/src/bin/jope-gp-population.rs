//! Deterministic metadata inspection only; never admits or evaluates a fit.
use anyhow::Result;
use clap::Parser;
use serde_json::json;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    rng_seed: u64,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let population = jope_parallel::gp::metadata_initial_population(a.rng_seed)?;
    let rows:Vec<_>=population.into_iter().enumerate().map(|(i,genome)|json!({"index":i,"program_sha256":jope_parallel::manifest::hash(genome.as_bytes()),"genome":genome})).collect();
    println!(
        "{}",
        serde_json::to_string(
            &json!({"schema":"jope.gp_population_metadata.v1","rng_seed":a.rng_seed,"vocabulary":null,"population":rows,"execution_order":(0..80).collect::<Vec<_>>(),"numerical_work_started":false})
        )?
    );
    Ok(())
}
