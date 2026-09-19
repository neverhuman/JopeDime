use std::path::PathBuf;

use clap::Parser;
use crate::hyperion_gate::{run_gate_command, JopeHyperionTrainOnlyRegistryV2};
use crate::jope_dlgp_best5::validate_hyperion_campaign_registry_ids;

#[derive(Debug, Parser)]
#[command(
    name = "jope_hyperion_gate",
    about = "Train-only paired CV5/CV7 Hyperion feature gate"
)]
struct Args {
    /// Opaque dataset identifier in the authorized train-only registry.
    #[arg(long)]
    dataset_id: String,
    /// JopeHyperionTrainOnlyRegistry.v2 authority.
    #[arg(long)]
    train_only_registry: PathBuf,
    /// Frozen production-compatible AFPM predictor JSON.
    #[arg(long)]
    frozen_afpm_predictor: PathBuf,
    /// New or immutable artifact directory for receipts and the bound predictor.
    #[arg(long)]
    artifact_root: PathBuf,
}

pub fn main(args: Vec<String>) -> anyhow::Result<()> {
    let args = Args::parse_from(&args);
    let registry_bytes = match std::fs::read(&args.train_only_registry) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("jope_hyperion_gate: read train-only registry: {error}");
            std::process::exit(2);
        }
    };
    let registry: JopeHyperionTrainOnlyRegistryV2 = match serde_json::from_slice(&registry_bytes) {
        Ok(registry) => registry,
        Err(error) => {
            eprintln!("jope_hyperion_gate: decode train-only registry: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = validate_hyperion_campaign_registry_ids(
        &registry
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_id.clone())
            .collect::<Vec<_>>(),
    ) {
        eprintln!("jope_hyperion_gate: {error}");
        std::process::exit(2);
    }
    match run_gate_command(
        &args.dataset_id,
        &args.train_only_registry,
        &args.frozen_afpm_predictor,
        &args.artifact_root,
    ) {
        Ok(decision) => {
            println!(
                "{}",
                serde_json::to_string(&decision).expect("decision is serializable")
            );
        }
        Err(error) => {
            eprintln!("jope_hyperion_gate: {error}");
            std::process::exit(2);
        }
    }
    Ok(())
}
