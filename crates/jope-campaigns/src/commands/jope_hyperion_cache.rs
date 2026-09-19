use std::env;
use std::path::PathBuf;

use clap::Parser;
use crate::hyperion_gate::{
    build_registry_cache, freeze_precomputed_train_feature_coverage,
    verify_precomputed_train_feature_coverage, JopeHyperionTrainOnlyRegistryV2,
};
use crate::jope_dlgp_best5::validate_hyperion_campaign_registry_ids;

#[derive(Debug, Parser)]
#[command(
    name = "jope_hyperion_cache",
    about = "Freeze/verify precomputed train-side Hyperion matrices or build fresh encoder caches"
)]
struct Args {
    /// JopeHyperionTrainOnlyRegistry.v2 authority.
    #[arg(long)]
    train_only_registry: PathBuf,
    /// Optional opaque member to backfill. Omit to freeze complete coverage.
    #[arg(long)]
    dataset_id: Option<String>,
}

pub fn main(args: Vec<String>) -> anyhow::Result<()> {
    let mut raw = args;
    let operation = raw
        .get(1)
        .filter(|value| matches!(value.as_str(), "freeze-registry" | "build" | "verify"))
        .cloned();
    if operation.is_some() {
        raw.remove(1);
    }
    let args = Args::parse_from(raw);
    let registry_bytes = match std::fs::read(&args.train_only_registry) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("jope_hyperion_cache: read train-only registry: {error}");
            std::process::exit(2);
        }
    };
    let registry: JopeHyperionTrainOnlyRegistryV2 = match serde_json::from_slice(&registry_bytes) {
        Ok(registry) => registry,
        Err(error) => {
            eprintln!("jope_hyperion_cache: decode train-only registry: {error}");
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
        eprintln!("jope_hyperion_cache: {error}");
        std::process::exit(2);
    }
    let result: Result<String, String> = match operation.as_deref() {
        Some("freeze-registry") => freeze_precomputed_train_feature_coverage(
            &args.train_only_registry,
        )
        .and_then(|coverage| {
            serde_json::to_string(&coverage)
                .map_err(|error| format!("encode precomputed coverage: {error}"))
        }),
        Some("verify") => verify_precomputed_train_feature_coverage(&args.train_only_registry)
            .and_then(|coverage| {
                serde_json::to_string(&coverage)
                    .map_err(|error| format!("encode precomputed coverage: {error}"))
            }),
        Some("build") | None => {
            build_registry_cache(&args.train_only_registry, args.dataset_id.as_deref()).and_then(
                |coverage| {
                    serde_json::to_string(&coverage)
                        .map_err(|error| format!("encode encoder cache coverage: {error}"))
                },
            )
        }
        Some(_) => unreachable!(),
    };
    match result {
        Ok(coverage) => println!("{coverage}"),
        Err(error) => {
            eprintln!("jope_hyperion_cache: {error}");
            std::process::exit(2);
        }
    }
    Ok(())
}
