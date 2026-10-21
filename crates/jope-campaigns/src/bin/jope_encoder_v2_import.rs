//! Offline xbabe2 importer for fetched Encoder V2 SCQ output capsules.

use anyhow::{Context, Result};
use clap::Parser;
use jope_dime_regressor::encoder_v2_execution::{
    EncoderAttemptReceiptV2, EncoderExecutionReceiptV2,
};
use jope_dime_regressor::encoder_v2_identity::{decode_dataset_input, decode_encoder_identity};
use jope_dime_regressor::encoder_v2_store::{EncoderV2Store, PublishOutcome};
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::V1EncodingQuarantineV1;
use serde_json::json;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn authoritative_store_str => paths::flywheel_encodings_root().join("expanded-v2")
);

#[derive(Debug, Parser)]
#[command(name = "jope-encoder-v2-import")]
struct Args {
    #[arg(long, default_value = authoritative_store_str())]
    store: PathBuf,
    #[arg(long)]
    transaction_id: String,
    #[arg(long)]
    encoder_identity: PathBuf,
    #[arg(long)]
    dataset_input: PathBuf,
    #[arg(long)]
    execution_receipt: PathBuf,
    #[arg(long)]
    attempt_receipt: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    scq_job_identity_sha256: String,
    #[arg(long)]
    quarantine_index: Option<PathBuf>,
    #[arg(long)]
    v1_source_identity: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let encoder = decode_encoder_identity(&fs::read(&args.encoder_identity)?)?;
    let input = decode_dataset_input(&fs::read(&args.dataset_input)?)?;
    let execution: EncoderExecutionReceiptV2 = serde_json::from_reader(std::io::BufReader::new(
        File::open(&args.execution_receipt).context("open fetched execution receipt")?,
    ))?;
    let attempt: EncoderAttemptReceiptV2 = serde_json::from_reader(std::io::BufReader::new(
        File::open(&args.attempt_receipt).context("open fetched attempt receipt")?,
    ))?;
    let quarantine = args
        .quarantine_index
        .as_ref()
        .map(|path| -> Result<V1EncodingQuarantineV1> {
            let index: V1EncodingQuarantineV1 =
                serde_json::from_reader(std::io::BufReader::new(File::open(path)?))?;
            index.validate()?;
            Ok(index)
        })
        .transpose()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let store = EncoderV2Store::new(&args.store);
    let outcome = store.publish_success(
        &args.transaction_id,
        &encoder,
        &input,
        &execution,
        &attempt,
        &fs::read(&args.output)?,
        &args.scq_job_identity_sha256,
        now,
        args.v1_source_identity.as_deref(),
        quarantine.as_ref(),
        |_| Ok(()),
    )?;
    let (disposition, binding) = match outcome {
        PublishOutcome::Published(binding) => ("published", binding),
        PublishOutcome::Converged(binding) => ("converged", binding),
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema": "JopeEncoderV2ImportResult.v1",
            "disposition": disposition,
            "cache_key_sha256": binding.cache_key_sha256,
            "output_sha256": binding.output_sha256,
            "authoritative_store": args.store,
        }))?
    );
    Ok(())
}
