use crate::target_encoder::TargetTask;
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod dispatch;
mod input;
mod regression_cache;
#[cfg(test)]
mod tests;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Encode the final (label) column of a CSV as the governed fused-512 target vector.
    EncodeTarget {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        task: TargetTask,
        #[arg(long, default_value = ".")]
        bundle_root: PathBuf,
    },
    Infer {
        #[arg(
            long,
            default_value = "artifacts/starforge/chimera_classification_expert_a.safetensors"
        )]
        weights: PathBuf,
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        test: PathBuf,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long, default_value = "f32")]
        dtype: String,
        #[arg(long, default_value_t = 1000)]
        max_context: usize,
        #[arg(long, default_value_t = 1000)]
        max_test: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    Bench {
        #[arg(
            long,
            default_value = "artifacts/starforge/chimera_classification_expert_a.safetensors"
        )]
        weights: PathBuf,
        #[arg(long, default_value = "data/binary")]
        root: PathBuf,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long, default_value = "f32")]
        dtype: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value_t = 1000)]
        max_context: usize,
        #[arg(long, default_value_t = 1000)]
        max_test: usize,
        #[arg(long, default_value_t = 0.25)]
        train_frac: f64,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value = "artifacts/starforge/meteor_results_rust.json")]
        out: PathBuf,
    },
    BenchRegression {
        #[arg(
            long,
            default_value = "artifacts/starforge/starlight_core_regressor.safetensors"
        )]
        weights: PathBuf,
        #[arg(long, default_value = "data/regression")]
        root: PathBuf,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value_t = 1000)]
        max_context: usize,
        #[arg(long, default_value_t = 1000)]
        max_test: usize,
        #[arg(long, default_value_t = 256)]
        max_features: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(
            long,
            default_value = "/tmp/starforge_regressor_verify/rust_bench_regression.json"
        )]
        out: PathBuf,
    },
    DumpRegressionFeatures {
        #[arg(
            long,
            default_value = "artifacts/starforge/starlight_core_regressor.safetensors"
        )]
        starlight_weights: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/foundation/tabicl-regressor-v2-chimera-shuffled.safetensors"
        )]
        tabicl_weights: PathBuf,
        #[arg(
            long,
            default_value = "artifacts/starforge/chimera_stackfusionreg.safetensors"
        )]
        chimera_weights: PathBuf,
        #[arg(long)]
        dataset_dir: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, default_value_t = 1000)]
        max_context: usize,
        #[arg(long, default_value_t = 1000)]
        max_test: usize,
        #[arg(long, default_value_t = 256)]
        max_features: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
}

pub fn run() -> Result<()> {
    dispatch::run()
}
