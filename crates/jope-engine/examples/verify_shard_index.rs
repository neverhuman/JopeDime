//! Read-only verifier for a historical shard index. Usage:
//!   verify_shard_index <index.json> <shard_root> <expected_shards>
//! Prints the verified count and exits non-zero on the first mismatch.

use anyhow::{Context, Result};
use jope_parallel::label_admission::verify_shard_index;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 4, "usage: verify_shard_index <index.json> <shard_root> <expected_shards>");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&args[1]).with_context(|| format!("read {}", args[1]))?,
    )?;
    let expected: usize = args[3].parse()?;
    let verified = verify_shard_index(&index, Path::new(&args[2]), expected)?;
    println!("verified_shards={verified} expected={expected}");
    Ok(())
}
