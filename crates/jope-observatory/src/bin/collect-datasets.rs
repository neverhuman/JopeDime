//! Compact census of an operator-provided exact-version dataset metadata catalog. No data rows.
use anyhow::{bail, Context, Result};
use chrono::Utc;
use clap::Parser;
use jope_observatory::{digest, read_bounded};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    sha256: String,
    #[arg(long)]
    source_s3_uri: String,
    #[arg(long)]
    source_version_id: String,
    #[arg(long)]
    source_modified_at: String,
    #[arg(long)]
    output: PathBuf,
}
fn census(bytes: &[u8]) -> Result<Value> {
    let mut ids = BTreeSet::new();
    let mut licenses = BTreeMap::<String, usize>::new();
    let mut origins = BTreeMap::<String, usize>::new();
    for line in std::str::from_utf8(bytes)?
        .lines()
        .filter(|s| !s.trim().is_empty())
    {
        let row: Value = serde_json::from_str(line)?;
        if row["schema"] != "jope.dataset-catalog/v1" {
            bail!("unknown dataset catalog schema");
        }
        let id = row["id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 256)
            .context("dataset ID missing")?;
        if !ids.insert(id.to_owned()) {
            bail!("duplicate dataset ID requires explicit upstream reconciliation");
        }
        *licenses
            .entry(
                row["provenance"]["license"]["status"]
                    .as_str()
                    .unwrap_or("unknown")
                    .into(),
            )
            .or_default() += 1;
        *origins
            .entry(row["origin"].as_str().unwrap_or("unknown").into())
            .or_default() += 1;
        if ids.len() > 100_000 {
            bail!("dataset count exceeds bounds");
        }
    }
    if ids.is_empty() {
        bail!("empty catalog cannot establish a dataset census");
    }
    Ok(json!({"count":ids.len(),"license_status":licenses,"origins":origins}))
}
fn main() -> Result<()> {
    let a = Args::parse();
    if !a
        .source_s3_uri
        .starts_with("s3://veox-jopedime-datasets-use2/catalog/")
        || a.source_version_id.is_empty()
    {
        bail!("ordinary dataset-catalog URI and explicit VersionId required");
    }
    let bytes = read_bounded(&a.catalog, 16 * 1024 * 1024)?;
    if digest(&bytes) != a.sha256 {
        bail!("catalog SHA256 mismatch");
    }
    let mut dataset = census(&bytes)?;
    dataset["name"] = json!(a.source_s3_uri);
    dataset["sha256"] = json!(a.sha256);
    dataset["version_id"] = json!(a.source_version_id);
    dataset["catalog_modified_at"] = json!(a.source_modified_at);
    dataset["observed_at"] = json!(Utc::now().to_rfc3339());
    dataset["status"] = json!("version-pinned");
    dataset["scope"]=json!("Dataset IDs in locally SHA256-verified exact-version metadata copy. Remote version/latest-key observation belongs to the operator receipt; no training data rows read.");
    let out = json!({"schema":"jope.dataset-observation/v1","datasets":[dataset]});
    let temp = a
        .output
        .with_extension(format!("{}.new", std::process::id()));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    fs::write(&temp, serde_json::to_vec_pretty(&out)?)?;
    fs::rename(&temp, &a.output)?;
    println!("dataset census written to {}", a.output.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_counts_ids_and_preserves_unknown_license() {
        let v=census(b"{\"schema\":\"jope.dataset-catalog/v1\",\"id\":\"a\",\"origin\":\"real\"}\n{\"schema\":\"jope.dataset-catalog/v1\",\"id\":\"b\",\"origin\":\"real\",\"provenance\":{\"license\":{\"status\":\"recorded\"}}}\n").unwrap();
        assert_eq!(v["count"], 2);
        assert_eq!(v["license_status"]["unknown"], 1);
        assert_eq!(v["license_status"]["recorded"], 1);
    }
    #[test]
    fn duplicate_or_wrong_schema_cannot_make_a_census() {
        assert!(census(b"{\"schema\":\"unknown\",\"id\":\"a\"}").is_err());
        assert!(census(b"").is_err());
        let row = "{\"schema\":\"jope.dataset-catalog/v1\",\"id\":\"same\"}\n";
        assert!(census(row.repeat(2).as_bytes()).is_err());
    }
}
