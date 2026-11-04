use super::*;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::jope_generalized_gp::{raw_sha256, ArtifactStoreV1};
use serde::Serialize;

#[derive(Serialize)]
struct ProposalCatalogV1<'a> {
    schema: &'static str,
    campaign_freeze_sha256: &'a str,
    task: &'static str,
    datasets: Vec<ProposalCatalogDatasetV1<'a>>,
}

#[derive(Serialize)]
struct ProposalCatalogDatasetV1<'a> {
    dataset_sha256: &'a str,
    training_path: String,
    training_sha256: &'a str,
    metadata_path: String,
    metadata_sha256: &'a str,
    feature_count: u32,
}

pub fn proposal_job_identity(campaign: &CampaignFreezeV1) -> Result<String, String> {
    Ok(raw_sha256(
        format!("best5-propose-all-v1|{}", campaign.identity()?).as_bytes(),
    ))
}

fn write_once(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            file.sync_all()
                .map_err(|error| format!("sync {}: {error}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path)
                .map_err(|read| format!("read existing {}: {read}", path.display()))?;
            if present == bytes {
                Ok(())
            } else {
                Err(format!(
                    "immutable worker input collision at {}",
                    path.display()
                ))
            }
        }
        Err(error) => Err(format!("create {}: {error}", path.display())),
    }
}

/// Executes the single all-dataset inference or loads the already-published
/// batch on retry. Once the job identity exists, the runtime is never invoked.
pub fn propose_all(
    campaign: &CampaignFreezeV1,
    artifact_root: &Path,
    work_root: &Path,
) -> Result<ProposalBatchFreezeV1, String> {
    campaign.validate()?;
    campaign
        .proposal_generator_artifact
        .verify("proposal generator")?;
    campaign
        .proposal_runtime_artifact
        .verify("proposal runtime")?;
    campaign.checkpoint.verify_artifact()?;
    let store = ArtifactStoreV1::new(artifact_root);
    let job_identity = proposal_job_identity(campaign)?;
    if store.contains(&job_identity)? {
        let batch: ProposalBatchFreezeV1 = store.load(&job_identity)?;
        batch.validate(campaign)?;
        let canonical_identity = batch.identity(campaign)?;
        if !store.contains(&canonical_identity)? {
            // A retry may observe the publish-once job alias after a crash
            // immediately before canonical indexing. Repair that index from
            // the validated published bytes; never rerun inference.
            store.publish(&canonical_identity, &batch)?;
        }
        return Ok(batch);
    }
    let campaign_sha256 = campaign.identity()?;
    let directory = work_root.join(&job_identity[..16]);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create proposal work directory: {error}"))?;
    let catalog = ProposalCatalogV1 {
        schema: "JopeDlgpBest5ProposalCatalog.v1",
        campaign_freeze_sha256: &campaign_sha256,
        task: "regression",
        datasets: campaign
            .datasets
            .iter()
            .map(|dataset| ProposalCatalogDatasetV1 {
                dataset_sha256: &dataset.dataset_sha256,
                training_path: Path::new(&campaign.sotarena_root)
                    .join(&dataset.train_relative_path)
                    .to_string_lossy()
                    .into_owned(),
                training_sha256: &dataset.train_sha256,
                metadata_path: Path::new(&campaign.sotarena_root)
                    .join(&dataset.metadata_relative_path)
                    .to_string_lossy()
                    .into_owned(),
                metadata_sha256: &dataset.metadata_sha256,
                feature_count: dataset.feature_count,
            })
            .collect(),
    };
    let catalog_path = directory.join("train-only-catalog.json");
    write_once(
        &catalog_path,
        &crate::search_v3::canonical_json_bytes(&catalog)?,
    )?;
    let output_path = directory.join("proposal-batch.json");
    if output_path.exists() {
        return Err("unpublished proposal output collision requires operator review".into());
    }
    let output = Command::new(&campaign.proposal_runtime_artifact.path)
        .args([
            "--generator",
            &campaign.proposal_generator_artifact.path,
            "--checkpoint",
            &campaign.checkpoint.artifact_path,
            "--catalog",
            catalog_path
                .to_str()
                .ok_or_else(|| "proposal catalog path is not UTF-8".to_string())?,
            "--output",
            output_path
                .to_str()
                .ok_or_else(|| "proposal output path is not UTF-8".to_string())?,
            "--proposals-per-dataset",
            "64",
        ])
        .output()
        .map_err(|error| format!("launch frozen proposal runtime: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "frozen proposal runtime failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut batch: ProposalBatchFreezeV1 = serde_json::from_slice(
        &fs::read(&output_path).map_err(|error| format!("read proposal output: {error}"))?,
    )
    .map_err(|error| format!("decode proposal output: {error}"))?;
    batch.normalize_completion_order(campaign);
    batch.validate(campaign)?;
    let canonical_identity = batch.identity(campaign)?;
    // Publish the deterministic job alias first. Its existence is the
    // durable no-rerun marker; canonical indexing is replay-safe afterward.
    store.publish(&job_identity, &batch)?;
    store.publish(&canonical_identity, &batch)?;
    Ok(batch)
}
