use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::*;
use crate::jope_generalized_gp::{
    raw_sha256, DatasetRowV1, DatasetTableV1, GpTaskV1, SealedTestTableV1, SplitRowV1,
};

#[derive(Deserialize)]
struct SourceInputManifestV1 {
    datasets: Vec<SourceInputDatasetV1>,
}

#[derive(Deserialize)]
struct SourceInputDatasetV1 {
    dataset_hash: String,
    features: u32,
    meta_sha256: String,
    train_rows: u32,
    train_sha256: String,
    test_sha256: String,
}

#[derive(Deserialize)]
struct SourceObservationSnapshotV1 {
    snapshot_sha256: String,
}

#[derive(Deserialize)]
struct RegistryV1 {
    datasets: Vec<RegistryDatasetV1>,
}

#[derive(Deserialize)]
struct RegistryDatasetV1 {
    id: String,
    features: u32,
    files: RegistryFilesV1,
    upstream: RegistryUpstreamV1,
}

#[derive(Deserialize)]
struct RegistryFilesV1 {
    train: RegistryFileV1,
    test: RegistryFileV1,
    split_metadata: RegistryFileV1,
}

#[derive(Deserialize)]
struct RegistryFileV1 {
    path: String,
    sha256: String,
    #[serde(default)]
    rows: Option<u32>,
}

#[derive(Deserialize)]
struct RegistryUpstreamV1 {
    dataset_sha256: String,
    #[serde(default)]
    source_dataset_id: String,
    #[serde(default)]
    source_platform: Option<String>,
}

fn read_bound(path: &Path, expected: &str, name: &str) -> Result<Vec<u8>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("read {name} {}: {error}", path.display()))?;
    if raw_sha256(&bytes) != expected {
        return Err(format!(
            "{name} hash differs from its frozen registry binding"
        ));
    }
    Ok(bytes)
}

fn normalize_source_family(upstream: &RegistryUpstreamV1) -> Result<String, String> {
    let normalized = upstream
        .source_platform
        .as_deref()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if !normalized.is_empty() {
        return Ok(normalized);
    }
    if upstream
        .source_dataset_id
        .to_ascii_lowercase()
        .contains("hyper")
    {
        Ok("local-hyper".into())
    } else {
        Err(format!(
            "dataset {} lacks a source_platform and is not a legacy Hyper dataset",
            upstream.source_dataset_id
        ))
    }
}

fn safe_join(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err("registry path is not a safe relative path".into());
    }
    Ok(root.join(relative))
}

pub fn freeze_campaign(source: &CampaignSourceV1) -> Result<CampaignFreezeV1, String> {
    source.verify()?;
    let manifest: SourceInputManifestV1 = serde_json::from_slice(
        &fs::read(&source.input_manifest.path)
            .map_err(|error| format!("read source input manifest: {error}"))?,
    )
    .map_err(|error| format!("decode source input manifest: {error}"))?;
    let registry: RegistryV1 = serde_json::from_slice(
        &fs::read(&source.registry.path)
            .map_err(|error| format!("read source registry: {error}"))?,
    )
    .map_err(|error| format!("decode source registry: {error}"))?;
    let wins: SixWinManifestV1 = serde_json::from_slice(
        &fs::read(&source.six_win_manifest.path)
            .map_err(|error| format!("read source six-win manifest: {error}"))?,
    )
    .map_err(|error| format!("decode source six-win manifest: {error}"))?;
    wins.validate()?;
    let observation: SourceObservationSnapshotV1 = serde_json::from_slice(
        &fs::read(&source.observation_snapshot.path)
            .map_err(|error| format!("read source observation snapshot: {error}"))?,
    )
    .map_err(|error| format!("decode source observation snapshot: {error}"))?;
    crate::search_v3::validate_sha256(
        "observation snapshot protocol",
        &observation.snapshot_sha256,
    )?;
    if manifest.datasets.len() != ALL_DATASETS {
        return Err("source input manifest must contain exactly 42 datasets".into());
    }
    let by_id = registry
        .datasets
        .iter()
        .map(|dataset| (dataset.id.as_str(), dataset))
        .collect::<BTreeMap<_, _>>();
    if by_id.len() != registry.datasets.len() {
        return Err("source registry contains duplicate dataset IDs".into());
    }
    let win_ids = wins
        .dataset_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let root = Path::new(&source.sotarena_root);
    let mut datasets = Vec::with_capacity(ALL_DATASETS);
    for (index, input) in manifest.datasets.iter().enumerate() {
        let registered = by_id
            .get(input.dataset_hash.as_str())
            .ok_or_else(|| format!("registry omits frozen dataset {}", input.dataset_hash))?;
        if registered.features != input.features
            || registered.files.train.sha256 != input.train_sha256
            || registered.files.test.sha256 != input.test_sha256
            || registered.files.split_metadata.sha256 != input.meta_sha256
            || registered.files.train.rows != Some(input.train_rows)
        {
            return Err(format!(
                "registry bindings drifted for frozen dataset {}",
                input.dataset_hash
            ));
        }
        for (name, file) in [
            ("training", &registered.files.train),
            ("test", &registered.files.test),
            ("metadata", &registered.files.split_metadata),
        ] {
            let path = safe_join(root, &file.path)?;
            read_bound(&path, &file.sha256, name)?;
        }
        datasets.push(FrozenDatasetV1 {
            dataset_id: input.dataset_hash.clone(),
            dataset_sha256: registered.upstream.dataset_sha256.clone(),
            train_relative_path: registered.files.train.path.clone(),
            test_relative_path: registered.files.test.path.clone(),
            metadata_relative_path: registered.files.split_metadata.path.clone(),
            train_sha256: input.train_sha256.clone(),
            test_sha256: input.test_sha256.clone(),
            metadata_sha256: input.meta_sha256.clone(),
            train_rows: input.train_rows,
            feature_count: input.features,
            source_family: normalize_source_family(&registered.upstream)?,
            stage: if index < PILOT_DATASETS {
                DatasetStageV1::Pilot
            } else {
                DatasetStageV1::Confirmation
            },
            current_sotarena_win: win_ids.contains(input.dataset_hash.as_str()),
        });
    }
    if !win_ids
        .iter()
        .all(|id| datasets.iter().any(|dataset| dataset.dataset_id == *id))
    {
        return Err("six-win manifest contains a dataset outside the frozen panel".into());
    }
    let campaign = CampaignFreezeV1 {
        schema: CampaignFreezeV1::SCHEMA.into(),
        campaign_id: CAMPAIGN_ID.into(),
        input_manifest_sha256: source.input_manifest.sha256.clone(),
        panel_sha256: source.panel.sha256.clone(),
        observation_snapshot_sha256: observation.snapshot_sha256,
        generated_registry_sha256: source.registry.sha256.clone(),
        evaluator_sha256: source.evaluator.sha256.clone(),
        math_policy_sha256: source.math_policy.sha256.clone(),
        backend_policy_sha256: source.backend_policy.sha256.clone(),
        six_win_manifest_sha256: source.six_win_manifest.sha256.clone(),
        proposal_generator_sha256: source.proposal_generator.sha256.clone(),
        proposal_runtime_sha256: source.proposal_runtime.sha256.clone(),
        sotarena_root: source.sotarena_root.clone(),
        input_manifest_artifact: source.input_manifest.clone(),
        panel_artifact: source.panel.clone(),
        observation_snapshot_artifact: source.observation_snapshot.clone(),
        six_win_manifest_artifact: source.six_win_manifest.clone(),
        registry_artifact: source.registry.clone(),
        evaluator_artifact: source.evaluator.clone(),
        math_policy_artifact: source.math_policy.clone(),
        backend_policy_artifact: source.backend_policy.clone(),
        proposal_generator_artifact: source.proposal_generator.clone(),
        proposal_runtime_artifact: source.proposal_runtime.clone(),
        checkpoint: source.checkpoint.clone(),
        datasets,
        search_seeds: SEARCH_SEEDS.to_vec(),
        bootstrap_seed: BOOTSTRAP_SEED,
        budgets: Best5ArmV1::ALL
            .into_iter()
            .map(|arm| (arm, ArmBudgetV1::frozen()))
            .collect(),
        scq: ScqPolicyV1::frozen(),
        production_pointer_writes: false,
        production_learning_ingest: false,
        legacy_strategy_state_reused: false,
    };
    campaign.validate()?;
    Ok(campaign)
}

pub fn verify_campaign_artifacts(campaign: &CampaignFreezeV1) -> Result<(), String> {
    campaign.validate()?;
    for (name, artifact) in [
        ("input manifest", &campaign.input_manifest_artifact),
        ("panel", &campaign.panel_artifact),
        (
            "observation snapshot",
            &campaign.observation_snapshot_artifact,
        ),
        ("six-win manifest", &campaign.six_win_manifest_artifact),
        ("registry", &campaign.registry_artifact),
        ("evaluator", &campaign.evaluator_artifact),
        ("math policy", &campaign.math_policy_artifact),
        ("backend policy", &campaign.backend_policy_artifact),
        ("proposal generator", &campaign.proposal_generator_artifact),
        ("proposal runtime", &campaign.proposal_runtime_artifact),
    ] {
        artifact.verify(name)?;
    }
    campaign.checkpoint.verify_artifact()?;
    let root = Path::new(&campaign.sotarena_root);
    for dataset in &campaign.datasets {
        for (name, relative, expected) in [
            (
                "training",
                dataset.train_relative_path.as_str(),
                dataset.train_sha256.as_str(),
            ),
            (
                "test",
                dataset.test_relative_path.as_str(),
                dataset.test_sha256.as_str(),
            ),
            (
                "metadata",
                dataset.metadata_relative_path.as_str(),
                dataset.metadata_sha256.as_str(),
            ),
        ] {
            read_bound(&safe_join(root, relative)?, expected, name)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Best5TrainingTableV1 {
    pub table: DatasetTableV1,
    pub target_rows: Vec<TargetRowV1>,
    pub training_artifact: FrozenFileV1,
    pub metadata_artifact: FrozenFileV1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Best5SealedTableV1 {
    pub table: SealedTestTableV1,
    pub target_rows: Vec<TargetRowV1>,
    pub test_artifact: FrozenFileV1,
}

fn file_artifact(path: &Path, expected: &str) -> Result<FrozenFileV1, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if raw_sha256(&bytes) != expected {
        return Err(format!("{} changed after campaign freeze", path.display()));
    }
    Ok(FrozenFileV1 {
        path: path.to_string_lossy().into_owned(),
        sha256: expected.into(),
        bytes: bytes.len() as u64,
    })
}

/// Builds the only training table accepted by Best5. It opens the frozen
/// training and metadata files and deliberately never resolves the test path.
pub fn build_training_table(
    campaign: &CampaignFreezeV1,
    dataset_id: &str,
) -> Result<Best5TrainingTableV1, String> {
    campaign.validate()?;
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| format!("dataset {dataset_id} is not in the campaign"))?;
    let root = Path::new(&campaign.sotarena_root);
    let train_path = safe_join(root, &dataset.train_relative_path)?;
    let metadata_path = safe_join(root, &dataset.metadata_relative_path)?;
    let training_artifact = file_artifact(&train_path, &dataset.train_sha256)?;
    let metadata_artifact = file_artifact(&metadata_path, &dataset.metadata_sha256)?;
    let bytes =
        fs::read(&train_path).map_err(|error| format!("read frozen training table: {error}"))?;
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    for (ordinal, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if raw.is_empty() && ordinal + 1 == bytes.split(|byte| *byte == b'\n').count() {
            continue;
        }
        if raw.is_empty() {
            return Err("training CSV contains an empty source row".into());
        }
        let line = std::str::from_utf8(raw)
            .map_err(|error| format!("training CSV is not UTF-8: {error}"))?;
        let columns = line.trim_end_matches('\r').split(',').collect::<Vec<_>>();
        if columns.len() != dataset.feature_count as usize + 1 {
            return Err(format!(
                "training row {ordinal} has {} columns, expected {}",
                columns.len(),
                dataset.feature_count + 1
            ));
        }
        let target = columns
            .last()
            .unwrap()
            .trim()
            .parse::<f64>()
            .map_err(|error| format!("parse target at row {ordinal}: {error}"))?;
        if !target.is_finite() {
            return Err(format!("target at row {ordinal} is nonfinite"));
        }
        let mut feature_bits = Vec::with_capacity(dataset.feature_count as usize * 8);
        for value in &columns[..columns.len() - 1] {
            let parsed = value
                .trim()
                .parse::<f64>()
                .map_err(|error| format!("parse feature at row {ordinal}: {error}"))?;
            if !parsed.is_finite() {
                return Err(format!("feature at row {ordinal} is nonfinite"));
            }
            feature_bits.extend_from_slice(&parsed.to_bits().to_le_bytes());
        }
        let stable_row_id = raw_sha256(
            [
                dataset.dataset_sha256.as_bytes(),
                &(ordinal as u64).to_le_bytes(),
                raw,
            ]
            .concat()
            .as_slice(),
        );
        let atomic_group = raw_sha256(&feature_bits);
        let split = SplitRowV1 {
            stable_row_id: stable_row_id.clone(),
            atomic_group: atomic_group.clone(),
            regression_target: Some(target),
            binary_target: None,
        };
        rows.push(DatasetRowV1 {
            split,
            csv_line: line.into(),
        });
        targets.push(TargetRowV1 {
            stable_row_id,
            atomic_group,
            target,
        });
    }
    if rows.len() != dataset.train_rows as usize {
        return Err(format!(
            "training table has {} rows, expected {}",
            rows.len(),
            dataset.train_rows
        ));
    }
    let table = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: dataset.dataset_sha256.clone(),
        training_sha256: dataset.train_sha256.clone(),
        rows,
    };
    table.validate()?;
    Ok(Best5TrainingTableV1 {
        table,
        target_rows: targets,
        training_artifact,
        metadata_artifact,
    })
}

/// Opens sealed bytes only after the caller has atomically published and
/// supplied the dataset-level claim identity.
pub fn build_claimed_sealed_table(
    campaign: &CampaignFreezeV1,
    dataset_id: &str,
    claim: &SealedClaimV1,
) -> Result<Best5SealedTableV1, String> {
    campaign.validate()?;
    claim.validate()?;
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| format!("dataset {dataset_id} is not in the campaign"))?;
    if claim.campaign_freeze_sha256 != campaign.identity()?
        || claim.dataset_sha256 != dataset.dataset_sha256
    {
        return Err("sealed claim does not bind the requested campaign dataset".into());
    }
    let test_path = safe_join(
        Path::new(&campaign.sotarena_root),
        &dataset.test_relative_path,
    )?;
    // This is intentionally the first function in the Best5 layer that opens
    // the test path.
    let test_artifact = file_artifact(&test_path, &dataset.test_sha256)?;
    let bytes = fs::read(&test_path).map_err(|error| format!("read sealed test table: {error}"))?;
    let raw_rows = bytes.split(|byte| *byte == b'\n').collect::<Vec<_>>();
    let mut row_ids = Vec::new();
    let mut targets = Vec::new();
    for (ordinal, raw) in raw_rows.iter().enumerate() {
        if raw.is_empty() && ordinal + 1 == raw_rows.len() {
            continue;
        }
        if raw.is_empty() {
            return Err("sealed test CSV contains an empty source row".into());
        }
        let line = std::str::from_utf8(raw)
            .map_err(|error| format!("sealed test CSV is not UTF-8: {error}"))?;
        let columns = line.trim_end_matches('\r').split(',').collect::<Vec<_>>();
        if columns.len() != dataset.feature_count as usize + 1 {
            return Err(format!(
                "sealed test row {ordinal} has the wrong column count"
            ));
        }
        let target = columns
            .last()
            .unwrap()
            .trim()
            .parse::<f64>()
            .map_err(|error| format!("parse sealed target at row {ordinal}: {error}"))?;
        if !target.is_finite() {
            return Err(format!("sealed target at row {ordinal} is nonfinite"));
        }
        let stable_row_id = raw_sha256(
            [
                dataset.dataset_sha256.as_bytes(),
                b"sealed-test".as_slice(),
                &(ordinal as u64).to_le_bytes(),
                *raw,
            ]
            .concat()
            .as_slice(),
        );
        row_ids.push(stable_row_id.clone());
        targets.push(TargetRowV1 {
            stable_row_id,
            atomic_group: raw_sha256(b"sealed-test-row"),
            target,
        });
    }
    let table = SealedTestTableV1 {
        dataset_sha256: dataset.dataset_sha256.clone(),
        test_csv_path: test_path.to_string_lossy().into_owned(),
        test_sha256: dataset.test_sha256.clone(),
        test_row_ids: row_ids,
    };
    table.validate_metadata()?;
    Ok(Best5SealedTableV1 {
        table,
        target_rows: targets,
        test_artifact,
    })
}
