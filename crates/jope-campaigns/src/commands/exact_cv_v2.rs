//! Hidden command adapter for the additive one-fold V2 proof evaluator.

use crate::data::read_regression_csv_bytes;
use crate::evaluation_v2::{
    evaluate_regression_fold_v2, validate_sha256, EvaluationOutcomeV2, EvaluatorIdentityV1,
    ExplicitFoldV1, ObjectiveIdentityV1, RunContractV2, TrainOnlyRegressionView,
};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const MANIFEST_SCHEMA: &str = "JopeDimeEvaluationManifest.v2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationManifestV2 {
    schema: String,
    dataset: String,
    train_sha256: String,
    stable_row_ids: Vec<String>,
    canonical_genome: String,
    candidate_sha256: String,
    fold: ExplicitFoldRequestV1,
    objective_policy: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExplicitFoldRequestV1 {
    fold_id: u32,
    train_rows: Vec<usize>,
    validation_rows: Vec<usize>,
}

pub(crate) fn verify_manifest(
    manifest_path: &Path,
    registry: &Path,
    requested_dataset: Option<&str>,
) -> Result<()> {
    let outcome = evaluate_manifest(manifest_path, registry, requested_dataset)?;
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &outcome.semantic)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    let mut stderr = io::stderr().lock();
    serde_json::to_writer(&mut stderr, &outcome.observation)?;
    stderr.write_all(b"\n")?;
    stderr.flush()?;
    Ok(())
}

fn evaluate_manifest(
    manifest_path: &Path,
    registry: &Path,
    requested_dataset: Option<&str>,
) -> Result<EvaluationOutcomeV2> {
    let manifest_bytes =
        fs::read(manifest_path).context("read authorized V2 evaluation manifest")?;
    let manifest: EvaluationManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse V2 evaluation manifest")?;
    if manifest.schema != MANIFEST_SCHEMA {
        bail!(
            "evaluation manifest must be {MANIFEST_SCHEMA}, found {}",
            manifest.schema
        );
    }
    validate_dataset_id(&manifest.dataset)?;
    if requested_dataset.is_some_and(|requested| requested != manifest.dataset) {
        bail!("global dataset selection differs from the evaluation manifest");
    }
    validate_sha256("train CSV", &manifest.train_sha256)?;
    validate_sha256("candidate", &manifest.candidate_sha256)?;

    // The manifest cannot select a path. The caller-authorized registry and
    // opaque dataset ID are the sole authority for the training role.
    let train_path = registry.join(&manifest.dataset).join("train.csv");
    let train_bytes = fs::read(&train_path).context("read authorized training CSV")?;
    let actual_train_sha256 = hex::encode(Sha256::digest(&train_bytes));
    if actual_train_sha256 != manifest.train_sha256 {
        bail!("manifested train CSV differs from its pinned SHA-256");
    }
    let data = read_regression_csv_bytes(&train_bytes, "authorized train.csv")?;
    let view =
        TrainOnlyRegressionView::new(&data.x, &data.y, &manifest.stable_row_ids, data.features)?;
    let actual_candidate_sha256 = hex::encode(Sha256::digest(manifest.canonical_genome.as_bytes()));
    if actual_candidate_sha256 != manifest.candidate_sha256 {
        bail!("manifested candidate differs from its pinned semantic SHA-256");
    }
    let fold = ExplicitFoldV1::new(
        manifest.fold.fold_id,
        manifest.fold.train_rows,
        manifest.fold.validation_rows,
        &view,
    )?;
    let evaluator = EvaluatorIdentityV1::current();
    let objective = ObjectiveIdentityV1::from_policy(&manifest.objective_policy)?;
    let contract =
        RunContractV2::strict_legacy(view.feature_view_sha256().to_string(), evaluator, objective)?;
    evaluate_regression_fold_v2(&view, &contract, &fold, &manifest.canonical_genome)
}

fn validate_dataset_id(dataset: &str) -> Result<()> {
    if dataset.is_empty()
        || dataset.len() > 128
        || !dataset
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("dataset ID must be an opaque 1..=128 character ASCII identifier");
    }
    Ok(())
}

pub(crate) fn legacy_robust_fitness(r2: f64, folds: &[f64]) -> f64 {
    super::exact_cv::robust_fitness(r2, folds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn manifested_evaluation_is_path_free_and_repeatable() {
        let root = std::env::temp_dir().join(format!(
            "jope-dime-evaluation-v2-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let dataset = "fixture";
        let train_root = root.join(dataset);
        fs::create_dir_all(&train_root).unwrap();
        let train_path = train_root.join("train.csv");
        let mut train = String::new();
        for row in 0..12 {
            let a = row as f64 / 12.0;
            train.push_str(&format!("{a},{},{}\n", 1.0 - a, 2.0 * a - 0.25));
        }
        fs::write(&train_path, train.as_bytes()).unwrap();
        let train_sha = hex::encode(Sha256::digest(train.as_bytes()));
        let candidate = "(pipe X (ridge))";
        let manifest = json!({
            "schema": MANIFEST_SCHEMA,
            "dataset": dataset,
            "train_sha256": train_sha,
            "stable_row_ids": (0..12).map(|row| format!("row-{row:02}")).collect::<Vec<_>>(),
            "canonical_genome": candidate,
            "candidate_sha256": hex::encode(Sha256::digest(candidate.as_bytes())),
            "fold": {
                "fold_id": 0,
                "train_rows": (0..8).collect::<Vec<_>>(),
                "validation_rows": (8..12).collect::<Vec<_>>()
            },
            "objective_policy": "legacy-robust-r2-v1"
        });
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let first = evaluate_manifest(&manifest_path, &root, Some(dataset)).unwrap();
        let second = evaluate_manifest(&manifest_path, &root, Some(dataset)).unwrap();
        assert_eq!(first.semantic, second.semantic);
        let semantic_json = serde_json::to_string(&first.semantic).unwrap();
        assert!(!semantic_json.contains(root.to_str().unwrap()));
        assert!(!semantic_json.contains("train.csv"));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dataset_authority_rejects_path_selection() {
        assert!(validate_dataset_id("../test.csv").is_err());
        assert!(validate_dataset_id("a/b").is_err());
        assert!(validate_dataset_id("fixture_01").is_ok());
    }
}
