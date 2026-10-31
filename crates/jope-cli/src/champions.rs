//! Local champion bank and a label-recording train step. Live S3 is fail-closed on this SHA.

use crate::admission::verified_datasets;
use crate::kernel::{search_validated_dataset, unsupported_regression_reason, SearchReport};
use crate::router::{dataset_embedding, fit_centroid_router, RouterArtifact};
use crate::schema::Dataset;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

const S3_CLOSED: &str = "live S3 champion pull/push is not implemented for this command; host credentials do not enable this route. Use --local.";

#[derive(Debug, Serialize, Clone)]
/// One admitted dataset's selected genome and validation score.
pub struct ChampionRow {
    pub dataset_id: String,
    pub task: String,
    pub scenario: String,
    pub genome: String,
    pub hold_r2: f64,
}

/// An admitted fixture that fit/search/route cannot use, and why. It is reported, never searched.
#[derive(Debug, Serialize, Clone)]
pub struct UnsupportedDataset {
    pub dataset_id: String,
    pub task: String,
    pub scenario: String,
    pub reason: String,
}

#[derive(Debug, Serialize)]
/// Local search results for every admitted fixture, including unsupported cases.
pub struct ChampionBank {
    pub schema: &'static str,
    pub ok: bool,
    pub env: &'static str,
    /// Every admitted dataset: `supported_count + unsupported_count`.
    pub dataset_count: usize,
    pub supported_count: usize,
    pub unsupported_count: usize,
    pub rows: Vec<ChampionRow>,
    pub unsupported: Vec<UnsupportedDataset>,
}

#[derive(Debug, Serialize)]
/// Local labels and centroid router fitted from the same admitted fixture snapshot.
pub struct TrainStepReport {
    pub schema: &'static str,
    pub ok: bool,
    pub env: &'static str,
    pub dataset_count: usize,
    pub supported_count: usize,
    pub unsupported_count: usize,
    pub labels: Vec<ChampionRow>,
    pub unsupported: Vec<UnsupportedDataset>,
    pub router: RouterArtifact,
    pub note: &'static str,
}

/// Return the current explicit refusal for unsupported live S3 champion operations.
pub fn refuse_s3() -> Result<ChampionBank> {
    bail!("{S3_CLOSED}")
}

/// Search the admitted snapshot once. No catalog or fixture path is reopened after admission.
pub fn local_champions(root: &Path) -> Result<ChampionBank> {
    let (_, datasets) =
        verified_datasets(root).context("admit fixture catalog before searching champions")?;
    champion_bank(&datasets, search_validated_dataset)
}

fn champion_bank(
    datasets: &[Dataset],
    evaluate: impl Fn(&Dataset) -> Result<SearchReport>,
) -> Result<ChampionBank> {
    let mut rows = Vec::new();
    let mut unsupported = Vec::new();
    for data in datasets {
        if let Some(reason) = unsupported_regression_reason(data) {
            unsupported.push(UnsupportedDataset {
                dataset_id: data.id.clone(),
                task: data.task.name().into(),
                scenario: data.scenario.clone(),
                reason,
            });
            continue;
        }
        // A supported fixture that fails evaluation is an error, never an unsupported row.
        let found = evaluate(data).with_context(|| format!("search {}", data.id))?;
        rows.push(ChampionRow {
            dataset_id: data.id.clone(),
            task: data.task.name().into(),
            scenario: data.scenario.clone(),
            genome: found.genome,
            hold_r2: found.hold_r2,
        });
    }
    ensure!(
        rows.len() + unsupported.len() == datasets.len(),
        "champion support counts do not cover the admitted datasets"
    );
    Ok(ChampionBank {
        schema: "jope.champions.v1",
        ok: true,
        env: "local",
        dataset_count: datasets.len(),
        supported_count: rows.len(),
        unsupported_count: unsupported.len(),
        rows,
        unsupported,
    })
}

/// Label and embed the same admitted snapshot; unsupported fixtures are excluded and reported.
pub fn local_train_step(root: &Path) -> Result<TrainStepReport> {
    let (_, datasets) =
        verified_datasets(root).context("admit fixture catalog before fitting local router")?;
    train_step(&datasets, search_validated_dataset)
}

fn train_step(
    datasets: &[Dataset],
    evaluate: impl Fn(&Dataset) -> Result<SearchReport>,
) -> Result<TrainStepReport> {
    let bank = champion_bank(datasets, evaluate)?;
    let mut examples = Vec::new();
    for row in &bank.rows {
        let data = datasets
            .iter()
            .find(|data| data.id == row.dataset_id)
            .with_context(|| format!("missing admitted dataset for {}", row.dataset_id))?;
        examples.push((dataset_embedding(data)?, row.genome.clone()));
    }
    let router = fit_centroid_router(&examples)?;
    ensure!(
        router.classes.iter().map(|class| class.n).sum::<usize>() == bank.supported_count,
        "router training count differs from the supported champion count"
    );
    Ok(TrainStepReport {
        schema: "jope.train-step.v1",
        ok: true,
        env: "local",
        dataset_count: bank.dataset_count,
        supported_count: bank.supported_count,
        unsupported_count: bank.unsupported_count,
        labels: bank.rows,
        unsupported: bank.unsupported,
        router,
        note: "fits a local 128-d centroid baseline on supported champion genomes; does not update jope-policy checkpoint weights or accepted policy weights, and does not write S3",
    })
}

/// Serialize `value` as pretty JSON, creating parent directories as needed.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_fixtures, generator::dataset, schema::Task};
    use std::collections::BTreeSet;

    #[test]
    fn champions_and_training_keep_the_admitted_snapshot_after_paths_change() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("fixtures");
        generate_fixtures(&root).unwrap();
        let (receipt, datasets) = verified_datasets(&root).unwrap();
        let bank = champion_bank(&datasets, search_validated_dataset).unwrap();
        let step = train_step(&datasets, search_validated_dataset).unwrap();

        // Replace the admitted root after admission: the public consumers must refuse the
        // unverified replacement, while the snapshot reproduces identical reports.
        fs::remove_dir_all(&root).unwrap();
        fs::create_dir(&root).unwrap();
        fs::write(root.join("catalog.json"), b"unverified replacement").unwrap();
        assert!(local_champions(&root).is_err());
        assert!(local_train_step(&root).is_err());
        assert_eq!(
            serde_json::to_value(champion_bank(&datasets, search_validated_dataset).unwrap())
                .unwrap(),
            serde_json::to_value(&bank).unwrap()
        );
        assert_eq!(
            serde_json::to_value(train_step(&datasets, search_validated_dataset).unwrap()).unwrap(),
            serde_json::to_value(&step).unwrap()
        );

        // Coverage is total and has no duplicates.
        assert_eq!(bank.dataset_count, receipt.dataset_count);
        assert_eq!(bank.supported_count, bank.rows.len());
        assert_eq!(bank.unsupported_count, bank.unsupported.len());
        assert_eq!(
            bank.dataset_count,
            bank.supported_count + bank.unsupported_count
        );
        assert!(bank.supported_count > 0 && bank.unsupported_count > 0);
        let mut covered = BTreeSet::new();
        for id in bank
            .rows
            .iter()
            .map(|row| &row.dataset_id)
            .chain(bank.unsupported.iter().map(|row| &row.dataset_id))
        {
            assert!(covered.insert(id), "duplicate coverage of {id}");
        }
        assert_eq!(covered, datasets.iter().map(|data| &data.id).collect());
        assert!(bank.rows.iter().all(|row| row.task == "regression"));
        assert_eq!(step.dataset_count, bank.dataset_count);
        assert_eq!(step.supported_count, step.labels.len());
        assert_eq!(step.unsupported_count, step.unsupported.len());
        assert_eq!(
            step.supported_count + step.unsupported_count,
            step.dataset_count
        );
        assert_eq!(
            step.router
                .classes
                .iter()
                .map(|class| class.n)
                .sum::<usize>(),
            step.supported_count
        );
    }

    #[test]
    fn supported_evaluation_failure_aborts_champions_and_training() {
        let failing = dataset(Task::Regression, 1);
        crate::admission::validate_dataset(&failing).unwrap();
        assert!(unsupported_regression_reason(&failing).is_none());
        let failing_id = failing.id.clone();
        let datasets = vec![
            dataset(Task::Regression, 0),
            dataset(Task::Binary, 0),
            failing,
        ];
        // A valid supported fixture fails after an earlier result and an exclusion exist.
        // Injecting the evaluator keeps this deterministic without a production flag or timer.
        let evaluate = |data: &Dataset| {
            if data.id == failing_id {
                bail!("injected evaluation failure");
            }
            search_validated_dataset(data)
        };
        for error in [
            champion_bank(&datasets, evaluate).unwrap_err(),
            train_step(&datasets, evaluate).unwrap_err(),
        ] {
            let message = format!("{error:#}");
            assert!(
                message.contains(&format!("search {failing_id}")),
                "{message}"
            );
            assert!(message.contains("injected evaluation failure"), "{message}");
        }
    }
}
