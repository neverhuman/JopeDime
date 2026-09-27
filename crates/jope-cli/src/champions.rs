//! Local champion bank and a label-recording train step. Live S3 is fail-closed on this SHA.

use crate::admission::verify_inputs;
use crate::kernel::{read_dataset, search_dataset};
use crate::router::{dataset_embedding, fit_centroid_router, RouterArtifact};
use crate::schema::Catalog;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

const S3_CLOSED: &str = "live S3 champion pull/push is not implemented for this command; host credentials do not enable this route. Use --local.";

#[derive(Debug, Serialize, Clone)]
pub struct ChampionRow {
    pub dataset_id: String,
    pub task: String,
    pub scenario: String,
    pub genome: String,
    pub hold_r2: f64,
}

#[derive(Debug, Serialize)]
pub struct ChampionBank {
    pub schema: &'static str,
    pub ok: bool,
    pub env: &'static str,
    pub dataset_count: usize,
    pub rows: Vec<ChampionRow>,
}

#[derive(Debug, Serialize)]
pub struct TrainStepReport {
    pub schema: &'static str,
    pub ok: bool,
    pub env: &'static str,
    pub labels: Vec<ChampionRow>,
    pub router: RouterArtifact,
    pub note: &'static str,
}

pub fn refuse_s3() -> Result<ChampionBank> {
    bail!("{S3_CLOSED}")
}

pub fn local_champions(root: &Path) -> Result<ChampionBank> {
    verify_inputs(root).context("admit fixture catalog before searching champions")?;
    let catalog: Catalog = serde_json::from_slice(&fs::read(root.join("catalog.json"))?)
        .context("parse catalog.json")?;
    let mut rows = Vec::new();
    for entry in catalog.datasets {
        let found = search_dataset(&root.join(&entry.path))
            .with_context(|| format!("search {}", entry.id))?;
        rows.push(ChampionRow {
            dataset_id: entry.id,
            task: entry.task.name().into(),
            scenario: entry.scenario,
            genome: found.genome,
            hold_r2: found.hold_r2,
        });
    }
    Ok(ChampionBank {
        schema: "jope.champions.v1",
        ok: true,
        env: "local",
        dataset_count: rows.len(),
        rows,
    })
}

pub fn local_train_step(root: &Path) -> Result<TrainStepReport> {
    let bank = local_champions(root)?;
    let catalog: Catalog = serde_json::from_slice(&fs::read(root.join("catalog.json"))?)
        .context("parse catalog.json")?;
    let mut examples = Vec::new();
    for entry in &catalog.datasets {
        let data = read_dataset(&root.join(&entry.path))?;
        let embedding = dataset_embedding(&data)?;
        let genome = bank
            .rows
            .iter()
            .find(|row| row.dataset_id == entry.id)
            .map(|row| row.genome.clone())
            .with_context(|| format!("missing champion for {}", entry.id))?;
        examples.push((embedding, genome));
    }
    let router = fit_centroid_router(&examples)?;
    Ok(TrainStepReport {
        schema: "jope.train-step.v1",
        ok: true,
        env: "local",
        labels: bank.rows,
        router,
        note: "fits a 128-d centroid router on champion genomes; does not update jope-policy checkpoint weights or write S3",
    })
}

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
