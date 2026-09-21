//! Local champion bank and a label-recording train step. Live S3 is fail-closed on this SHA.

use crate::admission::verify_inputs;
use crate::kernel::search_dataset;
use crate::schema::Catalog;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

const S3_CLOSED: &str = "live S3 champion pull/push is not wired on this SHA (no AWS credentials, no bucket allow-list). Use --local.";

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
    Ok(TrainStepReport {
        schema: "jope.train-step.v1",
        ok: true,
        env: "local",
        labels: bank.rows,
        note: "records champion genomes as router labels; does not update jope-policy weights or write S3",
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
