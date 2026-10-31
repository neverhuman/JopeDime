//! `jope.bench-report/v2`: one file per dataset with the task and its metrics, the plan, every
//! cut id, every method's screen and deep scores and timings, and where it was produced.
//!
//! v2 (from v1): `task` is one of the benchmarked tasks and `score_metric`, `ladder_metric` and
//! `metric_direction` name what the numbers are; every outcome carries `score` plus a `metrics`
//! map instead of fixed `rmse`/`r2` fields; deep results carry `fold_score`, `oof_score` and an
//! `oof` map, and a `reason` when rejected. No v1 report was ever committed.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cuts::{Cut, FoldPlan};
use crate::load::{sha256_hex, Dataset};
use crate::protocol::{MethodResult, Protocol};
use crate::task::Task;

/// Schema id.
pub const SCHEMA: &str = "jope.bench-report/v2";

/// A cut, without its row lists.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CutInfo {
    /// Held-out fold.
    pub fold: usize,
    /// Cut identity.
    pub cut_id: String,
    /// Training rows.
    pub train_rows: usize,
    /// Held-out rows.
    pub hold_rows: usize,
}

/// Where the report was produced.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Host {
    /// Logical cores available to the process.
    pub cores: usize,
    /// Rayon threads the benchmark ran with.
    pub workers: usize,
}

/// What the caller knows about the run that the data does not carry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stamp {
    /// Rayon threads the benchmark ran with.
    pub workers: usize,
    /// Git head of the tree that ran, when known.
    pub git_sha: Option<String>,
    /// Search budget, when the search ran.
    pub search: Option<crate::methods::SearchConfig>,
}

/// The report.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Report {
    /// [`SCHEMA`].
    pub schema: String,
    /// [`crate::EVALUATOR`].
    pub evaluator: String,
    /// Catalog id.
    pub dataset_id: String,
    /// Catalog hash.
    pub dataset_hash: String,
    /// The task ([`Task::as_str`]).
    pub task: String,
    /// The primary score every `score` / `oof_score` is ([`Task::score_metric`]).
    pub score_metric: String,
    /// The metric a ladder built from this report ranks by ([`crate::task::Task::ladder_metric`]).
    pub ladder_metric: String,
    /// `minimize` / `maximize`, for `ladder_metric`.
    pub metric_direction: String,
    /// SHA-256 of `train.csv`.
    pub train_sha256: String,
    /// Training rows.
    pub rows: usize,
    /// Feature columns.
    pub features: usize,
    /// Fold plan.
    pub plan: FoldPlan,
    /// Every cut.
    pub cuts: Vec<CutInfo>,
    /// Protocol parameters.
    pub protocol: Protocol,
    /// Search budget, when the search ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<crate::methods::SearchConfig>,
    /// Every method, screen-rank order.
    pub methods: Vec<MethodResult>,
    /// Host stamp.
    pub host: Host,
    /// Git head of the tree that ran, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    /// Unix seconds when the report was written.
    pub generated_at_unix: u64,
}

impl Report {
    /// Assemble a report.
    pub fn new(
        dataset: &Dataset,
        task: Task,
        plan: &FoldPlan,
        cuts: &[Cut],
        protocol: &Protocol,
        methods: Vec<MethodResult>,
        stamp: Stamp,
    ) -> Self {
        let (ladder_metric, direction) = task.ladder_metric();
        Self {
            schema: SCHEMA.to_owned(),
            evaluator: crate::EVALUATOR.to_owned(),
            dataset_id: dataset.row.id.clone(),
            dataset_hash: dataset.row.dataset_hash.clone(),
            task: task.as_str().to_owned(),
            score_metric: task.score_metric().to_owned(),
            ladder_metric: ladder_metric.to_owned(),
            metric_direction: direction.as_str().to_owned(),
            train_sha256: dataset.train_sha256.clone(),
            rows: dataset.x.len(),
            features: dataset.x.first().map(Vec::len).unwrap_or(0),
            plan: plan.clone(),
            cuts: cuts
                .iter()
                .map(|cut| CutInfo {
                    fold: cut.fold,
                    cut_id: cut.cut_id.clone(),
                    train_rows: cut.train.len(),
                    hold_rows: cut.hold.len(),
                })
                .collect(),
            protocol: protocol.clone(),
            search: stamp.search,
            methods,
            host: Host {
                cores: std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
                workers: stamp.workers,
            },
            git_sha: stamp.git_sha,
            generated_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// The report's file name under an output directory.
    pub fn file_name(&self) -> String {
        format!("{}.bench-report.json", self.dataset_hash)
    }

    /// Write pretty JSON plus a trailing newline; returns the file's SHA-256.
    pub fn write(&self, path: &Path) -> Result<String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, &text).with_context(|| format!("write {}", path.display()))?;
        Ok(sha256_hex(text.as_bytes()))
    }

    /// Read a report.
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let report: Self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        anyhow::ensure!(
            report.schema == SCHEMA,
            "{}: schema {} is not {SCHEMA}",
            path.display(),
            report.schema
        );
        Ok(report)
    }

    /// Every method that has a deep score, best first.
    pub fn deep_ranked(&self) -> Vec<&MethodResult> {
        let mut ranked: Vec<&MethodResult> = self
            .methods
            .iter()
            .filter(|m| m.deep_rank.is_some())
            .collect();
        ranked.sort_by_key(|m| m.deep_rank);
        ranked
    }
}
