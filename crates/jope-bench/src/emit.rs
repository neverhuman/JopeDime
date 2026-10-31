//! Reports → a local `jope-arena` track: `scores.csv` rows under protocol `same-cut`, plus the
//! track and registry files the ladder code needs, so `jope-arena elo` ranks the benchmark with
//! the same Bradley–Terry fit that ranks the published one. Two tracks come out of one run:
//! `<track>-screen` (every method, one cut) and `<track>-deep` (the finalists, every cut). The
//! ladder metric and its direction follow the reports' task
//! ([`crate::task::Task::ladder_metric`]): RMSE (minimize) for regression, ROC AUC and macro-F1
//! (maximize) for binary and multiclass.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use jope_arena::pyfmt::py_repr;
use jope_arena::scores::{write_scores, ScoreRow};
use serde_json::json;

use crate::load::sha256_hex;
use crate::report::Report;
use crate::task::Task;

/// The protocol every emitted row carries.
pub const PROTOCOL: &str = "same-cut";

/// Which stage's number becomes the row value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Screen cut ladder metric, every method.
    Screen,
    /// Pooled out-of-fold ladder metric, finalists only; others are `missing`.
    Deep,
}

impl Stage {
    /// Track suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Screen => "screen",
            Self::Deep => "deep",
        }
    }
}

/// The task every report in the set carries (they must agree: one track per task).
pub fn common_task(reports: &[(PathBuf, Report)]) -> Result<Task> {
    ensure!(!reports.is_empty(), "no reports to emit");
    let task = Task::parse(&reports[0].1.task)
        .with_context(|| format!("report task {:?} is not benchmarked", reports[0].1.task))?;
    ensure!(
        reports.iter().all(|(_, r)| r.task == task.as_str()),
        "reports mix tasks; one track per task"
    );
    Ok(task)
}

/// One `scores.csv` row per (dataset, method).
pub fn score_rows(
    reports: &[(PathBuf, Report)],
    root: &Path,
    track_id: &str,
    stage: Stage,
) -> Result<Vec<ScoreRow>> {
    let task = common_task(reports)?;
    let (metric, _) = task.ladder_metric();
    let mut rows = Vec::new();
    for (path, report) in reports {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let source_sha256 = sha256_hex(&bytes);
        let source_uri = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let source_revision = format!(
            "seed={};k={};screen_fold={};top={}",
            report.plan.seed, report.plan.k, report.protocol.screen_fold, report.protocol.top
        );
        for method in &report.methods {
            let origin = match (&method.method.adapter, &method.method.native) {
                (Some(spec), _) => format!(
                    "external adapter {} ({}), controller-scored; ",
                    spec.name,
                    spec.version.as_deref().unwrap_or("version unknown")
                ),
                (_, Some(spec)) => format!(
                    "native regressor {} (source {}, seed {}), controller-scored; ",
                    spec.name, spec.source_sha256, spec.seed
                ),
                _ => String::new(),
            };
            let (value, status, notes) = match stage {
                Stage::Screen => match method.screen.metrics.get(metric) {
                    Some(v) => (
                        py_repr(*v),
                        "ok",
                        format!("screen cut {}", &method.screen_cut_id[..12]),
                    ),
                    None => (
                        String::new(),
                        "failed",
                        method
                            .screen
                            .reason
                            .clone()
                            .or_else(|| method.screen.note.clone())
                            .unwrap_or_default(),
                    ),
                },
                Stage::Deep => match method.deep.as_ref() {
                    Some(deep) => match deep.oof.get(metric) {
                        Some(v) => (
                            py_repr(*v),
                            "ok",
                            format!(
                                "pooled OOF over {} folds; fold {} sd {}",
                                deep.folds.len(),
                                task.score_metric(),
                                deep.fold_sd.map(py_repr).unwrap_or_default()
                            ),
                        ),
                        None => (
                            String::new(),
                            "failed",
                            deep.reason.clone().unwrap_or_default(),
                        ),
                    },
                    None => (String::new(), "missing", "not a screen finalist".to_owned()),
                },
            };
            let row = ScoreRow {
                track_id: track_id.to_owned(),
                dataset_id: report.dataset_hash.clone(),
                method_id: method.method.id.clone(),
                protocol: PROTOCOL.to_owned(),
                metric: metric.to_owned(),
                value,
                status: status.to_owned(),
                benchmark_valid: "True".to_owned(),
                source_kind: "bench-report".to_owned(),
                source_revision: source_revision.clone(),
                source_sha256: source_sha256.clone(),
                source_uri: source_uri.clone(),
                notes: format!("{origin}{notes}"),
            };
            row.validate()
                .with_context(|| format!("{}/{}", row.dataset_id, row.method_id))?;
            rows.push(row);
        }
    }
    rows.sort_by(|a, b| (&a.dataset_id, &a.method_id).cmp(&(&b.dataset_id, &b.method_id)));
    Ok(rows)
}

/// Write `tracks/<id>/track.json`, both registries and `results/<id>/scores.csv` under
/// `track_root`, for one stage. Returns the scores path.
pub fn write_track(
    track_root: &Path,
    track_base: &str,
    stage: Stage,
    reports: &[(PathBuf, Report)],
    reports_root: &Path,
) -> Result<PathBuf> {
    let task = common_task(reports)?;
    let (metric, direction) = task.ladder_metric();
    let track_id = format!("{track_base}-{}", stage.suffix());
    let rows = score_rows(reports, reports_root, &track_id, stage)?;

    let tracks_dir = track_root.join("tracks").join(&track_id);
    let registry_dir = track_root.join("registry");
    let results_dir = track_root.join("results").join(&track_id);
    for dir in [&tracks_dir, &registry_dir, &results_dir] {
        std::fs::create_dir_all(dir)?;
    }
    let dataset_registry = format!("registry/datasets.{track_id}.json");
    let track = json!({
        "schema": "sotarena.track.v1",
        "id": track_id,
        "name": format!("jope same-cut ({}) · {}", stage.suffix(), task.as_str()),
        "public_name": "same-cut",
        "task": task.as_str(),
        "primary_metric": metric,
        "metric_direction": direction.as_str(),
        "dataset_registry": dataset_registry,
        "method_registry": "registry/methods.json",
        "scores": format!("results/{track_id}/scores.csv"),
        "protocols": [PROTOCOL],
        "split": {"family": "same-cut", "excluded_datasets": [], "restricted_dataset_count": 0, "test_labels_public": true},
        "leaderboard_policy": {
            "dataset_weighting": "uniform",
            "pairwise_tie_relative_tolerance": 1e-12,
            "elo_method": "regularized-bradley-terry-elo-scale",
            "protocols_ranked_together": true,
            "official_output": format!("results/{track_id}/leaderboard.csv"),
            "partial_coverage_is_not_full_rank": true
        },
        "stage": stage.suffix(),
        "evaluator": crate::EVALUATOR,
    });
    write_json(&tracks_dir.join("track.json"), &track)?;

    let mut method_ids: Vec<&str> = rows.iter().map(|r| r.method_id.as_str()).collect();
    method_ids.sort_unstable();
    method_ids.dedup();
    let methods = json!({
        "schema": "sotarena.method-registry.v1",
        "methods": method_ids.iter().map(|id| json!({"id": id, "name": id})).collect::<Vec<_>>(),
    });
    write_json(&registry_dir.join("methods.json"), &methods)?;

    let datasets = json!({
        "schema": "sotarena.dataset-registry.v1",
        "datasets": reports.iter().map(|(_, r)| json!({
            "id": r.dataset_hash,
            "name": r.dataset_id,
            "task": r.task,
            "rows": r.rows,
            "features": r.features,
            "train_sha256": r.train_sha256,
        })).collect::<Vec<_>>(),
    });
    write_json(&track_root.join(&dataset_registry), &datasets)?;

    let scores_path = results_dir.join("scores.csv");
    std::fs::write(&scores_path, write_scores(&rows)?)
        .with_context(|| format!("write {}", scores_path.display()))?;
    Ok(scores_path)
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}
