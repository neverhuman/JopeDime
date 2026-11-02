use super::*;

/// Highest `attempts/<n>/` id present under a run dir, or `None` when the run
/// has not been staged yet. One cheap `read_dir` of a directory that holds a
/// handful of entries — used as the cache key for every resolved progress /
/// metrics path so an in-place SCQ requeue invalidates them.
pub(super) fn newest_attempt_id(run_dir: &Path) -> Option<u64> {
    fs::read_dir(run_dir.join("attempts"))
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
        })
        .max()
}

/// The `attempts/<k>/` id a path under `run_dir` belongs to. A path resolved
/// by `resolve_progress_file` may come from an older attempt (the newest one
/// has no progress file yet), so it may be cached against the newest attempt
/// only when this equals `newest_attempt_id`.
pub(super) fn attempt_of(run_dir: &Path, path: &Path) -> Option<u64> {
    path.strip_prefix(run_dir.join("attempts"))
        .ok()?
        .components()
        .next()?
        .as_os_str()
        .to_str()?
        .parse()
        .ok()
}

/// Identity of the metrics file a cached parse was read from. A parse is
/// reused only when the resolved path, its mtime and its length all still
/// match: mtimes come from the kernel's coarse clock (1 ms ticks), so a
/// requeued attempt's file can carry the dead attempt's exact mtime, and only
/// the path tells the two apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MetricsStamp {
    pub(super) path: PathBuf,
    pub(super) modified: SystemTime,
    pub(super) len: u64,
}

impl MetricsStamp {
    pub(super) fn of(path: &Path) -> io::Result<Self> {
        let metadata = fs::metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            modified: metadata.modified()?,
            len: metadata.len(),
        })
    }
}

pub(super) fn resolve_progress_file(run_dir: &Path) -> Option<PathBuf> {
    let attempts = run_dir.join("attempts");
    let mut attempt_ids = fs::read_dir(&attempts)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
        })
        .collect::<Vec<_>>();
    attempt_ids.sort_unstable_by(|left, right| right.cmp(left));
    for attempt in attempt_ids {
        let outputs = attempts.join(attempt.to_string()).join("outputs");
        let mut best: Option<(SystemTime, PathBuf)> = None;
        walk_for_progress(&outputs, 0, &mut best);
        if let Some((_, path)) = best {
            return Some(path);
        }
    }
    None
}

pub(super) fn walk_for_progress(
    dir: &Path,
    depth: usize,
    best: &mut Option<(SystemTime, PathBuf)>,
) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()).take(256) {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk_for_progress(&path, depth + 1, best);
        } else if file_type.is_file() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "train_progress.json" || name == "progress.json" {
                let modified = entry
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH);
                if best.as_ref().is_none_or(|(prior, _)| modified > *prior) {
                    *best = Some((modified, path));
                }
            }
        }
    }
}

pub(super) fn read_trainer_progress(path: &Path, detail: &mut TrainerDetail) {
    let Ok(bytes) = fs::read(path) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    detail.phase = string_at(&value, &["phase"]).unwrap_or("-").to_owned();
    detail.epoch = u64_at(&value, &["epoch"]).unwrap_or(0);
    detail.epochs = u64_at(&value, &["epochs"]).unwrap_or(0);
    detail.batch = u64_at(&value, &["batch"]).unwrap_or(0);
    detail.total_batches = u64_at(&value, &["total_batches"]).unwrap_or(0);
    detail.pct = f64_at(&value, &["pct"]).unwrap_or(0.0);
    detail.running_loss = f64_at(&value, &["running_loss"]);
    detail.val_loss = f64_at(&value, &["val_loss"]);
    detail.label_watermark = u64_at(&value, &["active_label_watermark"]).unwrap_or(0);
    detail.elapsed_sec = f64_at(&value, &["elapsed_sec"]).unwrap_or(0.0);
    detail.validation_batch = u64_at(&value, &["validation_batch"]).unwrap_or(0);
    detail.validation_batches = u64_at(&value, &["validation_batches"]).unwrap_or(0);
    detail.progress_at_ms = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
}

pub(super) fn parse_metric_row(row: &Value, fallback_epoch: u64) -> TrainerMetricPoint {
    TrainerMetricPoint {
        epoch: u64_at(row, &["epoch"]).unwrap_or(fallback_epoch),
        train_loss: f64_at(row, &["train_loss"]).unwrap_or(0.0),
        val_loss: f64_at(row, &["val_loss"]).unwrap_or(0.0),
        bad_passes: u64_at(row, &["macro_validation_bad_passes"]).unwrap_or(0),
        overfit_gap: f64_at(row, &["overfit_gap"]).unwrap_or(0.0),
        val_pairwise: f64_at(row, &["val_pairwise_win_accuracy"]),
        val_binary_pairwise: f64_at(row, &["val_binary_pairwise_win"])
            .or_else(|| f64_at(row, &["val_binary_pairwise_win_accuracy"])),
        val_regression_pairwise: f64_at(row, &["val_regression_pairwise_win"])
            .or_else(|| f64_at(row, &["val_regression_pairwise_win_accuracy"])),
        val_reward_cv: f64_at(row, &["val_reward_cv_correlation"]),
        val_top1_regret: f64_at(row, &["val_top1_historical_regret"]),
        val_conditional_advantage: f64_at(row, &["val_conditional_advantage"]),
        val_ablation_pairwise: f64_at(row, &["val_ablation_pairwise_win"]),
    }
}

/// Full-history read (call sites gate on mtime), downsampled for charting while
/// always preserving the newest point.
pub(super) fn read_metrics_history(path: &Path) -> Vec<TrainerMetricPoint> {
    read_metrics_points(path, DL_HISTORY_MAX)
}

pub(super) fn read_metrics_points(path: &Path, cap: usize) -> Vec<TrainerMetricPoint> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let points = text
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            serde_json::from_str::<Value>(line)
                .ok()
                .map(|row| parse_metric_row(&row, index as u64 + 1))
        })
        .collect::<Vec<_>>();
    downsample_points(points, cap)
}

pub(super) fn downsample_points(
    points: Vec<TrainerMetricPoint>,
    cap: usize,
) -> Vec<TrainerMetricPoint> {
    if points.len() <= cap || cap < 2 {
        return points;
    }
    let last = points.len() - 1;
    let stride = (points.len() as f64 / (cap - 1) as f64).ceil() as usize;
    let mut sampled = points
        .iter()
        .step_by(stride.max(1))
        .cloned()
        .collect::<Vec<_>>();
    if sampled.last().map(|point| point.epoch) != Some(points[last].epoch) {
        sampled.push(points[last].clone());
    }
    sampled
}
