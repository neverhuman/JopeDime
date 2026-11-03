//! The benchmark clone's official ladder: `results/<track>[/<version>]/scores.csv` →
//! `leaderboard.csv`, byte-for-byte what `sotarena/leaderboard.py` wrote.
//!
//! A track is `tracks/<track>/track.json`; its `dataset_registry` counts the datasets and
//! `registry/methods.json` names the methods. Results live either directly in
//! `results/<track>/` (the upstream layout) or in `results/<track>/<version>/` (what
//! [`crate::publish`] writes); `--version` selects one, otherwise the newest version directory,
//! otherwise the flat layout.
//!
//! Two ladders are computed from the same rows ([`Ladder`]):
//!
//! * **published** — every rankable row, exactly as the benchmark ranked them. This is what the
//!   benchmark published and what the regression fixture reproduces byte-for-byte.
//! * **benchmark-valid** — only rows whose `benchmark_valid` is `True` after an optional
//!   [`Corrections`] file has been applied. This is the held-out claim; a row whose source run
//!   this repository's own campaign code labels test-adaptive never enters it.
//!
//! Either way the `benchmark_valid` cell of a ladder row is computed from the rows that entered
//! the fit (`True` only when every one of the method's ranked rows is valid), never assumed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::corrections::{Corrections, CorrectionsMeta};
use crate::elo::Direction;
use crate::io::{read_bytes, read_json, read_text, resolve_input, sha256_hex};
use crate::scores::parse_scores;

/// `tracks/<track>/track.json` (the fields the ladder needs; the rest is kept as `extra`).
#[derive(Clone, Debug, Deserialize)]
pub struct Track {
    /// `sotarena.track.v1`.
    pub schema: String,
    /// Track id.
    pub id: String,
    /// Task id.
    pub task: String,
    /// Metric id.
    pub primary_metric: String,
    /// `minimize` / `maximize`.
    pub metric_direction: String,
    /// Registry path relative to the root.
    pub dataset_registry: String,
    /// Method registry path relative to the root.
    #[serde(default = "default_method_registry")]
    pub method_registry: String,
    /// Scores path relative to the root (upstream layout).
    pub scores: String,
    /// Ledger path relative to the root (upstream layout).
    #[serde(default)]
    pub resolution_ledger: Option<String>,
    /// Everything else.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn default_method_registry() -> String {
    "registry/methods.json".to_owned()
}

/// Which rows enter the fit.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Ladder {
    /// Every rankable row, as the benchmark ranked them.
    Published,
    /// Only rows whose `benchmark_valid` is `True` (after corrections).
    BenchmarkValid,
}

impl Ladder {
    /// The spelling used in snapshots and on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::BenchmarkValid => "benchmark-valid",
        }
    }

    /// The `scope` cell every row of this ladder carries.
    pub fn scope(self) -> &'static str {
        match self {
            Self::Published => "official unified benchmark comparison",
            Self::BenchmarkValid => "benchmark-valid rows only; protocol corrections applied",
        }
    }
}

/// Where one results snapshot lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultsLocation {
    /// The directory holding `scores.csv`.
    pub dir: PathBuf,
    /// The version label (`None` for the upstream flat layout).
    pub version: Option<String>,
}

/// Read `tracks/<track>/track.json`.
pub fn load_track(root: &Path, track_id: &str) -> Result<Track> {
    let path = root.join("tracks").join(track_id).join("track.json");
    let track: Track =
        read_json(&path).with_context(|| format!("track {track_id} under {}", root.display()))?;
    ensure!(
        track.schema == "sotarena.track.v1",
        "{}: unexpected schema {}",
        path.display(),
        track.schema
    );
    ensure!(
        track.id == track_id,
        "{}: track id {} does not match directory",
        path.display(),
        track.id
    );
    Ok(track)
}

/// Version directories under `results/<track>/` that hold a `scores.csv`, sorted.
pub fn list_versions(root: &Path, track_id: &str) -> Result<Vec<String>> {
    let dir = root.join("results").join(track_id);
    let mut versions = Vec::new();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && resolve_input(&entry.path().join("scores.csv")).is_some()
            {
                versions.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
    }
    versions.sort();
    Ok(versions)
}

/// Resolve the results directory for `track_id` and an optional version.
pub fn resolve_results(
    root: &Path,
    track_id: &str,
    version: Option<&str>,
) -> Result<ResultsLocation> {
    let base = root.join("results").join(track_id);
    if let Some(version) = version {
        let dir = base.join(version);
        ensure!(
            resolve_input(&dir.join("scores.csv")).is_some(),
            "no scores.csv under {} (versions present: {:?})",
            dir.display(),
            list_versions(root, track_id)?
        );
        return Ok(ResultsLocation {
            dir,
            version: Some(version.to_owned()),
        });
    }
    if let Some(latest) = list_versions(root, track_id)?.pop() {
        return Ok(ResultsLocation {
            dir: base.join(&latest),
            version: Some(latest),
        });
    }
    ensure!(
        resolve_input(&base.join("scores.csv")).is_some(),
        "no results for track {track_id} under {} (neither {}/scores.csv nor a version directory)",
        root.display(),
        base.display()
    );
    Ok(ResultsLocation {
        dir: base,
        version: None,
    })
}

/// One `leaderboard.csv` row, with every cell already spelled the way Python wrote it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LeaderboardRow {
    /// 1-based rank by rating.
    pub elo_rank: usize,
    /// Rank among full-coverage methods, or empty.
    pub full_coverage_rank: String,
    /// Method id.
    pub method_id: String,
    /// Display name from the method registry (falls back to the id).
    pub method: String,
    /// `round(rating, 1)`.
    pub elo: String,
    /// `round(mean rank, 3)`.
    pub average_rank: String,
    /// Datasets where the method is strictly first (ties broken by method id).
    pub dataset_wins: usize,
    /// Datasets scored.
    pub datasets_scored: usize,
    /// Datasets in the registry.
    pub datasets_total: usize,
    /// `round(100 · scored / total, 1)`.
    pub coverage_pct: String,
    /// `True` / `False`.
    pub full_coverage: String,
    /// Unresolved ledger cells for the method.
    pub pending_cells: usize,
    /// `scored/total[ · pending pending]`.
    pub coverage_status: String,
    /// `|`-joined sorted protocols.
    pub protocols: String,
    /// `True` when every ranked row of the method is `benchmark_valid`, else `False`. The
    /// benchmark's own `leaderboard.py` wrote `True` unconditionally; the published fixture is
    /// all-`True` so the two agree on it.
    pub benchmark_valid: String,
    /// [`Ladder::scope`].
    pub scope: String,
}

/// A recomputed ladder plus the inputs it was built from.
#[derive(Clone, Debug)]
pub struct Leaderboard {
    /// Track id.
    pub track_id: String,
    /// Results version label, if versioned.
    pub version: Option<String>,
    /// Which rows entered the fit.
    pub ladder: Ladder,
    /// SHA-256 of the `scores.csv` bytes the rows were read from.
    pub scores_sha256: String,
    /// The corrections applied before ranking, if any.
    pub corrections: Option<CorrectionsMeta>,
    /// Rankable rows left out because they were not `benchmark_valid` (always 0 when
    /// published).
    pub excluded_rows: usize,
    /// Rows in rank order.
    pub rows: Vec<LeaderboardRow>,
    /// Unrounded ratings by method id.
    pub ratings: BTreeMap<String, f64>,
    /// Rankable score rows that entered the fit.
    pub rankable_rows: usize,
    /// Datasets with at least one rankable row.
    pub datasets_scored: usize,
}

/// `_pending_by_method` from `leaderboard.py`: unresolved ledger cells per method.
pub fn pending_by_method(ledger_text: &str) -> Result<BTreeMap<String, usize>> {
    let mut pending: BTreeMap<String, usize> = BTreeMap::new();
    for (index, line) in ledger_text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cell: Value = serde_json::from_str(line)
            .with_context(|| format!("resolution ledger line {}", index + 1))?;
        let resolved = cell["resolved"].as_bool().with_context(|| {
            format!(
                "resolution ledger line {}: no boolean `resolved`",
                index + 1
            )
        })?;
        if !resolved {
            let method = cell["method_id"]
                .as_str()
                .with_context(|| format!("resolution ledger line {}: no method_id", index + 1))?;
            *pending.entry(method.to_owned()).or_default() += 1;
        }
    }
    Ok(pending)
}

/// Inputs of [`build_leaderboard`] beyond the score rows.
#[derive(Clone, Debug)]
pub struct Inputs<'a> {
    /// Track id every row must carry.
    pub track_id: &'a str,
    /// Results version label.
    pub version: Option<&'a str>,
    /// Which rows enter the fit.
    pub ladder: Ladder,
    /// SHA-256 of the `scores.csv` bytes.
    pub scores_sha256: &'a str,
    /// Corrections already applied to the rows, for the record.
    pub corrections: Option<CorrectionsMeta>,
    /// Method id → display name.
    pub method_names: &'a BTreeMap<String, String>,
    /// Method id → unresolved ledger cells.
    pub pending: &'a BTreeMap<String, usize>,
    /// Datasets in the registry.
    pub total_datasets: usize,
    /// Whether lower or higher values win (the track's `metric_direction`).
    pub direction: Direction,
}

/// `build_leaderboard` from `leaderboard.py`, over already-loaded inputs.
pub use build::build_leaderboard;

/// Method id → display name from `registry/methods.json`.
pub fn load_method_names(root: &Path, track: &Track) -> Result<BTreeMap<String, String>> {
    let registry: Value = read_json(&root.join(&track.method_registry))?;
    let methods = registry["methods"]
        .as_array()
        .with_context(|| format!("{}: no `methods` array", track.method_registry))?;
    let mut names = BTreeMap::new();
    for method in methods {
        let id = method["id"].as_str().context("method without id")?;
        let name = method["name"].as_str().unwrap_or(id);
        names.insert(id.to_owned(), name.to_owned());
    }
    Ok(names)
}

/// Dataset ids from the track's registry.
pub fn load_dataset_ids(root: &Path, track: &Track) -> Result<Vec<String>> {
    let registry: Value = read_json(&root.join(&track.dataset_registry))?;
    let datasets = registry["datasets"]
        .as_array()
        .with_context(|| format!("{}: no `datasets` array", track.dataset_registry))?;
    datasets
        .iter()
        .map(|dataset| {
            dataset["id"]
                .as_str()
                .map(str::to_owned)
                .context("dataset without id")
        })
        .collect()
}

/// Recompute a ladder for `track_id` under `root`, applying `corrections` (a
/// `jope.arena-protocol-corrections/v1` file) to the rows first when given.
pub fn compute(
    root: &Path,
    track_id: &str,
    version: Option<&str>,
    ladder: Ladder,
    corrections: Option<&Path>,
) -> Result<(Leaderboard, ResultsLocation)> {
    let track = load_track(root, track_id)?;
    let location = resolve_results(root, track_id, version)?;
    let scores_path = location.dir.join("scores.csv");
    let scores_bytes = read_bytes(&scores_path)?;
    let scores_sha256 = sha256_hex(&scores_bytes);
    let mut scores = parse_scores(
        std::str::from_utf8(&scores_bytes)
            .with_context(|| format!("{} is not UTF-8", scores_path.display()))?,
    )?;
    let corrections = match corrections {
        Some(path) => {
            let (corrections, sha256) = Corrections::load(path)?;
            let (applied, rows_corrected) = corrections
                .apply(track_id, &scores_sha256, &mut scores)
                .with_context(|| format!("apply {}", path.display()))?;
            Some(CorrectionsMeta {
                file: path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                sha256,
                applied,
                rows_corrected,
            })
        }
        None => None,
    };
    let method_names = load_method_names(root, &track)?;
    let dataset_ids = load_dataset_ids(root, &track)?;
    let ledger_path = location.dir.join("resolution-ledger.jsonl");
    let pending = if resolve_input(&ledger_path).is_some() {
        pending_by_method(&read_text(&ledger_path)?)?
    } else if track.resolution_ledger.is_some() && location.version.is_none() {
        let path = root.join(track.resolution_ledger.as_deref().unwrap_or_default());
        if resolve_input(&path).is_some() {
            pending_by_method(&read_text(&path)?)?
        } else {
            BTreeMap::new()
        }
    } else {
        BTreeMap::new()
    };
    let direction = match track.metric_direction.as_str() {
        "minimize" => Direction::Minimize,
        "maximize" => Direction::Maximize,
        other => bail!(
            "track {track_id} ranks by {other:?}; metric_direction must be minimize or maximize"
        ),
    };
    let leaderboard = build_leaderboard(
        &scores,
        Inputs {
            track_id,
            version: location.version.as_deref(),
            ladder,
            scores_sha256: &scores_sha256,
            corrections,
            method_names: &method_names,
            pending: &pending,
            total_datasets: dataset_ids.len(),
            direction,
        },
    )?;
    Ok((leaderboard, location))
}

/// `write_leaderboard` from `leaderboard.py`: the CSV text.
pub fn to_csv(rows: &[LeaderboardRow]) -> Result<String> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new());
    for row in rows {
        writer.serialize(row)?;
    }
    Ok(String::from_utf8(writer.into_inner()?)?)
}

/// Parse a `leaderboard.csv`.
pub fn parse_csv(text: &str) -> Result<Vec<LeaderboardRow>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(text.as_bytes());
    let mut rows = Vec::new();
    for (index, record) in reader.deserialize::<LeaderboardRow>().enumerate() {
        rows.push(record.with_context(|| format!("leaderboard.csv row {}", index + 2))?);
    }
    Ok(rows)
}

/// Schema id of [`Snapshot`]. v2 added `ladder`, `corrections` and `excluded_rows`.
pub const SNAPSHOT_SCHEMA: &str = "jope.arena-elo-snapshot/v2";

/// The JSON snapshot committed as `assets/regression/elo-<track>[.<ladder>].json`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    /// [`SNAPSHOT_SCHEMA`].
    pub schema: String,
    /// Track id.
    pub track_id: String,
    /// Results version label.
    pub version: Option<String>,
    /// Which rows entered the fit.
    pub ladder: Ladder,
    /// SHA-256 of the `scores.csv` bytes the ladder was computed from.
    pub scores_sha256: String,
    /// The corrections applied before ranking, if any.
    pub corrections: Option<CorrectionsMeta>,
    /// Rankable rows left out as not `benchmark_valid`.
    pub excluded_rows: usize,
    /// SHA-256 of the `leaderboard.csv` text produced.
    pub leaderboard_sha256: String,
    /// Rankable rows.
    pub rankable_rows: usize,
    /// Datasets scored.
    pub datasets_scored: usize,
    /// Methods rated.
    pub methods: usize,
    /// The ladder.
    pub rows: Vec<LeaderboardRow>,
}

impl Snapshot {
    /// Build from a computed ladder.
    pub fn new(leaderboard: &Leaderboard) -> Result<Self> {
        let csv = to_csv(&leaderboard.rows)?;
        Ok(Self {
            schema: SNAPSHOT_SCHEMA.to_owned(),
            track_id: leaderboard.track_id.clone(),
            version: leaderboard.version.clone(),
            ladder: leaderboard.ladder,
            scores_sha256: leaderboard.scores_sha256.clone(),
            corrections: leaderboard.corrections.clone(),
            excluded_rows: leaderboard.excluded_rows,
            leaderboard_sha256: sha256_hex(csv.as_bytes()),
            rankable_rows: leaderboard.rankable_rows,
            datasets_scored: leaderboard.datasets_scored,
            methods: leaderboard.rows.len(),
            rows: leaderboard.rows.clone(),
        })
    }
}

mod build;

#[cfg(test)]
mod tests;
