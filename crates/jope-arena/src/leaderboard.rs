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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::corrections::{Corrections, CorrectionsMeta};
use crate::elo::{fit_elo, matches_from_scores, TIE_RELATIVE_TOLERANCE};
use crate::io::{read_bytes, read_json, read_text, resolve_input, sha256_hex};
use crate::pyfmt::{py_bool, py_repr, py_round};
use crate::scores::{parse_scores, ScoreRow};

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
}

/// `build_leaderboard` from `leaderboard.py`, over already-loaded inputs.
pub fn build_leaderboard(scores: &[ScoreRow], inputs: Inputs<'_>) -> Result<Leaderboard> {
    let track_id = inputs.track_id;
    let total_datasets = inputs.total_datasets;
    let (method_names, pending) = (inputs.method_names, inputs.pending);
    let mut by_dataset: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut protocols_by_method: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut valid_by_method: BTreeMap<String, bool> = BTreeMap::new();
    let mut rankable_rows = 0;
    let mut excluded_rows = 0;
    for row in scores {
        if !row.is_rankable() {
            continue;
        }
        ensure!(
            row.track_id == track_id,
            "score row for {} inside track {track_id}",
            row.track_id
        );
        let valid = match row.benchmark_valid.as_str() {
            "True" => true,
            "False" => false,
            other => bail!(
                "benchmark_valid {other:?} for {}/{} is neither True nor False",
                row.dataset_id,
                row.method_id
            ),
        };
        if inputs.ladder == Ladder::BenchmarkValid && !valid {
            excluded_rows += 1;
            continue;
        }
        let value: f64 = row.value.parse().with_context(|| {
            format!(
                "non-numeric value {:?} for {}/{}",
                row.value, row.dataset_id, row.method_id
            )
        })?;
        by_dataset
            .entry(row.dataset_id.clone())
            .or_default()
            .insert(row.method_id.clone(), value);
        protocols_by_method
            .entry(row.method_id.clone())
            .or_default()
            .insert(row.protocol.clone());
        *valid_by_method.entry(row.method_id.clone()).or_insert(true) &= valid;
        rankable_rows += 1;
    }
    let matches = matches_from_scores(&by_dataset, TIE_RELATIVE_TOLERANCE);
    let ratings = fit_elo(&matches);

    let mut ranks: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut wins: BTreeMap<&str, usize> = BTreeMap::new();
    for dataset_scores in by_dataset.values() {
        let mut ordered: Vec<(&str, f64)> = dataset_scores
            .iter()
            .map(|(m, v)| (m.as_str(), *v))
            .collect();
        // Python: sorted by (value, method_id); values are finite so total_cmp is the same order.
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        for (rank, (method, _)) in ordered.iter().enumerate() {
            ranks.entry(method).or_default().push((rank + 1) as f64);
        }
        if let Some((winner, _)) = ordered.first() {
            *wins.entry(winner).or_default() += 1;
        }
    }

    let mut order: Vec<(&String, &f64)> = ratings.iter().collect();
    order.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let mut rows = Vec::with_capacity(order.len());
    let mut full_coverage_rank = 0;
    for (elo_rank, (method_id, rating)) in order.into_iter().enumerate() {
        let method_ranks = ranks
            .get(method_id.as_str())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let scored = method_ranks.len();
        let full_coverage = scored == total_datasets;
        let pending_cells = pending.get(method_id).copied().unwrap_or(0);
        if full_coverage {
            full_coverage_rank += 1;
        }
        let mut rank_sum = 0.0;
        for rank in method_ranks {
            rank_sum += rank;
        }
        rows.push(LeaderboardRow {
            elo_rank: elo_rank + 1,
            full_coverage_rank: if full_coverage {
                full_coverage_rank.to_string()
            } else {
                String::new()
            },
            method_id: method_id.clone(),
            method: method_names
                .get(method_id)
                .cloned()
                .unwrap_or_else(|| method_id.clone()),
            elo: py_repr(py_round(*rating, 1)),
            average_rank: py_repr(py_round(rank_sum / scored as f64, 3)),
            dataset_wins: wins.get(method_id.as_str()).copied().unwrap_or(0),
            datasets_scored: scored,
            datasets_total: total_datasets,
            coverage_pct: py_repr(py_round(100.0 * scored as f64 / total_datasets as f64, 1)),
            full_coverage: py_bool(full_coverage).to_owned(),
            pending_cells,
            coverage_status: if pending_cells > 0 {
                format!("{scored}/{total_datasets} · {pending_cells} pending")
            } else {
                format!("{scored}/{total_datasets}")
            },
            protocols: protocols_by_method
                .get(method_id)
                .map(|set| set.iter().cloned().collect::<Vec<_>>().join("|"))
                .unwrap_or_default(),
            benchmark_valid: py_bool(valid_by_method.get(method_id).copied().unwrap_or(false))
                .to_owned(),
            scope: inputs.ladder.scope().to_owned(),
        });
    }
    Ok(Leaderboard {
        track_id: track_id.to_owned(),
        version: inputs.version.map(str::to_owned),
        ladder: inputs.ladder,
        scores_sha256: inputs.scores_sha256.to_owned(),
        corrections: inputs.corrections,
        excluded_rows,
        rows,
        ratings,
        rankable_rows,
        datasets_scored: by_dataset.len(),
    })
}

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
    if track.metric_direction != "minimize" {
        bail!(
            "track {track_id} ranks by {} but the official ladder only knows minimize (lower wins)",
            track.metric_direction
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scores::parse_scores;

    fn row(dataset: &str, method: &str, value: &str, status: &str) -> String {
        valid_row(dataset, method, value, status, "True")
    }

    fn valid_row(dataset: &str, method: &str, value: &str, status: &str, valid: &str) -> String {
        format!("t,{dataset},{method},public-development,rmse,{value},{status},{valid},k,,,,\n")
    }

    fn inputs<'a>(
        ladder: Ladder,
        names: &'a BTreeMap<String, String>,
        pending: &'a BTreeMap<String, usize>,
    ) -> Inputs<'a> {
        Inputs {
            track_id: "t",
            version: Some("v1"),
            ladder,
            scores_sha256: "s",
            corrections: None,
            method_names: names,
            pending,
            total_datasets: 3,
        }
    }

    #[test]
    fn ladder_matches_the_python_reference_shape() {
        let mut text = crate::scores::COLUMNS.join(",");
        text.push('\n');
        text += &row("d1", "a", "1.0", "ok");
        text += &row("d1", "b", "2.0", "ok");
        text += &row("d2", "a", "1.0", "ok");
        text += &row("d2", "b", "1.0", "ok");
        text += &row("d2", "c", "", "missing");
        text += &row("d3", "c", "0.5", "ok");
        text += &row("d3", "b", "0.7", "ok");
        let scores = parse_scores(&text).unwrap();
        let names = BTreeMap::from([("a".to_owned(), "Method A".to_owned())]);
        let pending = BTreeMap::from([("c".to_owned(), 1)]);
        let board =
            build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap();
        assert_eq!(board.rankable_rows, 6);
        assert_eq!(board.datasets_scored, 3);
        assert_eq!(board.excluded_rows, 0);
        assert_eq!(board.scores_sha256, "s");
        assert!(board.rows.iter().all(|r| r.benchmark_valid == "True"));
        let ids: Vec<&str> = board.rows.iter().map(|r| r.method_id.as_str()).collect();
        assert_eq!(
            ids,
            ["c", "a", "b"],
            "c only ever beat b; a beat b once and tied once"
        );
        let a = &board.rows[1];
        assert_eq!(a.method, "Method A");
        assert_eq!(a.dataset_wins, 2, "a wins d1 and the d2 tie by id");
        assert_eq!(a.datasets_scored, 2);
        assert_eq!(a.full_coverage, "False");
        assert_eq!(a.full_coverage_rank, "");
        assert_eq!(a.coverage_pct, "66.7");
        assert_eq!(a.average_rank, "1.0");
        assert_eq!(a.coverage_status, "2/3");
        let b = &board.rows[2];
        assert_eq!(b.full_coverage, "True");
        assert_eq!(b.full_coverage_rank, "1");
        assert_eq!(b.average_rank, "2.0");
        assert_eq!(b.coverage_pct, "100.0");
        let c = &board.rows[0];
        assert_eq!(c.pending_cells, 1);
        assert_eq!(c.coverage_status, "1/3 · 1 pending");
        assert_eq!(c.dataset_wins, 1);
        let csv = to_csv(&board.rows).unwrap();
        assert!(csv.starts_with("elo_rank,full_coverage_rank,method_id,method,elo,average_rank,dataset_wins,datasets_scored,datasets_total,coverage_pct,full_coverage,pending_cells,coverage_status,protocols,benchmark_valid,scope\n"));
        assert_eq!(parse_csv(&csv).unwrap(), board.rows);
        let mean: f64 = board.ratings.values().sum::<f64>() / 3.0;
        assert!((mean - 1500.0).abs() < 1e-9);
        let snapshot = Snapshot::new(&board).unwrap();
        assert_eq!(snapshot.schema, SNAPSHOT_SCHEMA);
        assert_eq!(snapshot.ladder, Ladder::Published);
        assert_eq!(snapshot.leaderboard_sha256, sha256_hex(csv.as_bytes()));
        let text = serde_json::to_string(&snapshot).unwrap();
        assert!(text.contains("\"ladder\":\"published\""), "{text}");
        assert_eq!(serde_json::from_str::<Snapshot>(&text).unwrap(), snapshot);
    }

    #[test]
    fn benchmark_valid_ladder_drops_invalid_rows_and_published_ladder_marks_them() {
        // Method `a` wins d1 with a test-adaptive row; `b` is the honest full-coverage entry.
        let mut text = crate::scores::COLUMNS.join(",");
        text.push('\n');
        text += &valid_row("d1", "a", "0.1", "ok", "False");
        text += &row("d1", "b", "0.5", "ok");
        text += &row("d2", "a", "0.4", "ok");
        text += &row("d2", "b", "0.5", "ok");
        text += &row("d3", "b", "0.5", "ok");
        text += &valid_row("d3", "c", "0.2", "ok", "maybe");
        let scores = parse_scores(&text).unwrap();
        let names = BTreeMap::new();
        let pending = BTreeMap::new();
        let error =
            build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap_err();
        assert!(
            error.to_string().contains("neither True nor False"),
            "{error}"
        );
        let scores: Vec<ScoreRow> = scores.into_iter().filter(|r| r.method_id != "c").collect();

        let published =
            build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap();
        assert_eq!(published.rankable_rows, 5);
        assert_eq!(published.excluded_rows, 0);
        let a = published.rows.iter().find(|r| r.method_id == "a").unwrap();
        assert_eq!(a.datasets_scored, 2);
        assert_eq!(
            a.benchmark_valid, "False",
            "one invalid row marks the method's published standing"
        );
        assert_eq!(
            published.rows[0].method_id, "a",
            "a wins both datasets as published"
        );
        let b = published.rows.iter().find(|r| r.method_id == "b").unwrap();
        assert_eq!(b.benchmark_valid, "True");
        assert_eq!(published.rows[0].scope, Ladder::Published.scope());

        let valid =
            build_leaderboard(&scores, inputs(Ladder::BenchmarkValid, &names, &pending)).unwrap();
        assert_eq!(valid.rankable_rows, 4);
        assert_eq!(valid.excluded_rows, 1);
        assert_eq!(valid.ladder, Ladder::BenchmarkValid);
        let a = valid.rows.iter().find(|r| r.method_id == "a").unwrap();
        assert_eq!(
            a.datasets_scored, 1,
            "the invalid d1 row never entered the fit"
        );
        assert_eq!(a.full_coverage, "False");
        assert_eq!(a.benchmark_valid, "True");
        assert_eq!(a.dataset_wins, 1, "a still wins d2 honestly");
        assert!(valid.rows.iter().all(|r| r.benchmark_valid == "True"));
        assert_eq!(valid.rows[0].scope, Ladder::BenchmarkValid.scope());
        let snapshot = Snapshot::new(&valid).unwrap();
        assert_eq!(snapshot.excluded_rows, 1);
        assert!(serde_json::to_string(&snapshot)
            .unwrap()
            .contains("\"ladder\":\"benchmark-valid\""));
    }

    #[test]
    fn pending_counts_only_unresolved_cells() {
        let ledger = "{\"method_id\":\"x\",\"resolved\":false}\n{\"method_id\":\"x\",\"resolved\":true}\n\n{\"method_id\":\"y\",\"resolved\":false}\n";
        let pending = pending_by_method(ledger).unwrap();
        assert_eq!(
            pending,
            BTreeMap::from([("x".to_owned(), 1), ("y".to_owned(), 1)])
        );
        assert!(pending_by_method("{\"method_id\":\"x\"}\n").is_err());
    }

    #[test]
    fn results_resolution_prefers_versions_then_flat() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let base = root.join("results/tr");
        fs::create_dir_all(base.join("v1")).unwrap();
        fs::create_dir_all(base.join("v2")).unwrap();
        fs::create_dir_all(base.join("empty")).unwrap();
        assert!(resolve_results(root, "tr", None).is_err());
        fs::write(base.join("scores.csv"), "").unwrap();
        assert_eq!(
            resolve_results(root, "tr", None).unwrap(),
            ResultsLocation {
                dir: base.clone(),
                version: None
            }
        );
        fs::write(base.join("v1/scores.csv"), "").unwrap();
        crate::io::write_gz(&base.join("v2/scores.csv.gz"), b"").unwrap();
        assert_eq!(list_versions(root, "tr").unwrap(), ["v1", "v2"]);
        assert_eq!(
            resolve_results(root, "tr", None)
                .unwrap()
                .version
                .as_deref(),
            Some("v2")
        );
        assert_eq!(
            resolve_results(root, "tr", Some("v1")).unwrap().dir,
            base.join("v1")
        );
        assert!(resolve_results(root, "tr", Some("empty"))
            .unwrap_err()
            .to_string()
            .contains("versions present"));
    }
}
