//! Per-task ELO over the strict corpus: is the model actually contributing?
//!
//! The flywheel had no answer to that question. `scripts/render_elo_ranking.py`
//! is a static figure generator with hard-coded numbers, and the TUI never
//! mentioned ELO at all, so "the deep learning proposes, genetic programming
//! evolves, the best one wins" was an assertion nobody could check per task.
//!
//! A first raw measurement said the model loses ~95% of head-to-head dataset
//! comparisons against GP (multiclass 3.1%, binary 4.7%). That number is real
//! but it is NOT a verdict, because `gp_origin` holds ~8.1M observations against
//! ~0.9M for the two DL classes: roughly 14x more draws per dataset, and
//! best-of-N rises with N for free. An uncontrolled comparison therefore
//! measures attempt count as much as proposal quality -- the same defect that
//! made the old "matched" GP comparator untrustworthy.
//!
//! So this job equalizes draws per arm per dataset before comparing, and
//! publishes the controlled AND the raw figure side by side. The gap between
//! them is itself the finding.

use anyhow::{Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_runtime::{atomic_json, now_ms};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Task -> the ONE metric that task's observations are scored on. Mirrors
/// ENABLED_TASK_METRICS in scripts/test_exact_train.py; an observation whose
/// metric drifted from its task's contract is not comparable and is excluded.
const TASK_METRICS: [(&str, &str); 4] = [
    ("regression", "r2"),
    ("binary", "auc"),
    ("multiclass", "macro_f1"),
    ("timeseries", "negative_mase"),
];

/// Every task gets a row even with zero observations. A dark lane must LOOK
/// dark; rendering only the tasks that happen to have data is how "timeseries
/// produces nothing" hid in plain sight.
const ALL_TASKS: [&str; 4] = ["regression", "binary", "multiclass", "timeseries"];

const BASE_RATING: f64 = 1200.0;
const K_FACTOR: f64 = 32.0;
/// Below this many matches a rating is noise, and is published as
/// `insufficient_evidence` rather than as a number someone will quote.
const MIN_MATCHES: usize = 8;
/// Shrinkage toward BASE_RATING: a 3-match arm must not outrank a 3000-match
/// one on luck. rating' = base + (rating - base) * m / (m + SHRINKAGE).
const SHRINKAGE: f64 = 24.0;

#[derive(Parser, Debug)]
#[command(name = "jope-task-elo")]
#[command(about = "Draw-controlled per-task ELO for proposal producers and genomes")]
struct Args {
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite"
    )]
    corpus: PathBuf,
    #[arg(
        long,
        default_value = "/home/ubuntu/jope-flywheel/test_exact_gen0/ops/task-elo.v1.json"
    )]
    output: PathBuf,
    /// Cap the genome leaderboard published per task.
    #[arg(long, default_value_t = 32)]
    top_genomes: usize,
    /// Stop after this many datasets per task (smoke runs only).
    #[arg(long)]
    limit_datasets: Option<usize>,
}

/// Which producer a candidate came from. `origin_class` is the corpus's own
/// label; `candidate_source` distinguishes the robust bank inside gp_origin.
fn producer_of(origin_class: &str, candidate_source: &str) -> &'static str {
    if candidate_source.contains("robust_gp_seed_bank")
        || candidate_source.contains("cross_dataset_repeated_robust_gp_winner")
        || candidate_source.contains("task_common_winner")
    {
        return "robust_bank";
    }
    match origin_class {
        "dl_origin" => "dl_origin",
        "dl_descendant" => "dl_descendant",
        _ => "gp_origin",
    }
}

/// Deterministic per-(task,dataset) stream position. No RNG: the same corpus
/// must always yield the same sample, or ELO stops being replayable.
fn stride_seed(task: &str, dataset: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in task.bytes().chain(b"/".iter().copied()).chain(dataset.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash | 1
}

/// Take exactly `n` entries spread deterministically across `values`.
fn sample_n(values: &[f64], n: usize, seed: u64) -> Vec<f64> {
    if values.len() <= n {
        return values.to_vec();
    }
    let start = (seed as usize) % values.len();
    let stride = values.len() / n.max(1);
    (0..n)
        .map(|index| values[(start + index * stride.max(1)) % values.len()])
        .collect()
}

fn expected(rating: f64, opponent: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((opponent - rating) / 400.0))
}

#[derive(Clone, Debug, Default)]
struct Rating {
    value: f64,
    matches: usize,
    wins: f64,
}

impl Rating {
    fn new() -> Self {
        Self {
            value: BASE_RATING,
            matches: 0,
            wins: 0.0,
        }
    }

    fn shrunk(&self) -> f64 {
        let m = self.matches as f64;
        BASE_RATING + (self.value - BASE_RATING) * m / (m + SHRINKAGE)
    }
}

/// One ELO match between two rated entities. `score` is 1 / 0.5 / 0.
fn play(table: &mut BTreeMap<String, Rating>, left: &str, right: &str, score: f64) {
    let left_rating = table.entry(left.to_owned()).or_insert_with(Rating::new).value;
    let right_rating = table
        .entry(right.to_owned())
        .or_insert_with(Rating::new)
        .value;
    let left_expected = expected(left_rating, right_rating);
    {
        let entry = table.get_mut(left).expect("left present");
        entry.value = left_rating + K_FACTOR * (score - left_expected);
        entry.matches += 1;
        entry.wins += score;
    }
    {
        let entry = table.get_mut(right).expect("right present");
        entry.value = right_rating + K_FACTOR * ((1.0 - score) - (1.0 - left_expected));
        entry.matches += 1;
        entry.wins += 1.0 - score;
    }
}

fn outcome(left: f64, right: f64) -> f64 {
    if (left - right).abs() <= f64::EPSILON {
        0.5
    } else if left > right {
        1.0
    } else {
        0.0
    }
}

#[derive(Default)]
struct TaskAccumulator {
    controlled: BTreeMap<String, Rating>,
    raw: BTreeMap<String, Rating>,
    genomes: BTreeMap<String, Rating>,
    datasets: usize,
    observations: usize,
    scores: Vec<f64>,
    runtimes: Vec<f64>,
    nonfinite: usize,
}

/// Fold one dataset's observations into the task's accumulators.
fn absorb_dataset(
    accumulator: &mut TaskAccumulator,
    task: &str,
    dataset: &str,
    rows: &[(String, String, String, f64, f64)],
) {
    if rows.is_empty() {
        return;
    }
    accumulator.datasets += 1;
    accumulator.observations += rows.len();

    let mut by_producer: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut genome_best: BTreeMap<&str, f64> = BTreeMap::new();
    for (semantic, origin, source, score, runtime) in rows {
        if !score.is_finite() {
            accumulator.nonfinite += 1;
            continue;
        }
        accumulator.scores.push(*score);
        if runtime.is_finite() {
            accumulator.runtimes.push(*runtime);
        }
        by_producer
            .entry(producer_of(origin, source))
            .or_default()
            .push(*score);
        let slot = genome_best.entry(semantic.as_str()).or_insert(*score);
        if *score > *slot {
            *slot = *score;
        }
    }
    if by_producer.is_empty() {
        return;
    }

    // RAW: best-of-arm as the corpus happens to hold it. This is the number
    // that flatters whichever producer got more attempts.
    let raw_best: Vec<(&str, f64)> = by_producer
        .iter()
        .map(|(name, scores)| {
            (
                *name,
                scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        })
        .collect();
    for left in 0..raw_best.len() {
        for right in (left + 1)..raw_best.len() {
            let score = outcome(raw_best[left].1, raw_best[right].1);
            play(
                &mut accumulator.raw,
                raw_best[left].0,
                raw_best[right].0,
                score,
            );
        }
    }

    // DRAW-CONTROLLED: every producer gets the SAME number of attempts on this
    // dataset before best-of is taken, so the comparison measures proposal
    // quality rather than proposal volume.
    let quota = by_producer
        .values()
        .map(|scores| scores.len())
        .min()
        .unwrap_or(0);
    if quota > 0 {
        let seed = stride_seed(task, dataset);
        let controlled_best: Vec<(&str, f64)> = by_producer
            .iter()
            .map(|(name, scores)| {
                let sampled = sample_n(scores, quota, seed);
                (
                    *name,
                    sampled.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            })
            .collect();
        for left in 0..controlled_best.len() {
            for right in (left + 1)..controlled_best.len() {
                let score = outcome(controlled_best[left].1, controlled_best[right].1);
                play(
                    &mut accumulator.controlled,
                    controlled_best[left].0,
                    controlled_best[right].0,
                    score,
                );
            }
        }
    }

    // Genome leaderboard: adjacent pairs in score order. O(k) per dataset
    // rather than the O(k^2) full round robin, which at ~363 rows per dataset
    // across ~25k datasets would never finish.
    let mut ranked: Vec<(&str, f64)> = genome_best.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(right.0))
    });
    for window in ranked.windows(2) {
        let score = outcome(window[0].1, window[1].1);
        play(&mut accumulator.genomes, window[0].0, window[1].0, score);
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted.get(index).copied()
}

fn rating_rows(table: &BTreeMap<String, Rating>, limit: Option<usize>) -> Vec<Value> {
    let mut rows: Vec<(&String, &Rating)> = table.iter().collect();
    rows.sort_by(|left, right| {
        right
            .1
            .shrunk()
            .partial_cmp(&left.1.shrunk())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(right.0))
    });
    if let Some(limit) = limit {
        rows.truncate(limit);
    }
    rows.into_iter()
        .map(|(name, rating)| {
            let sufficient = rating.matches >= MIN_MATCHES;
            json!({
                "name": name,
                // A rating below the match floor is NOT published as a number:
                // someone will quote it, and at 3 matches it is noise.
                "elo": if sufficient { Some((rating.shrunk() * 100.0).round() / 100.0) } else { None },
                "insufficient_evidence": !sufficient,
                "matches": rating.matches,
                "win_rate": if rating.matches > 0 {
                    Some((rating.wins / rating.matches as f64 * 1000.0).round() / 1000.0)
                } else { None },
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let connection = Connection::open_with_flags(
        &args.corpus,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("open corpus read-only {}", args.corpus.display()))?;

    let mut tasks: BTreeMap<String, TaskAccumulator> = BTreeMap::new();
    for task in ALL_TASKS {
        tasks.insert(task.to_owned(), TaskAccumulator::default());
    }

    for (task, metric) in TASK_METRICS {
        // Scan in PRIMARY KEY order (task, dataset_hash, semantic_sha256) so
        // SQLite streams the index instead of materializing a sort. A GROUP BY
        // over 9M rows exceeded a ten-minute budget; this does not.
        let mut statement = connection.prepare(
            "SELECT dataset_hash, semantic_sha256, origin_class, candidate_source, score, runtime_seconds
             FROM observations WHERE task=?1 AND metric=?2 ORDER BY dataset_hash",
        )?;
        let mut rows = statement.query(rusqlite::params![task, metric])?;

        let accumulator = tasks.get_mut(task).expect("task seeded");
        let mut current = String::new();
        let mut batch: Vec<(String, String, String, f64, f64)> = Vec::new();
        let mut seen_datasets = 0usize;
        while let Some(row) = rows.next()? {
            let dataset: String = row.get(0)?;
            if dataset != current {
                if !batch.is_empty() {
                    absorb_dataset(accumulator, task, &current, &batch);
                    batch.clear();
                    seen_datasets += 1;
                    if args.limit_datasets.is_some_and(|cap| seen_datasets >= cap) {
                        break;
                    }
                }
                current = dataset;
            }
            batch.push((
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get::<_, Option<f64>>(5)?.unwrap_or(f64::NAN),
            ));
        }
        if !batch.is_empty() {
            absorb_dataset(accumulator, task, &current, &batch);
        }
    }

    let mut task_documents = serde_json::Map::new();
    for task in ALL_TASKS {
        let accumulator = tasks.get_mut(task).expect("task seeded");
        accumulator.scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        accumulator
            .runtimes
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        task_documents.insert(
            task.to_owned(),
            json!({
                "observations": accumulator.observations,
                "datasets": accumulator.datasets,
                // An empty lane says so out loud rather than rendering blank.
                "state": if accumulator.observations == 0 { "no_observations" } else { "live" },
                "producers_draw_controlled": rating_rows(&accumulator.controlled, None),
                "producers_raw": rating_rows(&accumulator.raw, None),
                "genomes": rating_rows(&accumulator.genomes, Some(args.top_genomes)),
                "performance": {
                    "best_score": accumulator.scores.last().copied(),
                    "median_score": percentile(&accumulator.scores, 0.5),
                    "p10_score": percentile(&accumulator.scores, 0.10),
                    "median_runtime_seconds": percentile(&accumulator.runtimes, 0.5),
                    "p95_runtime_seconds": percentile(&accumulator.runtimes, 0.95),
                    "nonfinite_scores": accumulator.nonfinite,
                },
            }),
        );
    }

    let document = json!({
        "schema": "JopeTaskElo.v1",
        "generated_at_ms": now_ms(),
        "base_rating": BASE_RATING,
        "k_factor": K_FACTOR,
        "min_matches": MIN_MATCHES,
        "shrinkage": SHRINKAGE,
        "note": "draw_controlled equalizes attempts per producer per dataset before best-of; \
                 raw does not and therefore rewards attempt count. Quote the controlled figure.",
        "tasks": Value::Object(task_documents),
    });
    atomic_json(&args.output, &document)?;
    println!("{}", serde_json::to_string(&document)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_bank_candidates_are_their_own_producer_not_generic_gp() {
        assert_eq!(
            producer_of("gp_origin", "cross_dataset_repeated_robust_gp_winner"),
            "robust_bank"
        );
        assert_eq!(
            producer_of("gp_origin", "robust_gp_seed_bank_coverage_probe"),
            "robust_bank"
        );
        assert_eq!(
            producer_of("gp_origin", "deterministic_rust_mutation_crossover"),
            "gp_origin"
        );
        assert_eq!(
            producer_of("dl_origin", "current_checkpoint_dl_proposal"),
            "dl_origin"
        );
        assert_eq!(producer_of("dl_descendant", "whatever"), "dl_descendant");
    }

    #[test]
    fn sampling_is_deterministic_and_exactly_the_quota() {
        let values: Vec<f64> = (0..100).map(|value| value as f64).collect();
        let seed = stride_seed("multiclass", "00b383718f6f4fd8");
        let first = sample_n(&values, 7, seed);
        let again = sample_n(&values, 7, seed);
        assert_eq!(first.len(), 7);
        assert_eq!(first, again, "ELO must be replayable from the same corpus");
        // A different dataset must not reuse the same stream position.
        let other = sample_n(&values, 7, stride_seed("multiclass", "0f7f4024a9b28e9c"));
        assert_ne!(first, other);
        // Fewer values than the quota returns everything, never pads.
        assert_eq!(sample_n(&values[..3], 7, seed).len(), 3);
    }

    /// The whole reason this binary exists: with equal attempts the arms are
    /// judged on quality, and the raw view must visibly disagree when one arm
    /// simply had more draws.
    #[test]
    fn draw_control_changes_the_verdict_when_attempts_are_lopsided() {
        let mut accumulator = TaskAccumulator::default();
        // GP gets 20 mediocre attempts; DL gets 2 good ones. Best-of-20 beats
        // best-of-2 on volume alone, but per-attempt the DL is stronger.
        let mut rows: Vec<(String, String, String, f64, f64)> = Vec::new();
        for index in 0..20 {
            rows.push((
                format!("gp{index}"),
                "gp_origin".into(),
                "deterministic_rust_mutation_crossover".into(),
                0.50 + index as f64 * 0.005,
                1.0,
            ));
        }
        for index in 0..2 {
            rows.push((
                format!("dl{index}"),
                "dl_origin".into(),
                "current_checkpoint_dl_proposal".into(),
                0.58 + index as f64 * 0.001,
                1.0,
            ));
        }
        absorb_dataset(&mut accumulator, "binary", "dataset0", &rows);

        let raw_gp = accumulator.raw.get("gp_origin").expect("gp raw").value;
        let raw_dl = accumulator.raw.get("dl_origin").expect("dl raw").value;
        assert!(
            raw_gp > raw_dl,
            "raw best-of-20 (0.595) should beat best-of-2 (0.581)"
        );

        let ctl_gp = accumulator
            .controlled
            .get("gp_origin")
            .expect("gp controlled")
            .value;
        let ctl_dl = accumulator
            .controlled
            .get("dl_origin")
            .expect("dl controlled")
            .value;
        assert!(
            ctl_dl > ctl_gp,
            "with 2 attempts each the DL's better proposals must win: \
             dl={ctl_dl} gp={ctl_gp}"
        );
    }

    #[test]
    fn thin_evidence_is_published_as_absent_not_as_a_number() {
        let mut table = BTreeMap::new();
        table.insert(
            "dl_origin".to_owned(),
            Rating {
                value: 1600.0,
                matches: 3,
                wins: 3.0,
            },
        );
        table.insert(
            "gp_origin".to_owned(),
            Rating {
                value: 1300.0,
                matches: 500,
                wins: 300.0,
            },
        );
        let rows = rating_rows(&table, None);
        let thin = rows
            .iter()
            .find(|row| row["name"] == "dl_origin")
            .expect("dl row");
        assert!(thin["elo"].is_null(), "3 matches must not publish a rating");
        assert_eq!(thin["insufficient_evidence"], json!(true));
        let solid = rows
            .iter()
            .find(|row| row["name"] == "gp_origin")
            .expect("gp row");
        assert!(solid["elo"].is_number());
        assert_eq!(solid["insufficient_evidence"], json!(false));
    }

    #[test]
    fn shrinkage_keeps_a_lucky_short_run_below_a_proven_one() {
        let lucky = Rating {
            value: 1500.0,
            matches: 4,
            wins: 4.0,
        };
        let proven = Rating {
            value: 1400.0,
            matches: 2_000,
            wins: 1_400.0,
        };
        assert!(
            proven.shrunk() > lucky.shrunk(),
            "4 wins must not outrank 2000 matches: lucky={} proven={}",
            lucky.shrunk(),
            proven.shrunk()
        );
    }
}
