use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

use crate::elo::{fit_elo, matches_from_scores_directed, Direction, TIE_RELATIVE_TOLERANCE};
use crate::pyfmt::{py_bool, py_repr, py_round};
use crate::scores::ScoreRow;

use super::{Inputs, Ladder, Leaderboard, LeaderboardRow};

pub fn build_leaderboard(scores: &[ScoreRow], inputs: Inputs<'_>) -> Result<Leaderboard> {
    let track_id = inputs.track_id;
    let total_datasets = inputs.total_datasets;
    let (method_names, pending) = (inputs.method_names, inputs.pending);
    let mut by_dataset: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut protocols_by_method: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut valid_by_method: BTreeMap<String, bool> = BTreeMap::new();
    let mut rankable_rows = 0;
    let mut excluded_rows = 0;
    let mut delivered: BTreeSet<(&str, &str)> = BTreeSet::new();
    for row in scores {
        if !row.is_rankable() {
            continue;
        }
        ensure!(
            row.track_id == track_id,
            "score row for {} inside track {track_id}",
            row.track_id
        );
        // W12-T04: one delivery per rankable (dataset, method) cell, whatever its validity. A
        // second delivery can neither add a match nor replace the first, and the independent
        // replay (`jope_records::elo_history::parse_scores`) refuses it the same way.
        ensure!(
            delivered.insert((row.dataset_id.as_str(), row.method_id.as_str())),
            "duplicate delivery of {}/{} cannot add or replace a match",
            row.dataset_id,
            row.method_id
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
    let matches =
        matches_from_scores_directed(&by_dataset, inputs.direction, TIE_RELATIVE_TOLERANCE);
    let ratings = fit_elo(&matches);

    let mut ranks: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut wins: BTreeMap<&str, usize> = BTreeMap::new();
    for dataset_scores in by_dataset.values() {
        let mut ordered: Vec<(&str, f64)> = dataset_scores
            .iter()
            .map(|(m, v)| (m.as_str(), *v))
            .collect();
        // Python: sorted by (value, method_id); values are finite so total_cmp is the same order.
        // A maximize track ranks the largest value first, ties still by method id.
        ordered.sort_by(|a, b| {
            let by_value = match inputs.direction {
                Direction::Minimize => a.1.total_cmp(&b.1),
                Direction::Maximize => b.1.total_cmp(&a.1),
            };
            by_value.then_with(|| a.0.cmp(b.0))
        });
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
