//! Stage 5: pick the program to emit, from what the search actually measured.
//!
//! Only programs measured on every requested cut are eligible. Among them,
//! the lowest mean held-out RMSE wins, with canonical genome as the tie-breaker.
//! Partial coverage is recorded, never compared with a different cut set. If
//! no program covers every cut, selection fails rather than emitting an
//! unqualified partial winner.

use anyhow::{ensure, Result};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    search::{CutSearch, Observation},
    sha256_hex,
};

/// Generation 0's seed.
pub use crate::search::BOOTSTRAP_GENOME;

/// How the emitted program was chosen.
pub const MIN_MEAN_VALIDATION_RMSE: &str = "all_cuts_then_min_mean_validation_rmse_then_genome";
/// Historical v1 method; v2 refuses selection with no complete coverage.
pub const UNMEASURED_FALLBACK: &str = "unmeasured_fallback_bootstrap_genome";

/// One program's measurement on one cut.
#[derive(Clone, Copy, Debug)]
struct Measurement {
    rmse: f64,
    hold_r2: f64,
    train_r2: f64,
    seconds: f64,
}

/// The winning program and its evidence.
#[derive(Clone, Debug)]
pub struct Best {
    pub genome: String,
    pub program_sha256: String,
    /// Mean held-out RMSE over the cuts this program was measured on, in the
    /// dataset's own target units.
    pub validation_rmse: f64,
    /// Mean held-out R^2 over the same cuts.
    pub validation_r2: f64,
    /// Mean in-sample R^2, for the gap against `validation_r2`.
    pub train_r2: f64,
    /// Total fit-and-score seconds this program cost.
    pub seconds: f64,
    pub cuts_measured: usize,
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub method: &'static str,
    /// The genome handed to the emitters, always present.
    pub genome: String,
    pub program_sha256: String,
    pub best: Option<Best>,
    pub reason: Option<String>,
    pub compared_cuts: Vec<u64>,
    pub coverage: Vec<crate::receipt::Coverage>,
}

/// Apply the selection rule over every cut's observations.
pub fn select(cuts: &[CutSearch]) -> Result<Selection> {
    let per_cut: Vec<(usize, &[Observation])> = cuts
        .iter()
        .map(|c| (c.cut, c.observations.as_slice()))
        .collect();
    select_from(&per_cut)
}

/// [`select`] over `(cut index, observations)` pairs, for callers that re-derive a selection
/// from published evidence rather than from a live search.
pub fn select_from(cuts: &[(usize, &[Observation])]) -> Result<Selection> {
    let (ranked, coverage, partial) = rank(cuts)?;
    let searched = cuts.len();
    let best = ranked.into_iter().next().ok_or_else(|| anyhow::anyhow!(
        "no program measured on all {searched} requested cuts; {partial} incomplete program(s); inspect observations.json"
    ))?;
    let reason = (partial > 0).then(|| {
        format!(
            "excluded {partial} program(s) without measurements on all {searched} requested cuts"
        )
    });
    Ok(Selection {
        method: MIN_MEAN_VALIDATION_RMSE,
        genome: best.genome.clone(),
        program_sha256: best.program_sha256.clone(),
        best: Some(best),
        reason,
        compared_cuts: cuts
            .iter()
            .map(|&(c, _)| c as u64)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        coverage,
    })
}

/// Every program measured on all requested cuts, best first under the selection rule (lowest
/// mean held-out RMSE, then canonical genome). The first entry is what [`select`] picks; the
/// rest are the ranked runners-up a later round may be seeded with.
pub fn ranking(cuts: &[(usize, &[Observation])]) -> Result<Vec<Best>> {
    Ok(rank(cuts)?.0)
}

type Ranked = (Vec<Best>, Vec<crate::receipt::Coverage>, usize);

fn rank(cuts: &[(usize, &[Observation])]) -> Result<Ranked> {
    let expected: BTreeSet<usize> = cuts.iter().map(|&(c, _)| c).collect();
    ensure!(
        !expected.is_empty() && expected.len() == cuts.len(),
        "selection needs distinct cuts"
    );
    let mut attempted: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    // genome -> cut -> that program's measurement on that cut.
    let mut measured: BTreeMap<String, BTreeMap<usize, Measurement>> = BTreeMap::new();
    for &(cut_index, observations) in cuts {
        for observation in observations {
            ensure!(
                observation.cut == cut_index,
                "observation cut identity mismatch"
            );
            ensure!(
                attempted
                    .entry(observation.genome.clone())
                    .or_default()
                    .insert(cut_index),
                "duplicate program attempt on cut {}",
                cut_index
            );
            let (Some(rmse), Some(hold_r2), Some(train_r2)) =
                (observation.rmse, observation.hold_r2, observation.train_r2)
            else {
                continue;
            };
            if !observation.measured()
                || !rmse.is_finite()
                || rmse < 0.0
                || !hold_r2.is_finite()
                || !train_r2.is_finite()
            {
                continue;
            }
            let entry = measured
                .entry(observation.genome.clone())
                .or_default()
                .entry(observation.cut);
            let candidate = Measurement {
                rmse,
                hold_r2,
                train_r2,
                seconds: observation.seconds,
            };
            entry.or_insert(candidate);
        }
    }

    let searched = cuts.len();
    let coverage: Vec<crate::receipt::Coverage> = attempted
        .iter()
        .map(|(genome, attempts)| {
            let measured_cuts: BTreeSet<usize> = measured
                .get(genome)
                .map(|m| m.keys().copied().collect())
                .unwrap_or_default();
            crate::receipt::Coverage {
                program_sha256: sha256_hex(genome.as_bytes()),
                measured_cuts: measured_cuts.iter().map(|&c| c as u64).collect(),
                failed_cuts: attempts
                    .difference(&measured_cuts)
                    .map(|&c| c as u64)
                    .collect(),
                missing_cuts: expected.difference(attempts).map(|&c| c as u64).collect(),
            }
        })
        .collect();
    let partial = coverage
        .iter()
        .filter(|c| c.measured_cuts.len() != searched)
        .count();
    let mut ranked: Vec<Best> = measured
        .into_iter()
        .filter(|(_, per_cut)| per_cut.len() == searched)
        .map(|(genome, per_cut)| {
            let n = per_cut.len() as f64;
            Best {
                program_sha256: sha256_hex(genome.as_bytes()),
                genome,
                validation_rmse: per_cut.values().map(|m| m.rmse).sum::<f64>() / n,
                validation_r2: per_cut.values().map(|m| m.hold_r2).sum::<f64>() / n,
                train_r2: per_cut.values().map(|m| m.train_r2).sum::<f64>() / n,
                seconds: per_cut.values().map(|m| m.seconds).sum::<f64>(),
                cuts_measured: per_cut.len(),
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.validation_rmse
            .total_cmp(&b.validation_rmse)
            .then(a.genome.cmp(&b.genome))
    });
    Ok((ranked, coverage, partial))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{Observation, MEASURED, PARSE_FAILED};

    fn cut(index: usize, observations: Vec<Observation>) -> CutSearch {
        CutSearch {
            cut: index,
            output: std::path::PathBuf::from("."),
            train_rows: 10,
            hold_rows: 4,
            generations: 1,
            completed_evaluations: observations.len() as u64,
            successful_measurements: observations.iter().filter(|o| o.measured()).count() as u64,
            failed_or_unavailable_measurements: observations
                .iter()
                .filter(|o| !o.measured())
                .count() as u64,
            stopped_by: crate::search::STOPPED_BY_EVALUATIONS,
            seconds: 1.0,
            exploration: crate::receipt::Phase::not_requested(),
            comparison: crate::receipt::Phase::not_requested(),
            best_validation_rmse: None,
            best_validation_r2: None,
            best_program: None,
            observations,
            search_sha256: String::new(),
            observations_sha256: String::new(),
            rng_seed: 0,
            budget: crate::search::Budget::default(),
        }
    }

    fn measured(genome: &str, cut: usize, rmse: f64) -> Observation {
        Observation {
            genome: genome.to_owned(),
            program_sha256: sha256_hex(genome.as_bytes()),
            cut,
            generation: 0,
            origin: "random",
            train_r2: Some(0.9),
            hold_r2: Some(0.5),
            rmse: Some(rmse),
            seconds: 0.5,
            status: MEASURED,
            failure: None,
        }
    }

    fn failed(genome: &str, cut: usize) -> Observation {
        Observation {
            rmse: None,
            hold_r2: None,
            train_r2: None,
            status: PARSE_FAILED,
            failure: Some("unbalanced parentheses".to_owned()),
            ..measured(genome, cut, 0.0)
        }
    }

    #[test]
    fn an_unmeasured_search_fails_closed() {
        let error = select(&[cut(0, vec![failed("(pipe X (ridge))", 0)])]).unwrap_err();
        assert!(error.to_string().contains("no program measured on all"));
    }

    #[test]
    fn complete_cover_still_ranks_above_a_better_partial_cover() {
        // select() itself still prefers full cover. The product path must
        // remeasure top-k first so a real champion is on every cut.
        let wide = "(pipe X (ridge))";
        let narrow = "(pipe X (standardize) (ridge))";
        let selection = select(&[
            cut(0, vec![measured(wide, 0, 1.0), measured(narrow, 0, 0.1)]),
            cut(1, vec![measured(wide, 1, 1.2)]),
        ])
        .unwrap();
        assert_eq!(selection.method, MIN_MEAN_VALIDATION_RMSE);
        let best = selection.best.unwrap();
        assert_eq!(best.genome, wide);
        assert_eq!(best.cuts_measured, 2);
        assert!((best.validation_rmse - 1.1).abs() < 1e-12);
        assert_eq!(best.program_sha256, sha256_hex(wide.as_bytes()));
        assert!(selection.reason.unwrap().contains("excluded 1 program"));
        assert_eq!(selection.compared_cuts, vec![0, 1]);
    }

    #[test]
    fn the_lowest_mean_wins_among_programs_measured_on_the_same_cuts() {
        let worse = "(pipe X (ridge))";
        let better = "(pipe X (standardize) (ridge))";
        let selection = select(&[
            cut(0, vec![measured(worse, 0, 1.0), measured(better, 0, 0.4)]),
            cut(1, vec![measured(worse, 1, 1.2), measured(better, 1, 0.6)]),
        ])
        .unwrap();
        let best = selection.best.unwrap();
        assert_eq!(best.genome, better);
        assert!((best.validation_rmse - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ties_break_deterministically_on_the_genome() {
        let a = "(pipe X (a_head))";
        let b = "(pipe X (b_head))";
        let forward = select(&[cut(0, vec![measured(a, 0, 1.0), measured(b, 0, 1.0)])])
            .unwrap()
            .genome;
        let reversed = select(&[cut(0, vec![measured(b, 0, 1.0), measured(a, 0, 1.0)])])
            .unwrap()
            .genome;
        assert_eq!(forward, reversed);
        assert_eq!(forward, a);
    }

    #[test]
    fn different_partial_cut_sets_are_never_compared() {
        let error = select(&[
            cut(0, vec![measured("a", 0, 1.0)]),
            cut(1, vec![measured("b", 1, 0.1)]),
        ])
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("no program measured on all 2 requested cuts"));
    }

    #[test]
    fn a_failed_observation_never_contributes_a_score() {
        let genome = "(pipe X (ridge))";
        let selection = select(&[cut(
            0,
            vec![measured(genome, 0, 2.0), failed("(pipe X (bogus", 0)],
        )])
        .unwrap();
        let best = selection.best.unwrap();
        assert_eq!(best.genome, genome);
        assert_eq!(best.cuts_measured, 1);
    }
}
