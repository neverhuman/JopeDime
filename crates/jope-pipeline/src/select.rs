//! Stage 5: pick the program to emit, from what the search actually measured.
//!
//! The rule is the plain one named in the receipt: over the cuts that were
//! searched, a program's score is the **mean of its held-out RMSE**, and the
//! lowest mean wins. Every RMSE in that mean was computed on rows the fit that
//! produced it never saw ([`crate::prepare`] builds the folds; [`crate::search`]
//! never crosses them).
//!
//! A program measured on more cuts beats one measured on fewer, because a mean
//! over one cut is not comparable with a mean over five. Ties break on the
//! canonical genome, so the choice is a function of the measurements alone and
//! two runs of the same command select the same program.
//!
//! This is *not* a fold-pure lower-confidence-bound selection. No such selector
//! exists in this workspace and none is invented here; [`Selection::method`] says
//! exactly what was applied.
//!
//! [`UNMEASURED_FALLBACK`] survives as a guard, not as a normal path. Since the
//! search fails loudly when a cut measures nothing, a run reaching this crate's
//! [`select`] always carries at least one measurement; the fallback is what keeps
//! that an assertion rather than a panic.

use anyhow::Result;
use std::collections::BTreeMap;

use crate::{search::CutSearch, sha256_hex};

/// Generation 0's seed, and the genome emitted if nothing at all measured.
pub use crate::search::BOOTSTRAP_GENOME;

/// How the emitted program was chosen.
pub const MIN_MEAN_VALIDATION_RMSE: &str = "min_mean_validation_rmse";
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
}

/// Apply the selection rule over every cut's observations.
pub fn select(cuts: &[CutSearch]) -> Result<Selection> {
    // genome -> cut -> that program's best measurement on that cut.
    let mut measured: BTreeMap<String, BTreeMap<usize, Measurement>> = BTreeMap::new();
    for cut in cuts {
        for observation in &cut.observations {
            let (Some(rmse), Some(hold_r2), Some(train_r2)) =
                (observation.rmse, observation.hold_r2, observation.train_r2)
            else {
                continue;
            };
            if !observation.measured() || !rmse.is_finite() || rmse < 0.0 {
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
            entry
                .and_modify(|kept| {
                    // One cut may measure the same genome only once, but keep the
                    // rule total anyway: the better score on that cut stands.
                    if candidate.rmse < kept.rmse {
                        *kept = candidate;
                    }
                })
                .or_insert(candidate);
        }
    }

    let searched = cuts.len();
    let ranked = measured
        .into_iter()
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
        .min_by(|a, b| {
            b.cuts_measured
                .cmp(&a.cuts_measured)
                .then(a.validation_rmse.total_cmp(&b.validation_rmse))
                .then(a.genome.cmp(&b.genome))
        });

    let Some(best) = ranked else {
        return Ok(Selection {
            method: UNMEASURED_FALLBACK,
            genome: BOOTSTRAP_GENOME.to_owned(),
            program_sha256: sha256_hex(BOOTSTRAP_GENOME.as_bytes()),
            best: None,
            reason: Some(format!(
                "no successful measurement on any of the {searched} searched cut(s); \
                 emitting the bootstrap genome {BOOTSTRAP_GENOME} instead"
            )),
        });
    };
    let reason = (best.cuts_measured < searched).then(|| {
        format!(
            "measured on {} of {searched} searched cut(s); the mean is over those cuts only",
            best.cuts_measured
        )
    });
    Ok(Selection {
        method: MIN_MEAN_VALIDATION_RMSE,
        genome: best.genome.clone(),
        program_sha256: best.program_sha256.clone(),
        best: Some(best),
        reason,
    })
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
    fn an_unmeasured_search_falls_back_and_says_so() {
        let selection = select(&[cut(0, vec![failed("(pipe X (ridge))", 0)])]).unwrap();
        assert_eq!(selection.method, UNMEASURED_FALLBACK);
        assert_eq!(selection.genome, BOOTSTRAP_GENOME);
        assert!(selection.best.is_none());
        assert!(selection
            .reason
            .unwrap()
            .contains("no successful measurement"));
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
        assert!(selection.reason.is_none());
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
    fn a_partial_cut_cover_is_recorded_in_the_reason() {
        let only = "(pipe X (ridge))";
        let selection = select(&[cut(0, vec![measured(only, 0, 1.0)]), cut(1, vec![])]).unwrap();
        assert_eq!(selection.best.unwrap().cuts_measured, 1);
        assert!(selection.reason.unwrap().contains("1 of 2"));
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
