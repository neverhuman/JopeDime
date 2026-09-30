//! Candidate admission for the library search under a fault policy (kernel2 WP-1.5).
//!
//! `crate::evolve::FaultPolicy` decides how the search treats the faults of a candidate's `map`
//! and `derive` stages. Under its non-legacy policies the search checks each program with
//! [`validate_scope`](super::scope::validate_scope) before fitting it, fits and predicts every
//! fold with the traced entry points of [`super::telemetry`], and turns the stage counts into the
//! totals, rates, penalty and order defined here. This module fits nothing and reads no data: it
//! is the arithmetic of the policy, kept apart so that it can be tested on constructed counts.
//!
//! The definitions:
//!
//! - **Passes.** A candidate's totals pool every traced pass of one evaluation: the training pass
//!   ([`fit_traced`](super::telemetry::fit_traced)) and the held-out pass
//!   ([`predict_traced`](super::telemetry::predict_traced)) of every fold.
//! - **Cells.** Only the cells a `map` or `derive` expression generated count. The input sanitize
//!   and the stage epilogues are left out, so a program without an expression stage has no cells,
//!   no faults and rates of 0.
//! - **Faulted cells.** In one stage pass, a cell counts once for each severity it carries,
//!   regardless of how many classes of that severity it raises. A constant output of at least
//!   two cells counts all its cells as Domain-faulted.
//! - **Rates.** A severity's faulted cells over the evaluated cells, and 0 when no cell was
//!   evaluated.
//! - **Hard.** A Hard class in any pass rejects the candidate.
//! - **Penalty.** `s - lambda_domain * domain_rate - lambda_saturation * saturation_rate`, in that
//!   order, where a term whose rate is 0 is left out. A candidate without a fault therefore keeps
//!   its score bit for bit, whatever the lambdas.
//! - **Ties.** Two scores within [`TIE_TOLERANCE`] of each other are tied. Among tied candidates,
//!   the one with fewer faulted cells (Domain and Saturation together) comes first.

use super::scope::ScopeError;
use super::telemetry::{PipelineTelemetry, StageFaultCounts};
use crate::math_expr::{FaultSet, FaultSeverity};
use std::fmt;

/// Two scores within this distance of each other are tied: [`prefer_fewer_faults`] and
/// [`outranks`] then order them by their faulted cells.
pub const TIE_TOLERANCE: f64 = 1e-12;

/// The fault totals of one evaluation of one candidate, pooled over the `map` and `derive` stages
/// of every traced pass (see the module documentation for each count).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaultTotals {
    /// Cells the expression stages generated.
    pub evaluated_cells: u64,
    /// Domain-faulted cells, constant stage passes included.
    pub domain_cells: u64,
    /// Saturation-faulted cells.
    pub saturation_cells: u64,
    /// Expression stage passes whose output was constant, each counted as Domain-faulted.
    pub constant_passes: u64,
    /// Rows with a generated cell whose faults include a Hard class.
    pub hard_rows: u64,
    /// Every Hard class a pass raised.
    pub hard_classes: FaultSet,
}

impl FaultTotals {
    /// Adds one traced pass: each of its stages.
    pub fn add_pass(&mut self, pass: &PipelineTelemetry) {
        for stage in &pass.stages {
            self.add_stage(stage);
        }
    }

    /// Adds one stage of one pass. A stage that generated no cell, which is every op but `map` and
    /// `derive`, adds nothing.
    pub fn add_stage(&mut self, stage: &StageFaultCounts) {
        let cells = stage.evaluated_cells;
        if cells == 0 {
            return;
        }
        let constant = stage.constant_output && cells >= 2;
        let domain = if constant {
            cells
        } else {
            stage.domain_cells.min(cells)
        };
        let saturation = stage.saturation_cells.min(cells);
        self.evaluated_cells = self.evaluated_cells.saturating_add(cells);
        self.domain_cells = self.domain_cells.saturating_add(domain);
        self.saturation_cells = self.saturation_cells.saturating_add(saturation);
        self.constant_passes = self.constant_passes.saturating_add(u64::from(constant));
        self.hard_rows = self.hard_rows.saturating_add(stage.hard_rows);
        for class in FaultSeverity::Hard.classes().iter() {
            if stage.faulted(class) > 0 {
                self.hard_classes = self.hard_classes.union(class.set());
            }
        }
    }

    /// Domain-faulted cells over evaluated cells; 0 when no cell was evaluated.
    pub fn domain_rate(&self) -> f64 {
        rate(self.domain_cells, self.evaluated_cells)
    }

    /// Saturation-faulted cells over evaluated cells; 0 when no cell was evaluated.
    pub fn saturation_rate(&self) -> f64 {
        rate(self.saturation_cells, self.evaluated_cells)
    }

    /// Domain and Saturation faulted cells together: the count the tie-break compares.
    pub fn faulted_cells(&self) -> u64 {
        self.domain_cells.saturating_add(self.saturation_cells)
    }

    /// Whether a pass raised a Hard class.
    pub fn is_hard(&self) -> bool {
        !self.hard_classes.is_empty()
    }

    /// `score - lambda_domain * domain_rate - lambda_saturation * saturation_rate`, evaluated in
    /// that order, where a term whose rate is 0 is left out, so a candidate without a fault gets
    /// `score` back bit for bit. `None` when the result is not finite.
    pub fn penalize(&self, score: f64, lambda_domain: f64, lambda_saturation: f64) -> Option<f64> {
        let mut penalized = score;
        if self.domain_cells > 0 {
            penalized -= lambda_domain * self.domain_rate();
        }
        if self.saturation_cells > 0 {
            penalized -= lambda_saturation * self.saturation_rate();
        }
        penalized.is_finite().then_some(penalized)
    }
}

fn rate(cells: u64, evaluated: u64) -> f64 {
    if evaluated == 0 {
        0.0
    } else {
        cells as f64 / evaluated as f64
    }
}

/// Why a non-legacy fault policy rejected a candidate. Its `Display` form is `scope: ` followed
/// by the scope error, or `hard fault: ` followed by the Hard classes.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The scope check refused the program that would have been fitted. Nothing was fitted.
    Scope(ScopeError),
    /// A traced pass raised these Hard classes. The evaluation stopped at that pass.
    HardFault(FaultSet),
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scope(error) => write!(f, "scope: {error}"),
            Self::HardFault(classes) => write!(f, "hard fault: {classes}"),
        }
    }
}

impl std::error::Error for Rejection {}

/// Reorders `ranked`, already sorted best first by `score`, so that among tied candidates the one
/// with fewer faulted cells comes first.
///
/// The list is cut into runs. A run starts at a candidate and holds every following candidate
/// whose score is within [`TIE_TOLERANCE`] of the run's first score. Each run is sorted stably by
/// `faulted_cells`: candidates with equal counts keep their order, so a list whose counts are all
/// equal is left as it is. Anchoring each run at its first score keeps the order well defined
/// where ties chain (a ties b and b ties c, but a does not tie c).
pub fn prefer_fewer_faults<T>(
    ranked: &mut [T],
    score: impl Fn(&T) -> f64,
    faulted_cells: impl Fn(&T) -> u64,
) {
    let mut start = 0;
    while start < ranked.len() {
        let anchor = score(&ranked[start]);
        let mut end = start + 1;
        while end < ranked.len() && (score(&ranked[end]) - anchor).abs() <= TIE_TOLERANCE {
            end += 1;
        }
        ranked[start..end].sort_by_key(|candidate| faulted_cells(candidate));
        start = end;
    }
}

/// Whether a candidate with `score` and `faulted_cells` beats one with `other_score` and
/// `other_cells`: with fewer faulted cells when the two scores are within [`TIE_TOLERANCE`] of
/// each other and the counts differ, and otherwise with a strictly higher score.
pub fn outranks(score: f64, faulted_cells: u64, other_score: f64, other_cells: u64) -> bool {
    if (score - other_score).abs() <= TIE_TOLERANCE && faulted_cells != other_cells {
        faulted_cells < other_cells
    } else {
        score > other_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_expr::FaultClass;

    /// One stage pass with `cells` generated cells. Classes of the same severity overlap on the
    /// same cells in these constructed counts.
    fn stage(cells: u64, faults: &[(FaultClass, u64)], constant: bool) -> StageFaultCounts {
        let mut counts = StageFaultCounts {
            evaluated_cells: cells,
            constant_output: constant,
            ..StageFaultCounts::default()
        };
        for &(class, count) in faults {
            counts.faulted_cells[class.index()] = count;
            match class.severity() {
                FaultSeverity::Hard => counts.hard_rows = counts.hard_rows.max(count),
                FaultSeverity::Domain => counts.domain_cells = counts.domain_cells.max(count),
                FaultSeverity::Saturation => {
                    counts.saturation_cells = counts.saturation_cells.max(count)
                }
            }
        }
        counts
    }

    fn pass(stages: Vec<StageFaultCounts>, input_sanitized: u64) -> PipelineTelemetry {
        PipelineTelemetry {
            input_sanitized,
            stages,
        }
    }

    #[test]
    fn stages_without_expression_cells_add_nothing() {
        // A non-expression stage reports only its epilogue, and the input sanitize is outside
        // every stage: neither is an expression fault.
        let quiet = StageFaultCounts {
            epilogue_sanitized: 7,
            ..StageFaultCounts::default()
        };
        let mut totals = FaultTotals::default();
        totals.add_pass(&pass(vec![quiet.clone(), quiet], 11));
        assert_eq!(totals, FaultTotals::default());
        assert_eq!(totals.domain_rate().to_bits(), 0.0_f64.to_bits());
        assert_eq!(totals.saturation_rate().to_bits(), 0.0_f64.to_bits());
        assert!(!totals.is_hard());
        // Without a fault the score comes back bit for bit, whatever the lambdas: no lambda is
        // multiplied, so a NaN, an infinite or a negative one cannot reach it.
        for score in [0.0, -0.0, 0.75, -3.25, 1e-300] {
            for lambdas in [(0.05, 0.0), (f64::NAN, f64::INFINITY), (-1.0, -0.0)] {
                assert_eq!(
                    totals
                        .penalize(score, lambdas.0, lambdas.1)
                        .map(f64::to_bits),
                    Some(score.to_bits())
                );
            }
        }
    }

    #[test]
    fn rates_pool_every_expression_stage_of_every_pass() {
        let mut totals = FaultTotals::default();
        totals.add_pass(&pass(
            vec![
                stage(10, &[(FaultClass::LogDomain, 3)], false),
                StageFaultCounts::default(),
            ],
            0,
        ));
        totals.add_pass(&pass(
            vec![stage(4, &[(FaultClass::PowExpClamped, 1)], false)],
            0,
        ));
        assert_eq!(totals.evaluated_cells, 14);
        assert_eq!(totals.domain_cells, 3);
        assert_eq!(totals.saturation_cells, 1);
        assert_eq!(totals.faulted_cells(), 4);
        assert_eq!(totals.domain_rate().to_bits(), (3.0_f64 / 14.0).to_bits());
        assert_eq!(
            totals.saturation_rate().to_bits(),
            (1.0_f64 / 14.0).to_bits()
        );
        assert!(!totals.is_hard());
    }

    #[test]
    fn each_severity_counts_a_cell_once_across_its_classes() {
        // `col(0)` on a NaN cell raises two Domain classes, and `exp(31)` two Saturation classes:
        // each such cell counts once under each class, but once for its severity.
        let mut totals = FaultTotals::default();
        totals.add_stage(&stage(
            2,
            &[
                (FaultClass::NonfiniteInput, 2),
                (FaultClass::NonfiniteSanitized, 2),
                (FaultClass::ExpArgClamped, 2),
                (FaultClass::MagnitudeClamped, 2),
            ],
            false,
        ));
        assert_eq!(totals.domain_cells, 2);
        assert_eq!(totals.saturation_cells, 2);
        assert_eq!(totals.domain_rate().to_bits(), 1.0_f64.to_bits());
        // Below the stage's cell count, two Domain classes on one cell still count once.
        let mut totals = FaultTotals::default();
        totals.add_stage(&stage(
            4,
            &[(FaultClass::LogDomain, 1), (FaultClass::DivNearZero, 1)],
            false,
        ));
        assert_eq!(totals.domain_cells, 1);
    }

    #[test]
    fn a_constant_stage_pass_is_domain_faulted_from_two_cells() {
        let mut totals = FaultTotals::default();
        totals.add_stage(&stage(5, &[(FaultClass::LogDomain, 5)], true));
        totals.add_stage(&stage(3, &[], true));
        assert_eq!(totals.evaluated_cells, 8);
        assert_eq!(totals.domain_cells, 8);
        assert_eq!(totals.constant_passes, 2);
        assert_eq!(totals.domain_rate().to_bits(), 1.0_f64.to_bits());
        // One cell is trivially constant, and is no evidence of a constant stage.
        let mut single = FaultTotals::default();
        single.add_stage(&stage(1, &[], true));
        assert_eq!(single.evaluated_cells, 1);
        assert_eq!(single.domain_cells, 0);
        assert_eq!(single.constant_passes, 0);
    }

    #[test]
    fn hard_classes_are_collected_and_named() {
        let mut totals = FaultTotals::default();
        totals.add_stage(&stage(6, &[(FaultClass::DivNearZero, 1)], false));
        assert!(!totals.is_hard());
        totals.add_stage(&stage(
            6,
            &[
                (FaultClass::ColumnOob, 6),
                (FaultClass::InvalidClipBounds, 2),
            ],
            false,
        ));
        assert!(totals.is_hard());
        assert_eq!(totals.hard_rows, 6);
        assert_eq!(
            totals.hard_classes,
            FaultSet::COLUMN_OOB.union(FaultSet::INVALID_CLIP_BOUNDS)
        );
        assert_eq!(
            Rejection::HardFault(totals.hard_classes).to_string(),
            "hard fault: {INVALID_CLIP_BOUNDS, COLUMN_OOB}"
        );
    }

    #[test]
    fn the_penalty_is_the_documented_formula() {
        let mut totals = FaultTotals::default();
        totals.add_stage(&stage(
            12,
            &[
                (FaultClass::LogDomain, 3),
                (FaultClass::SigmoidArgClamped, 1),
            ],
            false,
        ));
        let expected: f64 = 0.8 - 0.05 * (3.0 / 12.0) - 0.02 * (1.0 / 12.0);
        assert_eq!(
            totals.penalize(0.8, 0.05, 0.02).map(f64::to_bits),
            Some(expected.to_bits())
        );
        // A zero lambda leaves a faulted score where it was.
        assert_eq!(
            totals.penalize(0.8, 0.0, 0.0).map(f64::to_bits),
            Some(0.8_f64.to_bits())
        );
        // Only the Saturation term applies when there is no Domain fault.
        let mut saturated = FaultTotals::default();
        saturated.add_stage(&stage(4, &[(FaultClass::ExpArgClamped, 1)], false));
        assert_eq!(
            saturated.penalize(0.5, f64::NAN, 0.1).map(f64::to_bits),
            Some((0.5 - 0.1 * 0.25_f64).to_bits())
        );
        // A score the penalty makes non-finite is no score.
        assert_eq!(totals.penalize(0.8, f64::INFINITY, 0.0), None);
        assert_eq!(totals.penalize(0.8, f64::NAN, 0.0), None);
    }

    #[test]
    fn ties_prefer_fewer_faulted_cells_within_anchored_runs() {
        // (score, faulted cells, original position)
        let mut ranked = vec![
            (1.0, 3, 0),
            (1.0 - 5e-13, 0, 1),
            (1.0 - 1.5e-12, 0, 2),
            (0.5, 2, 3),
            (0.5, 2, 4),
            (0.5, 1, 5),
        ];
        prefer_fewer_faults(&mut ranked, |c| c.0, |c| c.1);
        let order: Vec<usize> = ranked.iter().map(|c| c.2).collect();
        // Run [0, 1] is anchored at 1.0; position 2 is 1.5e-12 below the anchor and starts its own
        // run although it is within the tolerance of position 1. In the last run the equal counts
        // keep their order.
        assert_eq!(order, vec![1, 0, 2, 5, 3, 4]);

        let mut equal = vec![(0.3, 4, 0), (0.3, 4, 1), (0.2, 0, 2)];
        prefer_fewer_faults(&mut equal, |c| c.0, |c| c.1);
        assert_eq!(equal.iter().map(|c| c.2).collect::<Vec<_>>(), vec![0, 1, 2]);
        let mut empty: Vec<(f64, u64)> = Vec::new();
        prefer_fewer_faults(&mut empty, |c| c.0, |c| c.1);
    }

    #[test]
    fn pairwise_ties_are_decided_by_faulted_cells() {
        assert!(outranks(0.5, 1, 0.5 + 5e-13, 2));
        assert!(!outranks(0.5 + 5e-13, 2, 0.5, 1));
        // Equal counts fall back to the strict comparison, as do scores outside the tolerance.
        assert!(!outranks(0.5, 1, 0.5, 1));
        assert!(outranks(0.5 + 5e-13, 1, 0.5, 1));
        assert!(outranks(0.6, 9, 0.5, 0));
        assert!(!outranks(0.5, 0, 0.6, 9));
    }
}
