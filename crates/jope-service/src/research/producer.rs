//! CPU capture control; the caller still owns corpus admission and durable commit.

use super::{RecordedOutcome, RecordedProgram, ResearchRound, Rung0Measurement};
use anyhow::{ensure, Context, Result};
use std::time::Instant;

/// Evaluate an observed slate without changing its behavior-policy admission.
///
/// Inputs must contain actual producer observations and no dispatched outcomes.
/// An evaluator error or unwind leaves an unresolved attempt: retain this object's
/// inspection evidence, and never turn that unknown outcome into a training label.
/// This object does not persist attempts or charge the discovery store's budget.
pub struct RoundCapture {
    round: ResearchRound,
    population: Vec<usize>,
    max_attempts: usize,
    unresolved: bool,
}

impl RoundCapture {
    pub fn new(input: ResearchRound, max_attempts: usize) -> Result<Self> {
        ensure!(
            input.dispatched.is_empty() && input.outcomes.is_empty(),
            "capture input already contains evaluation attempts"
        );
        input.validate()?;
        let population = if input.evaluate_all_proposals {
            (0..input.proposals.len()).collect()
        } else {
            input.policy.admitted.clone()
        };
        Ok(Self {
            round: input,
            population,
            max_attempts,
            unresolved: false,
        })
    }

    /// Reserve one attempt, then call the evaluator and measure its wall time.
    ///
    /// Explicit worker failures are observations and consume an attempt. Return
    /// `Err` for an unknown/aborted result; it blocks further calls and `finish`.
    /// `Ok(false)` means the population or attempt budget is exhausted.
    pub fn evaluate_next(
        &mut self,
        evaluator: impl FnOnce(usize, &RecordedProgram) -> Result<Rung0Measurement>,
    ) -> Result<bool> {
        ensure!(
            !self.unresolved,
            "capture has an unresolved evaluation attempt"
        );
        let next = self.round.dispatched.len();
        if next >= self.max_attempts || next >= self.population.len() {
            return Ok(false);
        }
        let index = self.population[next];
        self.round.dispatched.push(index);
        self.unresolved = true;
        let start = Instant::now();
        let measurement = evaluator(index, &self.round.proposals[index])?;
        let wall_ms = u64::try_from(start.elapsed().as_millis())
            .context("evaluation wall time exceeds record range")?;
        if let Rung0Measurement::Success {
            surrogate,
            objective,
            ..
        } = &measurement
        {
            ensure!(
                surrogate.is_finite() && objective.is_finite(),
                "nonfinite successful measurement"
            );
        }
        self.round.outcomes.push(RecordedOutcome {
            proposal_index: index,
            wall_ms,
            measurement,
        });
        self.unresolved = false;
        Ok(true)
    }

    /// Actual reserved attempts, including an unresolved last attempt on error.
    pub fn attempted_indices(&self) -> &[usize] {
        &self.round.dispatched
    }

    /// Only actual returned measurements; unknown attempts have no fabricated row.
    pub fn outcomes(&self) -> &[RecordedOutcome] {
        &self.round.outcomes
    }

    /// Finish a complete or explicitly budget-truncated capture.
    ///
    /// The result must be committed with the discovery checkpoint before it is
    /// appended to `RecordWriter`. Finishing alone does not establish that commit.
    pub fn finish(mut self) -> Result<ResearchRound> {
        ensure!(
            !self.unresolved,
            "capture has an unresolved evaluation attempt"
        );
        self.round.capture_complete = self.round.dispatched.len() == self.population.len();
        self.round.validate()?;
        Ok(self.round)
    }
}
