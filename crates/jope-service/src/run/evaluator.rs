use jope_api::{error::ErrorCode, ids::RunId, Effort};
use std::{thread, time::Duration};

/// The attempt is never reused, including after recovery of uncommitted rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptKey {
    pub run: RunId,
    pub round: u64,
    pub slot: u32,
    pub attempt: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationRequest {
    pub key: AttemptKey,
    pub seed: u64,
    /// Captured at round admission; subsequent effort changes do not alter it.
    pub effort: Effort,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EvaluationOutcome {
    Scored(f64),
    Failed(ErrorCode),
    /// Reserved interruption marker; rejected as a result and never committed.
    DiscardedAtPauseDeadline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub key: AttemptKey,
    pub outcome: EvaluationOutcome,
}

/// Blocking worker adapter; the caller chooses its execution/threading mechanism.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, request: &EvaluationRequest) -> Evaluation;
}

/// Controller fixture only. These seeded values are not dataset/model scores.
pub struct FakeEvaluator {
    delay: Duration,
}

impl FakeEvaluator {
    pub fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

impl Evaluator for FakeEvaluator {
    fn evaluate(&self, request: &EvaluationRequest) -> Evaluation {
        thread::sleep(self.delay);
        // SplitMix64: a pure seed projection, independent of attempt or arrival.
        let mut value = request.seed.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        Evaluation {
            key: request.key.clone(),
            outcome: EvaluationOutcome::Scored((value >> 11) as f64 / ((1_u64 << 53) as f64)),
        }
    }
}
