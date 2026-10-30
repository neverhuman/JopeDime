//! GP generation recorder (TELEMETRY_SPEC rows G1-G12 and the E2 status split).
//!
//! The flywheel's evolution loop calls `add` with each evaluator result and `finish` at the end
//! of a generation; the summary (diversity, entropy, novelty) comes from
//! [`crate::api::GpSession::summarize`]. Results come from
//! `research/flywheel_selfplay/rust_evaluator.py`: `status` is `completed` or `failed`
//! (timeouts, nonzero exits and parse failures are not told apart there), `fitness` is
//! lower-is-better with `100.0` for failures, and `proj16` is a 16-D float projection whose
//! signs are packed into bits here.

use crate::api::{CandidateIn, GenerationIn, GpSession};
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const GEN_KIND: &str = "jope.telemetry.gp-generation/v1";

/// Bit i is set when component i of the 16-D projection is positive.
pub fn proj16_bits(proj16: &[f64]) -> Option<u16> {
    if proj16.len() != 16 {
        return None;
    }
    Some(
        proj16
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .fold(0u16, |bits, (i, _)| bits | (1 << i)),
    )
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|f| f.is_finite())
}

/// The G4 proxy available today: the candidate's error profile on its own context.
pub fn error_profile(
    cv_mean_log_ratio: Option<f64>,
    cv_tail_log_ratio: Option<f64>,
    cv_log_ratio_sd: Option<f64>,
    test_rmse_ratio: Option<f64>,
) -> Vec<Option<f64>> {
    let test = finite(test_rmse_ratio);
    vec![
        finite(cv_mean_log_ratio),
        finite(cv_tail_log_ratio),
        finite(cv_log_ratio_sd),
        test.filter(|t| *t > 0.0).map(f64::ln),
    ]
}

/// `completed` stays as is; a failed result is split by its return code when the worker
/// recorded one, and stays `failed` (cause unknown) otherwise.
pub fn status_of(status: &str, rc: Option<i64>) -> &'static str {
    if status == "completed" {
        return "completed";
    }
    match rc {
        None => "failed",
        Some(-1) => "timeout",
        Some(0) => "parse_fail",
        Some(_) => "nonzero_rc",
    }
}

pub fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One evaluator result, already read out of its dict.
#[derive(Clone, Debug, Default)]
pub struct EvalResult {
    pub dsl: String,
    pub program_sha256: Option<String>,
    pub operator: String,
    pub status: String,
    pub rc: Option<i64>,
    pub cv_gate_passed: bool,
    pub fitness: f64,
    pub complexity: Option<u32>,
    pub proj16: Option<Vec<f64>>,
    pub cv_mean_log_ratio: Option<f64>,
    pub cv_tail_log_ratio: Option<f64>,
    pub cv_log_ratio_sd: Option<f64>,
    pub test_rmse_ratio: Option<f64>,
}

/// Collects one generation of one context. `bank` and `corpus` are the program hashes in the
/// champion bank and the corpus at round start; a candidate in neither is `novel`.
pub struct GenerationRecorder {
    pub context: String,
    pub round: u32,
    bank: HashSet<String>,
    corpus: HashSet<String>,
    cands: Vec<CandidateIn>,
    proposals: u64,
    seen_rejects: u64,
}

impl GenerationRecorder {
    pub fn new(
        context: String,
        round: u32,
        bank: impl IntoIterator<Item = String>,
        corpus: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            context,
            round,
            bank: bank.into_iter().collect(),
            corpus: corpus.into_iter().collect(),
            cands: Vec::new(),
            proposals: 0,
            seen_rejects: 0,
        }
    }

    /// Proposal-time counters for G2: candidates proposed, and those dropped by `seen_dsl`.
    pub fn proposals(&mut self, n: u64, seen_rejects: u64) {
        self.proposals += n;
        self.seen_rejects += seen_rejects;
    }

    pub fn provenance(&self, program_sha256: &str) -> &'static str {
        if self.bank.contains(program_sha256) {
            "bank"
        } else if self.corpus.contains(program_sha256) {
            "replayed"
        } else {
            "novel"
        }
    }

    pub fn add(
        &mut self,
        r: EvalResult,
        parent_dsl: Option<String>,
        improved: Option<bool>,
        hwm_winner: bool,
        behavior: Option<Vec<Option<f64>>>,
    ) {
        let sha = r
            .program_sha256
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| sha256_hex(&r.dsl));
        let behavior = behavior.unwrap_or_else(|| {
            error_profile(
                r.cv_mean_log_ratio,
                r.cv_tail_log_ratio,
                r.cv_log_ratio_sd,
                r.test_rmse_ratio,
            )
        });
        self.cands.push(CandidateIn {
            provenance: self.provenance(&sha).to_owned(),
            dsl: r.dsl,
            parent_dsl,
            operator: r.operator,
            status: status_of(&r.status, r.rc).to_owned(),
            cv_gate_passed: r.cv_gate_passed,
            fitness: r.fitness,
            improved,
            hwm_winner,
            complexity: r.complexity,
            proj16_bits: r.proj16.as_deref().and_then(proj16_bits),
            behavior,
        });
    }

    pub fn finish(
        &mut self,
        generation: u32,
        session: &mut GpSession,
    ) -> Result<crate::gp::GenSummary> {
        let g = GenerationIn {
            context: self.context.clone(),
            round: self.round,
            generation,
            proposals: self.proposals,
            seen_rejects: self.seen_rejects,
            candidates: std::mem::take(&mut self.cands),
        };
        self.proposals = 0;
        self.seen_rejects = 0;
        session.summarize(&g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_the_python_hooks() {
        let mut p = vec![-1.0; 16];
        p[0] = 0.5;
        p[15] = 2.0;
        assert_eq!(proj16_bits(&p), Some(0x8001));
        assert_eq!(proj16_bits(&[1.0; 3]), None);
        assert_eq!(status_of("completed", Some(-1)), "completed");
        assert_eq!(status_of("failed", None), "failed");
        assert_eq!(status_of("failed", Some(-1)), "timeout");
        assert_eq!(status_of("failed", Some(2)), "nonzero_rc");
        assert_eq!(status_of("failed", Some(0)), "parse_fail");
        let e = error_profile(Some(0.1), Some(f64::NAN), None, Some(1.0));
        assert_eq!(e, vec![Some(0.1), None, None, Some(0.0)]);
        assert_eq!(error_profile(None, None, None, Some(-1.0))[3], None);
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn summarizes_a_generation_with_provenance() {
        let mut session = GpSession::new(None, Some(7), false);
        let bank_dsl = "ttsln";
        let mut rec =
            GenerationRecorder::new("ctx".into(), 2, [sha256_hex(bank_dsl)], std::iter::empty());
        rec.proposals(4, 1);
        rec.add(
            EvalResult {
                dsl: bank_dsl.into(),
                status: "completed".into(),
                fitness: 0.5,
                ..Default::default()
            },
            None,
            Some(true),
            false,
            None,
        );
        rec.add(
            EvalResult {
                dsl: "ttsln | clip".into(),
                status: "failed".into(),
                rc: Some(-1),
                fitness: 100.0,
                ..Default::default()
            },
            Some(bank_dsl.into()),
            Some(false),
            false,
            None,
        );
        let s = rec.finish(3, &mut session).unwrap();
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["generation"], 3);
        assert_eq!(v["round"], 2);
        assert!(rec.cands.is_empty() && rec.proposals == 0);
    }
}
