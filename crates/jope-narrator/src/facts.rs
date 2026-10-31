//! `jope.lineage-fact/v1`: what the search tells the narrator, as owned values.
//!
//! A fact is a copy of something the search already decided or measured. Nothing here is read
//! back by the search, and a fact never carries an instruction. Programs travel as canonical
//! text plus the sha256 of that text, the same identity `Observation::program_sha256` and the
//! search-event log use, so a fact joins to the published evidence without a lookup table.
//!
//! Privacy follows `jope.invent.progress.v1`'s rule for what leaves the process without opt-in:
//! no column names and no cell values. Programs refer to columns by index only (`c3`), which is
//! already how canonical program text spells them.

use serde::{Deserialize, Serialize};

/// The schema of one journaled fact envelope ([`Envelope`]).
pub const FACT_SCHEMA: &str = "jope.lineage-fact/v1";

/// A program by identity and canonical text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramRef {
    /// Lowercase hex sha256 of `canonical`'s bytes.
    pub sha256: String,
    /// Canonical program text whose bytes produce `sha256`.
    pub canonical: String,
}

/// One elite carried into the next generation, with the measurement it was ranked by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Survivor {
    /// Lowercase hex sha256 identity of the carried program.
    pub program_sha256: String,
    /// Held-out RMSE on this cut, in target units.
    pub rmse: f64,
    /// What the search ranks by: the objective's loss, or the RMSE itself. Lower is better.
    pub score: f64,
}

/// What happened. Every variant names its cut, so one stream can carry a multi-cut run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LineageFact {
    /// One cut's exploration begins. Constants are the search's own, copied, not restated.
    SearchStarted {
        cut: usize,
        engine: String,
        train_rows: usize,
        hold_rows: usize,
        features: usize,
        population: usize,
        elites: usize,
        max_transforms: usize,
        islands: usize,
        migration_interval: usize,
        /// This cut's exploration allowance: half the budget when comparison is reserved.
        exploration_evaluations: usize,
        exploration_seconds: f64,
        #[serde(with = "crate::decimal")]
        rng_seed: u64,
        /// The regression objective's name when the search ranks by one; `None` ranks by RMSE.
        objective: Option<String>,
        /// Generation-0 seeds supplied by the caller (`incumbent` origin), as canonical text.
        incumbents: usize,
    },
    /// A program was proposed: every draw, including duplicates and operator failures, in the
    /// order the search drew them. Emitted before anything about the child is measured.
    Drawn {
        cut: usize,
        /// `exploration`, `migration` or `comparison`: the first segment of `coordinate`.
        phase: String,
        /// The search-event coordinate of this draw, verbatim.
        coordinate: String,
        generation: usize,
        island: usize,
        /// Population slot, when the draw fills one.
        slot: Option<usize>,
        /// Distinct-child attempt within the slot (0-based), when the draw retries.
        attempt: Option<u32>,
        /// `bootstrap`, `incumbent`, `random`, `mutate`, `crossover`, `random_fallback`,
        /// `migration` or `remeasure`.
        origin: String,
        /// The operator's stream seed; with the parents it replays the child exactly.
        #[serde(with = "crate::decimal_option")]
        seed: Option<u64>,
        /// Ordered parents; a crossover lists both choices even when they are the same program.
        parents: Vec<ProgramRef>,
        /// The proposed program; `None` when the operator failed.
        child: Option<ProgramRef>,
        /// `candidate`, `duplicate`, `operator_failed`, `migration_admitted` or
        /// `migration_rejected`.
        outcome: String,
        detail: Option<String>,
    },
    /// The admission decision for one drawn program, in slot order.
    Admitted {
        cut: usize,
        generation: usize,
        island: usize,
        program_sha256: String,
        /// `admitted`, `duplicate`, `evaluations`, `seconds` or `cancelled`.
        disposition: String,
    },
    /// One finished evaluation, exactly as `observations.json` will publish it (minus the text,
    /// which the `Drawn` fact already carried).
    Measured {
        cut: usize,
        /// `exploration` or `comparison`.
        phase: String,
        generation: usize,
        island: Option<usize>,
        program_sha256: String,
        origin: String,
        /// `measured`, `parse_failed`, `fit_failed`, `score_failed` or `panicked`.
        status: String,
        rmse: Option<f64>,
        hold_r2: Option<f64>,
        train_r2: Option<f64>,
        objective_loss: Option<f64>,
        /// Wall seconds of the fit; a real measurement, so two runs never agree on it.
        seconds: f64,
        failure: Option<String>,
    },
    /// The elites one island carries into the next generation, best first.
    GenerationClosed {
        cut: usize,
        generation: usize,
        island: usize,
        survivors: Vec<Survivor>,
        /// Evaluations spent on this cut so far.
        evaluations: usize,
        /// Consecutive generations that spent no evaluation (the search stops at its limit).
        barren: usize,
    },
    /// An island's champion offered to another island's elites.
    Migrated {
        cut: usize,
        after_generation: usize,
        from_island: usize,
        to_island: usize,
        program_sha256: String,
        rmse: f64,
        admitted: bool,
    },
    /// Exploration on one cut ended, and why.
    ExplorationStopped {
        cut: usize,
        /// `evaluations`, `seconds` or `search_space_exhausted`.
        reason: String,
        generations: usize,
        evaluations: usize,
    },
    /// The cross-cut comparison pool, before any of it is re-measured.
    ComparisonStarted {
        cuts: Vec<usize>,
        program_sha256s: Vec<String>,
    },
    /// The tap could not keep up and discarded this many facts since the last one delivered.
    /// Synthesized by the receiving side, never by the search.
    TapDropped {
        #[serde(with = "crate::decimal")]
        dropped: u64,
    },
    Started {
        cut: usize,
        phase: String,
        generation: usize,
        island: Option<usize>,
        program_sha256: String,
    },
    Selected {
        program_sha256: String,
        rule: String,
    },
    CandidatesOffered {
        program_sha256s: Vec<String>,
        rule: String,
    },
    Emitted {
        program_sha256: String,
        languages: Vec<String>,
    },
    RunStopped {
        status: String,
    },
    Prediction {
        status: String,
        program_sha256: Option<String>,
    },
    PredictionBound {
        status: String,
        model_sha256: String,
        input_sha256: String,
        rows: usize,
        seconds: Option<f64>,
    },
}

impl LineageFact {
    /// A cheap upper-bound-ish byte estimate for queue accounting, without serializing.
    pub fn approx_bytes(&self) -> usize {
        const BASE: usize = 192;
        let program = |p: &ProgramRef| p.sha256.len() + p.canonical.len() + 32;
        BASE + match self {
            LineageFact::Drawn {
                coordinate,
                parents,
                child,
                detail,
                ..
            } => {
                coordinate.len()
                    + parents.iter().map(program).sum::<usize>()
                    + child.as_ref().map_or(0, program)
                    + detail.as_ref().map_or(0, String::len)
            }
            LineageFact::Measured { failure, .. } => failure.as_ref().map_or(0, String::len) + 64,
            LineageFact::GenerationClosed { survivors, .. } => survivors.len() * 112,
            LineageFact::ComparisonStarted {
                program_sha256s, ..
            } => program_sha256s.len() * 68,
            _ => 64,
        }
    }

    /// The fact's `kind` tag, as serialized.
    pub fn kind(&self) -> &'static str {
        match self {
            LineageFact::SearchStarted { .. } => "search_started",
            LineageFact::Drawn { .. } => "drawn",
            LineageFact::Admitted { .. } => "admitted",
            LineageFact::Measured { .. } => "measured",
            LineageFact::GenerationClosed { .. } => "generation_closed",
            LineageFact::Migrated { .. } => "migrated",
            LineageFact::ExplorationStopped { .. } => "exploration_stopped",
            LineageFact::ComparisonStarted { .. } => "comparison_started",
            LineageFact::TapDropped { .. } => "tap_dropped",
            LineageFact::Started { .. } => "started",
            LineageFact::Selected { .. } => "selected",
            LineageFact::CandidatesOffered { .. } => "candidates_offered",
            LineageFact::Emitted { .. } => "emitted",
            LineageFact::RunStopped { .. } => "run_stopped",
            LineageFact::Prediction { .. } => "prediction",
            LineageFact::PredictionBound { .. } => "prediction_bound",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_round_trip_with_their_kind_tag() {
        let fact = LineageFact::Drawn {
            cut: 0,
            phase: "exploration".into(),
            coordinate: "exploration/3/0/5/1".into(),
            generation: 3,
            island: 0,
            slot: Some(5),
            attempt: Some(1),
            origin: "mutate".into(),
            seed: Some(42),
            parents: vec![ProgramRef {
                sha256: "a".repeat(64),
                canonical: "(pipe X (ridge))".into(),
            }],
            child: None,
            outcome: "operator_failed".into(),
            detail: Some("parser refused operator output".into()),
        };
        let text = serde_json::to_string(&fact).unwrap();
        assert!(text.starts_with("{\"kind\":\"drawn\""), "{text}");
        assert_eq!(serde_json::from_str::<LineageFact>(&text).unwrap(), fact);
        assert_eq!(fact.kind(), "drawn");
        assert!(fact.approx_bytes() > text.len() / 2);
    }

    #[test]
    fn unknown_fields_are_refused() {
        let bad = r#"{"kind":"tap_dropped","dropped":"1","extra":true}"#;
        assert!(serde_json::from_str::<LineageFact>(bad).is_err());
    }
}
