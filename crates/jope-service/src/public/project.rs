//! Projection of internal evaluation evidence into public events (spec §8.1–8.4).
//!
//! Inputs are internal views supplied by the discovery loop; they may carry internal
//! ordinals and flags, but leave this module only through `jope_api` bodies, which have
//! no fields for programs, operators, lineage, rounds or population indices.

use super::phrase::{
    lint_public_phrase, render_phrase, BannedWordError, CatalogEntry, EventDetails, Template,
};
use jope_api::{
    ids::{AlgorithmId, RunId},
    ComparisonEvidence, EventBody, ExperimentStartedBody, Language, ReasoningBody,
};
use sha2::{Digest, Sha256};

/// SHA-256 identity of the problem: dataset matrix, target, task and split policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProblemDigest(pub [u8; 32]);

/// SHA-256 identity of the fold assignment a score was computed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldsDigest(pub [u8; 32]);

/// Everything two scores must share before they may be compared (§8.1 rule 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreContext {
    pub problem: ProblemDigest,
    pub folds: FoldsDigest,
    /// Objective metric name, for example `rmse`.
    pub metric: String,
    /// Public evaluation scope, for example `cv_fit_5fold`.
    pub scope: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub context: ScoreContext,
    pub value: f64,
}

/// Position of an evaluation within its run. Internal only: public experiment IDs are
/// a one-way digest of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationOrdinal(pub u64);

/// A proposal admitted for evaluation, with the catalog wording for its change.
#[derive(Clone, Copy, Debug)]
pub struct AdmittedProposal<'a> {
    pub ordinal: EvaluationOrdinal,
    pub template: Template,
    pub event_digest: [u8; 32],
    pub entry: CatalogEntry<'a>,
    pub details: EventDetails<'a>,
}

/// An evaluation that finished. Rung 0 is screening; a measured claim needs rung ≥ 1.
#[derive(Clone, Debug)]
pub struct CompletedEvaluation {
    pub ordinal: EvaluationOrdinal,
    pub rung: u8,
    pub score: Score,
}

/// The champion recorded by the store. Only a committed champion supports a claim.
#[derive(Clone, Debug)]
pub struct Champion<'a> {
    pub algorithm_id: AlgorithmId,
    pub head: CatalogEntry<'a>,
    pub committed: bool,
    pub score: Score,
}

/// Finalization output: a holdout score on rows the search never used.
#[derive(Clone, Debug)]
pub struct Finalization {
    pub holdout: Score,
    pub holdout_rows: u64,
    pub excluded_from_search: bool,
}

/// Outcome of comparing exported predictors with the engine on check rows.
#[derive(Clone, Copy, Debug)]
pub struct ParityReceipt<'a> {
    pub languages: &'a [Language],
    pub check_rows: u64,
    pub within_tolerance: bool,
}

/// A public claim and the evidence its level requires (§8.2).
#[derive(Clone, Debug)]
pub enum Claim<'a> {
    Hypothesis(AdmittedProposal<'a>),
    Measured {
        evaluation: CompletedEvaluation,
        best: Option<Score>,
    },
    Accepted {
        champion: Champion<'a>,
        previous: Option<Score>,
    },
    Final(Finalization),
    Export(ParityReceipt<'a>),
    /// A gap in the phrasing catalog: never an invented mechanism.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionError {
    /// A score claimed about this run was computed on a different problem.
    ProblemMismatch,
    /// The evidence does not support the claimed level.
    MissingEvidence(&'static str),
    NonFiniteScore,
    Wording(BannedWordError),
}

impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProblemMismatch => write!(f, "score belongs to a different problem"),
            Self::MissingEvidence(what) => write!(f, "claim requires {what}"),
            Self::NonFiniteScore => write!(f, "score must be finite"),
            Self::Wording(error) => write!(f, "banned public word: {}", error.word),
        }
    }
}
impl std::error::Error for ProjectionError {}

impl From<BannedWordError> for ProjectionError {
    fn from(error: BannedWordError) -> Self {
        Self::Wording(error)
    }
}

/// Projects one run's internal evidence into public event bodies.
#[derive(Clone, Debug)]
pub struct Projector {
    problem: ProblemDigest,
}

impl Projector {
    pub fn new(_run: RunId, problem: ProblemDigest) -> Self {
        Self { problem }
    }

    /// Run-independent opaque ID from the problem and ordinal (owner R2, §5.6).
    /// The one-way digest never encodes lineage, round, slot or parents (§8.1).
    pub fn experiment_id(&self, ordinal: EvaluationOrdinal) -> String {
        let mut hash = Sha256::new();
        hash.update(b"jope.public-experiment/v1\0");
        hash.update(self.problem.0);
        hash.update([0]);
        hash.update(ordinal.0.to_be_bytes());
        let digest = hash.finalize();
        let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        format!("exp_{hex}")
    }

    pub fn experiment_started(
        &self,
        proposal: &AdmittedProposal<'_>,
    ) -> Result<EventBody, ProjectionError> {
        Ok(EventBody::ExperimentStarted(ExperimentStartedBody {
            experiment_id: self.experiment_id(proposal.ordinal),
            hypothesis: hypothesis(proposal)?,
        }))
    }

    pub fn project(&self, claim: Claim<'_>) -> Result<EventBody, ProjectionError> {
        let body = match claim {
            Claim::Hypothesis(proposal) => {
                reasoning("hypothesis", None, hypothesis(&proposal)?, None)
            }
            Claim::Measured { evaluation, best } => {
                if evaluation.rung == 0 {
                    return Err(ProjectionError::MissingEvidence(
                        "a completed rung-1 evaluation",
                    ));
                }
                self.own(&evaluation.score)?;
                let evidence = compare(&evaluation.score, best.as_ref())?;
                let summary = match evidence.before {
                    Some(before) => format!(
                        "Cross-validated {} {} vs best {} on the same folds.",
                        evidence.metric,
                        value(evaluation.score.value),
                        value(before)
                    ),
                    None => format!(
                        "Cross-validated {} {}; comparison unavailable.",
                        evidence.metric,
                        value(evaluation.score.value)
                    ),
                };
                reasoning("measured", None, summary, Some(evidence))
            }
            Claim::Accepted { champion, previous } => {
                if !champion.committed {
                    return Err(ProjectionError::MissingEvidence("a committed champion"));
                }
                self.own(&champion.score)?;
                let evidence = compare(&champion.score, previous.as_ref())?;
                let after = champion.score.value;
                let change = match evidence.before {
                    Some(before) if before != 0.0 => {
                        let percent = (after / before.abs() - before.signum()) * 100.0;
                        if percent.is_finite() {
                            format!(", {percent:+.1}%")
                        } else {
                            String::new()
                        }
                    }
                    Some(_) => String::new(),
                    None => "; comparison unavailable".to_owned(),
                };
                let summary = format!(
                    "New best: {} (CV {} {}{change}).",
                    champion.head.public_name,
                    evidence.metric,
                    value(after)
                );
                reasoning(
                    "accepted",
                    Some(champion.algorithm_id),
                    summary,
                    Some(evidence),
                )
            }
            Claim::Final(finalization) => {
                if finalization.holdout_rows == 0 || !finalization.excluded_from_search {
                    return Err(ProjectionError::MissingEvidence(
                        "a holdout score on rows not used during search",
                    ));
                }
                self.own(&finalization.holdout)?;
                let summary = format!(
                    "Holdout {} {} on {} rows not used during search.",
                    finalization.holdout.context.metric,
                    value(finalization.holdout.value),
                    finalization.holdout_rows
                );
                reasoning("final", None, summary, None)
            }
            Claim::Export(receipt) => {
                if !receipt.within_tolerance
                    || receipt.languages.is_empty()
                    || receipt.check_rows == 0
                {
                    return Err(ProjectionError::MissingEvidence(
                        "a passing parity receipt on check rows",
                    ));
                }
                let summary = format!(
                    "{} {} the engine within the release tolerance on {} check rows.",
                    language_list(receipt.languages),
                    if receipt.languages.len() == 1 {
                        "predictor matches"
                    } else {
                        "predictors match"
                    },
                    receipt.check_rows
                );
                reasoning("export", None, summary, None)
            }
            Claim::Unknown => reasoning(
                "unknown",
                None,
                "Evaluating a new algorithm variant.".to_owned(),
                None,
            ),
        };
        lint_public_phrase(&body.summary)?;
        Ok(EventBody::Reasoning(body))
    }

    /// The ProblemDigest check: claims about this run must use this run's problem.
    fn own(&self, score: &Score) -> Result<(), ProjectionError> {
        if !score.value.is_finite() {
            return Err(ProjectionError::NonFiniteScore);
        }
        if score.context.problem != self.problem {
            return Err(ProjectionError::ProblemMismatch);
        }
        Ok(())
    }
}

/// Compare only when problem, folds, metric and scope all match (§8.1 rule 4).
pub fn compare(
    current: &Score,
    baseline: Option<&Score>,
) -> Result<ComparisonEvidence, ProjectionError> {
    if !current.value.is_finite() || baseline.is_some_and(|b| !b.value.is_finite()) {
        return Err(ProjectionError::NonFiniteScore);
    }
    let comparable = baseline.filter(|b| b.context == current.context);
    Ok(ComparisonEvidence {
        metric: current.context.metric.clone(),
        scope: current.context.scope.clone(),
        before: comparable.map(|b| b.value),
        after: Some(current.value),
        comparable: comparable.is_some(),
    })
}

fn hypothesis(proposal: &AdmittedProposal<'_>) -> Result<String, ProjectionError> {
    Ok(render_phrase(
        proposal.template,
        &proposal.event_digest,
        proposal.entry,
        proposal.details,
    )?)
}

fn reasoning(
    stage: &str,
    algorithm_id: Option<AlgorithmId>,
    summary: String,
    evidence: Option<ComparisonEvidence>,
) -> ReasoningBody {
    ReasoningBody {
        stage: stage.to_owned(),
        algorithm_id,
        summary,
        evidence,
    }
}

fn value(value: f64) -> String {
    format!("{value:.3}")
}

fn language_list(languages: &[Language]) -> String {
    let names: Vec<&str> = languages
        .iter()
        .map(|language| match language {
            Language::Python => "Python",
            Language::Rust => "Rust",
            Language::Javascript => "JavaScript",
            Language::Typescript => "TypeScript",
        })
        .collect();
    match names.split_last() {
        Some((last, [])) => (*last).to_owned(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

// Compile-time pin: the projector can only emit the reviewed public `jope_api` types.
const _: fn(&Projector, Claim<'static>) -> Result<EventBody, ProjectionError> = Projector::project;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const RUN: &str = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0";
    const PROBLEM: ProblemDigest = ProblemDigest([7; 32]);

    fn projector() -> Projector {
        Projector::new(RUN.parse().unwrap(), PROBLEM)
    }
    fn context() -> ScoreContext {
        ScoreContext {
            problem: PROBLEM,
            folds: FoldsDigest([3; 32]),
            metric: "rmse".into(),
            scope: "cv_fit_5fold".into(),
        }
    }
    fn score(value: f64) -> Score {
        Score {
            context: context(),
            value,
        }
    }
    fn entry() -> CatalogEntry<'static> {
        CatalogEntry {
            public_name: "log scaling",
            purpose: "reduce skew in income features",
        }
    }
    fn proposal(ordinal: u64) -> AdmittedProposal<'static> {
        AdmittedProposal {
            ordinal: EvaluationOrdinal(ordinal),
            template: Template::StageInsert,
            event_digest: [ordinal as u8; 32],
            entry: entry(),
            details: EventDetails::default(),
        }
    }
    fn champion(committed: bool, value: f64) -> Champion<'static> {
        Champion {
            algorithm_id: "alg_0007".parse().unwrap(),
            head: CatalogEntry {
                public_name: "gradient-boosted trees",
                purpose: "capture interactions",
            },
            committed,
            score: score(value),
        }
    }
    fn reasoning_of(body: EventBody) -> ReasoningBody {
        match body {
            EventBody::Reasoning(body) => body,
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn every_claim_level_requires_its_evidence() {
        let p = projector();
        let measured = |rung| Claim::Measured {
            evaluation: CompletedEvaluation {
                ordinal: EvaluationOrdinal(1),
                rung,
                score: score(0.493),
            },
            best: Some(score(0.487)),
        };
        assert_eq!(
            p.project(measured(0)).unwrap_err(),
            ProjectionError::MissingEvidence("a completed rung-1 evaluation")
        );
        assert_eq!(
            reasoning_of(p.project(measured(1)).unwrap()).stage,
            "measured"
        );

        let accepted = |committed| Claim::Accepted {
            champion: champion(committed, 0.487),
            previous: Some(score(0.512)),
        };
        assert_eq!(
            p.project(accepted(false)).unwrap_err(),
            ProjectionError::MissingEvidence("a committed champion")
        );
        let body = reasoning_of(p.project(accepted(true)).unwrap());
        assert_eq!(body.stage, "accepted");
        assert_eq!(body.algorithm_id.unwrap().to_string(), "alg_0007");
        assert_eq!(
            body.summary,
            "New best: gradient-boosted trees (CV rmse 0.487, -4.9%)."
        );

        let finalization = |rows, excluded| {
            Claim::Final(Finalization {
                holdout: score(0.491),
                holdout_rows: rows,
                excluded_from_search: excluded,
            })
        };
        for (rows, excluded) in [(0, true), (4128, false)] {
            assert!(matches!(
                p.project(finalization(rows, excluded)).unwrap_err(),
                ProjectionError::MissingEvidence(_)
            ));
        }
        assert_eq!(
            reasoning_of(p.project(finalization(4128, true)).unwrap()).summary,
            "Holdout rmse 0.491 on 4128 rows not used during search."
        );

        static BOTH: [Language; 2] = [Language::Python, Language::Rust];
        let export = |languages: &'static [Language], rows, within| {
            Claim::Export(ParityReceipt {
                languages,
                check_rows: rows,
                within_tolerance: within,
            })
        };
        for claim in [
            export(&BOTH, 64, false),
            export(&[], 64, true),
            export(&BOTH, 0, true),
        ] {
            assert!(matches!(
                p.project(claim).unwrap_err(),
                ProjectionError::MissingEvidence(_)
            ));
        }
        assert_eq!(
            reasoning_of(p.project(export(&BOTH, 64, true)).unwrap()).summary,
            "Python and Rust predictors match the engine within the release tolerance on 64 check rows."
        );

        let hypothesis = reasoning_of(p.project(Claim::Hypothesis(proposal(5))).unwrap());
        assert_eq!(hypothesis.stage, "hypothesis");
        assert!(hypothesis.summary.contains("log scaling"));
        let unknown = reasoning_of(p.project(Claim::Unknown).unwrap());
        assert_eq!(
            (unknown.stage.as_str(), unknown.summary.as_str()),
            ("unknown", "Evaluating a new algorithm variant.")
        );
    }

    #[test]
    fn accepted_percentage_handles_finite_extremes() {
        for (before, after, percentage) in [
            (1e308, -1e308, Some("-200.0%")),
            (-1e308, 1e308, Some("+200.0%")),
            (f64::MIN_POSITIVE, f64::MAX, None),
        ] {
            let body = reasoning_of(
                projector()
                    .project(Claim::Accepted {
                        champion: champion(true, after),
                        previous: Some(score(before)),
                    })
                    .unwrap(),
            );
            assert!(!body.summary.contains("inf") && !body.summary.contains("NaN"));
            match percentage {
                Some(expected) => assert!(body.summary.contains(expected)),
                None => assert!(!body.summary.contains('%')),
            }
            let evidence = body.evidence.unwrap();
            assert!(evidence.comparable);
            assert_eq!(
                (evidence.before, evidence.after),
                (Some(before), Some(after))
            );
        }
    }

    #[test]
    fn mismatched_comparisons_are_unavailable_and_matching_ones_compare() {
        let current = score(0.493);
        let matching = compare(&current, Some(&score(0.487))).unwrap();
        assert!(matching.comparable);
        assert_eq!(
            (matching.before, matching.after),
            (Some(0.487), Some(0.493))
        );

        let mut baselines = Vec::new();
        for change in 0..4 {
            let mut context = context();
            match change {
                0 => context.problem = ProblemDigest([8; 32]),
                1 => context.folds = FoldsDigest([4; 32]),
                2 => context.metric = "mae".into(),
                _ => context.scope = "cv_fit_3fold".into(),
            }
            baselines.push(Score {
                context,
                value: 0.487,
            });
        }
        for baseline in baselines.iter().map(Some).chain([None]) {
            let evidence = compare(&current, baseline).unwrap();
            assert!(!evidence.comparable);
            assert_eq!(evidence.before, None);
            let body = reasoning_of(
                projector()
                    .project(Claim::Measured {
                        evaluation: CompletedEvaluation {
                            ordinal: EvaluationOrdinal(2),
                            rung: 1,
                            score: current.clone(),
                        },
                        best: baseline.cloned(),
                    })
                    .unwrap(),
            );
            assert_eq!(
                body.summary,
                "Cross-validated rmse 0.493; comparison unavailable."
            );
            assert!(!body.evidence.unwrap().comparable);
        }
        let accepted = reasoning_of(
            projector()
                .project(Claim::Accepted {
                    champion: champion(true, 0.487),
                    previous: Some(baselines[1].clone()),
                })
                .unwrap(),
        );
        assert!(accepted.summary.ends_with("; comparison unavailable)."));
        assert!(compare(&score(f64::NAN), None).is_err());
    }

    #[test]
    fn claims_about_this_run_must_use_its_problem_digest() {
        let mut foreign = score(0.5);
        foreign.context.problem = ProblemDigest([9; 32]);
        let p = projector();
        assert_eq!(
            p.project(Claim::Measured {
                evaluation: CompletedEvaluation {
                    ordinal: EvaluationOrdinal(3),
                    rung: 1,
                    score: foreign.clone(),
                },
                best: None,
            })
            .unwrap_err(),
            ProjectionError::ProblemMismatch
        );
        let mut champion = champion(true, 0.5);
        champion.score = foreign.clone();
        assert_eq!(
            p.project(Claim::Accepted {
                champion,
                previous: None
            })
            .unwrap_err(),
            ProjectionError::ProblemMismatch
        );
        assert_eq!(
            p.project(Claim::Final(Finalization {
                holdout: foreign,
                holdout_rows: 10,
                excluded_from_search: true,
            }))
            .unwrap_err(),
            ProjectionError::ProblemMismatch
        );
    }

    #[test]
    fn experiment_ids_are_opaque_deterministic_and_unique() {
        let p = projector();
        let other = Projector::new("run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap(), PROBLEM);
        let other_problem = Projector::new(RUN.parse().unwrap(), ProblemDigest([8; 32]));
        let mut seen = BTreeSet::new();
        for ordinal in 0..10_000 {
            let id = p.experiment_id(EvaluationOrdinal(ordinal));
            assert_eq!(id.len(), 20);
            assert!(id.starts_with("exp_"));
            assert!(id[4..]
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
            assert_ne!(id[4..], format!("{ordinal:016x}"));
            assert_eq!(id, p.experiment_id(EvaluationOrdinal(ordinal)));
            assert_eq!(id, other.experiment_id(EvaluationOrdinal(ordinal)));
            assert_ne!(id, other_problem.experiment_id(EvaluationOrdinal(ordinal)));
            assert!(seen.insert(id));
        }
        match p.experiment_started(&proposal(9)).unwrap() {
            EventBody::ExperimentStarted(body) => {
                assert_eq!(body.experiment_id, p.experiment_id(EvaluationOrdinal(9)));
                assert!(body.hypothesis.contains("log scaling"));
            }
            other => panic!("expected experiment.started, got {other:?}"),
        }
    }

    #[test]
    fn banned_wording_is_refused() {
        let mut banned = proposal(1);
        banned.entry.public_name = "parent pipeline";
        assert!(matches!(
            projector().project(Claim::Hypothesis(banned)),
            Err(ProjectionError::Wording(_))
        ));
        assert!(projector().experiment_started(&banned).is_err());
    }

    /// Type inspection over the live `jope_api` schemas: no public payload has a
    /// field for programs, operators, lineage or search positions (§8.1 rule 2).
    #[test]
    fn no_public_payload_schema_has_a_field_for_programs_lineage_or_search_positions() {
        const FORBIDDEN: &[&str] = &[
            "dsl",
            "program",
            "operator",
            "parent",
            "lineage",
            "round",
            "population",
            "elite",
            "genome",
            "generation",
            "slot",
            "attempt",
            "island",
            "offspring",
        ];
        // The planned pipeline depth R in `Plan`: a count, not a round index.
        const ALLOWED: &[&str] = &["rounds_in_flight"];
        fn collect(value: &serde_json::Value, names: &mut BTreeSet<String>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, child) in map {
                        if key == "properties" {
                            if let serde_json::Value::Object(properties) = child {
                                names.extend(properties.keys().cloned());
                            }
                        }
                        collect(child, names);
                    }
                }
                serde_json::Value::Array(items) => items.iter().for_each(|v| collect(v, names)),
                _ => {}
            }
        }
        let mut names = BTreeSet::new();
        let mut inspected = BTreeSet::new();
        for operation in jope_api::registry::OPERATIONS {
            for payload in operation
                .request_types
                .iter()
                .chain(operation.response_types)
            {
                if let Ok(schema) = payload.resolve_schema() {
                    inspected.insert(payload.name);
                    collect(&serde_json::to_value(schema).unwrap(), &mut names);
                }
            }
        }
        for required in ["PublicEvent", "TelemetryEvent", "RunSnapshot", "Algorithm"] {
            assert!(
                inspected.contains(required),
                "{required} schema not inspected"
            );
        }
        assert!(names.contains("summary") && names.contains("experiment_id"));
        for name in names.iter().filter(|n| !ALLOWED.contains(&n.as_str())) {
            for token in name.to_ascii_lowercase().split('_') {
                for forbidden in FORBIDDEN {
                    assert!(
                        token != *forbidden && token != format!("{forbidden}s"),
                        "public field `{name}` can carry `{forbidden}`"
                    );
                }
            }
        }
    }
}
