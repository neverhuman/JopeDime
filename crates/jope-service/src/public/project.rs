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
mod tests;
