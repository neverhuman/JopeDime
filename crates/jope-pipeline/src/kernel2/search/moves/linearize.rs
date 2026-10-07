//! The bounded minimax linearization pass's proposals: what a parent offers, from its text and
//! each `derive` stage's training input, and the same re-derived from a record of the fits.

// laneD:B19-minimax-linearize

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    dsl::{DslExpr, ParamValue},
    kernel::{
        expression_search::{
            LinearFit, LinearizationEvidence, MinimaxStop, SubtreeLinearization,
            LINEARIZATION_DEGREE, LINEARIZE_FIT_REFUSED, LINEARIZE_INPUT_FAULTS,
            LINEARIZE_UNSAMPLED,
        },
        search::admit_child,
    },
    math_expr::{ExpressionScope, MathExpr},
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    kernel2::{
        refinement::{parse_untrusted_pipeline, training_prefix},
        Kernel2Evaluator,
    },
    prepare::Cut,
};

/// The refusal of every subtree of a `derive` stage whose training input could not be built
/// without a fault: none of them is sampled. It is feat-math's [`LINEARIZE_INPUT_FAULTS`], which
/// the recorded linearization takes for every subtree of an expression or none.
pub const PREFIX_FAULTS: &str = LINEARIZE_INPUT_FAULTS;

/// The linearization pass's record of one subtree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearizationRecord {
    /// The `derive` stage's index in the parent.
    pub stage: usize,
    /// The subtree's position in the stage's expression: child indices from its root.
    pub path: Vec<usize>,
    /// The subtree, as printed.
    pub subtree: String,
    /// The subtree's nodes.
    pub nodes: usize,
    /// The fitted degree, always 1: the pass offers lines only.
    pub degree: usize,
    /// The minimax fit on the stage's training input; absent when the subtree was not sampled or
    /// the fit refused its samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<FitRecord>,
    /// With a fit, the minimax bound the line is held to on the training samples:
    /// `max_error (1 + 1e-6) + 64 ε max_abs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<f64>,
    /// The lowered line's nodes, once it was built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_nodes: Option<usize>,
    /// The line's own largest error over the training samples, once measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lowered_error: Option<f64>,
    /// The stage's expression with the line in place of the subtree, admitted by the profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    /// Why no line is offered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// A degree-1 discrete minimax fit, as published. Its errors are floating-point residuals on the
/// training samples, not certified bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitRecord {
    /// Why the fit stopped: `exact_fit`, `converged` or `noise_limited` (its bracket closed), or
    /// `stalled`, `degenerate` or `iteration_limit` (the best fit seen; no line is offered).
    pub stop: String,
    /// `[a0, a1]` of the line `a0 + a1 x`; absent when one is not finite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coefficients: Option<[f64; 2]>,
    /// The training range of the subtree's variable.
    pub lo: f64,
    pub hi: f64,
    /// The bracket `[levelled, max_error]` of the fit's error over the samples.
    pub levelled: f64,
    pub max_error: f64,
    /// The largest absolute value of the subtree over the samples.
    pub max_abs: f64,
    pub iterations: usize,
    pub steps: usize,
}

impl FitRecord {
    fn new(fit: &LinearFit) -> Self {
        Self {
            stop: fit.stop.as_str().into(),
            coefficients: fit
                .coefficients
                .iter()
                .all(|c| c.is_finite())
                .then_some(fit.coefficients),
            lo: fit.lo,
            hi: fit.hi,
            levelled: fit.levelled,
            max_error: fit.max_error,
            max_abs: fit.max_abs,
            iterations: fit.iterations,
            steps: fit.steps,
        }
    }

    /// The fit this record names. A coefficient that was not finite is read back as NaN, which
    /// the lowering refuses as it refused the original.
    fn fit(&self) -> Result<LinearFit> {
        let stop = MinimaxStop::ALL
            .into_iter()
            .find(|s| s.as_str() == self.stop)
            .context("unknown minimax stop")?;
        Ok(LinearFit {
            stop,
            coefficients: self.coefficients.unwrap_or([f64::NAN; 2]),
            lo: self.lo,
            hi: self.hi,
            levelled: self.levelled,
            max_error: self.max_error,
            max_abs: self.max_abs,
            iterations: self.iterations,
            steps: self.steps,
        })
    }
}

impl LinearizationRecord {
    fn new(stage: usize, l: &SubtreeLinearization) -> Result<Self> {
        Ok(Self {
            stage,
            path: l.path.clone(),
            subtree: l.subtree.to_source()?,
            nodes: l.nodes,
            degree: LINEARIZATION_DEGREE,
            fit: l.fit.as_ref().map(FitRecord::new),
            bound: l.fit.as_ref().map(LinearFit::bound),
            line_nodes: l.line_nodes,
            lowered_error: l.lowered_error,
            expression: l.expression.as_ref().map(MathExpr::to_source).transpose()?,
            refusal: l.refusal.clone(),
        })
    }

    /// What this record keeps of the training measurements. Without a fit, its refusal must be
    /// one a subtree can have before any fit: not sampled, a faulting training input, or the
    /// fit's own refusal.
    fn evidence(&self) -> Result<LinearizationEvidence> {
        if self.fit.is_none() {
            let refusal = self.refusal.as_deref().unwrap_or_default();
            ensure!(
                refusal == LINEARIZE_UNSAMPLED
                    || refusal == PREFIX_FAULTS
                    || refusal
                        .strip_prefix(LINEARIZE_FIT_REFUSED)
                        .is_some_and(|rest| rest.starts_with(": ")),
                "a linearization without a fit names no refusal it can have"
            );
        }
        Ok(LinearizationEvidence {
            fit: self.fit.as_ref().map(FitRecord::fit).transpose()?,
            lowered_error: self.lowered_error,
            refusal: self.refusal.clone(),
        })
    }
}

/// One subtree's proposal for one parent, before any search state is read.
#[derive(Debug)]
pub(crate) struct LinearProposal {
    pub(crate) linearization: LinearizationRecord,
    /// The parent with the stage's expression replaced, when the line was offered.
    pub(crate) program: Option<String>,
    /// Why pipeline admission refused that program.
    pub(crate) reason: Option<String>,
    /// The line was offered and the program passed pipeline admission.
    pub(crate) admitted: bool,
}

/// The scope a `derive` stage's expression is admitted under: the profile's pool, from the
/// program text alone, as the rewrite pass admits it.
fn scope(evaluator: &Kernel2Evaluator) -> Result<ExpressionScope> {
    let pool = evaluator
        .profile()
        .expr
        .as_ref()
        .context("expression pool required")?;
    Ok(ExpressionScope {
        columns: pool.column_upper_exclusive,
        elementwise: false,
        max_nodes: pool.max_nodes,
        max_depth: pool.max_depth,
    })
}

fn derive_expression(pipe: &DslExpr, index: usize) -> Result<Option<MathExpr>> {
    let stage = &pipe.stages[index];
    if stage.op != "derive" {
        return Ok(None);
    }
    let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
        bail!("expression stage without an expression");
    };
    Ok(Some(MathExpr::parse(source)?))
}

/// Whether `parent` has a subtree the pass would consider, from its text alone. `check` reads the
/// parents back from evidence, which is untrusted, so the text is parsed within
/// [`parse_untrusted_pipeline`]'s bounds, and a program over them offers nothing.
pub(crate) fn offers(evaluator: &Kernel2Evaluator, parent: &str) -> Result<bool> {
    let Ok(pipe) = parse_untrusted_pipeline(parent) else {
        return Ok(false);
    };
    let scope = scope(evaluator)?;
    for index in 0..pipe.stages.len() {
        if let Some(expression) = derive_expression(&pipe, index)? {
            if !expression
                .linearizable_v2(evaluator.profile(), scope)?
                .is_empty()
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Every considered subtree of every `derive` stage of `parent`, in stage order and then the
/// order `linearize` gives, each assembled into a program and checked by the pipeline admission
/// every measured program passes when its line was offered. `rederive` calls it on parents read
/// back from evidence, so `parent` is parsed within [`parse_untrusted_pipeline`]'s bounds first.
fn assemble(
    evaluator: &Kernel2Evaluator,
    parent: &str,
    width: usize,
    mut linearize: impl FnMut(
        &DslExpr,
        usize,
        &MathExpr,
        ExpressionScope,
    ) -> Result<Vec<SubtreeLinearization>>,
) -> Result<Vec<LinearProposal>> {
    let pipe = parse_untrusted_pipeline(parent)
        .context("the linearization pass needs a pipeline program")?;
    let profile = evaluator.profile();
    let scope = scope(evaluator)?;
    let mut out = Vec::new();
    for index in 0..pipe.stages.len() {
        let Some(expression) = derive_expression(&pipe, index)? else {
            continue;
        };
        let considered = linearize(&pipe, index, &expression, scope)
            .context("the linearization pass refused an admitted parent")?;
        for l in considered {
            let (program, reason) = match &l.expression {
                None => (None, None),
                Some(expression) => {
                    let mut child = pipe.clone();
                    child.stages[index]
                        .params
                        .insert("expr".into(), ParamValue::Str(expression.to_source()?));
                    let program = GenomeIr::Pipeline(child).canonical();
                    let reason = admit_child(&program, profile, width)
                        .err()
                        .map(|error| format!("{error:#}"));
                    (Some(program), reason)
                }
            };
            out.push(LinearProposal {
                linearization: LinearizationRecord::new(index, &l)?,
                admitted: program.is_some() && reason.is_none(),
                program,
                reason,
            });
        }
    }
    Ok(out)
}

/// What `parent` offers: each `derive` stage's subtrees, sampled on that stage's training input
/// (the earlier stages fitted and applied on the cut's training rows and labels alone). No
/// held-out row or label is read. `width` is the input's column count.
pub(crate) fn propose(
    evaluator: &Kernel2Evaluator,
    parent: &str,
    cut: &Cut,
    width: usize,
) -> Result<Vec<LinearProposal>> {
    let profile = evaluator.profile();
    assemble(
        evaluator,
        parent,
        width,
        |pipe, index, expression, scope| {
            let prefix = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                training_prefix(pipe, index, &cut.train_x, &cut.train_y)
            }));
            match prefix {
                Ok(Ok(rows)) => expression.linearize_v2(profile, scope, &rows),
                // The stage's training input faults: every subtree is considered, none sampled.
                _ => expression.linearize_recorded_v2(profile, scope, &mut |_| {
                    Ok(LinearizationEvidence {
                        fit: None,
                        lowered_error: None,
                        refusal: Some(PREFIX_FAULTS.into()),
                    })
                }),
            }
        },
    )
}

/// The proposals of `parent` re-derived from `records`, its published linearization records in
/// order: everything but each fit and the line's measured error, which the records keep, is
/// recomputed from the parent's text. The records must account for every considered subtree, and
/// a faulting training input ([`PREFIX_FAULTS`]) refuses every subtree of its stage or none, as
/// `propose` refuses them.
pub(crate) fn rederive(
    evaluator: &Kernel2Evaluator,
    parent: &str,
    width: usize,
    records: &[LinearizationRecord],
) -> Result<Vec<LinearProposal>> {
    let mut stages: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for record in records {
        let (subtrees, faulted) = stages.entry(record.stage).or_default();
        *subtrees += 1;
        *faulted += usize::from(record.refusal.as_deref() == Some(PREFIX_FAULTS));
    }
    ensure!(
        stages
            .values()
            .all(|&(subtrees, faulted)| faulted == 0 || faulted == subtrees),
        "a faulting training input refuses only some of its stage's subtrees"
    );
    let profile = evaluator.profile();
    let mut used = 0usize;
    let proposals = assemble(evaluator, parent, width, |_, index, expression, scope| {
        let start = used;
        let considered = expression.linearize_recorded_v2(profile, scope, &mut |i| {
            let record = records
                .get(start + i)
                .context("a linearization record is missing")?;
            ensure!(
                record.stage == index,
                "a linearization record names another stage"
            );
            record.evidence()
        })?;
        used += considered.len();
        Ok(considered)
    })?;
    ensure!(
        proposals.len() == records.len(),
        "linearization records do not account for their parent"
    );
    Ok(proposals)
}
