//! Controlled realization. Every constructor chooses factual content before independent
//! wording. No renderer has access to future facts, training data, or the search RNG.
use crate::{diff_programs, digest, AtomicEdit, LineageFact, Links, ProgramDelta, ProgramSnapshot};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;

mod renderer;
pub const RENDERER: &str = "controlled-search-commentary/1";
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    /// Kind of fact asserted by this claim.
    pub category: String,
    /// Stable machine-readable claim identifier.
    pub predicate: String,
    /// Evidence identifiers supporting the claim.
    pub evidence: Vec<String>,
    /// Predicate-specific structured evidence details.
    pub details: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Narration {
    /// Human-readable bounded commentary.
    pub text: String,
    /// Overall category of the commentary.
    pub category: String,
    /// Machine-readable claims supporting the commentary.
    pub claims: Vec<Claim>,
    /// Exact structural difference when the source facts support one.
    pub delta: Option<ProgramDelta>,
    /// Renderer wording variant used for replay, when applicable.
    pub variant: Option<u8>,
}

/// Caller-supplied metrics compare only under an identical explicit scope. Values retain
/// bits and direction; percentages require positive baselines and are never inferred here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    /// Name of the measured quantity.
    pub name: String,
    /// Comparison direction: `minimize` or `maximize`.
    pub direction: String,
    /// Explicit scope that must match before comparing metrics.
    pub scope: String,
    /// Finite `f64` value encoded as 16 lowercase hexadecimal bits.
    pub value_bits: String,
}
impl Metric {
    pub fn new(name: &str, direction: &str, scope: &str, value: f64) -> Result<Self> {
        ensure!(
            value.is_finite() && matches!(direction, "minimize" | "maximize") && !scope.is_empty(),
            "invalid metric"
        );
        Ok(Self {
            name: name.into(),
            direction: direction.into(),
            scope: scope.into(),
            value_bits: format!("{:016x}", value.to_bits()),
        })
    }
    pub fn value(&self) -> Result<f64> {
        ensure!(self.value_bits.len() == 16, "invalid metric bits");
        let v = f64::from_bits(u64::from_str_radix(&self.value_bits, 16)?);
        ensure!(v.is_finite(), "nonfinite metric");
        Ok(v)
    }
    pub fn improvement_over(&self, baseline: &Self) -> Result<f64> {
        ensure!(
            self.name == baseline.name
                && self.direction == baseline.direction
                && self.scope == baseline.scope
                && !self.scope.is_empty(),
            "incompatible comparison scope"
        );
        let delta = baseline.value()? - self.value()?;
        match self.direction.as_str() {
            "minimize" => Ok(delta),
            "maximize" => Ok(-delta),
            _ => anyhow::bail!("invalid metric direction"),
        }
    }
}

pub(crate) struct Renderer {
    recent: VecDeque<usize>,
}
impl Renderer {
    pub(crate) fn new() -> Self {
        Self {
            recent: VecDeque::new(),
        }
    }
    pub(crate) fn render(
        &mut self,
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
    ) -> Narration {
        self.render_with_variant(fact, links, evidence, None)
    }
    pub(crate) fn rederive(
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
        variant: Option<u8>,
    ) -> Result<Narration> {
        ensure!(variant.is_none_or(|v| v < 8), "invalid grammar variant");
        ensure!(
            matches!(fact, LineageFact::Drawn { .. }) || variant.is_none(),
            "unexpected grammar variant"
        );
        Ok(Self::new().render_with_variant(fact, links, evidence, variant))
    }
}

fn short(value: &str) -> String {
    safe(&value.chars().take(12).collect::<String>())
}
pub fn safe(value: &str) -> String {
    let mut out = value
        .chars()
        .take(256)
        .flat_map(|c| {
            if c.is_control() || matches!(c,'\u{202a}'..='\u{202e}'|'\u{2066}'..='\u{2069}') {
                c.escape_unicode().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect::<String>();
    if value.chars().nth(256).is_some() {
        out.push('…');
    }
    out
}
fn stage_text(stage: &crate::StageView) -> String {
    let params = stage
        .params
        .iter()
        .map(|(k, v)| format!("{}={}", safe(k), safe(&v.display())))
        .collect::<Vec<_>>()
        .join(", ");
    if params.is_empty() {
        safe(&stage.op)
    } else {
        format!("{}({params})", safe(&stage.op))
    }
}

fn value(value: &Option<crate::ExactValue>) -> String {
    value
        .as_ref()
        .map(|v| safe(&v.display()))
        .unwrap_or_else(|| "omitted".into())
}
fn edit_text(edit: &AtomicEdit) -> String {
    match edit {
        AtomicEdit::Parameter {
            at,
            op,
            key,
            before,
            after,
        } => format!(
            "stage[{at}] {}.{} changes {} → {}",
            safe(op),
            safe(key),
            value(before),
            value(after)
        ),
        AtomicEdit::Insert { at, stage } => format!("insert {} at stage[{at}]", stage_text(stage)),
        AtomicEdit::Delete { at, before } => {
            format!("remove {} at stage[{at}]", stage_text(before))
        }
        AtomicEdit::Replace { at, before, after } => format!(
            "stage[{at}] replaces {} with {}",
            stage_text(before),
            stage_text(after)
        ),
        AtomicEdit::Swap { at, left, right } => format!(
            "swap stage[{at}] {} and stage[{}] {}",
            stage_text(left),
            at + 1,
            stage_text(right)
        ),
        AtomicEdit::ReplaceProgram { .. } => {
            "replace the whole captured program; finer semantic coverage unavailable".into()
        }
    }
}
fn hypothesis(delta: &ProgramDelta) -> String {
    if let [AtomicEdit::Parameter { op, key, .. }] = delta.edits.as_slice() {
        if matches!(op.as_str(), "ridge" | "logistic") && key == "lambda_frac" {
            return "Observer question: would this different ridge regularization setting help validation? The measured result is pending.".into();
        }
        if key == "seed" {
            return "Observer question: how will this recorded seeded construction measure? A different seed is not evidence of improved quality. Validation is pending.".into();
        }
    }
    "Observer question: how will the changed recipe measure as a whole? Validation is pending, and any eventual gain cannot yet be attributed to an individual edit.".into()
}
pub(crate) fn resolution(edit: &AtomicEdit) -> Option<(String, bool)> {
    let AtomicEdit::Parameter {
        op,
        key,
        before,
        after,
        ..
    } = edit
    else {
        return None;
    };
    let number = |v: &Option<crate::ExactValue>, default: f64| -> Option<f64> {
        match v {
            None => Some(default),
            Some(v) => match v.to_param().ok()? {
                feat_math::dsl::ParamValue::Int(n) => Some(n as f64),
                feat_math::dsl::ParamValue::Float(n) if n.is_finite() => Some(n),
                _ => None,
            },
        }
    };
    if matches!(op.as_str(), "winsor_clip" | "winsor_moments") && key == "q" {
        let (a, b) = (
            number(before, 0.05)?.clamp(0.0, 0.25),
            number(after, 0.05)?.clamp(0.0, 0.25),
        );
        let unchanged = a.to_bits() == b.to_bits();
        return Some((
            if unchanged {
                format!("Under the legacy planner, both requests resolve to q={a}; this parameter has no resolved change.")
            } else {
                format!("The legacy planner resolves q from {a} to {b}.")
            },
            unchanged,
        ));
    }
    let (default, min, max) = match (op.as_str(), key.as_str()) {
        ("quantile_bins", "bins") => (8, 2, 64),
        ("cheb", "degree") => (4, 1, 8),
        ("bin_onehot", "bins") => (6, 2, 32),
        ("hinge", "knots") => (4, 1, 16),
        ("rff_cos" | "rff_sin" | "oblique_hinge", "n") => (32, 1, 256),
        ("rand_proj", "k") => (16, 1, 256),
        ("landmark_dist" | "nystrom", "k") => (16, 1, 128),
        ("pair_prod", "topk") => (8, 2, 24),
        ("target_bin_encode", "bins") => (8, 2, 32),
        ("spline", "knots") => (5, 3, 12),
        ("svd_project", "k") => (8, 1, 64),
        ("gbm" | "histgbm", "rounds") => (150, 1, 500),
        ("gbm" | "histgbm", "depth") => (3, 1, 8),
        ("gbm" | "histgbm", "bins") => (64, 2, 255),
        _ => return None,
    };
    let resolve = |v: &Option<crate::ExactValue>| -> Option<usize> {
        Some(
            match v {
                None => default,
                Some(v) => v.to_param().ok()?.as_usize().unwrap_or(default),
            }
            .clamp(min, max),
        )
    };
    let (a, b) = (resolve(before)?, resolve(after)?);
    Some((
        if a == b {
            format!("Under the legacy planner, both requests resolve to {}={a}; this parameter has no resolved change.",safe(key))
        } else {
            format!("The legacy planner resolves {} from {a} to {b}.", safe(key))
        },
        a == b,
    ))
}

pub(crate) fn implementation_notes(source: &str) -> Vec<String> {
    let Ok(feat_math::GenomeIr::Pipeline(p)) = feat_math::GenomeIr::parse(source) else {
        return Vec::new();
    };
    let Some(head) = p.stages.last() else {
        return Vec::new();
    };
    let mut notes = Vec::new();
    let alias = match head.op.as_str() {
        "logistic" => Some("ridge"),
        "gbm" => Some("histgbm"),
        "ensemble" => Some("ensemble_ridge_gbm"),
        _ => None,
    };
    if let Some(alias) = alias {
        notes.push(format!(
            "The legacy planner resolves the requested {} head to {alias}.",
            safe(&head.op)
        ));
    }
    if matches!(head.op.as_str(), "ridge" | "logistic") {
        let fraction = head.get_f64("lambda_frac", 0.02);
        if fraction.is_finite() {
            notes.push(format!("Ridge lambda_frac={fraction} sets a fitting penalty max(lambda_frac × fitting rows, 0.001)."));
        }
    }
    notes
}
