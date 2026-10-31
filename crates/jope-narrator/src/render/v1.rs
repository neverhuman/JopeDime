//! Controlled realization. Every constructor chooses factual content before independent
//! wording. No renderer has access to future facts, training data, or the search RNG.
use crate::{diff_programs, digest, AtomicEdit, LineageFact, Links, ProgramDelta, ProgramSnapshot};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;

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
    fn render_with_variant(
        &mut self,
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
        forced: Option<u8>,
    ) -> Narration {
        let evidence_refs = std::iter::once(evidence.to_string())
            .chain(
                [
                    links.search_event_id.clone(),
                    links.observation_sha256.clone(),
                    links.draw_id.clone(),
                    links.request_id.clone(),
                    links.attempt_id.clone(),
                ]
                .into_iter()
                .flatten(),
            )
            .collect::<Vec<_>>();
        let claim = |category: &str, predicate: &str, details: Value| Claim {
            category: category.into(),
            predicate: predicate.into(),
            evidence: evidence_refs.clone(),
            details,
        };
        let mut out = Narration {
            text: String::new(),
            category: "status_fact".into(),
            claims: Vec::new(),
            delta: None,
            variant: None,
        };
        use LineageFact::*;
        match fact {
            SearchStarted {
                cut,
                objective,
                train_rows,
                hold_rows,
                ..
            } => {
                out.text=format!("Live search commentary, derived from program edits and measurements. Cut {cut}: {train_rows} fitting rows and {hold_rows} validation rows. Ranking uses {}.",objective.as_ref().map(|s|format!("the recorded {} objective",safe(s))).unwrap_or_else(||"validation RMSE".into()));
                out.claims.push(claim("status_fact","search_started",json!({"cut":cut,"objective":objective,"train_rows":train_rows,"hold_rows":hold_rows})));
            }
            Drawn {
                cut,
                coordinate,
                origin,
                parents,
                child,
                outcome,
                generation,
                island,
                phase,
                ..
            } => {
                let variants = [
                    "Compared with its recorded parent",
                    "Keeping the parent as reference",
                    "The new variation",
                    "This candidate",
                    "The local change",
                    "Against the parent recipe",
                    "At this search step",
                    "The next recorded draw",
                ];
                let seed = digest(evidence.as_bytes());
                let mut chosen =
                    usize::from_str_radix(&seed[..8], 16).unwrap_or(0) % variants.len();
                for _ in 0..variants.len() {
                    if !self.recent.contains(&chosen) {
                        break;
                    }
                    chosen = (chosen + 1) % variants.len();
                }
                if let Some(v) = forced {
                    chosen = usize::from(v);
                }
                out.variant = Some(chosen as u8);
                self.recent.push_back(chosen);
                if self.recent.len() > 4 {
                    self.recent.pop_front();
                }
                let id = child
                    .as_ref()
                    .map(|p| short(&p.sha256))
                    .unwrap_or_else(|| "unavailable".into());
                out.text = if phase == "comparison" {
                    format!(
                        "Cut {cut}, comparison pool (coordinate island {island}), candidate {id}: "
                    )
                } else {
                    format!(
                        "Cut {cut}, generation {generation}, island {island}, {} candidate {id}: ",
                        safe(phase)
                    )
                };
                if !parents.is_empty() {
                    out.text.push_str(&format!(
                        "ordered parents [{}]. ",
                        parents
                            .iter()
                            .map(|p| short(&p.sha256))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                if let Some(child) = child {
                    for note in implementation_notes(&child.canonical) {
                        out.claims.push(claim("implementation_fact","legacy_head_resolution",json!({"rule_version":"legacy-plan-7848c77/1","source":"crates/feat-math/src/spec.rs::plan_head_stage; crates/feat-math/src/heads.rs::RidgeModel::fit","note":note})));
                        out.text.push_str(&note);
                        out.text.push(' ');
                    }
                    if let Some(parent) = parents.first() {
                        match ProgramSnapshot::from_legacy(&parent.canonical).and_then(|p| {
                            ProgramSnapshot::from_legacy(&child.canonical)
                                .and_then(|c| diff_programs(&p, &c))
                        }) {
                            Ok(delta) => {
                                out.category = "structural_fact".into();
                                if delta.edits.is_empty() {
                                    out.text.push_str("the captured program is unchanged from the recorded parent.");
                                } else if delta.coverage == "whole_program_only" {
                                    out.text.push_str("the program changed; detailed semantic analysis is unavailable for this representation.");
                                } else {
                                    out.text.push_str(variants[chosen]);
                                    out.text.push_str(": ");
                                    out.text.push_str(
                                        &delta
                                            .edits
                                            .iter()
                                            .take(6)
                                            .map(edit_text)
                                            .collect::<Vec<_>>()
                                            .join("; "),
                                    );
                                    out.text.push('.');
                                    if delta.edits.len() > 6 {
                                        out.text.push_str(&format!(" {} additional exact edits are retained in the audit record.",delta.edits.len()-6));
                                    }
                                    for edit in &delta.edits {
                                        if let Some((text, _)) = resolution(edit) {
                                            out.text.push(' ');
                                            out.text.push_str(&text);
                                            out.claims.push(claim("implementation_fact","legacy_parameter_resolution",json!({"rule_version":"legacy-plan-7848c77/1","source":"crates/feat-math/src/spec.rs::plan_op","edit":edit,"resolution":text})));
                                        }
                                    }
                                }
                                out.claims.push(claim(
                                    "structural_fact",
                                    "endpoint_difference",
                                    json!({"parent_identity":delta.parent.identity,"child_identity":delta.child.identity,"coverage":delta.coverage,"patch":"narration.delta"}),
                                ));
                                if parents.len() > 1 {
                                    if parents[0] == parents[1] {
                                        out.text.push_str(" The first two ordered parent roles selected the same program; neither role is discarded.");
                                    }
                                    for (role, other) in parents.iter().enumerate().skip(1) {
                                        match ProgramSnapshot::from_legacy(&other.canonical)
                                            .and_then(|p| {
                                                ProgramSnapshot::from_legacy(&child.canonical)
                                                    .and_then(|c| diff_programs(&p, &c))
                                            }) {
                                            Ok(other_delta) => {
                                                out.text.push_str(&format!(" Relative to parent role {} ({}), {} exact endpoint edits are retained.",role+1,short(&other.sha256),other_delta.edits.len()));
                                                out.claims.push(claim(
                                                    "structural_fact",
                                                    "additional_parent_difference",
                                                    json!({"role":role+1,"delta":other_delta}),
                                                ));
                                            }
                                            Err(_) => out.claims.push(claim(
                                                "limitation",
                                                "additional_parent_analysis_unavailable",
                                                json!({"role":role+1,"parent":other.sha256}),
                                            )),
                                        }
                                    }
                                    out.text.push_str(" Endpoint differences do not identify a unique crossover splice.");
                                    out.claims.push(claim("limitation","crossover_endpoint_provenance",json!({"parents":parents.iter().map(|p|&p.sha256).collect::<Vec<_>>()})));
                                }
                                if outcome == "candidate"
                                    && !delta.edits.is_empty()
                                    && !delta.edits.iter().all(|e| {
                                        resolution(e).is_some_and(|(_, unchanged)| unchanged)
                                    })
                                    && matches!(origin.as_str(), "mutate" | "crossover")
                                {
                                    let question = hypothesis(&delta);
                                    out.text.push(' ');
                                    out.text.push_str(&question);
                                    out.claims.push(claim("prospective_hypothesis","test_changed_recipe",json!({"question":question,"knowledge_cut":evidence,"origin":"observer_interpretation","status":"pending"})));
                                }
                                out.delta = Some(delta);
                            }
                            Err(_) => {
                                out.text.push_str("the recorded recipe is available; exact structural analysis is unavailable under current limits.");
                                out.claims.push(claim(
                                    "limitation",
                                    "analysis_unavailable",
                                    json!({"coordinate":coordinate}),
                                ));
                            }
                        }
                    } else {
                        out.text.push_str(&format!(
                            "a {} proposal was recorded without parent references.",
                            safe(origin)
                        ));
                        out.claims.push(claim(
                            "structural_fact",
                            "parentless_proposal",
                            json!({"program_sha256":child.sha256,"origin":origin}),
                        ));
                    }
                } else {
                    out.text.push_str(
                        "construction returned no program; no evaluation result is implied.",
                    );
                }
                out.claims.push(claim(
                    "observed_decision",
                    "draw_outcome",
                    json!({"coordinate":coordinate,"outcome":outcome,"origin":origin}),
                ));
                if outcome != "candidate" {
                    out.text
                        .push_str(&format!(" Recorded draw outcome: {}.", safe(outcome)));
                }
            }
            Admitted {
                cut,
                program_sha256,
                disposition,
                ..
            } => {
                out.text=format!("Cut {cut}, candidate {}: admission disposition {}. Admission alone does not establish that fitting started.",short(program_sha256),safe(disposition));
                out.claims.push(claim(
                    "observed_decision",
                    "admission",
                    json!({"cut":cut,"program_sha256":program_sha256,"disposition":disposition}),
                ));
            }
            Started {
                cut,
                program_sha256,
                ..
            } => {
                out.text = format!(
                    "Cut {cut}, candidate {}: evaluation started; results pending.",
                    short(program_sha256)
                );
                out.claims.push(claim(
                    "status_fact",
                    "evaluation_started",
                    json!({"cut":cut,"program_sha256":program_sha256}),
                ));
            }
            Measured {
                cut,
                phase,
                program_sha256,
                status,
                rmse,
                objective_loss,
                hold_r2,
                train_r2,
                seconds,
                generation,
                island,
                origin,
                ..
            } => {
                out.category = "measured_fact".into();
                out.text = format!(
                    "Cut {cut}, candidate {}: {} evaluation status {}",
                    short(program_sha256),
                    safe(phase),
                    safe(status)
                );
                if let Some(v) = objective_loss {
                    out.text.push_str(&format!("; recorded objective loss {v}"));
                }
                if let Some(v) = rmse {
                    out.text.push_str(&format!("; validation RMSE {v}"));
                }
                if let Some(v) = hold_r2 {
                    out.text.push_str(&format!("; validation R² {v}"));
                }
                if let Some(v) = train_r2 {
                    out.text.push_str(&format!("; fitting R² {v}"));
                }
                out.text.push_str(&format!("; measured runtime {seconds}s (generation {generation}, island {}, origin {}).",island.map_or_else(||"unavailable".into(),|i|i.to_string()),safe(origin)));
                out.claims.push(claim("measured_fact","evaluation_result",json!({"cut":cut,"phase":phase,"program_sha256":program_sha256,"status":status,"rmse_bits":rmse.map(|v|format!("{:016x}",v.to_bits())),"objective_loss_bits":objective_loss.map(|v|format!("{:016x}",v.to_bits()))})));
            }
            GenerationClosed {
                cut,
                generation,
                island,
                survivors,
                evaluations,
                barren,
            } => {
                out.text=format!("Cut {cut}, island {island}: generation {generation} closed after {evaluations} evaluations; {} recorded survivors.",survivors.len());
                if let Some(top) = survivors.first() {
                    out.text.push_str(&format!(
                        " Leading survivor {} has ranking loss {} and validation RMSE {}.",
                        short(&top.program_sha256),
                        top.score,
                        top.rmse
                    ));
                }
                if *barren > 0 {
                    out.text.push_str(&format!(
                        " {barren} consecutive generations spent no evaluation."
                    ));
                }
                out.claims.push(claim("aggregate_fact","generation_closed",json!({"cut":cut,"generation":generation,"island":island,"survivors":survivors.len(),"evaluations":evaluations})));
            }
            Migrated {
                cut,
                from_island,
                to_island,
                admitted,
                program_sha256,
                rmse,
                after_generation,
            } => {
                out.text = format!(
                    "Cut {cut}, after generation {after_generation}: candidate {} (validation RMSE {rmse}) migration from island {from_island} to {to_island} was {}.",
                    short(program_sha256),
                    if *admitted { "admitted" } else { "refused" }
                );
                out.claims.push(claim(
                    "observed_decision",
                    "migration",
                    json!({"cut":cut,"from":from_island,"to":to_island,"admitted":admitted}),
                ));
            }
            ExplorationStopped {
                cut,
                reason,
                evaluations,
                ..
            } => {
                out.text=format!("Cut {cut}: exploration stopped after {evaluations} evaluations; recorded stop reason {}.",safe(reason));
                out.claims.push(claim(
                    "observed_decision",
                    "exploration_stopped",
                    json!({"cut":cut,"reason":reason,"evaluations":evaluations}),
                ));
            }
            ComparisonStarted {
                cuts,
                program_sha256s,
            } => {
                out.text=format!("Cross-cut comparison started: {} recorded programs across {} cuts. Complete comparison results pending.",program_sha256s.len(),cuts.len());
                out.claims.push(claim(
                    "status_fact",
                    "comparison_started",
                    json!({"cuts":cuts,"programs":program_sha256s}),
                ));
            }
            TapDropped { dropped } => {
                out.category = "limitation".into();
                out.text=format!("Evidence gap: {dropped} observer records were dropped. Complete experiment counts and history cannot be inferred across this gap.");
                out.claims.push(claim(
                    "limitation",
                    "capture_gap",
                    json!({"dropped":dropped.to_string()}),
                ));
            }
            Selected {
                program_sha256,
                rule,
            } => {
                out.category = "observed_decision".into();
                out.text = format!(
                    "Candidate {} selected under recorded rule {}.",
                    short(program_sha256),
                    safe(rule)
                );
                out.claims.push(claim(
                    "observed_decision",
                    "selected",
                    json!({"program_sha256":program_sha256,"rule":rule}),
                ));
            }
            CandidatesOffered {
                program_sha256s,
                rule,
            } => {
                out.category = "observed_decision".into();
                out.text=format!("{} ranked candidates offered under recorded rule {}. The consuming engine has its own final selection and fitting steps.",program_sha256s.len(),safe(rule));
                out.claims.push(claim(
                    "observed_decision",
                    "candidates_offered",
                    json!({"program_sha256s":program_sha256s,"rule":rule}),
                ));
            }
            Emitted {
                program_sha256,
                languages,
            } => {
                out.text=format!("Candidate {}: source emission completed for {}. Compiler checks and predictive validation have separate status.",short(program_sha256),languages.iter().map(|v|safe(v)).collect::<Vec<_>>().join(", "));
                out.claims.push(claim(
                    "status_fact",
                    "source_emitted",
                    json!({"program_sha256":program_sha256,"languages":languages}),
                ));
            }
            RunStopped { status } => {
                out.text = format!("Recorded run status: {}.", safe(status));
                out.claims
                    .push(claim("status_fact", "run_status", json!({"status":status})));
            }
            Prediction {
                status,
                program_sha256,
            } => {
                out.text = format!(
                    "Fixed-model prediction status: {}{}.",
                    safe(status),
                    program_sha256
                        .as_ref()
                        .map(|p| format!("; program {}", short(p)))
                        .unwrap_or_default()
                );
                out.claims.push(claim(
                    "status_fact",
                    "prediction_status",
                    json!({"status":status,"program_sha256":program_sha256}),
                ));
            }
            PredictionBound {
                status,
                model_sha256,
                input_sha256,
                rows,
                seconds,
            } => {
                out.text = format!(
                    "Fixed-model prediction {}: model {}, input {}, {rows} rows.",
                    safe(status),
                    short(model_sha256),
                    short(input_sha256)
                );
                if let Some(seconds) = seconds {
                    out.text
                        .push_str(&format!(" Measured prediction runtime {seconds}s."));
                }
                out.text.push_str(" Prediction uses fixed fitted state.");
                out.claims.push(claim("status_fact","prediction_bound",json!({"status":status,"model_sha256":model_sha256,"input_sha256":input_sha256,"rows":rows,"seconds_bits":seconds.map(|v|format!("{:016x}",v.to_bits()))})));
            }
        }
        if matches!(fact, LineageFact::Measured { .. })
            && links.observation_sha256.is_some()
            && links.draw_id.is_none()
            && links.request_id.is_none()
            && links.attempt_id.is_none()
        {
            out.text.push_str(" Native draw/request/attempt references are unavailable; the observation digest was computed from the retained boundary data.");
            out.claims.push(claim(
                "limitation",
                "derived_boundary_observation",
                json!({"observation_sha256":links.observation_sha256,"native_refs":"unavailable"}),
            ));
        }
        out
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
fn resolution(edit: &AtomicEdit) -> Option<(String, bool)> {
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

fn implementation_notes(source: &str) -> Vec<String> {
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
