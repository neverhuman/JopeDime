//! Checks of a private fragment table (section 5 of the design).
//!
//! A reference is a macro, so the proof obligation for every recorded occurrence is the
//! byte-exact round trip. Everything after the canonical text is a function of that text and
//! the data, so comparing fits of the same text twice tests engine determinism and is labelled
//! that way. The checks that are not vacuous are context compatibility and the pinned-digest
//! control tests. Context compatibility counts only real fits of contexts whose text differs
//! from every recorded host, compared both as fitted and with the task's default head written
//! out; a context that repeats a host is recorded as not independent and never satisfies it.
//!
//! No check passes without being exercised. A check class that examined nothing records a
//! `<class>_not_exercised` failure (`context_not_exercised` for context compatibility), so
//! an empty artifact list, a fragment whose every fit was refused, or a missing context can
//! never be reported or tallied as a pass. A table with no fragments never passes, and the
//! source check ([`verify_source`]) never confirms a table that records no occurrence.

use crate::expand::{expand, rewrite, At};
use crate::fragment::{classification_schema, input_stage};
use crate::source::{QualifiedProgram, QualifiedSet, TaskLabel};
use crate::synthetic::{Shape, Synthetic};
use crate::table::{MotifDef, MotifTable};
use crate::{Code, NAME_PREFIX};
use anyhow::{ensure, Result};
use feat_math::advanced_v3::{
    emit_bound, emit_fitted_cpp17, fit_table, predict_multiclass_table, predict_table,
    supports_fitted_cpp17, validate_v3_genome, BoundModel, DataSchema,
};
use feat_math::kernel::pipeline::PipelineSlots;
use feat_math::transpile::{self, EmitOptions};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Private label of the validation report. There is no schema file for it.
pub const VALIDATION_SCHEMA: &str = "jope.motif-validation/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
/// Options controlling how many recorded hosts are fitted for each fragment.
pub struct ValidateOptions {
    /// Distinct recorded hosts fitted per fragment, in provenance order.
    pub hosts_per_fragment: u32,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        Self {
            hosts_per_fragment: 3,
        }
    }
}

/// The private validation report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Private validation result bound to a table and, optionally, a fresh source read.
pub struct Validation {
    /// Validation report schema identifier.
    pub schema: String,
    /// Digest of the table that was checked.
    pub table_sha256: String,
    /// True only when a fresh read of the source (`validate --state`) confirmed every recorded
    /// occurrence ([`verify_source`]). False when the source was not read, when nothing was
    /// examined, or when any occurrence is no longer current.
    pub source_verified: bool,
    /// `Some` exactly when the source was read: whether its digest is unchanged. The digest
    /// covers the source's exclusion reasons too, so it is recorded, never relied on.
    pub source_digest_equal: Option<bool>,
    /// Results for each fragment in table order.
    pub motifs: Vec<MotifResult>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Validation outcomes for one fragment.
pub struct MotifResult {
    /// Zero-based index of the fragment in table order.
    pub index: u32,
    /// Whether rewriting and re-expanding its recorded host is exact.
    pub round_trip: CheckResult,
    /// Whether repeated evaluation is deterministic.
    pub determinism: CheckResult,
    /// Whether independent compatible contexts were fitted successfully.
    pub context_compatibility: CheckResult,
    /// Whether emitted artifacts omit the reserved fragment reference name.
    pub name_absence: CheckResult,
    /// Hosts whose fit was refused on every table. Never counted as a pass.
    pub not_exercised: u32,
    /// Backend capability observations for the attempted contexts and tasks.
    pub backend: Vec<BackendRow>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Aggregate outcome for one validation check class.
pub struct CheckResult {
    /// True only when at least one case was exercised and none failed.
    pub passed: bool,
    /// Number of cases examined by the check.
    pub exercised: u32,
    /// Fixed-code failures observed while running the check.
    pub failures: Vec<Failure>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A fixed validation failure and its optional local context.
pub struct Failure {
    /// Fixed failure code; it contains no private source value.
    pub code: String,
    /// Host recipe digest associated with the failure, when available.
    pub recipe_sha256: Option<String>,
    /// Context label associated with the failure, when available.
    pub context: Option<String>,
    /// Table label associated with the failure, when available.
    pub table: Option<String>,
    /// Task associated with the failure, when available.
    pub task: Option<TaskLabel>,
}

/// Backend coverage of one context and task. Recorded, never gating: coverage is not parity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendRow {
    /// Synthetic context label used for this observation.
    pub context: String,
    /// Task evaluated in the context.
    pub task: TaskLabel,
    /// False when the context text, as fitted or with the task's default head written out,
    /// equals a recorded host. Such a fit repeats a host and never satisfies context
    /// compatibility.
    pub independent: bool,
    /// The legacy planner accepts the context text (every row).
    pub legacy_plan: bool,
    /// Q3 only: whether the legacy fit on S2's numeric rows succeeded. `None` when no legacy
    /// fit was attempted: on the Q1 and Q2 rows, or when the planner refused the text.
    pub legacy_fit: Option<bool>,
    /// Whether the legacy C++17 emitter accepted the context.
    pub legacy_cpp17: bool,
    /// Whether the legacy JavaScript emitter accepted the context.
    pub legacy_javascript: bool,
    /// Whether the bound artifact emitter accepted the context.
    pub bound_emit: bool,
    /// Whether fitted C++17 emission succeeded.
    pub fitted_cpp17: bool,
}

impl CheckResult {
    /// Seal a check. A check that examined nothing never passes: it records `unexercised`
    /// as a failure instead.
    fn finish(mut self, unexercised: &str) -> Self {
        if self.exercised == 0 {
            self.failures.push(Failure::new(unexercised));
        }
        self.passed = self.exercised > 0 && self.failures.is_empty();
        self
    }
}

impl Failure {
    fn new(code: &str) -> Self {
        Self {
            code: code.to_owned(),
            recipe_sha256: None,
            context: None,
            table: None,
            task: None,
        }
    }

    fn recipe(mut self, recipe_sha256: &str) -> Self {
        self.recipe_sha256 = Some(recipe_sha256.to_owned());
        self
    }

    fn context(mut self, context: &str) -> Self {
        self.context = Some(context.to_owned());
        self
    }

    fn table(mut self, shape: Shape) -> Self {
        self.table = Some(shape.as_str().to_owned());
        self
    }

    fn task(mut self, task: TaskLabel) -> Self {
        self.task = Some(task);
        self
    }
}

impl MotifResult {
    /// V0 passed (or no result would exist) and C1 to C4 passed. Each of them passed only
    /// after examining at least one occurrence, fit, context or artifact.
    pub fn passed(&self) -> bool {
        self.round_trip.passed
            && self.determinism.passed
            && self.context_compatibility.passed
            && self.name_absence.passed
    }
}

impl Validation {
    /// At least one fragment was checked, every fragment passed, and, when the source was read,
    /// it confirmed every occurrence. An empty table never passes: nothing was examined.
    pub fn passed(&self) -> bool {
        !self.motifs.is_empty()
            && self.motifs.iter().all(MotifResult::passed)
            && (self.source_digest_equal.is_none() || self.source_verified)
    }
}

/// Confirm a table against a fresh read of its source. `source_verified` is true only when the
/// table records at least one occurrence and every occurrence's record is in the fresh set with
/// the same comparison, recipe, task, dataset, cut and label, a non-adaptive label, and a
/// program byte-equal to the recorded host. The digest comparison is recorded separately.
///
/// Returns the source check's failure code, `None` exactly when the source was confirmed:
/// `source_not_exercised` when the table records no occurrence, so nothing was examined, and
/// `source_not_confirmed` when an occurrence is no longer current.
pub fn verify_source(
    validation: &mut Validation,
    table: &MotifTable,
    set: &QualifiedSet,
) -> Option<&'static str> {
    let fresh: BTreeMap<&str, &QualifiedProgram> = set
        .programs()
        .iter()
        .map(|program| (program.record_sha256(), program))
        .collect();
    let mut examined = 0_usize;
    let mut current = true;
    for motif in &table.body.motifs {
        for occurrence in &motif.provenance {
            examined += 1;
            current &= fresh
                .get(occurrence.record_sha256.as_str())
                .is_some_and(|program| {
                    program.comparison_sha256() == occurrence.comparison_sha256
                        && program.recipe_sha256() == occurrence.recipe_sha256
                        && program.task() == occurrence.task
                        && program.dataset_sha256() == occurrence.dataset_sha256
                        && program.cut_sha256() == occurrence.cut_sha256
                        && *program.label() == occurrence.label
                        && !program.label().is_adaptive()
                        && table
                            .body
                            .hosts
                            .get(&occurrence.recipe_sha256)
                            .is_some_and(|host| host == program.program())
                });
        }
    }
    validation.source_digest_equal = Some(set.source_sha256() == table.body.source.sha256);
    validation.source_verified = examined > 0 && current;
    if examined == 0 {
        Some("source_not_exercised")
    } else if current {
        None
    } else {
        Some("source_not_confirmed")
    }
}

/// Check a table: V0 first (any failure is an error), then C1 to C4 for every fragment.
pub fn validate(table: &MotifTable, options: &ValidateOptions) -> Result<Validation> {
    ensure!(options.hosts_per_fragment >= 1, Code("invalid_parameters"));
    table.verify()?;
    let tables = Synthetic::all()?;
    let mut motifs = Vec::with_capacity(table.body.motifs.len());
    for (index, motif) in table.body.motifs.iter().enumerate() {
        let (result, _artifacts) = check_motif(table, index, motif, options, &tables)?;
        motifs.push(result);
    }
    Ok(Validation {
        schema: VALIDATION_SCHEMA.to_owned(),
        table_sha256: table.table_sha256.clone(),
        source_verified: false,
        source_digest_equal: None,
        motifs,
    })
}

/// One emitted artifact and its kind.
#[derive(Clone, Debug)]
struct Artifact {
    kind: &'static str,
    text: String,
}

/// Fitted state and probe predictions of one fit, as bytes and bit patterns.
#[derive(Clone, Debug)]
struct Fitted {
    model: BoundModel,
    state: Vec<u8>,
    predictions: Vec<u64>,
    probabilities: Vec<Vec<u64>>,
}

impl Fitted {
    fn same(&self, other: &Self) -> bool {
        self.state == other.state
            && self.predictions == other.predictions
            && self.probabilities == other.probabilities
    }
}

enum Outcome {
    Refused,
    Fitted(Box<Fitted>),
    Nondeterministic,
}

/// Fit on one synthetic table and predict its probe. Probe predictions must be finite, and
/// multiclass probabilities must be valid (the engine refuses invalid ones).
fn fit_once(ir: &GenomeIr, task: TaskLabel, data: &Synthetic) -> Result<Fitted> {
    let spec = task.task_spec()?;
    let model = fit_table(ir, &spec, &data.schema, &data.train, data.targets(task))?;
    let state = serde_json::to_vec(&model)?;
    let predictions = predict_table(&model, &data.probe)?;
    ensure!(
        predictions.iter().all(|value| value.is_finite()),
        "non-finite probe prediction"
    );
    let probabilities: Vec<Vec<u64>> = if task == TaskLabel::Multiclass {
        predict_multiclass_table(&model, &data.probe)?
            .probabilities
            .iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect())
            .collect()
    } else {
        Vec::new()
    };
    Ok(Fitted {
        predictions: predictions.iter().map(|value| value.to_bits()).collect(),
        model,
        state,
        probabilities,
    })
}

/// Fit twice and compare fitted state bytes and prediction bits.
fn fit_twice(ir: &GenomeIr, task: TaskLabel, data: &Synthetic) -> Outcome {
    let Ok(first) = fit_once(ir, task, data) else {
        return Outcome::Refused;
    };
    match fit_once(ir, task, data) {
        Ok(second) if first.same(&second) => Outcome::Fitted(Box::new(first)),
        _ => Outcome::Nondeterministic,
    }
}

fn data(tables: &[Synthetic], shape: Shape) -> Result<&Synthetic> {
    tables
        .iter()
        .find(|table| table.shape == shape)
        .ok_or(Code("io"))
        .map_err(Into::into)
}

#[derive(Clone, Copy, Debug, Default)]
struct LegacyCoverage {
    plan: bool,
    cpp17: bool,
    javascript: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct BoundCoverage {
    bound_emit: bool,
    fitted_cpp17: bool,
}

/// Legacy training-program artifacts: Python, Rust and manifest with the default and the
/// fast-runtime profile, plus C++17 and JavaScript where covered.
fn legacy_artifacts(ir: &GenomeIr, out: &mut Vec<Artifact>) -> LegacyCoverage {
    let plan = transpile::supports(ir).is_ok();
    if plan {
        for fast_runtime in [false, true] {
            let options = EmitOptions {
                fast_runtime,
                ..EmitOptions::default()
            };
            if let Ok(emitted) = transpile::emit(ir, &options) {
                out.push(Artifact {
                    kind: "legacy_python",
                    text: emitted.python,
                });
                out.push(Artifact {
                    kind: "legacy_rust",
                    text: emitted.rust,
                });
                out.push(Artifact {
                    kind: "legacy_manifest",
                    text: emitted.manifest,
                });
            }
        }
    }
    let cpp17 = transpile::supports_cpp17(ir).is_ok();
    if let Ok(source) = transpile::emit_cpp17(ir, &EmitOptions::default()) {
        out.push(Artifact {
            kind: "legacy_cpp17",
            text: source,
        });
    }
    let javascript = transpile::supports_javascript(ir).is_ok();
    if let Ok(source) = transpile::emit_javascript(ir, &EmitOptions::default()) {
        out.push(Artifact {
            kind: "legacy_javascript",
            text: source,
        });
    }
    LegacyCoverage {
        plan,
        cpp17,
        javascript,
    }
}

/// Artifacts of one fitted model: the bound prediction program and the fitted C++17 source
/// and codec, where supported.
fn bound_artifacts(model: &BoundModel, out: &mut Vec<Artifact>) -> BoundCoverage {
    let bound_emit = match emit_bound(model, &EmitOptions::default()) {
        Ok(emitted) => {
            out.push(Artifact {
                kind: "bound_python",
                text: emitted.python,
            });
            out.push(Artifact {
                kind: "bound_rust",
                text: emitted.rust,
            });
            out.push(Artifact {
                kind: "bound_manifest",
                text: emitted.manifest,
            });
            true
        }
        Err(_) => false,
    };
    let fitted_cpp17 = supports_fitted_cpp17(model).is_ok();
    if let Ok(artifact) = emit_fitted_cpp17(model) {
        out.push(Artifact {
            kind: "fitted_cpp17_source",
            text: artifact.source,
        });
        out.push(Artifact {
            kind: "fitted_cpp17_codec",
            text: artifact.codec,
        });
    }
    BoundCoverage {
        bound_emit,
        fitted_cpp17,
    }
}

/// C1: for every recorded occurrence, rewrite at its offset and everywhere, expand, and
/// require the host bytes and recipe digest back. The unexpanded text must be refused by the
/// slot contract and the schema-bound grammar.
fn round_trip(table: &MotifTable, motif: &MotifDef) -> Result<CheckResult> {
    let schema = classification_schema();
    let mut result = CheckResult::default();
    for occurrence in &motif.provenance {
        let host = table
            .body
            .hosts
            .get(&occurrence.recipe_sha256)
            .ok_or(Code("provenance_inconsistent"))?;
        let offset = usize::try_from(occurrence.stage_offset)?;
        let single = rewrite(host, motif, At::Offset(offset));
        let every = rewrite(host, motif, At::All);
        for (label, rewritten) in [("rewrite_offset", &single), ("rewrite_all", &every)] {
            let code = match rewritten {
                Err(_) => Some("canonical_mismatch"),
                Ok(text) => match expand(text, table) {
                    Err(_) => Some("canonical_mismatch"),
                    Ok(expanded) if expanded.canonical != *host => Some("canonical_mismatch"),
                    Ok(expanded) if expanded.recipe_sha256 != occurrence.recipe_sha256 => {
                        Some("recipe_hash_mismatch")
                    }
                    Ok(_) => None,
                },
            };
            if let Some(code) = code {
                result.failures.push(
                    Failure::new(code)
                        .recipe(&occurrence.recipe_sha256)
                        .context(label)
                        .task(occurrence.task),
                );
            }
        }
        let accepted = single
            .as_ref()
            .is_ok_and(|text| unexpanded_is_accepted(text, occurrence.task, &schema));
        if accepted {
            result.failures.push(
                Failure::new("unexpanded_accepted")
                    .recipe(&occurrence.recipe_sha256)
                    .task(occurrence.task),
            );
        }
        result.exercised += 1;
    }
    Ok(result.finish("round_trip_not_exercised"))
}

/// The slot contract or the schema-bound grammar accepts text that still holds a reference.
fn unexpanded_is_accepted(text: &str, task: TaskLabel, schema: &DataSchema) -> bool {
    let Ok(ir) = GenomeIr::parse(text) else {
        return false;
    };
    let slots = matches!(
        &ir,
        GenomeIr::Pipeline(expr) if PipelineSlots::from_pipeline(expr).is_ok()
    );
    let grammar = task
        .task_spec()
        .is_ok_and(|spec| validate_v3_genome(&ir, &spec, schema).is_ok());
    slots || grammar
}

/// C4: no artifact may contain the name, the full id, its 12- or 16-hex prefix or the
/// reserved prefix. At least one artifact must have been emitted: a fragment for which no
/// emitter produced anything (for example one observed only with multiclass hosts, which the
/// scalar bound and fitted C++ emitters refuse and whose native text the legacy planner
/// refuses) fails with `name_absence_not_exercised`.
fn name_absence(motif: &MotifDef, artifacts: &[Artifact]) -> CheckResult {
    let mut needles = vec![motif.name.clone(), motif.id.clone(), NAME_PREFIX.to_owned()];
    for length in [12, 16] {
        if let Some(prefix) = motif.id.get(..length) {
            needles.push(prefix.to_owned());
        }
    }
    let mut result = CheckResult::default();
    for artifact in artifacts {
        result.exercised += 1;
        if needles
            .iter()
            .any(|needle| artifact.text.contains(needle.as_str()))
        {
            result
                .failures
                .push(Failure::new("name_in_emission").context(artifact.kind));
        }
    }
    result.finish("name_absence_not_exercised")
}

/// The head the engine adds when a program names none (`advanced_v3/pipeline.rs:364-370`).
fn default_head(task: TaskLabel) -> &'static str {
    match task {
        TaskLabel::Regression => "ridge",
        TaskLabel::Binary => "logit_ridge",
        TaskLabel::Multiclass => "multiclass_ridge",
    }
}

/// Mutable state of the fit-based checks of one fragment.
struct MotifRun<'a> {
    table: &'a MotifTable,
    motif: &'a MotifDef,
    tables: &'a [Synthetic],
    determinism: CheckResult,
    context: CheckResult,
    not_exercised: u32,
    backend: Vec<BackendRow>,
    artifacts: Vec<Artifact>,
}

impl MotifRun<'_> {
    /// C2 on hosts: up to `limit` distinct hosts in provenance order, fitted on S2 and
    /// otherwise S3. A host refused on both counts as not exercised.
    fn hosts(&mut self, limit: usize) -> Result<()> {
        let table = self.table;
        let motif = self.motif;
        let tables = self.tables;
        let mut chosen: Vec<(&str, TaskLabel)> = Vec::new();
        for occurrence in &motif.provenance {
            if chosen.len() == limit {
                break;
            }
            let recipe = occurrence.recipe_sha256.as_str();
            if !chosen.iter().any(|(seen, _)| *seen == recipe) {
                chosen.push((recipe, occurrence.task));
            }
        }
        for (recipe, task) in chosen {
            let host = table
                .body
                .hosts
                .get(recipe)
                .ok_or(Code("provenance_inconsistent"))?;
            let ir = GenomeIr::parse(host).map_err(|_| Code("provenance_inconsistent"))?;
            legacy_artifacts(&ir, &mut self.artifacts);
            let mut exercised = false;
            for shape in [Shape::S2, Shape::S3] {
                match fit_twice(&ir, task, data(tables, shape)?) {
                    Outcome::Refused => {}
                    Outcome::Nondeterministic => {
                        self.determinism.exercised += 1;
                        self.determinism.failures.push(
                            Failure::new("fit_not_deterministic")
                                .recipe(recipe)
                                .context("host")
                                .table(shape)
                                .task(task),
                        );
                        exercised = true;
                        break;
                    }
                    Outcome::Fitted(fitted) => {
                        self.determinism.exercised += 1;
                        bound_artifacts(&fitted.model, &mut self.artifacts);
                        exercised = true;
                        break;
                    }
                }
            }
            if !exercised {
                self.not_exercised += 1;
            }
        }
        Ok(())
    }

    /// The context, as fitted or with the task's default head written out, equals a recorded
    /// host. Such a fit repeats a host: it is determinism and name-absence evidence, but not
    /// an independent context.
    fn repeats_a_host(&self, texts: &[&str]) -> bool {
        let hosts = &self.table.body.hosts;
        texts.iter().any(|text| {
            GenomeIr::parse(text).is_ok_and(|ir| {
                let canonical = ir.canonical();
                hosts.values().any(|host| *host == canonical)
            })
        })
    }

    /// C3 with C2 on every successful context fit. For every task where a host qualified:
    /// Q1 `(pipe X <input> E)` with the default head, Q2 with a leading `(standardize)`, and
    /// for regression the legacy-dialect Q3 `(pipe X E (ridge))`.
    ///
    /// For each task, Q1 must fit on S2 or S3 (`context_not_fitted`), and at least one
    /// context that does not repeat a recorded host must fit on S2 or S3
    /// (`context_not_independent`). Only fits of such independent contexts count as
    /// exercised. Everything else is recorded coverage.
    fn contexts(&mut self) -> Result<()> {
        let motif = self.motif;
        let tables = self.tables;
        let input = input_stage()?;
        let schema = classification_schema();
        for task in &motif.support.tasks {
            let task = *task;
            let spec = task.task_spec()?;
            let prefixes = [
                ("q1", input.clone()),
                ("q2", format!("{input} (standardize)")),
            ];
            let mut q1_fitted = false;
            let mut independent_fitted = false;
            for (label, prefix) in prefixes {
                let text = format!("(pipe X {prefix} {})", motif.expansion);
                let explicit = format!(
                    "(pipe X {prefix} {} ({}))",
                    motif.expansion,
                    default_head(task)
                );
                let refusal = |code: &str| Failure::new(code).context(label).task(task);
                let Ok(ir) = GenomeIr::parse(&text) else {
                    self.context.failures.push(refusal("parse_failed"));
                    continue;
                };
                let canonical = ir.canonical();
                let again = GenomeIr::parse(&canonical).map(|parsed| parsed.canonical());
                if again.ok().as_deref() != Some(canonical.as_str()) {
                    self.context.failures.push(refusal("not_idempotent"));
                    continue;
                }
                let slotted = matches!(
                    &ir,
                    GenomeIr::Pipeline(expr) if PipelineSlots::from_pipeline(expr).is_ok()
                );
                if !slotted {
                    self.context.failures.push(refusal("slot_refused"));
                    continue;
                }
                if validate_v3_genome(&ir, &spec, &schema).is_err() {
                    self.context.failures.push(refusal("grammar_refused"));
                    continue;
                }
                let independent = !self.repeats_a_host(&[text.as_str(), explicit.as_str()]);
                let legacy = legacy_artifacts(&ir, &mut self.artifacts);
                let mut primary = false;
                let mut bound = BoundCoverage::default();
                for data in tables {
                    match fit_twice(&ir, task, data) {
                        Outcome::Refused => {}
                        Outcome::Nondeterministic => {
                            self.determinism.exercised += 1;
                            self.determinism.failures.push(
                                Failure::new("fit_not_deterministic")
                                    .context(label)
                                    .table(data.shape)
                                    .task(task),
                            );
                        }
                        Outcome::Fitted(fitted) => {
                            self.determinism.exercised += 1;
                            if independent {
                                self.context.exercised += 1;
                            }
                            let coverage = bound_artifacts(&fitted.model, &mut self.artifacts);
                            if !primary && matches!(data.shape, Shape::S2 | Shape::S3) {
                                primary = true;
                                bound = coverage;
                            }
                        }
                    }
                }
                if label == "q1" {
                    q1_fitted = primary;
                }
                independent_fitted |= independent && primary;
                self.backend.push(BackendRow {
                    context: label.to_owned(),
                    task,
                    independent,
                    legacy_plan: legacy.plan,
                    legacy_fit: None,
                    legacy_cpp17: legacy.cpp17,
                    legacy_javascript: legacy.javascript,
                    bound_emit: bound.bound_emit,
                    fitted_cpp17: bound.fitted_cpp17,
                });
            }
            if !q1_fitted {
                self.context
                    .failures
                    .push(Failure::new("context_not_fitted").context("q1").task(task));
            } else if !independent_fitted {
                self.context
                    .failures
                    .push(Failure::new("context_not_independent").task(task));
            }
            if task == TaskLabel::Regression {
                self.legacy_context()?;
            }
        }
        Ok(())
    }

    /// Q3: legacy planner coverage and, where the planner accepts the text, a legacy fit on
    /// S2's numeric rows. Recorded only.
    fn legacy_context(&mut self) -> Result<()> {
        let text = format!("(pipe X {} (ridge))", self.motif.expansion);
        let Ok(ir) = GenomeIr::parse(&text) else {
            return Ok(());
        };
        let independent = !self.repeats_a_host(&[text.as_str()]);
        let legacy = legacy_artifacts(&ir, &mut self.artifacts);
        let s2 = data(self.tables, Shape::S2)?;
        let legacy_fit = legacy.plan.then(|| {
            s2.numeric_rows()
                .is_some_and(|rows| feat_math::fit(&ir, &rows, &s2.regression).is_ok())
        });
        self.backend.push(BackendRow {
            context: String::from("q3"),
            task: TaskLabel::Regression,
            independent,
            legacy_plan: legacy.plan,
            legacy_fit,
            legacy_cpp17: legacy.cpp17,
            legacy_javascript: legacy.javascript,
            bound_emit: false,
            fitted_cpp17: false,
        });
        Ok(())
    }
}

/// C1 to C4 for one fragment, with every artifact the checks produced.
fn check_motif(
    table: &MotifTable,
    index: usize,
    motif: &MotifDef,
    options: &ValidateOptions,
    tables: &[Synthetic],
) -> Result<(MotifResult, Vec<Artifact>)> {
    let round_trip = round_trip(table, motif)?;
    let mut run = MotifRun {
        table,
        motif,
        tables,
        determinism: CheckResult::default(),
        context: CheckResult::default(),
        not_exercised: 0,
        backend: Vec::new(),
        artifacts: Vec::new(),
    };
    run.hosts(usize::try_from(options.hosts_per_fragment)?)?;
    run.contexts()?;
    let name_absence = name_absence(motif, &run.artifacts);
    let result = MotifResult {
        index: u32::try_from(index)?,
        round_trip,
        determinism: run.determinism.finish("determinism_not_exercised"),
        context_compatibility: run.context.finish("context_not_exercised"),
        name_absence,
        not_exercised: run.not_exercised,
        backend: run.backend,
    };
    Ok((result, run.artifacts))
}

#[cfg(test)]
mod tests;
