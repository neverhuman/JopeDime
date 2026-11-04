//! Host classification, transform runs, windows and the registry-derived fragment signature.

use crate::source::{QualifiedProgram, TaskLabel};
use crate::table::{Occurrence, SignatureV1, TaskMask};
use crate::{sha256_hex, Code, NAME_PREFIX};
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{validate_v3_genome, ColumnSpec, DataSchema, RowSemantics};
use feat_math::dsl::{parse_dsl, ParamValue, Stage};
use feat_math::kernel::pipeline::PipelineSlots;
use feat_math::{members, registry, GenomeIr, CHAMPION_GENOME};
use std::collections::BTreeSet;
use std::ops::Range;

/// The input stage of every native recipe, as the frozen panel writes it
/// (`crates/jope-bench/src/panel/methods.rs:20-21`).
pub const INPUT_STAGE: &str =
    "(input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256)";

/// Slot of every fragment: rows in, rows out, with the target visible to fitted stages.
pub const SLOT: &str = "transform_run";

/// [`INPUT_STAGE`] in canonical text.
pub fn input_stage() -> Result<String> {
    let parsed = parse_dsl(&format!("(pipe X {INPUT_STAGE})"))?;
    parsed
        .stages
        .first()
        .map(ToString::to_string)
        .context("input stage")
}

/// Six numeric columns `c0..c5` with independent rows.
pub fn classification_schema() -> DataSchema {
    DataSchema {
        columns: (0..6)
            .map(|index| ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// How one qualified program was classified. The first rule that applies decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostClass {
    /// Optimization-visible test labels; excluded by default.
    ExcludedAdaptive,
    /// The text contains the reserved prefix anywhere.
    SkipReserved,
    /// Not a `(pipe X ...)` program.
    SkipForm,
    /// The slot contract refuses it.
    SkipSlots,
    /// The schema-bound grammar refuses it for its task on the classification schema.
    SkipUnsupported,
    /// A host: its transform body is mined.
    Used,
}

/// A classified program and, for a host, its transform body.
#[derive(Clone, Debug)]
pub struct Classified {
    /// Why the qualified program was used or skipped.
    pub class: HostClass,
    /// Absolute index of the first body stage: one for each present target or input stage.
    pub body_start: usize,
    /// Transform stages extracted from the host program.
    pub body: Vec<Stage>,
}

impl Classified {
    fn skipped(class: HostClass) -> Self {
        Self {
            class,
            body_start: 0,
            body: Vec::new(),
        }
    }
}

/// Classify one program (section 4, step 2). A digest or canonical-form mismatch is an
/// error, never a skip.
pub fn classify(program: &QualifiedProgram) -> Result<Classified> {
    if program.label().is_adaptive() {
        return Ok(Classified::skipped(HostClass::ExcludedAdaptive));
    }
    let text = program.program();
    ensure!(
        sha256_hex(text.as_bytes()) == program.recipe_sha256(),
        Code("recipe_hash_mismatch")
    );
    let ir = GenomeIr::parse(text).map_err(|_| Code("noncanonical_program"))?;
    ensure!(ir.canonical() == text, Code("noncanonical_program"));
    if text.contains(NAME_PREFIX) {
        return Ok(Classified::skipped(HostClass::SkipReserved));
    }
    let GenomeIr::Pipeline(expr) = &ir else {
        return Ok(Classified::skipped(HostClass::SkipForm));
    };
    let Ok(slots) = PipelineSlots::from_pipeline(expr) else {
        return Ok(Classified::skipped(HostClass::SkipSlots));
    };
    let spec = program.task().task_spec()?;
    if validate_v3_genome(&ir, &spec, &classification_schema()).is_err() {
        return Ok(Classified::skipped(HostClass::SkipUnsupported));
    }
    Ok(Classified {
        class: HostClass::Used,
        body_start: usize::from(slots.target.is_some()) + usize::from(slots.input.is_some()),
        body: slots.transforms,
    })
}

/// A body stage may join a run only when its op is not expression-addressed (`map`,
/// `derive`), not order-dependent (`lag`, `roll`) and has an executable, non-quarantined
/// registry row from which its signature can be derived.
pub fn stage_eligible(stage: &Stage) -> bool {
    if matches!(stage.op.as_str(), "map" | "derive" | "lag" | "roll") {
        return false;
    }
    registry::operator(&stage.op)
        .is_some_and(|row| row.lifecycle.executable && !row.lifecycle.quarantined)
}

/// Maximal runs of eligible stages; every other stage breaks a run.
pub fn runs(body: &[Stage]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut open: Option<usize> = None;
    for (index, stage) in body.iter().enumerate() {
        match (stage_eligible(stage), open) {
            (true, None) => open = Some(index),
            (false, Some(start)) => {
                out.push(start..index);
                open = None;
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        out.push(start..body.len());
    }
    out
}

/// Every contiguous window of two to `max_stages` stages inside one run, as
/// `(start, length)`.
pub fn windows(run: &Range<usize>, max_stages: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let longest = max_stages.min(run.end.saturating_sub(run.start));
    for length in 2..=longest {
        for start in run.start..=run.end - length {
            out.push((start, length));
        }
    }
    out
}

/// The canonical text of every stage.
pub fn stage_texts(stages: &[Stage]) -> Vec<String> {
    stages.iter().map(ToString::to_string).collect()
}

/// Stage texts joined by one space: the identity text of a window.
pub fn join(stages: &[Stage]) -> String {
    stage_texts(stages).join(" ")
}

/// The stages of an expansion text.
pub fn expansion_stages(expansion: &str) -> Result<Vec<Stage>> {
    Ok(parse_dsl(&format!("(pipe X {expansion})"))?.stages)
}

/// A recorded host: its stage texts and the range of its transform body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostBody {
    /// Canonical text of every stage in the host.
    pub texts: Vec<String>,
    /// Half-open range of transform stages within `texts`.
    pub body: Range<usize>,
}

/// `inner` is a strictly shorter contiguous sub-run of `outer`.
pub fn is_sub_run(inner: &[String], outer: &[String]) -> bool {
    !inner.is_empty()
        && inner.len() < outer.len()
        && outer.windows(inner.len()).any(|window| window == inner)
}

/// Canonical texts of the non-empty member pipes and of the reigning control program.
/// Roster recipes are not listed: they have no transform body.
pub fn known_seeds() -> Vec<String> {
    let mut out: Vec<String> = members::CORE_V1
        .iter()
        .filter(|member| !member.pipe.is_empty())
        .filter_map(|member| GenomeIr::parse(member.pipe).ok())
        .map(|ir| ir.canonical())
        .collect();
    if let Ok(ir) = GenomeIr::parse(CHAMPION_GENOME.trim()) {
        out.push(ir.canonical());
    }
    out
}

/// The planner's append rule: `append=true` or `append=1`.
pub fn appends_input(stage: &Stage) -> bool {
    matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
}

/// The signature of an expansion, derived from the registry and the provenance. It stores
/// only local mirrors and strings, never registry types.
pub fn signature(stages: &[Stage], provenance: &[Occurrence]) -> Result<SignatureV1> {
    ensure!(!stages.is_empty(), Code("signature_drift"));
    let mut ops = Vec::with_capacity(stages.len());
    let mut semantic_keys = Vec::with_capacity(stages.len());
    let mut shapes = Vec::with_capacity(stages.len());
    let mut appends = Vec::with_capacity(stages.len());
    let mut tags = BTreeSet::new();
    let mut fitted = false;
    let mut requires_labels = false;
    let mut declared_tasks = TaskMask {
        regression: true,
        binary: true,
        multiclass: true,
    };
    let mut all_registry_generatable = true;
    for stage in stages {
        let row = registry::operator(&stage.op)
            .filter(|row| row.lifecycle.executable && !row.lifecycle.quarantined)
            .ok_or(Code("signature_drift"))?;
        ops.push(stage.op.clone());
        semantic_keys.push(row.semantic_key.to_owned());
        fitted |= row.fitted;
        requires_labels |= row.requires_labels;
        declared_tasks.regression &= row.tasks.regression;
        declared_tasks.binary &= row.tasks.binary_classification;
        declared_tasks.multiclass &= row.tasks.multiclass;
        shapes.push(serde_json::to_string(&row.shape)?);
        appends.push(appends_input(stage));
        tags.extend(row.tags.iter().map(|tag| (*tag).to_owned()));
        all_registry_generatable &= row.lifecycle.generatable;
    }
    let observed: BTreeSet<TaskLabel> = provenance.iter().map(|o| o.task).collect();
    Ok(SignatureV1 {
        slot: SLOT.to_owned(),
        stages: u32::try_from(stages.len())?,
        ops,
        semantic_keys,
        fitted,
        requires_labels,
        declared_tasks,
        observed_tasks: observed.into_iter().collect(),
        shapes,
        appends,
        tags: tags.into_iter().collect(),
        all_registry_generatable,
    })
}

/// The typing probe `(pipe X <input> E)` passes the slot contract and the schema-bound
/// grammar for `task` on the classification schema.
pub fn typed_for(expansion: &str, task: TaskLabel) -> bool {
    let Ok(input) = input_stage() else {
        return false;
    };
    let Ok(ir) = GenomeIr::parse(&format!("(pipe X {input} {expansion})")) else {
        return false;
    };
    let GenomeIr::Pipeline(expr) = &ir else {
        return false;
    };
    PipelineSlots::from_pipeline(expr).is_ok()
        && task
            .task_spec()
            .is_ok_and(|spec| validate_v3_genome(&ir, &spec, &classification_schema()).is_ok())
}

#[cfg(test)]
mod tests;
