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
mod tests {
    use super::*;
    use crate::fixture;
    use crate::mine::mine;
    use crate::table::MineParameters;

    fn stages(text: &str) -> Vec<Stage> {
        expansion_stages(text).expect("stages parse")
    }

    fn classify_text(text: &str, task: TaskLabel) -> Classified {
        classify(&fixture::qualified(text, task, "d1", "r1")).expect("classifies")
    }

    #[test]
    fn roster_recipe_is_a_host_with_an_empty_body() {
        for (task, heads) in fixture::roster_heads() {
            for head in heads {
                let classified = classify_text(&fixture::roster(head), task);
                assert_eq!(classified.class, HostClass::Used, "{head} for {task:?}");
                assert!(classified.body.is_empty());
                assert_eq!(classified.body_start, 1);
            }
        }
        let input = input_stage().expect("input stage");
        assert_eq!(
            input,
            "(input categorical=onehot max_onehot=256 min_count=1 missing=indicator numeric=median)"
        );
    }

    #[test]
    fn roster_recipe_with_a_transform_body_is_mined_not_skipped() {
        let text = fixture::v3("(standardize) (cheb degree=2)", "tab_boost");
        let classified = classify_text(&text, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::Used);
        assert_eq!(
            stage_texts(&classified.body),
            ["(standardize)", "(cheb degree=2)"]
        );
        let set = fixture::set_of(vec![
            fixture::qualified(&text, TaskLabel::Regression, "d1", "r1"),
            fixture::qualified(&text, TaskLabel::Regression, "d2", "r2"),
        ]);
        let table = mine(&set, None, MineParameters::default()).expect("mines");
        assert_eq!(table.body.motifs.len(), 1);
        assert_eq!(
            table.body.motifs[0].expansion,
            "(standardize) (cheb degree=2)"
        );
    }

    #[test]
    fn transform_body_excludes_target_input_and_learner() {
        let text = fixture::canonical(&format!(
            "(pipe X (target kind=log1p) {INPUT_STAGE} (standardize) (cheb degree=2) (ridge))"
        ));
        let classified = classify_text(&text, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::Used);
        assert_eq!(classified.body_start, 2);
        assert_eq!(
            stage_texts(&classified.body),
            ["(standardize)", "(cheb degree=2)"]
        );
        let legacy = fixture::canonical("(pipe X (standardize) (cheb degree=2) (ridge))");
        let classified = classify_text(&legacy, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::Used);
        assert_eq!(classified.body_start, 0);
        assert_eq!(classified.body.len(), 2);
    }

    #[test]
    fn runs_break_at_expression_order_dependent_and_unregistered_stages() {
        let body = stages(
            "(standardize) (cheb degree=2) (map expr=\"sin(x)\") (rank) (sigmoid) \
             (symlog) (blom) (rank) (maxabs_scale) (standardize) (lag k=1) (cheb degree=2) \
             (rank) (derive expr=\"c0\") (roll k=2) (robust_scale)",
        );
        assert_eq!(runs(&body), [0..2, 3..5, 6..8, 9..10, 11..13, 15..16]);
        for op in ["map", "derive", "lag", "roll", "symlog", "maxabs_scale"] {
            assert!(!stage_eligible(&Stage::new(op)), "{op}");
        }
        for op in [
            "standardize",
            "cheb",
            "pair_prod",
            "target_bin_encode",
            "sigmoid",
        ] {
            assert!(stage_eligible(&Stage::new(op)), "{op}");
        }
        // Quarantined rows are never eligible.
        assert!(!stage_eligible(&Stage::new("rff_orthogonal")));
        assert_eq!(windows(&(0..2), 6), [(0, 2)]);
        assert!(windows(&(9..10), 6).is_empty());
        assert_eq!(
            windows(&(0..4), 3),
            [(0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
        );

        // Under ordered rows the grammar admits a lag host; its run still breaks at the lag.
        let ordered = DataSchema {
            row_semantics: RowSemantics::Ordered {
                group_by: Vec::new(),
                order_by: vec![String::from("c0")],
            },
            ..classification_schema()
        };
        let lagged = fixture::v3(
            "(standardize) (cheb degree=2) (lag k=1) (rank) (sigmoid)",
            "ridge",
        );
        let ir = GenomeIr::parse(&lagged).expect("parses");
        let spec = TaskLabel::Regression.task_spec().expect("spec");
        assert!(validate_v3_genome(&ir, &spec, &ordered).is_ok());
        assert!(validate_v3_genome(&ir, &spec, &classification_schema()).is_err());
        let GenomeIr::Pipeline(expr) = &ir else {
            panic!("a pipeline")
        };
        let slots = PipelineSlots::from_pipeline(expr).expect("slots");
        assert_eq!(runs(&slots.transforms), [0..2, 3..5]);
    }

    #[test]
    fn unslotted_programs_are_skipped_not_mined() {
        for body in ["(rff_orthogonal n=8) (standardize)", "(bag) (standardize)"] {
            let classified = classify_text(&fixture::v3(body, "ridge"), TaskLabel::Regression);
            assert_eq!(classified.class, HostClass::SkipSlots, "{body}");
        }
        let middle = fixture::canonical(&format!(
            "(pipe X {INPUT_STAGE} (ridge) (standardize) (ridge))"
        ));
        let classified = classify_text(&middle, TaskLabel::Regression);
        assert_eq!(classified.class, HostClass::SkipSlots);
    }

    #[test]
    fn non_pipeline_programs_are_skipped() {
        for text in [fixture::SYMBOLIC, fixture::PORTFOLIO] {
            let classified = classify_text(text, TaskLabel::Regression);
            assert_eq!(classified.class, HostClass::SkipForm, "{text}");
        }
    }

    #[test]
    fn reserved_prefix_anywhere_skips_the_host() {
        let parameter = fixture::v3(
            "(cheb append=true degree=3 tag=xmotif_) (pair_prod append=true topk=24)",
            "ridge",
        );
        let reference = fixture::v3("(motif_0123456789abcdef) (standardize)", "ridge");
        for text in [parameter, reference] {
            let classified = classify_text(&text, TaskLabel::Regression);
            assert_eq!(classified.class, HostClass::SkipReserved, "{text}");
        }
    }

    #[test]
    fn window_text_is_an_exact_substring_of_the_canonical_host() {
        let mut checked = 0;
        for program in fixture::lines() {
            let classified = classify(&program).expect("classifies");
            if classified.class != HostClass::Used {
                continue;
            }
            let host_stages = match GenomeIr::parse(program.program()).expect("parses") {
                GenomeIr::Pipeline(expr) => expr.stages,
                _ => panic!("a used host is a pipeline"),
            };
            for run in runs(&classified.body) {
                for (start, length) in windows(&run, 6) {
                    let window = &classified.body[start..start + length];
                    let key = join(window);
                    assert!(program.program().contains(&key), "{key}");
                    let absolute = classified.body_start + start;
                    assert_eq!(
                        stage_texts(&host_stages[absolute..absolute + length]),
                        stage_texts(window)
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn signature_is_registry_derived() {
        let body = stages(fixture::E1);
        let table = fixture::table();
        assert_eq!(table.body.motifs[0].expansion, fixture::E1);
        let occurrences = &table.body.motifs[0].provenance;
        let signature = signature(&body, occurrences).expect("signature");
        assert_eq!(signature.slot, "transform_run");
        assert_eq!(signature.stages, 2);
        assert_eq!(signature.ops, ["cheb", "pair_prod"]);
        assert_eq!(signature.semantic_keys, ["cheb", "pair_prod"]);
        assert!(signature.fitted);
        assert!(signature.requires_labels);
        assert!(signature.tags.iter().any(|tag| tag == "target_aware"));
        assert_eq!(
            signature.tags,
            ["basis", "expansion", "interaction", "target_aware"]
        );
        assert_eq!(signature.appends, [true, true]);
        assert_eq!(
            signature.declared_tasks,
            TaskMask {
                regression: true,
                binary: true,
                multiclass: false,
            }
        );
        assert_eq!(
            signature.observed_tasks,
            [
                TaskLabel::Regression,
                TaskLabel::Binary,
                TaskLabel::Multiclass
            ]
        );
        assert!(signature.all_registry_generatable);
        let unregistered = stages("(symlog) (standardize)");
        assert!(super::signature(&unregistered, occurrences).is_err());
    }

    #[test]
    fn signature_round_trips_through_json_without_registry_types() {
        let body = stages(fixture::E1);
        let signature = signature(&body, &[]).expect("signature");
        assert_eq!(
            signature.shapes,
            [r#"{"multiply":"degree"}"#, r#"{"top_pairs":"topk"}"#]
        );
        assert!(signature.observed_tasks.is_empty());
        let text = serde_json::to_string(&signature).expect("serialize");
        let back: SignatureV1 = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, signature);
        let preserve = super::signature(&stages("(standardize) (sigmoid)"), &[]).expect("sig");
        assert_eq!(preserve.shapes, [r#""preserve""#, r#""preserve""#]);
        assert!(!preserve.requires_labels);
        assert_eq!(preserve.appends, [false, false]);
    }
}
