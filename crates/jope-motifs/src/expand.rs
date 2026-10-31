//! Expansion and rewrite of fragment references.
//!
//! A reference is a parameterless stage named with the reserved prefix. It is expanded before
//! canonicalization; the recipe digest is always the sha256 of the expanded canonical text,
//! the same value the native controller records.

use crate::fragment::{self, expansion_stages};
use crate::table::{MotifDef, MotifTable};
use crate::{sha256_hex, Code, NAME_PREFIX};
use anyhow::{bail, ensure, Result};
use feat_math::dsl::{parse_dsl, DslExpr, ParamValue, Stage};
use feat_math::kernel::pipeline::PipelineSlots;
use feat_math::GenomeIr;
use std::collections::BTreeSet;

/// The expanded canonical program, its recipe digest and the ids of the fragments used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expanded {
    /// Expanded, canonical program text.
    pub canonical: String,
    /// SHA-256 digest of the expanded canonical text.
    pub recipe_sha256: String,
    /// Sorted, distinct fragment ids.
    pub used: Vec<String>,
}

/// Expand every reference in `src` and return the canonical program.
///
/// Without the reserved prefix this is the identity path: the canonical text of `src`.
/// Otherwise the program must be a pipeline; a reference must be a parameterless stage that
/// names a table entry; no string parameter may carry the prefix; and the result must be a
/// canonical, slotted pipeline without the prefix.
pub fn expand(src: &str, table: &MotifTable) -> Result<Expanded> {
    if !src.contains(NAME_PREFIX) {
        let canonical = GenomeIr::parse(src)
            .map_err(|_| Code("expansion_not_canonical"))?
            .canonical();
        return Ok(Expanded {
            recipe_sha256: sha256_hex(canonical.as_bytes()),
            canonical,
            used: Vec::new(),
        });
    }
    ensure!(
        src.trim_start().starts_with("(pipe"),
        Code("reference_position")
    );
    let parsed = parse_dsl(src.trim()).map_err(|_| Code("expansion_not_canonical"))?;
    let mut stages: Vec<Stage> = Vec::with_capacity(parsed.stages.len());
    let mut used = BTreeSet::new();
    for stage in parsed.stages {
        let prefixed_string = stage
            .params
            .values()
            .any(|value| matches!(value, ParamValue::Str(text) if text.contains(NAME_PREFIX)));
        ensure!(!prefixed_string, Code("reference_position"));
        if !stage.op.starts_with(NAME_PREFIX) {
            stages.push(stage);
            continue;
        }
        ensure!(stage.params.is_empty(), Code("reference_position"));
        let Some(motif) = table.body.motifs.iter().find(|m| m.name == stage.op) else {
            bail!(Code("unknown_reference"))
        };
        let expansion =
            expansion_stages(&motif.expansion).map_err(|_| Code("expansion_not_canonical"))?;
        stages.extend(expansion);
        used.insert(motif.id.clone());
    }
    let canonical = DslExpr { stages }.to_string();
    let reparsed = GenomeIr::parse(&canonical).map_err(|_| Code("expansion_not_canonical"))?;
    ensure!(
        reparsed.canonical() == canonical,
        Code("expansion_not_canonical")
    );
    // A reference left after one expansion is reported as nested before the slot check,
    // which would otherwise refuse the unknown stage first.
    ensure!(!canonical.contains(NAME_PREFIX), Code("nested_reference"));
    let GenomeIr::Pipeline(expr) = &reparsed else {
        bail!(Code("reference_position"))
    };
    ensure!(
        PipelineSlots::from_pipeline(expr).is_ok(),
        Code("reference_position")
    );
    Ok(Expanded {
        recipe_sha256: sha256_hex(canonical.as_bytes()),
        canonical,
        used: used.into_iter().collect(),
    })
}

/// Which occurrences of the expansion a rewrite replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    /// The first occurrence.
    First,
    /// Every non-overlapping occurrence, left to right.
    All,
    /// The occurrence that starts at this absolute stage index.
    Offset(usize),
}

/// Replace occurrences of a fragment's stages in `host` with its reference. The result is
/// not canonical and is never persisted; it exists to be expanded again.
pub fn rewrite(host: &str, motif: &MotifDef, at: At) -> Result<String> {
    let parsed = parse_dsl(host.trim()).map_err(|_| Code("rewrite_mismatch"))?;
    let pattern = fragment::stage_texts(
        &expansion_stages(&motif.expansion).map_err(|_| Code("expansion_not_canonical"))?,
    );
    ensure!(!pattern.is_empty(), Code("expansion_not_canonical"));
    let texts = fragment::stage_texts(&parsed.stages);
    let width = pattern.len();
    let chosen: Vec<usize> = match at {
        At::Offset(offset) => {
            let end = offset.checked_add(width).ok_or(Code("rewrite_mismatch"))?;
            ensure!(
                texts.get(offset..end) == Some(pattern.as_slice()),
                Code("rewrite_mismatch")
            );
            vec![offset]
        }
        At::First | At::All => {
            let mut starts = Vec::new();
            let mut index = 0;
            while index + width <= texts.len() {
                if texts[index..index + width] == pattern[..] {
                    starts.push(index);
                    if at == At::First {
                        break;
                    }
                    index += width;
                } else {
                    index += 1;
                }
            }
            starts
        }
    };
    ensure!(!chosen.is_empty(), Code("rewrite_mismatch"));
    let mut stages = Vec::with_capacity(parsed.stages.len());
    let mut next = 0;
    for start in chosen {
        stages.extend(parsed.stages[next..start].iter().cloned());
        stages.push(Stage::new(motif.name.clone()));
        next = start + width;
    }
    stages.extend(parsed.stages[next..].iter().cloned());
    Ok(DslExpr { stages }.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;
    use crate::fragment::{classification_schema, input_stage};
    use crate::source::TaskLabel;
    use feat_math::advanced_v3::validate_v3_genome;

    fn shared(table: &MotifTable) -> &MotifDef {
        table
            .body
            .motifs
            .iter()
            .find(|m| m.expansion == fixture::E1)
            .expect("the shared fragment")
    }

    fn expand_code(src: &str, table: &MotifTable) -> Option<&'static str> {
        expand(src, table).err().and_then(|error| code_of(&error))
    }

    #[test]
    fn expand_is_identity_without_references() {
        let table = fixture::table();
        for host in table.body.hosts.values() {
            let expanded = expand(host, &table).expect("expands");
            assert_eq!(&expanded.canonical, host);
            assert_eq!(expanded.recipe_sha256, sha256_hex(host.as_bytes()));
            assert!(expanded.used.is_empty());
        }
        let written = feat_math::members::CORE_V1[1].pipe;
        let expanded = expand(written, &table).expect("expands");
        assert_eq!(expanded.canonical, fixture::canonical(written));
        assert_ne!(expanded.canonical, written);
        assert_eq!(
            expand_code("(pipe X", &table),
            Some("expansion_not_canonical")
        );
    }

    #[test]
    fn expand_splices_exact_stage_text() {
        let table = fixture::table();
        let motif = shared(&table);
        let input = input_stage().expect("input stage");
        let written = format!("(pipe X {input} ({}) (ridge))", motif.name);
        let expanded = expand(&written, &table).expect("expands");
        assert_eq!(
            expanded.canonical,
            format!("(pipe X {input} {} (ridge))", fixture::E1)
        );
        assert_eq!(
            expanded.recipe_sha256,
            sha256_hex(expanded.canonical.as_bytes())
        );
        assert_eq!(expanded.used, std::slice::from_ref(&motif.id));
        assert!(expanded.canonical.contains(&motif.expansion));
    }

    #[test]
    fn head_position_reference_expands_to_trailing_transforms_with_the_default_head() {
        let table = fixture::table();
        let motif = shared(&table);
        let input = input_stage().expect("input stage");
        let expanded =
            expand(&format!("(pipe X {input} ({}))", motif.name), &table).expect("expands");
        assert_eq!(
            expanded.canonical,
            format!("(pipe X {input} {})", fixture::E1)
        );
        let ir = GenomeIr::parse(&expanded.canonical).expect("parses");
        let GenomeIr::Pipeline(expr) = &ir else {
            panic!("a pipeline")
        };
        let slots = PipelineSlots::from_pipeline(expr).expect("slots");
        assert_eq!(slots.transforms.len(), 2);
        assert_eq!(slots.learner.op, "ridge");
        for task in [TaskLabel::Regression, TaskLabel::Binary] {
            let spec = task.task_spec().expect("spec");
            assert!(validate_v3_genome(&ir, &spec, &classification_schema()).is_ok());
        }
    }

    #[test]
    fn references_before_target_or_input_or_after_the_learner_are_refused() {
        let table = fixture::table();
        let name = &shared(&table).name;
        let input = input_stage().expect("input stage");
        for written in [
            format!("(pipe X ({name}) {input} (ridge))"),
            format!("(pipe X ({name}) (target kind=log1p) (ridge))"),
            format!("(pipe X {input} (ridge) ({name}))"),
        ] {
            assert_eq!(
                expand_code(&written, &table),
                Some("reference_position"),
                "{written}"
            );
        }
    }

    #[test]
    fn unknown_or_parameterised_references_are_refused() {
        let table = fixture::table();
        let name = &shared(&table).name;
        let input = input_stage().expect("input stage");
        let unknown = format!("(pipe X {input} (motif_0000000000000000) (ridge))");
        assert_eq!(expand_code(&unknown, &table), Some("unknown_reference"));
        let parameterised = format!("(pipe X {input} ({name} degree=3) (ridge))");
        assert_eq!(
            expand_code(&parameterised, &table),
            Some("reference_position")
        );
    }

    #[test]
    fn references_in_parameters_portfolio_members_or_closed_form_text_are_refused() {
        let table = fixture::table();
        let name = &shared(&table).name;
        let input = input_stage().expect("input stage");
        for written in [
            format!("(pipe X {input} (cheb degree=3 tag={name}) (ridge))"),
            format!("(pipe X {input} (portfolio preset=core tag=x{name}))"),
            format!("(portfolio (member (pipe X ({name}) (ridge))) (blend))"),
            format!("(sym affine \"col(0) + {name}\")"),
        ] {
            assert_eq!(
                expand_code(&written, &table),
                Some("reference_position"),
                "{written}"
            );
        }
    }

    #[test]
    fn expanded_text_never_contains_the_reserved_prefix() {
        let table = fixture::table();
        let input = input_stage().expect("input stage");
        let keyed = format!("(pipe X {input} (cheb motif_k=1) (ridge))");
        assert_eq!(expand_code(&keyed, &table), Some("nested_reference"));

        let mut nested = table.clone();
        let inner = nested.body.motifs[1].name.clone();
        nested.body.motifs[0].expansion = format!("({inner}) (standardize)");
        let outer = nested.body.motifs[0].name.clone();
        let written = format!("(pipe X {input} ({outer}) (ridge))");
        assert_eq!(expand_code(&written, &nested), Some("nested_reference"));

        for motif in &table.body.motifs {
            for host in table.body.hosts.values() {
                if let Ok(rewritten) = rewrite(host, motif, At::All) {
                    let expanded = expand(&rewritten, &table).expect("expands");
                    assert!(!expanded.canonical.contains(NAME_PREFIX));
                }
            }
        }
    }

    #[test]
    fn rewrite_first_all_and_offset_round_trip() {
        let table = fixture::table();
        let motif = shared(&table);
        let input = input_stage().expect("input stage");
        let host = fixture::canonical(&format!(
            "(pipe X {input} {} (standardize) {} (ridge))",
            fixture::E1,
            fixture::E1
        ));
        let name = &motif.name;
        let first = rewrite(&host, motif, At::First).expect("first");
        assert_eq!(
            first,
            format!(
                "(pipe X {input} ({name}) (standardize) {} (ridge))",
                fixture::E1
            )
        );
        let all = rewrite(&host, motif, At::All).expect("all");
        assert_eq!(
            all,
            format!("(pipe X {input} ({name}) (standardize) ({name}) (ridge))")
        );
        let second = rewrite(&host, motif, At::Offset(4)).expect("offset");
        assert_eq!(
            second,
            format!(
                "(pipe X {input} {} (standardize) ({name}) (ridge))",
                fixture::E1
            )
        );
        for rewritten in [&first, &all, &second] {
            let expanded = expand(rewritten, &table).expect("expands");
            assert_eq!(expanded.canonical, host);
            assert_eq!(expanded.recipe_sha256, sha256_hex(host.as_bytes()));
        }
        for bad in [At::Offset(2), At::Offset(40)] {
            let error = rewrite(&host, motif, bad).expect_err("no occurrence there");
            assert_eq!(code_of(&error), Some("rewrite_mismatch"));
        }
        let absent = fixture::canonical("(pipe X (standardize) (ridge))");
        let error = rewrite(&absent, motif, At::First).expect_err("no occurrence");
        assert_eq!(code_of(&error), Some("rewrite_mismatch"));
    }
}
