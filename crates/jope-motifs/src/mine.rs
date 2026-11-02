//! Build a private fragment table from qualified programs (section 4 of the design).
//!
//! The statistic is presence among qualified programs. It is never described as uplift or
//! usefulness, and nothing here is a qualification or promotion gate: the dataset and
//! family minimums only decide which entries enter the table, and they are recorded.

use crate::expand::{expand, rewrite, At};
use crate::fragment::{self, HostClass};
use crate::source::{FamilyMap, LabelOrigin, QualifiedSet, TaskLabel};
use crate::table::{
    motif_id, motif_name, motif_order, occurrence_order, support_of, EngineBinding, InputCounts,
    Lifecycle, MineParameters, MotifDef, MotifTable, Occurrence, Promotion, SourceBinding, Support,
    TableBody,
};
use crate::Code;
use anyhow::{ensure, Result};
use feat_math::dsl::Stage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Private label of the input list written beside a mined table. There is no schema file.
pub const INPUTS_SCHEMA: &str = "jope.motif-inputs/v1";

/// How one admitted input was used by the table builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputClass {
    /// Admitted as a host eligible for fragment mining.
    Used,
    /// Excluded because optimization could have seen test labels.
    ExcludedAdaptive,
    /// Skipped because it contains a reserved fragment name.
    SkippedReserved,
    /// Skipped because it is not a pipeline program.
    SkippedForm,
    /// Skipped because its input/transform slots are unsupported.
    SkippedSlots,
    /// Skipped because the task-bound grammar refused the program.
    SkippedUnsupported,
}

impl From<HostClass> for InputClass {
    fn from(class: HostClass) -> Self {
        match class {
            HostClass::Used => Self::Used,
            HostClass::ExcludedAdaptive => Self::ExcludedAdaptive,
            HostClass::SkipReserved => Self::SkippedReserved,
            HostClass::SkipForm => Self::SkippedForm,
            HostClass::SkipSlots => Self::SkippedSlots,
            HostClass::SkipUnsupported => Self::SkippedUnsupported,
        }
    }
}

/// The identity of one admitted input: its record, comparison, recipe, dataset, cut, task and
/// label provenance, and how the table builder used it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputRecord {
    /// Digest of the qualified source record.
    pub record_sha256: String,
    /// Digest of the measured comparison identity.
    pub comparison_sha256: String,
    /// Digest of the canonical program text.
    pub recipe_sha256: String,
    /// Digest identifying the source dataset.
    pub dataset_sha256: String,
    /// Digest identifying the dataset cut.
    pub cut_sha256: String,
    /// Task under which the program was qualified.
    pub task: TaskLabel,
    /// Label provenance from the qualified record.
    pub label: LabelOrigin,
    /// Mining disposition assigned to this input.
    pub class: InputClass,
}

/// Every admitted input of one mined table, including those no fragment kept, so the input
/// comparison identities survive even when the table is empty. Private, like the table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputManifest {
    /// Private input-manifest schema identifier.
    pub schema: String,
    /// Digest of the table this input list accompanies.
    pub table_sha256: String,
    /// Qualified source identity and counts.
    pub source: SourceBinding,
    /// Sorted by record digest.
    pub records: Vec<InputRecord>,
}

/// The input list of a table mined from `set`. Each input is classified again, and the class
/// counts and source binding must be exactly those the table records
/// (`provenance_inconsistent` otherwise).
pub fn input_manifest(set: &QualifiedSet, table: &MotifTable) -> Result<InputManifest> {
    let mut counts = InputCounts {
        qualified: u32::try_from(set.programs().len())?,
        dropped_untyped: table.body.inputs.dropped_untyped,
        ..InputCounts::default()
    };
    let mut records = BTreeMap::new();
    for program in set.programs() {
        let class = fragment::classify(program)?.class;
        let count = match class {
            HostClass::Used => &mut counts.used,
            HostClass::ExcludedAdaptive => &mut counts.excluded_adaptive,
            HostClass::SkipReserved => &mut counts.skipped_reserved,
            HostClass::SkipForm => &mut counts.skipped_form,
            HostClass::SkipSlots => &mut counts.skipped_slots,
            HostClass::SkipUnsupported => &mut counts.skipped_unsupported,
        };
        *count += 1;
        records.insert(
            program.record_sha256().to_owned(),
            InputRecord {
                record_sha256: program.record_sha256().to_owned(),
                comparison_sha256: program.comparison_sha256().to_owned(),
                recipe_sha256: program.recipe_sha256().to_owned(),
                dataset_sha256: program.dataset_sha256().to_owned(),
                cut_sha256: program.cut_sha256().to_owned(),
                task: program.task(),
                label: program.label().clone(),
                class: InputClass::from(class),
            },
        );
    }
    let source = SourceBinding {
        schema: set.source_schema().to_owned(),
        sha256: set.source_sha256().to_owned(),
        programs: counts.qualified,
        excluded_at_source: set.excluded_at_source(),
    };
    ensure!(
        counts == table.body.inputs
            && source == table.body.source
            && records.len() == set.programs().len(),
        Code("provenance_inconsistent")
    );
    Ok(InputManifest {
        schema: INPUTS_SCHEMA.to_owned(),
        table_sha256: table.table_sha256.clone(),
        source,
        records: records.into_values().collect(),
    })
}

/// Every occurrence of one window text.
struct Window {
    stages: Vec<Stage>,
    occurrences: Vec<Occurrence>,
}

/// A window text that passed the inclusion filter.
struct Included {
    key: String,
    stages: Vec<Stage>,
    texts: Vec<String>,
    datasets: BTreeSet<String>,
    occurrences: Vec<Occurrence>,
    support: Support,
}

/// Mine the transform runs of every host into a sealed, quarantined table. The output bytes
/// do not depend on the input order.
pub fn mine(
    set: &QualifiedSet,
    families: Option<&FamilyMap>,
    parameters: MineParameters,
) -> Result<MotifTable> {
    parameters.check()?;
    // The recorded map digest must be the digest of the map actually used: both absent, or
    // both present and equal.
    ensure!(
        parameters.family_map_sha256.as_deref() == families.map(FamilyMap::sha256),
        Code("invalid_parameters")
    );
    let max_stages = usize::try_from(parameters.max_stages)?;
    let mut inputs = InputCounts {
        qualified: u32::try_from(set.programs().len())?,
        ..InputCounts::default()
    };

    let mut windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut host_texts: BTreeMap<String, String> = BTreeMap::new();
    for program in set.programs() {
        let classified = fragment::classify(program)?;
        match classified.class {
            HostClass::ExcludedAdaptive => inputs.excluded_adaptive += 1,
            HostClass::SkipReserved => inputs.skipped_reserved += 1,
            HostClass::SkipForm => inputs.skipped_form += 1,
            HostClass::SkipSlots => inputs.skipped_slots += 1,
            HostClass::SkipUnsupported => inputs.skipped_unsupported += 1,
            HostClass::Used => inputs.used += 1,
        }
        if classified.class != HostClass::Used {
            continue;
        }
        for run in fragment::runs(&classified.body) {
            for (start, length) in fragment::windows(&run, max_stages) {
                let stages = &classified.body[start..start + length];
                let key = fragment::join(stages);
                debug_assert!(program.program().contains(&key));
                let family = families
                    .map(|map| {
                        map.label(program.dataset_sha256())
                            .map(str::to_owned)
                            .ok_or(Code("invalid_family_map"))
                    })
                    .transpose()?;
                let occurrence = Occurrence {
                    record_sha256: program.record_sha256().to_owned(),
                    comparison_sha256: program.comparison_sha256().to_owned(),
                    dataset_sha256: program.dataset_sha256().to_owned(),
                    cut_sha256: program.cut_sha256().to_owned(),
                    recipe_sha256: program.recipe_sha256().to_owned(),
                    task: program.task(),
                    stage_offset: u32::try_from(classified.body_start + start)?,
                    family,
                    label: program.label().clone(),
                };
                windows
                    .entry(key)
                    .or_insert_with(|| Window {
                        stages: stages.to_vec(),
                        occurrences: Vec::new(),
                    })
                    .occurrences
                    .push(occurrence);
                host_texts.insert(
                    program.recipe_sha256().to_owned(),
                    program.program().to_owned(),
                );
            }
        }
    }

    let with_families = families.is_some();
    let mut included = Vec::new();
    for (key, window) in windows {
        let support = support_of(&window.occurrences, &host_texts, with_families)?;
        let families_pass =
            !with_families || support.families.unwrap_or(0) >= parameters.min_families;
        if support.datasets < parameters.min_datasets || !families_pass {
            continue;
        }
        included.push(Included {
            texts: fragment::stage_texts(&window.stages),
            datasets: window
                .occurrences
                .iter()
                .map(|o| o.dataset_sha256.clone())
                .collect(),
            key,
            stages: window.stages,
            occurrences: window.occurrences,
            support,
        });
    }

    // Closed-fragment pruning: longest first, so a window is dropped when a kept, strictly
    // longer window contains it with the identical dataset set.
    included.sort_by(|a, b| {
        b.texts
            .len()
            .cmp(&a.texts.len())
            .then_with(|| a.key.cmp(&b.key))
    });
    let mut retained: Vec<Included> = Vec::new();
    for window in included {
        let covered = retained.iter().any(|longer| {
            fragment::is_sub_run(&window.texts, &longer.texts) && longer.datasets == window.datasets
        });
        if !covered {
            retained.push(window);
        }
    }

    let mut motifs = Vec::with_capacity(retained.len());
    for window in retained {
        let typed = window
            .support
            .tasks
            .iter()
            .all(|task| fragment::typed_for(&window.key, *task));
        if !typed {
            inputs.dropped_untyped += 1;
            continue;
        }
        let signature = fragment::signature(&window.stages, &window.occurrences)?;
        let id = motif_id(&window.key);
        let mut provenance = window.occurrences;
        provenance.sort_by(occurrence_order);
        motifs.push(MotifDef {
            name: motif_name(&id),
            id,
            expansion: window.key,
            signature,
            lifecycle: Lifecycle::Quarantined,
            promotion: Promotion::required_not_run(),
            support: window.support,
            provenance,
        });
    }
    motifs.sort_by(motif_order);

    let referenced: BTreeSet<String> = motifs
        .iter()
        .flat_map(|motif| motif.provenance.iter().map(|o| o.recipe_sha256.clone()))
        .collect();
    let hosts: BTreeMap<String, String> = host_texts
        .into_iter()
        .filter(|(recipe, _)| referenced.contains(recipe))
        .collect();
    let body = TableBody {
        engine: EngineBinding::current(),
        source: SourceBinding {
            schema: set.source_schema().to_owned(),
            sha256: set.source_sha256().to_owned(),
            programs: inputs.qualified,
            excluded_at_source: set.excluded_at_source(),
        },
        parameters,
        inputs,
        motifs,
        hosts,
    };
    let table = MotifTable::seal(body)?;

    // Self-check: every occurrence rewrites and expands back to its host, byte for byte.
    for motif in &table.body.motifs {
        for occurrence in &motif.provenance {
            let host = table
                .body
                .hosts
                .get(&occurrence.recipe_sha256)
                .ok_or(Code("rewrite_mismatch"))?;
            let offset = usize::try_from(occurrence.stage_offset)?;
            let rewritten =
                rewrite(host, motif, At::Offset(offset)).map_err(|_| Code("rewrite_mismatch"))?;
            let expanded = expand(&rewritten, &table).map_err(|_| Code("rewrite_mismatch"))?;
            ensure!(expanded.canonical == *host, Code("rewrite_mismatch"));
        }
    }
    Ok(table)
}

#[cfg(test)]
#[path = "mine/tests.rs"]
mod tests;
