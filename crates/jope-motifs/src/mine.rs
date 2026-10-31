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
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;
    use crate::source::QualifiedProgram;
    use crate::{sha256_hex, NAME_PREFIX};

    fn host(body: &str, dataset: &str, record: &str) -> QualifiedProgram {
        fixture::qualified(
            &fixture::v3(body, "ridge"),
            TaskLabel::Regression,
            dataset,
            record,
        )
    }

    #[test]
    fn shared_body_is_mined_once_with_dataset_support() {
        let table = fixture::table();
        let shared: Vec<&MotifDef> = table
            .body
            .motifs
            .iter()
            .filter(|motif| motif.expansion == fixture::E1)
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].support.datasets, 4);
        assert!(table
            .body
            .motifs
            .iter()
            .all(|motif| motif.support.datasets >= 2));
        assert_eq!(table.body.motifs.len(), 3);
        assert!(table.verify().is_ok());
    }

    #[test]
    fn support_counts_distinct_datasets_records_and_recipes() {
        let table = fixture::table();
        let motif = &table.body.motifs[0];
        assert_eq!(motif.expansion, fixture::E1);
        assert_eq!(motif.support.datasets, 4);
        assert_eq!(motif.support.records, 5);
        assert_eq!(motif.support.recipes, 4);
        assert_eq!(motif.support.families, None);
        assert_eq!(
            motif.support.tasks,
            [
                TaskLabel::Regression,
                TaskLabel::Binary,
                TaskLabel::Multiclass
            ]
        );
        assert_eq!(motif.provenance.len(), 5);
        assert_eq!(table.body.inputs.qualified, 14);
        assert_eq!(table.body.inputs.used, 9);
        assert_eq!(table.body.inputs.excluded_adaptive, 1);
        assert_eq!(table.body.inputs.skipped_form, 1);
        assert_eq!(table.body.inputs.skipped_reserved, 1);
        assert_eq!(table.body.inputs.skipped_slots, 1);
        assert_eq!(table.body.inputs.skipped_unsupported, 1);
        assert_eq!(table.body.inputs.dropped_untyped, 0);
    }

    #[test]
    fn one_recipe_on_many_datasets_is_visible_in_the_recipe_count() {
        let body = "(standardize) (sigmoid)";
        let set = fixture::set_of(vec![
            host(body, "d1", "r1"),
            host(body, "d2", "r2"),
            host(body, "d3", "r3"),
        ]);
        let table = mine(&set, None, MineParameters::default()).expect("mines");
        assert_eq!(table.body.motifs.len(), 1);
        let support = &table.body.motifs[0].support;
        assert_eq!(support.datasets, 3);
        assert_eq!(support.records, 3);
        assert_eq!(support.recipes, 1);
        assert_eq!(table.body.hosts.len(), 1);
    }

    #[test]
    fn known_seed_hosts_are_flagged() {
        // Two member pipes share the fixture body; both hosts are known seeds.
        let quad = feat_math::members::CORE_V1
            .iter()
            .filter(|member| member.pipe.contains("pair_prod"))
            .map(|member| member.pipe)
            .collect::<Vec<_>>();
        assert_eq!(quad.len(), 2);
        let set = fixture::set_of(vec![
            fixture::qualified(quad[0], TaskLabel::Regression, "d1", "r1"),
            fixture::qualified(quad[1], TaskLabel::Regression, "d2", "r2"),
        ]);
        let seeded = mine(&set, None, MineParameters::default()).expect("mines");
        assert_eq!(seeded.body.motifs.len(), 1);
        assert_eq!(seeded.body.motifs[0].expansion, fixture::E1);
        assert!(seeded.body.motifs[0].support.every_host_is_a_known_seed);

        let table = fixture::table();
        assert!(!table.body.motifs[0].support.every_host_is_a_known_seed);
        assert!(table
            .body
            .motifs
            .iter()
            .all(|motif| !motif.support.every_host_is_a_known_seed));
    }

    #[test]
    fn families_are_counted_only_with_a_family_map() {
        let plain = fixture::table();
        assert!(plain.body.parameters.family_map_sha256.is_none());
        for motif in &plain.body.motifs {
            assert_eq!(motif.support.families, None);
            assert!(motif.provenance.iter().all(|o| o.family.is_none()));
        }

        let map = fixture::family_map();
        let parameters = MineParameters {
            family_map_sha256: Some(map.sha256().to_owned()),
            ..MineParameters::default()
        };
        let labelled = mine(&fixture::set(), Some(&map), parameters.clone()).expect("mines");
        assert!(labelled.verify().is_ok());
        assert_eq!(labelled, fixture::labelled_table());
        assert_eq!(
            labelled.body.parameters.family_map_sha256.as_deref(),
            Some(map.sha256())
        );
        let shared = &labelled.body.motifs[0];
        assert_eq!(shared.expansion, fixture::E1);
        assert_eq!(shared.support.families, Some(2));
        for motif in &labelled.body.motifs[1..] {
            assert_eq!(motif.support.families, Some(1));
        }

        let strict = MineParameters {
            min_families: 2,
            ..parameters
        };
        let filtered = mine(&fixture::set(), Some(&map), strict).expect("mines");
        assert_eq!(filtered.body.motifs.len(), 1);
        assert_eq!(filtered.body.motifs[0].expansion, fixture::E1);

        let error = mine(&fixture::set(), Some(&map), MineParameters::default())
            .expect_err("a map without its digest is refused");
        assert_eq!(code_of(&error), Some("invalid_parameters"));

        // The recorded digest is tied to the map bytes: a foreign digest is refused, and a
        // digest without a map is refused.
        let foreign = MineParameters {
            family_map_sha256: Some(fixture::digest("family-map")),
            ..MineParameters::default()
        };
        let error = mine(&fixture::set(), Some(&map), foreign.clone())
            .expect_err("a foreign map digest is refused");
        assert_eq!(code_of(&error), Some("invalid_parameters"));
        let error = mine(&fixture::set(), None, foreign).expect_err("a digest without a map");
        assert_eq!(code_of(&error), Some("invalid_parameters"));

        // A parsed map carries the digest of the bytes it was read from.
        let bytes = format!(
            "{{\n  \"{}\": \"{}\"\n}}\n",
            fixture::dataset("d1"),
            fixture::FAMILY_A
        )
        .into_bytes();
        let parsed = FamilyMap::parse(&bytes).expect("parses");
        assert_eq!(parsed.sha256(), sha256_hex(&bytes));
        assert_eq!(
            parsed.label(&fixture::dataset("d1")),
            Some(fixture::FAMILY_A)
        );
        assert!(FamilyMap::parse(b"{\"abc\":\"canaryfamily01\"}").is_err());
    }

    #[test]
    fn closed_subfragments_are_pruned_unless_support_grows() {
        let three = "(standardize) (cheb degree=2) (sigmoid)";
        let closed = fixture::set_of(vec![host(three, "d1", "r1"), host(three, "d2", "r2")]);
        let table = mine(&closed, None, MineParameters::default()).expect("mines");
        let expansions: Vec<&str> = table
            .body
            .motifs
            .iter()
            .map(|m| m.expansion.as_str())
            .collect();
        assert_eq!(expansions, [three]);

        let grown = fixture::set_of(vec![
            host(three, "d1", "r1"),
            host(three, "d2", "r2"),
            host("(standardize) (cheb degree=2)", "d3", "r3"),
        ]);
        let table = mine(&grown, None, MineParameters::default()).expect("mines");
        let expansions: Vec<&str> = table
            .body
            .motifs
            .iter()
            .map(|m| m.expansion.as_str())
            .collect();
        assert_eq!(expansions, ["(standardize) (cheb degree=2)", three]);
        assert!(table.verify().is_ok());
    }

    #[test]
    fn inclusion_filters_are_recorded_in_parameters() {
        let parameters = MineParameters {
            min_datasets: 3,
            ..MineParameters::default()
        };
        let table = mine(&fixture::set(), None, parameters.clone()).expect("mines");
        assert_eq!(table.body.parameters, parameters);
        assert_eq!(table.body.motifs.len(), 1);
        assert_eq!(table.body.motifs[0].expansion, fixture::E1);

        let none = MineParameters {
            min_datasets: 5,
            ..MineParameters::default()
        };
        let empty = mine(&fixture::set(), None, none.clone()).expect("mines");
        assert_eq!(empty.body.parameters, none);
        assert!(empty.body.motifs.is_empty());
        assert!(empty.body.hosts.is_empty());
        assert!(empty.verify().is_ok());

        let short = MineParameters {
            max_stages: 2,
            ..MineParameters::default()
        };
        let table = mine(&fixture::set(), None, short).expect("mines");
        assert!(table.body.motifs.iter().all(|m| m.signature.stages == 2));

        for bad in [
            MineParameters {
                min_datasets: 0,
                ..MineParameters::default()
            },
            MineParameters {
                max_stages: 1,
                ..MineParameters::default()
            },
            MineParameters {
                max_stages: 7,
                ..MineParameters::default()
            },
            MineParameters {
                min_families: 2,
                ..MineParameters::default()
            },
            MineParameters {
                min_families: 0,
                ..MineParameters::default()
            },
        ] {
            let error = mine(&fixture::set(), None, bad).expect_err("refused");
            assert_eq!(code_of(&error), Some("invalid_parameters"));
        }
    }

    #[test]
    fn adaptive_label_rows_are_excluded_and_counted() {
        let table = fixture::table();
        assert_eq!(table.body.inputs.excluded_adaptive, 1);
        let adaptive_dataset = fixture::dataset("d7");
        for motif in &table.body.motifs {
            for occurrence in &motif.provenance {
                assert_ne!(occurrence.dataset_sha256, adaptive_dataset);
                assert!(!occurrence.label.is_adaptive());
            }
        }

        let body = "(standardize) (sigmoid)";
        let only_adaptive_second = fixture::set_of(vec![
            host(body, "d1", "r1"),
            fixture::qualified_with(
                &fixture::v3(body, "ridge"),
                TaskLabel::Regression,
                "d2",
                "r2",
                fixture::adaptive_label(),
            ),
        ]);
        let table = mine(&only_adaptive_second, None, MineParameters::default()).expect("mines");
        assert!(table.body.motifs.is_empty());
        assert_eq!(table.body.inputs.excluded_adaptive, 1);
    }

    #[test]
    fn mining_is_deterministic_and_input_order_independent() {
        let first = fixture::table().to_json().expect("json");
        let second = fixture::table().to_json().expect("json");
        assert_eq!(first, second);
        let mut lines = fixture::lines();
        lines.reverse();
        let reversed = mine(&fixture::set_of(lines), None, MineParameters::default())
            .expect("mines")
            .to_json()
            .expect("json");
        assert_eq!(first, reversed);
    }

    #[test]
    fn noncanonical_or_hash_mismatched_input_fails_closed() {
        let good = host("(standardize) (sigmoid)", "d1", "r1");
        let written = "(pipe X (cheb degree=3 append=true) (ridge))";
        let noncanonical = good.tampered(written, &sha256_hex(written.as_bytes()));
        let set = fixture::set_of(vec![noncanonical]);
        let error = mine(&set, None, MineParameters::default()).expect_err("refused");
        assert_eq!(code_of(&error), Some("noncanonical_program"));

        let mismatched = good.tampered(good.program(), &fixture::digest("other"));
        let set = fixture::set_of(vec![mismatched]);
        let error = mine(&set, None, MineParameters::default()).expect_err("refused");
        assert_eq!(code_of(&error), Some("recipe_hash_mismatch"));

        // Adaptive rows are excluded before any re-check, by the first rule.
        let adaptive = fixture::qualified_with(
            good.program(),
            TaskLabel::Regression,
            "d2",
            "r2",
            fixture::adaptive_label(),
        )
        .tampered(good.program(), &fixture::digest("other"));
        let table = mine(
            &fixture::set_of(vec![adaptive]),
            None,
            MineParameters::default(),
        )
        .expect("excluded, not an error");
        assert_eq!(table.body.inputs.excluded_adaptive, 1);
    }

    #[test]
    fn every_occurrence_rewrites_and_expands_back_to_its_host() {
        let table = fixture::table();
        let mut checked = 0;
        for motif in &table.body.motifs {
            for occurrence in &motif.provenance {
                let host = &table.body.hosts[&occurrence.recipe_sha256];
                let offset = usize::try_from(occurrence.stage_offset).expect("offset");
                for at in [At::Offset(offset), At::All, At::First] {
                    let rewritten = rewrite(host, motif, at).expect("rewrites");
                    assert!(rewritten.contains(&motif.name));
                    assert!(!rewritten.contains(&motif.expansion));
                    let expanded = expand(&rewritten, &table).expect("expands");
                    assert_eq!(&expanded.canonical, host);
                    assert_eq!(expanded.recipe_sha256, occurrence.recipe_sha256);
                    assert!(!expanded.canonical.contains(NAME_PREFIX));
                    assert_eq!(expanded.used, std::slice::from_ref(&motif.id));
                }
                checked += 1;
            }
        }
        assert!(checked >= 9);
    }

    #[test]
    fn input_manifest_keeps_every_input_identity_even_for_an_empty_table() {
        let set = fixture::set();
        let table = fixture::table();
        let inputs = input_manifest(&set, &table).expect("inputs");
        assert_eq!(inputs.schema, INPUTS_SCHEMA);
        assert_eq!(inputs.table_sha256, table.table_sha256);
        assert_eq!(inputs.source, table.body.source);
        assert_eq!(inputs.records.len(), 14);
        assert!(inputs
            .records
            .windows(2)
            .all(|pair| pair[0].record_sha256 < pair[1].record_sha256));
        for line in fixture::lines() {
            let record = inputs
                .records
                .iter()
                .find(|record| record.record_sha256 == line.record_sha256())
                .expect("every input is listed");
            assert_eq!(record.comparison_sha256, line.comparison_sha256());
            assert_eq!(record.recipe_sha256, line.recipe_sha256());
            assert_eq!(record.dataset_sha256, line.dataset_sha256());
            assert_eq!(record.cut_sha256, line.cut_sha256());
            assert_eq!(record.task, line.task());
            assert_eq!(&record.label, line.label());
        }
        let count = |class: InputClass| {
            inputs
                .records
                .iter()
                .filter(|record| record.class == class)
                .count()
        };
        assert_eq!(count(InputClass::Used), 9);
        assert_eq!(count(InputClass::ExcludedAdaptive), 1);
        assert_eq!(count(InputClass::SkippedForm), 1);
        assert_eq!(count(InputClass::SkippedReserved), 1);
        assert_eq!(count(InputClass::SkippedSlots), 1);
        assert_eq!(count(InputClass::SkippedUnsupported), 1);

        // It round-trips through JSON.
        let bytes = serde_json::to_vec(&inputs).expect("json");
        let back: InputManifest = serde_json::from_slice(&bytes).expect("parses");
        assert_eq!(back, inputs);

        // A table with no fragment still lists every input it examined.
        let parameters = MineParameters {
            min_datasets: 5,
            ..MineParameters::default()
        };
        let empty = mine(&set, None, parameters).expect("mines");
        assert!(empty.body.motifs.is_empty());
        let listed = input_manifest(&set, &empty).expect("inputs");
        assert_eq!(listed.records, inputs.records);
        assert_eq!(listed.table_sha256, empty.table_sha256);

        // A table mined from other inputs is refused.
        let other = fixture::set_of(fixture::small_lines());
        let error = input_manifest(&other, &table).expect_err("refused");
        assert_eq!(code_of(&error), Some("provenance_inconsistent"));
    }

    #[test]
    fn current_roster_shape_yields_an_empty_table() {
        let mut lines = Vec::new();
        for (task, heads) in fixture::roster_heads() {
            for (index, head) in heads.iter().enumerate() {
                for dataset in ["d1", "d2"] {
                    let record = format!("{}-{head}-{index}-{dataset}", task.as_str());
                    lines.push(fixture::qualified(
                        &fixture::roster(head),
                        task,
                        dataset,
                        &record,
                    ));
                }
            }
        }
        let count = u32::try_from(lines.len()).expect("count");
        let table = mine(&fixture::set_of(lines), None, MineParameters::default()).expect("mines");
        assert!(table.body.motifs.is_empty());
        assert!(table.body.hosts.is_empty());
        assert_eq!(table.body.inputs.qualified, count);
        assert_eq!(table.body.inputs.used, count);
        assert_eq!(table.body.inputs.skipped_unsupported, 0);
        assert!(table.verify().is_ok());
    }
}
