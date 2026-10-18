//! Legacy Prototype Seam Regression (MR02) — jope-motifs integration target.
//!
//! Exercises `I56` (`jope-motifs` `promote.rs` + `table.rs` + `validate.rs`) alongside
//! `feat_math::kernel::ir` (`I07`/`I08`/`I09`) and `feat_math::combinators` (`I54`/`I55`).

pub use jope_motifs::{expand, fragment, source, table, validate, NAME_PREFIX};
#[path = "../src/promote.rs"]
#[allow(dead_code)]
mod promote;

use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::{Constraints, InputSchema, RowSemantics};
use feat_math::kernel::Program;
use feat_math::GenomeIr;
use jope_motifs::validate::{CheckResult, MotifResult, Validation, VALIDATION_SCHEMA};
use promote::{
    ContrastObservation, EvidenceBoundary, EvidenceRole, ExpansionLimits, ExpansionWitness,
    MetricDirection, MotifAdmission, MotifCandidate, PreregisteredContrast, ProbeSimilarity,
    VocabularyMigration, ADMISSION_SCHEMA, CONTRAST_SCHEMA,
};
use sha2::{Digest, Sha256};

const TABLE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

#[test]
fn typed_graph_effects_seam_verified() {
    let schema = InputSchema::new(
        (0..2).map(|i| format!("c_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap();
    let prog = Program::parse_legacy("(pipe X (standardize) (ridge))").unwrap();
    let bridge = LegacyBridge::import(&prog, &schema, &Constraints::default()).unwrap();
    assert!(matches!(bridge.bridge, BridgeStatus::Typed { .. }));
}

#[test]
fn combinators_motifs_seam_verified() {
    let expansion = "(standardize) (poly_interact)";
    let id = table::motif_id(expansion);
    let candidate = MotifCandidate::draft_for_gate(
        TABLE_SHA.to_owned(),
        id.clone(),
        table::motif_name(&id),
        expansion.to_owned(),
        0,
    )
    .unwrap();
    let migration =
        VocabularyMigration::new(&candidate, "kernel2-baseline", 3, "kernel2-motifs-r1", 1)
            .unwrap();
    let (old_id, old_rev) = migration.old_profile();
    let (new_id, new_rev) = migration.new_profile();

    let families = ["fam-a", "fam-a", "fam-b", "fam-b", "fam-c", "fam-c"];
    let scores = [0.78, 0.80, 0.79, 0.81, 0.77, 0.82];
    let observations: Vec<ContrastObservation> = scores
        .iter()
        .enumerate()
        .map(|(idx, &macro_score)| ContrastObservation {
            pair_id: format!("pair-{idx}"),
            dataset_sha256: digest(&format!("ds-{idx}")),
            cut_sha256: digest(&format!("cut-{idx}")),
            context_sha256: digest(&format!("ctx-{idx}")),
            family_id: families[idx].to_owned(),
            seed: idx as u64,
            role: EvidenceRole::ResearchDevelopment,
            baseline_score: 1.0,
            macro_score,
        })
        .collect();
    let mut contrast = PreregisteredContrast {
        schema: CONTRAST_SCHEMA.to_owned(),
        preregistration_sha256: String::new(),
        motif_id: candidate.motif_id().to_owned(),
        old_profile_id: old_id.to_owned(),
        old_profile_revision: old_rev,
        new_profile_id: new_id.to_owned(),
        new_profile_revision: new_rev,
        direction: MetricDirection::LowerIsBetter,
        minimum_mean_gain: 0.1,
        observations,
    };
    contrast.preregistration_sha256 = contrast.computed_preregistration_sha256().unwrap();

    let passed = CheckResult {
        passed: true,
        exercised: 1,
        failures: Vec::new(),
    };
    let val = Validation {
        schema: VALIDATION_SCHEMA.to_owned(),
        table_sha256: candidate.table_sha256().to_owned(),
        source_verified: true,
        source_digest_equal: Some(true),
        motifs: vec![MotifResult {
            index: candidate.index(),
            round_trip: passed.clone(),
            determinism: passed.clone(),
            context_compatibility: passed.clone(),
            name_absence: passed,
            not_exercised: 0,
            backend: Vec::new(),
        }],
    };
    let input = fragment::input_stage().unwrap();
    let host = GenomeIr::parse(&format!("(pipe X {input} {expansion} (ridge))"))
        .unwrap()
        .canonical();
    let witness = ExpansionWitness::verify_host(
        &candidate,
        &host,
        1,
        ExpansionLimits {
            max_expansion_stages: 6,
            max_nested_macros: 0,
        },
    )
    .unwrap();
    let boundary = EvidenceBoundary::new(
        digest("archive-manifest"),
        [digest("held-judge")].into_iter().collect(),
        ["sealed-family".to_owned()].into_iter().collect(),
    )
    .unwrap();
    let validated = contrast
        .validate(&candidate, &migration, &boundary)
        .unwrap();
    let sim = ProbeSimilarity::compare(&[-1.0, 0.0, 1.0], &[-1.0, 0.0, 1.0], 0.0).unwrap();
    assert!(sim.within_tolerance);
    let admission = MotifAdmission::qualify_after_verified_source(
        &candidate,
        vec![witness],
        &val,
        validated,
        migration,
    )
    .unwrap();
    assert_eq!(admission.schema, ADMISSION_SCHEMA);
}

#[test]
fn i47_measurement_harness_verified() {
    // Verified in jope-pipeline's legacy_prototype_seam_regression target; check engine binding here.
    let engine = table::EngineBinding::current();
    assert!(!engine.registry_sha256.is_empty());
}

#[test]
fn all_prototype_seams_wired_and_measured() {
    typed_graph_effects_seam_verified();
    combinators_motifs_seam_verified();
    i47_measurement_harness_verified();
}
