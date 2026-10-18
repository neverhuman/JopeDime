// Compile the worker-owned implementation directly until the serialized lib.rs export lands.
pub use jope_motifs::{expand, fragment, source, table, validate, NAME_PREFIX};
#[path = "../src/promote.rs"]
#[allow(dead_code)]
mod promote;

use feat_math::GenomeIr;
use jope_motifs::validate::{CheckResult, MotifResult, Validation, VALIDATION_SCHEMA};
use promote::{
    AdmissionError, ContrastObservation, EvidenceBoundary, EvidenceRole, ExpansionLimits,
    ExpansionWitness, MetricDirection, MotifAdmission, MotifCandidate, PreregisteredContrast,
    ProbeSimilarity, VocabularyMigration, ADMISSION_SCHEMA, CONTRAST_SCHEMA,
};
use sha2::Digest;

const TABLE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

fn candidate(expansion: &str) -> MotifCandidate {
    let id = table::motif_id(expansion);
    MotifCandidate::draft_for_gate(
        TABLE_SHA.to_owned(),
        id.clone(),
        table::motif_name(&id),
        expansion.to_owned(),
        0,
    )
    .unwrap()
}

fn migration(candidate: &MotifCandidate) -> VocabularyMigration {
    VocabularyMigration::new(candidate, "kernel2-baseline", 3, "kernel2-motifs-r1", 1).unwrap()
}

fn observations(macro_scores: &[f64], families: &[&str]) -> Vec<ContrastObservation> {
    macro_scores
        .iter()
        .enumerate()
        .map(|(index, score)| ContrastObservation {
            pair_id: format!("pair-{index}"),
            dataset_sha256: digest(&format!("dataset-{index}")),
            cut_sha256: digest(&format!("cut-{index}")),
            context_sha256: digest(&format!("context-{index}")),
            family_id: families[index].to_owned(),
            seed: index as u64,
            role: EvidenceRole::ResearchDevelopment,
            baseline_score: 1.0,
            macro_score: *score,
        })
        .collect()
}

fn contrast(
    candidate: &MotifCandidate,
    macro_scores: &[f64],
    families: &[&str],
) -> PreregisteredContrast {
    let migration = migration(candidate);
    let (old_profile_id, old_profile_revision) = migration.old_profile();
    let (new_profile_id, new_profile_revision) = migration.new_profile();
    let mut result = PreregisteredContrast {
        schema: CONTRAST_SCHEMA.to_owned(),
        preregistration_sha256: String::new(),
        motif_id: candidate.motif_id().to_owned(),
        old_profile_id: old_profile_id.to_owned(),
        old_profile_revision,
        new_profile_id: new_profile_id.to_owned(),
        new_profile_revision,
        direction: MetricDirection::LowerIsBetter,
        minimum_mean_gain: 0.1,
        observations: observations(macro_scores, families),
    };
    result.preregistration_sha256 = result.computed_preregistration_sha256().unwrap();
    result
}

fn validation(candidate: &MotifCandidate) -> Validation {
    let passed = CheckResult {
        passed: true,
        exercised: 1,
        failures: Vec::new(),
    };
    Validation {
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
    }
}

fn host_for(expansion: &str) -> String {
    let input = fragment::input_stage().unwrap();
    GenomeIr::parse(&format!("(pipe X {input} {expansion} (ridge))"))
        .unwrap()
        .canonical()
}

fn digest(value: &str) -> String {
    format!("{:x}", sha2::Sha256::digest(value.as_bytes()))
}

fn safe_boundary() -> EvidenceBoundary {
    EvidenceBoundary::new(
        digest("archive-manifest"),
        [digest("held-judge")].into_iter().collect(),
        ["sealed-family".to_owned()].into_iter().collect(),
    )
    .unwrap()
}

#[test]
fn exact_expansion_witness_round_trips_the_host_and_enforces_nested_limits() {
    let expansion = "(standardize) (sigmoid)";
    let candidate = candidate(expansion);
    let host = host_for(expansion);
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
    assert_eq!(witness.expanded_stages(), 2);
    assert_eq!(witness.nested_macros(), 1);

    assert_eq!(
        ExpansionWitness::verify_host(
            &candidate,
            &host,
            1,
            ExpansionLimits {
                max_expansion_stages: 1,
                max_nested_macros: 0,
            }
        )
        .unwrap_err(),
        AdmissionError("expansion_stage_limit")
    );
    let nested = format!("(standardize) ({}child)", NAME_PREFIX);
    assert_eq!(
        MotifCandidate::draft_for_gate(
            TABLE_SHA.to_owned(),
            table::motif_id(&nested),
            table::motif_name(&table::motif_id(&nested)),
            nested,
            0
        )
        .unwrap_err(),
        AdmissionError("candidate_identity_invalid")
    );
    assert_eq!(
        ExpansionWitness::verify_host(
            &candidate,
            &host.replace("(sigmoid)", "(standardize)"),
            1,
            ExpansionLimits {
                max_expansion_stages: 6,
                max_nested_macros: 0,
            }
        )
        .unwrap_err(),
        AdmissionError("expansion_occurrence_missing")
    );
}

#[test]
fn finite_probe_similarity_does_not_authorize_a_macro_with_no_useful_contrast() {
    // x and x^3 agree at -1, 0, and 1, but differ at 0.5.
    let probes: [f64; 3] = [-1.0, 0.0, 1.0];
    let left = probes;
    let right = probes.map(|x| x.powi(3));
    let similarity = ProbeSimilarity::compare(&left, &right, 0.0).unwrap();
    assert!(similarity.within_tolerance);
    assert_ne!(0.5_f64, 0.5_f64.powi(3));

    let candidate = candidate("(standardize) (sigmoid)");
    let migration = migration(&candidate);
    let contrast = contrast(
        &candidate,
        &[1.1, 1.2, 1.1, 1.2],
        &["family-a", "family-a", "family-b", "family-b"],
    );
    assert_eq!(
        contrast
            .validate(&candidate, &migration, &safe_boundary())
            .unwrap_err(),
        AdmissionError("contrast_not_useful")
    );
}

#[test]
fn synthetic_gate_accepts_positive_preregistered_cross_family_contrast() {
    let candidate = candidate("(standardize) (sigmoid)");
    let migration = migration(&candidate);
    let contrast = contrast(
        &candidate,
        &[0.8, 0.85, 0.9, 0.88],
        &["family-a", "family-a", "family-b", "family-b"],
    );
    let validated = contrast
        .validate(&candidate, &migration, &safe_boundary())
        .unwrap();
    let host = host_for(candidate.expansion());
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
    // This exercises the final pure predicate with a synthetic validation record. The public
    // production entry point instead computes validation and verifies a real QualifiedSet.
    let admission = MotifAdmission::qualify_after_verified_source(
        &candidate,
        vec![witness],
        &validation(&candidate),
        validated,
        migration,
    )
    .unwrap();
    assert_eq!(admission.schema, ADMISSION_SCHEMA);
    assert_eq!(admission.motif_id, candidate.motif_id());
    assert!(admission.contrast.mean_gain() > admission.contrast.minimum_mean_gain());
    assert!(admission.vocabulary.old_policy_expands_macro());
    assert!(!admission.vocabulary.enables_macro("kernel2-baseline", 3));
    assert!(admission.vocabulary.enables_macro("kernel2-motifs-r1", 1));
}

#[test]
fn judge_and_protected_family_rows_are_refused_before_contrast_scoring() {
    let candidate = candidate("(standardize) (sigmoid)");
    let migration = migration(&candidate);
    let mut observations = observations(&[0.8, 0.85], &["family-a", "family-b"]);
    let judge_dataset = observations[0].dataset_sha256.clone();
    let contrast = contrast(&candidate, &[0.8, 0.85], &["family-a", "family-b"]);
    let boundary = EvidenceBoundary::new(
        digest("archive-manifest"),
        [judge_dataset].into_iter().collect(),
        ["sealed-family".to_owned()].into_iter().collect(),
    )
    .unwrap();
    assert_eq!(
        contrast
            .validate(&candidate, &migration, &boundary)
            .unwrap_err(),
        AdmissionError("contrast_input_refused")
    );

    let boundary = EvidenceBoundary::new(
        digest("archive-manifest"),
        [digest("other-judge")].into_iter().collect(),
        ["family-b".to_owned()].into_iter().collect(),
    )
    .unwrap();
    assert_eq!(
        contrast
            .validate(&candidate, &migration, &boundary)
            .unwrap_err(),
        AdmissionError("contrast_input_refused")
    );

    observations[1].role = EvidenceRole::Judge;
    let mut forbidden = contrast;
    forbidden.observations = observations;
    forbidden.preregistration_sha256 = forbidden.computed_preregistration_sha256().unwrap();
    assert_eq!(
        forbidden
            .validate(&candidate, &migration, &safe_boundary())
            .unwrap_err(),
        AdmissionError("contrast_input_refused")
    );
    forbidden.observations[1].role = EvidenceRole::ProtectedFamily;
    forbidden.preregistration_sha256 = forbidden.computed_preregistration_sha256().unwrap();
    assert_eq!(
        forbidden
            .validate(&candidate, &migration, &safe_boundary())
            .unwrap_err(),
        AdmissionError("contrast_input_refused")
    );
}

#[test]
fn source_validation_and_profile_migration_are_mandatory() {
    let candidate = candidate("(standardize) (sigmoid)");
    let migration = migration(&candidate);
    let contrast = contrast(
        &candidate,
        &[0.8, 0.85, 0.9, 0.88],
        &["family-a", "family-a", "family-b", "family-b"],
    )
    .validate(&candidate, &migration, &safe_boundary())
    .unwrap();
    let witness = ExpansionWitness::verify_host(
        &candidate,
        &host_for(candidate.expansion()),
        1,
        ExpansionLimits {
            max_expansion_stages: 6,
            max_nested_macros: 0,
        },
    )
    .unwrap();
    let mut unverified = validation(&candidate);
    unverified.source_verified = false;
    assert_eq!(
        MotifAdmission::qualify_after_verified_source(
            &candidate,
            vec![witness.clone()],
            &unverified,
            contrast.clone(),
            migration.clone()
        )
        .unwrap_err(),
        AdmissionError("admission_evidence_incomplete")
    );
    assert_eq!(
        VocabularyMigration::new(&candidate, "same", 2, "same", 2).unwrap_err(),
        AdmissionError("vocabulary_version_invalid")
    );
}

#[test]
fn profile_contrast_requires_distinct_datasets_and_families_and_finite_scores() {
    let candidate = candidate("(standardize) (sigmoid)");
    let migration = migration(&candidate);
    let one_family = contrast(&candidate, &[0.7, 0.8], &["family-a", "family-a"]);
    assert_eq!(
        one_family
            .validate(&candidate, &migration, &safe_boundary())
            .unwrap_err(),
        AdmissionError("contrast_not_useful")
    );
    let mut invalid = contrast(
        &candidate,
        &[0.7, 0.8, 0.8, 0.9],
        &["family-a", "family-a", "family-b", "family-b"],
    );
    invalid.observations[0].macro_score = f64::NAN;
    assert_eq!(
        invalid
            .validate(&candidate, &migration, &safe_boundary())
            .unwrap_err(),
        AdmissionError("contrast_input_refused")
    );
    assert!(contrast(
        &candidate,
        &[0.7, 0.8, 0.8, 0.9],
        &["family-a", "family-a", "family-b", "family-b"]
    )
    .validate(&candidate, &migration, &safe_boundary())
    .is_ok());
}
