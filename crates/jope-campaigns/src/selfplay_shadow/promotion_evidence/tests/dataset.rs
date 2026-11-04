use super::*;

fn provenance(license: Option<&str>) -> DatasetProvenanceV2 {
    DatasetProvenanceV2 {
        schema: DATASET_PROVENANCE_V2_SCHEMA.to_owned(),
        provider: "OpenML".to_owned(),
        provider_version: "1".to_owned(),
        upstream_root: "openml:1".to_owned(),
        license_spdx: license.map(str::to_owned),
        manual_license_approved: false,
        raw_sha256: BTreeMap::from([("data".to_owned(), digest("raw"))]),
        origin: DatasetOriginV2::Real,
        generator_superfamily: None,
        exact_family_sha256: digest("family"),
        near_duplicate_cluster_sha256: digest("cluster"),
    }
}

fn binary_profile() -> DatasetProfileV1 {
    DatasetProfileV1 {
        schema: DATASET_PROFILE_V1_SCHEMA.to_owned(),
        rows: 1_000,
        encoded_features: 20,
        missing_fraction: 0.0,
        class_count: Some(2),
        minority_majority_ratio: Some(0.2),
        numeric_target_distinct_finite: None,
        finite_target_fraction: None,
        size_slice: DatasetSizeSliceV1::Small,
        missingness_slice: MissingnessSliceV1::None,
        target_shape_slice: TargetShapeSliceV1::Balanced,
        embedding_novelty: 1.0,
        sensitive_or_pii_like: false,
        ambiguous_target: false,
        sparse_text: false,
        grouped_or_time_series: false,
        duplicate_or_leakage_prone: false,
    }
}

#[test]
fn license_and_task_eligibility_fail_closed() {
    assert!(
        binary_profile()
            .eligibility(Task::Binary, &provenance(Some("CC-BY-4.0")))
            .unwrap()
            .eligible
    );
    assert!(
        !binary_profile()
            .eligibility(Task::Binary, &provenance(None))
            .unwrap()
            .eligible
    );
}

#[test]
fn public_discovery_is_versioned_and_cross_provider_duplicates_cluster() {
    let openml = deterministic_openml_discovery(vec![
        OpenMlDatasetMetadataV1 {
            dataset_id: 2,
            version: 1,
            name: "later".to_owned(),
            license: Some("CC-BY-4.0".to_owned()),
            default_target_attribute: Some("target".to_owned()),
            original_data_url: None,
            download_url: "https://example.test/2".to_owned(),
            rows: 1_000,
            features: 10,
            missing_values: 0,
        },
        OpenMlDatasetMetadataV1 {
            dataset_id: 1,
            version: 1,
            name: "first".to_owned(),
            license: Some("CC0-1.0".to_owned()),
            default_target_attribute: Some("target".to_owned()),
            original_data_url: None,
            download_url: "https://example.test/1".to_owned(),
            rows: 1_000,
            features: 10,
            missing_values: 0,
        },
    ]);
    assert_eq!(openml[0].dataset_id, 1);
    let raw = BTreeMap::from([("data".to_owned(), digest("same bytes"))]);
    assert_eq!(
        cross_provider_duplicate_cluster(&raw).unwrap(),
        cross_provider_duplicate_cluster(&raw).unwrap()
    );
}

#[test]
fn synthetic_family_identity_has_only_stable_components() {
    let family = SyntheticFamilyKeyV2 {
        generator_name: "make_classification".to_owned(),
        generator_version: "2".to_owned(),
        scenario: "covariate_shift".to_owned(),
        target_family: "binary".to_owned(),
        feature_family_set: BTreeSet::from(["numeric".to_owned()]),
        shift_pathology_class: "missing_at_random".to_owned(),
    };
    assert_eq!(
        family.exact_family_sha256().unwrap(),
        family.clone().exact_family_sha256().unwrap()
    );
}
