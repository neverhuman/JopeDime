use super::*;

#[test]
fn real_regression_benchmark_has_no_300_row_cut() {
    let mut datasets = Vec::new();
    for index in 0..42 {
        datasets.push(CatalogDatasetV1 {
            dataset_sha256: digest(&format!("sotarena-{index}")),
            task: GpTaskV1::Regression,
            partition: CatalogPartitionV1::Development,
            origin: DatasetOriginV1::Real,
            source_family: format!("family-{}", index % 5),
            row_count: if index < 20 { 20 + index } else { 500 + index },
            feature_count: 2 + index,
            missing_fraction: index as f64 / 100.0,
            target_distribution: index as f64 / 10.0,
            sotarena_dataset: true,
            current_strict_win: index < 6,
        });
    }
    datasets.push(CatalogDatasetV1 {
        dataset_sha256: digest("synthetic"),
        task: GpTaskV1::Regression,
        partition: CatalogPartitionV1::Development,
        origin: DatasetOriginV1::Synthetic,
        source_family: "generator".into(),
        row_count: 10_000,
        feature_count: 10,
        missing_fraction: 0.0,
        target_distribution: 0.0,
        sotarena_dataset: false,
        current_strict_win: false,
    });
    let freeze = freeze_catalog(digest("observations"), datasets).unwrap();
    assert_eq!(freeze.datasets.len(), 42);
    assert_eq!(
        freeze
            .datasets
            .iter()
            .filter(|row| row.row_count < 300)
            .count(),
        20
    );
    assert_eq!(
        freeze
            .datasets
            .iter()
            .filter(|row| row.current_strict_win)
            .count(),
        6
    );
    assert_eq!(freeze.counts.development_regression, 42);
}

#[test]
fn pilot_and_tranche_freezes_are_deterministic_and_phase_ordered() {
    let mut datasets = Vec::new();
    for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
        for index in 0..12 {
            datasets.push(CatalogDatasetV1 {
                dataset_sha256: digest(&format!("coverage-{task:?}-{index}")),
                task,
                partition: CatalogPartitionV1::Development,
                origin: DatasetOriginV1::Real,
                source_family: format!("family-{}", index % 3),
                row_count: 10 + index,
                feature_count: 1 + index,
                missing_fraction: index as f64 / 20.0,
                target_distribution: index as f64 / 11.0,
                sotarena_dataset: index < 2,
                current_strict_win: false,
            });
        }
        datasets.push(CatalogDatasetV1 {
            dataset_sha256: digest(&format!("validation-{task:?}")),
            task,
            partition: CatalogPartitionV1::Validation,
            origin: DatasetOriginV1::Real,
            source_family: "validation-family".into(),
            row_count: 8,
            feature_count: 2,
            missing_fraction: 0.0,
            target_distribution: 0.5,
            sotarena_dataset: false,
            current_strict_win: false,
        });
    }
    let catalog = freeze_catalog(digest("pilot observations"), datasets).unwrap();
    let first = select_max_min_pilot(&catalog, 2).unwrap();
    let second = select_max_min_pilot(&catalog, 2).unwrap();
    assert_eq!(first, second);
    assert!(first.replacement_allowed());
    assert_eq!(first.selected[&GpTaskV1::Regression].len(), 10);
    assert_eq!(first.alternates[&GpTaskV1::Binary].len(), 2);

    let validation_without_development = build_tranche_plan(
        &catalog,
        &digest("tranche campaign"),
        CampaignArmV1::RotatingMean,
        CatalogPartitionV1::Validation,
        &digest("pilot comparison"),
        None,
        Vec::new(),
    );
    assert!(validation_without_development
        .unwrap_err()
        .contains("before development completion"));
    let validation = build_tranche_plan(
        &catalog,
        &digest("tranche campaign"),
        CampaignArmV1::RotatingMean,
        CatalogPartitionV1::Validation,
        &digest("pilot comparison"),
        Some(digest("development complete")),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(validation.tranches.len(), 1);
    assert_eq!(validation.tranches[0].dataset_sha256s.len(), 2);
}
