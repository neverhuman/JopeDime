fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn full_dataset_digest(id: &str) -> String {
    format!("{id}{}", &digest(id)[16..])
}

fn archive(offset: usize) -> Vec<String> {
    let mut values = (0..20)
        .map(|index| digest(&format!("archive-{}", index + offset)))
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn checkpoint(
    evaluations: u32,
    prefix_prediction_roots: BTreeMap<u32, String>,
    cv7_rmse: Vec<f64>,
    archive: Vec<String>,
) -> BudgetCheckpointV1 {
    BudgetCheckpointV1 {
        evaluations,
        prefix_prediction_roots,
        population_sha256: digest(&format!("population-{evaluations}")),
        selection_sha256: digest(&format!("selection-{evaluations}")),
        receipt_root_sha256: digest(&format!("receipts-{evaluations}")),
        prediction_root_sha256: digest(&format!("predictions-{evaluations}")),
        cv7_rmse,
        archive_top20_semantic_sha256s: archive,
        logical_operations: evaluations as u64,
        cache_hits: 0,
        cpu_seconds: evaluations as f64,
        gpu_seconds: 0.0,
    }
}

fn ladder(continue_to_10k: bool) -> BudgetLadderBundleV1 {
    let roots = BTreeMap::from([(1_000, digest("predictions-1000"))]);
    let one = checkpoint(1_000, roots, vec![1.1; 7], archive(0));
    let roots = BTreeMap::from([
        (1_000, digest("predictions-1000")),
        (2_000, digest("predictions-2000")),
    ]);
    let two = checkpoint(2_000, roots, vec![1.0; 7], archive(0));
    let five_values = if continue_to_10k {
        vec![0.998, 0.998, 0.998, 0.998, 1.0, 1.0, 1.0]
    } else {
        vec![1.01; 7]
    };
    let roots = BTreeMap::from([
        (1_000, digest("predictions-1000")),
        (2_000, digest("predictions-2000")),
        (5_000, digest("predictions-5000")),
    ]);
    let five = checkpoint(5_000, roots, five_values, archive(0));
    let mut checkpoints = vec![one, two.clone(), five.clone()];
    if continue_to_10k {
        let roots = BTreeMap::from([
            (1_000, digest("predictions-1000")),
            (2_000, digest("predictions-2000")),
            (5_000, digest("predictions-5000")),
            (10_000, digest("predictions-10000")),
        ]);
        checkpoints.push(checkpoint(10_000, roots, vec![0.99; 7], archive(0)));
    }
    BudgetLadderBundleV1 {
        schema: BudgetLadderBundleV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("campaign"),
        dataset_sha256: full_dataset_digest(STAGE1_DATASET_IDS[0]),
        master_seed: 101,
        license_track: LicenseTrackV1::ProductionCompatible,
        metric_sha256: digest("metric"),
        evaluator_sha256: digest("evaluator"),
        split_sha256: digest("split"),
        test_csv_read: false,
        checkpoints,
        promotion: BudgetPromotionDecisionV1 {
            source_2k_checkpoint_sha256: two.identity().unwrap(),
            source_5k_checkpoint_sha256: five.identity().unwrap(),
            continue_to_10k,
            improved_folds_at_least_point_one_percent: if continue_to_10k { 4 } else { 0 },
            archive_jaccard_micros: 1_000_000,
            cv7_mean_ratio_micros: if continue_to_10k { 998_857 } else { 1_010_000 },
        },
    }
}

fn license() -> LicenseIdentityV1 {
    LicenseIdentityV1 {
        spdx_expression: "research-only".into(),
        license_text_sha256: digest("license"),
        upstream_source: "https://example.invalid/upstream".into(),
        upstream_revision: "commit-1".into(),
    }
}

fn tabfm() -> FrozenPredictorComponentV1 {
    FrozenPredictorComponentV1::ExternalModel {
        family: ExternalModelFamilyV1::TabFm,
        artifact_sha256: digest("tabfm-model"),
        artifact_bytes: 42,
        inference_protocol_sha256: digest("tabfm-inference"),
        license: license(),
        complexity: 100,
    }
}
