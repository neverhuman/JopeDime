use super::*;

#[test]
fn target_decile_splits_are_exact_distinct_and_deterministic() {
    let targets = (0..103)
        .map(|index| (index as f64).sin() + index as f64 / 10.0)
        .collect::<Vec<_>>();
    let first = build_splits("fixture", "dataset", "train", &targets, GENERATIONS).unwrap();
    let second = build_splits("fixture", "dataset", "train", &targets, GENERATIONS).unwrap();
    assert_eq!(
        receipt_digest(&first).unwrap(),
        receipt_digest(&second).unwrap()
    );
    assert_eq!(first.splits.len(), GENERATIONS);
    let mut hashes = HashSet::new();
    let mut seeds = HashSet::new();
    for split in first.splits {
        assert_eq!(split.validation_indices.len(), targets.len().div_ceil(5));
        assert_eq!(
            split.train_indices.len() + split.validation_indices.len(),
            targets.len()
        );
        let train = split.train_indices.into_iter().collect::<HashSet<_>>();
        assert!(split
            .validation_indices
            .iter()
            .all(|index| !train.contains(index)));
        assert!(hashes.insert(split.split_sha256));
        assert!(seeds.insert(split.seed));
    }
}
#[test]
fn final_top10_cv_folds_are_deterministic_disjoint_and_exhaustive() {
    let targets = (0..103)
        .map(|index| (index as f64).cos() + index as f64 / 7.0)
        .collect::<Vec<_>>();
    let first = build_final_cv_splits("fixture", "dataset", "train", &targets).unwrap();
    let second = build_final_cv_splits("fixture", "dataset", "train", &targets).unwrap();
    assert_eq!(
        receipt_digest(&first).unwrap(),
        receipt_digest(&second).unwrap()
    );
    assert_eq!(first.splits.len(), FINAL_CV_FOLDS);
    let mut validation_rows = HashSet::new();
    let mut hashes = HashSet::new();
    let mut seeds = HashSet::new();
    for split in first.splits {
        assert_eq!(
            split.train_indices.len() + split.validation_indices.len(),
            targets.len()
        );
        let train = split.train_indices.into_iter().collect::<HashSet<_>>();
        for row in split.validation_indices {
            assert!(!train.contains(&row));
            assert!(validation_rows.insert(row));
        }
        assert!(hashes.insert(split.split_sha256));
        assert!(seeds.insert(split.seed));
    }
    assert_eq!(validation_rows.len(), targets.len());
}
#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn production_sotarena_registry_freezes_exact_37_to_requested_10() {
    let qualified_root = std::env::var_os("JOPE_TEST_SOTARENA_ROOT").map(PathBuf::from);
    let sotarena_root = qualified_root
        .as_deref()
        .unwrap_or_else(|| Path::new(default_sotarena_root_str()));
    assert!(
            Path::new(default_catalog_pointer_str()).is_file(),
            "unavailable: production context catalog pointer {} (provision <JOPE_STATE>/legacy/test_exact_gen0)",
            default_catalog_pointer_str()
        );
    let (_, catalog) =
        load_catalog(Path::new(default_catalog_pointer_str())).expect("production context catalog");
    let disclosure = CorpusExclusions {
        schema: "fixture".to_string(),
        run_id: "fixture".to_string(),
        checkpoint_sha256: "a".repeat(64),
        training_corpus_train_sha256: "b".repeat(64),
        training_corpus_validation_sha256: "c".repeat(64),
        dataset_ids: Vec::new(),
        source_families: Vec::new(),
        target_identities: Vec::new(),
    };
    let (eligibility, selected) = build_eligibility(
        "fixture",
        sotarena_root,
        &catalog,
        &disclosure,
        GENERATIONS,
        FROZEN_DATASETS,
        Some(EXPECTED_ELIGIBLE),
    )
    .expect("production Golden42 eligibility");
    assert_eq!(
        eligibility.iter().filter(|row| row.eligible).count(),
        EXPECTED_ELIGIBLE
    );
    assert_eq!(
        selected
            .iter()
            .map(|(entry, _, _, _)| entry.dataset_hash.as_str())
            .collect::<Vec<_>>(),
        FROZEN_DATASETS
    );
    assert!(selected.iter().all(|(_, _, _, reports)| {
        reports.tabpfn_v3_path.is_file()
            && reports.tabfm_path.is_file()
            && reports.tabfm_ensemble_path.is_file()
    }));
}
#[test]
fn paired_fitness_clips_and_uses_population_sd() {
    let scores = [-2.0, 0.0, 1.0];
    let clipped = [-1.0_f64, 0.0, 1.0];
    let mean = 0.0;
    let sd = (clipped
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    assert_eq!(paired_robust_fitness(&scores).unwrap(), mean - 0.25 * sd);
    assert!(paired_robust_fitness(&[]).is_err());
}
#[test]
fn ranking_ties_use_worst_complexity_then_semantic_hash() {
    let a = scored(member(0, "(pipe X (ridge l2=0.1))", 4), 0.5, 0.1);
    let b = scored(member(1, "(pipe X (ridge l2=0.2))", 3), 0.5, 0.2);
    let c = scored(member(2, "(pipe X (ridge l2=0.3))", 2), 0.5, 0.2);
    let mut values = [a, b, c.clone()];
    values.sort_by(rank_scored);
    assert_eq!(values[0].member.semantic_sha256, c.member.semantic_sha256);

    let mut same = [
        scored(member(0, "(pipe X (ridge l2=0.4))", 2), 0.5, 0.2),
        scored(member(1, "(pipe X (ridge l2=0.5))", 2), 0.5, 0.2),
    ];
    same.sort_by(rank_scored);
    assert!(same[0].member.semantic_sha256 < same[1].member.semantic_sha256);
}
#[test]
fn cache_identity_includes_split_and_kernel() {
    let first = cache_key("train", "semantic", "split-a").unwrap();
    let second = cache_key("train", "semantic", "split-b").unwrap();
    assert_ne!(first, second);
    assert_eq!(first, cache_key("train", "semantic", "split-a").unwrap());
}
#[test]
fn split_and_stage_caches_are_metric_bit_identical_to_plain_fit() {
    let data = RegressionData {
        x: (0..100)
            .map(|index| vec![index as f64 / 10.0, (index as f64).sin()])
            .collect(),
        y: (0..100)
            .map(|index| 0.5 * index as f64 + (index as f64).cos())
            .collect(),
        features: 2,
    };
    let splits = build_splits("fixture", "dataset", "train", &data.y, GENERATIONS).unwrap();
    let dataset = DatasetStudyEntry {
        dataset_hash: "dataset".to_string(),
        source_family: "fixture".to_string(),
        target_identity: "fixture".to_string(),
        train_path: PathBuf::new(),
        train_sha256: "a".repeat(64),
        rows: 100,
        features: 2,
        encoding: EncodingSlice {
            blob_path: PathBuf::new(),
            blob_sha256: "b".repeat(64),
            offset_bytes: 0,
            length_bytes: CONTEXT_BYTES,
            context_sha256: "c".repeat(64),
        },
        split_manifest_path: PathBuf::new(),
        split_manifest_sha256: "d".repeat(64),
        generation_zero_path: PathBuf::new(),
        generation_zero_sha256: "e".repeat(64),
        transfer_ranking_path: PathBuf::new(),
        transfer_ranking_sha256: "f".repeat(64),
        baseline_reports: SotarenaReportHashes {
            tabpfn_v3_path: PathBuf::new(),
            tabpfn_v3_sha256: "1".repeat(64),
            tabfm_path: PathBuf::new(),
            tabfm_sha256: "2".repeat(64),
            tabfm_ensemble_path: PathBuf::new(),
            tabfm_ensemble_sha256: "3".repeat(64),
        },
    };
    let evaluator = StudyEvalContext::new(&data, &dataset, &splits).unwrap();
    evaluator.begin_generation(0);
    let genome = GenomeIr::parse("(pipe X (standardize append=true) (ridge l2=0.25))").unwrap();
    let cached_first = evaluator.evaluate(&genome, &splits.splits[0]).unwrap();
    let cached_second = evaluator.evaluate(&genome, &splits.splits[0]).unwrap();
    let (train_x, train_y) = slice_data(&data, &splits.splits[0].train_indices).unwrap();
    let (validation_x, validation_y) =
        slice_data(&data, &splits.splits[0].validation_indices).unwrap();
    let predictions = fit(&genome, &train_x, &train_y)
        .unwrap()
        .predict(&validation_x)
        .unwrap();
    let plain = metrics(&validation_y, &predictions).unwrap();
    assert_eq!((cached_first.0, cached_first.1, cached_first.2), plain);
    assert_eq!((cached_second.0, cached_second.1, cached_second.2), plain);
    assert!(evaluator.stage_cache.stats().0 > 0);

    let sources = [
        "(pipe X (standardize append=true) (ridge l2=0.25))",
        "(pipe X (standardize append=true) (ridge l2=0.75))",
        "(pipe X (ridge l2=0.5))",
    ];
    let mut cached_rank = Vec::new();
    let mut uncached_rank = Vec::new();
    for (slot, source) in sources.iter().enumerate() {
        let genome = GenomeIr::parse(source).unwrap();
        let cached_r2 = splits.splits[..2]
            .iter()
            .map(|split| evaluator.evaluate(&genome, split).unwrap().1)
            .collect::<Vec<_>>();
        let uncached_r2 = splits.splits[..2]
            .iter()
            .map(|split| {
                let (train_x, train_y) = slice_data(&data, &split.train_indices).unwrap();
                let (validation_x, validation_y) =
                    slice_data(&data, &split.validation_indices).unwrap();
                let predictions = fit(&genome, &train_x, &train_y)
                    .unwrap()
                    .predict(&validation_x)
                    .unwrap();
                metrics(&validation_y, &predictions).unwrap().1
            })
            .collect::<Vec<_>>();
        assert_eq!(cached_r2, uncached_r2);
        let make_scored = |scores: Vec<f64>| ScoredMember {
            member: member(slot, source, genome_complexity(&genome)),
            observations: Vec::new(),
            robust_fitness: Some(paired_robust_fitness(&scores).unwrap()),
            worst_split_r2: scores.iter().copied().min_by(f64::total_cmp),
            split_r2: scores,
            failure_reason: None,
        };
        cached_rank.push(make_scored(cached_r2));
        uncached_rank.push(make_scored(uncached_r2));
    }
    cached_rank.sort_by(rank_scored);
    uncached_rank.sort_by(rank_scored);
    assert_eq!(
        cached_rank
            .iter()
            .map(|row| &row.member.semantic_sha256)
            .collect::<Vec<_>>(),
        uncached_rank
            .iter()
            .map(|row| &row.member.semantic_sha256)
            .collect::<Vec<_>>()
    );
}
