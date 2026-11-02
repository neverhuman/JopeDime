use super::*;
use tempfile::tempdir;

fn canonical(source: &str) -> String {
    GenomeIr::parse(source).unwrap().canonical()
}

fn member(slot: usize, source: &str, complexity: usize) -> GenomeMember {
    let canonical_genome = canonical(source);
    GenomeMember {
        slot,
        semantic_sha256: semantic_identity(&canonical_genome),
        canonical_genome,
        complexity,
        source: "fixture".to_string(),
        source_evidence: Value::Null,
        ancestry: Ancestry {
            operation: "fixture".to_string(),
            parents: Vec::new(),
            roots: BTreeSet::from(["dl".to_string()]),
            retry_count: 0,
        },
    }
}

fn scored(member: GenomeMember, fitness: f64, worst: f64) -> ScoredMember {
    ScoredMember {
        member,
        observations: Vec::new(),
        split_r2: vec![fitness],
        robust_fitness: Some(fitness),
        worst_split_r2: Some(worst),
        failure_reason: None,
    }
}

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

#[test]
fn sealed_receipts_detect_all_required_tamper_classes() {
    let directory = tempdir().unwrap();
    for artifact in [
        "checkpoint",
        "proposal",
        "transfer-seed",
        "split",
        "observation",
        "generation-checkpoint",
        "baseline-report",
    ] {
        let path = directory.path().join(format!("{artifact}.json"));
        write_receipt(
            &path,
            &json!({"schema": "Fixture.v2", "artifact": artifact, "value": 1}),
        )
        .unwrap();
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["value"] = json!(2);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            read_receipt::<Value>(&path).is_err(),
            "{artifact} tampering was not rejected"
        );
    }
}

#[test]
fn checkpoint_overlap_is_disclosed_without_zero_shot_rejection() {
    let directory = tempdir().unwrap();
    let registry = directory.path().join("registry");
    let benchmark_id = GOLDEN42[0];
    let benchmark_root = registry.join(benchmark_id);
    fs::create_dir_all(&benchmark_root).unwrap();
    let meta = json!({
        "source_family": "same-family",
        "source_platform": "fixture",
        "source_dataset_id": "same-source",
        "source_dataset_name": "same-name",
        "source_target_column_original": "y",
        "target_column": "target",
        "target_names": ["target"],
    });
    fs::write(
        benchmark_root.join("meta.json"),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();
    let train = (0..100)
        .map(|index| format!("{},{}\n", index as f64, index as f64 / 10.0))
        .collect::<String>();
    fs::write(benchmark_root.join("train.csv"), train).unwrap();
    let train_hash = sha256_file(&benchmark_root.join("train.csv")).unwrap();
    let context_hash = "a".repeat(64);
    let catalog = Catalog {
        dataset_dimensions: 1536,
        target_dimensions: 512,
        record_stride_bytes: CONTEXT_BYTES,
        entries: vec![CatalogEntry {
            dataset_hash: benchmark_id.to_string(),
            task: "regression".to_string(),
            train_csv: benchmark_root.join("train.csv"),
            train_sha256: train_hash,
            source_family: "same-family".to_string(),
            context_sha256: context_hash,
            catalog_index: 0,
        }],
    };
    let other_id = "checkpoint-source";
    let other_root = registry.join(other_id);
    fs::create_dir_all(&other_root).unwrap();
    fs::write(
        other_root.join("meta.json"),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();
    let corpus_train = directory.path().join("train.jsonl");
    let corpus_validation = directory.path().join("validation.jsonl");
    fs::write(
        &corpus_train,
        format!("{{\"dataset_id\":\"{other_id}\"}}\n{{\"dataset_id\":\"{benchmark_id}\"}}\n"),
    )
    .unwrap();
    fs::write(&corpus_validation, "").unwrap();
    let checkpoint = CheckpointProvenance {
        accepted_pointer_path: PathBuf::new(),
        accepted_pointer_sha256: String::new(),
        checkpoint_path: PathBuf::new(),
        checkpoint_sha256: "b".repeat(64),
        model_version: "fixture".to_string(),
        training_corpus_manifest_path: PathBuf::new(),
        training_corpus_manifest_sha256: String::new(),
        training_corpus_train_path: corpus_train,
        training_corpus_train_sha256: "c".repeat(64),
        training_corpus_validation_path: corpus_validation,
        training_corpus_validation_sha256: "d".repeat(64),
    };
    let exclusions = checkpoint_exclusions("fixture", &registry, &checkpoint, &catalog).unwrap();
    assert!(exclusions
        .source_families
        .contains(&"same-family".to_string()));
    assert!(exclusions
        .target_identities
        .contains(&target_identity(&registry, benchmark_id).unwrap()));
    let (records, selected) = build_eligibility(
        "fixture",
        &registry,
        &catalog,
        &exclusions,
        GENERATIONS,
        &[],
        None,
    )
    .unwrap();
    let record = records
        .iter()
        .find(|record| record.dataset_hash == benchmark_id)
        .unwrap();
    assert!(record.checkpoint_source_family_overlap);
    assert!(record.checkpoint_exact_overlap);
    assert!(record.checkpoint_target_identity_overlap);
    assert!(!record
        .reasons
        .iter()
        .any(|reason| reason.starts_with("checkpoint_corpus_")));
    assert!(selected.is_empty());
}

fn proposal_row(dataset: &str, checkpoint: &str, context: &str, index: usize) -> ProposalRow {
    let canonical_genome = canonical(&format!("(pipe X (ridge l2={}))", index as f64 + 0.125));
    ProposalRow {
        task: "regression".to_string(),
        dataset_hash: dataset.to_string(),
        semantic_sha256: semantic_identity(&canonical_genome),
        canonical_genome,
        model_checkpoint_sha256: checkpoint.to_string(),
        context_sha256: Some(context.to_string()),
        proposal_score: Some(100.0 - index as f64),
        selection_score: None,
        reward_lcb: Some(100.0 - index as f64),
        instability_probability: Some(0.0),
        slot: index,
        extra: BTreeMap::new(),
    }
}

#[test]
fn generation_zero_preserves_quotas_deduplicates_and_fails_closed() {
    let directory = tempdir().unwrap();
    let proposals_path = directory.path().join("proposals.jsonl");
    let dataset_hash = "fixture";
    let checkpoint_hash = "1".repeat(64);
    let context_hash = "2".repeat(64);
    let proposals = (0..64)
        .map(|index| proposal_row(dataset_hash, &checkpoint_hash, &context_hash, index))
        .collect::<Vec<_>>();
    let proposal_bytes = proposals
        .iter()
        .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
        .collect::<String>();
    fs::write(&proposals_path, proposal_bytes).unwrap();
    let proposal_hash = sha256_file(&proposals_path).unwrap();
    let mut global = Vec::new();
    global.push(GlobalRankedGenome {
        rank: 0,
        canonical_genome: proposals[0].canonical_genome.clone(),
        semantic_sha256: proposals[0].semantic_sha256.clone(),
        mean_percentile: 1.0,
        worst_percentile: 1.0,
        strict_win_count: 1,
        dataset_count: 1,
        source_dataset_ids: vec!["source".to_string()],
        source_families: vec!["source-family".to_string()],
    });
    for index in 0..30 {
        let canonical_genome = canonical(&format!(
            "(pipe X (elasticnet alpha={}))",
            index as f64 + 0.75
        ));
        global.push(GlobalRankedGenome {
            rank: index + 1,
            semantic_sha256: semantic_identity(&canonical_genome),
            canonical_genome,
            mean_percentile: 1.0 - index as f64 / 100.0,
            worst_percentile: 0.5,
            strict_win_count: 0,
            dataset_count: 1,
            source_dataset_ids: vec!["source".to_string()],
            source_families: vec!["source-family".to_string()],
        });
    }
    let ranking = GlobalRankingReceipt {
        schema: "TrainOnlyTransferGenomeRanking.v2".to_string(),
        run_id: "fixture".to_string(),
        target_dataset_id: dataset_hash.to_string(),
        target_source_family: "fixture".to_string(),
        observation_db_sha256: "3".repeat(64),
        source_catalog_sha256: "4".repeat(64),
        excluded_dataset_ids: Vec::new(),
        excluded_source_family: "fixture".to_string(),
        direct_dl_sources_excluded: true,
        ranking_policy: "fixture".to_string(),
        genomes: global,
    };
    let entry = CatalogEntry {
        dataset_hash: dataset_hash.to_string(),
        task: "regression".to_string(),
        train_csv: PathBuf::new(),
        train_sha256: "5".repeat(64),
        source_family: "fixture".to_string(),
        context_sha256: context_hash.clone(),
        catalog_index: 0,
    };
    let data = RegressionData {
        x: vec![vec![0.0]; 100],
        y: (0..100).map(|value| value as f64).collect(),
        features: 1,
    };
    let encoding = EncodingSlice {
        blob_path: PathBuf::new(),
        blob_sha256: "6".repeat(64),
        offset_bytes: 0,
        length_bytes: CONTEXT_BYTES,
        context_sha256: context_hash,
    };
    let checkpoint = CheckpointProvenance {
        accepted_pointer_path: PathBuf::new(),
        accepted_pointer_sha256: String::new(),
        checkpoint_path: PathBuf::new(),
        checkpoint_sha256: checkpoint_hash,
        model_version: "fixture".to_string(),
        training_corpus_manifest_path: PathBuf::new(),
        training_corpus_manifest_sha256: String::new(),
        training_corpus_train_path: PathBuf::new(),
        training_corpus_train_sha256: String::new(),
        training_corpus_validation_path: PathBuf::new(),
        training_corpus_validation_sha256: String::new(),
    };
    let generation = generation_zero(
        "fixture",
        &entry,
        &data,
        &encoding,
        &checkpoint,
        &proposals_path,
        &proposal_hash,
        &ranking,
        "7",
    )
    .unwrap();
    assert_eq!(generation.members.len(), POPULATION);
    assert!(generation.members[..DL_QUOTA]
        .iter()
        .all(|member| member.source == "latest_checkpoint_dl"));
    assert!(generation.members[DL_QUOTA..]
        .iter()
        .all(|member| member.source == "target_clean_cross_dataset_transfer"));
    assert_eq!(
        generation
            .members
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<HashSet<_>>()
            .len(),
        POPULATION
    );

    let mut underfilled = ranking;
    underfilled.genomes.truncate(30);
    assert!(generation_zero(
        "fixture",
        &entry,
        &data,
        &encoding,
        &checkpoint,
        &proposals_path,
        &proposal_hash,
        &underfilled,
        "7",
    )
    .is_err());
}

#[test]
fn transfer_ranking_binds_sql_shaped_target_values_and_excludes_direct_dl_sources() {
    let directory = tempdir().unwrap();
    let database = directory.path().join("observations.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE observations(
                    task TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    canonical_genome TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    score REAL NOT NULL,
                    origin_class TEXT NOT NULL,
                    PRIMARY KEY(task,dataset_hash,semantic_sha256)
                ) WITHOUT ROWID;",
        )
        .unwrap();
    let entries = ["target' OR 1=1 --", "same-family-source", "clean-source"]
        .into_iter()
        .enumerate()
        .map(|(index, dataset)| CatalogEntry {
            dataset_hash: dataset.to_string(),
            task: "regression".to_string(),
            train_csv: PathBuf::new(),
            train_sha256: "a".repeat(64),
            source_family: if index < 2 {
                "family-a' OR 1=1 --"
            } else {
                "family-b"
            }
            .to_string(),
            context_sha256: "b".repeat(64),
            catalog_index: index,
        })
        .collect::<Vec<_>>();
    for index in 0..35 {
        let genome = canonical(&format!(
            "(pipe X (elasticnet alpha={} l1_ratio=0.5))",
            index as f64 + 0.75
        ));
        for dataset in ["target' OR 1=1 --", "same-family-source", "clean-source"] {
            connection
                    .execute(
                        "INSERT INTO observations VALUES('regression',?1,?2,?3,'transfer',?4,'gp_origin')",
                        rusqlite::params![dataset, semantic_identity(&genome), genome, index as f64],
                    )
                    .unwrap();
        }
    }
    let dl_genome = canonical("(pipe X (ridge l2=99))");
    connection
            .execute(
                "INSERT INTO observations VALUES('regression','clean-source',?1,?2,'checkpoint_dl',999.0,'dl_origin')",
                rusqlite::params![semantic_identity(&dl_genome), dl_genome],
            )
            .unwrap();
    drop(connection);
    let catalog = Catalog {
        dataset_dimensions: 1536,
        target_dimensions: 512,
        record_stride_bytes: CONTEXT_BYTES,
        entries: entries.clone(),
    };
    let rankings = build_transfer_rankings(
        "fixture",
        &database,
        &sha256_file(&database).unwrap(),
        &catalog,
        &"c".repeat(64),
        &entries[..1],
    )
    .unwrap();
    let ranking = &rankings["target' OR 1=1 --"];
    assert!(ranking.genomes.len() >= GLOBAL_QUOTA);
    assert_eq!(
        ranking.excluded_dataset_ids,
        vec![
            "same-family-source".to_string(),
            "target' OR 1=1 --".to_string(),
        ]
    );
    assert!(
        ranking.genomes.iter().all(|genome| {
            genome.source_dataset_ids == vec!["clean-source".to_string()]
                && genome.source_families == vec!["family-b".to_string()]
                && genome.semantic_sha256 != semantic_identity(&dl_genome)
        }),
        "unexpected ranking row: {:?}",
        ranking.genomes.iter().find(|genome| {
            genome.source_dataset_ids != vec!["clean-source".to_string()]
                || genome.source_families != vec!["family-b".to_string()]
                || genome.semantic_sha256 == semantic_identity(&dl_genome)
        })
    );
}

#[test]
fn refill_is_deterministic_preserves_three_elites_and_history_uniqueness() {
    let mut ranked = (0..POPULATION)
        .map(|index| {
            scored(
                member(
                    index,
                    &format!("(pipe X (ridge l2={}))", index as f64 + 0.25),
                    1,
                ),
                1.0 - index as f64 / 100.0,
                0.5 - index as f64 / 1000.0,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(rank_scored);
    let initial = ranked
        .iter()
        .map(|member| member.member.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut first_history = initial.clone();
    let mut second_history = initial;
    let first = refill_population("fixture", "dataset", 4, 1, &ranked, &mut first_history).unwrap();
    let second =
        refill_population("fixture", "dataset", 4, 1, &ranked, &mut second_history).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), POPULATION);
    assert_eq!(
        first[..ELITES]
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<Vec<_>>(),
        ranked[..ELITES]
            .iter()
            .map(|member| &member.member.semantic_sha256)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<HashSet<_>>()
            .len(),
        POPULATION
    );
    assert_eq!(first_history, second_history);
    assert_eq!(first_history.len(), POPULATION + (POPULATION - ELITES));
}

#[test]
#[ignore = "legacy v1 strict-firewall fixture; v2 production registry is covered separately"]
fn preparation_completes_without_test_or_baseline_files() {
    let directory = tempdir().unwrap();
    let registry = directory.path().join("registry");
    fs::create_dir_all(&registry).unwrap();
    let input_root = directory.path().join("inputs");
    fs::create_dir_all(&input_root).unwrap();

    let checkpoint_path = input_root.join("checkpoint.model");
    fs::write(&checkpoint_path, b"fixture checkpoint").unwrap();
    let checkpoint_sha256 = sha256_file(&checkpoint_path).unwrap();
    let accepted_pointer = input_root.join("accepted.json");
    fs::write(
        &accepted_pointer,
        serde_json::to_vec(&json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "tasks": {
                "regression": {
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let corpus_train = input_root.join("corpus.train.jsonl");
    let corpus_validation = input_root.join("corpus.val.jsonl");
    fs::write(&corpus_train, "").unwrap();
    fs::write(&corpus_validation, "").unwrap();
    let corpus_manifest = input_root.join("corpus-manifest.json");
    fs::write(
        &corpus_manifest,
        serde_json::to_vec(&json!({
            "outputs": {
                "corpus.train.jsonl": sha256_file(&corpus_train).unwrap(),
                "corpus.val.jsonl": sha256_file(&corpus_validation).unwrap(),
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let provenance = input_root.join("provenance.json");
    fs::write(
        &provenance,
        serde_json::to_vec(&json!({
            "checkpoint_sha": checkpoint_sha256,
            "corpus_manifest": corpus_manifest,
            "checkpoint_config": {"model_version": "fixture-v1"},
        }))
        .unwrap(),
    )
    .unwrap();

    let context_slice = vec![0_u8; CONTEXT_BYTES];
    let context_sha256 = sha256_bytes(&context_slice);
    let mut context_blob = Vec::new();
    let mut catalog_entries = Vec::new();
    for (catalog_index, dataset_hash) in GOLDEN42[..10].iter().enumerate() {
        let dataset_root = registry.join(dataset_hash);
        fs::create_dir_all(&dataset_root).unwrap();
        let train = (0..100)
            .map(|row| {
                format!(
                    "{},{},{}\n",
                    row as f64 / 100.0,
                    (row as f64).sin(),
                    row as f64 / 10.0 + (row as f64).cos()
                )
            })
            .collect::<String>();
        let train_path = dataset_root.join("train.csv");
        fs::write(&train_path, train).unwrap();
        fs::write(
            dataset_root.join("meta.json"),
            serde_json::to_vec(&json!({
                "source_family": format!("fixture-family-{catalog_index}"),
                "source_platform": "fixture",
                "source_dataset_id": format!("fixture-source-{catalog_index}"),
                "source_dataset_name": format!("fixture-name-{catalog_index}"),
                "source_target_column_original": "y",
                "target_column": "target",
                "target_names": ["target"],
            }))
            .unwrap(),
        )
        .unwrap();
        catalog_entries.push(json!({
            "dataset_hash": dataset_hash,
            "task": "regression",
            "train_csv": train_path,
            "train_sha256": sha256_file(&train_path).unwrap(),
            "source_family": format!("fixture-family-{catalog_index}"),
            "context_sha256": context_sha256,
            "catalog_index": catalog_index,
        }));
        context_blob.extend_from_slice(&context_slice);
    }
    let context_data = input_root.join("contexts.bin");
    fs::write(&context_data, &context_blob).unwrap();
    let catalog_path = input_root.join("catalog.json");
    fs::write(
        &catalog_path,
        serde_json::to_vec(&json!({
            "dataset_dimensions": 1536,
            "target_dimensions": 512,
            "record_stride_bytes": CONTEXT_BYTES,
            "entries": catalog_entries,
        }))
        .unwrap(),
    )
    .unwrap();
    let catalog_pointer = input_root.join("catalog-pointer.json");
    fs::write(
        &catalog_pointer,
        serde_json::to_vec(&json!({
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path).unwrap(),
            "data": context_data,
            "data_sha256": sha256_file(&context_data).unwrap(),
        }))
        .unwrap(),
    )
    .unwrap();

    let proposals_path = input_root.join("proposals.jsonl");
    let mut proposal_lines = String::new();
    for dataset_hash in &GOLDEN42[..10] {
        for index in 0..64 {
            let row = proposal_row(dataset_hash, &checkpoint_sha256, &context_sha256, index);
            writeln!(proposal_lines, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
    }
    fs::write(&proposals_path, proposal_lines).unwrap();

    let observation_db = input_root.join("observations.sqlite");
    let connection = Connection::open(&observation_db).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE observations(
                    task TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    canonical_genome TEXT NOT NULL,
                    candidate_source TEXT NOT NULL,
                    score REAL NOT NULL,
                    origin_class TEXT NOT NULL
                );",
        )
        .unwrap();
    for index in 0..35 {
        let genome = canonical(&format!(
            "(pipe X (elasticnet alpha={} l1_ratio=0.5))",
            index as f64 + 0.75
        ));
        connection
                .execute(
                    "INSERT INTO observations VALUES('regression','external',?1,?2,'global',?3,'gp_origin')",
                    rusqlite::params![
                        semantic_identity(&genome),
                        genome,
                        index as f64
                    ],
                )
                .unwrap();
    }
    drop(connection);

    let study_root = directory.path().join("studies");
    let args = PrepareArgs {
        run_id: "fixture-study".to_string(),
        study_root: study_root.clone(),
        checkpoint_pointer: accepted_pointer,
        checkpoint_provenance: provenance,
        catalog_pointer,
        observation_db,
        dl_proposals: Some(proposals_path),
        proposal_generator: PathBuf::new(),
        python: PathBuf::new(),
        proposal_device: "cpu".to_string(),
        sotarena_root: registry.clone(),
        workers: 1,
        smoke: false,
        proposal_manifest_only: false,
    };
    let manifest_path = prepare(&registry, args.clone()).unwrap();
    let first_bytes = fs::read(&manifest_path).unwrap();
    let (manifest, _): (TrainOnlyStudyManifest, _) = read_receipt(&manifest_path).unwrap();
    assert_eq!(manifest.datasets.len(), 10);
    assert!(manifest.firewall.train_only);
    let lower = String::from_utf8_lossy(&first_bytes).to_ascii_lowercase();
    assert!(!lower.contains("/test.csv"));
    assert!(!lower.contains("tabpfn"));
    assert!(!lower.contains("tabfm"));
    assert!(registry.read_dir().unwrap().all(|entry| !entry
        .unwrap()
        .path()
        .join("test.csv")
        .exists()));

    let repeated = prepare(&registry, args).unwrap();
    assert_eq!(manifest_path, repeated);
    assert_eq!(first_bytes, fs::read(repeated).unwrap());
}

#[test]
fn score_refuses_before_validation_receipt_exists() {
    let directory = tempdir().unwrap();
    let error = score(directory.path(), "fixture", directory.path(), 1).unwrap_err();
    assert!(format!("{error:#}").contains("requires a successful independent validation"));
    assert!(!directory.path().join("fixture/test.csv").exists());
}
