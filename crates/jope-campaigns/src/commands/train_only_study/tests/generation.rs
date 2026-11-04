use super::*;

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
