#[cfg(test)]
mod corpus_cache_tests {
    use super::*;

    #[test]
    fn cache_is_shared_by_campaign_cycles_and_isolated_for_other_runs() {
        let root = Path::new("/workspace");
        let path = |s: &str| corpus_cache_path(root, &root.join(s)).unwrap();
        assert_eq!(
            path("runs/campaign/cycle-0000/corpus-2"),
            path("runs/campaign/cycle-0001/corpus-2")
        );
        assert_ne!(
            path("runs/campaign/cycle-0000/corpus-2"),
            path("runs/campaign/expansion/cycle-0000/corpus-2")
        );
        assert_eq!(
            path("runs/standalone/corpus-2"),
            root.join("runs/standalone/corpus-scan-cache")
        );
        assert_eq!(
            path("runs/cycle-0000/corpus-2"),
            root.join("runs/cycle-0000/corpus-scan-cache")
        );
        assert!(corpus_cache_path(root, &root.join("data/corpus-2")).is_err());
    }

    #[test]
    fn pending_jobs_require_current_process_bound_activity() {
        let pending = vec![PathBuf::from("runs/pending")];
        let recent_unbound =
            json!({"timestamp":100,"expires_at":110,"last_progress_at":100,"phase":"running"});
        assert!(check_gp_health(&recent_unbound, &pending, Duration::ZERO, 101).is_err());
        assert!(check_gp_health(&Value::Null, &pending, Duration::from_secs(9), 100).is_ok());
        assert!(check_gp_health(&Value::Null, &pending, Duration::from_secs(10), 100).is_err());
        assert!(check_gp_health(&recent_unbound, &[], Duration::from_secs(60), 160).is_ok());
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("heartbeat.json");
        let heartbeat = StageHeartbeat::start(&path, json!({"phase":"running"})).unwrap();
        let saved: Value = read_json(&path).unwrap();
        let timestamp = saved["timestamp"].as_u64().unwrap();
        assert!(check_gp_health(&saved, &pending, Duration::from_secs(30), timestamp + 9).is_ok());
        assert!(
            check_gp_health(&saved, &pending, Duration::from_secs(30), timestamp + 10).is_err()
        );
        heartbeat.finish().unwrap();
    }

    #[test]
    fn atomic_campaign_pack_creates_private_cache_and_preserves_full_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let vocab = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
        let mut contexts = Vec::new();
        let mut observations = Vec::new();
        for family in ["a", "b"] {
            let prepared = root.join(family);
            write_json(&prepared.join("prepared.json"),&json!({"schema":"parallel.prepared.v1","dataset":family,"family":family,
                "task":"regression","meta_partition":"meta_train","seed":17,"clean_evaluation":true,"clean_status":"fixture",
                "original_train_sha256":"original","original_test_sha256":"sealed","selected_train_indices":[0,1],
                "selected_test_indices":[],"target_last":true,"rows_total":2,"cuts":[{"index":0,"train_indices":[0],
                "validation_indices":[1],"training_sha256":"train","validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
            let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
            let vector = prepared.join("context.bin");
            fs::write(&vector, b"saved fixture context").unwrap();
            let vector_sha = file_hash(&vector).unwrap();
            write_json(&vector.with_extension("json"),&json!({"schema":"parallel.encoding.v1","dataset":family,"family":family,
                "partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,"training_sha256":"train",
                "selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing","encoder_spec_sha256":"assembly",
                "numerical_backend":"fixture","dataset_dimensions":1536,"target_dimensions":512,"output_sha256":vector_sha})).unwrap();
            contexts.push(json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train","path":vector.strip_prefix(root).unwrap(),
                "sha256":vector_sha,"prepared":family,"prepared_sha256":prepared_sha}));
            // Misleading directory names must not filter valid row identities.
            let source = root.join(format!("observations/not-{family}/cut-999"));
            let rows=["standardize","robust_scale","norm_row","row_stats","rank","blom","sigmoid","step"].iter().enumerate().map(|(n,t)|{
                let genome=format!("(pipe X ({t}) (ridge))");json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                    "training_sha256":"train","validation_sha256":"validation","genome":genome,"program_sha256":hash(genome.as_bytes()),
                    "metrics":{"rmse":n as f64+1.},"runtime_seconds":0.1,"evaluation_bias":"fixture","failure":null})}).collect::<Vec<_>>();
            write_json(&source.join("observations.json"), &rows).unwrap();
            write_json(&source.join("search.json"), &json!({})).unwrap();
            observations.push(source);
        }
        let index = root.join("contexts.json");
        write_json(&index,&json!({"schema":"parallel.context_index.v1","qualification":true,"assembly_sha256":"assembly","encodings":contexts})).unwrap();
        let full = root.join("full");
        corpus::build(root, &index, &vocab, &observations, &full, true).unwrap();
        let first = root.join("runs/campaign/cycle-0000/corpus-2");
        let second = root.join("runs/campaign/cycle-0001/corpus-2");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::create_dir_all(second.parent().unwrap()).unwrap();
        build_corpus_atomic(root, &index, &vocab, &observations, &first, None).unwrap();
        let cache = corpus_cache_path(root, &first).unwrap();
        assert!(cache.is_dir());
        let inventory = || {
            walkdir::WalkDir::new(&cache)
                .into_iter()
                .map(|e| e.unwrap().into_path())
                .filter(|p| p.is_file())
                .map(|p| {
                    let sha = file_hash(&p).unwrap();
                    (p, sha)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let before = inventory();
        assert!(!before.is_empty());
        build_corpus_atomic(root, &index, &vocab, &observations, &second, None).unwrap();
        assert_eq!(before, inventory());
        for name in [
            "records.bin",
            "labels.jsonl",
            "index.json",
            "provenance.json",
            "vocabulary.json",
        ] {
            assert_eq!(
                fs::read(full.join(name)).unwrap(),
                fs::read(first.join(name)).unwrap()
            );
            assert_eq!(
                fs::read(full.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        assert!(!root.join("runs/corpus-progress.json").exists());
        assert!(!second.with_file_name(".corpus-2.staging").exists());
        fs::write(second.join("records.bin"), b"corrupt committed output").unwrap();
        assert!(build_corpus_atomic(root, &index, &vocab, &observations, &second, None).is_err());
        assert_eq!(
            fs::read(second.join("records.bin")).unwrap(),
            b"corrupt committed output"
        );
    }
}
