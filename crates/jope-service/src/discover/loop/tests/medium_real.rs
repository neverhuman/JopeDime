use super::*;

#[test]
#[ignore = "dedicated real-data quality gate: first cargo build -p jope-service --locked, then run this exact test with --ignored"]
fn medium_real_qlt05_regression_beats_ridge_cv() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 800, Effort::Medium, 128);
    let binary = native_worker(directory.path());
    let workers = pool(&binary, 8);
    let store = Arc::new(Store::open(directory.path().join("quality")).unwrap());
    let mut d = Discovery::open(store, p.clone()).unwrap();
    let ridge = NativeSpace.canonical("(pipe X (ridge))").unwrap();
    let request = d
        .request(
            999,
            0,
            1,
            &Candidate {
                algorithm_id: "alg_9999".into(),
                dsl: ridge,
            },
        )
        .unwrap();
    let baseline = workers.evaluate(&request);
    assert_eq!(baseline.failure, None);
    finish(&mut d, &NativeSpace, &workers, 8);
    let champion = d
        .champion()
        .expect("medium search must produce a finite rung-1 model");
    println!("DSC08 QLT05 source={TRAIN_SHA}, FIT=800, seed=73, folds=5, medium128/R2; ridge_cv_rmse={}, champion_cv_rmse={}, admissions={}, reason={:?}, nondeterminism={:?}",baseline.scores[2],champion.value,d.admitted(),d.completed(),d.nondeterminism());
    assert!(
        champion.loss < baseline.scores[1],
        "medium must strictly improve ridge on the same pinned CV folds"
    );
    // Quality may be time-bound under the explicit spec §5.6 exception.
    // Every actual limit must be disclosed; only the separate zero-limit
    // S2 runs establish deterministic replay equality.
    let projection = d.projection().unwrap();
    let limits: Vec<String> = projection
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| {
            matches!(
                e["body"]["failure"]["reason"].as_str(),
                Some(reason) if reason == public_failure(WorkerFailure::Timeout) || reason == public_failure(WorkerFailure::MemoryLimit)
            )
        })
        .map(|e| e["body"]["experiment_id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        limits,
        d.nondeterminism().experiment_ids,
        "all real worker limits must be listed, in commit order"
    );
    assert_eq!(limits.len() as u64, d.nondeterminism().count.0);
    d.store.verify_blob(&champion.model).unwrap();
    if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
        let receipt = PathBuf::from(receipt);
        fs::create_dir_all(&receipt).unwrap();
        fs::copy(&p.matrix_path, receipt.join("quality-fit.jmat")).unwrap();
        fs::copy(
            directory.path().join("fit-source-rows.json"),
            receipt.join("quality-fit-source-rows.json"),
        )
        .unwrap();
        fs::write(receipt.join("quality.json"),serde_json::to_vec_pretty(&json!({"plan":p,"source_train_sha256":TRAIN_SHA,"ridge_scores":baseline.scores,"champion":{"candidate":&champion.candidate,"loss":champion.loss,"value":champion.value,"predictions_digest":&champion.predictions_digest,"model":&champion.model},"admissions":d.admitted(),"completion_reason":d.completed(),"budget_bound_by":d.budget_bound_by(),"nondeterminism":d.nondeterminism(),"projection":d.projection().unwrap()})).unwrap()).unwrap();
    }
}
