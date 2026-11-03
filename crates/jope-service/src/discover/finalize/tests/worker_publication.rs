use super::*;

#[test]
fn a_controller_change_before_holdout_prevents_further_reads_and_late_publication() {
    struct Pause {
        store: Arc<Store>,
        native: Native,
        count: AtomicUsize,
    }
    impl Executor for Pause {
        fn evaluate(&self, request: &Request) -> Response {
            let response = self.native.evaluate(request);
            if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
                let log = EventLog::new(Arc::clone(&self.store));
                let mut snapshot = log.snapshot(&request.attempt.run).unwrap().unwrap();
                let record = self
                    .store
                    .get(&RecordKey::Run(request.attempt.run.clone()))
                    .unwrap()
                    .unwrap();
                let revision = snapshot.revision.0;
                snapshot.revision.0 += 1;
                snapshot.state = RunState::Paused;
                log.commit(
                    RunUpdate {
                        expected_revision: Some(revision),
                        snapshot,
                        checkpoint: record.data["checkpoint"].clone(),
                        blobs: record.blobs,
                    },
                    vec![],
                    |_| Ok(()),
                )
                .unwrap();
            }
            response
        }
        fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
            bail!("unused")
        }
    }
    let f = fixture(false);
    search(&f, &Native::default());
    let executor = Pause {
        store: Arc::clone(&f.store),
        native: Native::default(),
        count: AtomicUsize::new(0),
    };
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    assert!(finalizer
        .finalize(&f.request, &executor)
        .unwrap_err()
        .to_string()
        .contains("controller changed"));
    assert_eq!(
        finalizer
            .journal()
            .snapshot(&f.plan.run_id)
            .unwrap()
            .unwrap()
            .state,
        RunState::Paused
    );
    assert!(!f.request.model_root.exists());
    let claim = f.store.get(&claim_key(&f.plan.run_id)).unwrap().unwrap();
    assert_eq!(claim.data["audit"]["entries"].as_array().unwrap().len(), 1);
}

#[test]
#[ignore = "operational gate: build actual jopedime, set JOPE_DSC09_APPLIANCE_EXE and execute explicitly"]
fn real_appliance_workers_finalize_and_jopedime_predict_loads_the_model_object() {
    let executable = PathBuf::from(
        std::env::var_os("JOPE_DSC09_APPLIANCE_EXE")
            .expect("actual built appliance executable required"),
    );
    let f = fixture(false);
    let executor =
        crate::discover::r#loop::ProcessExecutor::new(&executable, 1, &f.request.scratch_root)
            .unwrap();
    search(&f, &executor);
    let receipt = Finalizer::new(Arc::clone(&f.store))
        .finalize(&f.request, &executor)
        .unwrap();
    let part = f.request.test.as_ref().unwrap();
    let table = load(part, &f.plan, f.request.max_input_bytes).unwrap();
    let csv_path = f.directory.path().join("predict.csv");
    let mut csv = table.headers.join(",") + "\n";
    for row in &table.rows {
        csv.push_str(
            &row.iter()
                .map(|v| v.as_deref().unwrap())
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }
    fs::write(&csv_path, csv).unwrap();
    let output_path = f.directory.path().join("predictions.csv");
    let result = std::process::Command::new(&executable)
        .args(["predict", "--model"])
        .arg(&receipt.model_path)
        .args(["--input"])
        .arg(&csv_path)
        .args(["--output"])
        .arg(&output_path)
        .args(["--threads", "1", "--device", "cpu", "--json"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let cli_receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(cli_receipt["outcome"], "succeeded");
    assert!(output_path.is_file());
    let object = ModelObject::read(&receipt.model_path, HANDOFF_MAX).unwrap();
    let expected = predict(&object, &table).unwrap();
    let actual =
        jope_runtime::table::read_csv(&output_path, &jope_runtime::limits::Limits::default())
            .unwrap();
    let observed: Vec<f64> = actual
        .rows
        .iter()
        .map(|r| r[0].as_deref().unwrap().parse().unwrap())
        .collect();
    assert_eq!(expected, observed);
    if let Some(root) = std::env::var_os("JOPE_DSC09_RECEIPT_DIR") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("finalization.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        fs::write(root.join("predict-receipt.json"), &result.stdout).unwrap();
        fs::copy(&output_path, root.join("predictions.csv")).unwrap();
        fs::copy(&csv_path, root.join("predict-input.csv")).unwrap();
        // Preserve executable model parts and the exact TRAIN-derived input
        // mappings before fixture cleanup; receipts alone are not replayable.
        object.save(&root.join("models"), HANDOFF_MAX).unwrap();
        for part in [&f.request.fit, &f.request.holdout]
            .into_iter()
            .chain(f.request.test.as_ref())
        {
            fs::copy(&part.path, root.join(part.path.file_name().unwrap())).unwrap();
        }
        fs::write(
            root.join("source.json"),
            serde_json::to_vec_pretty(&json!({"source_commit":f.plan.binary_version,"source_train_sha256":TRAIN_SHA,"source_train_path":"test-support/datasets/v1/d2-regression/train.csv","external_holdout_or_test_reads":0,"plan":f.plan})).unwrap(),
        ).unwrap();
        fs::write(
            root.join("input-identities.json"),
            serde_json::to_vec_pretty(&f.request).unwrap(),
        )
        .unwrap();
    }
}
#[test]
fn fit_holdout_publication_keeps_the_genuine_fit_model_and_replay_does_not_refit() {
    let f = fixture(false);
    let native = Native::default();
    search(&f, &native);
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    let receipt = finalizer.finalize(&f.request, &native).unwrap();
    assert_eq!(receipt.refit, Refit::FitHoldout);
    let fit_id: ArtifactId = format!("art_{}", &receipt.holdout_model_digest[7..31])
        .parse()
        .unwrap();
    assert_ne!(fit_id, receipt.model_artifact);
    let fit_record = f.store.get(&RecordKey::Artifact(fit_id)).unwrap().unwrap();
    let primary = f
        .store
        .get(&RecordKey::Artifact(receipt.model_artifact.clone()))
        .unwrap()
        .unwrap();
    assert_eq!(fit_record.data["refit"], json!(Refit::Fit));
    assert_eq!(
        fit_record.data["model_digest"],
        receipt.holdout_model_digest
    );
    assert_eq!(primary.data["refit"], json!(Refit::FitHoldout));
    let fit_object = ModelObject::read(
        &f.request.model_root.join(&receipt.holdout_model_digest),
        HANDOFF_MAX,
    )
    .unwrap();
    assert_eq!(
        f.store.read_blob(&fit_record.blobs[0]).unwrap(),
        fit_object.manifest_bytes()
    );
    let before = native.calls.lock().unwrap().len();
    let replay = finalizer.finalize(&f.request, &native).unwrap();
    assert_eq!(receipt.model_digest, replay.model_digest);
    assert_eq!(native.calls.lock().unwrap().len(), before);
}

#[test]
fn repeated_frozen_model_preserves_measured_parity_and_refuses_other_drift() {
    let f = fixture(false);
    let key = RecordKey::Artifact("art_00112233445566778899aabb".parse().unwrap());
    let frozen = Record::metadata(
        json!({"kind":"finalized_model","model_digest":"frozen","refit":"fit","manifest":{}}),
        vec![],
    );
    f.store
        .transaction(|tx| put_frozen_model(tx, key.clone(), frozen.clone()))
        .unwrap();
    let mut measured = frozen.clone();
    measured.data["parity_receipt"] = json!("art_aabbccddeeff001122334455");
    f.store.transaction(|tx| tx.put(&key, &measured)).unwrap();
    f.store
        .transaction(|tx| put_frozen_model(tx, key.clone(), frozen.clone()))
        .unwrap();
    assert_eq!(f.store.get(&key).unwrap().unwrap(), measured);
    let mut changed = frozen;
    changed.data["refit"] = json!("fit_holdout");
    assert!(f
        .store
        .transaction(|tx| put_frozen_model(tx, key.clone(), changed))
        .is_err());
    assert_eq!(f.store.get(&key).unwrap().unwrap(), measured);
}
