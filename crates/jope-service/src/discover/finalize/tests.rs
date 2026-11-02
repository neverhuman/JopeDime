use super::*;

mod tests {
    use super::*;
    use crate::discover::r#loop::{evaluate_native, Discovery, HeuristicAdmission, SearchSpace};
    use crate::worker::{Response, Scores};
    use rand_chacha::ChaCha12Rng;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };
    const TRAIN: &[u8] =
        include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    const TRAIN_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

    struct Ridge;
    impl SearchSpace for Ridge {
        fn canonical(&self, dsl: &str) -> Result<String> {
            Ok(feat_math::GenomeIr::parse(dsl)?.canonical())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(vec![self.canonical("(pipe X (ridge))")?])
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }
    #[derive(Default)]
    struct Native {
        calls: Mutex<Vec<(u64, u32, PathBuf)>>,
        failure: Option<WorkerFailure>,
    }
    impl Executor for Native {
        fn evaluate(&self, request: &Request) -> Response {
            self.calls.lock().unwrap().push((
                request.attempt.round,
                request.attempt.slot,
                request.matrix_path.clone(),
            ));
            if let Some(reason) = self.failure {
                return Response::failed(request.attempt.clone(), reason, 0);
            }
            let matrix = Matrix::open(fs::read(&request.matrix_path).unwrap()).unwrap();
            match evaluate_native(request, &matrix) {
                Ok(Scores {
                    values,
                    fold_scores,
                    predictions_digest,
                }) => Response {
                    attempt: request.attempt.clone(),
                    scores: values,
                    fold_scores,
                    predictions_digest,
                    wall_ms: 0,
                    failure: None,
                },
                Err(reason) => Response::failed(request.attempt.clone(), reason, 0),
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            let temp = tempfile::tempdir()?;
            let output = temp.path().join("model.json");
            let mut request = request.clone();
            request.fold_spec["refit_output"] = json!(output);
            request.fold_spec["refit_max_bytes"] = json!(HANDOFF_MAX);
            ensure!(
                self.evaluate(&request).failure.is_none(),
                "native fixture fit failed"
            );
            let envelope: Envelope = serde_json::from_slice(&fs::read(output)?)?;
            Ok(envelope.model_json.into_bytes())
        }
    }
    struct Fixture {
        directory: tempfile::TempDir,
        store: Arc<Store>,
        plan: Plan,
        request: FinalizationRequest,
    }
    fn fixture(missing: bool) -> Fixture {
        assert_eq!(digest(TRAIN), TRAIN_SHA);
        let directory = tempfile::tempdir().unwrap();
        let mut headers: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
        headers.push("target".into());
        let mut csv = headers.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend_from_slice(TRAIN);
        let table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        // Source is only the approved TRAIN object; logical roles use disjoint
        // source ordinals. No external HOLDOUT/TEST/sealed object is opened.
        let (roster, _) = crate::discover::splits::random_split(
            table.rows.len(),
            table.rows.len() * 4 / 5,
            73,
            "dsc09/train-only-fixture",
        )
        .unwrap();
        let dataset: jope_api::ids::DatasetId = "ds_21fbfe8b53dbcd605d79486b".parse().unwrap();
        let part = |name: &str, range: std::ops::Range<usize>, absent: bool| {
            let source: Vec<_> = range.map(|i| roster[i]).collect();
            let mut selected = table.clone();
            selected.rows = source.iter().map(|&row| table.rows[row].clone()).collect();
            if absent {
                selected.rows[0][0] = None;
            }
            let bytes = crate::dataset::jmat::encode(&selected).unwrap();
            let path = directory.path().join(name);
            fs::write(&path, &bytes).unwrap();
            Partition {
                path,
                sha256: digest(&bytes),
                rows: source
                    .into_iter()
                    .map(|row| AuditRowId {
                        dataset: dataset.clone(),
                        row: row as u64,
                    })
                    .collect(),
            }
        };
        let fit = part("fit.jmat", 0..40, missing);
        let holdout = part("holdout.jmat", 40..50, false);
        let test = part("test.jmat", 50..60, false);
        let plan = Plan {
            run_id: RunId::new(),
            dataset_id: dataset,
            problem_digest: digest(b"DSC09 TRAIN-only logical finalization fixture"),
            bundle_digest: digest(b"heuristic M1"),
            binary_version: std::env::var("JOPE_DSC09_SOURCE_SHA")
                .unwrap_or_else(|_| "dsc09-unit-fixture".into()),
            matrix_path: fit.path.clone(),
            matrix_sha256: fit.sha256.clone(),
            target: "target".into(),
            features: headers[..21].to_vec(),
            objective: "rmse".into(),
            seed: 73,
            planned_workers: 8,
            effort: jope_api::Effort::Fast,
            evaluations: 1,
            wall_seconds: 120,
            evaluation_wall_ms: 10_000,
            evaluation_rss_bytes: 512 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let request = FinalizationRequest {
            run_id: plan.run_id.clone(),
            fit,
            holdout,
            test: Some(test),
            refit: Refit::FitHoldout,
            model_root: directory.path().join("models"),
            scratch_root: directory.path().join("scratch"),
            max_input_bytes: 1024 * 1024,
            max_model_bytes: HANDOFF_MAX,
        };
        Fixture {
            directory,
            store,
            plan,
            request,
        }
    }
    fn search(f: &Fixture, executor: &impl Executor) -> Champion {
        let mut d = Discovery::open(Arc::clone(&f.store), f.plan.clone()).unwrap();
        d.advance(&Ridge, &HeuristicAdmission, executor, 1, usize::MAX)
            .unwrap();
        assert!(d.completed().is_some());
        d.champion().unwrap().clone()
    }
    fn claim_key(run: &RunId) -> RecordKey {
        RecordKey::Command {
            principal: "discover.finalization.v1".into(),
            idempotency_key: run.to_string(),
        }
    }

    #[test]
    fn finalization_audits_each_real_partition_and_declares_the_refit_and_score_scopes() {
        for kind in [Refit::Fit, Refit::FitHoldout] {
            let mut f = fixture(false);
            f.request.refit = kind;
            let native = Native::default();
            let selected = search(&f, &native);
            assert!(native
                .calls
                .lock()
                .unwrap()
                .iter()
                .all(|(_, _, path)| path == &f.plan.matrix_path));
            let finalizer = Finalizer::new(Arc::clone(&f.store));
            let receipt = finalizer.finalize(&f.request, &native).unwrap();
            assert_eq!(receipt.scores.cv, Some(selected.value));
            assert_eq!(
                receipt.completion_reason,
                CompletionReason::BudgetEvaluations
            );
            assert_eq!(receipt.budget_bound_by.as_deref(), Some("evaluations"));
            assert_eq!(receipt.budget_used.evaluations.0, 1);
            assert_eq!(receipt.nondeterminism.count.0, 0);
            assert!(receipt.scores.holdout.unwrap().is_finite());
            assert!(receipt.scores.test.unwrap().is_finite());
            let exported = ModelObject::read(&receipt.model_path, HANDOFF_MAX).unwrap();
            exported.verify_executable().unwrap();
            assert_eq!(
                exported.manifest().metadata["refit"],
                serde_json::to_value(kind).unwrap()
            );
            assert_eq!(
                exported.manifest().metadata["training_rows"],
                if kind == Refit::Fit { 40 } else { 50 }
            );
            assert_eq!(
                receipt.model_digest == receipt.holdout_model_digest,
                kind == Refit::Fit
            );
            let entries = receipt.audit["entries"].as_array().unwrap();
            assert_eq!(
                entries
                    .iter()
                    .filter(|e| e["phase"] == "test_score")
                    .count(),
                1
            );
            for entry in entries {
                let phase = entry["phase"].as_str().unwrap();
                for row in entry["rows"].as_array().unwrap() {
                    let id = row["row"].as_u64().unwrap();
                    if f.request.holdout.rows.iter().any(|r| r.row == id) {
                        assert!(matches!(phase, "holdout_score" | "finalization_refit"));
                    }
                    if f.request
                        .test
                        .as_ref()
                        .unwrap()
                        .rows
                        .iter()
                        .any(|r| r.row == id)
                    {
                        assert_eq!(phase, "test_score");
                    }
                }
            }
            let snapshot = finalizer
                .journal()
                .snapshot(&f.plan.run_id)
                .unwrap()
                .unwrap();
            assert_eq!(snapshot.state, RunState::Completed);
            assert_eq!(snapshot.phase, RunPhase::Finalizing);
            assert_eq!(
                snapshot.best.unwrap().algorithm_id.to_string(),
                selected.candidate.algorithm_id
            );
            let events = finalizer
                .journal()
                .page(&f.plan.run_id, 0, 256, &[])
                .unwrap()
                .unwrap();
            assert_eq!(
                events
                    .events
                    .iter()
                    .filter(|e| matches!(e.event, EventBody::RunFinished(_)))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn completed_replay_and_recovery_do_not_read_or_score_protected_partitions_again() {
        let f = fixture(false);
        let native = Native::default();
        search(&f, &native);
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let first = finalizer.finalize(&f.request, &native).unwrap();
        let calls = native.calls.lock().unwrap().len();
        fs::rename(
            &f.request.holdout.path,
            f.directory.path().join("holdout-unavailable"),
        )
        .unwrap();
        fs::rename(
            &f.request.test.as_ref().unwrap().path,
            f.directory.path().join("test-unavailable"),
        )
        .unwrap();
        drop(finalizer);
        drop(f.store);
        let recovered = Arc::new(Store::open(f.directory.path().join("store")).unwrap());
        let finalizer = Finalizer::new(recovered);
        let replay = finalizer.finalize(&f.request, &native).unwrap();
        assert_eq!(first.model_digest, replay.model_digest);
        assert_eq!(calls, native.calls.lock().unwrap().len());
        let mut changed = f.request.clone();
        changed.refit = Refit::Fit;
        assert!(finalizer.finalize(&changed, &native).is_err());
        assert_eq!(calls, native.calls.lock().unwrap().len());

        // Run provenance belongs to the run-scoped receipt, not the canonical
        // predictor identity: the same problem and fit must have one digest.
        let other = fixture(false);
        let other_native = Native::default();
        search(&other, &other_native);
        let independent = Finalizer::new(Arc::clone(&other.store))
            .finalize(&other.request, &other_native)
            .unwrap();
        assert_ne!(first.run_id, independent.run_id);
        assert_eq!(first.run_id, f.request.run_id);
        assert_eq!(independent.run_id, other.request.run_id);
        assert_eq!(first.model_digest, independent.model_digest);
        assert_eq!(first.model_artifact, independent.model_artifact);
        assert_eq!(first.holdout_model_digest, independent.holdout_model_digest);
    }

    #[test]
    fn failed_bounded_fit_preserves_audit_and_never_retries_or_reads_holdout_or_test() {
        let f = fixture(false);
        search(&f, &Native::default());
        let failing = Native {
            failure: Some(WorkerFailure::Timeout),
            ..Native::default()
        };
        fs::rename(
            &f.request.holdout.path,
            f.directory.path().join("holdout-unavailable"),
        )
        .unwrap();
        fs::rename(
            &f.request.test.as_ref().unwrap().path,
            f.directory.path().join("test-unavailable"),
        )
        .unwrap();
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        assert!(finalizer.finalize(&f.request, &failing).is_err());
        let record = f.store.get(&claim_key(&f.plan.run_id)).unwrap().unwrap();
        assert_eq!(record.data["audit"]["entries"].as_array().unwrap().len(), 1);
        assert_eq!(
            record.data["audit"]["entries"][0]["phase"],
            "finalization_fit"
        );
        assert_eq!(
            finalizer
                .journal()
                .snapshot(&f.plan.run_id)
                .unwrap()
                .unwrap()
                .state,
            RunState::Failed
        );
        assert!(finalizer.finalize(&f.request, &Native::default()).is_err());
        assert!(!f.request.model_root.exists());
    }

    #[test]
    fn bad_rosters_refuse_before_claim_and_missing_values_refuse_without_format_drift() {
        let f = fixture(false);
        let native = Native::default();
        search(&f, &native);
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let mut bad = f.request.clone();
        bad.holdout.rows[0] = bad.fit.rows[0].clone();
        assert!(finalizer.finalize(&bad, &native).is_err());
        assert!(f.store.get(&claim_key(&f.plan.run_id)).unwrap().is_none());
        let f = fixture(true);
        let native = Native::default();
        search(&f, &native);
        let calls = native.calls.lock().unwrap().len();
        let error = Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap_err();
        assert!(error.to_string().contains("complete finite numeric"));
        assert_eq!(calls, native.calls.lock().unwrap().len());
    }

    #[test]
    fn unspecified_test_is_absent_and_an_interrupted_claim_never_becomes_a_zero_score() {
        let mut f = fixture(false);
        f.request.test = None;
        let native = Native::default();
        search(&f, &native);
        let receipt = Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap();
        assert_eq!(receipt.scores.test, None);
        assert_eq!(receipt.test_scope, None);
        let f = fixture(false);
        search(&f, &native);
        let run = f
            .store
            .get(&RecordKey::Run(f.plan.run_id.clone()))
            .unwrap()
            .unwrap();
        let identity = digest(
            &serde_json::to_vec(&(
                &f.request,
                &run.data["checkpoint"]["plan"],
                &run.data["checkpoint"]["champion"],
            ))
            .unwrap(),
        );
        f.store
            .transaction(|tx| {
                tx.put(
                    &claim_key(&f.plan.run_id),
                    &Record::metadata(json!({"identity":identity,"state":"started"}), vec![]),
                )
            })
            .unwrap();
        let before = native.calls.lock().unwrap().len();
        assert!(Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap_err()
            .to_string()
            .contains("unknown"));
        assert_eq!(before, native.calls.lock().unwrap().len());
        // An operator's explicit Continue is a new authorized attempt, while the
        // interrupted attempt remains durably unknown rather than a zero score.
        let revision = Finalizer::new(Arc::clone(&f.store))
            .journal()
            .snapshot(&f.plan.run_id)
            .unwrap()
            .unwrap()
            .revision
            .0;
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let resumed = finalizer
            .finalize_after_explicit_continue(&f.request, &native)
            .unwrap();
        assert!(resumed.scores.holdout.unwrap().is_finite());
        let history = f
            .store
            .get(&RecordKey::Command {
                principal: "discover.finalization.interrupted.v1".into(),
                idempotency_key: format!("{}:{revision}", f.plan.run_id),
            })
            .unwrap()
            .unwrap();
        assert_eq!(history.data["outcome"], "unknown");
        assert_eq!(history.data["prior"]["data"]["state"], "started");
        let calls = native.calls.lock().unwrap().len();
        assert_eq!(
            finalizer
                .finalize(&f.request, &native)
                .unwrap()
                .model_digest,
            resumed.model_digest
        );
        assert_eq!(calls, native.calls.lock().unwrap().len());
    }

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
            super::super::r#loop::ProcessExecutor::new(&executable, 1, &f.request.scratch_root)
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
}
