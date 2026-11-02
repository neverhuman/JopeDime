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

    mod partition_and_replay;
    mod worker_publication;
}
