use super::*;
pub(super) fn fixture(selected_task: Task, improved: usize) -> Fixture {
    fixture_with(selected_task, improved, Some(context_plan()))
}

pub(super) fn fixture_with(
    selected_task: Task,
    improved: usize,
    context: Option<ContextPlan>,
) -> Fixture {
    fixture_spec(selected_task, context, None, &exact_on(improved))
}

pub(super) fn teaching_fixture_root(requested: Option<&Path>) -> anyhow::Result<PathBuf> {
    let root = match requested {
        Some(root) => {
            anyhow::ensure!(root.is_absolute(), "teaching fixture root must be absolute");
            root.to_path_buf()
        }
        None => {
            let root =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp/teaching-fixtures");
            fs::create_dir_all(&root)?;
            root
        }
    };
    let root = fs::canonicalize(root)?;
    anyhow::ensure!(root.is_dir(), "teaching fixture root must be a directory");
    anyhow::ensure!(
        !root.starts_with("/tmp"),
        "teaching fixture root must be outside /tmp"
    );
    Ok(root)
}

#[test]
fn teaching_fixture_root_preserves_default_and_refuses_invalid_overrides() {
    let default = teaching_fixture_root(None).unwrap();
    assert_eq!(
        default,
        fs::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp/teaching-fixtures")
        )
        .unwrap()
    );
    let directory = tempfile::tempdir_in(&default).unwrap();
    assert_eq!(
        teaching_fixture_root(Some(directory.path())).unwrap(),
        fs::canonicalize(directory.path()).unwrap()
    );
    for invalid in [
        PathBuf::from("relative/teaching"),
        PathBuf::from("/tmp"),
        PathBuf::from("/tmp/../tmp"),
        directory.path().join("missing"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
    ] {
        assert!(
            teaching_fixture_root(Some(&invalid)).is_err(),
            "{invalid:?}"
        );
    }
}

// All inputs/predictions are synthetic; these tests establish admission mechanics,
// not corpus admission, real fitted-model accuracy, or a completed live search.
// `fold_targets` overrides the balanced default; `predict` maps (fold, held rows,
// train-only baseline) to the candidate's retained predictions.
pub(super) fn fixture_spec(
    selected_task: Task,
    context: Option<ContextPlan>,
    fold_targets: Option<Vec<Vec<String>>>,
    predict: Predict,
) -> Fixture {
    // Set once by the invoking test process; malformed explicit roots never fall back.
    // This relocates only synthetic fixtures, not production authority or fit inputs.
    let requested = std::env::var_os("JOPE_TEACHING_TEST_ROOT");
    let fixture_root = teaching_fixture_root(requested.as_deref().map(Path::new)).unwrap();
    let directory = tempfile::tempdir_in(fixture_root).unwrap();
    let state = State::at(directory.path());
    let classes: Vec<String> = match selected_task {
        Task::Regression => vec![],
        Task::Binary => vec!["negative".into(), "positive".into()],
        Task::Multiclass => vec!["α".into(), "001".into(), "class-C".into()],
        _ => unreachable!(),
    };
    let comparison = HwmIdentity {
        task: selected_task.as_str().into(),
        dataset_hash: "d".repeat(64),
        cut_id: "c".repeat(64),
        evaluator_sha256: "e".repeat(64),
        vocabulary_sha256: "a".repeat(64),
        protocol: "synthetic-train-oof-v1".into(),
        comparison: Some(ComparisonContext {
            numerical_policy: NUMERICAL_POLICY.into(),
            stage: "train_oof".into(),
            aggregation: "mean_fold_loss".into(),
            label_role: "development".into(),
            sealed: false,
            benchmark_valid: false,
            recipe_encoding: "native-canonical-dsl-sha256/v1".into(),
            class_order: classes.clone(),
            checkpoint_sha256: None,
        }),
    };
    let fold_targets = fold_targets.unwrap_or_else(|| {
        vec![
            if classes.is_empty() {
                vec!["-1".to_string(), "1".to_string()]
            } else {
                classes.clone()
            };
            5
        ]
    });
    assert_eq!(fold_targets.len(), 5);
    let rows = fold_targets
        .iter()
        .enumerate()
        .flat_map(|(fold, targets)| {
            targets
                .iter()
                .enumerate()
                .map(move |(index, target)| TargetRow {
                    id: format!("{:064x}", fold * 100 + index),
                    group: format!("{:064x}", 1000 + fold * 100 + index),
                    fold: fold as u8,
                    target: target.clone(),
                })
        })
        .collect::<Vec<_>>();
    let prepared_rows = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            json!({
                "id":row.id, "source_index":index, "group":row.group, "fold":row.fold,
                "target":row.target, "values":[index.to_string()]
            })
        })
        .collect::<Vec<_>>();
    let table = json!({"schema":"jope.prepared-table/v1", "dataset":"fixture", "task":selected_task,
            "source_sha256":comparison.dataset_hash, "partition":"development",
            "features":[{"name":"x", "kind":"numeric"}], "classes":classes, "rows":prepared_rows});
    let development_sha = state.put_blob(&canonical_json(&table).unwrap()).unwrap();
    let dataset = json!({"id":"fixture", "task":selected_task,
            "metric":jope_pipeline::experiment::Metric::for_task(selected_task).unwrap(),
            "source":{"url":"synthetic", "sha256":comparison.dataset_hash, "bytes":1,
                "metadata_url":"synthetic", "metadata_sha256":"9".repeat(64)},
            "rows":rows.len()+1, "features":[{"name":"x", "kind":"numeric"}], "classes":classes,
            "group_columns":[], "split":{"protocol":"synthetic", "sha256":comparison.cut_id,
                "development_sha256":development_sha, "sealed_sha256":"8".repeat(64),
                "development_rows":rows.len(), "sealed_rows":1,
                "fold_rows":fold_targets.iter().map(Vec::len).collect::<Vec<_>>()},
            "license":{"original":"synthetic", "evidence_url":"synthetic", "repository":"synthetic", "note":"synthetic"},
            "checkpoint_exposure":"synthetic", "description":"synthetic", "group_limitations":"synthetic"});
    let mut datasets = Vec::new();
    for (kind_index, kind) in [Task::Regression, Task::Binary, Task::Multiclass]
        .into_iter()
        .enumerate()
    {
        for index in 0..10 {
            let mut value = dataset.clone();
            if !(kind == selected_task && index == 0) {
                value["id"] = json!(format!("synthetic-{kind_index}-{index}"));
                value["task"] = json!(kind);
                value["metric"] = json!(jope_pipeline::experiment::Metric::for_task(kind).unwrap());
                value["source"]["sha256"] =
                    json!(format!("{:064x}", kind_index * 10 + index + 100));
                value["classes"] = match kind {
                    Task::Regression => json!([]),
                    Task::Binary => json!(["0", "1"]),
                    _ => json!(["0", "1", "2"]),
                };
            }
            datasets.push(value);
        }
    }
    let panel = json!({"schema":"jope.benchmark-panel/v1", "panel_id":"synthetic",
            "repository":"synthetic", "revision":"a".repeat(40), "seed":7,
            "selection":"explicit synthetic qualification fixture", "folds":5, "datasets":datasets});
    let plan = FreezePlan {
        comparison: comparison.clone(),
        original_label_role: "test".into(),
        panel_sha256: state.put_blob(&canonical_json(&panel).unwrap()).unwrap(),
        dataset_id: "fixture".into(),
        evaluation_seed: 7,
        context,
    };
    let freeze = freeze_optimization(&state, &plan).unwrap();
    let frozen = load_freeze(&state, &freeze).unwrap();
    assert_eq!(frozen.label_role, "development");
    let experiment: Experiment = serde_json::from_value(json!({
            "schema":"jope.experiment/v1", "panel_sha256":plan.panel_sha256,
            "methods_sha256":"7".repeat(64), "source_sha":"a".repeat(40),
            "checkpoints":{"teaching_freeze":freeze.freeze_sha256,"teaching_vocabulary":comparison.vocabulary_sha256}, "seed":7,
            "resources":{"workers":1,"cpus":1,"memory_bytes":1048576,"swap_bytes":0,
                "gpu_tasks":0,"threads_per_worker":1,"total_seconds":60,"reporting_seconds":5,"per_fit_seconds":30},
            "expected_artifacts":["synthetic"],"protocol":"synthetic-train-oof-v1", "numerical_policy":NUMERICAL_POLICY
        })).unwrap();
    let window = ExecutionWindow {
        experiment_sha256: experiment.identity().unwrap(),
        started_unix_ms: frozen.created_at_unix * 1000,
        deadline_unix_ms: frozen.created_at_unix * 1000 + 60000,
    };
    let base = OofEvidence {
        freeze,
        experiment_blob_sha256: state
            .put_blob(&canonical_json(&experiment).unwrap())
            .unwrap(),
        window_sha256: state.put_blob(&canonical_json(&window).unwrap()).unwrap(),
        fold_predictions_sha256: Vec::new(),
        inner: None,
        context: Some(handed_off(
            &state,
            valid_context(&state, &comparison, &rows),
            &comparison,
            &experiment,
            &window,
        )),
    };
    let (record, evidence) = candidate_in(
        &state,
        &comparison,
        &rows,
        selected_task,
        &classes,
        &experiment,
        &window,
        frozen.created_at_unix,
        base,
        "",
        predict,
    );
    Fixture {
        _directory: directory,
        state,
        plan,
        record,
        evidence,
        rows,
        task: selected_task,
        classes,
        experiment,
        window,
        created_at: frozen.created_at_unix,
    }
}

/// One complete synthetic candidate: per-fold fitted state, request, job, worker receipt,
/// result and predictions, plus the record claiming their mean fold score. An empty `tag`
/// keeps the original fixture's program, job directories and record blob.
#[allow(clippy::too_many_arguments)]
pub(super) fn candidate_in(
    state: &State,
    comparison: &HwmIdentity,
    rows: &[TargetRow],
    selected_task: Task,
    classes: &[String],
    experiment: &Experiment,
    window: &ExecutionWindow,
    created_at: u64,
    mut evidence: OofEvidence,
    tag: &str,
    predict: Predict,
) -> (HwmRecord, OofEvidence) {
    let suffix = if tag.is_empty() {
        String::new()
    } else {
        format!("-{tag}")
    };
    let dsl = format!("(pipe X (synthetic-fixture{suffix}))");
    let program = sha256_hex(dsl.as_bytes());
    let classes = classes.to_vec();
    let mut all_targets = Vec::new();
    let mut all_predictions = Vec::new();
    let mut scores = Vec::new();
    for fold in 0..5 {
        let (train, hold) = fold_rows(rows, fold);
        let predictions = predict(
            fold,
            &hold,
            baseline(selected_task, &classes, &train, hold.len()).unwrap(),
        );
        // Each fold's fitted state is its own subject, never shared across folds.
        let fitted_bytes = canonical_json(&json!({"schema":"jope.panel-fitted-rust/v1",
                "method":{"dsl":dsl}, "model":{"genome_sha256":program}, "fold":fold,
                "fixture_note":"synthetic artifact consistency test, not a live fit"}))
        .unwrap();
        let fitted = state.put_blob(&fitted_bytes).unwrap();
        let labels: Vec<_> = hold.iter().map(|row| row.target.clone()).collect();
        let score = score_predictions(selected_task, &classes, &labels, &predictions).unwrap();
        let predictions_sha256 = state
            .put_blob(&serde_json::to_vec(&predictions).unwrap())
            .unwrap();
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: experiment.identity().unwrap(),
            dataset_sha256: comparison.dataset_hash.clone(),
            split_sha256: comparison.cut_id.clone(),
            fitting_rows_sha256: rows_sha(&train).unwrap(),
            scoring_rows_sha256: rows_sha(&hold).unwrap(),
            fitting_rows: train.len(),
            scoring_rows: hold.len(),
            recipe_sha256: program.clone(),
            evaluator_sha256: comparison.evaluator_sha256.clone(),
            checkpoint_sha256: None,
            task: selected_task,
            metric: score.metric,
            classes: classes.clone(),
            fold,
            stage: EvaluationStage::TrainOof,
            seed: 7,
            numerical_policy: NUMERICAL_POLICY.into(),
        };
        let job_directory = state.root().join(format!("fixture-worker{suffix}-{fold}"));
        fs::create_dir(&job_directory).unwrap();
        let input_path = job_directory.join("input.json");
        fs::write(&input_path, canonical_json(&request).unwrap()).unwrap();
        let output_path = job_directory.join("predictions.json");
        fs::write(&output_path, serde_json::to_vec(&predictions).unwrap()).unwrap();
        let fitted_path = job_directory.join("fitted.bin");
        fs::write(&fitted_path, &fitted_bytes).unwrap();
        let fixture_program = Artifact {
            path: input_path.clone(),
            sha256: sha256_hex(&fs::read(&input_path).unwrap()),
        };
        let job = WorkerJob {
            request: request.clone(),
            directory: job_directory,
            outputs: vec!["predictions.json".into(), "fitted.bin".into()],
            arm: None,
            command: WorkerCommand {
                program: fixture_program.clone(),
                args: vec![],
                inputs: vec![fixture_program],
                environment: BTreeMap::new(),
            },
        };
        let outcome = WorkerOutcome {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: job.identity().unwrap(),
            request_sha256: request.identity().unwrap(),
            command_status: EvaluationStatus::Completed,
            reason: None,
            exit_code: Some(0),
            signal: None,
            outputs: vec![
                Artifact {
                    path: output_path,
                    sha256: predictions_sha256.clone(),
                },
                Artifact {
                    path: fitted_path,
                    sha256: fitted.clone(),
                },
            ],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                wall_seconds: Some(0.01),
                cpu_seconds: None,
                accounting_reason: "synthetic fixture, CPU unmeasured".into(),
                ..Default::default()
            },
        };
        let result = EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            worker_receipt_sha256: state.put_blob(&canonical_json(&outcome).unwrap()).unwrap(),
            reused_from_request_sha256: None,
            request_sha256: request.identity().unwrap(),
            status: EvaluationStatus::Completed,
            reason: None,
            primary: Some(score.value),
            secondary: BTreeMap::new(),
            scored_rows: hold.len(),
            predictions_sha256: Some(predictions_sha256.clone()),
            fitted_state_sha256: Some(fitted.clone()),
            consumption: Consumption {
                wall_seconds: Some(0.01),
                cpu_seconds: None,
                accounting_reason: "synthetic fixture, CPU unmeasured".into(),
                ..Default::default()
            },
        };
        let prediction = FoldPredictions {
            schema: PREDICTIONS.into(),
            comparison_sha256: comparison.sha256().unwrap(),
            program_sha256: program.clone(),
            fold,
            training_rows_sha256: request.fitting_rows_sha256.clone(),
            scoring_rows_sha256: request.scoring_rows_sha256.clone(),
            row_ids: hold.iter().map(|row| row.id.clone()).collect(),
            predictions_sha256,
            evaluation_request_sha256: state.put_blob(&canonical_json(&request).unwrap()).unwrap(),
            worker_job_sha256: state.put_blob(&canonical_json(&job).unwrap()).unwrap(),
            fitted_artifact_sha256: fitted.clone(),
            execution_receipt_sha256: state.put_blob(&canonical_json(&result).unwrap()).unwrap(),
            started_receipt_sha256: state
                .put_blob(
                    &canonical_json(&json!({"job":job,
                    "job_sha256":job.identity().unwrap(),"started_unix_ms":window.started_unix_ms,
                    "fit_deadline_unix_ms":window.started_unix_ms+30000,"cgroup":"synthetic"}))
                    .unwrap(),
                )
                .unwrap(),
            fit_started_unix: created_at,
        };
        evidence.fold_predictions_sha256.push(
            state
                .put_blob(&canonical_json(&prediction).unwrap())
                .unwrap(),
        );
        all_targets.extend(labels);
        all_predictions.push(predictions);
        scores.push(score.value);
    }
    score_predictions(
        selected_task,
        &classes,
        &all_targets,
        &combine(&all_predictions).unwrap(),
    )
    .unwrap();
    let metric = match selected_task {
        Task::Regression => Metric::Rmse,
        Task::Binary => Metric::Auc,
        _ => Metric::Logloss,
    };
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: comparison.sha256().unwrap(),
        identity: comparison.clone(),
        legacy_identity_sha256: None,
        genome: Genome {
            dsl,
            program_sha256: program,
            semantic_sha256: None,
        },
        score: Score {
            metric,
            value: scores.iter().sum::<f64>() / 5.0,
            direction: metric.direction(),
            units: if metric == Metric::Rmse {
                "original_target_units"
            } else {
                "dimensionless"
            }
            .into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: "e".repeat(64),
            version: "synthetic".into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: true,
        provenance: Provenance {
            source: "synthetic".into(),
            run_id: "fixture".into(),
            git_sha: "unbound".into(),
            bundle_id: "unbound".into(),
            record_sha256: state.put_blob(&candidate_blob(tag)).unwrap(),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: now_unix(),
    };
    (record, evidence)
}

pub(super) fn candidate_blob(tag: &str) -> Vec<u8> {
    if tag.is_empty() {
        b"synthetic candidate".to_vec()
    } else {
        format!("synthetic candidate {tag}").into_bytes()
    }
}
