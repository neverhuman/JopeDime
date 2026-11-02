use super::*;

impl RunService {
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }
    pub(crate) fn new(
        store: Arc<Store>,
        config: Config,
        home: PathBuf,
        executable: PathBuf,
        activity: Arc<EngineActivity>,
        lease: Arc<EngineLock>,
    ) -> anyhow::Result<Self> {
        let binary = format!(
            "{}:{}",
            env!("CARGO_PKG_VERSION"),
            hash(&std::fs::read(&executable)?)
        );
        Ok(Self {
            journal: Arc::new(EventLog::new(Arc::clone(&store))),
            store,
            config,
            home,
            executable,
            binary,
            activity,
            admission: Mutex::new(()),
            _lease: lease,
        })
    }

    pub(crate) fn q1_evaluate(&self, raw: Value) -> Result<Vec<u8>> {
        self.q1_evaluate_using(raw, || {
            ProcessExecutor::new(&self.executable, 1, &self.home.join("worker-scratch"))
        })
    }

    pub(crate) fn q1_evaluate_using<E: crate::discover::r#loop::Executor>(
        &self,
        raw: Value,
        make_executor: impl FnOnce() -> anyhow::Result<E>,
    ) -> Result<Vec<u8>> {
        let started_at = std::time::Instant::now();
        let request: Q1Request = serde_json::from_value(raw.clone())
            .map_err(|_| invalid("Expected a closed jope.q1-request/v1 object."))?;
        validate_q1_request(&request, &raw)?;
        let admission = self
            .admission
            .lock()
            .map_err(|_| io("Run admission lock is poisoned."))?;
        if let Some(record) = self
            .store
            .get(&q1_command_key(&request.request_sha256))
            .map_err(io)?
        {
            return q1_replay_or_unknown(record.data)?
                .ok_or_else(|| io("Q1 replay bytes missing."));
        }
        if self.activity.active_runs() != 0 {
            return Err(unavailable("The bounded evaluation backend is busy."));
        }
        // This spec only selects the trusted backend. No public-run split is
        // performed: the Q1 matrix and every evaluation fold come from the
        // independently pinned protocol manifest below.
        let backend_spec = RunSpec {
            dataset: format!("ds_{}", &request.dataset.dataset_sha256[..24])
                .parse()
                .map_err(io)?,
            target: String::new(),
            task: Task::Regression,
            ignore_columns: vec![],
            test_dataset: None,
            split: SplitSpec {
                kind: SplitKind::Random,
                holdout_fraction: 0.5,
                group_column: None,
                time_column: None,
            },
            objective: ObjectiveSpec::Name("rmse".into()),
            effort: Effort::High,
            budget: Budget {
                evaluations: Some(DecimalU64(request.budget)),
                wall_seconds: Some(DecimalU64(600)),
            },
            landmarks: LandmarkMode::Full,
            guidance: if request.guidance == "learned" {
                Guidance::Learned
            } else {
                Guidance::Heuristic
            },
            languages: vec![Language::Rust],
            require_all_languages: true,
            device: "cpu".into(),
            seed: DecimalU64(request.seed),
            dry_run: false,
            plan_digest: None,
        };
        let mut backend = BackendPlan::resolve(&backend_spec, self.activity.admitted_ranker(), 1)
            .map_err(|error| {
            issue(
                crate::run::service::error_code(&error),
                "The requested Q1 backend could not be prepared.",
            )
        })?;
        if backend.learned()
            && (backend.bundle_digest != request.pins["bundle_sha256"]
                || request.pins["dope_sha256"]
                    != "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814")
        {
            return Err(invalid(
                "Q1 bundle/DOPE pins differ from the admitted evaluator.",
            ));
        }

        let protocol_sha = &request.pins["protocol_sha256"];
        let manifest = load_q1_manifest(&self.home, protocol_sha)?;
        if manifest.schema != "jope.q1-protocol-manifest/v1"
            || manifest.protocol_sha256 != *protocol_sha
            || manifest.source_commit != request.source_commit
            || manifest.plan_sha256 != request.plan_sha256
            || manifest.pins != request.pins
        {
            return Err(invalid("Q1 protocol manifest identity mismatch."));
        }
        let binding = manifest
            .datasets
            .get(&request.dataset.id)
            .ok_or_else(|| unavailable("Dataset is absent from the pinned protocol manifest."))?;
        validate_q1_binding(binding, &request.dataset)?;
        validate_q1_folds(binding)?;
        if binding.train_blob_bytes > self.config.limits.max_dataset_bytes.value
            || binding.train_rows > self.config.limits.rung1_max_rows.value
        {
            return Err(unavailable(
                "Pinned Q1 training partition exceeds the local engine limits.",
            ));
        }
        let available = SystemSensors::without_gpu(&self.home)
            .mem_available()
            .map_err(|_| unavailable("Memory admission sensor unavailable."))?;
        if binding
            .train_blob_bytes
            .saturating_mul(4)
            .saturating_add(WORKER_RAM)
            > available
        {
            return Err(unavailable(
                "Insufficient available memory for the pinned Q1 training partition.",
            ));
        }

        let train_sha = parse_digest(&binding.train_blob_sha256)
            .ok_or_else(|| invalid("Pinned training blob hash is malformed."))?;
        let train_blob = crate::store::BlobRef {
            sha256: train_sha,
            bytes: binding.train_blob_bytes,
        };
        let train_csv = self.store.read_blob(&train_blob).map_err(io)?;
        let table = q1_fit::parse(&self.home, binding, &train_csv).map_err(|_| {
            invalid("Pinned training partition CSV or native import metadata mismatch.")
        })?;
        if table.rows.len() as u64 != binding.train_rows
            || binding.feature_columns.len() as u64 != request.dataset.features
            || table.headers.len() != binding.feature_columns.len() + 1
            || !table.headers.contains(&binding.target)
            || binding
                .feature_columns
                .iter()
                .any(|column| !table.headers.contains(column))
        {
            return Err(invalid(
                "Pinned training partition shape or columns mismatch.",
            ));
        }
        let matrix_bytes = jmat::encode(&table)
            .map_err(|_| invalid("Pinned training partition cannot be encoded."))?;
        let matrix_sha = hash(&matrix_bytes);
        let matrix = jmat::Matrix::open(matrix_bytes)
            .map_err(|_| invalid("Pinned training matrix is invalid."))?;
        let run_id = jope_api::ids::RunId::new();
        let matrix_blob = self.store.publish_blob(matrix.bytes()).map_err(io)?;
        // High patience cannot stop a 24-admission policy early; evaluations
        // and wall time remain explicitly capped, and every worker uses the pinned folds.
        let search_plan = SearchPlan {
            run_id: jope_api::ids::RunId::new(),
            dataset_id: backend_spec.dataset,
            problem_digest: request.dataset.dataset_sha256.clone(),
            bundle_digest: request.pins["bundle_sha256"].clone(),
            binary_version: self.binary.clone(),
            matrix_path: self.store.verified_blob_path(&matrix_blob).map_err(io)?,
            matrix_sha256: matrix_sha.clone(),
            target: binding.target.clone(),
            features: binding.feature_columns.clone(),
            objective: "rmse".into(),
            seed: request.seed,
            planned_workers: 1,
            effort: Effort::High,
            evaluations: request.budget,
            wall_seconds: 600,
            evaluation_wall_ms: 600_000,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(matrix.bytes(), &search_plan)
            .map_err(|error| {
                issue(
                    crate::run::service::error_code(&error),
                    "Pinned Q1 FIT binding failed.",
                )
            })?;
        let inner = make_executor().map_err(io)?;
        let search_root = self.home.join("q1/search");
        match std::fs::create_dir(&search_root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !std::fs::symlink_metadata(&search_root)
                    .map_err(io)?
                    .file_type()
                    .is_dir()
                {
                    return Err(invalid("Q1 search root is not a real directory."));
                }
            }
            Err(error) => return Err(io(error)),
        }
        // Give each pre-reservation setup attempt its own private journal
        // directory. A crash before request reservation must not poison a
        // later retry with an already-created directory.
        let search_home =
            search_root.join(format!("{}-{}", request.request_sha256, search_plan.run_id));
        std::fs::create_dir(&search_home).map_err(io)?;
        let search_store = Arc::new(Store::open(search_home.join("store")).map_err(io)?);
        let mut discovery = Discovery::open(search_store, search_plan).map_err(io)?;
        let started = self.reserve_q1_request(&request, &run_id)?;
        if let Some(bytes) = started {
            return Ok(bytes);
        }

        let _activity = self.activity.begin_run();
        drop(admission);
        let executor = Q1PinnedExecutor {
            inner,
            folds: json!({"version":1,"matrix_sha256":matrix_sha,"target":binding.target,
                "features":binding.feature_columns,"folds":binding.folds}),
            started_at,
            attempts: Mutex::new(vec![]),
        };
        // Match the proposal construction and numerical recipe across arms.
        // The verified backend chooses only the admission policy.
        let space = BackendSpace {
            native: NativeSpace,
            learned: true,
        };
        let mut terminal_failure = None;
        while discovery.completed().is_none() {
            if let Err(error) =
                discovery.advance(&space, &backend.admission, &executor, 1, usize::MAX)
            {
                if discovery.outcome_unknown() {
                    return Err(io(error));
                }
                terminal_failure = Some(format!(
                    "Q1 search stopped before completing its admitted budget: {error:#}"
                ));
                break;
            }
        }
        let used = discovery.snapshot().budget_used.evaluations.0;
        let attempts: Vec<_> = executor
            .attempts
            .into_inner()
            .map_err(|_| io("Q1 inventory poisoned."))?
            .into_iter()
            .map(|(_, receipt)| receipt)
            .collect();
        if attempts.len() as u64 != used || used > request.budget {
            terminal_failure.get_or_insert_with(|| {
                format!(
                    "Q1 worker inventory mismatch: {} actual worker attempts for {used} committed admissions (budget {}).",
                    attempts.len(), request.budget
                )
            });
        }
        let mut attempt_events = Vec::with_capacity(attempts.len() * 2);
        for attempt in &attempts {
            attempt_events.push(EventBody::ExperimentStarted(ExperimentStartedBody {
                experiment_id: format!("q1_{}", attempt.attempt),
                hypothesis: "Pinned Q1 evaluation attempt".into(),
            }));
            attempt_events.push(EventBody::ExperimentFinished(ExperimentFinishedBody {
                experiment_id: format!("q1_{}", attempt.attempt),
                outcome: if attempt.outcome == EvaluationAttemptOutcome::Completed {
                    ExperimentOutcome::NotImproved
                } else {
                    ExperimentOutcome::Failed
                },
                metric: request.metric.clone(),
                value: attempt.loss,
                failure: attempt.failure.as_ref().map(|failure| PublicFailure {
                    code: ErrorCode::NumericalFailure,
                    reason: format!("{}: {}", failure.code, failure.reason),
                }),
                attempt_receipt: Some(Box::new(attempt.clone())),
            }));
        }

        let charged = attempts.len() as u64;
        let champion = attempts
            .iter()
            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
            .min_by(|left, right| left.loss.unwrap().total_cmp(&right.loss.unwrap()));
        let champion_program = champion.and_then(|attempt| attempt.program_sha256.clone());
        let receipt = EvaluationReceipt {
            schema: "jope.q1-evaluation-receipt/v1".into(),
            request_sha256: request.request_sha256.clone(),
            identity: EvaluationIdentity {
                source_commit: request.source_commit.clone(),
                plan_sha256: request.plan_sha256.clone(),
                pins: request.pins.clone(),
                dataset_id: request.dataset.id.clone(),
                dataset_sha256: request.dataset.dataset_sha256.clone(),
                cut_sha256: request.dataset.cut_sha256.clone(),
                guidance: request.guidance.clone(),
                metric: request.metric.clone(),
                seed: request.seed,
                requested_budget: request.budget,
                dataset_binding_sha256: binding.dataset_binding_sha256.clone(),
            },
            charged,
            attempts,
            champion_program,
            adapter: format!("jope-service/{}", self.binary),
        };
        let receipt_bytes = serde_json::to_vec(&receipt).map_err(io)?;
        self.commit_q1_receipt(
            &request,
            &run_id,
            receipt,
            receipt_bytes.clone(),
            Q1ReceiptCompletion {
                attempt_events,
                elapsed: started_at.elapsed(),
                terminal_failure,
            },
        )?;
        Ok(receipt_bytes)
    }
}
