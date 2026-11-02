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

    pub(super) fn q1_evaluate(&self, raw: Value) -> Result<Vec<u8>> {
        self.q1_evaluate_using(raw, || {
            ProcessExecutor::new(&self.executable, 1, &self.home.join("worker-scratch"))
        })
    }

    pub(super) fn q1_evaluate_using<E: crate::discover::r#loop::Executor>(
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

    pub(super) fn reserve_q1_request(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
    ) -> Result<Option<Vec<u8>>> {
        let key = q1_command_key(&request.request_sha256);
        if let Some(record) = self.store.get(&key).map_err(io)? {
            return q1_replay_or_unknown(record.data);
        }
        let snapshot = q1_running_snapshot(run_id.clone(), request.budget);
        let result = self.journal.commit(
            RunUpdate {
                expected_revision: None,
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            vec![EventBody::RunState(RunStateBody {
                state: RunState::Running,
                phase: RunPhase::Encoding,
                pending: Some("pinned_training_evaluation".into()),
                reason: None,
            })],
            |tx| {
                if tx.get(&key)?.is_some() {
                    return Err(crate::store::StoreError::Invalid("Q1 request already reserved"));
                }
                tx.put(
                    &key,
                    &crate::store::Record::metadata(
                        json!({"status":"started_outcome_unknown","request_sha256":request.request_sha256,"run_id":run_id}),
                        vec![],
                    ),
                )
            },
        );
        match result {
            Ok(_) => Ok(None),
            Err(_) => match self.store.get(&key).map_err(io)? {
                Some(record) => q1_replay_or_unknown(record.data),
                None => Err(io("Could not durably reserve Q1 request.")),
            },
        }
    }

    pub(super) fn commit_q1_receipt(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
        receipt: EvaluationReceipt,
        receipt_bytes: Vec<u8>,
        completion: Q1ReceiptCompletion,
    ) -> Result<()> {
        let Q1ReceiptCompletion {
            mut attempt_events,
            elapsed,
            terminal_failure,
        } = completion;
        let mut snapshot = self
            .journal
            .snapshot(run_id)
            .map_err(io)?
            .ok_or_else(|| io("Q1 reservation disappeared."))?;
        let expected_revision = snapshot.revision.0;
        let completed = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
            .count();
        let loss = receipt
            .attempts
            .iter()
            .filter_map(|attempt| attempt.loss)
            .min_by(f64::total_cmp);
        snapshot.revision.0 = expected_revision
            .checked_add(1)
            .ok_or_else(|| io("Q1 receipt revision exhausted."))?;
        snapshot.state = if terminal_failure.is_some() || completed == 0 {
            RunState::Failed
        } else {
            RunState::Completed
        };
        snapshot.phase = RunPhase::Finalizing;
        snapshot.pending = None;
        snapshot.budget_used = BudgetUsed {
            evaluations: DecimalU64(receipt.charged),
            wall_seconds: format!("{:.3}", elapsed.as_secs_f64()),
        };
        snapshot.completion_reason = Some(CompletionReason::BudgetEvaluations);
        snapshot.evaluation_receipt = Some(receipt.clone());
        if let Some(reason) = terminal_failure.as_deref() {
            attempt_events.push(EventBody::RunState(RunStateBody {
                state: RunState::Failed,
                phase: RunPhase::Finalizing,
                pending: None,
                reason: Some(reason.to_owned()),
            }));
        }
        attempt_events.push(EventBody::RunFinished(RunFinishedBody {
            completion_reason: CompletionReason::BudgetEvaluations,
            champion: None,
            scores: FinalScores {
                cv: loss,
                holdout: None,
                test: None,
            },
            budget_used: snapshot.budget_used.clone(),
            nondeterminism: Nondeterminism {
                count: DecimalU64(0),
                experiment_ids: vec![],
            },
            evaluation_receipt: Some(Box::new(receipt.clone())),
        }));
        let command_key = q1_command_key(&request.request_sha256);
        let receipt_text = String::from_utf8(receipt_bytes.clone())
            .map_err(|_| io("Serialized Q1 receipt was not UTF-8."))?;
        self.journal.commit(
            RunUpdate {
                expected_revision: Some(expected_revision),
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            attempt_events,
            |tx| {
                let command = tx.get(&command_key)?
                    .ok_or(crate::store::StoreError::Invalid("Q1 reservation missing"))?;
                if command.data["status"] != "started_outcome_unknown" {
                    return Err(crate::store::StoreError::Invalid("Q1 reservation state changed"));
                }
                tx.put(
                    &RecordKey::Evaluation { run: run_id.clone(), number: 0 },
                    &crate::store::Record::metadata(json!({
                        "request_sha256":request.request_sha256,
                        "receipt_bytes":receipt_text,
                        "charged":receipt.charged,
                        "completed":completed,
                        "terminal_failure":terminal_failure,
                        "fold_scores_completed":receipt.attempts.iter()
                            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
                            .map(|attempt| attempt.fold_scores.len()).sum::<usize>(),
                    }), vec![]),
                )?;
                for attempt in &receipt.attempts {
                    tx.put(
                        &RecordKey::Evaluation { run: run_id.clone(), number: attempt.attempt + 1 },
                        &crate::store::Record::metadata(serde_json::to_value(attempt)?, vec![]),
                    )?;
                }
                tx.put(
                    &command_key,
                    &crate::store::Record::metadata(json!({
                        "status":"completed",
                        "request_sha256":request.request_sha256,
                        "run_id":run_id,
                        "receipt_bytes":receipt_text,
                        "terminal_failure":terminal_failure,
                    }), vec![]),
                )
            },
        ).map_err(io)?;
        Ok(())
    }

    pub(super) fn prepare(&self, spec: &RunSpec) -> Result<Prepared> {
        if spec.task != Task::Regression {
            return Err(unavailable("This run backend requires task=regression."));
        }
        if jope_runtime::backend::DeviceRequest::parse(&spec.device).is_none() {
            return Err(invalid("Unknown device."));
        }
        let mut backend =
            BackendPlan::resolve(spec, self.activity.admitted_ranker(), 1).map_err(|error| {
                let code = crate::run::service::error_code(&error);
                // Keep the public catalog code without exposing private backend diagnostics.
                issue(
                    code,
                    format!("{code}: The requested run backend could not be prepared."),
                )
            })?;
        if spec.test_dataset.is_some()
            || spec.split.kind != SplitKind::Random
            || spec.split.group_column.is_some()
            || spec.split.time_column.is_some()
        {
            return Err(unavailable(
                "This run backend supports random FIT/internal assessment splits without a separate test dataset.",
            ));
        }
        let language_filter = run_language_filter(&spec.languages)?;
        let (objective, resolved_objective) = crate::run::service::run_objective(&spec.objective)
            .map_err(|_| {
            invalid("The objective is invalid or requires unsupported ordered/auxiliary inputs.")
        })?;
        if !spec.split.holdout_fraction.is_finite()
            || !(0.0..1.0).contains(&spec.split.holdout_fraction)
            || spec.split.holdout_fraction == 0.0
        {
            return Err(invalid(
                "holdout_fraction must be finite and strictly between zero and one.",
            ));
        }
        let record = self
            .store
            .get(&RecordKey::Dataset(spec.dataset.clone()))
            .map_err(io)?
            .ok_or_else(|| issue(ErrorCode::NotFound, "Registered dataset does not exist."))?;
        let dataset: RegisteredDataset = serde_json::from_value(record.data).map_err(io)?;
        if dataset.id != spec.dataset {
            return Err(io("Dataset identity mismatch."));
        }
        if dataset.csv.bytes > self.config.limits.max_dataset_bytes.value {
            return Err(invalid("Dataset exceeds configured maximum bytes."));
        }
        let fit_estimate =
            ((dataset.rows as f64) * (1.0 - spec.split.holdout_fraction)).floor() as u64;
        if fit_estimate > self.config.limits.rung1_max_rows.value {
            return Err(invalid("FIT exceeds limits.rung1_max_rows."));
        }
        let available = SystemSensors::without_gpu(&self.home)
            .mem_available()
            .map_err(|_| unavailable("Memory admission sensor unavailable."))?;
        if dataset
            .csv
            .bytes
            .saturating_mul(4)
            .saturating_add(WORKER_RAM)
            > available
        {
            return Err(issue(
                ErrorCode::BackendUnavailable,
                "Insufficient available memory for this bounded run.",
            ));
        }
        let source = self.store.read_blob(&dataset.csv).map_err(io)?;
        let mut table = jope_runtime::table::parse_csv(&source, &Default::default()).map_err(io)?;
        let matrix =
            jmat::Matrix::open(self.store.read_blob(&dataset.matrix).map_err(io)?).map_err(io)?;
        let target = table
            .headers
            .iter()
            .position(|v| v == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        if spec.ignore_columns.contains(&spec.target) {
            return Err(invalid("The target cannot be ignored."));
        }
        let ignored: std::collections::BTreeSet<_> = spec.ignore_columns.iter().collect();
        if ignored.len() != spec.ignore_columns.len()
            || ignored.iter().any(|v| !table.headers.contains(v))
        {
            return Err(invalid(
                "Ignored columns must be distinct registered columns.",
            ));
        }
        let selected: Vec<_> = table
            .headers
            .iter()
            .enumerate()
            .filter(|(_, name)| !ignored.contains(name))
            .map(|(i, _)| i)
            .collect();
        if selected.len() < 2
            || selected
                .iter()
                .any(|&i| matrix.columns()[i].kind != jmat::ColumnKind::Numeric)
            || table.rows.iter().any(|row| {
                row[target]
                    .as_deref()
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .is_none_or(|v| !v.is_finite())
            })
        {
            return Err(invalid(
                "At least one numeric feature and a finite numeric target in every row are required.",
            ));
        }
        let features: Vec<_> = selected
            .iter()
            .filter(|&&i| i != target)
            .map(|&i| table.headers[i].clone())
            .collect();
        table.headers = selected.iter().map(|&i| table.headers[i].clone()).collect();
        table.rows = table
            .rows
            .into_iter()
            .map(|row| selected.iter().map(|&i| row[i].clone()).collect())
            .collect();
        let fit_count =
            ((table.rows.len() as f64) * (1.0 - spec.split.holdout_fraction)).floor() as usize;
        let effort = SearchEffort::from_name(match spec.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .unwrap()
        .limits_with_budget(BudgetOverrides {
            evaluations: spec.budget.evaluations.map(|v| v.0),
            wall_seconds: spec.budget.wall_seconds.map(|v| v.0),
            ..Default::default()
        });
        if effort.evaluations == 0
            || effort.wall_seconds == 0
            || fit_count < usize::from(effort.rung1_folds)
            || fit_count > self.config.limits.rung1_max_rows.value as usize
            || fit_count >= table.rows.len()
        {
            return Err(invalid(
                "Budget or FIT row count is outside the backend's configured limits.",
            ));
        }
        let (fit_ids, holdout_ids) = random_split(
            table.rows.len(),
            fit_count,
            spec.seed.0,
            "public-run/fit-v1",
        )
        .map_err(io)?;
        let encode = |ids: &[usize]| -> Result<Vec<u8>> {
            let mut part = table.clone();
            part.rows = ids.iter().map(|&i| table.rows[i].clone()).collect();
            jmat::encode(&part).map_err(io)
        };
        let fit = encode(&fit_ids)?;
        let holdout = encode(&holdout_ids)?;
        let governor = Governor::new(
            std::num::NonZeroU64::new(u64::MAX).unwrap(),
            NonZeroU32::new(1).unwrap(),
            self.config.limits.disk_reserve.value,
        );
        let estimates = governor.preflight(
            Requirements {
                matrix_ram_bytes: dataset.csv.bytes.saturating_mul(4),
                per_worker_ram_bytes: WORKER_RAM,
                matrix_disk_bytes: (fit.len() + holdout.len()) as u64,
                artifact_bytes: MODEL_LIMIT,
                ..Default::default()
            },
            &mut SystemSensors::without_gpu(&self.home),
        )?;
        let timeout = self
            .config
            .limits
            .eval_timeout
            .value
            .as_millis()
            .min(u128::from(effort.wall_seconds) * 1000)
            .max(1) as u64;
        let mut normalized = serde_json::to_value(spec).map_err(io)?;
        normalized["dry_run"] = json!(false);
        normalized["plan_digest"] = Value::Null;
        let plan_digest = hash(
            &serde_json::to_vec(
                &json!({"backend":"cpu-ridge-seeded-rmse/v1", "spec":normalized,
            "dataset":dataset.csv, "fit":hash(&fit), "binary":self.binary, "timeout_ms":timeout,
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates,
            "ranker_bundle":backend.bundle_digest,"placement":backend.placement}),
            )
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let mut public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"regression","inferred":false,"reason":"Explicit regression task."},
            "objective":resolved_objective,
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effort.rung1_folds,"rounds_in_flight":1},
            "guidance":backend.guidance,"landmarks":{"mode":spec.landmarks,"sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":language_filter.excluded_ops(),"intersection_empty":false},
            "placement":backend.placement,
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":backend.warnings})).map_err(io)?;
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: backend.bundle_digest.clone(),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective,
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: effort.evaluations,
            wall_seconds: effort.wall_seconds,
            evaluation_wall_ms: timeout,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(&fit, &search)
            .map_err(|_| invalid("Unable to prepare numeric FIT landmarks."))?;
        public.landmarks.sample_rows = backend.landmark_rows as u32;
        let rows = |ids: Vec<usize>| {
            ids.into_iter()
                .map(|row| AuditRowId {
                    dataset: spec.dataset.clone(),
                    row: row as u64,
                })
                .collect()
        };
        Ok(Prepared {
            public,
            search,
            fit,
            holdout,
            fit_rows: rows(fit_ids),
            holdout_rows: rows(holdout_ids),
            backend,
        })
    }
}
