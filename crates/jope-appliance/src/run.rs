use super::*;

/// State shared by the fitting and prediction commands, so a failure still yields a receipt.
pub(super) struct Run {
    command: &'static str,
    started: Instant,
    common: CommonArgs,
    pub(super) receipt: Receipt,
    encoding_package: Option<Package>,
    /// The run's budget once prepared, so a failed run still records what it consumed.
    budget: Option<Budget>,
    /// Whether `--receipt` names no input or output (checked before any work).
    pub(super) receipt_allowed: bool,
}

impl Run {
    pub(super) fn new(command: &'static str, common: &CommonArgs) -> Self {
        Self {
            command,
            started: Instant::now(),
            common: common.clone(),
            budget: None,
            receipt_allowed: false,
            encoding_package: None,
            receipt: Receipt {
                schema: jope_runtime::receipt::SCHEMA,
                command: command.to_owned(),
                product_version: VERSION,
                runtime_abi: RUNTIME_ABI,
                source_sha40: bootstrap::source_sha40(),
                package: None,
                device: backend::DeviceChoice {
                    requested: common.device.clone(),
                    selected: "none",
                    reason: "not selected".into(),
                },
                task: None,
                method: None,
                inputs: Vec::new(),
                outputs: Vec::new(),
                ignored_columns: Vec::new(),
                budget: Budget::new(None, 1).record(),
                timings: Timings::default(),
                selection: None,
                landmark_encoding: None,
                outcome: "failed",
                error: None,
            },
        }
    }

    fn limits(&self) -> Limits {
        Limits {
            max_input_bytes: self.common.max_input_mib.saturating_mul(1 << 20),
            max_rows: self.common.max_rows,
            ..Limits::default()
        }
    }

    fn budget(&self) -> anyhow::Result<Budget> {
        let limit = match self.common.time_limit_seconds {
            None => None,
            // Negative, NaN, infinite or too large for a Duration: an input error, never a panic.
            Some(s) => Some(Duration::try_from_secs_f64(s).map_err(|_| {
                Failure::input("--time-limit-seconds must be a non-negative number of seconds")
            })?),
        };
        Ok(Budget::new(limit, self.common.threads))
    }

    /// Authenticate the package and select the device: the preamble of every run.
    pub(super) fn prepare(&mut self, landmark_encoding: bool) -> anyhow::Result<Budget> {
        let request = DeviceRequest::parse(&self.common.device)
            .ok_or_else(|| Failure::input(format!("unknown device {:?}", self.common.device)))?;
        let package = bootstrap::open()?;
        if let Some(p) = &package {
            let s = bootstrap::summary(p);
            self.receipt.package = Some(PackageRecord {
                profile: s.profile,
                scope: s.scope,
                manifest_sha256: s.manifest_sha256,
                signer: s.signer,
                signer_role: s.signer_role,
            });
        }
        self.receipt.device = if landmark_encoding {
            if !cfg!(feature = "dope-encoding") {
                return Err(Failure::backend(
                    "landmark encoding requires a dope-cuda build",
                ));
            }
            backend::DeviceChoice {
                requested: request.as_string(),
                selected: "none",
                reason: "Dope initialization pending; native fitting uses CPU".into(),
            }
        } else {
            // Only device selection establishes that a backend is unavailable.
            // Preserve all other backend failures' own messages in finish().
            backend::select(request).map_err(|error| {
                Failure::backend(format!(
                    "{}: {error:#}",
                    jope_api::error::ErrorCode::BackendUnavailable
                ))
            })?
        };
        let budget = self.budget()?;
        self.receipt.budget = budget.record();
        self.budget = Some(budget.clone());
        self.encoding_package = package;
        Ok(budget)
    }

    /// Refuse a receipt or output path that names an input, before any work. A receipt that
    /// would overwrite an input or the output is never written.
    pub(super) fn refuse_aliasing(
        &mut self,
        output: &Path,
        inputs: &[&Path],
    ) -> anyhow::Result<()> {
        self.receipt_allowed = false;
        if let Some(receipt) = self.common.receipt.clone() {
            let mut protected = inputs.to_vec();
            protected.push(output);
            io::refuse_aliasing(&receipt, &protected)?;
        }
        self.receipt_allowed = true;
        io::refuse_aliasing(output, inputs)
    }

    fn created_by(&self) -> CreatedBy {
        CreatedBy {
            source_sha40: bootstrap::source_sha40(),
            package_manifest_sha256: self
                .receipt
                .package
                .as_ref()
                .map(|p| p.manifest_sha256.clone()),
            package_profile: self.receipt.package.as_ref().map(|p| p.profile.clone()),
        }
    }

    fn read_csv(&mut self, role: &'static str, path: &Path) -> anyhow::Result<table::CsvTable> {
        let started = Instant::now();
        let csv = table::read_csv(path, &self.limits())?;
        self.receipt.inputs.push(FileRecord {
            role,
            path: path.display().to_string(),
            bytes: csv.bytes,
            sha256: csv.sha256.clone(),
        });
        self.receipt.timings.read_seconds += started.elapsed().as_secs_f64();
        Ok(csv)
    }

    pub(super) fn fit(
        &mut self,
        train: &TrainArgs,
        budget: &Budget,
    ) -> anyhow::Result<FittedModel> {
        let choice = TaskChoice::parse(&train.task)
            .ok_or_else(|| Failure::input(format!("unknown task {:?}", train.task)))?;
        let fitness = train
            .fitness
            .as_deref()
            .map(load_fitness)
            .transpose()
            .map_err(|e| Failure::input(format!("fitness: {e:#}")))?;
        let calibration = jope_runtime::calibration::Calibration::parse(&train.calibration)
            .ok_or_else(|| {
                Failure::input(format!("unknown calibration {:?}", train.calibration))
            })?;
        let csv = self.read_csv("train", &train.train)?;
        let auxiliary = fitness
            .as_ref()
            .map(jope_runtime::fitness::Fitness::columns)
            .unwrap_or_default();
        let training = table::training_with_auxiliary(
            &csv,
            &train.target,
            choice,
            &self.limits(),
            &auxiliary,
        )?;
        self.receipt.task = Some(TaskRecord {
            task: task::name(training.kind).to_owned(),
            reason: training.task_reason.clone(),
            target: training.target.clone(),
            classes: training.classes.len(),
            rows: training.targets.len(),
            features: training.schema.columns.len(),
        });
        let options = FitOptions {
            method: train.method.clone(),
            candidates: train.candidates.clone(),
            folds: train.folds,
            fitness,
            calibration,
            validation: if train.validation == "forward" {
                jope_runtime::automl::Validation::Forward { gap: train.gap }
            } else {
                if train.gap != 0 {
                    return Err(Failure::input("--gap requires --validation forward"));
                }
                jope_runtime::automl::Validation::Iid {}
            },
        };
        let encoding = landmarks::start(
            train,
            &self.common,
            &training,
            self.encoding_package.as_ref(),
            &csv.sha256,
        )?;
        if train.landmarks.enabled() {
            self.receipt.device = backend::DeviceChoice {
                requested: self.common.device.clone(),
                selected: if self.common.device == "cpu" {
                    "cpu"
                } else {
                    "cuda"
                },
                reason: "Dope inference on selected device; native fitting on CPU".into(),
            };
        }
        let fitted = api::fit(&training, &options, budget, &csv.sha256, self.created_by());
        if fitted.is_err() {
            landmarks::cancel(&encoding);
        }
        self.receipt.landmark_encoding = landmarks::finish(encoding)?;
        let outcome = fitted?;
        self.receipt.timings.select_seconds = outcome.select_seconds;
        self.receipt.timings.fit_seconds = outcome.fit_seconds;
        self.receipt.method = Some(MethodRecord {
            requested: train.method.clone(),
            selected: outcome.method.id.clone(),
            recipe_sha256: outcome.fitted.header.recipe_sha256.clone(),
        });
        self.receipt.selection = outcome.selection;
        Ok(outcome.fitted)
    }

    pub(super) fn predict(
        &mut self,
        fitted: &FittedModel,
        input: &Path,
        output: &Path,
        budget: &Budget,
    ) -> anyhow::Result<()> {
        let csv = self.read_csv("input", input)?;
        let started = Instant::now();
        let (rows, ignored) = table::prediction_ignoring(
            &csv,
            fitted.schema(),
            &fitted.header.target,
            &fitted.header.auxiliary_columns,
        )?;
        self.receipt.ignored_columns = ignored;
        let predictions = api::predict(fitted, &rows, budget)?;
        self.receipt.timings.predict_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let bytes = api::predictions_csv(&predictions)?;
        self.receipt
            .outputs
            .push(io::write_atomic("predictions", output, &bytes)?);
        self.receipt.timings.write_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Predict with a `math-pipeline/v1` object. It carries no fitted header, so the receipt
    /// records the decoder and the genome's digest as the method and leaves the task unset.
    pub(super) fn predict_math_pipeline(
        &mut self,
        object: &jope_runtime::model_object::ModelObject,
        model: &Path,
        input: &Path,
        output: &Path,
        started: Instant,
    ) -> anyhow::Result<()> {
        let pipeline = object.math_pipeline()?;
        self.receipt.inputs.push(FileRecord {
            role: "model",
            path: model.display().to_string(),
            bytes: object.manifest_bytes().len() as u64,
            sha256: jope_runtime::sha256_hex(object.manifest_bytes()),
        });
        self.receipt.timings.read_seconds += started.elapsed().as_secs_f64();
        self.receipt.method = Some(MethodRecord {
            requested: "model".into(),
            selected: jope_runtime::model_object::MATH_PIPELINE.into(),
            recipe_sha256: jope_runtime::sha256_hex(pipeline.state.dsl.as_bytes()),
        });
        let csv = self.read_csv("input", input)?;
        let started = Instant::now();
        let (values, ignored) = pipeline.predict_table(&csv)?;
        self.receipt.ignored_columns = ignored;
        self.receipt.timings.predict_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let bytes = api::predictions_csv(&methods::native::Predictions::Values(values))?;
        self.receipt
            .outputs
            .push(io::write_atomic("predictions", output, &bytes)?);
        self.receipt.timings.write_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Predict with a Platt-calibrated object through `ModelObject::predict`, which applies the
    /// stored map to the native head. The receipt takes the method and task from the fitted
    /// header; a bound head has none, so it records the format and genome and leaves the task unset.
    pub(super) fn predict_calibrated(
        &mut self,
        object: &jope_runtime::model_object::ModelObject,
        model: &Path,
        input: &Path,
        output: &Path,
        budget: &Budget,
        started: Instant,
    ) -> anyhow::Result<()> {
        let contract = object
            .calibrated_input()?
            .ok_or_else(|| Failure::integrity("model object has no Platt calibration"))?;
        self.receipt.inputs.push(FileRecord {
            role: "model",
            path: model.display().to_string(),
            bytes: object.manifest_bytes().len() as u64,
            sha256: jope_runtime::sha256_hex(object.manifest_bytes()),
        });
        self.receipt.timings.read_seconds += started.elapsed().as_secs_f64();
        match &contract.header {
            Some(header) => {
                self.receipt.method = Some(MethodRecord {
                    requested: "model".into(),
                    selected: header.method.id.clone(),
                    recipe_sha256: header.recipe_sha256.clone(),
                });
                self.receipt.task = Some(TaskRecord {
                    task: header.task.clone(),
                    reason: "from the model file".into(),
                    target: header.target.clone(),
                    classes: header.class_order.len(),
                    rows: header.training_procedure.rows,
                    features: header.input_schema.len(),
                });
            }
            None => {
                self.receipt.method = Some(MethodRecord {
                    requested: "model".into(),
                    selected: object.manifest().format.clone(),
                    recipe_sha256: contract.genome_sha256.clone(),
                });
            }
        }
        let csv = self.read_csv("input", input)?;
        let started = Instant::now();
        let (rows, ignored) = contract.table(&csv)?;
        self.receipt.ignored_columns = ignored;
        let predictions = api::predict_object(object, &rows, budget)?;
        self.receipt.timings.predict_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let bytes = api::predictions_csv(&predictions)?;
        self.receipt
            .outputs
            .push(io::write_atomic("predictions", output, &bytes)?);
        self.receipt.timings.write_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Finish the receipt, write and print it, and return the exit status.
    pub(super) fn finish(mut self, result: anyhow::Result<()>) -> ExitCode {
        if let Some(budget) = &self.budget {
            self.receipt.budget = budget.record();
        }
        self.receipt.timings.total_seconds = self.started.elapsed().as_secs_f64();
        let code = match &result {
            Ok(()) => {
                self.receipt.outcome = "succeeded";
                0
            }
            Err(error) => {
                let kind = kind_of(error);
                self.receipt.error = Some(ErrorRecord {
                    kind,
                    exit_code: kind.exit_code(),
                    message: format!("{error:#}"),
                });
                kind.exit_code()
            }
        };
        let value = json!(self.receipt);
        let mut code = code;
        if let Some(path) = self
            .common
            .receipt
            .as_ref()
            .filter(|_| self.receipt_allowed)
        {
            let text = serde_json::to_vec_pretty(&value).expect("receipt serializes");
            if let Err(error) = io::write_atomic("receipt", path, &text) {
                eprintln!("jain: {error:#}");
                if code == 0 {
                    code = FailureKind::Output.exit_code();
                }
            }
        }
        PENDING_EXIT.store(code, std::sync::atomic::Ordering::SeqCst);
        if self.common.json {
            print_json(&value);
        } else {
            match &result {
                Ok(()) => summary_line(self.command, &self.receipt),
                Err(_) => {
                    if let Some(error) = &self.receipt.error {
                        eprintln!("jopedime {}: {}", self.command, error.message);
                    }
                }
            }
        }
        ExitCode::from(code)
    }
}
