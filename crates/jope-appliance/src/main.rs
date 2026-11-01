//! `jopedime`: one signed executable that fits and predicts tabular models with native Rust
//! methods (deploy/appliance/README.md).
//!
//! Every command that fits, predicts, verifies or prints notices first authenticates the
//! executable's own package (footer, manifest, signature, executable region); an integrity
//! failure is fatal there and never becomes a fallback. `doctor` reports it with exit 3;
//! `--version --json` reports it in its output. Exit statuses follow SPECIFICATION.md §13.3:
//! 0 success, 2 input, 3 integrity, 4 backend, 5 budget, 6 numerical, 7 output.
#![forbid(unsafe_code)]

mod bootstrap;
mod cli;
#[path = "cmd/mod.rs"]
mod cmd;
#[path = "cmd/common.rs"]
mod common;
#[cfg(feature = "client")]
#[path = "terminal/draft.rs"]
mod draft;
mod io;
mod landmarks;
#[cfg(feature = "mcp")]
pub mod mcp;
mod model;
#[cfg(feature = "research")]
#[path = "cmd/research.rs"]
mod research;
#[cfg(feature = "client")]
#[path = "terminal/session.rs"]
mod session;
#[path = "terminal/theme.rs"]
pub mod terminal;
#[cfg(feature = "engine")]
mod worker;

use clap::Parser;
use cli::{Cli, Command, CommonArgs, TrainArgs};
use jope_assets::Package;
use jope_runtime::api::{self, FitOptions};
use jope_runtime::backend::{self, DeviceRequest};
use jope_runtime::budget::Budget;
use jope_runtime::error::kind_of;
use jope_runtime::fitted::{CreatedBy, FittedModel};
use jope_runtime::limits::Limits;
use jope_runtime::methods;
use jope_runtime::receipt::{
    ErrorRecord, FileRecord, MethodRecord, PackageRecord, Receipt, TaskRecord, Timings,
};
use jope_runtime::table;
use jope_runtime::task::{self, TaskChoice};
use jope_runtime::{Failure, FailureKind, RUNTIME_ABI};
use serde_json::{json, Value};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The status a run will exit with, so a closed stdout while printing its receipt keeps it.
static PENDING_EXIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Write to stdout. A closed pipe (`jopedime licenses | head`) ends the program quietly instead
/// of panicking; any other write failure is an output failure.
fn out(text: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if let Err(error) = stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        if error.kind() == std::io::ErrorKind::BrokenPipe {
            std::process::exit(i32::from(
                PENDING_EXIT.load(std::sync::atomic::Ordering::SeqCst),
            ));
        }
        eprintln!("jopedime: stdout: {error}");
        std::process::exit(i32::from(FailureKind::Output.exit_code()));
    }
}

fn print_json(value: &Value) {
    let mut text = serde_json::to_string_pretty(value).expect("JSON values serialize");
    text.push('\n');
    out(&text);
}

fn version_value() -> Value {
    let package = Package::open_self();
    let package = match &package {
        Ok(Some(p)) => json!(bootstrap::summary(p)),
        Ok(None) => Value::Null,
        Err(error) => json!({"error": error.to_string()}),
    };
    json!({
        "product": "jopedime",
        "version": VERSION,
        "source_sha40": bootstrap::source_sha40(),
        "runtime_abi": RUNTIME_ABI,
        "package_format": jope_assets::MANIFEST_SCHEMA,
        "target": env!("JOPE_BUILD_TARGET"),
        "features": {
            "engine": cfg!(feature = "engine"),
            "mcp": cfg!(feature = "mcp"),
            "client": cfg!(feature = "client"),
            "embedded": bootstrap::EMBEDDED,
            "dev_keys": jope_assets::verify::TRUSTS_DEVELOPMENT_KEY,
            "cuda": backend::CUDA_COMPILED,
            "dope_cuda": cfg!(feature = "dope-cuda"),
        },
        "trusts_development_key": jope_assets::verify::TRUSTS_DEVELOPMENT_KEY,
        "package": package,
    })
}

fn unexecutable(promised: &[String]) -> anyhow::Result<Vec<(String, String)>> {
    #[cfg(feature = "engine")]
    {
        let mut blocked = jope_service::proposal::uncompiled(promised)?;
        if promised
            .iter()
            .any(|id| id == jope_service::proposal::METHOD)
        {
            let package = Package::open_self().map_err(bootstrap::integrity)?;
            let record = jope_service::proposal::record(package.as_ref())
                .map_err(|error| Failure::integrity(format!("{error:#}")))?;
            if !record.executable {
                blocked.push((record.id, record.reason.unwrap_or_default()));
            }
        }
        Ok(blocked)
    }
    #[cfg(not(feature = "engine"))]
    methods::unexecutable(promised)
}

fn doctor() -> anyhow::Result<(Value, bool)> {
    let started = Instant::now();
    let (package, promised, package_ok) = match Package::open_self() {
        Ok(Some(p)) => {
            let summary = bootstrap::summary(&p);
            let methods = summary.methods.clone();
            (
                json!({"status": "authenticated", "summary": summary}),
                methods,
                true,
            )
        }
        Ok(None) if bootstrap::EMBEDDED => (
            json!({"status": "missing", "reason": "release build without its package"}),
            Vec::new(),
            false,
        ),
        Ok(None) => (json!({"status": "unpackaged"}), Vec::new(), true),
        Err(error) => (
            json!({"status": "refused", "reason": error.to_string()}),
            Vec::new(),
            false,
        ),
    };
    let unexecutable: Vec<Value> = unexecutable(&promised)?
        .into_iter()
        .map(|(id, reason)| json!({"id": id, "reason": reason}))
        .collect();
    let ok = package_ok && unexecutable.is_empty();
    let value = json!({
        "ok": ok,
        "version": VERSION,
        "package": package,
        "promised_methods": promised,
        "unexecutable_promised_methods": unexecutable,
        "devices": backend::probe(),
        "native_device": "cpu",
        "dope_cuda_compiled": cfg!(feature = "dope-cuda"),
        "limits": Limits::default(),
        "cache": "none: packaged assets are authenticated and decoded in memory when a method \
                  needs them; this build keeps no derived cache",
        "worker_processes": "none: fitting runs in this process with a cooperative deadline",
        "seconds": started.elapsed().as_secs_f64(),
    });
    Ok((value, ok))
}

fn methods_value(task_filter: Option<&str>) -> anyhow::Result<Value> {
    if let Some(name) = task_filter {
        if !matches!(name, "regression" | "binary" | "multiclass") {
            return Err(Failure::input(format!("unknown task {name:?}")));
        }
    }
    let promised: Vec<String> = match Package::open_self() {
        Ok(Some(p)) => p.manifest().methods.clone(),
        Ok(None) => Vec::new(),
        Err(error) => return Err(bootstrap::integrity(error)),
    };
    let records = methods::records()?;
    #[cfg(feature = "engine")]
    let records = {
        let mut records = records;
        let package = Package::open_self().map_err(bootstrap::integrity)?;
        records.push(
            jope_service::proposal::record(package.as_ref())
                .map_err(|error| Failure::integrity(format!("{error:#}")))?,
        );
        records
    };
    let records: Vec<Value> = records
        .into_iter()
        .filter(|r| task_filter.is_none_or(|t| r.task == t))
        .map(|r| {
            let mut value = json!(r);
            value["promised_by_package"] = json!(promised.contains(&r.id));
            value
        })
        .collect();
    Ok(json!({
        "auto": {
            "id": methods::AUTO,
            "policy": jope_runtime::automl::SELECTION_POLICY,
            "fold_policy": jope_runtime::automl::FOLD_POLICY,
            "default_folds": jope_runtime::automl::DEFAULT_FOLDS,
            "metrics": {"regression": "rmse", "binary": "auroc", "multiclass": "log_loss"},
        },
        "methods": records,
    }))
}

fn licenses(json_out: bool) -> anyhow::Result<()> {
    let package = bootstrap::open()?;
    let mut notices = Vec::new();
    let mut assets = Vec::new();
    if let Some(package) = &package {
        for entry in &package.manifest().entries {
            assets.push(json!({
                "id": entry.id,
                "origin": entry.origin,
                "spdx": entry.license.spdx,
                "redistribution": entry.license.redistribution,
            }));
            if entry.role == "notice" {
                let bytes = package
                    .read_entry(&entry.id)
                    .map_err(bootstrap::integrity)?;
                notices.push((
                    entry.id.clone(),
                    String::from_utf8_lossy(bytes.bytes()).into_owned(),
                ));
            }
        }
    }
    if json_out {
        print_json(&json!({
            "product_license": "Apache-2.0",
            "packaged": package.is_some(),
            "assets": assets,
            "notices": notices.iter().map(|(id, text)| json!({"id": id, "text": text})).collect::<Vec<_>>(),
        }));
    } else {
        out(&format!(
            "jopedime {VERSION} is licensed under Apache-2.0.\n"
        ));
        if package.is_none() {
            out("This is an unpackaged development build: it embeds no notices.\n");
        }
        for (id, text) in notices {
            out(&format!("\n==> {id} <==\n{text}\n"));
        }
    }
    Ok(())
}

fn verify_self(json_out: bool) -> anyhow::Result<()> {
    let package = Package::open_self()
        .map_err(bootstrap::integrity)?
        .ok_or_else(|| Failure::integrity("this executable has no package to verify"))?;
    let report = package.verify_full().map_err(bootstrap::integrity)?;
    let value = json!({
        "ok": true,
        "summary": bootstrap::summary(&package),
        "report": report,
        "note": "verify-self detects corruption; it cannot vouch for an executable that may \
                 itself be modified. Verify a download with `jope-dist verify` or SHA256SUMS.",
    });
    if json_out {
        print_json(&value);
    } else {
        eprintln!(
            "verified {} bytes: executable, {} assets, manifest, signature ({})",
            report.file_bytes,
            report.entries,
            package.signer().id
        );
    }
    Ok(())
}

/// State shared by the fitting and prediction commands, so a failure still yields a receipt.
struct Run {
    command: &'static str,
    started: Instant,
    common: CommonArgs,
    receipt: Receipt,
    encoding_package: Option<Package>,
    /// The run's budget once prepared, so a failed run still records what it consumed.
    budget: Option<Budget>,
    /// Whether `--receipt` names no input or output (checked before any work).
    receipt_allowed: bool,
}

impl Run {
    fn new(command: &'static str, common: &CommonArgs) -> Self {
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
    fn prepare(&mut self, landmark_encoding: bool) -> anyhow::Result<Budget> {
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
    fn refuse_aliasing(&mut self, output: &Path, inputs: &[&Path]) -> anyhow::Result<()> {
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

    fn fit(&mut self, train: &TrainArgs, budget: &Budget) -> anyhow::Result<FittedModel> {
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

    fn predict(
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
    fn predict_math_pipeline(
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
    fn predict_calibrated(
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
    fn finish(mut self, result: anyhow::Result<()>) -> ExitCode {
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
                eprintln!("jopedime: {error:#}");
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

fn summary_line(command: &str, receipt: &Receipt) {
    let method = receipt
        .method
        .as_ref()
        .map(|m| format!(" method={}", m.selected))
        .unwrap_or_default();
    let task = receipt
        .task
        .as_ref()
        .map(|t| match command {
            "predict" => format!(" task={}", t.task),
            _ => format!(" task={} rows={}", t.task, t.rows),
        })
        .unwrap_or_default();
    let objective = receipt
        .selection
        .as_ref()
        .filter(|s| s.fitness.is_some())
        .map(|s| {
            let calibrated = s
                .calibration
                .as_ref()
                .and_then(|c| c.output_transform)
                .map_or("", |t| {
                    if t.intercept == 0.0 && t.slope == 1.0 {
                        " floored"
                    } else {
                        " calibrated"
                    }
                });
            format!(" objective={}{calibrated}", s.metric)
        })
        .unwrap_or_default();
    let outputs: Vec<&str> = receipt.outputs.iter().map(|o| o.path.as_str()).collect();
    eprintln!(
        "jopedime {command}:{task}{method}{objective} device={} wrote {} in {:.2}s",
        receipt.device.selected,
        outputs.join(", "),
        receipt.timings.total_seconds
    );
}

fn run_fit(
    train: &TrainArgs,
    output: Option<&Path>,
    model_root: &Path,
    common: &CommonArgs,
) -> ExitCode {
    let mut run = Run::new("fit", common);
    let result = (|| -> anyhow::Result<()> {
        let landmark_inputs = train.landmarks.protected_inputs();
        let mut protected = vec![train.train.as_path()];
        protected.extend(landmark_inputs.iter().map(std::path::PathBuf::as_path));
        if let Some(output) = output {
            run.refuse_aliasing(output, &protected)?;
        } else {
            jope_runtime::output_safety::refuse_directory_aliasing(model_root, &protected)?;
            protected.push(model_root);
            if let Some(receipt) = &common.receipt {
                io::refuse_aliasing(receipt, &protected)?;
            }
            run.receipt_allowed = true;
        }
        landmarks::refuse_aliasing(train, common, &[output.unwrap_or(model_root)])?;
        let b = run.prepare(train.landmarks.enabled())?;
        let fitted = run.fit(train, &b)?;
        let started = Instant::now();
        if let Some(output) = output {
            let bytes = fitted.to_bytes()?;
            run.receipt
                .outputs
                .push(io::write_atomic("model", output, &bytes)?);
        } else {
            let object = jope_runtime::model_object::ModelObject::from_fitted(&fitted)?;
            let path = object.save(model_root, Limits::default().max_model_bytes)?;
            run.receipt.outputs.push(FileRecord {
                role: "model",
                path: path.display().to_string(),
                bytes: object.manifest_bytes().len() as u64,
                sha256: jope_runtime::sha256_hex(object.manifest_bytes()),
            });
        }
        run.receipt.timings.write_seconds += started.elapsed().as_secs_f64();
        Ok(())
    })();
    run.finish(result)
}

fn run_predict(model: &str, input: &Path, output: &Path, common: &CommonArgs) -> ExitCode {
    #[cfg(feature = "mcp")]
    if let Some((run_id, algorithm_id)) = model
        .split_once(':')
        .or_else(|| model.strip_prefix("run_").map(|_| (model, "")))
    {
        if run_id.parse::<jope_api::ids::RunId>().is_ok() {
            return cmd::dataset::report(run_predict_remote(
                run_id,
                (!algorithm_id.is_empty()).then_some(algorithm_id),
                input,
                output,
            ));
        }
    }
    let model = Path::new(model);
    let mut run = Run::new("predict", common);
    let result = (|| -> anyhow::Result<()> {
        run.refuse_aliasing(output, &[model, input])?;
        let b = run.prepare(false)?;
        let started = Instant::now();
        let limit = Limits::default().max_model_bytes;
        // Only the manifest's `format` and capabilities are read here; `ModelObject::read` and
        // `load_fitted` each verify every part, so the full read happens once, on the path that
        // uses it. A Platt-calibrated object predicts whole: `load_fitted` refuses it.
        let route = if model.is_dir() {
            object_route(model)?
        } else {
            ObjectRoute::Fitted
        };
        if route != ObjectRoute::Fitted {
            let object = jope_runtime::model_object::ModelObject::read(model, limit)?;
            return match route {
                ObjectRoute::MathPipeline => {
                    run.predict_math_pipeline(&object, model, input, output, started)
                }
                _ => run.predict_calibrated(&object, model, input, output, &b, started),
            };
        }
        let fitted = jope_runtime::model_object::load_fitted(model, limit)?;
        let identity_path = if model.is_dir() {
            model.join("manifest.json")
        } else {
            model.to_owned()
        };
        let bytes = jope_runtime::model_object::read_bounded(&identity_path, limit)?;
        run.receipt.inputs.push(FileRecord {
            role: "model",
            path: model.display().to_string(),
            bytes: bytes.len() as u64,
            sha256: jope_runtime::sha256_hex(&bytes),
        });
        run.receipt.timings.read_seconds += started.elapsed().as_secs_f64();
        run.receipt.method = Some(MethodRecord {
            requested: "model".into(),
            selected: fitted.header.method.id.clone(),
            recipe_sha256: fitted.header.recipe_sha256.clone(),
        });
        run.receipt.task = Some(TaskRecord {
            task: fitted.header.task.clone(),
            reason: "from the model file".into(),
            target: fitted.header.target.clone(),
            classes: fitted.header.class_order.len(),
            rows: fitted.header.training_procedure.rows,
            features: fitted.header.input_schema.len(),
        });
        run.predict(&fitted, input, output, &b)?;
        Ok(())
    })();
    run.finish(result)
}

#[cfg(feature = "mcp")]
fn run_predict_remote(
    run: &str,
    algorithm: Option<&str>,
    input: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    use jope_api::{
        ids::{AlgorithmId, RunId},
        PredictRequest, RunSnapshot,
    };
    use jope_client::{files, Transport};
    use std::{io::Read, str::FromStr};
    let client = cmd::dataset::resolve_client()?;
    let run_id = RunId::from_str(run).map_err(|_| anyhow::anyhow!("invalid run id"))?;
    let snapshot: RunSnapshot = client.run_status(&run_id)?;
    let algorithm = match algorithm {
        Some(value) => {
            AlgorithmId::from_str(value).map_err(|_| anyhow::anyhow!("invalid algorithm id"))?
        }
        None => {
            snapshot
                .best
                .ok_or_else(|| anyhow::anyhow!("run has no best algorithm"))?
                .algorithm_id
        }
    };
    let record = client.algorithm_get(&run_id, &algorithm)?;
    let artifact = record
        .model_artifact
        .ok_or_else(|| anyhow::anyhow!("algorithm has no model artifact"))?;
    let register_key = cmd::common::resolve_idempotency_key(None);
    cmd::common::write_idempotency_key(&mut std::io::stderr().lock(), &register_key)?;
    let name = input
        .file_name()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "prediction.csv".into());
    let registered = files::upload_csv(&client, &name, std::fs::File::open(input)?, &register_key)?
        .json::<jope_api::DatasetProfile>()?;
    let predict_key = cmd::common::resolve_idempotency_key(None);
    cmd::common::write_idempotency_key(&mut std::io::stderr().lock(), &predict_key)?;
    let request = PredictRequest {
        model_artifact: artifact,
        dataset: registered.dataset_id,
    };
    let mut response = client.model_predict(&request, &predict_key)?;
    let mut bytes = Vec::new();
    response
        .body
        .by_ref()
        .take((1_u64 << 30) + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= (1_u64 << 30),
        "prediction output exceeds 1 GiB"
    );
    if output == Path::new("-") {
        use std::io::Write;
        std::io::stdout().lock().write_all(&bytes)?;
    } else {
        io::write_atomic("predictions", output, &bytes)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObjectRoute {
    MathPipeline,
    Calibrated,
    Fitted,
}

fn object_route(object_dir: &Path) -> anyhow::Result<ObjectRoute> {
    use jope_runtime::model_object::{
        calibration_platt, read_bounded, MANIFEST, MATH_PIPELINE, MAX_MANIFEST_BYTES,
    };
    #[derive(serde::Deserialize)]
    struct Format {
        format: String,
        #[serde(default)]
        required_capabilities: Vec<String>,
    }
    let bytes = read_bounded(&object_dir.join(MANIFEST), MAX_MANIFEST_BYTES)?;
    let manifest: Format = serde_json::from_slice(&bytes)
        .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
    Ok(if manifest.format == MATH_PIPELINE {
        ObjectRoute::MathPipeline
    } else if manifest
        .required_capabilities
        .iter()
        .any(|c| c == calibration_platt::CAPABILITY)
    {
        ObjectRoute::Calibrated
    } else {
        ObjectRoute::Fitted
    })
}

fn run_fit_predict(
    train: &TrainArgs,
    input: &Path,
    output: &Path,
    common: &CommonArgs,
) -> ExitCode {
    let mut run = Run::new("fit-predict", common);
    let result = (|| -> anyhow::Result<()> {
        let landmark_inputs = train.landmarks.protected_inputs();
        let mut protected = vec![train.train.as_path(), input];
        protected.extend(landmark_inputs.iter().map(std::path::PathBuf::as_path));
        run.refuse_aliasing(output, &protected)?;
        landmarks::refuse_aliasing(train, common, &[input, output])?;
        let b = run.prepare(train.landmarks.enabled())?;
        let fitted = run.fit(train, &b)?;
        run.predict(&fitted, input, output, &b)?;
        Ok(())
    })();
    run.finish(result)
}

fn report(result: anyhow::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("jopedime: {error:#}");
            ExitCode::from(
                common::report_exit_code(&error).unwrap_or_else(|| kind_of(&error).exit_code()),
            )
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--version` alone, and `--version --json` (SPECIFICATION.md §3.3).
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--version"] | ["-V"] => {
            out(&format!("jopedime {VERSION}\n"));
            return ExitCode::SUCCESS;
        }
        ["--version", "--json"] | ["--json", "--version"] => {
            print_json(&version_value());
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    #[cfg(feature = "client")]
    if let Some(attach) = session::requested(&args) {
        return report(session::run(attach));
    }
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "research")]
        Command::Research(args) => report(research::run(args)),
        #[cfg(feature = "engine")]
        Command::Worker { protocol } => report(worker::run(protocol)),
        Command::Version(flag) => {
            if flag.json {
                print_json(&version_value());
            } else {
                out(&format!("jopedime {VERSION}\n"));
            }
            ExitCode::SUCCESS
        }
        Command::Doctor(flag) => match doctor() {
            Ok((value, ok)) => {
                let code = if ok {
                    0
                } else {
                    FailureKind::Integrity.exit_code()
                };
                PENDING_EXIT.store(code, std::sync::atomic::Ordering::SeqCst);
                if flag.json {
                    print_json(&value);
                } else {
                    eprintln!(
                        "jopedime doctor: {} (package: {}; devices: {})",
                        if ok { "ok" } else { "NOT OK" },
                        value["package"]["status"].as_str().unwrap_or("?"),
                        value["devices"]["reason"].as_str().unwrap_or("?")
                    );
                    for item in value["unexecutable_promised_methods"]
                        .as_array()
                        .into_iter()
                        .flatten()
                    {
                        eprintln!(
                            "  promised but not executable: {} ({})",
                            item["id"], item["reason"]
                        );
                    }
                }
                ExitCode::from(code)
            }
            Err(error) => report(Err(error)),
        },
        Command::Methods { task, json } => report(methods_value(task.as_deref()).map(|value| {
            if json.json {
                print_json(&value);
            } else {
                for m in value["methods"].as_array().into_iter().flatten() {
                    out(&format!(
                        "{:<36} {:<11} {:<7} {}
",
                        m["id"].as_str().unwrap_or(""),
                        m["task"].as_str().unwrap_or(""),
                        m["kind"].as_str().unwrap_or(""),
                        if m["executable"] == true {
                            "executable"
                        } else {
                            "not executable"
                        }
                    ));
                }
                out(&format!(
                    "auto: bounded selection ({})\n",
                    jope_runtime::automl::SELECTION_POLICY
                ));
            }
        })),
        Command::Fit {
            train,
            output,
            model_root,
            common,
        } => run_fit(&train, output.as_deref(), &model_root, &common),
        Command::Model(args) => report(model::run(args)),
        #[cfg(feature = "mcp")]
        Command::Run(args) => match cmd::run::dispatch(args) {
            Err(error) => match cmd::run::dispatch_exit_code(&error) {
                Some(code) => {
                    if code != 130 {
                        eprintln!("jopedime: {error:#}");
                    }
                    ExitCode::from(code)
                }
                None => report(Err(error)),
            },
            result => report(result),
        },
        #[cfg(feature = "mcp")]
        Command::Status(args) => report((|| {
            let client = cmd::client()?;
            cmd::status::run(args, &client)
        })()),
        #[cfg(feature = "mcp")]
        Command::Events(args) => report((|| {
            let client = cmd::client()?;
            cmd::events::run(args, &client)
        })()),
        #[cfg(feature = "client")]
        Command::Control {
            run,
            action,
            body,
            idempotency_key,
        } => report(cmd::control::run(run, action, body, idempotency_key)),
        #[cfg(feature = "client")]
        Command::Restart {
            run,
            new_seed,
            warm,
            body,
            idempotency_key,
        } => report(cmd::restart::run(
            run,
            new_seed,
            warm,
            body,
            idempotency_key,
        )),
        #[cfg(feature = "client")]
        Command::Runs { command } => report(cmd::runs::run(command)),
        #[cfg(feature = "client")]
        Command::Code { run, alg, lang } => report(cmd::code::run(run, alg, lang)),
        #[cfg(feature = "mcp")]
        Command::Export {
            run,
            alg,
            kind,
            lang,
            out,
            force,
            idempotency_key,
            json,
        } => cmd::dataset::report(cmd::export::run(cmd::export::Options {
            run: &run,
            alg: alg.as_deref(),
            kind,
            language: lang,
            out: &out,
            force,
            idempotency_key: idempotency_key.as_deref(),
            json_out: json,
        })),
        #[cfg(feature = "mcp")]
        Command::Dataset { command } => cmd::dataset::report(cmd::dataset::run(command)),
        #[cfg(feature = "mcp")]
        Command::Objective {
            command: cli::ObjectiveCommand::Validate { spec, json },
        } => cmd::dataset::report(cmd::objective::run(&spec, json)),
        #[cfg(feature = "mcp")]
        Command::Gc {
            dry_run,
            idempotency_key,
            json,
        } => cmd::dataset::report(cmd::gc::run(dry_run, idempotency_key.as_deref(), json)),
        Command::Predict {
            model,
            input,
            output,
            common,
        } => run_predict(&model, &input, &output, &common),
        Command::FitPredict {
            train,
            input,
            output,
            common,
        } => run_fit_predict(&train, &input, &output, &common),
        Command::Fitness { command } => report(match command {
            cli::FitnessCommand::List => {
                print_json(&jope_runtime::service::catalog());
                Ok(())
            }
            cli::FitnessCommand::Validate { spec } => (|| {
                let fitness = load_fitness(&spec)?;
                print_json(&jope_runtime::service::describe(&fitness)?);
                Ok(())
            })(),
            cli::FitnessCommand::Schema => {
                print_json(&jope_runtime::fitness::schema());
                Ok(())
            }
            cli::FitnessCommand::Evaluate { input } => (|| {
                use std::io::Read;
                // Read at most one byte past the limit, so an oversized file is never loaded.
                let mut bytes = Vec::new();
                std::fs::File::open(input)?
                    .take((jope_runtime::service::MAX_MESSAGE_BYTES + 1) as u64)
                    .read_to_end(&mut bytes)?;
                anyhow::ensure!(
                    bytes.len() <= jope_runtime::service::MAX_MESSAGE_BYTES,
                    "evaluation request too large"
                );
                let request = serde_json::from_slice(&bytes)?;
                print_json(&jope_runtime::service::evaluate(request)?);
                Ok(())
            })(),
        }),
        #[cfg(feature = "mcp")]
        Command::Mcp => report(cmd::mcp_entry::run()),
        // Without the shared client there is no discovery: serve in process, as before #656.
        #[cfg(not(feature = "mcp"))]
        Command::Mcp => report(serve(true)),
        Command::Api => report(serve(false)),
        Command::Serve {
            status,
            stop,
            idle_exit,
        } => ExitCode::from(cmd::serve::run(status, stop, idle_exit.as_deref())),
        Command::VerifySelf(flag) => report(verify_self(flag.json)),
        Command::Licenses(flag) => report(licenses(flag.json)),
    }
}

fn load_fitness(text: &str) -> anyhow::Result<jope_runtime::fitness::Fitness> {
    let value = if let Some(path) = text.strip_prefix('@') {
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut bytes = Vec::new();
        file.by_ref()
            .take((jope_runtime::fitness::MAX_SPEC_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        String::from_utf8(bytes)?
    } else {
        text.to_owned()
    };
    jope_runtime::fitness::Fitness::parse(&value)
}

fn serve(mcp: bool) -> anyhow::Result<()> {
    use std::io::{BufRead, Read, Write};
    let package = bootstrap::open()?;
    let mut server = jope_runtime::service::Server::new(CreatedBy {
        source_sha40: bootstrap::source_sha40(),
        package_manifest_sha256: package
            .as_ref()
            .map(|p| bootstrap::summary(p).manifest_sha256),
        package_profile: package.as_ref().map(|p| p.manifest().profile.clone()),
    });
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    loop {
        let mut bytes = Vec::new();
        input
            .by_ref()
            .take((jope_runtime::service::MAX_MESSAGE_BYTES + 1) as u64)
            .read_until(b'\n', &mut bytes)?;
        if bytes.is_empty() {
            break;
        }
        let response = if bytes.len() > jope_runtime::service::MAX_MESSAGE_BYTES {
            // Skip the rest of the oversized line and answer it; the session goes on.
            if !bytes.ends_with(b"\n") {
                let mut rest = Vec::new();
                loop {
                    rest.clear();
                    let read = input
                        .by_ref()
                        .take(64 * 1024)
                        .read_until(b'\n', &mut rest)?;
                    if read == 0 || rest.ends_with(b"\n") {
                        break;
                    }
                }
            }
            Some(
                serde_json::json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,
                "message":format!("message larger than {} bytes",
                    jope_runtime::service::MAX_MESSAGE_BYTES)}}),
            )
        } else {
            match serde_json::from_slice(&bytes) {
                Ok(value) => server.dispatch(value, mcp),
                Err(error) => Some(
                    serde_json::json!({"jsonrpc":"2.0", "id":null,"error":{"code":-32700,"message":error.to_string()}}),
                ),
            }
        };
        if let Some(response) = response {
            serde_json::to_writer(&mut output, &response)?;
            writeln!(output)?;
            output.flush()?;
        }
    }
    Ok(())
}
