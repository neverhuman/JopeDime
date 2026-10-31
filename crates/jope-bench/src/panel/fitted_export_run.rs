//! Execute retained export plans through the original campaign pool and deadline.
//! Reporting authenticates receipts and never launches a compiler or predictor.
use super::{
    campaign::{self, CampaignPlan, Runtime},
    fitted_export::{self, Entry, PreparedExport},
    inference,
};
use anyhow::{ensure, Context, Result};
use clap::Args;
use jope_pipeline::{
    execution::{Artifact, ExecutionPool, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{EvaluationStage, EvaluationStatus},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Args)]
pub struct QualifyArgs {
    #[arg(long)]
    pub prepared: PathBuf,
    #[arg(long)]
    pub prepared_sha256: String,
    /// JSON map from generated command names to pinned campaign Runtime records.
    #[arg(long)]
    pub tools: PathBuf,
    #[arg(long)]
    pub tools_sha256: String,
    /// W9-T05: a backend whose package is required (repeatable); every other emitted backend
    /// is optional. Frozen with the run.
    #[arg(long = "require", value_name = "BACKEND")]
    pub require: Vec<String>,
    #[arg(long, hide = true)]
    pub supervisor_unit: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunPlan {
    schema: String,
    prepared: Artifact,
    prepared_sha256: String,
    tools: Artifact,
    runtimes: BTreeMap<String, Runtime>,
    /// W9-T05: backends whose package is required. Absent in runs frozen before packages were
    /// sealed (every backend optional).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    required: Vec<inference::Backend>,
}

/// W9-T05: the installed packages of an export directory, one per qualified backend.
pub const PACKAGES: &str = "packages";
pub const PACKAGE_RECORD_SCHEMA: &str = "jope.fitted-package-record/v1";

/// W9-T05: the record `qualify-export` writes for one backend's installed package. Its
/// identity is re-derived from the authenticated qualification receipts whenever it is used
/// (`load_recorded`), so the record names a package; it does not vouch for one. A loaded
/// package is not a release: release stays a separate, explicitly authorized state (W9-T07).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageRecord {
    pub schema: String,
    pub prepared_sha256: String,
    pub backend: inference::Backend,
    /// SHA-256 of the installed `package.json` bytes.
    pub identity: String,
    pub required: bool,
    /// W9-T02: the fitted subject the package serves.
    pub fitted_subject: FittedSubject,
    /// W9-T02: the authorized final whole-data refit the export binds. Always `None` today:
    /// no whole-data refit is defined or authorized (owner decision D19), so the package serves
    /// the selected recipe's fold fit and cannot be presented as the final model.
    pub final_refit_sha256: Option<String>,
    pub release: String,
}

/// W9-T02: the exported fitted subject, kept apart from any final refit: the selected
/// recipe's existing fit on one development fold (`export-winner --fold`), never a refit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedSubject {
    pub kind: String,
    pub method: String,
    pub fold: u8,
    /// Identity of the selected fold's original fit request.
    pub request_sha256: String,
    pub fitted_sha256: String,
}
fn fitted_subject(prepared: &PreparedExport) -> Result<FittedSubject> {
    Ok(FittedSubject {
        kind: "selected_fold_fit".into(),
        method: prepared.observation.method.clone(),
        fold: prepared.observation.request.fold,
        request_sha256: prepared.original_job.request.identity()?,
        fitted_sha256: prepared
            .observation
            .result
            .fitted_state_sha256
            .clone()
            .context("fitted subject has no fitted state")?,
    })
}
/// The record the authenticated seal of `backend` derives.
fn expected_record(
    prepared: &PreparedExport,
    backend: inference::Backend,
    manifest: &inference::package::PackageManifest,
) -> Result<PackageRecord> {
    let fitted_subject = fitted_subject(prepared)?;
    ensure!(
        manifest.fitted_sha256 == fitted_subject.fitted_sha256,
        "package serves another fitted state than the exported subject"
    );
    Ok(PackageRecord {
        schema: PACKAGE_RECORD_SCHEMA.into(),
        prepared_sha256: prepared.identity()?,
        backend,
        identity: inference::package::identity(manifest)?,
        required: manifest.required,
        fitted_subject,
        final_refit_sha256: None,
        release: "not-decided".into(),
    })
}
fn package_directory(directory: &Path, backend: inference::Backend) -> PathBuf {
    directory.join(PACKAGES).join(fitted_export::name(backend))
}
fn record_path(directory: &Path, backend: inference::Backend) -> PathBuf {
    directory
        .join(PACKAGES)
        .join(format!("{}.record.json", fitted_export::name(backend)))
}
fn backend_named(name: &str) -> Result<inference::Backend> {
    fitted_export::ORDER
        .into_iter()
        .find(|backend| fitted_export::name(*backend) == name)
        .with_context(|| format!("unknown backend {name}"))
}

/// W9-T05: the package of one backend, sealed from its authenticated qualification receipts
/// (report, execution receipt, pinned tool inventory and the qualified native outputs), never
/// from a caller's claim. `None` unless the authenticated status is `qualified`.
type Sealed = (
    inference::package::PackageManifest,
    BTreeMap<String, Vec<u8>>,
);
fn sealed_package(
    plan: &RunPlan,
    prepared: &PreparedExport,
    entry: &Entry,
) -> Result<Option<Sealed>> {
    use inference::package::{built_files, seal, DependencyProfile};
    let Entry::Prepared {
        backend,
        bundle,
        qualification,
    } = entry
    else {
        return Ok(None);
    };
    let report = backend_report(plan, prepared, entry)?;
    if report["status"] != "qualified" {
        return Ok(None);
    }
    let (input, job) = derived(plan, prepared, entry)?;
    let (_, journal) = campaign(prepared)?;
    let outcome = original_receipt(&journal, &job)?
        .context("qualified backend lost its execution receipt")?;
    let result: inference::QualificationResult = serde_json::from_value(report["parity"].clone())?;
    let (build, serve) = DependencyProfile::expected_tools(qualification)?;
    let tool = |name: &String| -> Result<(String, Artifact)> {
        let runtime = input
            .runtimes
            .get(name)
            .context("qualified tool has no pinned inventory")?;
        Ok((name.clone(), runtime.program.clone()))
    };
    let profile = DependencyProfile {
        runtime: bundle.manifest.runtime.clone(),
        host_boundary: bundle.manifest.host_boundary.clone(),
        numerical_policy: bundle.manifest.numerical_policy.clone(),
        build: build.iter().map(tool).collect::<Result<_>>()?,
        build_commands: qualification.commands[..qualification.commands.len() - 1].to_vec(),
        serve: serve.iter().map(tool).collect::<Result<_>>()?,
    };
    let mut built = BTreeMap::new();
    let mut executed = BTreeMap::new();
    for name in built_files(*backend) {
        let path = job.directory.join(name);
        let output = outcome
            .outputs
            .iter()
            .find(|artifact| artifact.path == path)
            .context("qualified native output absent from its execution receipt")?;
        built.insert((*name).to_owned(), fs::read(&path)?);
        executed.insert((*name).to_owned(), output.sha256.clone());
    }
    Ok(Some(seal(
        bundle,
        qualification,
        &result,
        profile,
        &built,
        &executed,
        plan.required.contains(backend),
    )?))
}

/// W9-T05: seal, install (once) and record the package of every qualified backend. An
/// existing installation must load under the re-derived identity; an existing record must
/// equal the re-derived one.
fn record_packages(
    directory: &Path,
    plan: &RunPlan,
    prepared: &PreparedExport,
) -> Result<Vec<PackageRecord>> {
    let mut records = Vec::new();
    for entry in &prepared.entries {
        let Some((manifest, files)) = sealed_package(plan, prepared, entry)? else {
            continue;
        };
        let backend = entry.backend();
        let identity = inference::package::identity(&manifest)?;
        let installed = package_directory(directory, backend);
        if !installed.try_exists()? {
            fs::create_dir_all(directory.join(PACKAGES))?;
            ensure!(
                inference::package::install(&installed, &manifest, &files)? == identity,
                "installed package identity differs from its seal"
            );
        }
        inference::package::load(&installed, &identity)
            .context("installed package differs from its authenticated seal")?;
        let record = expected_record(prepared, backend, &manifest)?;
        campaign::immutable(&record_path(directory, backend), &record)?;
        records.push(record);
    }
    Ok(records)
}

/// W9-T05: required/optional outcome of the recorded packages
/// (`inference::package::assess`); every emitted backend not required is optional.
fn assess_records(directory: &Path, plan: &RunPlan, prepared: &PreparedExport) -> Result<Value> {
    let request = inference::package::PackageRequest {
        required: plan.required.clone(),
        optional: prepared
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::Prepared { .. }))
            .map(Entry::backend)
            .filter(|backend| !plan.required.contains(backend))
            .collect(),
    };
    let mut outcomes = Vec::new();
    for backend in request.required.iter().chain(&request.optional) {
        let path = record_path(directory, *backend);
        let outcome = if path.try_exists()? {
            let record: PackageRecord = read(&path)?;
            inference::package::load(&package_directory(directory, *backend), &record.identity)
                .map_err(|error| format!("{error:#}"))
        } else {
            Err("not qualified: no package".to_owned())
        };
        outcomes.push((*backend, outcome));
    }
    inference::package::assess(&request, &outcomes)
}

/// Authenticate a retained export and its frozen run, and seal `backend`'s package from the
/// authenticated receipts.
fn authenticated_seal(
    prepared_path: &Path,
    prepared_sha256: &str,
    backend: inference::Backend,
) -> Result<(PathBuf, PreparedExport, Sealed)> {
    let prepared: PreparedExport = read(&fitted_export::artifact(prepared_path)?.path)?;
    ensure!(
        prepared.identity()? == prepared_sha256,
        "prepared export identity changed"
    );
    prepared.verify()?;
    let (_, journal) = campaign(&prepared)?;
    let directory = fitted_export::location(&journal, &prepared.observation)?;
    let plan: RunPlan = read(&directory.join("run.json"))?;
    verify_plan(&plan, &prepared)?;
    let entry = prepared
        .entries
        .iter()
        .find(|entry| entry.backend() == backend)
        .context("backend absent from the export")?;
    let sealed = sealed_package(&plan, &prepared, entry)?
        .context("backend is not qualified by its authenticated receipts")?;
    Ok((directory, prepared, sealed))
}

/// W9-T05: load a backend's installed package under the identity of its record, after
/// requiring that record to be the one the authenticated qualification receipts re-derive.
/// The identity is never taken from the caller.
pub fn load_recorded(
    prepared_path: &Path,
    prepared_sha256: &str,
    backend: inference::Backend,
) -> Result<inference::package::LoadedPackage> {
    let (directory, prepared, (manifest, _)) =
        authenticated_seal(prepared_path, prepared_sha256, backend)?;
    let recorded: PackageRecord = read(&record_path(&directory, backend))?;
    let expected = expected_record(&prepared, backend, &manifest)?;
    // The identity is compared by `load_authenticated`; everything else here.
    ensure!(
        PackageRecord {
            identity: expected.identity.clone(),
            ..recorded.clone()
        } == expected,
        "package record is not the one its authenticated qualification derives"
    );
    inference::package::load_authenticated(
        &package_directory(&directory, backend),
        &recorded.identity,
        &manifest,
    )
}

#[derive(Args)]
pub struct InstallArgs {
    #[arg(long)]
    pub prepared: PathBuf,
    #[arg(long)]
    pub prepared_sha256: String,
    /// `python`, `rust`, `cpp17`, `cpp20` or `javascript`.
    #[arg(long)]
    pub backend: String,
    /// A new directory to install into.
    #[arg(long)]
    pub destination: PathBuf,
}

/// W9-T05 `install-export`: install one backend's package, sealed from the authenticated
/// qualification receipts, into a new directory and load it back under the re-derived
/// identity (which must also be the recorded one). Nothing is emitted, compiled, fitted or
/// predicted; release is not decided here.
pub fn install_export(args: InstallArgs) -> Result<Value> {
    let backend = backend_named(&args.backend)?;
    let recorded = load_recorded(&args.prepared, &args.prepared_sha256, backend)?;
    let (_, prepared, (manifest, files)) =
        authenticated_seal(&args.prepared, &args.prepared_sha256, backend)?;
    let record = expected_record(&prepared, backend, &manifest)?;
    let identity = inference::package::install(&args.destination, &manifest, &files)?;
    ensure!(
        identity == recorded.identity,
        "installed package differs from the recorded package"
    );
    let loaded = inference::package::load_authenticated(&args.destination, &identity, &manifest)?;
    Ok(
        json!({"ok":true,"backend":loaded.manifest.backend,"identity":identity,"destination":args.destination,"fitted_subject":record.fitted_subject,"final_refit_sha256":record.final_refit_sha256,"required":loaded.manifest.required,"release":"not-decided","fits":0,"compiler_commands":0}),
    )
}
/// Only public model code and unlabeled prediction inputs reach the bounded child.
/// Native expected predictions, campaign data paths and scoring labels stay out.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerInput {
    schema: String,
    files: BTreeMap<String, String>,
    commands: Vec<Vec<String>>,
    runtimes: BTreeMap<String, Runtime>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandReceipt {
    arguments: Vec<String>,
    program: Artifact,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    wall_seconds: f64,
    exit_code: Option<i32>,
    signal: Option<i32>,
    stdout: Artifact,
    stderr: Artifact,
}
fn equal(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        fs::metadata(path)?.len() <= 256 * 1024 * 1024,
        "export receipt exceeds 256 MiB"
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn campaign(prepared: &PreparedExport) -> Result<(CampaignPlan, PathBuf)> {
    let plan: CampaignPlan = read(&prepared.campaign_plan.path)?;
    let journal = prepared
        .original_outcome
        .path
        .parent()
        .and_then(Path::parent)
        .context("original journal missing")?
        .to_path_buf();
    ensure!(
        prepared.original_outcome.path
            == journal
                .join("finished")
                .join(format!("{}.json", prepared.original_job.identity()?)),
        "original journal changed"
    );
    Ok((plan, journal))
}
fn verify_plan(plan: &RunPlan, prepared: &PreparedExport) -> Result<()> {
    ensure!(
        plan.schema == "jope.fitted-export-run/v1"
            && plan.prepared_sha256 == prepared.identity()?,
        "export run binding changed"
    );
    plan.prepared.verify()?;
    plan.tools.verify()?;
    ensure!(
        equal(&read::<PreparedExport>(&plan.prepared.path)?, prepared)?
            && equal(
                &read::<BTreeMap<String, Runtime>>(&plan.tools.path)?,
                &plan.runtimes
            )?,
        "export input inventory changed"
    );
    Ok(())
}
fn needed(commands: &[Vec<String>]) -> impl Iterator<Item = &str> {
    commands
        .iter()
        .filter_map(|args| args.first().map(String::as_str))
        .filter(|name| *name != "./qualification")
}
fn unavailable(plan: &RunPlan, commands: &[Vec<String>]) -> Option<String> {
    for name in needed(commands) {
        let Some(runtime) = plan.runtimes.get(name) else {
            return Some(format!("tool {name} has no pinned inventory"));
        };
        if let Some(reason) = &runtime.unavailable {
            return Some(format!("{name}: {reason}"));
        }
        if runtime.version.trim().is_empty() || !runtime.argument_prefix.is_empty() {
            return Some(format!(
                "{name}: missing version or unsupported argument prefix"
            ));
        }
        for item in std::iter::once(&runtime.program)
            .chain(&runtime.inputs)
            .chain(&runtime.package_files)
        {
            if let Err(error) = item.verify() {
                return Some(format!("{name}: {error:#}"));
            }
        }
    }
    None
}
fn derived(
    plan: &RunPlan,
    prepared: &PreparedExport,
    entry: &Entry,
) -> Result<(WorkerInput, WorkerJob)> {
    let Entry::Prepared { qualification, .. } = entry else {
        anyhow::bail!("unsupported export has no execution job")
    };
    let (campaign, journal) = campaign(prepared)?;
    let mut runtimes = BTreeMap::new();
    for name in needed(&qualification.commands) {
        runtimes.insert(
            name.to_owned(),
            plan.runtimes
                .get(name)
                .context("missing tool inventory")?
                .clone(),
        );
    }
    let input = WorkerInput {
        schema: "jope.fitted-export-worker/v1".into(),
        files: qualification.files.clone(),
        commands: qualification.commands.clone(),
        runtimes,
    };
    let mut request = prepared.observation.request.clone();
    request.stage = EvaluationStage::Diagnostic;
    request.recipe_sha256 = qualification.identity()?;
    request.evaluator_sha256 = campaign::digest(&(&plan.tools, &input))?;
    request.numerical_policy = qualification.backend.numerical_policy();
    let id = request.identity()?;
    let input_path = journal.join("export-inputs").join(format!("{id}.json"));
    // write_private uses pretty JSON with a final newline.
    let mut bytes = serde_json::to_vec_pretty(&input)?;
    bytes.push(b'\n');
    let mut inputs = vec![
        Artifact {
            path: input_path.clone(),
            sha256: super::methods::digest(&bytes),
        },
        plan.tools.clone(),
    ];
    for runtime in input.runtimes.values() {
        inputs.push(runtime.program.clone());
        inputs.extend(runtime.inputs.clone());
        inputs.extend(runtime.package_files.clone());
    }
    inputs.sort_by(|a, b| a.path.cmp(&b.path));
    inputs.dedup_by(|a, b| a.path == b.path && a.sha256 == b.sha256);
    let mut args = campaign.runtimes["rust"].argument_prefix.clone();
    args.extend([
        "fitted-export-worker".into(),
        "--input".into(),
        input_path.display().to_string(),
        "--output".into(),
        "predictions.json".into(),
    ]);
    let mut outputs: Vec<PathBuf> = vec!["predictions.json".into(), "commands.json".into()];
    outputs.extend(input.files.keys().map(PathBuf::from));
    for index in 0..input.commands.len() {
        outputs.extend([
            format!("command-{index}.stdout").into(),
            format!("command-{index}.stderr").into(),
        ]);
    }
    match qualification.backend {
        inference::Backend::Rust => outputs.push("qualification".into()),
        inference::Backend::Cpp17 | inference::Backend::Cpp20 => outputs.push("model.so".into()),
        _ => {}
    }
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program: prepared.program.clone(),
            args,
            inputs,
            environment: BTreeMap::from([
                ("PATH".into(), "/usr/bin:/bin".into()),
                ("LC_ALL".into(), "C.UTF-8".into()),
            ]),
        },
        directory: journal.join("jobs").join(id),
        outputs,
        arm: None,
    };
    job.identity()?;
    for runtime in input.runtimes.values() {
        let mut environment_guard = job.clone();
        environment_guard.command.environment = runtime.environment.clone();
        environment_guard.identity()?;
    }
    Ok((input, job))
}
fn input_path(job: &WorkerJob) -> Result<&Path> {
    job.command
        .inputs
        .iter()
        .find(|item| {
            item.path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|n| n == "export-inputs")
        })
        .map(|item| item.path.as_path())
        .context("worker payload missing")
}
fn original_receipt(journal: &Path, job: &WorkerJob) -> Result<Option<WorkerOutcome>> {
    let path = journal
        .join("finished")
        .join(format!("{}.json", job.identity()?));
    if !path.try_exists()? {
        return Ok(None);
    }
    let outcome: WorkerOutcome = read(&path)?;
    ensure!(
        outcome.job_sha256 == job.identity()?
            && outcome.request_sha256 == job.request.identity()?,
        "qualification execution identity changed"
    );
    outcome.verify_outputs()?;
    Ok(Some(outcome))
}
fn backend_report(plan: &RunPlan, prepared: &PreparedExport, entry: &Entry) -> Result<Value> {
    let Entry::Prepared {
        backend,
        qualification,
        ..
    } = entry
    else {
        let Entry::Unsupported { backend, reason } = entry else {
            unreachable!()
        };
        return Ok(json!({"backend":backend,"status":"unsupported","reason":reason}));
    };
    let dependency_failure = unavailable(plan, &qualification.commands);
    if needed(&qualification.commands).any(|name| !plan.runtimes.contains_key(name)) {
        return Ok(
            json!({"backend":backend,"status":"unavailable","reason":dependency_failure,"cost":null}),
        );
    }
    let (input, job) = derived(plan, prepared, entry)?;
    let (_, journal) = campaign(prepared)?;
    let id = job.identity()?;
    let Some(outcome) = original_receipt(&journal, &job)? else {
        let started_path = journal.join("started").join(format!("{id}.json"));
        let started = started_path.try_exists()?;
        if started {
            let value: Value = read(&started_path)?;
            ensure!(
                value["job_sha256"] == id && value["job"] == serde_json::to_value(&job)?,
                "incomplete qualification lost started identity"
            );
        }
        return Ok(
            json!({"backend":backend,"status":if started {"unknown_started_without_finished"} else if dependency_failure.is_some() {"unavailable"} else {"unattempted"},"reason":dependency_failure,"job_sha256":id,"cost":null}),
        );
    };
    ensure!(
        equal(&read::<WorkerInput>(input_path(&job)?)?, &input)?,
        "qualification payload changed"
    );
    let started = journal.join("started").join(format!("{id}.json"));
    if started.try_exists()? {
        let value: Value = read(&started)?;
        ensure!(
            value["job_sha256"] == id && value["job"] == serde_json::to_value(&job)?,
            "started qualification changed"
        );
    } else {
        ensure!(
            outcome.command_status != EvaluationStatus::Completed
                && outcome.consumption.gate_released_unix_ms.is_none()
                && outcome.outputs.is_empty()
                && outcome.exit_code.is_none()
                && outcome.signal.is_none()
                && outcome.consumption.cpu_seconds.is_none_or(|v| v == 0.)
                && outcome.consumption.wall_seconds.is_none_or(|v| v == 0.),
            "launched qualification lost started receipt"
        );
    }
    ensure!(
        [
            outcome.consumption.cpu_seconds,
            outcome.consumption.wall_seconds
        ]
        .into_iter()
        .flatten()
        .all(|v| v.is_finite() && v >= 0.),
        "invalid qualification execution cost"
    );
    let mut result = json!({"backend":backend,"status":"failed","execution_status":outcome.command_status,"reason":outcome.reason,"job_sha256":id,"request_sha256":job.request.identity()?,"outcome":fitted_export::artifact(&journal.join("finished").join(format!("{id}.json")))?,"cost":outcome.consumption});
    if outcome.command_status != EvaluationStatus::Completed {
        return Ok(result);
    }
    ensure!(
        dependency_failure.is_none(),
        "completed qualification has stale tool dependencies: {dependency_failure:?}"
    );
    super::receipt_timing::completed_fit(&journal, &job, &outcome)?;
    let dependencies =
        jope_pipeline::execution::inventory_dependencies(std::slice::from_ref(&job))?;
    ensure!(
        dependencies.len() == 1 && dependencies[0].available,
        "qualification input closure changed"
    );
    let commands: Vec<CommandReceipt> = read(&job.directory.join("commands.json"))?;
    ensure!(
        commands.len() == qualification.commands.len(),
        "incomplete compiler/predictor commands"
    );
    for (index, (actual, expected)) in commands.iter().zip(&qualification.commands).enumerate() {
        ensure!(
            actual.stdout.path == job.directory.join(format!("command-{index}.stdout"))
                && actual.stderr.path == job.directory.join(format!("command-{index}.stderr")),
            "compiler/predictor log location changed"
        );
        ensure!(
            &actual.arguments == expected
                && actual.exit_code == Some(0)
                && actual.signal.is_none()
                && actual.finished_unix_ms >= actual.started_unix_ms
                && actual.wall_seconds.is_finite()
                && actual.wall_seconds >= 0.,
            "compiler/predictor receipt changed"
        );
        let program = if expected[0] == "./qualification" {
            fitted_export::artifact(&job.directory.join("qualification"))?
        } else {
            input.runtimes[&expected[0]].program.clone()
        };
        ensure!(
            equal(&actual.program, &program)?,
            "executed compiler/predictor differs"
        );
        actual.program.verify()?;
        actual.stdout.verify()?;
        actual.stderr.verify()?;
        ensure!(
            actual.started_unix_ms
                >= outcome
                    .consumption
                    .gate_released_unix_ms
                    .context("qualification gate missing")?
                && actual.finished_unix_ms
                    <= outcome
                        .consumption
                        .observed_exit_unix_ms
                        .context("qualification exit missing")?,
            "compiler/predictor escaped bounded worker"
        );
    }
    for (name, source) in &input.files {
        ensure!(
            fs::read(job.directory.join(name))? == source.as_bytes(),
            "executed public model or unlabeled input changed"
        );
    }
    let output = fs::read(job.directory.join("predictions.json"))?;
    ensure!(
        super::methods::digest(&output) == commands.last().context("zero commands")?.stdout.sha256,
        "prediction output differs from executed program stdout"
    );
    match inference::verify_qualification(qualification, &qualification.identity()?, &output) {
        Ok(parity) => {
            result["status"] = json!("qualified");
            result["parity"] = serde_json::to_value(parity)?;
            result["bounded_execution_verified"] = json!(true);
        }
        Err(error) => {
            result["status"] = json!("failed_parity");
            result["reason"] = json!(format!("{error:#}"));
        }
    }
    Ok(result)
}
pub fn recover(directory: &Path, prepared: &PreparedExport) -> Result<Value> {
    let path = directory.join("run.json");
    let mut reports = Vec::new();
    if path.try_exists()? {
        let plan: RunPlan = read(&path)?;
        verify_plan(&plan, prepared)?;
        let mut previous_settled = true;
        for entry in &prepared.entries {
            let mut report = backend_report(&plan, prepared, entry)?;
            if report["status"] == "qualified" {
                ensure!(
                    previous_settled,
                    "qualification order lacks prior backend evidence"
                );
            }
            if !previous_settled && report["status"] == "unattempted" {
                report["reason"] =
                    json!("prior backend is not qualified or explicitly unsupported");
            }
            previous_settled &=
                report["status"] == "qualified" || report["status"] == "unsupported";
            reports.push(report);
        }
    } else {
        reports = prepared.entries.iter().map(|entry|match entry {Entry::Unsupported{backend,reason}=>json!({"backend":backend,"status":"unsupported","reason":reason}),Entry::Prepared{backend,qualification,..}=>json!({"backend":backend,"status":"unattempted","qualification_sha256":qualification.identity().ok(),"cost":null})}).collect();
    }
    let (plan, journal) = campaign(prepared)?;
    Ok(
        json!({"all_backends_qualified":reports.iter().all(|r|r["status"]=="qualified"),"backends":reports,"experiment_sha256":plan.experiment.identity()?,"original_window":read::<Value>(&journal.join("window.json"))?,"budget":"original campaign execution cutoff; no additional allowance","cost_scope":if plan.scope=="asm-panel" {"qualification costs are a subset of this campaign's worker command accounting; do not add twice"} else {"qualification costs are additional to comparator fit-observation accounting"},"preparation_cost":{"wall_seconds":prepared.preparation_wall_seconds,"cpu_seconds":prepared.preparation_cpu_seconds,"scope":"preparation through public materialization and verification is additional to worker counters; final receipt publication excluded","receipt_publication_cpu_seconds":null},"controller_cpu_seconds":null,"fits":0}),
    )
}
fn may_submit(status: &Value) -> bool {
    status == "unattempted" || status == "unknown_started_without_finished"
}

pub fn run(args: QualifyArgs) -> Result<Value> {
    let prepared_artifact = fitted_export::artifact(&args.prepared)?;
    let prepared: PreparedExport = read(&prepared_artifact.path)?;
    ensure!(
        prepared.identity()? == args.prepared_sha256,
        "prepared export identity changed"
    );
    prepared.verify()?;
    let tools = fitted_export::artifact(&args.tools)?;
    ensure!(
        tools.sha256 == args.tools_sha256,
        "tool inventory checksum changed"
    );
    let mut required = args
        .require
        .iter()
        .map(|name| backend_named(name))
        .collect::<Result<Vec<_>>>()?;
    required.sort_by_key(|backend| fitted_export::ORDER.iter().position(|b| b == backend));
    required.dedup();
    let plan = RunPlan {
        schema: "jope.fitted-export-run/v1".into(),
        prepared: prepared_artifact.clone(),
        prepared_sha256: args.prepared_sha256.clone(),
        runtimes: read(&tools.path)?,
        tools,
        required,
    };
    verify_plan(&plan, &prepared)?;
    let (campaign, journal) = campaign(&prepared)?;
    let directory = fitted_export::location(&journal, &prepared.observation)?;
    ensure!(
        prepared_artifact.path == directory.join("prepared.json"),
        "export is outside the original campaign"
    );
    ensure!(
        equal(
            &fitted_export::artifact(&std::env::current_exe()?)?,
            &prepared.program
        )?,
        "running CLI differs from original fitted source"
    );
    campaign::immutable(&directory.join("run.json"), &plan)?;
    if prepared
        .entries
        .iter()
        .all(|entry| matches!(entry, Entry::Unsupported { .. }))
    {
        return Ok(
            json!({"ok":true,"execution_started":false,"report":recover(&directory,&prepared)?}),
        );
    }
    if let Some(unit) = args.supervisor_unit {
        let mut pool = ExecutionPool::open(
            &campaign.experiment,
            &journal,
            jope_pipeline::supervisor::for_unit(&unit)?,
        )?;
        let cancellation = jope_pipeline::supervisor::cancellation()?;
        prepared.verify_executable_plans()?;
        for entry in &prepared.entries {
            if matches!(entry, Entry::Unsupported { .. }) {
                continue;
            }
            let report = backend_report(&plan, &prepared, entry)?;
            if report["status"] == "qualified" {
                continue;
            }
            if !may_submit(&report["status"]) {
                break;
            }
            let (input, job) = derived(&plan, &prepared, entry)?;
            fs::create_dir_all(journal.join("export-inputs"))?;
            campaign::immutable(input_path(&job)?, &input)?;
            pool.run(vec![job], &cancellation)?;
            if backend_report(&plan, &prepared, entry)?["status"] != "qualified" {
                break;
            }
        }
        // W9-T05: every qualified backend is sealed, installed and recorded from its
        // authenticated receipts; required and optional outcomes are assessed, never released.
        let records = record_packages(&directory, &plan, &prepared)?;
        let assessment = assess_records(&directory, &plan, &prepared)?;
        let report = recover(&directory, &prepared)?;
        return Ok(
            json!({"ok":report["backends"].as_array().context("missing backends")?.iter().all(|r|r["status"]=="qualified" || r["status"]=="unsupported"),"report":report,"packages":records,"package_assessment":assessment}),
        );
    }
    let mut arguments = campaign.runtimes["rust"].argument_prefix.clone();
    arguments.extend([
        "qualify-export".into(),
        "--prepared".into(),
        prepared_artifact.path.display().to_string(),
        "--prepared-sha256".into(),
        args.prepared_sha256,
        "--tools".into(),
        plan.tools.path.display().to_string(),
        "--tools-sha256".into(),
        args.tools_sha256,
    ]);
    let launch = jope_pipeline::supervisor::launch(
        &campaign.experiment,
        &journal,
        &prepared.program,
        &arguments,
    )?;
    let report = recover(&directory, &prepared)?;
    let complete = report["backends"]
        .as_array()
        .context("missing backends")?
        .iter()
        .all(|r| r["status"] == "qualified" || r["status"] == "unsupported");
    Ok(json!({"ok":launch.exit_code==Some(0) && complete,"launch":launch,"report":report}))
}
fn normal(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        && path.components().count() == 1
}
fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
/// This endpoint is always scheduled by ExecutionPool: its descendants inherit
/// the same worker cgroup, one-thread environment, memory cap and cancellation.
pub fn worker(input: &Path, output: &Path) -> Result<()> {
    ensure!(
        output == Path::new("predictions.json"),
        "unexpected qualification output"
    );
    let input: WorkerInput = read(input)?;
    ensure!(
        input.schema == "jope.fitted-export-worker/v1" && !input.commands.is_empty(),
        "invalid qualification worker input"
    );
    for (name, source) in &input.files {
        ensure!(
            normal(name)
                && ![
                    "commands.json",
                    "predictions.json",
                    "stdout.log",
                    "stderr.log"
                ]
                .contains(&name.as_str()),
            "unsafe generated filename"
        );
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(name)?;
        file.write_all(source.as_bytes())?;
    }
    let directory = std::env::current_dir()?;
    let mut receipts = Vec::new();
    for (index, args) in input.commands.iter().enumerate() {
        ensure!(!args.is_empty(), "empty generated command");
        let mut command;
        let program;
        if args[0] == "./qualification" {
            ensure!(
                index > 0 && args.len() == 1,
                "unexpected generated executable"
            );
            program = fitted_export::artifact(&directory.join("qualification"))?;
            command = Command::new(&program.path);
        } else {
            let runtime = input
                .runtimes
                .get(&args[0])
                .context("unregistered generated command")?;
            ensure!(
                runtime.unavailable.is_none() && runtime.argument_prefix.is_empty(),
                "unavailable generated command"
            );
            for item in std::iter::once(&runtime.program)
                .chain(&runtime.inputs)
                .chain(&runtime.package_files)
            {
                item.verify()?;
            }
            program = runtime.program.clone();
            command = Command::new(&program.path);
            command.envs(&runtime.environment);
        }
        let stdout = directory.join(format!("command-{index}.stdout"));
        let stderr = directory.join(format!("command-{index}.stderr"));
        command
            .args(&args[1..])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&stdout)?)
            .stderr(fs::File::create(&stderr)?);
        let started_unix_ms = now_ms()?;
        let clock = Instant::now();
        let status = command.status()?;
        let finished_unix_ms = now_ms()?;
        use std::os::unix::process::ExitStatusExt;
        receipts.push(CommandReceipt {
            arguments: args.clone(),
            program,
            started_unix_ms,
            finished_unix_ms,
            wall_seconds: clock.elapsed().as_secs_f64(),
            exit_code: status.code(),
            signal: status.signal(),
            stdout: fitted_export::artifact(&stdout)?,
            stderr: fitted_export::artifact(&stderr)?,
        });
        // Incremental failure evidence survives an unsuccessful later command.
        fs::write("commands.json", serde_json::to_vec_pretty(&receipts)?)?;
        ensure!(
            status.success(),
            "generated command {index} failed: {status}"
        );
        if index + 1 == input.commands.len() {
            fs::copy(stdout, output)?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interrupted_export_reenters_pool_recovery_without_retrying_terminal_results() {
        assert!(may_submit(&json!("unattempted")));
        assert!(may_submit(&json!("unknown_started_without_finished")));
        for status in [
            "qualified",
            "failed",
            "failed_parity",
            "unavailable",
            "unsupported",
        ] {
            assert!(!may_submit(&json!(status)));
        }
    }

    #[test]
    fn export_child_closure_omits_reference_predictions_and_targets() {
        let value = serde_json::to_value(WorkerInput {
            schema: "jope.fitted-export-worker/v1".into(),
            files: BTreeMap::new(),
            commands: vec![vec!["python3".into(), "qualification.py".into()]],
            runtimes: BTreeMap::new(),
        })
        .unwrap();
        assert_eq!(value.as_object().unwrap().len(), 4);
        assert!(value.get("expected").is_none());
        assert!(value.get("campaign").is_none());
    }
    #[test]
    fn export_generated_filenames_cannot_escape_or_be_nested() {
        for name in ["../model.py", "/model.py", "a/b.py", ".", ""] {
            assert!(!normal(name));
        }
        assert!(normal("model.py"));
    }
    #[test]
    fn generated_executable_is_not_a_toolchain_dependency() {
        assert_eq!(
            needed(&[
                vec!["rustc".into(), "qualification.rs".into()],
                vec!["./qualification".into()]
            ])
            .collect::<Vec<_>>(),
            vec!["rustc"]
        );
    }
}
