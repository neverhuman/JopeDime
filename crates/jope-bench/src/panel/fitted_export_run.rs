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
mod run_support;
use run_support::{
    assess_records, backend_report, campaign, derived, equal, input_path, original_receipt, read,
    record_packages, verify_plan, CommandReceipt, WorkerInput,
};

mod runner;
pub use runner::{recover, run, worker};

#[cfg(test)]
mod tests;
