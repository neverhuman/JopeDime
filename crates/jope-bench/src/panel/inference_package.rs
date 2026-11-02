//! W9-T05 backend-specific fitted inference packages and their install/load verification.
//!
//! A package binds the exact shipping bytes of one emitted public bundle, plus any native
//! library built from it during qualification, to the fitted subject, the frozen native
//! reference, the qualification plan and result, and a phase-specific dependency profile.
//! It reuses the existing authorities: `PublicManifest`/`PublicBundle` for public source,
//! `QualificationPlan` for the private replay and `QualificationResult` for parity.
//!
//! Loading re-hashes the bytes actually present. A changed, missing, extra, linked or
//! substituted file, a stale package reference, a receipt for another backend, bundle or
//! fitted state, or a changed declared serving dependency refuses before anything is
//! served. Sealing and loading never emit, compile, fit or predict. A loaded package is
//! not a release decision; release remains a separate, explicitly authorized state.
use super::{
    digest, verify_plan_binding, Backend, PublicBundle, PublicManifest, QualificationPlan,
    QualificationResult,
};
use anyhow::{ensure, Context, Result};
use jope_pipeline::execution::Artifact;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const PACKAGE_SCHEMA: &str = "jope.fitted-inference-package/v1";
pub const PACKAGE_FILE: &str = "package.json";
const MAX_PACKAGE_FILE: u64 = 256 * 1024 * 1024;

/// Native shipping outputs built from public source during qualification.
pub fn built_files(backend: Backend) -> &'static [&'static str] {
    match backend {
        Backend::Cpp17 | Backend::Cpp20 => &["model.so"],
        _ => &[],
    }
}

/// Phase-specific dependencies. `build` tools produced native shipping bytes (or the
/// qualification harness) with exactly `build_commands`; `serve` tools must be present
/// and unchanged to load and predict. All are pinned by absolute path and digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyProfile {
    pub runtime: String,
    pub host_boundary: String,
    pub numerical_policy: String,
    pub build: BTreeMap<String, Artifact>,
    pub build_commands: Vec<Vec<String>>,
    pub serve: BTreeMap<String, Artifact>,
}

impl DependencyProfile {
    /// The build/serve split implied by a verified qualification plan. The final plan
    /// command is the predictor; the Rust harness binary is not a serving dependency.
    pub fn expected_tools(
        plan: &QualificationPlan,
    ) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
        let (predictor, build) = plan
            .commands
            .split_last()
            .context("qualification has no predictor command")?;
        let build = build
            .iter()
            .map(|argv| argv.first().cloned().context("empty build command"))
            .collect::<Result<BTreeSet<_>>>()?;
        let predictor = predictor.first().context("empty predictor command")?;
        let serve = if predictor == "./qualification" {
            BTreeSet::new()
        } else {
            BTreeSet::from([predictor.clone()])
        };
        Ok((build, serve))
    }
}

/// Which backends a frozen export request requires; the rest are optional.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageRequest {
    pub required: Vec<Backend>,
    pub optional: Vec<Backend>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageManifest {
    pub schema: String,
    pub backend: Backend,
    pub required: bool,
    pub fitted_sha256: String,
    pub bundle_sha256: String,
    pub public: PublicManifest,
    /// Digest of the native predictions frozen in the plan before lowering.
    pub native_reference_sha256: String,
    pub qualification_plan_sha256: String,
    pub qualification: QualificationResult,
    pub profile: DependencyProfile,
    /// Every shipping file other than `package.json`, by name and SHA-256.
    pub files: BTreeMap<String, String>,
}

/// A verified, read-only view of an installed package.
#[derive(Clone, Debug)]
pub struct LoadedPackage {
    pub identity: String,
    pub manifest: PackageManifest,
    pub bundle: PublicBundle,
}

fn cardinality(plan: &QualificationPlan) -> usize {
    match &plan.expected {
        super::Predictions::Values(values) => values.len(),
        super::Predictions::Probabilities { values, .. } => values.iter().map(Vec::len).sum(),
    }
}

/// Internal consistency of a manifest against its public bundle; shared by seal and load.
fn verify_bindings(manifest: &PackageManifest, bundle: &PublicBundle) -> Result<()> {
    ensure!(
        manifest.schema == PACKAGE_SCHEMA,
        "unsupported package schema"
    );
    bundle.validate()?;
    let backend = bundle.manifest.backend;
    let result = &manifest.qualification;
    ensure!(
        manifest.backend == backend
            && manifest.bundle_sha256 == bundle.identity()?
            && result.schema == "jope.fitted-qualification-result/v1"
            && result.backend == backend
            && result.bundle_sha256 == manifest.bundle_sha256
            && result.fitted_sha256 == manifest.fitted_sha256
            && result.plan_sha256 == manifest.qualification_plan_sha256
            && result.parity_policy == backend.parity_policy()
            && result.execution_receipt_required
            && result.values > 0
            && result.maximum_absolute_error.is_finite(),
        "qualification receipt is bound to a different backend, bundle, state or plan"
    );
    let profile = &manifest.profile;
    ensure!(
        profile.runtime == bundle.manifest.runtime
            && profile.host_boundary == bundle.manifest.host_boundary
            && profile.numerical_policy == bundle.manifest.numerical_policy,
        "dependency profile differs from the public backend contract"
    );
    let mut expected: BTreeSet<String> = bundle.files.keys().cloned().collect();
    expected.extend(built_files(backend).iter().map(|name| (*name).to_owned()));
    ensure!(
        manifest.files.keys().cloned().collect::<BTreeSet<_>>() == expected
            && !expected.contains(PACKAGE_FILE),
        "package shipping file inventory differs from its backend contract"
    );
    for (name, sha) in &bundle.manifest.files {
        ensure!(
            manifest.files.get(name) == Some(sha),
            "package public file differs from its bundle"
        );
    }
    Ok(())
}

/// Seal a package from an already executed qualification. `built` holds the native
/// shipping bytes and `executed_outputs` the digests recorded by that execution's
/// receipt; a built file that is not the one qualified cannot be sealed.
pub fn seal(
    bundle: &PublicBundle,
    plan: &QualificationPlan,
    result: &QualificationResult,
    profile: DependencyProfile,
    built: &BTreeMap<String, Vec<u8>>,
    executed_outputs: &BTreeMap<String, String>,
    required: bool,
) -> Result<(PackageManifest, BTreeMap<String, Vec<u8>>)> {
    verify_plan_binding(plan, bundle)?;
    ensure!(
        result.plan_sha256 == plan.identity()?
            && result.input_sha256 == plan.input_sha256
            && result.values == cardinality(plan),
        "qualification result is not this plan's complete replay"
    );
    let (build, serve) = DependencyProfile::expected_tools(plan)?;
    let (_, build_commands) = plan
        .commands
        .split_last()
        .context("qualification has no predictor command")?;
    ensure!(
        profile.build.keys().cloned().collect::<BTreeSet<_>>() == build
            && profile.serve.keys().cloned().collect::<BTreeSet<_>>() == serve
            && profile.build_commands == build_commands,
        "dependency profile differs from the executed qualification"
    );
    for tool in profile.build.values().chain(profile.serve.values()) {
        ensure!(tool.path.is_absolute(), "dependency path is not absolute");
        tool.verify()?;
    }
    let backend = bundle.manifest.backend;
    ensure!(
        built
            .keys()
            .map(String::as_str)
            .eq(built_files(backend).iter().copied()),
        "native shipping outputs differ from the backend contract"
    );
    let mut files: BTreeMap<String, Vec<u8>> = bundle
        .files
        .iter()
        .map(|(name, text)| (name.clone(), text.as_bytes().to_vec()))
        .collect();
    for (name, bytes) in built {
        ensure!(
            !bytes.is_empty() && executed_outputs.get(name) == Some(&digest(bytes)),
            "native shipping output was not the qualified output"
        );
        files.insert(name.clone(), bytes.clone());
    }
    let manifest = PackageManifest {
        schema: PACKAGE_SCHEMA.into(),
        backend,
        required,
        fitted_sha256: plan.fitted_sha256.clone(),
        bundle_sha256: bundle.identity()?,
        public: bundle.manifest.clone(),
        native_reference_sha256: digest(&serde_json::to_vec(&plan.expected)?),
        qualification_plan_sha256: plan.identity()?,
        qualification: result.clone(),
        profile,
        files: files
            .iter()
            .map(|(name, bytes)| (name.clone(), digest(bytes)))
            .collect(),
    };
    verify_bindings(&manifest, bundle)?;
    Ok((manifest, files))
}

/// The package identity of a sealed manifest: the digest of the exact `package.json` bytes
/// [`install`] writes and a later [`load`] must present.
pub fn identity(manifest: &PackageManifest) -> Result<String> {
    Ok(digest(&serde_json::to_vec_pretty(manifest)?))
}

/// W9-T05: load `directory` under a recorded identity only when that identity is the one
/// `expected` has, where `expected` is the manifest the caller re-derived by sealing from
/// authenticated qualification evidence. A record naming any other package is refused before
/// anything is read from the installation, so the identity is never taken on trust.
pub fn load_authenticated(
    directory: &Path,
    recorded: &str,
    expected: &PackageManifest,
) -> Result<LoadedPackage> {
    ensure!(
        recorded == identity(expected)?,
        "package record is not the one its authenticated qualification derives"
    );
    load(directory, recorded)
}

/// Install a sealed package into a new empty directory. Returns the package identity,
/// the digest of the exact `package.json` bytes that a later load must present.
pub fn install(
    directory: &Path,
    manifest: &PackageManifest,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<String> {
    use std::io::Write;
    ensure!(
        files.keys().eq(manifest.files.keys())
            && files
                .iter()
                .all(|(name, bytes)| manifest.files[name] == digest(bytes)),
        "installed bytes differ from the sealed manifest"
    );
    fs::create_dir(directory)?;
    let bytes = serde_json::to_vec_pretty(manifest)?;
    debug_assert_eq!(digest(&bytes), identity(manifest)?);
    for (name, content) in files
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .chain(std::iter::once((PACKAGE_FILE, bytes.as_slice())))
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(name))?;
        file.write_all(content)?;
        file.sync_all()?;
    }
    Ok(digest(&bytes))
}

fn read_regular(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.file_type().is_file() && metadata.len() <= MAX_PACKAGE_FILE,
        "package entry is not a bounded regular file: {}",
        path.display()
    );
    Ok(fs::read(path)?)
}

/// Verify an installed package before serving. `identity` is the reference retained
/// from sealing/release; the bytes present must reproduce it and every binding.
pub fn load(directory: &Path, identity: &str) -> Result<LoadedPackage> {
    let bytes = read_regular(&directory.join(PACKAGE_FILE))?;
    ensure!(
        digest(&bytes) == identity,
        "stale or substituted package reference"
    );
    let manifest: PackageManifest =
        serde_json::from_slice(&bytes).context("decode package manifest")?;
    let mut present = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let name = entry?.file_name();
        present.insert(name.to_str().context("non-UTF-8 package entry")?.to_owned());
    }
    let mut expected: BTreeSet<String> = manifest.files.keys().cloned().collect();
    expected.insert(PACKAGE_FILE.into());
    ensure!(
        present == expected,
        "installed files differ from the package: missing {:?}, unexpected {:?}",
        expected.difference(&present).collect::<Vec<_>>(),
        present.difference(&expected).collect::<Vec<_>>()
    );
    let mut public = BTreeMap::new();
    for (name, sha) in &manifest.files {
        let content = read_regular(&directory.join(name))?;
        ensure!(digest(&content) == *sha, "shipping file changed: {name}");
        if manifest.public.files.contains_key(name) {
            public.insert(name.clone(), String::from_utf8(content)?);
        }
    }
    let bundle = PublicBundle {
        manifest: manifest.public.clone(),
        files: public,
    };
    verify_bindings(&manifest, &bundle)?;
    for tool in manifest.profile.serve.values() {
        tool.verify().context("serving dependency changed")?;
    }
    Ok(LoadedPackage {
        identity: identity.to_owned(),
        manifest,
        bundle,
    })
}

/// Separate required from optional outcomes. Any required backend that did not load
/// blocks; optional failures leave an explicitly partial result. Nothing here promotes.
pub fn assess(
    request: &PackageRequest,
    outcomes: &[(Backend, std::result::Result<LoadedPackage, String>)],
) -> Result<serde_json::Value> {
    let requested: Vec<&Backend> = request.required.iter().chain(&request.optional).collect();
    for (index, backend) in requested.iter().enumerate() {
        ensure!(
            !requested[..index].contains(backend),
            "backend requested twice"
        );
    }
    ensure!(
        outcomes
            .iter()
            .all(|(backend, _)| requested.contains(&backend)),
        "outcome for a backend outside the frozen request"
    );
    let status = |backend: &Backend| -> Result<std::result::Result<(), String>> {
        let matches: Vec<_> = outcomes.iter().filter(|(b, _)| b == backend).collect();
        ensure!(matches.len() <= 1, "duplicate backend outcome");
        Ok(match matches.first() {
            None => Err("absent".into()),
            Some((_, Ok(package))) => {
                ensure!(
                    package.manifest.backend == *backend
                        && package.manifest.required == request.required.contains(backend),
                    "package backend or requirement differs from the frozen request"
                );
                Ok(())
            }
            Some((_, Err(reason))) => Err(reason.clone()),
        })
    };
    let mut required_failures = Vec::new();
    let mut optional_failures = Vec::new();
    for backend in &request.required {
        if let Err(reason) = status(backend)? {
            required_failures.push(serde_json::json!({"backend":backend,"reason":reason}));
        }
    }
    for backend in &request.optional {
        if let Err(reason) = status(backend)? {
            optional_failures.push(serde_json::json!({"backend":backend,"reason":reason}));
        }
    }
    let state = if !required_failures.is_empty() {
        "blocked"
    } else if !optional_failures.is_empty() {
        "partial"
    } else {
        "complete"
    };
    Ok(
        serde_json::json!({"state":state,"required_failures":required_failures,
        "optional_failures":optional_failures,"release":"not-decided"}),
    )
}

#[cfg(test)]
#[path = "inference_package/tests.rs"]
mod tests;
