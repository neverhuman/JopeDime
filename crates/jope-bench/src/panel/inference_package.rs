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
mod tests {
    use super::super::tests::{emitted, fixture};
    use super::super::{qualification_plan, verify_qualification, FittedRustArtifact, Task};
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    fn which(name: &str) -> PathBuf {
        let path = std::env::var_os("PATH").expect("PATH");
        std::env::split_paths(&path)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
            .map(|found| fs::canonicalize(found).unwrap())
            .unwrap_or_else(|| panic!("required tool {name} unavailable"))
    }
    fn pin(path: &Path) -> Artifact {
        Artifact {
            path: path.to_owned(),
            sha256: digest(&fs::read(path).unwrap()),
        }
    }
    /// A labelled change applied to a copy of an installed package directory.
    type Mutation = (&'static str, Box<dyn Fn(&Path)>);
    struct Qualified {
        bundle: PublicBundle,
        plan: QualificationPlan,
        result: QualificationResult,
        built: BTreeMap<String, Vec<u8>>,
        executed: BTreeMap<String, String>,
    }
    /// Execute the private plan exactly as the qualification route does and keep the
    /// digests its execution produced for native shipping outputs.
    fn qualify(root: &Path, artifact: &FittedRustArtifact, backend: Backend) -> Qualified {
        let bundle = emitted(artifact, backend);
        let rows = vec![
            vec![Some("3.25".into()), Some("NA".into())],
            vec![None, None],
            vec![Some("-1".into()), Some("unseen".into())],
        ];
        let plan = qualification_plan(artifact, &bundle, &rows).unwrap();
        let directory = root.join(format!("execute-{backend:?}-{}", &plan.fitted_sha256[..8]));
        fs::create_dir_all(&directory).unwrap();
        for (name, text) in &plan.files {
            fs::write(directory.join(name), text).unwrap();
        }
        let mut output = Vec::new();
        for argv in &plan.commands {
            let run = Command::new(&argv[0])
                .args(&argv[1..])
                .current_dir(&directory)
                .output()
                .unwrap();
            assert!(
                run.status.success(),
                "{}",
                String::from_utf8_lossy(&run.stderr)
            );
            output = run.stdout;
        }
        let result = verify_qualification(&plan, &plan.identity().unwrap(), &output).unwrap();
        let mut built = BTreeMap::new();
        let mut executed = BTreeMap::new();
        for name in built_files(backend) {
            let bytes = fs::read(directory.join(name)).unwrap();
            executed.insert((*name).to_owned(), digest(&bytes));
            built.insert((*name).to_owned(), bytes);
        }
        Qualified {
            bundle,
            plan,
            result,
            built,
            executed,
        }
    }
    fn profile(q: &Qualified) -> DependencyProfile {
        let (build, serve) = DependencyProfile::expected_tools(&q.plan).unwrap();
        DependencyProfile {
            runtime: q.bundle.manifest.runtime.clone(),
            host_boundary: q.bundle.manifest.host_boundary.clone(),
            numerical_policy: q.bundle.manifest.numerical_policy.clone(),
            build: build.iter().map(|n| (n.clone(), pin(&which(n)))).collect(),
            build_commands: q.plan.commands[..q.plan.commands.len() - 1].to_vec(),
            serve: serve.iter().map(|n| (n.clone(), pin(&which(n)))).collect(),
        }
    }
    fn seal_default(q: &Qualified, required: bool) -> (PackageManifest, BTreeMap<String, Vec<u8>>) {
        seal(
            &q.bundle,
            &q.plan,
            &q.result,
            profile(q),
            &q.built,
            &q.executed,
            required,
        )
        .unwrap()
    }
    fn copy(from: &Path, to: &Path) {
        fs::create_dir(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    fn scratch() -> tempfile::TempDir {
        let scratch =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
        fs::create_dir_all(&scratch).unwrap();
        tempfile::tempdir_in(scratch).unwrap()
    }
    /// Loaded package must serve the qualified predictions from its installed bytes.
    fn serve_python(directory: &Path, module: &str) -> Vec<u8> {
        let run = Command::new("python3")
            .arg("-c")
            .arg(format!("import json,{module};print(json.dumps({module}.predict_raw([['3.25','NA'],[None,None],['-1','unseen']]),ensure_ascii=False,allow_nan=False))"))
            .current_dir(directory)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        run.stdout
    }

    #[test]
    fn intact_packages_load_and_tampered_installations_refuse_before_serving() {
        let root = scratch();
        let (a, _) = fixture(Task::Regression, "ridge");
        let (b, _) = fixture(Task::Binary, "logit_ridge");
        let a_bytes = a.bytes().unwrap();
        let mut refused = Vec::new();
        let mut accepted = Vec::new();
        for backend in [Backend::Python, Backend::Rust, Backend::Cpp17] {
            let q = qualify(root.path(), &a, backend);
            let (manifest, files) = seal_default(&q, true);
            let intact = root.path().join(format!("{backend:?}-intact"));
            let identity = install(&intact, &manifest, &files).unwrap();
            let loaded = load(&intact, &identity).unwrap();
            assert_eq!(loaded.manifest.fitted_sha256, q.plan.fitted_sha256);
            assert_eq!(
                loaded.bundle.identity().unwrap(),
                q.bundle.identity().unwrap()
            );
            if backend != Backend::Rust {
                // The installed bytes, not the build directory, serve the qualified output.
                let module = if backend == Backend::Python {
                    "model"
                } else {
                    "boundary"
                };
                let served = serve_python(&intact, module);
                assert_eq!(digest(&served), q.result.prediction_sha256);
            }
            let main = match backend {
                Backend::Python => "model.py",
                Backend::Rust => "model.rs",
                _ => "model.cpp",
            };
            let mut mutations: Vec<Mutation> = vec![
                (
                    "changed-source",
                    Box::new(move |d: &Path| {
                        let mut text = fs::read(d.join(main)).unwrap();
                        text.push(b' ');
                        fs::write(d.join(main), text).unwrap();
                    }),
                ),
                (
                    "missing-wrapper",
                    Box::new(move |d: &Path| {
                        let wrapper = if main == "model.cpp" {
                            "boundary.py"
                        } else {
                            main
                        };
                        fs::remove_file(d.join(wrapper)).unwrap();
                    }),
                ),
                (
                    "extra-file",
                    Box::new(|d: &Path| {
                        fs::write(d.join("sitecustomize.py"), "import os\n").unwrap()
                    }),
                ),
                (
                    "linked-file",
                    Box::new(move |d: &Path| {
                        // Identical bytes, but reached through a link outside the package.
                        let outside = d.with_extension("link-target");
                        fs::copy(d.join(main), &outside).unwrap();
                        fs::remove_file(d.join(main)).unwrap();
                        std::os::unix::fs::symlink(&outside, d.join(main)).unwrap();
                    }),
                ),
                (
                    "rebound-wrong-backend-receipt",
                    Box::new(|d: &Path| {
                        let mut value: serde_json::Value =
                            serde_json::from_slice(&fs::read(d.join(PACKAGE_FILE)).unwrap())
                                .unwrap();
                        value["qualification"]["backend"] = serde_json::json!("javascript");
                        fs::write(
                            d.join(PACKAGE_FILE),
                            serde_json::to_vec_pretty(&value).unwrap(),
                        )
                        .unwrap();
                    }),
                ),
                (
                    "rebound-other-state-receipt",
                    Box::new(|d: &Path| {
                        let mut value: serde_json::Value =
                            serde_json::from_slice(&fs::read(d.join(PACKAGE_FILE)).unwrap())
                                .unwrap();
                        value["qualification"]["fitted_sha256"] = serde_json::json!("0".repeat(64));
                        fs::write(
                            d.join(PACKAGE_FILE),
                            serde_json::to_vec_pretty(&value).unwrap(),
                        )
                        .unwrap();
                    }),
                ),
            ];
            if matches!(backend, Backend::Cpp17) {
                mutations.push((
                    "substituted-executable",
                    Box::new(|d: &Path| {
                        let mut bytes = fs::read(d.join("model.so")).unwrap();
                        let last = bytes.len() - 1;
                        bytes[last] ^= 1;
                        fs::write(d.join("model.so"), bytes).unwrap();
                    }),
                ));
                mutations.push((
                    "missing-stage-codec",
                    Box::new(|d: &Path| fs::remove_file(d.join("codec.json")).unwrap()),
                ));
                mutations.push((
                    "missing-native-library",
                    Box::new(|d: &Path| fs::remove_file(d.join("model.so")).unwrap()),
                ));
            }
            for (label, mutate) in &mutations {
                let directory = root.path().join(format!("{backend:?}-{label}"));
                copy(&intact, &directory);
                mutate(&directory);
                // A rebound manifest is presented with its own new digest, so the internal
                // bindings, not only the retained reference, must refuse it.
                let presented = if label.starts_with("rebound") {
                    digest(&fs::read(directory.join(PACKAGE_FILE)).unwrap())
                } else {
                    identity.clone()
                };
                match load(&directory, &presented) {
                    Ok(_) => accepted.push(format!("{backend:?}/{label}")),
                    Err(_) => refused.push(format!("{backend:?}/{label}")),
                }
            }
            // Stale reference: a different package installed where A's reference points.
            let other = qualify(root.path(), &b, backend);
            let (other_manifest, other_files) = seal_default(&other, true);
            let stale = root.path().join(format!("{backend:?}-stale"));
            install(&stale, &other_manifest, &other_files).unwrap();
            assert!(load(&stale, &identity).is_err());
            // Seal-time refusals: wrong-backend receipt, receipt from another state/plan,
            // unqualified native bytes and a profile that is not the executed one.
            let rust = qualify(
                root.path(),
                &a,
                if backend == Backend::Rust {
                    Backend::Python
                } else {
                    Backend::Rust
                },
            );
            assert!(seal(
                &q.bundle,
                &q.plan,
                &rust.result,
                profile(&q),
                &q.built,
                &q.executed,
                true
            )
            .is_err());
            assert!(seal(
                &q.bundle,
                &q.plan,
                &other.result,
                profile(&q),
                &q.built,
                &q.executed,
                true
            )
            .is_err());
            let mut changed = profile(&q);
            changed.build_commands.push(vec!["true".into()]);
            assert!(seal(
                &q.bundle,
                &q.plan,
                &q.result,
                changed,
                &q.built,
                &q.executed,
                true
            )
            .is_err());
            if backend == Backend::Cpp17 {
                let mut built = q.built.clone();
                built.get_mut("model.so").unwrap().push(0);
                assert!(seal(
                    &q.bundle,
                    &q.plan,
                    &q.result,
                    profile(&q),
                    &built,
                    &q.executed,
                    true
                )
                .is_err());
                assert!(seal(
                    &q.bundle,
                    &q.plan,
                    &q.result,
                    profile(&q),
                    &BTreeMap::new(),
                    &q.executed,
                    true
                )
                .is_err());
            }
            // The intact installation still loads; nothing above altered it.
            load(&intact, &identity).unwrap();
        }
        assert_eq!(
            a.bytes().unwrap(),
            a_bytes,
            "packaging is read-only for the fitted subject"
        );
        assert!(
            accepted.is_empty(),
            "accepted tampered packages: {accepted:?}"
        );
        assert_eq!(refused.len(), 6 + 6 + 9);
    }

    /// Reproduction of the gap this module closes: the existing public manifest binds only
    /// source files, so a substituted native library beside intact sources passes
    /// `PublicBundle::validate` and silently serves other predictions. `load` refuses it.
    #[test]
    fn public_manifest_alone_cannot_see_a_substituted_native_library() {
        let root = scratch();
        let (a, _) = fixture(Task::Regression, "ridge");
        let (b, _) = fixture(Task::Binary, "logit_ridge");
        let q = qualify(root.path(), &a, Backend::Cpp17);
        let other = qualify(root.path(), &b, Backend::Cpp17);
        let (manifest, files) = seal_default(&q, true);
        let installed = root.path().join("installed");
        let identity = install(&installed, &manifest, &files).unwrap();
        assert_eq!(
            digest(&serve_python(&installed, "boundary")),
            q.result.prediction_sha256
        );
        fs::write(installed.join("model.so"), &other.built["model.so"]).unwrap();
        let public = PublicBundle {
            manifest: manifest.public.clone(),
            files: q
                .bundle
                .files
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        fs::read_to_string(installed.join(name)).unwrap(),
                    )
                })
                .collect(),
        };
        public.validate().unwrap();
        assert_ne!(
            digest(&serve_python(&installed, "boundary")),
            q.result.prediction_sha256
        );
        let refused = load(&installed, &identity).unwrap_err();
        assert!(format!("{refused:#}").contains("shipping file changed: model.so"));
    }

    #[test]
    fn changed_serving_dependency_refuses_and_required_failures_block() {
        let root = scratch();
        let (a, _) = fixture(Task::Regression, "ridge");
        let q = qualify(root.path(), &a, Backend::Python);
        // A pinned stand-in for the declared interpreter, so it can be changed safely.
        let tool = root.path().join("python3-pinned");
        fs::write(&tool, b"pinned interpreter bytes").unwrap();
        let tool = fs::canonicalize(&tool).unwrap();
        let mut declared = profile(&q);
        declared.serve.insert("python3".into(), pin(&tool));
        let (manifest, files) = seal(
            &q.bundle,
            &q.plan,
            &q.result,
            declared,
            &q.built,
            &q.executed,
            true,
        )
        .unwrap();
        let python = root.path().join("python");
        let identity = install(&python, &manifest, &files).unwrap();
        let loaded = load(&python, &identity).unwrap();
        fs::write(&tool, b"changed interpreter bytes").unwrap();
        let refused = load(&python, &identity).unwrap_err();
        assert!(format!("{refused:#}").contains("serving dependency changed"));

        let js = qualify(root.path(), &a, Backend::JavaScript);
        let (manifest, files) = seal_default(&js, false);
        let javascript = root.path().join("javascript");
        let js_identity = install(&javascript, &manifest, &files).unwrap();
        let js_loaded = load(&javascript, &js_identity).unwrap();
        let request = PackageRequest {
            required: vec![Backend::Python],
            optional: vec![Backend::JavaScript],
        };
        let complete = assess(
            &request,
            &[
                (Backend::Python, Ok(loaded.clone())),
                (Backend::JavaScript, Ok(js_loaded.clone())),
            ],
        )
        .unwrap();
        assert_eq!(complete["state"], "complete");
        let partial = assess(
            &request,
            &[
                (Backend::Python, Ok(loaded.clone())),
                (Backend::JavaScript, Err("node absent".into())),
            ],
        )
        .unwrap();
        assert_eq!(partial["state"], "partial");
        assert_eq!(partial["optional_failures"][0]["backend"], "javascript");
        let blocked = assess(
            &request,
            &[
                (Backend::Python, Err(format!("{refused:#}"))),
                (Backend::JavaScript, Ok(js_loaded.clone())),
            ],
        )
        .unwrap();
        assert_eq!(blocked["state"], "blocked");
        // An absent required backend blocks; nothing is inferred from another backend.
        assert_eq!(
            assess(&request, &[(Backend::JavaScript, Ok(js_loaded.clone()))]).unwrap()["state"],
            "blocked"
        );
        // A package sealed as optional cannot satisfy a required slot, nor can a stray backend.
        let swapped = PackageRequest {
            required: vec![Backend::JavaScript],
            optional: vec![Backend::Python],
        };
        assert!(assess(&swapped, &[(Backend::JavaScript, Ok(js_loaded.clone()))]).is_err());
        assert!(assess(&request, &[(Backend::Rust, Err("x".into()))]).is_err());
        for value in [complete, partial, blocked] {
            assert_eq!(value["release"], "not-decided");
        }
    }
}
