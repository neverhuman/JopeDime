//! Rust-first, content-addressed self-play runtime bundler.
//!
//! Operational Python is forbidden. The only Python copied into a production
//! bundle is an exact allowlist for the PyTorch training, proposal, and
//! dataset-encoding boundaries.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

const DEFAULT_SOURCE: &str = "/home/ubuntu/jope-dime";
const DEFAULT_PRIME: &str = "/home/ubuntu/jope-prime";
const DEFAULT_ENCODING_DRIVER: &str = "/home/ubuntu/jope-flywheel/encodings/_driver";
const DEFAULT_RUNNER: &str = "/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh";
const DEFAULT_SCQ_PROJECTS: &str = "/home/ubuntu/smart_cluster/etc/projects.toml";
const DEFAULT_OUTPUT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles";
const DEFAULT_SERVICE_MANIFEST: &str = "manifests/selfplay/service-manifest.v3.json";

/// These are the complete Python exception. Adding a file requires a code
/// review of this exact list and an ML-only role.
const PYTORCH_ALLOWLIST: &[(&str, &str, &str)] = &[
    (
        "source:scripts/test_exact_common.py",
        "pytorch/shared/test_exact_common.py",
        "pytorch_training_proposal_support",
    ),
    (
        "source:scripts/test_exact_dl_generate.py",
        "pytorch/proposals/test_exact_dl_generate.py",
        "pytorch_proposal_inference",
    ),
    (
        "source:scripts/test_exact_context_proposals.py",
        "pytorch/proposals/test_exact_context_proposals.py",
        "pytorch_proposal_inference",
    ),
    (
        "source:scripts/test_exact_canary_propose.py",
        "pytorch/proposals/test_exact_canary_propose.py",
        "pytorch_proposal_inference",
    ),
    (
        "source:scripts/test_exact_packed_corpus.py",
        "pytorch/training/test_exact_packed_corpus.py",
        "pytorch_training_support",
    ),
    (
        "source:scripts/test_exact_train.py",
        "pytorch/training/test_exact_train.py",
        "pytorch_training",
    ),
    (
        "prime:scripts/prime_policy_trainer.py",
        "pytorch/training/prime_policy_trainer.py",
        "pytorch_training",
    ),
    (
        "prime:scripts/ratchet_dl_update.py",
        "pytorch/training/ratchet_dl_update.py",
        "pytorch_training",
    ),
    (
        "encoding:encode_batch.py",
        "pytorch/encoding/encode_batch.py",
        "pytorch_dataset_encoding",
    ),
    (
        "encoding:foundation_backfill.py",
        "pytorch/encoding/foundation_backfill.py",
        "pytorch_dataset_encoding",
    ),
    (
        "prime:scripts/encode_external.py",
        "pytorch/encoding/encode_external.py",
        "pytorch_dataset_encoding",
    ),
];

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-bundle",
    about = "Build a Rust-first content-addressed self-play runtime bundle"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_SOURCE)]
    source: PathBuf,
    #[arg(long, default_value = DEFAULT_PRIME)]
    prime: PathBuf,
    #[arg(long, default_value = DEFAULT_ENCODING_DRIVER)]
    encoding_driver: PathBuf,
    #[arg(long, default_value = DEFAULT_RUNNER)]
    runner: PathBuf,
    #[arg(long, default_value = DEFAULT_SCQ_PROJECTS)]
    scq_projects: PathBuf,
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output_root: PathBuf,
    #[arg(long)]
    service_manifest: Option<PathBuf>,
    /// Atomically advance output_root/active.json after full verification.
    #[arg(long, conflicts_with = "activate_existing")]
    activate: bool,
    /// Re-verify and atomically activate an already built content hash without
    /// rebuilding from a source tree that may be changing concurrently.
    #[arg(long, value_name = "SHA256", conflicts_with = "activate")]
    activate_existing: Option<String>,
}

#[derive(Clone, Debug)]
struct Candidate {
    source: PathBuf,
    target: String,
    python_role: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
struct InventoryRow {
    source: String,
    target: String,
    sha256: String,
    bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_role: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServiceManifest {
    schema: String,
    binaries: Vec<String>,
    services: Vec<ServiceRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServiceRow {
    unit: String,
    binary: String,
    owner: String,
    active: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(bundle_sha256) = args.activate_existing.as_deref() {
        ensure!(
            bundle_sha256.len() == 64
                && bundle_sha256
                    .bytes()
                    .all(|byte| { byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) }),
            "activate-existing requires a lowercase SHA-256"
        );
        let manifest_path = args.output_root.join(bundle_sha256).join("bundle.json");
        let manifest: Value = serde_json::from_reader(
            File::open(&manifest_path)
                .with_context(|| format!("open staged bundle {}", manifest_path.display()))?,
        )?;
        verify_existing(
            manifest_path
                .parent()
                .context("staged bundle manifest has no parent")?,
            &manifest,
        )?;
        activate_manifest(&args.output_root, &manifest)?;
        println!("{}", serde_json::to_string(&manifest)?);
        return Ok(());
    }
    let manifest = build(&args)?;
    if args.activate {
        activate_manifest(&args.output_root, &manifest)?;
    }
    println!("{}", serde_json::to_string(&manifest)?);
    Ok(())
}

fn activate_manifest(output_root: &Path, manifest: &Value) -> Result<()> {
    let bundle_sha256 = manifest["bundle_sha256"].as_str().context("bundle hash")?;
    let manifest_path = output_root.join(bundle_sha256).join("bundle.json");
    let pointer = json!({
        "schema": "JopeSelfplayRuntimeBundlePointer.v2",
        "bundle": manifest_path,
        "bundle_manifest_sha256": sha256_file(&manifest_path)?,
        "bundle_sha256": bundle_sha256,
        "runtime": "rust",
    });
    atomic_json(&output_root.join("active.json"), &pointer)?;
    activate_current_bundle(output_root, bundle_sha256)
}

fn build(args: &Args) -> Result<Value> {
    let service_manifest_path = args
        .service_manifest
        .clone()
        .unwrap_or_else(|| args.source.join(DEFAULT_SERVICE_MANIFEST));
    let service_manifest: ServiceManifest =
        serde_json::from_reader(File::open(&service_manifest_path).with_context(|| {
            format!(
                "open declarative service manifest {}",
                service_manifest_path.display()
            )
        })?)?;
    validate_service_manifest(&service_manifest)?;
    let candidates = sources(args, &service_manifest, &service_manifest_path)?;
    let inventory = inventory(&candidates)?;
    let identity = inventory_identity(&inventory)?;
    let final_dir = args.output_root.join(&identity);
    let manifest_path = final_dir.join("bundle.json");
    if manifest_path.is_file() {
        let manifest: Value = serde_json::from_reader(File::open(&manifest_path)?)?;
        verify_existing(&final_dir, &manifest)?;
        return Ok(manifest);
    }

    fs::create_dir_all(&args.output_root)?;
    let temporary = args
        .output_root
        .join(format!(".{identity}.tmp-{}", std::process::id()));
    if temporary.exists() {
        bail!("temporary bundle already exists: {}", temporary.display());
    }
    fs::create_dir(&temporary)?;
    let result = (|| -> Result<Value> {
        for (candidate, row) in candidates.iter().zip(&inventory) {
            let destination = temporary.join(&row.target);
            ensure_bundle_target(&temporary, &destination)?;
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&candidate.source, &destination).with_context(|| {
                format!(
                    "copy {} to {}",
                    candidate.source.display(),
                    destination.display()
                )
            })?;
            ensure!(
                sha256_file(&destination)? == row.sha256,
                "bundle copy hash mismatch: {}",
                row.target
            );
        }
        let python_targets: Vec<_> = inventory
            .iter()
            .filter_map(|row| {
                row.python_role
                    .map(|role| json!({"target": row.target, "role": role}))
            })
            .collect();
        let active_units = service_manifest
            .services
            .iter()
            .filter(|row| row.active)
            .map(|row| row.unit.as_str())
            .collect::<Vec<_>>();
        let manifest = json!({
            "schema": "RuntimeBundle.v3",
            "bundle_sha256": identity,
            "runtime": "rust",
            "operational_python_allowed": false,
            "pytorch_python_allowlist": python_targets,
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-exact-reused",
            "active_units": active_units,
            "service_manifest": service_manifest,
            "service_manifest_sha256": sha256_file(&service_manifest_path)?,
            "files": inventory,
        });
        atomic_json(&temporary.join("bundle.json"), &manifest)?;
        fs::rename(&temporary, &final_dir)?;
        verify_existing(&final_dir, &manifest)?;
        Ok(manifest)
    })();
    if result.is_err() && temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    result
}

fn sources(
    args: &Args,
    service_manifest: &ServiceManifest,
    service_manifest_path: &Path,
) -> Result<Vec<Candidate>> {
    let mut selected = Vec::new();
    for name in ["Cargo.toml", "Cargo.lock"] {
        add(
            &mut selected,
            args.source.join(name),
            format!("runtime/{name}"),
            None,
        );
    }
    collect_extension(
        &args.source.join("src"),
        &args.source,
        "rs",
        "runtime",
        None,
        &mut selected,
    )?;
    for binary in &service_manifest.binaries {
        add(
            &mut selected,
            args.source.join("target/release").join(binary),
            format!("runtime/bin/{binary}"),
            None,
        );
    }
    collect_extension(
        &args.source.join("systemd"),
        &args.source,
        "service",
        "runtime",
        None,
        &mut selected,
    )?;
    collect_extension(
        &args.source.join("systemd"),
        &args.source,
        "timer",
        "runtime",
        None,
        &mut selected,
    )?;
    for config in [
        "manifests/selfplay/scq-nodes.v3.toml",
        "manifests/selfplay/scq-projects.v3.toml",
        "manifests/selfplay/freshness-mcp-runtime.v2.json",
        "manifests/selfplay/causal-shadow.v2.json",
    ] {
        add(
            &mut selected,
            args.source.join(config),
            format!("runtime/{config}"),
            None,
        );
    }
    add(
        &mut selected,
        service_manifest_path.to_path_buf(),
        format!("runtime/{DEFAULT_SERVICE_MANIFEST}"),
        None,
    );
    add(
        &mut selected,
        args.runner.clone(),
        "runtime/scripts/fweval_runner.sh".to_owned(),
        None,
    );
    add(
        &mut selected,
        args.scq_projects.clone(),
        "external/scq-projects.active.toml".to_owned(),
        None,
    );
    for config in [
        "prime.toml",
        "registry/operator_registry.json",
        "registry/compatibility/gen31_policy_v3_vocab.json",
        "registry/compatibility/prime_arena_suites_v1.json",
    ] {
        add(
            &mut selected,
            args.prime.join(config),
            config.to_owned(),
            None,
        );
        add(
            &mut selected,
            args.prime.join(config),
            format!("pytorch/{config}"),
            None,
        );
    }
    for (locator, target, role) in PYTORCH_ALLOWLIST {
        let source = resolve_locator(locator, &args.source, &args.prime, &args.encoding_driver)?;
        add(&mut selected, source, (*target).to_owned(), Some(*role));
    }
    add(
        &mut selected,
        args.encoding_driver.join("binexp/runner.sh"),
        "pytorch/encoding/runner.sh".to_owned(),
        None,
    );
    collect_extension(
        &args.prime.join("src/jope"),
        &args.prime.join("src"),
        "py",
        "pytorch",
        Some("pytorch_dependency_package"),
        &mut selected,
    )?;

    selected.retain(|row| row.source.is_file());
    selected.sort_by(|left, right| left.target.cmp(&right.target));
    let mut targets = BTreeSet::new();
    for row in &selected {
        ensure!(
            targets.insert(row.target.clone()),
            "duplicate bundle target {}",
            row.target
        );
        validate_python_candidate(row, &args.source, &args.prime, &args.encoding_driver)?;
        if row.target.ends_with(".service") {
            validate_systemd_unit(&fs::read_to_string(&row.source)?)
                .with_context(|| format!("validate {}", row.source.display()))?;
        }
        if row.target.ends_with(".sh") {
            let contents = fs::read_to_string(&row.source)?;
            if row.target == "pytorch/encoding/runner.sh" {
                validate_encoding_runner(&contents)
                    .with_context(|| format!("validate {}", row.source.display()))?;
            } else {
                validate_no_python_runtime(&contents)
                    .with_context(|| format!("validate {}", row.source.display()))?;
            }
        }
    }
    for binary in &service_manifest.binaries {
        ensure!(
            selected
                .iter()
                .any(|row| row.target == format!("runtime/bin/{binary}")),
            "required release binary is missing: {binary}; run cargo build --release --bins"
        );
    }
    Ok(selected)
}

fn add(
    selected: &mut Vec<Candidate>,
    source: PathBuf,
    target: String,
    python_role: Option<&'static str>,
) {
    selected.push(Candidate {
        source,
        target,
        python_role,
    });
}

fn collect_extension(
    directory: &Path,
    relative_root: &Path,
    extension: &str,
    target_prefix: &str,
    python_role: Option<&'static str>,
    selected: &mut Vec<Candidate>,
) -> Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(directory)?.collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_extension(
                &path,
                relative_root,
                extension,
                target_prefix,
                python_role,
                selected,
            )?;
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            let relative = path.strip_prefix(relative_root)?.to_path_buf();
            add(
                selected,
                path,
                format!("{target_prefix}/{}", relative.display()),
                python_role,
            );
        }
    }
    Ok(())
}

fn resolve_locator(
    locator: &str,
    source: &Path,
    prime: &Path,
    encoding_driver: &Path,
) -> Result<PathBuf> {
    if let Some(relative) = locator.strip_prefix("source:") {
        return Ok(source.join(relative));
    }
    if let Some(relative) = locator.strip_prefix("prime:") {
        return Ok(prime.join(relative));
    }
    if let Some(relative) = locator.strip_prefix("encoding:") {
        return Ok(encoding_driver.join(relative));
    }
    bail!("invalid Python allowlist locator {locator}")
}

fn validate_python_candidate(
    candidate: &Candidate,
    source: &Path,
    prime: &Path,
    encoding_driver: &Path,
) -> Result<()> {
    if candidate
        .source
        .extension()
        .and_then(|value| value.to_str())
        != Some("py")
        && !candidate.target.ends_with(".py")
    {
        ensure!(
            candidate.python_role.is_none(),
            "non-Python file has Python role: {}",
            candidate.target
        );
        return Ok(());
    }
    let allowlisted = PYTORCH_ALLOWLIST.iter().any(|(locator, target, role)| {
        resolve_locator(locator, source, prime, encoding_driver)
            .map(|allowed| {
                allowed == candidate.source
                    && *target == candidate.target
                    && Some(*role) == candidate.python_role
            })
            .unwrap_or(false)
    });
    let dependency_package = candidate.python_role == Some("pytorch_dependency_package")
        && candidate.source.starts_with(prime.join("src/jope"))
        && candidate.target.starts_with("pytorch/jope/")
        && candidate.target.ends_with(".py");
    ensure!(
        allowlisted || dependency_package,
        "operational or unallowlisted Python is forbidden: {}",
        candidate.target
    );
    Ok(())
}

fn validate_systemd_unit(contents: &str) -> Result<()> {
    for line in contents.lines().map(str::trim) {
        if line.starts_with("ExecStart=")
            || line.starts_with("ExecStartPre=")
            || line.starts_with("ExecStartPost=")
        {
            if line.contains("/jope_selfplay_supply ") {
                validate_supply_pytorch_boundary(line)?;
            } else {
                validate_no_python_runtime(line)?;
            }
            ensure!(
                !line.contains("/target-") && !line.contains("/home/ubuntu/jope-dime/target/"),
                "systemd runtime references a mutable or split target directory"
            );
            ensure!(
                !line.contains("/home/ubuntu/jope-prime/src")
                    && !line.contains("/home/ubuntu/jope-dime/scripts"),
                "systemd runtime references a live source directory"
            );
        }
    }
    Ok(())
}

fn validate_supply_pytorch_boundary(line: &str) -> Result<()> {
    let normalized = line.to_ascii_lowercase();
    ensure!(
        !normalized.contains("/usr/bin/python")
            && !normalized.contains("python3")
            && normalized.contains(
                "--proposal-generator /home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch/proposals/test_exact_context_proposals.py"
            )
            && normalized.contains(
                "--proposal-legacy-generator /home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch/proposals/test_exact_dl_generate.py"
            )
            && normalized.contains(
                "--proposal-common /home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch/shared/test_exact_common.py"
            )
            && normalized.contains(
                "--proposal-pythonpath /home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch"
            ),
        "supply may name only the bundled PyTorch proposal boundary"
    );
    Ok(())
}

fn activate_current_bundle(output_root: &Path, bundle_sha256: &str) -> Result<()> {
    ensure!(
        output_root
            .join(bundle_sha256)
            .join("bundle.json")
            .is_file(),
        "cannot activate an absent runtime bundle"
    );
    let current = output_root.join("current");
    if current.exists() && !current.is_symlink() {
        bail!(
            "runtime bundle current path is not an atomic symlink: {}",
            current.display()
        );
    }
    let temporary = output_root.join(format!(".current.tmp-{}", std::process::id()));
    if temporary.exists() || temporary.is_symlink() {
        fs::remove_file(&temporary)?;
    }
    symlink(bundle_sha256, &temporary)?;
    fs::rename(&temporary, &current)?;
    ensure!(
        current.join("bundle.json").is_file(),
        "activated runtime bundle symlink is broken"
    );
    Ok(())
}

fn validate_no_python_runtime(contents: &str) -> Result<()> {
    let normalized = contents.to_ascii_lowercase();
    if normalized.contains("python") || normalized.contains(".py") {
        bail!("Python runtime reference is forbidden outside the PyTorch boundary");
    }
    Ok(())
}

fn validate_encoding_runner(contents: &str) -> Result<()> {
    let normalized = contents.to_ascii_lowercase();
    ensure!(
        normalized.contains("jope_runtime_pythonpath")
            && normalized.contains("jope_encode_external_path")
            && normalized.contains("python3 \"$driver\"")
            && !normalized.contains("export pythonpath=/home/ubuntu/jope-prime/src")
            && !normalized.contains("/home/ubuntu/jope-flywheel/encodings/_driver"),
        "encoding launcher escaped the bundled PyTorch boundary"
    );
    Ok(())
}

fn inventory(candidates: &[Candidate]) -> Result<Vec<InventoryRow>> {
    candidates
        .iter()
        .map(|candidate| {
            Ok(InventoryRow {
                source: candidate.source.display().to_string(),
                target: candidate.target.clone(),
                sha256: sha256_file(&candidate.source)?,
                bytes: candidate.source.metadata()?.len(),
                python_role: candidate.python_role,
            })
        })
        .collect()
}

fn inventory_identity(inventory: &[InventoryRow]) -> Result<String> {
    let rows: Vec<Value> = inventory
        .iter()
        .map(|row| {
            json!({
                "target": row.target,
                "sha256": row.sha256,
                "bytes": row.bytes,
                "python_role": row.python_role,
            })
        })
        .collect();
    Ok(sha256_bytes(&serde_json::to_vec(&rows)?))
}

fn verify_existing(root: &Path, manifest: &Value) -> Result<()> {
    ensure!(manifest["runtime"] == "rust", "bundle runtime is not Rust");
    ensure!(
        manifest["operational_python_allowed"] == false,
        "bundle permits operational Python"
    );
    let rows = manifest["files"].as_array().context("manifest files")?;
    let identity_rows: Vec<_> = rows
        .iter()
        .map(|row| {
            json!({
                "target": row["target"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let computed_identity = sha256_bytes(&serde_json::to_vec(&identity_rows)?);
    ensure!(
        manifest["bundle_sha256"].as_str() == Some(computed_identity.as_str()),
        "bundle manifest identity drift"
    );
    ensure!(
        root.file_name().and_then(|name| name.to_str()) == Some(computed_identity.as_str()),
        "bundle directory does not match its identity"
    );
    for row in rows {
        let target = row["target"].as_str().context("file target")?;
        let expected = row["sha256"].as_str().context("file sha256")?;
        ensure!(
            sha256_file(&root.join(target))? == expected,
            "existing bundle content drift: {target}"
        );
        if target.ends_with(".service") {
            validate_systemd_unit(&fs::read_to_string(root.join(target))?)?;
        }
        if target.ends_with(".py") {
            let role = row
                .get("python_role")
                .and_then(Value::as_str)
                .context("unclassified Python in bundle")?;
            validate_manifest_python_target(target, role)?;
        }
        if target.ends_with(".sh") {
            let contents = fs::read_to_string(root.join(target))?;
            if target == "pytorch/encoding/runner.sh" {
                validate_encoding_runner(&contents)?;
            } else {
                validate_no_python_runtime(&contents)?;
            }
        }
    }
    for required in [
        "pytorch/jope/__init__.py",
        "pytorch/jope/policy/model_v3.py",
        "pytorch/jope/policy/decode.py",
        "pytorch/proposals/test_exact_context_proposals.py",
    ] {
        ensure!(
            root.join(required).is_file(),
            "bundle lacks required JOPE proposal dependency: {required}"
        );
    }
    Ok(())
}

fn validate_manifest_python_target(target: &str, role: &str) -> Result<()> {
    if role == "pytorch_dependency_package"
        && target.starts_with("pytorch/jope/")
        && target.ends_with(".py")
    {
        return Ok(());
    }
    ensure!(
        PYTORCH_ALLOWLIST
            .iter()
            .any(|(_, allowed_target, allowed_role)| {
                target == *allowed_target && role == *allowed_role
            }),
        "unallowlisted Python in bundle: {target}"
    );
    Ok(())
}

fn validate_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    ensure!(
        manifest.schema == "JopeSelfplayServiceManifest.v3",
        "unsupported service manifest schema"
    );
    let binaries = manifest.binaries.iter().collect::<BTreeSet<_>>();
    ensure!(
        binaries.len() == manifest.binaries.len() && !binaries.is_empty(),
        "service manifest binary inventory is empty or duplicated"
    );
    let mut units = BTreeSet::new();
    for row in &manifest.services {
        ensure!(
            row.unit.ends_with(".service")
                && !row.unit.contains('/')
                && !row.owner.trim().is_empty(),
            "service manifest contains an invalid unit or owner"
        );
        ensure!(
            units.insert(row.unit.as_str()),
            "service manifest repeats unit {}",
            row.unit
        );
        ensure!(
            binaries.contains(&row.binary),
            "service {} references unbundled binary {}",
            row.unit,
            row.binary
        );
    }
    Ok(())
}

fn ensure_bundle_target(root: &Path, target: &Path) -> Result<()> {
    ensure!(
        target.starts_with(root),
        "bundle target escaped temporary directory"
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut input =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    serde_json::to_writer(&mut output, value)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operational_python_is_rejected() {
        let source = Path::new("/repo");
        let prime = Path::new("/prime");
        let encoding = Path::new("/encoding");
        let candidate = Candidate {
            source: source.join("scripts/queue.py"),
            target: "runtime/scripts/queue.py".to_owned(),
            python_role: None,
        };
        assert!(validate_python_candidate(&candidate, source, prime, encoding).is_err());
    }

    #[test]
    fn exact_pytorch_allowlist_is_accepted() {
        let source = Path::new("/repo");
        let prime = Path::new("/prime");
        let encoding = Path::new("/encoding");
        let candidate = Candidate {
            source: source.join("scripts/test_exact_dl_generate.py"),
            target: "pytorch/proposals/test_exact_dl_generate.py".to_owned(),
            python_role: Some("pytorch_proposal_inference"),
        };
        validate_python_candidate(&candidate, source, prime, encoding).unwrap();
        let context_candidate = Candidate {
            source: source.join("scripts/test_exact_context_proposals.py"),
            target: "pytorch/proposals/test_exact_context_proposals.py".to_owned(),
            python_role: Some("pytorch_proposal_inference"),
        };
        validate_python_candidate(&context_candidate, source, prime, encoding).unwrap();
        let trainer = Candidate {
            source: source.join("scripts/test_exact_train.py"),
            target: "pytorch/training/test_exact_train.py".to_owned(),
            python_role: Some("pytorch_training"),
        };
        validate_python_candidate(&trainer, source, prime, encoding).unwrap();
        let dependency = Candidate {
            source: prime.join("src/jope/data/active.py"),
            target: "pytorch/jope/data/active.py".to_owned(),
            python_role: Some("pytorch_dependency_package"),
        };
        validate_python_candidate(&dependency, source, prime, encoding).unwrap();
        let encoder = Candidate {
            source: encoding.join("encode_batch.py"),
            target: "pytorch/encoding/encode_batch.py".to_owned(),
            python_role: Some("pytorch_dataset_encoding"),
        };
        validate_python_candidate(&encoder, source, prime, encoding).unwrap();
    }

    #[test]
    fn python_systemd_execstart_is_rejected() {
        let unit = "[Service]\nExecStart=/usr/bin/python3 /runtime/queue.py\n";
        assert!(validate_systemd_unit(unit).is_err());
        validate_systemd_unit("[Service]\nExecStart=/runtime/bin/jope_selfplay_queue --submit\n")
            .unwrap();
        assert!(validate_systemd_unit(
            "[Service]\nExecStart=/home/ubuntu/jope-dime/target-promotion/release/jope_selfplay_promotion\n"
        )
        .is_err());
    }

    #[test]
    fn manifest_python_role_cannot_expand_the_allowlist() {
        assert!(
            validate_manifest_python_target("runtime/scripts/queue.py", "pytorch_training")
                .is_err()
        );
        validate_manifest_python_target(
            "pytorch/training/prime_policy_trainer.py",
            "pytorch_training",
        )
        .unwrap();
    }

    #[test]
    fn content_identity_is_ordered_and_hash_sensitive() {
        let rows = vec![InventoryRow {
            source: "/irrelevant/source".to_owned(),
            target: "runtime/bin/queue".to_owned(),
            sha256: "a".repeat(64),
            bytes: 42,
            python_role: None,
        }];
        let first = inventory_identity(&rows).unwrap();
        let mut changed_source = rows.clone();
        changed_source[0].source = "/elsewhere/same-content".to_owned();
        assert_eq!(first, inventory_identity(&changed_source).unwrap());
        let mut changed_hash = rows;
        changed_hash[0].sha256 = "b".repeat(64);
        assert_ne!(first, inventory_identity(&changed_hash).unwrap());
    }

    #[test]
    fn declarative_service_manifest_rejects_split_inventory() {
        let valid = ServiceManifest {
            schema: "JopeSelfplayServiceManifest.v3".to_owned(),
            binaries: vec!["queue".to_owned()],
            services: vec![ServiceRow {
                unit: "queue.service".to_owned(),
                binary: "queue".to_owned(),
                owner: "submission".to_owned(),
                active: true,
            }],
        };
        validate_service_manifest(&valid).unwrap();
        let mut invalid = valid;
        invalid.services[0].binary = "missing".to_owned();
        assert!(validate_service_manifest(&invalid).is_err());
    }
}
