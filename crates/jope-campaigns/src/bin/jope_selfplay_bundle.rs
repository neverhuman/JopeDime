//! Rust-first, content-addressed self-play runtime bundler.
//!
//! Operational Python is forbidden. The only Python copied into a production
//! bundle is an exact allowlist for the PyTorch training, proposal, and
//! dataset-encoding boundaries.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::dime_activation::{
    abort_activation_transaction, execute_activation_transaction, ActivationLock,
    ActivationOperationV1, ActivationRequest, DimeCandidateReviewGateV1,
    DimeExternalActivationGateV1, SystemdUserHost,
};
use jope_dime_regressor::encoder_v2_identity::ArtifactIdentity;
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::{
    classify_rollback_capsule_schema, immutable_flywheel_root_str, RollbackBaselineDerivationV1,
    RollbackBundleFileV2, RollbackCapsuleV2, RollbackSystemdArtifactV2,
    RollbackUnitTransformationV1, ROLLBACK_BASELINE_DERIVATION_SCHEMA,
    ROLLBACK_RUNTIME_BUNDLE_SHA256,
};
use jope_dime_regressor::selfplay_data::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

mod activation;
mod build;
mod inventory;
mod rollback;
mod synchronization;
#[cfg(test)]
mod tests;

use activation::*;
use build::*;
use inventory::*;
use rollback::*;
use synchronization::*;

jope_paths::once_str!(
    fn default_source_str => paths::jope_home()
);
jope_paths::once_str!(
    /// The live Python source tree a unit must never import from.
    fn python_src_str => paths::python_src()
);
jope_paths::once_str!(
    /// The live self-play scripts a unit must never run from.
    fn research_scripts_str => paths::research_scripts_root()
);
const PRIME_SOURCE_BUNDLE_SHA256: &str =
    "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4";
// The rollback runtime happens to have supplied the immutable Prime support
// tree, but those are separate authorities. Never derive one identity from
// the other.
const ROLLBACK_BUNDLE_SHA256: &str = ROLLBACK_RUNTIME_BUNDLE_SHA256;
const SUPERSEDED_STAGING_BUNDLE_SHA256: &str =
    "412c5ed5560222842e41dbd479e673cf99cd293cc965f01504bb38c1a4dd7372";
const HISTORICAL_ROLLBACK_PARENT_SHA256: &str =
    "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4";
const REVIEWED_ROLLBACK_UNIT_COMMIT: &str = "7e1d0429021ab11e8ec63141669f352d32eebe38";
jope_paths::once_str!(
    fn rollback_working_directory_from_str => paths::jope_home()
);
const ROLLBACK_REBASE_UNITS: &[&str] = &[
    "jope-selfplay-controller.service",
    "jope-selfplay-corpus-freezer.service",
    "jope-selfplay-freshness.service",
    "jope-selfplay-ingestion.service",
    "jope-selfplay-reserve.service",
    "jope-selfplay-watchdog.service",
    "jope-testonly-monitor.service",
];
jope_paths::once_str!(
    fn default_prime_str => paths::runtime_bundles_root().join("9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4/pytorch")
);
jope_paths::once_str!(
    fn active_bundle_pytorch_root_str => paths::runtime_bundle_pytorch()
);
jope_paths::once_str!(
    fn default_encoding_driver_str => paths::flywheel_encodings_root().join("_driver")
);
jope_paths::once_str!(
    fn default_runner_str => paths::fweval_runner()
);
jope_paths::once_str!(
    fn default_scq_projects_str => paths::scq_projects()
);
jope_paths::once_str!(
    fn default_output_str => paths::runtime_bundles_root()
);
const DEFAULT_SERVICE_MANIFEST: &str = "manifests/selfplay/service-manifest.v3.json";
jope_paths::once_str!(
    fn default_user_unit_root_str => paths::user_unit_root()
);
jope_paths::once_str!(
    fn default_activation_transaction_root_str => paths::flywheel_root().join("ops/dime-activation")
);

/// GPU nodes that can host bundle-pinned GPU jobs (cohort proposer, trainer).
/// Their scq specs pin `[env] PYTHONPATH` and a smoke test to the bundle path
/// on the EXECUTION node, so a bundle that exists only on this host kills wave
/// production the moment another workload takes the local GPU (2026-08-03:
/// six vanished proposer dispatches, supply retry budget exhausted).
/// Activation refuses to advance the pointer until the bundle is present on
/// every listed node. The production node set is not runtime-overridable.
const GPU_SYNC_NODES: &[&str] = &["xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"];

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
    #[arg(long, default_value = default_source_str())]
    source: PathBuf,
    #[arg(long, default_value = default_prime_str())]
    prime: PathBuf,
    #[arg(long, default_value = default_encoding_driver_str())]
    encoding_driver: PathBuf,
    #[arg(long, default_value = default_runner_str())]
    runner: PathBuf,
    #[arg(long, default_value = default_scq_projects_str())]
    scq_projects: PathBuf,
    #[arg(long, default_value = default_output_str())]
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
    /// Verify a staged content-addressed bundle without changing any pointer.
    #[arg(
        long,
        value_name = "SHA256",
        conflicts_with_all = ["activate", "activate_existing", "synchronize_existing"]
    )]
    verify_existing: Option<String>,
    /// Synchronize and hash-verify a staged bundle on all five nodes without
    /// changing the active runtime pointer.
    #[arg(
        long,
        value_name = "SHA256",
        conflicts_with_all = ["activate", "activate_existing", "verify_existing"]
    )]
    synchronize_existing: Option<String>,
    /// Content-addressed independent candidate review and test evidence.
    #[arg(long, value_name = "PATH")]
    candidate_review_gate: Option<PathBuf>,
    /// Composite runtime/unit rollback authority required for activation.
    #[arg(long, value_name = "PATH")]
    rollback_capsule: Option<PathBuf>,
    /// Verify a composite rollback capsule without changing runtime pointers.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = [
            "activate",
            "activate_existing",
            "verify_existing",
            "synchronize_existing"
        ]
    )]
    verify_rollback_capsule: Option<PathBuf>,
    /// Build the strict unit-only rollback baseline from the historical parent.
    #[arg(long, conflicts_with_all = [
        "activate", "activate_existing", "verify_existing", "synchronize_existing",
        "verify_rollback_capsule", "prepare_evaluator_retention"
    ])]
    derive_rollback_baseline: bool,
    /// Idempotently retain all referenced evaluators without activating a bundle.
    #[arg(
        long,
        value_name = "SHA256",
        conflicts_with_all = [
            "activate", "activate_existing", "verify_existing", "synchronize_existing",
            "verify_rollback_capsule", "derive_rollback_baseline"
        ]
    )]
    prepare_evaluator_retention: Option<String>,
    /// Fresh external proof of SCQ companion-core support and cancellation ownership.
    #[arg(long, value_name = "PATH")]
    external_activation_gate: Option<PathBuf>,
    #[arg(long, default_value = default_user_unit_root_str())]
    user_unit_root: PathBuf,
    #[arg(long, default_value = default_activation_transaction_root_str())]
    activation_transaction_root: PathBuf,
    /// Capsule-gated rollback of runtime, DIME units, and service state only.
    #[arg(long, conflicts_with_all = [
        "activate", "activate_existing", "verify_existing", "synchronize_existing",
        "verify_rollback_capsule", "derive_rollback_baseline", "prepare_evaluator_retention",
        "abort_activation_transaction"
    ])]
    rollback_to_baseline: bool,
    /// Restore exact DIME activation preimages after an interrupted transaction.
    #[arg(long, value_name = "PATH", conflicts_with_all = [
        "activate", "activate_existing", "verify_existing", "synchronize_existing",
        "verify_rollback_capsule", "derive_rollback_baseline", "prepare_evaluator_retention",
        "rollback_to_baseline"
    ])]
    abort_activation_transaction: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Candidate {
    source: PathBuf,
    target: String,
    python_role: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct InventoryRow {
    source: String,
    target: String,
    sha256: String,
    bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    python_role: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PrimeSupportFile {
    source_bundle_target: String,
    bundle_target: String,
    sha256: String,
    bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PrimeSupportProvenance {
    source_bundle_sha256: &'static str,
    source_bundle_manifest_sha256: String,
    package_tree_sha256: String,
    files: Vec<PrimeSupportFile>,
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
    if let Some(transaction_root) = &args.abort_activation_transaction {
        let _lock =
            ActivationLock::acquire(&args.activation_transaction_root.join("activation.lock"))?;
        let mut host = SystemdUserHost;
        let transaction = abort_activation_transaction(
            &args.output_root,
            &args.user_unit_root,
            transaction_root,
            &mut host,
        )?;
        println!("{}", serde_json::to_string(&transaction)?);
        return Ok(());
    }
    if args.rollback_to_baseline {
        let (capsule, evidence_root) = load_activation_rollback_capsule(&args)?;
        let manifest = verify_intrinsic_existing(&args.output_root, ROLLBACK_BUNDLE_SHA256)?;
        activate_manifest(
            &args,
            &manifest,
            &capsule,
            &evidence_root,
            ActivationOperationV1::Rollback,
        )?;
        println!("{}", serde_json::to_string(&manifest)?);
        return Ok(());
    }
    if args.derive_rollback_baseline {
        let derivation = derive_rollback_baseline(&args.source, &args.output_root)?;
        println!("{}", serde_json::to_string(&derivation)?);
        return Ok(());
    }
    if let Some(bundle_sha256) = &args.prepare_evaluator_retention {
        require_bundle_sha256(bundle_sha256)?;
        verify_intrinsic_existing(&args.output_root, bundle_sha256)?;
        let retention = retain_evaluators(&args.output_root, bundle_sha256)?;
        println!("{}", serde_json::to_string(&retention)?);
        return Ok(());
    }
    if let Some(path) = &args.verify_rollback_capsule {
        let value: Value = serde_json::from_reader(std::io::BufReader::new(
            File::open(path)
                .with_context(|| format!("open rollback capsule {}", path.display()))?,
        ))?;
        ensure!(
            classify_rollback_capsule_schema(&value)? == "authorizing-v2",
            "JopeRollbackCapsule.v1 is historical evidence and can never authorize activation"
        );
        let capsule: RollbackCapsuleV2 = serde_json::from_value(value)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        let evidence_root = path.parent().unwrap_or_else(|| Path::new("."));
        let digest = verify_rollback_capsule(&args.output_root, evidence_root, &capsule, now)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "schema": "JopeRollbackCapsuleVerification.v2",
                "rollback_capsule_sha256": digest,
                "runtime_bundle_sha256": capsule.runtime_bundle_sha256,
                "systemd_closure_sha256": capsule.systemd_closure_canonical_json_sha256,
                "verified": true,
                "pointer_changed": false,
            }))?
        );
        return Ok(());
    }
    if let Some(bundle_sha256) = args
        .verify_existing
        .as_deref()
        .or(args.synchronize_existing.as_deref())
        .or(args.activate_existing.as_deref())
    {
        ensure!(
            bundle_sha256.len() == 64
                && bundle_sha256
                    .bytes()
                    .all(|byte| { byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) }),
            "existing bundle verification requires a lowercase SHA-256"
        );
        let manifest_path = args.output_root.join(bundle_sha256).join("bundle.json");
        let manifest: Value = serde_json::from_reader(std::io::BufReader::new(
            File::open(&manifest_path)
                .with_context(|| format!("open staged bundle {}", manifest_path.display()))?,
        ))?;
        verify_intrinsic_existing(&args.output_root, bundle_sha256)?;
        if args.verify_existing.is_some() {
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "schema": "JopeBundleVerification.v1",
                    "bundle_sha256": bundle_sha256,
                    "verified": true,
                    "rollback_authorization_checked": false,
                    "pointer_changed": false,
                }))?
            );
            return Ok(());
        }
        if args.synchronize_existing.is_some() {
            let mut receipt =
                sync_bundle_to_gpu_nodes(&args.output_root, bundle_sha256, &manifest)?;
            ensure!(
                receipt.get("complete").and_then(Value::as_bool) == Some(true),
                "bundle synchronization is incomplete"
            );
            receipt["fresh_live_probe"] =
                fresh_live_bundle_probe(&args.output_root, bundle_sha256, &manifest)?;
            let receipt_path = args
                .output_root
                .join("synchronization")
                .join(format!("{bundle_sha256}.v1.json"));
            atomic_json(&receipt_path, &receipt)?;
            println!("{}", serde_json::to_string(&receipt)?);
            return Ok(());
        }
        let (capsule, evidence_root) = load_activation_rollback_capsule(&args)?;
        activate_manifest(
            &args,
            &manifest,
            &capsule,
            &evidence_root,
            ActivationOperationV1::Activate,
        )?;
        println!("{}", serde_json::to_string(&manifest)?);
        return Ok(());
    }
    let manifest = build(&args)?;
    if args.activate {
        let (capsule, evidence_root) = load_activation_rollback_capsule(&args)?;
        activate_manifest(
            &args,
            &manifest,
            &capsule,
            &evidence_root,
            ActivationOperationV1::Activate,
        )?;
    }
    println!("{}", serde_json::to_string(&manifest)?);
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
