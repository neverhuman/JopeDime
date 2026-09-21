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
use jope_dime_regressor::recovery_contracts::{
    classify_rollback_capsule_schema, RollbackBaselineDerivationV1, RollbackBundleFileV2,
    RollbackCapsuleV2, RollbackSystemdArtifactV2, RollbackUnitTransformationV1,
    IMMUTABLE_FLYWHEEL_ROOT, ROLLBACK_BASELINE_DERIVATION_SCHEMA, ROLLBACK_RUNTIME_BUNDLE_SHA256,
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

const DEFAULT_SOURCE: &str = "/home/ubuntu/jope-dime";
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
const ROLLBACK_WORKING_DIRECTORY_FROM: &str = "/home/ubuntu/jope-dime";
const ROLLBACK_REBASE_UNITS: &[&str] = &[
    "jope-selfplay-controller.service",
    "jope-selfplay-corpus-freezer.service",
    "jope-selfplay-freshness.service",
    "jope-selfplay-ingestion.service",
    "jope-selfplay-reserve.service",
    "jope-selfplay-watchdog.service",
    "jope-testonly-monitor.service",
];
const DEFAULT_PRIME: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4/pytorch";
const ACTIVE_BUNDLE_PYTORCH_ROOT: &str =
    "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch";
const DEFAULT_ENCODING_DRIVER: &str = "/home/ubuntu/jope-flywheel/encodings/_driver";
const DEFAULT_RUNNER: &str = "/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh";
const DEFAULT_SCQ_PROJECTS: &str = "/home/ubuntu/smart_cluster/etc/projects.toml";
const DEFAULT_OUTPUT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles";
const DEFAULT_SERVICE_MANIFEST: &str = "manifests/selfplay/service-manifest.v3.json";
const DEFAULT_USER_UNIT_ROOT: &str = "/home/ubuntu/.config/systemd/user";
const DEFAULT_ACTIVATION_TRANSACTION_ROOT: &str =
    "/home/ubuntu/jope-flywheel/test_exact_gen0/ops/dime-activation";

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
    #[arg(long, default_value = DEFAULT_USER_UNIT_ROOT)]
    user_unit_root: PathBuf,
    #[arg(long, default_value = DEFAULT_ACTIVATION_TRANSACTION_ROOT)]
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
        let value: Value = serde_json::from_reader(
            File::open(path)
                .with_context(|| format!("open rollback capsule {}", path.display()))?,
        )?;
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
        let manifest: Value = serde_json::from_reader(
            File::open(&manifest_path)
                .with_context(|| format!("open staged bundle {}", manifest_path.display()))?,
        )?;
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

fn load_activation_rollback_capsule(args: &Args) -> Result<(RollbackCapsuleV2, PathBuf)> {
    let path = args
        .rollback_capsule
        .as_ref()
        .context("activation requires --rollback-capsule")?;
    let value: Value = serde_json::from_reader(
        File::open(path).with_context(|| format!("open rollback capsule {}", path.display()))?,
    )
    .context("decode rollback capsule")?;
    ensure!(
        classify_rollback_capsule_schema(&value)? == "authorizing-v2",
        "JopeRollbackCapsule.v1 is historical evidence and can never authorize activation"
    );
    Ok((
        serde_json::from_value(value)?,
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    ))
}

fn activate_manifest(
    args: &Args,
    manifest: &Value,
    rollback_capsule: &RollbackCapsuleV2,
    evidence_root: &Path,
    operation: ActivationOperationV1,
) -> Result<()> {
    let output_root = &args.output_root;
    let bundle_sha256 = manifest["bundle_sha256"].as_str().context("bundle hash")?;
    validate_activation_target(operation, bundle_sha256)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let candidate_review_gate_sha256 = if operation == ActivationOperationV1::Activate {
        Some(verify_candidate_review_gate(args, manifest, now)?)
    } else {
        None
    };
    let rollback_capsule_sha256 =
        verify_rollback_capsule(output_root, evidence_root, rollback_capsule, now)?;
    let rollback_path = output_root.join(ROLLBACK_BUNDLE_SHA256).join("bundle.json");
    let rollback_manifest: Value = serde_json::from_reader(
        File::open(&rollback_path)
            .with_context(|| format!("open rollback bundle {}", rollback_path.display()))?,
    )?;
    verify_rollback_candidate(
        rollback_path
            .parent()
            .context("rollback manifest has no bundle directory")?,
        &rollback_manifest,
        ROLLBACK_BUNDLE_SHA256,
    )?;
    let rollback_synchronization =
        sync_bundle_to_gpu_nodes(output_root, ROLLBACK_BUNDLE_SHA256, &rollback_manifest)?;
    ensure!(
        rollback_synchronization
            .get("complete")
            .and_then(Value::as_bool)
            == Some(true),
        "rollback bundle is not verified on all five nodes"
    );
    let rollback_live_probe =
        fresh_live_bundle_probe(output_root, ROLLBACK_BUNDLE_SHA256, &rollback_manifest)?;
    // Refuse to advance any activation pointer until the bundle exists on
    // every GPU node — a locally-activated, unsynced bundle killed wave
    // production on 2026-08-03 (see GPU_SYNC_NODES).
    let synchronization = sync_bundle_to_gpu_nodes(output_root, bundle_sha256, manifest)?;
    let candidate_live_probe = fresh_live_bundle_probe(output_root, bundle_sha256, manifest)?;
    let synchronization_path = output_root
        .join("synchronization")
        .join(format!("{bundle_sha256}.v1.json"));
    atomic_json(&synchronization_path, &synchronization)?;
    let retention = retain_evaluators(output_root, bundle_sha256)?;
    ensure!(
        retention
            .get("all_referenced_evaluators_retained")
            .and_then(Value::as_bool)
            == Some(true),
        "evaluator retention is incomplete"
    );
    let external_gate_sha256 = if operation == ActivationOperationV1::Activate {
        let path = args
            .external_activation_gate
            .as_ref()
            .context("forward DIME activation requires --external-activation-gate")?;
        let gate: DimeExternalActivationGateV1 = serde_json::from_reader(
            File::open(path)
                .with_context(|| format!("open external activation gate {}", path.display()))?,
        )?;
        gate.validate_at(path.parent().unwrap_or_else(|| Path::new(".")), now)?;
        ensure!(
            gate.candidate_bundle_sha256 == bundle_sha256,
            "external activation gate is bound to another candidate"
        );
        Some(sha256_file(path)?)
    } else {
        None
    };
    let manifest_path = output_root.join(bundle_sha256).join("bundle.json");
    let pointer = json!({
        "schema": "JopeSelfplayRuntimeBundlePointer.v2",
        "bundle": manifest_path,
        "bundle_manifest_sha256": sha256_file(&manifest_path)?,
        "bundle_sha256": bundle_sha256,
        "synchronization_receipt": synchronization_path,
        "synchronization_receipt_sha256": sha256_file(&synchronization_path)?,
        "synchronized_nodes": GPU_SYNC_NODES,
        "rollback_bundle_sha256": ROLLBACK_BUNDLE_SHA256,
        "rollback_capsule_sha256": rollback_capsule_sha256,
        "rollback_systemd_closure_sha256": rollback_capsule.systemd_closure_canonical_json_sha256,
        "rollback_synchronization": rollback_synchronization,
        "rollback_fresh_live_probe": rollback_live_probe,
        "candidate_fresh_live_probe": candidate_live_probe,
        "candidate_review_gate_sha256": candidate_review_gate_sha256,
        "external_activation_gate_sha256": external_gate_sha256,
        "runtime": "rust",
    });
    let operation_name = match operation {
        ActivationOperationV1::Activate => "activate",
        ActivationOperationV1::Rollback => "rollback",
    };
    let transaction_root = args.activation_transaction_root.join(format!(
        "{operation_name}-{bundle_sha256}-{}",
        &rollback_capsule_sha256[..12]
    ));
    let _lock = ActivationLock::acquire(&args.activation_transaction_root.join("activation.lock"))?;
    let request = ActivationRequest {
        output_root,
        user_unit_root: &args.user_unit_root,
        transaction_root: &transaction_root,
        candidate_bundle_sha256: bundle_sha256,
        rollback_capsule_sha256: &rollback_capsule_sha256,
        candidate_review_gate_sha256: candidate_review_gate_sha256.as_deref(),
        external_gate_sha256: external_gate_sha256.as_deref(),
        active_document: pointer,
        operation,
    };
    let mut host = SystemdUserHost;
    let transaction = execute_activation_transaction(&request, &mut host, None)?;
    ensure!(
        transaction.stage == jope_dime_regressor::dime_activation::ActivationStageV1::Complete,
        "DIME activation transaction did not complete"
    );
    Ok(())
}

fn verify_candidate_review_gate(args: &Args, manifest: &Value, now_unix_ms: u64) -> Result<String> {
    let path = args
        .candidate_review_gate
        .as_ref()
        .context("forward activation requires --candidate-review-gate")?;
    let gate: DimeCandidateReviewGateV1 = serde_json::from_reader(
        File::open(path)
            .with_context(|| format!("open candidate review gate {}", path.display()))?,
    )?;
    gate.validate_at(path.parent().unwrap_or_else(|| Path::new(".")), now_unix_ms)?;
    let bundle_sha256 = manifest
        .get("bundle_sha256")
        .and_then(Value::as_str)
        .context("candidate manifest lacks bundle identity")?;
    let integration_commit = manifest
        .get("integration_commit")
        .and_then(Value::as_str)
        .context("candidate manifest lacks integration commit")?;
    let manifest_path = args.output_root.join(bundle_sha256).join("bundle.json");
    ensure!(
        gate.candidate_bundle_sha256 == bundle_sha256
            && gate.integration_commit == integration_commit
            && gate.bundle_manifest_file_sha256 == sha256_file(&manifest_path)?
            && gate.bundle_manifest_canonical_json_sha256 == canonical_json_sha256(manifest)?,
        "candidate review gate is bound to different manifest bytes or provenance"
    );
    let object = format!("{integration_commit}^{{commit}}");
    let reachable = Command::new("git")
        .arg("-C")
        .arg(&args.source)
        .args(["cat-file", "-e", &object])
        .output()
        .with_context(|| format!("check reviewed integration commit {integration_commit}"))?;
    ensure!(
        reachable.status.success(),
        "reviewed integration commit is not reachable: {}",
        String::from_utf8_lossy(&reachable.stderr).trim()
    );
    sha256_file(path)
}

fn validate_activation_target(operation: ActivationOperationV1, bundle_sha256: &str) -> Result<()> {
    if operation == ActivationOperationV1::Rollback {
        ensure!(
            bundle_sha256 == ROLLBACK_BUNDLE_SHA256,
            "explicit rollback target is not the pinned baseline"
        );
    } else {
        ensure!(
            bundle_sha256 != SUPERSEDED_STAGING_BUNDLE_SHA256,
            "412c5ed5 is retained historical staging evidence and is never activatable"
        );
    }
    Ok(())
}

fn collect_evaluator_digests(value: &Value, output: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "evaluator_sha256"
                        | "original_evaluator_sha256"
                        | "expected_evaluator_sha256"
                        | "evaluator_binary_sha256"
                ) {
                    if let Some(digest) = value.as_str() {
                        if digest.len() == 64
                            && digest
                                .bytes()
                                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                        {
                            output.insert(digest.to_owned());
                        }
                    }
                }
                collect_evaluator_digests(value, output);
            }
        }
        Value::Array(rows) => {
            for row in rows {
                collect_evaluator_digests(row, output);
            }
        }
        _ => {}
    }
}

fn collect_json_references(path: &Path, output: &mut BTreeSet<String>) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let value: Value = serde_json::from_reader(
        File::open(path).with_context(|| format!("open evaluator reference {}", path.display()))?,
    )?;
    collect_evaluator_digests(&value, output);
    Ok(())
}

fn evaluator_reference_index(root: &Path) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let mut references = BTreeMap::<String, BTreeSet<String>>::new();
    for (class, path) in [
        (
            "repair_state",
            root.join("ops/causal-pair-repair-state.v1.json"),
        ),
        (
            "repair_request",
            root.join("ops/causal-pair-repair-queue.v1.json"),
        ),
        (
            "unreplayable_disposition",
            root.join("training/unreplayable-pair-disposition.v2.json"),
        ),
    ] {
        let mut digests = BTreeSet::new();
        collect_json_references(&path, &mut digests)?;
        references.insert(class.to_owned(), digests);
    }
    let epoch_directory = root.join("training/causal-evidence-epochs");
    let mut epoch_digests = BTreeSet::new();
    if epoch_directory.is_dir() {
        for entry in fs::read_dir(epoch_directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) == Some("json") {
                collect_json_references(&path, &mut epoch_digests)?;
            }
        }
    }
    references.insert("evidence_epoch".to_owned(), epoch_digests);
    Ok(references)
}

/// Retain every evaluator still available in a content-addressed vault before
/// advancing the active bundle pointer. Hard links make later bundle cleanup
/// safe without copying or changing the evaluator bytes.
fn retain_evaluators(output_root: &Path, prepared_bundle_sha256: &str) -> Result<Value> {
    let root = output_root
        .parent()
        .context("runtime bundle root has no flywheel parent")?;
    let current_target = fs::read_link(output_root.join("current"))
        .context("evaluator retention requires the live current-bundle symlink")?;
    let active_bundle_sha256 = current_target
        .file_name()
        .and_then(|name| name.to_str())
        .context("current-bundle symlink has no content identity")?;
    require_bundle_sha256(active_bundle_sha256)?;
    require_bundle_sha256(prepared_bundle_sha256)?;
    let vault = root.join("training/evaluator-vault");
    fs::create_dir_all(&vault)?;
    let mut retained = BTreeMap::<String, BTreeSet<String>>::new();
    for entry in fs::read_dir(output_root)? {
        let bundle = entry?;
        if !bundle.file_type()?.is_dir() {
            continue;
        }
        let bundle_sha256 = bundle.file_name().to_string_lossy().into_owned();
        if bundle_sha256.len() != 64 {
            continue;
        }
        let evaluator = bundle.path().join("runtime/bin/corpus_eval");
        if !evaluator.is_file() {
            continue;
        }
        let evaluator_sha256 = sha256_file(&evaluator)?;
        let directory = vault.join(&evaluator_sha256);
        fs::create_dir_all(&directory)?;
        let target = directory.join("corpus_eval");
        if target.is_file() {
            ensure!(
                sha256_file(&target)? == evaluator_sha256,
                "evaluator vault content drift for {evaluator_sha256}"
            );
        } else {
            fs::hard_link(&evaluator, &target).with_context(|| {
                format!(
                    "hard-link evaluator {} into {}",
                    evaluator.display(),
                    target.display()
                )
            })?;
        }
        retained
            .entry(evaluator_sha256)
            .or_default()
            .insert(bundle_sha256);
    }
    ensure!(
        retained
            .values()
            .any(|bundles| bundles.contains(prepared_bundle_sha256))
            && retained
                .values()
                .any(|bundles| bundles.contains(active_bundle_sha256)),
        "live or prepared bundle does not retain runtime/bin/corpus_eval"
    );
    let references = evaluator_reference_index(root)?;
    let referenced = references
        .values()
        .flat_map(|digests| digests.iter().cloned())
        .collect::<BTreeSet<_>>();
    let unavailable = referenced
        .iter()
        .filter(|digest| !retained.contains_key(*digest))
        .cloned()
        .collect::<Vec<_>>();
    let all_referenced_evaluators_retained = unavailable.is_empty();
    let generated_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let index = json!({
        "schema": "JopeEvaluatorRetentionIndex.v1",
        "generated_at_ms": generated_at_ms,
        "active_bundle_sha256": active_bundle_sha256,
        "prepared_for_bundle_sha256": prepared_bundle_sha256,
        "vault": vault,
        "retained": retained.into_iter().map(|(evaluator_sha256, bundle_sha256)| {
            let path = root.join("training/evaluator-vault").join(&evaluator_sha256).join("corpus_eval");
            json!({
                "evaluator_sha256": evaluator_sha256,
                "path": path,
                "bundle_sha256": bundle_sha256,
            })
        }).collect::<Vec<_>>(),
        "reference_classes": references,
        "referenced_evaluator_sha256": referenced,
        "unavailable_referenced_evaluator_sha256": unavailable,
        "all_referenced_evaluators_retained": all_referenced_evaluators_retained,
        "cleanup_policy": "vault hard links are outside runtime-bundle cleanup scope",
    });
    let index_path = root.join("training/evaluator-retention-index.v1.json");
    if index_path.is_file() {
        let existing: Value = serde_json::from_reader(File::open(&index_path)?)?;
        let stable = |mut value: Value| {
            if let Some(object) = value.as_object_mut() {
                object.remove("generated_at_ms");
            }
            value
        };
        if stable(existing.clone()) == stable(index.clone()) {
            return Ok(existing);
        }
    }
    atomic_json(&index_path, &index)?;
    Ok(index)
}

fn build(args: &Args) -> Result<Value> {
    let integration_commit = clean_detached_commit(&args.source)?;
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
    validate_recovery_service_manifest(&service_manifest)?;
    let candidates = sources(args, &service_manifest, &service_manifest_path)?;
    let inventory = inventory(&candidates)?;
    let prime_support = prime_support_provenance(args, &candidates, &inventory)?;
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
            "integration_commit": integration_commit,
            "clean_detached_source": true,
            "operational_python_allowed": false,
            "pytorch_python_allowlist": python_targets,
            "prime_support": prime_support,
            "required_sync_nodes": GPU_SYNC_NODES,
            "five_node_synchronization": {
                "schema": "JopeBundleSynchronizationContract.v1",
                "receipt_schema": "JopeBundleSynchronizationReceipt.v1",
                "receipt_relative_path": format!("synchronization/{identity}.v1.json"),
                "required_nodes": GPU_SYNC_NODES,
                "activation_requires_complete": true,
            },
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

fn git_output(source: &Path, arguments: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(source)
        .args(arguments)
        .output()
        .with_context(|| format!("run git in {}", source.display()))?;
    ensure!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn clean_detached_commit(source: &Path) -> Result<String> {
    ensure!(
        git_output(source, &["status", "--porcelain"])?.is_empty(),
        "runtime bundle source worktree is dirty"
    );
    ensure!(
        git_output(source, &["rev-parse", "--abbrev-ref", "HEAD"])? == "HEAD",
        "runtime bundle source must be a detached reviewed commit"
    );
    let commit = git_output(source, &["rev-parse", "HEAD"])?;
    ensure!(
        commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "runtime bundle integration commit is malformed"
    );
    Ok(commit)
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
        "manifests/selfplay/proposal-selection-policy.v1.json",
        "manifests/dlgp-budget-first/protocol.v1.json",
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
        args.source
            .join("manifests/selfplay/proposal-selection-policy.v1.json"),
        "pytorch/proposals/proposal-selection-policy.v1.json".to_owned(),
        None,
    );
    add(
        &mut selected,
        args.encoding_driver.join("binexp/runner.sh"),
        "pytorch/encoding/runner.sh".to_owned(),
        None,
    );
    collect_extension(
        &args.prime.join("jope"),
        &args.prime,
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
        let relative = match relative {
            "scripts/prime_policy_trainer.py" => "training/prime_policy_trainer.py",
            "scripts/ratchet_dl_update.py" => "training/ratchet_dl_update.py",
            "scripts/encode_external.py" => "encoding/encode_external.py",
            _ => relative,
        };
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
        && candidate.source.starts_with(prime.join("jope"))
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
    for line in systemd_logical_lines(contents)? {
        let line = line.trim();
        if let Some(working_directory) = line.strip_prefix("WorkingDirectory=") {
            ensure!(
                working_directory == IMMUTABLE_FLYWHEEL_ROOT,
                "systemd WorkingDirectory is not the canonical immutable flywheel root"
            );
        }
        if line.starts_with("ExecStart=")
            || line.starts_with("ExecStartPre=")
            || line.starts_with("ExecStartPost=")
        {
            if line.contains("/jope_selfplay_supply ") {
                validate_supply_pytorch_boundary(line)?;
            } else {
                validate_systemd_python_boundary(line)?;
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

fn systemd_logical_lines(contents: &str) -> Result<Vec<String>> {
    let mut logical = Vec::new();
    let mut continued = String::new();
    for physical in contents.lines() {
        let physical = physical.trim();
        let (fragment, continues) = physical
            .strip_suffix('\\')
            .map(|fragment| (fragment.trim_end(), true))
            .unwrap_or((physical, false));
        if !continued.is_empty() && !fragment.is_empty() {
            continued.push(' ');
        }
        continued.push_str(fragment);
        if !continues {
            logical.push(std::mem::take(&mut continued));
        }
    }
    ensure!(
        continued.is_empty(),
        "systemd unit ends with a dangling line continuation"
    );
    Ok(logical)
}

fn validate_systemd_python_boundary(line: &str) -> Result<()> {
    let allowed_root = Path::new(ACTIVE_BUNDLE_PYTORCH_ROOT);
    for (index, token) in line.split_ascii_whitespace().enumerate() {
        let value = token
            .split_once('=')
            .map(|(_, value)| value)
            .unwrap_or(token)
            .trim_matches(['\'', '"']);
        let path = Path::new(value);
        let basename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(value)
            .to_ascii_lowercase();
        let is_python_interpreter = !basename.contains(".py")
            && (basename.starts_with("python") || basename.starts_with("pypy"));
        ensure!(
            !is_python_interpreter,
            "systemd may not invoke a Python interpreter directly"
        );
        if value.to_ascii_lowercase().contains(".py") {
            ensure!(
                index != 0
                    && !value.contains(':')
                    && path.is_absolute()
                    && path.starts_with(allowed_root)
                    && path.extension().and_then(|extension| extension.to_str()) == Some("py")
                    && !path.components().any(|component| {
                        matches!(
                            component,
                            std::path::Component::ParentDir | std::path::Component::CurDir
                        )
                    }),
                "Python script reference escaped the active bundle PyTorch boundary"
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

fn sync_bundle_to_gpu_nodes(
    output_root: &Path,
    bundle_sha256: &str,
    manifest: &Value,
) -> Result<Value> {
    let source = output_root.join(bundle_sha256);
    let local_manifest_sha256 = sha256_file(&source.join("bundle.json"))?;
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("bundle manifest lacks file inventory for synchronization")?;
    let expected = rows
        .iter()
        .map(|row| {
            Ok((
                row.get("target")
                    .and_then(Value::as_str)
                    .context("bundle file lacks target")?,
                row.get("sha256")
                    .and_then(Value::as_str)
                    .context("bundle file lacks digest")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let hostname = local_hostname()?;
    let mut receipts = Vec::new();
    for node in GPU_SYNC_NODES {
        let (transport, observed) = if node_is_local(node, &hostname) {
            let mut observed = expected
                .iter()
                .map(|(target, _)| sha256_file(&source.join(target)))
                .collect::<Result<Vec<_>>>()?;
            observed.push(local_manifest_sha256.clone());
            ("local", observed)
        } else {
            let sync = std::process::Command::new("rsync")
                .arg("-a")
                .arg(&source)
                .arg(format!("{node}:{}/", output_root.display()))
                .output()
                .with_context(|| format!("spawn rsync of bundle {bundle_sha256} to {node}"))?;
            ensure!(
                sync.status.success(),
                "bundle sync to {node} failed: {}",
                String::from_utf8_lossy(&sync.stderr)
            );
            let mut command = std::process::Command::new("ssh");
            command.args(["-o", "BatchMode=yes", node, "sha256sum"]);
            for (target, _) in &expected {
                command.arg(source.join(target));
            }
            command.arg(source.join("bundle.json"));
            let check = command
                .output()
                .with_context(|| format!("spawn bundle sync verification on {node}"))?;
            ensure!(
                check.status.success(),
                "bundle {bundle_sha256} hash verification failed on {node}: {}",
                String::from_utf8_lossy(&check.stderr).trim()
            );
            let observed = String::from_utf8(check.stdout)?
                .lines()
                .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
                .collect::<Vec<_>>();
            ("ssh+rsync", observed)
        };
        ensure!(
            observed.len() == expected.len() + 1
                && observed
                    .iter()
                    .zip(expected.iter())
                    .all(|(actual, (_, digest))| actual == digest)
                && observed.last().map(String::as_str) == Some(local_manifest_sha256.as_str()),
            "bundle {bundle_sha256} failed complete asset verification on {node}"
        );
        receipts.push(json!({
            "node": node,
            "bundle_sha256": bundle_sha256,
            "bundle_manifest_sha256": local_manifest_sha256,
            "files_verified": expected.len(),
            "transport": transport,
            "verified": true,
        }));
    }
    Ok(json!({
        "schema": "JopeBundleSynchronizationReceipt.v1",
        "generated_at_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "bundle_sha256": bundle_sha256,
        "required_nodes": GPU_SYNC_NODES,
        "receipts": receipts,
        "complete": receipts.len() == GPU_SYNC_NODES.len(),
    }))
}

fn node_command(
    node: &str,
    local_hostname: &str,
    program: &str,
    arguments: &[String],
) -> Result<std::process::Output> {
    let mut command = if node_is_local(node, local_hostname) {
        Command::new(program)
    } else {
        let mut command = Command::new("ssh");
        command.args(["-o", "BatchMode=yes", node, program]);
        command
    };
    command.args(arguments);
    command
        .output()
        .with_context(|| format!("run fresh activation probe on {node}: {program}"))
}

/// Repeat a fresh, verifier-challenged live probe during activation. This does
/// not consume the capsule's supplied node receipts and therefore cannot turn
/// a stale or fabricated receipt into live-node authority.
fn fresh_live_bundle_probe(
    output_root: &Path,
    bundle_sha256: &str,
    manifest: &Value,
) -> Result<Value> {
    let started_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let challenge_nonce_sha256 = sha256_bytes(
        format!(
            "jope-live-activation-probe-v2\0{bundle_sha256}\0{started_at_ms}\0{}",
            std::process::id()
        )
        .as_bytes(),
    );
    let root = output_root.join(bundle_sha256);
    let expected = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("bundle manifest lacks live-probe inventory")?
        .iter()
        .map(|row| {
            Ok((
                row.get("target")
                    .and_then(Value::as_str)
                    .context("live-probe target is absent")?
                    .to_owned(),
                row.get("sha256")
                    .and_then(Value::as_str)
                    .context("live-probe digest is absent")?
                    .to_owned(),
                row.get("bytes")
                    .and_then(Value::as_u64)
                    .context("live-probe size is absent")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let systemd = expected
        .iter()
        .filter(|(target, _, _)| {
            target.starts_with("runtime/systemd/")
                && (target.ends_with(".service") || target.ends_with(".timer"))
        })
        .map(|(target, digest, bytes)| json!({"target": target, "sha256": digest, "bytes": bytes}))
        .collect::<Vec<_>>();
    ensure!(
        !systemd.is_empty(),
        "live probe found no bundled systemd artifacts"
    );
    let manifest_file_sha256 = sha256_file(&root.join("bundle.json"))?;
    let manifest_canonical_json_sha256 = canonical_json_sha256(manifest)?;
    let hostname = local_hostname()?;
    let paths = expected
        .iter()
        .map(|(target, _, _)| root.join(target).display().to_string())
        .chain(std::iter::once(
            root.join("bundle.json").display().to_string(),
        ))
        .collect::<Vec<_>>();
    let mut receipts = Vec::new();
    for node in GPU_SYNC_NODES {
        let node_started_at_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let observed_hostname = node_command(node, &hostname, "hostname", &[])?;
        ensure!(
            observed_hostname.status.success(),
            "fresh hostname probe failed on {node}"
        );
        let observed_hostname = String::from_utf8(observed_hostname.stdout)?
            .trim()
            .to_owned();
        ensure!(
            observed_hostname.split('.').next() == Some(*node),
            "fresh activation probe observed the wrong hostname for {node}"
        );
        let hashes = node_command(node, &hostname, "sha256sum", &paths)?;
        ensure!(
            hashes.status.success(),
            "fresh bundle hash probe failed on {node}"
        );
        let observed = String::from_utf8(hashes.stdout.clone())?
            .lines()
            .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
            .collect::<Vec<_>>();
        ensure!(
            observed.len() == expected.len() + 1
                && observed
                    .iter()
                    .zip(expected.iter())
                    .all(|(actual, (_, digest, _))| actual == digest)
                && observed.last().map(String::as_str) == Some(manifest_file_sha256.as_str()),
            "fresh activation probe found incomplete or altered bytes on {node}"
        );
        let tool_path = "/usr/bin/sha256sum".to_owned();
        let tool_hash = node_command(
            node,
            &hostname,
            "sha256sum",
            std::slice::from_ref(&tool_path),
        )?;
        ensure!(
            tool_hash.status.success(),
            "fresh probe tool hash failed on {node}"
        );
        let tool_sha256 = String::from_utf8(tool_hash.stdout.clone())?
            .split_whitespace()
            .next()
            .context("fresh probe tool digest is absent")?
            .to_owned();
        require_bundle_sha256(&tool_sha256)?;
        let version = node_command(node, &hostname, "sha256sum", &["--version".to_owned()])?;
        ensure!(
            version.status.success(),
            "fresh probe tool version failed on {node}"
        );
        let version_text = String::from_utf8(version.stdout.clone())?;
        let raw_output_sha256 = sha256_bytes(
            &[
                observed_hostname.as_bytes(),
                &hashes.stdout,
                &hashes.stderr,
                &tool_hash.stdout,
                &version.stdout,
            ]
            .concat(),
        );
        receipts.push(json!({
            "node": node,
            "observed_hostname": observed_hostname,
            "challenge_nonce_sha256": challenge_nonce_sha256,
            "runtime_bundle_sha256": bundle_sha256,
            "bundle_manifest_file_sha256": manifest_file_sha256,
            "bundle_manifest_canonical_json_sha256": manifest_canonical_json_sha256,
            "complete_file_inventory": expected.iter().map(|(target, digest, bytes)| {
                json!({"target": target, "sha256": digest, "bytes": bytes})
            }).collect::<Vec<_>>(),
            "complete_systemd_artifact_inventory": systemd,
            "tool": {
                "path": tool_path,
                "executable_sha256": tool_sha256,
                "version": version_text.lines().next().unwrap_or_default(),
            },
            "started_at_unix_ms": node_started_at_ms,
            "finished_at_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "raw_output_sha256": raw_output_sha256,
        }));
    }
    ensure!(
        receipts.len() == GPU_SYNC_NODES.len(),
        "fresh five-node probe is incomplete"
    );
    let receipt = json!({
        "schema": "JopeLiveBundleProbe.v2",
        "challenge_nonce_sha256": challenge_nonce_sha256,
        "runtime_bundle_sha256": bundle_sha256,
        "generated_at_unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "receipts": receipts,
    });
    let path = output_root
        .join("activation-probes")
        .join(format!("{bundle_sha256}-{challenge_nonce_sha256}.v2.json"));
    atomic_json(&path, &receipt)?;
    Ok(json!({
        "receipt": path,
        "receipt_file_sha256": sha256_file(&path)?,
        "challenge_nonce_sha256": challenge_nonce_sha256,
    }))
}

fn local_hostname() -> Result<String> {
    let output = std::process::Command::new("hostname")
        .output()
        .context("read local hostname for bundle synchronization")?;
    ensure!(
        output.status.success(),
        "local hostname lookup failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let hostname = String::from_utf8(output.stdout)?.trim().to_owned();
    ensure!(
        !hostname.is_empty(),
        "local hostname lookup returned empty output"
    );
    Ok(hostname)
}

fn node_is_local(node: &str, hostname: &str) -> bool {
    hostname.split('.').next() == Some(node)
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

fn prime_support_provenance(
    args: &Args,
    candidates: &[Candidate],
    inventory: &[InventoryRow],
) -> Result<PrimeSupportProvenance> {
    let source_bundle = args
        .prime
        .parent()
        .context("Prime capsule has no source bundle directory")?;
    ensure!(
        source_bundle.file_name().and_then(|name| name.to_str())
            == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "Prime support must come from immutable bundle {PRIME_SOURCE_BUNDLE_SHA256}"
    );
    let manifest_path = source_bundle.join("bundle.json");
    let manifest: Value = serde_json::from_reader(
        File::open(&manifest_path)
            .with_context(|| format!("open Prime source bundle {}", manifest_path.display()))?,
    )?;
    ensure!(
        manifest.get("bundle_sha256").and_then(Value::as_str) == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "Prime source bundle identity drift"
    );
    let source_rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("Prime source bundle lacks file inventory")?;
    let mut files = Vec::new();
    let mut seen = BTreeSet::new();
    for (candidate, row) in candidates.iter().zip(inventory) {
        if !candidate.source.starts_with(&args.prime) {
            continue;
        }
        let relative = candidate.source.strip_prefix(&args.prime)?;
        let source_target = format!("pytorch/{}", relative.display());
        let source_row = source_rows
            .iter()
            .find(|source| source.get("target").and_then(Value::as_str) == Some(&source_target))
            .with_context(|| format!("Prime source bundle lacks {source_target}"))?;
        ensure!(
            source_row.get("sha256").and_then(Value::as_str) == Some(row.sha256.as_str())
                && source_row.get("bytes").and_then(Value::as_u64) == Some(row.bytes),
            "Prime source bundle content drift for {source_target}"
        );
        if seen.insert((source_target.clone(), row.target.clone())) {
            files.push(PrimeSupportFile {
                source_bundle_target: source_target,
                bundle_target: row.target.clone(),
                sha256: row.sha256.clone(),
                bytes: row.bytes,
            });
        }
    }
    ensure!(!files.is_empty(), "Prime support capsule is empty");
    files.sort_by(|left, right| {
        left.source_bundle_target
            .cmp(&right.source_bundle_target)
            .then_with(|| left.bundle_target.cmp(&right.bundle_target))
    });
    let package_tree_sha256 = prime_support_identity(&files)?;
    Ok(PrimeSupportProvenance {
        source_bundle_sha256: PRIME_SOURCE_BUNDLE_SHA256,
        source_bundle_manifest_sha256: sha256_file(&manifest_path)?,
        package_tree_sha256,
        files,
    })
}

fn prime_support_identity(files: &[PrimeSupportFile]) -> Result<String> {
    let rows = files
        .iter()
        .map(|row| {
            json!([
                row.source_bundle_target,
                row.bundle_target,
                row.sha256,
                row.bytes,
            ])
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&serde_json::to_vec(&rows)?))
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

fn require_bundle_sha256(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "bundle identity is not a lowercase SHA-256"
    );
    Ok(())
}

fn manifest_inventory_identity(rows: &[Value]) -> Result<String> {
    let identity_rows = rows
        .iter()
        .map(|row| {
            json!({
                "target": row.get("target").cloned().unwrap_or(Value::Null),
                "sha256": row.get("sha256").cloned().unwrap_or(Value::Null),
                "bytes": row.get("bytes").cloned().unwrap_or(Value::Null),
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&serde_json::to_vec(&identity_rows)?))
}

fn rollback_derivation_path(output_root: &Path, bundle_sha256: &str) -> PathBuf {
    output_root
        .join("rollback-derivations")
        .join(format!("{bundle_sha256}.v1.json"))
}

fn verify_intrinsic_existing(output_root: &Path, bundle_sha256: &str) -> Result<Value> {
    require_bundle_sha256(bundle_sha256)?;
    let root = output_root.join(bundle_sha256);
    let path = root.join("bundle.json");
    let manifest: Value = serde_json::from_reader(
        File::open(&path).with_context(|| format!("open staged bundle {}", path.display()))?,
    )?;
    if manifest
        .get("clean_detached_source")
        .and_then(Value::as_bool)
        == Some(true)
    {
        verify_existing(&root, &manifest)?;
    } else {
        ensure!(
            bundle_sha256 == HISTORICAL_ROLLBACK_PARENT_SHA256
                || rollback_derivation_path(output_root, bundle_sha256).is_file(),
            "legacy-provenance bundle is neither the historical rollback parent nor a strict derivation"
        );
        verify_rollback_candidate(&root, &manifest, bundle_sha256)?;
        let rows = manifest
            .get("files")
            .and_then(Value::as_array)
            .context("legacy bundle lacks file inventory")?;
        ensure!(
            manifest_inventory_identity(rows)? == bundle_sha256,
            "legacy bundle inventory identity drift"
        );
        if bundle_sha256 != HISTORICAL_ROLLBACK_PARENT_SHA256 {
            verify_rollback_derivation(output_root, bundle_sha256)?;
        }
    }
    Ok(manifest)
}

fn rollback_bundle_files(manifest: &Value) -> Result<Vec<RollbackBundleFileV2>> {
    let mut files = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?
        .iter()
        .map(|row| {
            Ok(RollbackBundleFileV2 {
                target: row
                    .get("target")
                    .and_then(Value::as_str)
                    .context("rollback file target is absent")?
                    .to_owned(),
                bytes: row
                    .get("bytes")
                    .and_then(Value::as_u64)
                    .context("rollback file size is absent")?,
                sha256: row
                    .get("sha256")
                    .and_then(Value::as_str)
                    .context("rollback file digest is absent")?
                    .to_owned(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    files.sort_by(|left, right| left.target.cmp(&right.target));
    ensure!(
        files.windows(2).all(|pair| pair[0].target < pair[1].target),
        "rollback file inventory is duplicated"
    );
    Ok(files)
}

fn parse_systemd_artifact(
    root: &Path,
    target: &str,
    expected_sha256: &str,
) -> Result<RollbackSystemdArtifactV2> {
    let unit = Path::new(target)
        .file_name()
        .and_then(|name| name.to_str())
        .context("systemd target has no unit name")?
        .to_owned();
    let kind = if unit.ends_with(".service") {
        "service"
    } else if unit.ends_with(".timer") {
        "timer"
    } else {
        bail!("unsupported systemd artifact: {unit}")
    };
    let contents = fs::read_to_string(root.join(target))?;
    ensure!(
        sha256_bytes(contents.as_bytes()) == expected_sha256,
        "systemd artifact digest differs from bundle inventory: {unit}"
    );
    let logical = systemd_logical_lines(&contents)?;
    let exec_start_argv = if kind == "service" {
        let line = logical
            .iter()
            .find_map(|line| line.trim().strip_prefix("ExecStart="))
            .with_context(|| format!("rollback service has no ExecStart: {unit}"))?;
        line.split_ascii_whitespace()
            .map(|argument| argument.trim_matches(['\'', '"']).to_owned())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let working_directory = logical
        .iter()
        .find_map(|line| line.trim().strip_prefix("WorkingDirectory="))
        .map(str::to_owned);
    let mut artifact = RollbackSystemdArtifactV2 {
        unit,
        kind: kind.to_owned(),
        bundled_target: target.to_owned(),
        unit_file_sha256: expected_sha256.to_owned(),
        resolved_effective_unit_sha256: String::new(),
        exec_start_argv,
        working_directory,
        drop_ins: Vec::<ArtifactIdentity>::new(),
    };
    artifact.resolved_effective_unit_sha256 = artifact.computed_effective_unit_sha256()?;
    Ok(artifact)
}

fn systemd_closure(root: &Path, manifest: &Value) -> Result<Vec<RollbackSystemdArtifactV2>> {
    let mut artifacts = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?
        .iter()
        .filter_map(|row| {
            let target = row.get("target")?.as_str()?;
            (target.starts_with("runtime/systemd/")
                && (target.ends_with(".service") || target.ends_with(".timer")))
            .then_some((target, row.get("sha256")?.as_str()?))
        })
        .map(|(target, digest)| parse_systemd_artifact(root, target, digest))
        .collect::<Result<Vec<_>>>()?;
    artifacts.sort_by(|left, right| left.unit.cmp(&right.unit));
    ensure!(
        artifacts.windows(2).all(|pair| pair[0].unit < pair[1].unit),
        "rollback systemd inventory is duplicated"
    );
    Ok(artifacts)
}

fn unchanged_inventory_root(rows: &[Value], changed: &BTreeSet<String>) -> Result<String> {
    canonical_json_sha256(&Value::Array(
        rows.iter()
            .filter(|row| {
                row.get("target")
                    .and_then(Value::as_str)
                    .is_some_and(|target| !changed.contains(target))
            })
            .cloned()
            .collect(),
    ))
}

fn git_blob(source: &Path, commit: &str, path: &str) -> Result<Vec<u8>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(source)
        .arg("show")
        .arg(format!("{commit}:{path}"))
        .output()
        .with_context(|| format!("read reviewed source {commit}:{path}"))?;
    ensure!(
        output.status.success(),
        "reviewed source is unreachable: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output.stdout)
}

fn transform_rollback_unit(unit: &str, before: &[u8], reviewed: &[u8]) -> Result<Vec<u8>> {
    let before_text = String::from_utf8(before.to_vec())?;
    let from_line = format!("WorkingDirectory={ROLLBACK_WORKING_DIRECTORY_FROM}");
    let to_line = format!("WorkingDirectory={IMMUTABLE_FLYWHEEL_ROOT}");
    ensure!(
        before_text
            .lines()
            .filter(|line| *line == from_line)
            .count()
            == 1,
        "rollback parent unit does not contain exactly one permitted line: {unit}"
    );
    let reviewed_text = String::from_utf8(reviewed.to_vec())?;
    ensure!(
        reviewed_text.lines().any(|line| line == to_line),
        "reviewed source does not contain the canonical WorkingDirectory: {unit}"
    );
    Ok(before_text.replacen(&from_line, &to_line, 1).into_bytes())
}

fn derive_rollback_baseline(
    source: &Path,
    output_root: &Path,
) -> Result<RollbackBaselineDerivationV1> {
    let parent_root = output_root.join(HISTORICAL_ROLLBACK_PARENT_SHA256);
    let mut manifest = verify_intrinsic_existing(output_root, HISTORICAL_ROLLBACK_PARENT_SHA256)?;
    let parent_manifest_path = parent_root.join("bundle.json");
    let parent_manifest_file_sha256 = sha256_file(&parent_manifest_path)?;
    let parent_manifest_canonical_json_sha256 = canonical_json_sha256(&manifest)?;
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback parent has no file inventory")?
        .clone();
    let changed_targets = ROLLBACK_REBASE_UNITS
        .iter()
        .map(|unit| format!("runtime/systemd/{unit}"))
        .collect::<BTreeSet<_>>();
    ensure!(
        changed_targets.len() == 7,
        "rollback rebase unit allowlist is not exactly seven"
    );

    let staging = output_root.join(format!(".rollback-rebase.tmp-{}", std::process::id()));
    ensure!(
        !staging.exists(),
        "rollback rebase staging directory already exists"
    );
    fs::create_dir_all(&staging)?;
    let result = (|| -> Result<RollbackBaselineDerivationV1> {
        let mut transformations = Vec::new();
        for row in &rows {
            let target = row
                .get("target")
                .and_then(Value::as_str)
                .context("rollback parent row lacks target")?;
            let destination = staging.join(target);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(parent_root.join(target), &destination)?;
        }
        let manifest_rows = manifest
            .get_mut("files")
            .and_then(Value::as_array_mut)
            .context("rollback parent file inventory is not mutable")?;
        for unit in ROLLBACK_REBASE_UNITS {
            let target = format!("runtime/systemd/{unit}");
            let path = staging.join(&target);
            let before = fs::read(&path)?;
            let reviewed = git_blob(
                source,
                REVIEWED_ROLLBACK_UNIT_COMMIT,
                &format!("systemd/{unit}"),
            )?;
            let after = transform_rollback_unit(unit, &before, &reviewed)?;
            fs::write(&path, &after)?;
            let before_sha256 = sha256_bytes(&before);
            let after_sha256 = sha256_bytes(&after);
            let manifest_row = manifest_rows
                .iter_mut()
                .find(|row| row.get("target").and_then(Value::as_str) == Some(target.as_str()))
                .with_context(|| format!("rollback parent inventory omits {target}"))?;
            ensure!(
                manifest_row.get("sha256").and_then(Value::as_str) == Some(before_sha256.as_str()),
                "rollback parent unit inventory drift: {unit}"
            );
            manifest_row["sha256"] = Value::String(after_sha256.clone());
            manifest_row["bytes"] = Value::from(after.len() as u64);
            transformations.push(RollbackUnitTransformationV1 {
                target,
                before_sha256,
                after_sha256,
                reviewed_source_sha256: sha256_bytes(&reviewed),
                from: ROLLBACK_WORKING_DIRECTORY_FROM.to_owned(),
                to: IMMUTABLE_FLYWHEEL_ROOT.to_owned(),
            });
        }
        transformations.sort_by(|left, right| left.target.cmp(&right.target));
        let derived_bundle_sha256 = manifest_inventory_identity(manifest_rows)?;
        manifest["bundle_sha256"] = Value::String(derived_bundle_sha256.clone());
        atomic_json(&staging.join("bundle.json"), &manifest)?;
        let final_root = output_root.join(&derived_bundle_sha256);
        if final_root.exists() {
            let existing = verify_intrinsic_existing(output_root, &derived_bundle_sha256)?;
            ensure!(
                existing == manifest,
                "existing derived rollback bundle manifest drift"
            );
            fs::remove_dir_all(&staging)?;
        } else {
            fs::rename(&staging, &final_root)?;
        }
        let derivation = RollbackBaselineDerivationV1 {
            schema: ROLLBACK_BASELINE_DERIVATION_SCHEMA.to_owned(),
            parent_bundle_sha256: HISTORICAL_ROLLBACK_PARENT_SHA256.to_owned(),
            derived_bundle_sha256: derived_bundle_sha256.clone(),
            parent_manifest_file_sha256,
            parent_manifest_canonical_json_sha256,
            reviewed_source_commit: REVIEWED_ROLLBACK_UNIT_COMMIT.to_owned(),
            transformations,
            unchanged_inventory_root_sha256: unchanged_inventory_root(&rows, &changed_targets)?,
        };
        derivation.validate()?;
        atomic_json(
            &rollback_derivation_path(output_root, &derived_bundle_sha256),
            &serde_json::to_value(&derivation)?,
        )?;
        verify_rollback_derivation(output_root, &derived_bundle_sha256)?;
        Ok(derivation)
    })();
    if result.is_err() && staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    result
}

fn verify_rollback_derivation(output_root: &Path, derived_bundle_sha256: &str) -> Result<()> {
    let derivation_path = rollback_derivation_path(output_root, derived_bundle_sha256);
    let derivation: RollbackBaselineDerivationV1 = serde_json::from_reader(
        File::open(&derivation_path)
            .with_context(|| format!("open rollback derivation {}", derivation_path.display()))?,
    )?;
    derivation.validate()?;
    ensure!(
        derivation.derived_bundle_sha256 == derived_bundle_sha256,
        "rollback derivation output identity mismatch"
    );
    let parent_root = output_root.join(&derivation.parent_bundle_sha256);
    let derived_root = output_root.join(derived_bundle_sha256);
    let parent_manifest_path = parent_root.join("bundle.json");
    let parent_manifest: Value = serde_json::from_reader(File::open(&parent_manifest_path)?)?;
    let derived_manifest: Value =
        serde_json::from_reader(File::open(derived_root.join("bundle.json"))?)?;
    ensure!(
        sha256_file(&parent_manifest_path)? == derivation.parent_manifest_file_sha256
            && canonical_json_sha256(&parent_manifest)?
                == derivation.parent_manifest_canonical_json_sha256,
        "rollback derivation parent manifest drift"
    );
    let changed = derivation
        .transformations
        .iter()
        .map(|row| row.target.clone())
        .collect::<BTreeSet<_>>();
    let parent_rows = parent_manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback parent lacks files")?;
    let derived_rows = derived_manifest
        .get("files")
        .and_then(Value::as_array)
        .context("derived rollback lacks files")?;
    ensure!(
        parent_rows.len() == derived_rows.len()
            && unchanged_inventory_root(parent_rows, &changed)?
                == derivation.unchanged_inventory_root_sha256,
        "rollback derivation inventory shape/root drift"
    );
    for (parent, derived) in parent_rows.iter().zip(derived_rows) {
        let target = parent
            .get("target")
            .and_then(Value::as_str)
            .context("rollback parent target is absent")?;
        ensure!(
            derived.get("target").and_then(Value::as_str) == Some(target),
            "rollback derivation reordered or replaced an inventory target"
        );
        if !changed.contains(target) {
            ensure!(
                parent == derived,
                "rollback derivation changed a non-unit manifest row: {target}"
            );
            ensure!(
                sha256_file(&parent_root.join(target))? == sha256_file(&derived_root.join(target))?,
                "rollback derivation changed non-unit bytes: {target}"
            );
            continue;
        }
        let transformation = derivation
            .transformations
            .iter()
            .find(|row| row.target == target)
            .context("rollback derivation changed an unrecorded target")?;
        let before = fs::read(parent_root.join(target))?;
        let after = fs::read(derived_root.join(target))?;
        ensure!(
            sha256_bytes(&before) == transformation.before_sha256
                && sha256_bytes(&after) == transformation.after_sha256,
            "rollback derivation unit digest drift: {target}"
        );
        let expected = String::from_utf8(before)?.replacen(
            &format!("WorkingDirectory={}", transformation.from),
            &format!("WorkingDirectory={}", transformation.to),
            1,
        );
        ensure!(
            expected.as_bytes() == after.as_slice(),
            "rollback derivation made a change other than the permitted WorkingDirectory replacement"
        );
    }
    ensure!(
        manifest_inventory_identity(derived_rows)? == derived_bundle_sha256,
        "derived rollback bundle identity drift"
    );
    Ok(())
}

fn verify_rollback_candidate(root: &Path, manifest: &Value, expected_sha256: &str) -> Result<()> {
    ensure!(
        manifest.get("bundle_sha256").and_then(Value::as_str) == Some(expected_sha256)
            && root.file_name().and_then(|name| name.to_str()) == Some(expected_sha256),
        "rollback candidate identity drift"
    );
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?;
    let targets = rows
        .iter()
        .map(|row| {
            let target = row
                .get("target")
                .and_then(Value::as_str)
                .context("rollback file target is absent")?;
            let expected = row
                .get("sha256")
                .and_then(Value::as_str)
                .context("rollback file digest is absent")?;
            ensure!(
                sha256_file(&root.join(target))? == expected,
                "rollback bundle content drift: {target}"
            );
            Ok(target)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let service: ServiceManifest = serde_json::from_value(
        manifest
            .get("service_manifest")
            .cloned()
            .context("rollback bundle lacks service manifest")?,
    )?;
    validate_service_manifest(&service)?;
    for row in &service.services {
        ensure!(
            targets.contains(format!("runtime/bin/{}", row.binary).as_str())
                && targets.contains(format!("runtime/systemd/{}", row.unit).as_str()),
            "rollback service/unit closure is incomplete for {}",
            row.unit
        );
    }
    ensure!(
        targets.contains("runtime/bin/corpus_eval"),
        "rollback bundle lacks evaluator"
    );
    Ok(())
}

fn verify_rollback_capsule(
    output_root: &Path,
    evidence_root: &Path,
    capsule: &RollbackCapsuleV2,
    now_unix_ms: u64,
) -> Result<String> {
    capsule.validate_evidence_at(evidence_root, now_unix_ms)?;
    let runtime_root = output_root.join(&capsule.runtime_bundle_sha256);
    let runtime_manifest_path = runtime_root.join("bundle.json");
    ensure!(
        sha256_file(&runtime_manifest_path)? == capsule.runtime_bundle_manifest_file_sha256,
        "rollback runtime manifest digest mismatch"
    );
    let runtime_manifest: Value = serde_json::from_reader(File::open(&runtime_manifest_path)?)?;
    ensure!(
        canonical_json_sha256(&runtime_manifest)?
            == capsule.runtime_bundle_manifest_canonical_json_sha256,
        "rollback runtime manifest canonical digest mismatch"
    );
    verify_rollback_candidate(
        &runtime_root,
        &runtime_manifest,
        &capsule.runtime_bundle_sha256,
    )?;

    let service_value = runtime_manifest
        .get("service_manifest")
        .cloned()
        .context("rollback runtime lacks service manifest")?;
    ensure!(
        canonical_json_sha256(&service_value)?
            == capsule.embedded_service_manifest_canonical_json_sha256,
        "rollback embedded service manifest canonical digest mismatch"
    );
    let service: ServiceManifest = serde_json::from_value(service_value)?;
    let service_target = "runtime/manifests/selfplay/service-manifest.v3.json";
    let service_path = runtime_root.join(service_target);
    ensure!(
        sha256_file(&service_path)? == capsule.embedded_service_manifest_file_sha256
            && runtime_manifest
                .get("service_manifest_sha256")
                .and_then(Value::as_str)
                == Some(capsule.embedded_service_manifest_file_sha256.as_str()),
        "rollback service-manifest raw-file digest mismatch"
    );
    let expected_units = service
        .services
        .iter()
        .map(|row| row.unit.as_str())
        .collect::<BTreeSet<_>>();
    let capsule_services = capsule
        .systemd_closure
        .iter()
        .filter(|row| row.kind == "service")
        .map(|row| row.unit.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        capsule_services == expected_units,
        "rollback systemd closure does not contain the complete service inventory"
    );
    let actual_files = rollback_bundle_files(&runtime_manifest)?;
    ensure!(
        actual_files == capsule.files,
        "rollback capsule file inventory differs from the bundle manifest"
    );
    let actual_systemd = systemd_closure(&runtime_root, &runtime_manifest)?;
    ensure!(
        actual_systemd == capsule.systemd_closure,
        "rollback capsule systemd closure differs from bundled unit bytes"
    );
    capsule.digest_at(evidence_root, now_unix_ms)
}

fn verify_existing(root: &Path, manifest: &Value) -> Result<()> {
    ensure!(manifest["runtime"] == "rust", "bundle runtime is not Rust");
    ensure!(
        manifest["operational_python_allowed"] == false,
        "bundle permits operational Python"
    );
    ensure!(
        manifest
            .get("clean_detached_source")
            .and_then(Value::as_bool)
            == Some(true)
            && manifest
                .get("integration_commit")
                .and_then(Value::as_str)
                .is_some_and(|commit| {
                    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
                }),
        "bundle lacks clean detached integration provenance"
    );
    ensure!(
        manifest
            .pointer("/prime_support/source_bundle_sha256")
            .and_then(Value::as_str)
            == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "bundle does not bind immutable Prime support"
    );
    let required_nodes = manifest
        .get("required_sync_nodes")
        .and_then(Value::as_array)
        .context("bundle lacks five-node synchronization contract")?;
    ensure!(
        required_nodes
            == &GPU_SYNC_NODES
                .iter()
                .map(|node| Value::String((*node).to_owned()))
                .collect::<Vec<_>>(),
        "bundle synchronization node set drift"
    );
    ensure!(
        manifest
            .pointer("/five_node_synchronization/schema")
            .and_then(Value::as_str)
            == Some("JopeBundleSynchronizationContract.v1")
            && manifest
                .pointer("/five_node_synchronization/receipt_schema")
                .and_then(Value::as_str)
                == Some("JopeBundleSynchronizationReceipt.v1")
            && manifest
                .pointer("/five_node_synchronization/activation_requires_complete")
                .and_then(Value::as_bool)
                == Some(true)
            && manifest
                .pointer("/five_node_synchronization/required_nodes")
                .and_then(Value::as_array)
                == Some(required_nodes)
            && manifest
                .pointer("/five_node_synchronization/receipt_relative_path")
                .and_then(Value::as_str)
                == manifest
                    .get("bundle_sha256")
                    .and_then(Value::as_str)
                    .map(|identity| format!("synchronization/{identity}.v1.json"))
                    .as_deref(),
        "bundle synchronization receipt contract drift"
    );
    let rows = manifest["files"].as_array().context("manifest files")?;
    let service_manifest: ServiceManifest = serde_json::from_value(
        manifest
            .get("service_manifest")
            .cloned()
            .context("bundle lacks embedded service manifest")?,
    )?;
    validate_recovery_service_manifest(&service_manifest)?;
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
    for service in service_manifest.services.iter().filter(|row| row.active) {
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/bin/{}", service.binary).as_str())
            }) && rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/systemd/{}", service.unit).as_str())
            }),
            "bundle service/unit closure is incomplete for {}",
            service.unit
        );
    }
    let prime_files = manifest
        .pointer("/prime_support/files")
        .and_then(Value::as_array)
        .context("bundle lacks Prime per-file provenance")?;
    let mut canonical_prime_files =
        serde_json::from_value::<Vec<PrimeSupportFile>>(Value::Array(prime_files.clone()))?;
    canonical_prime_files.sort_by(|left, right| {
        left.source_bundle_target
            .cmp(&right.source_bundle_target)
            .then_with(|| left.bundle_target.cmp(&right.bundle_target))
    });
    let computed_prime_tree_sha256 = prime_support_identity(&canonical_prime_files)?;
    ensure!(
        manifest
            .pointer("/prime_support/package_tree_sha256")
            .and_then(Value::as_str)
            == Some(computed_prime_tree_sha256.as_str()),
        "Prime package-tree digest drift"
    );
    for prime in prime_files {
        let target = prime
            .get("bundle_target")
            .and_then(Value::as_str)
            .context("Prime support row lacks bundle target")?;
        let digest = prime
            .get("sha256")
            .and_then(Value::as_str)
            .context("Prime support row lacks digest")?;
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str) == Some(target)
                    && row.get("sha256").and_then(Value::as_str) == Some(digest)
            }),
            "Prime support row is absent from bundle inventory: {target}"
        );
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

fn validate_recovery_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    validate_service_manifest(manifest)?;
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-reporting.service" && row.binary == "jope_reporting" && row.active
        }),
        "recovery bundle must install and activate jope-reporting.service"
    );
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-selfplay-expansion.service"
                && row.binary == "jope_selfplay_expansion_driver"
                && !row.active
        }),
        "recovery bundle must carry expansion as an inactive Encoder V2-gated unit"
    );
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
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn synchronization_does_not_ssh_the_scheduler_local_node() {
        let hostname = "xbabe2";
        let local = GPU_SYNC_NODES
            .iter()
            .copied()
            .filter(|node| node_is_local(node, hostname))
            .collect::<Vec<_>>();
        assert_eq!(local, vec!["xbabe2"]);
        assert!(node_is_local("xbabe2", "xbabe2.cluster.local"));
        for remote in ["xbabe0", "xbabe1", "xbabe3", "xbabe4"] {
            assert!(!node_is_local(remote, hostname));
        }
    }

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
            source: prime.join("jope/data/active.py"),
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
        assert!(validate_systemd_unit(
            "[Service]\nWorkingDirectory=/home/ubuntu/jope-dime\nExecStart=/runtime/bin/tool\n"
        )
        .is_err());
    }

    #[test]
    fn systemd_accepts_only_bundle_local_pytorch_scripts() {
        let watchdog = include_str!(
            "../../../../deploy/systemd/selfplay/jope-selfplay-trainer-watchdog.service"
        );
        validate_systemd_unit(watchdog).unwrap();

        let inside = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {ACTIVE_BUNDLE_PYTORCH_ROOT}/training/test_exact_train.py\n"
        );
        validate_systemd_unit(&inside).unwrap();

        let outside =
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script /tmp/training.py\n";
        assert!(validate_systemd_unit(outside).is_err());

        let mixed = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {ACTIVE_BUNDLE_PYTORCH_ROOT}/training/test_exact_train.py --common-script /tmp/common.py\n"
        );
        assert!(validate_systemd_unit(&mixed).is_err());

        let traversal = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {ACTIVE_BUNDLE_PYTORCH_ROOT}/../outside.py\n"
        );
        assert!(validate_systemd_unit(&traversal).is_err());

        let continued = "[Service]\nExecStart=/runtime/bin/controller \\\n+  --trainer-script /tmp/training.py\n";
        assert!(validate_systemd_unit(continued).is_err());

        for interpreter in ["/usr/bin/python3-dbg", "/usr/bin/pypy3"] {
            let unit = format!("[Service]\nExecStart={interpreter} --version\n");
            assert!(validate_systemd_unit(&unit).is_err());
        }

        let packed = format!(
            "[Service]\nExecStart=/runtime/bin/controller --pythonpath {ACTIVE_BUNDLE_PYTORCH_ROOT}/training/test_exact_train.py:/tmp/outside.py\n"
        );
        assert!(validate_systemd_unit(&packed).is_err());

        let direct_script = format!(
            "[Service]\nExecStart={ACTIVE_BUNDLE_PYTORCH_ROOT}/training/test_exact_train.py\n"
        );
        assert!(validate_systemd_unit(&direct_script).is_err());
    }

    #[test]
    fn every_tracked_systemd_unit_satisfies_the_runtime_boundary() {
        let systemd = Path::new(env!("CARGO_MANIFEST_DIR")).join("systemd");
        for entry in fs::read_dir(&systemd).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("service") {
                continue;
            }
            let contents = fs::read_to_string(&path).unwrap();
            validate_systemd_unit(&contents)
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
        }
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

    #[test]
    fn recovery_manifest_rejects_missing_reporting_or_active_expansion() {
        let mut manifest = ServiceManifest {
            schema: "JopeSelfplayServiceManifest.v3".to_owned(),
            binaries: vec![
                "jope_reporting".to_owned(),
                "jope_selfplay_expansion_driver".to_owned(),
            ],
            services: vec![ServiceRow {
                unit: "jope-selfplay-expansion.service".to_owned(),
                binary: "jope_selfplay_expansion_driver".to_owned(),
                owner: "encoder_v2_identity_gate_inactive".to_owned(),
                active: false,
            }],
        };
        assert!(validate_recovery_service_manifest(&manifest).is_err());

        manifest.services.push(ServiceRow {
            unit: "jope-reporting.service".to_owned(),
            binary: "jope_reporting".to_owned(),
            owner: "recovery_reporting".to_owned(),
            active: true,
        });
        validate_recovery_service_manifest(&manifest).unwrap();

        manifest.services[0].active = true;
        assert!(validate_recovery_service_manifest(&manifest).is_err());
    }

    #[test]
    fn rollback_identity_mismatch_is_rejected_before_activation() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest = json!({"bundle_sha256": "b".repeat(64)});
        assert!(
            verify_rollback_candidate(temporary.path(), &manifest, ROLLBACK_BUNDLE_SHA256).is_err()
        );
    }

    #[test]
    fn superseded_staging_bundle_is_verifiable_but_never_activatable() {
        assert!(validate_activation_target(
            ActivationOperationV1::Activate,
            SUPERSEDED_STAGING_BUNDLE_SHA256
        )
        .is_err());
        assert!(validate_activation_target(
            ActivationOperationV1::Rollback,
            SUPERSEDED_STAGING_BUNDLE_SHA256
        )
        .is_err());
        validate_activation_target(ActivationOperationV1::Rollback, ROLLBACK_BUNDLE_SHA256)
            .unwrap();
    }

    #[test]
    fn rollback_rebase_is_exactly_the_seven_working_directory_replacements() {
        assert_eq!(ROLLBACK_REBASE_UNITS.len(), 7);
        assert_eq!(
            ROLLBACK_REBASE_UNITS.iter().collect::<BTreeSet<_>>().len(),
            7
        );
        let before = format!(
            "[Unit]\nDescription=unchanged\n[Service]\nWorkingDirectory={ROLLBACK_WORKING_DIRECTORY_FROM}\nExecStart=/runtime/bin/tool --unchanged\n"
        );
        let reviewed = format!(
            "[Unit]\nDescription=reviewed-but-not-copied\n[Service]\nWorkingDirectory={IMMUTABLE_FLYWHEEL_ROOT}\nExecStart=/different/tool\n"
        );
        for unit in ROLLBACK_REBASE_UNITS {
            let after = transform_rollback_unit(unit, before.as_bytes(), reviewed.as_bytes())
                .expect("strict unit transformation");
            let expected = before.replacen(
                &format!("WorkingDirectory={ROLLBACK_WORKING_DIRECTORY_FROM}"),
                &format!("WorkingDirectory={IMMUTABLE_FLYWHEEL_ROOT}"),
                1,
            );
            assert_eq!(after, expected.as_bytes());
            assert!(String::from_utf8(after)
                .unwrap()
                .contains("ExecStart=/runtime/bin/tool --unchanged"));
        }

        let no_source_line = b"[Service]\nExecStart=/runtime/bin/tool\n";
        assert!(transform_rollback_unit(
            ROLLBACK_REBASE_UNITS[0],
            no_source_line,
            reviewed.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn activation_retains_all_available_evaluators_and_indexes_missing_history() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let bundles = root.join("runtime-bundles");
        let active = "a".repeat(64);
        let previous = "b".repeat(64);
        for (bundle, bytes) in [
            (&active, b"active".as_slice()),
            (&previous, b"previous".as_slice()),
        ] {
            let evaluator = bundles.join(bundle).join("runtime/bin/corpus_eval");
            fs::create_dir_all(evaluator.parent().unwrap()).unwrap();
            fs::write(&evaluator, bytes).unwrap();
        }
        std::os::unix::fs::symlink(&active, bundles.join("current")).unwrap();
        let missing = "c".repeat(64);
        atomic_json(
            &root.join("ops/causal-pair-repair-state.v1.json"),
            &json!({
                "schema": "CausalPairRepairExecutorState.v1",
                "entries": {"pair": {"original_evaluator_sha256": missing}},
            }),
        )
        .unwrap();
        let index = retain_evaluators(&bundles, &active).unwrap();
        let replayed = retain_evaluators(&bundles, &active).unwrap();
        assert_eq!(replayed, index);
        assert_eq!(index["schema"], "JopeEvaluatorRetentionIndex.v1");
        assert_eq!(index["active_bundle_sha256"], active);
        assert_eq!(index["prepared_for_bundle_sha256"], active);
        assert_eq!(index["all_referenced_evaluators_retained"], false);
        assert_eq!(index["unavailable_referenced_evaluator_sha256"][0], missing);
        for (bundle, bytes) in [
            (&active, b"active".as_slice()),
            (&previous, b"previous".as_slice()),
        ] {
            let digest = sha256_bytes(bytes);
            let source = bundles.join(bundle).join("runtime/bin/corpus_eval");
            let retained = root
                .join("training/evaluator-vault")
                .join(digest)
                .join("corpus_eval");
            assert_eq!(
                sha256_file(&retained).unwrap(),
                sha256_file(&source).unwrap()
            );
            assert_eq!(
                source.metadata().unwrap().ino(),
                retained.metadata().unwrap().ino()
            );
        }
        assert!(root
            .join("training/evaluator-retention-index.v1.json")
            .is_file());
    }
}
