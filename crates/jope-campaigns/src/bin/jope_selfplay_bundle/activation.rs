use super::*;

pub(crate) fn load_activation_rollback_capsule(
    args: &Args,
) -> Result<(RollbackCapsuleV2, PathBuf)> {
    let path = args
        .rollback_capsule
        .as_ref()
        .context("activation requires --rollback-capsule")?;
    let value: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(path).with_context(|| format!("open rollback capsule {}", path.display()))?,
    ))
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

pub(crate) fn activate_manifest(
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
    let rollback_manifest: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(&rollback_path)
            .with_context(|| format!("open rollback bundle {}", rollback_path.display()))?,
    ))?;
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
        let gate: DimeExternalActivationGateV1 = serde_json::from_reader(std::io::BufReader::new(
            File::open(path)
                .with_context(|| format!("open external activation gate {}", path.display()))?,
        ))?;
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

pub(crate) fn verify_candidate_review_gate(
    args: &Args,
    manifest: &Value,
    now_unix_ms: u64,
) -> Result<String> {
    let path = args
        .candidate_review_gate
        .as_ref()
        .context("forward activation requires --candidate-review-gate")?;
    let gate: DimeCandidateReviewGateV1 = serde_json::from_reader(std::io::BufReader::new(
        File::open(path)
            .with_context(|| format!("open candidate review gate {}", path.display()))?,
    ))?;
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

pub(crate) fn validate_activation_target(
    operation: ActivationOperationV1,
    bundle_sha256: &str,
) -> Result<()> {
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

pub(crate) fn collect_evaluator_digests(value: &Value, output: &mut BTreeSet<String>) {
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

pub(crate) fn collect_json_references(path: &Path, output: &mut BTreeSet<String>) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }
    let value: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(path).with_context(|| format!("open evaluator reference {}", path.display()))?,
    ))?;
    collect_evaluator_digests(&value, output);
    Ok(())
}

pub(crate) fn evaluator_reference_index(root: &Path) -> Result<BTreeMap<String, BTreeSet<String>>> {
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
pub(crate) fn retain_evaluators(output_root: &Path, prepared_bundle_sha256: &str) -> Result<Value> {
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
        let existing: Value =
            serde_json::from_reader(std::io::BufReader::new(File::open(&index_path)?))?;
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
