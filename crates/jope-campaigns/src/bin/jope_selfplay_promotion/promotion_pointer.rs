use super::*;

pub(super) fn flock_exclusive(file: &fs::File, blocking: bool) -> Result<()> {
    if let Err(error) = flock::exclusive(file, blocking) {
        bail!("flock failed: {}", error);
    }
    Ok(())
}

/// Port of `test_exact_accept.promote`.
pub(super) fn promote(
    evidence_path: &Path,
    gate_path: &Path,
    accepted_pointer: &Path,
    immutable_root: &Path,
    enforce_pointer_ancestor: bool,
) -> Result<Value> {
    let evidence = read_json(evidence_path)?;
    // The per-task anchor file lives beside the global accepted pointer; it
    // decides which NEW branches gate (advisory-until-anchor).
    let accepted_by_task =
        load_accepted_by_task(&accepted_pointer.with_file_name("accepted-by-task.json"))?;
    let result = evaluate_gate(&evidence, accepted_by_task.as_ref())?;
    atomic_json(gate_path, &result)?;
    if result.get("pass") != Some(&Value::Bool(true)) {
        return Ok(result);
    }
    let checkpoint_sha = get_str(&result, "checkpoint_sha256")?.to_owned();
    let checkpoint = PathBuf::from(
        evidence
            .pointer("/candidate/checkpoint_path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("evidence has no candidate checkpoint path"))?,
    );
    if sha256_file(&checkpoint)? != checkpoint_sha {
        bail!("accepted checkpoint content hash mismatch");
    }
    let immutable_checkpoint = immutable_root
        .join("checkpoints")
        .join(format!("{checkpoint_sha}.model.pt"));
    fs::create_dir_all(immutable_checkpoint.parent().unwrap())?;
    if !immutable_checkpoint.exists() {
        let temporary = immutable_checkpoint.with_file_name(format!(
            ".{}.tmp-{}",
            immutable_checkpoint.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        fs::copy(&checkpoint, &temporary)?;
        if sha256_file(&temporary)? != checkpoint_sha {
            let _ = fs::remove_file(&temporary);
            bail!("checkpoint changed while copying to immutable storage");
        }
        fs::rename(&temporary, &immutable_checkpoint)?;
    } else if sha256_file(&immutable_checkpoint)? != checkpoint_sha {
        bail!("immutable checkpoint path contains different bytes");
    }
    let evidence_hash = sha256_file(evidence_path)?;
    let immutable_evidence = immutable_root
        .join("gates")
        .join(format!("{evidence_hash}.evidence.json"));
    fs::create_dir_all(immutable_evidence.parent().unwrap())?;
    if !immutable_evidence.exists() {
        fs::copy(evidence_path, &immutable_evidence)?;
    }
    if sha256_file(&immutable_evidence)? != evidence_hash {
        bail!("immutable evidence copy hash mismatch");
    }
    let gate_hash = sha256_file(gate_path)?;
    let immutable_gate = immutable_root
        .join("gates")
        .join(format!("{gate_hash}.gate.json"));
    if !immutable_gate.exists() {
        fs::copy(gate_path, &immutable_gate)?;
    }
    if sha256_file(&immutable_gate)? != gate_hash {
        bail!("immutable gate copy hash mismatch");
    }
    let lock_path = accepted_pointer.with_extension(format!(
        "{}.lock",
        accepted_pointer
            .extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    fs::create_dir_all(lock_path.parent().unwrap())?;
    let lock = fs::File::create(&lock_path)?;
    flock_exclusive(&lock, true)?;
    if accepted_pointer.is_file() {
        let current = read_json(accepted_pointer)?;
        let current_sha = get_str(&current, "checkpoint_sha256")?;
        if current_sha == checkpoint_sha {
            return Ok(result);
        }
        if enforce_pointer_ancestor && current_sha != get_str(&result, "accepted_ancestor_sha256")?
        {
            bail!("accepted ancestor changed before pointer promotion");
        }
    }
    atomic_json(
        accepted_pointer,
        &json!({
            "schema": "JopeSelfplayAcceptedCheckpoint.v2",
            "checkpoint": immutable_checkpoint.to_string_lossy(),
            "checkpoint_sha256": checkpoint_sha,
            "accepted_ancestor_sha256": result["accepted_ancestor_sha256"],
            "gate": immutable_gate.to_string_lossy(),
            "gate_sha256": gate_hash,
            "evidence": immutable_evidence.to_string_lossy(),
            "evidence_sha256": evidence_hash,
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-exact-reused",
            "promotion_scope": "internal-selected-test-flywheel-only",
        }),
    )?;
    Ok(result)
}

pub(super) fn handle_advance_accepted_pointer(request: &Value, root: &Path) -> Result<Value> {
    let evidence = PathBuf::from(get_str(
        request
            .pointer("/artifacts/build_acceptance_evidence")
            .ok_or_else(|| anyhow!("missing build_acceptance_evidence artifact"))?,
        "artifact_path",
    )?);
    let gate = PathBuf::from(get_str(
        request
            .pointer("/artifacts/run_checkpoint_gate")
            .ok_or_else(|| anyhow!("missing run_checkpoint_gate artifact"))?,
        "artifact_path",
    )?);
    let learning = root.join("training/learning.json");
    let immutable = root.join("training/immutable");
    let result = promote(&evidence, &gate, &learning, &immutable, true)?;
    if result.get("pass") != Some(&Value::Bool(true)) {
        bail!("controller attempted to promote a rejected checkpoint");
    }
    // Compatibility pointer for existing consumers; the learning pointer is
    // authoritative and is always advanced first.
    atomic_json(&root.join("training/accepted.json"), &read_json(&learning)?)?;
    let mut milestone = Value::Null;
    if result.get("milestone_pass") == Some(&Value::Bool(true)) {
        let milestone_pointer = root.join("training/milestone.json");
        let milestone_result = promote(&evidence, &gate, &milestone_pointer, &immutable, false)?;
        milestone = json!({
            "advanced": true,
            "checkpoint_sha256": milestone_result["checkpoint_sha256"],
            "pointer": milestone_pointer.to_string_lossy(),
            "pointer_sha256": sha256_file(&milestone_pointer)?,
        });
    }
    Ok(json!({
        "advanced": true,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "promotion_tier": result["promotion_tier"],
        "learning_pointer": learning.to_string_lossy(),
        "learning_pointer_sha256": sha256_file(&learning)?,
        "milestone": milestone,
    }))
}

pub(super) fn handle_run_sotarena_qualification(request: &Value, root: &Path) -> Result<Value> {
    let scan = request
        .pointer("/artifacts/freeze_winner_scan")
        .ok_or_else(|| anyhow!("missing freeze_winner_scan artifact"))?;
    let ledger_path = PathBuf::from(get_str(scan, "ledger")?);
    if sha256_file(&ledger_path)? != get_str(scan, "ledger_sha256")? {
        bail!("winner scan ledger changed before qualification enqueue");
    }
    let ledger = read_json(&ledger_path)?;
    let mut selected = Vec::new();
    for row in ledger
        .get("datasets")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("winner ledger has no datasets"))?
    {
        if row.get("qualification_state").and_then(Value::as_str) != Some("discovered") {
            continue;
        }
        let candidate = row.get("candidate").filter(|c| !c.is_null());
        selected.push(json!({
            "task": row["task"],
            "dataset_hash": row["dataset_hash"],
            "semantic_sha256": candidate
                .and_then(|c| c.get("semantic_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "receipt_sha256": candidate
                .and_then(|c| c.get("receipt_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "checkpoint_sha256": candidate
                .and_then(|c| c.get("model_checkpoint_sha256"))
                .cloned()
                .unwrap_or(Value::Null),
            "qualification_state": row["qualification_state"],
            "evidence_lane": "adaptive",
            "report_only": true,
        }));
    }
    let tier = request
        .pointer("/artifacts/run_checkpoint_gate/promotion_tier")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let sealed = if tier == "milestone" {
        milestone_status(
            request
                .pointer("/artifacts/advance_accepted_pointer/checkpoint_sha256")
                .and_then(Value::as_str)
                .unwrap_or(""),
        )?
    } else {
        Value::Null
    };
    let record = json!({
        "schema": "JopeSotarenaQualificationQueue.v1",
        "winner_ledger": ledger_path.to_string_lossy(),
        "winner_ledger_sha256": scan["ledger_sha256"],
        "candidate_checkpoint_sha256": request
            .pointer("/artifacts/advance_accepted_pointer/checkpoint_sha256")
            .cloned()
            .unwrap_or(Value::Null),
        "promotion_tier": tier,
        "adaptive_evidence": true,
        "zero_shot_claim_eligible": false,
        "report_only": true,
        "sealed_audit": sealed,
        "candidates": selected,
    });
    let path = root.join("qualifications").join(format!(
        "watermark-{:09}.json",
        get_i64(request, "watermark")?
    ));
    atomic_json(&path, &record)?;
    Ok(json!({
        "queue": path.to_string_lossy(),
        "queue_sha256": sha256_file(&path)?,
        "enqueued": selected.len(),
        "report_only": true,
    }))
}

/// Fail-closed production adapter for externally produced phase evidence
/// (used for every phase when running against a non-production root).
pub(super) fn file_handler(name: &str, request: &Value, root: &Path) -> Result<Value> {
    let path = watermark_dir(root, get_i64(request, "watermark")?).join(format!("{name}.json"));
    if !path.is_file() {
        return Err(pending(format!("waiting for {}", path.display())));
    }
    let mut value = read_json(&path)?;
    value["artifact_path"] = Value::String(path.to_string_lossy().into_owned());
    value["artifact_sha256"] = Value::String(sha256_file(&path)?);
    Ok(value)
}

pub(super) fn dispatch_phase(
    phase: &str,
    request: &Value,
    root: &Path,
    production: bool,
) -> Result<Value> {
    if !production {
        return file_handler(phase, request, root);
    }
    if PYTHON_PHASES.contains(&phase) {
        return python_handler(phase, request, root);
    }
    match phase {
        "launch_checkpoint_tournament" => handle_launch_checkpoint_tournament(request, root),
        "fetch_candidate_checkpoints" => handle_fetch_candidate_checkpoints(request, root),
        "generate_gate_proposals" => handle_generate_gate_proposals(request, root),
        "build_acceptance_evidence" => handle_build_acceptance_evidence(request, root),
        "run_checkpoint_gate" => handle_run_checkpoint_gate(request, root),
        "advance_accepted_pointer" => handle_advance_accepted_pointer(request, root),
        "run_sotarena_qualification" => handle_run_sotarena_qualification(request, root),
        other => file_handler(other, request, root),
    }
}
