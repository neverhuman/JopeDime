use super::*;

// ---------------------------------------------------------------------------
// Per-task accepted pointers
// ---------------------------------------------------------------------------

/// Per-task accepted checkpoints. Deliberately a SEPARATE, additive file:
/// `training/accepted.json` and `training/learning.json` stay the ONE global
/// warm-start lineage pointer read by `jope_selfplay_freshness.rs:384`,
/// `jope_selfplay_supply.rs:542` (Codex-owned), `jope_selfplay_trainer_watchdog.rs:1055`,
/// `selfplay_runtime.rs:83` and `promotion.rs:accepted_checkpoint`, and they
/// alone define the freshness epoch. Nothing in this file participates in epoch
/// identity, so advancing a branch never retires a queued job.
///
/// The schema string is `JopeSelfplayAcceptedByTask.v1` — the value already on disk
/// and already written by `jope_task_learning_controller.rs` and `jope_selfplay_tui.rs`.
/// It was previously `AcceptedByTask.v2`, which is the *shadow* namespace's constant
/// (`selfplay_shadow/contracts.rs:7`) for a different artifact
/// (`accepted/by-task.v2.json`) with a different shape. Because
/// `advance_task_accepted_pointers` rewrites `document["schema"]` unconditionally — even
/// on a cycle where every branch is unchanged — the first promotion run would have
/// rewritten the live pointer to a string that four readers reject outright:
/// `jope_task_learning_controller.rs:1055`, `jope_causal_evidence.rs:302`,
/// `scripts/test_exact_train.py:843`, `train_only_study.rs:1437`. Two of those run as
/// live services against this root, so the break would have been immediate and fleet-wide.
pub(super) const TASK_ACCEPTED_SCHEMA: &str = "JopeSelfplayAcceptedByTask.v1";

pub(super) fn task_accepted_pointer(root: &Path) -> PathBuf {
    root.join("training/accepted-by-task.json")
}

/// Content-addressed copy into the immutable store, hash-verified both sides.
/// Same contract as the copy inside `promote`, factored so a per-task advance
/// cannot leave a pointer aimed at a scheduler run directory the GC will reap.
pub(super) fn publish_immutable_checkpoint(
    immutable_root: &Path,
    checkpoint: &Path,
    checkpoint_sha: &str,
) -> Result<PathBuf> {
    let destination = immutable_root
        .join("checkpoints")
        .join(format!("{checkpoint_sha}.model.pt"));
    fs::create_dir_all(destination.parent().unwrap())?;
    if destination.exists() {
        if sha256_file(&destination)? != checkpoint_sha {
            bail!("immutable checkpoint path contains different bytes");
        }
        return Ok(destination);
    }
    if sha256_file(checkpoint)? != checkpoint_sha {
        bail!("checkpoint content hash mismatch before immutable publish");
    }
    let temporary = destination.with_file_name(format!(
        ".{}.tmp-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    fs::copy(checkpoint, &temporary)?;
    if sha256_file(&temporary)? != checkpoint_sha {
        let _ = fs::remove_file(&temporary);
        bail!("checkpoint changed while copying to immutable storage");
    }
    fs::rename(&temporary, &destination)?;
    Ok(destination)
}

/// Advance `training/accepted-by-task.json` from one gate result.
///
/// A branch advances only when its OWN verdict passed. A branch that did not
/// pass keeps whatever it already had; a branch that has never been recorded is
/// bootstrapped to the global accepted checkpoint so every reader always
/// resolves to something content-verified. The compare-and-swap is per branch:
/// the entry must currently name the gate's accepted ancestor (or be a
/// bootstrap), which is the same ancestor guard `promote` applies globally.
pub(super) fn advance_task_accepted_pointers(
    root: &Path,
    evidence: &Value,
    gate: &Value,
    global_path: &Path,
    global_sha: &str,
) -> Result<Value> {
    let verdicts = gate
        .get("task_verdicts")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("gate result carries no task_verdicts"))?;
    let ancestor = get_str(gate, "accepted_ancestor_sha256")?.to_owned();
    let candidate_sha = get_str(gate, "checkpoint_sha256")?.to_owned();
    let candidate_path = evidence
        .pointer("/candidate/checkpoint_path")
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let watermark = evidence
        .get("corpus_watermark")
        .cloned()
        .unwrap_or(Value::Null);
    let evidence_sha256 = match evidence.get("artifact_sha256").and_then(Value::as_str) {
        Some(sha256) => sha256.to_owned(),
        None => sha256_bytes(&serde_json::to_vec(evidence)?),
    };
    let immutable = root.join("training/immutable");
    let pointer_path = task_accepted_pointer(root);
    fs::create_dir_all(pointer_path.parent().unwrap())?;
    let lock_path = pointer_path.with_extension("json.lock");
    let lock = fs::File::create(&lock_path)?;
    flock_exclusive(&lock, true)?;

    let mut document = if pointer_path.is_file() {
        read_json(&pointer_path)?
    } else {
        json!({"schema": TASK_ACCEPTED_SCHEMA, "tasks": {}})
    };
    if document.get("tasks").and_then(Value::as_object).is_none() {
        document["tasks"] = json!({});
    }
    let mut advanced: Vec<String> = Vec::new();
    let mut unchanged: Vec<String> = Vec::new();
    let mut refused: Map<String, Value> = Map::new();
    for task in TASK_BRANCHES {
        let existing = document
            .pointer(&format!("/tasks/{task}"))
            .cloned()
            .filter(|value| value.is_object());
        let passed = verdicts
            .get(task)
            .and_then(|verdict| verdict.get("pass"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !passed {
            if existing.is_none() {
                // Advisory-until-anchor: a NEW task is never bootstrapped into
                // this file — presence of its entry IS the anchor that turns
                // its verdict gating (`task_verdict_is_gating`, and the same
                // presence rule in the task learning controller's
                // `accepted_pointers`). Only a genuinely passing verdict below
                // may create it.
                if !matches!(task, "regression" | "binary") {
                    continue;
                }
                document["tasks"][task] = json!({
                    "task": task,
                    "checkpoint": global_path.to_string_lossy(),
                    "checkpoint_sha256": global_sha,
                    "generation": 0,
                    "parent_checkpoint_sha256": Value::Null,
                    "evidence_sha256": Value::Null,
                    "proposal_sha256": Value::Null,
                    "corpus_watermark": watermark,
                    "rollback": Value::Null,
                    "source": "global_accepted_bootstrap",
                    "promotion_tier": Value::Null,
                    "updated_at_ms": now_ms(),
                });
            }
            unchanged.push(task.to_owned());
            continue;
        }
        let current = existing
            .as_ref()
            .and_then(|value| value.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if current == candidate_sha {
            unchanged.push(task.to_owned());
            continue;
        }
        if !current.is_empty() && current != ancestor {
            refused.insert(
                task.to_owned(),
                json!({"reason": "task ancestor changed before pointer advance",
                       "current": current, "expected": ancestor}),
            );
            continue;
        }
        let Some(source) = candidate_path.as_ref() else {
            refused.insert(
                task.to_owned(),
                json!({"reason": "evidence has no candidate checkpoint path"}),
            );
            continue;
        };
        let published = publish_immutable_checkpoint(&immutable, source, &candidate_sha)?;
        let generation = existing
            .as_ref()
            .and_then(|value| value.get("generation"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        let proposal_sha256 = evidence
            .get("proposal_sets")
            .and_then(Value::as_array)
            .and_then(|sets| {
                sets.iter()
                    .find(|set| set.get("task").and_then(Value::as_str) == Some(task))
            })
            .and_then(|set| set.get("proposals_sha256"))
            .cloned()
            .unwrap_or(Value::Null);
        let rollback = existing.clone().unwrap_or_else(|| {
            json!({
                "checkpoint": global_path.to_string_lossy(),
                "checkpoint_sha256": global_sha,
                "generation": 0,
            })
        });
        document["tasks"][task] = json!({
            "task": task,
            "checkpoint": published.to_string_lossy(),
            "checkpoint_sha256": candidate_sha,
            "generation": generation,
            "parent_checkpoint_sha256": ancestor,
            "evidence_sha256": evidence_sha256,
            "proposal_sha256": proposal_sha256,
            "corpus_watermark": watermark,
            "rollback": rollback,
            "promotion_tier": verdicts[task]["promotion_tier"],
            "pairwise_lift": verdicts[task]["pairwise_lift"],
            "source": "per_task_gate",
            "updated_at_ms": now_ms(),
        });
        advanced.push(task.to_owned());
    }
    document["schema"] = json!(TASK_ACCEPTED_SCHEMA);
    document["global_pointer"] = json!(root.join("training/accepted.json").to_string_lossy());
    document["global_checkpoint_sha256"] = json!(global_sha);
    document["updated_at_ms"] = json!(now_ms());
    atomic_json(&pointer_path, &document)?;
    Ok(json!({
        "pointer": pointer_path.to_string_lossy(),
        "pointer_sha256": sha256_file(&pointer_path)?,
        "advanced": advanced,
        "unchanged": unchanged,
        "refused": Value::Object(refused),
    }))
}

pub(super) fn handle_run_checkpoint_gate(request: &Value, root: &Path) -> Result<Value> {
    let evidence = request
        .pointer("/artifacts/build_acceptance_evidence")
        .ok_or_else(|| anyhow!("missing build_acceptance_evidence artifact"))?;
    // Advisory-until-anchor: new-task verdicts gate only once
    // accepted-by-task.json carries their anchor.
    let accepted_by_task = load_accepted_by_task(&task_accepted_pointer(root))?;
    let result = evaluate_gate(evidence, accepted_by_task.as_ref())?;
    let destination =
        watermark_dir(root, get_i64(request, "watermark")?).join("checkpoint-gate.json");
    atomic_json(&destination, &result)?;
    let mut out = result;
    // Per-task pointers are a RECORD, not a precondition: this phase must keep
    // producing its artifact even if the branch pointer cannot be written, or a
    // filesystem problem in an additive lane would stall the whole tournament.
    out["task_accepted_pointers"] = match accepted_checkpoint()
        .and_then(|(path, sha)| advance_task_accepted_pointers(root, evidence, &out, &path, &sha))
    {
        Ok(value) => value,
        Err(error) => json!({"error": error.to_string()}),
    };
    out["artifact_path"] = Value::String(destination.to_string_lossy().into_owned());
    out["artifact_sha256"] = Value::String(sha256_file(&destination)?);
    Ok(out)
}
