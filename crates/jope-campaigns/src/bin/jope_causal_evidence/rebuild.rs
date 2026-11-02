use super::*;

pub(crate) fn rebuild(
    root: &Path,
) -> Result<(CausalEvidenceLedgerV2, CausalEvidenceLedgerV3, Value)> {
    let (accepted, accepted_lineage_valid) = accepted_parents(root)?;
    let waves = wave_states(root)?;
    let mut requested_slots = Task::ALL
        .into_iter()
        .map(|task| (task, 0))
        .collect::<BTreeMap<_, _>>();
    for wave in waves.values() {
        if !wave.completed {
            continue;
        }
        for task in Task::ALL {
            if accepted.get(&task).map(Sha256Digest::as_str)
                == Some(wave.checkpoint_sha256.as_str())
            {
                *requested_slots.entry(task).or_default() +=
                    wave.requested_slots.get(&task).copied().unwrap_or_default();
            }
        }
    }
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let connection = Connection::open_with_flags(
        &corpus,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let receipts = receipt_rows(&connection, &accepted, accepted_lineage_valid, &waves)?;
    let generated_at_ms = now_ms();
    let mut ledger_v2 = build_causal_evidence_ledger_v2(
        &accepted,
        &requested_slots,
        receipts.clone(),
        generated_at_ms,
    )?;
    apply_permanent_repair_failures(root, &mut ledger_v2)?;
    let (epochs, policy_pins_match, logged_slots, permanent_missing_slots) =
        load_promotion_epochs(root, &accepted)?;
    let ledger_v3 = build_causal_evidence_ledger_v3(
        ledger_v2.clone(),
        &epochs,
        &receipts,
        &policy_pins_match,
        &logged_slots,
        &permanent_missing_slots,
        generated_at_ms,
    )?;
    let binary = &ledger_v2.tasks[&Task::Binary];
    let regression = &ledger_v2.tasks[&Task::Regression];
    let unreplayable = unreplayable_pair_disposition(root, generated_at_ms)?;
    let audit = json!({
        "schema": "CausalEvidenceAudit.v3",
        "generated_at_ms": generated_at_ms,
        "authoritative": true,
        "source": "causal_evaluation_receipts_v2 joined to immutable wave manifests",
        "all_history_ledger": root.join("training/causal-evidence-ledger.v2.json"),
        "promotion_ledger": root.join("training/causal-evidence-ledger.v3.json"),
        "non_authoritative_pre_v2_ledger": root.join("training/causal-evidence-ledger.v1.json"),
        "promotion_authority": "only CausalEvidenceLedger.v3 promotion_epoch can open production",
        "historical_direct_outcome_lower_bound_non_authoritative": {
            "binary": HISTORICAL_BINARY_LOWER_BOUND,
            "regression": HISTORICAL_REGRESSION_LOWER_BOUND,
            "note": "286/776 is preserved only as a historical direct-outcome lower bound and cannot open a gate.",
        },
        "receipt_level_complete_pairs": {
            "binary": binary.complete_pairs,
            "regression": regression.complete_pairs,
        },
        "requested_slot_validity": {
            "binary": binary.proposal_validity,
            "regression": regression.proposal_validity,
        },
        "pair_repair_queue_depth": ledger_v2.pair_repairs.len(),
        "unreplayable_pair_disposition": unreplayable,
        "production_gate_open": {
            "binary": ledger_v3.promotion_epoch.get(&Task::Binary).is_some_and(|row| row.production_gate_open),
            "regression": ledger_v3.promotion_epoch.get(&Task::Regression).is_some_and(|row| row.production_gate_open),
        },
        "gate_reasons": {
            "binary": ledger_v3.promotion_epoch.get(&Task::Binary).map(|row| &row.gate_reasons),
            "regression": ledger_v3.promotion_epoch.get(&Task::Regression).map(|row| &row.gate_reasons),
        },
        "legacy_cumulative_gate_non_authoritative": {
            "binary": binary.production_gate_open,
            "regression": regression.production_gate_open,
        }
    });
    Ok((ledger_v2, ledger_v3, audit))
}

pub(crate) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable unreplayable-pair object collision at {}",
            path.display()
        );
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable object has no parent")?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    let mut permissions = file.metadata()?.permissions();
    permissions.set_mode(0o444);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

pub(crate) fn retained_evaluator(root: &Path, digest: &str) -> Result<bool> {
    let path = root
        .join("training/evaluator-vault")
        .join(digest)
        .join("corpus_eval");
    Ok(path.is_file() && sha256_file(&path)? == digest)
}

pub(crate) fn failure_class(root: &Path, entry: &Value) -> Result<&'static str> {
    let Some(evaluator) = entry
        .get("original_evaluator_sha256")
        .and_then(Value::as_str)
    else {
        return Ok("expected_evaluator_unrecorded");
    };
    ensure!(
        evaluator.len() == 64
            && evaluator
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "repair state contains an invalid expected evaluator digest"
    );
    if !retained_evaluator(root, evaluator)? {
        return Ok("evaluator_binary_unavailable");
    }
    if entry
        .get("identical_failures")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        >= 3
    {
        return Ok("repeated_identical_failure");
    }
    Ok("permanent_repair_failure")
}

/// Publish permanent causal failures without evaluator substitution or outcome
/// imputation. Pair identities live in a content-addressed JSONL object so the
/// summary stays small and can be grouped without exposing a mutable ID list.
pub(crate) fn unreplayable_pair_disposition(root: &Path, generated_at_ms: u128) -> Result<Value> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    let mut identities = BTreeMap::<(String, String), Value>::new();
    let mut groups = BTreeMap::<(String, String), usize>::new();
    if path.is_file() {
        let state = read_json(&path)?;
        ensure!(
            state.get("schema").and_then(Value::as_str) == Some("CausalPairRepairExecutorState.v1"),
            "causal pair repair state schema drift"
        );
        if let Some(entries) = state.get("entries").and_then(Value::as_object) {
            for entry in entries.values() {
                if entry.get("status").and_then(Value::as_str) != Some("permanent_failure") {
                    continue;
                }
                let pair_id = entry
                    .get("pair_id")
                    .and_then(Value::as_str)
                    .context("permanent failure lacks pair identity")?;
                let lane = entry
                    .get("missing_lane")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let expected = entry
                    .get("original_evaluator_sha256")
                    .and_then(Value::as_str);
                let class = failure_class(root, entry)?;
                *groups
                    .entry((
                        expected.unwrap_or("unrecorded").to_owned(),
                        class.to_owned(),
                    ))
                    .or_default() += 1;
                identities.insert(
                    (pair_id.to_owned(), lane.to_owned()),
                    json!({
                        "pair_id": pair_id,
                        "task": entry.get("task").and_then(Value::as_str),
                        "missing_lane": lane,
                        "expected_evaluator_sha256": expected,
                        "failure_class": class,
                        "promotion_eligible": false,
                    }),
                );
            }
        }
    }
    let mut bytes = Vec::new();
    for identity in identities.values() {
        bytes.extend(serde_json::to_vec(identity)?);
        bytes.push(b'\n');
    }
    let identities_sha256 = sha256_bytes(&bytes);
    let identities_path = root
        .join("training/unreplayable-pair-identities")
        .join(format!("{identities_sha256}.v2.jsonl"));
    write_immutable(&identities_path, &bytes)?;
    let grouped_counts = groups
        .into_iter()
        .map(|((evaluator, class), count)| {
            json!({
                "expected_evaluator_sha256": (evaluator != "unrecorded").then_some(evaluator),
                "failure_class": class,
                "count": count,
            })
        })
        .collect::<Vec<_>>();
    let disposition = json!({
        "schema": "JopeUnreplayablePairDisposition.v2",
        "generated_at_ms": generated_at_ms,
        "policy": "explicit-unreplayable-no-imputation",
        "source_repair_state": path,
        "source_repair_state_sha256": path.is_file().then(|| sha256_file(&path)).transpose()?,
        "pair_identities_path": identities_path,
        "pair_identities_sha256": identities_sha256,
        "count": identities.len(),
        "unique_pair_count": identities.keys().map(|(pair, _)| pair).collect::<BTreeSet<_>>().len(),
        "grouped_counts": grouped_counts,
        "promotion_eligible": false,
        "evaluator_substitution_allowed": false,
        "outcome_imputation_allowed": false,
    });
    atomic_json(
        &root.join("training/unreplayable-pair-disposition.v2.json"),
        &disposition,
    )?;
    Ok(disposition)
}
