use super::*;

pub(super) type EpochAccounting = (
    BTreeMap<Task, CausalEvidenceEpochV1>,
    BTreeMap<Task, bool>,
    BTreeMap<Task, usize>,
    BTreeMap<Task, usize>,
);

pub(super) fn load_promotion_epochs(
    root: &Path,
    accepted: &BTreeMap<Task, Sha256Digest>,
) -> Result<EpochAccounting> {
    let mut epochs = BTreeMap::new();
    let mut policy_pins_match = BTreeMap::new();
    let mut logged_slots = BTreeMap::new();
    let mut permanent_missing_slots = BTreeMap::new();
    for task in Task::ALL {
        let path = root
            .join("training/causal-evidence-epochs")
            .join(format!("{}.v1.json", task.as_str()));
        if !path.is_file() {
            continue;
        }
        let epoch: CausalEvidenceEpochV1 = serde_json::from_value(read_json(&path)?)?;
        epoch.validate()?;
        ensure!(
            epoch.task == task
                && accepted.get(&task) == Some(&epoch.accepted_parent_checkpoint_sha256),
            "{task} promotion epoch is not pinned to the current accepted parent"
        );
        let accounting_path = root
            .join("training/causal-evidence-epoch-accounting")
            .join(format!("{}.v1.json", task.as_str()));
        let mut pins = false;
        let mut slots = 0;
        let mut missing = 0;
        if accounting_path.is_file() {
            let accounting = read_json(&accounting_path)?;
            ensure!(
                accounting.get("schema").and_then(Value::as_str)
                    == Some("CausalEvidenceEpochAccounting.v1")
                    && accounting.get("epoch_sha256").and_then(Value::as_str)
                        == Some(epoch.identity()?.as_str()),
                "{task} epoch accounting identity drift"
            );
            pins = accounting
                .get("policy_pins_match")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            slots = accounting
                .get("logged_slots")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            missing = accounting
                .get("permanent_missing_slots")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
        }
        policy_pins_match.insert(task, pins);
        logged_slots.insert(task, slots);
        permanent_missing_slots.insert(task, missing);
        epochs.insert(task, epoch);
    }
    Ok((
        epochs,
        policy_pins_match,
        logged_slots,
        permanent_missing_slots,
    ))
}

pub(super) fn apply_permanent_repair_failures(
    root: &Path,
    ledger: &mut CausalEvidenceLedgerV2,
) -> Result<()> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    if !path.is_file() {
        return Ok(());
    }
    let state = read_json(&path)?;
    ensure!(
        state.get("schema").and_then(Value::as_str) == Some("CausalPairRepairExecutorState.v1"),
        "causal pair repair state schema drift"
    );
    let permanent = state
        .get("entries")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(_, entry)| {
            if entry.get("status").and_then(Value::as_str) != Some("permanent_failure") {
                return None;
            }
            Some((
                entry.get("pair_id")?.as_str()?.to_owned(),
                entry.get("missing_lane")?.as_str()?.to_owned(),
            ))
        })
        .collect::<std::collections::BTreeSet<_>>();
    for repair in &mut ledger.pair_repairs {
        if permanent.contains(&(
            repair.pair_id.as_str().to_owned(),
            match repair.missing_lane {
                CausalLane::NativeDlDirect => "native_dl_direct",
                CausalLane::MatchedGpComparator => "matched_gp_comparator",
            }
            .to_owned(),
        )) {
            repair.permanent_failure = true;
        }
    }
    Ok(())
}
