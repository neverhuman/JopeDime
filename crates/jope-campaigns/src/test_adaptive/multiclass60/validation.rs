use super::*;

pub(super) fn validate_ledger(
    root: &Path,
    value: &Value,
    dataset: &DatasetBinding,
) -> Result<u128> {
    if value.get("schema").and_then(Value::as_str) != Some(LEDGER_SCHEMA)
        || value.get("dataset_hash").and_then(Value::as_str) != Some(&dataset.dataset_hash)
        || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || value.get("selection_bias").and_then(Value::as_str) != Some("test-selected")
        || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || value
            .get("official_combined_rank_eligible")
            .and_then(Value::as_bool)
            != Some(false)
        || value.get("comparator_sha256").and_then(Value::as_str)
            != Some(dataset.comparator_sha256.as_str())
    {
        bail!(
            "{} champion ledger identity or labels drift",
            dataset.dataset_hash
        );
    }
    let status = value.get("status").and_then(Value::as_str);
    if !matches!(
        status,
        Some("strict_winner" | "ceiling_matched" | "exhausted")
    ) {
        bail!("{} is not terminal", dataset.dataset_hash);
    }
    if value.get("final_incumbent_count").and_then(Value::as_u64)
        != Some(dataset.completed_incumbent_count as u64)
    {
        bail!(
            "{} winner was not measured against its frozen comparator report set",
            dataset.dataset_hash
        );
    }
    let generation = value
        .get("generation")
        .and_then(Value::as_u64)
        .context("champion ledger lacks generation")? as usize;
    let completed_generations = value
        .get("completed_generations")
        .and_then(Value::as_u64)
        .context("champion ledger lacks completed_generations")?
        as usize;
    if completed_generations == 0
        || completed_generations > MAX_GENERATIONS
        || completed_generations != generation + 1
        || (status == Some("exhausted") && completed_generations != MAX_GENERATIONS)
    {
        bail!("{} generation budget is invalid", dataset.dataset_hash);
    }
    for initial in 0..completed_generations {
        let checkpoint = load_checkpoint(root, dataset, initial)?;
        if initial == MAX_GENERATIONS - 1 && !checkpoint.initial_budget_complete {
            bail!(
                "{} final checkpoint lacks budget completion",
                dataset.dataset_hash
            );
        }
    }
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, generation);
    if value.get("checkpoint_sha256").and_then(Value::as_str)
        != Some(sha256_file(&checkpoint_file)?.as_str())
    {
        bail!("{} ledger checkpoint hash mismatch", dataset.dataset_hash);
    }
    let replays = value
        .get("replays")
        .and_then(Value::as_array)
        .context("champion ledger lacks replay evidence")?;
    if status == Some("exhausted") {
        if !replays.is_empty() {
            bail!(
                "{} exhausted ledger unexpectedly has replays",
                dataset.dataset_hash
            );
        }
        let incumbent = dataset
            .strongest_available_incumbent
            .as_ref()
            .context("dataset lacks frozen incumbent")?;
        if value
            .get("macro_f1")
            .and_then(Value::as_f64)
            .is_some_and(|score| {
                score > incumbent.macro_f1 || (score == 1.0 && incumbent.macro_f1 == 1.0)
            })
        {
            bail!(
                "{} best-effort ledger contains a qualifying winner",
                dataset.dataset_hash
            );
        }
        return value
            .get("terminal_at_unix_ms")
            .and_then(Value::as_u64)
            .map(u128::from)
            .context("best-effort ledger lacks terminal timestamp");
    }
    let nodes = replays
        .iter()
        .filter_map(|row| row.get("node").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let hashes = replays
        .iter()
        .filter_map(|row| row.get("prediction_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let probability_hashes = replays
        .iter()
        .filter_map(|row| row.get("probability_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let primary = replays
        .iter()
        .filter_map(|row| {
            row.get("macro_f1")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    let secondary = replays
        .iter()
        .filter_map(|row| {
            row.get("accuracy")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    if replays.len() != 3
        || nodes.len() != 3
        || hashes.len() != 1
        || probability_hashes.len() != 1
        || primary.len() != 1
        || secondary.len() != 1
    {
        bail!(
            "{} lacks three-node bit-identical replay",
            dataset.dataset_hash
        );
    }
    for replay in replays {
        let path = replay
            .get("receipt_path")
            .and_then(Value::as_str)
            .context("replay lacks receipt path")?;
        if replay.get("receipt_sha256").and_then(Value::as_str)
            != Some(sha256_file(Path::new(path))?.as_str())
        {
            bail!("{} replay receipt hash mismatch", dataset.dataset_hash);
        }
    }
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset lacks frozen incumbent")?;
    let champion = value
        .get("macro_f1")
        .and_then(Value::as_f64)
        .filter(|score| score.is_finite())
        .context("champion ledger lacks finite macro-F1")?;
    let expected_status = if incumbent.macro_f1 == 1.0 && champion == 1.0 {
        "ceiling_matched"
    } else if champion > incumbent.macro_f1 {
        "strict_winner"
    } else {
        bail!(
            "{} ledger does not beat its comparator",
            dataset.dataset_hash
        );
    };
    if value.get("status").and_then(Value::as_str) != Some(expected_status) {
        bail!("{} champion status is inconsistent", dataset.dataset_hash);
    }
    value
        .get("terminal_at_unix_ms")
        .and_then(Value::as_u64)
        .map(u128::from)
        .context("champion ledger lacks terminal timestamp")
}

pub(super) fn validate_all(run_id: &str, campaign_root: &Path) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let ready = ready_dataset_bindings(&plan, &root)?;
    if ready.len() != plan.datasets.len() {
        bail!(
            "multiclass validation requires 272 frozen comparators, found {}",
            ready.len()
        );
    }
    let mut terminal_times = Vec::new();
    for dataset in &ready {
        let value = read_value(&ledger_path(&root, &dataset.dataset_hash))?;
        terminal_times.push(validate_ledger(&root, &value, dataset)?);
    }
    let latest_terminal = terminal_times.iter().max().copied();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "terminal_datasets": 272,
            "latest_terminal_at_unix_ms": latest_terminal,
            "all_datasets_included": true,
            "frozen_comparators": 272,
            "three_node_replay": true,
            "success": true,
        }))?
    );
    Ok(())
}
