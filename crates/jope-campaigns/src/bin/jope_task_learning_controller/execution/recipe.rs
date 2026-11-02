use super::*;

pub(super) fn trainer_recipe(
    corpus: &CorpusPointer,
    task: Task,
    mode: TrainerMode,
    accepted_parent: &Sha256Digest,
    last_recipe_e_watermark: u64,
    last_recipe_e_descendant_receipts: u64,
) -> Result<(&'static str, Value)> {
    let broad = || {
        json!({
            "recipe_name": "a",
            "broad_replay_fraction": 1.0,
            "exact_causal_pair_fraction": 0.0,
            "hard_novel_oversampling": false,
            "fixed_anchor_holdout": true,
        })
    };
    if mode != TrainerMode::Shadow {
        return Ok(("a", broad()));
    }
    let packed = read_json(&corpus.packed_manifest)?;
    let Some(relative) = packed.get("causal_pair_index").and_then(Value::as_str) else {
        return Ok(("a", broad()));
    };
    let path = corpus
        .packed_manifest
        .parent()
        .context("packed manifest has no parent")?
        .join(relative);
    let expected = packed
        .get("files")
        .and_then(|files| files.get(relative))
        .and_then(|row| row.get("sha256"))
        .and_then(Value::as_str)
        .context("packed causal pair index lacks a content pin")?;
    ensure!(
        path.is_file() && sha256_file(&path)? == expected,
        "packed causal pair index content drift"
    );
    let mut complete_parent_pairs = 0_usize;
    for line in BufReader::new(File::open(&path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        if row.get("task").and_then(Value::as_str) == Some(task.as_str())
            && row
                .get("accepted_parent_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(accepted_parent.as_str())
        {
            complete_parent_pairs += 1;
        }
    }
    if complete_parent_pairs < 32 {
        return Ok(("a", broad()));
    }
    if let Some(relative) = packed
        .get("recipe_e_evidence_index")
        .and_then(Value::as_str)
    {
        let path = corpus
            .packed_manifest
            .parent()
            .context("packed manifest has no parent")?
            .join(relative);
        let expected = packed
            .get("files")
            .and_then(|files| files.get(relative))
            .and_then(|row| row.get("sha256"))
            .and_then(Value::as_str)
            .context("packed recipe E evidence index lacks a content pin")?;
        ensure!(
            path.is_file() && sha256_file(&path)? == expected,
            "packed recipe E evidence index content drift"
        );
        let mut completed_descendant_rows = 0_u64;
        let mut completed_descendant_receipts = BTreeSet::new();
        let mut hard_negative_rows = 0_u64;
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(&line)?;
            if row.get("task").and_then(Value::as_str) != Some(task.as_str()) {
                continue;
            }
            match row.get("lane").and_then(Value::as_str) {
                Some("completed_descendant_trajectories") => {
                    let receipt = row
                        .get("descendant_credit_receipt_sha256")
                        .and_then(Value::as_str)
                        .context("recipe E descendant row lacks receipt identity")?;
                    Sha256Digest::parse(receipt.to_owned())?;
                    ensure!(
                        matches!(
                            row.get("descendant_horizon").and_then(Value::as_u64),
                            Some(1 | 2 | 4 | 8)
                        ),
                        "recipe E descendant row has an invalid horizon"
                    );
                    completed_descendant_rows += 1;
                    completed_descendant_receipts.insert(receipt.to_owned());
                }
                Some("validity_instability_hard_negatives") => {
                    hard_negative_rows += 1;
                }
                _ => bail!("packed recipe E evidence lane is invalid"),
            }
        }
        let completed_descendant_receipts = u64::try_from(completed_descendant_receipts.len())?;
        let new_eligible_labels = corpus.watermark.saturating_sub(last_recipe_e_watermark);
        let new_completed_descendant_receipts =
            completed_descendant_receipts.saturating_sub(last_recipe_e_descendant_receipts);
        let trigger_open =
            new_eligible_labels >= 50_000 || new_completed_descendant_receipts >= 5_000;
        if trigger_open && completed_descendant_rows > 0 && hard_negative_rows > 0 {
            return Ok((
                "e",
                json!({
                    "recipe_name": "e",
                    "uniform_broad_replay_fraction": 0.50,
                    "exact_current_parent_pair_fraction": 0.25,
                    "completed_descendant_trajectory_fraction": 0.15,
                    "validity_instability_hard_negative_fraction": 0.10,
                    "complete_parent_pairs": complete_parent_pairs,
                    "completed_descendant_rows": completed_descendant_rows,
                    "completed_descendant_receipts": completed_descendant_receipts,
                    "hard_negative_rows": hard_negative_rows,
                    "new_eligible_labels": new_eligible_labels,
                    "new_completed_descendant_receipts":
                        new_completed_descendant_receipts,
                    "accepted_parent_checkpoint_sha256": accepted_parent,
                    "shadow_only": true,
                    "fixed_anchor_holdout": true,
                    "test_adaptive": true,
                    "benchmark_valid": false,
                }),
            ));
        }
    }
    Ok((
        "d",
        json!({
            "recipe_name": "d",
            "broad_replay_fraction": 0.75,
            "exact_causal_pair_fraction": 0.25,
            "hard_novel_oversampling": true,
            "fixed_anchor_holdout": true,
            "complete_parent_pairs": complete_parent_pairs,
            "accepted_parent_checkpoint_sha256": accepted_parent,
        }),
    ))
}
