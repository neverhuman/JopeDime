use super::*;

pub(super) fn run_revert(
    args: &Args,
    batch_id: &str,
    receipt_dir: &Path,
) -> Result<serde_json::Value> {
    let journal_path = receipt_dir.join(format!("{batch_id}.journal.jsonl"));
    let file = File::open(&journal_path)
        .with_context(|| format!("open journal {}", journal_path.display()))?;

    let mut attempt_ids: Vec<i64> = Vec::new();
    let mut keeper_rows: Vec<(i64, String, Option<String>)> = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)?;
        match value.get("t").and_then(|v| v.as_str()) {
            Some("attempt_insert") => {
                attempt_ids.push(
                    value
                        .get("id")
                        .and_then(|v| v.as_i64())
                        .context("journal attempt_insert lacks id")?,
                );
            }
            Some("keeper_retire") => {
                keeper_rows.push((
                    value
                        .get("id")
                        .and_then(|v| v.as_i64())
                        .context("journal keeper_retire lacks id")?,
                    value
                        .get("prior_state")
                        .and_then(|v| v.as_str())
                        .context("journal keeper_retire lacks prior_state")?
                        .to_owned(),
                    value
                        .get("prior_failure_class")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned),
                ));
            }
            other => bail!("unknown journal record type {other:?}"),
        }
    }

    let mut deleted = 0usize;
    let mut restored = 0usize;
    if args.apply {
        if !attempt_ids.is_empty() {
            let mut registry = open_rw(
                &args.root.join("ops/pair-registry.sqlite"),
                args.busy_timeout_ms,
            )?;
            for chunk in attempt_ids.chunks(args.batch_rows.max(1)) {
                let tx = registry.transaction_with_behavior(TransactionBehavior::Immediate)?;
                {
                    // Only remove rows this batch created and that are still
                    // untouched, so a reversal can never destroy live progress.
                    let mut delete = tx.prepare_cached(
                        "DELETE FROM attempts WHERE id=? AND state IN ('reserved','infrastructure_retry') \
                         AND job_id IS NULL AND receipt_sha256 IS NULL \
                         AND detail_json LIKE '%wsA-three-ledger%'",
                    )?;
                    for id in chunk {
                        deleted += delete.execute(params![id])?;
                    }
                }
                tx.commit()?;
            }
        }
        if !keeper_rows.is_empty() {
            let mut backlog = open_rw(&args.root.join("ops/backlog.sqlite"), args.busy_timeout_ms)?;
            for chunk in keeper_rows.chunks(args.batch_rows.max(1)) {
                let tx = backlog.transaction_with_behavior(TransactionBehavior::Immediate)?;
                {
                    let mut update = tx.prepare_cached(
                        "UPDATE keeper_attempts SET state=?,failure_class=?,finalized_at_ms=NULL \
                         WHERE id=? AND state=?",
                    )?;
                    for (id, prior_state, prior_failure) in chunk {
                        restored += update.execute(params![
                            prior_state,
                            prior_failure,
                            id,
                            KEEPER_RETIRED_STATE
                        ])?;
                    }
                }
                tx.commit()?;
            }
        }
    }

    Ok(json!({
        "schema": "JopeLedgerReconcilerRevert.v1",
        "batch_id": batch_id,
        "apply": args.apply,
        "journal": journal_path.to_string_lossy(),
        "attempt_rows_in_journal": attempt_ids.len(),
        "keeper_rows_in_journal": keeper_rows.len(),
        "attempt_rows_deleted": deleted,
        "keeper_rows_restored": restored,
    }))
}
