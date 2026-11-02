use super::*;

pub(super) fn execute_plan(
    node: &str,
    runs_root: &str,
    root: &Path,
    plan: &Plan,
) -> Result<(usize, usize)> {
    // ARCHIVE-THEN-PRUNE. Nothing below this line may delete a directory whose
    // metrics are not already in training/history/.
    let guard = guard_metrics(node, runs_root, root, &plan.candidate_names);
    if !guard.blocked.is_empty() || guard.archived_rows > 0 || !guard.truncated.is_empty() {
        append_receipt(
            root,
            json!({
                "ts_ms": now_ms(),
                "node": node,
                "event": "metrics_guard",
                "candidates": plan.candidate_names.len(),
                "safe": guard.safe.len(),
                "blocked": guard.blocked.len(),
                "archived_rows": guard.archived_rows,
                "archived_bytes": guard.archived_bytes,
                "blocked_detail": guard.blocked,
                "truncated_tails": guard.truncated.len(),
                "truncated_detail": guard.truncated,
            }),
        )?;
    }
    let mut deleted = 0usize;
    for batch in guard.safe.chunks(DELETE_BATCH) {
        delete_batch(node, runs_root, batch)?;
        deleted += batch.len();
        append_receipt(
            root,
            json!({
                "ts_ms": now_ms(),
                "node": node,
                "runs_root": runs_root,
                "deleted_count": batch.len(),
                "bytes_estimate": plan.estimated_bytes / plan.candidate_names.len().max(1) as u64 * batch.len() as u64,
                "first": batch.first(),
                "last": batch.last(),
            }),
        )?;
    }
    Ok((deleted, guard.blocked.len()))
}
