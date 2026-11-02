use super::*;

pub(super) fn cmd_runs(
    node: &str,
    runs_root: &str,
    root: &Path,
    prefix: &str,
    min_age_hours: u64,
    keep_newest: usize,
    execute: bool,
) -> Result<()> {
    let plan = build_runs_plan(node, runs_root, root, prefix, min_age_hours, keep_newest)?;
    let plan_path = write_plan(root, &format!("gc-plan-{node}.v1.json"), &plan)?;
    let mut summary = json!({
        "mode": if execute { "execute" } else { "audit" },
        "node": node,
        "plan": plan_path.to_string_lossy(),
        "total_dirs_seen": plan.total_dirs,
        "candidates": plan.candidate_names.len(),
        "estimated_bytes": plan.estimated_bytes,
        "foreign_prefix_counts": plan.foreign_prefix_counts,
    });
    if execute {
        let (deleted, blocked) = execute_plan(node, runs_root, root, &plan)?;
        summary["deleted"] = json!(deleted);
        summary["blocked_unarchived_metrics"] = json!(blocked);
    }
    println!("{summary}");
    Ok(())
}

pub(super) fn cmd_receipts(
    root: &Path,
    min_age_hours: u64,
    keep_newest: usize,
    execute: bool,
) -> Result<()> {
    let receipts_root = root.join("receipts");
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(&receipts_root)?.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("jte-") || !sane_name(&name) {
            continue;
        }
        let mtime_ms = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        dirs.push(DirEntryInfo { name, mtime_ms });
    }
    let total_dirs = dirs.len();
    dirs.sort_by_key(|d| std::cmp::Reverse(d.mtime_ms));
    let protected: HashSet<String> = dirs
        .iter()
        .take(keep_newest)
        .map(|d| d.name.clone())
        .collect();
    let names: Vec<String> = dirs.iter().map(|d| d.name.clone()).collect();
    let ingested = ingested_set(root, &names);
    let now = now_ms();
    let min_age_secs = min_age_hours * 3600;
    let mut entries = Vec::new();
    let mut candidates = Vec::new();
    for dir in &dirs {
        let age_secs = now.saturating_sub(dir.mtime_ms) / 1000;
        // Receipts fall off only on the strong gate: evidence is in the corpus.
        if age_secs < min_age_secs || protected.contains(&dir.name) || !ingested.contains(&dir.name)
        {
            continue;
        }
        entries.push(json!({
            "dir": receipts_root.join(&dir.name).to_string_lossy(),
            "mtime_ms": dir.mtime_ms,
            "basis": ["prefix:jte-", format!("age_secs:{age_secs}"), "not_in_keep_newest", "evidence:ingested_jobs"],
        }));
        candidates.push(dir.name.clone());
    }
    let estimated_bytes =
        estimate_bytes(LOCAL_NODE, &receipts_root.to_string_lossy(), &candidates)?;
    let plan = Plan {
        schema: "JopeSelfplayGcPlan.v1",
        node: "xbabe2:receipts".to_string(),
        entries,
        candidate_names: candidates,
        estimated_bytes,
        foreign_prefix_counts: HashMap::new(),
        total_dirs,
    };
    let plan_path = write_plan(root, "gc-plan-receipts.v1.json", &plan)?;
    let mut summary = json!({
        "mode": if execute { "execute" } else { "audit" },
        "node": "xbabe2:receipts",
        "plan": plan_path.to_string_lossy(),
        "total_dirs_seen": plan.total_dirs,
        "candidates": plan.candidate_names.len(),
        "estimated_bytes": plan.estimated_bytes,
    });
    if execute {
        let mut deleted = 0usize;
        for batch in plan.candidate_names.chunks(DELETE_BATCH) {
            for name in batch {
                let path = receipts_root.join(name);
                if path.is_dir() {
                    std::fs::remove_dir_all(&path)?;
                }
            }
            deleted += batch.len();
            append_receipt(
                root,
                json!({
                    "ts_ms": now_ms(),
                    "node": "xbabe2:receipts",
                    "deleted_count": batch.len(),
                    "bytes_estimate": plan.estimated_bytes / plan.candidate_names.len().max(1) as u64 * batch.len() as u64,
                    "first": batch.first(),
                    "last": batch.last(),
                }),
            )?;
        }
        summary["deleted"] = json!(deleted);
    }
    println!("{summary}");
    Ok(())
}
