use super::*;

/// Manifests that still hold active `keeper_attempts` rows. Driven off the
/// covering index `keeper_attempt_manifest_state(manifest_sha256,state)` so
/// this stays an index scan rather than a 4.7 GB table walk.
fn active_keeper_manifests(backlog: &Connection, limit: usize) -> Result<BTreeMap<String, usize>> {
    let mut statement = backlog.prepare(
        "SELECT manifest_sha256, COUNT(*) FROM keeper_attempts \
         WHERE state IN ('checked','submitting','submitted','deferred_horizon') \
         GROUP BY manifest_sha256",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
    })?;
    let mut out = BTreeMap::new();
    let mut total = 0usize;
    for row in rows {
        let (manifest, count) = row?;
        if total >= limit {
            break;
        }
        total += count;
        out.insert(manifest, count);
    }
    Ok(out)
}

pub(super) fn run_reap(
    backlog: &mut Connection,
    registry: &Connection,
    args: &Args,
    _batch_id: &str,
    journal: &mut Journal,
) -> Result<ReapStats> {
    let mut stats = ReapStats::default();
    let candidates = active_keeper_manifests(backlog, args.max_reap)?;
    stats.manifests_examined = candidates.len();

    // Two independent release signals. The registry one alone was NOT enough:
    // measured in production, it retired 0 rows across 297 receipts while the
    // active set regrew 0 -> 78,261 in three hours, because a normally-COMPLETED
    // pair keeps its manifest binding (pairs state='completed': 2,591,302 bound,
    // 0 unbound). "Zero bound pairs" only ever describes the retirement path,
    // never the far more common success path.
    //
    // Signal 1 (backlog, authoritative for "the work is done"): the manifest's
    // claim reached a terminal state. 'ingested' means the evidence is already
    // in the corpus, so the attempt has no business holding a slot in the
    // keeper_one_active_attempt uniqueness index. Same database as the rows we
    // retire, so there is no cross-database race.
    //
    // Signal 2 (registry, the original): zero bound pairs, i.e. freshness or
    // jope_queue_control nulled the binding when releasing the pair.
    const TERMINAL_CLAIM_STATES: [&str; 3] =
        ["ingested", "superseded_unstarted", "rejected_task_invalid"];

    // Scoped so the cached statements release their immutable borrow of
    // `backlog` before the retirement transaction below takes it mutably.
    let mut released: Vec<String> = Vec::new();
    {
        let mut claim_statement =
            backlog.prepare_cached("SELECT state FROM claims WHERE manifest_sha256=?")?;
        let mut bound_statement =
            registry.prepare_cached("SELECT COUNT(*) FROM pairs WHERE manifest_sha256=?")?;
        for manifest in candidates.keys() {
            let claim_state: Option<String> = claim_statement
                .query_row([manifest], |row| row.get(0))
                .optional()?;
            let claim_terminal = claim_state
                .as_deref()
                .is_some_and(|state| TERMINAL_CLAIM_STATES.contains(&state));
            if claim_terminal {
                released.push(manifest.clone());
                continue;
            }
            let bound: i64 = bound_statement.query_row([manifest], |row| row.get(0))?;
            if bound == 0 {
                released.push(manifest.clone());
            }
        }
    }
    stats.manifests_released = released.len();
    if released.is_empty() || !args.apply {
        stats.retired = released
            .iter()
            .map(|manifest| candidates.get(manifest).copied().unwrap_or(0))
            .sum();
        return Ok(stats);
    }

    for chunk in released.chunks(args.batch_rows.max(1)) {
        let mut journalled: Vec<serde_json::Value> = Vec::new();
        let tx = backlog.transaction_with_behavior(TransactionBehavior::Immediate)?;
        {
            let mut select = tx.prepare_cached(
                "SELECT id,state,failure_class FROM keeper_attempts \
                 WHERE manifest_sha256=? \
                 AND state IN ('checked','submitting','submitted','deferred_horizon')",
            )?;
            let mut update = tx.prepare_cached(
                "UPDATE keeper_attempts SET state=?,finalized_at_ms=? WHERE id=?",
            )?;
            let timestamp = now_ms() as i64;
            for manifest in chunk {
                let rows = select
                    .query_map([manifest], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                for (id, prior_state, prior_failure) in rows {
                    update.execute(params![KEEPER_RETIRED_STATE, timestamp, id])?;
                    journalled.push(json!({
                        "t": "keeper_retire",
                        "id": id,
                        "prior_state": prior_state,
                        "prior_failure_class": prior_failure,
                    }));
                }
            }
        }
        for entry in &journalled {
            journal.record(entry)?;
        }
        journal.sync()?;
        tx.commit()?;
        stats.retired += journalled.len();
    }
    Ok(stats)
}
