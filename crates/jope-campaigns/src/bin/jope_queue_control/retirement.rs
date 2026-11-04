use super::*;

pub(crate) fn apply_retirement_registry(
    pair_registry: &Path,
    backlog: &Path,
    candidates: &[HoldCandidate],
    results: &mut [RetirementResult],
    preview_sha256: &str,
) -> Result<usize> {
    if !pair_registry.is_file() || !backlog.is_file() {
        bail!("retirement requires pair registry and backlog databases");
    }
    let by_job = candidates
        .iter()
        .map(|candidate| (candidate.job_id.as_str(), candidate))
        .collect::<HashMap<_, _>>();
    let mut connection = Connection::open(pair_registry)
        .with_context(|| format!("open pair registry {}", pair_registry.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute_batch("PRAGMA cache_size=-262144; PRAGMA temp_store=MEMORY;")?;
    connection.execute(
        "ATTACH DATABASE ?1 AS backlog_db",
        [backlog.to_string_lossy().as_ref()],
    )?;
    let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else if sqlite_column_exists(&connection, "pairs", "owner")? {
        "owner"
    } else {
        bail!("pair registry has no ownership column");
    };
    let has_attempts = sqlite_table_exists(&connection, "attempts")?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TEMP TABLE retiring_jobs(
           manifest_sha256 TEXT PRIMARY KEY,
           job_id TEXT NOT NULL UNIQUE,
           detail_json TEXT NOT NULL
         ) WITHOUT ROWID;",
    )?;
    let mut accepted = BTreeSet::new();
    for result in results.iter_mut() {
        if result.command_error.is_some() {
            continue;
        }
        let safely_unstarted = result
            .state_after_cancel
            .as_deref()
            .is_none_or(|state| state == "cancelled");
        if !safely_unstarted || (!result.cancel_requested && !result.recovery_from_prior_preview) {
            result.registry_error = Some("cancel_not_confirmed_unstarted".to_owned());
            continue;
        }
        let candidate = by_job
            .get(result.job_id.as_str())
            .context("retirement candidate disappeared")?;
        let claim_state: Option<String> = transaction
            .query_row(
                "SELECT state FROM backlog_db.claims
                 WHERE job_id=?1 AND manifest_sha256=?2 AND manifest=?3",
                params![
                    candidate.job_id,
                    candidate.manifest_sha256,
                    candidate.manifest
                ],
                |row| row.get(0),
            )
            .optional()?;
        if claim_state.as_deref() == Some("superseded_unstarted") {
            continue;
        }
        if claim_state.as_deref() != Some("submitted") {
            result.registry_error = Some("backlog_claim_no_longer_submitted".to_owned());
            continue;
        }
        let detail = serde_json::to_string(&json!({
            "job_id": candidate.job_id,
            "manifest_sha256": candidate.manifest_sha256,
            "preview_sha256": preview_sha256,
            "reason": "superseded_unstarted",
        }))?;
        transaction.execute(
            "INSERT INTO retiring_jobs VALUES(?1,?2,?3)",
            params![candidate.manifest_sha256, candidate.job_id, detail],
        )?;
        accepted.insert(candidate.job_id.clone());
    }

    let timestamp = now_ms() as i64;
    transaction.execute(
        "INSERT INTO events(kind,detail_json,created_at_ms)
         SELECT 'superseded_unstarted',r.detail_json,?1 FROM retiring_jobs r",
        [timestamp],
    )?;
    if has_attempts {
        transaction.execute(
            "UPDATE attempts SET state='superseded_unstarted',
               receipt_sha256=COALESCE(receipt_sha256,?1),
               job_id=COALESCE(
                 job_id,
                 (SELECT r.job_id FROM retiring_jobs r
                  WHERE r.manifest_sha256=attempts.manifest_sha256)
               ),finalized_at_ms=?2
             WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)
               AND state IN ('reserved','submitted','held')
               AND finalized_at_ms IS NULL",
            params![preview_sha256, timestamp],
        )?;
    }
    let bound_pairs: usize = transaction.query_row(
        "SELECT COUNT(*) FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)",
        [],
        |row| row.get(0),
    )?;
    let missing_manifest: Option<String> = transaction
        .query_row(
            "SELECT r.job_id FROM retiring_jobs r
             WHERE NOT EXISTS(
               SELECT 1 FROM pairs p INDEXED BY pair_manifest
               WHERE p.manifest_sha256=r.manifest_sha256
             ) LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(job_id) = missing_manifest {
        bail!("{job_id}: pair retirement has no pair-registry provenance");
    }
    // Sparse blocker repair is fastest through pair_manifest; mass retirement
    // must scan the WITHOUT ROWID primary table to avoid hundreds of thousands
    // of random primary-row fetches. Both paths retain immutable old bindings
    // until the next reservation, avoiding secondary-index churn.
    let update = if candidates.len() <= 128 {
        format!(
            "UPDATE pairs INDEXED BY pair_manifest
             SET state='available',{owner_column}='available',
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               failure_class=NULL,updated_at_ms=?1
             WHERE manifest_sha256 IN(
               SELECT manifest_sha256 FROM retiring_jobs
             )
               AND state IN ('reserved','submitted','held')
               AND receipt_sha256 IS NULL
               AND (
                 job_id IS NULL OR job_id=(
                   SELECT r.job_id FROM retiring_jobs r
                   WHERE r.manifest_sha256=pairs.manifest_sha256
                 )
               )"
        )
    } else {
        format!(
            "UPDATE pairs NOT INDEXED
             SET state='available',{owner_column}='available',
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               failure_class=NULL,updated_at_ms=?1
             WHERE state IN ('reserved','submitted','held')
               AND receipt_sha256 IS NULL
               AND EXISTS(
                 SELECT 1 FROM retiring_jobs r
                 WHERE r.manifest_sha256=pairs.manifest_sha256
                   AND (pairs.job_id IS NULL OR pairs.job_id=r.job_id)
               )"
        )
    };
    let updated_pairs = transaction.execute(&update, [timestamp])?;
    // The pre-update count is index-only. Equality proves every pair bound to a retiring
    // manifest passed the active-state, empty-receipt, and job provenance
    // predicates above. Any authoritative or mismatched row rolls back all
    // pair, attempt, event, and claim changes.
    if bound_pairs != updated_pairs {
        bail!(
            "pair retirement provenance mismatch: {updated_pairs} active rows \
             but {bound_pairs} total manifest-bound rows"
        );
    }
    let remaining_bound: usize = transaction.query_row(
        "SELECT COUNT(*) FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)",
        [],
        |row| row.get(0),
    )?;
    if remaining_bound != 0 {
        bail!("pair retirement did not clear active manifest bindings");
    }
    let updated_claims = transaction.execute(
        "UPDATE backlog_db.claims
         SET state='superseded_unstarted',updated_at_ms=?1
         WHERE state='submitted' AND EXISTS(
           SELECT 1 FROM retiring_jobs r
           WHERE r.job_id=backlog_db.claims.job_id
             AND r.manifest_sha256=backlog_db.claims.manifest_sha256
         )",
        [timestamp],
    )?;
    if updated_claims != accepted.len() {
        bail!("bulk backlog retirement count changed");
    }
    transaction.commit()?;
    for result in results.iter_mut() {
        if accepted.contains(&result.job_id) {
            result.claim_updated = true;
        }
    }
    Ok(updated_pairs)
}
