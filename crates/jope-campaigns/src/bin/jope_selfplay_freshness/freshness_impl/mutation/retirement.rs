use super::*;

impl Supervisor {
    pub(crate) fn apply_stale_retirement(
        &mut self,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Value> {
        if let Some(prior) =
            self.prior_mutation("apply_stale_retirement", preview_sha256, idempotency_key)?
        {
            return Ok(prior);
        }
        let preview = self.load_preview(preview_sha256, Some("stale_retirement"))?;
        let actions = preview
            .payload
            .get("actions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut inventory = self
            .inventory_with_manifest_policy(true)?
            .into_iter()
            .map(|job| (job.job_id.clone(), job))
            .collect::<BTreeMap<_, _>>();
        let mut held = 0_u64;
        let mut retired = 0_u64;
        let mut skipped = Vec::new();

        for action in &actions {
            let Some(job_id) = action.get("job_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(current) = inventory.get(job_id) else {
                skipped.push(json!({"job_id": job_id, "reason": "not_active"}));
                continue;
            };
            if !retirement_scoped_job(current)
                || current.epoch == self.state.active_epoch
                || NON_INTERRUPTIBLE_STATES.contains(&current.state.as_str())
            {
                skipped.push(json!({"job_id": job_id, "reason": "started_or_current"}));
                continue;
            }
            if USEFUL_QUEUE_STATES.contains(&current.state.as_str()) {
                self.scheduler.hold(job_id)?;
                self.event(
                    "stale_job_held",
                    Some(&current.epoch),
                    Some(job_id),
                    json!({"automatic": false}),
                )?;
                held += 1;
            }
        }

        inventory = self
            .inventory_with_manifest_policy(true)?
            .into_iter()
            .map(|job| (job.job_id.clone(), job))
            .collect();
        for action in &actions {
            let Some(job_id) = action.get("job_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(current) = inventory.get(job_id) else {
                continue;
            };
            if !retirement_scoped_job(current) {
                skipped.push(json!({"job_id": job_id, "reason": "protected_project"}));
                continue;
            }
            if !matches!(current.state.as_str(), "held" | "cancelled" | "checked") {
                continue;
            }
            let expected_claim = if current.state == "checked" {
                "checked"
            } else {
                "submitted"
            };
            if current.epoch == self.state.active_epoch
                || current.manifest_sha256.is_empty()
                || current.claim_state != expected_claim
            {
                skipped.push(json!({"job_id": job_id, "reason": "provenance_audit_failed"}));
                continue;
            }
            let states = self.manifest_state_counts(&current.manifest_sha256)?;
            if states.keys().all(|state| state == "available") && !states.is_empty() {
                self.mark_claim_superseded(job_id, &current.manifest_sha256)?;
                continue;
            }
            let expected_pair_state = if current.state == "checked" {
                "reserved"
            } else {
                "submitted"
            };
            if states.len() != 1 || !states.contains_key(expected_pair_state) {
                skipped.push(json!({"job_id": job_id, "reason": "pair_state_audit_failed"}));
                continue;
            }
            let retirement = content_hash(&json!({
                "preview": preview_sha256,
                "job_id": job_id,
                "manifest_sha256": current.manifest_sha256
            }))?;

            // This second scheduler read is the hard guard against racing a
            // dispatch.  Only held/cancelled/checked jobs can cross it.
            let latest = self
                .inventory()?
                .into_iter()
                .find(|job| job.job_id == job_id);
            let Some(latest) = latest else {
                skipped.push(json!({"job_id": job_id, "reason": "state_changed_before_cancel"}));
                continue;
            };
            if !retirement_scoped_job(&latest)
                || !matches!(latest.state.as_str(), "held" | "cancelled" | "checked")
                || latest.epoch == self.state.active_epoch
            {
                skipped.push(json!({"job_id": job_id, "reason": "state_changed_before_cancel"}));
                continue;
            }
            if latest.state == "held" {
                self.scheduler.cancel(job_id)?;
            }
            let count = self.supersede_pairs(
                &current.manifest_sha256,
                job_id,
                expected_pair_state,
                &retirement,
            )?;
            self.mark_claim_superseded(job_id, &current.manifest_sha256)?;
            self.event(
                "stale_job_retired",
                Some(&current.epoch),
                Some(job_id),
                json!({"retirement_receipt_sha256": retirement, "pairs": count}),
            )?;
            retired += 1;
        }
        let result = json!({
            "preview_receipt_sha256": preview_sha256,
            "held": held,
            "retired": retired,
            "skipped": skipped,
            "running_interrupted": 0
        });
        self.finish_mutation(
            "apply_stale_retirement",
            preview_sha256,
            idempotency_key,
            result,
        )
    }

    pub(crate) fn manifest_state_counts(
        &self,
        manifest_sha256: &str,
    ) -> Result<BTreeMap<String, u64>> {
        if !self.pair_registry.is_file() {
            bail!("pair registry is missing");
        }
        Ok(sqlite_rows(
            &self.pair_registry,
            &format!(
                "SELECT state,COUNT(*) AS count FROM pairs \
                 WHERE manifest_sha256={} GROUP BY state",
                sql_quote(manifest_sha256)
            ),
        )?
        .into_iter()
        .filter_map(|row| {
            Some((
                row.get("state")?.as_str()?.to_owned(),
                row.get("count")?.as_u64()?,
            ))
        })
        .collect())
    }

    pub(crate) fn supersede_pairs(
        &self,
        manifest_sha256: &str,
        job_id: &str,
        expected_state: &str,
        retirement_sha256: &str,
    ) -> Result<u64> {
        if !matches!(expected_state, "reserved" | "submitted") {
            bail!("unsupported retirement pair state");
        }
        if !is_sha256(retirement_sha256) {
            bail!("retirement receipt is not a SHA-256 value");
        }
        let mut connection = open_sqlite(&self.pair_registry)?;
        let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
            "reservation_owner"
        } else if sqlite_column_exists(&connection, "pairs", "owner")? {
            "owner"
        } else {
            bail!("pair registry has no ownership column");
        };
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let pair_query = if expected_state == "submitted" {
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state
             FROM pairs WHERE manifest_sha256=?1 AND state='submitted' AND job_id=?2"
        } else {
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state
             FROM pairs WHERE manifest_sha256=?1 AND state='reserved' AND job_id IS NULL"
        };
        let mut statement = transaction.prepare(pair_query)?;
        let retiring = statement
            .query_map(params![manifest_sha256, job_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        if retiring.is_empty() {
            let prior: i64 = transaction.query_row(
                "SELECT COUNT(*) FROM attempts WHERE manifest_sha256=?1 AND job_id=?2
                   AND state='superseded_unstarted' AND receipt_sha256=?3",
                params![manifest_sha256, job_id, retirement_sha256],
                |row| row.get(0),
            )?;
            if prior == 0 {
                bail!("retirement selected no protocol pairs");
            }
            transaction.commit()?;
            return Ok(prior as u64);
        }
        let timestamp = now_ms();
        let detail = serde_json::to_string(&json!({
            "manifest_sha256": manifest_sha256,
            "job_id": job_id,
            "retirement_receipt_sha256": retirement_sha256
        }))?;
        for (task, dataset, semantic, protocol, state) in &retiring {
            let active = transaction
                .query_row(
                    "SELECT id,state,job_id FROM attempts
                     WHERE task=?1 AND dataset_hash=?2 AND semantic_sha256=?3
                       AND protocol_sha256=?4
                       AND state IN ('reserved','submitted') AND finalized_at_ms IS NULL",
                    params![task, dataset, semantic, protocol],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                        ))
                    },
                )
                .optional()?
                .context("retirement active-attempt audit failed")?;
            if active.1 != *state || (state == "submitted" && active.2.as_deref() != Some(job_id)) {
                bail!("retirement active-attempt audit failed");
            }
            if transaction.execute(
                "UPDATE attempts SET state='superseded_unstarted',
                   receipt_sha256=?1,job_id=COALESCE(job_id,?2),finalized_at_ms=?3
                 WHERE id=?4 AND state=?5 AND finalized_at_ms IS NULL",
                params![retirement_sha256, job_id, timestamp, active.0, state],
            )? != 1
            {
                bail!("retirement lost active attempt");
            }
            transaction.execute(
                "INSERT INTO events(
                   kind,task,dataset_hash,semantic_sha256,protocol_sha256,
                   detail_json,created_at_ms
                 ) VALUES('superseded_unstarted',?1,?2,?3,?4,?5,?6)",
                params![task, dataset, semantic, protocol, detail, timestamp],
            )?;
            let update = format!(
                "UPDATE pairs SET state='available',{owner_column}='available',
                   manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
                   receipt_sha256=NULL,failure_class=NULL,updated_at_ms=?1
                 WHERE task=?2 AND dataset_hash=?3 AND semantic_sha256=?4
                   AND protocol_sha256=?5 AND state=?6"
            );
            if transaction.execute(
                &update,
                params![timestamp, task, dataset, semantic, protocol, state],
            )? != 1
            {
                bail!("retirement lost pair ownership");
            }
        }
        transaction.commit()?;
        Ok(retiring.len() as u64)
    }

    pub(crate) fn mark_claim_superseded(&self, job_id: &str, manifest_sha256: &str) -> Result<()> {
        if !self.backlog.is_file() {
            return Ok(());
        }
        sqlite_exec(
            &self.backlog,
            &format!(
                "BEGIN IMMEDIATE;\
                 UPDATE claims SET state='superseded_unstarted',updated_at_ms={} \
                 WHERE job_id={} AND manifest_sha256={} \
                 AND state IN ('submitted','checked');\
                 COMMIT;",
                now_ms(),
                sql_quote(job_id),
                sql_quote(manifest_sha256)
            ),
        )
    }
}
