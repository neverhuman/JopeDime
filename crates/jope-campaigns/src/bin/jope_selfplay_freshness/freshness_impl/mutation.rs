use super::*;

impl Supervisor {
    pub(super) fn prior_mutation(
        &self,
        tool: &str,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Option<Value>> {
        if idempotency_key.trim().is_empty() {
            bail!("mutating freshness tools require an idempotency key");
        }
        if let Some(prior) = self.state.mutations.get(idempotency_key) {
            if prior.tool != tool || prior.preview_sha256 != preview_sha256 {
                bail!("idempotency key was used for a different mutation");
            }
            return Ok(Some(prior.result.clone()));
        }
        self.load_preview(preview_sha256, None)?;
        Ok(None)
    }

    pub(super) fn prior_auto_retirement_mutation(
        &self,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Option<Value>> {
        if let Some(prior) =
            self.prior_mutation("apply_stale_retirement", preview_sha256, idempotency_key)?
        {
            return Ok(Some(prior));
        }

        let digest = idempotency_key
            .rsplit_once('-')
            .map(|(_, digest)| digest)
            .filter(|digest| is_sha256(digest))
            .context("automatic retirement key lacks a valid digest")?;
        let legacy_suffix = format!("-{digest}");
        Ok(self
            .state
            .mutations
            .iter()
            .find(|(key, mutation)| {
                key.ends_with(&legacy_suffix)
                    && mutation.tool == "apply_stale_retirement"
                    && mutation.preview_sha256 == preview_sha256
            })
            .map(|(_, mutation)| mutation.result.clone()))
    }

    pub(super) fn finish_mutation(
        &mut self,
        tool: &str,
        preview_sha256: &str,
        idempotency_key: &str,
        result: Value,
    ) -> Result<Value> {
        self.state.mutations.insert(
            idempotency_key.to_owned(),
            Mutation {
                tool: tool.to_owned(),
                preview_sha256: preview_sha256.to_owned(),
                result: result.clone(),
                created_at_ms: now_ms(),
            },
        );
        self.save_state()?;
        Ok(result)
    }

    pub(super) fn apply_stale_retirement(
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

    pub(super) fn manifest_state_counts(
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

    pub(super) fn supersede_pairs(
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

    pub(super) fn mark_claim_superseded(&self, job_id: &str, manifest_sha256: &str) -> Result<()> {
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

    pub(super) fn admission_mutation(
        &mut self,
        paused: bool,
        preview_sha256: &str,
        idempotency_key: &str,
    ) -> Result<Value> {
        let tool = if paused {
            "pause_admission"
        } else {
            "resume_admission"
        };
        if let Some(prior) = self.prior_mutation(tool, preview_sha256, idempotency_key)? {
            return Ok(prior);
        }
        if !self.pair_registry.is_file() {
            bail!("pair registry is missing");
        }
        let previous = sqlite_rows(
            &self.pair_registry,
            "SELECT value FROM controls WHERE key='admission_paused'",
        )?
        .first()
        .and_then(|row| row.get("value"))
        .and_then(Value::as_str)
            == Some("1");
        let reason = format!("freshness:{idempotency_key}");
        let event_kind = if paused {
            "admission_paused"
        } else {
            "admission_resumed"
        };
        let detail = sql_quote(&serde_json::to_string(
            &json!({"reason": reason, "changed": previous != paused}),
        )?);
        sqlite_exec(
            &self.pair_registry,
            &format!(
                "BEGIN IMMEDIATE;\
                 INSERT OR IGNORE INTO controls VALUES('admission_paused','0');\
                 UPDATE controls SET value={} WHERE key='admission_paused';\
                 INSERT INTO events(kind,detail_json,created_at_ms) VALUES({},{},{});\
                 COMMIT;",
                sql_quote(if paused { "1" } else { "0" }),
                sql_quote(event_kind),
                detail,
                now_ms()
            ),
        )?;
        self.state.admission_paused = paused;
        let result = json!({"admission_paused": paused, "changed": previous != paused});
        self.finish_mutation(tool, preview_sha256, idempotency_key, result)
    }

    pub(super) fn request_micro_wave(
        &mut self,
        preview_sha256: &str,
        idempotency_key: &str,
        reason: &str,
    ) -> Result<Value> {
        if let Some(prior) =
            self.prior_mutation("request_micro_wave", preview_sha256, idempotency_key)?
        {
            return Ok(prior);
        }
        let request = json!({
            "schema": "JopeMicroWaveRequest.v1",
            "checkpoint_epoch": self.state.active_epoch,
            "checkpoint_sha256": self.state.active_checkpoint,
            "requested_at_ms": now_ms(),
            "reason": reason,
            "idempotency_key": idempotency_key
        });
        let directory = self.root.join("ops/micro-wave-requests");
        fs::create_dir_all(&directory)?;
        let target = directory.join(format!("{}.json", content_hash(&request)?));
        atomic_json(&target, &request)?;
        let mut result = request;
        result.as_object_mut().unwrap().insert(
            "request".to_owned(),
            json!(target.to_string_lossy().to_string()),
        );
        self.finish_mutation(
            "request_micro_wave",
            preview_sha256,
            idempotency_key,
            result,
        )
    }

    pub(super) fn hold_stale_queued_from_inventory(
        &self,
        inventory: &[InventoryJob],
    ) -> Result<u64> {
        let mut held = 0;
        for job in inventory {
            if !retirement_scoped_job(job)
                || job.epoch == self.state.active_epoch
                || !USEFUL_QUEUE_STATES.contains(&job.state.as_str())
            {
                continue;
            }
            self.scheduler.hold(&job.job_id)?;
            self.event(
                "stale_job_held",
                Some(&job.epoch),
                Some(&job.job_id),
                json!({"automatic": true}),
            )?;
            held += 1;
        }
        Ok(held)
    }

    pub(super) fn auto_retire_stale(&mut self, preview: &Value) -> Result<Value> {
        let preview_sha256 = preview
            .get("preview_receipt_sha256")
            .and_then(Value::as_str)
            .context("automatic retirement preview lacks receipt SHA-256")?;
        let idempotency_key =
            auto_retirement_idempotency_key(&self.state.active_epoch, preview_sha256)?;
        let result = match self.prior_auto_retirement_mutation(preview_sha256, &idempotency_key)? {
            Some(prior) => prior,
            None => self.apply_stale_retirement(preview_sha256, &idempotency_key)?,
        };
        Ok(json!({
            "idempotency_key": idempotency_key,
            "preview_receipt_sha256": preview_sha256,
            "result": result
        }))
    }

    pub(super) fn cycle(&mut self, audit_only: bool, auto_retire_stale: bool) -> Result<Value> {
        self.sync_checkpoint()?;
        let mut inventory = self.inventory()?;
        let automatically_held = if audit_only {
            0
        } else {
            self.hold_stale_queued_from_inventory(&inventory)?
        };
        if automatically_held > 0 {
            inventory = self.inventory()?;
        }
        let preview = self.preview_stale_retirement_from_inventory(&inventory)?;
        let automatic_retirement = if auto_retire_stale && !audit_only {
            let result = self.auto_retire_stale(&preview)?;
            inventory = self.inventory()?;
            Some(result)
        } else {
            None
        };
        let mut result = self.freshness_status_from_inventory(&inventory);
        result.as_object_mut().unwrap().extend(Map::from_iter([
            (
                "latest_preview_receipt_sha256".to_owned(),
                preview["preview_receipt_sha256"].clone(),
            ),
            ("audit_only".to_owned(), json!(audit_only)),
            (
                "auto_retire_stale".to_owned(),
                json!(auto_retire_stale && !audit_only),
            ),
            (
                "automatic_retirement".to_owned(),
                automatic_retirement.unwrap_or(Value::Null),
            ),
            ("automatically_held".to_owned(), json!(automatically_held)),
            (
                "loaded_bundle_sha256".to_owned(),
                std::env::var("JOPE_LOADED_BUNDLE_SHA256").map_or(Value::Null, Value::String),
            ),
        ]));
        atomic_json(&self.root.join("ops/freshness-status.json"), &result)?;
        Ok(result)
    }

    pub(super) fn dispatch(&mut self, method: &str, arguments: &Value) -> Result<Value> {
        if !TOOL_NAMES.contains(&method) {
            bail!("unsupported freshness method {method}");
        }
        let get_string = |name: &str| -> Result<&str> {
            arguments
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .with_context(|| format!("missing {name}"))
        };
        match method {
            "get_freshness_status" => self.freshness_status(),
            "get_active_checkpoint" => Ok(self.active_checkpoint()),
            "list_queue_epochs" => self.queue_epochs(),
            "explain_job_provenance" => self.explain_job(get_string("job_id")?),
            "preview_stale_retirement" => self.preview_stale_retirement(),
            "apply_stale_retirement" => self.apply_stale_retirement(
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "pause_admission" => self.admission_mutation(
                true,
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "resume_admission" => self.admission_mutation(
                false,
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
            ),
            "request_micro_wave" => self.request_micro_wave(
                get_string("preview_receipt_sha256")?,
                get_string("idempotency_key")?,
                arguments
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("operator_request"),
            ),
            _ => unreachable!(),
        }
    }

    pub(super) fn prune_state(&mut self) {
        let cutoff = now_ms() - PREVIEW_MAX_AGE_MS;
        self.state
            .previews
            .retain(|_, preview| preview.created_at_ms >= cutoff);
    }
}
