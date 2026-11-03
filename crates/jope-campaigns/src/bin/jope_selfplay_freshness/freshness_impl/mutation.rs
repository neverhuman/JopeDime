use super::*;

#[path = "mutation/retirement.rs"]
mod retirement;

impl Supervisor {
    pub(crate) fn prior_mutation(
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

    pub(crate) fn prior_auto_retirement_mutation(
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

    pub(crate) fn finish_mutation(
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

    pub(crate) fn admission_mutation(
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

    pub(crate) fn request_micro_wave(
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

    pub(crate) fn hold_stale_queued_from_inventory(
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

    pub(crate) fn auto_retire_stale(&mut self, preview: &Value) -> Result<Value> {
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

    pub(crate) fn cycle(&mut self, audit_only: bool, auto_retire_stale: bool) -> Result<Value> {
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

    pub(crate) fn dispatch(&mut self, method: &str, arguments: &Value) -> Result<Value> {
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

    pub(crate) fn prune_state(&mut self) {
        let cutoff = now_ms() - PREVIEW_MAX_AGE_MS;
        self.state
            .previews
            .retain(|_, preview| preview.created_at_ms >= cutoff);
    }
}
