use super::*;

impl Supervisor {
    pub(crate) fn active_checkpoint(&self) -> Value {
        let epoch = null_if_empty(&self.state.active_epoch);
        let checkpoint = null_if_empty(&self.state.active_checkpoint);
        let activated = self
            .state
            .checkpoint_activated_at_ms
            .get(&self.state.active_epoch)
            .copied();
        json!({
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": checkpoint,
            "activated_at_ms": activated,
            "admission_paused": self.state.admission_paused
        })
    }

    pub(crate) fn queue_epochs(&self) -> Result<Value> {
        let inventory = self.inventory()?;
        Ok(self.queue_epochs_from_inventory(&inventory))
    }

    pub(crate) fn queue_epochs_from_inventory(&self, inventory: &[InventoryJob]) -> Value {
        let mut counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        for job in inventory {
            *counts
                .entry(job.epoch.clone())
                .or_default()
                .entry(job.state.clone())
                .or_default() += 1;
        }
        let epochs = counts
            .into_iter()
            .map(|(epoch, states)| {
                let useful_depth = USEFUL_QUEUE_STATES
                    .iter()
                    .map(|state| states.get(*state).copied().unwrap_or(0))
                    .sum::<u64>();
                json!({
                    "epoch": epoch,
                    "active": epoch == self.state.active_epoch,
                    "states": states,
                    "useful_depth": useful_depth,
                    "held": states.get("held").copied().unwrap_or(0),
                    "running": states.get("running").copied().unwrap_or(0),
                    "dispatching": states.get("dispatching").copied().unwrap_or(0)
                })
            })
            .collect::<Vec<_>>();
        json!({
            "active_epoch": null_if_empty(&self.state.active_epoch),
            "epochs": epochs
        })
    }

    pub(crate) fn freshness_status(&self) -> Result<Value> {
        let inventory = self.inventory()?;
        Ok(self.freshness_status_from_inventory(&inventory))
    }

    pub(crate) fn freshness_status_from_inventory(&self, inventory: &[InventoryJob]) -> Value {
        let queue = self.queue_epochs_from_inventory(inventory);
        let epochs = queue["epochs"].as_array().cloned().unwrap_or_default();
        let stale = epochs
            .iter()
            .filter(|row| row.get("active").and_then(Value::as_bool) != Some(true))
            .collect::<Vec<_>>();
        let activated = self
            .state
            .checkpoint_activated_at_ms
            .get(&self.state.active_epoch)
            .copied();
        let mut first_running = None;
        for job in inventory {
            if job.epoch != self.state.active_epoch || job.state != "running" {
                continue;
            }
            let observed = job
                .scheduler
                .created_at_ms
                .or(job.scheduler.submitted_at_ms)
                .unwrap_or(0);
            first_running = Some(first_running.map_or(observed, |prior: i64| prior.min(observed)));
        }
        let latency = activated
            .zip(first_running)
            .map(|(activated, running)| (running - activated).max(0));
        let stale_useful = stale
            .iter()
            .filter_map(|row| row.get("useful_depth").and_then(Value::as_u64))
            .sum::<u64>();
        let stale_held = stale
            .iter()
            .filter_map(|row| row.get("held").and_then(Value::as_u64))
            .sum::<u64>();
        let mut status = self.active_checkpoint();
        let object = status.as_object_mut().expect("active checkpoint is object");
        object.extend(Map::from_iter([
            ("schema".to_owned(), json!("JopeSelfplayFreshnessStatus.v1")),
            ("queue_epochs".to_owned(), json!(epochs)),
            ("stale_epoch_count".to_owned(), json!(stale.len())),
            ("stale_useful_depth".to_owned(), json!(stale_useful)),
            ("stale_held".to_owned(), json!(stale_held)),
            (
                "checkpoint_to_first_job_latency_ms".to_owned(),
                json!(latency),
            ),
            (
                "first_job_deadline_ms".to_owned(),
                json!(FIRST_JOB_DEADLINE_MS),
            ),
            (
                "healthy".to_owned(),
                json!(latency.is_none_or(|value| value <= FIRST_JOB_DEADLINE_MS)),
            ),
        ]));
        status
    }

    pub(crate) fn explain_job(&self, wanted: &str) -> Result<Value> {
        let job = self
            .inventory()?
            .into_iter()
            .find(|job| job.job_id == wanted)
            .with_context(|| format!("unknown job {wanted}"))?;
        let pairs = if self.pair_registry.is_file() {
            sqlite_rows(
                &self.pair_registry,
                &format!(
                    "SELECT * FROM pairs WHERE manifest_sha256={} \
                     ORDER BY task,dataset_hash,semantic_sha256",
                    sql_quote(&job.manifest_sha256)
                ),
            )?
        } else {
            Vec::new()
        };
        Ok(json!({
            "job": job.as_value(),
            "active_epoch": null_if_empty(&self.state.active_epoch),
            "pair_count": pairs.len(),
            "pairs": pairs
        }))
    }

    pub(crate) fn write_preview(&mut self, kind: &str, payload: Value) -> Result<Value> {
        self.prune_state();
        let timestamp = now_ms();
        if let Some((digest, preview)) = self
            .state
            .previews
            .iter()
            .find(|(_, preview)| {
                preview.kind == kind
                    && preview.payload == payload
                    && timestamp - preview.created_at_ms <= PREVIEW_MAX_AGE_MS
            })
            .map(|(digest, preview)| (digest.clone(), preview.clone()))
        {
            let mut result = serde_json::to_value(preview)?;
            result
                .as_object_mut()
                .expect("preview is object")
                .insert("preview_receipt_sha256".to_owned(), json!(digest));
            return Ok(result);
        }
        let preview = Preview {
            schema: "JopeFreshnessPreview.v1".to_owned(),
            kind: kind.to_owned(),
            created_at_ms: timestamp,
            payload,
        };
        let encoded = serde_json::to_value(&preview)?;
        let digest = content_hash(&encoded)?;
        self.state.previews.entry(digest.clone()).or_insert(preview);
        self.save_state()?;
        let mut result = encoded;
        result
            .as_object_mut()
            .expect("preview is object")
            .insert("preview_receipt_sha256".to_owned(), json!(digest));
        Ok(result)
    }

    pub(crate) fn preview_stale_retirement(&mut self) -> Result<Value> {
        let inventory = self.inventory()?;
        self.preview_stale_retirement_from_inventory(&inventory)
    }

    pub(crate) fn preview_stale_retirement_from_inventory(
        &mut self,
        inventory: &[InventoryJob],
    ) -> Result<Value> {
        let mut actions = Vec::new();
        let mut preserved = Vec::new();
        for job in inventory {
            if job.epoch == self.state.active_epoch {
                continue;
            }
            if !retirement_scoped_job(job) {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "protected_project"
                }));
                continue;
            }
            if NON_INTERRUPTIBLE_STATES.contains(&job.state.as_str()) {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "started"
                }));
                continue;
            }
            if !matches!(
                job.state.as_str(),
                "queued" | "starting" | "held" | "cancelled" | "checked"
            ) {
                continue;
            }
            let expected_claim = if job.state == "checked" {
                "checked"
            } else {
                "submitted"
            };
            if job.claim_state != expected_claim {
                preserved.push(json!({
                    "job_id": job.job_id,
                    "state": job.state,
                    "reason": "claim_not_confirmed_unstarted"
                }));
                continue;
            }
            let action = match job.state.as_str() {
                "cancelled" => "finalize_cancelled",
                "checked" => "retire_checked_unstarted",
                _ => "cancel_after_hold",
            };
            actions.push(json!({
                "job_id": job.job_id,
                "epoch": job.epoch,
                "observed_state": job.state,
                "action": action,
                "manifest": job.manifest,
                "manifest_sha256": job.manifest_sha256,
                "claim_state": job.claim_state
            }));
        }
        actions.sort_by(|left, right| left["job_id"].as_str().cmp(&right["job_id"].as_str()));
        preserved.sort_by(|left, right| left["job_id"].as_str().cmp(&right["job_id"].as_str()));
        self.write_preview(
            "stale_retirement",
            json!({
                "active_epoch": null_if_empty(&self.state.active_epoch),
                "actions": actions,
                "preserved_started": preserved
            }),
        )
    }

    pub(crate) fn load_preview(
        &self,
        digest: &str,
        expected_kind: Option<&str>,
    ) -> Result<Preview> {
        if !is_sha256(digest) {
            bail!("preview receipt must be a SHA-256 value");
        }
        let preview = self
            .state
            .previews
            .get(digest)
            .cloned()
            .context("unknown preview receipt")?;
        if expected_kind.is_some_and(|kind| kind != preview.kind) {
            bail!("preview receipt kind mismatch");
        }
        if now_ms() - preview.created_at_ms > PREVIEW_MAX_AGE_MS {
            bail!("preview receipt expired");
        }
        Ok(preview)
    }
}
