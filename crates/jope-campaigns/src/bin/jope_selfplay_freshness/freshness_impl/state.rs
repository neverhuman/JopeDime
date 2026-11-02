use super::*;

impl Supervisor {
    pub(super) fn open(root: PathBuf, scq: PathBuf) -> Result<Self> {
        let ops = root.join("ops");
        fs::create_dir_all(&ops)?;
        let state_path = ops.join("freshness-state.v1.json");
        let state = if state_path.is_file() {
            let parsed: DurableState = serde_json::from_slice(&fs::read(&state_path)?)
                .with_context(|| format!("parse {}", state_path.display()))?;
            if parsed.schema != "JopeFreshnessState.v1" {
                bail!(
                    "{}: unsupported freshness state schema",
                    state_path.display()
                );
            }
            parsed
        } else {
            import_legacy_state(&ops.join("freshness.sqlite"))?
        };
        Ok(Self {
            backlog: ops.join("backlog.sqlite"),
            pair_registry: ops.join("pair-registry.sqlite"),
            events_path: ops.join("freshness-events.v1.jsonl"),
            state_path,
            scheduler: Box::new(ScqScheduler { binary: scq }),
            state,
            verified_checkpoints: BTreeSet::new(),
            manifest_cache: RefCell::new(BTreeMap::new()),
            root,
        })
    }

    pub(super) fn save_state(&self) -> Result<()> {
        atomic_json(&self.state_path, &serde_json::to_value(&self.state)?)
    }

    pub(super) fn event(
        &self,
        kind: &str,
        epoch: Option<&str>,
        job_id: Option<&str>,
        detail: Value,
    ) -> Result<()> {
        let event = json!({
            "schema": "JopeFreshnessAuditEvent.v1",
            "kind": kind,
            "epoch": epoch,
            "job_id": job_id,
            "detail": detail,
            "created_at_ms": now_ms()
        });
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&self.events_path)?;
        // Serialize record and newline before appending; write_all may retry short writes.
        // An I/O failure can still leave an incomplete record.
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        file.write_all(&line)?;
        file.sync_data()?;
        Ok(())
    }

    pub(super) fn pointer_checkpoint(&mut self) -> Result<Option<String>> {
        for relative in ["training/learning.json", "training/accepted.json"] {
            let path = self.root.join(relative);
            if !path.is_file() {
                continue;
            }
            let pointer: Value =
                serde_json::from_slice(&fs::read(&path)?).with_context(|| relative.to_owned())?;
            let digest = pointer
                .get("checkpoint_sha256")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !is_sha256(digest) {
                continue;
            }
            if let Some(checkpoint) = pointer.get("checkpoint").and_then(Value::as_str) {
                let checkpoint = Path::new(checkpoint);
                if checkpoint.is_file() && !self.verified_checkpoints.contains(digest) {
                    if sha256_file(checkpoint)? != digest {
                        bail!("{relative}: checkpoint pointer hash drift");
                    }
                    self.verified_checkpoints.insert(digest.to_owned());
                }
            }
            return Ok(Some(digest.to_owned()));
        }
        let proposal_pointer = self.root.join("proposals/current.json");
        if proposal_pointer.is_file() {
            let value: Value = serde_json::from_slice(&fs::read(&proposal_pointer)?)?;
            if value.get("legacy_adopted").and_then(Value::as_bool) == Some(true) {
                let digest = value
                    .get("checkpoint_sha256")
                    .and_then(Value::as_str)
                    .context("legacy proposal pointer lacks checkpoint_sha256")?;
                if !is_sha256(digest) {
                    bail!("legacy proposal checkpoint hash is invalid");
                }
                let checkpoint =
                    paths::legacy_checkpoints_root().join(format!("{digest}.model.pt"));
                if !checkpoint.is_file() {
                    bail!("legacy-adopted checkpoint artifact is missing");
                }
                if !self.verified_checkpoints.contains(digest)
                    && sha256_file(&checkpoint)? != digest
                {
                    bail!("legacy-adopted checkpoint artifact failed verification");
                }
                self.verified_checkpoints.insert(digest.to_owned());
                return Ok(Some(digest.to_owned()));
            }
        }
        Ok(None)
    }

    pub(super) fn sync_checkpoint(&mut self) -> Result<Value> {
        let Some(checkpoint) = self.pointer_checkpoint()? else {
            return Ok(self.active_checkpoint());
        };
        let epoch = epoch_for_checkpoint(&checkpoint)?;
        if epoch == self.state.active_epoch {
            return Ok(self.active_checkpoint());
        }
        let timestamp = now_ms();
        let previous = self.state.active_epoch.clone();
        if !previous.is_empty() {
            self.state
                .checkpoint_superseded_at_ms
                .entry(previous.clone())
                .or_insert(timestamp);
        }
        self.state.active_epoch = epoch.clone();
        self.state.active_checkpoint = checkpoint;
        self.state
            .checkpoint_activated_at_ms
            .entry(epoch.clone())
            .or_insert(timestamp);
        self.event(
            "checkpoint_advanced",
            Some(&epoch),
            None,
            json!({"previous_epoch": null_if_empty(&previous)}),
        )?;
        self.save_state()?;
        Ok(self.active_checkpoint())
    }

    pub(super) fn claims(&self) -> Result<BTreeMap<String, BacklogClaim>> {
        if !self.backlog.is_file() {
            return Ok(BTreeMap::new());
        }
        let connection =
            Connection::open_with_flags(&self.backlog, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| format!("open {}", self.backlog.display()))?;
        connection.busy_timeout(Duration::from_secs(60))?;
        let owner_expression = if sqlite_column_exists(&connection, "claims", "reservation_owner")?
        {
            "reservation_owner"
        } else {
            "NULL"
        };
        let mut statement = connection.prepare(&format!(
            "SELECT job_id,manifest,manifest_sha256,state,{owner_expression} FROM claims"
        ))?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                BacklogClaim {
                    manifest: PathBuf::from(row.get::<_, String>(1)?),
                    manifest_sha256: row.get(2)?,
                    state: row.get(3)?,
                    reservation_owner: row.get(4)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<BTreeMap<_, _>>>()?)
    }

    pub(super) fn inventory(&self) -> Result<Vec<InventoryJob>> {
        self.inventory_with_manifest_policy(false)
    }

    pub(super) fn inventory_with_manifest_policy(
        &self,
        force_manifest_audit: bool,
    ) -> Result<Vec<InventoryJob>> {
        let claims = self.claims()?;
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for scheduler in self.scheduler.jobs()? {
            let job_id = scheduler.id.clone();
            let Some(claim) = claims.get(&job_id) else {
                continue;
            };
            if let Some(job) = self.inventory_job(scheduler, claim, false, force_manifest_audit)? {
                seen.insert(job_id);
                result.push(job);
            }
        }
        for (job_id, claim) in claims {
            if seen.contains(&job_id) || claim.state != "checked" {
                continue;
            }
            if let Some(job) = self.inventory_job(
                SchedulerJob {
                    id: job_id,
                    state: "checked".to_owned(),
                    ..SchedulerJob::default()
                },
                &claim,
                true,
                force_manifest_audit,
            )? {
                result.push(job);
            }
        }
        Ok(result)
    }

    pub(super) fn inventory_job(
        &self,
        scheduler: SchedulerJob,
        claim: &BacklogClaim,
        outside_scq: bool,
        force_manifest_audit: bool,
    ) -> Result<Option<InventoryJob>> {
        let job_id = scheduler.id.clone();
        if claim.manifest.as_os_str().is_empty() {
            return Ok(None);
        }
        let manifest = claim.manifest.clone();
        if !manifest.is_file() {
            return Ok(None);
        }
        let (freshness_scoped, epoch) = self.audit_manifest(
            &job_id,
            &manifest,
            &claim.manifest_sha256,
            force_manifest_audit,
        )?;
        if !freshness_scoped {
            return Ok(None);
        }
        Ok(Some(InventoryJob {
            state: scheduler.state.clone(),
            epoch,
            manifest,
            manifest_sha256: claim.manifest_sha256.clone(),
            claim_state: claim.state.clone(),
            reservation_owner: claim.reservation_owner.clone(),
            scheduler,
            outside_scq,
            job_id,
        }))
    }

    pub(super) fn audit_manifest(
        &self,
        job_id: &str,
        manifest: &Path,
        expected_sha256: &str,
        force: bool,
    ) -> Result<(bool, String)> {
        let metadata = manifest
            .metadata()
            .with_context(|| format!("stat {}", manifest.display()))?;
        let fingerprint = ManifestFingerprint {
            bytes: metadata.len(),
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        };
        if !force {
            if let Some(cached) = self.manifest_cache.borrow().get(manifest) {
                if cached.expected_sha256 == expected_sha256 && cached.fingerprint == fingerprint {
                    return Ok((cached.freshness_scoped, cached.epoch.clone()));
                }
            }
        }
        let observed_sha256 = sha256_file(manifest)?;
        if expected_sha256 != observed_sha256 {
            bail!("{job_id}: backlog manifest hash drift");
        }
        let parsed: Value = serde_json::from_slice(&fs::read(manifest)?)
            .with_context(|| format!("parse {}", manifest.display()))?;
        let freshness_scoped = freshness_scoped_manifest(&parsed);
        let epoch = if freshness_scoped {
            manifest_epoch(&parsed)
        } else {
            String::new()
        };
        self.manifest_cache.borrow_mut().insert(
            manifest.to_path_buf(),
            CachedManifestAudit {
                expected_sha256: expected_sha256.to_owned(),
                fingerprint,
                freshness_scoped,
                epoch: epoch.clone(),
            },
        );
        Ok((freshness_scoped, epoch))
    }
}
