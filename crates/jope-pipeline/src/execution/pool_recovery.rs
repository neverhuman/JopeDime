use super::*;

impl ExecutionPool {
    pub(super) fn cached(&self, job: &WorkerJob) -> Result<Option<WorkerOutcome>> {
        let id = job.identity()?;
        let path = self.finished_path(&id);
        if !path.exists() {
            return Ok(None);
        }
        let prior: WorkerOutcome = read_json(&path)?;
        ensure!(
            prior.job_sha256 == id && prior.request_sha256 == job.request.identity()?,
            "outdated terminal identity"
        );
        if prior.command_status == EvaluationStatus::Completed {
            job.verify_inputs()?;
        }
        prior.verify_outputs()?;
        Ok(Some(prior))
    }
    pub(super) fn interrupted(&self, job: &WorkerJob, drain: bool) -> Result<WorkerOutcome> {
        let id = job.identity()?;
        let original: Started = read_json(&self.started_path(&id))?;
        ensure!(
            original.job_sha256 == id && original.job.identity()? == id,
            "outdated started identity"
        );
        let recovery = self.recover_group(&original, drain)?;
        let mut outcome = WorkerOutcome::unstarted(
            job,
            EvaluationStatus::Cancelled,
            format!("controller_interrupted: original attempt not retried{recovery}"),
        )?;
        outcome.consumption = ObservedConsumption {
            accounting_reason:
                "original controller interrupted; final counters and exit time unknown".into(),
            ..Default::default()
        };
        Ok(outcome)
    }
    pub(super) fn cleanup_path(&self, id: &str) -> PathBuf {
        self.directory.join("cleanup").join(format!("{id}.json"))
    }
    pub(super) fn clear_cleanup(&self, id: &str) -> Result<()> {
        let path = self.cleanup_path(id);
        if path.exists() {
            fs::remove_file(path)?;
            File::open(self.directory.join("cleanup"))?.sync_all()?;
        }
        Ok(())
    }
    /// Inspect every prior owned group, even if its job is absent from this batch
    /// or already has a terminal receipt. A finished receipt is never proof that
    /// all descendants disappeared. This also recovers pre-Started setup failures.
    pub(super) fn reconcile_cleanup(&self) -> Result<Option<String>> {
        let mut records: BTreeMap<String, Started> = BTreeMap::new();
        for directory in ["started", "cleanup"] {
            for entry in fs::read_dir(self.directory.join(directory))? {
                let entry = entry?;
                let original: Started = read_json(&entry.path())?;
                ensure!(
                    original.job_sha256 == original.job.identity()?
                        && entry.file_name() == format!("{}.json", original.job_sha256).as_str(),
                    "invalid cleanup identity"
                );
                records.insert(original.job_sha256.clone(), original);
            }
        }
        let mut blocked = None;
        for (id, original) in records {
            let recovery = self.recover_group(&original, true)?;
            if recovery.is_empty() {
                self.clear_cleanup(&id)?;
            } else {
                blocked = Some(format!("prior fit {id} cleanup pending{recovery}"));
            }
        }
        Ok(blocked)
    }
    pub(super) fn recover_group(&self, original: &Started, drain: bool) -> Result<String> {
        let id = original.job.identity()?;
        ensure!(original.job_sha256 == id, "outdated cleanup job identity");
        let mut recovery = String::new();
        if original.cgroup.exists() {
            let expected_name = format!("fit-{id}");
            let parent = original.cgroup.parent();
            let owned_path = parent
                .is_some_and(|p| p.parent() == Some(Path::new("/sys/fs/cgroup/system.slice")))
                && original
                    .cgroup
                    .file_name()
                    .is_some_and(|n| n == expected_name.as_str())
                && fs::canonicalize(&original.cgroup).ok().as_ref() == Some(&original.cgroup);
            let same_unit = original.cgroup == self.supervisor.cgroup_root.join(&expected_name);
            let old_inactive = if owned_path && !same_unit {
                parent
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .filter(|n| n.starts_with("jope-asm-") && n.ends_with(".service"))
                    .and_then(|unit| systemd_properties(unit).ok())
                    .is_some_and(|props| {
                        matches!(
                            props.get("ActiveState").map(String::as_str),
                            Some("inactive" | "failed")
                        ) && props.get("MainPID").map(String::as_str) == Some("0")
                            && props.get("ControlGroup").is_some_and(|value| {
                                parent
                                    == Some(
                                        Path::new("/sys/fs/cgroup")
                                            .join(value.trim_start_matches('/'))
                                            .as_path(),
                                    )
                            })
                    })
            } else {
                false
            };
            if drain && owned_path && (same_unit || old_inactive) {
                if let Err(error) = drain_cgroup(&original.cgroup) {
                    recovery = format!("; recovery_not_drained: {error:#}");
                }
            } else if named_counter(&original.cgroup, "cgroup.events", "populated").ok() != Some(0)
            {
                recovery =
                    "; recovery_not_drained: foreign, active or unverifiable prior group".into();
            }
        }
        Ok(recovery)
    }
    pub(super) fn acquire_host(&mut self) -> Result<()> {
        if self.host_lock.is_some() {
            return Ok(());
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(HOST_LOCK)
            .context("shared host lock is not provisioned")?;
        let guard = Flock::lock(file, FlockArg::LockExclusiveNonblock)
            .map_err(|(_, error)| anyhow::anyhow!("exclusive CPU host is busy: {error}"))?;
        let full = bounded_output(Command::new("/usr/bin/docker").args([
            "ps",
            "--filter",
            "label=jope.runner=1",
            "--filter",
            "label=jope.lane=full",
            "--format",
            "{{.ID}}",
        ]))
        .context("runner census")?;
        ensure!(
            full.status.success() && full.stdout.iter().all(u8::is_ascii_whitespace),
            "full-CPU runner container active or census unavailable"
        );
        // The old controller's flock can disappear before systemd finishes
        // killing its descendants. Recover only this journal's safely owned
        // inactive groups, then refuse every other populated ASM service.
        ensure!(
            self.reconcile_cleanup()?.is_none(),
            "prior owned fit cleanup remains unresolved"
        );
        require_no_other_asm_groups(
            Path::new("/sys/fs/cgroup/system.slice"),
            &self.supervisor.cgroup_root,
        )?;
        self.host_lock = Some(guard);
        Ok(())
    }
    pub(super) fn started_path(&self, id: &str) -> PathBuf {
        self.directory.join("started").join(format!("{id}.json"))
    }
    pub(super) fn finished_path(&self, id: &str) -> PathBuf {
        self.directory.join("finished").join(format!("{id}.json"))
    }
    pub(super) fn arm_deadline(&self, arm: &ArmBudget) -> Result<u64> {
        ensure!(
            crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
            "invalid arm allowance"
        );
        let path = self
            .directory
            .join("arms")
            .join(format!("{}.json", arm.id_sha256));
        let window = if path.exists() {
            let window: ArmWindow = read_json(&path)?;
            ensure!(
                window.id_sha256 == arm.id_sha256
                    && window.seconds == arm.seconds
                    && window.started_unix_ms.checked_add(arm.seconds * 1000)
                        == Some(window.deadline_unix_ms),
                "search arm allowance changed"
            );
            window
        } else {
            let started_unix_ms = now_ms()?;
            let window = ArmWindow {
                id_sha256: arm.id_sha256.clone(),
                seconds: arm.seconds,
                started_unix_ms,
                deadline_unix_ms: started_unix_ms + arm.seconds * 1000,
            };
            immutable(&path, &window)?;
            window
        };
        ensure!(
            now_ms()? >= window.started_unix_ms,
            "clock precedes original arm start"
        );
        Ok(window.deadline_unix_ms)
    }
}
