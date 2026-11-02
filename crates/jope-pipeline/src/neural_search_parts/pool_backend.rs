impl<J, S, P: PoolAccess> Backend for PoolBackend<'_, J, S, P>
where
    J: FnMut(&Candidate, &EvaluationRequest) -> Result<WorkerJob>,
    S: FnMut(&Candidate, &[WorkerOutcome], &SelectionPlan) -> Result<Feedback>,
{
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        self.pool.begin_arm(arm)
    }
    fn cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
    fn produce(&mut self, job: WorkerJob) -> Result<WorkerOutcome> {
        let mut outcomes = self.pool.run(vec![job], self.cancelled)?;
        ensure!(outcomes.len() == 1, "missing policy execution outcome");
        Ok(outcomes.remove(0))
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>> {
        selection.validate()?;
        let mut requests = BTreeMap::new();
        let mut jobs = vec![];
        let mut preparation_failures = BTreeMap::new();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let mut prepared = vec![];
            for (fold, expected) in selection
                .selection_requests(&candidate.program_sha256, &candidate.id_sha256)?
                .into_iter()
                .enumerate()
            {
                let mut job = match (self.make_job)(candidate, &expected) {
                    Ok(job) => job,
                    Err(error) => {
                        preparation_failures.insert(
                            candidate.id_sha256.clone(),
                            Feedback {
                                candidate_sha256: candidate.id_sha256.clone(),
                                status: EvaluationStatus::Unavailable,
                                loss: None,
                                physical_fit_starts: Some(0),
                                worker_receipt_sha256: None,
                                replay_evidence_sha256: None,
                                fitted_artifact_sha256: None,
                                reason: Some(format!("job preparation unavailable: {error:#}")),
                                oof: None,
                                retryable_inventory: true,
                            },
                        );
                        break;
                    }
                };
                ensure!(
                    job.request.identity()? == expected.identity()?,
                    "fit factory changed train-side request/seed/recipe"
                );
                job.arm = Some(arm.clone());
                prepared.push((fold, expected.identity()?, job));
            }
            if preparation_failures.contains_key(&candidate.id_sha256) {
                continue;
            }
            for (fold, id, job) in prepared {
                ensure!(
                    requests.insert(id, (candidate_index, fold)).is_none(),
                    "candidate jobs reused request identity"
                );
                jobs.push(job);
            }
        }
        let mut outcomes: Vec<BTreeMap<usize, WorkerOutcome>> =
            (0..candidates.len()).map(|_| BTreeMap::new()).collect();
        for outcome in self.pool.run(jobs, self.cancelled)? {
            let (candidate, fold) = requests
                .remove(&outcome.request_sha256)
                .context("unrequested worker outcome")?;
            ensure!(
                outcomes[candidate].insert(fold, outcome).is_none(),
                "duplicate inner fit outcome"
            );
        }
        ensure!(requests.is_empty(), "missing inner fit outcome");
        candidates
            .iter()
            .zip(outcomes)
            .map(|(candidate, folds)| {
                if let Some(failure) = preparation_failures.remove(&candidate.id_sha256) {
                    return Ok(failure);
                }
                let folds = folds.into_values().collect::<Vec<_>>();
                ensure!(folds.len() == 5, "incomplete inner fold inventory");
                let receipt_sha = sha256_hex(&serde_json::to_vec(&folds)?);
                let replay_sha = fold_replay_evidence(&folds)?;
                let retryable = folds.iter().any(retryable_inventory);
                if let Some(failed) = folds
                    .iter()
                    .find(|o| o.command_status != EvaluationStatus::Completed)
                {
                    return Ok(Feedback {
                        candidate_sha256: candidate.id_sha256.clone(),
                        status: failed.command_status,
                        loss: None,
                        physical_fit_starts: folds.iter().all(unstarted).then_some(0),
                        worker_receipt_sha256: Some(receipt_sha),
                        replay_evidence_sha256: Some(replay_sha),
                        fitted_artifact_sha256: None,
                        reason: failed.reason.clone(),
                        oof: None,
                        retryable_inventory: retryable,
                    });
                }
                let mut feedback = match (self.score)(candidate, &folds, selection) {
                    Ok(feedback) => feedback,
                    Err(error) => Feedback {
                        candidate_sha256: candidate.id_sha256.clone(),
                        status: EvaluationStatus::Failed,
                        loss: None,
                        physical_fit_starts: None,
                        worker_receipt_sha256: Some(receipt_sha.clone()),
                        replay_evidence_sha256: None,
                        fitted_artifact_sha256: None,
                        reason: Some(format!("controller OOF scoring failed: {error:#}")),
                        oof: None,
                        retryable_inventory: false,
                    },
                };
                ensure!(
                    feedback.candidate_sha256 == candidate.id_sha256,
                    "scorer changed candidate identity"
                );
                feedback.worker_receipt_sha256 = Some(receipt_sha);
                feedback.replay_evidence_sha256 = Some(replay_sha);
                feedback.retryable_inventory = false;
                feedback.enforce_selection(selection, candidate);
                Ok(feedback)
            })
            .collect()
    }
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &serde_json::Value) -> Result<()> {
        persist_immutable(&self.journal.join(&arm.id_sha256), name, value)
    }
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>> {
        let path = self
            .journal
            .join(&arm.id_sha256)
            .join(format!("{name}.json"));
        if !path.exists() {
            return Ok(None);
        }
        ensure!(
            std::fs::symlink_metadata(&path)?.file_type().is_file(),
            "search journal entry is not a physical file"
        );
        Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
    }
}

/// Authenticate the ordered inner-fold evidence for live scoring and report replay.
pub fn fold_replay_evidence(folds: &[WorkerOutcome]) -> Result<String> {
    let mut evidence = Vec::with_capacity(folds.len());
    ensure!(folds.len() == 5, "incomplete replay fold inventory");
    let mut requests = BTreeSet::new();
    for (fold, outcome) in folds.iter().enumerate() {
        ensure!(
            requests.insert(&outcome.request_sha256),
            "duplicate replay fold"
        );
        // A completed fold remains scientifically material even if another fold
        // never started. Authenticate its artifacts before the failure return.
        outcome.verify_outputs()?;
        let consumption = &outcome.consumption;
        let proven_nonattempt = matches!(
            outcome.command_status,
            EvaluationStatus::Unavailable
                | EvaluationStatus::Cancelled
                | EvaluationStatus::BudgetExhausted
        ) && outcome.exit_code.is_none()
            && outcome.signal.is_none()
            && outcome.outputs.is_empty()
            && outcome.stdout.is_none()
            && outcome.stderr.is_none()
            && consumption.gate_released_unix_ms.is_none()
            && consumption.observed_exit_unix_ms.is_none()
            && consumption.fit_memory_limit_bytes.is_none()
            && consumption.peak_rss_bytes.is_none()
            && consumption.cgroup_peak_memory_bytes.is_none()
            && consumption.sampled_peak_rss_bytes.is_none()
            && ((consumption.accounting_reason == "not_started: gate never released"
                && consumption.cpu_seconds == Some(0.)
                && consumption.wall_seconds == Some(0.))
                || (consumption.accounting_reason
                    == "launch failed before gate release; gate/setup CPU unavailable"
                    && consumption.cpu_seconds.is_none()
                    && consumption.wall_seconds.is_none()));
        evidence.push(if proven_nonattempt {
            serde_json::json!({"kind":"proven_unadmitted", "fold":fold,
                "job":outcome.job_sha256, "request":outcome.request_sha256})
        } else {
            serde_json::to_value(outcome)?
        });
    }
    Ok(sha256_hex(&serde_json::to_vec(&(
        "asm.fold-replay/v1",
        evidence,
    ))?))
}

fn retryable_inventory(outcome: &WorkerOutcome) -> bool {
    matches!(
        outcome.command_status,
        EvaluationStatus::Unavailable | EvaluationStatus::Cancelled
    ) && outcome.consumption.gate_released_unix_ms.is_none()
        && outcome.exit_code.is_none()
        && outcome.signal.is_none()
        && !outcome.reason.as_ref().is_some_and(|r| {
            r.starts_with("controller_interrupted:") || r.starts_with("original attempt unknown;")
        })
}

fn unstarted(outcome: &WorkerOutcome) -> bool {
    outcome.consumption.gate_released_unix_ms.is_none()
        && outcome.exit_code.is_none()
        && outcome.signal.is_none()
        && outcome.consumption.cpu_seconds == Some(0.)
        && outcome
            .consumption
            .accounting_reason
            .starts_with("not_started:")
}

fn persist_immutable(directory: &Path, name: &str, value: &serde_json::Value) -> Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    ensure!(
        directory.is_absolute() && !directory.starts_with("/tmp"),
        "search journal needs durable absolute location"
    );
    ensure!(
        !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'),
        "invalid search record name"
    );
    std::fs::create_dir_all(directory)?;
    ensure!(
        std::fs::canonicalize(directory)? == directory,
        "search journal traverses symlink"
    );
    let path = directory.join(format!("{name}.json"));
    let bytes = serde_json::to_vec(value)?;
    if path.exists() {
        ensure!(
            std::fs::symlink_metadata(&path)?.file_type().is_file(),
            "search record is not a physical file"
        );
        ensure!(
            std::fs::read(&path)? == bytes,
            "search resume evidence differs: {}",
            path.display()
        );
        return Ok(());
    }
    let temporary = directory.join(format!(
        ".{name}.{}-{}.pending",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .mode(0o600)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &path)?;
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
}
