use super::*;

impl Keeper {
    pub(super) fn validate_claim_manifest(
        &mut self,
        claim: &Claim,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<ValidManifest> {
        let wave = manifest_wave(&claim.manifest)?;
        if !waves.contains(&wave) {
            bail!("claim manifest wave is not current-checkpoint immutable supply");
        }
        let cache_key = (claim.manifest.clone(), claim.manifest_sha256.clone());
        if let Some(manifest) = self.manifest_cache.get(&cache_key) {
            return Ok(manifest.clone());
        }
        let manifest = validate_manifest(&claim.manifest, checkpoint_sha256, epoch, &wave)?;
        if manifest.sha256 != claim.manifest_sha256 {
            bail!("claim manifest hash drift");
        }
        self.manifest_cache.insert(cache_key, manifest.clone());
        Ok(manifest)
    }

    pub(super) fn ready_claims(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<Vec<ReadyClaim>> {
        let mut ready = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting")
                || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let manifest =
                match self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            ready.push(ReadyClaim {
                claim: claim.clone(),
                manifest,
                owner,
            });
        }
        Ok(ready)
    }

    pub(super) fn rebalance_launch_ready(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
        limit: usize,
    ) -> Result<()> {
        let mut eligible = Vec::new();
        for claim in claims.values() {
            if !matches!(
                claim.state.as_str(),
                "checked" | "submitting" | "deferred_horizon"
            ) || scheduler.contains_key(claim.job_id.as_str())
            {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                (claim.state == "submitting").then_some(claim.job_id.as_str()),
            )?;
            eligible.push((claim, manifest.generated_at_ms));
        }
        eligible.sort_by(|(left, left_time), (right, right_time)| {
            // A crash-window `submitting` claim keeps priority so its
            // deterministic SCQ identity is reconciled before fresh work.
            let left_submit = left.state == "submitting";
            let right_submit = right.state == "submitting";
            right_submit
                .cmp(&left_submit)
                .then_with(|| left_time.cmp(right_time))
                .then_with(|| left.job_id.cmp(&right.job_id))
        });
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (index, (claim, _)) in eligible.into_iter().enumerate() {
            let state = if index < limit {
                if claim.state == "submitting" {
                    "submitting"
                } else {
                    "checked"
                }
            } else {
                "deferred_horizon"
            };
            tx.execute(
                "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                params![state, now_ms() as i64, claim.manifest_sha256],
            )?;
            tx.execute(
                "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                 AND state IN ('checked','submitting','deferred_horizon')",
                params![state, claim.manifest_sha256],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub(super) fn count_deferred_current(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut count = 0;
        for claim in claims.values() {
            if claim.state != "deferred_horizon" || scheduler.contains_key(claim.job_id.as_str()) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            self.validate_pair_reservations(&manifest, claim.owner.as_deref(), None)?;
            count += 1;
        }
        Ok(count)
    }

    pub(super) fn validate_pair_reservations(
        &mut self,
        manifest: &ValidManifest,
        claimed_owner: Option<&str>,
        allow_submitted_job: Option<&str>,
    ) -> Result<String> {
        let cache_key = (
            manifest.sha256.clone(),
            claimed_owner.map(str::to_owned),
            allow_submitted_job.map(str::to_owned),
        );
        if let Some(owner) = self.reservation_cache.get(&cache_key) {
            return Ok(owner.clone());
        }
        let owner_col = self.pair_schema.owner_column;
        let query = format!(
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state,{owner_col},\
             manifest_path,manifest_sha256,job_id FROM pairs WHERE manifest_sha256=?"
        );
        let expected_pairs = manifest
            .pairs
            .iter()
            .map(|pair| {
                (
                    pair.task.clone(),
                    pair.dataset_hash.clone(),
                    pair.semantic_sha256.clone(),
                    pair.protocol_sha256.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        let (owner, observed_pairs) = {
            let mut statement = self.pairs.prepare_cached(&query)?;
            let rows = statement.query_map([manifest.sha256.as_str()], |row| {
                Ok((
                    (
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ),
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            })?;
            let mut owner = None;
            let mut observed_pairs = BTreeSet::new();
            for row in rows {
                let (pair, state, pair_owner, path, digest, submitted_job) = row?;
                observed_pairs.insert(pair);
                let launch_ready = matches!(state.as_str(), "reserved" | "infrastructure_retry");
                let crash_adoptable = state == "submitted"
                    && allow_submitted_job.is_some()
                    && submitted_job.as_deref() == allow_submitted_job;
                if !launch_ready && !crash_adoptable {
                    bail!("manifest protocol pair is not launch-ready");
                }
                if path.as_deref() != Some(manifest.path.to_string_lossy().as_ref())
                    || digest.as_deref() != Some(manifest.sha256.as_str())
                {
                    bail!("pair/manifest binding drift");
                }
                if let Some(expected) = claimed_owner {
                    if pair_owner != expected {
                        bail!("claim/pair owner drift");
                    }
                }
                if owner
                    .replace(pair_owner.clone())
                    .is_some_and(|prior| prior != pair_owner)
                {
                    bail!("manifest spans reservation owners");
                }
            }
            (owner, observed_pairs)
        };
        if observed_pairs != expected_pairs {
            bail!("manifest reserved protocol-pair set mismatch");
        }
        let owner = owner.context("manifest has no reserved protocol pairs")?;
        self.reservation_cache.insert(cache_key, owner.clone());
        Ok(owner)
    }

    pub(super) fn prepare_current_manifests(
        &mut self,
        limit: usize,
        checkpoint_sha256: &str,
        epoch: &str,
        wave: &str,
        claims: &HashMap<String, Claim>,
    ) -> Result<PrepareOutcome> {
        let directory = wave_manifest_directory(&self.args.root, wave);
        let mut paths = fs::read_dir(&directory)?
            .filter_map(|entry| entry.ok().map(|value| value.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .collect::<Vec<_>>();
        paths.sort();
        let claimed = claims
            .values()
            .map(|claim| claim.manifest.clone())
            .collect::<BTreeSet<_>>();
        let rejected = load_rejected_manifests(&self.backlog)?;
        let candidate_paths = paths
            .into_iter()
            .filter(|path| !claimed.contains(path))
            .filter(|path| !rejected.contains(path))
            .collect::<Vec<_>>();
        let unclaimed = candidate_paths.len();
        // Admission must not wait for an eager sweep of an entire micro-wave.
        // Validate enough manifests to fill this cycle plus bounded headroom for
        // invalid or concurrently reserved pairs; later cycles consume the rest.
        let validation_budget = bounded_validation_budget(unclaimed, limit);
        let operator_registry = &self.operator_registry;
        let validation_results = self.workers.install(|| {
            candidate_paths
                .par_iter()
                .take(validation_budget)
                .map(|path| {
                    let manifest = match validate_manifest(path, checkpoint_sha256, epoch, wave) {
                        Ok(manifest) => manifest,
                        Err(error) => return (path.clone(), Err(error), true, None),
                    };
                    let mut preflight = match preflight_manifest(operator_registry, &manifest, None)
                    {
                        Ok(outcome) => outcome,
                        Err(error) => return (path.clone(), Err(error), true, None),
                    };
                    if preflight.disposition == TaskPreflightDisposition::PermanentRejection {
                        let reason = format!("task preflight rejected: {:?}", preflight.reasons);
                        return (
                            path.clone(),
                            Err(anyhow::anyhow!(reason)),
                            true,
                            Some(preflight),
                        );
                    }
                    if preflight.disposition == TaskPreflightDisposition::BoundedScreening {
                        if let Err(error) = bounded_regression_screening(&self.args.evaluator, path)
                        {
                            preflight = match TaskPreflightOutcomeV1::failed_regression_screening(
                                &preflight,
                                REGRESSION_SCREENING_BUDGET_SECONDS,
                            ) {
                                Ok(outcome) => outcome,
                                Err(outcome_error) => {
                                    return (
                                        path.clone(),
                                        Err(outcome_error),
                                        true,
                                        Some(preflight),
                                    )
                                }
                            };
                            return (path.clone(), Err(error), true, Some(preflight));
                        }
                        preflight = match preflight_manifest(
                            operator_registry,
                            &manifest,
                            Some(REGRESSION_SCREENING_BUDGET_SECONDS),
                        ) {
                            Ok(outcome) => outcome,
                            Err(error) => return (path.clone(), Err(error), true, Some(preflight)),
                        };
                    }
                    match validate_with_evaluator(&self.args.evaluator, path) {
                        Ok(()) => (path.clone(), Ok(manifest), false, Some(preflight)),
                        Err(error) => {
                            let permanent = evaluator_failure_is_permanent(&error);
                            (path.clone(), Err(error), permanent, Some(preflight))
                        }
                    }
                })
                .collect::<Vec<_>>()
        });
        let validation_failures = validation_results
            .iter()
            .filter(|(_, result, _, _)| result.is_err())
            .count();
        let validated = validation_results.len().saturating_sub(validation_failures);
        let mut preflight_accepted = 0;
        let mut preflight_permanent_rejections = 0;
        let mut preflight_bounded_screenings = 0;
        let mut candidate_evaluations_preflighted = 0_u64;
        let mut candidate_evaluations_avoided = 0_u64;
        let mut predicted_cpu_seconds_preflighted = 0_u64;
        let mut predicted_cpu_seconds_avoided = 0_u64;
        for (path, result, permanent, preflight) in &validation_results {
            if let Some(preflight) = preflight {
                record_preflight_receipt(&self.backlog, path, preflight)?;
                match preflight.disposition {
                    TaskPreflightDisposition::Accepted => preflight_accepted += 1,
                    TaskPreflightDisposition::PermanentRejection => {
                        preflight_permanent_rejections += 1
                    }
                    TaskPreflightDisposition::BoundedScreening => {}
                    TaskPreflightDisposition::RetryableFailure => {}
                }
                preflight_bounded_screenings +=
                    usize::from(preflight.bounded_screening_budget_seconds.is_some());
                candidate_evaluations_preflighted = candidate_evaluations_preflighted
                    .saturating_add(preflight.requested_work.candidate_evaluation_count);
                candidate_evaluations_avoided = candidate_evaluations_avoided.saturating_add(
                    preflight
                        .estimated_compute_avoided
                        .candidate_evaluation_count,
                );
                predicted_cpu_seconds_preflighted = predicted_cpu_seconds_preflighted
                    .saturating_add(preflight.requested_work.predicted_cpu_seconds);
                predicted_cpu_seconds_avoided = predicted_cpu_seconds_avoided
                    .saturating_add(preflight.estimated_compute_avoided.predicted_cpu_seconds);
            }
            if *permanent {
                let error = result
                    .as_ref()
                    .expect_err("permanent validation outcome is an error");
                reject_manifest(
                    &self.backlog,
                    path,
                    None,
                    &format!("validation_failed:{error:#}"),
                )?;
            }
        }
        let mut manifests = validation_results
            .into_iter()
            .filter_map(|(_, result, _, _)| result.ok())
            .collect::<Vec<_>>();
        manifests.sort_by(|left, right| left.path.cmp(&right.path));

        let owner_column = self.pair_schema.owner_column;
        let tx = self
            .pairs
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let halted: Option<String> = tx
            .query_row(
                "SELECT value FROM controls WHERE key='admission_halted'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if halted.as_deref().is_some_and(|value| value != "0") {
            bail!("pair admission is halted by the uniqueness authority");
        }
        let mut accepted = Vec::new();
        let mut conflicts = Vec::new();
        let mut reservation_conflicts = 0;
        for manifest in manifests {
            if accepted.len() == limit {
                break;
            }
            let owner = match reserve_manifest_pairs_in_tx(&tx, owner_column, &manifest)? {
                Some(value) => value,
                None => {
                    reservation_conflicts += 1;
                    conflicts.push((manifest.path.clone(), manifest.sha256.clone()));
                    continue;
                }
            };
            accepted.push((manifest, owner));
        }
        tx.commit()?;
        for (path, digest) in conflicts {
            reject_manifest(
                &self.backlog,
                &path,
                Some(&digest),
                "pair_reservation_conflict",
            )?;
        }

        // The runner and evaluator are the same files for every spec in this cycle: hash each
        // once here instead of twice per manifest inside render_spec (up to 1024 manifests per
        // cycle). They are hashed again after preparation, so a binary replaced while specs were
        // being written fails the cycle. A wave with nothing accepted renders no spec and, as
        // before, hashes nothing.
        let prepared_rows = if accepted.is_empty() {
            Vec::new()
        } else {
            let runner_sha256 = sha256_file(&self.args.runner)?;
            let evaluator_sha256 = sha256_file(&self.args.evaluator)?;
            let pinned_tools = PinnedTools {
                runner: (self.args.runner.as_path(), &runner_sha256),
                evaluator: (self.args.evaluator.as_path(), &evaluator_sha256),
            };
            let rows = self.workers.install(|| {
                accepted
                    .into_par_iter()
                    .map(|(manifest, owner)| {
                        let job_id = format!("jte-{}", &manifest.sha256[..24]);
                        let spec = self
                            .args
                            .root
                            .join("specs/evaluations")
                            .join(format!("{job_id}.toml"));
                        render_spec(&spec, &job_id, &manifest, &pinned_tools)?;
                        // `submit-batch` runs SCQ's complete authoritative
                        // validation path for every spec before it mutates the
                        // scheduler. Running `scq check` here repeated the same
                        // controller round trip once per manifest and could leave
                        // the cluster starved while a large fresh wave waited.
                        // The manifest/evaluator/input hashes have already been
                        // validated above; permanent batch rejections remain
                        // recorded by `submit_claims`.
                        Ok((manifest, owner, job_id, spec))
                    })
                    .collect::<Vec<Result<_>>>()
            });
            let rows = rows.into_iter().collect::<Result<Vec<_>>>()?;
            pinned_tools.verify_unchanged()?;
            rows
        };
        let prepared = prepared_rows.len();
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (manifest, owner, job_id, spec) in &prepared_rows {
                tx.execute(
                    "INSERT INTO claims(manifest_sha256,manifest,job_id,spec,state,attempts,\
                     reservation_owner,updated_at_ms) VALUES(?,?,?,?, 'checked',0,?,?)",
                    params![
                        manifest.sha256,
                        manifest.path.to_string_lossy(),
                        job_id,
                        spec.to_string_lossy(),
                        owner,
                        now_ms() as i64
                    ],
                )?;
                append_keeper_attempts(&tx, manifest, owner, job_id, "checked", 1)?;
            }
            tx.commit()?;
        }
        for (manifest, owner, _, _) in prepared_rows {
            self.manifest_cache.insert(
                (manifest.path.clone(), manifest.sha256.clone()),
                manifest.clone(),
            );
            self.reservation_cache
                .insert((manifest.sha256.clone(), Some(owner.clone()), None), owner);
        }
        Ok(PrepareOutcome {
            unclaimed,
            validated,
            validation_failures,
            preflight_accepted,
            preflight_permanent_rejections,
            preflight_bounded_screenings,
            candidate_evaluations_preflighted,
            candidate_evaluations_avoided,
            predicted_cpu_seconds_preflighted,
            predicted_cpu_seconds_avoided,
            reservation_conflicts,
            prepared,
        })
    }

    pub(super) fn adopt_scheduler_jobs(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut rows = Vec::new();
        for claim in claims.values() {
            if !matches!(claim.state.as_str(), "checked" | "submitting") {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                continue;
            };
            if !scheduler_job_is_adoptable(job) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            rows.push((manifest, owner, claim.job_id.clone()));
        }
        if rows.is_empty() {
            return Ok(0);
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut attempts = tx.prepare_cached(
                    "SELECT COUNT(*) FROM keeper_attempts WHERE manifest_sha256=? AND job_id=? \
                     AND state IN ('checked','submitting','submitted')",
                )?;
                for (manifest, _, job_id) in &rows {
                    let count: i64 =
                        attempts.query_row(params![manifest.sha256, job_id], |row| row.get(0))?;
                    if count != manifest.pairs.len() as i64 {
                        bail!("scheduler adoption has incomplete active attempt history");
                    }
                }
            }
            tx.commit()?;
        }
        let submissions = rows
            .iter()
            .map(|(manifest, owner, job_id)| (manifest, owner.as_str(), job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&submissions)?;
        Ok(rows.len())
    }

    pub(super) fn record_submissions(
        &mut self,
        submissions: &[(&ValidManifest, &str, &str)],
    ) -> Result<()> {
        if submissions.is_empty() {
            return Ok(());
        }
        {
            let tx = self
                .pairs
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (manifest, owner, job_id) in submissions {
                ensure_native_reserved_attempts(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_pairs_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
                update_native_attempts_submitted(&tx, &self.pair_schema, manifest, owner, job_id)?;
            }
            tx.commit()?;
        }
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            {
                let mut update_claim = tx.prepare_cached(
                    "UPDATE claims SET state='submitted',attempts=attempts+1,\
                     reservation_owner=?,updated_at_ms=? WHERE manifest_sha256=?",
                )?;
                let mut update_attempt = tx.prepare_cached(
                    "UPDATE keeper_attempts SET state='submitted',job_id=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                )?;
                for (manifest, owner, job_id) in submissions {
                    update_claim.execute(params![owner, now_ms() as i64, manifest.sha256])?;
                    update_attempt.execute(params![job_id, manifest.sha256])?;
                }
            }
            tx.commit()?;
        }
        Ok(())
    }

    pub(super) fn submit_claims(&mut self, claims: &[ReadyClaim]) -> Result<(usize, usize, u64)> {
        {
            let tx = self
                .backlog
                .transaction_with_behavior(TransactionBehavior::Immediate)?;
            for row in claims {
                let changed = tx.execute(
                    "UPDATE claims SET state='submitting',updated_at_ms=? \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    params![now_ms() as i64, row.manifest.sha256],
                )?;
                if changed != 1 {
                    bail!("launch-ready claim changed before submission");
                }
                let attempts_changed = tx.execute(
                    "UPDATE keeper_attempts SET state='submitting' \
                     WHERE manifest_sha256=? AND state IN ('checked','submitting')",
                    [row.manifest.sha256.as_str()],
                )?;
                if attempts_changed != row.manifest.pairs.len() {
                    bail!("launch-ready claim has incomplete active attempt history");
                }
            }
            tx.commit()?;
        }
        let batch_outcomes = submit_batch(&self.args.scq, claims)?;
        let outcomes = claims
            .iter()
            .cloned()
            .zip(batch_outcomes)
            .map(|(row, (accepted, transient, detail))| (row, accepted, transient, detail))
            .collect::<Vec<_>>();
        let mut submitted = 0;
        let mut failed = 0;
        let mut submitted_candidate_evaluations = 0_u64;
        let accepted_rows = outcomes
            .iter()
            .filter(|(_, accepted, _, _)| *accepted)
            .map(|(row, _, _, _)| (&row.manifest, row.owner.as_str(), row.claim.job_id.as_str()))
            .collect::<Vec<_>>();
        self.record_submissions(&accepted_rows)?;
        for (row, accepted, transient, detail) in outcomes {
            if accepted {
                submitted += 1;
                submitted_candidate_evaluations =
                    submitted_candidate_evaluations.saturating_add(row.manifest.pairs.len() as u64);
            } else {
                failed += 1;
                self.backlog.execute(
                    "UPDATE claims SET state=?,updated_at_ms=? WHERE manifest_sha256=?",
                    params![
                        if transient {
                            "checked".to_owned()
                        } else {
                            format!(
                                "submit_failed:{}",
                                detail.chars().take(500).collect::<String>()
                            )
                        },
                        now_ms() as i64,
                        row.manifest.sha256
                    ],
                )?;
                self.backlog.execute(
                    "UPDATE keeper_attempts SET state=? WHERE manifest_sha256=? \
                     AND state='submitting'",
                    params![
                        if transient {
                            "checked"
                        } else {
                            "submit_failed"
                        },
                        row.manifest.sha256
                    ],
                )?;
            }
        }
        Ok((submitted, failed, submitted_candidate_evaluations))
    }

    pub(super) fn prepare_infrastructure_retries(
        &mut self,
        claims: &HashMap<String, Claim>,
        scheduler: &HashMap<&str, &SchedulerJob>,
        checkpoint_sha256: &str,
        epoch: &str,
        waves: &BTreeSet<String>,
    ) -> Result<usize> {
        let mut recovered = 0;
        for claim in claims.values() {
            if claim.state != "submitted" || claim.attempts >= 5 {
                continue;
            }
            let Some(job) = scheduler.get(claim.job_id.as_str()) else {
                // Missing scheduler state is ambiguous. Never infer an
                // infrastructure failure and never retry it.
                continue;
            };
            if job.state != "failed"
                || !job
                    .failure_class
                    .as_deref()
                    .is_some_and(is_infrastructure_failure_class)
            {
                continue;
            }
            if authoritative_receipt_exists(&self.args.root, &claim.job_id) {
                continue;
            }
            let Ok(manifest) = self.validate_claim_manifest(claim, checkpoint_sha256, epoch, waves)
            else {
                continue;
            };
            let owner = self.validate_pair_reservations(
                &manifest,
                claim.owner.as_deref(),
                Some(claim.job_id.as_str()),
            )?;
            let retry_number = claim.attempts;
            let new_job_id = format!("{}-r{retry_number}", claim.job_id);
            let new_spec = claim.spec.with_file_name(format!("{new_job_id}.toml"));
            rewrite_spec_job_id(&claim.spec, &new_spec, &claim.job_id, &new_job_id)?;
            scq_check(&self.args.scq, &new_spec)?;
            let failure = job.failure_class.as_deref().unwrap_or("infrastructure");
            self.mark_infrastructure_retry(&manifest, failure, &new_job_id, &owner)?;
            self.backlog.execute(
                "UPDATE claims SET job_id=?,spec=?,state='checked',updated_at_ms=? \
                 WHERE manifest_sha256=? AND state='submitted' AND job_id=?",
                params![
                    new_job_id,
                    new_spec.to_string_lossy(),
                    now_ms() as i64,
                    manifest.sha256,
                    claim.job_id
                ],
            )?;
            recovered += 1;
        }
        Ok(recovered)
    }

    pub(super) fn mark_infrastructure_retry(
        &mut self,
        manifest: &ValidManifest,
        failure: &str,
        new_job_id: &str,
        owner: &str,
    ) -> Result<()> {
        let tx = self
            .pairs
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let retry_increment = if self.pair_schema.has_infrastructure_retries {
            ",infrastructure_retries=infrastructure_retries+1"
        } else {
            ""
        };
        let changed = tx.execute(
            &format!(
                "UPDATE pairs SET state='infrastructure_retry',failure_class=?{retry_increment},\
                 job_id=NULL,updated_at_ms=? WHERE manifest_sha256=? AND state='submitted'"
            ),
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        if changed != manifest.pairs.len() {
            let prior: i64 = tx.query_row(
                "SELECT COUNT(*) FROM pairs WHERE manifest_sha256=? \
                 AND state='infrastructure_retry' AND job_id IS NULL",
                [manifest.sha256.as_str()],
                |row| row.get(0),
            )?;
            if changed != 0 || prior != manifest.pairs.len() as i64 {
                bail!("infrastructure retry pair transition count mismatch");
            }
        }
        finalize_native_attempts_for_retry(&tx, &self.pair_schema, manifest, failure)?;
        append_native_retry_attempts(&tx, &self.pair_schema, manifest, owner, new_job_id)?;
        tx.commit()?;
        let tx = self
            .backlog
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "UPDATE keeper_attempts SET state='infrastructure_retry',failure_class=?,\
             finalized_at_ms=? WHERE manifest_sha256=? AND state='submitted'",
            params![failure, now_ms() as i64, manifest.sha256],
        )?;
        append_keeper_attempts(&tx, manifest, owner, new_job_id, "checked", 1)?;
        tx.commit()?;
        Ok(())
    }
}
