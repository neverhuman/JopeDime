use super::*;

impl Keeper {
    pub(crate) fn prepare_current_manifests(
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
}
