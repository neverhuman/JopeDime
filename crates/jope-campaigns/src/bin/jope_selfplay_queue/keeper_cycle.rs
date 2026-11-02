use super::*;

impl Keeper {
    pub(super) fn open(args: Args) -> Result<Self> {
        fs::create_dir_all(args.root.join("ops"))?;
        let backlog = Connection::open(args.root.join("ops/backlog.sqlite"))?;
        backlog.busy_timeout(Duration::from_secs(60))?;
        initialize_backlog(&backlog)?;
        let terminal_seen = load_terminal_seen(&backlog)?;
        let pairs = Connection::open(args.root.join("ops/pair-registry.sqlite"))?;
        pairs.busy_timeout(Duration::from_secs(60))?;
        let pair_schema = inspect_pair_schema(&pairs)?;
        let operator_registry_path = args.operator_registry.clone().unwrap_or_else(|| {
            args.root
                .join("runtime-bundles/current/registry/operator_registry.json")
        });
        let operator_registry = GeneratedOperatorRegistry::from_bytes(
            &fs::read(&operator_registry_path).with_context(|| {
                format!(
                    "read generated operator registry {}",
                    operator_registry_path.display()
                )
            })?,
        )?;
        let workers = ThreadPoolBuilder::new()
            .num_threads(keeper_workers())
            .thread_name(|index| format!("jope-keeper-worker-{index}"))
            .build()?;
        Ok(Self {
            args,
            backlog,
            pairs,
            pair_schema,
            manifest_cache: HashMap::new(),
            reservation_cache: HashMap::new(),
            terminal_seen,
            operator_registry,
            workers,
        })
    }

    pub(super) fn cycle(&mut self) -> Result<CycleReport> {
        self.manifest_cache.clear();
        self.reservation_cache.clear();
        let scheduler_value = scq_json(&self.args.scq, &["ls"])?;
        let jobs = parse_jobs(&scheduler_value)?;
        let nodes_value = scq_json(&self.args.scq, &["nodes"])?;
        let nodes = parse_nodes(&nodes_value)?;
        let effective_cluster_slots = nodes.iter().map(|row| row.slot_target).sum();
        let burn = record_burn_rate(&mut self.backlog, &jobs, &mut self.terminal_seen)?;
        let targets = dynamic_queue_targets(&burn, effective_cluster_slots)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let gpu_pause = self.args.root.join("ops/gpu-lane.pause.json");
        if gpu_pause.is_file() {
            return Ok(empty_report(
                self.args.submit,
                targets,
                nodes,
                "GPU lane admission transaction paused evaluator queue submission",
            ));
        }
        let checkpoint = active_checkpoint(&self.args.root)?;

        let Some((checkpoint_sha256, checkpoint_epoch)) = checkpoint else {
            return Ok(empty_report(
                self.args.submit,
                targets,
                nodes,
                "active checkpoint pointer is absent",
            ));
        };
        let claims = load_claims(&self.backlog)?;
        let current_waves =
            valid_micro_waves(&self.args.root, &checkpoint_sha256, &checkpoint_epoch)?;
        let Some(newest_wave) = current_waves.last().map(|wave| wave.wave.clone()) else {
            let mut report = empty_report(
                self.args.submit,
                targets,
                nodes,
                "no complete current-checkpoint 128-dataset micro-wave",
            );
            report.active_checkpoint_sha256 = Some(checkpoint_sha256);
            report.checkpoint_epoch = Some(checkpoint_epoch);
            return Ok(report);
        };
        let current_wave_names = current_waves
            .iter()
            .map(|wave| wave.wave.clone())
            .collect::<BTreeSet<_>>();
        let rejected = load_rejected_manifests(&self.backlog)?;
        let wave_inventory =
            current_admission_waves(&self.args.root, &current_waves, &claims, &rejected)?;
        if wave_inventory.len() > MAX_ADMISSION_WAVES {
            bail!(
                "current checkpoint has {} unclaimed waves; maximum admission overlap is {}",
                wave_inventory.len(),
                MAX_ADMISSION_WAVES
            );
        }
        let (mut admission_waves, _) = partition_protected_reserve(wave_inventory);
        let scheduler_by_id = jobs
            .iter()
            .map(|job| (job.id.as_str(), job))
            .collect::<HashMap<_, _>>();
        let mut manifests = HashMap::new();
        let mut useful = BTreeSet::new();
        let mut stale_or_unproven = 0_usize;
        let mut counts = BTreeMap::new();
        let mut max_queue_age = 0.0_f64;
        let now_seconds = (now_ms() / 1_000) as i64;
        for job in &jobs {
            if job.project != PROJECT || !ACTIVE_JOB_STATES.contains(&job.state.as_str()) {
                continue;
            }
            *counts.entry(job.state.clone()).or_insert(0) += 1;
            let valid = match claims.get(&job.id) {
                Some(claim) if matches!(claim.state.as_str(), "submitted" | "submitting") => self
                    .validate_claim_manifest(
                        claim,
                        &checkpoint_sha256,
                        &checkpoint_epoch,
                        &current_wave_names,
                    ),
                _ => Err(anyhow::anyhow!("scheduler job has no active claim")),
            };
            match valid {
                Ok(manifest) => {
                    manifests.insert(job.id.clone(), manifest);
                    useful.insert(job.id.clone());
                    if QUEUE_STATES.contains(&job.state.as_str()) {
                        if let Some(timestamp) =
                            job.queue_ts.as_deref().and_then(parse_rfc3339_seconds)
                        {
                            max_queue_age =
                                max_queue_age.max((now_seconds - timestamp).max(0) as f64);
                        }
                    }
                }
                Err(_) => stale_or_unproven += 1,
            }
        }

        let queued = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "queued")
            .count();
        let starting = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "starting")
            .count();
        let queued_starting = queued.saturating_add(starting);
        let running = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "running")
            .count();
        let dispatching = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "dispatching")
            .count();
        let held = jobs
            .iter()
            .filter(|job| useful.contains(&job.id) && job.state == "held")
            .count();
        let candidates_in_state = |state: &str| -> u64 {
            jobs.iter()
                .filter(|job| useful.contains(&job.id) && job.state == state)
                .filter_map(|job| manifests.get(&job.id))
                .map(|manifest| manifest.pairs.len() as u64)
                .sum()
        };
        let mut queued_candidate_evaluations = candidates_in_state("queued");
        let starting_candidate_evaluations = candidates_in_state("starting");
        let dispatching_candidate_evaluations = candidates_in_state("dispatching");

        // 2026-08-01 incident: a host CPU-overload episode (load ~194 on
        // xbabe2) made the scheduler guard-hold ~996 jope-selfplay-eval jobs
        // ("priority preemption for cpu-overload-guard", "CPU target
        // reached"). SCQ's `held` state is sticky — no daemon releases it —
        // so the flywheel sat at 3 running jobs on an otherwise idle cluster
        // (329/408 cores free) for ~3 hours until an operator released them
        // by hand. The keeper is the one process that knows which held jobs
        // are still provenance-useful, so it now releases them, rate-limited,
        // whenever the useful queue is under target. If the host is genuinely
        // overloaded the guard simply holds them again; a release is one
        // cheap socket call, so the worst case is a slow benign loop.
        let mut released_held = 0;
        if self.args.submit && !self.args.preserve_held {
            let budget = held_release_budget(queued_starting, targets.target, held);
            for job in jobs
                .iter()
                .filter(|job| useful.contains(&job.id) && job.state == "held")
            {
                if released_held >= budget {
                    break;
                }
                if scq_release(&self.args.scq, &job.id).is_ok() {
                    released_held += 1;
                }
            }
        }

        let mut adopted = 0;
        let mut retries = 0;
        if self.args.submit {
            adopted = self.adopt_scheduler_jobs(
                &claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
            retries = self.prepare_infrastructure_retries(
                &claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
        }

        let mut refreshed_claims = load_claims(&self.backlog)?;
        if self.args.submit {
            self.rebalance_launch_ready(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
                targets.target.max(0) as usize,
            )?;
            refreshed_claims = load_claims(&self.backlog)?;
        }
        let mut ready = self.ready_claims(
            &refreshed_claims,
            &scheduler_by_id,
            &checkpoint_sha256,
            &checkpoint_epoch,
            &current_wave_names,
        )?;
        let launch_ready_limit = targets.target.max(0) as usize;
        ready.sort_by(|left, right| {
            left.manifest
                .generated_at_ms
                .cmp(&right.manifest.generated_at_ms)
                .then_with(|| left.claim.job_id.cmp(&right.claim.job_id))
        });
        if ready.len() > launch_ready_limit {
            ready.truncate(launch_ready_limit);
        }
        let mut deferred = self.count_deferred_current(
            &refreshed_claims,
            &scheduler_by_id,
            &checkpoint_sha256,
            &checkpoint_epoch,
            &current_wave_names,
        )?;

        let needed = queue_refill_needed(queued_starting as i64, targets.target, targets.hard_cap)
            .map_err(|error| anyhow::anyhow!(error.to_string()))? as usize;
        let outside_room = launch_ready_limit.saturating_sub(ready.len());
        // Never make launch-ready work wait behind validation of the next
        // batch. Submit it now; the following five-second cycle prepares more.
        let prepare_limit = if ready.is_empty() {
            needed.min(outside_room).min(self.args.max_submit_per_cycle)
        } else {
            0
        };
        let mut prepare = PrepareOutcome::default();
        if self.args.submit && prepare_limit > 0 {
            let mut remaining = prepare_limit;
            for wave in &admission_waves {
                if remaining == 0 {
                    break;
                }
                let outcome = self.prepare_current_manifests(
                    remaining,
                    &checkpoint_sha256,
                    &checkpoint_epoch,
                    &wave.wave,
                    &refreshed_claims,
                )?;
                remaining = remaining.saturating_sub(outcome.prepared);
                prepare.merge(outcome);
                refreshed_claims = load_claims(&self.backlog)?;
            }
            refreshed_claims = load_claims(&self.backlog)?;
            ready = self.ready_claims(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
            ready.sort_by(|left, right| left.claim.job_id.cmp(&right.claim.job_id));
            if ready.len() > launch_ready_limit {
                ready.truncate(launch_ready_limit);
            }
            deferred = self.count_deferred_current(
                &refreshed_claims,
                &scheduler_by_id,
                &checkpoint_sha256,
                &checkpoint_epoch,
                &current_wave_names,
            )?;
        }

        let submit_count = needed
            .min(self.args.max_submit_per_cycle)
            .min(ready.len())
            .min((targets.hard_cap as usize).saturating_sub(queued_starting));
        let (submitted, submission_failures, submitted_candidate_evaluations) =
            if self.args.submit && submit_count > 0 {
                self.submit_claims(&ready[..submit_count])?
            } else {
                (0, 0, 0)
            };
        let report_claims = load_claims(&self.backlog)?;
        let report_rejected = load_rejected_manifests(&self.backlog)?;
        let wave_inventory = current_admission_waves(
            &self.args.root,
            &current_waves,
            &report_claims,
            &report_rejected,
        )?;
        if wave_inventory.len() > MAX_ADMISSION_WAVES {
            bail!(
                "current checkpoint has {} unclaimed waves after admission; maximum overlap is {}",
                wave_inventory.len(),
                MAX_ADMISSION_WAVES
            );
        }
        let complete_current_checkpoint_waves =
            wave_inventory.iter().filter(|wave| wave.complete).count();
        let partial_current_checkpoint_waves = wave_inventory
            .len()
            .saturating_sub(complete_current_checkpoint_waves);
        let (next_admission_waves, protected_reserve_waves) =
            partition_protected_reserve(wave_inventory);
        admission_waves = next_admission_waves;
        let admission_manifests: usize = admission_waves
            .iter()
            .map(|wave| wave.unclaimed_manifests)
            .sum();
        let reserve_manifests: usize = protected_reserve_waves
            .iter()
            .map(|wave| wave.unclaimed_manifests)
            .sum();
        let unclaimed_candidate_evaluations = admission_waves
            .iter()
            .chain(protected_reserve_waves.iter())
            .map(|wave| wave.candidate_evaluation_count)
            .sum::<u64>();
        let unclaimed_current_checkpoint_manifests =
            admission_manifests.saturating_add(reserve_manifests);
        let launch_ready = ready.len().saturating_sub(submitted);
        let launch_ready_candidate_evaluations = ready
            .iter()
            .map(|row| row.manifest.pairs.len() as u64)
            .sum::<u64>()
            .saturating_sub(submitted_candidate_evaluations);
        let validated_ready = unclaimed_current_checkpoint_manifests.saturating_add(launch_ready);
        let validated_ready_candidate_evaluations =
            unclaimed_candidate_evaluations.saturating_add(launch_ready_candidate_evaluations);
        // Successful submissions happened after the scheduler snapshot above.
        // Count them as queued immediately so the published identity remains
        // `queued + starting + dispatching + validated_ready` without a
        // one-cycle hole that could trigger an unnecessary reserve build.
        let queued = queued.saturating_add(submitted);
        queued_candidate_evaluations =
            queued_candidate_evaluations.saturating_add(submitted_candidate_evaluations);
        let queued_starting = queued.saturating_add(starting);
        let useful_work = queued
            .saturating_add(starting)
            .saturating_add(dispatching)
            .saturating_add(validated_ready);
        let candidate_evaluation_work = queued_candidate_evaluations
            .saturating_add(starting_candidate_evaluations)
            .saturating_add(dispatching_candidate_evaluations)
            .saturating_add(validated_ready_candidate_evaluations);
        let candidates_per_manifest = if useful_work == 0 {
            0.0
        } else {
            candidate_evaluation_work as f64 / useful_work as f64
        };

        Ok(CycleReport {
            schema: "JopeSelfplayQueueKeeper.v3",
            generated_at_ms: now_ms(),
            mode: if self.args.submit { "submit" } else { "audit" },
            admission_allowed: true,
            degraded: false,
            error: None,
            fail_closed_reason: None,
            active_checkpoint_sha256: Some(checkpoint_sha256),
            checkpoint_epoch: Some(checkpoint_epoch),
            newest_micro_wave: Some(newest_wave),
            admission_waves,
            complete_reserve_waves: protected_reserve_waves.len(),
            protected_reserve_waves,
            complete_current_checkpoint_waves,
            partial_current_checkpoint_waves,
            reserve_manifests,
            admission_manifests,
            unclaimed_current_checkpoint_manifests,
            queued_target: targets.target,
            queued_hard_cap: targets.hard_cap,
            p95_60m_jobs_per_minute: targets.p95_jobs_per_minute,
            p95_60m_candidate_evaluations_per_minute: targets.p95_jobs_per_minute
                * candidates_per_manifest,
            candidate_throughput_estimator:
                "p95_job_completions_times_current_candidate_evaluations_per_useful_manifest",
            queued,
            starting,
            queued_starting,
            running,
            dispatching,
            held,
            released_held,
            queue_maximum_age_seconds: max_queue_age,
            stale_or_unproven_active: stale_or_unproven,
            launch_ready,
            validated_ready,
            useful_work,
            queued_candidate_evaluations,
            starting_candidate_evaluations,
            dispatching_candidate_evaluations,
            validated_ready_candidate_evaluations,
            candidate_evaluation_work,
            launch_ready_deferred_by_horizon: deferred,
            manifests_unclaimed: prepare.unclaimed,
            manifests_validated: prepare.validated,
            manifest_validation_failures: prepare.validation_failures,
            preflight_accepted: prepare.preflight_accepted,
            preflight_permanent_rejections: prepare.preflight_permanent_rejections,
            preflight_bounded_screenings: prepare.preflight_bounded_screenings,
            candidate_evaluations_preflighted: prepare.candidate_evaluations_preflighted,
            candidate_evaluations_avoided: prepare.candidate_evaluations_avoided,
            predicted_cpu_seconds_preflighted: prepare.predicted_cpu_seconds_preflighted,
            predicted_cpu_seconds_avoided: prepare.predicted_cpu_seconds_avoided,
            pair_reservation_conflicts: prepare.reservation_conflicts,
            prepared: prepare.prepared,
            infrastructure_retries_prepared: retries,
            adopted_after_restart: adopted,
            submitted,
            submission_failures,
            effective_cluster_slots,
            nodes,
            state_counts: counts,
        })
    }
}
