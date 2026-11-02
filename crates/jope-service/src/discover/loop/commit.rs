use super::*;

impl Discovery {
    fn commit_control(&mut self, next: Checkpoint, state: RunState) -> Result<()> {
        ensure!(!self.poisoned, "commit outcome unknown: reopen the run");
        let mut snapshot = self.snapshot.clone();
        snapshot.state = state;
        snapshot.revision.0 = snapshot
            .revision
            .0
            .checked_add(1)
            .context("revision exhausted")?;
        snapshot.budget_used.wall_seconds = format!("{:.3}", next.elapsed_ms as f64 / 1000.0);
        #[cfg(feature = "research")]
        if next.capture.is_some() {
            snapshot.budget_used.evaluations = DecimalU64(rung0_budget_used(&next));
        }
        if next.completed.is_some() {
            snapshot.phase = RunPhase::Finalizing;
        }
        self.poisoned = true;
        let result = self.log.commit(
            RunUpdate {
                expected_revision: Some(self.snapshot.revision.0),
                snapshot,
                checkpoint: serde_json::to_value(&next)?,
                blobs: checkpoint_blobs(&next)?,
            },
            vec![],
            |_| Ok(()),
        )?;
        self.snapshot = result.snapshot;
        self.checkpoint = next;
        self.poisoned = false;
        Ok(())
    }

    /// Execute a fixed planned wave, or its retained suffix after recovery.
    /// `stop_after_rounds` permits a supervisor to request a bounded pause without
    /// changing the plan or event stream. Already computed siblings are discarded.
    pub fn advance(
        &mut self,
        space: &impl SearchSpace,
        admission: &impl RoundAdmission,
        executor: &impl Executor,
        live_workers: usize,
        stop_after_rounds: usize,
    ) -> Result<Vec<PublicEvent>> {
        ensure!(!self.poisoned, "commit outcome unknown: reopen the run");
        ensure!(
            self.snapshot.state == RunState::Running,
            "continue a paused run by reopening it"
        );
        ensure!(
            live_workers > 0 && live_workers <= 1024 && stop_after_rounds > 0,
            "invalid execution capacity"
        );
        if self.completed().is_some() {
            return Ok(vec![]);
        }
        let started = ElapsedSpan::start();
        let elapsed_before = self.checkpoint.elapsed_at(wall_clock_ms()?)?;
        if elapsed_before >= self.checkpoint.plan.wall_seconds.saturating_mul(1000) {
            let mut next = self.checkpoint.clone();
            next.elapsed_ms = elapsed_before;
            next.active_since_ms = None;
            next.completed = Some(CompletionReason::BudgetTime);
            discard_pending(&mut next, None, &[], &[], &mut vec![])?;
            self.commit_control(next, RunState::Running)?;
            return Ok(vec![]);
        }
        let mut next = self.checkpoint.clone();
        #[cfg(feature = "research")]
        ensure!(
            next.capture.as_ref().is_none_or(|s| !s.tainted),
            "capture has unknown outcomes; retained for inspection only"
        );
        if next.pending.is_empty() {
            #[cfg(feature = "research")]
            if let Some(context) = self.capture_context.as_deref() {
                plan_wave_inner(&mut next, space, admission, Some((&self.store, context)))?;
            } else {
                plan_wave(&mut next, space, admission)?;
            }
            #[cfg(not(feature = "research"))]
            plan_wave(&mut next, space, admission)?;
        }
        #[cfg(feature = "research")]
        let capture_work = if self.capture_context.is_some() {
            let result = self.evaluate_research_wave(&mut next, executor, live_workers);
            if result.is_err() {
                self.poisoned = true;
            }
            Some(result?)
        } else {
            None
        };
        let mut work = vec![];
        #[cfg(feature = "research")]
        let captured = capture_work.is_some();
        #[cfg(not(feature = "research"))]
        let captured = false;
        if !captured {
            for round in &next.pending {
                for (slot, candidate) in round.candidates.iter().enumerate() {
                    work.push(self.request(round.number, slot as u32, 0, candidate)?);
                }
            }
        }
        let outcomes;
        #[cfg(feature = "research")]
        if let Some((requests, responses)) = capture_work {
            work = requests;
            outcomes = responses;
        } else {
            outcomes = parallel_evaluate(executor, &work, live_workers);
        }
        #[cfg(not(feature = "research"))]
        {
            outcomes = parallel_evaluate(executor, &work, live_workers);
        }
        let mut offset = 0;
        let mut published = vec![];
        for _ in 0..stop_after_rounds.min(next.pending.len()) {
            let round = next.pending.remove(0);
            let end = offset + round.candidates.len();
            let mut events = vec![];
            let mut records = vec![];
            for (slot, candidate) in round.candidates.iter().enumerate() {
                let response = &outcomes[offset + slot];
                let id = experiment(&next.plan, round.number, slot, 0)?;
                record_result(&mut next, candidate, response, &id, &mut events)?;
                records.push(json!({"round":round.number,"slot":slot,"rung":0,"candidate":candidate,"worker_attempt":response.attempt,
                    "scores":response.scores,"predictions_digest":response.predictions_digest,"failure":response.failure}));
            }
            let best = round
                .candidates
                .iter()
                .filter_map(|c| {
                    let response =
                        &outcomes[offset + round.candidates.iter().position(|v| v == c)?];
                    response.failure.is_none().then(|| (c, response.scores[1]))
                })
                .min_by(|(a, x), (b, y)| {
                    x.partial_cmp(y)
                        .expect("validated finite objective")
                        .then_with(|| a.algorithm_id.cmp(&b.algorithm_id))
                });
            let mut promoted = false;
            if let Some((candidate, _)) = best.filter(|(c, _)| {
                next.elites
                    .rank_by_objective(&c.algorithm_id)
                    .is_some_and(|rank| rank < 4)
            }) {
                let request = self.request(round.number, 0, 1, candidate)?;
                #[cfg(feature = "research")]
                let mut response =
                    self.evaluate_research_aux(&mut next, executor, request.clone())?;
                #[cfg(not(feature = "research"))]
                let mut response = checked_response(&request, executor.evaluate(&request));
                let cv_response = response.clone();
                let mut refit_failure = None;
                let improving = response.failure.is_none()
                    && next
                        .champion
                        .as_ref()
                        .is_none_or(|c| response.scores[1] < c.loss);
                let bytes = if improving {
                    let refit = self.request(round.number, 0, 2, candidate)?;
                    // Numerical refit failure belongs to this candidate. Store IO
                    // failure below still aborts the round and requires recovery.
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(feature = "research")]
                        {
                            self.fit_research_aux(&mut next, executor, refit.clone())
                        }
                        #[cfg(not(feature = "research"))]
                        {
                            executor.fitted_model(&refit)
                        }
                    })) {
                        Ok(Ok(bytes)) if !bytes.is_empty() => Some(bytes),
                        Ok(Err(error)) if error.is::<RefitIoError>() => return Err(error),
                        failure => {
                            let worker = match &failure {
                                Ok(Err(error)) => error
                                    .downcast_ref::<RefitWorkerError>()
                                    .map(|e| e.0.clone()),
                                _ => None,
                            };
                            let kind = worker
                                .as_ref()
                                .and_then(|r| r.failure)
                                .unwrap_or(WorkerFailure::NonFinite);
                            refit_failure = worker;
                            response =
                                Response::failed(request.attempt.clone(), kind, response.wall_ms);
                            None
                        }
                    }
                } else {
                    None
                };
                response.fold_scores = cv_response.fold_scores.clone();
                let id = experiment(&next.plan, round.number, 0, 1)?;
                add_events(&mut next, candidate, &response, &id, &mut events)?;
                records.push(json!({"round":round.number,"slot":0,"rung":1,"candidate":candidate,"worker_attempt":response.attempt,
                    "scores":cv_response.scores,"predictions_digest":cv_response.predictions_digest,
                    "failure":response.failure,"cv_failure":cv_response.failure,"refit_failure":refit_failure}));
                if let Some(bytes) = bytes {
                    // Publication precedes the metadata reference. An aborted transaction
                    // may leave an unreferenced CAS object, never a partial champion.
                    let model = self.store.publish_blob(bytes.as_slice())?;
                    let previous = next.champion.as_ref().map(|c| c.value);
                    let artifact: ArtifactId = format!("art_{}", &digest(&bytes)[..24]).parse()?;
                    next.champion = Some(Champion {
                        candidate: candidate.clone(),
                        loss: response.scores[1],
                        value: response.scores[2],
                        predictions_digest: response.predictions_digest.clone(),
                        model,
                    });
                    let summary =
                        accepted_summary(&next.plan, candidate, response.scores[2], previous)?;
                    events.push(EventBody::AlgorithmAccepted(AlgorithmAcceptedBody {
                        algorithm_id: candidate.algorithm_id.parse()?,
                        metric: public_metric(&next.plan)?,
                        cv_value: response.scores[2],
                        previous_value: previous,
                        summary,
                        model_artifact: Some(artifact),
                        excerpt: String::new(),
                    }));
                    promoted = true;
                }
            }
            next.stalled = if promoted {
                0
            } else {
                next.stalled.saturating_add(1)
            };
            next.next_round = round
                .number
                .checked_add(1)
                .context("round counter exhausted")?;
            next.elapsed_ms =
                elapsed_before.saturating_add(started.millis().min(u128::from(u64::MAX)) as u64);
            next.completed = if next.stalled >= next.plan.limits().patience {
                Some(CompletionReason::Converged)
            } else if next.elapsed_ms >= next.plan.wall_seconds.saturating_mul(1000) {
                Some(CompletionReason::BudgetTime)
            } else if next.pending.is_empty()
                && rung0_budget_used(&next) >= rung0_budget_limit(&next)
            {
                Some(CompletionReason::BudgetEvaluations)
            } else {
                None
            };
            if next.completed.is_some() {
                discard_pending(&mut next, Some(executor), &work, &outcomes, &mut events)?;
            }
            next.active_since_ms = if next.completed.is_none() {
                Some(wall_clock_ms()?)
            } else {
                None
            };
            #[cfg(feature = "research")]
            if let Some(capture) = next.capture.as_mut() {
                if let Err(error) = capture.finish_round(&self.store, round.number) {
                    self.poisoned = true;
                    return Err(error);
                }
            }
            #[cfg(all(test, feature = "research"))]
            if next.capture.is_some() {
                capture::boundary("before_round");
            }
            let committed = self.commit_round(next.clone(), events, records)?;
            #[cfg(all(test, feature = "research"))]
            if next.capture.is_some() {
                capture::boundary("after_round");
            }
            published.extend(committed);
            offset = end;
            if self.completed().is_some() {
                break;
            }
        }
        Ok(published)
    }

    fn request(
        &mut self,
        round: u64,
        slot: u32,
        rung: u8,
        candidate: &Candidate,
    ) -> Result<Request> {
        self.attempt = self
            .attempt
            .checked_add(1)
            .context("session attempt counter exhausted")?;
        let plan = &self.checkpoint.plan;
        let request = Request {
            attempt: Attempt {
                run: plan.run_id.clone(),
                round,
                slot: slot + u32::from(rung) * 65_536,
                attempt: (u64::from(self.checkpoint.epoch) << 32) | u64::from(self.attempt),
            },
            matrix_path: plan.matrix_path.clone(),
            fold_spec: plan.fold_spec(rung)?,
            dsl: candidate.dsl.clone(),
            objective: json!(plan.objective),
            seed: evaluation_seed(plan.seed, round, slot, rung),
            limits: worker::Limits {
                wall_ms: plan.evaluation_wall_ms,
                rss_bytes: plan.evaluation_rss_bytes,
            },
        };
        #[cfg(feature = "research")]
        let request = {
            let mut request = request;
            if rung == 0 {
                if let Some(context) = &self.capture_context {
                    context.provider.prepare_request(&mut request)?;
                }
            }
            request
        };
        Ok(request)
    }

    fn commit_round(
        &mut self,
        next: Checkpoint,
        events: Vec<EventBody>,
        records: Vec<Value>,
    ) -> Result<Vec<PublicEvent>> {
        let mut snapshot = self.snapshot.clone();
        snapshot.revision.0 += 1;
        snapshot.phase = if next.completed.is_some() {
            RunPhase::Finalizing
        } else {
            RunPhase::Searching
        };
        snapshot.budget_used = BudgetUsed {
            evaluations: DecimalU64(rung0_budget_used(&next)),
            wall_seconds: format!("{:.3}", next.elapsed_ms as f64 / 1000.0),
        };
        snapshot.best = next
            .champion
            .as_ref()
            .map(|c| -> Result<_> {
                Ok(BestAlgorithm {
                    algorithm_id: c.candidate.algorithm_id.parse()?,
                    metric: public_metric(&next.plan)?,
                    cv_value: c.value,
                    model_artifact: Some(format!("art_{}", &hex_blob(&c.model)[..24]).parse()?),
                })
            })
            .transpose()?;
        let run = next.plan.run_id.clone();
        let round = next.next_round - 1;
        let champion = next.champion.clone();
        self.poisoned = true;
        let committed = self.log.commit(RunUpdate {
            expected_revision: Some(self.snapshot.revision.0), snapshot,
            checkpoint: serde_json::to_value(&next)?, blobs: checkpoint_blobs(&next)?,
        }, events, |tx| {
            if let Some(champion) = champion {
                let key = RecordKey::Artifact(format!("art_{}", &hex_blob(&champion.model)[..24]).parse()
                    .map_err(|_| StoreError::Invalid("artifact identity invalid"))?);
                if let Some(old) = tx.get(&key)? {
                    if old.blobs != [champion.model] { return Err(StoreError::Integrity("artifact identity collision")); }
                } else {
                    tx.put(&key, &Record::metadata(json!({"kind":"m1-native-fit", "model_digest":hex_blob(&champion.model)}), vec![champion.model]))?;
                }
            }
            for (slot, data) in records.into_iter().enumerate() {
                let number = round.checked_mul(65_536).and_then(|n| n.checked_add(slot as u64)).ok_or(StoreError::Invalid("evaluation key exhausted"))?;
                let key = RecordKey::Evaluation { run: run.clone(), number };
                if tx.get(&key)?.is_some() { return Err(StoreError::Integrity("round evaluation already committed")); }
                tx.put(&key, &Record::metadata(data, vec![]))?;
            }
            Ok(())
        })?;
        self.snapshot = committed.snapshot;
        self.checkpoint = next;
        self.poisoned = false;
        Ok(committed.events)
    }
}
