use super::*;

#[cfg(feature = "research")]
impl Discovery {
    // Publish only the capture journal over the last committed GP state. The
    // local candidate may already contain uncommitted elite/promotion changes.
    pub(super) fn journal_research(&mut self, next: &mut Checkpoint) -> Result<()> {
        let now = wall_clock_ms()?;
        let mut durable = self.checkpoint.clone();
        durable.capture = next.capture.clone();
        durable.elapsed_ms = durable.elapsed_at(now)?;
        durable.active_since_ms = Some(now);
        next.elapsed_ms = durable.elapsed_ms;
        next.active_since_ms = durable.active_since_ms;
        self.commit_control(durable, RunState::Running)
    }

    pub(super) fn evaluate_research_wave(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        workers: usize,
    ) -> Result<(Vec<Request>, Vec<Response>)> {
        let context = Arc::clone(
            self.capture_context
                .as_ref()
                .context("capture context missing")?,
        );
        let pending = next.pending.clone();
        let mut fresh = vec![];
        for round in &pending {
            let captured = next
                .capture
                .as_ref()
                .unwrap()
                .rounds
                .get(&round.number)
                .context("pending capture round missing")?
                .clone();
            ensure!(
                round.capture.as_ref() == Some(&captured.slate),
                "pending capture slate mismatch"
            );
            let slate: Slate = CaptureState::read(&self.store, &captured.slate)?;
            if !captured.attempts.is_empty() {
                ensure!(
                    captured.attempts.len() == slate.dispatch.len()
                        && captured.attempts.iter().all(|a| a.result.is_some()),
                    "unknown capture attempt; retry refused"
                );
                continue;
            }
            let mut admitted = BTreeMap::new();
            for (slot, candidate) in round.candidates.iter().enumerate() {
                let index = slate
                    .dsls
                    .iter()
                    .position(|dsl| dsl == &candidate.dsl)
                    .context("admitted candidate absent from slate")?;
                admitted.insert(
                    index,
                    (slot, self.request(round.number, slot as u32, 0, candidate)?),
                );
            }
            for index in &slate.dispatch {
                let (admitted_slot, request) = if let Some((slot, request)) = admitted.remove(index)
                {
                    (Some(slot), request)
                } else {
                    ensure!(
                        *index < 16 && round.number > 0,
                        "label-only slot outside approved v1 bounds"
                    );
                    let state = next.capture.as_mut().unwrap();
                    state.label_counter = state
                        .label_counter
                        .checked_add(1)
                        .context("label attempt counter exhausted")?;
                    let slot = 32 + *index as u32;
                    let mut request = Request {
                        attempt: Attempt {
                            run: next.plan.run_id.clone(),
                            round: round.number,
                            slot,
                            attempt: (u64::from(next.epoch) << 32) | u64::from(state.label_counter),
                        },
                        matrix_path: next.plan.matrix_path.clone(),
                        fold_spec: next.plan.fold_spec(0)?,
                        dsl: slate.dsls[*index].clone(),
                        objective: json!(next.plan.objective),
                        seed: context.provider.label_seed(round.number, *index)?,
                        limits: worker::Limits {
                            wall_ms: next.plan.evaluation_wall_ms,
                            rss_bytes: next.plan.evaluation_rss_bytes,
                        },
                    };
                    context.provider.prepare_request(&mut request)?;
                    (None, request)
                };
                let capture = next.capture.as_mut().unwrap();
                let reference = capture.publish(&self.store, &request)?;
                let attempts = &mut capture.rounds.get_mut(&round.number).unwrap().attempts;
                let position = attempts.len();
                attempts.push(CapturedAttempt {
                    proposal_index: *index,
                    admitted_slot,
                    request: reference,
                    result: None,
                });
                fresh.push((round.number, position, request));
            }
            ensure!(
                admitted.is_empty(),
                "admitted request not reserved for capture"
            );
        }
        if !fresh.is_empty() {
            let now = wall_clock_ms()?;
            next.elapsed_ms = next.elapsed_at(now)?;
            next.active_since_ms = Some(now);
            if next.elapsed_ms >= next.plan.wall_seconds.saturating_mul(1000) {
                next.completed = Some(CompletionReason::BudgetTime);
                next.active_since_ms = None;
                discard_pending(next, None, &[], &[], &mut vec![])?;
                self.commit_control(next.clone(), RunState::Running)?;
                return Ok((vec![], vec![]));
            }
            // Slates, exact requests, roots and every reservation are durable
            // before the first executor call. No GP result is installed here.
            #[cfg(test)]
            capture::boundary("before_reservation");
            self.commit_control(next.clone(), RunState::Running)?;
            #[cfg(test)]
            capture::boundary("after_reservation");
            let requests: Vec<_> = fresh.iter().map(|(_, _, r)| r.clone()).collect();
            parallel_evaluate_research(
                executor,
                &requests,
                workers,
                next.plan
                    .wall_seconds
                    .saturating_mul(1000)
                    .saturating_sub(next.elapsed_at(wall_clock_ms()?)?),
                |index, response, wall_ms| {
                    let (number, position, request) = &fresh[index];
                    ensure!(
                        response.failure != Some(WorkerFailure::StaleAttempt),
                        "stale outcome is unknown, not a failed label"
                    );
                    let measurement = if let Some(code) = response.failure {
                        Rung0Measurement::Failed { code }
                    } else {
                        let measurement = context.provider.measurement(request, &response)?;
                        ensure!(
                            matches!(&measurement, Rung0Measurement::Success { surrogate, objective, .. }
                        if *surrogate == response.scores[0] && *objective == response.scores[1]),
                            "actual surrogate/objective or gate unavailable"
                        );
                        measurement
                    };
                    let capture = next.capture.as_mut().unwrap();
                    let attempt = &capture.rounds[number].attempts[*position];
                    ensure!(attempt.result.is_none(), "duplicate capture outcome");
                    let measured = Measured {
                        response,
                        outcome: RecordedOutcome {
                            proposal_index: attempt.proposal_index,
                            wall_ms,
                            measurement,
                        },
                    };
                    let reference = capture.publish(&self.store, &measured)?;
                    capture.rounds.get_mut(number).unwrap().attempts[*position].result =
                        Some(reference);
                    self.journal_research(next)?;
                    #[cfg(test)]
                    capture::boundary("after_outcome");
                    Ok(())
                },
            )?;
        }
        let capture = next.capture.as_ref().unwrap();
        let mut requests = vec![];
        let mut responses = vec![];
        for pending in &next.pending {
            let round = &capture.rounds[&pending.number];
            let mut ordered = BTreeMap::new();
            for attempt in &round.attempts {
                let request: Request = CaptureState::read(&self.store, &attempt.request)?;
                let measured: Measured = CaptureState::read(
                    &self.store,
                    attempt.result.as_ref().context("unknown capture outcome")?,
                )?;
                ensure!(
                    request.attempt == measured.response.attempt
                        && measured.response.valid()
                        && (measured.response.failure.is_some()
                            || measured.response.scores.len() == 3),
                    "invalid saved capture response"
                );
                if let Some(slot) = attempt.admitted_slot {
                    ordered.insert(slot, (request, measured.response));
                }
            }
            ensure!(
                ordered.len() == pending.candidates.len(),
                "admitted capture inventory mismatch"
            );
            for (expected, (slot, (request, response))) in ordered.into_iter().enumerate() {
                ensure!(
                    slot == expected && request.dsl == pending.candidates[slot].dsl,
                    "admitted capture order mismatch"
                );
                requests.push(request);
                responses.push(response);
            }
        }
        Ok((requests, responses))
    }

    pub(super) fn prepare_research_aux(
        &mut self,
        next: &mut Checkpoint,
        request: Request,
    ) -> Result<(Request, Option<usize>)> {
        let wall_available = next.capture.is_none()
            || next.elapsed_at(wall_clock_ms()?)? < next.plan.wall_seconds.saturating_mul(1000);
        let Some(capture) = next.capture.as_mut() else {
            return Ok((request, None));
        };
        for (index, attempt) in capture.auxiliary.iter().enumerate() {
            let saved: Request = CaptureState::read(&self.store, &attempt.request)?;
            if saved.attempt.round == request.attempt.round
                && saved.attempt.slot == request.attempt.slot
            {
                let mut old = serde_json::to_value(&saved)?;
                let mut new = serde_json::to_value(&request)?;
                old.as_object_mut().unwrap().remove("attempt");
                new.as_object_mut().unwrap().remove("attempt");
                ensure!(
                    old == new && attempt.result.is_some(),
                    "unknown or changed auxiliary request"
                );
                return Ok((saved, Some(index)));
            }
        }
        ensure!(
            capture.auxiliary.len() < capture.config.max_rounds.saturating_mul(2),
            "auxiliary attempt bound exhausted"
        );
        ensure!(
            wall_available,
            "capture wall budget exhausted before auxiliary dispatch"
        );
        let reference = capture.publish(&self.store, &request)?;
        let index = capture.auxiliary.len();
        capture.auxiliary.push(AuxiliaryAttempt {
            request: reference,
            result: None,
            model: None,
        });
        self.journal_research(next)?;
        Ok((request, Some(index)))
    }

    pub(super) fn finish_research_aux(
        &mut self,
        next: &mut Checkpoint,
        index: usize,
        result: AuxiliaryResult,
        model: Option<Vec<u8>>,
    ) -> Result<()> {
        let capture = next.capture.as_mut().unwrap();
        if let Some(bytes) = model {
            let total = capture
                .bytes
                .checked_add(bytes.len() as u64)
                .context("capture byte counter exhausted")?;
            ensure!(
                total <= capture.config.max_bytes,
                "capture model-root budget exhausted"
            );
            capture.auxiliary[index].model =
                Some(self.store.publish_blob(std::io::Cursor::new(bytes))?);
            capture.bytes = total;
        }
        let reference = capture.publish(&self.store, &result)?;
        capture.auxiliary[index].result = Some(reference);
        self.journal_research(next)
    }

    pub(super) fn evaluate_research_aux(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Response> {
        if next.capture.is_none() {
            return Ok(checked_response(&request, executor.evaluate(&request)));
        }
        let result = self.evaluate_research_aux_inner(next, executor, request);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    pub(super) fn evaluate_research_aux_inner(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Response> {
        let (request, index) = self.prepare_research_aux(next, request)?;
        let Some(index) = index else {
            return Ok(checked_response(&request, executor.evaluate(&request)));
        };
        if let Some(reference) = &next.capture.as_ref().unwrap().auxiliary[index].result {
            let result: AuxiliaryResult = CaptureState::read(&self.store, reference)?;
            if let AuxiliaryResult::Evaluation { response } = result {
                ensure!(
                    response.attempt == request.attempt && response.valid(),
                    "auxiliary response identity mismatch"
                );
                return Ok(response);
            }
            bail!("wrong auxiliary result kind");
        }
        let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            research_response(&request, executor.evaluate(&request))
        }))
        .map_err(|_| anyhow::anyhow!("auxiliary evaluator aborted; outcome remains unknown"))??;
        ensure!(
            response.failure != Some(WorkerFailure::StaleAttempt),
            "auxiliary stale response remains unknown"
        );
        self.finish_research_aux(
            next,
            index,
            AuxiliaryResult::Evaluation {
                response: response.clone(),
            },
            None,
        )?;
        Ok(response)
    }

    pub(super) fn capture_io(&mut self, error: anyhow::Error) -> anyhow::Error {
        self.poisoned = true;
        RefitIoError(std::io::Error::other(error.to_string())).into()
    }

    pub(super) fn fit_research_aux(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Vec<u8>> {
        let (request, index) = self
            .prepare_research_aux(next, request)
            .map_err(|error| self.capture_io(error))?;
        let Some(index) = index else {
            return executor.fitted_model(&request);
        };
        let attempt = next.capture.as_ref().unwrap().auxiliary[index].clone();
        if let Some(reference) = attempt.result {
            let result = CaptureState::read(&self.store, &reference)
                .map_err(|error| self.capture_io(error))?;
            return match result {
                AuxiliaryResult::Fit => {
                    let model = attempt
                        .model
                        .context("captured model root missing")
                        .map_err(|error| self.capture_io(error))?;
                    self.store
                        .read_blob(&model)
                        .map_err(|error| self.capture_io(error.into()))
                }
                AuxiliaryResult::Failure {
                    worker: Some(response),
                } => Err(RefitWorkerError(response).into()),
                AuxiliaryResult::Failure { worker: None } => {
                    bail!("retained numerical refit failure")
                }
                _ => Err(self.capture_io(anyhow::anyhow!("wrong auxiliary result kind"))),
            };
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.fitted_model(&request)
        }));
        match result {
            Ok(Ok(bytes)) if !bytes.is_empty() => {
                self.finish_research_aux(next, index, AuxiliaryResult::Fit, Some(bytes.clone()))
                    .map_err(|error| self.capture_io(error))?;
                Ok(bytes)
            }
            Ok(Err(error)) if error.is::<RefitIoError>() => {
                self.poisoned = true;
                Err(error)
            }
            Ok(Err(error)) => {
                let worker = error
                    .downcast_ref::<RefitWorkerError>()
                    .map(|e| e.0.clone());
                self.finish_research_aux(next, index, AuxiliaryResult::Failure { worker }, None)
                    .map_err(|error| self.capture_io(error))?;
                Err(error)
            }
            Ok(Ok(_)) => {
                self.finish_research_aux(
                    next,
                    index,
                    AuxiliaryResult::Failure { worker: None },
                    None,
                )
                .map_err(|error| self.capture_io(error))?;
                bail!("empty fitted model")
            }
            Err(_) => {
                Err(self.capture_io(anyhow::anyhow!("refit aborted; outcome remains unknown")))
            }
        }
    }
}
