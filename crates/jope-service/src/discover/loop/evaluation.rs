use super::*;

pub(super) fn checked_response(request: &Request, response: Response) -> Response {
    if response.attempt != request.attempt
        || !response.valid()
        || (response.failure.is_none() && response.scores.len() != 3)
    {
        Response::failed(request.attempt.clone(), WorkerFailure::Protocol, 0)
    } else {
        response
    }
}

pub(super) fn parallel_evaluate(
    executor: &impl Executor,
    requests: &[Request],
    workers: usize,
) -> Vec<Response> {
    let cursor = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; requests.len()]);
    std::thread::scope(|scope| {
        for _ in 0..workers.min(requests.len()) {
            scope.spawn(|| loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(request) = requests.get(index) else {
                    break;
                };
                let response = checked_response(request, executor.evaluate(request));
                results.lock().expect("result mutex poisoned")[index] = Some(response);
            });
        }
    });
    results
        .into_inner()
        .expect("result mutex poisoned")
        .into_iter()
        .map(|r| r.expect("worker result missing"))
        .collect()
}

pub(super) fn record_result(
    state: &mut Checkpoint,
    candidate: &Candidate,
    response: &Response,
    id: &str,
    events: &mut Vec<EventBody>,
) -> Result<()> {
    add_events(state, candidate, response, id, events)?;
    if response.failure.is_none() {
        state.elites.insert(Elite {
            algorithm_id: candidate.algorithm_id.clone(),
            surrogate: response.scores[0],
            objective: response.scores[1],
        })?;
    }
    Ok(())
}

pub(super) fn add_events(
    state: &mut Checkpoint,
    candidate: &Candidate,
    response: &Response,
    id: &str,
    events: &mut Vec<EventBody>,
) -> Result<()> {
    if matches!(
        response.failure,
        Some(WorkerFailure::Timeout | WorkerFailure::MemoryLimit)
    ) {
        state.nondeterminism.push(id.into());
    }
    events.push(EventBody::ExperimentStarted(ExperimentStartedBody {
        experiment_id: id.into(),
        hypothesis: projected_summary(&state.plan, project::Claim::Unknown)?,
    }));
    events.push(EventBody::ExperimentFinished(ExperimentFinishedBody {
        experiment_id: id.into(),
        outcome: if response.failure.is_none() {
            ExperimentOutcome::NotImproved
        } else {
            ExperimentOutcome::Failed
        },
        metric: public_metric(&state.plan)?,
        value: (response.attempt.slot / 65536 == 1)
            .then(|| response.scores.get(2).copied())
            .flatten(),
        failure: response.failure.map(|failure| PublicFailure {
            code: match failure {
                WorkerFailure::NonFinite => jope_api::error::ErrorCode::FitnessUndefined,
                WorkerFailure::Unavailable => jope_api::error::ErrorCode::BackendUnavailable,
                WorkerFailure::InvalidRequest => jope_api::error::ErrorCode::InvalidInput,
                _ => jope_api::error::ErrorCode::NumericalFailure,
            },
            reason: public_failure(failure),
        }),
        attempt_receipt: Some(Box::new(jope_api::EvaluationAttempt {
            // The public receipt identifies the logical evaluation, just like its
            // experiment ID. Worker restart epochs remain in private run records.
            attempt: evaluation_ordinal(response.attempt.round, response.attempt.slot)?,
            outcome: if response.failure.is_some() {
                jope_api::EvaluationAttemptOutcome::Failed
            } else {
                jope_api::EvaluationAttemptOutcome::Completed
            },
            fold_scores: response.fold_scores.clone(),
            loss: response
                .scores
                .get(1)
                .copied()
                .filter(|_| response.failure.is_none()),
            program_sha256: response
                .failure
                .is_none()
                .then(|| digest(candidate.dsl.as_bytes())),
            failure: response.failure.map(|failure| jope_api::EvaluationFailure {
                code: format!("{failure:?}").to_ascii_lowercase(),
                reason: public_failure(failure),
            }),
        })),
    }));
    Ok(())
}

pub(super) fn public_failure(failure: WorkerFailure) -> String {
    let reason = match failure {
        WorkerFailure::Timeout => "the evaluation reached its time limit",
        WorkerFailure::MemoryLimit => "the evaluation exceeded its memory limit",
        WorkerFailure::NonFinite => "the objective was undefined for its predictions",
        WorkerFailure::Unavailable => "the evaluation backend was unavailable",
        WorkerFailure::InvalidRequest => "the approach could not be evaluated",
        _ => "the evaluation did not complete",
    };
    let name = &feat_math::public_names::heads::HEADS[0];
    phrase::render_phrase(
        phrase::Template::Failure,
        &Sha256::digest(reason.as_bytes()).into(),
        phrase::CatalogEntry {
            public_name: name.public_name,
            purpose: name.purpose,
        },
        phrase::EventDetails {
            reason: Some(reason),
            ..Default::default()
        },
    )
    .expect("reviewed failure phrase")
}

pub(super) fn discard_pending(
    state: &mut Checkpoint,
    executor: Option<&dyn Executor>,
    requests: &[Request],
    responses: &[Response],
    events: &mut Vec<EventBody>,
) -> Result<()> {
    let reason = state
        .completed
        .context("discard requires a completion reason")?;
    for pending in std::mem::take(&mut state.pending) {
        #[cfg(feature = "research")]
        if let Some(capture) = state.capture.as_mut() {
            if let Some(round) = capture.rounds.get_mut(&pending.number) {
                round.aborted = true;
            }
            capture.tainted = true;
        }
        for (slot, candidate) in pending.candidates.into_iter().enumerate() {
            let index = requests
                .iter()
                .position(|r| r.attempt.round == pending.number && r.attempt.slot == slot as u32);
            if let (Some(executor), Some(index)) = (executor, index) {
                executor.revoke(&requests[index].attempt);
            }
            let response = index.and_then(|i| responses.get(i)).cloned();
            if let Some(response) = &response {
                let id = experiment(&state.plan, pending.number, slot, 0)?;
                add_events(state, &candidate, response, &id, events)?;
            }
            state.discarded.push(DiscardedEvaluation {
                round: pending.number,
                slot: slot as u32,
                candidate,
                reason,
                response,
            });
        }
    }
    Ok(())
}
