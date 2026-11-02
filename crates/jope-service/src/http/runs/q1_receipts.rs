use super::*;

impl RunService {
    pub(crate) fn reserve_q1_request(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
    ) -> Result<Option<Vec<u8>>> {
        let key = q1_command_key(&request.request_sha256);
        if let Some(record) = self.store.get(&key).map_err(io)? {
            return q1_replay_or_unknown(record.data);
        }
        let snapshot = q1_running_snapshot(run_id.clone(), request.budget);
        let result = self.journal.commit(
            RunUpdate {
                expected_revision: None,
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            vec![EventBody::RunState(RunStateBody {
                state: RunState::Running,
                phase: RunPhase::Encoding,
                pending: Some("pinned_training_evaluation".into()),
                reason: None,
            })],
            |tx| {
                if tx.get(&key)?.is_some() {
                    return Err(crate::store::StoreError::Invalid("Q1 request already reserved"));
                }
                tx.put(
                    &key,
                    &crate::store::Record::metadata(
                        json!({"status":"started_outcome_unknown","request_sha256":request.request_sha256,"run_id":run_id}),
                        vec![],
                    ),
                )
            },
        );
        match result {
            Ok(_) => Ok(None),
            Err(_) => match self.store.get(&key).map_err(io)? {
                Some(record) => q1_replay_or_unknown(record.data),
                None => Err(io("Could not durably reserve Q1 request.")),
            },
        }
    }

    pub(crate) fn commit_q1_receipt(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
        receipt: EvaluationReceipt,
        receipt_bytes: Vec<u8>,
        completion: Q1ReceiptCompletion,
    ) -> Result<()> {
        let Q1ReceiptCompletion {
            mut attempt_events,
            elapsed,
            terminal_failure,
        } = completion;
        let mut snapshot = self
            .journal
            .snapshot(run_id)
            .map_err(io)?
            .ok_or_else(|| io("Q1 reservation disappeared."))?;
        let expected_revision = snapshot.revision.0;
        let completed = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
            .count();
        let loss = receipt
            .attempts
            .iter()
            .filter_map(|attempt| attempt.loss)
            .min_by(f64::total_cmp);
        snapshot.revision.0 = expected_revision
            .checked_add(1)
            .ok_or_else(|| io("Q1 receipt revision exhausted."))?;
        snapshot.state = if terminal_failure.is_some() || completed == 0 {
            RunState::Failed
        } else {
            RunState::Completed
        };
        snapshot.phase = RunPhase::Finalizing;
        snapshot.pending = None;
        snapshot.budget_used = BudgetUsed {
            evaluations: DecimalU64(receipt.charged),
            wall_seconds: format!("{:.3}", elapsed.as_secs_f64()),
        };
        snapshot.completion_reason = Some(CompletionReason::BudgetEvaluations);
        snapshot.evaluation_receipt = Some(receipt.clone());
        if let Some(reason) = terminal_failure.as_deref() {
            attempt_events.push(EventBody::RunState(RunStateBody {
                state: RunState::Failed,
                phase: RunPhase::Finalizing,
                pending: None,
                reason: Some(reason.to_owned()),
            }));
        }
        attempt_events.push(EventBody::RunFinished(RunFinishedBody {
            completion_reason: CompletionReason::BudgetEvaluations,
            champion: None,
            scores: FinalScores {
                cv: loss,
                holdout: None,
                test: None,
            },
            budget_used: snapshot.budget_used.clone(),
            nondeterminism: Nondeterminism {
                count: DecimalU64(0),
                experiment_ids: vec![],
            },
            evaluation_receipt: Some(Box::new(receipt.clone())),
        }));
        let command_key = q1_command_key(&request.request_sha256);
        let receipt_text = String::from_utf8(receipt_bytes.clone())
            .map_err(|_| io("Serialized Q1 receipt was not UTF-8."))?;
        self.journal.commit(
            RunUpdate {
                expected_revision: Some(expected_revision),
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            attempt_events,
            |tx| {
                let command = tx.get(&command_key)?
                    .ok_or(crate::store::StoreError::Invalid("Q1 reservation missing"))?;
                if command.data["status"] != "started_outcome_unknown" {
                    return Err(crate::store::StoreError::Invalid("Q1 reservation state changed"));
                }
                tx.put(
                    &RecordKey::Evaluation { run: run_id.clone(), number: 0 },
                    &crate::store::Record::metadata(json!({
                        "request_sha256":request.request_sha256,
                        "receipt_bytes":receipt_text,
                        "charged":receipt.charged,
                        "completed":completed,
                        "terminal_failure":terminal_failure,
                        "fold_scores_completed":receipt.attempts.iter()
                            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
                            .map(|attempt| attempt.fold_scores.len()).sum::<usize>(),
                    }), vec![]),
                )?;
                for attempt in &receipt.attempts {
                    tx.put(
                        &RecordKey::Evaluation { run: run_id.clone(), number: attempt.attempt + 1 },
                        &crate::store::Record::metadata(serde_json::to_value(attempt)?, vec![]),
                    )?;
                }
                tx.put(
                    &command_key,
                    &crate::store::Record::metadata(json!({
                        "status":"completed",
                        "request_sha256":request.request_sha256,
                        "run_id":run_id,
                        "receipt_bytes":receipt_text,
                        "terminal_failure":terminal_failure,
                    }), vec![]),
                )
            },
        ).map_err(io)?;
        Ok(())
    }
}
