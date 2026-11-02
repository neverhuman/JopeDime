use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Reserved,
    Submitted,
    Terminal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueAttempt {
    pub attempt_number: u32,
    pub reservation_owner: String,
    pub state: AttemptState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_receipt_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueReservationBook {
    #[serde(default)]
    pub pairs: Vec<QueuePairAttempts>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueuePairAttempts {
    pub evaluation_key: EvaluationKeyV2,
    pub attempts: Vec<QueueAttempt>,
}

impl QueueReservationBook {
    pub fn validate(&self) -> Result<()> {
        let mut keys = BTreeSet::new();
        for entry in &self.pairs {
            entry.evaluation_key.validate()?;
            ensure!(
                keys.insert(entry.evaluation_key.clone()),
                "queue reservation book repeats an evaluation key"
            );
            let mut active = 0;
            for (index, attempt) in entry.attempts.iter().enumerate() {
                ensure!(
                    attempt.attempt_number as usize == index + 1,
                    "queue attempt numbers are not contiguous"
                );
                ensure!(
                    !attempt.reservation_owner.trim().is_empty(),
                    "queue attempt owner is empty"
                );
                match attempt.state {
                    AttemptState::Reserved => {
                        active += 1;
                        ensure!(
                            attempt.evaluator_run_id.is_none()
                                && attempt.final_receipt_sha256.is_none(),
                            "reserved attempt has execution or terminal identity"
                        );
                    }
                    AttemptState::Submitted => {
                        active += 1;
                        ensure!(
                            attempt.evaluator_run_id.is_some()
                                && attempt.final_receipt_sha256.is_none(),
                            "submitted attempt identity is incomplete"
                        );
                    }
                    AttemptState::Terminal => ensure!(
                        attempt.evaluator_run_id.is_some()
                            && attempt.final_receipt_sha256.is_some(),
                        "terminal attempt is not exactly-once finalized"
                    ),
                }
            }
            ensure!(active <= 1, "evaluation pair has duplicate active attempts");
        }
        Ok(())
    }

    pub fn reserve(&mut self, key: EvaluationKeyV2, owner: &str) -> Result<u32> {
        self.validate()?;
        ensure!(!owner.trim().is_empty(), "reservation owner is empty");
        key.validate()?;
        let position = match self
            .pairs
            .iter()
            .position(|entry| entry.evaluation_key == key)
        {
            Some(position) => position,
            None => {
                self.pairs.push(QueuePairAttempts {
                    evaluation_key: key,
                    attempts: Vec::new(),
                });
                self.pairs.len() - 1
            }
        };
        let attempts = &mut self.pairs[position].attempts;
        if let Some(active) = attempts
            .iter()
            .find(|attempt| attempt.state != AttemptState::Terminal)
        {
            ensure!(
                active.reservation_owner == owner,
                "evaluation pair is reserved by another owner"
            );
            return Ok(active.attempt_number);
        }
        ensure!(
            attempts.is_empty(),
            "terminal evaluations require explicit retry, not a fresh reservation"
        );
        attempts.push(QueueAttempt {
            attempt_number: 1,
            reservation_owner: owner.to_owned(),
            state: AttemptState::Reserved,
            evaluator_run_id: None,
            final_receipt_sha256: None,
        });
        Ok(1)
    }

    pub fn submit(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
        evaluator_run_id: &str,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            !evaluator_run_id.trim().is_empty(),
            "evaluator run id is empty"
        );
        let attempt = self.attempt_mut(key, owner, attempt_number)?;
        match attempt.state {
            AttemptState::Reserved => {
                attempt.state = AttemptState::Submitted;
                attempt.evaluator_run_id = Some(evaluator_run_id.to_owned());
                Ok(())
            }
            AttemptState::Submitted => {
                ensure!(
                    attempt.evaluator_run_id.as_deref() == Some(evaluator_run_id),
                    "idempotent submission changed evaluator run identity"
                );
                Ok(())
            }
            AttemptState::Terminal => bail!("terminal attempt cannot be resubmitted"),
        }
    }

    pub fn finalize(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
        receipt_sha256: Sha256Digest,
    ) -> Result<()> {
        self.validate()?;
        let attempt = self.attempt_mut(key, owner, attempt_number)?;
        match attempt.state {
            AttemptState::Submitted => {
                attempt.state = AttemptState::Terminal;
                attempt.final_receipt_sha256 = Some(receipt_sha256);
                Ok(())
            }
            AttemptState::Terminal => {
                ensure!(
                    attempt.final_receipt_sha256.as_ref() == Some(&receipt_sha256),
                    "attempt was already finalized with a different receipt"
                );
                Ok(())
            }
            AttemptState::Reserved => bail!("unsubmitted attempt cannot be finalized"),
        }
    }

    pub fn retry(&mut self, key: &EvaluationKeyV2, owner: &str) -> Result<u32> {
        self.validate()?;
        ensure!(!owner.trim().is_empty(), "reservation owner is empty");
        let attempts = self
            .pairs
            .iter_mut()
            .find(|entry| &entry.evaluation_key == key)
            .map(|entry| &mut entry.attempts)
            .ok_or_else(|| anyhow::anyhow!("evaluation pair has no prior attempt"))?;
        let previous = attempts
            .last()
            .ok_or_else(|| anyhow::anyhow!("evaluation pair has no prior attempt"))?;
        ensure!(
            previous.state == AttemptState::Terminal && previous.final_receipt_sha256.is_some(),
            "retry requires an exactly-once finalized prior attempt"
        );
        let attempt_number = previous
            .attempt_number
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("attempt number overflow"))?;
        attempts.push(QueueAttempt {
            attempt_number,
            reservation_owner: owner.to_owned(),
            state: AttemptState::Reserved,
            evaluator_run_id: None,
            final_receipt_sha256: None,
        });
        Ok(attempt_number)
    }

    pub fn active_attempts(&self) -> usize {
        self.pairs
            .iter()
            .flat_map(|entry| entry.attempts.iter())
            .filter(|attempt| attempt.state != AttemptState::Terminal)
            .count()
    }

    fn attempt_mut(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
    ) -> Result<&mut QueueAttempt> {
        let attempt = self
            .pairs
            .iter_mut()
            .find(|entry| &entry.evaluation_key == key)
            .and_then(|entry| {
                entry
                    .attempts
                    .iter_mut()
                    .find(|attempt| attempt.attempt_number == attempt_number)
            })
            .ok_or_else(|| anyhow::anyhow!("queue attempt does not exist"))?;
        ensure!(
            attempt.reservation_owner == owner,
            "queue attempt owner mismatch"
        );
        Ok(attempt)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueStatusEnvelope {
    pub schema: String,
    pub generated_at_ms: u64,
    pub admission_allowed: bool,
    pub degraded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_closed_reason: Option<String>,
    pub queued: usize,
    pub running: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupplyAdmission {
    Admit,
    HealthyHold(String),
    Degraded(String),
}

pub fn interpret_queue_status(status: &QueueStatusEnvelope) -> Result<SupplyAdmission> {
    ensure!(
        status.schema == "JopeSelfplayQueueKeeper.v3",
        "queue status envelope schema is not v3"
    );
    ensure!(
        status.generated_at_ms > 0,
        "queue status timestamp is absent"
    );
    if status.admission_allowed {
        ensure!(
            !status.degraded && status.error.is_none(),
            "admission cannot be allowed while queue status is degraded"
        );
        return Ok(SupplyAdmission::Admit);
    }
    let reason = status
        .fail_closed_reason
        .clone()
        .or_else(|| status.error.clone())
        .unwrap_or_else(|| "queue admission is held".to_owned());
    if status.degraded {
        Ok(SupplyAdmission::Degraded(reason))
    } else {
        Ok(SupplyAdmission::HealthyHold(reason))
    }
}
