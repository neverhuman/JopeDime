impl Store {
    /// Atomically allocate a never-reused fence and persist `Starting` before a
    /// process may be launched.
    pub fn begin_attempt_before_launch(
        &self,
        job_id: JobId,
        acquired_at: Timestamp,
        expires_at: Timestamp,
    ) -> StoreResult<LaunchReceipt> {
        if expires_at <= acquired_at {
            return Err(StoreError::InvalidRecord(
                "attempt lease expiry must be later than acquisition".into(),
            ));
        }
        let transaction = self.database.begin_write().store()?;
        let job_key = job_id.to_string();
        let mut job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(job_id))?;
            job
        };
        let previous_job = job.clone();
        if job.state != JobState::Admitted {
            return Err(StoreError::InvalidTransition(format!(
                "job must be admitted before launch, found {:?}",
                job.state
            )));
        }
        if job.active_attempt.is_some() {
            return Err(StoreError::InvalidTransition(
                "admitted job already has an active attempt".into(),
            ));
        }
        if acquired_at < job.updated_at {
            return Err(StoreError::InvalidRecord(
                "attempt acquisition timestamp precedes the job's last transition".into(),
            ));
        }

        let number_value = job
            .attempt_count
            .checked_add(1)
            .ok_or(StoreError::AttemptCounterExhausted(job_id))?;
        let number = AttemptNumber::new(number_value)
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
        let fence_value = {
            let mut table = transaction.open_table(META).store()?;
            let current = table
                .get(NEXT_FENCE_KEY)
                .store()?
                .map(|value| decode::<u64>(value.value()))
                .transpose()?
                .unwrap_or(FIRST_FENCE);
            let next = current
                .checked_add(1)
                .ok_or(StoreError::FenceCounterExhausted)?;
            let bytes = encode(&next)?;
            table.insert(NEXT_FENCE_KEY, bytes.as_slice()).store()?;
            current
        };
        let fence = FenceToken::new(fence_value)
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
        let fenced = FencedAttempt::new(job_id, number, fence);
        let lease = AttemptLease {
            attempt: fenced,
            acquired_at,
            expires_at,
        };
        let attempt = AttemptRecord {
            attempt: fenced,
            state: JobState::Starting,
            lease,
            pid: None,
            cgroup_path: None,
            started_at: None,
            finished_at: None,
            failure: None,
        };
        attempt
            .validate()
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;

        let transition = FencedTransition {
            job_id,
            from: JobState::Admitted,
            to: JobState::Starting,
            attempt: Some(fenced),
            at: acquired_at,
            reason: Some("fenced attempt lease committed before launch".into()),
        };
        job.state = JobState::Starting;
        job.active_attempt = Some(fenced);
        job.attempt_count = number_value;
        job.updated_at = acquired_at;
        {
            let bytes = encode(&job)?;
            let mut table = transaction.open_table(JOBS).store()?;
            table.insert(job_key.as_str(), bytes.as_slice()).store()?;
        }
        {
            let key = attempt_table_key(job_id, number);
            let bytes = encode(&attempt)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        update_indexes(&transaction, &previous_job, &job)?;
        insert_pending_transition(&transaction, &transition)?;
        transaction.commit().store()?;
        Ok(LaunchReceipt {
            job,
            attempt,
            transition,
        })
    }

    /// Persist a fence-checked transition before its external effect.
    pub fn transition_before_effect(
        &self,
        mutation: TransitionMutation,
    ) -> StoreResult<TransitionReceipt> {
        validate_transition_mutation(&mutation)?;
        let transaction = self.database.begin_write().store()?;
        let job_key = mutation.transition.job_id.to_string();
        let mut job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(mutation.transition.job_id))?;
            job
        };
        let previous_job = job.clone();
        mutation.transition.validate_against(&job)?;
        if mutation.transition.at < job.updated_at {
            return Err(StoreError::InvalidTransition(
                "transition timestamp precedes the current durable state".into(),
            ));
        }

        let mut attempt = match mutation.transition.attempt {
            Some(fenced) => {
                let key = attempt_table_key(fenced.job_id, fenced.number);
                let table = transaction.open_table(ATTEMPTS).store()?;
                let record = table
                    .get(key.as_str())
                    .store()?
                    .map(|value| decode::<AttemptRecord>(value.value()))
                    .transpose()?
                    .ok_or(StoreError::AttemptNotFound {
                        job_id: fenced.job_id,
                        attempt: fenced.number.get(),
                    })?;
                if record.attempt != fenced {
                    return Err(StoreError::Transition(TransitionError::SupersededFence {
                        expected: record.attempt,
                        presented: fenced,
                    }));
                }
                if record.state != mutation.transition.from {
                    return Err(StoreError::InvalidTransition(format!(
                        "attempt state {:?} does not match transition source {:?}",
                        record.state, mutation.transition.from
                    )));
                }
                Some(record)
            }
            None => None,
        };

        job.state = mutation.transition.to;
        job.updated_at = mutation.transition.at;
        job.terminal = mutation.terminal.clone();
        if mutation
            .failure
            .is_some_and(FailureClass::consumes_failure_budget)
        {
            job.failure_count = job.failure_count.saturating_add(1);
        }
        if let Some(record) = attempt.as_mut() {
            record.state = mutation.transition.to;
            if mutation.transition.to == JobState::Running && record.started_at.is_none() {
                record.started_at = Some(mutation.transition.at);
            }
            record.failure = mutation.failure;
            if retires_attempt(mutation.transition.to) {
                record.finished_at = Some(mutation.transition.at);
                job.active_attempt = None;
            }
        }
        job.validate()
            .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;

        {
            let bytes = encode(&job)?;
            let mut table = transaction.open_table(JOBS).store()?;
            table.insert(job_key.as_str(), bytes.as_slice()).store()?;
        }
        if let Some(record) = &attempt {
            let key = attempt_table_key(record.attempt.job_id, record.attempt.number);
            let bytes = encode(record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        update_indexes(&transaction, &previous_job, &job)?;
        insert_pending_transition(&transaction, &mutation.transition)?;
        transaction.commit().store()?;
        Ok(TransitionReceipt {
            job,
            attempt,
            transition: mutation.transition,
        })
    }
}
