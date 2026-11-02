/// Atomically create a queued job and its idempotency binding.
pub fn submit(&self, job: &JobRecord) -> StoreResult<SubmitOutcome> {
    job.validate()
        .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
    if job.state != JobState::Queued
        || job.active_attempt.is_some()
        || job.attempt_count != 0
        || job.terminal.is_some()
    {
        return Err(StoreError::InvalidRecord(
            "new jobs must be pristine queued records".into(),
        ));
    }

    let digest = digest(&job.spec)?;
    let scope = IdempotencyScope::for_submission(job.submitted_by.clone(), &job.spec);
    let idempotency_key = idempotency_table_key(&scope)?;
    let job_key = job.id.to_string();
    let transaction = self.database.begin_write().store()?;
    let existing_binding = {
        let table = transaction.open_table(IDEMPOTENCY).store()?;
        let binding = table
            .get(idempotency_key.as_str())
            .store()?
            .map(|value| decode::<IdempotencyRecord>(value.value()))
            .transpose()?;
        binding
    };

    if let Some(binding) = existing_binding {
        if binding.scope != scope {
            return Err(StoreError::Corrupt(
                "idempotency index key does not match its stored scope".into(),
            ));
        }
        if binding.spec_digest != digest {
            return Err(StoreError::IdempotencyConflict {
                existing_job: binding.job_id,
            });
        }
        let existing = {
            let table = transaction.open_table(JOBS).store()?;
            let binding_key = binding.job_id.to_string();
            let job = table
                .get(binding_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::Corrupt(
                    "idempotency binding references a missing job".into(),
                ))?;
            job
        };
        transaction.abort().store()?;
        return Ok(SubmitOutcome::Existing(existing));
    }

    {
        let mut table = transaction.open_table(JOBS).store()?;
        if table.get(job_key.as_str()).store()?.is_some() {
            return Err(StoreError::JobAlreadyExists(job.id));
        }
        let bytes = encode(job)?;
        table.insert(job_key.as_str(), bytes.as_slice()).store()?;
    }
    {
        let binding = IdempotencyRecord {
            scope,
            job_id: job.id,
            spec_digest: digest,
            created_at: job.created_at,
        };
        let bytes = encode(&binding)?;
        let mut table = transaction.open_table(IDEMPOTENCY).store()?;
        table
            .insert(idempotency_key.as_str(), bytes.as_slice())
            .store()?;
    }
    index_job(&transaction, job)?;
    transaction.commit().store()?;
    Ok(SubmitOutcome::Created(job.clone()))
}
