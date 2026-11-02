impl Store {
    pub fn get_job(&self, job_id: JobId) -> StoreResult<Option<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(JOBS).store()?;
        let key = job_id.to_string();
        table
            .get(key.as_str())
            .store()?
            .map(|value| decode(value.value()))
            .transpose()
    }

    pub fn all_jobs(&self) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(JOBS).store()?;
        let mut jobs = Vec::with_capacity(table.len().store()? as usize);
        for row in table.iter().store()? {
            let (_key, value) = row.store()?;
            jobs.push(decode(value.value())?);
        }
        jobs.sort_by_key(|job: &JobRecord| (job.created_at, job.id));
        Ok(jobs)
    }

    /// Return queued jobs in scheduler order without scanning terminal or
    /// running records. The index is maintained in the same transaction as
    /// every job transition.
    pub fn queued_jobs(&self, limit: usize) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let index = transaction.open_table(QUEUED_INDEX).store()?;
        let jobs = transaction.open_table(JOBS).store()?;
        let mut result = Vec::new();
        for row in index.iter().store()?.take(limit) {
            let (_key, value) = row.store()?;
            let job_id = indexed_job_id(value.value())?;
            let key = job_id.to_string();
            let Some(job) = jobs
                .get(key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
            else {
                return Err(StoreError::Corrupt(
                    "queued index references a missing job".to_owned(),
                ));
            };
            if job.state != JobState::Queued {
                return Err(StoreError::Corrupt(
                    "queued index references a non-queued job".to_owned(),
                ));
            }
            result.push(job);
        }
        Ok(result)
    }

    /// Return queued/admitted jobs whose indexed deadline has expired.
    pub fn expired_deadline_jobs(
        &self,
        now: Timestamp,
        limit: usize,
    ) -> StoreResult<Vec<JobRecord>> {
        let transaction = self.database.begin_read().store()?;
        let index = transaction.open_table(DEADLINE_INDEX).store()?;
        let jobs = transaction.open_table(JOBS).store()?;
        let end = deadline_index_key(now, "~");
        let mut result = Vec::new();
        for row in index.range(..=end.as_str()).store()?.take(limit) {
            let (_key, value) = row.store()?;
            let job_id = indexed_job_id(value.value())?;
            let key = job_id.to_string();
            let Some(job) = jobs
                .get(key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
            else {
                return Err(StoreError::Corrupt(
                    "deadline index references a missing job".to_owned(),
                ));
            };
            if matches!(job.state, JobState::Queued | JobState::Admitted)
                && job.spec.deadline.is_some_and(|deadline| deadline <= now)
            {
                result.push(job);
            }
        }
        Ok(result)
    }

    pub fn get_attempt(&self, attempt: FencedAttempt) -> StoreResult<Option<AttemptRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(ATTEMPTS).store()?;
        let key = attempt_table_key(attempt.job_id, attempt.number);
        let record = table
            .get(key.as_str())
            .store()?
            .map(|value| decode::<AttemptRecord>(value.value()))
            .transpose()?;
        match record {
            Some(record) if record.attempt == attempt => Ok(Some(record)),
            Some(_) => Err(StoreError::Corrupt(
                "attempt key and stored fence disagree".into(),
            )),
            None => Ok(None),
        }
    }

    pub fn attempts_for_job(&self, job_id: JobId) -> StoreResult<Vec<AttemptRecord>> {
        let transaction = self.database.begin_read().store()?;
        let table = transaction.open_table(ATTEMPTS).store()?;
        let prefix = format!("{job_id}/");
        let mut attempts = Vec::new();
        for row in table.iter().store()? {
            let (key, value) = row.store()?;
            if key.value().starts_with(&prefix) {
                attempts.push(decode(value.value())?);
            }
        }
        attempts.sort_by_key(|record: &AttemptRecord| record.attempt.number);
        Ok(attempts)
    }
}
