impl Store {
    pub fn renew_lease(
        &self,
        presented: FencedAttempt,
        expires_at: Timestamp,
    ) -> StoreResult<AttemptRecord> {
        let transaction = self.database.begin_write().store()?;
        let job_key = presented.job_id.to_string();
        let job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(presented.job_id))?;
            job
        };
        let expected = job
            .active_attempt
            .ok_or(StoreError::NoActiveAttempt(presented.job_id))?;
        if expected != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected,
                presented,
            }));
        }
        let key = attempt_table_key(presented.job_id, presented.number);
        let mut record = {
            let table = transaction.open_table(ATTEMPTS).store()?;
            let record = table
                .get(key.as_str())
                .store()?
                .map(|value| decode::<AttemptRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::AttemptNotFound {
                    job_id: presented.job_id,
                    attempt: presented.number.get(),
                })?;
            record
        };
        record.lease.renew(presented, expires_at)?;
        {
            let bytes = encode(&record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(record)
    }

    pub fn record_spawned(
        &self,
        presented: FencedAttempt,
        pid: u32,
        cgroup_path: String,
    ) -> StoreResult<AttemptRecord> {
        if pid == 0 {
            return Err(StoreError::InvalidRecord(
                "spawned process id must be greater than zero".into(),
            ));
        }
        if cgroup_path.is_empty() {
            return Err(StoreError::InvalidRecord("cgroup path is empty".into()));
        }
        let transaction = self.database.begin_write().store()?;
        let job_key = presented.job_id.to_string();
        let job = {
            let table = transaction.open_table(JOBS).store()?;
            let job = table
                .get(job_key.as_str())
                .store()?
                .map(|value| decode::<JobRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::JobNotFound(presented.job_id))?;
            job
        };
        let expected = job
            .active_attempt
            .ok_or(StoreError::NoActiveAttempt(presented.job_id))?;
        if expected != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected,
                presented,
            }));
        }
        let key = attempt_table_key(presented.job_id, presented.number);
        let mut record = {
            let table = transaction.open_table(ATTEMPTS).store()?;
            let record = table
                .get(key.as_str())
                .store()?
                .map(|value| decode::<AttemptRecord>(value.value()))
                .transpose()?
                .ok_or(StoreError::AttemptNotFound {
                    job_id: presented.job_id,
                    attempt: presented.number.get(),
                })?;
            record
        };
        if record.attempt != presented {
            return Err(StoreError::Transition(TransitionError::SupersededFence {
                expected: record.attempt,
                presented,
            }));
        }
        record.pid = Some(pid);
        record.cgroup_path = Some(cgroup_path);
        {
            let bytes = encode(&record)?;
            let mut table = transaction.open_table(ATTEMPTS).store()?;
            table.insert(key.as_str(), bytes.as_slice()).store()?;
        }
        transaction.commit().store()?;
        Ok(record)
    }
}
