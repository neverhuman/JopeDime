pub fn get_resource_history(
    &self,
    fingerprint: &str,
) -> StoreResult<Option<ResourceHistoryRecord>> {
    validate_fingerprint(fingerprint)?;
    let transaction = self.database.begin_read().store()?;
    let table = transaction.open_table(RESOURCE_HISTORY).store()?;
    table
        .get(fingerprint)
        .store()?
        .map(|value| decode(value.value()))
        .transpose()
}

pub fn update_resource_history<F>(
    &self,
    fingerprint: &str,
    at: Timestamp,
    update: F,
) -> StoreResult<ResourceHistoryRecord>
where
    F: FnOnce(&mut ResourceHistoryRecord),
{
    validate_fingerprint(fingerprint)?;
    let transaction = self.database.begin_write().store()?;
    let mut record = {
        let table = transaction.open_table(RESOURCE_HISTORY).store()?;
        let record = table
            .get(fingerprint)
            .store()?
            .map(|value| decode::<ResourceHistoryRecord>(value.value()))
            .transpose()?
            .unwrap_or(ResourceHistoryRecord::new(fingerprint, at)?);
        record
    };
    if at < record.updated_at {
        return Err(StoreError::InvalidRecord(
            "resource-history timestamp moved backwards".into(),
        ));
    }
    update(&mut record);
    record.updated_at = at;
    if record.fingerprint != fingerprint {
        return Err(StoreError::InvalidRecord(
            "resource-history update changed its fingerprint".into(),
        ));
    }
    let bytes = encode(&record)?;
    {
        let mut table = transaction.open_table(RESOURCE_HISTORY).store()?;
        table.insert(fingerprint, bytes.as_slice()).store()?;
    }
    transaction.commit().store()?;
    Ok(record)
}

/// Return state events that were committed with redb but have not yet been
/// acknowledged as durable in the event segment log.
pub fn pending_transition_events(&self) -> StoreResult<Vec<PendingTransitionEvent>> {
    let transaction = self.database.begin_read().store()?;
    let table = transaction.open_table(TRANSITION_OUTBOX).store()?;
    let mut pending = Vec::with_capacity(table.len().store()? as usize);
    for row in table.iter().store()? {
        let (_key, value) = row.store()?;
        pending.push(decode(value.value())?);
    }
    pending.sort_by_key(|event: &PendingTransitionEvent| {
        (
            event.transition.at,
            event.transition.job_id,
            event.transition.from,
            event.transition.to,
        )
    });
    Ok(pending)
}

/// Idempotently acknowledge publication after the event segment append is
/// durable. A mismatched value under the deterministic key is corruption.
pub fn acknowledge_transition_event(&self, transition: &FencedTransition) -> StoreResult<()> {
    let key = transition_outbox_key(transition)?;
    let transaction = self.database.begin_write().store()?;
    {
        let mut table = transaction.open_table(TRANSITION_OUTBOX).store()?;
        let pending = match table.get(key.as_str()).store()? {
            Some(stored) => Some(decode::<PendingTransitionEvent>(stored.value())?),
            None => None,
        };
        if let Some(pending) = pending {
            if pending.transition != *transition {
                return Err(StoreError::Corrupt(
                    "transition outbox key does not match its value".to_owned(),
                ));
            }
            table.remove(key.as_str()).store()?;
        }
    }
    transaction.commit().store()?;
    Ok(())
}

pub fn put_meta<T: Serialize>(&self, key: &str, value: &T) -> StoreResult<()> {
    validate_meta_key(key)?;
    let bytes = encode(value)?;
    let transaction = self.database.begin_write().store()?;
    {
        let mut table = transaction.open_table(META).store()?;
        table.insert(key, bytes.as_slice()).store()?;
    }
    transaction.commit().store()?;
    Ok(())
}

pub fn get_meta<T: DeserializeOwned>(&self, key: &str) -> StoreResult<Option<T>> {
    validate_meta_key(key)?;
    let transaction = self.database.begin_read().store()?;
    let table = transaction.open_table(META).store()?;
    table
        .get(key)
        .store()?
        .map(|value| decode(value.value()))
        .transpose()
}
