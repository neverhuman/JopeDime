impl Store {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).store()?;
        }
        let database = Database::create(&path).store()?;
        let mut transaction = database.begin_write().store()?;
        transaction.set_durability(Durability::Immediate);
        {
            let _table = transaction.open_table(JOBS).store()?;
        }
        {
            let _table = transaction.open_table(ATTEMPTS).store()?;
        }
        {
            let _table = transaction.open_table(IDEMPOTENCY).store()?;
        }
        {
            let _table = transaction.open_table(RESOURCE_HISTORY).store()?;
        }
        {
            let _table = transaction.open_table(TRANSITION_OUTBOX).store()?;
        }
        {
            let _table = transaction.open_table(META).store()?;
        }
        {
            let _table = transaction.open_table(QUEUED_INDEX).store()?;
        }
        {
            let _table = transaction.open_table(DEADLINE_INDEX).store()?;
        }
        let jobs = {
            let table = transaction.open_table(JOBS).store()?;
            let mut jobs = Vec::with_capacity(table.len().store()? as usize);
            for row in table.iter().store()? {
                let (_key, value) = row.store()?;
                jobs.push(decode::<JobRecord>(value.value())?);
            }
            jobs
        };
        rebuild_indexes(&transaction, &jobs)?;
        transaction.commit().store()?;
        Ok(Self { database, path })
    }
}
