//! Explicit store deletion and content-addressed blob collection.
//!
//! These operations are intentionally detached from the HTTP router. The next
//! router-state task supplies the engine-owned `Store` and dispatches requests.

use super::{db, definition, BlobRef, Record, RecordKey, Store, StoreError, Table, Write};
use jope_api::{ids::DatasetId, ids::RunId, RunState};
use redb::ReadableTable;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeleteDataset {
    Deleted,
    NotFound,
    Referenced,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeleteRun {
    Deleted,
    NotFound,
    Active,
    RevisionConflict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GcReport {
    /// Blobs selected by the reference scan. On a dry run these are only listed;
    /// on a real run they are exactly the files removed.
    pub blobs: Vec<BlobRef>,
    pub bytes: u64,
}

impl Store {
    /// Delete a dataset record only when no run metadata or run blob references
    /// point at it. Blob removal is deferred to explicit GC.
    pub fn delete_dataset(&self, id: &DatasetId) -> super::Result<DeleteDataset> {
        let _guard = self
            .writer
            .lock()
            .map_err(|_| StoreError::Integrity("writer lock poisoned"))?;
        let mut transaction = self.database.begin_write().map_err(db)?;
        transaction.set_durability(redb::Durability::Immediate);

        let dataset_key = RecordKey::Dataset(id.clone()).bytes()?;
        let dataset = {
            let table = transaction
                .open_table(definition(Table::Datasets))
                .map_err(db)?;
            let Some(value) = table.get(dataset_key.as_slice()).map_err(db)? else {
                return Ok(DeleteDataset::NotFound);
            };
            serde_json::from_slice::<Record>(value.value())?
        };

        {
            let runs = transaction
                .open_table(definition(Table::Runs))
                .map_err(db)?;
            for row in runs.iter().map_err(db)? {
                let (_, value) = row.map_err(db)?;
                let run: Record = serde_json::from_slice(value.value())?;
                if references_dataset(&run.data, id)
                    || run.blobs.iter().any(|blob| dataset.blobs.contains(blob))
                {
                    return Ok(DeleteDataset::Referenced);
                }
            }
        }

        let removed = {
            let mut datasets = transaction
                .open_table(definition(Table::Datasets))
                .map_err(db)?;
            let removed = datasets.remove(dataset_key.as_slice()).map_err(db)?;
            removed.is_some()
        };
        if !removed {
            return Err(StoreError::Integrity("dataset disappeared during deletion"));
        }
        transaction.commit().map_err(db)?;
        super::boundary("after_commit");
        Ok(DeleteDataset::Deleted)
    }

    /// Delete a quiescent run and its durable event/evaluation children. The
    /// run's blob references cease protecting data only after the same commit.
    pub fn delete_run(&self, id: &RunId) -> super::Result<DeleteRun> {
        let _guard = self
            .writer
            .lock()
            .map_err(|_| StoreError::Integrity("writer lock poisoned"))?;
        let mut transaction = self.database.begin_write().map_err(db)?;
        transaction.set_durability(redb::Durability::Immediate);
        let mut write = Write {
            inner: transaction,
            cas: &self.cas,
        };
        let result = delete_run_in(&mut write, id, None)?;
        let transaction = write.inner;
        if result != DeleteRun::Deleted {
            return Ok(result);
        }
        transaction.commit().map_err(db)?;
        super::boundary("after_commit");
        Ok(result)
    }

    /// List or remove unreferenced regular CAS files under one writer lock.
    /// The lock prevents Store transactions and blob publication from racing
    /// the reference scan and removal.
    pub fn gc(&self, dry_run: bool) -> super::Result<GcReport> {
        let _guard = self
            .writer
            .lock()
            .map_err(|_| StoreError::Integrity("writer lock poisoned"))?;
        let mut referenced = BTreeSet::new();
        let read = self.database.begin_read().map_err(db)?;
        for kind in Table::ALL {
            let table = read.open_table(definition(kind)).map_err(db)?;
            for row in table.iter().map_err(db)? {
                let (_, value) = row.map_err(db)?;
                let record: Record = serde_json::from_slice(value.value())?;
                for blob in record.blobs {
                    self.cas.verify(&blob)?;
                    referenced.insert(blob.sha256);
                }
            }
        }
        drop(read);

        let candidates = scan_unreferenced(self, &referenced)?;
        let bytes = candidates.iter().try_fold(0_u64, |total, (_, blob)| {
            total
                .checked_add(blob.bytes)
                .ok_or(StoreError::Invalid("GC byte count overflow"))
        })?;
        if !dry_run {
            for (path, blob) in &candidates {
                self.cas.verify(blob)?;
                fs::remove_file(path)?;
                super::cas::sync_dir(path.parent().unwrap())?;
            }
        }
        Ok(GcReport {
            blobs: candidates.into_iter().map(|(_, blob)| blob).collect(),
            bytes,
        })
    }
}

/// Apply run deletion inside the caller's write transaction. This lets the
/// handler commit the deletion and its idempotency receipt as one effect.
pub(crate) fn delete_run_in(
    write: &mut Write<'_>,
    id: &RunId,
    expected_revision: Option<u64>,
) -> super::Result<DeleteRun> {
    let run_key = RecordKey::Run(id.clone()).bytes()?;
    let record = {
        let runs = write
            .inner
            .open_table(definition(Table::Runs))
            .map_err(db)?;
        let Some(value) = runs.get(run_key.as_slice()).map_err(db)? else {
            return Ok(DeleteRun::NotFound);
        };
        serde_json::from_slice::<Record>(value.value())?
    };
    let Some(run) = record.run.as_ref() else {
        return Err(StoreError::Integrity("run table record has no run state"));
    };
    if expected_revision.is_some_and(|revision| revision != run.revision) {
        return Ok(DeleteRun::RevisionConflict);
    }
    if matches!(run.state, RunState::Running | RunState::Pausing) {
        return Ok(DeleteRun::Active);
    }

    write
        .inner
        .open_table(definition(Table::Runs))
        .map_err(db)?
        .remove(run_key.as_slice())
        .map_err(db)?;
    let mut prefix = id.to_string().into_bytes();
    prefix.push(0);
    remove_children(&mut write.inner, Table::Events, &prefix)?;
    remove_children(&mut write.inner, Table::Evaluations, &prefix)?;
    Ok(DeleteRun::Deleted)
}

fn references_dataset(value: &Value, id: &DatasetId) -> bool {
    match value {
        Value::String(value) => value == &id.to_string(),
        Value::Array(values) => values.iter().any(|value| references_dataset(value, id)),
        Value::Object(values) => values.values().any(|value| references_dataset(value, id)),
        _ => false,
    }
}

fn remove_children(
    transaction: &mut redb::WriteTransaction,
    kind: Table,
    prefix: &[u8],
) -> super::Result<()> {
    let mut table = transaction.open_table(definition(kind)).map_err(db)?;
    let keys = table
        .iter()
        .map_err(db)?
        .map(|row| {
            let (key, _) = row.map_err(db)?;
            Ok::<_, StoreError>(key.value().to_vec())
        })
        .collect::<super::Result<Vec<_>>>()?
        .into_iter()
        .filter(|key| key.starts_with(prefix))
        .collect::<Vec<_>>();
    for key in keys {
        table.remove(key.as_slice()).map_err(db)?;
    }
    Ok(())
}

fn scan_unreferenced(
    store: &Store,
    referenced: &BTreeSet<[u8; 32]>,
) -> super::Result<Vec<(PathBuf, BlobRef)>> {
    let root = store.cas.home.join("cas/sha256");
    ensure_real_dir(&root)?;
    let mut candidates = BTreeMap::new();
    for first in fs::read_dir(&root)? {
        let first = first?;
        if !first.file_type()?.is_dir() || !is_shard(first.file_name().to_str()) {
            return Err(StoreError::Integrity("invalid CAS shard entry"));
        }
        for second in fs::read_dir(first.path())? {
            let second = second?;
            if !second.file_type()?.is_dir() || !is_shard(second.file_name().to_str()) {
                return Err(StoreError::Integrity("invalid CAS shard entry"));
            }
            for entry in fs::read_dir(second.path())? {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    return Err(StoreError::Integrity("non-file entry in CAS shard"));
                }
                let digest = digest_from_path(&first.path(), &second.path(), &entry.path())?;
                if referenced.contains(&digest) {
                    continue;
                }
                let blob = BlobRef {
                    sha256: digest,
                    bytes: entry.metadata()?.len(),
                };
                store.cas.verify(&blob)?;
                candidates.insert(digest, (entry.path(), blob));
            }
        }
    }
    Ok(candidates.into_values().collect())
}

fn ensure_real_dir(path: &Path) -> super::Result<()> {
    if !fs::symlink_metadata(path)?.file_type().is_dir() {
        return Err(StoreError::Integrity("CAS shard is not a real directory"));
    }
    Ok(())
}

fn is_shard(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.len() == 2 && is_lowercase_hex(value))
}

fn is_lowercase_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest_from_path(first: &Path, second: &Path, file: &Path) -> super::Result<[u8; 32]> {
    let first = first
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(StoreError::Integrity("invalid CAS shard name"))?;
    let second = second
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(StoreError::Integrity("invalid CAS shard name"))?;
    let tail = file
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(StoreError::Integrity("invalid CAS blob name"))?;
    if !is_shard(Some(first))
        || !is_shard(Some(second))
        || tail.len() != 60
        || !is_lowercase_hex(tail)
    {
        return Err(StoreError::Integrity("invalid CAS blob path"));
    }
    let hex = format!("{first}{second}{tail}");
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&hex[start..start + 2], 16)
            .map_err(|_| StoreError::Integrity("invalid CAS blob digest"))?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Cursor;
    use tempfile::TempDir;

    fn store() -> (TempDir, Store) {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }

    fn dataset() -> DatasetId {
        "ds_00112233445566778899aabb".parse().unwrap()
    }

    fn run() -> RunId {
        "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap()
    }

    fn add_dataset(store: &Store, id: &DatasetId, blobs: Vec<BlobRef>) {
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Dataset(id.clone()),
                    &Record::metadata(json!({"id": id}), blobs),
                )
            })
            .unwrap();
    }

    #[test]
    fn dataset_delete_refuses_run_reference_then_succeeds_after_run_delete() {
        let (_dir, store) = store();
        let id = dataset();
        add_dataset(&store, &id, vec![]);
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(run()),
                    &Record::run(json!({"dataset": id}), vec![], RunState::Completed, 1),
                )
            })
            .unwrap();
        assert_eq!(
            store.delete_dataset(&id).unwrap(),
            DeleteDataset::Referenced
        );
        assert!(store
            .get(&RecordKey::Dataset(id.clone()))
            .unwrap()
            .is_some());
        assert_eq!(store.delete_run(&run()).unwrap(), DeleteRun::Deleted);
        assert_eq!(store.delete_dataset(&id).unwrap(), DeleteDataset::Deleted);
        assert!(store.get(&RecordKey::Dataset(id)).unwrap().is_none());
    }

    #[test]
    fn run_delete_refuses_active_run_and_removes_quiescent_children() {
        let (_dir, store) = store();
        let id = run();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(id.clone()),
                    &Record::run(json!({}), vec![], RunState::Running, 1),
                )
            })
            .unwrap();
        assert_eq!(store.delete_run(&id).unwrap(), DeleteRun::Active);
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Event {
                        run: id.clone(),
                        sequence: 1,
                    },
                    &Record::metadata(json!({"event": true}), vec![]),
                )?;
                write.put(
                    &RecordKey::Evaluation {
                        run: id.clone(),
                        number: 1,
                    },
                    &Record::metadata(json!({"evaluation": true}), vec![]),
                )?;
                write.put(
                    &RecordKey::Run(id.clone()),
                    &Record::run(json!({}), vec![], RunState::Completed, 2),
                )
            })
            .unwrap();
        assert_eq!(store.delete_run(&id).unwrap(), DeleteRun::Deleted);
        assert!(store.get(&RecordKey::Run(id.clone())).unwrap().is_none());
        assert!(store
            .get(&RecordKey::Event {
                run: id.clone(),
                sequence: 1
            })
            .unwrap()
            .is_none());
        assert!(store
            .get(&RecordKey::Evaluation { run: id, number: 1 })
            .unwrap()
            .is_none());
    }

    #[test]
    fn gc_dry_run_matches_removal_and_keeps_run_accepted_model_blob() {
        let (_dir, store) = store();
        let orphan = store.publish_blob(Cursor::new(b"orphan blob")).unwrap();
        let accepted_model = store
            .publish_blob(Cursor::new(b"accepted model bytes"))
            .unwrap();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(run()),
                    &Record::run(
                        json!({"accepted_model": "model-v1"}),
                        vec![accepted_model.clone()],
                        RunState::Completed,
                        1,
                    ),
                )
            })
            .unwrap();
        let dry = store.gc(true).unwrap();
        assert_eq!(dry.blobs, vec![orphan.clone()]);
        assert!(store.verify_blob(&orphan).is_ok());
        assert!(store.verify_blob(&accepted_model).is_ok());

        let removed = store.gc(false).unwrap();
        assert_eq!(removed, dry);
        assert!(store.verify_blob(&orphan).is_err());
        assert!(store.verify_blob(&accepted_model).is_ok());
    }

    #[test]
    fn gc_rejects_malformed_utf8_cas_name_without_poisoning_writer() {
        let (_dir, store) = store();
        let shard = store.cas.home.join("cas/sha256/aa/bb");
        fs::create_dir_all(&shard).unwrap();
        let malformed_tail = format!("aé{}", "a".repeat(57));
        assert_eq!(malformed_tail.len(), 60);
        fs::write(shard.join(malformed_tail), b"malformed blob").unwrap();

        assert!(matches!(store.gc(true), Err(StoreError::Integrity(_))));

        let id = dataset();
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Dataset(id.clone()),
                    &Record::metadata(json!({"id": id}), vec![]),
                )
            })
            .unwrap();
        assert!(store.get(&RecordKey::Dataset(id)).unwrap().is_some());
    }
}
