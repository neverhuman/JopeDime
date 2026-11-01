//! Version1 metadata/CAS store. Only explicit BlobRef values are content links.
//! All application metadata changes in a transaction closure commit atomically.
//! Controller checkpoints/fencing, event publication and GC adapters remain their
//! owners' work; opening this store never dispatches workers or publishes events.

mod cas;
pub mod commands;
pub mod gc;
mod record;
use cas::{sync_dir, Cas};
use jope_api::RunState;
pub use record::*;
use redb::{Database, Durability, ReadableTable, TableDefinition, TableHandle};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, io::Read, path::Path, sync::Mutex};

pub const SCHEMA_VERSION: u64 = 1;
type BytesTable = TableDefinition<'static, &'static [u8], &'static [u8]>;
const META: BytesTable = TableDefinition::new("meta");
const VERSION_KEY: &[u8] = b"schema_version";
const STORE_ID_KEY: &[u8] = b"store_id";

fn definition(table: Table) -> BytesTable {
    TableDefinition::new(table.name())
}

#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Database(String),
    Json(serde_json::Error),
    SchemaVersion { found: u64, supported: u64 },
    Integrity(&'static str),
    Invalid(&'static str),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "store IO: {e}"),
            Self::Database(e) => write!(f, "store database: {e}"),
            Self::Json(e) => write!(f, "store metadata JSON: {e}"),
            Self::SchemaVersion { found, supported } => write!(
                f,
                "unsupported store schema version {found}; supported {supported}"
            ),
            Self::Integrity(e) => write!(f, "store integrity: {e}"),
            Self::Invalid(e) => write!(f, "invalid store operation: {e}"),
        }
    }
}
impl std::error::Error for StoreError {}
impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
pub type Result<T> = std::result::Result<T, StoreError>;
fn db(e: impl std::fmt::Display) -> StoreError {
    StoreError::Database(e.to_string())
}

#[cfg(feature = "failpoints")]
static BOUNDARY_COUNTS: [std::sync::atomic::AtomicU64; 11] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 11];

#[cfg(feature = "failpoints")]
fn boundary(name: &str) {
    let selected = std::env::var("JOPE_STORE_FAILPOINT").as_deref() == Ok(name);
    let trace = std::env::var("JOPE_STORE_TRACE_BOUNDARIES").as_deref() == Ok("1");
    if !selected && !trace {
        return;
    }
    let index = match name {
        "after_write" => 0,
        "after_file_sync" => 1,
        "after_hash_verify" => 2,
        "after_rename" => 3,
        "after_directory_sync" => 4,
        "after_commit" => 5,
        "before_commit" => 6,
        "before_schema_commit" => 7,
        "after_schema_commit" => 8,
        "before_recovery_commit" => 9,
        "after_recovery_commit" => 10,
        _ => return,
    };
    let ordinal = BOUNDARY_COUNTS[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if trace {
        eprintln!("JOPE_STORE_BOUNDARY:{name}:{ordinal}");
    }
    let requested = std::env::var("JOPE_STORE_FAILPOINT_OCCURRENCE")
        .map(|value| value.parse::<u64>().expect("positive failpoint occurrence"))
        .unwrap_or(1);
    assert!(requested > 0, "positive failpoint occurrence");
    if selected && ordinal == requested {
        eprintln!("JOPE_STORE_CRASH:{name}");
        // Qualification alone may stop this writer at the real boundary while
        // its supervisor sends SIGKILL to the actual background engine. Normal
        // failpoint behavior remains abort; ordinary release builds omit it.
        if std::env::var("JOPE_STORE_FAILPOINT_PARK").as_deref() == Ok("1") {
            loop {
                std::thread::park();
            }
        }
        std::process::abort();
    }
}
#[cfg(not(feature = "failpoints"))]
fn boundary(_: &str) {}

pub struct Store {
    database: Database,
    cas: Cas,
    writer: Mutex<()>,
    store_id: [u8; 16],
}

impl Store {
    pub fn open(home: impl AsRef<Path>) -> Result<Self> {
        let home = home.as_ref();
        if !home.exists() {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(home)?;
        }
        if !fs::symlink_metadata(home)?.file_type().is_dir() {
            return Err(StoreError::Integrity("home is not a real directory"));
        }
        let home = home.canonicalize()?;
        let path = home.join("state.redb");
        if let Ok(meta) = fs::symlink_metadata(&path) {
            if !meta.file_type().is_file() {
                return Err(StoreError::Integrity("state.redb is not a regular file"));
            }
        }
        let database = Database::create(&path).map_err(db)?;
        let read = database.begin_read().map_err(db)?;
        let tables: Vec<String> = read
            .list_tables()
            .map_err(db)?
            .map(|t| t.name().to_owned())
            .collect();
        let store_id = if tables.is_empty() {
            drop(read);
            let store_id = ulid::Ulid::new().to_bytes();
            let mut write = database.begin_write().map_err(db)?;
            write.set_durability(Durability::Immediate);
            {
                let mut meta = write.open_table(META).map_err(db)?;
                meta.insert(VERSION_KEY, SCHEMA_VERSION.to_le_bytes().as_slice())
                    .map_err(db)?;
                meta.insert(STORE_ID_KEY, store_id.as_slice()).map_err(db)?;
            }
            for table in Table::ALL {
                write.open_table(definition(table)).map_err(db)?;
            }
            boundary("before_schema_commit");
            write.commit().map_err(db)?;
            boundary("after_schema_commit");
            sync_dir(&home)?;
            store_id
        } else {
            let meta = read.open_table(META).map_err(db)?;
            let version = meta
                .get(VERSION_KEY)
                .map_err(db)?
                .ok_or(StoreError::Integrity("schema version missing"))?;
            let bytes: [u8; 8] = version
                .value()
                .try_into()
                .map_err(|_| StoreError::Integrity("invalid schema version bytes"))?;
            let found = u64::from_le_bytes(bytes);
            if found != SCHEMA_VERSION {
                return Err(StoreError::SchemaVersion {
                    found,
                    supported: SCHEMA_VERSION,
                });
            }
            let expected: HashSet<&str> = Table::ALL
                .iter()
                .map(|t| t.name())
                .chain(["meta"])
                .collect();
            if tables.iter().map(String::as_str).collect::<HashSet<_>>() != expected {
                return Err(StoreError::Integrity("schema table inventory mismatch"));
            }
            let id = meta
                .get(STORE_ID_KEY)
                .map_err(db)?
                .ok_or(StoreError::Integrity("store ID missing"))?;
            id.value()
                .try_into()
                .map_err(|_| StoreError::Integrity("invalid store ID bytes"))?
        };
        let store = Self {
            database,
            cas: Cas { home },
            writer: Mutex::new(()),
            store_id,
        };
        store.cas.layout()?;
        store.recover()?;
        Ok(store)
    }

    pub fn store_id(&self) -> [u8; 16] {
        self.store_id
    }

    pub fn publish_blob(&self, data: impl Read) -> Result<BlobRef> {
        let _guard = self
            .writer
            .lock()
            .map_err(|_| StoreError::Integrity("writer lock poisoned"))?;
        self.cas.publish(data)
    }

    pub fn verify_blob(&self, reference: &BlobRef) -> Result<()> {
        self.cas.verify(reference)
    }

    /// A verified immutable file for the worker's existing FIT matrix mapping.
    pub(crate) fn verified_blob_path(&self, reference: &BlobRef) -> Result<std::path::PathBuf> {
        self.cas.verify(reference)?;
        Ok(self.cas.path(reference))
    }

    /// Read an immutable CAS blob and verify its complete content identity on the bytes returned.
    pub fn read_blob(&self, reference: &BlobRef) -> Result<Vec<u8>> {
        let digest: String = reference
            .sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let path = self
            .cas
            .home
            .join("cas/sha256")
            .join(&digest[..2])
            .join(&digest[2..4])
            .join(&digest[4..]);
        for parent in [
            self.cas.home.join("cas"),
            self.cas.home.join("cas/sha256"),
            path.parent().unwrap().parent().unwrap().to_path_buf(),
            path.parent().unwrap().to_path_buf(),
        ] {
            if !fs::symlink_metadata(parent)?.file_type().is_dir() {
                return Err(StoreError::Integrity("CAS parent is not a real directory"));
            }
        }
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.file_type().is_file() || metadata.len() != reference.bytes {
            return Err(StoreError::Integrity("blob hash/size mismatch"));
        }
        let bytes = fs::read(path)?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if bytes.len() as u64 != reference.bytes || digest != reference.sha256 {
            return Err(StoreError::Integrity("blob hash/size mismatch"));
        }
        Ok(bytes)
    }

    /// Record references are verified before insertion. A closure error aborts metadata.
    /// A commit IO error has an unknown outcome; reopen/read back before retrying.
    pub fn transaction<T>(&self, action: impl FnOnce(&mut Write<'_>) -> Result<T>) -> Result<T> {
        let _guard = self
            .writer
            .lock()
            .map_err(|_| StoreError::Integrity("writer lock poisoned"))?;
        let mut inner = self.database.begin_write().map_err(db)?;
        inner.set_durability(Durability::Immediate);
        let mut write = Write {
            inner,
            cas: &self.cas,
        };
        let result = action(&mut write)?;
        boundary("before_commit");
        write.inner.commit().map_err(db)?;
        boundary("after_commit");
        Ok(result)
    }

    pub fn get(&self, key: &RecordKey) -> Result<Option<Record>> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(definition(key.table())).map_err(db)?;
        let value = table.get(key.bytes()?.as_slice()).map_err(db)?;
        value
            .map(|v| serde_json::from_slice(v.value()).map_err(StoreError::from))
            .transpose()
    }

    /// Exclusive, bounded dataset-ID traversal in metadata key order.
    pub(crate) fn dataset_ids_after(
        &self,
        after: Option<&jope_api::ids::DatasetId>,
        limit: usize,
    ) -> Result<Vec<jope_api::ids::DatasetId>> {
        if !(1..=257).contains(&limit) {
            return Err(StoreError::Invalid("dataset page size"));
        }
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(definition(Table::Datasets)).map_err(db)?;
        let start = after.map(ToString::to_string).unwrap_or_default();
        let mut ids = Vec::new();
        for row in table.range::<&[u8]>(start.as_bytes()..).map_err(db)? {
            let (key, _) = row.map_err(db)?;
            if key.value() == start.as_bytes() {
                continue;
            }
            let id = std::str::from_utf8(key.value())
                .map_err(|_| StoreError::Integrity("invalid dataset key"))?
                .parse()
                .map_err(|_| StoreError::Integrity("invalid dataset ID"))?;
            ids.push(id);
            if ids.len() == limit {
                break;
            }
        }
        Ok(ids)
    }

    pub(crate) fn run_ids_after(
        &self,
        after: Option<&jope_api::ids::RunId>,
        limit: usize,
    ) -> Result<Vec<jope_api::ids::RunId>> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(definition(Table::Runs)).map_err(db)?;
        let start = after.map(ToString::to_string).unwrap_or_default();
        let mut ids = Vec::new();
        for row in table.range::<&[u8]>(start.as_bytes()..).map_err(db)? {
            let (key, _) = row.map_err(db)?;
            if key.value() == start.as_bytes() {
                continue;
            }
            let id = std::str::from_utf8(key.value())
                .map_err(|_| StoreError::Integrity("invalid run key"))?
                .parse()
                .map_err(|_| StoreError::Integrity("invalid run ID"))?;
            ids.push(id);
            if ids.len() == limit {
                break;
            }
        }
        Ok(ids)
    }

    fn recover(&self) -> Result<()> {
        let read = self.database.begin_read().map_err(db)?;
        let mut verified = HashSet::new();
        let mut interrupted = Vec::new();
        for kind in Table::ALL {
            let table = read.open_table(definition(kind)).map_err(db)?;
            for row in table.iter().map_err(db)? {
                let (key, value) = row.map_err(db)?;
                let record: Record = serde_json::from_slice(value.value())?;
                validate(kind, &record)?;
                for reference in &record.blobs {
                    if verified.insert(reference.clone()) {
                        self.cas.verify(reference)?;
                    }
                }
                if record
                    .run
                    .as_ref()
                    .is_some_and(|r| matches!(r.state, RunState::Running | RunState::Pausing))
                {
                    interrupted.push(key.value().to_vec());
                }
            }
        }
        drop(read);
        // Preserve staging on reference corruption; only a verified store clears it.
        self.cas.clear_staging()?;
        if !interrupted.is_empty() {
            let mut write = self.database.begin_write().map_err(db)?;
            write.set_durability(Durability::Immediate);
            {
                let mut table = write.open_table(definition(Table::Runs)).map_err(db)?;
                for key in interrupted {
                    let mut record: Record = {
                        let value = table
                            .get(key.as_slice())
                            .map_err(db)?
                            .ok_or(StoreError::Integrity("recovery run disappeared"))?;
                        serde_json::from_slice(value.value())?
                    };
                    let run = record.run.as_mut().unwrap();
                    run.state = RunState::Paused;
                    run.interrupted = true;
                    run.revision = run
                        .revision
                        .checked_add(1)
                        .ok_or(StoreError::Integrity("recovery revision exhausted"))?;
                    table
                        .insert(key.as_slice(), serde_json::to_vec(&record)?.as_slice())
                        .map_err(db)?;
                }
            }
            boundary("before_recovery_commit");
            write.commit().map_err(db)?;
            boundary("after_recovery_commit");
        }
        Ok(())
    }
}

fn validate(table: Table, record: &Record) -> Result<()> {
    if (table == Table::Runs) != record.run.is_some()
        || record.run.as_ref().is_some_and(|run| run.revision == 0)
    {
        return Err(StoreError::Invalid(
            "run table requires positive revision/state; other tables have no run metadata",
        ));
    }
    Ok(())
}

/// An atomic write view; no raw database/table handles escape the wrapper.
pub struct Write<'a> {
    inner: redb::WriteTransaction,
    cas: &'a Cas,
}

impl Write<'_> {
    pub fn get(&self, key: &RecordKey) -> Result<Option<Record>> {
        let table = self.inner.open_table(definition(key.table())).map_err(db)?;
        let value = table.get(key.bytes()?.as_slice()).map_err(db)?;
        value
            .map(|v| serde_json::from_slice(v.value()).map_err(StoreError::from))
            .transpose()
    }

    pub fn put(&mut self, key: &RecordKey, record: &Record) -> Result<()> {
        validate(key.table(), record)?;
        for reference in &record.blobs {
            self.cas.verify(reference)?;
        }
        let encoded = serde_json::to_vec(record)?;
        self.inner
            .open_table(definition(key.table()))
            .map_err(db)?
            .insert(key.bytes()?.as_slice(), encoded.as_slice())
            .map_err(db)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod blob_read_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_blob_returns_exact_bytes_and_refuses_identity_mismatch_or_corruption() {
        let root = tempfile::tempdir().unwrap();
        let store = Store::open(root.path()).unwrap();
        let expected = b"stored model object bytes";
        let reference = store.publish_blob(Cursor::new(expected)).unwrap();
        assert_eq!(store.read_blob(&reference).unwrap(), expected);

        let wrong_size = BlobRef {
            sha256: reference.sha256,
            bytes: reference.bytes + 1,
        };
        assert!(matches!(
            store.read_blob(&wrong_size),
            Err(StoreError::Integrity(_))
        ));

        let digest: String = reference
            .sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let path = root
            .path()
            .join("cas/sha256")
            .join(&digest[..2])
            .join(&digest[2..4])
            .join(&digest[4..]);
        fs::write(&path, b"stored model object bytex").unwrap();
        assert!(matches!(
            store.read_blob(&reference),
            Err(StoreError::Integrity(_))
        ));
    }
}
