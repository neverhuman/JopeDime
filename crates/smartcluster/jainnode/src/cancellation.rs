use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex as StdMutex},
};

use async_trait::async_trait;
use jain_smartcluster_fabric::{sha256, valid_identifier, Digest32};
use nix::unistd::Uid;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify};

use crate::{CancellationPort, DockerEnginePort, ExecutionError};

const CANCELLATION_SCHEMA: u16 = 1;
const MAX_CANCELLATION_RECORD_BYTES: usize = 4096;
const MAX_CANCELLATION_RECORDS: usize = 100_000;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CancellationKey {
    run_id: String,
    attempt: u32,
    fence: u64,
}

impl CancellationKey {
    fn new(run_id: &str, attempt: u32, fence: u64) -> Result<Self, ExecutionError> {
        if !valid_identifier(run_id) || attempt == 0 || fence == 0 {
            return Err(ExecutionError::CancellationState);
        }
        Ok(Self {
            run_id: run_id.to_owned(),
            attempt,
            fence,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CancellationRecord {
    schema_version: u16,
    run_id: String,
    attempt: u32,
    fence: u64,
}

impl CancellationRecord {
    fn from_key(key: &CancellationKey) -> Self {
        Self {
            schema_version: CANCELLATION_SCHEMA,
            run_id: key.run_id.clone(),
            attempt: key.attempt,
            fence: key.fence,
        }
    }

    fn validate(&self) -> Result<CancellationKey, ExecutionError> {
        if self.schema_version != CANCELLATION_SCHEMA {
            return Err(ExecutionError::CancellationState);
        }
        CancellationKey::new(&self.run_id, self.attempt, self.fence)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StartState {
    Ready,
    Started,
    Cancelled,
}

struct CancellationGate {
    state: Mutex<StartState>,
    notify: Notify,
}

impl CancellationGate {
    fn new(state: StartState) -> Self {
        Self {
            state: Mutex::new(state),
            notify: Notify::new(),
        }
    }
}

/// Durable, fence-exact cancellation state shared by control and executor paths.
///
/// A cancellation is accepted only after its create-exclusive record and the
/// containing directory are synced. The same per-attempt gate serializes that
/// acceptance with release of executable bytes through `DockerEnginePort::start`.
pub struct DurableCancellationRegistry {
    root: PathBuf,
    gates: StdMutex<BTreeMap<CancellationKey, Arc<CancellationGate>>>,
}

impl DurableCancellationRegistry {
    pub fn open(root: PathBuf) -> Result<Self, ExecutionError> {
        validate_root(&root)?;
        validate_existing_records(&root)?;
        Ok(Self {
            root,
            gates: StdMutex::new(BTreeMap::new()),
        })
    }

    /// Accept one exact cancellation. Exact retries are idempotent; a distinct
    /// attempt or fence has an independent durable record.
    pub async fn accept(
        &self,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<(), ExecutionError> {
        let key = CancellationKey::new(run_id, attempt, fence)?;
        let gate = self.gate(&key)?;
        let mut state = gate.state.lock().await;
        if *state == StartState::Cancelled {
            return self.persist(&key);
        }
        self.persist(&key)?;
        *state = StartState::Cancelled;
        gate.notify.notify_waiters();
        Ok(())
    }

    fn gate(&self, key: &CancellationKey) -> Result<Arc<CancellationGate>, ExecutionError> {
        let mut gates = self
            .gates
            .lock()
            .map_err(|_| ExecutionError::CancellationState)?;
        if let Some(gate) = gates.get(key) {
            return Ok(Arc::clone(gate));
        }
        let state = match read_record_if_present(&self.record_path(key))? {
            Some(record) if record.validate()? == *key => StartState::Cancelled,
            Some(_) => return Err(ExecutionError::CancellationState),
            None => StartState::Ready,
        };
        let gate = Arc::new(CancellationGate::new(state));
        gates.insert(key.clone(), Arc::clone(&gate));
        Ok(gate)
    }

    fn persist(&self, key: &CancellationKey) -> Result<(), ExecutionError> {
        let record = CancellationRecord::from_key(key);
        let bytes = postcard::to_stdvec(&record).map_err(|_| ExecutionError::CancellationState)?;
        if bytes.is_empty() || bytes.len() > MAX_CANCELLATION_RECORD_BYTES {
            return Err(ExecutionError::CancellationState);
        }
        let destination = self.record_path(key);
        if let Some(existing) = read_record_if_present(&destination)? {
            return (existing == record)
                .then_some(())
                .ok_or(ExecutionError::CancellationState);
        }
        let pending = destination.with_extension("pending");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
            .open(&pending)
            .map_err(|_| ExecutionError::CancellationState)?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| ExecutionError::CancellationState)?;
        fs::rename(&pending, &destination).map_err(|_| ExecutionError::CancellationState)?;
        File::open(&self.root)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| ExecutionError::CancellationState)
    }

    fn record_path(&self, key: &CancellationKey) -> PathBuf {
        let record = CancellationRecord::from_key(key);
        let bytes = postcard::to_stdvec(&record).expect("bounded cancellation record serializes");
        self.root
            .join(format!("cancel-{}.record", hex_digest(sha256(&bytes))))
    }
}

#[async_trait]
impl CancellationPort for DurableCancellationRegistry {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<(), ExecutionError> {
        let key = CancellationKey::new(run_id, attempt, fence)?;
        let gate = self.gate(&key)?;
        let mut state = gate.state.lock().await;
        match *state {
            StartState::Cancelled => Err(ExecutionError::Cancelled),
            StartState::Ready | StartState::Started => {
                docker.start(container_id).await?;
                *state = StartState::Started;
                Ok(())
            }
        }
    }

    async fn cancelled(&self, run_id: &str, attempt: u32, fence: u64) {
        let Ok(key) = CancellationKey::new(run_id, attempt, fence) else {
            return;
        };
        let Ok(gate) = self.gate(&key) else {
            return;
        };
        loop {
            let notified = gate.notify.notified();
            if *gate.state.lock().await == StartState::Cancelled {
                return;
            }
            notified.await;
        }
    }
}

fn validate_root(root: &Path) -> Result<(), ExecutionError> {
    if !root.is_absolute()
        || root
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
    {
        return Err(ExecutionError::CancellationState);
    }
    let metadata = fs::symlink_metadata(root).map_err(|_| ExecutionError::CancellationState)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::effective().as_raw()
        || metadata.mode() & 0o777 != 0o700
        || root
            .canonicalize()
            .map_err(|_| ExecutionError::CancellationState)?
            != root
    {
        return Err(ExecutionError::CancellationState);
    }
    let mut current = PathBuf::from("/");
    for component in root.components().skip(1) {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)
            .map_err(|_| ExecutionError::CancellationState)?
            .file_type()
            .is_symlink()
        {
            return Err(ExecutionError::CancellationState);
        }
    }
    Ok(())
}

fn validate_existing_records(root: &Path) -> Result<(), ExecutionError> {
    let mut entries = fs::read_dir(root)
        .map_err(|_| ExecutionError::CancellationState)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ExecutionError::CancellationState)?;
    if entries.len() > MAX_CANCELLATION_RECORDS {
        return Err(ExecutionError::CancellationState);
    }
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| ExecutionError::CancellationState)?;
        let digest = name
            .strip_prefix("cancel-")
            .and_then(|value| value.strip_suffix(".record"))
            .filter(|value| {
                value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
            })
            .ok_or(ExecutionError::CancellationState)?;
        let (record, bytes) = read_record(&path)?;
        record.validate()?;
        if hex_digest(sha256(&bytes)) != digest {
            return Err(ExecutionError::CancellationState);
        }
    }
    Ok(())
}

fn read_record_if_present(path: &Path) -> Result<Option<CancellationRecord>, ExecutionError> {
    match fs::symlink_metadata(path) {
        Ok(_) => read_record(path).map(|(record, _)| Some(record)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(ExecutionError::CancellationState),
    }
}

fn read_record(path: &Path) -> Result<(CancellationRecord, Vec<u8>), ExecutionError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| ExecutionError::CancellationState)?;
    let opened = file
        .metadata()
        .map_err(|_| ExecutionError::CancellationState)?;
    let named = fs::symlink_metadata(path).map_err(|_| ExecutionError::CancellationState)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() as usize > MAX_CANCELLATION_RECORD_BYTES
    {
        return Err(ExecutionError::CancellationState);
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take((MAX_CANCELLATION_RECORD_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| ExecutionError::CancellationState)?;
    if bytes.is_empty() || bytes.len() > MAX_CANCELLATION_RECORD_BYTES {
        return Err(ExecutionError::CancellationState);
    }
    let record = postcard::from_bytes(&bytes).map_err(|_| ExecutionError::CancellationState)?;
    Ok((record, bytes))
}

fn hex_digest(digest: Digest32) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use tokio::sync::Notify;

    use super::*;
    use crate::{docker::DockerError, DockerLogEvent, DockerRunSpec, ExitStatus};

    #[derive(Default)]
    struct DockerMock {
        starts: AtomicUsize,
    }

    #[async_trait]
    impl DockerEnginePort for DockerMock {
        async fn create(&self, _: &DockerRunSpec) -> Result<String, DockerError> {
            unreachable!()
        }

        async fn start(&self, _: &str) -> Result<(), DockerError> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn logs(
            &self,
            _: &str,
        ) -> Result<tokio::sync::mpsc::Receiver<Result<DockerLogEvent, DockerError>>, DockerError>
        {
            unreachable!()
        }

        async fn wait(&self, _: &str) -> Result<ExitStatus, DockerError> {
            unreachable!()
        }

        async fn kill(&self, _: &str) -> Result<(), DockerError> {
            unreachable!()
        }

        async fn remove(&self, _: &str) -> Result<(), DockerError> {
            unreachable!()
        }
    }

    fn registry_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        root
    }

    #[tokio::test]
    async fn accepted_before_start_is_durable_and_exactly_fenced() {
        let root = registry_root();
        let registry = DurableCancellationRegistry::open(root.path().to_path_buf()).unwrap();
        registry.accept("run-1", 1, 7).await.unwrap();
        registry.accept("run-1", 1, 7).await.unwrap();
        let docker = DockerMock::default();
        assert!(matches!(
            registry
                .start_unless_cancelled(&docker, "container-1", "run-1", 1, 7)
                .await,
            Err(ExecutionError::Cancelled)
        ));
        assert_eq!(docker.starts.load(Ordering::SeqCst), 0);

        let reopened = DurableCancellationRegistry::open(root.path().to_path_buf()).unwrap();
        assert!(matches!(
            reopened
                .start_unless_cancelled(&docker, "container-1", "run-1", 1, 7)
                .await,
            Err(ExecutionError::Cancelled)
        ));
        reopened
            .start_unless_cancelled(&docker, "container-2", "run-1", 1, 8)
            .await
            .unwrap();
        reopened
            .start_unless_cancelled(&docker, "container-3", "run-1", 1, 8)
            .await
            .unwrap();
        assert_eq!(docker.starts.load(Ordering::SeqCst), 2);

        let record = fs::read_dir(root.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(&record, b"tampered").unwrap();
        assert!(registry.accept("run-1", 1, 7).await.is_err());
    }

    #[tokio::test]
    async fn cancellation_after_start_wakes_the_inflight_waiter() {
        let root = registry_root();
        let registry =
            Arc::new(DurableCancellationRegistry::open(root.path().to_path_buf()).unwrap());
        let docker = DockerMock::default();
        registry
            .start_unless_cancelled(&docker, "container-1", "run-2", 3, 9)
            .await
            .unwrap();
        let waiting = Arc::clone(&registry);
        let completed = Arc::new(Notify::new());
        let completed_task = Arc::clone(&completed);
        let task = tokio::spawn(async move {
            waiting.cancelled("run-2", 3, 9).await;
            completed_task.notify_one();
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        registry.accept("run-2", 3, 9).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), completed.notified())
            .await
            .unwrap();
        task.await.unwrap();
    }

    #[tokio::test]
    async fn corrupt_pending_and_unsafe_roots_fail_closed() {
        let root = registry_root();
        fs::write(root.path().join("cancel-corrupt.pending"), b"partial").unwrap();
        assert!(DurableCancellationRegistry::open(root.path().to_path_buf()).is_err());

        let unsafe_root = registry_root();
        fs::set_permissions(unsafe_root.path(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(DurableCancellationRegistry::open(unsafe_root.path().to_path_buf()).is_err());
    }
}
