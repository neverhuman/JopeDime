use std::{
    collections::BTreeMap,
    fs::{self, DirBuilder, File, Metadata, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
};

use nix::{
    errno::Errno,
    fcntl::{Flock, FlockArg},
    unistd::Uid,
};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::{journal_io, Digest32, FabricError};

const JOURNAL_FILE: &str = "replay-journal.postcard";
const JOURNAL_PENDING: &str = "replay-journal.pending";
const JOURNAL_LOCK: &str = "replay-journal.lock";
const MAX_REPLAY_JOURNAL_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttemptFence {
    pub attempt: u32,
    pub fence: u64,
    pub hub_epoch: u64,
    pub control_position: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct ReplayState {
    attempts: BTreeMap<Ulid, AttemptFence>,
    terminal_receipts: BTreeMap<Ulid, Digest32>,
}

pub struct ReplayJournal {
    root: PathBuf,
    root_device: u64,
    root_inode: u64,
    lock: Flock<File>,
    state: ReplayState,
}

impl ReplayJournal {
    pub fn open(root: PathBuf) -> Result<Self, FabricError> {
        if !valid_absolute_path(&root) {
            return Err(FabricError::JournalIo(
                "journal path is not absolute".to_owned(),
            ));
        }
        validate_existing_components(&root)?;
        DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&root)
            .map_err(journal_io)?;
        let root_metadata = validate_root(&root, None)?;
        let root_identity = (root_metadata.dev(), root_metadata.ino());
        let lock = open_journal_lock(&root, root_identity)?;
        let pending_path = root.join(JOURNAL_PENDING);
        if path_entry_exists(&pending_path)? {
            return Err(FabricError::JournalIo(
                "incomplete replay journal transition requires quarantine".to_owned(),
            ));
        }
        let journal_path = root.join(JOURNAL_FILE);
        let state = if path_entry_exists(&journal_path)? {
            let bytes = read_physical_state(&root, root_identity, &journal_path)?;
            postcard::from_bytes(&bytes)
                .map_err(|_| FabricError::JournalIo("invalid replay journal".to_owned()))?
        } else {
            ReplayState::default()
        };
        validate_journal_lock(&root, root_identity, &lock)?;
        Ok(Self {
            root,
            root_device: root_identity.0,
            root_inode: root_identity.1,
            lock,
            state,
        })
    }

    pub fn accept_assignment(
        &mut self,
        run_id: Ulid,
        fence: AttemptFence,
    ) -> Result<(), FabricError> {
        self.validate_ownership()?;
        if fence.attempt == 0
            || fence.fence == 0
            || fence.hub_epoch == 0
            || fence.control_position == 0
        {
            return Err(FabricError::FenceRejected);
        }
        if let Some(current) = self.state.attempts.get(&run_id) {
            if fence.hub_epoch < current.hub_epoch
                || (fence.hub_epoch == current.hub_epoch
                    && fence.control_position <= current.control_position)
                || fence.attempt < current.attempt
                || (fence.attempt == current.attempt && fence.fence != current.fence)
            {
                return Err(FabricError::ReplayedPosition);
            }
        }
        let mut next = self.state.clone();
        next.attempts.insert(run_id, fence);
        self.persist(&next)?;
        self.state = next;
        Ok(())
    }

    pub fn accept_cancel(&mut self, run_id: Ulid, fence: AttemptFence) -> Result<(), FabricError> {
        self.validate_ownership()?;
        let current = self
            .state
            .attempts
            .get_mut(&run_id)
            .ok_or(FabricError::FenceRejected)?;
        if fence.attempt != current.attempt
            || fence.fence != current.fence
            || fence.hub_epoch != current.hub_epoch
            || fence.control_position <= current.control_position
        {
            return Err(FabricError::FenceRejected);
        }
        let mut next = self.state.clone();
        next.attempts
            .get_mut(&run_id)
            .ok_or(FabricError::FenceRejected)?
            .control_position = fence.control_position;
        self.persist(&next)?;
        self.state = next;
        Ok(())
    }

    pub fn accept_terminal_receipt(
        &mut self,
        run_id: Ulid,
        receipt_digest: Digest32,
    ) -> Result<bool, FabricError> {
        self.validate_ownership()?;
        if receipt_digest == [0; 32] {
            return Err(FabricError::InvalidReceipt);
        }
        match self.state.terminal_receipts.get(&run_id) {
            Some(existing) if existing == &receipt_digest => Ok(false),
            Some(_) => Err(FabricError::ConflictingWorkerReport),
            None => {
                let mut next = self.state.clone();
                next.terminal_receipts.insert(run_id, receipt_digest);
                self.persist(&next)?;
                self.state = next;
                Ok(true)
            }
        }
    }

    fn persist(&self, state: &ReplayState) -> Result<(), FabricError> {
        self.validate_ownership()?;
        let root_identity = (self.root_device, self.root_inode);
        let bytes = postcard::to_stdvec(state)
            .map_err(|_| FabricError::JournalIo("replay journal encode failed".to_owned()))?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_REPLAY_JOURNAL_BYTES {
            return Err(FabricError::JournalIo(
                "replay journal exceeds its bounded representation".to_owned(),
            ));
        }
        let pending = self.root.join(JOURNAL_PENDING);
        let destination = self.root.join(JOURNAL_FILE);
        if path_entry_exists(&pending)? {
            return Err(FabricError::JournalIo(
                "incomplete replay journal transition requires quarantine".to_owned(),
            ));
        }
        if path_entry_exists(&destination)? {
            validate_state_file(&self.root, root_identity, &destination)?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
            .open(&pending)
            .map_err(journal_io)?;
        file.write_all(&bytes).map_err(journal_io)?;
        file.sync_all().map_err(journal_io)?;
        validate_opened_file(&pending, &file, bytes.len() as u64)?;
        validate_root(&self.root, Some(root_identity))?;
        fs::rename(&pending, &destination).map_err(journal_io)?;
        validate_state_file(&self.root, root_identity, &destination)?;
        File::open(&self.root)
            .and_then(|directory| directory.sync_all())
            .map_err(journal_io)?;
        validate_root(&self.root, Some(root_identity))?;
        validate_journal_lock(&self.root, root_identity, &self.lock)?;
        Ok(())
    }

    fn validate_ownership(&self) -> Result<(), FabricError> {
        let root_identity = (self.root_device, self.root_inode);
        validate_journal_lock(&self.root, root_identity, &self.lock)
    }
}

fn open_journal_lock(root: &Path, root_identity: (u64, u64)) -> Result<Flock<File>, FabricError> {
    validate_root(root, Some(root_identity))?;
    let path = root.join(JOURNAL_LOCK);
    let (file, created) = match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(&path)
    {
        Ok(file) => (file, true),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (
            OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
                .open(&path)
                .map_err(journal_io)?,
            false,
        ),
        Err(error) => return Err(journal_io(error)),
    };
    validate_opened_lock(&path, &file)?;
    if created {
        file.sync_all().map_err(journal_io)?;
        File::open(root)
            .and_then(|directory| directory.sync_all())
            .map_err(journal_io)?;
    }
    validate_root(root, Some(root_identity))?;
    let lock = Flock::lock(file, FlockArg::LockExclusiveNonblock).map_err(|(_, error)| {
        if error == Errno::EWOULDBLOCK {
            FabricError::JournalLocked
        } else {
            FabricError::JournalIo(format!("journal lock acquisition failed: {error}"))
        }
    })?;
    validate_journal_lock(root, root_identity, &lock)?;
    Ok(lock)
}

fn validate_journal_lock(
    root: &Path,
    root_identity: (u64, u64),
    lock: &Flock<File>,
) -> Result<(), FabricError> {
    validate_root(root, Some(root_identity))?;
    validate_opened_lock(&root.join(JOURNAL_LOCK), lock)?;
    validate_root(root, Some(root_identity))?;
    Ok(())
}

fn validate_opened_lock(path: &Path, file: &File) -> Result<(), FabricError> {
    let opened = file.metadata().map_err(journal_io)?;
    let named = fs::symlink_metadata(path).map_err(journal_io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() != 0
    {
        return Err(unsafe_journal("journal lock failed physical custody"));
    }
    Ok(())
}

fn validate_state_file(
    root: &Path,
    root_identity: (u64, u64),
    path: &Path,
) -> Result<(), FabricError> {
    validate_root(root, Some(root_identity))?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(journal_io)?;
    let length = file.metadata().map_err(journal_io)?.len();
    validate_opened_file(path, &file, length)?;
    validate_root(root, Some(root_identity))?;
    Ok(())
}

fn read_physical_state(
    root: &Path,
    root_identity: (u64, u64),
    path: &Path,
) -> Result<Vec<u8>, FabricError> {
    validate_root(root, Some(root_identity))?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(journal_io)?;
    let opened = file.metadata().map_err(journal_io)?;
    validate_opened_metadata(path, &opened, None)?;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_REPLAY_JOURNAL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(journal_io)?;
    if bytes.len() as u64 != opened.len() || bytes.len() as u64 > MAX_REPLAY_JOURNAL_BYTES {
        return Err(unsafe_journal("journal changed during bounded read"));
    }
    validate_opened_metadata(path, &opened, Some(bytes.len() as u64))?;
    validate_root(root, Some(root_identity))?;
    Ok(bytes)
}

fn validate_opened_file(path: &Path, file: &File, length: u64) -> Result<(), FabricError> {
    let opened = file.metadata().map_err(journal_io)?;
    validate_opened_metadata(path, &opened, Some(length))
}

fn validate_opened_metadata(
    path: &Path,
    opened: &Metadata,
    expected_length: Option<u64>,
) -> Result<(), FabricError> {
    let named = fs::symlink_metadata(path).map_err(journal_io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() > MAX_REPLAY_JOURNAL_BYTES
        || expected_length.is_some_and(|length| opened.len() != length)
    {
        return Err(unsafe_journal("journal file failed physical custody"));
    }
    Ok(())
}

fn validate_root(
    root: &Path,
    expected_identity: Option<(u64, u64)>,
) -> Result<Metadata, FabricError> {
    let metadata = fs::symlink_metadata(root).map_err(journal_io)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::effective().as_raw()
        || metadata.mode() & 0o777 != 0o700
        || root.canonicalize().map_err(journal_io)? != root
        || expected_identity.is_some_and(|identity| (metadata.dev(), metadata.ino()) != identity)
    {
        return Err(unsafe_journal("journal root failed physical custody"));
    }
    validate_existing_components(root)?;
    Ok(metadata)
}

fn validate_existing_components(path: &Path) -> Result<(), FabricError> {
    let mut current = PathBuf::from("/");
    for component in path.components().skip(1) {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(unsafe_journal("journal path contains a symbolic link"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(journal_io(error)),
        }
    }
    Ok(())
}

fn path_entry_exists(path: &Path) -> Result<bool, FabricError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(journal_io(error)),
    }
}

fn valid_absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|part| !matches!(part, Component::CurDir | Component::ParentDir))
}

fn unsafe_journal(message: &str) -> FabricError {
    FabricError::JournalIo(message.to_owned())
}
