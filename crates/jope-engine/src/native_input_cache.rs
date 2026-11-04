//! Process-local reuse for explicitly hash-bound native immutable inputs only.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::CString,
    fs::{File, Metadata},
    io::Read,
    marker::PhantomData,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Component, Path, PathBuf},
    rc::Rc,
};
const MAX_ENTRIES: usize = 262_144;
// Estimated entry bookkeeping, not an allocator or kernel memory quota.
const MAX_BYTES: usize = 128 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
    mode: u32,
    uid: u32,
    gid: u32,
    links: u64,
}
impl Identity {
    fn of(m: &Metadata) -> Self {
        Self {
            dev: m.dev(),
            ino: m.ino(),
            len: m.len(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
            links: m.nlink(),
        }
    }
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    root_dev: u64,
    root_ino: u64,
    path: PathBuf,
}
struct Entry {
    identity: Identity,
    digest: [u8; 32],
}
#[derive(Default)]
struct Cache {
    entries: BTreeMap<Key, Entry>,
    bytes: usize,
    full_bytes: u64,
    hits: u64,
}
impl Cache {
    fn retain_entry(
        &mut self,
        key: Key,
        entry: Option<Entry>,
        max_entries: usize,
        max_estimated_bytes: usize,
    ) {
        let cost = 256 + key.path.as_os_str().len();
        if self.entries.remove(&key).is_some() {
            self.bytes -= cost;
        }
        let Some(entry) = entry else {
            return;
        };
        if max_entries == 0 || cost > max_estimated_bytes {
            return;
        }
        while self.entries.len() >= max_entries || self.bytes > max_estimated_bytes - cost {
            let Some((old, _)) = self.entries.pop_first() else {
                break;
            };
            self.bytes -= 256 + old.path.as_os_str().len();
        }
        self.bytes += cost;
        self.entries.insert(key, entry);
    }
}
struct Inputs {
    root: PathBuf,
    directory: File,
    root_identity: (u64, u64),
    expected: BTreeMap<PathBuf, String>,
}
thread_local! {static INPUTS:RefCell<Option<Inputs>>=const{RefCell::new(None)};static CACHE:RefCell<Cache>=RefCell::new(Cache::default());}
/// Not Send: exact admission context must remain on its original verifier thread.
pub(crate) struct Scope {
    previous: Option<Inputs>,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        INPUTS.with(|i| *i.borrow_mut() = self.previous.take());
    }
}
fn open_at(directory: &File, relative: &Path) -> Result<File> {
    crate::manifest::relative(relative)?;
    let mut parent = directory.try_clone()?;
    let parts: Vec<_> = relative.components().collect();
    for (i, part) in parts.iter().enumerate() {
        let Component::Normal(name) = part else {
            anyhow::bail!("unsafe native input component")
        };
        let name = CString::new(name.as_encoded_bytes())?;
        let flags = libc::O_RDONLY
            | libc::O_CLOEXEC
            | libc::O_NOFOLLOW
            | libc::O_NONBLOCK
            | if i + 1 < parts.len() {
                libc::O_DIRECTORY
            } else {
                0
            };
        // SAFETY: parent owns the live directory descriptor, name is one
        // validated NUL-terminated component, and no create flag needs a mode.
        // NOFOLLOW rejects symlinks and NONBLOCK prevents a FIFO open from hanging.
        let fd = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: successful openat returned this fresh descriptor. Transfer
        // ownership once; assignment also closes the previous directory owner.
        parent = unsafe { File::from_raw_fd(fd) };
    }
    Ok(parent)
}
pub(crate) fn enter(root: &Path, expected: &BTreeMap<PathBuf, String>) -> Result<Scope> {
    ensure!(
        root.is_absolute() && root.canonicalize()? == root,
        "native cache requires physical absolute input root"
    );
    let slash = File::open("/")?;
    let directory = open_at(&slash, root.strip_prefix("/")?)?;
    ensure!(
        directory.metadata()?.is_dir(),
        "native cache root is not directory"
    );
    for (path, h) in expected {
        crate::manifest::relative(path)?;
        ensure!(
            h.len() == 64
                && h.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "native cache expected digest malformed"
        );
    }
    let m = directory.metadata()?;
    let input = Inputs {
        root: root.into(),
        directory,
        root_identity: (m.dev(), m.ino()),
        expected: expected.clone(),
    };
    let previous = INPUTS.with(|i| i.borrow_mut().replace(input));
    Ok(Scope {
        previous,
        _thread: PhantomData,
    })
}
fn after_read(inputs: &Inputs, path: &Path, file: &File, identity: &Identity) -> Result<()> {
    let named_root = open_at(&File::open("/")?, inputs.root.strip_prefix("/")?)?;
    let root_metadata = named_root.metadata()?;
    ensure!(
        (root_metadata.dev(), root_metadata.ino()) == inputs.root_identity,
        "native input root pathname replaced while verifying"
    );
    ensure!(
        Identity::of(&file.metadata()?) == *identity,
        "native input changed while hashing"
    );
    let named = open_at(&inputs.directory, path)?;
    ensure!(
        Identity::of(&named.metadata()?) == *identity,
        "native input pathname replaced while verifying"
    );
    Ok(())
}
/// fstatfs(2) `f_type` of ext4, the only filesystem qualified for reuse below.
const EXT4_SUPER_MAGIC: u64 = 0xef53;
/// fstatfs(2) `f_type` of an open file. The field is a signed word whose width
/// depends on the target, so a magic with its top bit set (btrfs, 0x9123683E, on a
/// 32-bit target) sign-extends here. Only equality with the ext4 and tmpfs magics,
/// both below 2^31, is ever tested, and those compare correctly on every target.
fn filesystem_magic(file: &File) -> Result<u64> {
    let status = nix::sys::statfs::fstatfs(file)?;
    Ok(status.filesystem_type().0 as u64)
}
// ext4's metadata clock can repeat within a scheduler tick. Never seed reuse
// from a recent timestamp: a same-tick write followed by mtime restoration can
// otherwise leave the complete fingerprint unchanged. Other filesystems keep
// full hashing until their timestamp behavior is separately qualified.
fn eligible_for_reuse(file: &File, metadata: &Metadata) -> Result<bool> {
    let magic = filesystem_magic(file)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let Some(cutoff) = now.checked_sub(2) else {
        return Ok(false);
    };
    Ok(magic == EXT4_SUPER_MAGIC
        && metadata.mode() & 0o222 == 0
        && metadata.ctime() >= 0
        && metadata.mtime() >= 0
        && metadata.ctime() as u64 <= cutoff
        && metadata.mtime() as u64 <= cutoff)
}
fn bound_hash(inputs: &Inputs, path: &Path, expected: &str) -> Result<String> {
    let mut file = open_at(&inputs.directory, path)?;
    let m = file.metadata()?;
    ensure!(m.is_file(), "native input is not regular");
    let identity = Identity::of(&m);
    let key = Key {
        root_dev: inputs.root_identity.0,
        root_ino: inputs.root_identity.1,
        path: path.into(),
    };
    let immutable = eligible_for_reuse(&file, &m)?;
    let cached = CACHE.with(|c| {
        let c = c.borrow();
        c.entries
            .get(&key)
            .filter(|e| immutable && e.identity == identity)
            .map(|e| e.digest)
    });
    if let Some(digest) = cached {
        after_read(inputs, path, &file, &identity)?;
        ensure!(
            hex::encode(digest) == expected,
            "cached native input differs from manifest"
        );
        CACHE.with(|c| c.borrow_mut().hits += 1);
        return Ok(hex::encode(digest));
    }
    let mut sha = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes = bytes
            .checked_add(n as u64)
            .context("native input length overflow")?;
        ensure!(bytes <= identity.len, "native input grew during hash");
        sha.update(&buffer[..n]);
    }
    ensure!(bytes == identity.len, "native input truncated during hash");
    after_read(inputs, path, &file, &identity)?;
    let digest: [u8; 32] = sha.finalize().into();
    let actual = hex::encode(digest);
    ensure!(
        actual == expected,
        "native input differs from exact manifest digest"
    );
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.full_bytes = c.full_bytes.saturating_add(bytes);
        c.retain_entry(
            key,
            immutable.then_some(Entry { identity, digest }),
            MAX_ENTRIES,
            MAX_BYTES,
        );
    });
    Ok(actual)
}
/// None means the original general-purpose hashing path remains authoritative.
pub(crate) fn hash_if_bound(path: &Path) -> Option<Result<String>> {
    INPUTS.with(|i| {
        let i = i.borrow();
        let inputs = i.as_ref()?;
        let relative = path.strip_prefix(&inputs.root).ok()?;
        let expected = inputs.expected.get(relative)?;
        Some(bound_hash(inputs, relative, expected))
    })
}
#[cfg(test)]
#[path = "native_input_cache/tests.rs"]
mod tests;
