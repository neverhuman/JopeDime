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
    let mut status = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: status provides aligned writable statfs storage and file owns a
    // live descriptor. fstatfs initializes the output only on its checked success.
    if unsafe { libc::fstatfs(file.as_raw_fd(), status.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: the preceding fstatfs succeeded; every statfs field is initialized.
    let status = unsafe { status.assume_init() };
    Ok(status.f_type as u64)
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
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    #[test]
    fn entry_and_estimated_byte_limits_evict_and_replacement_accounts_once() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("identity");
        std::fs::write(&p, b"x").unwrap();
        let identity = Identity::of(&std::fs::metadata(&p).unwrap());
        let entry = || Entry {
            identity: identity.clone(),
            digest: [7; 32],
        };
        let key = |name: &str| Key {
            root_dev: 1,
            root_ino: 2,
            path: name.into(),
        };
        let mut cache = Cache::default();
        for name in ["a", "b", "c"] {
            cache.retain_entry(key(name), Some(entry()), 2, 4096);
        }
        assert_eq!(cache.entries.len(), 2);
        assert!(!cache.entries.contains_key(&key("a")));
        assert_eq!(cache.bytes, 514);
        cache.retain_entry(key("c"), Some(entry()), 2, 4096);
        assert_eq!(cache.bytes, 514);
        cache.retain_entry(key("d"), Some(entry()), 10, 300);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.bytes, 257);
        cache.retain_entry(key("oversized"), Some(entry()), 10, 1);
        assert_eq!(cache.bytes, 257);
        cache.retain_entry(key("d"), None, 10, 300);
        assert!(cache.entries.is_empty());
        assert_eq!(cache.bytes, 0);
        cache.retain_entry(key("a"), Some(entry()), 0, 300);
        assert!(cache.entries.is_empty());
    }
    fn reset() {
        CACHE.with(|c| *c.borrow_mut() = Cache::default());
    }
    const TMPFS_MAGIC: u64 = 0x0102_1994;
    fn assert_filesystem(dir: &Path, expected: u64, name: &str) {
        let magic = filesystem_magic(&File::open(dir).unwrap()).unwrap();
        assert_eq!(
            magic,
            expected,
            "fixture {} must be on {name} (f_type {expected:#x}); observed f_type {magic:#x}",
            dir.display()
        );
    }
    /// Reuse is qualified on ext4 only, so the reuse tests place their fixture on ext4 on
    /// purpose instead of inheriting $TMPDIR (tmpfs in the CI runner container). They use
    /// the Cargo profile directory above this test binary's deps/, and assert its
    /// filesystem, so a non-ext4 target fails loudly rather than being skipped.
    fn ext4_fixture() -> tempfile::TempDir {
        let exe = std::env::current_exe().unwrap();
        let profile = exe
            .ancestors()
            .find(|dir| dir.file_name() == Some(std::ffi::OsStr::new("deps")))
            .and_then(Path::parent)
            .unwrap_or_else(|| {
                panic!(
                    "test binary {} is not under a deps/ directory",
                    exe.display()
                )
            });
        let base = profile.join("native-input-cache-tests");
        std::fs::create_dir_all(&base).unwrap();
        let t = tempfile::tempdir_in(base.canonicalize().unwrap()).unwrap();
        assert_filesystem(t.path(), EXT4_SUPER_MAGIC, "ext4");
        t
    }
    #[test]
    fn full_first_hash_then_identity_checked_reuse_and_scope_exit() {
        reset();
        let t = ext4_fixture();
        let p = t.path().join("a");
        std::fs::write(&p, vec![7u8; 4 * 1024 * 1024]).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        assert!(hash_if_bound(&p).is_none());
        std::thread::sleep(std::time::Duration::from_millis(2100));
        {
            let _s = enter(t.path(), &BTreeMap::from([("a".into(), h.clone())])).unwrap();
            assert_eq!(hash_if_bound(&p).unwrap().unwrap(), h);
            assert_eq!(hash_if_bound(&p).unwrap().unwrap(), h);
            CACHE.with(|c| {
                assert_eq!(c.borrow().full_bytes, 4 * 1024 * 1024);
                assert_eq!(c.borrow().hits, 1);
            });
        }
        assert!(hash_if_bound(&p).is_none());
    }
    /// The documented non-ext4 policy, pinned to tmpfs on purpose (/dev/shm is tmpfs on
    /// the host and in the runner container): an old, read-only, hash-bound input that
    /// ext4 would reuse is fully hashed on every read and never enters the cache.
    #[test]
    fn unqualified_filesystem_keeps_full_hashing_for_old_read_only_inputs() {
        reset();
        let t = tempfile::tempdir_in(Path::new("/dev/shm").canonicalize().unwrap()).unwrap();
        assert_filesystem(t.path(), TMPFS_MAGIC, "tmpfs");
        let p = t.path().join("a");
        let len = 4 * 1024 * 1024;
        std::fs::write(&p, vec![7u8; len]).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2100));
        let _s = enter(t.path(), &BTreeMap::from([("a".into(), h.clone())])).unwrap();
        assert_eq!(hash_if_bound(&p).unwrap().unwrap(), h);
        assert_eq!(hash_if_bound(&p).unwrap().unwrap(), h);
        CACHE.with(|c| {
            let c = c.borrow();
            assert!(c.entries.is_empty(), "non-ext4 inputs must not seed reuse");
            assert_eq!(c.hits, 0);
            assert_eq!(c.full_bytes, 2 * len as u64);
        });
    }
    #[test]
    fn writable_files_never_cache_and_wrong_expected_hash_never_enters() {
        reset();
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("a");
        std::fs::write(&p, b"original").unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        {
            let _s = enter(t.path(), &BTreeMap::from([("a".into(), h)])).unwrap();
            hash_if_bound(&p).unwrap().unwrap();
            hash_if_bound(&p).unwrap().unwrap();
            CACHE.with(|c| assert!(c.borrow().entries.is_empty()));
        }
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let _s = enter(t.path(), &BTreeMap::from([("a".into(), "a".repeat(64))])).unwrap();
        assert!(hash_if_bound(&p).unwrap().is_err());
        CACHE.with(|c| assert!(c.borrow().entries.is_empty()));
    }
    #[test]
    fn same_size_mutation_restored_mtime_and_rename_invalidate() {
        reset();
        let t = ext4_fixture();
        let p = t.path().join("a");
        std::fs::write(&p, b"original").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        let _s = enter(t.path(), &BTreeMap::from([("a".into(), h)])).unwrap();
        hash_if_bound(&p).unwrap().unwrap();
        CACHE.with(|c| {
            assert!(
                c.borrow().entries.is_empty(),
                "recent files must not seed reuse"
            )
        });
        std::thread::sleep(std::time::Duration::from_millis(2100));
        hash_if_bound(&p).unwrap().unwrap();
        CACHE.with(|c| assert_eq!(c.borrow().entries.len(), 1));
        let old = std::fs::metadata(&p).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        std::fs::write(&p, b"mutated!").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let name = CString::new(p.as_os_str().as_encoded_bytes()).unwrap();
        let times = [
            libc::timespec {
                tv_sec: 0,
                tv_nsec: libc::UTIME_OMIT,
            },
            libc::timespec {
                tv_sec: old.mtime(),
                tv_nsec: old.mtime_nsec(),
            },
        ];
        // SAFETY: name and both timespec entries are live through utimensat;
        // the kernel borrows the two bounded elements and this private fixture path.
        assert_eq!(
            unsafe { libc::utimensat(libc::AT_FDCWD, name.as_ptr(), times.as_ptr(), 0) },
            0
        );
        assert!(hash_if_bound(&p).unwrap().is_err());
        std::fs::rename(&p, t.path().join("old")).unwrap();
        std::fs::write(&p, b"changed!").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(hash_if_bound(&p).unwrap().is_err());
    }
    #[test]
    fn root_replacement_and_descriptor_mutation_are_detected() {
        reset();
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("root");
        std::fs::create_dir(&root).unwrap();
        let p = root.join("a");
        std::fs::write(&p, b"original").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        let _s = enter(&root, &BTreeMap::from([("a".into(), h)])).unwrap();
        hash_if_bound(&p).unwrap().unwrap();
        INPUTS.with(|i| {
            let i = i.borrow();
            let inputs = i.as_ref().unwrap();
            let file = open_at(&inputs.directory, Path::new("a")).unwrap();
            let identity = Identity::of(&file.metadata().unwrap());
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
            std::fs::write(&p, b"changed!").unwrap();
            assert!(after_read(inputs, Path::new("a"), &file, &identity).is_err());
        });
        std::fs::rename(&root, t.path().join("old-root")).unwrap();
        std::fs::create_dir(&root).unwrap();
        std::fs::write(&p, b"original").unwrap();
        assert!(hash_if_bound(&p).unwrap().is_err());
    }
    /// W1 route R9 through the receiver: `manifest::file_hash` is what the
    /// worker's input verification calls inside its scope
    /// (`smartcluster_worker`). Hit, miss and resume across scopes: a resumed
    /// scope reuses a digest only for an unchanged identity, and an input
    /// replaced between scopes is re-hashed and refused - never served from
    /// the earlier scope's entry.
    #[test]
    fn receiver_hit_miss_and_resume_across_scopes_never_serve_a_replaced_input() {
        reset();
        let t = ext4_fixture();
        let root = t.path().to_path_buf();
        let p = root.join("a");
        let len = 1024 * 1024;
        std::fs::write(&p, vec![3u8; len]).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o444)).unwrap();
        let h = crate::manifest::file_hash(&p).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2100));
        let expected = BTreeMap::from([("a".into(), h.clone())]);
        let counters = || CACHE.with(|c| (c.borrow().full_bytes, c.borrow().hits));
        // Cold scope: miss (full hash) then hit.
        {
            let _s = enter(&root, &expected).unwrap();
            assert_eq!(crate::manifest::file_hash(&p).unwrap(), h);
            assert_eq!(counters(), (len as u64, 0), "first read is a full hash");
            assert_eq!(crate::manifest::file_hash(&p).unwrap(), h);
            assert_eq!(counters(), (len as u64, 1), "unchanged identity is a hit");
        }
        // Resume: a new scope over the unchanged input reuses the entry.
        {
            let _s = enter(&root, &expected).unwrap();
            assert_eq!(crate::manifest::file_hash(&p).unwrap(), h);
            assert_eq!(counters(), (len as u64, 2), "resumed scope reuses");
        }
        // Between scopes the input is replaced (new inode, same length, old
        // enough to be eligible): the resumed scope must re-hash and refuse.
        let staged = root.join("a.new");
        std::fs::write(&staged, vec![4u8; len]).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o444)).unwrap();
        std::fs::rename(&staged, &p).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2100));
        {
            let _s = enter(&root, &expected).unwrap();
            let refused = format!(
                "{:#}",
                crate::manifest::file_hash(&p)
                    .expect_err("a replaced input was served from the cache")
            );
            // A cache-served answer would be the old, matching digest; the
            // refusal comes from the full hash of the replacement.
            assert!(
                refused.contains("differs from exact manifest digest"),
                "{refused}"
            );
            assert_eq!(counters().1, 2, "no cache hit was served");
        }
        // Positive control: re-bound to the replacement's digest, it is accepted.
        let replaced = crate::manifest::file_hash(&p).unwrap();
        assert_ne!(replaced, h);
        let _s = enter(&root, &BTreeMap::from([("a".into(), replaced.clone())])).unwrap();
        assert_eq!(crate::manifest::file_hash(&p).unwrap(), replaced);
    }
    #[test]
    fn symlinks_and_fifo_refuse_before_read() {
        reset();
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("a");
        let name = CString::new(p.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: name is NUL-terminated and names this test's private fixture;
        // mkfifo takes only a borrowed pathname and integer permission bits.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let _s = enter(t.path(), &BTreeMap::from([("a".into(), "a".repeat(64))])).unwrap();
        assert!(hash_if_bound(&p).unwrap().is_err());
        std::fs::remove_file(&p).unwrap();
        std::os::unix::fs::symlink("/dev/zero", &p).unwrap();
        assert!(hash_if_bound(&p).unwrap().is_err());
    }
}
