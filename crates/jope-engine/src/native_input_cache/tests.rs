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
// jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
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
        // SAFETY: `name` is a CString that outlives the call, so the path is NUL-terminated,
        // and `times` is the two-element timespec array utimensat reads.
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
            crate::manifest::file_hash(&p).expect_err("a replaced input was served from the cache")
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
