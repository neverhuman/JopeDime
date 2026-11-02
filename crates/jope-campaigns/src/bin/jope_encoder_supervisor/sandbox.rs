use super::*;

pub(super) const LANDLOCK_CREATE_RULESET_VERSION: u32 = 1;
pub(super) const LANDLOCK_RULE_PATH_BENEATH: u32 = 1;
pub(super) const ACCESS_EXECUTE: u64 = 1 << 0;
pub(super) const ACCESS_WRITE_FILE: u64 = 1 << 1;
pub(super) const ACCESS_READ_FILE: u64 = 1 << 2;
pub(super) const ACCESS_READ_DIR: u64 = 1 << 3;
pub(super) const ACCESS_REMOVE_DIR: u64 = 1 << 4;
pub(super) const ACCESS_REMOVE_FILE: u64 = 1 << 5;
pub(super) const ACCESS_MAKE_CHAR: u64 = 1 << 6;
pub(super) const ACCESS_MAKE_DIR: u64 = 1 << 7;
pub(super) const ACCESS_MAKE_REG: u64 = 1 << 8;
pub(super) const ACCESS_MAKE_SOCK: u64 = 1 << 9;
pub(super) const ACCESS_MAKE_FIFO: u64 = 1 << 10;
pub(super) const ACCESS_MAKE_BLOCK: u64 = 1 << 11;
pub(super) const ACCESS_MAKE_SYM: u64 = 1 << 12;
pub(super) const ACCESS_REFER: u64 = 1 << 13;
pub(super) const ACCESS_TRUNCATE: u64 = 1 << 14;

#[derive(Serialize)]
pub(super) struct SandboxPolicy<'a> {
    pub(super) schema: &'static str,
    pub(super) landlock_abi: i32,
    pub(super) runtime_read_execute_root: &'a Path,
    pub(super) input_read_root: &'a Path,
    pub(super) output_write_root: &'a Path,
    pub(super) system_read_execute_roots: &'a [PathBuf],
    pub(super) device_read_write_paths: &'a [PathBuf],
    pub(super) denied_classes: [&'static str; 5],
}

#[repr(C)]
struct LandlockRulesetAttr {
    handled_access_fs: u64,
}

#[repr(C)]
struct LandlockPathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
    _reserved: u32,
}

pub(super) struct LandlockRuleset {
    pub(super) fd: OwnedFd,
    pub(super) abi: i32,
}

impl LandlockRuleset {
    pub(super) fn new() -> Result<Self> {
        // SAFETY: LANDLOCK_CREATE_RULESET_VERSION requires a null attr and size 0, which are
        // passed; the kernel then only returns the ABI version and dereferences no memory.
        let abi = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<LandlockRulesetAttr>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        } as i32;
        ensure!(abi >= 3, "working Landlock ABI >=3 is required");
        let handled_access_fs = ACCESS_EXECUTE
            | ACCESS_WRITE_FILE
            | ACCESS_READ_FILE
            | ACCESS_READ_DIR
            | ACCESS_REMOVE_DIR
            | ACCESS_REMOVE_FILE
            | ACCESS_MAKE_CHAR
            | ACCESS_MAKE_DIR
            | ACCESS_MAKE_REG
            | ACCESS_MAKE_SOCK
            | ACCESS_MAKE_FIFO
            | ACCESS_MAKE_BLOCK
            | ACCESS_MAKE_SYM
            | ACCESS_REFER
            | ACCESS_TRUNCATE;
        let attr = LandlockRulesetAttr { handled_access_fs };
        // SAFETY: attr is a live repr(C) landlock_ruleset_attr (one u64) and its exact size is
        // passed, so the kernel reads only initialized bytes; flags are 0 and nothing is retained.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                &attr,
                std::mem::size_of::<LandlockRulesetAttr>(),
                0u32,
            )
        } as i32;
        if fd < 0 {
            return Err(std::io::Error::last_os_error()).context("create Landlock ruleset");
        }
        Ok(Self {
            // SAFETY: landlock_create_ruleset just returned this fresh nonnegative fd and no other
            // owner holds it; OwnedFd takes sole ownership and closes it exactly once.
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            abi,
        })
    }

    pub(super) fn allow_path(&self, path: &Path, requested_access: u64) -> Result<()> {
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("stat Landlock path {}", path.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "Landlock allow path is a symlink"
        );
        let access = if metadata.is_dir() {
            requested_access
        } else {
            requested_access & !(ACCESS_READ_DIR | ACCESS_REMOVE_DIR | ACCESS_MAKE_DIR)
        };
        let c_path = CString::new(path.as_os_str().as_bytes())?;
        // SAFETY: c_path is a live NUL-terminated CString borrowed for this call.
        // The c_int flags request neither O_CREAT nor O_TMPFILE (no mode argument).
        // open retains no pointer; negative results are checked before FD ownership.
        let path_fd = unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
        if path_fd < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("open Landlock path {}", path.display()));
        }
        // SAFETY: successful open returned a new nonnegative descriptor owned
        // only here. It has not been closed or wrapped elsewhere; this transfers
        // ownership exactly once to OwnedFd, which closes it after add_rule.
        let path_fd = unsafe { OwnedFd::from_raw_fd(path_fd) };
        let attr = LandlockPathBeneathAttr {
            allowed_access: access,
            parent_fd: path_fd.as_raw_fd(),
            _reserved: 0,
        };
        // SAFETY: the kernel copies just the 12 bytes of packed landlock_path_beneath_attr, which
        // match live attr's repr(C) prefix; self.fd and path_fd stay open for the call.
        let result = unsafe {
            libc::syscall(
                libc::SYS_landlock_add_rule,
                self.fd.as_raw_fd(),
                LANDLOCK_RULE_PATH_BENEATH,
                &attr,
                0u32,
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("add Landlock rule for {}", path.display()));
        }
        Ok(())
    }
}
