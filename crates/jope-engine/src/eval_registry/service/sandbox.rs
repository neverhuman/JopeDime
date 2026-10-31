//! Evaluator isolation for original fit and query children (J24/G38, CLOVER §12.4).
//!
//! An original child evaluates generated programs, so it must not reach the
//! network, read training-control or custody credentials, interfere with other
//! processes, or write outside its own output custody. Confinement has two layers.
//!
//! 1. Spawn. Between fork and exec the service clears the child's environment to
//!    an allowlist, marks every inherited descriptor except the child's own
//!    channel close-on-exec, bounds address space, CPU time, file size, open
//!    files and core dumps, sets `no_new_privs`, and installs a seccomp filter
//!    that refuses non-local sockets, io_uring, process creation, namespaces,
//!    mounts, keyrings, cross-process memory access, and changing another
//!    process's limits, scheduling or priority. The child cannot skip it.
//! 2. Lockdown. Before it fits or predicts, the child's own verified code applies
//!    a Landlock allowlist (read-only runtime files and its own `/proc/<pid>`,
//!    writes only beneath its own ticket directory, no TCP) and a second seccomp
//!    filter that refuses every new socket, exec, further seccomp filters, and
//!    signals to other processes.
//!
//! Without Landlock (an older kernel, allowed unless `require_fs_allowlist`)
//! only the filesystem allowlist is missing: the child can then read whatever
//! its user can, including credential files and other processes' `/proc`.
//!
//! The fit child must read its unconfined parent's `/proc/<pid>/exe` before it
//! trusts the service, which Landlock forbids, so the filesystem allowlist is
//! applied by the child rather than at spawn. The service attests the lockdown
//! from `/proc/<pid>/status` (both filters present) before it consumes a start
//! token or sends query inputs. Landlock binds only the calling thread on
//! current kernels, so lockdown refuses a child holding more threads than the
//! service allows; every later thread inherits the restriction.
use super::ServiceConfig;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    ffi::CString,
    fs,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
    time::Duration,
};

#[cfg(test)]
pub(crate) mod tests;

/// Carries the service's lockdown policy to its original child.
const POLICY_ENV: &str = "JOPE_EVAL_SANDBOX";
const POLICY_SCHEMA: &str = "jope.eval_sandbox.v1";
/// The only inherited variables. Loader inputs keep the child's mapped runtime
/// identical to the service's (a preloaded library runs inside the child, so it
/// is as trusted as the service itself); the Rust ones shape diagnostics and stacks.
const INHERITED_ENV: [&str; 5] = [
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "RUST_BACKTRACE",
    "RUST_LIB_BACKTRACE",
    "RUST_MIN_STACK",
];
const ADDRESS_SPACE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const OPEN_FILES: u64 = 256;
/// A query child writes only its sealed predictions (at most 1 MB).
const QUERY_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Readable after lockdown, with the executable and every mapped file. Of
/// `/proc`, only the child's own `/proc/<pid>` and the boot id its process
/// identity reads. Nothing beneath a home directory, `/etc`, `/run` or `/tmp`.
const RUNTIME_TREES: [&str; 10] = [
    "/usr",
    "/lib",
    "/lib64",
    "/lib32",
    "/etc/ld.so.cache",
    "/proc/self",
    "/proc/sys/kernel/random/boot_id",
    "/sys/devices/system/cpu",
    "/sys/fs/cgroup",
    "/dev/urandom",
];

// Landlock ABI (include/uapi/linux/landlock.h). Only rights named here are
// handled, so a newer kernel's additional rights never refuse the service.
const LANDLOCK_CREATE_RULESET_VERSION: libc::c_uint = 1;
const LANDLOCK_RULE_PATH_BENEATH: libc::c_int = 1;
const FS_EXECUTE: u64 = 1 << 0;
const FS_WRITE_FILE: u64 = 1 << 1;
const FS_READ_FILE: u64 = 1 << 2;
const FS_READ_DIR: u64 = 1 << 3;
const FS_REMOVE_FILE: u64 = 1 << 5;
const FS_MAKE_REG: u64 = 1 << 8;
const FS_REFER: u64 = 1 << 13;
const FS_TRUNCATE: u64 = 1 << 14;
const FS_IOCTL_DEV: u64 = 1 << 15;
/// Rights a rule on a regular file (rather than a directory) may carry.
const FS_FILE: u64 = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV;
const NET_BIND_CONNECT_TCP: u64 = 0b11;
const SCOPE_ABSTRACT_UNIX_SIGNAL: u64 = 0b11;
const KNOWN_ABI: u32 = 6;

#[repr(C)]
struct RulesetAttr {
    handled_access_fs: u64,
    handled_access_net: u64,
    scoped: u64,
}
#[repr(C, packed)]
struct PathBeneathAttr {
    allowed_access: u64,
    parent_fd: i32,
}

#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH: u32 = 0xC000_003E;
#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH: u32 = 0xC000_00B7;
/// No filter is defined for other architectures; binding refuses them.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const AUDIT_ARCH: u32 = 0;
// struct seccomp_data offsets; argument words are read little-endian.
const DATA_NR: u32 = 0;
const DATA_ARCH: u32 = 4;
const DATA_ARG0_LOW: u32 = 16;
/// The kernel's classic-BPF program length limit.
const BPF_MAXINSNS: usize = 4096;

/// Evaluator isolation settings bound into the service.
#[derive(Clone, Debug)]
pub struct WorkerSandbox {
    /// Refuse to bind unless this kernel enforces the Landlock filesystem allowlist.
    pub require_fs_allowlist: bool,
    /// Threads a child may hold at lockdown. Production children lock down
    /// single-threaded; libtest runs each fixture child on a second thread.
    pub(super) threads: u64,
    /// Fixture-only directories in which a child may also replace files.
    #[cfg(test)]
    pub(super) mutable_dirs: Vec<PathBuf>,
}
impl Default for WorkerSandbox {
    fn default() -> Self {
        Self {
            require_fs_allowlist: false,
            threads: if cfg!(test) { 2 } else { 1 },
            #[cfg(test)]
            mutable_dirs: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema: String,
    /// Zero when the kernel cannot enforce Landlock and the service allowed that.
    landlock_abi: u32,
    threads: u64,
    writable: Vec<PathBuf>,
    mutable: Vec<PathBuf>,
}

/// The kernel's Landlock ABI version, or zero when Landlock is unavailable.
pub(crate) fn landlock_abi() -> u32 {
    static ABI: OnceLock<u32> = OnceLock::new();
    *ABI.get_or_init(|| {
        // SAFETY: the version query takes no attribute pointer.
        let version = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<RulesetAttr>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        u32::try_from(version).unwrap_or(0)
    })
}

fn enforce(abi: u32, sandbox: &WorkerSandbox) -> Result<u32> {
    ensure!(
        AUDIT_ARCH != 0,
        "evaluator sandbox has no seccomp filter for this architecture"
    );
    ensure!(
        abi > 0 || !sandbox.require_fs_allowlist,
        "evaluator filesystem allowlist required but Landlock is unavailable"
    );
    Ok(abi.min(KNOWN_ABI))
}

/// Refuse to bind a service whose children this host cannot confine.
pub(super) fn check(sandbox: &WorkerSandbox) -> Result<()> {
    enforce(landlock_abi(), sandbox)?;
    status_field("/proc/thread-self/status", "Seccomp_filters:")?;
    Ok(())
}

/// Spawn-time confinement, prepared before fork so the child need not allocate.
pub(super) struct Confinement {
    filter: Vec<libc::sock_filter>,
    cpu_seconds: u64,
    file_bytes: u64,
    policy: String,
    inherited_filters: u64,
}
impl Confinement {
    /// A fit child may create and write files only in its own ticket's
    /// directory, so concurrent children cannot touch each other's artifacts.
    /// The service discards an artifact larger than `max_descriptor_bytes`.
    pub(super) fn fit(config: &ServiceConfig, custody: &Path) -> Result<Self> {
        Self::new(
            config,
            config.handshake_timeout + config.fit_deadline,
            config.max_descriptor_bytes,
            vec![custody.to_path_buf()],
        )
    }
    /// A query child writes no files; it returns predictions over its channel.
    pub(super) fn query(config: &ServiceConfig, remaining: Duration) -> Result<Self> {
        Self::new(config, remaining, QUERY_FILE_BYTES, Vec::new())
    }
    fn new(
        config: &ServiceConfig,
        wall: Duration,
        file_bytes: u64,
        writable: Vec<PathBuf>,
    ) -> Result<Self> {
        let sandbox = &config.sandbox;
        let c = &config.contract;
        // CPU seconds cannot exceed wall time across the main and pool threads.
        let threads = 1
            + u64::from(c.omp_threads)
            + u64::from(c.rayon_threads)
            + u64::from(c.blas_threads)
            + u64::from(c.mkl_threads);
        let policy = Policy {
            schema: POLICY_SCHEMA.into(),
            landlock_abi: enforce(landlock_abi(), sandbox)?,
            threads: sandbox.threads,
            writable,
            #[cfg(test)]
            mutable: sandbox.mutable_dirs.clone(),
            #[cfg(not(test))]
            mutable: Vec::new(),
        };
        Ok(Self {
            filter: spawn_filter(),
            cpu_seconds: (wall.as_secs() + 1).saturating_mul(threads),
            file_bytes,
            policy: serde_json::to_string(&policy)?,
            // A forked child inherits the spawning thread's filters.
            inherited_filters: status_field("/proc/thread-self/status", "Seccomp_filters:")?,
        })
    }
    /// Replace the inherited environment with the allowlist and the policy.
    /// Call before setting the child's own variables: it clears explicit ones.
    pub(super) fn configure(&self, command: &mut Command) {
        command.env_clear();
        for name in INHERITED_ENV {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command.env(POLICY_ENV, &self.policy);
    }
    pub(super) fn attestation(&self) -> Attestation {
        Attestation {
            filters: self.inherited_filters + 2,
        }
    }
    /// Run between fork and exec: async-signal-safe system calls only. `keep`
    /// is the child's private channel and stays inherited.
    pub(super) fn enter(&self, keep: RawFd) -> std::io::Result<()> {
        let error = std::io::Error::last_os_error;
        // SAFETY: plain system calls on integers and stack values.
        unsafe {
            if libc::syscall(
                libc::SYS_close_range,
                3u32,
                u32::MAX,
                libc::CLOSE_RANGE_CLOEXEC,
            ) != 0
                || libc::fcntl(keep, libc::F_SETFD, 0) != 0
                // An oversized write fails with EFBIG instead of killing the child.
                || libc::signal(libc::SIGXFSZ, libc::SIG_IGN) == libc::SIG_ERR
            {
                return Err(error());
            }
            for (resource, value) in [
                (libc::RLIMIT_AS, ADDRESS_SPACE_BYTES),
                (libc::RLIMIT_CPU, self.cpu_seconds),
                (libc::RLIMIT_FSIZE, self.file_bytes),
                (libc::RLIMIT_NOFILE, OPEN_FILES),
                (libc::RLIMIT_CORE, 0),
            ] {
                let limit = libc::rlimit {
                    rlim_cur: value,
                    rlim_max: value,
                };
                if libc::setrlimit(resource, &limit) != 0 {
                    return Err(error());
                }
            }
        }
        match install(&self.filter, 0) {
            0 => Ok(()),
            _ => Err(error()),
        }
    }
}

/// What the service expects of a child that completed its lockdown.
#[derive(Clone, Copy)]
pub(super) struct Attestation {
    filters: u64,
}
impl Attestation {
    pub(super) fn verify(self, pid: u32) -> Result<()> {
        let path = format!("/proc/{pid}/status");
        ensure!(
            status_field(&path, "NoNewPrivs:")? == 1
                && status_field(&path, "Seccomp:")? == 2
                && status_field(&path, "Seccomp_filters:")? == self.filters,
            "original evaluator child is not locked down"
        );
        Ok(())
    }
}

/// Called by the original child's verified code before it fits or predicts.
pub(super) fn lockdown() -> Result<()> {
    let policy = std::env::var(POLICY_ENV).context("evaluator sandbox policy absent")?;
    lockdown_with(&serde_json::from_str(&policy)?)
}
fn lockdown_with(policy: &Policy) -> Result<()> {
    ensure!(
        policy.schema == POLICY_SCHEMA,
        "unknown evaluator sandbox policy"
    );
    let threads = status_field("/proc/self/status", "Threads:")?;
    ensure!(
        threads <= policy.threads,
        "evaluator lockdown allows {} thread(s); found {threads}",
        policy.threads
    );
    if policy.landlock_abi > 0 {
        ensure!(
            landlock_abi() >= policy.landlock_abi,
            "evaluator kernel lacks the service's Landlock ABI"
        );
        restrict_filesystem(policy).context("evaluator filesystem allowlist")?;
    }
    let filter = lockdown_filter(std::process::id());
    match install(&filter, libc::SECCOMP_FILTER_FLAG_TSYNC) {
        0 => Ok(()),
        -1 => Err(std::io::Error::last_os_error()).context("evaluator lockdown filter"),
        thread => anyhow::bail!("evaluator thread {thread} cannot adopt the lockdown filter"),
    }
}

fn restrict_filesystem(policy: &Policy) -> Result<()> {
    let abi = policy.landlock_abi.min(KNOWN_ABI);
    let mut handled = (1 << 13) - 1; // ABI 1: execute through make_sym
    if abi >= 2 {
        handled |= FS_REFER;
    }
    if abi >= 3 {
        handled |= FS_TRUNCATE;
    }
    if abi >= 5 {
        handled |= FS_IOCTL_DEV;
    }
    let attr = RulesetAttr {
        handled_access_fs: handled,
        // No TCP rule is added, so every TCP bind and connect is refused.
        handled_access_net: if abi >= 4 { NET_BIND_CONNECT_TCP } else { 0 },
        scoped: if abi >= 6 {
            SCOPE_ABSTRACT_UNIX_SIGNAL
        } else {
            0
        },
    };
    // SAFETY: attr outlives the call; older kernels accept the zeroed tail.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &attr as *const RulesetAttr,
            std::mem::size_of::<RulesetAttr>(),
            0u32,
        )
    };
    ensure!(
        fd >= 0,
        "Landlock ruleset: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: a successful call returns a fresh descriptor owned here.
    let ruleset = unsafe { OwnedFd::from_raw_fd(fd as RawFd) };
    let read = FS_READ_FILE | FS_READ_DIR;
    let write = read | FS_WRITE_FILE | FS_MAKE_REG | FS_TRUNCATE;
    for tree in RUNTIME_TREES {
        allow(&ruleset, Path::new(tree), read & handled, true)?;
    }
    allow(
        &ruleset,
        Path::new("/dev/null"),
        (FS_READ_FILE | FS_WRITE_FILE | FS_IOCTL_DEV) & handled,
        false,
    )?;
    for file in runtime_files()? {
        allow(&ruleset, &file, FS_READ_FILE, true)?;
    }
    for dir in &policy.writable {
        allow(&ruleset, dir, write & handled, false)?;
    }
    for dir in &policy.mutable {
        allow(&ruleset, dir, (write | FS_REMOVE_FILE) & handled, false)?;
    }
    // SAFETY: plain system calls; the ruleset descriptor is live.
    unsafe {
        ensure!(
            libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0
                && libc::syscall(libc::SYS_landlock_restrict_self, ruleset.as_raw_fd(), 0u32) == 0,
            "Landlock restriction: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

/// The executable and every file it maps, read when the child re-verifies
/// its own loaded runtime after prediction.
fn runtime_files() -> Result<BTreeSet<PathBuf>> {
    let mut files = BTreeSet::from([fs::read_link("/proc/self/exe")?]);
    for line in fs::read_to_string("/proc/self/maps")?.lines() {
        if let Some(start) = line.find('/') {
            files.insert(PathBuf::from(&line[start..]));
        }
    }
    Ok(files)
}

fn allow(ruleset: &OwnedFd, path: &Path, rights: u64, optional: bool) -> Result<()> {
    let name = CString::new(path.as_os_str().as_bytes())?;
    // SAFETY: name is NUL-terminated and outlives the call.
    let fd = unsafe { libc::open(name.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        if optional && error.kind() == std::io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(error).with_context(|| format!("allowlist {}", path.display()));
    }
    // SAFETY: a successful open returns a fresh descriptor owned here.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    ensure!(
        // SAFETY: `stat` is aligned writable storage for one result and `fd`
        // owns a live descriptor for the opened allowlist path.
        unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) } == 0,
        "allowlist {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    // SAFETY: fstat returned 0, so it initialized every field of `stat`.
    let stat = unsafe { stat.assume_init() };
    let allowed = if stat.st_mode & libc::S_IFMT == libc::S_IFDIR {
        rights
    } else {
        rights & FS_FILE
    };
    if allowed == 0 {
        return Ok(());
    }
    let rule = PathBeneathAttr {
        allowed_access: allowed,
        parent_fd: fd.as_raw_fd(),
    };
    // SAFETY: rule and both descriptors outlive the call.
    let added = unsafe {
        libc::syscall(
            libc::SYS_landlock_add_rule,
            ruleset.as_raw_fd(),
            LANDLOCK_RULE_PATH_BENEATH,
            &rule as *const PathBeneathAttr,
            0u32,
        )
    };
    ensure!(
        added == 0,
        "allowlist {}: {}",
        path.display(),
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn status_field(path: &str, name: &str) -> Result<u64> {
    let status = fs::read_to_string(path)?;
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix(name))
        .with_context(|| format!("{path} lacks {name}"))?;
    Ok(value.trim().parse()?)
}

fn spawn_filter() -> Vec<libc::sock_filter> {
    let mut refused = vec![
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_process_madvise,
        libc::SYS_pidfd_getfd,
        libc::SYS_kcmp,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_fsopen,
        libc::SYS_fsconfig,
        libc::SYS_fsmount,
        libc::SYS_fspick,
        libc::SYS_move_mount,
        libc::SYS_open_tree,
        libc::SYS_mount_setattr,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_userfaultfd,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_name_to_handle_at,
        libc::SYS_open_by_handle_at,
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_send_signal,
        libc::SYS_process_mrelease,
        // Their targets are not the first argument; nothing here needs them.
        libc::SYS_setpriority,
        libc::SYS_ioprio_set,
    ];
    #[cfg(target_arch = "x86_64")]
    refused.extend([libc::SYS_fork, libc::SYS_vfork]);
    let mut filter = Filter::new()
        // Local IPC only; the fit child still connects to the service socket.
        .guard(libc::SYS_socket, libc::BPF_JEQ, libc::AF_UNIX as u32)
        // Threads only, never another process.
        .guard(libc::SYS_clone, libc::BPF_JSET, libc::CLONE_THREAD as u32);
    // Limits, scheduling and memory placement of the calling process only
    // (pid 0); glibc's getrlimit and setrlimit are prlimit64(0, ...).
    for call in [
        libc::SYS_prlimit64,
        libc::SYS_sched_setaffinity,
        libc::SYS_sched_setscheduler,
        libc::SYS_sched_setparam,
        libc::SYS_sched_setattr,
        libc::SYS_migrate_pages,
        libc::SYS_move_pages,
    ] {
        filter = filter.guard(call, libc::BPF_JEQ, 0);
    }
    filter
        // clone3 hides its flags from seccomp; libc falls back to clone.
        .refuse(&[libc::SYS_clone3], libc::ENOSYS)
        .refuse(&refused, libc::EPERM)
        .finish()
}

fn lockdown_filter(pid: u32) -> Vec<libc::sock_filter> {
    Filter::new()
        // A child may signal only itself.
        .guard(libc::SYS_kill, libc::BPF_JEQ, pid)
        .guard(libc::SYS_tgkill, libc::BPF_JEQ, pid)
        .guard(libc::SYS_rt_sigqueueinfo, libc::BPF_JEQ, pid)
        .guard(libc::SYS_rt_tgsigqueueinfo, libc::BPF_JEQ, pid)
        // No further filter, so the attested filter count stays exact.
        .refuse_when(libc::SYS_prctl, libc::PR_SET_SECCOMP as u32)
        .refuse(
            &[
                libc::SYS_socket,
                libc::SYS_socketpair,
                libc::SYS_connect,
                libc::SYS_bind,
                libc::SYS_listen,
                libc::SYS_accept,
                libc::SYS_accept4,
                libc::SYS_execve,
                libc::SYS_execveat,
                libc::SYS_tkill,
                libc::SYS_seccomp,
            ],
            libc::EPERM,
        )
        .finish()
}

/// Sets `no_new_privs` and installs `filter`; returns the raw `seccomp` result.
fn install(filter: &[libc::sock_filter], flags: libc::c_ulong) -> libc::c_long {
    let program = libc::sock_fprog {
        len: filter.len() as libc::c_ushort,
        filter: filter.as_ptr().cast_mut(),
    };
    // SAFETY: plain system calls; program borrows filter for their duration.
    unsafe {
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            return -1;
        }
        libc::syscall(
            libc::SYS_seccomp,
            libc::SECCOMP_SET_MODE_FILTER,
            flags,
            &program as *const libc::sock_fprog,
        )
    }
}

/// A classic-BPF seccomp program: an architecture check, then per-call rules.
struct Filter(Vec<libc::sock_filter>);
impl Filter {
    fn new() -> Self {
        let mut filter = Self(vec![
            load(DATA_ARCH),
            jump(libc::BPF_JEQ, AUDIT_ARCH, 1, 0),
            ret(libc::SECCOMP_RET_KILL_PROCESS),
            load(DATA_NR),
        ]);
        // x32 calls share the x86_64 architecture tag; refuse that whole ABI.
        #[cfg(target_arch = "x86_64")]
        filter.0.extend([
            jump(libc::BPF_JGE, 0x4000_0000, 0, 1),
            ret(errno(libc::EPERM)),
        ]);
        filter
    }
    fn refuse(mut self, calls: &[libc::c_long], code: i32) -> Self {
        for &call in calls {
            self.0
                .extend([jump(libc::BPF_JEQ, call as u32, 0, 1), ret(errno(code))]);
        }
        self
    }
    /// Allow `call` only when its first argument's low word passes `test`
    /// against `value`; refuse it with EPERM otherwise.
    fn guard(mut self, call: libc::c_long, test: u32, value: u32) -> Self {
        self.0.extend([
            jump(libc::BPF_JEQ, call as u32, 0, 4),
            load(DATA_ARG0_LOW),
            jump(test, value, 0, 1),
            ret(libc::SECCOMP_RET_ALLOW),
            ret(errno(libc::EPERM)),
        ]);
        self
    }
    /// Refuse `call` with EPERM when its first argument's low word is `value`.
    fn refuse_when(mut self, call: libc::c_long, value: u32) -> Self {
        self.0.extend([
            jump(libc::BPF_JEQ, call as u32, 0, 4),
            load(DATA_ARG0_LOW),
            jump(libc::BPF_JEQ, value, 0, 1),
            ret(errno(libc::EPERM)),
            ret(libc::SECCOMP_RET_ALLOW),
        ]);
        self
    }
    fn finish(mut self) -> Vec<libc::sock_filter> {
        self.0.push(ret(libc::SECCOMP_RET_ALLOW));
        assert!(self.0.len() <= BPF_MAXINSNS, "seccomp program too long");
        self.0
    }
}
fn load(offset: u32) -> libc::sock_filter {
    libc::sock_filter {
        code: (libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16,
        jt: 0,
        jf: 0,
        k: offset,
    }
}
fn jump(test: u32, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
    libc::sock_filter {
        code: (libc::BPF_JMP | test | libc::BPF_K) as u16,
        jt,
        jf,
        k,
    }
}
fn ret(k: u32) -> libc::sock_filter {
    libc::sock_filter {
        code: (libc::BPF_RET | libc::BPF_K) as u16,
        jt: 0,
        jf: 0,
        k,
    }
}
fn errno(code: i32) -> u32 {
    libc::SECCOMP_RET_ERRNO | (code as u32 & libc::SECCOMP_RET_DATA)
}
