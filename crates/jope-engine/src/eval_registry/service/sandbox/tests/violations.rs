use super::*;

pub(super) const PROBE_ENV: &str = "JOPE_SANDBOX_PROBE";
pub(super) const PROBE_TEST: &str = "eval_registry::service::sandbox::tests::probe_child";
pub(super) const REPORT: &str = "SANDBOX_PROBE ";

/// Records a violation unless `result` is a failure carrying `expected` errno.
fn refused(what: &str, result: i64, expected: i32, found: &mut Vec<String>) {
    let errno = std::io::Error::last_os_error().raw_os_error();
    if result >= 0 {
        found.push(format!("{what} succeeded"));
    } else if errno != Some(expected) {
        found.push(format!(
            "{what} failed with errno {errno:?}, not {expected}"
        ));
    }
}
pub(super) fn socket(family: i32, kind: i32) -> i64 {
    // SAFETY: plain system calls; a created socket is closed at once.
    unsafe {
        let fd = libc::socket(family, kind | libc::SOCK_CLOEXEC, 0);
        if fd >= 0 {
            libc::close(fd);
        }
        i64::from(fd)
    }
}
fn create(dir: &Path) -> i64 {
    let name = CString::new(
        dir.join(format!("probe-{}", std::process::id()))
            .as_os_str()
            .as_bytes(),
    )
    .unwrap();
    // SAFETY: name is NUL-terminated; a created file is closed at once.
    unsafe {
        let fd = libc::open(
            name.as_ptr(),
            libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_CLOEXEC,
            0o600,
        );
        if fd >= 0 {
            libc::close(fd);
        }
        i64::from(fd)
    }
}
fn open_read(path: &Path) -> i64 {
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: name is NUL-terminated; an opened file is closed at once.
    unsafe {
        let fd = libc::open(name.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
        if fd >= 0 {
            libc::close(fd);
        }
        i64::from(fd)
    }
}

/// What the spawn layer must already refuse, before the child locks down.
pub(super) fn spawn_violations() -> Vec<String> {
    let mut found = Vec::new();
    for (what, family, kind) in [
        ("TCP socket", libc::AF_INET, libc::SOCK_STREAM),
        ("UDP6 socket", libc::AF_INET6, libc::SOCK_DGRAM),
        ("netlink socket", libc::AF_NETLINK, libc::SOCK_RAW),
    ] {
        refused(what, socket(family, kind), libc::EPERM, &mut found);
    }
    // SAFETY: an unexpectedly created process exits at once without unwinding.
    let child_program = c"/bin/false";
    let child_argv = [child_program.as_ptr(), std::ptr::null()];
    let child_env = [std::ptr::null()];
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // A child here proves the sandbox failed to refuse fork. Replace it
        // with a program whose process status is explicitly unsuccessful.
        // SAFETY: all pointers refer to NUL-terminated static strings and
        // `child_argv`/`child_env` remain valid across this async-safe call.
        unsafe {
            libc::execve(
                child_program.as_ptr(),
                child_argv.as_ptr(),
                child_env.as_ptr(),
            );
            libc::abort();
        }
    }
    if pid > 0 {
        let mut status = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        if waited != pid {
            found.push(format!("fork child {pid} could not be reaped: {waited}"));
        } else if !libc::WIFEXITED(status) || libc::WEXITSTATUS(status) == 0 {
            found.push(format!("unexpected fork child status {status:#x}"));
        }
    }
    refused("fork", i64::from(pid), libc::EPERM, &mut found);
    let mut params = [0u64; 16];
    // SAFETY: params is a zeroed buffer larger than struct io_uring_params.
    let ring = unsafe { libc::syscall(libc::SYS_io_uring_setup, 1u32, params.as_mut_ptr()) };
    if ring >= 0 {
        unsafe { libc::close(ring as i32) };
    }
    refused("io_uring_setup", ring, libc::EPERM, &mut found);
    let mut byte = 0u8;
    let local = libc::iovec {
        iov_base: (&mut byte as *mut u8).cast(),
        iov_len: 1,
    };
    let remote = libc::iovec {
        iov_base: std::ptr::null_mut(),
        iov_len: 1,
    };
    // SAFETY: reads at most one byte into `byte`.
    let read = unsafe { libc::process_vm_readv(libc::getppid(), &local, 1, &remote, 1, 0) };
    refused("process_vm_readv", read as i64, libc::EPERM, &mut found);
    refused(
        "unshare",
        i64::from(unsafe { libc::unshare(libc::CLONE_NEWUSER) }),
        libc::EPERM,
        &mut found,
    );
    // Each call below is harmless if it is wrongly allowed: a read-only
    // prlimit, an invalid length or mode, or the caller's current priority.
    let parent = unsafe { libc::getppid() };
    let mut old = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    let other = unsafe {
        libc::syscall(
            libc::SYS_prlimit64,
            parent,
            libc::RLIMIT_NOFILE,
            std::ptr::null::<libc::rlimit>(),
            &mut old as *mut libc::rlimit,
        )
    };
    refused("prlimit another process", other, libc::EPERM, &mut found);
    let own = unsafe {
        libc::syscall(
            libc::SYS_prlimit64,
            0,
            libc::RLIMIT_NOFILE,
            std::ptr::null::<libc::rlimit>(),
            &mut old as *mut libc::rlimit,
        )
    };
    if own != 0 {
        found.push("cannot read its own limits".into());
    }
    let affinity = unsafe { libc::syscall(libc::SYS_sched_setaffinity, parent, 0, 0) };
    refused("pin another process", affinity, libc::EPERM, &mut found);
    let param = unsafe { libc::syscall(libc::SYS_sched_setparam, parent, 0) };
    refused("schedule another process", param, libc::EPERM, &mut found);
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
    refused(
        "setpriority",
        i64::from(unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) }),
        libc::EPERM,
        &mut found,
    );
    let ioprio = unsafe { libc::syscall(libc::SYS_ioprio_set, 0, 0, 0) };
    refused("ioprio_set", ioprio, libc::EPERM, &mut found);
    let pidfd = unsafe { libc::syscall(libc::SYS_pidfd_open, parent, 0) };
    if pidfd >= 0 {
        unsafe { libc::close(pidfd as i32) };
    }
    refused("pidfd_open", pidfd, libc::EPERM, &mut found);
    let threads = ["OMP", "RAYON", "OPENBLAS", "MKL"].map(|t| format!("{t}_NUM_THREADS"));
    for (name, _) in std::env::vars_os() {
        let name = name.to_string_lossy();
        if !(INHERITED_ENV.contains(&name.as_ref())
            || [POLICY_ENV, PROBE_ENV].contains(&name.as_ref())
            || threads.contains(&name.to_string())
            || name.starts_with("JOPE_SERVICE_")
            || name.starts_with("JOPE_QUERY_"))
        {
            found.push(format!("environment variable {name} inherited"));
        }
    }
    for (what, resource, expected) in [
        ("core", libc::RLIMIT_CORE, Some(0)),
        ("open files", libc::RLIMIT_NOFILE, Some(OPEN_FILES)),
        ("address space", libc::RLIMIT_AS, Some(ADDRESS_SPACE_BYTES)),
        ("cpu", libc::RLIMIT_CPU, None),
        ("file size", libc::RLIMIT_FSIZE, None),
    ] {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        unsafe { libc::getrlimit(resource, &mut limit) };
        let bounded = limit.rlim_cur == limit.rlim_max
            && limit.rlim_max != libc::RLIM_INFINITY
            && expected.is_none_or(|e| limit.rlim_max == e);
        if !bounded {
            found.push(format!(
                "{what} limit {}/{}",
                limit.rlim_cur, limit.rlim_max
            ));
        }
    }
    let status = "/proc/self/status";
    if status_field(status, "NoNewPrivs:").ok() != Some(1)
        || status_field(status, "Seccomp:").ok() != Some(2)
    {
        found.push("no_new_privs or seccomp filter mode absent".into());
    }
    found
}

/// What an evaluated program could still do after lockdown; empty when
/// confined. `writable`, when given, must accept a new file.
pub(crate) fn violations(
    unreadable: &[PathBuf],
    unwritable: &Path,
    writable: Option<&Path>,
) -> Vec<String> {
    let mut found = spawn_violations();
    refused(
        "UNIX socket",
        socket(libc::AF_UNIX, libc::SOCK_STREAM),
        libc::EPERM,
        &mut found,
    );
    let mut pair = [0; 2];
    refused(
        "socketpair",
        i64::from(unsafe {
            libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, pair.as_mut_ptr())
        }),
        libc::EPERM,
        &mut found,
    );
    let program = c"/bin/true";
    let argv = [program.as_ptr(), std::ptr::null()];
    // SAFETY: argv is NULL-terminated; success would replace this probe, which
    // then never reports and fails its test.
    let exec = unsafe { libc::execv(program.as_ptr(), argv.as_ptr()) };
    refused("execve", i64::from(exec), libc::EPERM, &mut found);
    refused(
        "signal parent",
        i64::from(unsafe { libc::kill(libc::getppid(), 0) }),
        libc::EPERM,
        &mut found,
    );
    if unsafe { libc::kill(libc::getpid(), 0) } != 0 {
        found.push("cannot signal itself".into());
    }
    if std::thread::spawn(|| 7).join().ok() != Some(7) {
        found.push("cannot start a thread".into());
    }
    let mut action = libc::SECCOMP_RET_ALLOW;
    // SECCOMP_GET_ACTION_AVAIL only queries; it is refused like every operation.
    let query = unsafe { libc::syscall(libc::SYS_seccomp, 2u32, 0u32, &mut action as *mut u32) };
    refused("seccomp", query, libc::EPERM, &mut found);
    let mode = unsafe { libc::prctl(libc::PR_SET_SECCOMP, 99, 0, 0, 0) };
    refused(
        "prctl(PR_SET_SECCOMP)",
        i64::from(mode),
        libc::EPERM,
        &mut found,
    );
    if landlock_abi() > 0 {
        for path in unreadable {
            let what = format!("read {}", path.display());
            refused(&what, open_read(path), libc::EACCES, &mut found);
        }
        let what = format!("create in {}", unwritable.display());
        refused(&what, create(unwritable), libc::EACCES, &mut found);
        if let Some(dir) = writable {
            if create(dir) < 0 {
                found.push(format!("cannot create in {}", dir.display()));
            }
        }
        for own in [
            "/proc/self/exe",
            "/proc/self/maps",
            "/proc/sys/kernel/random/boot_id",
        ] {
            if open_read(Path::new(own)) < 0 {
                found.push(format!("cannot read {own}"));
            }
        }
    }
    found
}
