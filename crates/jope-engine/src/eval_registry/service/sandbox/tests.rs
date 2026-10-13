//! Confinement probes. `probe_child` runs in a copy of this test binary spawned
//! through the service's own spawn path; `violations` also runs inside a
//! service-owned fixture fit child (service::tests::synthetic_worker).
use super::*;
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Lines, Write},
    os::unix::process::CommandExt,
    process::{Child, ChildStdin, ChildStdout, Stdio},
};

const PROBE_ENV: &str = "JOPE_SANDBOX_PROBE";
const PROBE_TEST: &str = "eval_registry::service::sandbox::tests::probe_child";
const REPORT: &str = "SANDBOX_PROBE ";

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
fn socket(family: i32, kind: i32) -> i64 {
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
fn spawn_violations() -> Vec<String> {
    let mut found = Vec::new();
    for (what, family, kind) in [
        ("TCP socket", libc::AF_INET, libc::SOCK_STREAM),
        ("UDP6 socket", libc::AF_INET6, libc::SOCK_DGRAM),
        ("netlink socket", libc::AF_NETLINK, libc::SOCK_RAW),
    ] {
        refused(what, socket(family, kind), libc::EPERM, &mut found);
    }
    // SAFETY: an unexpectedly created process exits at once without unwinding.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        unsafe { libc::_exit(0) };
    }
    if pid > 0 {
        unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) };
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
        if open_read(Path::new("/proc/self/exe")) < 0 {
            found.push("cannot read its own executable".into());
        }
    }
    found
}

#[test]
#[ignore = "sandbox probe child, launched only by the confinement tests"]
fn probe_child() {
    let Some(spec) = std::env::var_os(PROBE_ENV) else {
        return;
    };
    let spec: Value = serde_json::from_str(&spec.to_string_lossy()).unwrap();
    let paths = |key: &str| -> Vec<PathBuf> {
        spec[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| PathBuf::from(v.as_str().unwrap()))
            .collect()
    };
    let report = |value: Value| println!("{REPORT}{value}");
    let mut found = spawn_violations();
    if socket(libc::AF_UNIX, libc::SOCK_STREAM) < 0 {
        found.push("spawn layer refused a UNIX socket".into());
    }
    report(json!({"stage":"spawn","violations":found}));
    let mut stdin = std::io::stdin().lock();
    let mut line = String::new();
    stdin.read_line(&mut line).unwrap();
    if let Err(error) = lockdown() {
        report(json!({"stage":"refused","error":format!("{error:#}")}));
        return;
    }
    let writable = paths("writable");
    let found = violations(
        &paths("unreadable"),
        &paths("unwritable")[0],
        writable.first().map(PathBuf::as_path),
    );
    report(json!({"stage":"locked","violations":found}));
    // Stay alive, confined, until the parent has attested the lockdown.
    stdin.read_line(&mut line).unwrap();
}

struct Probe {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
    attestation: Attestation,
}
impl Probe {
    fn spawn(confinement: Confinement, spec: Value) -> Self {
        let attestation = confinement.attestation();
        let mut command = Command::new(std::env::current_exe().unwrap());
        // Explicit variables set before configuration are cleared too.
        command
            .env("AWS_SECRET_ACCESS_KEY", "probe-secret")
            .env("GH_TOKEN", "probe-token");
        confinement.configure(&mut command);
        command
            .args(["--exact", PROBE_TEST, "--ignored", "--nocapture"])
            .env(PROBE_ENV, spec.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // SAFETY: enter makes async-signal-safe system calls only.
        unsafe {
            command.pre_exec(move || confinement.enter(libc::STDIN_FILENO));
        }
        let mut child = command.spawn().unwrap();
        Self {
            stdin: child.stdin.take(),
            lines: BufReader::new(child.stdout.take().unwrap()).lines(),
            child,
            attestation,
        }
    }
    fn report(&mut self) -> Value {
        for line in self.lines.by_ref() {
            if let Some(report) = line.unwrap().strip_prefix(REPORT) {
                return serde_json::from_str(report).unwrap();
            }
        }
        panic!("sandbox probe exited without reporting");
    }
    fn release(&mut self) {
        self.stdin.as_mut().unwrap().write_all(b"next\n").unwrap();
    }
    fn finish(mut self) {
        drop(self.stdin.take());
        assert!(self.child.wait().unwrap().success());
    }
}

struct Custody {
    temp: tempfile::TempDir,
    config: ServiceConfig,
}
impl Custody {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("authority");
        fs::create_dir_all(root.join("worker-output")).unwrap();
        fs::write(root.join("authority.json"), b"{}").unwrap();
        fs::write(temp.path().join("secret"), b"credential").unwrap();
        let config = ServiceConfig::regression(
            root,
            temp.path().join("witness"),
            temp.path().join("service.sock"),
            std::env::current_exe().unwrap(),
            1,
        );
        Self { temp, config }
    }
    fn spec(&self, writable: bool) -> Value {
        let root = &self.config.authority_root;
        let output = root.join("worker-output");
        json!({
            "unreadable": [
                self.temp.path().join("secret"),
                root.join("authority.json"),
                "/etc/passwd",
                format!("/proc/{}/environ", std::process::id()),
            ],
            "unwritable": [if writable { root.clone() } else { output.clone() }],
            "writable": if writable { vec![output] } else { vec![] },
        })
    }
}

fn assert_confined(probe: &mut Probe) {
    let spawn = probe.report();
    assert_eq!(spawn["stage"], "spawn", "{spawn}");
    assert_eq!(spawn["violations"], json!([]), "{spawn}");
    let pid = probe.child.id();
    assert!(
        probe.attestation.verify(pid).is_err(),
        "attested before lockdown"
    );
    probe.release();
    let locked = probe.report();
    assert_eq!(locked["stage"], "locked", "{locked}");
    assert_eq!(locked["violations"], json!([]), "{locked}");
    probe.attestation.verify(pid).unwrap();
    println!("landlock_abi={}", landlock_abi());
}

#[test]
fn fit_child_is_confined_and_attested_only_after_lockdown() {
    let custody = Custody::new();
    let confinement = Confinement::fit(&custody.config).unwrap();
    let mut probe = Probe::spawn(confinement, custody.spec(true));
    assert_confined(&mut probe);
    probe.finish();
}

#[test]
fn query_child_is_confined_and_writes_no_files() {
    let custody = Custody::new();
    let confinement = Confinement::query(&custody.config, Duration::from_secs(60)).unwrap();
    let mut probe = Probe::spawn(confinement, custody.spec(false));
    assert_confined(&mut probe);
    probe.finish();
}

#[test]
fn lockdown_refuses_a_child_with_more_threads_than_allowed() {
    let mut custody = Custody::new();
    // libtest runs the probe on a second thread, which Landlock would not bind.
    custody.config.sandbox.threads = 1;
    let confinement = Confinement::fit(&custody.config).unwrap();
    let mut probe = Probe::spawn(confinement, custody.spec(true));
    assert_eq!(probe.report()["stage"], "spawn");
    probe.release();
    let refused = probe.report();
    assert_eq!(refused["stage"], "refused", "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("allows 1 thread(s); found 2"),
        "{refused}"
    );
    assert!(probe.attestation.verify(probe.child.id()).is_err());
    probe.finish();
}

#[test]
fn unconfined_processes_fail_attestation() {
    let custody = Custody::new();
    let attestation = Confinement::fit(&custody.config).unwrap().attestation();
    let error = attestation.verify(std::process::id()).unwrap_err();
    assert!(error.to_string().contains("not locked down"), "{error:#}");
}

#[test]
fn fs_allowlist_requirement_refuses_kernels_without_landlock() {
    let required = WorkerSandbox {
        require_fs_allowlist: true,
        ..WorkerSandbox::default()
    };
    let error = enforce(0, &required).unwrap_err();
    assert!(
        error.to_string().contains("Landlock is unavailable"),
        "{error:#}"
    );
    assert_eq!(enforce(0, &WorkerSandbox::default()).unwrap(), 0);
    assert_eq!(enforce(4, &required).unwrap(), 4);
    // Rights from ABIs this module does not name are never handled.
    assert_eq!(enforce(KNOWN_ABI + 3, &required).unwrap(), KNOWN_ABI);
    assert_eq!(check(&required).is_ok(), landlock_abi() > 0);
}

#[test]
fn lockdown_requires_the_services_policy() {
    let policy = Policy {
        schema: "jope.eval_sandbox.v0".into(),
        landlock_abi: 0,
        threads: u64::MAX,
        writable: Vec::new(),
        mutable: Vec::new(),
    };
    // The schema check precedes every restriction, so this process stays free.
    let error = lockdown_with(&policy).unwrap_err();
    assert!(error
        .to_string()
        .contains("unknown evaluator sandbox policy"));
    let encoded = json!({"schema":POLICY_SCHEMA,"landlock_abi":0,"threads":1,
        "writable":[],"mutable":[],"network":true});
    assert!(serde_json::from_value::<Policy>(encoded).is_err());
}
