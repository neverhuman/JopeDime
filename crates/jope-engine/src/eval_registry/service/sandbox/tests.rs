//! Confinement probes. `probe_child` runs in a copy of this test binary spawned
//! through the service's own spawn path; `violations` also runs inside a
//! service-owned fixture fit child (service::tests::synthetic_worker).
use super::*;

mod violations;
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Lines, Write},
    os::unix::process::CommandExt,
    process::{Child, ChildStdin, ChildStdout, Stdio},
};
pub(crate) use violations::violations;
use violations::{socket, spawn_violations, PROBE_ENV, PROBE_TEST, REPORT};

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
        fs::create_dir_all(root.join("worker-output/ticket-a")).unwrap();
        fs::write(root.join("worker-output/ticket-b.json"), b"{}").unwrap();
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
    fn ticket(&self) -> PathBuf {
        self.config.authority_root.join("worker-output/ticket-a")
    }
    fn spec(&self, writable: bool) -> Value {
        let root = &self.config.authority_root;
        let outputs = root.join("worker-output");
        let parent = std::process::id();
        json!({
            "unreadable": [
                self.temp.path().join("secret"),
                root.join("authority.json"),
                outputs.join("ticket-b.json"),
                "/etc/passwd",
                format!("/proc/{parent}/environ"),
                format!("/proc/{parent}/cmdline"),
            ],
            "unwritable": [if writable { outputs } else { self.ticket() }],
            "writable": if writable { vec![self.ticket()] } else { vec![] },
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
    let confinement = Confinement::fit(&custody.config, &custody.ticket()).unwrap();
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
    let confinement = Confinement::fit(&custody.config, &custody.ticket()).unwrap();
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
    let attestation = Confinement::fit(&custody.config, &custody.ticket())
        .unwrap()
        .attestation();
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
