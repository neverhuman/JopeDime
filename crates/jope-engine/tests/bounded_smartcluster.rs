//! W7-T05/T07 same-host multi-process fixture through the real supervisor.
//!
//! Every rank has the topology of the three-host proof, collapsed onto loopback: its own real
//! SmartCluster daemon (`scqd`, unprivileged embedded mode, one allocation at a time) runs the
//! rank as a trusted-service job whose program is the production `jope-native-loop` command line
//! as the test-only fixture executable (`support/native_loop_fixture.rs`), so the participant
//! freshly observes the checked-in fixture host (`support/fixture_host.rs`) and its own running
//! binary, not the machine the tests run on; its own real `jope-native-loop agent` process (the
//! product binary) serves that daemon's `SmartClusterSupervisor` (authenticated exchange: pinned
//! uid and executable digest) over the pinned-TLS transport; the coordinator's
//! `RemoteSupervisor` routes each rank to its agent, and the coordinator runs the real
//! rendezvous over the same transport. The host census is collected, on the same fixture host,
//! from the host's installation file, whose executable is the participant's. SSH is never
//! involved.
//!
//! This is a FIXTURE: three ranks under one boot on one host are not the three-host proof
//! (W7-T07), and the CPU proof workload is not numerical training.
#[path = "support/bounded_fixture.rs"]
mod fixture;
#[path = "support/fixture_host.rs"]
mod fixture_host;
#[path = "support/tls_fixture.rs"]
mod tls;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    agent::RemoteSupervisor,
    collect::{self, CommandOutput, CommandRunner},
    contract::{BoundedRunContract, Ceilings, WorkItem, WorkKind},
    controller::{self, Disposition, Outcome, Supervisor},
    driver::{self, CoordinateArgs},
    participant::{Installation, SmartClusterInstallation, INSTALLATION_SCHEMA},
    rendezvous,
    supervisor::{DaemonPin, HostSupervision, SmartClusterSupervisor, SERVICE},
    transport::HostCredential,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader},
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    sync::{atomic::AtomicBool, Arc},
    thread,
    time::{Duration, Instant},
};
use tls::*;

const MIB: u64 = 1024 * 1024;
const PROJECT: &str = "bounded-cpu-proof";
const RANKS: u32 = 3;
/// SmartCluster requires a job's tenant to be exactly the submitting peer's principal.
#[path = "bounded_smartcluster/mod.rs"]
mod bounded_smartcluster;
use bounded_smartcluster::*;

#[test]
fn smartcluster_ranks_commit_the_whole_group_exactly_once() {
    let host = Host::new(5);
    let (_stop, t) = coordinate_in_background(&host);
    let status = host.joined(t);
    assert_eq!(
        status.outcome,
        Some(Outcome::Committed),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_005);
    assert!(status
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Committed)));
    let verified = controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    assert_eq!(verified.results_verified, 3);
    host.assert_no_orphans();
    // The plan pins the digest of the participant executable the daemons run; every rank's
    // hello matched it from its own `/proc/self/exe` (the coordinator admits nothing else).
    let executable = sha(&fs::read(participant_binary()).unwrap());
    assert!(host
        .contract
        .participants
        .iter()
        .all(|p| p.host.executable_sha256 == executable));
    // Every rank ran as a SmartCluster job of the participant executable, one per daemon,
    // observed the fixture host, and its result records the job and attempt it proved it was
    // (W7-D1).
    for rank in 0..RANKS {
        let outbox = host.dispatch_dir(rank).join("outbox");
        assert!(outbox.join("envelope.json").exists() && outbox.join("result.bin").exists());
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(outbox.join("result.bin")).unwrap()).unwrap();
        assert_eq!(result["host"], fixture_host::HOST);
        assert_eq!(result["machine_id"], fixture_host::MACHINE_ID);
        assert_eq!(result["boot_id"], fixture_host::BOOT_ID);
        let mut own =
            SmartClusterSupervisor::new(host.ranks[rank as usize].supervision.clone()).unwrap();
        let handle = own
            .find(&rendezvous::dispatch_id(&host.sha, 1, rank).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(result["smartcluster_job"]["job_id"], handle.identity);
        assert_eq!(result["smartcluster_job"]["attempt"], 1);
        assert_eq!(result["smartcluster_job"]["tenant"], tenant());
    }
    // Running the coordinator again on a committed run changes nothing.
    let before = controller::status(&host.run_dir).unwrap().journal_records;
    let again = host.coordinated(
        "driver::coordinate on the committed run",
        driver::coordinate(&host.args(), host.supervisor(), &AtomicBool::new(false)),
    );
    assert_eq!(again.outcome, Some(Outcome::Committed));
    assert_eq!(again.committed_updates_total, 60_005);
    assert_eq!(
        controller::status(&host.run_dir).unwrap().journal_records,
        before
    );
    // A host supervisor acts only on its own plan's jobs: bound to another plan artifact on
    // the same daemon, it neither finds nor probes rank 0's job.
    let dispatch = rendezvous::dispatch_id(&host.sha, 1, 0).unwrap();
    let mut own = SmartClusterSupervisor::new(host.ranks[0].supervision.clone()).unwrap();
    let handle = own.find(&dispatch).unwrap().expect("rank 0's job");
    let other_plan = host.root.join("other-plan.json");
    fs::write(&other_plan, b"{\"another\":\"plan\"}").unwrap();
    let mut foreign = SmartClusterSupervisor::new(HostSupervision {
        plan: other_plan,
        ..host.ranks[0].supervision.clone()
    })
    .unwrap();
    assert_eq!(foreign.find(&dispatch).unwrap(), None);
    for e in [
        foreign.probe(&handle).err(),
        foreign.terminate(&handle).err(),
        foreign.retained_result(&handle).err(),
    ] {
        let e = e.expect("refused");
        assert!(
            e.to_string()
                .contains("is not a job of this supervisor's plan"),
            "{e}"
        );
    }
}

#[test]
fn a_killed_rank_drains_the_whole_group_through_smartcluster() {
    let host = Host::new(400);
    let (_stop, t) = coordinate_in_background(&host);
    host.wait_admitted();
    // SIGKILL rank 1's worker process, which rank 1's SmartCluster daemon started.
    let victim = host.dispatch_dir(1);
    let pid = {
        let mut found = None;
        wait(Duration::from_secs(10), || {
            found = host.rank_processes().into_iter().find(|pid| {
                fs::read(format!("/proc/{pid}/cmdline"))
                    .map(|c| String::from_utf8_lossy(&c).contains(victim.to_str().unwrap()))
                    .unwrap_or(false)
            });
            found.is_some()
        });
        found.unwrap()
    };
    unsafe { libc::kill(pid, libc::SIGKILL) };
    let status = host.joined(t);
    assert_eq!(
        status.outcome,
        Some(Outcome::Drained),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_000);
    assert_eq!(status.ranks[1].disposition, Some(Disposition::Failed));
    for r in [0, 2] {
        assert_eq!(status.ranks[r].disposition, Some(Disposition::Drained));
    }
    assert!(!host.run_dir.join("results").exists());
    controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    host.assert_no_orphans();
}

#[test]
fn cancellation_drains_the_whole_group_through_smartcluster() {
    let host = Host::new(400);
    let (_stop, t) = coordinate_in_background(&host);
    host.wait_admitted();
    let report = controller::cancel(&host.run_dir, &host.sha, "operator cancel", now_ms()).unwrap();
    assert!(report.delivered_to_live_coordinator);
    let status = host.joined(t);
    assert_eq!(
        status.outcome,
        Some(Outcome::Cancelled),
        "{status:?}\n{}",
        host.diagnose()
    );
    assert_eq!(status.committed_updates_total, 60_000);
    assert!(status
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    host.assert_no_orphans();
}

/// Rank 0's participant argv (as its supervisor would submit it), writing to `outbox`, with
/// its frozen launch record at `dir/launch.json`.
#[test]
fn an_inherited_flock_file_is_refused_without_releasing_the_parent_lock() {
    use std::os::fd::{AsFd, AsRawFd};

    let host = Host::new(5);
    let path = host.root.join("parent.lock");
    let lock = fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    lock.try_lock().unwrap();
    // SAFETY: fcntl(F_GETFD) borrows the open File's descriptor and neither closes it.
    let flags = unsafe { libc::fcntl(lock.as_raw_fd(), libc::F_GETFD) };
    assert!(flags >= 0);
    let contended = || {
        // A new open file description is essential: dup/try_clone would share the lock.
        let contender = fs::File::open(&path).unwrap();
        // std's try_lock is flock(LOCK_EX | LOCK_NB); WouldBlock is exactly EWOULDBLOCK.
        assert!(matches!(
            contender.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));
    };
    contended();

    // A syntactically valid claim reaches the real control-channel verifier. No job is
    // submitted: the inherited regular file must be refused before job lookup or work.
    let forged = [
        ("SCQ_JOB_ID", jain_smartcluster::JobId::new().to_string()),
        ("SCQ_ATTEMPT_NUMBER", "1".to_string()),
        ("SCQ_ATTEMPT_FENCE", "1".to_string()),
    ];
    let (ok, refusal) = run_outside(&host, "flock-file", &forged, Some(lock.as_fd()));
    assert!(!ok);
    assert!(
        refusal.contains("is not a SmartCluster control socket"),
        "{refusal}"
    );
    assert!(!refusal.contains("is not an open descriptor"), "{refusal}");
    let outbox = host.root.join("outside-flock-file/outbox");
    let files: Vec<_> = fs::read_dir(outbox)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(files, vec![std::ffi::OsString::from("refusal.txt")]);
    // The child refused an open file, and left the parent's descriptor and lock intact.
    assert_eq!(
        unsafe { libc::fcntl(lock.as_raw_fd(), libc::F_GETFD) },
        flags
    );
    contended();
    drop(lock);
    // No surviving child/helper duplicate may keep the lock after its owner releases it.
    let contender = fs::File::open(path).unwrap();
    contender.try_lock().unwrap();
}

#[test]
fn a_participant_proves_its_smartcluster_job_before_any_work() {
    use jain_smartcluster::{
        AttemptNumber, FenceToken, FencedAttempt, IdempotencyKey, JobSpec, JobState, TerminalStatus,
    };
    use jope_parallel::native_loop::bounded::{
        controller::RankLaunch,
        job_identity::{verify_job_record, ExpectedJob},
        supervisor::{DISPATCH_LABEL, PLAN_LABEL},
    };
    use std::os::fd::AsFd;
    let host = Host::new(5);
    let rank0 = &host.ranks[0].supervision;
    let dispatch = |rank| rendezvous::dispatch_id(&host.sha, 1, rank).unwrap();

    // Outside SmartCluster: started by hand, with no job environment.
    let (ok, refusal) = run_outside(&host, "no-job", &[], None);
    assert!(!ok);
    assert!(refusal.contains("SCQ_JOB_ID is unset"), "{refusal}");

    // Jobs of rank 0's own daemon, submitted directly with rank 0's frozen launch, whose
    // spec differs from its supervisor's only by `edit`.
    let client = jain_smartcluster::Client::new(rank0.daemon.socket.clone());
    let (mut genuine, mut other_job) = (None, None);
    type JobMutation = fn(&mut JobSpec, &str);
    let cases: [(&str, JobMutation, &str); 3] = [
        // Positive control: it proves its job, then finds no coordinator listening.
        ("genuine", |_, _| {}, "connect 127.0.0.1"),
        (
            "wrong-dispatch",
            |spec, other| {
                spec.labels.insert(DISPATCH_LABEL.into(), other.to_string());
            },
            "was submitted for another dispatch",
        ),
        (
            "other-plan",
            |spec, _| {
                spec.labels.insert(PLAN_LABEL.into(), sha(b"another plan"));
            },
            "was submitted for another plan",
        ),
    ];
    for (case, edit, needle) in cases {
        let supervisor = SmartClusterSupervisor::new(HostSupervision {
            state: host.root.join(format!("jobs-{case}")),
            ..rank0.clone()
        })
        .unwrap();
        let (dir, mut spec) = supervisor
            .launch_spec(&RankLaunch {
                dispatch_intent_unix_ms: now_ms(),
                contract: &host.contract,
                plan_sha256: &host.sha,
                global_rank: 0,
                dispatch_id: &dispatch(0),
            })
            .unwrap();
        spec.idempotency_key = IdempotencyKey::new(format!("w7-d1-{case}")).unwrap();
        edit(&mut spec, &dispatch(1));
        let submitted = block_on(client.submit(spec)).unwrap();
        let mut job = None;
        wait(Duration::from_secs(60), || {
            job = block_on(client.get(submitted.id)).unwrap();
            job.as_ref().is_some_and(|j| j.terminal.is_some())
        });
        let job = job.unwrap();
        assert!(
            matches!(job.terminal, Some(TerminalStatus::Failed { .. })),
            "{case}: {:?}\n{}",
            job.terminal,
            host.diagnose()
        );
        let refusal = fs::read_to_string(dir.join("outbox/refusal.txt")).unwrap();
        assert!(refusal.contains(needle), "{case}: {refusal}");
        assert!(!dir.join("outbox/admitted").exists());
        if case == "genuine" {
            genuine = Some(job);
        } else {
            other_job = Some(job.id);
        }
    }

    // Started by hand with a genuine job's environment: no control socket, then a control
    // socket whose peer (this test, its parent) is not the pinned daemon executable.
    let genuine = genuine.unwrap();
    let forged = [
        ("SCQ_JOB_ID", genuine.id.to_string()),
        ("SCQ_ATTEMPT_NUMBER", "1".to_string()),
        ("SCQ_ATTEMPT_FENCE", "1".to_string()),
    ];
    let (ok, refusal) = run_outside(&host, "no-control", &forged, None);
    assert!(!ok);
    assert!(refusal.contains("is not an open descriptor"), "{refusal}");
    let (ours, _theirs) = std::os::unix::net::UnixStream::pair().unwrap();
    let (ok, refusal) = run_outside(&host, "impostor-daemon", &forged, Some(ours.as_fd()));
    assert!(!ok);
    assert!(
        refusal.contains("is not the pinned SmartCluster executable"),
        "{refusal}"
    );

    // The daemon's record checks, on the genuine job's real record made live again.
    let attempt = FencedAttempt::new(
        genuine.id,
        AttemptNumber::new(1).unwrap(),
        FenceToken::new(7).unwrap(),
    );
    let mut live = genuine.clone();
    live.state = JobState::Running;
    live.terminal = None;
    live.active_attempt = Some(attempt);
    let arguments = match &live.spec.workload {
        jain_smartcluster::Workload::TrustedService { arguments, .. } => arguments
            .iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let expected = ExpectedJob {
        plan_sha256: host.sha.clone(),
        run: host.contract.run.clone(),
        global_rank: 0,
        dispatch_id: dispatch(0),
        tenant: tenant(),
        service: SERVICE.into(),
        arguments,
    };
    verify_job_record(&attempt, &live, &expected).unwrap();
    let refused = |claim: &FencedAttempt, job: &jain_smartcluster::JobRecord, e: &ExpectedJob| {
        format!("{:#}", verify_job_record(claim, job, e).unwrap_err())
    };
    // A replayed attempt: the same job under another fence or attempt number.
    for claim in [
        FencedAttempt::new(
            genuine.id,
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(6).unwrap(),
        ),
        FencedAttempt::new(
            genuine.id,
            AttemptNumber::new(2).unwrap(),
            FenceToken::new(7).unwrap(),
        ),
    ] {
        assert!(refused(&claim, &live, &expected).contains("stale or replayed attempt"));
    }
    // A job that is no longer live.
    assert!(refused(&attempt, &genuine, &expected).contains("not live"));
    // Another tenant, another service or other arguments.
    let other_tenant = ExpectedJob {
        tenant: "unix:0".into(),
        ..expected.clone()
    };
    assert!(refused(&attempt, &live, &other_tenant).contains("tenant"));
    let other_service = ExpectedJob {
        service: "another-service".into(),
        ..expected.clone()
    };
    assert!(refused(&attempt, &live, &other_service).contains("runs service"));
    let mut other_arguments = expected.clone();
    other_arguments.arguments.push("--extra".into());
    assert!(refused(&attempt, &live, &other_arguments).contains("arguments"));
    // Another job's answer to this job's claim.
    let mut another = live.clone();
    another.id = other_job.unwrap();
    assert!(refused(&attempt, &another, &expected).contains("answered with job"));

    // A supervision whose daemon is not the one the host installation pins is refused.
    let e = SmartClusterSupervisor::new(HostSupervision {
        daemon: host.ranks[1].supervision.daemon.clone(),
        ..rank0.clone()
    })
    .err()
    .unwrap();
    assert!(
        e.to_string().contains("pins another SmartCluster daemon"),
        "{e}"
    );
}

fn coordinate_cli(host: &Host) -> Child {
    let a = host.args();
    let log = fs::File::create(host.root.join(format!("coordinate-{}.log", now_ms()))).unwrap();
    Command::new(product_binary())
        .arg("coordinate")
        .args(["--run-dir", a.run_dir.to_str().unwrap()])
        .args(["--plan-sha256", &a.plan_sha256])
        .args(["--census", a.census.to_str().unwrap()])
        .args(["--listen", &a.listen.to_string()])
        .args(["--certificate", a.certificate.to_str().unwrap()])
        .args(["--certificate-key", a.certificate_key.to_str().unwrap()])
        .args(["--coordinator-key", a.coordinator_key.to_str().unwrap()])
        .args(["--heartbeat-ms", "100"])
        .args(["--agents", host.agents_file.to_str().unwrap()])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(log)
        .spawn()
        .unwrap()
}

#[test]
fn a_restarted_coordinator_reconciles_and_drains_with_no_orphan() {
    let host = Host::new(400);
    let mut first = coordinate_cli(&host);
    host.wait_admitted();
    // The coordinator process dies; its ranks lose control and stop themselves.
    unsafe { libc::kill(first.id() as i32, libc::SIGKILL) };
    let _ = first.wait();
    let pending = controller::status(&host.run_dir).unwrap();
    assert_eq!(pending.outcome, None);
    assert!(!pending.coordinator_live);
    host.assert_no_orphans();
    // The restarted coordinator reconciles every owned rank through the same agents.
    let out = coordinate_cli(&host).wait_with_output().unwrap();
    assert!(out.status.success(), "{}", host.diagnose());
    let status: controller::StatusReport = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(status.outcome, Some(Outcome::Drained));
    assert_eq!(status.committed_updates_total, 60_000);
    assert_eq!(
        status.original_deadline_unix_ms,
        host.contract.original_deadline_unix_ms
    );
    assert!(host.journal_contains("coordinator_restarted"));
    assert!(status.ranks.iter().all(|r| r.disposition.is_some()));
    controller::verify(&host.run_dir, Some(&host.sha)).unwrap();
    host.assert_no_orphans();
}
