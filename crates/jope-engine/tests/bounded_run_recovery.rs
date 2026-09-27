//! W7-T05 same-host process fixture: whole-group drain and coordinator recovery with real
//! rank processes (re-executions of this test binary that join the authenticated
//! rendezvous over a Unix socket). This is a FIXTURE ONLY: three processes under one boot
//! on one host are not the three-host proof (W7-T07), and no rank does numerical work.
#[path = "support/bounded_fixture.rs"]
mod fixture;
use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    contract::{BoundedRunContract, PlanRefusal, Predecessor},
    controller::{
        self, Coordinator, Disposition, Liveness, Outcome, RankLaunch, RankState, Supervisor,
        SupervisorHandle, EXIT_LOST_CONTROL,
    },
    rendezvous::{self, Connected, ControlFrame},
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::net::{UnixListener, UnixStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const CHILD: &str = "W7C_RANK_CHILD";

/// Rank process entry point. As an ordinary test (no `W7C_RANK_CHILD`) it does nothing.
#[test]
fn rank_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let code = match child_main() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("rank child refused or failed: {e:#}");
            76
        }
    };
    std::process::exit(code);
}
fn child_main() -> Result<i32> {
    let var = |k: &str| std::env::var(k).with_context(|| k.to_string());
    let plan_bytes = fs::read(var("W7C_PLAN")?)?;
    let contract = BoundedRunContract::parse(&plan_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let plan_sha = sha(&plan_bytes);
    let rank: u32 = var("W7C_RANK")?.parse()?;
    let key = SigningKey::from_bytes(&rank_seed(rank as usize));
    let outbox = PathBuf::from(var("W7C_OUTBOX")?);
    let root = PathBuf::from(var("W7C_ROOT")?);
    let mode = var("W7C_MODE")?;
    let socket = PathBuf::from(var("W7C_SOCKET")?);
    let start = Instant::now();
    let mut stream = loop {
        match UnixStream::connect(&socket) {
            Ok(s) => break s,
            Err(_) if start.elapsed() < Duration::from_secs(10) => {
                std::thread::sleep(Duration::from_millis(20))
            }
            Err(e) => return Err(e.into()),
        }
    };
    // The participant reports its own observed closure digest, never the planned one.
    let closure = rendezvous::observed_closure_digest(&root, &contract.artifact_closure)?;
    let member = rendezvous::join_unix(
        &mut stream,
        &contract,
        &plan_sha,
        &key,
        |challenge| hello_body(&contract, &plan_sha, challenge, rank, now_ms(), &closure),
        &now_ms,
    )?;
    fs::write(outbox.join("admitted"), member.admission_sha256())?;
    stream.set_read_timeout(Some(Duration::from_millis(
        contract.rendezvous.heartbeat_timeout_ms,
    )))?;
    let mut last = 0u64;
    let mut beats = 0u32;
    loop {
        let Ok(frame) = rendezvous::read_frame(&mut stream, contract.rendezvous.max_message_bytes)
        else {
            // EOF or heartbeat timeout: control is lost, stop without a result.
            return Ok(EXIT_LOST_CONTROL);
        };
        match serde_json::from_slice::<ControlFrame>(&frame)? {
            ControlFrame::Heartbeat { delivery_id } if delivery_id > last => {
                last = delivery_id;
                beats += 1;
                if mode == "complete" && beats >= 3 {
                    let (envelope, bytes) =
                        result(&contract, &plan_sha, member.admission_sha256(), rank, &key);
                    fs::write(outbox.join("result.bin.partial"), bytes)?;
                    fs::rename(outbox.join("result.bin.partial"), outbox.join("result.bin"))?;
                    fs::write(outbox.join("envelope.partial"), envelope)?;
                    fs::rename(
                        outbox.join("envelope.partial"),
                        outbox.join("envelope.json"),
                    )?;
                    return Ok(0);
                }
            }
            // A replayed or regressed heartbeat is not control.
            ControlFrame::Heartbeat { .. } => return Ok(EXIT_LOST_CONTROL),
            ControlFrame::Stop { .. } => return Ok(0),
        }
    }
}

// ------------------------------------------------ same-host process supervisor (fixture)

/// Fixture supervisor for same-host rank processes. Its records live on disk, like a
/// daemon's, so a restarted coordinator with a fresh supervisor instance can reconcile.
struct ProcessSupervisor {
    state: PathBuf,
    socket: PathBuf,
    plan: PathBuf,
    root: PathBuf,
    modes: BTreeMap<u32, &'static str>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct ProcessRecord {
    pid: i32,
    birth: u64,
    boot: String,
}
fn boot_id() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .trim()
        .into()
}
/// `(state, birth)` of a live /proc entry.
fn proc_stat(pid: i32) -> Option<(char, u64)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let tail = stat.rsplit_once(')')?.1;
    let f: Vec<&str> = tail.split_whitespace().collect();
    Some((f.first()?.chars().next()?, f.get(19)?.parse().ok()?))
}
impl ProcessSupervisor {
    fn dir(&self, handle: &SupervisorHandle) -> PathBuf {
        self.state.join(&handle.identity)
    }
    fn record(&self, handle: &SupervisorHandle) -> Result<ProcessRecord> {
        Ok(serde_json::from_slice(&fs::read(
            self.dir(handle).join("process.json"),
        )?)?)
    }
    fn exit_record(&self, handle: &SupervisorHandle) -> Option<Liveness> {
        let v: serde_json::Value =
            serde_json::from_slice(&fs::read(self.dir(handle).join("exit.json")).ok()?).ok()?;
        Some(Liveness::Exited {
            code: v["code"].as_i64().map(|c| c as i32),
            signal: v["signal"].as_i64().map(|s| s as i32),
        })
    }
}
impl Supervisor for ProcessSupervisor {
    fn launch(&mut self, l: &RankLaunch<'_>) -> Result<SupervisorHandle> {
        let handle = SupervisorHandle {
            supervisor: "fixture-same-host-process".into(),
            identity: l.dispatch_id.into(),
        };
        let dir = self.dir(&handle);
        anyhow::ensure!(!dir.exists(), "duplicate launch of {}", l.dispatch_id);
        fs::create_dir_all(dir.join("outbox"))?;
        let log = fs::File::create(dir.join("rank.log"))?;
        let child = Command::new(std::env::current_exe()?)
            .args([
                "rank_child",
                l.dispatch_id,
                "--exact",
                "--nocapture",
                "--test-threads",
                "1",
            ])
            .env(CHILD, "1")
            .env("W7C_PLAN", &self.plan)
            .env("W7C_RANK", l.global_rank.to_string())
            .env("W7C_OUTBOX", dir.join("outbox"))
            .env("W7C_ROOT", &self.root)
            .env("W7C_MODE", self.modes[&l.global_rank])
            .env("W7C_SOCKET", &self.socket)
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .spawn()?;
        let pid = child.id() as i32;
        let birth = proc_stat(pid).context("spawned rank vanished")?.1;
        fs::write(
            dir.join("process.json"),
            serde_json::to_vec(&ProcessRecord {
                pid,
                birth,
                boot: boot_id(),
            })?,
        )?;
        Ok(handle)
    }
    fn find(&mut self, dispatch_id: &str) -> Result<Option<SupervisorHandle>> {
        let handle = SupervisorHandle {
            supervisor: "fixture-same-host-process".into(),
            identity: dispatch_id.into(),
        };
        Ok(self
            .dir(&handle)
            .join("process.json")
            .exists()
            .then_some(handle))
    }
    fn probe(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        if let Some(exited) = self.exit_record(handle) {
            return Ok(exited);
        }
        let r = self.record(handle)?;
        anyhow::ensure!(r.boot == boot_id(), "record from another boot");
        let mut status = 0;
        let mut reaped = unsafe { libc::waitpid(r.pid, &mut status, libc::WNOHANG) };
        if reaped != r.pid {
            match proc_stat(r.pid) {
                Some((state, birth)) if birth == r.birth && state != 'Z' => {
                    return Ok(Liveness::Running)
                }
                // The rank became a zombie between the non-blocking wait and the /proc
                // read: reap it now (returns immediately), never report it as gone while
                // it is still unreaped.
                Some((_, birth)) if birth == r.birth => {
                    reaped = unsafe { libc::waitpid(r.pid, &mut status, 0) };
                }
                _ => {}
            }
        }
        if reaped != r.pid {
            return Ok(Liveness::Gone);
        }
        let (code, signal) = if libc::WIFEXITED(status) {
            (Some(libc::WEXITSTATUS(status)), None)
        } else {
            (None, Some(libc::WTERMSIG(status)))
        };
        fs::write(
            self.dir(handle).join("exit.json"),
            serde_json::to_vec(&serde_json::json!({"code": code, "signal": signal}))?,
        )?;
        Ok(Liveness::Exited { code, signal })
    }
    fn terminate(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        let r = self.record(handle)?;
        for (signal, grace) in [(libc::SIGTERM, 2_000u64), (libc::SIGKILL, 5_000)] {
            if self.probe(handle)? != Liveness::Running {
                break;
            }
            unsafe { libc::kill(r.pid, signal) };
            let start = Instant::now();
            while self.probe(handle)? == Liveness::Running
                && start.elapsed() < Duration::from_millis(grace)
            {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let last = self.probe(handle)?;
        anyhow::ensure!(last != Liveness::Running, "rank survived SIGKILL");
        Ok(last)
    }
    fn retained_result(&mut self, handle: &SupervisorHandle) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let outbox = self.dir(handle).join("outbox");
        match (
            fs::read(outbox.join("envelope.json")),
            fs::read(outbox.join("result.bin")),
        ) {
            (Ok(e), Ok(b)) => Ok(Some((e, b))),
            _ => Ok(None),
        }
    }
}

// ------------------------------------------------------------------------ harness

struct Harness {
    _t: tempfile::TempDir,
    root: PathBuf,
    dir: PathBuf,
    keys: Keys,
    contract: BoundedRunContract,
    sha: String,
    socket: PathBuf,
    modes: BTreeMap<u32, &'static str>,
}
impl Harness {
    fn new(modes: [&'static str; 3]) -> Self {
        let (t, root) = tempdir();
        write_closure(&root);
        let keys = keys(3);
        let now = now_ms();
        let mut contract = contract(now, 3, &keys);
        // A run name unique to this harness, so concurrent fixtures started in the same
        // millisecond never share plan digests, dispatch ids or rank command lines.
        let unique = root.file_name().unwrap().to_string_lossy();
        contract.run = format!("w7-fixture-{}", unique.trim_start_matches('.'));
        resign(&mut contract, &keys);
        let dir = root.join("run");
        let sha = controller::plan(&dir, &bytes(&contract), now, None)
            .unwrap()
            .plan_sha256;
        Self {
            socket: root.join("control.sock"),
            modes: modes
                .iter()
                .enumerate()
                .map(|(i, m)| (i as u32, *m))
                .collect(),
            _t: t,
            root,
            dir,
            keys,
            contract,
            sha,
        }
    }
    fn supervisor(&self) -> ProcessSupervisor {
        ProcessSupervisor {
            state: self.root.join("supervisor"),
            socket: self.socket.clone(),
            plan: self.dir.join("plan.json"),
            root: self.root.clone(),
            modes: self.modes.clone(),
        }
    }
    fn open(&self) -> Coordinator<ProcessSupervisor> {
        self.open_at(now_ms())
    }
    fn open_at(&self, now: u64) -> Coordinator<ProcessSupervisor> {
        Coordinator::open(&self.dir, &self.sha, self.supervisor(), now).unwrap()
    }
    /// The earliest participant grant expiry; the whole group's window closes there.
    fn first_expiry(&self) -> u64 {
        self.contract
            .participants
            .iter()
            .map(|p| p.grant.body.expires_unix_ms)
            .min()
            .unwrap()
    }
    /// Every rank completes with a retained authenticated result, then the coordinator
    /// crashes before acknowledging any of them. Returns the still-open control streams.
    fn completed_then_crashed(&self) -> Vec<Connected> {
        let mut c = self.open();
        let mut connected = self.start(&mut c);
        self.wait_admitted();
        for id in 1..=3 {
            heartbeat(&mut connected, id);
        }
        let pids = self.pids();
        wait(|| pids.iter().all(|p| exited(*p)));
        drop(c);
        let pending = controller::status(&self.dir).unwrap();
        assert_eq!(
            (pending.outcome, pending.committed_updates_total),
            (None, 60_000)
        );
        connected
    }
    /// Preflight, launch every rank, authenticate the full group, persist the admission
    /// and only then deliver it.
    fn start(&self, c: &mut Coordinator<ProcessSupervisor>) -> Vec<Connected> {
        let now = now_ms();
        c.preflight(
            &serde_json::to_vec(&census(&self.contract, now)).unwrap(),
            now,
        )
        .unwrap();
        let mut rv = c
            .open_rendezvous(
                &self.keys.coordinator,
                "fixture-coordinator-boot",
                &rendezvous::fresh_nonce().unwrap(),
                now,
            )
            .unwrap();
        let _ = fs::remove_file(&self.socket);
        let listener = UnixListener::bind(&self.socket).unwrap();
        c.dispatch_all(now).unwrap();
        let (admission, mut connected) =
            rendezvous::gather_unix(&listener, &mut rv, &self.keys.coordinator, &now_ms).unwrap();
        let persisted = c.record_admission(admission, now_ms()).unwrap();
        c.deliver_admission(&persisted, &mut connected, now_ms())
            .unwrap();
        connected
    }
    fn pids(&self) -> Vec<i32> {
        let mut pids = Vec::new();
        for rank in 0..3 {
            let id = rendezvous::dispatch_id(&self.sha, 1, rank).unwrap();
            if let Ok(b) = fs::read(self.root.join("supervisor").join(id).join("process.json")) {
                pids.push(serde_json::from_slice::<ProcessRecord>(&b).unwrap().pid);
            }
        }
        pids
    }
    fn wait_admitted(&self) {
        wait(|| {
            (0..3).all(|rank| {
                let id = rendezvous::dispatch_id(&self.sha, 1, rank).unwrap();
                self.root
                    .join("supervisor")
                    .join(id)
                    .join("outbox/admitted")
                    .exists()
            })
        });
    }
    /// No rank process of this run survives, reaped or otherwise.
    fn assert_no_orphans(&self) {
        let ids: Vec<String> = (0..3)
            .map(|r| rendezvous::dispatch_id(&self.sha, 1, r).unwrap())
            .collect();
        for pid in self.pids() {
            assert!(
                proc_stat(pid).is_none(),
                "rank process {pid} survives (or is unreaped)"
            );
        }
        for entry in fs::read_dir("/proc").unwrap().flatten() {
            if let Ok(cmdline) = fs::read(entry.path().join("cmdline")) {
                let cmdline = String::from_utf8_lossy(&cmdline);
                for id in &ids {
                    assert!(
                        !cmdline.contains(id.as_str()),
                        "orphan rank process {:?}",
                        entry.path()
                    );
                }
            }
        }
    }
    fn dispositions(&self) -> Vec<Option<Disposition>> {
        controller::status(&self.dir)
            .unwrap()
            .ranks
            .iter()
            .map(|r| r.disposition)
            .collect()
    }
}
fn wait(mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "fixture wait timed out"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn heartbeat(connected: &mut [Connected], id: u64) {
    let frame = serde_json::to_vec(&ControlFrame::Heartbeat { delivery_id: id }).unwrap();
    for c in connected.iter_mut() {
        let _ = rendezvous::write_frame(&mut c.stream, &frame, 64 * 1024);
    }
}
/// Heartbeat and poll until the attempt is terminal.
fn supervise(
    c: &mut Coordinator<ProcessSupervisor>,
    connected: &mut [Connected],
    next: &mut u64,
) -> controller::StatusReport {
    let start = Instant::now();
    loop {
        *next += 1;
        heartbeat(connected, *next);
        let s = c.poll(now_ms()).unwrap();
        if s.outcome.is_some() {
            return s;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "supervision timed out"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}
fn journal_contains(dir: &Path, needle: &str) -> bool {
    fs::read_dir(dir.join("journal"))
        .unwrap()
        .flatten()
        .any(|e| {
            fs::read_to_string(e.path())
                .unwrap_or_default()
                .contains(needle)
        })
}
fn journal_count(dir: &Path, needle: &str) -> usize {
    fs::read_dir(dir.join("journal"))
        .unwrap()
        .flatten()
        .filter(|e| {
            fs::read_to_string(e.path())
                .unwrap_or_default()
                .contains(needle)
        })
        .count()
}
/// Simulate a coordinator crash right after the record containing `needle` was published:
/// the journal is append-only, one record at a time, so a crash leaves exactly a prefix.
fn truncate_journal_after(dir: &Path, needle: &str) -> usize {
    let jdir = dir.join("journal");
    let mut records: Vec<PathBuf> = fs::read_dir(&jdir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    records.sort();
    let at = records
        .iter()
        .position(|p| fs::read_to_string(p).unwrap().contains(needle))
        .expect("record present");
    for later in &records[at + 1..] {
        fs::remove_file(later).unwrap();
    }
    records.len() - at - 1
}
fn exited(pid: i32) -> bool {
    proc_stat(pid).is_none_or(|(state, _)| state == 'Z')
}

// ------------------------------------------------------------------------ scenarios

#[test]
fn positive_control_commits_the_whole_group_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    let mut next = 0;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Committed));
    assert_eq!(s.committed_updates_total, 60_003);
    assert!(s.ranks.iter().all(|r| r.state == RankState::Committed));
    let journal = s.journal_records;
    drop(c);
    h.assert_no_orphans();
    // Reopening a terminal run neither restarts nor recommits anything.
    let mut again = h.open();
    assert!(!again.needs_reconcile());
    let s = again.reconcile(now_ms()).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_003)
    );
    drop(again);
    let v = controller::verify(&h.dir, Some(&h.sha)).unwrap();
    assert_eq!(v.results_verified, 3);
    assert!(!v.numerical_work_started);
}

#[test]
fn killed_rank_drains_the_whole_group_with_zero_commits() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    let mut next = 0;
    for _ in 0..3 {
        next += 1;
        heartbeat(&mut connected, next);
        assert!(c.poll(now_ms()).unwrap().outcome.is_none());
    }
    let victim = h.pids()[1];
    unsafe { libc::kill(victim, libc::SIGKILL) };
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        h.dispositions(),
        vec![
            Some(Disposition::Drained),
            Some(Disposition::Failed),
            Some(Disposition::Drained)
        ]
    );
    assert!(!h.dir.join("results").exists(), "no result was committed");
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn severed_control_stops_every_rank_and_drains_the_group() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let connected = h.start(&mut c);
    h.wait_admitted();
    let pids = h.pids();
    // Sever control: stop heartbeats while holding the channels open. Each rank stops
    // itself within the frozen heartbeat timeout; none is killed by the fixture.
    wait(|| pids.iter().all(|p| exited(*p)));
    let s = c.poll(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    assert!(
        journal_contains(&h.dir, "\"cause\":\"control_lost\""),
        "drain cause must be the lost control channel"
    );
    drop(connected);
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn coordinator_restart_reconciles_and_drains_with_original_deadline_and_counters() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    drop(c); // coordinator crash: no drain ran, ranks keep their channels
    let s = controller::status(&h.dir).unwrap();
    assert!(!s.coordinator_live);
    assert!(s.ranks.iter().all(|r| r.state == RankState::Unknown));
    heartbeat(&mut connected, 2);
    let mut c = h.open();
    assert!(c.needs_reconcile());
    assert!(c.poll(now_ms()).is_err());
    assert!(c.record_dispatch_intent(0, now_ms()).is_err());
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(
        s.original_deadline_unix_ms,
        h.contract.original_deadline_unix_ms
    );
    assert_eq!(
        (s.committed_updates_at_start, s.committed_updates_total),
        (60_000, 60_000)
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    drop(c);
    drop(connected);
    h.assert_no_orphans();

    // A successor attempt keeps the run, deadline, topology and committed counters.
    let outcome = controller::attempt_outcome(&h.dir, &h.sha).unwrap();
    assert!(outcome.terminal);
    let mut next = h.contract.clone();
    next.attempt = 2;
    next.predecessor = Some(Predecessor {
        plan_sha256: h.sha.clone(),
        attempt: 1,
        journal_tip_sha256: outcome.journal_tip_sha256.clone(),
        committed_updates: outcome.committed_updates_total,
    });
    resign(&mut next, &h.keys);
    let successor = h.root.join("attempt-2");
    let now = now_ms();
    assert!(controller::plan(&successor, &bytes(&next), now, None).is_err());
    assert!(!successor.exists());
    for (name, edit) in [
        (
            "renewed deadline",
            Box::new(|c: &mut BoundedRunContract| c.original_deadline_unix_ms += 60_000)
                as Box<dyn Fn(&mut BoundedRunContract)>,
        ),
        (
            "topology change",
            Box::new(|c: &mut BoundedRunContract| {
                c.participants[2].device.as_mut().unwrap().uuid = "GPU-fixture-9".into()
            }),
        ),
        (
            "counter reset",
            Box::new(|c: &mut BoundedRunContract| {
                c.counters.committed_updates_at_start = 0;
                c.predecessor.as_mut().unwrap().committed_updates = 0;
            }),
        ),
    ] {
        let mut bad = next.clone();
        edit(&mut bad);
        resign(&mut bad, &h.keys);
        let refusal = BoundedRunContract::parse(&bytes(&bad))
            .and_then(|b| b.validate_successor(&h.contract, &outcome));
        assert!(
            matches!(refusal, Err(PlanRefusal::Invalid(_))),
            "{name}: {refusal:?}"
        );
        assert!(
            controller::plan(&successor, &bytes(&bad), now, Some(&h.dir)).is_err(),
            "{name}"
        );
        assert!(!successor.exists(), "{name}");
    }
    let planned = controller::plan(&successor, &bytes(&next), now, Some(&h.dir)).unwrap();
    let s = controller::status(&successor).unwrap();
    assert_eq!(
        s.original_deadline_unix_ms,
        h.contract.original_deadline_unix_ms
    );
    assert_eq!(s.committed_updates_at_start, 60_000);
    assert_eq!(s.plan_sha256, planned.plan_sha256);
}

#[test]
fn restart_commits_a_complete_unacknowledged_group_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    for id in 1..=3 {
        heartbeat(&mut connected, id);
    }
    let pids = h.pids();
    wait(|| pids.iter().all(|p| exited(*p)));
    drop(c); // crashed before acknowledging any completion
    let pending = controller::status(&h.dir).unwrap();
    assert_eq!(pending.committed_updates_total, 60_000);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Committed));
    assert_eq!(s.committed_updates_total, 60_003);
    let journal = s.journal_records;
    drop(c);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_003)
    );
    drop(c);
    drop(connected);
    let commits = (1..=journal)
        .filter(|i| {
            fs::read_to_string(h.dir.join(format!("journal/{i:08}.json")))
                .unwrap()
                .contains("\"event\":\"group_committed\"")
        })
        .count();
    assert_eq!(commits, 1);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}

#[test]
fn restart_distinguishes_unacknowledged_completion_from_running_work() {
    let h = Harness::new(["complete", "complete", "hold"]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    for id in 1..=3 {
        heartbeat(&mut connected, id);
    }
    let pids = h.pids();
    wait(|| exited(pids[0]) && exited(pids[1]));
    drop(c);
    heartbeat(&mut connected, 4);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        h.dispositions(),
        vec![
            Some(Disposition::CompletedUnacknowledged),
            Some(Disposition::CompletedUnacknowledged),
            Some(Disposition::Drained)
        ]
    );
    assert!(
        !h.dir.join("results").exists(),
        "a partial group commits nothing"
    );
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn restart_before_admission_distinguishes_never_started_from_unrecorded_launch() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let now = now_ms();
    c.preflight(&serde_json::to_vec(&census(&h.contract, now)).unwrap(), now)
        .unwrap();
    let listener = UnixListener::bind(&h.socket).unwrap();
    // rank 0: journaled launch; rank 1: spawned but the launch record was never
    // written (crash window); rank 2: dispatch intent only.
    c.record_dispatch_intent(0, now).unwrap();
    c.launch(0, now).unwrap();
    let id1 = c.record_dispatch_intent(1, now).unwrap();
    let contract = c.contract().clone();
    let sha = c.plan_sha256().to_string();
    c.supervisor()
        .launch(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &contract,
            plan_sha256: &sha,
            global_rank: 1,
            dispatch_id: &id1,
        })
        .unwrap();
    c.record_dispatch_intent(2, now).unwrap();
    drop(c);
    let s = controller::status(&h.dir).unwrap();
    assert_eq!(
        s.ranks.iter().map(|r| r.state).collect::<Vec<_>>(),
        vec![RankState::Unknown; 3]
    );
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    drop(listener);
    assert_eq!(s.outcome, Some(Outcome::Drained));
    let d = h.dispositions();
    assert_eq!(
        d,
        vec![
            Some(Disposition::Drained),
            Some(Disposition::Drained),
            Some(Disposition::NeverStarted)
        ]
    );
    assert!(
        s.ranks[1].detail.contains("launch was never journaled"),
        "{}",
        s.ranks[1].detail
    );
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn cancellation_and_grant_expiry_drain_the_whole_group() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    let r = controller::cancel(&h.dir, &h.sha, "fixture cancellation", now_ms()).unwrap();
    assert!(r.delivered_to_live_coordinator);
    let mut next = 1;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    drop(c);
    drop(connected);
    h.assert_no_orphans();

    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    let expiry = h.contract.participants[0].grant.body.expires_unix_ms;
    let s = c.poll(expiry).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert!(journal_contains(&h.dir, "\"cause\":\"grant_expired\""));
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

// ------------------------------------- review of #111: grant expiry during recovery

/// Regression (independent review of #111, P1). `poll` drains the group once any
/// participant grant has expired, but `reconcile` adopted and committed retained results
/// after a restart without that gate: poll gave Drained/60 000 at expiry while reopen +
/// reconcile gave Committed/60 003. A restart must not widen the grant window.
#[test]
fn restart_at_grant_expiry_drains_retained_results_instead_of_committing() {
    let h = Harness::new(["complete"; 3]);
    let connected = h.completed_then_crashed();
    let expiry = h.first_expiry();
    assert!(expiry < h.contract.original_deadline_unix_ms);
    let mut c = h.open_at(expiry);
    assert!(c.needs_reconcile());
    let s = c.reconcile(expiry).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Drained), 60_000),
        "recovery bypassed grant expiry"
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::CompletedUnacknowledged)));
    assert!(journal_contains(&h.dir, "\"cause\":\"grant_expired\""));
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 0);
    assert_eq!(
        journal_count(&h.dir, "\"event\":\"result_acknowledged\""),
        0
    );
    assert!(!h.dir.join("results").exists(), "no result was adopted");
    // Reopening the terminal attempt later changes nothing.
    let journal = s.journal_records;
    drop(c);
    let mut again = h.open_at(expiry + 1);
    assert!(!again.needs_reconcile());
    let s = again.reconcile(expiry + 1).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_000)
    );
    drop(again);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

/// Positive control: the same crash reconciled one millisecond before the first grant
/// expiry commits the complete group exactly once.
#[test]
fn restart_just_before_execution_window_closes_commits_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let connected = h.completed_then_crashed();
    let inspected = h.open();
    let wall_end = inspected
        .journal()
        .iter()
        .filter_map(|r| match r.event {
            controller::Event::DispatchIntent { global_rank, .. } => Some(
                r.recorded_unix_ms
                    + h.contract
                        .participant(global_rank)
                        .unwrap()
                        .ceilings
                        .wall_ms,
            ),
            _ => None,
        })
        .min()
        .unwrap();
    drop(inspected);
    let before = h.first_expiry().min(wall_end) - 1;
    let mut c = h.open_at(before);
    let s = c.reconcile(before).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Committed), 60_003)
    );
    let journal = s.journal_records;
    drop(c);
    let mut again = h.open_at(h.first_expiry() + 1);
    let s = again.reconcile(h.first_expiry() + 1).unwrap();
    assert_eq!(
        (s.outcome, s.journal_records, s.committed_updates_total),
        (Some(Outcome::Committed), journal, 60_003)
    );
    drop(again);
    drop(connected);
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 1);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}

/// Positive control: a commit already journaled before the crash is finished after the
/// grant window has closed, never drained, and never journaled a second time.
#[test]
fn an_already_journaled_commit_is_finished_not_drained_after_expiry() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    let mut next = 0;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Committed));
    drop(c);
    // Crash immediately after the group_committed record was published.
    let removed = truncate_journal_after(&h.dir, "\"event\":\"group_committed\"");
    assert_eq!(removed, 4, "three dispositions and the terminal record");
    let s = controller::status(&h.dir).unwrap();
    assert_eq!((s.outcome, s.committed_updates_total), (None, 60_003));
    let later = h.first_expiry() + 1;
    let mut c = h.open_at(later);
    assert!(c.needs_reconcile());
    let s = c.reconcile(later).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Committed), 60_003)
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Committed)));
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 1);
    assert_eq!(journal_count(&h.dir, "\"event\":\"drain_started\""), 0);
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}
