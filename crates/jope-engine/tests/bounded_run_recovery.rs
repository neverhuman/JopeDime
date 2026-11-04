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
const CHILD_EXIT_STATUS: &str = "W7C_RANK_EXIT_STATUS";

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
    // Preserve the simulated rank's logical status without bypassing the test
    // harness. ProcessSupervisor consumes this only after the harness exits
    // successfully; a panic or failed assertion remains a harness failure.
    fs::write(
        std::env::var_os(CHILD_EXIT_STATUS).unwrap(),
        code.to_string(),
    )
    .unwrap();
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

#[path = "bounded_run_recovery/process_supervisor.rs"]
mod process_supervisor;
use process_supervisor::{proc_stat, ProcessRecord, ProcessSupervisor};

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

#[cfg(test)]
#[path = "bounded_run_recovery/scenarios.rs"]
mod scenarios;
