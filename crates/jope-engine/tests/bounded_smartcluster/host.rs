use super::*;

/// The coordinator's listen address, reserved for one [`Host`].
///
/// The agents learn the address from their supervision files before the coordinator binds
/// it, so it is chosen first. It used to be `127.0.0.1:0` bound and closed at once: from then
/// until the coordinator bound it, after the agents had started, the kernel could hand the
/// number to any process's `bind(:0)` (another fixture's reservation or agent) or autobound
/// `connect`, and the coordinator's bind then failed. The reservation closes those windows:
/// - the port lies outside the kernel's ephemeral range (`ip_local_port_range`), so no
///   `bind(:0)` or `connect` of any process is ever given it;
/// - a lock named after the port in the network namespace's abstract Unix socket namespace,
///   held for the host's whole life, keeps every other reservation off it: in this process,
///   in any other test process on the same network namespace, whatever its `TMPDIR`; the
///   kernel drops it when the holder exits;
/// - a bound, not listening, TCP socket without `SO_REUSEADDR` holds the address itself
///   until [`ListenPort::release`], immediately before a coordinator starts: every other bind
///   fails, and a participant's connection is refused exactly as if nothing were bound.
///
/// What remains is a process that binds this very number explicitly, bypassing the
/// reservation, between that release and the coordinator's bind, or between a killed
/// coordinator and its restart. The coordinator's bind then fails, and its error, naming the
/// address, is in the assertion message (`Host::coordinated`).
pub(crate) struct ListenPort {
    pub(crate) address: SocketAddr,
    pub(crate) _lock: std::os::unix::net::UnixDatagram,
    pub(crate) held: std::sync::Mutex<Option<std::os::fd::OwnedFd>>,
}
impl ListenPort {
    fn reserve() -> Self {
        use std::os::{linux::net::SocketAddrExt, unix::net};
        let range = fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range").unwrap();
        let (low, high) = match range.split_whitespace().collect::<Vec<_>>()[..] {
            [low, high] => (low.parse::<u16>().unwrap(), high.parse::<u16>().unwrap()),
            _ => panic!("unreadable ip_local_port_range: {range:?}"),
        };
        // Below the ephemeral range by at most 8192, or above it; never a privileged port.
        let candidates: Vec<u16> = (1024..=u16::MAX)
            .filter(|p| (*p < low && *p >= low.saturating_sub(8192)) || *p > high)
            .collect();
        assert!(
            !candidates.is_empty(),
            "no port outside the ephemeral range {low}-{high}"
        );
        // Start where concurrent processes are unlikely to meet; every candidate is tried.
        let start = (std::process::id() as usize).wrapping_mul(7919)
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as usize;
        for i in 0..candidates.len() {
            let port = candidates[(start + i) % candidates.len()];
            let name = format!("jope-test-loopback-port-{port}");
            let Ok(lock) = net::SocketAddr::from_abstract_name(name.as_bytes())
                .and_then(|a| net::UnixDatagram::bind_addr(&a))
            else {
                continue;
            };
            if let Ok(held) = hold_loopback_port(port) {
                return Self {
                    address: SocketAddr::from(([127, 0, 0, 1], port)),
                    _lock: lock,
                    held: std::sync::Mutex::new(Some(held)),
                };
            }
        }
        panic!(
            "no loopback port outside the ephemeral range {low}-{high} is free: {} tried",
            candidates.len()
        )
    }
    /// Let a coordinator that is about to start bind the address. The lock stays held.
    fn release(&self) {
        self.held.lock().unwrap().take();
    }
}
/// One rank's supervision: its own daemon and its own agent process.
pub(crate) struct RankHost {
    pub(crate) agent: Child,
    /// Kept open so the agent's stdout never breaks.
    pub(crate) _agent_stdout: BufReader<ChildStdout>,
    pub(crate) address: SocketAddr,
    pub(crate) supervision: HostSupervision,
    pub(crate) scqd: Scqd,
}
impl Drop for RankHost {
    fn drop(&mut self) {
        unsafe { libc::kill(self.agent.id() as i32, libc::SIGKILL) };
        let _ = self.agent.wait();
    }
}

/// Three CPU ranks of one host, each supervised through its own daemon and agent.
pub(crate) struct Host {
    pub(crate) _t: tempfile::TempDir,
    pub(crate) root: PathBuf,
    pub(crate) run_dir: PathBuf,
    pub(crate) sha: String,
    pub(crate) contract: BoundedRunContract,
    pub(crate) census: PathBuf,
    pub(crate) listen: ListenPort,
    pub(crate) agents_file: PathBuf,
    pub(crate) ranks: Vec<RankHost>,
}
impl Host {
    pub(crate) fn new(planned_updates: u64) -> Self {
        let (t, root) = tempdir();
        let participant = participant_binary();
        write_closure(&root);
        // This host's installation, certificate and keys.
        let cert = current("fixture-coordinator-host", 31);
        let etc = root.join("etc");
        fs::create_dir_all(&etc).unwrap();
        fs::write(etc.join("host.der"), &cert.der).unwrap();
        write_private(&etc.join("host.key"), &cert.pkcs8);
        for (name, bytes) in [
            ("libjope-runtime.so", b"fixture runtime".as_slice()),
            ("compiler", b"fixture compiler"),
            ("worker.json", b"{\"fixture\":\"policy\"}"),
        ] {
            fs::write(etc.join(name), bytes).unwrap();
        }
        let k = keys(RANKS as usize);
        write_private(&etc.join("coordinator.key"), &pkcs8([7; 32]));
        // One daemon per rank; each rank's installation pins its own daemon (W7-D1).
        let daemons: Vec<Scqd> = (0..RANKS)
            .map(|rank| Scqd::start(&root.join(format!("rank-{rank}/scqd"))))
            .collect();
        let pin = |scqd: &Scqd| SmartClusterInstallation {
            daemon: DaemonPin {
                socket: scqd.socket.clone(),
                uid: unsafe { libc::geteuid() },
                executable_sha256: scqd.executable_sha256.clone(),
            },
            service: SERVICE.into(),
        };
        let installation = Installation {
            schema: INSTALLATION_SCHEMA.into(),
            address: "127.0.0.1".into(),
            certificate: etc.join("host.der"),
            certificate_key: etc.join("host.key"),
            source_commit: sha(b"fixture-source")[..40].to_string(),
            executable: participant.clone(),
            runtime_libraries: vec![etc.join("libjope-runtime.so")],
            compiler: etc.join("compiler"),
            cuda_libraries: Vec::new(),
            nccl_libraries: Vec::new(),
            installed_policies: vec![etc.join("worker.json")],
            gpu_inventory: false,
            input_root: root.clone(),
            smartcluster: pin(&daemons[0]),
        };
        // The census this host's collector writes, observing the fixture host.
        let now = now_ms();
        let (census, _) = collect::collect(
            &installation.collect_spec(None),
            &fixture_host::with_commands(&NoGpu),
            now,
        )
        .unwrap();
        let census_path = root.join("census.json");
        collect::write_census(&census_path, &census).unwrap();
        let identity = census.hosts[0].identity.clone();
        // Three CPU ranks on this one host.
        let mut c = contract(now, RANKS as usize, &k);
        let unique = root.file_name().unwrap().to_string_lossy().to_string();
        c.run = format!("w7-scq-{}", unique.trim_start_matches('.'));
        c.coordinator_host = identity.host.clone();
        c.counters.planned_updates = planned_updates;
        c.work_census.clear();
        for i in 0..RANKS as usize {
            let p = &mut c.participants[i];
            p.host = identity.clone();
            p.local_rank = i as u32;
            p.device = None;
            p.ceilings = Ceilings {
                cpu_millicores: 250,
                ram_bytes: 128 * MIB,
                gpu_vram_bytes: None,
                wall_ms: 600_000,
            };
            for (kind, cpu, wall, updates) in [
                (WorkKind::Numerical, 60_000, 300_000, planned_updates),
                (WorkKind::Verification, 5_000, 30_000, 0),
                (WorkKind::ArtifactMovement, 5_000, 20_000, 0),
                (WorkKind::Cleanup, 1_000, 10_000, 0),
                (WorkKind::FailureAllowance, 10_000, 60_000, 0),
            ] {
                c.work_census.push(WorkItem {
                    id: format!(
                        "rank-{i}-{}",
                        serde_json::to_value(kind).unwrap().as_str().unwrap()
                    ),
                    global_rank: i as u32,
                    kind,
                    cpu_core_ms: cpu,
                    gpu_ms: 0,
                    wall_ms: wall,
                    peak_ram_bytes: 64 * MIB,
                    peak_vram_bytes: 0,
                    updates,
                });
            }
        }
        for i in 0..RANKS as usize {
            let p = c.participants[i].clone();
            c.participants[i].grant = grant(&c, &p, &k);
        }
        c.validate_structure().unwrap();
        let run_dir = root.join("run");
        let sha = controller::plan(&run_dir, &bytes(&c), now, None)
            .unwrap()
            .plan_sha256;
        let listen = ListenPort::reserve();
        // Per rank: its daemon, its supervision and its agent process.
        let mut ranks = Vec::new();
        for (rank, scqd) in (0..RANKS).zip(daemons) {
            let dir = root.join(format!("rank-{rank}"));
            let key = dir.join("participant.key");
            write_private(&key, &pkcs8(rank_seed(rank as usize)));
            let installation_path = dir.join("installation.json");
            let own = Installation {
                smartcluster: pin(&scqd),
                ..installation.clone()
            };
            fs::write(&installation_path, serde_json::to_vec(&own).unwrap()).unwrap();
            let supervision = HostSupervision {
                host: identity.host.clone(),
                daemon: own.smartcluster.daemon.clone(),
                tenant: tenant(),
                project: PROJECT.into(),
                service: SERVICE.into(),
                plan: run_dir.join("plan.json"),
                state: dir.join("supervisor"),
                coordinator: listen.address.to_string(),
                installation: installation_path.clone(),
                participant_keys: BTreeMap::from([(rank, key)]),
            };
            let supervision_path = dir.join("supervision.json");
            fs::write(&supervision_path, serde_json::to_vec(&supervision).unwrap()).unwrap();
            let log = fs::File::create(dir.join("agent.log")).unwrap();
            let mut agent = Command::new(product_binary())
                .arg("agent")
                .args(["--plan", run_dir.join("plan.json").to_str().unwrap()])
                .args(["--plan-sha256", &sha])
                .args(["--host", &identity.host])
                .args(["--rank", &rank.to_string()])
                .args(["--listen", "127.0.0.1:0"])
                .args(["--certificate", etc.join("host.der").to_str().unwrap()])
                .args(["--certificate-key", etc.join("host.key").to_str().unwrap()])
                .args(["--supervision", supervision_path.to_str().unwrap()])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(log)
                .spawn()
                .unwrap();
            let mut stdout = BufReader::new(agent.stdout.take().unwrap());
            let mut ready = String::new();
            stdout.read_line(&mut ready).unwrap();
            let ready: serde_json::Value = serde_json::from_str(&ready).unwrap_or_else(|e| {
                panic!(
                    "agent {rank} did not start ({e}): {}",
                    fs::read_to_string(dir.join("agent.log")).unwrap_or_default()
                )
            });
            assert_eq!(ready["agent"], "listening");
            let address: SocketAddr = ready["address"].as_str().unwrap().parse().unwrap();
            ranks.push(RankHost {
                agent,
                _agent_stdout: stdout,
                address,
                supervision,
                scqd,
            });
        }
        let agents: BTreeMap<String, String> = ranks
            .iter()
            .enumerate()
            .map(|(r, h)| (r.to_string(), h.address.to_string()))
            .collect();
        let agents_file = root.join("agents.json");
        fs::write(&agents_file, serde_json::to_vec(&agents).unwrap()).unwrap();
        Self {
            _t: t,
            root,
            run_dir,
            sha,
            contract: c,
            census: census_path,
            listen,
            agents_file,
            ranks,
        }
    }
    /// The coordinator's arguments, for a coordinator about to start: this releases the held
    /// listen address so that it can bind it ([`ListenPort`]).
    pub(crate) fn args(&self) -> CoordinateArgs {
        self.listen.release();
        let etc = self.root.join("etc");
        CoordinateArgs {
            run_dir: self.run_dir.clone(),
            plan_sha256: self.sha.clone(),
            census: self.census.clone(),
            listen: self.listen.address,
            certificate: etc.join("host.der"),
            certificate_key: etc.join("host.key"),
            coordinator_key: etc.join("coordinator.key"),
            heartbeat: Duration::from_millis(100),
        }
    }
    /// The coordinator's supervisor: every rank through its own agent.
    pub(crate) fn supervisor(&self) -> RemoteSupervisor {
        let etc = self.root.join("etc");
        RemoteSupervisor::new(
            &self.contract,
            &self.sha,
            HostCredential::load(&etc.join("host.der"), &etc.join("host.key")).unwrap(),
            self.ranks
                .iter()
                .enumerate()
                .map(|(r, h)| (r as u32, h.address))
                .collect(),
        )
        .unwrap()
    }
    pub(crate) fn dispatch_dir(&self, rank: u32) -> PathBuf {
        self.ranks[rank as usize]
            .supervision
            .state
            .join(rendezvous::dispatch_id(&self.sha, 1, rank).unwrap())
    }
    pub(crate) fn wait_admitted(&self) {
        let start = Instant::now();
        while !(0..RANKS).all(|r| self.dispatch_dir(r).join("outbox/admitted").exists()) {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "ranks were not admitted:\n{}",
                self.diagnose()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }
    /// A coordinator's report, or a panic with its error and [`Host::diagnose`] taken while
    /// the daemons, agents and run directory still exist (they go when the host drops).
    pub(crate) fn coordinated(
        &self,
        what: &str,
        report: anyhow::Result<controller::StatusReport>,
    ) -> controller::StatusReport {
        report.unwrap_or_else(|e| panic!("{what} failed: {e:#}\n{}", self.diagnose()))
    }
    /// [`Host::coordinated`] for a coordinator run on its own thread, which may also panic.
    pub(crate) fn joined(
        &self,
        coordinator: thread::JoinHandle<anyhow::Result<controller::StatusReport>>,
    ) -> controller::StatusReport {
        match coordinator.join() {
            Ok(report) => self.coordinated("driver::coordinate", report),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("a non-text panic");
                panic!(
                    "the coordinator thread panicked: {message}\n{}",
                    self.diagnose()
                )
            }
        }
    }
    /// The coordinator's listen address, the journal events, every rank's outbox (including a
    /// participant's recorded refusal), each daemon's and agent's log tail, each daemon's jobs
    /// and each coordinator process's log tail, for assertion messages.
    pub(crate) fn diagnose(&self) -> String {
        let mut out = format!("coordinator listen {}\n", self.listen.address);
        if let Ok(entries) = fs::read_dir(self.run_dir.join("journal")) {
            let mut names: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            names.sort();
            for p in names {
                let v: serde_json::Value =
                    serde_json::from_slice(&fs::read(&p).unwrap_or_default()).unwrap_or_default();
                out += &format!("journal {}\n", v["event"]);
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for rank in 0..RANKS {
            let outbox = self.dispatch_dir(rank).join("outbox");
            let files: Vec<String> = fs::read_dir(&outbox)
                .map(|d| {
                    d.flatten()
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            out += &format!("rank {rank} outbox {files:?}\n");
            if let Ok(refusal) = fs::read_to_string(outbox.join("refusal.txt")) {
                out += &format!("rank {rank} refusal: {refusal}\n");
            }
            let h = &self.ranks[rank as usize];
            for (name, path) in [
                ("scqd", h.scqd.dir.join("scqd.log")),
                ("agent", self.root.join(format!("rank-{rank}/agent.log"))),
            ] {
                let log = fs::read_to_string(path).unwrap_or_default();
                let tail: Vec<&str> = log.lines().rev().take(10).collect();
                out += &format!("rank {rank} {name} log tail: {tail:?}\n");
            }
            let listed = runtime.block_on(async {
                jain_smartcluster::Client::new(h.supervision.daemon.socket.clone())
                    .list(jain_smartcluster::ListRequest::default())
                    .await
            });
            match listed {
                Ok(page) => {
                    for job in page.jobs {
                        out += &format!(
                            "rank {rank} job {} state {:?} attempts {} terminal {:?}\n",
                            job.id, job.state, job.attempt_count, job.terminal
                        );
                    }
                }
                Err(e) => out += &format!("rank {rank} scqd list failed: {e}\n"),
            }
        }
        // `coordinate_cli`'s logs, oldest first.
        let mut logs: Vec<PathBuf> = fs::read_dir(&self.root)
            .map(|d| {
                d.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .is_some_and(|n| n.to_string_lossy().starts_with("coordinate-"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        logs.sort();
        for path in logs {
            let log = fs::read_to_string(&path).unwrap_or_default();
            let tail: Vec<&str> = log.lines().rev().take(10).collect();
            out += &format!("{} tail: {tail:?}\n", path.display());
        }
        out
    }
    /// Live processes whose command line names one of this run's dispatches.
    pub(crate) fn rank_processes(&self) -> Vec<i32> {
        let ids: Vec<String> = (0..RANKS)
            .map(|r| rendezvous::dispatch_id(&self.sha, 1, r).unwrap())
            .collect();
        let mut pids = Vec::new();
        for entry in fs::read_dir("/proc").unwrap().flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<i32>() else {
                continue;
            };
            if let Ok(cmdline) = fs::read(entry.path().join("cmdline")) {
                let cmdline = String::from_utf8_lossy(&cmdline);
                let state = fs::read_to_string(entry.path().join("stat")).unwrap_or_default();
                let zombie = state
                    .rsplit_once(')')
                    .is_some_and(|(_, t)| t.trim_start().starts_with('Z'));
                if !zombie && ids.iter().any(|id| cmdline.contains(id.as_str())) {
                    pids.push(pid);
                }
            }
        }
        pids
    }
    pub(crate) fn assert_no_orphans(&self) {
        wait(Duration::from_secs(20), || self.rank_processes().is_empty());
    }
    pub(crate) fn journal_contains(&self, needle: &str) -> bool {
        fs::read_dir(self.run_dir.join("journal"))
            .unwrap()
            .flatten()
            .any(|e| {
                fs::read_to_string(e.path())
                    .unwrap_or_default()
                    .contains(needle)
            })
    }
}
