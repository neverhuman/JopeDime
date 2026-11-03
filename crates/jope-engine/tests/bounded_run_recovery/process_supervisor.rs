use super::*;

// ------------------------------------------------ same-host process supervisor (fixture)

/// Fixture supervisor for same-host rank processes. Its records live on disk, like a
/// daemon's, so a restarted coordinator with a fresh supervisor instance can reconcile.
pub(super) struct ProcessSupervisor {
    pub(super) state: PathBuf,
    pub(super) socket: PathBuf,
    pub(super) plan: PathBuf,
    pub(super) root: PathBuf,
    pub(super) modes: BTreeMap<u32, &'static str>,
}
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct ProcessRecord {
    pub(super) pid: i32,
    pub(super) birth: u64,
    pub(super) boot: String,
}
fn boot_id() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .unwrap()
        .trim()
        .into()
}
/// `(state, birth)` of a live /proc entry.
pub(super) fn proc_stat(pid: i32) -> Option<(char, u64)> {
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
            .env(CHILD_EXIT_STATUS, dir.join("rank-exit-code"))
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
            let harness_code = libc::WEXITSTATUS(status);
            if harness_code == 0 {
                let rank_code: i32 = fs::read_to_string(self.dir(handle).join("rank-exit-code"))
                    .context("rank child harness succeeded without an exit-status record")?
                    .parse()
                    .context("invalid rank-child exit-status record")?;
                anyhow::ensure!(
                    (0..=255).contains(&rank_code),
                    "invalid rank exit code {rank_code}"
                );
                (Some(rank_code), None)
            } else {
                (Some(harness_code), None)
            }
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
