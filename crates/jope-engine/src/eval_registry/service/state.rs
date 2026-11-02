struct Pending {
    process: ProcessIdentity,
    send: SyncSender<(UnixStream, String)>,
}
pub(super) struct State {
    pub(super) native_fit_policy: Option<Arc<super::native_scq::policy::InstalledPolicy>>,
    pub(super) funding_installation: Option<super::funding_config::InstalledFunding>,
    pub(super) funding_active: AtomicUsize,
    pub(super) authority: Mutex<Authority>,
    pub(super) lease: Arc<Lease>,
    pub(super) dispatch_open: Mutex<bool>,
    pub(super) config: ServiceConfig,
    pub(super) identity: ServiceIdentity,
    pub(super) runtime: String,
    pub(super) pending: Mutex<BTreeMap<String, Pending>>,
    pub(super) requests: AtomicUsize,
    pub(super) worker_spawns: AtomicUsize,
    pub(super) committed: AtomicUsize,
    pub(super) reused: AtomicUsize,
    pub(super) active: AtomicUsize,
    pub(super) connections: AtomicUsize,
}
impl State {
    pub(super) fn dispatch_gate(&self) -> Result<std::sync::MutexGuard<'_, bool>> {
        let mut gate = lock(&self.dispatch_open)?;
        ensure!(*gate, "authority dispatch fenced");
        if let Err(error) = self
            .lease
            .verify()
            .and_then(|_| check_socket(&self.identity, &self.config.socket))
        {
            *gate = false;
            return Err(error);
        }
        Ok(gate)
    }
    pub(super) fn fence(&self) {
        if let Ok(mut gate) = self.dispatch_open.lock() {
            *gate = false;
        }
    }
    pub(super) fn stats(&self) -> ServiceStats {
        ServiceStats {
            authority_opens: 1,
            requests: self.requests.load(Ordering::SeqCst),
            worker_spawns: self.worker_spawns.load(Ordering::SeqCst),
            committed: self.committed.load(Ordering::SeqCst),
            reused: self.reused.load(Ordering::SeqCst),
            active: self.active.load(Ordering::SeqCst),
        }
    }
}

pub(super) struct Lease {
    file: File,
    path: PathBuf,
    identity: (u64, u64),
}
impl Lease {
    pub(super) fn new(root: &Path) -> Result<Self> {
        let path = root.join("native-service.lock");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)?;
        ensure!(
            file.try_lock().is_ok(),
            "another native authority service owns custody"
        );
        let m = file.metadata()?;
        Ok(Self {
            file,
            path,
            identity: (m.dev(), m.ino()),
        })
    }
    pub(super) fn verify(&self) -> Result<()> {
        let m = fs::symlink_metadata(&self.path)?;
        ensure!(
            m.is_file() && (m.dev(), m.ino()) == self.identity,
            "service lease replaced"
        );
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub use runtime::Service;

struct CountGuard<'a> {
    counter: &'a AtomicUsize,
}
impl Drop for CountGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

fn request_monotonic_ns() -> Result<u64> {
    crate::host_os::monotonic_ns().context("request monotonic clock unavailable")
}
fn remaining_request(deadline: u64) -> Result<Duration> {
    let remaining = deadline
        .checked_sub(request_monotonic_ns()?)
        .filter(|n| *n > 0)
        .context("original request deadline expired")?;
    Ok(Duration::from_nanos(remaining))
}
