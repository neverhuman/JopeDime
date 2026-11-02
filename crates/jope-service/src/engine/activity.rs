use super::*;

impl InProcessEngine {
    pub fn open_default() -> Result<Self, EngineError> {
        Self::open(jope_paths::jopedime_home())
    }
    pub fn open(home: impl AsRef<Path>) -> Result<Self, EngineError> {
        Self::open_with_worker(home, std::env::current_exe()?)
    }

    /// Trusted embeddings supply their authenticated appliance executable, which
    /// implements the existing worker protocol. This is not a remote API input.
    pub fn open_with_worker(
        home: impl AsRef<Path>,
        executable: impl AsRef<Path>,
    ) -> Result<Self, EngineError> {
        #[cfg(unix)]
        let ranker = crate::proposal::from_self().map_err(|error| {
            EngineError::Runtime(format!("embedded proposal integrity: {error:#}"))
        })?;
        #[cfg(not(unix))]
        let ranker: Option<crate::run::learned::AdmittedRanker> = None;
        let home = home.as_ref();
        let lock = Arc::new(EngineLock::acquire(home)?);
        let store = Arc::new(Store::open(home)?);
        Ok(Self {
            store,
            home: home.canonicalize()?,
            executable: executable.as_ref().to_path_buf(),
            runs: Mutex::new(None),
            activity: Arc::new(EngineActivity {
                ranker: ranker.map(Arc::new),
                ..EngineActivity::default()
            }),
            _lock: lock,
        })
    }

    /// Explicit trusted-host admission for qualification of the signed frozen
    /// candidate. Ordinary engines and Auto guidance retain their quality gates.
    pub fn open_with_ranker(
        home: impl AsRef<Path>,
        executable: impl AsRef<Path>,
        ranker: crate::run::learned::AdmittedRanker,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::open_with_worker(home, executable)?;
        engine.activity = Arc::new(EngineActivity {
            ranker: Some(Arc::new(ranker)),
            ..EngineActivity::default()
        });
        Ok(engine)
    }

    /// The same store-bound router serves embedded clients and loopback HTTP.
    pub fn router(&self, auth: LocalAuth) -> Result<Router, EngineError> {
        self.router_with_activity(auth, Arc::clone(&self.activity))
    }

    pub(crate) fn router_with_activity(
        &self,
        auth: LocalAuth,
        activity: Arc<EngineActivity>,
    ) -> Result<Router, EngineError> {
        let mut runs = self
            .runs
            .lock()
            .map_err(|_| EngineError::Runtime("run service lock poisoned".into()))?;
        if runs.is_none() {
            let environment = std::env::vars().collect();
            let config = Config::load(
                &self.home.join("config.toml"),
                &Default::default(),
                &environment,
            )
            .map_err(|e| EngineError::Runtime(e.to_string()))?;
            *runs = Some(Arc::new(
                crate::http::runs::RunService::new(
                    Arc::clone(&self.store),
                    config,
                    self.home.clone(),
                    self.executable.clone(),
                    activity,
                    Arc::clone(&self._lock),
                )
                .map_err(|e| EngineError::Runtime(e.to_string()))?,
            ));
        }
        let service = Arc::clone(runs.as_ref().expect("run service initialized"));
        Ok(router::router_with_state(
            auth,
            DatasetState {
                store: self.store_handle(),
                config: service.config().clone(),
                home: self.home.clone(),
            },
            Some(service),
        ))
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub(crate) fn store_handle(&self) -> Arc<Store> {
        Arc::clone(&self.store)
    }
}

/// Tracks run lifetimes and accepted TCP clients for idle shutdown.
pub struct EngineActivity {
    counts: Mutex<ActivityCounts>,
    ranker: Option<Arc<crate::run::learned::AdmittedRanker>>,
}

struct ActivityCounts {
    active_runs: usize,
    connected_clients: usize,
    last_activity: tokio::time::Instant,
}

impl Default for EngineActivity {
    fn default() -> Self {
        Self {
            ranker: None,
            counts: Mutex::new(ActivityCounts {
                active_runs: 0,
                connected_clients: 0,
                last_activity: tokio::time::Instant::now(),
            }),
        }
    }
}

impl EngineActivity {
    pub(crate) fn admitted_ranker(&self) -> Option<&crate::run::learned::AdmittedRanker> {
        self.ranker.as_deref()
    }
    pub(crate) fn active_runs(&self) -> usize {
        self.counts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .active_runs
    }
    pub fn begin_run(self: &Arc<Self>) -> ActivityGuard {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.active_runs += 1;
        counts.last_activity = tokio::time::Instant::now();
        drop(counts);
        ActivityGuard {
            activity: Arc::clone(self),
            kind: ActivityKind::Run,
        }
    }

    pub(crate) fn connected_client(self: &Arc<Self>) -> ActivityGuard {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.connected_clients += 1;
        counts.last_activity = tokio::time::Instant::now();
        drop(counts);
        ActivityGuard {
            activity: Arc::clone(self),
            kind: ActivityKind::Client,
        }
    }

    pub(crate) fn idle_for(&self, duration: Duration) -> bool {
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        counts.active_runs == 0
            && counts.connected_clients == 0
            && counts.last_activity.elapsed() >= duration
    }
}

enum ActivityKind {
    Run,
    Client,
}

/// Decrements the associated activity count when the run or client ends.
pub struct ActivityGuard {
    activity: Arc<EngineActivity>,
    kind: ActivityKind,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        let mut counts = self
            .activity
            .counts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match self.kind {
            ActivityKind::Run => counts.active_runs = counts.active_runs.saturating_sub(1),
            ActivityKind::Client => {
                counts.connected_clients = counts.connected_clients.saturating_sub(1)
            }
        }
        counts.last_activity = tokio::time::Instant::now();
    }
}

pub(crate) struct TrackedListener {
    pub(crate) listener: TcpListener,
    pub(crate) activity: Arc<EngineActivity>,
}

struct TrackedStream {
    stream: TcpStream,
    _activity: ActivityGuard,
}

impl AsyncRead for TrackedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for TrackedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

impl axum::serve::Listener for TrackedListener {
    type Io = TrackedStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.listener.accept().await {
                Ok((stream, address)) => {
                    return (
                        TrackedStream {
                            stream,
                            _activity: self.activity.connected_client(),
                        },
                        address,
                    );
                }
                Err(error) => {
                    tokio::time::sleep(if error.kind() == io::ErrorKind::Interrupted {
                        Duration::ZERO
                    } else {
                        Duration::from_millis(100)
                    })
                    .await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}
