use super::*;

pub struct Service {
    listener: UnixListener,
    state: Arc<State>,
    lease: Arc<Lease>,
}
impl Service {
    /// Does not initialize authority storage or replace a stale socket. Startup
    /// refuses unresolved original authorized/running tasks until reconciliation.
    pub fn bind(mut config: ServiceConfig) -> Result<Self> {
        // This release has no independently qualified cross-executable client or
        // worker allowlist. Run its CPU build explicitly, retaining the existing
        // identical-release checks instead of inheriting a neural executable.
        ensure!(
            !cfg!(feature = "neural"),
            "CPU fit authority requires a non-neural evaluator build"
        );
        ensure!(
            config.worker_executable.is_absolute(),
            "CPU evaluator binding must be an absolute path"
        );
        config.worker_executable = fs::canonicalize(&config.worker_executable)?;
        let worker_metadata = fs::metadata(&config.worker_executable)?;
        ensure!(
            worker_metadata.is_file() && worker_metadata.permissions().mode() & 0o111 != 0,
            "CPU evaluator binding must name an executable regular file"
        );
        let executable = verified_executable(&ProcessIdentity::current()?)?;
        ensure!(
            sha(&fs::read(&config.worker_executable)?) == executable,
            "service and explicitly bound CPU evaluator executable differ"
        );
        ensure!(
            (1..=104).contains(&config.max_workers)
                && !config.fit_deadline.is_zero()
                && !config.handshake_timeout.is_zero(),
            "invalid bounded service configuration"
        );
        sandbox::check(&config.sandbox)?;
        #[cfg(not(test))]
        ensure!(
            config.event_witness.is_some(),
            "explicit independent event witness configuration required"
        );
        config.authority_root = fs::canonicalize(&config.authority_root)?;
        config.witness = fs::canonicalize(&config.witness)?;
        config.socket = fs::canonicalize(
            config
                .socket
                .parent()
                .context("service socket parent missing")?,
        )?
        .join(
            config
                .socket
                .file_name()
                .context("service socket name missing")?,
        );
        let lease = Arc::new(Lease::new(&config.authority_root)?);
        let mut authority = Authority::open(&config.authority_root, &config.witness)?;
        if let Some(witness) = config.event_witness.clone() {
            authority.attach_event_witness(witness)?;
        }
        #[cfg(test)]
        ensure!(
            config.funding_fixture.is_none() || config.funding_config.is_none(),
            "private funding fixture cannot override an installed configuration"
        );
        let native_fit_policy = config
            .native_fit_policy
            .as_deref()
            .map(super::super::native_scq::policy::InstalledPolicy::load)
            .transpose()?
            .map(Arc::new);
        if let Some(policy) = &native_fit_policy {
            ensure!(
                policy.signed.body.authority_root == config.authority_root
                    && policy.signed.body.witness == config.witness,
                "native installed grant binds another authority/witness path"
            );
        }
        let funding_installation = config
            .funding_config
            .as_deref()
            .map(super::super::funding_config::InstalledFunding::load)
            .transpose()?;
        #[cfg(test)]
        let funding_installation = funding_installation.or_else(|| config.funding_fixture.clone());
        if let Some(installation) = &funding_installation {
            authority.attach_funding(installation.clone())?;
        }
        authority.require_reconciled_service_start()?;
        let runtime = current_runtime_digest(&config.contract)?;
        ensure!(
            !config.socket.exists() && !identity_path(&config.socket).exists(),
            "stale service socket/identity requires explicit reconciliation"
        );
        let output_dir = config.authority_root.join("worker-output");
        if !output_dir.exists() {
            fs::create_dir(&output_dir)?;
        }
        ensure!(
            fs::symlink_metadata(&output_dir)?.is_dir()
                && !fs::symlink_metadata(&output_dir)?.file_type().is_symlink(),
            "invalid worker output custody"
        );
        let listener = UnixListener::bind(&config.socket)?;
        fs::set_permissions(&config.socket, fs::Permissions::from_mode(0o600))?;
        listener.set_nonblocking(true)?;
        let meta = fs::symlink_metadata(&config.socket)?;
        let identity = ServiceIdentity {
            schema: "parallel.fit_service.v1".into(),
            process: ProcessIdentity::current()?,
            executable_sha256: executable,
            authority_instance: authority.instance().into(),
            authority_root: config.authority_root.clone(),
            witness: config.witness.clone(),
            manifest_sha256: sha(&fs::read(config.authority_root.join("authority.json"))?),
            socket_device: meta.dev(),
            socket_inode: meta.ino(),
        };
        write_new(
            &identity_path(&config.socket),
            &serde_json::to_vec(&identity)?,
        )?;
        let state = Arc::new(State {
            funding_installation,
            native_fit_policy,
            funding_active: AtomicUsize::new(0),
            authority: Mutex::new(authority),
            lease: lease.clone(),
            dispatch_open: Mutex::new(true),
            config,
            identity,
            runtime,
            pending: Mutex::new(BTreeMap::new()),
            requests: AtomicUsize::new(0),
            worker_spawns: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            connections: AtomicUsize::new(0),
        });
        Ok(Self {
            listener,
            state,
            lease,
        })
    }
    pub fn socket(&self) -> &Path {
        &self.state.config.socket
    }
    /// A graceful stop drains accepted children and commits their original
    /// receipts. Disconnected submitters do not cancel their supervisor threads.
    pub fn run(self, stop: Arc<AtomicBool>) -> Result<ServiceStats> {
        let mut handlers = Vec::new();
        let mut incident: Option<anyhow::Error> = None;
        while !stop.load(Ordering::SeqCst)
            || self.state.active.load(Ordering::SeqCst) > 0
            || self.state.connections.load(Ordering::SeqCst) > 0
        {
            if let Err(error) = self
                .lease
                .verify()
                .and_then(|_| check_socket(&self.state.identity, &self.state.config.socket))
            {
                self.state.fence();
                incident = Some(error);
                break;
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let state = self.state.clone();
                    let stopping = stop.load(Ordering::SeqCst);
                    // During drain only original workers awaiting their one
                    // consume handshake may create a handler. Read/submit floods
                    // must not keep the service alive after stop.
                    if stopping {
                        let owned_pending = authenticated_peer(&stream).ok().is_some_and(|peer| {
                            state.pending.lock().is_ok_and(|pending| {
                                pending.values().any(|slot| slot.process == peer)
                            })
                        });
                        if !owned_pending {
                            drop(stream);
                            continue;
                        }
                    }
                    if state.connections.fetch_add(1, Ordering::SeqCst)
                        >= state.config.max_workers * 2 + 16
                    {
                        state.connections.fetch_sub(1, Ordering::SeqCst);
                        drop(stream);
                        continue;
                    }
                    handlers.push(std::thread::spawn(move|| {
                        let _count=CountGuard{counter:&state.connections};
                        let mut stream=stream;
                        if let Err(error)=connection(&state,&mut stream,stopping) {let _=send_json(&mut stream,&json!({"ok":false,"error":format!("{error:#}"),"retry_permitted":false}),Duration::from_secs(5));}
                    }));
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(2))
                }
                Err(error) => {
                    self.state.fence();
                    incident = Some(error.into());
                    break;
                }
            }
            let mut i = 0;
            while i < handlers.len() {
                if handlers[i].is_finished() {
                    let handle = handlers.swap_remove(i);
                    if handle.join().is_err() {
                        incident = Some(anyhow::anyhow!("service handler panicked"));
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            if incident.is_some() {
                self.state.fence();
                break;
            }
        }
        for handler in handlers {
            if handler.join().is_err() {
                incident = Some(anyhow::anyhow!("service handler panicked during drain"));
            }
        }
        if let Some(error) = incident {
            return Err(error);
        }
        let stats = self.state.stats();
        check_socket(&self.state.identity, &self.state.config.socket)?;
        fs::remove_file(&self.state.config.socket)?;
        fs::remove_file(identity_path(&self.state.config.socket))?;
        Ok(stats)
    }
}
