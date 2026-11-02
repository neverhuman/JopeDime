use super::*;

struct StopState {
    token: String,
    port: u16,
    shutdown: watch::Sender<bool>,
}

async fn request_stop(
    Extension(state): Extension<Arc<StopState>>,
    request: Request,
) -> impl IntoResponse {
    let expected_host = format!("127.0.0.1:{}", state.port);
    let headers = request.headers();
    let host_ok = single_header(headers, header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value == expected_host
                || value.eq_ignore_ascii_case(&format!("localhost:{}", state.port))
        });
    let uri_authority_ok = request.uri().authority().is_none_or(|authority| {
        authority.as_str() == expected_host
            || authority
                .as_str()
                .eq_ignore_ascii_case(&format!("localhost:{}", state.port))
    });
    let auth_ok = single_header(headers, header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .is_some_and(|(scheme, value)| {
            scheme.eq_ignore_ascii_case("Bearer")
                && bool::from(value.as_bytes().ct_eq(state.token.as_bytes()))
        });
    if headers.contains_key(header::ORIGIN) || !host_ok || !uri_authority_ok || !auth_ok {
        return StatusCode::UNAUTHORIZED;
    }
    let _ = state.shutdown.send(true);
    StatusCode::NO_CONTENT
}

fn single_header(headers: &HeaderMap, name: HeaderName) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

/// Start the shared engine in this process. Store recovery occurs after lock acquisition.
pub fn serve(idle_exit: Option<Duration>) -> Result<(), EngineError> {
    let home = jope_paths::jopedime_home();
    let engine = InProcessEngine::open(&home)?;
    serve_engine(home, engine, idle_exit)
}

/// Internal opt-in for the owner's frozen r782 Q1 evaluator. Authentication is
/// not quality qualification or permission to launch a sealed-data study.
#[cfg(feature = "research")]
pub fn serve_q1_qualification(release: &Path) -> Result<usize, EngineError> {
    let home = jope_paths::jopedime_home();
    let (engine, forwards) = open_q1_engine(&home, release)?;
    serve_engine(home, engine, None)?;
    Ok(forwards.load(std::sync::atomic::Ordering::Relaxed))
}

/// Research-only infrastructure retry: the repaired host retains the original
/// Q1 numerical worker bytes. This does not grant a study admission.
#[cfg(feature = "research")]
pub fn serve_q1_infra_retry(release: &Path, worker: &Path) -> Result<usize, EngineError> {
    let worker = q1_retry_worker(worker)?;
    let home = jope_paths::jopedime_home();
    let (engine, forwards) = open_q1_engine_with_worker(&home, release, &worker)?;
    serve_engine(home, engine, None)?;
    Ok(forwards.load(std::sync::atomic::Ordering::Relaxed))
}

#[cfg(feature = "research")]
fn q1_retry_worker(worker: &Path) -> Result<PathBuf, EngineError> {
    const EXPECTED: &str = "4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177";
    if !worker.is_absolute() || !std::fs::symlink_metadata(worker)?.file_type().is_file() {
        return Err(EngineError::Runtime(
            "Q1 retry worker must be an absolute regular file".into(),
        ));
    }
    let bytes = std::fs::read(worker)?;
    if jope_runtime::sha256_hex(&bytes) != EXPECTED {
        return Err(EngineError::Runtime(
            "Q1 retry numerical evaluator pin mismatch".into(),
        ));
    }
    Ok(worker.to_path_buf())
}

#[cfg(feature = "research")]
pub(crate) fn open_q1_engine(
    home: &Path,
    release: &Path,
) -> Result<(InProcessEngine, Arc<std::sync::atomic::AtomicUsize>), EngineError> {
    open_q1_engine_with_worker(home, release, &std::env::current_exe()?)
}

#[cfg(feature = "research")]
fn open_q1_engine_with_worker(
    home: &Path,
    release: &Path,
    worker: &Path,
) -> Result<(InProcessEngine, Arc<std::sync::atomic::AtomicUsize>), EngineError> {
    // Authenticate before lock acquisition, store creation or daemon publication.
    let bundle = crate::ranker::cuda::release::load_r782_release(release)
        .map_err(|error| EngineError::Runtime(format!("frozen r782 integrity: {error:#}")))?;
    let ranker = crate::run::learned::AdmittedRanker::r782(bundle)
        .map_err(|error| EngineError::Runtime(format!("frozen r782 admission: {error:#}")))?;
    let forwards = ranker.execution_counter();
    let engine = InProcessEngine::open_with_ranker(home, worker, ranker)?;
    Ok((engine, forwards))
}

#[cfg(all(test, feature = "research"))]
#[test]
fn q1_retry_worker_refuses_changed_bytes_and_symlinks() {
    let directory = tempfile::tempdir().unwrap();
    let worker = directory.path().join("worker");
    std::fs::write(&worker, b"synthetic unpinned executable").unwrap();
    assert!(q1_retry_worker(&worker).is_err());
    assert!(q1_retry_worker(Path::new("worker")).is_err());
    #[cfg(unix)]
    {
        let link = directory.path().join("worker-link");
        std::os::unix::fs::symlink(&worker, &link).unwrap();
        assert!(q1_retry_worker(&link).is_err());
    }
    assert!(!directory.path().join("daemon.json").exists());
}

fn serve_engine(
    home: PathBuf,
    engine: InProcessEngine,
    idle_exit: Option<Duration>,
) -> Result<(), EngineError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| EngineError::Runtime(error.to_string()))?;
    runtime.block_on(serve_locked(
        home,
        engine,
        idle_exit.unwrap_or(DEFAULT_IDLE_EXIT),
    ))
}

pub(crate) async fn serve_locked(
    home: PathBuf,
    engine: InProcessEngine,
    idle_exit: Duration,
) -> Result<(), EngineError> {
    let activity = Arc::clone(&engine.activity);
    serve_with_activity(home, engine, idle_exit, activity).await
}

pub(crate) async fn serve_with_activity(
    home: PathBuf,
    engine: InProcessEngine,
    idle_exit: Duration,
    activity: Arc<EngineActivity>,
) -> Result<(), EngineError> {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    let address = listener.local_addr()?;
    // The socket is bound before the descriptor is atomically published.
    let record = daemon_file::write_daemon_file(
        &home,
        format!("http://127.0.0.1:{}", address.port()),
        env!("CARGO_PKG_VERSION"),
        "1.0",
        chrono::Utc::now().to_rfc3339(),
    )?;
    let bearer = BearerToken::from_hex(&record.token)
        .map_err(|error| EngineError::Daemon(error.to_string()))?;
    let auth = LocalAuth::new(bearer, address.port())
        .map_err(|error| EngineError::Daemon(error.to_string()))?;
    let (shutdown, mut shutdown_rx) = watch::channel(false);
    let state = Arc::new(StopState {
        token: record.token.clone(),
        port: address.port(),
        shutdown,
    });
    let app: Router = engine
        .router_with_activity(auth, Arc::clone(&activity))?
        .route(STOP_PATH, post(request_stop))
        .layer(Extension(state))
        .layer(Extension(Arc::clone(&activity)));
    let tracked = TrackedListener {
        listener,
        activity: Arc::clone(&activity),
    };
    let server = axum::serve(tracked, app).with_graceful_shutdown(async move {
        let idle = tokio::time::interval(Duration::from_millis(100));
        tokio::pin!(idle);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() { break; }
                }
                _ = idle.tick() => {
                    if activity.idle_for(idle_exit) {
                        break;
                    }
                }
            }
        }
    });
    let result = server
        .await
        .map_err(|error| EngineError::Runtime(error.to_string()));
    let cleanup = daemon_file::remove_daemon_file(&home).map_err(EngineError::Io);
    cleanup?;
    result
}
