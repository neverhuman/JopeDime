use anyhow::{ensure, Context, Result};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, Request, State,
    },
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use feat_math::source_stream::Occurrence;
use jope_observatory::gp_source::{self, complete_line, Event, Store};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Seek, SeekFrom},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

#[derive(Parser)]
#[command(
    about = "Stream every captured GP occurrence as full source and report compiler-qualified token yield"
)]
struct Args {
    /// External directory supplied to producers as JOPE_GP_SOURCE_DIR.
    #[arg(long)]
    journal_dir: PathBuf,
    /// External durable source/event/check storage. Use one service per store.
    #[arg(long)]
    state_dir: PathBuf,
    #[arg(long, default_value = "127.0.0.1:9876")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 15)]
    compile_timeout_seconds: u64,
    #[arg(long, default_value_t = 1_000_000)]
    max_occurrences: usize,
    /// Drain current journals, check every source, print metrics and exit.
    #[arg(long)]
    once: bool,
}
#[derive(Clone)]
struct App {
    store: Arc<Mutex<Store>>,
    authority: String,
    log: PathBuf,
    clients: Arc<tokio::sync::Semaphore>,
}
#[derive(Default, Deserialize)]
struct Cursor {
    #[serde(default)]
    after: u64,
}

async fn boundary(State(app): State<App>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok());
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|h| h.to_str().ok());
    if host != Some(app.authority.as_str())
        || origin.is_some_and(|o| {
            o != format!("http://{}", app.authority) && o != format!("https://{}", app.authority)
        })
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    response
}
async fn metrics(State(app): State<App>) -> Result<Json<serde_json::Value>, StatusCode> {
    tokio::task::spawn_blocking(move || {
        app.store
            .lock()
            .map(|s| Json(s.snapshot()))
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
    })
    .await
    .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
}
async fn stream(
    State(app): State<App>,
    Query(cursor): Query<Cursor>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Ok(permit) = app.clients.clone().try_acquire_owned() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let store = app.store.clone();
    let offset = match tokio::task::spawn_blocking(move || {
        store
            .lock()
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?
            .offset_after(cursor.after)
            .map_err(|_| StatusCode::BAD_REQUEST)
    })
    .await
    {
        Ok(Ok(offset)) => offset,
        Ok(Err(status)) => return status.into_response(),
        Err(_) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    upgrade
        .max_message_size(1024)
        .max_frame_size(1024)
        .on_upgrade(move |socket| replay(socket, app.log, offset, permit))
}
async fn replay(
    mut socket: WebSocket,
    log: PathBuf,
    offset: u64,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    let Ok(file) = File::open(log) else {
        return;
    };
    let mut reader = BufReader::new(file);
    if reader.seek(SeekFrom::Start(offset)).is_err() {
        return;
    }
    loop {
        let mut read = tokio::task::spawn_blocking(move || {
            let result = complete_line(&mut reader, true);
            (reader, result)
        });
        // Continue processing control frames even while replaying a large retained backlog.
        let result = loop {
            tokio::select! {
                result = &mut read => break result,
                message = socket.recv() => match message {
                    Some(Ok(Message::Ping(bytes))) => {
                        if !matches!(tokio::time::timeout(Duration::from_secs(10), socket.send(Message::Pong(bytes))).await, Ok(Ok(()))) { return; }
                    },
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                    _ => {},
                }
            }
        };
        let Ok((returned, result)) = result else {
            return;
        };
        reader = returned;
        match result {
            Ok(Some(bytes)) => {
                let Ok(event) = serde_json::from_slice::<Event>(&bytes) else {
                    return;
                };
                if event.sequence > 0 {
                    let Ok(line) = String::from_utf8(bytes) else {
                        return;
                    };
                    if !matches!(
                        tokio::time::timeout(
                            Duration::from_secs(10),
                            socket.send(Message::Text(line.into()))
                        )
                        .await,
                        Ok(Ok(()))
                    ) {
                        return;
                    }
                }
            }
            Ok(None) => {
                tokio::select! {
                    message = socket.recv() => match message {
                        Some(Ok(Message::Ping(bytes))) => { if !matches!(tokio::time::timeout(Duration::from_secs(10), socket.send(Message::Pong(bytes))).await, Ok(Ok(()))) {return;} },
                        Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return,
                        _ => {},
                    },
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
            Err(_) => return,
        }
    }
}
fn router(app: App) -> Router {
    Router::new()
        .route("/api/v1/gp/source", get(stream))
        .route("/api/v1/gp/metrics", get(metrics))
        .layer(middleware::from_fn_with_state(app.clone(), boundary))
        .with_state(app)
}

const JOURNAL_BATCH_RECORDS: usize = 8;

fn collect(
    root: &Path,
    offsets: &mut BTreeMap<PathBuf, u64>,
    store: &Arc<Mutex<Store>>,
    limit: usize,
) -> Result<usize> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "jsonl") {
            ensure!(
                entry.file_type()?.is_file(),
                "journal must be a regular file, not a symlink"
            );
            paths.push(entry.path());
        }
    }
    ensure!(
        paths.len() <= 4096,
        "more than 4096 producer journals; archive completed runs explicitly"
    );
    paths.sort();
    let mut partial = Vec::new();
    let mut backlog = Vec::new();
    let mut consumed = 0;
    for path in paths {
        let offset = offsets.get(&path).copied().unwrap_or(0);
        let metadata = fs::metadata(&path)?;
        if metadata.len() == offset {
            continue;
        }
        ensure!(
            metadata.len() >= offset,
            "producer journal truncated; preserve it"
        );
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset))?;
        // A live producer may append faster than we emit. Bound its turn so every
        // other producer is visited even when this file never reaches EOF.
        for _ in 0..JOURNAL_BATCH_RECORDS {
            let Some(bytes) = complete_line(&mut reader, true)? else {
                if reader.get_ref().metadata()?.len() > reader.stream_position()? {
                    partial.push(
                        path.file_name()
                            .context("journal file name")?
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
                break;
            };
            let occurrence: Occurrence =
                serde_json::from_slice(&bytes).context("invalid complete producer record")?;
            let known = store
                .lock()
                .map_err(|_| anyhow::anyhow!("source store poisoned"))?
                .contains(&occurrence)?;
            if !known {
                ensure!(
                    store
                        .lock()
                        .map_err(|_| anyhow::anyhow!("store poisoned"))?
                        .occurrence_count()
                        < limit,
                    "occurrence storage limit reached; producer journals remain retained"
                );
                let context = format!("occurrence {} from {}", occurrence.id, path.display());
                let (artifacts, elapsed) =
                    gp_source::emit(&occurrence).with_context(|| context.clone())?;
                store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("source store poisoned"))?
                    .publish(occurrence, artifacts, elapsed)
                    .with_context(|| context.clone())?;
            }
            offsets.insert(path.clone(), reader.stream_position()?);
            consumed += 1;
        }
        let position = reader.stream_position()?;
        let length = reader.get_ref().metadata()?.len();
        ensure!(
            length >= position,
            "producer journal truncated during collection; preserve it"
        );
        if length > position {
            backlog.push(
                path.file_name()
                    .context("journal file name")?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    store
        .lock()
        .map_err(|_| anyhow::anyhow!("store poisoned"))?
        .observed_ingress(partial, backlog);
    Ok(consumed)
}
fn check_one(
    store: &Arc<Mutex<Store>>,
    timeout: u64,
    cache: &mut BTreeMap<(String, String), gp_source::Check>,
) -> Result<bool> {
    let next = store
        .lock()
        .map_err(|_| anyhow::anyhow!("source store poisoned"))?
        .next_check();
    let Some((id, artifact, root)) = next else {
        return Ok(false);
    };
    let key = (
        artifact.language.clone(),
        artifact.sha256.clone().context("missing check hash")?,
    );
    let checked = if let Some(previous) = cache.get(&key) {
        let mut cached = previous.clone();
        cached.elapsed_ns = 0;
        cached
            .tool
            .push_str("; reused identical source check in this process");
        cached
    } else {
        let checked = gp_source::check(&root, &artifact, timeout)?;
        if checked.status == gp_source::Status::Passed {
            if cache.len() >= 4096 {
                cache.pop_first();
            }
            cache.insert(key, checked.clone());
        }
        checked
    };
    store
        .lock()
        .map_err(|_| anyhow::anyhow!("source store poisoned"))?
        .checked(id, &artifact, checked)?;
    Ok(true)
}

#[tokio::main(worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.listen.ip().is_loopback(),
        "bind only to loopback; use an SSH tunnel"
    );
    ensure!(
        (1..=300).contains(&args.compile_timeout_seconds) && args.max_occurrences > 0,
        "invalid compiler/storage bounds"
    );
    fs::create_dir_all(&args.journal_dir)?;
    fs::create_dir_all(&args.state_dir)?;
    ensure!(
        fs::canonicalize(&args.journal_dir)? != fs::canonicalize(&args.state_dir)?,
        "journal-dir and state-dir must differ"
    );
    let store = Arc::new(Mutex::new(Store::open(&args.state_dir)?));
    let mut offsets = BTreeMap::new();
    if args.once {
        while collect(
            &args.journal_dir,
            &mut offsets,
            &store,
            args.max_occurrences,
        )? > 0
        {}
        let mut cache = BTreeMap::new();
        while check_one(&store, args.compile_timeout_seconds, &mut cache)? {}
        println!(
            "{}",
            store
                .lock()
                .map_err(|_| anyhow::anyhow!("store poisoned"))?
                .snapshot()
        );
        return Ok(());
    }
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    let address = listener.local_addr()?;
    let app = App {
        store: store.clone(),
        authority: address.to_string(),
        log: args.state_dir.join("events.jsonl"),
        clients: Arc::new(tokio::sync::Semaphore::new(32)),
    };
    let stopped = Arc::new(AtomicBool::new(false));
    let (fatal_tx, mut fatal_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let producer = store.clone();
    let producer_stop = stopped.clone();
    let producer_tx = fatal_tx.clone();
    let collector = std::thread::spawn(move || {
        while !producer_stop.load(Ordering::Relaxed) {
            match collect(
                &args.journal_dir,
                &mut offsets,
                &producer,
                args.max_occurrences,
            ) {
                Ok(0) => std::thread::sleep(Duration::from_millis(100)),
                Ok(_) => {} // Process a backlog immediately, one bounded turn per journal.
                Err(error) => {
                    let _ = producer_tx.send(format!("source collector stopped: {error:#}"));
                    return;
                }
            }
        }
    });
    let compiler = store.clone();
    let compiler_stop = stopped.clone();
    let checker = std::thread::spawn(move || {
        let mut cache = BTreeMap::new();
        while !compiler_stop.load(Ordering::Relaxed) {
            match check_one(&compiler, args.compile_timeout_seconds, &mut cache) {
                Ok(true) => {}
                Ok(false) => std::thread::sleep(Duration::from_millis(100)),
                Err(error) => {
                    let _ = fatal_tx.send(format!("source validator stopped: {error:#}"));
                    return;
                }
            }
        }
    });
    eprintln!(
        "GP source WebSocket ws://{}/api/v1/gp/source ; metrics http://{}/api/v1/gp/metrics",
        address, address
    );
    let result = tokio::select! {
        result = axum::serve(listener, router(app)) => result.map_err(anyhow::Error::from),
        Some(error) = fatal_rx.recv() => Err(anyhow::anyhow!(error)),
        result = async {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                ensure!(!collector.is_finished() && !checker.is_finished(), "source worker exited unexpectedly");
            }
            #[allow(unreachable_code)] Ok::<(), anyhow::Error>(())
        } => result,
        signal = tokio::signal::ctrl_c() => signal.map_err(anyhow::Error::from),
    };
    stopped.store(true, Ordering::Relaxed);
    collector
        .join()
        .map_err(|_| anyhow::anyhow!("collector panicked"))?;
    checker
        .join()
        .map_err(|_| anyhow::anyhow!("compiler worker panicked"))?;
    result
}
