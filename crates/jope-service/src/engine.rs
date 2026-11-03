//! Process ownership and lifecycle for the shared JopeDime engine.

use axum::{
    extract::{Extension, Request},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::post,
    Router,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};
use subtle::ConstantTimeEq;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::watch,
};

use crate::{
    config::Config,
    daemon_file::{self, DaemonFile, DaemonFileStatus},
    http::{
        auth::{BearerToken, LocalAuth},
        router,
    },
    store::Store,
};

mod activity;
mod cli;
mod locking;
mod serve;
#[cfg(test)]
mod tests;

use activity::TrackedListener;
pub use activity::{ActivityGuard, EngineActivity};
pub use cli::run_cli;
#[cfg(test)]
use cli::{parse_duration, request_daemon_async};
pub use locking::{EngineError, EngineLock};
#[cfg(feature = "research")]
use serve::open_q1_engine;
pub use serve::serve;
#[cfg(test)]
use serve::{serve_locked, serve_with_activity};
#[cfg(feature = "research")]
pub use serve::{serve_q1_infra_retry, serve_q1_qualification};

const DEFAULT_IDLE_EXIT: Duration = Duration::from_secs(30 * 60);
const STOP_PATH: &str = "/__engine/stop";

/// The same home lock and recovery path used by background engines, without daemon publication.
pub struct InProcessEngine {
    // Fields drop in declaration order: close redb before releasing home ownership.
    store: Arc<Store>,
    home: PathBuf,
    executable: PathBuf,
    runs: Mutex<Option<Arc<crate::http::runs::RunService>>>,
    activity: Arc<EngineActivity>,
    _lock: Arc<EngineLock>,
}

/// Read-only request state lent from the one live engine/store instance.
#[derive(Clone)]
pub(crate) struct DatasetState {
    pub store: Arc<Store>,
    pub config: Config,
    pub home: PathBuf,
}
