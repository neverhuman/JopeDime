//! Dedicated OS-thread Tokio runtime driver and shared listener/response primitives for `serve` and `emu`.

use crate::error::CarbonErrCtx as _;
use std::future::Future;
use std::net::SocketAddr;
use std::thread::JoinHandle;

use axum::{
    body::Body,
    http::{Response, StatusCode},
};

pub(crate) fn bind_nonblocking_listener(
    addr: SocketAddr,
) -> Result<(std::net::TcpListener, SocketAddr), String> {
    let std_listener =
        std::net::TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))?;
    std_listener
        .set_nonblocking(true)
        .err_ctx("set_nonblocking")?;
    let bound_addr = std_listener.local_addr().err_ctx("local_addr")?;
    Ok((std_listener, bound_addr))
}

pub(crate) fn plain_axum_response(status: StatusCode, body: impl Into<Body>) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(body.into())
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

pub(crate) fn drive_dedicated_os_thread_runtime<F: Future>(
    rt: &tokio::runtime::Runtime,
    fut: F,
) -> F::Output {
    rt.block_on(fut)
}

pub(crate) fn spawn_background_tokio_thread<F>(
    thread_name: &str,
    worker_threads: usize,
    fut: F,
) -> Result<JoinHandle<()>, String>
where
    F: Future<Output = ()> + Send + 'static,
{
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .map_err(|e| format!("build {thread_name} tokio runtime: {e}"))?;
    std::thread::Builder::new()
        .name(thread_name.to_owned())
        .spawn(move || {
            drive_dedicated_os_thread_runtime(&rt, fut);
        })
        .map_err(|e| format!("spawn {thread_name} thread: {e}"))
}

pub(crate) struct BackgroundHttpTask {
    pub(crate) bound_addr: SocketAddr,
    pub(crate) http_origin: String,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl BackgroundHttpTask {
    pub(crate) fn new(
        bound_addr: SocketAddr,
        shutdown_tx: tokio::sync::oneshot::Sender<()>,
        join_handle: JoinHandle<()>,
    ) -> Self {
        Self {
            bound_addr,
            http_origin: format!("http://{bound_addr}"),
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }

    pub(crate) fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BackgroundHttpTask {
    fn drop(&mut self) {
        self.stop();
    }
}
