//! Synchronous client transports and local engine discovery for the JopeDime v1 API.
//!
//! The default build provides the registry-driven [`Transport`] boundary, the loopback-only
//! [`Http`] implementation, request/response types, file helpers, and explicit SSE streams. It
//! does not require an async runtime. The optional `engine` feature adds `InProcess` and its
//! Tokio-backed router adapter; builds without that feature do not depend on Tokio. The optional
//! `fake` feature adds deterministic event replay for tests and terminal clients.
//!
//! [`Http`] accepts only plain HTTP loopback authorities (`localhost` or `127.0.0.1`), disables
//! redirects and proxies, and is not a remote or TLS client. Remote endpoint selection and its
//! authentication policy belong to the embedding application.

//! # Example
//!
//! ```no_run
//! use jope_client::{Http, Request, Transport};
//!
//! let client = Http::new("http://127.0.0.1:8080", "<64-hex-token>")?;
//! let request = Request::new(1, 0, &[])?;
//! let system = client.execute(request)?.json::<jope_api::SystemInfo>()?;
//! # let _ = system;
//! # Ok::<(), jope_client::Error>(())
//! ```

pub mod discover;
pub mod files;
pub mod http;
#[cfg(feature = "engine")]
pub mod inproc;
pub mod sse;
pub mod transport;

#[cfg(feature = "fake")]
pub mod fake;

pub use http::Http;
#[cfg(feature = "engine")]
pub use inproc::InProcess;
pub use sse::{EventStream, SseReader, StreamEvent};
pub use transport::{Error, Request, Response, Result, RunResult, Transport};
