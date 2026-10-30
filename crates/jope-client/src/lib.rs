//! JopeDime v1 API client.

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
