#[cfg(feature = "client")]
pub mod code;
#[cfg(feature = "client")]
pub mod control;
#[cfg(feature = "mcp")]
pub mod events;
#[cfg(feature = "mcp")]
pub mod mcp_entry;
#[cfg(feature = "client")]
pub mod restart;
#[cfg(feature = "mcp")]
pub mod run;
#[cfg(feature = "client")]
pub mod runs;
#[cfg(feature = "mcp")]
pub(crate) use crate::common;
#[cfg(feature = "mcp")]
pub(crate) mod dataset;
#[cfg(feature = "mcp")]
pub(crate) mod export;
#[cfg(feature = "mcp")]
pub(crate) mod gc;
#[cfg(feature = "mcp")]
pub mod inproc;
#[cfg(feature = "mcp")]
pub(crate) mod objective;
pub mod serve;
#[cfg(feature = "mcp")]
pub mod status;

/// The shared engine for `status`, `events` and the other engine commands: started on demand.
#[cfg(all(feature = "mcp", feature = "engine"))]
pub fn client() -> anyhow::Result<jope_client::Http> {
    crate::common::resolve_client()
}

#[cfg(all(feature = "mcp", not(feature = "engine")))]
pub fn client() -> anyhow::Result<jope_client::Http> {
    anyhow::bail!("engine is unavailable in this build")
}
