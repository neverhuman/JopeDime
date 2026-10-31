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

#[cfg(all(feature = "mcp", feature = "engine"))]
pub fn client() -> anyhow::Result<jope_client::Http> {
    use jope_service::daemon_file::DaemonFileStatus;
    let home = jope_paths::jopedime_home();
    let daemon = jope_service::daemon_file::read_daemon_file(&home)
        .map_err(|error| anyhow::anyhow!("engine is unreachable: {error}"))?;
    let record = match daemon {
        DaemonFileStatus::Running(record) | DaemonFileStatus::Unknown(record) => record,
        DaemonFileStatus::Stale(_) => anyhow::bail!("engine is unreachable: daemon is stale"),
    };
    jope_client::Http::new(&record.url, &record.token)
        .map_err(|error| anyhow::anyhow!("engine is unreachable: {error}"))
}

#[cfg(all(feature = "mcp", not(feature = "engine")))]
pub fn client() -> anyhow::Result<jope_client::Http> {
    anyhow::bail!("engine is unavailable in this build")
}
