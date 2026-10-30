pub mod events;
pub mod run;
pub mod serve;
pub mod status;

#[cfg(feature = "engine")]
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

#[cfg(not(feature = "engine"))]
pub fn client() -> anyhow::Result<jope_client::Http> {
    anyhow::bail!("engine is unavailable in this build")
}
