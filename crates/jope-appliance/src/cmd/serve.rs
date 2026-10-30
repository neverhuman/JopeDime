//! CLI adapter for the service engine lifecycle.

pub fn run(status: bool, stop: bool, idle_exit: Option<&str>) -> anyhow::Result<()> {
    #[cfg(feature = "engine")]
    {
        jope_service::engine::run_cli(status, stop, idle_exit).map_err(anyhow::Error::new)
    }
    #[cfg(not(feature = "engine"))]
    {
        let _ = (status, stop, idle_exit);
        anyhow::bail!("no engine found; run `jopedime serve` with the engine feature")
    }
}
