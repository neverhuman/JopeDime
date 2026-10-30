#![cfg(feature = "mcp")]

//! CLI entrypoint for MCP over the shared client and engine discovery path.

use anyhow::{bail, Result};
use jope_client::{
    discover::{self, Engine, Mode, Options},
    Http,
};

fn options() -> Result<Options> {
    Ok(Options {
        home: jope_paths::jopedime_home(),
        executable: std::env::current_exe()?,
        endpoint: None,
        mode: Mode::Interactive,
        engine_available: cfg!(feature = "engine"),
    })
}

fn client(engine: Engine) -> Result<Http> {
    match engine {
        Engine::Background(background) => Ok(background.client),
        Engine::Endpoint(_) => {
            bail!("explicit MCP endpoints require remote bearer-token support, which is not available in this v1 client")
        }
        Engine::InProcess => {
            bail!("interactive MCP discovery unexpectedly selected an in-process engine")
        }
    }
}

fn adapter(engine: Engine) -> Result<crate::mcp::Adapter<Http>> {
    Ok(crate::mcp::Adapter::new(client(engine)?))
}

fn discover_engine(options: &Options, explicit_endpoint: bool) -> Result<Engine> {
    if !options.home.is_absolute() && !explicit_endpoint {
        bail!("shared MCP discovery requires an absolute user home directory");
    }
    discover::discover(options).map_err(anyhow::Error::from)
}

fn with_discovery_fallback<T>(
    engine: Result<Engine>,
    shared: impl FnOnce(Engine) -> Result<T>,
    fallback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match engine {
        Ok(engine) => shared(engine),
        Err(_) => fallback(),
    }
}

pub fn run() -> Result<()> {
    let engine = options().and_then(|options| {
        let explicit_endpoint =
            options.endpoint.is_some() || std::env::var_os("JOPEDIME_ENDPOINT").is_some();
        discover_engine(&options, explicit_endpoint)
    });
    with_discovery_fallback(
        engine,
        |engine| adapter(engine)?.serve_stdio(),
        || crate::serve(true),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn background() -> Engine {
        let client = Http::new("http://127.0.0.1:32123", &"a".repeat(64)).unwrap();
        Engine::Background(jope_client::discover::Background {
            client,
            url: "http://127.0.0.1:32123".into(),
            pid: 1,
            version: "test".into(),
            api_version: "1".into(),
            started_at: "test".into(),
        })
    }

    #[test]
    fn cli_discovery_is_interactive_and_uses_the_current_executable() {
        let options = options().unwrap();
        assert_eq!(options.mode, Mode::Interactive);
        assert!(options.endpoint.is_none());
        assert_eq!(options.executable, std::env::current_exe().unwrap());
        assert!(options.home.is_absolute());
        assert_eq!(options.engine_available, cfg!(feature = "engine"));
    }

    #[test]
    fn discovered_background_client_constructs_the_existing_adapter() {
        assert!(adapter(background()).is_ok());
    }

    #[test]
    fn discovery_failure_uses_the_existing_in_process_server() {
        let mut options = options().unwrap();
        options.home = std::path::PathBuf::from(".local/share/jopedime");
        let shared_called = std::cell::Cell::new(false);
        let fallback_called = std::cell::Cell::new(false);
        let result = with_discovery_fallback(
            discover_engine(&options, false),
            |_| {
                shared_called.set(true);
                Ok("shared client")
            },
            || {
                fallback_called.set(true);
                Ok("in-process server")
            },
        );

        assert_eq!(result.unwrap(), "in-process server");
        assert!(!shared_called.get());
        assert!(fallback_called.get());
    }

    #[test]
    fn shared_client_failure_does_not_silently_switch_servers() {
        let fallback_called = std::cell::Cell::new(false);
        let result = with_discovery_fallback(
            Ok(background()),
            |_| Err(anyhow::anyhow!("shared adapter failed")),
            || {
                fallback_called.set(true);
                Ok(())
            },
        );

        assert!(result.is_err());
        assert!(!fallback_called.get());
    }

    #[test]
    fn explicit_endpoint_does_not_fall_back_without_credentials() {
        let error = match client(Engine::Endpoint("https://example.test".into())) {
            Err(error) => error,
            Ok(_) => panic!("endpoint mode must require supported authentication"),
        };
        assert!(error.to_string().contains("remote bearer-token support"));
    }

    #[test]
    fn interactive_discovery_rejects_batch_only_in_process_mode() {
        let error = match client(Engine::InProcess) {
            Err(error) => error,
            Ok(_) => panic!("interactive mode must not select an in-process engine"),
        };
        assert!(error
            .to_string()
            .contains("unexpectedly selected an in-process"));
    }
}
