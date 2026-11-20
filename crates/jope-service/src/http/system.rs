//! System information returned by `GET /v1/system`.

use axum::Json;
use jope_api::{Capabilities, SystemInfo};

/// Supplies the build's current capability matrix to the system snapshot.
pub trait CapabilityProvider {
    fn capabilities(&self) -> Capabilities;
}

/// The HTTP skeleton has no engine integration yet, so it advertises no active capabilities.
struct StubCapabilities;

impl CapabilityProvider for StubCapabilities {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tasks: Vec::new(),
            languages: Vec::new(),
            devices: Vec::new(),
            learned_guidance: false,
        }
    }
}

pub fn snapshot(provider: &impl CapabilityProvider) -> SystemInfo {
    SystemInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        api_version: "1.0".to_owned(),
        health: "ok".to_owned(),
        local_paths: true,
        capabilities: provider.capabilities(),
        warnings: Vec::new(),
        telemetry: Vec::new(),
    }
}

/// Return the current process and engine capability snapshot.
pub async fn get() -> Json<SystemInfo> {
    Json(snapshot(&StubCapabilities))
}

pub(crate) struct LiveCapabilities {
    pub learned_guidance: bool,
}

impl CapabilityProvider for LiveCapabilities {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tasks: vec![
                jope_api::Task::Regression,
                jope_api::Task::Binary,
                jope_api::Task::Multiclass,
            ],
            languages: vec![
                jope_api::Language::Rust,
                jope_api::Language::Python,
                jope_api::Language::Javascript,
            ],
            devices: vec!["cpu".to_owned()],
            learned_guidance: self.learned_guidance,
        }
    }
}

/// Return the live bounded CPU engine capability snapshot when `RunService` is mounted.
pub(in crate::http) async fn get_live(
    axum::extract::Extension(runs): axum::extract::Extension<
        std::sync::Arc<super::runs::RunService>,
    >,
) -> Json<SystemInfo> {
    Json(snapshot(&LiveCapabilities {
        learned_guidance: runs.has_admitted_ranker(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn system_info_snapshot_matches_the_current_schema() {
        let response = get().await.0;
        let value = serde_json::to_value(&response).unwrap();
        let expected = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "api_version": "1.0",
            "health": "ok",
            "local_paths": true,
            "capabilities": {
                "tasks": [],
                "languages": [],
                "devices": [],
                "learned_guidance": false
            },
            "warnings": [],
            "telemetry": []
        });

        assert_eq!(value, expected);

        // SystemInfo is the source type for the generated JSON schema. Deserializing the
        // snapshot through it checks the schema's required fields and wire types.
        let validated: SystemInfo = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(validated).unwrap(), value);
    }
}
