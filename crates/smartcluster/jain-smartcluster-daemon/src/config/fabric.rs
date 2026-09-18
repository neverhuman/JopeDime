//! `[fabric]` table accepted for source/binary compatibility with deployed
//! `scqd.toml` files written for the July fabric-v2 daemon.
//!
//! Key names, types, and defaults mirror that daemon. This build contains no
//! fabric networking: the table is parsed and validated only, and
//! `enabled = true` refuses startup instead of being silently ignored.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::ConfigError;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FabricConfig {
    /// Must stay `false`: fabric networking is not implemented in this build.
    pub enabled: bool,
    /// Reserved. This build always admits work to its host executor.
    pub local_executor: bool,
    /// Reserved. Requires `enabled`.
    pub edge_enabled: bool,
    pub cluster_id: String,
    pub artifact_root: PathBuf,
    pub artifact_key_path: PathBuf,
    pub commitment_ttl_ms: u64,
    pub max_cpu_idle_slots: u16,
    pub max_gpu_idle_slots: u16,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            local_executor: true,
            edge_enabled: false,
            cluster_id: "cluster-not-configured".to_owned(),
            artifact_root: PathBuf::from("/var/lib/jain-smartcluster/artifacts"),
            artifact_key_path: PathBuf::from("/var/lib/jain-smartcluster/artifact.key"),
            commitment_ttl_ms: 30_000,
            max_cpu_idle_slots: 1,
            max_gpu_idle_slots: 1,
        }
    }
}

impl FabricConfig {
    pub(super) fn validate(&self) -> Result<(), ConfigError> {
        if self.enabled {
            return Err(ConfigError::FabricNotImplemented);
        }
        for (name, path) in [
            ("fabric.artifact_root", &self.artifact_root),
            ("fabric.artifact_key_path", &self.artifact_key_path),
        ] {
            if !path.is_absolute() {
                return Err(ConfigError::RelativePath {
                    name,
                    path: path.clone(),
                });
            }
        }
        if self.edge_enabled {
            return Err(ConfigError::InvalidFabric(
                "edge_enabled requires fabric.enabled".to_owned(),
            ));
        }
        if !self.local_executor {
            return Err(ConfigError::InvalidFabric(
                "local_executor = false requires fabric.enabled".to_owned(),
            ));
        }
        Ok(())
    }
}
