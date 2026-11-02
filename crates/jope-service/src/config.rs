//! Configuration loading and per-setting provenance for the JopeDime service.
//!
//! Environment overrides use `JOPEDIME_` plus the uppercase TOML path with underscores between
//! path components (for example, `JOPEDIME_DEFAULTS_EFFORT`). Command-line overrides use the
//! corresponding kebab-case flag (for example, `--effort`).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

use toml::Value;

const KEYS: [&str; 13] = [
    "defaults.effort",
    "defaults.languages",
    "defaults.reasoning",
    "defaults.workers",
    "defaults.device",
    "limits.max_dataset_bytes",
    "limits.disk_reserve",
    "limits.rung0_max_rows",
    "limits.rung1_max_rows",
    "limits.eval_timeout",
    "engine.idle_exit",
    "ui.color",
    "ui.theme",
];

/// Where an effective setting was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigOrigin {
    Flag,
    Environment,
    File,
    BuiltInDefault,
}

/// Compatibility name for a configuration value's source.
pub type Origin = ConfigOrigin;

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flag => f.write_str("flag"),
            Self::Environment => f.write_str("environment"),
            Self::File => f.write_str("file"),
            Self::BuiltInDefault => f.write_str("built-in default"),
        }
    }
}

/// A setting together with its source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Resolved<T> {
    pub value: T,
    pub origin: Origin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerCount {
    Auto,
    Count(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Defaults {
    pub effort: Resolved<String>,
    pub languages: Resolved<Vec<String>>,
    pub reasoning: Resolved<String>,
    pub workers: Resolved<WorkerCount>,
    pub device: Resolved<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Limits {
    pub max_dataset_bytes: Resolved<u64>,
    pub disk_reserve: Resolved<u64>,
    pub rung0_max_rows: Resolved<u64>,
    pub rung1_max_rows: Resolved<u64>,
    pub eval_timeout: Resolved<Duration>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Engine {
    pub idle_exit: Resolved<Duration>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ui {
    pub color: Resolved<String>,
    pub theme: Resolved<String>,
}

/// The effective configuration. Every value keeps its precedence origin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Config {
    pub defaults: Defaults,
    pub limits: Limits,
    pub engine: Engine,
    pub ui: Ui,
    pub warnings: Vec<ConfigWarning>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigWarning {
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError {
    pub key: String,
    pub message: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid configuration value for {}: {}",
            self.key, self.message
        )
    }
}

impl Error for ConfigError {}

impl Config {
    /// Read an optional configuration file and resolve it with explicit flag and environment
    /// overrides. Missing files use built-in defaults.
    pub fn load(
        path: &Path,
        flags: &BTreeMap<String, String>,
        environment: &BTreeMap<String, String>,
    ) -> Result<Self, ConfigError> {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => Some(contents),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(ConfigError {
                    key: "config.toml".into(),
                    message: format!("could not read {}: {error}", path.display()),
                });
            }
        };
        Self::from_sources(contents.as_deref(), flags, environment)
    }

    /// Resolve an optional TOML document and explicit overlays. Keys in `flags` are CLI option
    /// names such as `--effort`; keys in `environment` are exact `JOPEDIME_*` variable names.
    pub fn from_sources(
        file: Option<&str>,
        flags: &BTreeMap<String, String>,
        environment: &BTreeMap<String, String>,
    ) -> Result<Self, ConfigError> {
        let document = file
            .map(|text| {
                text.parse::<Value>().map_err(|error| ConfigError {
                    key: "config.toml".into(),
                    message: error.to_string(),
                })
            })
            .transpose()?
            .unwrap_or_else(|| Value::Table(Default::default()));
        let warnings = unknown_key_warnings(&document);

        Ok(Self {
            defaults: Defaults {
                effort: validate_choice(
                    "defaults.effort",
                    resolve_string(
                        "defaults.effort",
                        "--effort",
                        "JOPEDIME_DEFAULTS_EFFORT",
                        "medium",
                        &document,
                        flags,
                        environment,
                    )?,
                    &["fast", "medium", "high", "ultra"],
                )?,
                languages: resolve_languages(&document, flags, environment)?,
                reasoning: validate_choice(
                    "defaults.reasoning",
                    resolve_string(
                        "defaults.reasoning",
                        "--reasoning",
                        "JOPEDIME_DEFAULTS_REASONING",
                        "concise",
                        &document,
                        flags,
                        environment,
                    )?,
                    &["off", "concise", "detailed"],
                )?,
                workers: resolve_workers(&document, flags, environment)?,
                device: resolve_string(
                    "defaults.device",
                    "--device",
                    "JOPEDIME_DEFAULTS_DEVICE",
                    "auto",
                    &document,
                    flags,
                    environment,
                )?,
            },
            limits: Limits {
                max_dataset_bytes: resolve_bytes(
                    "limits.max_dataset_bytes",
                    "--max-dataset-bytes",
                    "JOPEDIME_LIMITS_MAX_DATASET_BYTES",
                    "4GiB",
                    &document,
                    flags,
                    environment,
                )?,
                disk_reserve: resolve_bytes(
                    "limits.disk_reserve",
                    "--disk-reserve",
                    "JOPEDIME_LIMITS_DISK_RESERVE",
                    "1GiB",
                    &document,
                    flags,
                    environment,
                )?,
                rung0_max_rows: resolve_rows(
                    "limits.rung0_max_rows",
                    "--rung0-max-rows",
                    "JOPEDIME_LIMITS_RUNG0_MAX_ROWS",
                    20_000,
                    &document,
                    flags,
                    environment,
                )?,
                rung1_max_rows: resolve_rows(
                    "limits.rung1_max_rows",
                    "--rung1-max-rows",
                    "JOPEDIME_LIMITS_RUNG1_MAX_ROWS",
                    100_000,
                    &document,
                    flags,
                    environment,
                )?,
                eval_timeout: resolve_duration(
                    "limits.eval_timeout",
                    "--eval-timeout",
                    "JOPEDIME_LIMITS_EVAL_TIMEOUT",
                    "30s",
                    &document,
                    flags,
                    environment,
                )?,
            },
            engine: Engine {
                idle_exit: resolve_duration(
                    "engine.idle_exit",
                    "--idle-exit",
                    "JOPEDIME_ENGINE_IDLE_EXIT",
                    "30m",
                    &document,
                    flags,
                    environment,
                )?,
            },
            ui: Ui {
                color: validate_choice(
                    "ui.color",
                    resolve_string(
                        "ui.color",
                        "--color",
                        "JOPEDIME_UI_COLOR",
                        "auto",
                        &document,
                        flags,
                        environment,
                    )?,
                    &["auto", "always", "never"],
                )?,
                theme: validate_choice(
                    "ui.theme",
                    resolve_string(
                        "ui.theme",
                        "--theme",
                        "JOPEDIME_UI_THEME",
                        "auto",
                        &document,
                        flags,
                        environment,
                    )?,
                    &["auto", "dark", "light"],
                )?,
            },
            warnings,
        })
    }

    /// Return the origin for a dotted Appendix C key.
    pub fn origin_for(&self, key: &str) -> Option<Origin> {
        Some(match key {
            "defaults.effort" => self.defaults.effort.origin,
            "defaults.languages" => self.defaults.languages.origin,
            "defaults.reasoning" => self.defaults.reasoning.origin,
            "defaults.workers" => self.defaults.workers.origin,
            "defaults.device" => self.defaults.device.origin,
            "limits.max_dataset_bytes" => self.limits.max_dataset_bytes.origin,
            "limits.disk_reserve" => self.limits.disk_reserve.origin,
            "limits.rung0_max_rows" => self.limits.rung0_max_rows.origin,
            "limits.rung1_max_rows" => self.limits.rung1_max_rows.origin,
            "limits.eval_timeout" => self.limits.eval_timeout.origin,
            "engine.idle_exit" => self.engine.idle_exit.origin,
            "ui.color" => self.ui.color.origin,
            "ui.theme" => self.ui.theme.origin,
            _ => return None,
        })
    }
}

mod helpers;
use helpers::*;

#[cfg(test)]
mod tests;
