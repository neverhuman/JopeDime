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
pub enum Origin {
    Flag,
    Environment,
    File,
    BuiltInDefault,
}

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

fn resolve_string(
    key: &str,
    flag: &str,
    env: &str,
    default: &str,
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<String>, ConfigError> {
    let (value, origin) = select_value(
        key,
        flag,
        env,
        Value::String(default.into()),
        file,
        flags,
        environment,
    )?;
    match value {
        Value::String(value) => Ok(Resolved { value, origin }),
        other => Err(type_error(key, "a string", &other)),
    }
}

fn validate_choice(
    key: &str,
    resolved: Resolved<String>,
    choices: &[&str],
) -> Result<Resolved<String>, ConfigError> {
    if choices.contains(&resolved.value.as_str()) {
        Ok(resolved)
    } else {
        Err(ConfigError {
            key: key.into(),
            message: format!("expected one of {}", choices.join(", ")),
        })
    }
}

fn resolve_languages(
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<Vec<String>>, ConfigError> {
    let (value, origin) = select_value(
        "defaults.languages",
        "--languages",
        "JOPEDIME_DEFAULTS_LANGUAGES",
        Value::Array(vec![Value::String("python".into())]),
        file,
        flags,
        environment,
    )?;
    let values = match value {
        Value::Array(values) => values,
        Value::String(value) => value
            .split(',')
            .map(|part| Value::String(part.trim().into()))
            .collect(),
        other => {
            return Err(type_error(
                "defaults.languages",
                "an array of strings",
                &other,
            ))
        }
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::String(value) if !value.is_empty() => Ok(value),
            other => Err(type_error(
                "defaults.languages",
                "non-empty strings",
                &other,
            )),
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|value| Resolved { value, origin })
}

fn resolve_workers(
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<WorkerCount>, ConfigError> {
    let (value, origin) = select_value(
        "defaults.workers",
        "--workers",
        "JOPEDIME_DEFAULTS_WORKERS",
        Value::String("auto".into()),
        file,
        flags,
        environment,
    )?;
    let workers = match value {
        Value::String(value) if value == "auto" => WorkerCount::Auto,
        Value::Integer(value) if value > 0 => usize::try_from(value)
            .map(WorkerCount::Count)
            .map_err(|_| ConfigError {
                key: "defaults.workers".into(),
                message: "worker count is out of range".into(),
            })?,
        Value::String(value) => value
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .map(WorkerCount::Count)
            .ok_or_else(|| ConfigError {
                key: "defaults.workers".into(),
                message: "expected `auto` or a positive integer".into(),
            })?,
        other => {
            return Err(type_error(
                "defaults.workers",
                "`auto` or a positive integer",
                &other,
            ))
        }
    };
    Ok(Resolved {
        value: workers,
        origin,
    })
}

fn resolve_bytes(
    key: &str,
    flag: &str,
    env: &str,
    default: &str,
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<u64>, ConfigError> {
    let (value, origin) = select_value(
        key,
        flag,
        env,
        Value::String(default.into()),
        file,
        flags,
        environment,
    )?;
    let bytes = match value {
        Value::String(value) => parse_bytes(&value).map_err(|message| ConfigError {
            key: key.into(),
            message,
        })?,
        Value::Integer(value) if value >= 0 => u64::try_from(value).map_err(|_| ConfigError {
            key: key.into(),
            message: "byte count is out of range".into(),
        })?,
        other => {
            return Err(type_error(
                key,
                "a byte-size string or non-negative integer",
                &other,
            ))
        }
    };
    Ok(Resolved {
        value: bytes,
        origin,
    })
}

fn resolve_rows(
    key: &str,
    flag: &str,
    env: &str,
    default: u64,
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<u64>, ConfigError> {
    let (value, origin) = select_value(
        key,
        flag,
        env,
        Value::Integer(default as i64),
        file,
        flags,
        environment,
    )?;
    let rows = match value {
        Value::Integer(value) if value >= 0 => u64::try_from(value).map_err(|_| ConfigError {
            key: key.into(),
            message: "row count is out of range".into(),
        })?,
        other => return Err(type_error(key, "a non-negative integer", &other)),
    };
    Ok(Resolved {
        value: rows,
        origin,
    })
}

fn resolve_duration(
    key: &str,
    flag: &str,
    env: &str,
    default: &str,
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<Resolved<Duration>, ConfigError> {
    let (value, origin) = select_value(
        key,
        flag,
        env,
        Value::String(default.into()),
        file,
        flags,
        environment,
    )?;
    let duration = match value {
        Value::String(value) => parse_duration(&value).map_err(|message| ConfigError {
            key: key.into(),
            message,
        })?,
        other => return Err(type_error(key, "a duration string", &other)),
    };
    Ok(Resolved {
        value: duration,
        origin,
    })
}

fn select_value(
    key: &str,
    flag: &str,
    env: &str,
    default: Value,
    file: &Value,
    flags: &BTreeMap<String, String>,
    environment: &BTreeMap<String, String>,
) -> Result<(Value, Origin), ConfigError> {
    if let Some(value) = flags.get(flag) {
        return Ok((overlay_value(key, value)?, Origin::Flag));
    }
    if let Some(value) = environment.get(env) {
        return Ok((overlay_value(key, value)?, Origin::Environment));
    }
    if let Some(value) = get_path(file, key)? {
        return Ok((value.clone(), Origin::File));
    }
    Ok((default, Origin::BuiltInDefault))
}

fn overlay_value(key: &str, value: &str) -> Result<Value, ConfigError> {
    let result = match key {
        "defaults.languages" => Value::Array(
            value
                .split(',')
                .map(|part| Value::String(part.trim().into()))
                .collect(),
        ),
        "limits.rung0_max_rows" | "limits.rung1_max_rows" | "defaults.workers" => value
            .parse::<i64>()
            .map(Value::Integer)
            .unwrap_or_else(|_| Value::String(value.into())),
        _ => Value::String(value.into()),
    };
    Ok(result)
}

fn get_path<'a>(value: &'a Value, path: &str) -> Result<Option<&'a Value>, ConfigError> {
    let mut node = value;
    let mut parent = String::new();
    for component in path.split('.') {
        let Some(table) = node.as_table() else {
            return Err(ConfigError {
                key: parent,
                message: "expected a table".into(),
            });
        };
        let Some(child) = table.get(component) else {
            return Ok(None);
        };
        if !parent.is_empty() {
            parent.push('.');
        }
        parent.push_str(component);
        node = child;
    }
    Ok(Some(node))
}

fn unknown_key_warnings(document: &Value) -> Vec<ConfigWarning> {
    let known = KEYS.iter().copied().collect::<BTreeSet<_>>();
    let prefixes = KEYS
        .iter()
        .flat_map(|key| {
            let mut prefixes = Vec::new();
            let mut path = String::new();
            for component in key.split('.') {
                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(component);
                prefixes.push(path.clone());
            }
            prefixes
        })
        .collect::<BTreeSet<_>>();
    let mut warnings = Vec::new();
    collect_unknown(document, "", &known, &prefixes, &mut warnings);
    warnings
}

fn collect_unknown(
    value: &Value,
    parent: &str,
    known: &BTreeSet<&str>,
    prefixes: &BTreeSet<String>,
    warnings: &mut Vec<ConfigWarning>,
) {
    let Some(table) = value.as_table() else {
        return;
    };
    for (name, value) in table {
        let path = if parent.is_empty() {
            name.clone()
        } else {
            format!("{parent}.{name}")
        };
        if !known.contains(path.as_str()) && !prefixes.contains(&path) {
            warnings.push(ConfigWarning {
                message: format!("unknown config key `{path}`"),
            });
        } else if prefixes.contains(&path) && !known.contains(path.as_str()) {
            collect_unknown(value, &path, known, prefixes, warnings);
        }
    }
}

fn parse_bytes(input: &str) -> Result<u64, String> {
    let input = input.trim();
    let split = input
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(input.len());
    if split == 0 {
        return Err("expected a non-negative byte count with an optional unit".into());
    }
    let amount = input[..split]
        .parse::<u64>()
        .map_err(|_| "byte count is out of range".to_owned())?;
    let multiplier = match &input[split..] {
        "" | "B" => 1,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        "TiB" => 1 << 40,
        unit => return Err(format!("unsupported byte-size unit `{unit}`")),
    };
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| "byte size is out of range".into())
}

fn parse_duration(input: &str) -> Result<Duration, String> {
    let input = input.trim();
    let split = input
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(input.len());
    if split == 0 {
        return Err("expected an integer duration with a unit".into());
    }
    let amount = input[..split]
        .parse::<u64>()
        .map_err(|_| "duration is out of range".to_owned())?;
    let nanos = match &input[split..] {
        "ms" => amount.checked_mul(1_000_000),
        "s" => amount.checked_mul(1_000_000_000),
        "m" => amount.checked_mul(60_000_000_000),
        "h" => amount.checked_mul(3_600_000_000_000),
        unit => return Err(format!("unsupported duration unit `{unit}`")),
    }
    .ok_or_else(|| "duration is out of range".to_owned())?;
    Ok(Duration::from_nanos(nanos))
}

fn type_error(key: &str, expected: &str, value: &Value) -> ConfigError {
    ConfigError {
        key: key.into(),
        message: format!("expected {expected}, got {value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).into(), (*value).into()))
            .collect()
    }

    fn setting_value(config: &Config, key: &str) -> String {
        match key {
            "defaults.effort" => config.defaults.effort.value.clone(),
            "defaults.languages" => config.defaults.languages.value.join(","),
            "defaults.reasoning" => config.defaults.reasoning.value.clone(),
            "defaults.workers" => match config.defaults.workers.value {
                WorkerCount::Auto => "auto".into(),
                WorkerCount::Count(count) => count.to_string(),
            },
            "defaults.device" => config.defaults.device.value.clone(),
            "limits.max_dataset_bytes" => config.limits.max_dataset_bytes.value.to_string(),
            "limits.disk_reserve" => config.limits.disk_reserve.value.to_string(),
            "limits.rung0_max_rows" => config.limits.rung0_max_rows.value.to_string(),
            "limits.rung1_max_rows" => config.limits.rung1_max_rows.value.to_string(),
            "limits.eval_timeout" => format!("{}s", config.limits.eval_timeout.value.as_secs()),
            "engine.idle_exit" => format!("{}s", config.engine.idle_exit.value.as_secs()),
            "ui.color" => config.ui.color.value.clone(),
            "ui.theme" => config.ui.theme.value.clone(),
            _ => panic!("unknown config key {key}"),
        }
    }

    #[test]
    fn defaults_match_appendix_c_and_keep_default_origins() {
        let config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
        assert_eq!(config.defaults.effort.value, "medium");
        assert_eq!(config.defaults.effort.origin, Origin::BuiltInDefault);
        assert_eq!(config.defaults.languages.value, ["python"]);
        assert_eq!(config.defaults.reasoning.value, "concise");
        assert_eq!(config.defaults.workers.value, WorkerCount::Auto);
        assert_eq!(config.defaults.device.value, "auto");
        assert_eq!(config.limits.max_dataset_bytes.value, 4 * (1 << 30));
        assert_eq!(config.limits.disk_reserve.value, 1 << 30);
        assert_eq!(config.limits.rung0_max_rows.value, 20_000);
        assert_eq!(config.limits.rung1_max_rows.value, 100_000);
        assert_eq!(config.limits.eval_timeout.value, Duration::from_secs(30));
        assert_eq!(config.engine.idle_exit.value, Duration::from_secs(30 * 60));
        assert_eq!(config.ui.color.value, "auto");
        assert_eq!(config.ui.theme.value, "auto");
    }

    #[test]
    fn each_key_obeys_flag_environment_file_default_precedence_and_origin() {
        let file = r#"
[defaults]
effort = "fast"
languages = ["rust"]
reasoning = "off"
workers = 2
device = "cpu"
[limits]
max_dataset_bytes = "2GiB"
disk_reserve = "2GiB"
rung0_max_rows = 10
rung1_max_rows = 20
eval_timeout = "5s"
[engine]
idle_exit = "5m"
[ui]
color = "always"
theme = "dark"
"#;
        let cases = [
            (
                "defaults.effort",
                "--effort",
                "JOPEDIME_DEFAULTS_EFFORT",
                "high",
                "fast",
            ),
            (
                "defaults.languages",
                "--languages",
                "JOPEDIME_DEFAULTS_LANGUAGES",
                "go,rust",
                "rust",
            ),
            (
                "defaults.reasoning",
                "--reasoning",
                "JOPEDIME_DEFAULTS_REASONING",
                "detailed",
                "off",
            ),
            (
                "defaults.workers",
                "--workers",
                "JOPEDIME_DEFAULTS_WORKERS",
                "4",
                "2",
            ),
            (
                "defaults.device",
                "--device",
                "JOPEDIME_DEFAULTS_DEVICE",
                "cuda",
                "cpu",
            ),
            (
                "limits.max_dataset_bytes",
                "--max-dataset-bytes",
                "JOPEDIME_LIMITS_MAX_DATASET_BYTES",
                "8GiB",
                "2147483648",
            ),
            (
                "limits.disk_reserve",
                "--disk-reserve",
                "JOPEDIME_LIMITS_DISK_RESERVE",
                "3GiB",
                "2147483648",
            ),
            (
                "limits.rung0_max_rows",
                "--rung0-max-rows",
                "JOPEDIME_LIMITS_RUNG0_MAX_ROWS",
                "30",
                "10",
            ),
            (
                "limits.rung1_max_rows",
                "--rung1-max-rows",
                "JOPEDIME_LIMITS_RUNG1_MAX_ROWS",
                "40",
                "20",
            ),
            (
                "limits.eval_timeout",
                "--eval-timeout",
                "JOPEDIME_LIMITS_EVAL_TIMEOUT",
                "10s",
                "5s",
            ),
            (
                "engine.idle_exit",
                "--idle-exit",
                "JOPEDIME_ENGINE_IDLE_EXIT",
                "10m",
                "300s",
            ),
            (
                "ui.color",
                "--color",
                "JOPEDIME_UI_COLOR",
                "never",
                "always",
            ),
            ("ui.theme", "--theme", "JOPEDIME_UI_THEME", "light", "dark"),
        ];
        for (key, flag, env, value, file_value) in cases {
            let flags = map(&[(flag, value)]);
            let environment = map(&[(env, "always-invalid-if-parsed")]);
            let config = Config::from_sources(Some(file), &flags, &environment).unwrap();
            assert_eq!(config.origin_for(key), Some(Origin::Flag), "{key}");
            assert_eq!(setting_value(&config, key), normalized(key, value), "{key}");

            let flags = BTreeMap::new();
            let environment = map(&[(env, value)]);
            let config = Config::from_sources(Some(file), &flags, &environment).unwrap();
            assert_eq!(config.origin_for(key), Some(Origin::Environment), "{key}");
            assert_eq!(setting_value(&config, key), normalized(key, value), "{key}");

            let config =
                Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap();
            assert_eq!(config.origin_for(key), Some(Origin::File), "{key}");
            assert_eq!(setting_value(&config, key), file_value, "{key}");

            let config = Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap();
            assert_eq!(
                config.origin_for(key),
                Some(Origin::BuiltInDefault),
                "{key}"
            );
        }
    }

    fn normalized(key: &str, value: &str) -> String {
        match key {
            "limits.max_dataset_bytes" | "limits.disk_reserve" => {
                parse_bytes(value).unwrap().to_string()
            }
            "limits.eval_timeout" | "engine.idle_exit" => {
                format!("{}s", parse_duration(value).unwrap().as_secs())
            }
            _ => value.to_owned(),
        }
    }

    #[test]
    fn byte_sizes_and_durations_parse() {
        let config = Config::from_sources(
            Some("[limits]\nmax_dataset_bytes = \"4GiB\"\neval_timeout = \"30m\""),
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(config.limits.max_dataset_bytes.value, 4 * (1 << 30));
        assert_eq!(
            config.limits.eval_timeout.value,
            Duration::from_secs(30 * 60)
        );
    }

    #[test]
    fn invalid_values_name_their_key() {
        let error = Config::from_sources(
            Some("[limits]\nmax_dataset_bytes = \"four gigs\""),
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("limits.max_dataset_bytes"));

        for (key, file) in [
            ("defaults.effort", "[defaults]\neffort = \"banana\""),
            ("defaults.reasoning", "[defaults]\nreasoning = \"banana\""),
            ("limits", "limits = 123"),
            ("defaults", "defaults = 123"),
            ("engine", "engine = 123"),
            ("ui", "ui = 123"),
        ] {
            let error =
                Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
            assert_eq!(error.key, key, "{file}");
            assert!(error.to_string().contains(key), "{file}");
        }
    }

    #[test]
    fn invalid_enum_values_are_rejected_at_each_precedence_source() {
        for (key, flag, env, file) in [
            (
                "defaults.effort",
                "--effort",
                "JOPEDIME_DEFAULTS_EFFORT",
                "[defaults]\neffort = \"banana\"",
            ),
            (
                "defaults.reasoning",
                "--reasoning",
                "JOPEDIME_DEFAULTS_REASONING",
                "[defaults]\nreasoning = \"banana\"",
            ),
        ] {
            let error =
                Config::from_sources(Some(file), &BTreeMap::new(), &BTreeMap::new()).unwrap_err();
            assert_eq!(error.key, key, "{file}");

            let error =
                Config::from_sources(None, &BTreeMap::new(), &map(&[(env, "banana")])).unwrap_err();
            assert_eq!(error.key, key, "{env}");

            let error = Config::from_sources(None, &map(&[(flag, "banana")]), &BTreeMap::new())
                .unwrap_err();
            assert_eq!(error.key, key, "{flag}");
        }
    }

    #[test]
    fn valid_flag_override_wins_over_invalid_lower_priority_file_value() {
        let file = "[defaults]\neffort = \"banana\"\nreasoning = \"banana\"";
        let flags = map(&[("--effort", "high"), ("--reasoning", "detailed")]);
        let config = Config::from_sources(Some(file), &flags, &BTreeMap::new()).unwrap();
        assert_eq!(config.defaults.effort.value, "high");
        assert_eq!(config.defaults.effort.origin, Origin::Flag);
        assert_eq!(config.defaults.reasoning.value, "detailed");
        assert_eq!(config.defaults.reasoning.origin, Origin::Flag);
    }

    #[test]
    fn unknown_keys_warn_without_rejecting_known_values() {
        let config = Config::from_sources(
            Some("[defaults]\neffort = \"high\"\nextra = true\n[future]\nvalue = 1"),
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(config.defaults.effort.value, "high");
        assert_eq!(config.warnings.len(), 2);
        assert!(config
            .warnings
            .iter()
            .any(|warning| warning.message.contains("defaults.extra")));
        assert!(config
            .warnings
            .iter()
            .any(|warning| warning.message.contains("future")));
    }
}
