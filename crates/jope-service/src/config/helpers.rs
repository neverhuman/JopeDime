use super::*;

pub(super) fn resolve_string(
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

pub(super) fn validate_choice(
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

pub(super) fn resolve_languages(
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

pub(super) fn resolve_workers(
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

pub(super) fn resolve_bytes(
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

pub(super) fn resolve_rows(
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

pub(super) fn resolve_duration(
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

pub(super) fn select_value(
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

pub(super) fn overlay_value(key: &str, value: &str) -> Result<Value, ConfigError> {
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

pub(super) fn get_path<'a>(value: &'a Value, path: &str) -> Result<Option<&'a Value>, ConfigError> {
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

pub(super) fn unknown_key_warnings(document: &Value) -> Vec<ConfigWarning> {
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

pub(super) fn collect_unknown(
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

pub(super) fn parse_bytes(input: &str) -> Result<u64, String> {
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

pub(super) fn parse_duration(input: &str) -> Result<Duration, String> {
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

pub(super) fn type_error(key: &str, expected: &str, value: &Value) -> ConfigError {
    ConfigError {
        key: key.into(),
        message: format!("expected {expected}, got {value}"),
    }
}
