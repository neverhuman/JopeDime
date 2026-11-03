use super::*;

pub(crate) const FAILING_CYCLES: u64 = 3;

/// Cadence budget for the content-frozen check, per file. Deliberately looser
/// than STATUS_AGES: this fires only when mtime is FRESH and the embedded
/// timestamp is old, which is the one thing mtime cannot see.
pub(crate) const CONTENT_FROZEN_SECS: &[(&str, u64)] = &[
    ("ops/queue-keeper-v3.json", 900),
    ("ops/micro-wave-supply-v2.json", 300),
    ("ops/task-learning-controller.v2.json", 300),
    ("ops/freshness-status.json", 600),
];

/// One live producer -> consumer contract.
pub(crate) struct Contract {
    /// Path relative to the flywheel root.
    pub(crate) file: &'static str,
    /// The one schema string a healthy document carries.
    pub(crate) schema: &'static str,
    /// Unit that writes it (for the alarm text — this is who to go look at).
    pub(crate) producer: &'static str,
    /// `(source file relative to src_root_str(), struct name)` of a consumer that
    /// deserializes this file into a strict struct. `None` means every known
    /// consumer reads it as a `serde_json::Value`, so the schema gate is the
    /// whole contract.
    pub(crate) consumer: Option<(&'static str, &'static str)>,
}

/// Verified against the live root on 2026-07-28 by reading the `schema` key of
/// every `$R/ops/*.json`. Consumers were found with
/// `grep -rn "ops/<file>" src/`.
pub(crate) const OPS_CONTRACTS: &[Contract] = &[
    Contract {
        file: "ops/queue-keeper-v3.json",
        schema: "JopeSelfplayQueueKeeper.v3",
        producer: "jope-selfplay-queue.service",
        // jope_selfplay_supply.rs:404-416 load_queue_status(); the struct whose
        // non-Option `queued_target` took supply down with the keeper.
        consumer: Some(("src/bin/jope_selfplay_supply.rs", "QueueStatus")),
    },
    Contract {
        file: "ops/micro-wave-supply-v2.json",
        schema: "JopeMicroWaveSupplyStatus.v2",
        producer: "jope-selfplay-supply.service",
        consumer: None,
    },
    Contract {
        file: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        producer: "jope-selfplay-trainer-watchdog.service",
        consumer: None,
    },
    Contract {
        file: "ops/trainer-watchdog-state.v1.json",
        schema: "JopeContinuousTrainerWatchdogState.v1",
        producer: "jope-selfplay-trainer-watchdog.service",
        // jope_selfplay_trainer_watchdog.rs:690 load_state().
        consumer: Some(("src/bin/jope_selfplay_trainer_watchdog.rs", "DurableState")),
    },
    Contract {
        file: "ops/freshness-status.json",
        schema: "JopeSelfplayFreshnessStatus.v1",
        producer: "jope-selfplay-freshness.service",
        consumer: None,
    },
    Contract {
        file: "ops/freshness-state.v1.json",
        schema: "JopeFreshnessState.v1",
        producer: "jope-selfplay-freshness.service",
        // jope_selfplay_freshness.rs:328 — survives a wrong-shaped document
        // only because the struct is #[serde(default)]; the schema gate there
        // runs AFTER the deserialize, so it can never reject one.
        consumer: Some(("src/bin/jope_selfplay_freshness.rs", "DurableState")),
    },
    Contract {
        file: "ops/expansion-driver-v1.json",
        schema: "JopeSelfplayExpansionDriver.v1",
        producer: "jope-selfplay-expansion.service",
        consumer: None,
    },
    Contract {
        file: "ops/trainer-warm-start-guard.v1.json",
        schema: "JopeTrainerWarmStartGuard.v1",
        producer: "jope-selfplay-trainer-watchdog.service",
        consumer: None,
    },
    Contract {
        file: "ops/training-replica.v1.json",
        schema: "JopeTrainingReplicaStatus.v1",
        producer: "jope-training-replica.service",
        consumer: None,
    },
    Contract {
        file: "ops/flywheel-state.json",
        schema: "JopeSelfplayControllerState.v1",
        producer: "jope-selfplay-promotion.service",
        consumer: None,
    },
    Contract {
        file: "ops/propose-driver-v1.json",
        schema: "JopeProposeDriver.v1",
        producer: "jope-selfplay-propose.service",
        consumer: None,
    },
    Contract {
        file: "ops/seeding-curve.v1.json",
        schema: "JopeSeedingCurve.v1",
        producer: "jope-selfplay-ingestion.service",
        consumer: None,
    },
    Contract {
        file: "ops/status-v3.json",
        schema: "JopeSelfplayOperationalStatus.v3",
        producer: "jope-selfplay-ingestion.service",
        consumer: None,
    },
];

/// The JSON shape a serde field demands. `Any` means "presence only" — a nested
/// struct or enum whose internals this check deliberately does not model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Any,
    Str,
    Bool,
    Num,
    Arr,
    Obj,
}

impl Kind {
    fn matches(self, value: &Value) -> bool {
        match self {
            Kind::Any => !value.is_null(),
            Kind::Str => value.is_string(),
            Kind::Bool => value.is_boolean(),
            Kind::Num => value.is_number(),
            Kind::Arr => value.is_array(),
            Kind::Obj => value.is_object(),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Kind::Any => "a value",
            Kind::Str => "string",
            Kind::Bool => "bool",
            Kind::Num => "number",
            Kind::Arr => "array",
            Kind::Obj => "object",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RequiredField {
    pub(crate) name: String,
    pub(crate) kind: Kind,
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn kind_of_type(ty: &str) -> Kind {
    let ty = ty.trim();
    if ty.starts_with("Option<") {
        return Kind::Any;
    }
    if ty.starts_with("Vec<") || ty.starts_with("[") {
        return Kind::Arr;
    }
    if ty.starts_with("HashMap<") || ty.starts_with("BTreeMap<") || ty.starts_with("Map<") {
        return Kind::Obj;
    }
    match ty {
        "String" | "&str" | "&'static str" | "PathBuf" | "Box<str>" => Kind::Str,
        "bool" => Kind::Bool,
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
        | "isize" | "f32" | "f64" => Kind::Num,
        _ => Kind::Any,
    }
}

/// Derive the fields a consumer's serde struct REQUIRES from that consumer's
/// own source. Returns `(declaration line, required fields)`.
///
/// Deliberately conservative: anything this parser is not sure about becomes
/// `Kind::Any` (presence only) or is dropped from the required set, so a false
/// alarm needs the source to actually say the field is mandatory. `None` means
/// the struct was not found or uses a serde attribute this cannot reason about
/// (`rename_all`), which the caller turns into its own visible alarm rather
/// than a silent skip.
pub(crate) fn derive_required_fields(
    source: &str,
    struct_name: &str,
) -> Option<(usize, Vec<RequiredField>)> {
    let lines: Vec<&str> = source.lines().collect();
    let head = lines.iter().position(|line| {
        let trimmed = line.trim_start();
        let trimmed = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
        trimmed.strip_prefix("struct ").and_then(|rest| {
            rest.split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
        }) == Some(struct_name)
    })?;

    // Container attributes sit immediately above the declaration.
    let mut container = String::new();
    let mut cursor = head;
    while cursor > 0 {
        let previous = lines[cursor - 1].trim();
        if previous.starts_with("#[") || previous.starts_with("///") || previous.starts_with("//") {
            container.push_str(previous);
            cursor -= 1;
        } else {
            break;
        }
    }
    if container.contains("rename_all") {
        return None;
    }
    // `#[serde(default)]` on the container makes every field optional — this is
    // exactly why jope_selfplay_freshness.rs:328 survives a wrong document.
    let container_default = container.contains("#[serde(default)]")
        || container.contains("serde(default,")
        || container.contains(", default)")
        || container.contains(", default,");

    let mut fields = Vec::new();
    let mut pending = String::new();
    for line in lines.iter().skip(head + 1) {
        let trimmed = line.trim();
        if trimmed == "}" {
            break;
        }
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("#[") {
            pending.push_str(trimmed);
            continue;
        }
        let attributes = std::mem::take(&mut pending);
        if attributes.contains("skip)")
            || attributes.contains("skip,")
            || attributes.contains("skip_deserializing")
        {
            continue;
        }
        let declaration = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
        let Some((raw_name, raw_type)) = declaration.split_once(':') else {
            continue;
        };
        let name = match attributes.find("rename = \"") {
            Some(index) => {
                let rest = &attributes[index + "rename = \"".len()..];
                match rest.find('"') {
                    Some(end) => rest[..end].to_owned(),
                    None => raw_name.trim().to_owned(),
                }
            }
            None => raw_name.trim().to_owned(),
        };
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let ty = raw_type.trim().trim_end_matches(',').trim();
        let optional =
            container_default || attributes.contains("default") || ty.starts_with("Option<");
        if optional {
            continue;
        }
        fields.push(RequiredField {
            name,
            kind: kind_of_type(ty),
        });
    }
    Some((head + 1, fields))
}

/// `(field, what is wrong)` for every required field the document breaks.
pub(crate) fn contract_violations(
    value: &Value,
    fields: &[RequiredField],
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for field in fields {
        match value.get(&field.name) {
            None => out.push((field.name.clone(), "missing".to_owned())),
            Some(found) if !field.kind.matches(found) => out.push((
                field.name.clone(),
                format!(
                    "expected {}, found {}",
                    field.kind.describe(),
                    json_kind(found)
                ),
            )),
            Some(_) => {}
        }
    }
    out
}

/// Persisted streak counters. Small, rewritten every scan, and never load
/// bearing: losing it costs at most FAILING_CYCLES minutes of escalation.
pub(crate) fn load_contract_state(path: &Path) -> serde_json::Map<String, Value> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| value.get("files").cloned())
        .and_then(|files| files.as_object().cloned())
        .unwrap_or_default()
}

pub(crate) fn truncate(text: &str, limit: usize) -> String {
    let cleaned = text.replace('\n', " ");
    match cleaned.char_indices().nth(limit) {
        Some((index, _)) => format!("{}…", &cleaned[..index]),
        None => cleaned,
    }
}
