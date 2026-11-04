pub const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
/// Constant identifying the byte cap.
pub const BYTE_CAP: u64 = 25_000_000_000;
/// Constant identifying the hot cap.
pub const HOT_CAP: u64 = 1_000_000_000_000;
/// Constant identifying the put cap.
pub const PUT_CAP: u64 = 1_500_000;
/// Constant identifying the keeper facts.
pub const KEEPER_FACTS: [&str; 4] = [
    "new_context_best",
    "parity_failure",
    "cited",
    "sealed_salt_audit",
];

/// Performs the unsigned operation and returns an error if it cannot be completed.
pub fn unsigned(value: &Value, name: &str) -> Result<u64> {
    value
        .as_u64()
        .with_context(|| format!("{name} must be a nonnegative integer"))
}
/// Performs the identity operation and returns an error if it cannot be completed.
pub fn identity(row: &Value) -> Result<[String; 3]> {
    let mut result = [String::new(), String::new(), String::new()];
    for (i, field) in ["bucket", "key", "version_id"].iter().enumerate() {
        result[i] = row[*field]
            .as_str()
            .filter(|s| !s.is_empty())
            .context("complete object version identity")?
            .to_owned();
    }
    Ok(result)
}

/// Performs the classify operation and returns an error if it cannot be completed.
pub fn classify(row: &Value, facts: Option<&Value>) -> Result<Value> {
    let id = identity(row)?;
    unsigned(&row["bytes"], "object bytes")?;
    let latest = row["is_latest"].as_bool().context("latest-version flag")?;
    let key = &id[1];
    let role = key.rsplit('/').next().unwrap_or_default();
    let mut kind = "keeper";
    let mut reason = "unmeasured_keeper_or_hold_facts";
    if id[2] == "null" {
        reason = "immutable_version_unavailable";
    } else if !latest {
        reason = "noncurrent_version_review";
    } else if key
        .split('/')
        .any(|p| ["sealed", "judge", "checkpoints", "gp-hwm", "provenance"].contains(&p))
    {
        reason = "protected_evidence";
    } else if [
        "score_records",
        "evaluation",
        "rust_parity",
        "input",
        "program",
        "rust_source",
        "record.json",
    ]
    .contains(&role)
    {
        reason = "permanent_ledger_or_small_artifact";
    } else if role != "rust_executable" {
        reason = "unknown_role";
    } else if let Some(facts) = facts.filter(|f| !f.is_null()) {
        ensure!(facts.is_object(), "keeper facts object");
        ensure!(
            facts["identity"] == json!(id),
            "keeper facts bind the exact physical version"
        );
        let measured = facts["legal_hold_checked"] == true
            && facts["retention_checked"] == true
            && KEEPER_FACTS.iter().all(|k| facts[*k].is_boolean())
            && facts["audit_salt_commitment"]
                .as_str()
                .is_some_and(is_hex64)
            && facts
                .as_object()
                .is_some_and(|o| o.contains_key("retention_until"));
        if measured {
            if facts["legal_hold"] != "OFF" || !facts["retention_until"].is_null() {
                reason = "protected_or_unknown_hold";
            } else if KEEPER_FACTS.iter().any(|k| facts[*k] == true) {
                reason = "verified_keeper_fact";
            } else {
                kind = "other";
                reason = "complete_negative_keeper_facts";
            }
        }
    }
    let mut result = row.as_object().context("object version row")?.clone();
    result.extend(
        json!({"role":role,"retention_class":kind,"reason":reason,
        "proposed_retention_days":if kind == "other" {Some(14)} else {None},
        "applied":false,"qualified":false})
        .as_object()
        .context("classification")?
        .clone(),
    );
    Ok(Value::Object(result))
}

/// Unknown counters and exhausted limits refuse a recommendation; no service is controlled.
pub fn budget_status(values: &Value) -> Result<Value> {
    let mut unknown = Vec::new();
    let mut alarm = Vec::new();
    let mut exhausted = Vec::new();
    for (name, cap) in [
        ("new_version_bytes", BYTE_CAP),
        ("raw_hot_bytes", HOT_CAP),
        ("puts", PUT_CAP),
    ] {
        if values[name].is_null() {
            unknown.push(name);
        } else {
            let n = unsigned(&values[name], name)?;
            if n >= cap * 4 / 5 {
                alarm.push(name);
            }
            if n >= cap {
                exhausted.push(name);
            }
        }
    }
    Ok(
        json!({"caps":{"new_version_bytes":BYTE_CAP,"raw_hot_bytes":HOT_CAP,"puts":PUT_CAP},
        "values":{"new_version_bytes":values["new_version_bytes"],"raw_hot_bytes":values["raw_hot_bytes"],"puts":values["puts"]},
        "admit_new_scoring":unknown.is_empty() && exhausted.is_empty(),
        "unknown":unknown,"alarm":alarm,"exhausted":exhausted,"control_applied":false}),
    )
}

/// Performs the bounded read operation and returns an error if it cannot be completed.
pub fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut raw = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut raw)?;
    ensure!(raw.len() as u64 <= limit, "bounded input bytes");
    Ok(raw)
}
