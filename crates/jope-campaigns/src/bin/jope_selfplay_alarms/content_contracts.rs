use super::*;

/// The whole point of workstream B: schema/state drift becomes a local, loud,
/// named failure instead of a silent cluster-wide outage.
pub(super) fn check_status_contracts(root: &Path, alarms: &mut Alarms, severity: &str) {
    let source_root =
        PathBuf::from(std::env::var("JOPE_SRC_ROOT").unwrap_or_else(|_| src_root_str().to_owned()));
    let state_path = root.join("ops/alarm-contract-state.v1.json");
    let previous = load_contract_state(&state_path);
    let mut next = serde_json::Map::new();
    let now = now_ms() as u64;

    for contract in OPS_CONTRACTS {
        let path = root.join(contract.file);
        let Ok(bytes) = std::fs::read(&path) else {
            // Absence is already covered by check_status_ages for the files it
            // watches; do not double-report here.
            continue;
        };
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                alarms.push(
                    &format!("status_unparseable:{}", contract.file),
                    severity,
                    truncate(&error.to_string(), 160),
                    format!("valid JSON written by {}", contract.producer),
                );
                continue;
            }
        };

        // VERSION GATE FIRST, on a Value — the correct pattern, as at
        // jope_selfplay_supply.rs:423-430 and jope_queue_control.rs:1026. The
        // inverted form (deserialize, then check `schema`) is unreachable for
        // exactly the documents it is supposed to reject.
        let found = value.get("schema").and_then(Value::as_str).unwrap_or("");
        if found != contract.schema {
            let streak = previous
                .get(contract.file)
                .and_then(|row| row.get("cycles"))
                .and_then(Value::as_u64)
                .unwrap_or(0)
                + 1;
            let since = previous
                .get(contract.file)
                .and_then(|row| row.get("broken_since_ms"))
                .and_then(Value::as_u64)
                .unwrap_or(now);
            next.insert(
                contract.file.to_owned(),
                json!({ "cycles": streak, "broken_since_ms": since }),
            );
            let detail = value
                .get("error")
                .and_then(Value::as_str)
                .map(|error| format!("; error={}", truncate(error, 160)))
                .unwrap_or_default();
            let name = if found.ends_with("Error.v1") {
                format!("status_error_doc:{}", contract.file)
            } else if found.is_empty() {
                format!("status_schema_missing:{}", contract.file)
            } else {
                format!("status_schema_mismatch:{}", contract.file)
            };
            alarms.push(
                &name,
                severity,
                format!("schema={found:?}{detail}"),
                format!("schema={:?} from {}", contract.schema, contract.producer),
            );
            if streak >= FAILING_CYCLES {
                alarms.push(
                    &format!("status_failing:{}", contract.file),
                    severity,
                    format!(
                        "{streak} consecutive scans off-schema, {}s since first failure",
                        now.saturating_sub(since) / 1000
                    ),
                    format!("<{FAILING_CYCLES} scans; {} is wedged", contract.producer),
                );
            }
            continue;
        }

        // In-schema but self-reporting degradation: the shape stayed stable, so
        // every consumer still parses, and the alarm still fires. This is the
        // pattern jope_selfplay_trainer_watchdog.rs:372 now uses, and the one
        // proposed for the queue keeper and supply.
        if value.get("degraded").and_then(Value::as_bool) == Some(true) {
            alarms.push(
                &format!("status_degraded:{}", contract.file),
                severity,
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .map_or_else(|| "degraded=true".to_owned(), |error| truncate(error, 160)),
                format!("{} completing its cycle", contract.producer),
            );
        }

        // Fresh mtime, frozen content. A daemon that rewrites an identical
        // document forever is invisible to every age-based guard in this file.
        if let Some((_, budget)) = CONTENT_FROZEN_SECS
            .iter()
            .find(|(f, _)| *f == contract.file)
        {
            let stamp = value
                .get("generated_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let mtime_age = file_age_secs(&path).unwrap_or(0);
            if stamp > 0 && mtime_age <= *budget {
                let content_age = now.saturating_sub(stamp) / 1000;
                if content_age > *budget {
                    alarms.push(
                        &format!("status_content_frozen:{}", contract.file),
                        severity,
                        format!("generated_at_ms {content_age}s old, mtime {mtime_age}s old"),
                        format!("generated_at_ms within {budget}s"),
                    );
                }
            }
        }

        // The consumer's own struct, applied to the bytes on disk.
        if let Some((relative, struct_name)) = contract.consumer {
            let source_path = source_root.join(relative);
            let Ok(source) = std::fs::read_to_string(&source_path) else {
                alarms.push(
                    &format!("contract_source_missing:{relative}"),
                    "warning",
                    format!("cannot read {}", source_path.display()),
                    "consumer source readable for contract derivation".into(),
                );
                continue;
            };
            let Some((line, required)) = derive_required_fields(&source, struct_name) else {
                alarms.push(
                    &format!("contract_underivable:{relative}"),
                    "warning",
                    format!("struct {struct_name} not found or uses rename_all"),
                    "a serde struct this check can model".into(),
                );
                continue;
            };
            for (field, problem) in contract_violations(&value, &required) {
                alarms.push(
                    &format!("contract_violation:{}#{field}", contract.file),
                    severity,
                    problem,
                    format!("required by {relative}:{line} struct {struct_name}"),
                );
            }
        }
    }

    let state = json!({
        "schema": "JopeSelfplayAlarmContractState.v1",
        "generated_at_ms": now,
        "files": Value::Object(next),
    });
    let temporary = root.join(format!(
        "ops/.alarm-contract-state.v1.json.{}",
        std::process::id()
    ));
    if let Ok(bytes) = serde_json::to_vec(&state) {
        if std::fs::write(&temporary, bytes).is_ok() {
            let _ = std::fs::rename(&temporary, &state_path);
        }
    }
}
